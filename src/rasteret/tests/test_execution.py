# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for core async utilities and loading."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr

from rasteret.core.collection import Collection
from rasteret.core.execution import _detect_target_crs, get_collection_xarray
from rasteret.core.utils import infer_data_source, run_sync


class TestRunSync:
    def test_outside_event_loop(self):
        async def coro():
            return 42

        assert run_sync(coro()) == 42

    def test_inside_running_loop(self):
        """Simulates Jupyter: run_sync dispatches to a thread."""
        result = None

        async def outer():
            nonlocal result
            result = run_sync(_inner())

        async def _inner():
            return "from-thread"

        asyncio.run(outer())
        assert result == "from-thread"


class TestInferDataSource:
    def _make_collection_and_infer(
        self, *, data_source: str = "", collection_col: str | None = None
    ) -> str:
        """Build a Collection in a temp dir and infer its data source.

        The temp dir must stay alive while infer_data_source reads the
        dataset, so we run the assertion inside the context manager.
        """
        cols = {
            "id": pa.array(["s1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "h"}}]),
            "year": pa.array([2024], type=pa.int32()),
            "month": pa.array([1], type=pa.int32()),
        }
        if collection_col:
            cols["collection"] = pa.array([collection_col])
        table = pa.table(cols)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "test"
            pq.write_to_dataset(
                table, root_path=str(path), partition_cols=["year", "month"]
            )
            c = Collection._load_cached(path)
            c.data_source = data_source
            return infer_data_source(c)

    def test_explicit_data_source(self):
        assert (
            self._make_collection_and_infer(data_source="landsat-c2-l2")
            == "landsat-c2-l2"
        )

    def test_column_based(self):
        assert (
            self._make_collection_and_infer(collection_col="my-custom-source")
            == "my-custom-source"
        )

    def test_default_fallback(self):
        assert self._make_collection_and_infer() == ""


class TestXarrayMerge:
    def test_merge_without_time_coord_does_not_crash(self):
        with (
            patch(
                "rasteret.core.execution._ensure_geoarrow",
                return_value=pa.array([]),
            ),
            patch(
                "rasteret.core.execution._load_collection_data",
                new=AsyncMock(return_value=([xr.Dataset()], [])),
            ),
        ):
            merged = get_collection_xarray(
                collection=None,  # collection is unused due patched _load_collection_data
                geometries=[],
                bands=["B04"],
                data_source="sentinel-2-l2a",
            )
        assert isinstance(merged, xr.Dataset)

    def test_no_valid_data_surfaces_first_error(self):
        first = ValueError("Missing band metadata or href for band 'B04' in record 'x'")
        with (
            patch(
                "rasteret.core.execution._ensure_geoarrow",
                return_value=pa.array([]),
            ),
            patch(
                "rasteret.core.execution._load_collection_data",
                new=AsyncMock(return_value=([], [("x", first)])),
            ),
        ):
            with pytest.raises(
                ValueError, match="first failure in record 'x'"
            ) as excinfo:
                get_collection_xarray(
                    collection=None,  # unused due to patched _load_collection_data
                    geometries=[],
                    bands=["B04"],
                    data_source="sentinel-2-l2a",
                )
        assert excinfo.value.__cause__ is first


class TestDetectTargetCrs:
    """Tests for _detect_target_crs multi-CRS auto-detection."""

    def _make_collection(self, epsg_values: list[int]) -> tuple[Collection, str]:
        """Build a Collection with given proj:epsg values.

        Returns (collection, tmpdir_path).  Caller must keep tmpdir alive.
        """
        n = len(epsg_values)
        table = pa.table(
            {
                "id": pa.array([f"r{i}" for i in range(n)]),
                "datetime": pa.array(
                    [datetime(2024, 1, 1)] * n, type=pa.timestamp("us")
                ),
                "geometry": pa.array([None] * n, type=pa.null()),
                "assets": pa.array([{"B04": {"href": "h"}}] * n),
                "proj:epsg": pa.array(epsg_values, type=pa.int32()),
            }
        )
        return table, epsg_values

    def test_single_crs_returns_none(self):
        table, _ = self._make_collection([32632, 32632, 32632])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "single_crs"
            path.mkdir()
            pq.write_table(table, str(path / "data.parquet"))
            c = Collection._load_cached(path)
            assert _detect_target_crs(c, {}) is None

    def test_multi_crs_returns_most_common(self):
        # 3× EPSG:32632, 1× EPSG:32633 → should pick 32632
        table, _ = self._make_collection([32632, 32632, 32632, 32633])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi_crs"
            path.mkdir()
            pq.write_table(table, str(path / "data.parquet"))
            c = Collection._load_cached(path)
            result = _detect_target_crs(c, {})
            assert result == 32632

    def test_multi_crs_equal_counts_picks_one(self):
        # 2× each → should pick one deterministically
        table, _ = self._make_collection([32632, 32632, 32633, 32633])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "equal_crs"
            path.mkdir()
            pq.write_table(table, str(path / "data.parquet"))
            c = Collection._load_cached(path)
            result = _detect_target_crs(c, {})
            assert result in (32632, 32633)
