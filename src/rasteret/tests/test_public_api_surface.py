# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rasteret
from rasteret.core.collection import Collection


def _write_minimal_partitioned_collection(dataset_path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
            "collection": pa.array(["sentinel-2-l2a"]),
            "year": pa.array([2024], type=pa.int32()),
            "month": pa.array([1], type=pa.int32()),
        }
    )
    pq.write_to_dataset(
        table, root_path=str(dataset_path), partition_cols=["year", "month"]
    )


def test_build_from_stac_returns_cached_collection_without_network() -> None:
    with TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)
        collection_name = Collection.create_name(
            "demo", ("2024-01-01", "2024-01-31"), "sentinel-2-l2a"
        )
        dataset_path = workspace / f"{collection_name}_stac"
        _write_minimal_partitioned_collection(dataset_path)

        collection = rasteret.build_from_stac(
            name="demo",
            stac_api="https://example-stac.invalid/v1",
            collection="sentinel-2-l2a",
            bbox=(0.0, 0.0, 1.0, 1.0),
            date_range=("2024-01-01", "2024-01-31"),
            workspace_dir=workspace,
        )

        assert isinstance(collection, Collection)
        assert collection.name == collection_name


def test_collection_analysis_methods_delegate_to_execution_layer() -> None:
    with TemporaryDirectory() as tmp_dir:
        dataset_path = Path(tmp_dir) / "example_stac"
        _write_minimal_partitioned_collection(dataset_path)
        collection = Collection._load_cached(dataset_path)

    with (
        patch(
            "rasteret.core.collection.get_collection_xarray",
            return_value="xarray-result",
        ) as mocked_xarray,
        patch(
            "rasteret.core.collection.get_collection_gdf", return_value="gdf-result"
        ) as mocked_gdf,
        patch(
            "rasteret.core.collection.get_collection_numpy",
            return_value="numpy-result",
        ) as mocked_numpy,
    ):
        xarray_result = collection.get_xarray(geometries=[], bands=["B04"])
        gdf_result = collection.get_gdf(geometries=[], bands=["B04"])
        numpy_result = collection.get_numpy(geometries=[], bands=["B04"])

    mocked_xarray.assert_called_once()
    mocked_gdf.assert_called_once()
    mocked_numpy.assert_called_once()
    assert xarray_result == "xarray-result"
    assert gdf_result == "gdf-result"
    assert numpy_result == "numpy-result"


def test_public_api_surface_is_collection_first() -> None:
    exported = set(dir(rasteret))
    assert "build_from_stac" in exported
    assert "load" in exported
    assert "build_from_table" in exported
    assert "Collection" in exported
    assert "BandRegistry" in exported

    assert "Rasteret" not in exported


def test_top_level_rasteret_class_is_not_exposed() -> None:
    with pytest.raises(AttributeError):
        _ = rasteret.Rasteret


def _write_minimal_flat_collection(path: Path) -> None:
    """Write a flat (non-Hive) parquet with required columns."""
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
            "scene_bbox": pa.array(
                [[0.0, 0.0, 1.0, 1.0]], type=pa.list_(pa.float64(), 4)
            ),
        }
    )
    pq.write_table(table, str(path))


def test_load_valid_file() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test.parquet"
        _write_minimal_flat_collection(path)
        collection = rasteret.load(path)
        assert isinstance(collection, Collection)
        assert collection.name == "test"


def test_load_rejects_missing_columns() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bad.parquet"
        table = pa.table({"id": pa.array(["scene-1"]), "foo": pa.array([1])})
        pq.write_table(table, str(path))
        with pytest.raises(ValueError, match="missing required columns"):
            rasteret.load(path)


def test_load_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        rasteret.load("/nonexistent/path.parquet")


def test_from_parquet_fallback_to_non_hive() -> None:
    """from_parquet should work on non-Hive partitioned parquet."""
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "flat.parquet"
        _write_minimal_flat_collection(path)
        collection = Collection.from_parquet(path)
        assert isinstance(collection, Collection)
