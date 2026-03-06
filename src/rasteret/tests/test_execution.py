# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for core async utilities and loading."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr

from rasteret.core.collection import Collection
from rasteret.core.execution import (
    _detect_target_crs,
    get_collection_numpy,
    get_collection_xarray,
)
from rasteret.core.geometry import ensure_point_geoarrow as _ensure_point_geoarrow
from rasteret.core.point_sampling import get_collection_point_samples
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


class TestGdfCrs:
    def test_gdf_merge_preserves_crs(self):
        import geopandas as gpd
        from shapely.geometry import box

        g1 = gpd.GeoDataFrame(
            {"band": ["B04"], "data": [np.ones((1, 1), dtype=np.uint8)]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:3857",
        )
        g2 = gpd.GeoDataFrame(
            {"band": ["B04"], "data": [np.ones((1, 1), dtype=np.uint8)]},
            geometry=[box(1, 1, 2, 2)],
            crs="EPSG:3857",
        )

        with (
            patch(
                "rasteret.core.execution._ensure_geoarrow",
                return_value=pa.array([]),
            ),
            patch(
                "rasteret.core.execution._load_collection_data",
                new=AsyncMock(return_value=([g1, g2], [])),
            ),
        ):
            from rasteret.core.execution import get_collection_gdf

            out = get_collection_gdf(
                collection=None,  # unused due to patched _load_collection_data
                geometries=[],
                bands=["B04"],
                data_source="sentinel-2-l2a",
            )
        assert str(out.crs) == "EPSG:3857"


class TestGeometryErrors:
    @pytest.mark.asyncio
    async def test_unsupported_geometry_error_is_not_silently_swallowed(self):
        """Geometry type errors should fail loudly, not become 'No valid data found'."""
        from rasteret.core.geometry import UnsupportedGeometryError
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import RasterInfo

        class _DummyReader:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        info = RasterInfo(
            id="r1",
            datetime=datetime(2024, 1, 1),
            bbox=[0.0, 0.0, 1.0, 1.0],
            footprint=None,
            crs=4326,
            cloud_cover=0.0,
            assets={"B04": {"href": "https://example.com/x.tif"}},
            band_metadata={"B04_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")

        with (
            patch("rasteret.fetch.cog.COGReader", return_value=_DummyReader()),
            patch.object(
                accessor,
                "_load_single_band",
                new=AsyncMock(
                    side_effect=UnsupportedGeometryError("Point geometry not supported")
                ),
            ),
        ):
            # If RasterAccessor swallows the exception, callers end up with a
            # misleading "No valid data found" error later.
            with pytest.raises(UnsupportedGeometryError, match="Unsupported geometry"):
                await accessor.load_bands(
                    geometries=pa.array([None]),
                    band_codes=["B04"],
                    max_concurrent=5,
                    for_xarray=False,
                )

    @pytest.mark.asyncio
    async def test_all_band_failures_raise_with_cause(self):
        """If every requested band fails for a geometry, surface the first error."""
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import RasterInfo

        class _DummyReader:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        info = RasterInfo(
            id="r2",
            datetime=datetime(2024, 1, 1),
            bbox=[0.0, 0.0, 1.0, 1.0],
            footprint=None,
            crs=4326,
            cloud_cover=0.0,
            assets={"B04": {"href": "https://example.com/x.tif"}},
            band_metadata={"B04_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")

        boom = ValueError("boom")
        with (
            patch("rasteret.fetch.cog.COGReader", return_value=_DummyReader()),
            patch.object(
                accessor, "_load_single_band", new=AsyncMock(side_effect=boom)
            ),
        ):
            with pytest.raises(
                RuntimeError, match="All geometry reads failed"
            ) as excinfo:
                await accessor.load_bands(
                    geometries=pa.array([None]),
                    band_codes=["B04"],
                    max_concurrent=5,
                    for_xarray=False,
                )
        assert excinfo.value.__cause__ is not None
        assert excinfo.value.__cause__.__cause__ is boom

    @pytest.mark.asyncio
    async def test_partial_band_failures_warn(self):
        """If some bands fail but others succeed, warn instead of silently dropping."""
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import RasterInfo

        class _DummyReader:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        info = RasterInfo(
            id="r3",
            datetime=datetime(2024, 1, 1),
            bbox=[0.0, 0.0, 1.0, 1.0],
            footprint=None,
            crs=4326,
            cloud_cover=0.0,
            assets={
                "B04": {"href": "https://example.com/x.tif"},
                "B08": {"href": "https://example.com/x.tif"},
            },
            band_metadata={"B04_metadata": {}, "B08_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")

        async def _side_effect(_geom_array, _geom_idx, band_code, *_args, **_kwargs):
            if band_code == "B04":
                return {
                    "data": np.ones((1, 1), dtype=np.uint8),
                    "transform": None,
                    "band": "B04",
                }
            raise ValueError("boom")

        with (
            patch("rasteret.fetch.cog.COGReader", return_value=_DummyReader()),
            patch.object(
                accessor, "_load_single_band", new=AsyncMock(side_effect=_side_effect)
            ),
            patch.object(
                accessor, "_merge_geodataframe_results", return_value=pd.DataFrame()
            ),
        ):
            with pytest.warns(
                RuntimeWarning, match="Partial read failures for record_id='r3'"
            ):
                await accessor.load_bands(
                    geometries=pa.array([None]),
                    band_codes=["B04", "B08"],
                    max_concurrent=5,
                    for_xarray=False,
                )

    @pytest.mark.asyncio
    async def test_partial_geometry_failures_warn(self):
        """If some geometries fail but others succeed, warn instead of silently dropping."""
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import RasterInfo

        class _DummyReader:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        info = RasterInfo(
            id="r4",
            datetime=datetime(2024, 1, 1),
            bbox=[0.0, 0.0, 1.0, 1.0],
            footprint=None,
            crs=4326,
            cloud_cover=0.0,
            assets={"B04": {"href": "https://example.com/x.tif"}},
            band_metadata={"B04_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")

        async def _side_effect(_geom_array, geom_idx, band_code, *_args, **_kwargs):
            if geom_idx == 0:
                return {
                    "data": np.ones((1, 1), dtype=np.uint8),
                    "transform": None,
                    "band": band_code,
                }
            raise ValueError("boom")

        with (
            patch("rasteret.fetch.cog.COGReader", return_value=_DummyReader()),
            patch.object(
                accessor, "_load_single_band", new=AsyncMock(side_effect=_side_effect)
            ),
            patch.object(
                accessor, "_merge_geodataframe_results", return_value=pd.DataFrame()
            ),
        ):
            with pytest.warns(
                RuntimeWarning, match="Partial read failures for record_id='r4'"
            ):
                await accessor.load_bands(
                    geometries=pa.array([None, None]),
                    band_codes=["B04"],
                    max_concurrent=5,
                    for_xarray=False,
                )


class TestRasterAccessorGdfCrs:
    @pytest.mark.asyncio
    async def test_load_bands_gdf_sets_crs_and_reprojects_geometry(self):
        """GeoDataFrame output should always have a CRS and match target_crs when given."""
        import geopandas as gpd
        from shapely.geometry import box

        from rasteret.core.geometry import coerce_to_geoarrow
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import RasterInfo

        class _DummyReader:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        info = RasterInfo(
            id="r-crs",
            datetime=datetime(2024, 1, 1),
            bbox=[0.0, 0.0, 1.0, 1.0],
            footprint=None,
            crs=3857,
            cloud_cover=0.0,
            assets={"B04": {"href": "https://example.com/x.tif"}},
            band_metadata={"B04_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")

        geom = box(0.0, 0.0, 1.0, 1.0)
        geometries = coerce_to_geoarrow(geom)

        async def _ok(*_args, **_kwargs):
            return {
                "data": np.ones((1, 1), dtype=np.uint8),
                "transform": None,
                "band": "B04",
            }

        with (
            patch("rasteret.fetch.cog.COGReader", return_value=_DummyReader()),
            patch.object(accessor, "_load_single_band", new=AsyncMock(side_effect=_ok)),
        ):
            out = await accessor.load_bands(
                geometries=geometries,
                band_codes=["B04"],
                max_concurrent=5,
                for_xarray=False,
                target_crs=3857,
                geometry_crs=4326,
            )

        assert isinstance(out, gpd.GeoDataFrame)
        assert str(out.crs) == "EPSG:3857"
        bounds = out.geometry.iloc[0].bounds
        # Rough sanity: degrees->meters for 1 degree at equator (~111km).
        assert bounds[2] > 100_000


class TestRasterAccessorPointSampling:
    @pytest.mark.asyncio
    async def test_sample_points_reads_single_pixel(self):
        from shapely.geometry import Point

        from rasteret.core.geometry import coerce_to_geoarrow
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import CogMetadata, RasterInfo

        class _DummyReader:
            def __init__(self):
                self.calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def read_merged_tile_samples(self, **kwargs):
                self.calls.append(kwargs)
                tile_key = kwargs["tiles"][0]
                return {tile_key: [np.full((10, 10), 123, dtype=np.uint16)]}

        info = RasterInfo(
            id="r-point",
            datetime=np.datetime64("2024-01-01T00:00:00", "ns"),
            bbox=[0.0, 0.0, 1.0, 1.0],
            footprint=None,
            crs=4326,
            cloud_cover=0.0,
            assets={"B04": {"href": "https://example.com/x.tif"}},
            band_metadata={"B04_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")
        points = coerce_to_geoarrow(Point(0.5, 0.5))

        meta = CogMetadata(
            width=10,
            height=10,
            tile_width=10,
            tile_height=10,
            dtype=np.dtype("uint16"),
            crs=4326,
            transform=[0.1, 0.0, 0.0, 0.0, -0.1, 1.0],
            tile_offsets=[0],
            tile_byte_counts=[1],
        )

        reader = _DummyReader()

        with (
            patch("rasteret.fetch.cog.COGReader", return_value=reader),
            patch.object(
                accessor,
                "try_get_band_cog_metadata",
                return_value=(meta, "https://example.com/x.tif", 0),
            ),
        ):
            rows = await accessor.sample_points(
                points=points,
                band_codes=["B04"],
                point_indices=[7],
                max_concurrent=5,
            )

        assert rows.num_rows == 1
        assert len(reader.calls) == 1
        assert rows.schema.field("datetime").type == pa.timestamp("us")
        assert rows.column("point_index")[0].as_py() == 7
        assert rows.column("record_id")[0].as_py() == "r-point"
        assert rows.column("band")[0].as_py() == "B04"
        assert rows.column("value")[0].as_py() == 123.0

    @pytest.mark.asyncio
    async def test_sample_points_batches_points_in_same_tile(self):
        from shapely.geometry import Point

        from rasteret.core.geometry import coerce_to_geoarrow
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import CogMetadata, RasterInfo

        class _DummyReader:
            def __init__(self):
                self.calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def read_merged_tile_samples(self, **kwargs):
                self.calls += 1
                tile_key = kwargs["tiles"][0]
                tile = np.arange(100, dtype=np.uint16).reshape(10, 10)
                return {tile_key: [tile]}

        info = RasterInfo(
            id="r-batch",
            datetime=np.datetime64("2024-01-01T00:00:00", "ns"),
            bbox=[0.0, 0.0, 10.0, 10.0],
            footprint=None,
            crs=4326,
            cloud_cover=0.0,
            assets={"B04": {"href": "https://example.com/x.tif"}},
            band_metadata={"B04_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")
        points = coerce_to_geoarrow([Point(1.5, 8.5), Point(2.5, 7.5)])

        meta = CogMetadata(
            width=10,
            height=10,
            tile_width=10,
            tile_height=10,
            dtype=np.dtype("uint16"),
            crs=4326,
            transform=[1.0, 0.0, 0.0, 0.0, -1.0, 10.0],
            tile_offsets=[0],
            tile_byte_counts=[1],
        )
        reader = _DummyReader()

        with (
            patch("rasteret.fetch.cog.COGReader", return_value=reader),
            patch.object(
                accessor,
                "try_get_band_cog_metadata",
                return_value=(meta, "https://example.com/x.tif", 0),
            ),
        ):
            rows = await accessor.sample_points(
                points=points,
                band_codes=["B04"],
                max_concurrent=5,
            )

        assert reader.calls == 1
        assert rows.num_rows == 2
        assert rows.column("value").to_pylist() == [11.0, 22.0]

    @pytest.mark.asyncio
    async def test_sample_points_groups_same_href_band_indices(self):
        from shapely.geometry import Point

        from rasteret.core.geometry import coerce_to_geoarrow
        from rasteret.core.raster_accessor import RasterAccessor
        from rasteret.types import CogMetadata, RasterInfo

        class _DummyReader:
            def __init__(self):
                self.calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def read_merged_tile_samples(self, **kwargs):
                self.calls.append(kwargs)
                tile_key = kwargs["tiles"][0]
                return {
                    tile_key: [
                        np.full((10, 10), 11, dtype=np.uint16),
                        np.full((10, 10), 22, dtype=np.uint16),
                    ]
                }

        info = RasterInfo(
            id="r-multi",
            datetime=np.datetime64("2024-01-01T00:00:00", "ns"),
            bbox=[0.0, 0.0, 10.0, 10.0],
            footprint=None,
            crs=4326,
            cloud_cover=0.0,
            assets={
                "B04": {"href": "https://example.com/x.tif", "band_index": 0},
                "B08": {"href": "https://example.com/x.tif", "band_index": 1},
            },
            band_metadata={"B04_metadata": {}, "B08_metadata": {}},
            collection="c",
        )
        accessor = RasterAccessor(info, data_source="")
        points = coerce_to_geoarrow(Point(1.5, 8.5))

        meta = CogMetadata(
            width=10,
            height=10,
            tile_width=10,
            tile_height=10,
            dtype=np.dtype("uint16"),
            crs=4326,
            transform=[1.0, 0.0, 0.0, 0.0, -1.0, 10.0],
            tile_offsets=[0],
            tile_byte_counts=[1],
            samples_per_pixel=2,
            planar_configuration=1,
        )
        reader = _DummyReader()

        with (
            patch("rasteret.fetch.cog.COGReader", return_value=reader),
            patch.object(
                accessor,
                "try_get_band_cog_metadata",
                side_effect=[
                    (meta, "https://example.com/x.tif", 0),
                    (meta, "https://example.com/x.tif", 1),
                ],
            ),
        ):
            rows = await accessor.sample_points(
                points=points,
                band_codes=["B04", "B08"],
                max_concurrent=5,
            )

        assert len(reader.calls) == 1
        assert reader.calls[0]["band_indices"] == [0, 1]
        assert rows.column("band").to_pylist() == ["B04", "B08"]
        assert rows.column("value").to_pylist() == [11.0, 22.0]


class TestNumpyOutput:
    def test_numpy_stack_multiband(self):
        frame = pd.DataFrame(
            {
                "band": ["B02", "B08", "B02", "B08"],
                "data": [
                    np.ones((2, 2), dtype=np.uint16),
                    np.full((2, 2), 2, dtype=np.uint16),
                    np.full((2, 2), 3, dtype=np.uint16),
                    np.full((2, 2), 4, dtype=np.uint16),
                ],
            }
        )
        with (
            patch(
                "rasteret.core.execution._ensure_geoarrow",
                return_value=pa.array([]),
            ),
            patch(
                "rasteret.core.execution._load_collection_data",
                new=AsyncMock(return_value=([frame], [])),
            ),
        ):
            out = get_collection_numpy(
                collection=None,  # unused due patched _load_collection_data
                geometries=[],
                bands=["B02", "B08"],
                data_source="sentinel-2-l2a",
            )

        assert out.shape == (2, 2, 2, 2)
        assert out.dtype == np.uint16
        assert np.all(out[0, 0] == 1)
        assert np.all(out[0, 1] == 2)
        assert np.all(out[1, 0] == 3)
        assert np.all(out[1, 1] == 4)

    def test_numpy_ragged_raises(self):
        frame = pd.DataFrame(
            {
                "band": ["B02", "B02"],
                "data": [
                    np.ones((2, 2), dtype=np.uint16),
                    np.ones((3, 3), dtype=np.uint16),
                ],
            }
        )
        with (
            patch(
                "rasteret.core.execution._ensure_geoarrow",
                return_value=pa.array([]),
            ),
            patch(
                "rasteret.core.execution._load_collection_data",
                new=AsyncMock(return_value=([frame], [])),
            ),
        ):
            with pytest.raises(ValueError, match="Ragged shapes"):
                get_collection_numpy(
                    collection=None,
                    geometries=[],
                    bands=["B02"],
                    data_source="sentinel-2-l2a",
                )


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
        # 3x EPSG:32632, 1x EPSG:32633 -> should pick 32632
        table, _ = self._make_collection([32632, 32632, 32632, 32633])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi_crs"
            path.mkdir()
            pq.write_table(table, str(path / "data.parquet"))
            c = Collection._load_cached(path)
            result = _detect_target_crs(c, {})
            assert result == 32632

    def test_multi_crs_equal_counts_picks_one(self):
        # 2x each -> should pick one deterministically
        table, _ = self._make_collection([32632, 32632, 32633, 32633])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "equal_crs"
            path.mkdir()
            pq.write_table(table, str(path / "data.parquet"))
            c = Collection._load_cached(path)
            result = _detect_target_crs(c, {})
            assert result in (32632, 32633)


class TestPointSampling:
    class _DummyCollection:
        def __init__(self, rasters):
            self._rasters = rasters
            self.data_source = ""
            self.dataset = None
            self.last_filters = None

        def subset(self, **_filters):
            self.last_filters = _filters
            return self

        async def iterate_rasters(self, _data_source=None):
            for raster in self._rasters:
                yield raster

    class _DummyRaster:
        def __init__(self, rows, *, raster_id="r", bbox=None):
            self.id = raster_id
            self.bbox = bbox
            self._rows = rows
            self.calls = []
            self.datetime = rows[0]["datetime"] if rows else None

        async def sample_points(self, **_kwargs):
            self.calls.append(_kwargs)
            return self._rows

    def test_ensure_point_geoarrow_arrow_table_xy(self):
        points = pa.table({"lon": [1.0, 2.0], "lat": [3.0, 4.0]})
        arr = _ensure_point_geoarrow(points)
        assert "geoarrow.point" in getattr(arr.type, "extension_name", "")

    def test_ensure_point_geoarrow_arrow_table_wkb_column(self):
        wkb_points = pa.array(
            [
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?"
                b"\x00\x00\x00\x00\x00\x00\x00@",
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@"
                b"\x00\x00\x00\x00\x00\x00\x08@",
            ],
            type=pa.binary(),
        )
        points = pa.table({"geom_wkb": wkb_points})
        arr = _ensure_point_geoarrow(points, geometry_column="geom_wkb")
        assert "geoarrow.point" in getattr(arr.type, "extension_name", "")

    def test_ensure_point_geoarrow_dataframe_xy(self):
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        arr = _ensure_point_geoarrow(frame)
        assert "geoarrow.point" in getattr(arr.type, "extension_name", "")

    def test_get_collection_point_samples_latest(self):
        rows_old = [
            {
                "point_index": 0,
                "point_x": 1.0,
                "point_y": 2.0,
                "point_crs": 4326,
                "record_id": "old",
                "datetime": np.datetime64("2024-01-01T00:00:00"),
                "collection": "c",
                "cloud_cover": 1.0,
                "band": "B04",
                "value": 10.0,
                "raster_crs": 32632,
            }
        ]
        rows_new = [
            {
                "point_index": 0,
                "point_x": 1.0,
                "point_y": 2.0,
                "point_crs": 4326,
                "record_id": "new",
                "datetime": np.datetime64("2024-01-02T00:00:00"),
                "collection": "c",
                "cloud_cover": 1.0,
                "band": "B04",
                "value": 11.0,
                "raster_crs": 32632,
            }
        ]
        collection = self._DummyCollection(
            [self._DummyRaster(rows_old), self._DummyRaster(rows_new)]
        )
        table = get_collection_point_samples(
            collection=collection,
            points=pa.table({"lon": [1.0], "lat": [2.0]}),
            bands=["B04"],
            match="latest",
        )
        assert table.num_rows == 1
        assert table.column("record_id")[0].as_py() == "new"
        assert table.column("value")[0].as_py() == 11.0
        assert len(collection._rasters[0].calls) == 0
        assert len(collection._rasters[1].calls) == 1

    def test_get_collection_point_samples_groups_points_by_raster_bbox(self):
        raster_left = self._DummyRaster(
            [
                {
                    "point_index": 0,
                    "point_x": 1.0,
                    "point_y": 2.0,
                    "point_crs": 4326,
                    "record_id": "left",
                    "datetime": np.datetime64("2024-01-01T00:00:00"),
                    "collection": "c",
                    "cloud_cover": 1.0,
                    "band": "B04",
                    "value": 10.0,
                    "raster_crs": 32632,
                }
            ],
            raster_id="left",
            bbox=[0.0, 0.0, 2.0, 3.0],
        )
        raster_right = self._DummyRaster(
            [
                {
                    "point_index": 1,
                    "point_x": 10.0,
                    "point_y": 20.0,
                    "point_crs": 4326,
                    "record_id": "right",
                    "datetime": np.datetime64("2024-01-01T00:00:00"),
                    "collection": "c",
                    "cloud_cover": 1.0,
                    "band": "B04",
                    "value": 20.0,
                    "raster_crs": 32633,
                }
            ],
            raster_id="right",
            bbox=[9.0, 19.0, 11.0, 21.0],
        )
        raster_empty = self._DummyRaster(
            [],
            raster_id="empty",
            bbox=[100.0, 100.0, 101.0, 101.0],
        )
        collection = self._DummyCollection([raster_left, raster_right, raster_empty])

        table = get_collection_point_samples(
            collection=collection,
            points=pa.table({"lon": [1.0, 10.0], "lat": [2.0, 20.0]}),
            bands=["B04"],
        )

        assert table.num_rows == 2
        assert collection.last_filters == {"bbox": (1.0, 2.0, 10.0, 20.0)}
        assert raster_left.calls[0]["point_indices"] == [0]
        assert raster_right.calls[0]["point_indices"] == [1]
        assert raster_empty.calls == []

    def test_get_collection_point_samples_reuses_single_reader(self):
        class _StubReader:
            def __init__(self, *_args, **_kwargs):
                self.sem = None

            async def __aenter__(self):
                self.sem = object()
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        rows_a = [
            {
                "point_index": 0,
                "point_x": 1.0,
                "point_y": 2.0,
                "point_crs": 4326,
                "record_id": "a",
                "datetime": np.datetime64("2024-01-01T00:00:00"),
                "collection": "c",
                "cloud_cover": 1.0,
                "band": "B04",
                "value": 10.0,
                "raster_crs": 32632,
            }
        ]
        rows_b = [
            {
                "point_index": 1,
                "point_x": 10.0,
                "point_y": 20.0,
                "point_crs": 4326,
                "record_id": "b",
                "datetime": np.datetime64("2024-01-02T00:00:00"),
                "collection": "c",
                "cloud_cover": 2.0,
                "band": "B04",
                "value": 20.0,
                "raster_crs": 32633,
            }
        ]
        raster_a = self._DummyRaster(rows_a, raster_id="a", bbox=[0.0, 0.0, 2.0, 3.0])
        raster_b = self._DummyRaster(
            rows_b, raster_id="b", bbox=[9.0, 19.0, 11.0, 21.0]
        )
        collection = self._DummyCollection([raster_a, raster_b])

        with patch("rasteret.core.point_sampling.COGReader", _StubReader):
            table = get_collection_point_samples(
                collection=collection,
                points=pa.table({"lon": [1.0, 10.0], "lat": [2.0, 20.0]}),
                bands=["B04"],
            )

        assert table.num_rows == 2
        assert len(raster_a.calls) == 1
        assert len(raster_b.calls) == 1
        assert "reader" in raster_a.calls[0]
        assert raster_a.calls[0]["reader"] is raster_b.calls[0]["reader"]

    def test_get_collection_point_samples_empty_returns_schema(self):
        collection = self._DummyCollection([self._DummyRaster([])])
        with patch(
            "rasteret.core.execution._ensure_geoarrow", return_value=pa.array([])
        ):
            table = get_collection_point_samples(
                collection=collection,
                points=[],
                bands=["B04"],
            )
        assert table.num_rows == 0
        assert "point_index" in table.schema.names
        assert "value" in table.schema.names
