# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for the ingest abstraction layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import tifffile as tf

from rasteret.core.collection import Collection
from rasteret.ingest.enrich import (
    add_band_metadata_columns,
    build_url_index_from_assets,
    enrich_table_with_cog_metadata,
)
from rasteret.ingest.normalize import build_collection_from_table
from rasteret.ingest.parquet_record_table import RecordTableBuilder

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _minimal_table(**overrides: pa.Array) -> pa.Table:
    """Return a minimal valid Arrow table with Collection contract columns."""
    defaults = {
        "id": pa.array(["scene-1", "scene-2"]),
        "datetime": pa.array(
            [datetime(2024, 1, 15), datetime(2024, 3, 20)],
            type=pa.timestamp("us"),
        ),
        "geometry": pa.array([None, None], type=pa.null()),
        "assets": pa.array(
            [
                {"B04": {"href": "https://example.com/s1.tif"}},
                {"B04": {"href": "https://example.com/s2.tif"}},
            ]
        ),
    }
    defaults.update(overrides)
    return pa.table(defaults)


def _write_manifest(path: Path, table: pa.Table | None = None) -> None:
    """Write a Parquet manifest from a table (default: minimal valid table)."""
    if table is None:
        table = _minimal_table()
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# build_collection_from_table
# ---------------------------------------------------------------------------


class TestBuildCollectionFromTable:
    def test_valid_table_returns_collection(self):
        table = _minimal_table()
        collection = build_collection_from_table(table, name="test")
        assert isinstance(collection, Collection)
        assert collection.name == "test"

    def test_missing_required_columns_raises(self):
        table = pa.table({"id": pa.array(["s1"]), "foo": pa.array([1])})
        with pytest.raises(ValueError, match="missing required columns"):
            build_collection_from_table(table)

    def test_auto_adds_year_month(self):
        table = _minimal_table()
        assert "year" not in table.schema.names
        assert "month" not in table.schema.names

        collection = build_collection_from_table(table)
        out = collection.dataset.to_table()
        assert "year" in out.schema.names
        assert "month" in out.schema.names
        assert out.column("year").to_pylist() == [2024, 2024]
        assert out.column("month").to_pylist() == [1, 3]

    def test_auto_adds_bbox(self):
        table = _minimal_table()
        assert "bbox" not in table.schema.names

        collection = build_collection_from_table(table)
        out = collection.dataset.to_table()
        assert "bbox" in out.schema.names

    def test_preserves_existing_year_month(self):
        table = _minimal_table()
        table = table.append_column("year", pa.array([2025, 2025], type=pa.int64()))
        table = table.append_column("month", pa.array([12, 12], type=pa.int64()))

        collection = build_collection_from_table(table)
        out = collection.dataset.to_table()
        assert out.column("year").to_pylist() == [2025, 2025]

    def test_date_range_sets_start_end(self):
        table = _minimal_table()
        collection = build_collection_from_table(
            table, date_range=("2024-01-01", "2024-06-30")
        )
        assert collection.start_date == datetime(2024, 1, 1)
        assert collection.end_date == datetime(2024, 6, 30)

    def test_workspace_dir_materialises(self, tmp_path):
        table = _minimal_table()
        out_dir = tmp_path / "output"
        build_collection_from_table(table, workspace_dir=out_dir)
        assert out_dir.exists()


# ---------------------------------------------------------------------------
# RecordTableBuilder
# ---------------------------------------------------------------------------


class TestRecordTableBuilder:
    def test_load_valid_manifest(self, tmp_path):
        path = tmp_path / "manifest.parquet"
        _write_manifest(path)
        builder = RecordTableBuilder(path)
        collection = builder.build()
        assert isinstance(collection, Collection)
        assert collection.name == "manifest"

    def test_workspace_dir_cloud_uri_not_mangled(self, tmp_path, monkeypatch):
        path = tmp_path / "manifest.parquet"
        _write_manifest(path)
        captured: dict[str, object] = {}

        def _fake_build_collection_from_table(table: pa.Table, **kwargs):
            captured["workspace_dir"] = kwargs.get("workspace_dir")
            return Collection(dataset=ds.dataset(table), name="manifest")

        monkeypatch.setattr(
            "rasteret.ingest.parquet_record_table.build_collection_from_table",
            _fake_build_collection_from_table,
        )

        builder = RecordTableBuilder(
            path,
            workspace_dir="s3://demo-bucket/workspace_records",
        )
        builder.build()
        assert captured["workspace_dir"] == "s3://demo-bucket/workspace_records"

    def test_column_remapping(self, tmp_path):
        table = pa.table(
            {
                "scene_id": pa.array(["s1"]),
                "timestamp": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
                "geometry": pa.array([None], type=pa.null()),
                "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
            }
        )
        path = tmp_path / "mapped.parquet"
        pq.write_table(table, str(path))

        builder = RecordTableBuilder(
            path, column_map={"scene_id": "id", "timestamp": "datetime"}
        )
        collection = builder.build()
        assert isinstance(collection, Collection)
        out = collection.dataset.to_table()
        # Column map acts as an alias map: preserve source columns.
        assert "scene_id" in out.schema.names
        assert "timestamp" in out.schema.names
        assert "id" in out.schema.names
        assert "datetime" in out.schema.names

    def test_missing_file_raises(self):
        builder = RecordTableBuilder(Path("/nonexistent/file.parquet"))
        with pytest.raises((FileNotFoundError, OSError)):
            builder.build()

    def test_missing_columns_after_remap_raises(self, tmp_path):
        table = pa.table({"foo": pa.array([1]), "bar": pa.array([2])})
        path = tmp_path / "bad.parquet"
        pq.write_table(table, str(path))

        builder = RecordTableBuilder(path, column_map={"foo": "id"})
        with pytest.raises(ValueError, match="missing required columns"):
            builder.build()

    def test_filter_expr_pushdown(self, tmp_path):
        path = tmp_path / "manifest.parquet"
        _write_manifest(path)
        builder = RecordTableBuilder(path, filter_expr=ds.field("id") == "scene-1")
        collection = builder.build()
        ids = collection.dataset.to_table(columns=["id"]).column("id").to_pylist()
        assert ids == ["scene-1"]


# ---------------------------------------------------------------------------
# RecordTableBuilder._prepare_table normalisation
# ---------------------------------------------------------------------------


class TestPrepareTable:
    """Tests for automatic type coercions and assets construction."""

    def test_integer_id_cast_to_string(self, tmp_path):
        table = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "datetime": pa.array(
                    [datetime(2024, 1, 1), datetime(2024, 6, 1)],
                    type=pa.timestamp("us"),
                ),
                "geometry": pa.array([None, None], type=pa.null()),
                "assets": pa.array(
                    [
                        {"B04": {"href": "https://example.com/a.tif"}},
                        {"B04": {"href": "https://example.com/b.tif"}},
                    ]
                ),
            }
        )
        path = tmp_path / "int_id.parquet"
        pq.write_table(table, str(path))
        builder = RecordTableBuilder(path)
        collection = builder.build()
        ids = collection.dataset.to_table(columns=["id"]).column("id").to_pylist()
        assert ids == ["1", "2"]
        assert pa.types.is_string(collection.dataset.schema.field("id").type)

    def test_integer_year_coerced_to_timestamp(self, tmp_path):
        table = pa.table(
            {
                "id": pa.array(["s1", "s2"]),
                "datetime": pa.array([2023, 2024], type=pa.int64()),
                "geometry": pa.array([None, None], type=pa.null()),
                "assets": pa.array(
                    [
                        {"B04": {"href": "https://example.com/a.tif"}},
                        {"B04": {"href": "https://example.com/b.tif"}},
                    ]
                ),
            }
        )
        path = tmp_path / "year_dt.parquet"
        pq.write_table(table, str(path))
        builder = RecordTableBuilder(path)
        collection = builder.build()
        out = collection.dataset.to_table(columns=["datetime"])
        assert pa.types.is_timestamp(out.schema.field("datetime").type)
        dt_values = out.column("datetime").to_pylist()
        assert dt_values[0].year == 2023
        assert dt_values[1].year == 2024

    def test_assets_constructed_from_href_column_and_band_index_map(self, tmp_path):
        table = pa.table(
            {
                "id": pa.array(["s1"]),
                "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
                "geometry": pa.array([None], type=pa.null()),
                "url": pa.array(["https://example.com/tile.tif"]),
            }
        )
        path = tmp_path / "href.parquet"
        pq.write_table(table, str(path))
        builder = RecordTableBuilder(
            path,
            href_column="url",
            band_index_map={"A0": 0, "A1": 1},
        )
        collection = builder.build()
        row = collection.dataset.to_table(columns=["assets"]).to_pylist()[0]
        assert "A0" in row["assets"]
        assert "A1" in row["assets"]
        assert row["assets"]["A0"]["href"] == "https://example.com/tile.tif"
        assert row["assets"]["A0"]["band_index"] == 0
        assert row["assets"]["A1"]["band_index"] == 1

    def test_url_rewrite_patterns_applied(self, tmp_path):
        table = pa.table(
            {
                "id": pa.array(["s1"]),
                "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
                "geometry": pa.array([None], type=pa.null()),
                "path": pa.array(["s3://my-bucket/tile.tif"]),
            }
        )
        path = tmp_path / "rewrite.parquet"
        pq.write_table(table, str(path))
        builder = RecordTableBuilder(
            path,
            href_column="path",
            band_index_map={"B0": 0},
            url_rewrite_patterns={"s3://my-bucket/": "https://cdn.example.com/"},
        )
        collection = builder.build()
        row = collection.dataset.to_table(columns=["assets"]).to_pylist()[0]
        assert row["assets"]["B0"]["href"] == "https://cdn.example.com/tile.tif"

    def test_proj_epsg_derived_from_crs_column(self, tmp_path):
        table = pa.table(
            {
                "id": pa.array(["s1"]),
                "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
                "geometry": pa.array([None], type=pa.null()),
                "assets": pa.array([{"B04": {"href": "https://example.com/a.tif"}}]),
                "crs": pa.array(["EPSG:32632"]),
            }
        )
        path = tmp_path / "crs.parquet"
        pq.write_table(table, str(path))
        builder = RecordTableBuilder(path)
        collection = builder.build()
        out = collection.dataset.to_table(columns=["proj:epsg", "crs"])
        assert out.column("proj:epsg").to_pylist() == [32632]
        assert out.column("crs").to_pylist() == ["EPSG:32632"]

    def test_missing_href_column_raises(self, tmp_path):
        table = pa.table(
            {
                "id": pa.array(["s1"]),
                "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
                "geometry": pa.array([None], type=pa.null()),
            }
        )
        path = tmp_path / "no_href.parquet"
        pq.write_table(table, str(path))
        builder = RecordTableBuilder(
            path,
            href_column="nonexistent",
            band_index_map={"A0": 0},
        )
        with pytest.raises(ValueError, match="href_column.*not found"):
            builder.build()

    def test_column_map_plus_prepare_table_full_pipeline(self, tmp_path):
        """Column map aliases + _prepare_table coercions work together."""
        table = pa.table(
            {
                "fid": pa.array([42], type=pa.int64()),
                "year": pa.array([2023], type=pa.int64()),
                "geom": pa.array([None], type=pa.null()),
                "url": pa.array(["https://example.com/tile.tif"]),
                "crs": pa.array(["EPSG:32632"]),
            }
        )
        path = tmp_path / "pipeline.parquet"
        pq.write_table(table, str(path))
        builder = RecordTableBuilder(
            path,
            column_map={"fid": "id", "geom": "geometry", "year": "datetime"},
            href_column="url",
            band_index_map={"A0": 0, "A1": 1},
        )
        collection = builder.build()
        out = collection.dataset.to_table()
        # Aliases preserve source columns
        assert "fid" in out.schema.names
        assert "geom" in out.schema.names
        assert "year" in out.schema.names
        # Contract columns present with correct types
        assert pa.types.is_string(out.schema.field("id").type)
        assert pa.types.is_timestamp(out.schema.field("datetime").type)
        assert "assets" in out.schema.names
        assert "proj:epsg" in out.schema.names
        assert "crs" in out.schema.names
        assert out.column("id").to_pylist() == ["42"]
        assert out.column("proj:epsg").to_pylist() == [32632]
        assert out.column("crs").to_pylist() == ["EPSG:32632"]


# ---------------------------------------------------------------------------
# build_from_table convenience function
# ---------------------------------------------------------------------------


class TestBuildFromTable:
    def test_build_from_table_top_level(self, tmp_path):
        import rasteret

        path = tmp_path / "demo.parquet"
        _write_manifest(path)
        collection = rasteret.build_from_table(path)
        assert isinstance(collection, Collection)

    def test_build_from_table_accepts_pyarrow_table(self):
        import rasteret

        table = _minimal_table()
        collection = rasteret.build_from_table(table)
        assert isinstance(collection, Collection)

    def test_build_from_table_accepts_arrow_capsule_stream_object(self):
        import rasteret

        table = _minimal_table()

        class _StreamObj:
            def __arrow_c_stream__(self, requested_schema=None):
                return table.to_reader().__arrow_c_stream__(requested_schema)

        collection = rasteret.build_from_table(_StreamObj())
        assert isinstance(collection, Collection)
        assert collection.dataset is not None
        assert collection.dataset.count_rows() == table.num_rows

    def test_build_from_table_accepts_arrow_capsule_array_object(self):
        import rasteret

        batch = _minimal_table().to_batches()[0]

        class _ArrayObj:
            def __arrow_c_array__(self, requested_schema=None):
                return batch.__arrow_c_array__(requested_schema)

        collection = rasteret.build_from_table(_ArrayObj())
        assert isinstance(collection, Collection)
        assert collection.dataset is not None
        assert collection.dataset.count_rows() == batch.num_rows

    def test_build_from_table_workspace_dir_persists(self, tmp_path):
        import rasteret

        manifest = tmp_path / "demo.parquet"
        _write_manifest(manifest)
        out_dir = tmp_path / "out_collection"

        collection = rasteret.build_from_table(
            manifest,
            name="demo",
            workspace_dir=out_dir,
        )
        assert isinstance(collection, Collection)
        # workspace_dir gets _records suffix for discoverability
        expected = out_dir / "demo_records"
        assert expected.exists()
        reloaded = Collection._load_cached(expected)
        assert reloaded.dataset is not None
        assert reloaded.dataset.count_rows() == collection.dataset.count_rows()

    def test_build_from_table_in_dir(self):
        import rasteret

        assert "build_from_table" in dir(rasteret)

    def test_build_from_table_with_filter_expr(self, tmp_path):
        import rasteret

        path = tmp_path / "demo.parquet"
        _write_manifest(path)
        collection = rasteret.build_from_table(
            path, filter_expr=ds.field("id") == "scene-2"
        )
        ids = collection.dataset.to_table(columns=["id"]).column("id").to_pylist()
        assert ids == ["scene-2"]

    def test_build_from_table_hf_uri_uses_hf_reader(self):
        import rasteret

        table = _minimal_table()
        filter_expr = ds.field("id") == "scene-1"
        captured: dict[str, object] = {}

        def _fake_load_hf_table(path: str, *, columns, filter_expr):
            captured["path"] = path
            captured["columns"] = columns
            captured["filter_expr"] = filter_expr
            return table

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "rasteret.ingest.parquet_record_table.load_hf_parquet_table",
                _fake_load_hf_table,
            )
            collection = rasteret.build_from_table(
                "hf://datasets/terrafloww/example/index.parquet",
                columns=["id", "datetime", "geometry", "assets"],
                filter_expr=filter_expr,
            )

        assert isinstance(collection, Collection)
        assert captured["path"] == "hf://datasets/terrafloww/example/index.parquet"
        assert captured["columns"] == ["id", "datetime", "geometry", "assets"]
        assert captured["filter_expr"] is filter_expr


# ---------------------------------------------------------------------------
# COG enrichment helpers
# ---------------------------------------------------------------------------


class TestBuildUrlIndex:
    def test_extracts_urls_from_assets(self):
        table = _minimal_table()
        url_index = build_url_index_from_assets(table)
        assert url_index == {
            "scene-1": {"B04": {"url": "https://example.com/s1.tif", "band_index": 0}},
            "scene-2": {"B04": {"url": "https://example.com/s2.tif", "band_index": 0}},
        }

    def test_filters_by_band_codes(self):
        table = _minimal_table(
            assets=pa.array(
                [
                    {
                        "B04": {"href": "https://example.com/s1_red.tif"},
                        "B08": {"href": "https://example.com/s1_nir.tif"},
                    },
                    {
                        "B04": {"href": "https://example.com/s2_red.tif"},
                        "B08": {"href": "https://example.com/s2_nir.tif"},
                    },
                ]
            )
        )
        url_index = build_url_index_from_assets(table, band_codes=["B08"])
        assert url_index == {
            "scene-1": {
                "B08": {"url": "https://example.com/s1_nir.tif", "band_index": 0}
            },
            "scene-2": {
                "B08": {"url": "https://example.com/s2_nir.tif", "band_index": 0}
            },
        }

    def test_empty_assets_returns_empty_index(self):
        table = _minimal_table(assets=pa.array([None, None]))
        url_index = build_url_index_from_assets(table)
        assert url_index == {}

    def test_extracts_band_index(self):
        table = _minimal_table(
            assets=pa.array(
                [
                    {"A0": {"href": "https://example.com/a.tif", "band_index": 7}},
                    {"A0": {"href": "https://example.com/b.tif"}},
                ]
            )
        )
        url_index = build_url_index_from_assets(table)
        assert url_index == {
            "scene-1": {"A0": {"url": "https://example.com/a.tif", "band_index": 7}},
            "scene-2": {"A0": {"url": "https://example.com/b.tif", "band_index": 0}},
        }


class TestAddBandMetadataColumns:
    def test_adds_struct_columns(self):
        table = _minimal_table()
        processed_items = [
            {
                "record_id": "scene-1",
                "band": "B04",
                "width": 10980,
                "height": 10980,
                "tile_width": 512,
                "tile_height": 512,
                "dtype": "uint16",
                "transform": (10.0, 600000.0, -10.0, 1400000.0),
                "predictor": 1,
                "compression": 8,
                "tile_offsets": [100, 200],
                "tile_byte_counts": [50, 50],
                "pixel_scale": (10.0, 10.0, 0.0),
                "tiepoint": (0.0, 0.0, 0.0, 600000.0, 1400000.0, 0.0),
            },
        ]
        result = add_band_metadata_columns(table, ["B04"], processed_items)
        assert "B04_metadata" in result.schema.names

        meta_col = result.column("B04_metadata").to_pylist()
        assert meta_col[0]["image_width"] == 10980
        assert meta_col[0]["tile_width"] == 512
        assert meta_col[1] is None  # scene-2 had no processed item

    def test_handles_multiple_bands(self):
        table = _minimal_table()
        processed_items = [
            {
                "record_id": "scene-1",
                "band": "B04",
                "width": 100,
                "height": 100,
                "tile_width": 64,
                "tile_height": 64,
                "dtype": "uint16",
                "predictor": 1,
                "compression": 8,
                "tile_offsets": [100],
                "tile_byte_counts": [50],
            },
            {
                "record_id": "scene-1",
                "band": "B08",
                "width": 100,
                "height": 100,
                "tile_width": 64,
                "tile_height": 64,
                "dtype": "uint16",
                "predictor": 1,
                "compression": 8,
                "tile_offsets": [200],
                "tile_byte_counts": [60],
            },
        ]
        result = add_band_metadata_columns(table, ["B04", "B08"], processed_items)
        assert "B04_metadata" in result.schema.names
        assert "B08_metadata" in result.schema.names


@pytest.mark.asyncio
async def test_enrich_slices_tile_tables_for_planar_separate_multisample(tmp_path):
    """Planar separate multi-sample TIFFs store tile tables for all samples.

    Rasteret's enrichment should slice TileOffsets/TileByteCounts per band_index
    so each {band}_metadata column contains a single-sample layout.
    """
    import numpy as np

    class LocalFileBackend:
        async def get_range(self, url: str, start: int, length: int) -> bytes:
            with open(url, "rb") as f:
                f.seek(start)
                return f.read(length)

    path = tmp_path / "planar_separate.tif"
    data = np.zeros((2, 128, 128), dtype=np.int8)
    extratags = [
        (33550, "d", 3, (1.0, 1.0, 0.0), False),  # ModelPixelScaleTag
        (33922, "d", 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), False),  # ModelTiepointTag
    ]
    tf.imwrite(
        path,
        data,
        tile=(64, 64),
        planarconfig="separate",
        extratags=extratags,
    )

    with tf.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.is_tiled
        offsets = list(page.dataoffsets)
        counts = list(page.databytecounts)
        tiles_x = (page.imagewidth + page.tilewidth - 1) // page.tilewidth
        tiles_y = (page.imagelength + page.tilelength - 1) // page.tilelength
        tiles_per_plane = tiles_x * tiles_y
        assert len(offsets) == tiles_per_plane * 2
        assert len(counts) == tiles_per_plane * 2

    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array(
                [
                    {
                        "A0": {"href": str(path), "band_index": 0},
                        "A1": {"href": str(path), "band_index": 1},
                    }
                ]
            ),
        }
    )

    url_index = build_url_index_from_assets(table)
    enriched = await enrich_table_with_cog_metadata(
        table,
        url_index,
        ["A0", "A1"],
        max_concurrent=1,
        batch_size=10,
        backend=LocalFileBackend(),
    )

    m0 = enriched.column("A0_metadata").to_pylist()[0]
    m1 = enriched.column("A1_metadata").to_pylist()[0]
    assert m0 is not None and m1 is not None

    assert m0["tile_offsets"] == offsets[:tiles_per_plane]
    assert m1["tile_offsets"] == offsets[tiles_per_plane : 2 * tiles_per_plane]
    assert m0["tile_byte_counts"] == counts[:tiles_per_plane]
    assert m1["tile_byte_counts"] == counts[tiles_per_plane : 2 * tiles_per_plane]


@pytest.mark.asyncio
async def test_enrich_backfills_proj_epsg_when_missing(monkeypatch):
    """Enrichment should backfill proj:epsg from parsed GeoTIFF CRS when possible."""

    class _Meta:
        width = 128
        height = 128
        tile_width = 64
        tile_height = 64
        dtype = "uint16"
        transform = (10.0, 0.0, 10.0, 0.0)
        predictor = 1
        compression = 8
        tile_offsets = [100, 200, 300, 400]
        tile_byte_counts = [10, 10, 10, 10]
        pixel_scale = None
        tiepoint = None
        crs = 32632
        nodata = None
        samples_per_pixel = 1
        planar_configuration = 1
        photometric = None
        extra_samples = None

    class _Parser:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, D401
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        async def process_cog_headers_batch(self, urls):  # noqa: ANN001
            return [_Meta() for _ in urls]

    monkeypatch.setattr("rasteret.ingest.enrich.AsyncCOGHeaderParser", _Parser)

    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"A0": {"href": "https://example.com/a.tif"}}]),
        }
    )

    url_index = build_url_index_from_assets(table)
    enriched = await enrich_table_with_cog_metadata(table, url_index, ["A0"])

    assert "proj:epsg" in enriched.schema.names
    assert "crs" in enriched.schema.names
    assert enriched.column("proj:epsg").to_pylist() == [32632]
    assert enriched.column("crs").to_pylist() == ["EPSG:32632"]


@pytest.mark.asyncio
async def test_enrich_keeps_crs_sidecars_consistent_when_proj_epsg_exists(monkeypatch):
    """If existing proj:epsg is preserved, crs should match that resolved value."""

    class _Meta:
        width = 128
        height = 128
        tile_width = 64
        tile_height = 64
        dtype = "uint16"
        transform = (10.0, 0.0, 10.0, 0.0)
        predictor = 1
        compression = 8
        tile_offsets = [100, 200, 300, 400]
        tile_byte_counts = [10, 10, 10, 10]
        pixel_scale = None
        tiepoint = None
        crs = 32632
        nodata = None
        samples_per_pixel = 1
        planar_configuration = 1
        photometric = None
        extra_samples = None

    class _Parser:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, D401
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        async def process_cog_headers_batch(self, urls):  # noqa: ANN001
            return [_Meta() for _ in urls]

    monkeypatch.setattr("rasteret.ingest.enrich.AsyncCOGHeaderParser", _Parser)

    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"A0": {"href": "https://example.com/a.tif"}}]),
            "proj:epsg": pa.array([32633], type=pa.int32()),
            "crs": pa.array(["EPSG:32633"], type=pa.string()),
        }
    )

    url_index = build_url_index_from_assets(table)
    enriched = await enrich_table_with_cog_metadata(table, url_index, ["A0"])

    assert enriched.column("proj:epsg").to_pylist() == [32633]
    assert enriched.column("crs").to_pylist() == ["EPSG:32633"]


class TestRecordTableBuilderEnrichCog:
    def test_enrich_cog_false_by_default(self, tmp_path):
        path = tmp_path / "manifest.parquet"
        _write_manifest(path)
        builder = RecordTableBuilder(path)
        assert not builder.enrich_cog

    def test_enrich_cog_params_pass_through(self):
        builder = RecordTableBuilder(
            "/tmp/fake.parquet",
            enrich_cog=True,
            band_codes=["B04"],
            max_concurrent=50,
        )
        assert builder.enrich_cog
        assert builder.band_codes == ["B04"]
        assert builder.max_concurrent == 50
