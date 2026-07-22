# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from affine import Affine

from rasteret.ingest.normalize import build_collection_from_table


def _collection_with_splits():
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2", "scene-3"]),
            "datetime": pa.array(
                [
                    datetime(2024, 1, 15),
                    datetime(2024, 1, 16),
                    datetime(2024, 1, 17),
                ],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/s1.tif"}},
                    {"B04": {"href": "https://example.com/s2.tif"}},
                    {"B04": {"href": "https://example.com/s3.tif"}},
                ]
            ),
            "split": pa.array(["train", "val", "test"]),
            "label": pa.array([0, 1, 2], type=pa.int64()),
        }
    )
    return build_collection_from_table(
        table,
        name="split-demo",
        description="Split demo collection",
        data_source="split-demo-source",
        date_range=("2024-01-15", "2024-01-17"),
    )


def _collection_with_bboxes():
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 16)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/s1.tif"}},
                    {"B04": {"href": "https://example.com/s2.tif"}},
                ]
            ),
            "bbox": pa.array(
                [
                    {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
                    {"xmin": 10.0, "ymin": 10.0, "xmax": 11.0, "ymax": 11.0},
                ],
                type=pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        }
    )
    return build_collection_from_table(table, name="bbox-demo")


def _collection_with_torchgeo_filters():
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2", "scene-3"]),
            "datetime": pa.array(
                [
                    datetime(2024, 1, 15),
                    datetime(2024, 2, 15),
                    datetime(2024, 3, 15),
                ],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/s1.tif"}},
                    {"B04": {"href": "https://example.com/s2.tif"}},
                    {"B04": {"href": "https://example.com/s3.tif"}},
                ]
            ),
            "bbox": pa.array(
                [
                    {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
                    {"xmin": 10.0, "ymin": 10.0, "xmax": 11.0, "ymax": 11.0},
                    {"xmin": 20.0, "ymin": 20.0, "xmax": 21.0, "ymax": 21.0},
                ],
                type=pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
            "eo:cloud_cover": pa.array([5.0, 40.0, 80.0], type=pa.float64()),
            "split": pa.array(["train", "val", "test"]),
        }
    )
    return build_collection_from_table(table, name="torchgeo-filter-demo")


def _collection_with_sampling_metadata():
    meta1 = {
        "transform": [10.0, 500000.0, -10.0, 1000000.0],
        "image_width": 128,
        "image_height": 128,
        "tile_width": 64,
        "tile_height": 64,
        "dtype": "uint16",
        "compress": 8,
        "predictor": 1,
        "tile_offsets": [0],
        "tile_byte_counts": [1],
    }
    meta2 = {
        **meta1,
        "transform": [10.0, 600000.0, -10.0, 1000000.0],
    }
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 16)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/s1.tif"}},
                    {"B04": {"href": "https://example.com/s2.tif"}},
                ]
            ),
            "proj:epsg": pa.array([32615, 32615], type=pa.int32()),
            "label": pa.array([10, 20], type=pa.int64()),
            "B04_metadata": pa.array([meta1, meta2]),
        }
    )
    return build_collection_from_table(table, name="sampling-demo")


def test_subset_single_split() -> None:
    collection = _collection_with_splits()
    filtered = collection.subset(split="train")
    ids = filtered.dataset.to_table(columns=["id"]).column("id").to_pylist()
    assert ids == ["scene-1"]


def test_subset_multi_split() -> None:
    collection = _collection_with_splits()
    filtered = collection.subset(split=("train", "val"))
    ids = filtered.dataset.to_table(columns=["id"]).column("id").to_pylist()
    assert ids == ["scene-1", "scene-2"]


def test_subset_split_matches_filter() -> None:
    collection = _collection_with_splits()
    filtered = collection.subset(split="test")
    ids = filtered.dataset.to_table(columns=["id"]).column("id").to_pylist()
    assert ids == ["scene-3"]


def test_subset_rejects_invalid_split_value() -> None:
    collection = _collection_with_splits()
    with pytest.raises(ValueError, match="Invalid split filter"):
        collection.subset(split=123)


def test_subset_rejects_missing_split_column() -> None:
    collection = _collection_with_splits()
    with pytest.raises(ValueError, match="no split column"):
        collection.subset(split="train", split_column="partition")


def test_select_split_is_convenience_wrapper() -> None:
    collection = _collection_with_splits()
    filtered = collection.select_split("val")
    ids = filtered.dataset.to_table(columns=["id"]).column("id").to_pylist()
    assert ids == ["scene-2"]


def test_subset_bbox_uses_bbox_struct() -> None:
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 16)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/s1.tif"}},
                    {"B04": {"href": "https://example.com/s2.tif"}},
                ]
            ),
            "bbox": pa.array(
                [
                    {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
                    {"xmin": 10.0, "ymin": 10.0, "xmax": 11.0, "ymax": 11.0},
                ],
                type=pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        }
    )
    collection = build_collection_from_table(table, name="bbox-demo")
    filtered = collection.subset(bbox=(0.5, 0.5, 2.0, 2.0))
    ids = filtered.dataset.to_table(columns=["id"]).column("id").to_pylist()
    assert ids == ["scene-1"]


def _collection_for_window_ordering():
    """2x2 COG metadata aligned with query bounds (0,0,2,2) at res=1, nodata=0."""
    meta = {
        "transform": [1.0, 0.0, -1.0, 2.0],
        "image_width": 2,
        "image_height": 2,
        "tile_width": 2,
        "tile_height": 2,
        "dtype": "uint16",
        "compress": 8,
        "predictor": 1,
        "tile_offsets": [0],
        "tile_byte_counts": [1],
        "nodata": 0.0,
    }
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 16)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/scene-1.tif"}},
                    {"B04": {"href": "https://example.com/scene-2.tif"}},
                ]
            ),
            "proj:epsg": pa.array([32615, 32615], type=pa.int32()),
            "B04_metadata": pa.array([meta, meta]),
        }
    )
    return build_collection_from_table(table, name="window-ordering-test")


def test_read_window_respects_requested_record_order(monkeypatch) -> None:
    """record_ids order controls mosaic priority: first ID wins at overlapping pixels."""
    collection = _collection_for_window_ordering()
    query_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)

    async def fake_read_cog(*args, **kwargs):
        url = args[0]
        if "scene-1" in url:
            data = np.array([[1, 0], [0, 0]], dtype=np.uint16)
        else:
            data = np.array([[2, 2], [2, 2]], dtype=np.uint16)
        return type("Result", (), {"data": data, "transform": query_transform})()

    monkeypatch.setattr("rasteret.core.window_read.read_cog", fake_read_cog)
    arr = collection.read_window(
        record_ids=["scene-1", "scene-2"],
        bounds=(0.0, 0.0, 2.0, 2.0),
        res=(1.0, 1.0),
        bands=["B04"],
    )

    assert arr.shape == (1, 2, 2)
    assert arr[0].tolist() == [[1, 2], [2, 2]]


def _fake_scene_reader(query_transform):
    async def fake_read_cog(*args, **kwargs):
        url = args[0]
        fill = 1 if "scene-1" in url or url.endswith("a.tif") else 2
        data = np.full((2, 2), fill, dtype=np.uint16)
        return type("Result", (), {"data": data, "transform": query_transform})()

    return fake_read_cog


def test_read_window_group_by_id_stacks_one_timestep_per_record(monkeypatch) -> None:
    """group_by='id' returns one timestep per record with no mosaicking, matching
    TorchGeo RasterDataset time_series=True (one T per file)."""
    collection = _collection_for_window_ordering()
    query_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    monkeypatch.setattr(
        "rasteret.core.window_read.read_cog", _fake_scene_reader(query_transform)
    )

    arr = collection.read_window(
        record_ids=["scene-1", "scene-2"],
        bounds=(0.0, 0.0, 2.0, 2.0),
        res=(1.0, 1.0),
        bands=["B04"],
        group_by="id",
    )

    assert arr.shape == (2, 1, 2, 2)  # [T, C, H, W], one T per record
    assert arr[0, 0].tolist() == [[1, 1], [1, 1]]
    assert arr[1, 0].tolist() == [[2, 2], [2, 2]]


def test_read_window_group_by_id_keeps_same_date_records_separate(monkeypatch) -> None:
    """Two records sharing a datetime: 'datetime' mosaics them into one timestep,
    'id' keeps them as separate timesteps (the TorchGeo-compatible behavior)."""
    meta = {
        "transform": [1.0, 0.0, -1.0, 2.0],
        "image_width": 2,
        "image_height": 2,
        "tile_width": 2,
        "tile_height": 2,
        "dtype": "uint16",
        "compress": 8,
        "predictor": 1,
        "tile_offsets": [0],
        "tile_byte_counts": [1],
        "nodata": 0.0,
    }
    table = pa.table(
        {
            "id": pa.array(["a", "b"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 15)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/a.tif"}},
                    {"B04": {"href": "https://example.com/b.tif"}},
                ]
            ),
            "proj:epsg": pa.array([32615, 32615], type=pa.int32()),
            "B04_metadata": pa.array([meta, meta]),
        }
    )
    collection = build_collection_from_table(table, name="same-date-test")
    query_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    monkeypatch.setattr(
        "rasteret.core.window_read.read_cog", _fake_scene_reader(query_transform)
    )
    kwargs = dict(
        record_ids=["a", "b"],
        bounds=(0.0, 0.0, 2.0, 2.0),
        res=(1.0, 1.0),
        bands=["B04"],
    )

    by_datetime = collection.read_window(**kwargs, group_by="datetime")
    by_id = collection.read_window(**kwargs, group_by="id")

    assert by_datetime.shape == (1, 1, 2, 2)  # same date -> one mosaicked timestep
    assert by_id.shape == (2, 1, 2, 2)  # one timestep per record


def test_read_window_reuses_reader_pool_across_calls(monkeypatch) -> None:
    """The reader pool is created once and reused across read_window calls."""
    collection = _collection_with_sampling_metadata()
    query_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    created_pools: list[tuple[int, object | None]] = []
    seen_readers: list[int] = []

    class FakePool:
        def __init__(self, *, max_concurrent: int, backend=None) -> None:
            created_pools.append((max_concurrent, backend))
            self.reader = object()

        def run(self, coro):
            return asyncio.run(coro)

        def close(self) -> None:
            return None

    async def fake_read_cog(*args, **kwargs):
        seen_readers.append(id(kwargs["reader"]))
        return type(
            "Result",
            (),
            {
                "data": np.array([[1, 1], [1, 1]], dtype=np.uint16),
                "transform": query_transform,
            },
        )()

    monkeypatch.setattr("rasteret.core.collection.AsyncCOGReaderPool", FakePool)
    monkeypatch.setattr("rasteret.core.window_read.read_cog", fake_read_cog)

    for _ in range(2):
        arr = collection.read_window(
            record_ids=["scene-1"],
            bounds=(0.0, 0.0, 2.0, 2.0),
            res=(1.0, 1.0),
            bands=["B04"],
        )
        assert arr.shape == (1, 2, 2)

    assert len(created_pools) == 1
    assert len(seen_readers) == 2
    assert seen_readers[0] == seen_readers[1]


def test_subset_preserves_collection_metadata() -> None:
    collection = _collection_with_splits()
    filtered = collection.subset(split="train")
    assert filtered.name == collection.name
    assert filtered.description == collection.description
    assert filtered.data_source == collection.data_source
    assert filtered.start_date == collection.start_date
    assert filtered.end_date == collection.end_date


def test_where_preserves_collection_metadata() -> None:
    collection = _collection_with_splits()
    filtered = collection.where(ds.field("split") == "val")
    assert filtered.data_source == collection.data_source


# ---------------------------------------------------------------------------
# Collection.footprints()
# ---------------------------------------------------------------------------


def _collection_for_footprints():
    """Two records in EPSG:32615 (UTM 15N): one tile at (0,0), one at (20,0).

    Each tile is 10 cols x 10 rows at res=2.0, so each covers a 20x20 box
    in native CRS. ``geometry`` is the WGS84 lon/lat polygon of those boxes
    (precomputed for the test); the point is to confirm ``footprints()``
    uses band metadata, not this column.
    """
    meta_a = {
        "transform": [2.0, 0.0, -2.0, 20.0],  # sx=2, tx=0, sy=-2, ty=20
        "image_width": 10,
        "image_height": 10,
        "tile_width": 10,
        "tile_height": 10,
        "dtype": "uint16",
        "compress": 8,
        "predictor": 1,
        "tile_offsets": [0],
        "tile_byte_counts": [1],
        "nodata": 0.0,
    }
    meta_b = {**meta_a, "transform": [2.0, 20.0, -2.0, 20.0]}  # shifted +20 in x
    table = pa.table(
        {
            "id": pa.array(["scene-a", "scene-b"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 16)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/a.tif"}},
                    {"B04": {"href": "https://example.com/b.tif"}},
                ]
            ),
            "proj:epsg": pa.array([32615, 32615], type=pa.int32()),
            "B04_metadata": pa.array([meta_a, meta_b]),
        }
    )
    return build_collection_from_table(table, name="footprints-test")


def test_footprints_uses_band_metadata_not_geometry_column() -> None:
    collection = _collection_for_footprints()
    gdf = collection.footprints(crs=32615)
    assert list(gdf.columns) == ["id", "datetime", "geometry"]
    assert gdf.crs.to_epsg() == 32615
    assert gdf["id"].tolist() == ["scene-a", "scene-b"]
    # Exact bounds derived from transform + width*height, not from any
    # WGS84-to-UTM reprojection.
    assert gdf.geometry.iloc[0].bounds == (0.0, 0.0, 20.0, 20.0)
    assert gdf.geometry.iloc[1].bounds == (20.0, 0.0, 40.0, 20.0)


def test_footprints_defaults_to_first_band_when_unspecified() -> None:
    collection = _collection_for_footprints()
    assert collection.bands == ["B04"]
    gdf = collection.footprints()
    # No crs given, all records single-CRS → tagged with native.
    assert gdf.crs.to_epsg() == 32615
    assert len(gdf) == 2


def test_footprints_reprojects_when_target_crs_differs() -> None:
    collection = _collection_for_footprints()
    gdf_native = collection.footprints(crs=32615)
    gdf_wgs84 = collection.footprints(crs=4326)
    assert gdf_wgs84.crs.to_epsg() == 4326
    # Reprojection should change the bounds.
    assert gdf_native.geometry.iloc[0].bounds != gdf_wgs84.geometry.iloc[0].bounds
    # Sanity-check the WGS84 bounds are in a plausible range (UTM 15N x=0
    # is west of the central meridian at -93°E, near -97.5°E at the equator).
    minx, miny, maxx, maxy = gdf_wgs84.geometry.iloc[0].bounds
    assert -180.0 < minx < maxx < 0.0
    assert 0.0 <= miny < maxy < 1.0


def test_footprints_skips_rows_with_null_metadata() -> None:
    """Rows with null band_metadata are dropped rather than raising."""
    meta = {
        "transform": [2.0, 0.0, -2.0, 20.0],
        "image_width": 10,
        "image_height": 10,
        "tile_width": 10,
        "tile_height": 10,
        "dtype": "uint16",
        "compress": 8,
        "predictor": 1,
        "tile_offsets": [0],
        "tile_byte_counts": [1],
        "nodata": 0.0,
    }
    table = pa.table(
        {
            "id": pa.array(["good", "bad"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 16)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/a.tif"}},
                    {"B04": {"href": "https://example.com/b.tif"}},
                ]
            ),
            "proj:epsg": pa.array([32615, 32615], type=pa.int32()),
            "B04_metadata": pa.array([meta, None]),
        }
    )
    collection = build_collection_from_table(table, name="footprints-null-meta")
    gdf = collection.footprints(crs=32615)
    assert gdf["id"].tolist() == ["good"]


def test_footprints_raises_on_unknown_band() -> None:
    collection = _collection_for_footprints()
    with pytest.raises(ValueError, match="Band"):
        collection.footprints(band="B99")
