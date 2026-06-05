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
