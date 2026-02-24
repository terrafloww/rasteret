# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

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


def test_subset_bbox_uses_scalar_bbox_columns() -> None:
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
            "scene_bbox": pa.array(
                [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]],
                type=pa.list_(pa.float64(), 4),
            ),
        }
    )
    collection = build_collection_from_table(table, name="bbox-demo")
    filtered = collection.subset(bbox=(0.5, 0.5, 2.0, 2.0))
    ids = filtered.dataset.to_table(columns=["id"]).column("id").to_pylist()
    assert ids == ["scene-1"]


def test_to_torchgeo_dataset_applies_split(monkeypatch) -> None:
    pytest.importorskip("torchgeo")
    import rasteret.integrations.torchgeo as torchgeo_adapter

    collection = _collection_with_splits()
    captured: dict[str, list[str]] = {}

    class DummyGeoDataset:
        def __init__(self, *, collection, **kwargs):
            _ = kwargs
            captured["ids"] = (
                collection.dataset.to_table(columns=["id"]).column("id").to_pylist()
            )

    monkeypatch.setattr(torchgeo_adapter, "RasteretGeoDataset", DummyGeoDataset)
    _ = collection.to_torchgeo_dataset(bands=["B04"], split="val")
    assert captured["ids"] == ["scene-2"]


def test_to_torchgeo_dataset_forwards_label_field(monkeypatch) -> None:
    pytest.importorskip("torchgeo")
    import rasteret.integrations.torchgeo as torchgeo_adapter

    collection = _collection_with_splits()
    captured: dict[str, object] = {}

    class DummyGeoDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(torchgeo_adapter, "RasteretGeoDataset", DummyGeoDataset)
    _ = collection.to_torchgeo_dataset(
        bands=["B04"], split="train", label_field="label"
    )
    assert captured["label_field"] == "label"


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
