# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import io
import sys
import types
from datetime import datetime

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from rasteret.core.collection import Collection
from rasteret.integrations.huggingface import (
    HFStreamingSource,
    head_hf_streaming_source,
    load_hf_parquet_table,
    open_hf_streaming_source,
    parse_hf_dataset_uri,
    resolve_hf_parquet_paths,
    subset_hf_streaming_source,
)


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


class _FakeHfFs:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def open(self, path: str, mode: str = "rb"):
        assert mode == "rb"
        return io.BytesIO(self.files[path])


def test_parse_hf_dataset_uri() -> None:
    repo_id, subpath, revision = parse_hf_dataset_uri(
        "hf://datasets/terrafloww/rasteret-collection@main/data/index.parquet"
    )
    assert repo_id == "terrafloww/rasteret-collection"
    assert revision == "main"
    assert subpath == "data/index.parquet"


def test_resolve_hf_parquet_paths_discovers_repo_files(monkeypatch) -> None:
    class _FakeApi:
        def list_repo_files(self, *, repo_id, repo_type, revision):
            assert repo_id == "terrafloww/demo"
            assert repo_type == "dataset"
            assert revision is None
            return [
                "README.md",
                "data/part-0001.parquet",
                "data/part-0002.parquet",
                "images/thumb.png",
            ]

    fake_hf_module = types.SimpleNamespace(HfApi=lambda: _FakeApi())
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    paths = resolve_hf_parquet_paths("hf://datasets/terrafloww/demo/data")
    assert paths == [
        "hf://datasets/terrafloww/demo/data/part-0001.parquet",
        "hf://datasets/terrafloww/demo/data/part-0002.parquet",
    ]


def test_load_hf_parquet_table_streams_arrow_batches(monkeypatch) -> None:
    table_bytes = _parquet_bytes(
        pa.table({"id": pa.array(["scene-1", "scene-2"]), "value": pa.array([1, 2])})
    )
    monkeypatch.setattr(
        "rasteret.integrations.huggingface.resolve_hf_parquet_paths",
        lambda _: ["hf://datasets/terrafloww/demo/data/part-0001.parquet"],
    )
    monkeypatch.setattr(
        "rasteret.integrations.huggingface._open_hf_filesystem",
        lambda: _FakeHfFs(
            {
                "datasets/terrafloww/demo/data/part-0001.parquet": table_bytes,
            }
        ),
    )

    table = load_hf_parquet_table(
        "hf://datasets/terrafloww/demo/data",
        columns=["id"],
        filter_expr=ds.field("id") == "scene-2",
    )

    assert table.column("id").to_pylist() == ["scene-2"]


def test_open_hf_streaming_source_reads_schema_from_parquet(monkeypatch) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    table_bytes = _parquet_bytes(schema.empty_table())

    monkeypatch.setattr(
        "rasteret.integrations.huggingface.resolve_hf_parquet_paths",
        lambda _: ["hf://datasets/terrafloww/demo/data/part-0001.parquet"],
    )
    monkeypatch.setattr(
        "rasteret.integrations.huggingface._open_hf_filesystem",
        lambda: _FakeHfFs(
            {
                "datasets/terrafloww/demo/data/part-0001.parquet": table_bytes,
            }
        ),
    )

    source = open_hf_streaming_source("hf://datasets/terrafloww/demo/data")
    assert isinstance(source, HFStreamingSource)
    assert source.schema == schema
    assert source.parquet_paths == (
        "hf://datasets/terrafloww/demo/data/part-0001.parquet",
    )


def test_subset_hf_streaming_source_builds_filters() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
            pa.field("eo:cloud_cover", pa.float64()),
            pa.field("split", pa.string()),
        ]
    )
    source = HFStreamingSource(
        path="hf://datasets/terrafloww/demo/data",
        parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
        schema=schema,
    )

    filtered = subset_hf_streaming_source(
        source,
        cloud_cover_lt=20,
        date_range=("2024-01-01", "2024-01-31"),
        bbox=(0.0, 0.0, 1.0, 1.0),
        split=("train", "val"),
    )

    assert filtered.filters is not None
    conjunction = filtered.filters[0]
    assert ("eo:cloud_cover", "<", 20.0) in conjunction
    assert ("split", "in", ["train", "val"]) in conjunction
    assert (("bbox", "xmax"), ">=", 0.0) in conjunction


def test_subset_hf_streaming_source_builds_struct_bbox_filters() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    source = HFStreamingSource(
        path="hf://datasets/terrafloww/demo/data",
        parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
        schema=schema,
    )

    filtered = subset_hf_streaming_source(source, bbox=(0.0, 0.0, 1.0, 1.0))

    assert filtered.filters == (
        (
            (("bbox", "xmax"), ">=", 0.0),
            (("bbox", "xmin"), "<=", 1.0),
            (("bbox", "ymax"), ">=", 0.0),
            (("bbox", "ymin"), "<=", 1.0),
        ),
    )


def test_head_hf_streaming_source_returns_arrow_table(monkeypatch) -> None:
    schema = pa.schema([pa.field("id", pa.string())])
    source = HFStreamingSource(
        path="hf://datasets/terrafloww/demo/data",
        parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
        schema=schema,
    )
    table_bytes = _parquet_bytes(
        pa.table({"id": pa.array(["scene-1", "scene-2", "scene-3"])})
    )
    monkeypatch.setattr(
        "rasteret.integrations.huggingface._open_hf_filesystem",
        lambda: _FakeHfFs(
            {
                "datasets/terrafloww/demo/data/part-0001.parquet": table_bytes,
            }
        ),
    )

    head = head_hf_streaming_source(source, n=2, columns=["id"])
    assert head.column("id").to_pylist() == ["scene-1", "scene-2"]


def test_hf_streaming_collection_iterates_rasters(monkeypatch) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field(
                "assets",
                pa.struct(
                    [pa.field("B04", pa.struct([pa.field("href", pa.string())]))]
                ),
            ),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
            pa.field("collection", pa.string()),
            pa.field("B04_metadata", pa.struct([pa.field("image_width", pa.int32())])),
        ]
    )
    source = HFStreamingSource(
        path="hf://datasets/terrafloww/demo/data",
        parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
        schema=schema,
    )
    collection = Collection(hf_streaming=source)

    monkeypatch.setattr(
        "rasteret.core.collection.iter_hf_arrow_tables",
        lambda *_args, **_kwargs: iter(
            [
                pa.table(
                    {
                        "id": pa.array(["scene-1"]),
                        "datetime": pa.array(
                            [datetime(2024, 1, 1)],
                            type=pa.timestamp("us"),
                        ),
                        "geometry": pa.array([None], type=pa.null()),
                        "assets": pa.array(
                            [{"B04": {"href": "https://example.com/test.tif"}}]
                        ),
                        "bbox": pa.array(
                            [
                                {
                                    "xmin": 0.0,
                                    "ymin": 0.0,
                                    "xmax": 1.0,
                                    "ymax": 1.0,
                                }
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
                        "collection": pa.array(["sentinel-2-l2a"]),
                        "B04_metadata": pa.array(
                            [{"image_width": 512}],
                            type=pa.struct([pa.field("image_width", pa.int32())]),
                        ),
                    }
                )
            ]
        ),
    )

    raster = asyncio.run(collection.get_first_raster())
    assert raster.id == "scene-1"
    assert raster.assets["B04"]["href"] == "https://example.com/test.tif"


def test_hf_streaming_collection_subset_head_uses_managed_filters(monkeypatch) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    source = HFStreamingSource(
        path="hf://datasets/terrafloww/demo/data",
        parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
        schema=schema,
    )
    collection = Collection(hf_streaming=source)
    table_bytes = _parquet_bytes(
        pa.table(
            {
                "id": pa.array(["scene-1", "scene-2"]),
                "bbox": pa.array(
                    [
                        {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
                        {"xmin": 10.0, "ymin": 10.0, "xmax": 11.0, "ymax": 11.0},
                    ],
                    type=schema.field("bbox").type,
                ),
            }
        )
    )
    monkeypatch.setattr(
        "rasteret.integrations.huggingface._open_hf_filesystem",
        lambda: _FakeHfFs(
            {
                "datasets/terrafloww/demo/data/part-0001.parquet": table_bytes,
            }
        ),
    )

    head = collection.subset(bbox=(0.0, 0.0, 1.0, 1.0)).head(1, columns=["id"])

    assert head.column("id").to_pylist() == ["scene-1"]


def test_hf_streaming_collection_subset_head_uses_struct_bbox_filters(
    monkeypatch,
) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    source = HFStreamingSource(
        path="hf://datasets/terrafloww/demo/data",
        parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
        schema=schema,
    )
    collection = Collection(hf_streaming=source)
    table_bytes = _parquet_bytes(
        pa.table(
            {
                "id": pa.array(["scene-1"]),
                "bbox": pa.array(
                    [{"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}],
                    type=schema.field("bbox").type,
                ),
            }
        )
    )
    monkeypatch.setattr(
        "rasteret.integrations.huggingface._open_hf_filesystem",
        lambda: _FakeHfFs(
            {
                "datasets/terrafloww/demo/data/part-0001.parquet": table_bytes,
            }
        ),
    )

    head = collection.subset(bbox=(0.0, 0.0, 1.0, 1.0)).head(1, columns=["id"])

    assert head.column("id").to_pylist() == ["scene-1"]


def test_hf_streaming_collection_rejects_where_expr() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    collection = Collection(
        hf_streaming=HFStreamingSource(
            path="hf://datasets/terrafloww/demo/data",
            parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
            schema=schema,
        )
    )

    with pytest.raises(NotImplementedError, match="HF streaming collections"):
        collection.where(ds.field("id") == "scene-1")


def test_hf_streaming_collection_len_is_not_available() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    collection = Collection(
        hf_streaming=HFStreamingSource(
            path="hf://datasets/terrafloww/demo/data",
            parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
            schema=schema,
        )
    )

    with pytest.raises(TypeError, match="len\\(\\) is not available"):
        len(collection)


def test_hf_streaming_collection_describe_bounds_and_epsg(monkeypatch) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field("assets", pa.null()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
            pa.field("proj:epsg", pa.int32()),
        ]
    )
    collection = Collection(
        hf_streaming=HFStreamingSource(
            path="hf://datasets/terrafloww/demo/data",
            parquet_paths=("hf://datasets/terrafloww/demo/data/part-0001.parquet",),
            schema=schema,
        )
    )

    monkeypatch.setattr(
        "rasteret.core.collection.iter_hf_arrow_tables",
        lambda *_args, **_kwargs: iter(
            [
                pa.table(
                    {
                        "bbox": pa.array(
                            [
                                {"xmin": 0.0, "ymin": 1.0, "xmax": 2.0, "ymax": 3.0},
                                {
                                    "xmin": 10.0,
                                    "ymin": 11.0,
                                    "xmax": 12.0,
                                    "ymax": 13.0,
                                },
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
                        "proj:epsg": pa.array([32632, 32633], type=pa.int32()),
                    }
                )
            ]
        ),
    )

    assert collection.bounds == (0.0, 1.0, 12.0, 13.0)
    assert collection.epsg == [32632, 32633]
    describe = collection.describe()
    assert describe["records"] == "?"
    assert describe["bounds"] == (0.0, 1.0, 12.0, 13.0)
