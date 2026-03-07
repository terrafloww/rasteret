# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pyarrow.dataset as ds

from rasteret.integrations.huggingface import (
    load_hf_parquet_table,
    parse_hf_dataset_uri,
    resolve_hf_parquet_paths,
)


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


def test_load_hf_parquet_table_uses_datasets_reader(monkeypatch) -> None:
    expected = pa.table({"id": pa.array(["scene-1"])})
    captured: dict[str, object] = {}

    class _FakeDatasetObj:
        data = types.SimpleNamespace(table=expected)

    class _FakeDataset:
        @classmethod
        def from_parquet(
            cls,
            path_or_paths,
            *,
            columns=None,
            filters=None,
            keep_in_memory=False,
        ):
            captured["paths"] = path_or_paths
            captured["columns"] = columns
            captured["filters"] = filters
            captured["keep_in_memory"] = keep_in_memory
            return _FakeDatasetObj()

    fake_datasets_module = types.SimpleNamespace(Dataset=_FakeDataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)
    monkeypatch.setattr(
        "rasteret.integrations.huggingface.resolve_hf_parquet_paths",
        lambda _: ["hf://datasets/terrafloww/demo/data/part-0001.parquet"],
    )

    filter_expr = ds.field("id") == "scene-1"
    table = load_hf_parquet_table(
        "hf://datasets/terrafloww/demo/data",
        columns=["id"],
        filter_expr=filter_expr,
    )

    assert table is expected
    assert captured["paths"] == ["hf://datasets/terrafloww/demo/data/part-0001.parquet"]
    assert captured["columns"] == ["id"]
    assert captured["filters"] is filter_expr
    assert captured["keep_in_memory"] is False
