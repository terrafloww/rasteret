# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

if TYPE_CHECKING:
    import pyarrow as pa


HF_DATASETS_URI_PREFIX = "hf://datasets/"


def is_hf_dataset_uri(path: str) -> bool:
    """Return ``True`` when *path* uses Hugging Face dataset URI format."""
    return path.startswith(HF_DATASETS_URI_PREFIX)


def parse_hf_dataset_uri(path: str) -> tuple[str, str, str | None]:
    """Parse ``hf://datasets/<org>/<name>[@revision][/subpath]`` URIs."""
    if not is_hf_dataset_uri(path):
        raise ValueError(
            f"Expected a Hugging Face datasets URI (hf://datasets/...), got: {path!r}"
        )

    tail = path[len(HF_DATASETS_URI_PREFIX) :].strip("/")
    parts = [part for part in tail.split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            "Invalid Hugging Face datasets URI. Expected "
            "'hf://datasets/<org>/<name>[/path]'."
        )

    org = parts[0]
    dataset_with_revision = parts[1]
    if "@" in dataset_with_revision:
        dataset_name, revision = dataset_with_revision.split("@", 1)
    else:
        dataset_name, revision = dataset_with_revision, None

    repo_id = f"{org}/{dataset_name}"
    subpath = "/".join(parts[2:])
    return repo_id, subpath, revision


def resolve_hf_parquet_paths(path: str) -> list[str]:
    """Resolve a Hugging Face dataset URI to one or more parquet file URIs."""
    repo_id, subpath, revision = parse_hf_dataset_uri(path)
    repo_locator = f"{repo_id}@{revision}" if revision else repo_id

    if subpath and subpath.lower().endswith(".parquet"):
        return [f"{HF_DATASETS_URI_PREFIX}{repo_locator}/{subpath}"]

    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "Hugging Face path support requires 'huggingface_hub'. "
            "Install rasteret with extras that include Hugging Face support."
        ) from exc

    api = HfApi()
    repo_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )

    prefix = f"{subpath.rstrip('/')}/" if subpath else ""
    parquet_files = sorted(
        file_path
        for file_path in repo_files
        if file_path.lower().endswith(".parquet")
        and (not prefix or file_path.startswith(prefix))
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in Hugging Face dataset '{repo_locator}' "
            f"under '{subpath or '/'}'."
        )

    return [
        f"{HF_DATASETS_URI_PREFIX}{repo_locator}/{file_path}"
        for file_path in parquet_files
    ]


def load_hf_parquet_table(
    path: str,
    *,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> pa.Table:
    """Load parquet data from Hugging Face using the ``datasets`` reader."""
    try:
        from datasets import Dataset as HFDataset
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "Hugging Face path support requires 'datasets'. "
            "Install rasteret with extras that include Hugging Face support."
        ) from exc

    parquet_paths = resolve_hf_parquet_paths(path)
    hf_dataset = HFDataset.from_parquet(
        parquet_paths,
        columns=columns,
        filters=filter_expr,
        keep_in_memory=False,
    )
    return hf_dataset.data.table


def open_hf_parquet_dataset(
    path: str,
    *,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> ds.Dataset:
    """Open Hugging Face parquet files as a pyarrow dataset."""
    table = load_hf_parquet_table(path, columns=columns, filter_expr=filter_expr)
    return ds.dataset(table)
