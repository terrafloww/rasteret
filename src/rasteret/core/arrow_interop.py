# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Shared Apache Arrow protocol helpers.

Rasteret accepts Arrow data from many producers, but the boundary should stay
the Arrow protocol rather than producer-specific branches.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.dataset as pads


def as_reader(value: Any) -> pa.RecordBatchReader:
    """Return *value* as a ``RecordBatchReader`` using public Arrow APIs."""
    if isinstance(value, pa.RecordBatchReader):
        return value
    if isinstance(value, pa.Table):
        return value.to_reader()
    if isinstance(value, pa.RecordBatch):
        return pa.RecordBatchReader.from_batches(value.schema, [value])
    if isinstance(value, pads.Dataset):
        return value.scanner().to_reader()

    stream_export = getattr(value, "__arrow_c_stream__", None)
    if callable(stream_export):
        return pa.RecordBatchReader.from_stream(value)

    table = as_table(value)
    return table.to_reader()


def as_dataset(value: Any) -> tuple[pads.Dataset, pa.Table | None]:
    """Return ``(dataset, materialized_table)`` for supported Arrow inputs."""
    if isinstance(value, pads.Dataset):
        return value, None
    if isinstance(value, pa.Table):
        return pads.dataset(value), value
    if isinstance(value, pa.RecordBatch):
        return pads.dataset(value), None
    if isinstance(value, pa.RecordBatchReader):
        return pads.dataset(value), None

    stream_export = getattr(value, "__arrow_c_stream__", None)
    if callable(stream_export):
        return pads.dataset(pa.RecordBatchReader.from_stream(value)), None

    table = as_table(value)
    return pads.dataset(table), table


def as_table(
    value: Any,
    *,
    columns: list[str] | None = None,
    filter_expr: Any | None = None,
) -> pa.Table:
    """Materialize a supported Arrow object as a ``pyarrow.Table``."""
    if columns is not None or filter_expr is not None:
        dataset, materialized = as_dataset(value)
        if materialized is not None and filter_expr is None:
            return materialized.select(columns) if columns is not None else materialized
        return dataset.to_table(columns=columns, filter=filter_expr)

    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.RecordBatchReader):
        return value.read_all()
    if isinstance(value, pads.Dataset):
        return value.to_table()

    stream_export = getattr(value, "__arrow_c_stream__", None)
    if callable(stream_export):
        return pa.RecordBatchReader.from_stream(value).read_all()

    for method_name in (
        "to_arrow_table",
        "fetch_arrow_table",
        "arrow",
        "to_arrow",
    ):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        exported = method()
        if exported is value:
            break
        try:
            return as_table(exported)
        except TypeError:
            continue

    try:
        return pa.table(value)
    except Exception as exc:
        raise TypeError(
            "Expected a pyarrow Table, RecordBatch, RecordBatchReader, Dataset, "
            "an Arrow PyCapsule producer, or an object exporting Arrow data."
        ) from exc


def as_table_or_none(value: Any) -> pa.Table | None:
    """Best-effort table materialization used for optional tabular inputs."""
    if isinstance(value, (bytes, bytearray, list, tuple)):
        return None
    if isinstance(value, dict) and "type" in value and "coordinates" in value:
        return None
    try:
        return as_table(value)
    except Exception:
        return None
