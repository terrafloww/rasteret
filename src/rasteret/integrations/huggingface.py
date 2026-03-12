# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pyarrow as pa


HF_DATASETS_URI_PREFIX = "hf://datasets/"
HFFieldRef = str | tuple[str, ...]
HFStreamingClause = tuple[HFFieldRef, str, object]
HFStreamingConjunction = tuple[HFStreamingClause, ...]
HFStreamingDNF = tuple[HFStreamingConjunction, ...]


@dataclass(frozen=True)
class HFStreamingSource:
    """Hugging Face streaming-backed metadata source for a Collection."""

    path: str
    parquet_paths: tuple[str, ...]
    schema: pa.Schema
    filters: HFStreamingDNF | None = None


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

    # Defaulting to repo root can unintentionally mix index-style and wide
    # runtime parquet files. When a conventional ``data/*.parquet`` layout is
    # present and no subpath is explicitly provided, prefer that surface.
    if not subpath:
        data_parquet_files = sorted(
            file_path
            for file_path in repo_files
            if file_path.lower().endswith(".parquet") and file_path.startswith("data/")
        )
    else:
        data_parquet_files = []

    parquet_files = (
        data_parquet_files
        if data_parquet_files
        else sorted(
            file_path
            for file_path in repo_files
            if file_path.lower().endswith(".parquet")
            and (not prefix or file_path.startswith(prefix))
        )
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


def _open_hf_filesystem():
    try:
        from huggingface_hub import HfFileSystem
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "Hugging Face path support requires 'huggingface_hub'. "
            "Install rasteret with extras that include Hugging Face support."
        ) from exc

    return HfFileSystem()


def _hf_filesystem_path(path: str) -> str:
    repo_id, subpath, revision = parse_hf_dataset_uri(path)
    repo_locator = f"{repo_id}@{revision}" if revision else repo_id
    return f"datasets/{repo_locator}/{subpath}"


def _open_hf_parquet_file(fs, path: str):
    return fs.open(_hf_filesystem_path(path), "rb")


def _field_ref_to_path(field: HFFieldRef) -> str:
    if isinstance(field, tuple):
        return ".".join(field)
    return field


def _field_ref_to_expr(field: HFFieldRef) -> ds.Expression:
    if isinstance(field, tuple):
        return ds.field(*field)
    return ds.field(field)


def _filter_columns(filters: HFStreamingDNF | None) -> list[str]:
    if not filters:
        return []
    columns: set[str] = set()
    for conjunction in filters:
        for field, _, _ in conjunction:
            if isinstance(field, tuple):
                columns.add(field[0])
            else:
                columns.add(field)
    return sorted(columns)


def _normalize_stat_value(value: object) -> object:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    return value


def _clause_to_expr(clause: HFStreamingClause) -> ds.Expression:
    field, op, value = clause
    expr = _field_ref_to_expr(field)
    if op == "==":
        return expr == value
    if op == ">=":
        return expr >= value
    if op == "<=":
        return expr <= value
    if op == ">":
        return expr > value
    if op == "<":
        return expr < value
    if op == "in":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("'in' filter expects a non-string sequence")
        return expr.isin(list(value))
    raise ValueError(f"Unsupported Hugging Face filter operator: {op!r}")


def _dnf_to_expr(filters: HFStreamingDNF | None) -> ds.Expression | None:
    if not filters:
        return None

    disjunction: ds.Expression | None = None
    for conjunction in filters:
        conjunction_expr: ds.Expression | None = None
        for clause in conjunction:
            clause_expr = _clause_to_expr(clause)
            conjunction_expr = (
                clause_expr
                if conjunction_expr is None
                else conjunction_expr & clause_expr
            )
        if conjunction_expr is None:
            continue
        disjunction = (
            conjunction_expr if disjunction is None else disjunction | conjunction_expr
        )
    return disjunction


def _column_stats_map(row_group: pq.RowGroupMetaData) -> dict[str, object]:
    out: dict[str, object] = {}
    for idx in range(row_group.num_columns):
        col = row_group.column(idx)
        out[col.path_in_schema] = col
    return out


def _row_group_clause_may_match(
    stats_column: object | None,
    op: str,
    value: object,
) -> bool:
    if stats_column is None:
        return True
    stats = getattr(stats_column, "statistics", None)
    if stats is None or not getattr(stats, "has_min_max", False):
        return True

    min_value = _normalize_stat_value(stats.min)
    max_value = _normalize_stat_value(stats.max)
    if min_value is None or max_value is None:
        return True

    if op == "==":
        return min_value <= value <= max_value
    if op == ">=":
        return max_value >= value
    if op == "<=":
        return min_value <= value
    if op == ">":
        return max_value > value
    if op == "<":
        return min_value < value
    if op == "in":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return True
        return any(min_value <= item <= max_value for item in value)
    return True


def _matching_row_groups(
    parquet_file: pq.ParquetFile,
    filters: HFStreamingDNF | None,
) -> list[int] | None:
    if not filters:
        return None

    matches: list[int] = []
    metadata = parquet_file.metadata
    for row_group_id in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_id)
        stats_map = _column_stats_map(row_group)
        row_group_matches = False
        for conjunction in filters:
            if all(
                _row_group_clause_may_match(
                    stats_map.get(_field_ref_to_path(field)),
                    op,
                    value,
                )
                for field, op, value in conjunction
            ):
                row_group_matches = True
                break
        if row_group_matches:
            matches.append(row_group_id)
    return matches


def open_hf_streaming_source(path: str) -> HFStreamingSource:
    """Open a Hugging Face dataset URI as a streaming-backed collection source."""
    parquet_paths = tuple(resolve_hf_parquet_paths(path))
    fs = _open_hf_filesystem()
    with _open_hf_parquet_file(fs, parquet_paths[0]) as f:
        schema = pq.ParquetFile(f).schema_arrow
    return HFStreamingSource(path=path, parquet_paths=parquet_paths, schema=schema)


def _combine_streaming_filters(
    left: HFStreamingDNF | None,
    right: HFStreamingDNF | None,
) -> HFStreamingDNF | None:
    if left is None:
        return right
    if right is None:
        return left
    return tuple(tuple((*lhs, *rhs)) for lhs in left for rhs in right)


def _bbox_struct_field(schema: pa.Schema) -> pa.Field | None:
    if "bbox" not in schema.names:
        return None
    field = schema.field("bbox")
    field_type = field.type
    if not pa.types.is_struct(field_type):
        return None
    child_names = {child.name for child in field_type}
    required = {"xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(child_names):
        return None
    return field


def _bbox_overlap_conjunction(
    schema: pa.Schema,
    bbox: tuple[float, float, float, float],
) -> HFStreamingConjunction:
    minx, miny, maxx, maxy = bbox
    bbox_struct = _bbox_struct_field(schema)
    if bbox_struct is None:
        raise ValueError(
            "bbox filtering requires a GeoParquet-style 'bbox' struct with "
            "xmin/ymin/xmax/ymax children."
        )
    return (
        (("bbox", "xmax"), ">=", float(minx)),
        (("bbox", "xmin"), "<=", float(maxx)),
        (("bbox", "ymax"), ">=", float(miny)),
        (("bbox", "ymin"), "<=", float(maxy)),
    )


def subset_hf_streaming_source(
    source: HFStreamingSource,
    *,
    cloud_cover_lt: float | None = None,
    date_range: tuple[str, str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geometries: Any = None,
    split: str | Sequence[str] | None = None,
    split_column: str = "split",
) -> HFStreamingSource:
    """Return a new streaming source with additional managed filters applied."""
    schema_names = set(source.schema.names)
    bbox_struct = _bbox_struct_field(source.schema)
    filters: HFStreamingDNF | None = None

    if cloud_cover_lt is not None:
        if "eo:cloud_cover" not in schema_names:
            raise ValueError("Collection has no cloud cover data")
        if not isinstance(cloud_cover_lt, (int, float)) or not (
            0 <= cloud_cover_lt <= 100
        ):
            raise ValueError(
                f"Invalid cloud_cover_lt={cloud_cover_lt!r}: must be between 0 and 100."
            )
        filters = _combine_streaming_filters(
            filters,
            ((("eo:cloud_cover", "<", float(cloud_cover_lt)),),),
        )

    if date_range is not None:
        if "datetime" not in schema_names:
            raise ValueError("Collection has no datetime data")
        start_raw, end_raw = date_range
        if not start_raw or not end_raw:
            raise ValueError("Invalid date range")
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        if start > end:
            raise ValueError("Invalid date range")
        filters = _combine_streaming_filters(
            filters,
            (
                (
                    ("datetime", ">=", start.to_pydatetime()),
                    ("datetime", "<=", end.to_pydatetime()),
                ),
            ),
        )

    if bbox is not None:
        if bbox_struct is None:
            raise ValueError(
                "bbox filtering requires a GeoParquet-style 'bbox' struct "
                "with xmin/ymin/xmax/ymax children."
            )
        if len(bbox) != 4:
            raise ValueError("Invalid bbox format")
        minx, miny, maxx, maxy = bbox
        if minx > maxx or miny > maxy:
            raise ValueError("Invalid bbox coordinates")
        filters = _combine_streaming_filters(
            filters,
            (_bbox_overlap_conjunction(source.schema, (minx, miny, maxx, maxy)),),
        )

    if geometries is not None:
        if bbox_struct is None:
            raise ValueError(
                "geometry filtering requires a GeoParquet-style 'bbox' "
                "struct with xmin/ymin/xmax/ymax children."
            )
        from rasteret.core.geometry import bbox_array, coerce_to_geoarrow

        geo_arr = coerce_to_geoarrow(geometries)
        xmin, ymin, xmax, ymax = bbox_array(geo_arr)
        geometry_filters: list[HFStreamingConjunction] = []
        for idx in range(len(xmin)):
            geometry_filters.append(
                _bbox_overlap_conjunction(
                    source.schema,
                    (
                        float(xmin[idx].as_py()),
                        float(ymin[idx].as_py()),
                        float(xmax[idx].as_py()),
                        float(ymax[idx].as_py()),
                    ),
                )
            )
        if geometry_filters:
            filters = _combine_streaming_filters(filters, tuple(geometry_filters))

    if split is not None:
        if split_column not in schema_names:
            raise ValueError(f"Collection has no split column: '{split_column}'")
        if isinstance(split, str):
            split_clause: HFStreamingConjunction = ((split_column, "==", split),)
        elif (
            isinstance(split, Sequence)
            and not isinstance(split, (str, bytes))
            and split
            and all(isinstance(value, str) for value in split)
        ):
            split_clause = ((split_column, "in", list(split)),)
        else:
            raise ValueError(
                "Invalid split filter. Use a split name or sequence of split names."
            )
        filters = _combine_streaming_filters(filters, (split_clause,))

    return replace(source, filters=_combine_streaming_filters(source.filters, filters))


def iter_hf_arrow_tables(
    source: HFStreamingSource,
    *,
    columns: list[str] | None = None,
    batch_size: int = 1024,
):
    """Yield Arrow tables from an HF streaming source."""
    filter_expr = _dnf_to_expr(source.filters)
    output_columns = columns
    read_columns = columns
    if columns is not None and source.filters:
        needed = set(columns)
        needed.update(_filter_columns(source.filters))
        read_columns = sorted(needed)
    fs = _open_hf_filesystem()
    for path in source.parquet_paths:
        with _open_hf_parquet_file(fs, path) as f:
            parquet_file = pq.ParquetFile(f)
            row_groups = _matching_row_groups(parquet_file, source.filters)
            if row_groups == []:
                continue
            for record_batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=read_columns,
                row_groups=row_groups,
            ):
                table = pa.Table.from_batches([record_batch])
                if filter_expr is not None:
                    table = _apply_arrow_filter(table, filter_expr)
                if output_columns is not None:
                    table = table.select(output_columns)
                if table.num_rows:
                    yield table


def head_hf_streaming_source(
    source: HFStreamingSource,
    *,
    n: int = 5,
    columns: list[str] | None = None,
) -> pa.Table:
    """Return the first *n* rows from a streaming source as a PyArrow table."""
    if n <= 0:
        schema = (
            source.schema
            if columns is None
            else pa.schema([source.schema.field(name) for name in columns])
        )
        return schema.empty_table()

    tables: list[pa.Table] = []
    remaining = n
    iterator = iter_hf_arrow_tables(source, columns=columns, batch_size=max(n, 1))
    try:
        for table in iterator:
            if table.num_rows >= remaining:
                tables.append(table.slice(0, remaining))
                remaining = 0
                break
            tables.append(table)
            remaining -= table.num_rows
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    if not tables:
        schema = (
            source.schema
            if columns is None
            else pa.schema([source.schema.field(name) for name in columns])
        )
        return schema.empty_table()
    return pa.concat_tables(tables, promote_options="default")


def load_hf_parquet_table(
    path: str,
    *,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> pa.Table:
    """Load parquet data from Hugging Face into a materialized Arrow table."""
    source = open_hf_streaming_source(path)
    tables = list(iter_hf_arrow_tables(source, columns=columns))
    if not tables:
        schema = source.schema
        if columns is not None:
            schema = pa.schema([schema.field(name) for name in columns])
        empty = schema.empty_table()
        if filter_expr is not None:
            return _apply_arrow_filter(empty, filter_expr)
        return empty
    table = pa.concat_tables(tables, promote_options="default")
    if filter_expr is not None:
        table = _apply_arrow_filter(table, filter_expr)
    return table


def _apply_arrow_filter(table: pa.Table, filter_expr: ds.Expression) -> pa.Table:
    """Apply an Arrow dataset filter with a clear compatibility error on failure."""
    try:
        return ds.dataset(table).to_table(filter=filter_expr)
    except pa.ArrowKeyError as exc:
        message = str(exc)
        if "No function registered with name" not in message:
            raise
        raise RuntimeError(
            "PyArrow comparison/filter kernels are unavailable in this environment. "
            "Install the full 'pyarrow' package (avoid minimal/core-only builds) "
            f"and retry. Original error: {message}"
        ) from exc


def open_hf_parquet_dataset(
    path: str,
    *,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> ds.Dataset:
    """Open Hugging Face parquet files as a pyarrow dataset."""
    table = load_hf_parquet_table(path, columns=columns, filter_expr=filter_expr)
    return ds.dataset(table)
