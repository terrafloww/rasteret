# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.


from __future__ import annotations

import json
import logging
import re
import struct
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from rasteret.core.execution import (
    _derive_query_bbox,
    get_collection_gdf,
    get_collection_numpy,
    get_collection_xarray,
)
from rasteret.core.geometry import intersect_bbox
from rasteret.core.parquet_read_planner import ParquetReadPlanner
from rasteret.core.point_sampling import get_collection_point_samples
from rasteret.core.raster_accessor import RasterAccessor
from rasteret.integrations.huggingface import (
    HFStreamingSource,
    head_hf_streaming_source,
    is_hf_dataset_uri,
    iter_hf_arrow_tables,
    load_hf_parquet_table,
    open_hf_streaming_source,
    subset_hf_streaming_source,
)
from rasteret.types import RasterInfo

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr

    from rasteret.core.display import DescribeResult
    from rasteret.integrations.torchgeo import RasteretGeoDataset

logger = logging.getLogger(__name__)
_UNSET_RECORD_INDEX_FILTER = object()
_PARQUET_DATASET_CACHE: dict[tuple[str, bool], tuple[object, ds.Dataset]] = {}
_PARQUET_DATASET_CACHE_LOCK = threading.Lock()


def _is_cloud_uri(path: str) -> bool:
    """Return True for s3://, gs://, az:// and similar cloud URIs."""
    return "://" in path and not path.startswith("file://")


def _local_dataset_fingerprint(path_str: str) -> tuple:
    """Best-effort local fingerprint for cache invalidation.

    Uses the dataset path plus the standard metadata sidecar when present.
    This keeps the cache stable across repeated reads while invalidating when a
    local file/directory is replaced or rewritten.
    """
    path = Path(path_str)
    if path.is_file():
        stat = path.stat()
        return ("file", str(path), stat.st_mtime_ns, stat.st_size)

    sidecar = path / "_metadata"
    if sidecar.is_file():
        sidecar_stat = sidecar.stat()
        dir_stat = path.stat()
        return (
            "dir+metadata",
            str(path),
            dir_stat.st_mtime_ns,
            sidecar_stat.st_mtime_ns,
            sidecar_stat.st_size,
        )

    dir_stat = path.stat()
    return ("dir", str(path), dir_stat.st_mtime_ns)


def _open_parquet_dataset(path_str: str, *, try_hive: bool = True) -> ds.Dataset:
    """Open a Parquet dataset from a local path or cloud URI.

    PyArrow's ``ds.dataset()`` handles both local paths and cloud URIs
    (s3://, gs://) when the string is passed directly.  We try Hive-style
    partitioning first, then fall back to plain Parquet.

    For local datasets, a standard Parquet ``_metadata`` sidecar is preferred
    when present so readers can avoid reopening every file footer. For cloud
    datasets, Rasteret relies on an in-process dataset cache instead of
    forcing a potentially large remote ``_metadata`` download.
    """
    cache_key = (path_str, try_hive)
    fingerprint: object
    if _is_cloud_uri(path_str):
        fingerprint = ("cloud", path_str)
    else:
        fingerprint = _local_dataset_fingerprint(path_str)

    with _PARQUET_DATASET_CACHE_LOCK:
        cached = _PARQUET_DATASET_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

    kwargs: dict[str, Any] = {
        "format": "parquet",
        "exclude_invalid_files": True,
    }
    metadata_path: str | None = None
    if not _is_cloud_uri(path_str):
        try:
            candidate = Path(path_str) / "_metadata"
            if candidate.is_file():
                metadata_path = str(candidate)
        except Exception:
            metadata_path = None

    dataset: ds.Dataset
    if metadata_path is not None:
        try:
            dataset = ds.parquet_dataset(
                metadata_path,
                partitioning="hive" if try_hive else None,
                partition_base_dir=path_str if try_hive else None,
            )
            with _PARQUET_DATASET_CACHE_LOCK:
                _PARQUET_DATASET_CACHE[cache_key] = (fingerprint, dataset)
            return dataset
        except (pa.ArrowInvalid, FileNotFoundError, OSError):
            pass

    if try_hive:
        try:
            dataset = ds.dataset(path_str, partitioning="hive", **kwargs)
            with _PARQUET_DATASET_CACHE_LOCK:
                _PARQUET_DATASET_CACHE[cache_key] = (fingerprint, dataset)
            return dataset
        except pa.ArrowInvalid:
            pass
    dataset = ds.dataset(path_str, **kwargs)
    with _PARQUET_DATASET_CACHE_LOCK:
        _PARQUET_DATASET_CACHE[cache_key] = (fingerprint, dataset)
    return dataset


def _stem_from_path(path_str: str) -> str:
    """Extract the filename stem from a local path or cloud URI."""
    from urllib.parse import urlparse

    parsed = urlparse(path_str)
    tail = parsed.path.rstrip("/") if parsed.scheme else path_str.rstrip("/")
    return Path(tail).stem if tail else ""


def _filesystem_source_path(path_str: str) -> str:
    parsed = urlparse(path_str)
    if not parsed.scheme:
        return path_str
    return f"{parsed.netloc}{parsed.path}"


# WKB geometry type id -> GeoParquet type name (OGC Simple Features).
_WKB_TYPE_NAMES: dict[int, str] = {
    1: "Point",
    2: "LineString",
    3: "Polygon",
    4: "MultiPoint",
    5: "MultiLineString",
    6: "MultiPolygon",
    7: "GeometryCollection",
}


def _geometry_types_from_wkb(col: pa.ChunkedArray) -> list[str]:
    """Return sorted GeoParquet ``geometry_types`` from a WKB column."""
    seen: set[str] = set()
    for chunk in col.chunks:
        for val in chunk:
            if val is None:
                continue
            raw = val.as_py()
            if raw is None or len(raw) < 5:
                continue
            fmt = "<I" if raw[0] == 1 else ">I"
            type_id = struct.unpack(fmt, raw[1:5])[0] & 0xFF
            name = _WKB_TYPE_NAMES.get(type_id)
            if name:
                seen.add(name)
    return sorted(seen)


def _bbox_overlap_expr(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    field_name: str = "bbox",
) -> ds.Expression:
    """Arrow expression testing whether a record's bbox overlaps the given bounds."""
    return (
        (ds.field(field_name, "xmax") >= minx)
        & (ds.field(field_name, "xmin") <= maxx)
        & (ds.field(field_name, "ymax") >= miny)
        & (ds.field(field_name, "ymin") <= maxy)
    )


def _bbox_struct_field(schema: pa.Schema, field_name: str = "bbox") -> pa.Field | None:
    if field_name not in schema.names:
        return None
    field = schema.field(field_name)
    if not pa.types.is_struct(field.type):
        return None
    child_names = {child.name for child in field.type}
    required = {"xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(child_names):
        return None
    return field


def _bbox_value_to_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        xmin = value.get("xmin")
        ymin = value.get("ymin")
        xmax = value.get("xmax")
        ymax = value.get("ymax")
        if None in (xmin, ymin, xmax, ymax):
            return None
        return [float(xmin), float(ymin), float(xmax), float(ymax)]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(v) for v in value]
    return None


def _and_filters(
    current: ds.Expression | None, new: ds.Expression | None
) -> ds.Expression | None:
    if new is None:
        return current
    return new if current is None else current & new


class Collection:
    """
    A collection of raster data with flexible initialization.

    Collections can be created from:
    - Local partitioned datasets
    - Single Arrow tables

    Collections maintain efficient partitioned storage when using files.

    Examples
    --------
    # From partitioned dataset
    >>> collection = Collection.from_parquet("path/to/dataset")

    # Filter and process
    >>> filtered = collection.subset(cloud_cover_lt=20)
    >>> ds = filtered.get_xarray(...)
    """

    def __init__(
        self,
        dataset: ds.Dataset | None = None,
        hf_streaming: HFStreamingSource | None = None,
        collection_path: str | None = None,
        record_index_path: str | None = None,
        record_index_field_roles: dict[str, str] | None = None,
        record_index_column_map: dict[str, str] | None = None,
        record_index_href_column: str | None = None,
        record_index_band_index_map: dict[str, int] | None = None,
        record_index_url_rewrite_patterns: dict[str, str] | None = None,
        record_index_filesystem: Any | None = None,
        surface_fields: dict[str, list[str]] | None = None,
        filter_capabilities: dict[str, list[str]] | None = None,
        record_index_filter_expr: ds.Expression | None = None,
        wide_filter_expr: ds.Expression | None = None,
        name: str = "",
        description: str = "",
        data_source: str = "",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ):
        """Initialize a Collection.

        Parameters
        ----------
        dataset : pyarrow.dataset.Dataset, optional
            Backing Arrow dataset. ``None`` creates an empty or non-Dataset-backed
            collection.
        hf_streaming : HFStreamingSource, optional
            Hugging Face streaming-backed metadata source.
        name : str
            Human-readable collection name.
        description : str
            Free-text description.
        data_source : str
            Data source identifier (e.g. ``"sentinel-2-l2a"``).
        start_date : datetime, optional
            Collection temporal start.
        end_date : datetime, optional
            Collection temporal end.
        """
        self.dataset = dataset
        self._hf_streaming = hf_streaming
        self.name = name
        self.description = description
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self._planner = ParquetReadPlanner(
            collection_path=collection_path,
            record_index_path=record_index_path,
            record_index_field_roles=record_index_field_roles or {},
            record_index_column_map=record_index_column_map or {},
            record_index_href_column=record_index_href_column,
            record_index_band_index_map=record_index_band_index_map,
            record_index_url_rewrite_patterns=record_index_url_rewrite_patterns or {},
            record_index_filesystem=record_index_filesystem,
            surface_fields=(
                {
                    surface: tuple(fields)
                    for surface, fields in (surface_fields or {}).items()
                }
                or None
            ),
            filter_capabilities=(
                {
                    surface: tuple(fields)
                    for surface, fields in (filter_capabilities or {}).items()
                }
                or None
            ),
            record_index_filter_expr=record_index_filter_expr,
            wide_filter_expr=wide_filter_expr,
        )
        self._record_index_dataset: ds.Dataset | None = None
        if self.dataset is not None and self._hf_streaming is not None:
            raise ValueError(
                "Collection cannot use both Dataset and HF streaming backends"
            )
        if self.dataset is not None:
            self._validate_parquet_dataset()

    def _view(
        self,
        dataset: ds.Dataset | None = None,
        *,
        hf_streaming: HFStreamingSource | None = None,
        collection_path: str | None | object = None,
        record_index_filter_expr: ds.Expression | None | object = None,
        wide_filter_expr: ds.Expression | None | object = None,
        drop_record_index: bool = False,
    ) -> Collection:
        """Return a new Collection view preserving this Collection's metadata."""
        resolved_collection_path = self._collection_path
        if collection_path is not None:
            resolved_collection_path = (
                None
                if collection_path is _UNSET_RECORD_INDEX_FILTER
                else collection_path
            )
        resolved_record_index_filter = self._record_index_filter_expr
        resolved_wide_filter = self._wide_filter_expr
        if record_index_filter_expr is not None:
            resolved_record_index_filter = (
                None
                if record_index_filter_expr is _UNSET_RECORD_INDEX_FILTER
                else record_index_filter_expr
            )
        if wide_filter_expr is not None:
            resolved_wide_filter = (
                None
                if wide_filter_expr is _UNSET_RECORD_INDEX_FILTER
                else wide_filter_expr
            )
        return Collection(
            dataset=dataset,
            hf_streaming=hf_streaming,
            collection_path=None if drop_record_index else resolved_collection_path,
            record_index_path=None if drop_record_index else self._record_index_path,
            record_index_field_roles=(
                None if drop_record_index else self._record_index_field_roles
            ),
            record_index_column_map=(
                None if drop_record_index else self._record_index_column_map
            ),
            record_index_href_column=(
                None if drop_record_index else self._record_index_href_column
            ),
            record_index_band_index_map=(
                None if drop_record_index else self._record_index_band_index_map
            ),
            record_index_url_rewrite_patterns=(
                None if drop_record_index else self._record_index_url_rewrite_patterns
            ),
            record_index_filesystem=(
                None if drop_record_index else self._record_index_filesystem
            ),
            surface_fields=None if drop_record_index else self._surface_fields,
            filter_capabilities=(
                None if drop_record_index else self._filter_capabilities
            ),
            record_index_filter_expr=(
                None if drop_record_index else resolved_record_index_filter
            ),
            wide_filter_expr=None if drop_record_index else resolved_wide_filter,
            name=self.name,
            description=self.description,
            data_source=self.data_source,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def _has_record_index(self) -> bool:
        return bool(self._record_index_path)

    @property
    def _collection_path(self) -> str | None:
        return self._planner.collection_path

    @property
    def _record_index_path(self) -> str | None:
        return self._planner.record_index_path

    @property
    def _record_index_column_map(self) -> dict[str, str]:
        return self._planner.record_index_column_map or {}

    @property
    def _record_index_field_roles(self) -> dict[str, str]:
        return self._planner.record_index_field_roles or {}

    @property
    def _record_index_href_column(self) -> str | None:
        return self._planner.record_index_href_column

    @property
    def _record_index_band_index_map(self) -> dict[str, int] | None:
        return self._planner.record_index_band_index_map

    @property
    def _record_index_url_rewrite_patterns(self) -> dict[str, str]:
        return self._planner.record_index_url_rewrite_patterns or {}

    @property
    def _record_index_filesystem(self) -> Any | None:
        return self._planner.record_index_filesystem

    @property
    def _record_index_filter_expr(self) -> ds.Expression | None:
        return self._planner.record_index_filter_expr

    @property
    def _wide_filter_expr(self) -> ds.Expression | None:
        return self._planner.wide_filter_expr

    @property
    def _surface_fields(self) -> dict[str, list[str]]:
        if not self._planner.surface_fields:
            return {}
        return {
            surface: list(fields)
            for surface, fields in self._planner.surface_fields.items()
        }

    @property
    def _filter_capabilities(self) -> dict[str, list[str]]:
        if not self._planner.filter_capabilities:
            return {}
        return {
            surface: list(fields)
            for surface, fields in self._planner.filter_capabilities.items()
        }

    def _record_index_inverse_map(self) -> dict[str, str]:
        return {dst: src for src, dst in self._record_index_column_map.items()}

    def _record_index_source_column(self, name: str) -> str:
        source = self._planner.source_field(name)
        if source != name:
            return source
        return self._record_index_inverse_map().get(name, name)

    def _surface_has_field(
        self,
        surface: str,
        canonical: str,
        schema: pa.Schema | None = None,
    ) -> bool:
        if surface == "index":
            source_name = self._record_index_source_column(canonical)
            if schema is not None and source_name in schema.names:
                return True
        elif surface == "collection":
            if schema is not None and canonical in schema.names:
                return True
        return self._planner.surface_has_field(surface, canonical)

    def _surface_supports_filter(
        self,
        surface: str,
        capability: str,
        schema: pa.Schema | None = None,
    ) -> bool:
        if self._planner.filter_capabilities and surface in self._filter_capabilities:
            return self._planner.surface_supports_filter(surface, capability)
        if capability == "bbox":
            field_name = (
                self._record_index_source_column("bbox")
                if surface == "index"
                else "bbox"
            )
            if (
                schema is not None
                and _bbox_struct_field(schema, field_name) is not None
            ):
                return True
            return self._planner.surface_supports_filter(surface, capability)
        return self._surface_has_field(surface, capability, schema=schema)

    def _open_record_index_dataset(self) -> ds.Dataset:
        if not self._record_index_path:
            raise ValueError("Collection has no record index")
        if is_hf_dataset_uri(self._record_index_path):
            raise ValueError("HF record indexes use table reads, not datasets")
        if self._record_index_dataset is None:
            source = self._record_index_path
            if self._record_index_filesystem is not None:
                source = _filesystem_source_path(source)
            self._record_index_dataset = ds.dataset(
                source,
                format="parquet",
                filesystem=self._record_index_filesystem,
            )
        return self._record_index_dataset

    def _record_index_supports_expr(self, expr: ds.Expression) -> bool:
        if not self._record_index_path:
            return False
        try:
            dataset = self._open_record_index_dataset()
            columns = [dataset.schema.names[0]] if dataset.schema.names else None
            dataset.scanner(columns=columns, filter=expr)
            return True
        except Exception:
            return False

    def _dataset_supports_expr(
        self, dataset: ds.Dataset | None, expr: ds.Expression
    ) -> bool:
        if dataset is None:
            return False
        try:
            columns = [dataset.schema.names[0]] if dataset.schema.names else None
            dataset.scanner(columns=columns, filter=expr)
            return True
        except Exception:
            return False

    def _data_dataset(self) -> ds.Dataset | None:
        if self.dataset is not None:
            return self.dataset
        if self._collection_path is None:
            return None
        self.dataset = _open_parquet_dataset(self._collection_path)
        return self.dataset

    def _record_index_required_raw_columns(
        self, columns: list[str] | None = None
    ) -> list[str] | None:
        if not self._record_index_path:
            return columns
        if columns is None:
            return None

        raw_columns: set[str] = set()
        for column in columns:
            if column == "assets":
                if self._record_index_href_column:
                    raw_columns.add(self._record_index_href_column)
                elif "assets" in self._open_record_index_dataset().schema.names:
                    raw_columns.add("assets")
                continue
            if column == "proj:epsg":
                raw_columns.add(self._record_index_source_column("proj:epsg"))
                continue
            raw_columns.add(self._record_index_source_column(column))
        return sorted(raw_columns)

    def _read_raw_record_index_table(
        self,
        raw_columns: list[str] | None = None,
        *,
        limit: int | None = None,
    ) -> pa.Table:
        if not self._record_index_path:
            raise ValueError("Collection has no record index")
        if is_hf_dataset_uri(self._record_index_path):
            return load_hf_parquet_table(
                self._record_index_path,
                columns=raw_columns,
                filter_expr=self._record_index_filter_expr,
            )
        dataset = self._open_record_index_dataset()
        if limit is not None:
            return dataset.head(
                limit,
                columns=raw_columns,
                filter=self._record_index_filter_expr,
            )
        return dataset.to_table(
            columns=raw_columns,
            filter=self._record_index_filter_expr,
        )

    def _read_record_index_table(
        self,
        columns: list[str] | None = None,
        *,
        limit: int | None = None,
    ) -> pa.Table:
        if not self._record_index_path:
            raise ValueError("Collection has no record index")
        raw_columns = self._record_index_required_raw_columns(columns)
        return self._read_raw_record_index_table(raw_columns, limit=limit)

    def _prepare_record_index_table(
        self,
        columns: list[str] | None = None,
        *,
        table: pa.Table | None = None,
        limit: int | None = None,
    ) -> pa.Table:
        from rasteret.ingest.parquet_record_table import (
            _apply_column_map_aliases,
            prepare_record_table,
        )

        table = table or self._read_record_index_table(columns=columns, limit=limit)
        table = _apply_column_map_aliases(table, self._record_index_column_map)
        table = prepare_record_table(
            table,
            href_column=self._record_index_href_column,
            band_index_map=self._record_index_band_index_map,
            url_rewrite_patterns=self._record_index_url_rewrite_patterns,
            required_columns=columns,
        )
        if columns is not None:
            keep = [column for column in columns if column in table.column_names]
            table = table.select(keep)
        return table

    def _materialize_record_index_collection(
        self, columns: list[str] | None = None
    ) -> Collection:
        if not self._record_index_path:
            raise ValueError("Collection has no record index")
        from rasteret.ingest.normalize import build_collection_from_table

        table = self._prepare_record_index_table(columns=columns)
        return build_collection_from_table(
            table,
            name=self.name,
            description=self.description,
            data_source=self.data_source,
        )

    def _filtered_data_dataset(self) -> ds.Dataset | None:
        dataset = self._data_dataset()
        if dataset is None:
            return None
        if not self._has_record_index():
            if self._wide_filter_expr is not None:
                return dataset.filter(self._wide_filter_expr)
            return dataset
        if self._record_index_filter_expr is None:
            return (
                dataset.filter(self._wide_filter_expr)
                if self._wide_filter_expr is not None
                else dataset
            )

        raw_columns = [self._record_index_source_column("id")]
        index_dataset = self._open_record_index_dataset()
        if "source_part" in index_dataset.schema.names:
            raw_columns.append("source_part")
        raw_index = self._read_raw_record_index_table(raw_columns)
        if raw_index.num_rows == 0:
            return dataset.filter(ds.field("id").isin(pa.array([], type=pa.string())))

        prepared = self._prepare_record_index_table(columns=["id"], table=raw_index)
        ids = prepared.column("id").combine_chunks()
        if len(ids) == 0:
            return dataset.filter(ds.field("id").isin(pa.array([], type=pa.string())))

        filtered_dataset = dataset
        if "source_part" in raw_index.column_names:
            requested_parts = {
                value
                for value in raw_index.column("source_part").to_pylist()
                if isinstance(value, str) and value
            }
            if requested_parts:
                fragment_paths = [
                    fragment.path
                    for fragment in dataset.get_fragments()
                    if any(
                        fragment.path == part or fragment.path.endswith("/" + part)
                        for part in requested_parts
                    )
                ]
                if fragment_paths:
                    filtered_dataset = ds.dataset(fragment_paths, format="parquet")
        final_filter = ds.field("id").isin(ids)
        if (
            self._wide_filter_expr is None
            and self._record_index_filter_expr is not None
            and self._dataset_supports_expr(
                filtered_dataset, self._record_index_filter_expr
            )
        ):
            final_filter = _and_filters(final_filter, self._record_index_filter_expr)
        final_filter = _and_filters(final_filter, self._wide_filter_expr)
        return filtered_dataset.filter(final_filter)

    @property
    def _schema(self) -> pa.Schema | None:
        dataset = self._data_dataset()
        if dataset is not None:
            return dataset.schema
        if self._hf_streaming is not None:
            return self._hf_streaming.schema
        return None

    def _iter_record_batches(
        self,
        *,
        columns: list[str],
        batch_size: int = 1024,
    ):
        dataset = self._filtered_data_dataset()
        if dataset is not None:
            scanner = dataset.scanner(columns=columns, batch_size=batch_size)
            yield from scanner.to_batches()
            return
        if self._hf_streaming is not None:
            for table in iter_hf_arrow_tables(
                self._hf_streaming,
                columns=columns,
                batch_size=batch_size,
            ):
                yield from table.to_batches()

    @staticmethod
    def _metadata_from_schema(dataset: ds.Dataset) -> dict[str, str]:
        """Extract Rasteret metadata stored by ``export()``."""
        raw = dataset.schema.metadata or {}
        out: dict[str, str] = {}
        for key in (b"name", b"data_source", b"description", b"date_range"):
            val = raw.get(key)
            if val:
                try:
                    out[key.decode()] = val.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    pass
        return out

    @classmethod
    def _load_cached(cls, path: str | Path) -> Collection:
        """Load a Collection from a workspace cache directory.

        Internal fast-path for ``build()`` / ``build_from_table()`` cache
        hits.  Trusts the data (no schema validation), strips workspace
        suffixes (``_stac``, ``_records``) from the name, and detects
        Hive partitioning.

        For user-facing loading, use :meth:`from_parquet` or
        :func:`rasteret.load` instead.
        """
        path_str = str(path)
        if not _is_cloud_uri(path_str):
            p = Path(path_str)
            if not p.exists():
                raise FileNotFoundError(f"Dataset not found at {path_str}")

        try:
            dataset = _open_parquet_dataset(path_str)
        except Exception as exc:
            raise FileNotFoundError(f"Cannot open Parquet at {path_str}") from exc

        meta = cls._metadata_from_schema(dataset)
        stem = _stem_from_path(path_str)
        name = meta.get("name") or stem.removesuffix("_stac").removesuffix("_records")

        start_date = None
        end_date = None
        dr = meta.get("date_range", "")
        if "," in dr:
            s, e = dr.split(",", 1)
            start_date = datetime.fromisoformat(s)
            end_date = datetime.fromisoformat(e)

        return cls(
            dataset=dataset,
            name=name,
            data_source=meta.get("data_source", ""),
            description=meta.get("description", ""),
            start_date=start_date,
            end_date=end_date,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        name: str = "",
        *,
        data_source: str = "",
        defer_dataset_open: bool = False,
        record_index_path: str | None = None,
        record_index_field_roles: dict[str, str] | None = None,
        record_index_column_map: dict[str, str] | None = None,
        record_index_href_column: str | None = None,
        record_index_band_index_map: dict[str, int] | None = None,
        record_index_url_rewrite_patterns: dict[str, str] | None = None,
        record_index_filesystem: Any | None = None,
        surface_fields: dict[str, list[str]] | None = None,
        filter_capabilities: dict[str, list[str]] | None = None,
    ) -> Collection:
        """Load a Collection from any Parquet file or directory.

        Accepts local paths **and** cloud URIs (``s3://``, ``gs://``).
        Tries Hive-style partitioning first (year/month), falls back to
        plain Parquet.  Validates that the core contract columns are present.

        See the `Schema Contract <../explanation/schema-contract/>`_ docs page.
        """
        path_str = str(path)
        if not _is_cloud_uri(path_str):
            p = Path(path_str)
            if not p.exists():
                raise FileNotFoundError(f"Parquet not found at {path_str}")

        if is_hf_dataset_uri(path_str):
            try:
                hf_streaming = open_hf_streaming_source(path_str)
            except Exception as exc:
                raise FileNotFoundError(f"Cannot open Parquet at {path_str}") from exc

            required = {"id", "datetime", "geometry", "assets"}
            missing = required - set(hf_streaming.schema.names)
            if missing or _bbox_struct_field(hf_streaming.schema) is None:
                raise ValueError(
                    f"Parquet is missing required columns: {missing or {'bbox'}}. "
                    "See the Schema Contract page in docs for the expected schema."
                )

            return cls(
                hf_streaming=hf_streaming,
                name=name or _stem_from_path(path_str),
                data_source=data_source,
                record_index_path=record_index_path,
                record_index_field_roles=record_index_field_roles,
                record_index_column_map=record_index_column_map,
                record_index_href_column=record_index_href_column,
                record_index_band_index_map=record_index_band_index_map,
                record_index_url_rewrite_patterns=record_index_url_rewrite_patterns,
                record_index_filesystem=record_index_filesystem,
                surface_fields=surface_fields,
                filter_capabilities=filter_capabilities,
            )

        dataset = None
        meta: dict[str, str] = {}
        if not defer_dataset_open:
            try:
                dataset = _open_parquet_dataset(path_str)
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise FileNotFoundError(f"Cannot open Parquet at {path_str}") from exc

            required = {"id", "datetime", "geometry", "assets"}
            missing = required - set(dataset.schema.names)
            if missing or _bbox_struct_field(dataset.schema) is None:
                raise ValueError(
                    f"Parquet is missing required columns: {missing or {'bbox'}}. "
                    "See the Schema Contract page in docs for the expected schema."
                )

            meta = cls._metadata_from_schema(dataset)
        resolved_name = name or meta.get("name") or _stem_from_path(path_str)

        start_date = None
        end_date = None
        dr = meta.get("date_range", "")
        if "," in dr:
            s, e = dr.split(",", 1)
            start_date = datetime.fromisoformat(s)
            end_date = datetime.fromisoformat(e)

        return cls(
            dataset=dataset,
            collection_path=path_str if defer_dataset_open else None,
            record_index_path=record_index_path,
            record_index_field_roles=record_index_field_roles,
            record_index_column_map=record_index_column_map,
            record_index_href_column=record_index_href_column,
            record_index_band_index_map=record_index_band_index_map,
            record_index_url_rewrite_patterns=record_index_url_rewrite_patterns,
            record_index_filesystem=record_index_filesystem,
            surface_fields=surface_fields,
            filter_capabilities=filter_capabilities,
            name=resolved_name,
            data_source=data_source or meta.get("data_source", ""),
            description=meta.get("description", ""),
            start_date=start_date,
            end_date=end_date,
        )

    def subset(
        self,
        *,
        cloud_cover_lt: float | None = None,
        date_range: tuple[str, str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        geometries: Any = None,
        split: str | Sequence[str] | None = None,
        split_column: str = "split",
    ) -> Collection:
        """Return a filtered view of this Collection.

        All provided criteria are combined with AND.

        Parameters
        ----------
        cloud_cover_lt : float, optional
            Keep records with ``eo:cloud_cover`` below this value (0--100).
        date_range : tuple of str, optional
            ``(start, end)`` ISO date strings for temporal filtering.
        bbox : tuple of float, optional
            ``(minx, miny, maxx, maxy)`` bounding box filter.
        geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict, optional
            Spatial filter; records whose bbox overlaps any geometry are kept.
            Accepts ``(minx, miny, maxx, maxy)`` bbox tuples, Arrow arrays
            (e.g. a geometry column read from GeoParquet), Shapely objects,
            raw WKB bytes, or GeoJSON dicts.
        split : str or sequence of str, optional
            Keep only rows matching the given split value(s).
        split_column : str
            Column name holding split labels. Defaults to ``"split"``.

        Returns
        -------
        Collection
            A new Collection with the filtered dataset view.
        """
        if self._hf_streaming is not None:
            if all(
                value is None
                for value in (
                    cloud_cover_lt,
                    date_range,
                    bbox,
                    geometries,
                    split,
                )
            ):
                raise ValueError("No filters provided")
            return self._view(
                hf_streaming=subset_hf_streaming_source(
                    self._hf_streaming,
                    cloud_cover_lt=cloud_cover_lt,
                    date_range=date_range,
                    bbox=bbox,
                    geometries=geometries,
                    split=split,
                    split_column=split_column,
                )
            )

        if self._has_record_index():
            filter_expr = self._record_index_filter_expr
            wide_filter_expr = self._wide_filter_expr
            index_dataset = self._open_record_index_dataset()
            wide_dataset = self.dataset
            index_schema = index_dataset.schema
            wide_schema = wide_dataset.schema if wide_dataset is not None else None

            if all(
                value is None
                for value in (
                    cloud_cover_lt,
                    date_range,
                    bbox,
                    geometries,
                    split,
                )
            ):
                raise ValueError("No filters provided")

            if cloud_cover_lt is not None:
                if not self._surface_supports_filter(
                    "index",
                    "eo:cloud_cover",
                    schema=index_schema,
                ):
                    filtered_dataset = self._filtered_data_dataset()
                    return self._view(
                        filtered_dataset.filter(
                            ds.field("eo:cloud_cover") < float(cloud_cover_lt)
                        )
                        if filtered_dataset is not None
                        else None,
                        record_index_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                        wide_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                        drop_record_index=True,
                    )
                if not isinstance(cloud_cover_lt, (int, float)) or not (
                    0 <= cloud_cover_lt <= 100
                ):
                    raise ValueError(
                        f"Invalid cloud_cover_lt={cloud_cover_lt!r}: must be between 0 and 100."
                    )
                filter_expr = _and_filters(
                    filter_expr, ds.field("eo:cloud_cover") < float(cloud_cover_lt)
                )
                if self._surface_supports_filter(
                    "collection",
                    "eo:cloud_cover",
                    schema=wide_schema,
                ):
                    wide_filter_expr = _and_filters(
                        wide_filter_expr,
                        ds.field("eo:cloud_cover") < float(cloud_cover_lt),
                    )

            if date_range is not None:
                start_raw, end_raw = date_range
                if not start_raw or not end_raw:
                    raise ValueError("Invalid date range")
                start = pd.Timestamp(start_raw)
                end = pd.Timestamp(end_raw)
                if start > end:
                    raise ValueError("Invalid date range")
                datetime_source = self._record_index_source_column("datetime")
                if datetime_source not in index_schema.names:
                    raise ValueError("Collection has no datetime data")
                dt_type = index_schema.field(datetime_source).type
                if pa.types.is_integer(dt_type):
                    filter_expr = _and_filters(
                        filter_expr, ds.field(datetime_source) >= int(start.year)
                    )
                    filter_expr = _and_filters(
                        filter_expr, ds.field(datetime_source) <= int(end.year)
                    )
                else:
                    start_scalar = pa.scalar(start.to_pydatetime(), type=dt_type)
                    end_scalar = pa.scalar(end.to_pydatetime(), type=dt_type)
                    filter_expr = _and_filters(
                        filter_expr,
                        (ds.field(datetime_source) >= start_scalar)
                        & (ds.field(datetime_source) <= end_scalar),
                    )
                if (
                    self._surface_supports_filter(
                        "collection",
                        "datetime",
                        schema=wide_schema,
                    )
                    and wide_schema is not None
                    and "datetime" in wide_schema.names
                ):
                    wide_ts_type = wide_schema.field("datetime").type
                    start_scalar = pa.scalar(start.to_pydatetime(), type=wide_ts_type)
                    end_scalar = pa.scalar(end.to_pydatetime(), type=wide_ts_type)
                    wide_filter_expr = _and_filters(
                        wide_filter_expr,
                        (ds.field("datetime") >= start_scalar)
                        & (ds.field("datetime") <= end_scalar),
                    )
                if self._surface_has_field("collection", "year", schema=wide_schema):
                    wide_filter_expr = _and_filters(
                        wide_filter_expr, ds.field("year") >= int(start.year)
                    )
                    wide_filter_expr = _and_filters(
                        wide_filter_expr, ds.field("year") <= int(end.year)
                    )

            if bbox is not None:
                if not self._surface_supports_filter(
                    "index", "bbox", schema=index_schema
                ):
                    raise ValueError(
                        "bbox filtering requires a root-level 'bbox' struct with "
                        "xmin/ymin/xmax/ymax children."
                    )
                if len(bbox) != 4:
                    raise ValueError("Invalid bbox format")
                minx, miny, maxx, maxy = bbox
                if minx > maxx or miny > maxy:
                    raise ValueError("Invalid bbox coordinates")
                filter_expr = _and_filters(
                    filter_expr,
                    _bbox_overlap_expr(
                        minx,
                        miny,
                        maxx,
                        maxy,
                        field_name=self._record_index_source_column("bbox"),
                    ),
                )
                if self._surface_supports_filter(
                    "collection",
                    "bbox",
                    schema=wide_schema,
                ):
                    wide_filter_expr = _and_filters(
                        wide_filter_expr, _bbox_overlap_expr(minx, miny, maxx, maxy)
                    )

            if geometries is not None:
                if not self._surface_supports_filter(
                    "index", "bbox", schema=index_schema
                ):
                    raise ValueError(
                        "geometry filtering requires a root-level 'bbox' struct with "
                        "xmin/ymin/xmax/ymax children."
                    )
                from rasteret.core.geometry import bbox_array, coerce_to_geoarrow

                geo_arr = coerce_to_geoarrow(geometries)
                xmin, ymin, xmax, ymax = bbox_array(geo_arr)
                geometry_filter: ds.Expression | None = None
                for i in range(len(xmin)):
                    geom_expr = _bbox_overlap_expr(
                        xmin[i].as_py(),
                        ymin[i].as_py(),
                        xmax[i].as_py(),
                        ymax[i].as_py(),
                        field_name=self._record_index_source_column("bbox"),
                    )
                    geometry_filter = (
                        geom_expr
                        if geometry_filter is None
                        else (geometry_filter | geom_expr)
                    )
                filter_expr = _and_filters(filter_expr, geometry_filter)
                if geometry_filter is not None and self._surface_supports_filter(
                    "collection",
                    "bbox",
                    schema=wide_schema,
                ):
                    wide_filter_expr = _and_filters(wide_filter_expr, geometry_filter)

            if split is not None:
                if split_column not in index_schema.names:
                    filtered_dataset = self._filtered_data_dataset()
                    return self._view(
                        Collection(dataset=filtered_dataset)
                        .subset(split=split, split_column=split_column)
                        .dataset
                        if filtered_dataset is not None
                        else None,
                        record_index_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                        wide_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                        drop_record_index=True,
                    )
                if isinstance(split, str):
                    split_expr = ds.field(split_column) == split
                elif (
                    isinstance(split, Sequence)
                    and not isinstance(split, (str, bytes))
                    and split
                    and all(isinstance(value, str) for value in split)
                ):
                    split_expr = ds.field(split_column).isin(list(split))
                else:
                    raise ValueError(
                        "Invalid split filter. Use a split name or sequence of split names."
                    )
                filter_expr = _and_filters(filter_expr, split_expr)
                if self._surface_supports_filter(
                    "collection",
                    split_column,
                    schema=wide_schema,
                ):
                    wide_filter_expr = _and_filters(wide_filter_expr, split_expr)

            return self._view(
                self.dataset,
                record_index_filter_expr=filter_expr,
                wide_filter_expr=wide_filter_expr,
            )

        if self.dataset is None:
            return self

        filter_expr: ds.Expression | None = None

        def _and(current: ds.Expression | None, new: ds.Expression) -> ds.Expression:
            return new if current is None else current & new

        if cloud_cover_lt is not None:
            if "eo:cloud_cover" not in self.dataset.schema.names:
                raise ValueError("Collection has no cloud cover data")
            if not isinstance(cloud_cover_lt, (int, float)) or not (
                0 <= cloud_cover_lt <= 100
            ):
                raise ValueError(
                    f"Invalid cloud_cover_lt={cloud_cover_lt!r}: must be between 0 and 100."
                )
            filter_expr = _and(
                filter_expr, ds.field("eo:cloud_cover") < float(cloud_cover_lt)
            )

        if date_range is not None:
            if "datetime" not in self.dataset.schema.names:
                raise ValueError("Collection has no datetime data")
            start_raw, end_raw = date_range
            if not start_raw or not end_raw:
                raise ValueError("Invalid date range")
            start = pd.Timestamp(start_raw)
            end = pd.Timestamp(end_raw)
            if start > end:
                raise ValueError("Invalid date range")

            ts_type = self.dataset.schema.field("datetime").type
            if not pa.types.is_timestamp(ts_type):
                raise ValueError("Collection datetime column is not a timestamp")
            start_scalar = pa.scalar(start.to_pydatetime(), type=ts_type)
            end_scalar = pa.scalar(end.to_pydatetime(), type=ts_type)
            date_filter = (ds.field("datetime") >= start_scalar) & (
                ds.field("datetime") <= end_scalar
            )
            filter_expr = _and(filter_expr, date_filter)

        if bbox is not None:
            if _bbox_struct_field(self.dataset.schema) is None:
                raise ValueError(
                    "bbox filtering requires a root-level 'bbox' struct with "
                    "xmin/ymin/xmax/ymax children. "
                    "Rebuild or re-normalize the collection with rasteret>=1.0.0."
                )
            if len(bbox) != 4:
                raise ValueError("Invalid bbox format")
            minx, miny, maxx, maxy = bbox
            if minx > maxx or miny > maxy:
                raise ValueError("Invalid bbox coordinates")
            filter_expr = _and(filter_expr, _bbox_overlap_expr(minx, miny, maxx, maxy))

        if geometries is not None:
            if _bbox_struct_field(self.dataset.schema) is None:
                raise ValueError(
                    "geometry filtering requires a root-level 'bbox' struct with "
                    "xmin/ymin/xmax/ymax children. "
                    "Rebuild or re-normalize the collection with rasteret>=1.0.0."
                )
            from rasteret.core.geometry import bbox_array, coerce_to_geoarrow

            geo_arr = coerce_to_geoarrow(geometries)
            xmin, ymin, xmax, ymax = bbox_array(geo_arr)

            geometry_filter: ds.Expression | None = None
            for i in range(len(xmin)):
                geom_expr = _bbox_overlap_expr(
                    xmin[i].as_py(),
                    ymin[i].as_py(),
                    xmax[i].as_py(),
                    ymax[i].as_py(),
                )
                geometry_filter = (
                    geom_expr
                    if geometry_filter is None
                    else (geometry_filter | geom_expr)
                )
            if geometry_filter is not None:
                filter_expr = _and(filter_expr, geometry_filter)

        if split is not None:
            if split_column not in self.dataset.schema.names:
                raise ValueError(f"Collection has no split column: '{split_column}'")
            if isinstance(split, str):
                split_expr = ds.field(split_column) == split
            elif (
                isinstance(split, Sequence)
                and not isinstance(split, (str, bytes))
                and split
                and all(isinstance(value, str) for value in split)
            ):
                split_expr = ds.field(split_column).isin(list(split))
            else:
                raise ValueError(
                    "Invalid split filter. Use a split name or sequence of split names."
                )
            filter_expr = _and(filter_expr, split_expr)

        if filter_expr is None:
            raise ValueError("No filters provided")

        return self._view(self.dataset.filter(filter_expr))

    def select_split(
        self,
        split: str | Sequence[str],
        *,
        split_column: str = "split",
    ) -> Collection:
        """Return a split-filtered view of this Collection.

        This is a convenience wrapper around ``subset(split=...)`` to keep the
        intent obvious in training code.
        """
        return self.subset(split=split, split_column=split_column)

    def where(self, expr: ds.Expression) -> Collection:
        """Return a filtered view using a raw Arrow dataset expression."""
        if self._hf_streaming is not None:
            raise NotImplementedError(
                "where(expr) is not supported for HF streaming collections. "
                "Use subset(...) with managed filters instead."
            )
        if self._has_record_index():
            index_expr = expr if self._record_index_supports_expr(expr) else None
            wide_expr = (
                expr if self._dataset_supports_expr(self.dataset, expr) else None
            )
            if index_expr is None and wide_expr is None:
                raise ValueError("where(expr) could not be applied to the collection")
            if index_expr is not None:
                return self._view(
                    self.dataset,
                    record_index_filter_expr=_and_filters(
                        self._record_index_filter_expr, index_expr
                    ),
                    wide_filter_expr=_and_filters(self._wide_filter_expr, wide_expr),
                )
            filtered_dataset = self._filtered_data_dataset()
            if filtered_dataset is None:
                return self
            return self._view(
                filtered_dataset.filter(expr),
                record_index_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                wide_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                drop_record_index=True,
            )
        if self.dataset is None:
            return self
        return self._view(self.dataset.filter(expr))

    def head(self, n: int = 5, columns: list[str] | None = None) -> pa.Table:
        """Return the first *n* metadata rows as a PyArrow table."""
        if n < 0:
            raise ValueError("head() requires n >= 0")
        if self._has_record_index():
            return self._prepare_record_index_table(columns=columns, limit=n)
        if self.dataset is not None:
            return self.dataset.head(n, columns=columns)
        if self._hf_streaming is not None:
            return head_hf_streaming_source(self._hf_streaming, n=n, columns=columns)
        schema = (
            pa.schema([])
            if columns is None
            else pa.schema([pa.field(name, pa.null()) for name in columns])
        )
        return schema.empty_table()

    @classmethod
    def list_collections(
        cls, workspace_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """List cached collections with summary metadata.

        Parameters
        ----------
        workspace_dir : Path, optional
            Directory to scan for cached collections. Defaults to
            ``~/rasteret_workspace``.

        Returns
        -------
        list of dict
            Each dict contains ``name``, ``kind``, ``data_source``,
            ``date_range``, ``size``, and ``created``.
        """
        if workspace_dir is None:
            workspace_dir = Path.home() / "rasteret_workspace"

        def _date_range(dataset: ds.Dataset) -> tuple[str, str] | None:
            if "datetime" not in dataset.schema.names:
                return None
            scanner = dataset.scanner(columns=["datetime"])
            min_value = None
            max_value = None
            for batch in scanner.to_batches():
                if batch.num_rows == 0:
                    continue
                column = batch.column(0)
                batch_min = pc.min(column).as_py()
                batch_max = pc.max(column).as_py()
                if batch_min is not None:
                    min_value = (
                        batch_min if min_value is None else min(min_value, batch_min)
                    )
                if batch_max is not None:
                    max_value = (
                        batch_max if max_value is None else max(max_value, batch_max)
                    )
            if min_value is None or max_value is None:
                return None
            return (min_value.date().isoformat(), max_value.date().isoformat())

        collections: list[dict[str, Any]] = []

        def _data_source_from_metadata(dataset: ds.Dataset) -> str | None:
            metadata = dataset.schema.metadata or {}
            value = metadata.get(b"data_source")
            if not value:
                return None
            try:
                decoded = value.decode("utf-8").strip()
            except (UnicodeDecodeError, AttributeError):
                return None
            return decoded or None

        # Look for cached directories
        for suffix in ("_stac", "_records"):
            dirs = workspace_dir.glob(f"*{suffix}")
            for cache_dir in dirs:
                try:
                    try:
                        dataset = ds.dataset(
                            str(cache_dir), format="parquet", partitioning="hive"
                        )
                    except pa.ArrowInvalid:
                        dataset = ds.dataset(str(cache_dir), format="parquet")
                    name = cache_dir.name.removesuffix(suffix)
                    date_range = _date_range(dataset)
                    data_source = _data_source_from_metadata(dataset) or (
                        name.split("_")[-1] if "_" in name else "unknown"
                    )

                    collections.append(
                        {
                            "name": name,
                            "kind": suffix.removeprefix("_"),
                            "data_source": data_source,
                            "date_range": date_range,
                            "size": dataset.count_rows(),
                            "created": cache_dir.stat().st_ctime,
                        }
                    )

                except (pa.ArrowInvalid, OSError) as exc:
                    logger.debug("Failed to read collection %s: %s", cache_dir, exc)
                    continue

        return collections

    def export(
        self,
        path: str | Path,
        partition_by: Sequence[str] = ("year", "month"),
    ) -> None:
        """Export the collection as a partitioned Parquet dataset.

        Use this to produce a portable copy of the collection that can
        be shared with teammates via :func:`rasteret.load`.

        Parameters
        ----------
        path : str or Path
            Output directory.  Accepts local paths and cloud URIs
            (``s3://``, ``gs://``).
        partition_by : sequence of str
            Columns to partition by. Defaults to ``("year", "month")``.
        """
        path_str = str(path)
        if not _is_cloud_uri(path_str):
            Path(path_str).mkdir(parents=True, exist_ok=True)

        if self.dataset is None:
            raise ValueError("No Pyarrow dataset provided")

        table = self.dataset.to_table()
        if _bbox_struct_field(table.schema) is None:
            bbox_idx = table.schema.get_field_index("bbox")
            if bbox_idx >= 0:
                bbox_field = table.schema.field(bbox_idx)
                if (
                    pa.types.is_list(bbox_field.type)
                    or pa.types.is_large_list(bbox_field.type)
                    or pa.types.is_fixed_size_list(bbox_field.type)
                ):
                    bbox_col = table.column(bbox_idx).combine_chunks()
                    bbox_struct = pa.StructArray.from_arrays(
                        [
                            pc.list_element(bbox_col, 0),
                            pc.list_element(bbox_col, 1),
                            pc.list_element(bbox_col, 2),
                            pc.list_element(bbox_col, 3),
                        ],
                        fields=[
                            pa.field("xmin", pa.float64()),
                            pa.field("ymin", pa.float64()),
                            pa.field("xmax", pa.float64()),
                            pa.field("ymax", pa.float64()),
                        ],
                    )
                    table = table.set_column(
                        bbox_idx,
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
                        bbox_struct,
                    )
            elif "geometry" in table.schema.names:
                from rasteret.ingest.normalize import _add_bbox_struct

                table = _add_bbox_struct(table)

        # Enhanced metadata with fallbacks
        custom_metadata = {
            b"description": (
                self.description.encode("utf-8") if self.description else b""
            ),
            b"created": datetime.now().isoformat().encode("utf-8"),
            b"name": self.name.encode("utf-8") if self.name else b"",
            b"data_source": (
                self.data_source.encode("utf-8") if self.data_source else b""
            ),
            b"date_range": (
                f"{self.start_date.isoformat()},{self.end_date.isoformat()}".encode(
                    "utf-8"
                )
                if self.start_date and self.end_date
                else b""
            ),
            b"rasteret_collection_version": b"1",
        }

        # Merge with existing metadata
        merged_metadata = {**custom_metadata, **(table.schema.metadata or {})}

        # GeoParquet metadata: declare the geometry column as WKB.
        #
        # Rasteret stores footprint geometries in CRS84 (lon/lat) for portability.
        # GeoParquet 1.1 treats missing `crs` as CRS84 by default.
        if "geometry" in table.schema.names and b"geo" not in merged_metadata:
            geom_types = _geometry_types_from_wkb(table.column("geometry"))
            geo = {
                "version": "1.1.0",
                "primary_column": "geometry",
                "columns": {
                    "geometry": {
                        "encoding": "WKB",
                        "geometry_types": geom_types,
                    }
                },
            }
            if _bbox_struct_field(table.schema) is not None:
                geo["columns"]["geometry"]["covering"] = {
                    "bbox": {
                        "xmin": ["bbox", "xmin"],
                        "ymin": ["bbox", "ymin"],
                        "xmax": ["bbox", "xmax"],
                        "ymax": ["bbox", "ymax"],
                    }
                }
            merged_metadata[b"geo"] = json.dumps(
                geo, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")

        table_with_metadata = table.replace_schema_metadata(merged_metadata)

        # Write dataset
        pq.write_to_dataset(
            table_with_metadata,
            root_path=path_str,
            partition_cols=partition_by,
            compression="zstd",
            compression_level=3,
            row_group_size=50_000,
            write_statistics=True,
            use_dictionary=True,
            write_batch_size=10000,
            basename_template="part-{i}.parquet",
        )

    async def iterate_rasters(
        self,
        data_source: str | None = None,
        bands: list[str] | None = None,
    ) -> AsyncIterator[RasterAccessor]:
        """Iterate through raster records in this Collection.

        Each Parquet row becomes a :class:`RasterAccessor` that provides
        async band-loading methods.

        Parameters
        ----------
        data_source : str, optional
            Data source identifier for band mapping. Defaults to
            ``self.data_source`` or inferred from the dataset.

        Yields
        ------
        RasterAccessor
        """
        required_fields = {"id", "datetime", "geometry", "assets", "bbox"}

        schema = self._schema
        if schema is None:
            return

        # Check required fields
        missing = required_fields - set(schema.names)
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        resolved_source = data_source or self.data_source or ""
        schema_names = set(schema.names)
        band_metadata_cols = [
            name for name in schema.names if name.endswith("_metadata")
        ]
        optional_cols = [
            name
            for name in ("proj:epsg", "eo:cloud_cover", "collection")
            if name in schema_names
        ]
        requested_band_metadata_cols: list[str] | None = None
        if bands:
            requested_band_metadata_cols = [
                f"{band}_metadata"
                for band in bands
                if f"{band}_metadata" in schema_names
            ]
        scan_cols = [
            "id",
            "datetime",
            "geometry",
            "assets",
            "bbox",
            *optional_cols,
            *(
                requested_band_metadata_cols
                if requested_band_metadata_cols is not None
                else band_metadata_cols
            ),
        ]

        batch_source: Collection = self
        if self.dataset is not None:
            scan_dataset = self._filtered_data_dataset()
            if scan_dataset is None:
                return
            batch_source = self._view(scan_dataset)

        for batch in batch_source._iter_record_batches(columns=scan_cols):
            ids = batch.column(batch.schema.get_field_index("id"))
            datetimes = batch.column(batch.schema.get_field_index("datetime"))
            geometries = batch.column(batch.schema.get_field_index("geometry"))
            assets = batch.column(batch.schema.get_field_index("assets"))
            bbox_col = batch.column(batch.schema.get_field_index("bbox"))

            crs_col = (
                batch.column(batch.schema.get_field_index("proj:epsg"))
                if "proj:epsg" in batch.schema.names
                else None
            )
            cloud_col = (
                batch.column(batch.schema.get_field_index("eo:cloud_cover"))
                if "eo:cloud_cover" in batch.schema.names
                else None
            )
            collection_col = (
                batch.column(batch.schema.get_field_index("collection"))
                if "collection" in batch.schema.names
                else None
            )
            band_cols = {
                name: batch.column(batch.schema.get_field_index(name))
                for name in (
                    requested_band_metadata_cols
                    if requested_band_metadata_cols is not None
                    else band_metadata_cols
                )
                if name in batch.schema.names
            }

            for idx in range(batch.num_rows):
                try:
                    band_metadata: dict[str, Any] = {}
                    for key, col in band_cols.items():
                        val = col[idx]
                        if val.is_valid:
                            py_val = val.as_py()
                            if py_val is not None:
                                band_metadata[key] = py_val

                    info = RasterInfo(
                        id=ids[idx].as_py(),
                        datetime=datetimes[idx].as_py(),
                        footprint=geometries[idx].as_py(),
                        bbox=_bbox_value_to_list(bbox_col[idx].as_py()) or [],
                        crs=crs_col[idx].as_py() if crs_col is not None else None,
                        cloud_cover=(
                            cloud_col[idx].as_py()
                            if cloud_col is not None and cloud_col[idx].is_valid
                            else 0
                        ),
                        assets=assets[idx].as_py(),
                        band_metadata=band_metadata,
                        collection=(
                            collection_col[idx].as_py()
                            if collection_col is not None
                            and collection_col[idx].is_valid
                            else resolved_source
                        ),
                    )
                    yield RasterAccessor(info, resolved_source)
                except (KeyError, TypeError, ValueError):
                    logger.exception(
                        "Failed to create RasterAccessor from collection row"
                    )
                    continue

    async def get_first_raster(self) -> RasterAccessor:
        """Return the first raster record in the collection.

        Returns
        -------
        RasterAccessor
        """
        async for raster in self.iterate_rasters():
            return raster
        raise ValueError("No raster records found in collection")

    def __len__(self) -> int:
        """Number of scene records in this collection."""
        if self._hf_streaming is not None:
            raise TypeError(
                "len() is not available for HF streaming collections. "
                "Use head() or explicit scans instead."
            )
        if self._has_record_index():
            if is_hf_dataset_uri(self._record_index_path or ""):
                return self._read_record_index_table(
                    columns=[self._record_index_source_column("id")]
                ).num_rows
            dataset = self._open_record_index_dataset()
            return dataset.count_rows(filter=self._record_index_filter_expr)
        if self.dataset is None:
            return 0
        return self.dataset.count_rows()

    @property
    def bands(self) -> list[str]:
        """Available band codes in this collection."""
        if self._has_record_index() and self._record_index_band_index_map:
            return list(self._record_index_band_index_map.keys())
        schema = self._schema
        if schema is None:
            return []
        return [
            c.removesuffix("_metadata") for c in schema.names if c.endswith("_metadata")
        ]

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """Spatial extent as ``(minx, miny, maxx, maxy)`` or ``None``."""
        if self._has_record_index():
            table = self._prepare_record_index_table(columns=["bbox"])
            if table.num_rows == 0 or "bbox" not in table.column_names:
                return None
            bbox_col = table.column("bbox")
            minx = pc.min(pc.struct_field(bbox_col, "xmin")).as_py()
            miny = pc.min(pc.struct_field(bbox_col, "ymin")).as_py()
            maxx = pc.max(pc.struct_field(bbox_col, "xmax")).as_py()
            maxy = pc.max(pc.struct_field(bbox_col, "ymax")).as_py()
            if None in (minx, miny, maxx, maxy):
                return None
            return (float(minx), float(miny), float(maxx), float(maxy))
        schema = self._schema
        if schema is None:
            return None
        if _bbox_struct_field(schema) is None:
            return None
        minx: float | None = None
        miny: float | None = None
        maxx: float | None = None
        maxy: float | None = None

        for batch in self._iter_record_batches(columns=["bbox"]):
            if batch.num_rows == 0:
                continue

            bbox_col = batch.column(batch.schema.get_field_index("bbox"))
            bminx = pc.min(pc.struct_field(bbox_col, "xmin")).as_py()
            bminy = pc.min(pc.struct_field(bbox_col, "ymin")).as_py()
            bmaxx = pc.max(pc.struct_field(bbox_col, "xmax")).as_py()
            bmaxy = pc.max(pc.struct_field(bbox_col, "ymax")).as_py()

            if bminx is not None:
                minx = bminx if minx is None else min(minx, bminx)
            if bminy is not None:
                miny = bminy if miny is None else min(miny, bminy)
            if bmaxx is not None:
                maxx = bmaxx if maxx is None else max(maxx, bmaxx)
            if bmaxy is not None:
                maxy = bmaxy if maxy is None else max(maxy, bmaxy)

        if None in (minx, miny, maxx, maxy):
            return None
        return (float(minx), float(miny), float(maxx), float(maxy))

    @property
    def epsg(self) -> list[int]:
        """Unique EPSG codes in this collection."""
        if self._has_record_index():
            try:
                table = self._prepare_record_index_table(columns=["proj:epsg"])
            except Exception:
                table = None
            if table is not None and "proj:epsg" in table.column_names:
                from rasteret.ingest.normalize import parse_epsg

                col = table.column("proj:epsg")
                unique = pc.unique(pc.drop_null(col))
                codes = {
                    int(parsed)
                    for value in unique
                    if value.is_valid
                    for parsed in [parse_epsg(value.as_py())]
                    if parsed is not None
                }
                return sorted(codes)
        schema = self._schema
        if schema is None:
            return []
        if "proj:epsg" not in schema.names:
            return []
        codes: set[int] = set()
        for batch in self._iter_record_batches(columns=["proj:epsg"]):
            col = batch.column(batch.schema.get_field_index("proj:epsg"))
            unique = pc.unique(pc.drop_null(col))
            for value in unique:
                if value.is_valid:
                    codes.add(int(value.as_py()))
        return sorted(codes)

    def __repr__(self) -> str:
        n_bands = len(self.bands)
        try:
            n_rows = len(self)
        except Exception:
            n_rows = "?"

        parts = [f"Collection({self.name!r}"]
        if self.data_source:
            parts.append(f"source={self.data_source!r}")
        parts.append(f"bands={n_bands}")
        parts.append(f"records={n_rows}")
        epsg = self.epsg
        if len(epsg) == 1:
            parts.append(f"crs={epsg[0]}")
        elif epsg:
            parts.append(f"crs={epsg}")
        if self.start_date and self.end_date:
            s = str(self.start_date)[:10]
            e = str(self.end_date)[:10]
            parts.append(f"{s}..{e}")
        return ", ".join(parts) + ")"

    def describe(self) -> DescribeResult:
        """Summary of this collection.

        Returns a :class:`~rasteret.core.display.DescribeResult` that renders
        as a clean table in terminals and as styled HTML in notebooks
        (Jupyter, marimo, Colab).

        The underlying data is accessible via ``.data`` or ``["key"]``.

        Examples
        --------
        >>> collection.describe()           # pretty table in REPL
        >>> collection.describe()["bands"]  # programmatic access
        >>> collection.describe().data      # full dict
        """
        from rasteret.core.display import build_describe_result

        dates = None
        if self.start_date and self.end_date:
            dates = (str(self.start_date)[:10], str(self.end_date)[:10])
        try:
            records = len(self)
        except Exception:
            records = "?"
        return build_describe_result(
            name=self.name,
            records=records,
            bands=self.bands,
            bounds=self.bounds,
            crs=self.epsg,
            dates=dates,
            source=self.data_source,
        )

    def _resolve_catalog_descriptor(self) -> Any | None:
        """Look up the catalog DatasetDescriptor for this collection."""
        if not self.data_source:
            return None
        from rasteret.catalog import DatasetRegistry

        desc = DatasetRegistry.get(self.data_source)
        if desc is None:
            for d in DatasetRegistry.list():
                if d.stac_collection == self.data_source:
                    return d
        return desc

    def compare_to_catalog(self) -> DescribeResult:
        """Compare this collection against its catalog source.

        Shows collection properties side-by-side with the catalog entry
        (bands coverage, date range vs source range, spatial coverage,
        auth requirements).

        Raises :class:`ValueError` if the collection has no catalog match.

        Returns a :class:`~rasteret.core.display.DescribeResult` that renders
        as a table in terminals and styled HTML in notebooks.

        Examples
        --------
        >>> collection.compare_to_catalog()        # pretty comparison table
        >>> collection.compare_to_catalog().data    # full dict with catalog info
        """
        from rasteret.core.display import build_catalog_comparison

        desc = self._resolve_catalog_descriptor()
        if desc is None:
            raise ValueError(
                f"No catalog entry found for data_source={self.data_source!r}. "
                "Use describe() for collection-only summary."
            )

        dates = None
        if self.start_date and self.end_date:
            dates = (str(self.start_date)[:10], str(self.end_date)[:10])

        return build_catalog_comparison(
            name=self.name,
            records=self.describe()["records"],
            bands=self.bands,
            bounds=self.bounds,
            crs=self.epsg,
            dates=dates,
            source=self.data_source,
            catalog_name=desc.name,
            catalog_bands=list(desc.band_map) if desc.band_map else [],
            catalog_temporal=desc.temporal_range,
            catalog_coverage=desc.spatial_coverage,
            catalog_auth=desc.requires_auth,
            catalog_license=desc.license,
        )

    def _validate_bands(self, bands: list[str]) -> None:
        """Raise ``ValueError`` eagerly if *bands* contains unknown names."""
        available = self.bands
        if not available:
            return
        invalid = [b for b in bands if b not in available]
        if invalid:
            raise ValueError(
                f"Band(s) not found: {invalid}. Available bands: {available}"
            )

    def _validate_parquet_dataset(self) -> None:
        """Basic dataset validation."""
        if not isinstance(self.dataset, ds.Dataset):
            raise TypeError("Expected pyarrow.dataset.Dataset")

    def _extract_band_metadata(self, row: dict) -> dict:
        """Extract band metadata from row."""
        return {k: v for k, v in row.items() if k.endswith("_metadata")}

    @classmethod
    def _format_date_range(
        cls, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> str:
        """Format date range for collection name."""
        if start_date.year == end_date.year:
            return f"{start_date.strftime('%Y%m')}-{end_date.strftime('%m')}"
        return f"{start_date.strftime('%Y%m')}-{end_date.strftime('%Y%m')}"

    @classmethod
    def _source_token(cls, data_source: str) -> str:
        """Convert data source id into a stable name token.

        Uses short tokens for canonical built-ins to keep cache paths
        compact, while preserving more detail for other sources
        (e.g. ``sentinel-1-grd``).
        """
        source = (data_source or "").strip().lower()
        short_names = {
            "sentinel-2-l2a": "sentinel",
            "landsat-c2-l2": "landsat",
        }
        if source in short_names:
            return short_names[source]

        # Use the source itself, normalised for filesystem safety.
        source = source.replace("/", "-")
        source = source.replace(" ", "-")
        source = re.sub(r"[^a-z0-9-]+", "-", source)
        source = re.sub(r"-{2,}", "-", source).strip("-")
        return source or "source"

    @classmethod
    def create_name(
        cls, custom_name: str, date_range: tuple[str, str], data_source: str
    ) -> str:
        """Create a standardized collection name.

        Parameters
        ----------
        custom_name : str
            User-chosen name component. Underscores are normalised to dashes.
        date_range : tuple of str
            ``(start, end)`` ISO date strings.
        data_source : str
            Data source identifier (e.g. ``"sentinel-2-l2a"``).

        Returns
        -------
        str
            Name in the format ``{custom}_{daterange}_{source}``.
        """
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        custom_token = custom_name.lower().replace(" ", "-").replace("_", "-")
        custom_token = re.sub(r"[^a-z0-9-]+", "-", custom_token)
        custom_token = re.sub(r"-{2,}", "-", custom_token).strip("-")
        if not custom_token:
            custom_token = "collection"

        name_parts = [
            custom_token,
            cls._format_date_range(start_date, end_date),
            cls._source_token(data_source),
        ]
        return "_".join(name_parts)

    @classmethod
    def parse_name(cls, name: str) -> dict[str, str | None]:
        """Parse a standardized collection name into its components.

        Parameters
        ----------
        name : str
            Collection name created by :meth:`create_name`.

        Returns
        -------
        dict
            Keys: ``custom_name``, ``data_source`` (``None`` if unparseable),
            ``name``.
        """
        try:
            # Remove _stac suffix if present
            clean = name.replace("_stac", "")

            # Split parts
            parts = clean.split("_")
            if len(parts) != 3:
                raise ValueError(f"Invalid name format: {clean}")

            custom_name, date_str, source = parts

            # Parse date range
            date_parts = date_str.split("-")
            if len(date_parts) != 2:
                raise ValueError(f"Invalid date format: {date_str}")

            return {
                "custom_name": custom_name,
                "data_source": source,
                "name": clean,
            }

        except ValueError as e:
            logger.debug("Failed to parse collection name %r: %s", name, e)
            return {"name": name, "custom_name": name, "data_source": None}

    def to_torchgeo_dataset(
        self,
        *,
        bands: list[str],
        chip_size: int | None = None,
        is_image: bool = True,
        allow_resample: bool = False,
        cloud_cover_lt: float | None = None,
        date_range: tuple[str, str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        split: str | Sequence[str] | None = None,
        split_column: str = "split",
        label_field: str | None = None,
        geometries: Any = None,
        geometries_crs: int = 4326,
        transforms: Any = None,
        max_concurrent: int = 50,
        cloud_config: Any = None,
        backend: Any = None,
        time_series: bool = False,
        target_crs: int | None = None,
    ) -> RasteretGeoDataset:
        """Create a TorchGeo GeoDataset backed by this Collection.

        This integration is optional and requires ``torchgeo`` and its
        dependencies.

        Parameters
        ----------
        bands : list of str
            Band codes to load (e.g. ``["B04", "B03", "B02"]``).
        chip_size : int, optional
            Spatial extent of each chip in pixels.
        is_image : bool
            If ``True`` (default), return chips as ``sample[\"image\"]``.
            If ``False``, return chips as ``sample[\"mask\"]`` (single-band data
            will have its channel dimension squeezed to match TorchGeo
            ``RasterDataset`` behavior).
        allow_resample : bool
            If ``True``, Rasteret will resample bands to the dataset grid when
            requested bands have different resolutions. This is opt-in because
            it may change pixel values (resampling) and can be slow.
        cloud_cover_lt : float, optional
            Keep only records with ``eo:cloud_cover`` below this value before
            constructing the TorchGeo dataset.
        date_range : tuple of str, optional
            Keep only records whose ``datetime`` falls within
            ``(start, end)`` before constructing the TorchGeo dataset.
        bbox : tuple of float, optional
            Spatial bbox filter applied before constructing the TorchGeo
            dataset.
        split : str or sequence of str, optional
            Filter to the given split(s) before creating the dataset.
        split_column : str
            Column holding split labels. Defaults to ``"split"``.
        label_field : str, optional
            Column name to include as ``sample["label"]``.
        geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict, optional
            Spatial extent for the dataset. Accepts ``(minx, miny, maxx, maxy)``
            bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects,
            raw WKB bytes, or GeoJSON dicts.
        geometries_crs : int
            EPSG code for *geometries*. Defaults to ``4326``.
        transforms : callable, optional
            TorchGeo-compatible transforms applied to each sample.
        max_concurrent : int
            Maximum concurrent HTTP requests.
        cloud_config : CloudConfig, optional
            Cloud configuration for URL rewriting.
        backend : StorageBackend, optional
            Pluggable I/O backend (e.g. ``ObstoreBackend``).
        time_series : bool
            When ``True``, stack all timesteps as ``[T, C, H, W]``.
        target_crs : int, optional
            Reproject all records to this EPSG code at read time.

        Returns
        -------
        RasteretGeoDataset
            A standard TorchGeo ``GeoDataset``. Pixel data is in the
            native COG dtype (e.g. ``uint16`` for Sentinel-2).
        """
        from rasteret.integrations.torchgeo import RasteretGeoDataset

        self._validate_bands(bands)

        selected_collection = self
        explicit_prefilter_kwargs: dict[str, Any] = {}
        if cloud_cover_lt is not None:
            explicit_prefilter_kwargs["cloud_cover_lt"] = cloud_cover_lt
        if date_range is not None:
            explicit_prefilter_kwargs["date_range"] = date_range
        if bbox is not None:
            explicit_prefilter_kwargs["bbox"] = bbox
        if split is not None:
            explicit_prefilter_kwargs["split"] = split
            explicit_prefilter_kwargs["split_column"] = split_column

        if explicit_prefilter_kwargs:
            selected_collection = self.subset(**explicit_prefilter_kwargs)

        if geometries is not None:
            derived_bbox = _derive_query_bbox(geometries, geometry_crs=geometries_crs)
            if derived_bbox is not None:
                merged_bbox = intersect_bbox(bbox, derived_bbox)
                if bbox is not None and merged_bbox is None:
                    selected_collection = selected_collection._view(
                        selected_collection.dataset.filter(ds.scalar(False))
                    )
                else:
                    try:
                        selected_collection = selected_collection.subset(
                            bbox=merged_bbox or derived_bbox
                        )
                    except ValueError as exc:
                        logger.debug(
                            "TorchGeo prefilter could not apply derived bbox %s: %s",
                            merged_bbox or derived_bbox,
                            exc,
                        )

        return RasteretGeoDataset(
            collection=selected_collection,
            bands=bands,
            chip_size=chip_size,
            is_image=is_image,
            allow_resample=allow_resample,
            label_field=label_field,
            geometries=geometries,
            geometries_crs=geometries_crs,
            transforms=transforms,
            cloud_config=cloud_config,
            max_concurrent=max_concurrent,
            backend=backend,
            time_series=time_series,
            target_crs=target_crs,
        )

    def _auto_backend(
        self,
        cloud_config: Any = None,
        data_source: str | None = None,
    ) -> Any:
        """Auto-create a backend from cloud_config if applicable."""
        from rasteret.cloud import CloudConfig, backend_config_from_cloud_config
        from rasteret.fetch.cog import _create_obstore_backend

        resolved_config = cloud_config or CloudConfig.get_config(
            data_source or self.data_source or ""
        )
        if resolved_config:
            cfg = backend_config_from_cloud_config(resolved_config)
            if cfg:
                return _create_obstore_backend(**cfg)
        return None

    def get_xarray(
        self,
        geometries: Any,
        bands: list[str],
        *,
        max_concurrent: int = 50,
        progress: bool | None = None,
        cloud_config: Any = None,
        data_source: str | None = None,
        backend: Any = None,
        target_crs: int | None = None,
        geometry_crs: int | None = 4326,
        all_touched: bool = False,
        xr_combine: str = "combine_first",
        **filters: Any,
    ) -> xr.Dataset:
        """Load selected bands into an xarray Dataset.

        Parameters
        ----------
        geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
            Area(s) of interest to load. Accepts ``(minx, miny, maxx, maxy)``
            bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects,
            raw WKB bytes, or GeoJSON dicts.
        bands : list of str
            Band codes to load.
        max_concurrent : int
            Maximum concurrent HTTP requests.
        cloud_config : CloudConfig, optional
            Cloud configuration for URL rewriting.
        data_source : str, optional
            Override the inferred data source.
        backend : StorageBackend, optional
            Pluggable I/O backend.
        target_crs : int, optional
            Reproject all records to this CRS before merging.
        all_touched : bool
            Passed through to polygon masking behavior. ``False`` matches
            rasterio default semantics.
        xr_combine : str
            Strategy for merging per-record xarray Datasets.
            ``"combine_first"`` (default) preserves all data and fills
            NaN gaps from subsequent records. ``"merge"`` uses
            ``xr.merge(join="outer")`` which raises on value conflicts.
            ``"merge_override"`` uses ``xr.merge(compat="override")``
            which silently picks one record's values in overlaps.
        progress : bool, optional
            If ``True``, show progress bars during remote reads. If ``None``,
            uses the global default set by :func:`rasteret.set_options`.
        filters : kwargs
            Additional keyword arguments passed to :meth:`subset`.

        Returns
        -------
        xarray.Dataset
            Band arrays in native COG dtype (e.g. ``uint16`` for
            Sentinel-2). CRS encoded via CF conventions (``spatial_ref``
            coordinate with WKT2, PROJJSON, GeoTransform). Multi-CRS
            queries are auto-reprojected to the most common CRS.
        """
        self._validate_bands(bands)
        if backend is None:
            backend = self._auto_backend(cloud_config, data_source)
        if progress is None:
            from rasteret.options import get_options

            progress = get_options().progress
        return get_collection_xarray(
            collection=self,
            geometries=geometries,
            bands=bands,
            data_source=data_source,
            max_concurrent=max_concurrent,
            progress=bool(progress),
            backend=backend,
            target_crs=target_crs,
            geometry_crs=geometry_crs,
            all_touched=all_touched,
            xr_combine=xr_combine,
            **filters,
        )

    def get_gdf(
        self,
        geometries: Any,
        bands: list[str],
        *,
        max_concurrent: int = 50,
        progress: bool | None = None,
        cloud_config: Any = None,
        data_source: str | None = None,
        backend: Any = None,
        target_crs: int | None = None,
        geometry_crs: int | None = 4326,
        all_touched: bool = False,
        **filters: Any,
    ) -> gpd.GeoDataFrame:
        """Load selected bands into a GeoDataFrame.

        Parameters
        ----------
        geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
            Area(s) of interest to load. Accepts ``(minx, miny, maxx, maxy)``
            bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects,
            raw WKB bytes, or GeoJSON dicts.
        bands : list of str
            Band codes to load.
        max_concurrent : int
            Maximum concurrent HTTP requests.
        cloud_config : CloudConfig, optional
            Cloud configuration for URL rewriting.
        data_source : str, optional
            Override the inferred data source.
        backend : StorageBackend, optional
            Pluggable I/O backend.
        target_crs : int, optional
            Reproject all records to this CRS before building the GeoDataFrame.
        all_touched : bool
            Passed through to polygon masking behavior. ``False`` matches
            rasterio default semantics.
        progress : bool, optional
            If ``True``, show progress bars during remote reads. If ``None``,
            uses the global default set by :func:`rasteret.set_options`.
        filters : kwargs
            Additional keyword arguments passed to :meth:`subset`.

        Returns
        -------
        geopandas.GeoDataFrame
            Band arrays in native COG dtype. Each row is a
            geometry-record pair with pixel data as columns.
        """
        self._validate_bands(bands)
        if backend is None:
            backend = self._auto_backend(cloud_config, data_source)
        if progress is None:
            from rasteret.options import get_options

            progress = get_options().progress
        return get_collection_gdf(
            collection=self,
            geometries=geometries,
            bands=bands,
            data_source=data_source,
            max_concurrent=max_concurrent,
            progress=bool(progress),
            backend=backend,
            target_crs=target_crs,
            geometry_crs=geometry_crs,
            all_touched=all_touched,
            **filters,
        )

    def get_numpy(
        self,
        geometries: Any,
        bands: list[str],
        *,
        max_concurrent: int = 50,
        progress: bool | None = None,
        cloud_config: Any = None,
        data_source: str | None = None,
        backend: Any = None,
        target_crs: int | None = None,
        geometry_crs: int | None = 4326,
        all_touched: bool = False,
        **filters: Any,
    ):
        """Load selected bands into NumPy arrays.

        Parameters
        ----------
        geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
            Area(s) of interest to load.
        bands : list of str
            Band codes to load.
        max_concurrent : int
            Maximum concurrent HTTP requests.
        cloud_config : CloudConfig, optional
            Cloud configuration for URL rewriting.
        data_source : str, optional
            Override the inferred data source.
        backend : StorageBackend, optional
            Pluggable I/O backend.
        target_crs : int, optional
            Reproject all records to this CRS before assembly.
        all_touched : bool
            Passed through to polygon masking behavior. ``False`` matches
            rasterio default semantics.
        progress : bool, optional
            If ``True``, show progress bars during remote reads. If ``None``,
            uses the global default set by :func:`rasteret.set_options`.
        filters : kwargs
            Additional keyword arguments passed to :meth:`subset`.

        Returns
        -------
        numpy.ndarray
            Single-band queries return ``[N, H, W]``.
            Multi-band queries return ``[N, C, H, W]`` in requested band order.
        """
        self._validate_bands(bands)
        if backend is None:
            backend = self._auto_backend(cloud_config, data_source)
        if progress is None:
            from rasteret.options import get_options

            progress = get_options().progress
        return get_collection_numpy(
            collection=self,
            geometries=geometries,
            bands=bands,
            data_source=data_source,
            max_concurrent=max_concurrent,
            progress=bool(progress),
            backend=backend,
            target_crs=target_crs,
            geometry_crs=geometry_crs,
            all_touched=all_touched,
            **filters,
        )

    def sample_points(
        self,
        points: Any,
        bands: list[str],
        *,
        geometry_column: str | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
        max_concurrent: int = 50,
        progress: bool | None = None,
        cloud_config: Any = None,
        data_source: str | None = None,
        backend: Any = None,
        geometry_crs: int | None = 4326,
        match: str = "all",
        max_distance_pixels: int = 0,
        return_neighbourhood: Literal["off", "always", "if_center_nodata"] = "off",
        **filters: Any,
    ) -> pa.Table:
        """Sample point values into an Arrow table.

        Parameters
        ----------
        points : Any
            Point input as Arrow/GeoArrow/WKB/Shapely/GeoJSON, or tabular input
            (Arrow table, pandas/GeoPandas, Polars, DuckDB/SedonaDB relation).
        bands : list of str
            Band codes to sample.
        geometry_column : str, optional
            Geometry column name when *points* is tabular. Column may contain WKB,
            GeoArrow points, or Shapely Point objects.
        x_column, y_column : str, optional
            Coordinate column names when *points* is tabular.
        max_concurrent : int
            Maximum concurrent HTTP requests.
        progress : bool, optional
            If ``True``, show progress bars during remote reads. If ``None``,
            uses the global default set by :func:`rasteret.set_options`.
        cloud_config : CloudConfig, optional
            Cloud configuration for URL rewriting.
        data_source : str, optional
            Override the inferred data source.
        backend : StorageBackend, optional
            Pluggable I/O backend.
        geometry_crs : int, optional
            CRS EPSG code of input points. Defaults to EPSG:4326.
        match : {"all", "latest"}
            ``"all"`` returns every matching record for each point.
            ``"latest"`` returns one row per ``(point_index, band)``.
        max_distance_pixels : int
            Maximum pixel distance for nodata fallback search, measured in
            Chebyshev distance (square rings). Rasteret samples the base pixel
            containing the point first; when that pixel is nodata and this is
            > 0, Rasteret searches outward in square rings up to this distance
            and picks the closest candidate by exact
            point-to-pixel-rectangle distance. ``0`` disables fallback and
            returns the base pixel value as-is.
        return_neighbourhood : {"off", "always", "if_center_nodata"}
            Controls whether a neighbourhood window is returned:
            ``"off"`` omits the window column.
            ``"always"`` returns the full window for every sampled row.
            ``"if_center_nodata"`` returns the full window only when the center
            pixel is nodata/NaN; other rows have a NULL window.
        filters : kwargs
            Additional keyword arguments passed to :meth:`subset`.

        Returns
        -------
        pyarrow.Table
            Table with sampled values and metadata columns.
        """
        self._validate_bands(bands)
        if backend is None:
            backend = self._auto_backend(cloud_config, data_source)
        if progress is None:
            from rasteret.options import get_options

            progress = get_options().progress
        if return_neighbourhood != "off" and max_distance_pixels <= 0:
            raise ValueError(
                "max_distance_pixels must be > 0 when return_neighbourhood is enabled"
            )
        return get_collection_point_samples(
            collection=self,
            points=points,
            bands=bands,
            geometry_column=geometry_column,
            x_column=x_column,
            y_column=y_column,
            data_source=data_source,
            max_concurrent=max_concurrent,
            progress=bool(progress),
            backend=backend,
            geometry_crs=geometry_crs,
            match=match,
            max_distance_pixels=max_distance_pixels,
            return_neighbourhood=return_neighbourhood,
            **filters,
        )

    def __dir__(self) -> list[str]:
        names = super().__dir__()
        return sorted(
            name
            for name in names
            if (name.startswith("__") and name.endswith("__"))
            or not name.startswith("_")
        )
