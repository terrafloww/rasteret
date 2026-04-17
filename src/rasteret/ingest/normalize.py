# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Shared normalisation layer: raw Arrow table -> Collection.

Every ingest driver calls :func:`build_collection_from_table` as its
final step to validate column contract, add partition columns, and
construct a :class:`~rasteret.core.collection.Collection`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

# Columns required by the Collection contract.
REQUIRED_COLUMNS = frozenset({"id", "datetime", "geometry", "assets"})

_BBOX_STRUCT_TYPE = pa.struct(
    [
        pa.field("xmin", pa.float64()),
        pa.field("ymin", pa.float64()),
        pa.field("xmax", pa.float64()),
        pa.field("ymax", pa.float64()),
    ]
)


def parse_epsg(crs_value: object) -> int | None:
    """Extract an integer EPSG code from a CRS value.

    Accepts ``int`` (returned as-is), ``"EPSG:32632"``-style strings,
    or ``None``.  Returns ``None`` when parsing fails.
    """
    if crs_value is None:
        return None
    if isinstance(crs_value, int):
        return crs_value
    if isinstance(crs_value, str):
        s = crs_value.strip().upper()
        if s.startswith("EPSG:"):
            try:
                return int(s.split(":", 1)[1])
            except ValueError:
                return None
        if s.isdigit():
            try:
                return int(s)
            except ValueError:
                return None
        try:
            from pyproj import CRS

            return CRS.from_user_input(crs_value).to_epsg()
        except Exception:
            return None
    if isinstance(crs_value, dict):
        try:
            from pyproj import CRS

            return CRS.from_json_dict(crs_value).to_epsg()
        except Exception:
            return None
    return None


def crs_code_from_epsg(epsg: int | None) -> str | None:
    """Return an authority-code CRS string for an EPSG integer."""
    if epsg is None:
        return None
    return f"EPSG:{int(epsg)}"


def normalize_crs_code(crs_value: object) -> str | None:
    """Normalize supported CRS inputs into an authority-code string."""
    epsg = parse_epsg(crs_value)
    if epsg is not None:
        return crs_code_from_epsg(epsg)
    if isinstance(crs_value, dict):
        try:
            from pyproj import CRS

            crs = CRS.from_json_dict(crs_value)
            authority = crs.to_authority()
            if authority is not None:
                return f"{authority[0]}:{authority[1]}"
            return crs.to_string()
        except Exception:
            return json.dumps(crs_value, sort_keys=True)
    if isinstance(crs_value, str):
        value = crs_value.strip()
        if not value:
            return None
        try:
            from pyproj import CRS

            crs = CRS.from_user_input(value)
            authority = crs.to_authority()
            if authority is not None:
                return f"{authority[0]}:{authority[1]}"
            return crs.to_string()
        except Exception:
            pass
        return value or None
    return None


def normalize_raster_crs_sidecars(
    table: pa.Table,
    *,
    required_columns: Sequence[str] | None = None,
) -> pa.Table:
    """Normalize row-level raster CRS sidecars from common source fields.

    Priority per row:
    1. ``proj:code``
    2. ``proj:epsg``
    3. ``crs``
    4. ``proj:wkt2``
    5. ``proj:projjson``
    """
    required = set(required_columns) if required_columns is not None else None
    names = set(table.schema.names)
    if required is not None and "proj:epsg" not in required and "crs" not in required:
        return table

    candidate_columns = [
        name
        for name in ("proj:code", "proj:epsg", "crs", "proj:wkt2", "proj:projjson")
        if name in names
    ]
    if not candidate_columns:
        return table

    candidate_values = {
        name: table.column(name).to_pylist() for name in candidate_columns
    }
    row_count = len(table)
    epsg_values: list[int | None] = []
    crs_values: list[str | None] = []

    for idx in range(row_count):
        epsg_value: int | None = None
        crs_value: str | None = None
        for name in ("proj:code", "proj:epsg", "crs", "proj:wkt2", "proj:projjson"):
            values = candidate_values.get(name)
            if values is None:
                continue
            raw = values[idx]
            if epsg_value is None:
                epsg_value = parse_epsg(raw)
            if crs_value is None:
                crs_value = normalize_crs_code(raw)
            if epsg_value is not None and crs_value is not None:
                break
        epsg_values.append(epsg_value)
        crs_values.append(crs_value or crs_code_from_epsg(epsg_value))

    if required is None or "proj:epsg" in required:
        epsg_col = pa.array(epsg_values, type=pa.int32())
        if "proj:epsg" in names:
            idx = table.schema.get_field_index("proj:epsg")
            table = table.set_column(idx, "proj:epsg", epsg_col)
        else:
            table = table.append_column("proj:epsg", epsg_col)

    if required is None or "crs" in required:
        crs_col = pa.array(crs_values, type=pa.string())
        if "crs" in names:
            idx = table.schema.get_field_index("crs")
            table = table.set_column(idx, "crs", crs_col)
        else:
            table = table.append_column("crs", crs_col)

    return table


def _add_bbox_struct(table: pa.Table) -> pa.Table:
    """Derive ``bbox`` struct from the ``geometry`` column.

    Uses GeoArrow for Arrow-native bbox extraction (no Shapely).
    Falls back to a null struct column if parsing fails.
    """
    try:
        from rasteret.core.geometry import bbox_array

        geom_col = table.column("geometry")
        xmin, ymin, xmax, ymax = bbox_array(geom_col)
        bbox = pa.StructArray.from_arrays(
            [xmin, ymin, xmax, ymax],
            fields=_BBOX_STRUCT_TYPE,
        )
        return table.append_column("bbox", bbox)
    except (KeyError, TypeError, ValueError, pa.ArrowInvalid) as exc:
        logger.warning(
            "Could not derive bbox from geometry: %s; adding null column", exc
        )
        nulls = pa.nulls(len(table), type=_BBOX_STRUCT_TYPE)
        return table.append_column("bbox", nulls)


def _drop_column_if_present(table: pa.Table, name: str) -> pa.Table:
    if name not in table.schema.names:
        return table
    return table.drop_columns([name])


def _ensure_footprint_geometry_crs84(table: pa.Table) -> tuple[pa.Table, bool]:
    """Return a table whose collection footprint geometry is CRS84.

    Rasteret's collection footprint is the index/filtering geometry and is
    expected to be lon/lat CRS84. Raster CRS remains in row-level sidecars.
    """
    if "geometry" not in table.schema.names:
        return table, False

    from rasteret.core.aoi import geoarrow_crs_from_field

    field = table.schema.field("geometry")
    source_crs = geoarrow_crs_from_field(field)
    if source_crs is None or source_crs == 4326:
        return table, False

    import geoarrow.pyarrow as ga
    import shapely
    from pyproj import Transformer
    from shapely.ops import transform

    transformer = Transformer.from_crs(source_crs, "OGC:CRS84", always_xy=True)
    geom_wkb = ga.as_wkb(table.column("geometry").combine_chunks()).to_pylist()
    geometries = shapely.from_wkb(geom_wkb)
    transformed = [
        transform(transformer.transform, geom) if geom is not None else None
        for geom in geometries
    ]
    crs84_wkb = pa.array(list(shapely.to_wkb(transformed)), type=pa.binary())
    index = table.schema.get_field_index("geometry")
    table = table.set_column(index, pa.field("geometry", pa.binary()), crs84_wkb)
    return table, True


def build_collection_from_table(
    table: pa.Table,
    *,
    name: str = "",
    description: str = "",
    data_source: str = "",
    date_range: tuple[str, str] | None = None,
    workspace_dir: str | Path | None = None,
    partition_cols: Sequence[str] = ("year", "month"),
) -> Any:
    """Normalise an Arrow table into a Collection.

    Validates the Collection contract columns, adds ``bbox``
    and partition columns when missing, and optionally materialises
    to Parquet.

    Parameters
    ----------
    table:
        Arrow table with at least the required columns.
    name:
        Human-readable collection name.
    description:
        Free-text description.
    data_source:
        Data source identifier (e.g. ``"sentinel-2-l2a"``).
    date_range:
        ``(start, end)`` ISO date strings.  Used for collection metadata.
    workspace_dir:
        If provided, persist the collection as partitioned Parquet here.
    partition_cols:
        Columns to partition by when writing Parquet.

    Returns
    -------
    Collection
    """
    from rasteret.core.collection import Collection

    missing = REQUIRED_COLUMNS - set(table.schema.names)
    if missing:
        raise ValueError(f"Table is missing required columns: {missing}")

    table, transformed_geometry = _ensure_footprint_geometry_crs84(table)
    if transformed_geometry:
        table = _drop_column_if_present(table, "bbox")

    table = normalize_raster_crs_sidecars(table)

    # Add canonical bbox if absent.
    if "bbox" not in table.schema.names:
        table = _add_bbox_struct(table)

    # Add year/month partition columns if absent.
    datetime_col = table.column("datetime")
    if "year" not in table.schema.names:
        # Ensure the column is a timestamp type.
        if not pa.types.is_timestamp(datetime_col.type):
            datetime_col = pc.cast(datetime_col, pa.timestamp("us"))
        table = table.append_column("year", pc.year(datetime_col))
    if "month" not in table.schema.names:
        if not pa.types.is_timestamp(datetime_col.type):
            datetime_col = pc.cast(datetime_col, pa.timestamp("us"))
        table = table.append_column("month", pc.month(datetime_col))

    start_date = datetime.fromisoformat(date_range[0]) if date_range else None
    end_date = datetime.fromisoformat(date_range[1]) if date_range else None

    collection = Collection(
        dataset=ds.dataset(table),
        name=name,
        description=description,
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
    )

    if workspace_dir:
        collection.export(workspace_dir)

    return collection
