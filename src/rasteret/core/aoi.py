# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Shared Arrow/GeoArrow AOI and point input planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import geoarrow.pyarrow as ga
import pyarrow as pa
import pyarrow.compute as pc
from pyproj import CRS

from rasteret.core.arrow_interop import as_table_or_none
from rasteret.core.geometry import coerce_to_geoarrow, resolve_xy_columns

AUTO_CRS = "auto"
GeometryCrs = int | str | None
GeometryCrsInput = GeometryCrs | Literal["auto"]


@dataclass(frozen=True)
class GeometryInputPlan:
    """Resolved geometry array plus optional row metadata."""

    geometries: pa.Array
    geometry_crs: GeometryCrs
    metadata: pa.Table | None = None
    geometry_column: str | None = None
    table: pa.Table | None = None


def is_auto_crs(value: GeometryCrsInput) -> bool:
    return value == AUTO_CRS


def _missing_geometry_column_error(
    *,
    geometry_column: str,
    names: list[str],
    container: str,
) -> ValueError:
    available = ", ".join(names) if names else "<none>"
    return ValueError(
        f"geometry_column='{geometry_column}' not found in {container}. "
        f"Available columns: {available}"
    )


def _geoarrow_extension_name(field: pa.Field) -> str:
    metadata = field.metadata or {}
    raw_name = metadata.get(b"ARROW:extension:name")
    if raw_name:
        return raw_name.decode("utf-8")
    return str(getattr(field.type, "extension_name", "") or "")


def is_geoarrow_geometry_field(field: pa.Field) -> bool:
    return _geoarrow_extension_name(field).startswith("geoarrow.")


def _geoarrow_extension_metadata(field: pa.Field) -> dict[str, Any]:
    metadata = field.metadata or {}
    raw = metadata.get(b"ARROW:extension:metadata")
    if raw:
        return json.loads(raw.decode("utf-8"))

    serialize = getattr(field.type, "__arrow_ext_serialize__", None)
    if callable(serialize):
        raw = serialize()
        if raw:
            return json.loads(raw.decode("utf-8"))
    return {}


def geometry_field_metadata(field: pa.Field) -> dict[str, Any]:
    """Return GeoArrow extension metadata for *field*, or ``{}``."""
    if not is_geoarrow_geometry_field(field):
        return {}
    return _geoarrow_extension_metadata(field)


def _crs_to_epsg_or_string(raw: Any) -> GeometryCrs:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        crs = CRS.from_user_input(value)
    elif isinstance(raw, dict):
        crs = CRS.from_json_dict(raw)
    else:
        crs = CRS.from_user_input(raw)

    authority = crs.to_authority()
    if authority == ("OGC", "CRS84"):
        return 4326
    epsg = crs.to_epsg()
    if epsg is not None:
        return int(epsg)
    return crs.to_wkt()


def geoarrow_crs_from_field(field: pa.Field) -> GeometryCrs:
    """Extract a CRS from a GeoArrow field, returning ``None`` if absent."""
    metadata = geometry_field_metadata(field)
    if not metadata:
        return None
    return _crs_to_epsg_or_string(metadata.get("crs"))


def validate_geoarrow_edges(field: pa.Field) -> None:
    metadata = geometry_field_metadata(field)
    edges = metadata.get("edges")
    if edges not in (None, "planar"):
        raise ValueError(
            f"Unsupported GeoArrow edges={edges!r}. Rasteret polygon masking and "
            "point sampling require planar coordinates."
        )


def resolve_geometry_crs(
    *,
    provided: GeometryCrsInput,
    field: pa.Field | None,
    tabular: bool,
) -> GeometryCrs:
    """Resolve user-provided or GeoArrow CRS for geometry reads."""
    if not is_auto_crs(provided):
        return _crs_to_epsg_or_string(provided)

    if field is not None:
        validate_geoarrow_edges(field)
        inferred = geoarrow_crs_from_field(field)
        if inferred is not None:
            return inferred

    if tabular:
        raise ValueError(
            "Tabular Arrow geometry input is missing GeoArrow CRS metadata. "
            "Pass geometry_crs=... explicitly, or attach CRS metadata before "
            "passing the table to Rasteret."
        )

    return 4326


def resolve_geometry_column(schema: pa.Schema, geometry_column: str | None) -> str:
    """Resolve or infer a GeoArrow geometry column."""
    names = schema.names
    if geometry_column is not None:
        if geometry_column not in names:
            raise _missing_geometry_column_error(
                geometry_column=geometry_column,
                names=names,
                container="table columns",
            )
        return geometry_column

    candidates = [field.name for field in schema if is_geoarrow_geometry_field(field)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple GeoArrow geometry columns found. Pass geometry_column=... "
            f"with one of: {', '.join(candidates)}"
        )
    raise TypeError(
        "Tabular geometry input needs geometry_column=... unless exactly one "
        "GeoArrow geometry field is present."
    )


def _metadata_table_from_table(
    table: pa.Table,
    *,
    exclude_columns: set[str],
    id_column: str,
    id_start: int,
) -> pa.Table | None:
    source_columns = [
        name for name in table.schema.names if name not in exclude_columns
    ]
    if id_column in source_columns:
        raise ValueError(
            f"Input metadata column '{id_column}' collides with Rasteret's output "
            f"column '{id_column}'. Rename or omit that input column before reading."
        )
    if not source_columns:
        return None
    metadata = table.select(source_columns)
    ids = pa.array(
        range(id_start, id_start + table.num_rows),
        type=pa.int64(),
    )
    return metadata.add_column(0, id_column, ids)


def _empty_polygon_array() -> pa.Array:
    polygon_type = ga.polygon()
    storage = pa.array([], type=polygon_type.storage_type)
    return pa.ExtensionArray.from_storage(polygon_type, storage)


def fail_on_metadata_collisions(
    metadata: pa.Table | None,
    *,
    output_columns: set[str],
    join_column: str,
) -> None:
    if metadata is None:
        return
    metadata_columns = set(metadata.schema.names)
    collisions = sorted((metadata_columns - {join_column}) & output_columns)
    if collisions:
        raise ValueError(
            "Input metadata columns collide with Rasteret output columns: "
            + ", ".join(collisions)
            + ". Rename those input columns or select a narrower metadata table."
        )


def prepare_geometry_input(
    geometries: Any,
    *,
    geometry_column: str | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    preserve_metadata: bool = False,
    id_column: str = "geometry_id",
    id_start: int = 1,
) -> GeometryInputPlan:
    """Normalize polygon AOIs into a GeoArrow array plus optional metadata."""
    table = as_table_or_none(geometries)
    if table is not None:
        resolved_column = resolve_geometry_column(table.schema, geometry_column)
        field = table.schema.field(resolved_column)
        resolved_crs = resolve_geometry_crs(
            provided=geometry_crs,
            field=field,
            tabular=True,
        )
        metadata = (
            _metadata_table_from_table(
                table,
                exclude_columns={resolved_column},
                id_column=id_column,
                id_start=id_start,
            )
            if preserve_metadata
            else None
        )
        return GeometryInputPlan(
            geometries=coerce_to_geoarrow(table.column(resolved_column)),
            geometry_crs=resolved_crs,
            metadata=metadata,
            geometry_column=resolved_column,
            table=table,
        )

    if geometry_column is not None:
        raise TypeError(
            "geometry_column requires an Arrow-native tabular input such as "
            "pyarrow.Table, pyarrow.RecordBatchReader, an Arrow C stream producer, "
            "or an exporter returning Arrow data."
        )

    geometry_array = (
        _empty_polygon_array()
        if isinstance(geometries, list) and len(geometries) == 0
        else coerce_to_geoarrow(geometries)
    )
    return GeometryInputPlan(
        geometries=geometry_array,
        geometry_crs=resolve_geometry_crs(
            provided=geometry_crs,
            field=None,
            tabular=False,
        ),
    )


def prepare_point_input(
    points: Any,
    *,
    geometry_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    preserve_metadata: bool = False,
) -> GeometryInputPlan:
    """Normalize point inputs into a GeoArrow point array plus metadata."""
    table = as_table_or_none(points)
    if table is not None:
        names = table.schema.names
        if geometry_column is not None or (
            x_column is None
            and y_column is None
            and any(is_geoarrow_geometry_field(field) for field in table.schema)
        ):
            resolved_column = resolve_geometry_column(table.schema, geometry_column)
            field = table.schema.field(resolved_column)
            resolved_crs = resolve_geometry_crs(
                provided=geometry_crs,
                field=field,
                tabular=True,
            )
            metadata = (
                _metadata_table_from_table(
                    table,
                    exclude_columns={resolved_column},
                    id_column="point_index",
                    id_start=0,
                )
                if preserve_metadata
                else None
            )
            return GeometryInputPlan(
                geometries=coerce_to_geoarrow(table.column(resolved_column)),
                geometry_crs=resolved_crs,
                metadata=metadata,
                geometry_column=resolved_column,
                table=table,
            )

        xy = resolve_xy_columns(names, x_column, y_column)
        if xy is None:
            raise TypeError(
                "Unsupported table input for point sampling. Provide geometry_column "
                "(WKB/GeoArrow point column) or x_column/y_column."
            )
        x_name, y_name = xy
        resolved_crs = resolve_geometry_crs(
            provided=geometry_crs,
            field=None,
            tabular=True,
        )
        x_values = pc.cast(table.column(x_name), pa.float64(), safe=False)
        y_values = pc.cast(table.column(y_name), pa.float64(), safe=False)
        metadata = (
            _metadata_table_from_table(
                table,
                exclude_columns=set(),
                id_column="point_index",
                id_start=0,
            )
            if preserve_metadata
            else None
        )
        return GeometryInputPlan(
            geometries=ga.make_point(x_values, y_values),
            geometry_crs=resolved_crs,
            metadata=metadata,
            table=table,
        )

    if geometry_column is not None:
        raise TypeError(
            "geometry_column requires an Arrow-native tabular point input. "
            "For raw WKB/Shapely/GeoJSON points, pass the geometry object directly."
        )
    if x_column is not None or y_column is not None:
        raise TypeError("x_column/y_column require a tabular Arrow point input.")

    if isinstance(points, list) and len(points) == 0:
        point_array = ga.make_point(
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.float64()),
        )
    else:
        point_array = coerce_to_geoarrow(points)
    return GeometryInputPlan(
        geometries=point_array,
        geometry_crs=resolve_geometry_crs(
            provided=geometry_crs,
            field=None,
            tabular=False,
        ),
    )
