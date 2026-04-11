# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Shared normalisation layer: raw Arrow table -> Collection.

Every ingest driver calls :func:`build_collection_from_table` as its
final step to validate column contract, add partition columns, and
construct a :class:`~rasteret.core.collection.Collection`.
"""

from __future__ import annotations

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
    if isinstance(crs_value, str):
        value = crs_value.strip()
        return value or None
    return None


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
