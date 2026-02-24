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

_BBOX_SCALAR_COLUMNS = ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy")


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


def _add_scene_bbox(table: pa.Table) -> pa.Table:
    """Derive ``scene_bbox`` from the ``geometry`` column.

    Uses GeoArrow for Arrow-native bbox extraction (no Shapely).
    Falls back to a null column if parsing fails.
    """
    try:
        from rasteret.core.geometry import bbox_array

        geom_col = table.column("geometry")
        xmin, ymin, xmax, ymax = bbox_array(geom_col)

        # Build scene_bbox as list<float64>[4] for backwards compatibility
        n = len(table)
        bboxes = []
        for i in range(n):
            x0, y0, x1, y1 = (
                xmin[i].as_py(),
                ymin[i].as_py(),
                xmax[i].as_py(),
                ymax[i].as_py(),
            )
            if x0 is None:
                bboxes.append(None)
            else:
                bboxes.append([x0, y0, x1, y1])

        return table.append_column(
            "scene_bbox", pa.array(bboxes, type=pa.list_(pa.float64(), 4))
        )
    except (KeyError, TypeError, ValueError, pa.ArrowInvalid) as exc:
        logger.warning(
            "Could not derive scene_bbox from geometry: %s; adding null column", exc
        )
        nulls = pa.array([None] * len(table), type=pa.list_(pa.float64(), 4))
        return table.append_column("scene_bbox", nulls)


def _add_bbox_scalar_columns(table: pa.Table) -> pa.Table:
    """Add scalar bbox columns derived from ``scene_bbox``.

    Rasteret keeps ``scene_bbox`` as a 4-element list for portability, but
    Arrow dataset filtering cannot efficiently filter inside list values.
    These scalar columns are the supported, pushdown-friendly filtering keys.
    """
    if "scene_bbox" not in table.schema.names:
        return table

    missing = [name for name in _BBOX_SCALAR_COLUMNS if name not in table.schema.names]
    if not missing:
        return table

    bbox = table.column("scene_bbox")
    try:
        minx = pc.list_element(bbox, 0)
        miny = pc.list_element(bbox, 1)
        maxx = pc.list_element(bbox, 2)
        maxy = pc.list_element(bbox, 3)
    except (KeyError, TypeError, ValueError, pa.ArrowInvalid) as exc:
        logger.warning("Could not derive scalar bbox columns from scene_bbox: %s", exc)
        return table

    mapping: dict[str, pa.Array] = {
        "bbox_minx": minx,
        "bbox_miny": miny,
        "bbox_maxx": maxx,
        "bbox_maxy": maxy,
    }
    for name, values in mapping.items():
        if name not in table.schema.names:
            table = table.append_column(name, values)
    return table


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

    Validates the Collection contract columns, adds ``scene_bbox``
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

    # Add scene_bbox if absent.
    if "scene_bbox" not in table.schema.names:
        table = _add_scene_bbox(table)
    table = _add_bbox_scalar_columns(table)

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
