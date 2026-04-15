# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Arrow-native geometry utilities.

All geometry handling in Rasteret flows through this module.
Internal representation is GeoArrow native ``pa.Array``
(``geoarrow.polygon``). Shapely is imported lazily for GeoJSON and
bbox conversions (it is already installed as a transitive dependency
of geopandas).

Conversion to GeoJSON dicts happens only at the rasterio
``geometry_mask()`` boundary.
"""

from __future__ import annotations

import logging
from typing import Any

import geoarrow.pyarrow as ga
import pyarrow as pa

Bbox = tuple[float, float, float, float]
logger = logging.getLogger(__name__)


class UnsupportedGeometryError(TypeError):
    """Raised when a geometry type is unsupported by Rasteret's mask path.

    Rasteret's raster masking relies on ``rasterio.features.geometry_mask``,
    which is driven by Polygon/MultiPolygon AOIs. Point sampling is a different
    operation (more like ``rasterio.sample``) and is intentionally handled as a
    separate feature.
    """


# ------------------------------------------------------------------
# Coercion: any geometry format -> GeoArrow native pa.Array
# ------------------------------------------------------------------


def coerce_to_geoarrow(geom: Any) -> pa.Array:
    """Convert any supported geometry input to a GeoArrow native array.

    Accepts
    -------
    - ``pa.Array`` or ``pa.ChunkedArray`` (WKB or native GeoArrow)
    - ``bytes`` (single WKB)
    - ``list[bytes]`` (multiple WKB)
    - ``Shapely Polygon`` or ``list[Shapely Polygon]``
    - ``dict`` with ``"type"`` and ``"coordinates"`` (GeoJSON)
    - ``(minx, miny, maxx, maxy)`` bbox tuple

    Returns
    -------
    pa.Array
        GeoArrow native polygon array.
    """
    if isinstance(geom, pa.ChunkedArray):
        geom = geom.combine_chunks()

    if isinstance(geom, pa.Array):
        return _ensure_native(geom)

    if isinstance(geom, bytes):
        arr = pa.array([geom], type=pa.binary())
        return _ensure_native(arr)

    if isinstance(geom, list):
        if not geom:
            raise ValueError("Empty geometry list")
        if isinstance(geom[0], bytes):
            arr = pa.array(geom, type=pa.binary())
            return _ensure_native(arr)
        # Check for Shapely objects
        if _is_shapely(geom[0]):
            return _from_shapely_list(geom)
        # Check for GeoJSON dicts
        if isinstance(geom[0], dict):
            return _from_geojson_list(geom)
        raise TypeError(f"Unsupported geometry list element type: {type(geom[0])}")

    if isinstance(geom, tuple) and len(geom) == 4:
        return _from_bbox(geom)

    if isinstance(geom, dict) and "type" in geom and "coordinates" in geom:
        return _from_geojson_list([geom])

    if _is_shapely(geom):
        return _from_shapely_list([geom])

    raise TypeError(f"Unsupported geometry type: {type(geom)}")


def _ensure_native(arr: pa.Array) -> pa.Array:
    """Convert WKB array to GeoArrow native if needed."""
    type_name = getattr(arr.type, "extension_name", "")
    if (
        "wkb" in type_name
        or pa.types.is_binary(arr.type)
        or pa.types.is_large_binary(arr.type)
    ):
        try:
            return ga.as_geoarrow(arr)
        except Exception as exc:
            raise TypeError(
                "Binary geometry input must be OGC WKB. "
                "If the column comes from DuckDB GEOMETRY, use "
                "ST_AsWKB(geom) in SQL before passing it to Rasteret."
            ) from exc
    return arr


def _is_shapely(obj: Any) -> bool:
    """Check if an object is a Shapely geometry without importing shapely."""
    return type(obj).__module__.startswith("shapely")


def _from_shapely_list(geoms: list) -> pa.Array:
    """Convert Shapely geometries to GeoArrow via WKB."""
    import shapely

    wkb_data = shapely.to_wkb(geoms)
    arr = pa.array(list(wkb_data), type=pa.binary())
    return ga.as_geoarrow(arr)


def _from_geojson_list(geojsons: list[dict]) -> pa.Array:
    """Convert GeoJSON dicts to GeoArrow via shapely (lazy import)."""
    wkb = geojson_dicts_to_wkb(geojsons)
    return ga.as_geoarrow(wkb)


def _from_bbox(bbox: tuple) -> pa.Array:
    """Convert (minx, miny, maxx, maxy) to a polygon GeoArrow array."""
    import shapely

    geom = shapely.box(*bbox)
    wkb_data = shapely.to_wkb(geom)
    arr = pa.array([wkb_data], type=pa.binary())
    return ga.as_geoarrow(arr)


# ------------------------------------------------------------------
# Bbox extraction
# ------------------------------------------------------------------


def bbox_array(
    geom_col: pa.Array | pa.ChunkedArray,
) -> tuple[pa.Array, pa.Array, pa.Array, pa.Array]:
    """Extract per-geometry bboxes as four Arrow arrays.

    Returns (xmin, ymin, xmax, ymax) arrays. Fully Arrow-native via
    ``geoarrow.pyarrow.box()``.
    """
    if isinstance(geom_col, pa.ChunkedArray):
        geom_col = geom_col.combine_chunks()
    native = _ensure_native(geom_col)
    boxes = ga.box(native).storage
    return (
        boxes.field("xmin"),
        boxes.field("ymin"),
        boxes.field("xmax"),
        boxes.field("ymax"),
    )


def bbox_single(geom_col: pa.Array, idx: int) -> Bbox:
    """Get bbox for one geometry at *idx* from a GeoArrow array."""
    storage = ga.box(geom_col).storage
    return (
        storage.field("xmin")[idx].as_py(),
        storage.field("ymin")[idx].as_py(),
        storage.field("xmax")[idx].as_py(),
        storage.field("ymax")[idx].as_py(),
    )


# ------------------------------------------------------------------
# CRS transform
# ------------------------------------------------------------------


def transform_coords(
    geom_col: pa.Array,
    idx: int,
    src_crs: int | str,
    dst_crs: int | str,
) -> dict:
    """CRS-transform one Polygon/MultiPolygon and return as GeoJSON dict.

    Uses ``pyproj.Transformer`` on coordinate lists. No Shapely.
    """
    from pyproj import Transformer

    src = f"EPSG:{src_crs}" if isinstance(src_crs, int) else src_crs
    dst = f"EPSG:{dst_crs}" if isinstance(dst_crs, int) else dst_crs
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    coords_py = geom_col.storage[idx].as_py()

    if not coords_py:
        raise ValueError("Empty geometry")

    if isinstance(coords_py, dict):
        raise UnsupportedGeometryError(
            "Unsupported geometry type for Rasteret masking: Point. "
            "Rasteret currently supports Polygon and MultiPolygon geometries for "
            "masking-based sampling."
        )
    if not isinstance(coords_py, (list, tuple)):
        raise UnsupportedGeometryError(
            f"Unsupported geometry type for Rasteret masking: {type(coords_py)!r}. "
            "Rasteret currently supports Polygon and MultiPolygon geometries for "
            "masking-based sampling."
        )

    # GeoArrow stores polygons as: rings -> points
    # and multipolygons as: polygons -> rings -> points.
    if isinstance(coords_py[0][0], dict):
        # Polygon: list[rings]
        transformed_rings: list[list[tuple[float, float]]] = []
        for ring in coords_py:
            xx = [pt["x"] for pt in ring]
            yy = [pt["y"] for pt in ring]
            tx, ty = transformer.transform(xx, yy)
            transformed_rings.append(list(zip(list(tx), list(ty))))
        return {"type": "Polygon", "coordinates": transformed_rings}

    if isinstance(coords_py[0][0], list):
        # MultiPolygon: list[polygons], polygon is list[rings]
        transformed_polys: list[list[list[tuple[float, float]]]] = []
        for poly in coords_py:
            transformed_rings = []
            for ring in poly:
                xx = [pt["x"] for pt in ring]
                yy = [pt["y"] for pt in ring]
                tx, ty = transformer.transform(xx, yy)
                transformed_rings.append(list(zip(list(tx), list(ty))))
            transformed_polys.append(transformed_rings)
        return {"type": "MultiPolygon", "coordinates": transformed_polys}

    raise UnsupportedGeometryError(
        "Unsupported geometry type for Rasteret masking. Rasteret currently "
        "supports Polygon and MultiPolygon geometries for masking-based sampling."
    )


# ------------------------------------------------------------------
# Rasterio bridge
# ------------------------------------------------------------------


def to_rasterio_geojson(geom_col: pa.Array, idx: int) -> dict:
    """Convert one GeoArrow Polygon/MultiPolygon at *idx* to a GeoJSON dict.

    This is the ONLY place geometry leaves Arrow format, at the
    ``rasterio.geometry_mask()`` boundary.
    """
    coords_py = geom_col.storage[idx].as_py()
    if not coords_py:
        raise ValueError("Empty geometry")

    if isinstance(coords_py, dict):
        raise UnsupportedGeometryError(
            "Unsupported geometry type for Rasteret masking: Point. "
            "Rasteret currently supports Polygon and MultiPolygon geometries for "
            "masking-based sampling."
        )
    if not isinstance(coords_py, (list, tuple)):
        raise UnsupportedGeometryError(
            f"Unsupported geometry type for Rasteret masking: {type(coords_py)!r}. "
            "Rasteret currently supports Polygon and MultiPolygon geometries for "
            "masking-based sampling."
        )

    if isinstance(coords_py[0][0], dict):
        rings = [[(pt["x"], pt["y"]) for pt in ring] for ring in coords_py]
        return {"type": "Polygon", "coordinates": rings}

    if isinstance(coords_py[0][0], list):
        polys = []
        for poly in coords_py:
            rings = [[(pt["x"], pt["y"]) for pt in ring] for ring in poly]
            polys.append(rings)
        return {"type": "MultiPolygon", "coordinates": polys}

    raise UnsupportedGeometryError(
        "Unsupported geometry type for Rasteret masking. Rasteret currently "
        "supports Polygon and MultiPolygon geometries for masking-based sampling."
    )


# ------------------------------------------------------------------
# Spatial predicates (pure arithmetic)
# ------------------------------------------------------------------


def bbox_intersects(a: Bbox, b: Bbox) -> bool:
    """Test whether two bboxes overlap. Pure arithmetic."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def geojson_dicts_to_wkb(geojsons: list[dict]) -> pa.Array:
    """Convert GeoJSON dicts to a GeoArrow WKB array.

    Uses shapely's vectorized ``from_geojson`` / ``to_wkb`` (lazy import).
    Shapely is already installed as a transitive dep of geopandas.
    """
    import json

    import shapely

    geojson_strings = [json.dumps(gj) for gj in geojsons]
    geoms = shapely.from_geojson(geojson_strings)
    wkb_data = shapely.to_wkb(geoms)
    return pa.array(list(wkb_data), type=pa.binary())


def bbox_from_geojson_coords(geojson: dict) -> Bbox:
    """Compute bbox from a GeoJSON dict's coordinates. No deps."""

    def iter_points(obj: Any):
        if isinstance(obj, (list, tuple)) and obj:
            # GeoJSON point: [x, y] (or (x, y))
            if isinstance(obj[0], (int, float)) and len(obj) >= 2:
                yield (obj[0], obj[1])
                return
            for child in obj:
                yield from iter_points(child)

    coords = geojson.get("coordinates", [])
    all_pts = list(iter_points(coords))
    if len(all_pts) == 0:
        raise ValueError("Empty geometry coordinates")
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return (min(xs), min(ys), max(xs), max(ys))


# ------------------------------------------------------------------
# Point sampling helpers
# ------------------------------------------------------------------


def resolve_xy_columns(
    names: list[str],
    x_column: str | None,
    y_column: str | None,
) -> tuple[str, str] | None:
    """Resolve x/y columns from explicit names or common defaults."""
    if x_column and y_column:
        if x_column in names and y_column in names:
            return x_column, y_column
        missing = [col for col in (x_column, y_column) if col not in names]
        raise ValueError(f"Missing point coordinate columns: {missing}")

    candidates = [
        ("x", "y"),
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("lng", "lat"),
    ]
    for x_name, y_name in candidates:
        if x_name in names and y_name in names:
            return x_name, y_name
    return None


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


def ensure_point_geoarrow(
    points: Any,
    *,
    geometry_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> pa.Array:
    """Normalize point inputs into a GeoArrow point array."""
    from rasteret.core.aoi import prepare_point_input

    return prepare_point_input(
        points,
        geometry_column=geometry_column,
        x_column=x_column,
        y_column=y_column,
        geometry_crs=4326,
        preserve_metadata=False,
    ).geometries


def transform_point_coords(
    points: pa.Array,
    *,
    geometry_crs: int | str | None,
    target_crs: int | str,
):
    """Return point coordinates in *target_crs* as numpy arrays."""
    import numpy as np

    point_xs_arr, point_ys_arr = ga.point_coords(points)
    point_xs = point_xs_arr.to_numpy(zero_copy_only=False)
    point_ys = point_ys_arr.to_numpy(zero_copy_only=False)

    if geometry_crs is None or geometry_crs == target_crs:
        return point_xs, point_ys

    from pyproj import Transformer

    src = f"EPSG:{geometry_crs}" if isinstance(geometry_crs, int) else geometry_crs
    dst = f"EPSG:{target_crs}" if isinstance(target_crs, int) else target_crs
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    x_coords, y_coords = transformer.transform(point_xs, point_ys)
    return np.asarray(x_coords), np.asarray(y_coords)


def intersect_bbox(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Return the bbox intersection, or ``None`` when disjoint."""
    if left is None:
        return right
    if right is None:
        return left

    minx = max(left[0], right[0])
    miny = max(left[1], right[1])
    maxx = min(left[2], right[2])
    maxy = min(left[3], right[3])
    if minx > maxx or miny > maxy:
        return None
    return (minx, miny, maxx, maxy)
