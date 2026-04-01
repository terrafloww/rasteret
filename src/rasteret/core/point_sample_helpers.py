# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from rasteret.core.geometry import transform_point_coords


def point_bounds_4326(
    points: pa.Array,
    *,
    geometry_crs: int | None,
):
    """Return point lon/lat coordinates and overall bbox in EPSG:4326."""
    if len(points) == 0:
        return None, None, None

    point_xs, point_ys = transform_point_coords(
        points,
        geometry_crs=geometry_crs,
        target_crs=4326,
    )
    bbox = (
        float(np.min(point_xs)),
        float(np.min(point_ys)),
        float(np.max(point_xs)),
        float(np.max(point_ys)),
    )
    return point_xs, point_ys, bbox


def candidate_point_indices_for_raster(
    *,
    raster_bbox: Any,
    point_xs,
    point_ys,
) -> list[int] | None:
    """Return absolute point indices whose lon/lat falls in *raster_bbox*."""
    if point_xs is None or point_ys is None:
        return None
    if raster_bbox is None or len(raster_bbox) != 4:
        return None

    minx, miny, maxx, maxy = raster_bbox
    mask = (
        (point_xs >= float(minx))
        & (point_xs <= float(maxx))
        & (point_ys >= float(miny))
        & (point_ys <= float(maxy))
    )
    return np.nonzero(mask)[0].astype(int).tolist()


def plan_point_tile_groups(
    sample_xs: np.ndarray,
    sample_ys: np.ndarray,
    *,
    scale_x: float,
    translate_x: float,
    scale_y: float,
    translate_y: float,
    raster_height: int,
    raster_width: int,
    tile_height: int,
    tile_width: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], np.ndarray],
    list[tuple[int, int]],
]:
    """Plan point sampling by grouping points into tiles.

    This is the point-sampling equivalent of a window planner: convert map-space
    point coordinates to pixel indices, drop out-of-bounds points, and group the
    remaining points by ``(tile_row, tile_col)``.

    Parameters
    ----------
    sample_xs, sample_ys:
        Point coordinates in the raster CRS (same CRS as the affine transform).
    scale_x, translate_x, scale_y, translate_y:
        North-up affine parameters as used by Rasteret's `normalize_transform`:
        ``x = scale_x * col + translate_x`` and ``y = scale_y * row + translate_y``.
        Rotated/sheared transforms are rejected earlier in the pipeline.
    raster_height, raster_width:
        Raster size in pixels.
    tile_height, tile_width:
        Tile size in pixels.

    Returns
    -------
    point_rows, point_cols:
        Integer pixel indices for each input point (shape ``(N,)``). Points
        outside raster extent are set to ``-1``.
    point_row_fs, point_col_fs:
        Floating pixel coordinates (shape ``(N,)``). Points outside raster extent
        are set to ``NaN``.
    tile_groups:
        Dict keyed by ``(tile_row, tile_col)`` with values shaped ``(M, 3)``:
        ``[point_index, row, col]`` rows for points that fall in that tile.
    tiles:
        List of tile keys (same set as ``tile_groups.keys()``) in deterministic
        ``(tile_row, tile_col)`` order.
    """
    if sample_xs.shape != sample_ys.shape:
        raise ValueError("sample_xs and sample_ys must have the same shape")

    # Vectorized inverse-affine for north-up rasters.
    # Equivalent to: (col_f, row_f) = (~Affine(scale_x,0,tx, 0,scale_y,ty)) * (x, y)
    col_f = (sample_xs - float(translate_x)) / float(scale_x)
    row_f = (sample_ys - float(translate_y)) / float(scale_y)
    cols = np.floor(col_f).astype(np.int64)
    rows = np.floor(row_f).astype(np.int64)

    in_bounds = (
        (rows >= 0)
        & (cols >= 0)
        & (rows < int(raster_height))
        & (cols < int(raster_width))
    )

    point_rows = np.where(in_bounds, rows, -1).astype(np.int64, copy=False)
    point_cols = np.where(in_bounds, cols, -1).astype(np.int64, copy=False)
    point_row_fs = np.where(in_bounds, row_f, np.nan).astype(np.float64, copy=False)
    point_col_fs = np.where(in_bounds, col_f, np.nan).astype(np.float64, copy=False)

    tile_groups: dict[tuple[int, int], np.ndarray] = {}
    tiles: list[tuple[int, int]] = []
    if not np.any(in_bounds):
        return point_rows, point_cols, point_row_fs, point_col_fs, tile_groups, tiles

    point_indices = np.flatnonzero(in_bounds).astype(np.int64)
    tile_rows = point_rows[point_indices] // int(tile_height)
    tile_cols = point_cols[point_indices] // int(tile_width)
    tile_pairs = np.stack((tile_rows, tile_cols), axis=1)

    unique_pairs, inverse = np.unique(tile_pairs, axis=0, return_inverse=True)
    for group_id, (tile_row, tile_col) in enumerate(unique_pairs.tolist()):
        selected = point_indices[inverse == group_id]
        # Store as a compact int64 matrix used directly by the sampling loop.
        tile_groups[(int(tile_row), int(tile_col))] = np.stack(
            (selected, point_rows[selected], point_cols[selected]),
            axis=1,
        ).astype(np.int64, copy=False)
        tiles.append((int(tile_row), int(tile_col)))

    return point_rows, point_cols, point_row_fs, point_col_fs, tile_groups, tiles


def chebyshev_ring_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered (dr, dc) offsets for a Chebyshev ring.

    Order matches PostGIS' perimeter scan:
    top row, bottom row, left column, right column.
    This keeps tie-breaks deterministic when multiple pixels are at the
    same exact distance from the query point.
    """
    if radius < 1:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    side_axis = np.arange(-radius + 1, radius, dtype=np.int64)

    dr = np.concatenate(
        (
            np.full(axis.size, -radius, dtype=np.int64),
            np.full(axis.size, radius, dtype=np.int64),
            side_axis,
            side_axis,
        )
    )
    dc = np.concatenate(
        (
            axis,
            axis,
            np.full(side_axis.size, -radius, dtype=np.int64),
            np.full(side_axis.size, radius, dtype=np.int64),
        )
    )
    return dr, dc


def ring_candidate_grid(
    base_rows: np.ndarray,
    base_cols: np.ndarray,
    row_offsets: np.ndarray,
    col_offsets: np.ndarray,
    *,
    raster_height: int,
    raster_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate (row, col) grids for a ring plus in-bounds mask.

    Shapes:
    - base_rows/base_cols: (N,)
    - row_offsets/col_offsets: (K,)
    - candidate_rows/candidate_cols/in_bounds: (N, K)
    """
    candidate_rows = base_rows[:, None] + row_offsets[None, :]
    candidate_cols = base_cols[:, None] + col_offsets[None, :]
    in_bounds = (
        (candidate_rows >= 0)
        & (candidate_cols >= 0)
        & (candidate_rows < raster_height)
        & (candidate_cols < raster_width)
    )
    return candidate_rows, candidate_cols, in_bounds


def square_neighborhood_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return row-major offsets for a full square neighborhood."""
    if radius < 0:
        raise ValueError("radius must be >= 0")

    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    row_offsets = np.repeat(axis, axis.size)
    col_offsets = np.tile(axis, axis.size)
    return row_offsets, col_offsets


def unique_tile_pairs_for_candidates(
    candidate_rows: np.ndarray,
    candidate_cols: np.ndarray,
    in_bounds: np.ndarray,
    *,
    tile_height: int,
    tile_width: int,
) -> np.ndarray:
    """Return unique ``(tile_row, tile_col)`` pairs for in-bounds candidates.

    This is used to prefetch any tiles required to evaluate candidate pixels.
    """
    if not np.any(in_bounds):
        return np.empty((0, 2), dtype=np.int64)

    tile_rows = candidate_rows[in_bounds] // tile_height
    tile_cols = candidate_cols[in_bounds] // tile_width
    return np.unique(np.stack((tile_rows, tile_cols), axis=1), axis=0)


def tile_grid_shape(
    *,
    raster_width: int,
    raster_height: int,
    tile_width: int,
    tile_height: int,
) -> tuple[int, int]:
    """Return ``(tiles_x, tiles_y)`` for a tiled raster.

    `tiles_x` and `tiles_y` are the number of tiles along each axis in the
    tile offset tables (ceil-div by tile size).
    """
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_width and tile_height must be > 0")
    tiles_x = (int(raster_width) + int(tile_width) - 1) // int(tile_width)
    tiles_y = (int(raster_height) + int(tile_height) - 1) // int(tile_height)
    return tiles_x, tiles_y


def tile_request_specs_for_pairs(
    tile_pairs: np.ndarray | list[tuple[int, int]],
    tile_offsets: Any,
    tile_byte_counts: Any,
    *,
    tiles_x: int,
    tiles_y: int,
) -> list[tuple[int, int, int, int]]:
    """Return request specs for tile pairs.

    Returns ``(tile_row, tile_col, offset, size)`` for each valid tile pair.
    The caller is responsible for attaching URL/metadata/band_index.

    This helper centralizes:
    - tile bounds checks (0 <= row < tiles_y, 0 <= col < tiles_x)
    - tile index computation (row * tiles_x + col)
    - offset/byte-count lookups and safe integer coercion
    """
    if tile_pairs is None:
        return []

    if isinstance(tile_pairs, np.ndarray):
        if tile_pairs.size == 0:
            return []
        pairs_iter = ((int(r), int(c)) for r, c in tile_pairs.tolist())
    else:
        if not tile_pairs:
            return []
        pairs_iter = ((int(r), int(c)) for r, c in tile_pairs)

    if tile_offsets is None or tile_byte_counts is None:
        return []

    specs: list[tuple[int, int, int, int]] = []
    offsets_len = len(tile_offsets)
    counts_len = len(tile_byte_counts)
    if offsets_len != counts_len:
        return []

    for tile_row, tile_col in pairs_iter:
        if (
            tile_row < 0
            or tile_col < 0
            or tile_row >= int(tiles_y)
            or tile_col >= int(tiles_x)
        ):
            continue
        tile_idx = tile_row * int(tiles_x) + tile_col
        if tile_idx < 0 or tile_idx >= offsets_len:
            continue
        specs.append(
            (
                tile_row,
                tile_col,
                int(tile_offsets[tile_idx]),
                int(tile_byte_counts[tile_idx]),
            )
        )

    return specs


def candidate_values_from_tile_cache(
    candidate_rows: np.ndarray,
    candidate_cols: np.ndarray,
    in_bounds: np.ndarray,
    *,
    tile_cache: dict[tuple[int, int], np.ndarray],
    tile_height: int,
    tile_width: int,
    tiles_x: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize candidate values from an in-memory tile cache.

    Returns:
    - `candidate_values`: float64 array (N, K) with sampled values or NaN where
      not sampled (missing tile, or out of bounds).
    - `candidate_sampled`: bool array (N, K) True where a value was read from a
      cached tile.

    Note: tiles are treated as full `tile_height x tile_width` arrays, consistent
    with Rasteret's tile materialization. Edge tiles may include pixels outside
    raster extent; callers should ensure candidates are in raster bounds.
    """
    candidate_values = np.full(candidate_rows.shape, np.nan, dtype=np.float64)
    candidate_sampled = np.zeros(candidate_rows.shape, dtype=bool)
    if not np.any(in_bounds):
        return candidate_values, candidate_sampled

    flat_indices = np.flatnonzero(in_bounds)
    flat_rows = candidate_rows.ravel()[flat_indices]
    flat_cols = candidate_cols.ravel()[flat_indices]
    tile_rows = flat_rows // tile_height
    tile_cols = flat_cols // tile_width
    local_rows = flat_rows - tile_rows * tile_height
    local_cols = flat_cols - tile_cols * tile_width
    tile_keys = tile_rows * tiles_x + tile_cols

    flat_candidate_values = candidate_values.ravel()
    flat_candidate_sampled = candidate_sampled.ravel()
    for tile_key in np.unique(tile_keys):
        tile_mask = tile_keys == tile_key
        tile_row = int(tile_rows[tile_mask][0])
        tile_col = int(tile_cols[tile_mask][0])
        tile_data = tile_cache.get((tile_row, tile_col))
        if tile_data is None:
            continue

        sampled_values = np.asarray(
            tile_data[local_rows[tile_mask], local_cols[tile_mask]],
            dtype=np.float64,
        )
        flat_selection = flat_indices[tile_mask]
        flat_candidate_values[flat_selection] = sampled_values
        flat_candidate_sampled[flat_selection] = True

    return candidate_values, candidate_sampled


def pixel_rect_distance_sq(
    point_rows: np.ndarray,
    point_cols: np.ndarray,
    candidate_rows: np.ndarray,
    candidate_cols: np.ndarray,
    *,
    pixel_width: float,
    pixel_height: float,
) -> np.ndarray:
    """Return squared map-space distance from query points to pixel rectangles.

    Distances are computed in pixel-coordinate space and then scaled by the
    raster's pixel size (Affine `a` and `e`). This yields a map-space distance
    compatible with "nearest non-nodata pixel" semantics in tools like PostGIS.
    """
    row0 = candidate_rows.astype(np.float64, copy=False)
    row1 = row0 + 1.0
    col0 = candidate_cols.astype(np.float64, copy=False)
    col1 = col0 + 1.0

    row_f = point_rows[:, None]
    col_f = point_cols[:, None]

    dy = np.maximum(np.maximum(row0 - row_f, row_f - row1), 0.0)
    dx = np.maximum(np.maximum(col0 - col_f, col_f - col1), 0.0)

    dy *= abs(float(pixel_height))
    dx *= abs(float(pixel_width))
    return dx * dx + dy * dy


def nodata_mask(values: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """Return True where *values* should be treated as nodata."""
    if nodata is None:
        return np.isnan(values)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isnan(values)
    return np.isnan(values) | (values == float(nodata))
