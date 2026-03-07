from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from affine import Affine


def _normalize_res(res: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(res, tuple):
        return float(res[0]), float(res[1])
    value = float(res)
    return value, value


def _normalize_transform(transform: Affine | np.ndarray | list[float]) -> Affine:
    if isinstance(transform, Affine):
        return transform
    values = np.asarray(transform).reshape(-1)
    if values.size < 6:
        raise ValueError("Transform must have at least 6 values.")
    return Affine(*values[:6].tolist())


def stitch_prediction_tiles(
    tiles: Iterable[
        dict[str, object] | tuple[np.ndarray, Affine | np.ndarray | list[float]]
    ],
    *,
    roi_bounds: tuple[float, float, float, float],
    res: float | tuple[float, float],
    reducer: str = "overwrite",
    fill_value: float = np.nan,
    out_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Stitch georeferenced prediction tiles into a single north-up canvas.

    Parameters
    ----------
    tiles:
        Iterable of either:
        - ``{"prediction": np.ndarray, "transform": Affine|array_like}``, or
        - ``(prediction, transform)`` tuples.
    roi_bounds:
        ``(xmin, ymin, xmax, ymax)`` of output ROI in dataset CRS.
    res:
        Pixel size in CRS units (``(xres, yres)`` or scalar).
    reducer:
        ``"overwrite"`` (last-write-wins) or ``"mean"`` (average overlaps).
    fill_value:
        Fill value for pixels with no contributing tile.
    out_shape:
        Optional ``(height, width)`` override. If omitted, derived from ROI + res.
    """
    if reducer not in {"overwrite", "mean"}:
        raise ValueError("reducer must be 'overwrite' or 'mean'.")

    roi_xmin, roi_ymin, roi_xmax, roi_ymax = map(float, roi_bounds)
    res_x, res_y = _normalize_res(res)

    if out_shape is None:
        out_w = int(round((roi_xmax - roi_xmin) / res_x))
        out_h = int(round((roi_ymax - roi_ymin) / res_y))
    else:
        out_h, out_w = int(out_shape[0]), int(out_shape[1])

    if reducer == "mean":
        sum_grid = np.zeros((out_h, out_w), dtype=np.float64)
        count_grid = np.zeros((out_h, out_w), dtype=np.uint32)
    else:
        stitched = np.full((out_h, out_w), fill_value, dtype=np.float32)

    for tile in tiles:
        if isinstance(tile, dict):
            prediction = np.asarray(tile["prediction"], dtype=np.float32)
            transform = _normalize_transform(tile["transform"])  # type: ignore[arg-type]
        else:
            prediction = np.asarray(tile[0], dtype=np.float32)
            transform = _normalize_transform(tile[1])

        row = int(round((roi_ymax - float(transform.f)) / res_y))
        col = int(round((float(transform.c) - roi_xmin) / res_x))
        tile_h, tile_w = prediction.shape

        r0 = max(0, row)
        c0 = max(0, col)
        r1 = min(out_h, row + tile_h)
        c1 = min(out_w, col + tile_w)
        if r1 <= r0 or c1 <= c0:
            continue

        patch = prediction[r0 - row : r1 - row, c0 - col : c1 - col]

        if reducer == "mean":
            valid = np.isfinite(patch)
            if np.any(valid):
                block_sum = sum_grid[r0:r1, c0:c1]
                block_count = count_grid[r0:r1, c0:c1]
                block_sum[valid] += patch[valid]
                block_count[valid] += 1
                sum_grid[r0:r1, c0:c1] = block_sum
                count_grid[r0:r1, c0:c1] = block_count
        else:
            stitched[r0:r1, c0:c1] = patch

    if reducer == "mean":
        stitched = np.full((out_h, out_w), fill_value, dtype=np.float32)
        valid = count_grid > 0
        stitched[valid] = (sum_grid[valid] / count_grid[valid]).astype(np.float32)

    return stitched
