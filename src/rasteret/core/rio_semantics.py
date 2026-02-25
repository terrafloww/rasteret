# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Rasterio/GDAL semantic helpers (integration boundary).

Rasteret's core reader operates on a raster's native pixel grid. Some integrations
(notably TorchGeo) define *query-grid* semantics: given bounds + resolution from a
sampler, return pixels on that exact grid.

Rasterio exposes two different, widely-used semantics:
  - ``rasterio.merge.merge`` (gdal_merge-style window alignment)
  - ``rasterio.warp.reproject`` (warp semantics based on destination pixel centers)

These can differ sometimes in output. TorchGeo's
``RasterDataset`` uses ``merge(bounds=..., res=...)`` in its read path, so Rasteret
needs to match *merge semantics* for TorchGeo interop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from affine import Affine


@dataclass(frozen=True)
class MergeGrid:
    """Query grid definition (bounds + resolution)."""

    bounds: tuple[float, float, float, float]  # (left, bottom, right, top)
    res: tuple[float, float]  # (xres, yres) in CRS units, positive numbers

    @property
    def transform(self) -> Affine:
        left, _bottom, _right, top = self.bounds
        xres, yres = self.res
        return Affine(xres, 0.0, float(left), 0.0, -yres, float(top))

    @property
    def shape(self) -> tuple[int, int]:
        """Match rasterio.merge.merge output shape computation (round)."""
        left, bottom, right, top = self.bounds
        xres, yres = self.res
        width = int(round((right - left) / xres))
        height = int(round((top - bottom) / yres))
        return height, width


def merge_semantic_resample_single_source(
    src_crop: np.ndarray,
    *,
    src_crop_transform: Affine,
    src_full_transform: Affine,
    src_full_width: int,
    src_full_height: int,
    src_crs: int,
    grid: MergeGrid,
    resampling: Literal["nearest", "bilinear"],
    src_nodata: float | int | None,
) -> np.ndarray:
    """Resample *src_crop* onto *grid* using rasterio.merge.merge semantics.

    This intentionally delegates the semantics to ``rasterio.merge.merge``.
    Rasterio's merge behavior is what TorchGeo uses, and it is known to differ
    from warp/reproject behavior by one pixel at extent boundaries.
    """
    from rasterio.crs import CRS as RioCRS
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.merge import merge as rio_merge

    if src_crop.ndim != 2:
        raise ValueError(f"Expected 2-D src_crop, got shape={src_crop.shape}")

    # rasterio.merge.merge cannot merge upside-down rasters directly (south-up).
    # TorchGeo's merge-based semantics in practice behave like a north-up view.
    #
    # Normalize south-up sources into a north-up equivalent *without resampling*:
    # flip data vertically and adjust transforms to keep georeferencing correct.
    if float(src_crop_transform.e) > 0.0:
        src_crop = np.ascontiguousarray(src_crop[::-1, :])
        src_crop_transform = Affine(
            float(src_crop_transform.a),
            0.0,
            float(src_crop_transform.c),
            0.0,
            -float(src_crop_transform.e),
            float(src_crop_transform.f)
            + float(src_crop.shape[0]) * float(src_crop_transform.e),
        )
        src_full_transform = Affine(
            float(src_full_transform.a),
            0.0,
            float(src_full_transform.c),
            0.0,
            -float(src_full_transform.e),
            float(src_full_transform.f)
            + float(src_full_height) * float(src_full_transform.e),
        )

    if (
        src_nodata is not None
        and isinstance(src_nodata, float)
        and np.isnan(src_nodata)
    ):
        # A NaN nodata is only meaningful for floating rasters.
        if src_crop.dtype.kind != "f":
            src_nodata = None
    dst_h, dst_w = grid.shape
    if dst_h <= 0 or dst_w <= 0:
        return np.zeros((0, 0), dtype=src_crop.dtype)

    # Clip the crop to the physical extent of the full raster.
    #
    # Rasteret's window-mode reader can return a *boundless* crop that extends
    # beyond the raster's physical extent (filled with 0/nodata). If we pass
    # this boundless crop directly to rasterio.merge.merge, merge will treat
    # those extended bounds as valid source bounds and can shift boundary
    # behavior (notably for sub-pixel-aligned query grids like NAIP).
    #
    # We therefore slice away any rows/cols that are fully outside the physical
    # extent so the in-memory dataset's bounds match the real raster's bounds.
    a = float(src_crop_transform.a)
    e = float(src_crop_transform.e)
    if a <= 0.0 or e >= 0.0:
        raise ValueError(
            "Expected a north-up transform for merge semantics " f"(a={a}, e={e})."
        )

    crop_w = int(src_crop.shape[1])
    crop_h = int(src_crop.shape[0])
    crop_left = float(src_crop_transform.c)
    crop_top = float(src_crop_transform.f)
    crop_right = crop_left + crop_w * a
    crop_bottom = crop_top + crop_h * e

    full_left = float(src_full_transform.c)
    full_top = float(src_full_transform.f)
    full_right = full_left + int(src_full_width) * float(src_full_transform.a)
    full_bottom = full_top + int(src_full_height) * float(src_full_transform.e)

    inter_left = max(crop_left, full_left)
    inter_right = min(crop_right, full_right)
    inter_top = min(crop_top, full_top)
    inter_bottom = max(crop_bottom, full_bottom)

    if inter_left < inter_right and inter_bottom < inter_top:
        import numpy as _np

        col0 = int(_np.ceil((inter_left - crop_left) / a - 1e-9))
        col1 = int(_np.floor((inter_right - crop_left) / a + 1e-9))
        row0 = int(_np.ceil((crop_top - inter_top) / abs(e) - 1e-9))
        row1 = int(_np.floor((crop_top - inter_bottom) / abs(e) + 1e-9))

        col0 = max(0, min(crop_w, col0))
        col1 = max(col0, min(crop_w, col1))
        row0 = max(0, min(crop_h, row0))
        row1 = max(row0, min(crop_h, row1))

        if row0 != 0 or col0 != 0 or row1 != crop_h or col1 != crop_w:
            src_crop = src_crop[row0:row1, col0:col1]
            src_crop_transform = Affine(
                a,
                0.0,
                crop_left + col0 * a,
                0.0,
                e,
                crop_top + row0 * e,
            )

    profile = {
        "driver": "GTiff",
        "height": int(src_crop.shape[0]),
        "width": int(src_crop.shape[1]),
        "count": 1,
        "dtype": src_crop.dtype,
        "crs": RioCRS.from_epsg(int(src_crs)),
        "transform": src_crop_transform,
        **({"nodata": src_nodata} if src_nodata is not None else {}),
    }

    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(src_crop, 1)

        with mem.open() as ds:
            data, _out_transform = rio_merge(
                [ds],
                bounds=grid.bounds,
                res=grid.res,
                indexes=[1],
                resampling=getattr(Resampling, resampling),
            )
            return data.squeeze()
