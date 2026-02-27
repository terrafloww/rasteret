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

    # Fast path: if the query grid is exactly pixel-aligned with the source grid
    # at the same resolution and using nearest resampling, we can copy directly
    # from the fetched crop into the destination canvas and skip MemoryFile+merge.
    #
    # This is conservative by design: on any ambiguity we fall back to rasterio.
    if resampling == "nearest":
        tol = 1e-9

        def _as_int_if_close(value: float) -> int | None:
            rounded = int(round(float(value)))
            if abs(float(value) - float(rounded)) <= tol:
                return rounded
            return None

        def _fill_value_for_dtype(
            dtype: np.dtype, nodata: float | int | None
        ) -> float | int:
            if nodata is None:
                return np.array(0, dtype=dtype).item()
            if isinstance(nodata, float) and np.isnan(nodata):
                if dtype.kind != "f":
                    raise ValueError("NaN nodata is only valid for float dtype")
                return np.nan
            if dtype.kind in {"i", "u", "b"}:
                # Integer-like output: require a finite integer nodata representable
                # by dtype; otherwise delegate to rasterio path.
                if isinstance(nodata, float):
                    if not np.isfinite(nodata) or not float(nodata).is_integer():
                        raise ValueError("Non-integer nodata for integer dtype")
                    nodata_i = int(nodata)
                else:
                    nodata_i = int(nodata)
                info = np.iinfo(dtype)
                if nodata_i < int(info.min) or nodata_i > int(info.max):
                    raise ValueError("nodata outside integer dtype range")
                return np.array(nodata_i, dtype=dtype).item()
            if dtype.kind == "f":
                nodata_f = float(nodata)
                if np.isfinite(nodata_f):
                    info = np.finfo(dtype)
                    if nodata_f < float(info.min) or nodata_f > float(info.max):
                        raise ValueError("nodata outside float dtype range")
                return np.array(nodata_f, dtype=dtype).item()
            raise ValueError(f"Unsupported dtype for fast path: {dtype}")

        full_a = float(src_full_transform.a)
        full_b = float(src_full_transform.b)
        full_d = float(src_full_transform.d)
        full_e = float(src_full_transform.e)

        crop_a = float(src_crop_transform.a)
        crop_b = float(src_crop_transform.b)
        crop_d = float(src_crop_transform.d)
        crop_e = float(src_crop_transform.e)

        if (
            abs(full_b) <= tol
            and abs(full_d) <= tol
            and abs(crop_b) <= tol
            and abs(crop_d) <= tol
            and full_a > 0.0
            and full_e < 0.0
            and abs(crop_a - full_a) <= tol
            and abs(crop_e - full_e) <= tol
            and abs(float(grid.res[0]) - full_a) <= tol
            and abs(float(grid.res[1]) - abs(full_e)) <= tol
        ):
            full_left = float(src_full_transform.c)
            full_top = float(src_full_transform.f)
            crop_left = float(src_crop_transform.c)
            crop_top = float(src_crop_transform.f)
            grid_left, _grid_bottom, _grid_right, grid_top = grid.bounds

            q_col0 = _as_int_if_close((float(grid_left) - full_left) / full_a)
            q_row0 = _as_int_if_close((full_top - float(grid_top)) / abs(full_e))
            c_col0 = _as_int_if_close((crop_left - full_left) / full_a)
            c_row0 = _as_int_if_close((full_top - crop_top) / abs(full_e))

            if (
                q_col0 is not None
                and q_row0 is not None
                and c_col0 is not None
                and c_row0 is not None
            ):
                try:
                    fill_value = _fill_value_for_dtype(src_crop.dtype, src_nodata)
                except ValueError:
                    fill_value = None

                if fill_value is not None:
                    dst_h, dst_w = grid.shape
                    if dst_h <= 0 or dst_w <= 0:
                        return np.zeros((0, 0), dtype=src_crop.dtype)

                    out = np.full((dst_h, dst_w), fill_value, dtype=src_crop.dtype)

                    # Relative offset from the requested query-grid top-left pixel
                    # to the provided source crop top-left pixel.
                    rel_col = int(q_col0) - int(c_col0)
                    rel_row = int(q_row0) - int(c_row0)

                    src_h, src_w = src_crop.shape

                    src_r0 = max(0, rel_row)
                    src_c0 = max(0, rel_col)
                    dst_r0 = max(0, -rel_row)
                    dst_c0 = max(0, -rel_col)

                    copy_h = min(src_h - src_r0, dst_h - dst_r0)
                    copy_w = min(src_w - src_c0, dst_w - dst_c0)

                    if copy_h > 0 and copy_w > 0:
                        out[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w] = (
                            src_crop[src_r0 : src_r0 + copy_h, src_c0 : src_c0 + copy_w]
                        )
                    return out

    from rasterio.crs import CRS as RioCRS
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.merge import merge as rio_merge

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
