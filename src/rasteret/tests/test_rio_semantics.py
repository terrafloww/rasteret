# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from rasteret.core.rio_semantics import MergeGrid, merge_semantic_resample_single_source

pytest.importorskip("rasterio")


def _merge_oracle(
    arr: np.ndarray,
    *,
    transform: Affine,
    crs_epsg: int,
    bounds: tuple[float, float, float, float],
    res: tuple[float, float],
) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.merge import merge as rio_merge

    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": arr.dtype,
        "crs": rasterio.crs.CRS.from_epsg(int(crs_epsg)),
        "transform": transform,
    }

    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(arr, 1)
        with mem.open() as ds:
            data, _out_transform = rio_merge(
                [ds],
                bounds=bounds,
                res=res,
                indexes=[1],
                resampling=Resampling.nearest,
            )
            return data.squeeze()


def test_merge_semantics_matches_rasterio_merge_at_extent_boundary() -> None:
    rng = np.random.default_rng(17)
    src = rng.integers(0, 1000, size=(9, 11), dtype=np.uint16)
    src_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4200000.0)
    crs = 3857
    grid = MergeGrid(
        bounds=(500020.0, 4199905.0, 500110.0, 4200005.0),
        res=(10.0, 10.0),
    )
    oracle = _merge_oracle(
        src, transform=src_transform, crs_epsg=crs, bounds=grid.bounds, res=grid.res
    )

    out = merge_semantic_resample_single_source(
        src,
        src_crop_transform=src_transform,
        src_full_transform=src_transform,
        src_full_width=11,
        src_full_height=9,
        src_crs=crs,
        grid=grid,
        resampling="nearest",
        src_nodata=None,
    )

    assert out.shape == oracle.shape
    np.testing.assert_array_equal(out, oracle)


def test_merge_semantics_with_crop_transform_translation() -> None:
    full = (np.arange(100, dtype=np.uint16).reshape(10, 10) + 1).astype(np.uint16)
    full_transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    crs = 3857

    crop = full[1:6, 3:7]
    crop_transform = Affine(1.0, 0.0, 103.0, 0.0, -1.0, 199.0)

    grid = MergeGrid(bounds=(103.0, 194.6, 107.0, 198.6), res=(1.0, 1.0))
    oracle = _merge_oracle(
        full,
        transform=full_transform,
        crs_epsg=crs,
        bounds=grid.bounds,
        res=grid.res,
    )

    out = merge_semantic_resample_single_source(
        crop,
        src_crop_transform=crop_transform,
        src_full_transform=full_transform,
        src_full_width=10,
        src_full_height=10,
        src_crs=crs,
        grid=grid,
        resampling="nearest",
        src_nodata=None,
    )

    assert out.shape == oracle.shape
    np.testing.assert_array_equal(out, oracle)


def test_merge_semantics_aligned_nearest_uses_fast_path(monkeypatch) -> None:
    src = (np.arange(25, dtype=np.uint16).reshape(5, 5) + 1).astype(np.uint16)
    src_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
    crs = 3857
    grid = MergeGrid(bounds=(1.0, 1.0, 4.0, 4.0), res=(1.0, 1.0))

    import rasterio.io

    class _BoomMemoryFile:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Expected fast path without MemoryFile fallback")

    monkeypatch.setattr(rasterio.io, "MemoryFile", _BoomMemoryFile)

    out = merge_semantic_resample_single_source(
        src,
        src_crop_transform=src_transform,
        src_full_transform=src_transform,
        src_full_width=5,
        src_full_height=5,
        src_crs=crs,
        grid=grid,
        resampling="nearest",
        src_nodata=0,
    )

    expected = src[1:4, 1:4]
    np.testing.assert_array_equal(out, expected)
