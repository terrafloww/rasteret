# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile as tf

from rasteret.fetch.cog import read_cog
from rasteret.types import CogMetadata


def _first(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return value[0]
    if isinstance(value, np.ndarray):
        return value.flat[0]
    return value


def _cog_metadata_from_tifffile_page(
    page: "tf.TiffPage",
    *,
    transform_override: list[float] | None = None,
) -> CogMetadata:
    # NOTE: This helper does NOT apply PixelIsPoint correction - it is only
    # used with PixelIsArea test fixtures.  The real parser in header_parser.py
    # handles the correction via get_raster_type_from_geokeys().
    tags = page.tags

    width = int(_first(tags[256].value))
    height = int(_first(tags[257].value))
    tile_width = int(_first(tags.get(322, width).value if 322 in tags else width))
    tile_height = int(_first(tags.get(323, height).value if 323 in tags else height))
    predictor = int(_first(tags.get(317, 1).value if 317 in tags else 1))
    compression = int(_first(tags.get(259, 1).value if 259 in tags else 1))
    samples_per_pixel = int(_first(tags.get(277, 1).value if 277 in tags else 1))
    planar_configuration = int(_first(tags.get(284, 1).value if 284 in tags else 1))
    photometric = int(_first(tags.get(262).value)) if 262 in tags else None
    extra_samples = (
        tuple(tags.get(338).value) if 338 in tags and tags.get(338).value else None
    )

    tile_offsets = list(tags.get(324).value) if 324 in tags else []
    tile_byte_counts = list(tags.get(325).value) if 325 in tags else []

    pixel_scale = tuple(tags.get(33550).value) if 33550 in tags else None
    tiepoint = tuple(tags.get(33922).value) if 33922 in tags else None

    transform = transform_override
    if transform is None and pixel_scale and tiepoint:
        transform = [
            float(pixel_scale[0]),
            float(tiepoint[3]),
            -float(pixel_scale[1]),
            float(tiepoint[4]),
        ]

    return CogMetadata(
        width=width,
        height=height,
        tile_width=tile_width,
        tile_height=tile_height,
        dtype=np.dtype(page.dtype),
        crs=None,
        predictor=predictor,
        transform=transform,
        compression=compression,
        tile_offsets=[int(v) for v in tile_offsets],
        tile_byte_counts=[int(v) for v in tile_byte_counts],
        pixel_scale=pixel_scale,
        tiepoint=tiepoint,
        samples_per_pixel=samples_per_pixel,
        planar_configuration=planar_configuration,
        photometric=photometric,
        extra_samples=extra_samples,
    )


@pytest.mark.asyncio
async def test_read_cog_local_tiled_geotiff_roundtrip(tmp_path: Path) -> None:
    data = (np.arange(128 * 128, dtype=np.uint16).reshape(128, 128) % 1024).astype(
        np.uint16
    )

    path = tmp_path / "tiled_geotiff.tif"
    scale_x = 10.0
    scale_y = 10.0
    tie_x = 1000.0
    tie_y = 2000.0
    extratags = [
        # ModelPixelScaleTag (33550) - DOUBLE * 3
        (33550, "d", 3, (scale_x, scale_y, 0.0), False),
        # ModelTiepointTag (33922) - DOUBLE * 6
        (33922, "d", 6, (0.0, 0.0, 0.0, tie_x, tie_y, 0.0), False),
    ]

    tf.imwrite(
        path,
        data,
        tile=(64, 64),
        photometric="minisblack",
        compression=None,
        predictor=False,
        extratags=extratags,
    )

    with tf.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.is_tiled
        meta = _cog_metadata_from_tifffile_page(page)

    result = await read_cog(str(path), meta)
    out, out_transform = result.data, result.transform
    assert out.shape == data.shape
    assert out.dtype == data.dtype
    np.testing.assert_array_equal(out, data)

    assert out_transform is not None
    assert out_transform.a == pytest.approx(scale_x)
    assert out_transform.c == pytest.approx(tie_x)
    assert out_transform.e == pytest.approx(-scale_y)
    assert out_transform.f == pytest.approx(tie_y)


@pytest.mark.asyncio
async def test_read_cog_local_non_tiled_geotiff_raises(
    tmp_path: Path,
) -> None:
    data = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    path = tmp_path / "striped_geotiff.tif"

    extratags = [
        (33550, "d", 3, (1.0, 1.0, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), False),
    ]

    tf.imwrite(
        path,
        data,
        photometric="minisblack",
        compression=None,
        predictor=False,
        extratags=extratags,
    )

    with tf.TiffFile(path) as tif:
        meta = _cog_metadata_from_tifffile_page(tif.pages[0])

    # This file is not tiled, so it lacks TileOffsets/TileByteCounts.
    assert not meta.tile_offsets
    assert not meta.tile_byte_counts

    with pytest.raises(ValueError, match=r"requires a tiled GeoTIFF"):
        await read_cog(str(path), meta)


@pytest.mark.asyncio
async def test_read_cog_local_tiled_non_geotiff_fixture_smoke(
    async_tiff_fixtures: Path,
) -> None:
    fixture = async_tiff_fixtures / "tiled-oversize-gray-i8.tif"

    with tf.TiffFile(fixture) as tif:
        page = tif.pages[0]
        assert page.is_tiled
        # These async-tiff fixtures are image TIFFs (not GeoTIFF), so we supply
        # a synthetic transform to exercise local tile I/O + decoding.
        meta = _cog_metadata_from_tifffile_page(
            page, transform_override=[1.0, 0.0, -1.0, 0.0]
        )

    result = await read_cog(str(fixture), meta)
    out, out_transform = result.data, result.transform
    assert out_transform is not None
    assert out.shape[0] >= meta.height
    assert out.shape[1] >= meta.width

    expected = tf.imread(fixture).astype(np.float32)
    np.testing.assert_array_equal(out[: meta.height, : meta.width], expected)


@pytest.mark.asyncio
async def test_read_cog_local_chunky_multisample_band_selection(
    tmp_path: Path,
) -> None:
    data = np.zeros((128, 128, 3), dtype=np.uint8)
    data[:, :, 0] = 10
    data[:, :, 1] = 20
    data[:, :, 2] = 30

    path = tmp_path / "chunky_rgb.tif"
    extratags = [
        (33550, "d", 3, (1.0, 1.0, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, 1000.0, 2000.0, 0.0), False),
    ]

    tf.imwrite(
        path,
        data,
        tile=(64, 64),
        photometric="rgb",
        planarconfig="contig",
        compression=None,
        predictor=False,
        extratags=extratags,
    )

    with tf.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.is_tiled
        meta = _cog_metadata_from_tifffile_page(page)

    with pytest.raises(NotImplementedError, match=r"require an explicit band_index"):
        await read_cog(str(path), meta)

    for idx, expected in enumerate([10, 20, 30]):
        result = await read_cog(str(path), meta, band_index=idx)
        out, out_transform = result.data, result.transform
        assert out_transform is not None
        assert out.shape == (128, 128)
        np.testing.assert_array_equal(
            out, np.full((128, 128), expected, dtype=out.dtype)
        )
