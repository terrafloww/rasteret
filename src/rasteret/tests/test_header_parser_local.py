# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from pathlib import Path

import pytest
import tifffile as tf

from rasteret.fetch.header_parser import AsyncCOGHeaderParser


async def _read_local_range(url: str, start: int, size: int) -> bytes:
    path = Path(url)
    with path.open("rb") as f:
        f.seek(start)
        return f.read(size)


@pytest.mark.asyncio
async def test_header_parser_matches_tiled_fixture_metadata(
    tmp_path: Path,
) -> None:
    import numpy as np

    fixture = tmp_path / "tiled-rgb-u8.tif"
    data = np.zeros((128, 128, 3), dtype=np.uint8)
    extratags = [
        (33550, "d", 3, (1.0, 1.0, 0.0), False),  # ModelPixelScaleTag
        (33922, "d", 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), False),  # ModelTiepointTag
    ]
    tf.imwrite(fixture, data, tile=(64, 64), photometric="rgb", extratags=extratags)

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]

    meta = await parser.parse_cog_header(str(fixture))
    assert meta is not None

    with tf.TiffFile(fixture) as tif:
        page = tif.pages[0]
        assert page.is_tiled

        assert meta.width == page.imagewidth
        assert meta.height == page.imagelength
        assert meta.tile_width == page.tilewidth
        assert meta.tile_height == page.tilelength

        assert meta.tile_offsets is not None
        assert meta.tile_byte_counts is not None
        assert len(meta.tile_offsets) == len(page.dataoffsets)
        assert len(meta.tile_byte_counts) == len(page.databytecounts)
        assert meta.tile_offsets[:3] == list(page.dataoffsets)[:3]
        assert meta.tile_byte_counts[:3] == list(page.databytecounts)[:3]


@pytest.mark.asyncio
async def test_header_parser_skips_non_tiled_bigtiff(
    async_tiff_bigtiff_fixtures: Path,
) -> None:
    """Non-tiled BigTIFF fixtures should be rejected (not tiled => unsupported).

    This validates that the parser correctly handles BigTIFF headers
    (endianness, 64-bit offsets) without crashing, while properly
    rejecting non-tiled files that aren't usable as COGs.
    """
    fixtures = [
        async_tiff_bigtiff_fixtures / "BigTIFF.tif",
        async_tiff_bigtiff_fixtures / "BigTIFFLong.tif",
        async_tiff_bigtiff_fixtures / "BigTIFFMotorola.tif",
    ]

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]

    for path in fixtures:
        with pytest.raises(NotImplementedError, match=r"tiled GeoTIFF/COG"):
            await parser.parse_cog_header(str(path))


@pytest.mark.asyncio
async def test_header_parser_parses_model_transformation_tag(tmp_path: Path) -> None:
    """ModelTransformationTag (34264) should populate the affine transform."""
    import numpy as np

    fixture = tmp_path / "model_transform.tif"
    data = np.zeros((128, 128), dtype=np.uint8)

    # Row-major 4x4 matrix:
    # x = 10 * col + 500000
    # y = 10 * row + 1000000  (south-up / bottom-up orientation)
    model_transform = (
        10.0,
        0.0,
        0.0,
        500000.0,
        0.0,
        10.0,
        0.0,
        1000000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    tf.imwrite(
        fixture,
        data,
        tile=(64, 64),
        extratags=[(34264, "d", 16, model_transform, False)],
    )

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]

    meta = await parser.parse_cog_header(str(fixture))
    assert meta is not None
    assert meta.transform == (10.0, 500000.0, 10.0, 1000000.0)


# ---------------------------------------------------------------------------
# PixelIsPoint / PixelIsArea correction (GDAL RFC 33)
# ---------------------------------------------------------------------------


def _geokey_directory(*keys: tuple[int, int, int, int]) -> tuple[int, ...]:
    """Build a GeoKeyDirectory (tag 34735) value tuple."""
    header = (1, 1, 0, len(keys))
    body: tuple[int, ...] = ()
    for k in keys:
        body += k
    return header + body


@pytest.mark.asyncio
async def test_pixel_is_point_correction(tmp_path: Path) -> None:
    """PixelIsPoint (key 1025 = 2) should shift origin by half a pixel."""
    import numpy as np

    fixture = tmp_path / "pixel_is_point.tif"
    data = np.zeros((128, 128), dtype=np.uint8)

    geokeys = _geokey_directory((1025, 0, 1, 2))  # RasterPixelIsPoint
    extratags = [
        (33550, "d", 3, (30.0, 30.0, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, 459600.0, 4264200.0, 0.0), False),
        (34735, "H", len(geokeys), geokeys, False),
    ]
    tf.imwrite(fixture, data, tile=(64, 64), extratags=extratags)

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]

    meta = await parser.parse_cog_header(str(fixture))
    assert meta is not None
    # translate_x = 459600 - 30/2 = 459585
    # translate_y = 4264200 - (-30)/2 = 4264215
    assert meta.transform == pytest.approx((30.0, 459585.0, -30.0, 4264215.0))
    # Raw tags untouched.
    assert meta.tiepoint[3] == pytest.approx(459600.0)
    assert meta.tiepoint[4] == pytest.approx(4264200.0)


@pytest.mark.asyncio
async def test_pixel_is_area_no_correction(tmp_path: Path) -> None:
    """PixelIsArea (key 1025 = 1) should NOT shift the origin."""
    import numpy as np

    fixture = tmp_path / "pixel_is_area.tif"
    data = np.zeros((128, 128), dtype=np.uint8)

    geokeys = _geokey_directory((1025, 0, 1, 1))  # RasterPixelIsArea
    extratags = [
        (33550, "d", 3, (30.0, 30.0, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, 459600.0, 4264200.0, 0.0), False),
        (34735, "H", len(geokeys), geokeys, False),
    ]
    tf.imwrite(fixture, data, tile=(64, 64), extratags=extratags)

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]

    meta = await parser.parse_cog_header(str(fixture))
    assert meta is not None
    assert meta.transform == pytest.approx((30.0, 459600.0, -30.0, 4264200.0))


@pytest.mark.asyncio
async def test_tiepoint_nonzero_ij_offsets_adjust_translation(tmp_path: Path) -> None:
    """Non-zero (I,J) tiepoint must be converted to pixel(0,0) translation."""
    import numpy as np

    fixture = tmp_path / "tiepoint_ij.tif"
    data = np.zeros((64, 64), dtype=np.uint8)

    # Pixel scale: 10m. Tiepoint says pixel (I=2, J=3) maps to (X=1000, Y=2000).
    # Therefore pixel (0,0) should map to:
    #   X0 = 1000 - 2*10 = 980
    #   Y0 = 2000 - 3*(-10) = 2030  (north-up scale_y = -10)
    extratags = [
        (33550, "d", 3, (10.0, 10.0, 0.0), False),
        (33922, "d", 6, (2.0, 3.0, 0.0, 1000.0, 2000.0, 0.0), False),
    ]
    tf.imwrite(fixture, data, tile=(32, 32), extratags=extratags)

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]

    meta = await parser.parse_cog_header(str(fixture))
    assert meta is not None
    assert meta.transform == pytest.approx((10.0, 980.0, -10.0, 2030.0))


@pytest.mark.asyncio
async def test_header_parser_raises_on_unsupported_dtype(tmp_path: Path) -> None:
    """Unsupported SampleFormat/BitsPerSample should hard-fail (not default uint8)."""
    import numpy as np

    fixture = tmp_path / "unsupported_dtype.tif"
    data = np.zeros((128, 128), dtype=np.uint8)
    tf.imwrite(fixture, data, tile=(64, 64))

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]
    parser.dtype_map = {}  # type: ignore[assignment]

    with pytest.raises(NotImplementedError, match=r"Unsupported TIFF dtype"):
        await parser.parse_cog_header(str(fixture))


@pytest.mark.asyncio
async def test_header_parser_raises_on_jpeg_compression(tmp_path: Path) -> None:
    """JPEG-compressed TIFFs must hard-fail until decode is implemented."""
    import numpy as np

    fixture = tmp_path / "jpeg_compressed.tif"
    data = np.zeros((128, 128), dtype=np.uint8)

    try:
        tf.imwrite(fixture, data, tile=(64, 64), compression="jpeg")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"JPEG compression not available in this environment: {exc}")

    parser = AsyncCOGHeaderParser(max_concurrent=1, batch_size=1)
    parser._fetch_byte_range = _read_local_range  # type: ignore[method-assign]

    with pytest.raises(NotImplementedError, match=r"JPEG compression"):
        await parser.parse_cog_header(str(fixture))
