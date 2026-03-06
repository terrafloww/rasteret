# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for the COG reader: codecs, dtypes, predictors, tile math.

Real-world fixture tests sourced from the async-tiff project (MIT,
image-tiff fixtures under libtiff permissive license).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from rasteret.fetch.cog import (
    COGReader,
    COGTileRequest,
    apply_mask_and_crop,
    compute_tile_indices,
    merge_tiles,
    read_cog,
)
from rasteret.types import CogMetadata

# ---------------------------------------------------------------------------
# Optional test dependency
# ---------------------------------------------------------------------------

try:
    import tifffile  # type: ignore
except ImportError:  # pragma: no cover
    tifffile = None

has_tifffile = tifffile is not None
needs_tifffile = pytest.mark.skipif(
    not has_tifffile, reason="Install tifffile to run TIFF fixture tests"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_tiff_metadata(path: Path) -> dict:
    """Read tile offsets/byte counts from a TIFF for test metadata."""
    assert tifffile is not None
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        return {
            "width": page.imagewidth,
            "height": page.imagelength,
            "tile_width": page.tilewidth,
            "tile_height": page.tilelength,
            "dtype": str(page.dtype),
            "compression": page.compression.value,
            "predictor": page.predictor.value
            if hasattr(page.predictor, "value")
            else int(page.predictor),
            "tile_offsets": list(page.dataoffsets),
            "tile_byte_counts": list(page.databytecounts),
            "samples_per_pixel": page.samplesperpixel,
        }


def _make_cog_metadata(meta: dict, crs: int = 32643) -> CogMetadata:
    """Build CogMetadata from tiff metadata dict."""
    return CogMetadata(
        width=meta["width"],
        height=meta["height"],
        tile_width=meta["tile_width"],
        tile_height=meta["tile_height"],
        dtype=np.dtype(meta["dtype"]),
        crs=crs,
        compression=meta["compression"],
        predictor=meta["predictor"],
        tile_offsets=meta["tile_offsets"],
        tile_byte_counts=meta["tile_byte_counts"],
        transform=[10.0, 500000.0, -10.0, 1000000.0],
    )


def _read_raw_tile(path: Path, tile_index: int, meta: dict) -> bytes:
    """Read raw (compressed) tile bytes from a TIFF on disk."""
    offset = meta["tile_offsets"][tile_index]
    size = meta["tile_byte_counts"][tile_index]
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def _reference_tile_data(path: Path, page_index: int = 0) -> np.ndarray:
    """Use tifffile to decode the full image (ground truth)."""
    assert tifffile is not None
    with tifffile.TiffFile(str(path)) as tif:
        return tif.pages[page_index].asarray()


# ---------------------------------------------------------------------------
# Pure logic tests (no I/O)
# ---------------------------------------------------------------------------


class TestMergeRanges:
    def test_empty(self):
        reader = COGReader.__new__(COGReader)
        assert reader.merge_ranges([]) == []

    def test_single_request(self):
        reader = COGReader.__new__(COGReader)
        meta = CogMetadata(
            width=16,
            height=16,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint16"),
            crs=32643,
            tile_offsets=[100],
            tile_byte_counts=[512],
        )
        req = COGTileRequest(url="x", offset=100, size=512, row=0, col=0, metadata=meta)
        merged = reader.merge_ranges([req])
        assert merged == [(100, 612)]

    def test_adjacent_ranges_merge(self):
        reader = COGReader.__new__(COGReader)
        meta = CogMetadata(
            width=16,
            height=16,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint16"),
            crs=32643,
            tile_offsets=[100, 700],
            tile_byte_counts=[512, 512],
        )
        r1 = COGTileRequest(url="x", offset=100, size=512, row=0, col=0, metadata=meta)
        r2 = COGTileRequest(url="x", offset=700, size=512, row=0, col=1, metadata=meta)
        merged = reader.merge_ranges([r1, r2], gap_threshold=100)
        assert merged == [(100, 1212)]

    def test_distant_ranges_stay_separate(self):
        reader = COGReader.__new__(COGReader)
        meta = CogMetadata(
            width=16,
            height=16,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint16"),
            crs=32643,
            tile_offsets=[100, 100000],
            tile_byte_counts=[512, 512],
        )
        r1 = COGTileRequest(url="x", offset=100, size=512, row=0, col=0, metadata=meta)
        r2 = COGTileRequest(
            url="x", offset=100000, size=512, row=0, col=1, metadata=meta
        )
        merged = reader.merge_ranges([r1, r2], gap_threshold=1024)
        assert len(merged) == 2


def test_cog_reader_reads_local_file_ranges(tmp_path: Path) -> None:
    payload = b"0123456789abcdef"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    async def _read() -> bytes:
        async with COGReader(max_concurrent=1) as reader:
            return await reader._read_range(str(path), 2, 6)

    out = asyncio.run(_read())
    assert out == payload[2:6]

    async def _read_file_uri() -> bytes:
        async with COGReader(max_concurrent=1) as reader:
            return await reader._read_range(path.as_uri(), 2, 6)

    out_uri = asyncio.run(_read_file_uri())
    assert out_uri == payload[2:6]


@pytest.mark.asyncio
async def test_read_tile_samples_reads_once_for_multiple_band_indices():
    reader = COGReader.__new__(COGReader)
    reader.sem = asyncio.Semaphore(1)

    meta = CogMetadata(
        width=4,
        height=4,
        tile_width=4,
        tile_height=4,
        dtype=np.dtype("uint16"),
        crs=4326,
        tile_offsets=[16],
        tile_byte_counts=[8],
        samples_per_pixel=2,
        planar_configuration=1,
    )

    read_calls: list[tuple[str, int, int]] = []

    async def _fake_read_range(url: str, start: int, end: int) -> bytes:
        read_calls.append((url, start, end))
        return b"x" * (end - start)

    def _fake_process(
        data: bytes,
        metadata: CogMetadata,
        band_index: int | None = None,
    ) -> np.ndarray:
        # No compression in this fixture: `_decompress_tile_sync` is the identity.
        assert data == b"x" * 8
        assert metadata is meta
        return np.full((4, 4), int(band_index), dtype=np.int16)

    reader._read_range = _fake_read_range  # type: ignore[method-assign]
    reader._process_tile_sync = _fake_process  # type: ignore[method-assign]

    tiles = await reader.read_tile_samples(
        url="https://example.com/example.tif",
        metadata=meta,
        tile_row=0,
        tile_col=0,
        band_indices=[0, 1],
    )

    assert read_calls == [("https://example.com/example.tif", 16, 24)]
    assert len(tiles) == 2
    np.testing.assert_array_equal(tiles[0], np.zeros((4, 4), dtype=np.int16))
    np.testing.assert_array_equal(tiles[1], np.ones((4, 4), dtype=np.int16))


class TestMergeTiles:
    def test_single_tile(self):
        tile = np.ones((16, 16), dtype=np.float32)
        merged, bounds, tile_mask = merge_tiles(
            {(0, 0): tile}, (16, 16), dtype=tile.dtype, fill_value=0
        )
        assert merged.shape == (16, 16)
        assert bounds == (0, 0, 0, 0)
        np.testing.assert_array_equal(merged, tile)
        assert tile_mask.shape == merged.shape
        assert np.all(tile_mask)

    def test_two_by_two(self):
        tiles = {
            (0, 0): np.full((8, 8), 1.0, dtype=np.float32),
            (0, 1): np.full((8, 8), 2.0, dtype=np.float32),
            (1, 0): np.full((8, 8), 3.0, dtype=np.float32),
            (1, 1): np.full((8, 8), 4.0, dtype=np.float32),
        }
        merged, bounds, tile_mask = merge_tiles(
            tiles, (8, 8), dtype=np.dtype("float32"), fill_value=0
        )
        assert merged.shape == (16, 16)
        assert bounds == (0, 0, 1, 1)
        np.testing.assert_array_equal(merged[:8, :8], 1.0)
        np.testing.assert_array_equal(merged[:8, 8:], 2.0)
        np.testing.assert_array_equal(merged[8:, :8], 3.0)
        np.testing.assert_array_equal(merged[8:, 8:], 4.0)
        assert tile_mask.shape == merged.shape
        assert np.all(tile_mask)

    def test_empty_tiles(self):
        merged, _, tile_mask = merge_tiles(
            {}, (16, 16), dtype=np.dtype("uint8"), fill_value=0
        )
        assert merged.size == 0
        assert tile_mask.size == 0


# ---------------------------------------------------------------------------
# Error path tests (no fixtures needed)
# ---------------------------------------------------------------------------


class TestUnsupportedCodecs:
    def test_unsupported_compression_raises(self):
        reader = COGReader.__new__(COGReader)
        meta = CogMetadata(
            width=16,
            height=16,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint16"),
            crs=32643,
            compression=99999,
        )
        with pytest.raises(NotImplementedError, match="Unsupported TIFF compression"):
            reader._decompress_tile_sync(b"\x00" * 512, meta)

    def test_unsupported_predictor_raises(self):
        reader = COGReader.__new__(COGReader)
        meta = CogMetadata(
            width=16,
            height=16,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint16"),
            crs=32643,
            compression=1,
            predictor=99,
        )
        raw = np.zeros(256, dtype="uint16").tobytes()
        with pytest.raises(NotImplementedError, match="Unsupported TIFF predictor"):
            reader._process_tile_sync(raw, meta)


class TestReadCogTileDataPreflight:
    def test_requires_tile_offsets(self):
        meta = CogMetadata(
            width=32,
            height=32,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint8"),
            crs=4326,
            transform=[10.0, 0.0, -10.0, 0.0],
            tile_offsets=None,
            tile_byte_counts=None,
        )

        async def _run() -> None:
            await read_cog("file:///dev/null", meta)

        with pytest.raises(ValueError, match="tiled GeoTIFF"):
            asyncio.run(_run())

    def test_rejects_mismatched_tables(self):
        meta = CogMetadata(
            width=32,
            height=32,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint8"),
            crs=4326,
            transform=[10.0, 0.0, -10.0, 0.0],
            tile_offsets=[0, 10],
            tile_byte_counts=[5],
        )

        async def _run() -> None:
            await read_cog("file:///dev/null", meta)

        with pytest.raises(ValueError, match="length mismatch"):
            asyncio.run(_run())

    def test_rejects_short_offset_table(self):
        meta = CogMetadata(
            width=32,
            height=32,
            tile_width=16,
            tile_height=16,
            dtype=np.dtype("uint8"),
            crs=4326,
            transform=[10.0, 0.0, -10.0, 0.0],
            tile_offsets=[0, 10, 20],
            tile_byte_counts=[5, 5, 5],
        )

        async def _run() -> None:
            await read_cog("file:///dev/null", meta)

        with pytest.raises(ValueError, match="tile offset table is shorter"):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# apply_mask_and_crop tests (no I/O)
# ---------------------------------------------------------------------------


class TestApplyMaskAndCrop:
    def _make_data_and_transform(
        self, rows: int = 100, cols: int = 100
    ) -> tuple[np.ndarray, Affine]:
        """Create a synthetic raster grid and matching affine transform."""
        data = np.ones((rows, cols), dtype=np.float32)
        # 10m pixels, origin at (500000, 1000000) in a north-up raster
        transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 1000000.0)
        return data, transform

    def test_normal_mask(self):
        """Triangle inside raster -> cropped array with fill outside polygon."""
        data, transform = self._make_data_and_transform()
        # Triangle inside the raster; its bounding box will include corners
        # that are outside the triangle, producing NaN pixels.
        geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    (500200.0, 999500.0),
                    (500500.0, 999500.0),
                    (500350.0, 999800.0),
                    (500200.0, 999500.0),
                ]
            ],
        }

        result, result_transform = apply_mask_and_crop(data, geojson, transform)

        assert result.ndim == 2
        # Result should be smaller than the full raster
        assert result.shape[0] < data.shape[0]
        assert result.shape[1] < data.shape[1]
        # Should contain both valid (1.0) and filled pixels
        assert np.any(result == 1.0)
        assert np.any(result == 0.0)
        # Transform should be offset from the original
        assert result_transform.c >= transform.c
        assert result_transform.f <= transform.f

    def test_empty_mask_returns_nan(self):
        """Geometry does not intersect any pixels -> fill-valued array."""
        data, transform = self._make_data_and_transform()
        # Geometry far outside the raster extent
        geojson = {
            "type": "Polygon",
            "coordinates": [
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
            ],
        }

        result, result_transform = apply_mask_and_crop(data, geojson, transform)

        assert result.shape == data.shape
        assert np.all(result == 0.0)
        # Transform should be unchanged
        assert result_transform == transform

    def test_full_coverage(self):
        """Geometry covers the entire raster -> all pixels valid, no NaN."""
        data, transform = self._make_data_and_transform(rows=10, cols=10)
        # Box that fully covers the 10x10 raster
        geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    (499990.0, 999890.0),
                    (500110.0, 999890.0),
                    (500110.0, 1000010.0),
                    (499990.0, 1000010.0),
                    (499990.0, 999890.0),
                ]
            ],
        }

        result, result_transform = apply_mask_and_crop(data, geojson, transform)

        assert result.shape == data.shape
        assert not np.any(result == 0.0)


# ---------------------------------------------------------------------------
# compute_tile_indices tests (no I/O)
# ---------------------------------------------------------------------------


class TestComputeTileIndices:
    def test_single_tile_intersection(self):
        """Geometry bbox intersects exactly one tile."""
        bbox = (500050.0, 999900.0, 500150.0, 999990.0)
        transform = [10.0, 500000.0, -10.0, 1000000.0]
        indices = compute_tile_indices(
            bbox, transform, tile_size=(256, 256), image_size=(256, 256)
        )
        assert len(indices) >= 1
        assert (0, 0) in indices

    def test_south_up_positive_scale_y(self):
        """Positive scale_y (south-up / bottom-up rasters) should still work."""
        bbox = (500050.0, 1000010.0, 500150.0, 1000100.0)
        transform = [10.0, 500000.0, 10.0, 1000000.0]
        indices = compute_tile_indices(
            bbox, transform, tile_size=(256, 256), image_size=(256, 256)
        )
        assert len(indices) >= 1
        assert (0, 0) in indices

    def test_multiple_tile_intersection(self):
        """Geometry bbox spans across multiple tiles."""
        bbox = (500000.0, 996000.0, 504000.0, 1000000.0)
        transform = [10.0, 500000.0, -10.0, 1000000.0]
        # 1000x1000 image with 256x256 tiles -> 4x4 tile grid
        indices = compute_tile_indices(
            bbox, transform, tile_size=(256, 256), image_size=(1000, 1000)
        )
        assert len(indices) > 1

    def test_no_intersection(self):
        """Geometry bbox completely outside raster extent -> empty list."""
        bbox = (0.0, 0.0, 1.0, 1.0)
        transform = [10.0, 500000.0, -10.0, 1000000.0]
        indices = compute_tile_indices(
            bbox, transform, tile_size=(256, 256), image_size=(256, 256)
        )
        assert indices == []


# ---------------------------------------------------------------------------
# Real-world fixture tests: async-tiff image-tiff dataset
# ---------------------------------------------------------------------------


@needs_tifffile
class TestAsyncTiffFixtures:
    """Roundtrip tests against real TIFF files from the async-tiff repo.

    For each fixture, we:
      1. Read raw tile bytes from the TIFF on disk.
      2. Pass through COGReader._decompress_tile_sync + _process_tile_sync.
      3. Compare with tifffile's full decode (ground truth).

    These catch codec/predictor edge cases that synthetic data might miss:
    wrap-around in integer differencing, LZW dictionary resets, oversized
    tiles, rectangular (non-square) tiles, etc.
    """

    def test_tiled_oversize_lzw_int8_predictor2(self, async_tiff_fixtures: Path):
        """512x512 tile in a 499x374 image: oversized tile, LZW + predictor=2."""
        path = async_tiff_fixtures / "tiled-oversize-gray-i8.tif"
        meta_dict = _read_tiff_metadata(path)

        assert meta_dict["tile_width"] == 512
        assert meta_dict["tile_height"] == 512
        assert meta_dict["compression"] == 5  # LZW
        assert meta_dict["predictor"] == 2

        raw = _read_raw_tile(path, 0, meta_dict)
        cog_meta = _make_cog_metadata(meta_dict)

        reader = COGReader.__new__(COGReader)
        decompressed = reader._decompress_tile_sync(raw, cog_meta)
        result = reader._process_tile_sync(decompressed, cog_meta)

        assert result.dtype == np.int8
        assert result.shape == (512, 512)

        # Compare with tifffile ground truth (top-left 512x512 region, padded)
        ref = _reference_tile_data(path)
        # The image is 499x374, tile is 512x512; tifffile only gives us 374x499.
        # The tile itself may be padded. Compare the valid region.
        h, w = ref.shape
        np.testing.assert_array_equal(result[:h, :w], ref)

    def test_read_cog_local_tiled_tiff(self, async_tiff_fixtures: Path):
        """End-to-end local path read via read_cog on a tiled (non-COG) TIFF."""
        path = async_tiff_fixtures / "tiled-oversize-gray-i8.tif"
        meta_dict = _read_tiff_metadata(path)
        cog_meta = _make_cog_metadata(meta_dict)

        async def _run() -> np.ndarray:
            return (await read_cog(str(path), cog_meta)).data

        result = asyncio.run(_run())

        assert result.dtype == np.int8
        assert result.shape == (meta_dict["height"], meta_dict["width"])

        ref = _reference_tile_data(path)
        h, w = ref.shape
        np.testing.assert_array_almost_equal(result[:h, :w], ref)

    def test_predictor3_gray_f32_stripped(self, async_tiff_fixtures: Path):
        """Float predictor=3 with real LZW data (stripped, single band).

        The file uses strips, not tiles. We treat each strip as a tile for
        testing the decompress + process pipeline.
        """
        path = async_tiff_fixtures / "predictor-3-gray-f32.tif"

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            assert page.compression.name in {"LZW", "ADOBE_DEFLATE", "DEFLATE"}
            pred = (
                page.predictor.value
                if hasattr(page.predictor, "value")
                else int(page.predictor)
            )
            assert pred == 3
            assert str(page.dtype) == "float32"

            # Read the first strip
            strip_offset = page.dataoffsets[0]
            strip_bytecount = page.databytecounts[0]
            rows_per_strip = page.rowsperstrip
            width = page.imagewidth

        with open(path, "rb") as f:
            f.seek(strip_offset)
            raw_strip = f.read(strip_bytecount)

        # Build metadata for this strip-as-tile
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            comp_val = page.compression.value

        meta = CogMetadata(
            width=width,
            height=rows_per_strip,
            tile_width=width,
            tile_height=rows_per_strip,
            dtype=np.dtype("float32"),
            crs=None,
            compression=comp_val,
            predictor=3,
            tile_offsets=[strip_offset],
            tile_byte_counts=[strip_bytecount],
            transform=[1.0, 0.0, -1.0, float(rows_per_strip)],
        )

        reader = COGReader.__new__(COGReader)
        decompressed = reader._decompress_tile_sync(raw_strip, meta)
        result = reader._process_tile_sync(decompressed, meta)

        assert result.dtype == np.float32  # native dtype is float32
        assert result.shape == (rows_per_strip, width)

        # Compare with tifffile ground truth (first strip)
        ref = _reference_tile_data(path)
        np.testing.assert_array_almost_equal(result, ref[:rows_per_strip, :])

    def test_int16_zstd_stripped(self, async_tiff_fixtures: Path):
        """ZSTD-compressed int16: real-world data, stripped layout."""
        path = async_tiff_fixtures / "int16_zstd.tif"

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            assert str(page.dtype) == "int16"
            comp_val = page.compression.value

            strip_offset = page.dataoffsets[0]
            strip_bytecount = page.databytecounts[0]
            rows_per_strip = page.rowsperstrip
            width = page.imagewidth

        with open(path, "rb") as f:
            f.seek(strip_offset)
            raw_strip = f.read(strip_bytecount)

        meta = CogMetadata(
            width=width,
            height=rows_per_strip,
            tile_width=width,
            tile_height=rows_per_strip,
            dtype=np.dtype("int16"),
            crs=None,
            compression=comp_val,
            predictor=1,
            tile_offsets=[strip_offset],
            tile_byte_counts=[strip_bytecount],
        )

        reader = COGReader.__new__(COGReader)
        decompressed = reader._decompress_tile_sync(raw_strip, meta)
        result = reader._process_tile_sync(decompressed, meta)

        assert result.dtype == np.int16

        ref = _reference_tile_data(path)
        np.testing.assert_array_equal(result, ref[:rows_per_strip, :])

    def test_issue69_lzw_uint16(self, async_tiff_fixtures: Path):
        """LZW uint16 regression fixture from image-tiff (issue 69)."""
        path = async_tiff_fixtures / "issue_69_lzw.tiff"

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            assert page.compression.value == 5  # LZW
            assert str(page.dtype) == "uint16"

            strip_offset = page.dataoffsets[0]
            strip_bytecount = page.databytecounts[0]
            rows_per_strip = page.rowsperstrip
            width = page.imagewidth
            height = page.imagelength

        # This file may have fewer rows in the last strip
        actual_rows = min(rows_per_strip, height)

        with open(path, "rb") as f:
            f.seek(strip_offset)
            raw_strip = f.read(strip_bytecount)

        meta = CogMetadata(
            width=width,
            height=actual_rows,
            tile_width=width,
            tile_height=actual_rows,
            dtype=np.dtype("uint16"),
            crs=None,
            compression=5,
            predictor=1,
            tile_offsets=[strip_offset],
            tile_byte_counts=[strip_bytecount],
        )

        reader = COGReader.__new__(COGReader)
        decompressed = reader._decompress_tile_sync(raw_strip, meta)
        result = reader._process_tile_sync(decompressed, meta)

        assert result.dtype == np.uint16

        ref = _reference_tile_data(path)
        np.testing.assert_array_equal(result, ref[:actual_rows, :])

    def test_random_fp16_predictor2(self, async_tiff_fixtures: Path):
        """Float16 with integer predictor (pred=2), edge case dtype."""
        path = async_tiff_fixtures / "random-fp16-pred2.tiff"

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            pred = (
                page.predictor.value
                if hasattr(page.predictor, "value")
                else int(page.predictor)
            )
            assert pred == 2
            assert str(page.dtype) == "float16"

            strip_offset = page.dataoffsets[0]
            strip_bytecount = page.databytecounts[0]
            rows_per_strip = page.rowsperstrip
            width = page.imagewidth
            height = page.imagelength
            comp_val = page.compression.value

        actual_rows = min(rows_per_strip, height)

        with open(path, "rb") as f:
            f.seek(strip_offset)
            raw_strip = f.read(strip_bytecount)

        meta = CogMetadata(
            width=width,
            height=actual_rows,
            tile_width=width,
            tile_height=actual_rows,
            dtype=np.dtype("float16"),
            crs=None,
            compression=comp_val,
            predictor=2,
            tile_offsets=[strip_offset],
            tile_byte_counts=[strip_bytecount],
        )

        reader = COGReader.__new__(COGReader)
        decompressed = reader._decompress_tile_sync(raw_strip, meta)
        result = reader._process_tile_sync(decompressed, meta)

        assert result.dtype == np.float16

        ref = _reference_tile_data(path)
        np.testing.assert_array_almost_equal(result, ref[:actual_rows, :])

    def test_random_fp16_predictor3(self, async_tiff_fixtures: Path):
        """Float16 with floating-point predictor (pred=3), edge case."""
        path = async_tiff_fixtures / "random-fp16-pred3.tiff"

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            pred = (
                page.predictor.value
                if hasattr(page.predictor, "value")
                else int(page.predictor)
            )
            assert pred == 3
            assert str(page.dtype) == "float16"

            strip_offset = page.dataoffsets[0]
            strip_bytecount = page.databytecounts[0]
            rows_per_strip = page.rowsperstrip
            width = page.imagewidth
            height = page.imagelength
            comp_val = page.compression.value

        actual_rows = min(rows_per_strip, height)

        with open(path, "rb") as f:
            f.seek(strip_offset)
            raw_strip = f.read(strip_bytecount)

        meta = CogMetadata(
            width=width,
            height=actual_rows,
            tile_width=width,
            tile_height=actual_rows,
            dtype=np.dtype("float16"),
            crs=None,
            compression=comp_val,
            predictor=3,
            tile_offsets=[strip_offset],
            tile_byte_counts=[strip_bytecount],
        )

        reader = COGReader.__new__(COGReader)
        decompressed = reader._decompress_tile_sync(raw_strip, meta)
        result = reader._process_tile_sync(decompressed, meta)

        assert result.dtype == np.float16

        ref = _reference_tile_data(path)
        np.testing.assert_array_almost_equal(result, ref[:actual_rows, :])
