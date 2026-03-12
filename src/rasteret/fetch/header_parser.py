# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Async tiled GeoTIFF/COG header parser with connection pooling and caching.

This module parses tiled GeoTIFF headers via HTTP range requests to extract
tile layout metadata (TileOffsets/TileByteCounts) without downloading full files.

COGs are the primary target for remote access, but many non-COG *tiled* GeoTIFFs
also work since the same TIFF tags drive tile layout.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Any

from cachetools import TTLCache

from rasteret.cloud import StorageBackend
from rasteret.types import CogMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TIFF / GeoTIFF tag IDs
# ---------------------------------------------------------------------------
# Baseline TIFF 6.0
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_SAMPLES_PER_PIXEL = 277
TAG_PLANAR_CONFIGURATION = 284
TAG_PREDICTOR = 317
TAG_TILE_WIDTH = 322
TAG_TILE_LENGTH = 323
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_SAMPLE_FORMAT = 339
TAG_EXTRA_SAMPLES = 338

# GeoTIFF
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_MODEL_TRANSFORM = 34264
TAG_GEO_KEY_DIRECTORY = 34735
TAG_GEO_DOUBLE_PARAMS = 34736
TAG_GEO_ASCII_PARAMS = 34737

# GDAL extensions
TAG_GDAL_METADATA = 42112
TAG_GDAL_NODATA = 42113

COMPRESSION_JPEG = 7


def _parse_nodata(raw: str) -> float | int | None:
    """Parse a GDAL_NODATA ASCII string into a numeric value.

    Returns ``None`` when the string is empty or unparseable.
    """
    raw = raw.strip().rstrip("\x00")
    if not raw:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    # Preserve integer type when the value is integral (e.g. -128, 0, 65535).
    if not (value != value):  # not NaN
        int_val = int(value)
        if float(int_val) == value:
            return int_val
    return value


def get_crs_from_tiff_tags(tags: dict[int, Any]) -> int | None:
    """Extract CRS (EPSG code) from GeoTIFF tags.

    Tries three methods in order:
    1. GeoAsciiParamsTag (34737) - WKT string with ``ID["EPSG",...]``
    2. GeoKeyDirectory (34735) with GeoDoubleParamsTag (34736) lookup
    3. GeoKeyDirectory (34735) direct value

    Parameters
    ----------
    tags : dict
        Parsed TIFF tag dictionary.

    Returns
    -------
    int or None
        EPSG code if found, ``None`` otherwise.
    """
    # Method 1: GeoTiff WKT string (GeoAsciiParamsTag)
    if TAG_GEO_ASCII_PARAMS in tags:
        wkt = tags[TAG_GEO_ASCII_PARAMS]
        # _parse_tiff_tag_value returns tuples; unwrap to string.
        if isinstance(wkt, tuple):
            wkt = wkt[0] if wkt else ""
        try:
            import re

            # Look for EPSG code in WKT string
            epsg_match = re.search(r'ID\["EPSG",(\d+)\]', wkt)
            if epsg_match:
                return int(epsg_match.group(1))
        except (ValueError, IndexError, TypeError, re.error) as e:
            logger.debug(f"Failed to parse WKT string: {e}")

    # Method 2: GeoKey directory (tag 34735) with GeoDoubleParams (34736) support
    geokeys = tags.get(TAG_GEO_KEY_DIRECTORY)
    geo_doubles = tags.get(TAG_GEO_DOUBLE_PARAMS)
    if geokeys:
        try:
            num_keys = geokeys[3]
            # Collect both keys; ProjectedCRSGeoKey (3072) takes priority
            # over GeographicTypeGeoKey (2048) per GeoTIFF spec.
            crs_candidates: dict[int, int] = {}
            for i in range(4, 4 + (4 * num_keys), 4):
                key_id = geokeys[i]
                tiff_tag_loc = geokeys[i + 1]
                count = geokeys[i + 2]
                value = geokeys[i + 3]

                if key_id in (3072, 2048):
                    if tiff_tag_loc == 0 and count == 1:  # Direct value
                        crs_candidates[key_id] = int(value)
                    elif (
                        tiff_tag_loc == TAG_GEO_DOUBLE_PARAMS
                        and geo_doubles
                        and count == 1
                    ):
                        idx = int(value)
                        if 0 <= idx < len(geo_doubles):
                            crs_candidates[key_id] = int(geo_doubles[idx])

            if 3072 in crs_candidates:
                return crs_candidates[3072]
            if 2048 in crs_candidates:
                return crs_candidates[2048]
        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.debug(f"Failed to parse GeoKey directory: {e}")

    return None


def get_raster_type_from_geokeys(tags: dict[int, Any]) -> int:
    """Return GTRasterTypeGeoKey (1025) from tag 34735: 1=Area, 2=Point.

    Defaults to 1 (PixelIsArea) per OGC spec.
    See GDAL RFC 33: https://gdal.org/development/rfc/rfc33_gtiff_pixelispoint.html
    """
    geokeys = tags.get(TAG_GEO_KEY_DIRECTORY)
    if not geokeys:
        return 1  # default: PixelIsArea

    try:
        num_keys = geokeys[3]
        for i in range(4, 4 + (4 * num_keys), 4):
            key_id = geokeys[i]
            tiff_tag_loc = geokeys[i + 1]
            count = geokeys[i + 2]
            value = geokeys[i + 3]

            if key_id == 1025 and tiff_tag_loc == 0 and count == 1:
                return int(value)
    except (IndexError, KeyError, ValueError, TypeError) as e:
        logger.debug("Failed to parse GTRasterTypeGeoKey: %s", e)

    return 1  # default: PixelIsArea


class AsyncCOGHeaderParser:
    """Optimized async parser for tiled GeoTIFF/COG headers."""

    def __init__(
        self,
        max_concurrent: int = 300,
        batch_size: int = 100,
        cache_ttl: int = 3600,  # 1 hour
        retry_attempts: int = 3,
        backend: StorageBackend | None = None,
    ):
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.batch_size = batch_size
        if backend is None:
            # Default to the same obstore-backed router used by COGReader so
            # header enrichment supports s3://, gs://, and Azure Blob URLs
            # (not just HTTPS).
            from rasteret.fetch.cog import _create_obstore_backend

            self._backend = _create_obstore_backend()
        else:
            self._backend = backend

        # Rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Caching
        self.header_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.dtype_map = {
            (1, 8): "uint8",
            (1, 16): "uint16",
            (1, 32): "uint32",
            (1, 64): "uint64",
            (2, 8): "int8",
            (2, 16): "int16",
            (2, 32): "int32",
            (2, 64): "int64",
            (3, 16): "float16",
            (3, 32): "float32",
            (3, 64): "float64",
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def process_cog_headers_batch(
        self,
        urls: list[str],
    ) -> list[CogMetadata | None]:
        """Process multiple URLs in parallel with smart batching."""

        results = []
        total = len(urls)

        num_batches = (total + self.batch_size - 1) // self.batch_size
        if num_batches == 1:
            logger.info(f"Processing {total} COG headers (single batch)")
        else:
            logger.info(
                f"Processing {total} COG headers"
                f" in {num_batches} batches of {self.batch_size}"
            )

        for i in range(0, total, self.batch_size):
            batch = urls[i : min(i + self.batch_size, total)]
            batch_start = time.time()

            tasks = [self.parse_cog_header(url) for url in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            batch_time = time.time() - batch_start
            batch_num = i // self.batch_size + 1
            remaining = total - (i + len(batch))
            status = "Done" if remaining == 0 else f"{remaining} remaining"
            logger.info(
                f"Batch {batch_num}/{num_batches} "
                f"({len(batch)} headers) {batch_time:.2f}s: {status}"
            )

        return results

    async def _fetch_byte_range(self, url: str, start: int, size: int) -> bytes:
        """Fetch a byte range from a URL via the configured backend."""

        cache_key = f"{url}:{start}:{size}"

        if cache_key in self.header_cache:
            return self.header_cache[cache_key]

        for attempt in range(self.retry_attempts):
            try:
                async with self.semaphore:
                    data = await self._backend.get_range(url, start=start, length=size)
                    if len(data) != size:
                        raise IOError(
                            "Truncated header range response for "
                            f"{url}: requested bytes={start}..{start + size}, "
                            f"expected {size} bytes, got {len(data)}."
                        )
                    self.header_cache[cache_key] = data
                    return data

            except (OSError, IOError, Exception) as e:
                if attempt == self.retry_attempts - 1:
                    raise IOError(f"Failed to fetch bytes from {url}: {e}")
                await asyncio.sleep(1 * (attempt + 1))

    async def parse_cog_header(self, url: str) -> CogMetadata | None:
        """Parse COG header from URL."""
        try:
            # Read initial header bytes
            header_bytes = await self._fetch_byte_range(url, 0, 16)

            # Check byte order
            big_endian = header_bytes[0:2] == b"MM"
            endian = ">" if big_endian else "<"

            # Parse version and IFD offset
            version = struct.unpack(f"{endian}H", header_bytes[2:4])[0]
            if version == 42:
                ifd_offset = struct.unpack(f"{endian}L", header_bytes[4:8])[0]
                entry_size = 12
                offset_size = 4
            elif version == 43:
                ifd_offset = struct.unpack(f"{endian}Q", header_bytes[8:16])[0]
                entry_size = 20
                offset_size = 8
            else:
                raise ValueError(f"Unsupported TIFF version: {version}")

            # Read IFD entries
            ifd_count_size = 2 if version == 42 else 8
            ifd_count_bytes = await self._fetch_byte_range(
                url, ifd_offset, ifd_count_size
            )
            entry_count = (
                struct.unpack(f"{endian}H", ifd_count_bytes)[0]
                if version == 42
                else struct.unpack(f"{endian}Q", ifd_count_bytes)[0]
            )

            ifd_bytes = await self._fetch_byte_range(
                url, ifd_offset + ifd_count_size, entry_count * entry_size
            )

            # Parse tags
            tags = {}
            for i in range(entry_count):
                entry = ifd_bytes[i * entry_size : (i + 1) * entry_size]
                tag = struct.unpack(f"{endian}H", entry[0:2])[0]
                type_id = struct.unpack(f"{endian}H", entry[2:4])[0]
                if version == 42:
                    count = struct.unpack(f"{endian}L", entry[4:8])[0]
                    value_or_offset = entry[8:12]
                else:
                    count = struct.unpack(f"{endian}Q", entry[4:12])[0]
                    value_or_offset = entry[12:20]

                tags[tag] = await self._parse_tiff_tag_value(
                    url,
                    tag,
                    type_id,
                    int(count),
                    value_or_offset,
                    endian,
                    offset_size=offset_size,
                )

            # Extract essential metadata
            image_width = tags.get(TAG_IMAGE_WIDTH)[0]
            image_height = tags.get(TAG_IMAGE_LENGTH)[0]
            tile_width = tags.get(TAG_TILE_WIDTH, [image_width])[0]
            tile_height = tags.get(TAG_TILE_LENGTH, [image_height])[0]

            compression = tags.get(TAG_COMPRESSION, (1,))[0]
            predictor = tags.get(TAG_PREDICTOR, (1,))[0]

            # Data type
            sample_format = tags.get(TAG_SAMPLE_FORMAT, (1,))[0]
            bits_per_sample = tags.get(TAG_BITS_PER_SAMPLE, (8,))[0]
            dtype_key = (sample_format, bits_per_sample)
            dtype = self.dtype_map.get(dtype_key)
            if dtype is None:
                raise NotImplementedError(
                    "Unsupported TIFF dtype: "
                    f"SampleFormat={sample_format}, BitsPerSample={bits_per_sample}"
                )

            # Band/sample layout
            samples_per_pixel = tags.get(TAG_SAMPLES_PER_PIXEL, (1,))[0]
            planar_configuration = tags.get(TAG_PLANAR_CONFIGURATION, (1,))[0]
            photometric = tags.get(TAG_PHOTOMETRIC, (None,))[0]
            extra_samples = tags.get(TAG_EXTRA_SAMPLES)

            # GDAL nodata (ASCII string like "-128" or "nan")
            nodata = None
            raw_nodata = tags.get(TAG_GDAL_NODATA)
            if raw_nodata is not None:
                nodata_str = (
                    raw_nodata[0] if isinstance(raw_nodata, tuple) else raw_nodata
                )
                if isinstance(nodata_str, (bytes, bytearray)):
                    try:
                        nodata_str = nodata_str.decode("ascii", errors="ignore")
                    except (UnicodeDecodeError, AttributeError):
                        nodata_str = ""
                if isinstance(nodata_str, str) and nodata_str:
                    nodata = _parse_nodata(nodata_str)
            else:
                # Some GeoTIFFs store nodata in GDALMetadata XML
                # instead of the dedicated GDAL_NODATA tag.
                raw_xml = tags.get(TAG_GDAL_METADATA)
                xml_str = raw_xml[0] if isinstance(raw_xml, tuple) else raw_xml
                if isinstance(xml_str, (bytes, bytearray)):
                    try:
                        xml_str = xml_str.decode("utf-8", errors="ignore")
                    except (UnicodeDecodeError, AttributeError):
                        xml_str = ""
                if isinstance(xml_str, str) and xml_str.strip():
                    try:
                        import xml.etree.ElementTree as ET

                        root = ET.fromstring(xml_str)
                        for item in root.findall(".//Item"):
                            name = item.attrib.get("name", "")
                            if name in {"NODATA_VALUES", "NODATA_VALUE"}:
                                text = (item.text or "").strip()
                                if text:
                                    # NODATA_VALUES may contain multiple values; take the first.
                                    nodata = _parse_nodata(text.split()[0])
                                    break
                    except ET.ParseError:
                        # Best-effort: ignore malformed GDALMetadata XML.
                        logger.debug(
                            "Malformed GDALMetadata XML, skipping nodata extraction"
                        )

            # Tile layout (only present in tiled TIFFs/COGs)
            tile_offsets = list(tags.get(TAG_TILE_OFFSETS, []))
            tile_byte_counts = list(tags.get(TAG_TILE_BYTE_COUNTS, []))

            if not tile_offsets or not tile_byte_counts:
                raise NotImplementedError(
                    "Rasteret requires a tiled GeoTIFF/COG "
                    "(no TileOffsets/TileByteCounts)."
                )
            if compression == COMPRESSION_JPEG:
                raise NotImplementedError(
                    "TIFF JPEG compression is not supported yet. "
                    "Some TIFFs also use shared JPEGTables (tag 347), which requires "
                    "concatenating the tables with each tile stream during decode."
                )

            # Geotransform
            pixel_scale = tags.get(TAG_MODEL_PIXEL_SCALE)
            tiepoint = tags.get(TAG_MODEL_TIEPOINT)
            model_transform = tags.get(TAG_MODEL_TRANSFORM)

            # Calculate transform
            transform = None
            if model_transform:
                # GeoTIFF ModelTransformationTag is a 4x4 matrix mapping raster
                # coordinates (col, row, z, 1) to model space (x, y, z, 1).
                #
                # Rasteret supports axis-aligned transforms only (north-up or
                # south-up, no rotation/shear). For those cases:
                #   x = m00 * col + m03
                #   y = m11 * row + m13
                try:
                    values = list(model_transform)
                    if len(values) != 16:
                        raise ValueError(
                            f"ModelTransformationTag must have 16 values, got {len(values)}"
                        )
                    m00, m01, m02, m03 = values[0:4]
                    m10, m11, m12, m13 = values[4:8]
                    m20, m21, m22, m23 = values[8:12]
                    m30, m31, m32, m33 = values[12:16]

                    rotated = any(
                        abs(float(v)) > 1e-12
                        for v in (m01, m02, m10, m12, m20, m21, m23, m30, m31, m32)
                    )
                    if rotated or abs(float(m33) - 1.0) > 1e-12:
                        raise ValueError("rotated/sheared ModelTransformationTag")

                    transform = (
                        float(m00),
                        float(m03),
                        float(m11),
                        float(m13),
                    )
                except (TypeError, ValueError) as exc:
                    raise NotImplementedError(
                        f"Unsupported ModelTransformationTag (rotation/shear): {exc}"
                    ) from exc
            elif pixel_scale and tiepoint:
                scale_x = float(pixel_scale[0])
                scale_y = -float(pixel_scale[1])
                # ModelTiepointTag values are (I, J, K, X, Y, Z) tuples. The
                # tiepoint may refer to a raster point (I, J) that is not the
                # origin. Convert to the PixelIsArea-style affine where
                # pixel (0, 0) maps to (translate_x, translate_y).
                try:
                    n_tp = len(tiepoint)
                except TypeError as exc:
                    raise ValueError(
                        "Invalid ModelTiepointTag: expected a sequence of values"
                    ) from exc
                if n_tp < 6 or n_tp % 6 != 0:
                    raise ValueError(
                        "Invalid ModelTiepointTag: expected 6*N values "
                        f"(I, J, K, X, Y, Z), got {n_tp}"
                    )

                translate_x = None
                translate_y = None
                for idx in range(0, n_tp, 6):
                    i0 = float(tiepoint[idx + 0])
                    j0 = float(tiepoint[idx + 1])
                    x0 = float(tiepoint[idx + 3])
                    y0 = float(tiepoint[idx + 4])
                    tx = x0 - i0 * scale_x
                    ty = y0 - j0 * scale_y
                    if translate_x is None:
                        translate_x, translate_y = tx, ty
                    else:
                        assert translate_y is not None
                        # Multiple tiepoints must imply the same origin for an
                        # axis-aligned PixelScale+Tiepoint transform.
                        if abs(tx - translate_x) > 1e-6 or abs(ty - translate_y) > 1e-6:
                            raise NotImplementedError(
                                "GeoTIFF ModelTiepointTag contains multiple tiepoints "
                                "that do not imply a single consistent origin. "
                                "Rasteret supports axis-aligned transforms only."
                            )
                assert translate_x is not None and translate_y is not None
                transform = (scale_x, translate_x, scale_y, translate_y)
            else:
                raise NotImplementedError(
                    "Missing GeoTIFF georeferencing tags (no ModelTransformationTag "
                    "and no ModelPixelScaleTag+ModelTiepointTag)."
                )

            # --- PixelIsPoint correction (GDAL RFC 33) ---
            # When GTRasterTypeGeoKey == 2, the tiepoint references the pixel
            # centre, not the upper-left corner.  Shift the origin by half a
            # pixel so that CogMetadata.transform is always PixelIsArea-based.
            if transform is not None:
                raster_type = get_raster_type_from_geokeys(tags)
                if raster_type == 2:  # RasterPixelIsPoint
                    scale_x, translate_x, scale_y, translate_y = transform
                    translate_x -= scale_x / 2
                    translate_y -= scale_y / 2
                    transform = (scale_x, translate_x, scale_y, translate_y)
                    logger.debug("Applied PixelIsPoint correction for %s", url)

            crs = get_crs_from_tiff_tags(tags)

            return CogMetadata(
                width=image_width,
                height=image_height,
                tile_width=tile_width,
                tile_height=tile_height,
                dtype=dtype,
                transform=transform,
                predictor=predictor,
                compression=compression,
                tile_offsets=tile_offsets,
                tile_byte_counts=tile_byte_counts,
                crs=crs,
                pixel_scale=pixel_scale,
                tiepoint=tiepoint,
                nodata=nodata,
                samples_per_pixel=samples_per_pixel,
                planar_configuration=planar_configuration,
                photometric=photometric,
                extra_samples=extra_samples,
            )

        except NotImplementedError:
            raise
        except (
            struct.error,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            IOError,
        ) as e:
            logger.exception("Failed to parse header for %s: %s", url, e)
            raise

    async def _parse_tiff_tag_value(
        self,
        url: str,
        tag: int,
        type_id: int,
        count: int,
        value_or_offset: bytes,
        endian: str,
        *,
        offset_size: int,
    ) -> tuple:
        """Parse a TIFF tag value based on its type."""
        type_sizes: dict[int, int] = {
            1: 1,  # BYTE
            2: 1,  # ASCII
            3: 2,  # SHORT
            4: 4,  # LONG
            5: 8,  # RATIONAL (2x LONG)
            6: 1,  # SBYTE
            7: 1,  # UNDEFINED
            8: 2,  # SSHORT
            9: 4,  # SLONG
            10: 8,  # SRATIONAL (2x SLONG)
            11: 4,  # FLOAT
            12: 8,  # DOUBLE
            16: 8,  # LONG8 (BigTIFF)
            17: 8,  # SLONG8 (BigTIFF)
        }

        if type_id not in type_sizes:
            raise ValueError(f"Unsupported TIFF tag type: {type_id} for tag {tag}")

        total_size = type_sizes[type_id] * count

        if total_size <= offset_size:
            data = value_or_offset[:offset_size]
        else:
            offset_fmt = "L" if offset_size == 4 else "Q"
            offset = struct.unpack(
                f"{endian}{offset_fmt}", value_or_offset[:offset_size]
            )[0]
            data = await self._fetch_byte_range(url, int(offset), total_size)

        if type_id == 1:  # BYTE
            return struct.unpack(f"{endian}{count}B", data[:total_size])
        if type_id == 2:  # ASCII
            raw = data[:total_size]
            return (raw[: max(0, count - 1)].decode("ascii", errors="replace"),)
        if type_id == 3:  # SHORT
            return struct.unpack(f"{endian}{count}H", data[:total_size])
        if type_id == 4:  # LONG
            return struct.unpack(f"{endian}{count}L", data[:total_size])
        if type_id == 5:  # RATIONAL
            vals = struct.unpack(f"{endian}{count*2}L", data[:total_size])
            return tuple(vals[i] / vals[i + 1] for i in range(0, len(vals), 2))
        if type_id == 6:  # SBYTE
            return struct.unpack(f"{endian}{count}b", data[:total_size])
        if type_id == 8:  # SSHORT
            return struct.unpack(f"{endian}{count}h", data[:total_size])
        if type_id == 9:  # SLONG
            return struct.unpack(f"{endian}{count}l", data[:total_size])
        if type_id == 10:  # SRATIONAL
            vals = struct.unpack(f"{endian}{count*2}l", data[:total_size])
            return tuple(vals[i] / vals[i + 1] for i in range(0, len(vals), 2))
        if type_id == 11:  # FLOAT
            return struct.unpack(f"{endian}{count}f", data[:total_size])
        if type_id == 12:  # DOUBLE
            return struct.unpack(f"{endian}{count}d", data[:total_size])
        if type_id == 16:  # LONG8
            return struct.unpack(f"{endian}{count}Q", data[:total_size])
        if type_id == 17:  # SLONG8
            return struct.unpack(f"{endian}{count}q", data[:total_size])

        raise ValueError(f"Unsupported TIFF tag type: {type_id} for tag {tag}")
