"""Async COG header parsing with caching."""

from __future__ import annotations
import asyncio
import struct
import time
from typing import Dict, List, Optional, Set, Any

import httpx
from cachetools import TTLCache, LRUCache

from rasteret.types import CogMetadata
from rasteret.cloud import CloudProvider, CloudConfig
from rasteret.logging import setup_logger

logger = setup_logger()


def get_crs_from_tiff_tags(tags: Dict[int, Any]) -> Optional[str]:
    """
    Extract CRS from GeoTIFF tags using multiple methods.

    Args:
        tags: Dictionary of TIFF tags

    Returns:
        Optional[str]: EPSG code string like "EPSG:32643" if found, None otherwise
    """
    # Method 1: GeoTiff WKT string (tag 34737 - GeoAsciiParamsTag)
    if 34737 in tags:
        wkt = tags[34737]
        try:
            import re

            # Look for EPSG code in WKT string
            epsg_match = re.search(r'ID\["EPSG",(\d+)\]', wkt)
            if epsg_match:
                return int(epsg_match.group(1))
        except Exception as e:
            logger.debug(f"Failed to parse WKT string: {e}")

    # Method 2: GeoKey directory (tag 34735)
    geokeys = tags.get(34735)
    if geokeys:
        try:
            num_keys = geokeys[3]
            for i in range(4, 4 + (4 * num_keys), 4):
                key_id = geokeys[i]
                tiff_tag_loc = geokeys[i + 1]
                count = geokeys[i + 2]
                value = geokeys[i + 3]

                if key_id in (3072, 2048):  # ProjectedCRS or GeographicCRS
                    if tiff_tag_loc == 0 and count == 1:  # Direct value
                        return int(value)
        except Exception as e:
            logger.debug(f"Failed to parse GeoKey directory: {e}")

    return None


class AsyncCOGHeaderParser:
    """Optimized async parser for COG headers with connection pooling and caching."""

    def __init__(
        self,
        max_concurrent: int = 50,
        cache_ttl: int = 3600,  # 1 hour
        retry_attempts: int = 3,
        cloud_provider: Optional[CloudProvider] = None,
        cloud_config: Optional[CloudConfig] = None,
    ):
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.cloud_provider = cloud_provider
        self.cloud_config = cloud_config

        # Connection optimization
        self.connector = httpx.Limits(
            max_keepalive_connections=max_concurrent,
            max_connections=max_concurrent,
            keepalive_expiry=120,
        )

        # Rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests: Set[str] = set()

        # Caching
        self.header_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.dns_cache = LRUCache(maxsize=500)

        self.client = None
        self.dtype_map = {
            (1, 8): "uint8",
            (1, 16): "uint16",
            (2, 8): "int8",
            (2, 16): "int16",
            (3, 32): "float32",
        }

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            limits=self.connector,
            timeout=30.0,
            http2=True,
            headers={"Connection": "keep-alive", "Keep-Alive": "timeout=120"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def process_cog_headers_batch(
        self, urls: List[str], batch_size: int = 10
    ) -> List[Optional[CogMetadata]]:
        """Process multiple URLs in parallel with smart batching."""

        results = []
        total = len(urls)

        logger.info(
            f"Processing {total} COG headers {'(single batch)' if total <= batch_size else f'in {(total + batch_size - 1) // batch_size} batches of {batch_size}'}"
        )

        for i in range(0, total, batch_size):
            batch = urls[i : min(i + batch_size, total)]
            batch_start = time.time()

            tasks = [self.parse_cog_header(url) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to process header: {result}")
                    results.append(None)
                else:
                    results.append(result)

            batch_time = time.time() - batch_start
            remaining = total - (i + len(batch))
            batch_msg = (
                f"Processed batch {i//batch_size + 1}/{(total + batch_size - 1) // batch_size} "
                f"({len(batch)} {'header' if len(batch) == 1 else 'headers'}) "
                f"in {batch_time:.2f}s. "
                f"{'Completed!' if remaining == 0 else f'Remaining: {remaining}'}"
            )
            logger.info(batch_msg)

        return results

    async def _fetch_byte_range(self, url: str, start: int, size: int) -> bytes:
        """Fetch a byte range from a URL."""

        cache_key = f"{url}:{start}:{size}"

        if cache_key in self.header_cache:
            return self.header_cache[cache_key]

        while url in self.active_requests:
            await asyncio.sleep(0.1)

        self.active_requests.add(url)
        try:
            headers = {"Range": f"bytes={start}-{start + size - 1}"}

            for attempt in range(self.retry_attempts):
                try:
                    async with self.semaphore:
                        response = await self.client.get(url, headers=headers)
                        if response.status_code != 206:
                            raise IOError(
                                f"Range request failed: {response.status_code}"
                            )

                        data = response.content
                        self.header_cache[cache_key] = data
                        return data

                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise IOError(f"Failed to fetch bytes from {url}: {e}")
                    await asyncio.sleep(1 * (attempt + 1))
        finally:
            self.active_requests.remove(url)

    async def parse_cog_header(self, url: str) -> Optional[CogMetadata]:
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
            elif version == 43:
                ifd_offset = struct.unpack(f"{endian}Q", header_bytes[8:16])[0]
                entry_size = 20
            else:
                raise ValueError(f"Unsupported TIFF version: {version}")

            # Read IFD entries
            ifd_count_size = 2 if version == 42 else 8
            ifd_count_bytes = await self._fetch_byte_range(
                url, ifd_offset, ifd_count_size
            )
            entry_count = struct.unpack(f"{endian}H", ifd_count_bytes)[0]

            ifd_bytes = await self._fetch_byte_range(
                url, ifd_offset + ifd_count_size, entry_count * entry_size
            )

            # Parse tags
            tags = {}
            for i in range(entry_count):
                entry = ifd_bytes[i * entry_size : (i + 1) * entry_size]
                tag = struct.unpack(f"{endian}H", entry[0:2])[0]
                type_id = struct.unpack(f"{endian}H", entry[2:4])[0]
                count = struct.unpack(f"{endian}L", entry[4:8])[0]
                value_or_offset = entry[8:12] if version == 42 else entry[16:24]

                tags[tag] = await self._parse_tiff_tag_value(
                    url, tag, type_id, count, value_or_offset, endian
                )

            # Extract essential metadata
            image_width = tags.get(256)[0]  # ImageWidth
            image_height = tags.get(257)[0]  # ImageLength
            tile_width = tags.get(322, [image_width])[0]  # TileWidth
            tile_height = tags.get(323, [image_height])[0]  # TileLength

            compression = tags.get(259, (1,))[0]  # Compression
            predictor = tags.get(317, (1,))[0]  # Predictor

            # Data type
            sample_format = tags.get(339, (1,))[0]
            bits_per_sample = tags.get(258, (8,))[0]
            dtype = self.dtype_map.get((sample_format, bits_per_sample), "uint8")

            # Tile layout
            tile_offsets = list(tags.get(324, []))  # TileOffsets
            tile_byte_counts = list(tags.get(325, []))  # TileByteCounts

            # Geotransform
            pixel_scale = tags.get(33550)  # ModelPixelScaleTag
            tiepoint = tags.get(33922)  # ModelTiepointTag

            # Calculate transform
            transform = None
            if pixel_scale and tiepoint:
                scale_x, scale_y = pixel_scale[0], -pixel_scale[1]
                translate_x, translate_y = tiepoint[3], tiepoint[4]
                transform = (scale_x, translate_x, scale_y, translate_y)

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
            )

        except Exception as e:
            logger.error(f"Failed to parse header for {url}: {str(e)}")
            return None

    async def _parse_tiff_tag_value(
        self,
        url: str,
        tag: int,
        type_id: int,
        count: int,
        value_or_offset: bytes,
        endian: str,
    ) -> tuple:
        """Parse a TIFF tag value based on its type."""
        # Handle single values
        if count == 1:
            if type_id == 3:  # SHORT
                return (struct.unpack(f"{endian}H", value_or_offset[:2])[0],)
            elif type_id == 4:  # LONG
                return (struct.unpack(f"{endian}L", value_or_offset[:4])[0],)
            elif type_id == 5:  # RATIONAL
                offset = struct.unpack(f"{endian}L", value_or_offset[:4])[0]
                data = await self._fetch_byte_range(url, offset, 8)
                nums = struct.unpack(f"{endian}LL", data)
                return (float(nums[0]) / nums[1],)

        # Handle offset values
        offset = struct.unpack(f"{endian}L", value_or_offset[:4])[0]
        size = {
            1: 1,  # BYTE
            2: 1,  # ASCII
            3: 2,  # SHORT
            4: 4,  # LONG
            5: 8,  # RATIONAL
            12: 8,  # DOUBLE
        }[type_id] * count

        data = await self._fetch_byte_range(url, offset, size)

        if type_id == 1:  # BYTE
            return struct.unpack(f"{endian}{count}B", data)
        elif type_id == 2:  # ASCII
            return (data[: count - 1].decode("ascii"),)
        elif type_id == 3:  # SHORT
            return struct.unpack(f"{endian}{count}H", data)
        elif type_id == 4:  # LONG
            return struct.unpack(f"{endian}{count}L", data)
        elif type_id == 5:  # RATIONAL
            vals = struct.unpack(f"{endian}{count*2}L", data)
            return tuple(vals[i] / vals[i + 1] for i in range(0, len(vals), 2))
        elif type_id == 12:  # DOUBLE
            return struct.unpack(f"{endian}{count}d", data)
