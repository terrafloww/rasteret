"""Optimized COG reading using byte ranges."""

from __future__ import annotations
import asyncio
import httpx
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import imagecodecs
import numpy as np
from affine import Affine
from shapely.geometry import Polygon, box
from rasterio.mask import geometry_mask

from rasteret.types import CogMetadata
from rasteret.core.utils import wgs84_to_utm_convert_poly
from rasteret.logging import setup_logger

logger = setup_logger("INFO", customname="rasteret.fetch.cog")


@dataclass
class COGTileRequest:
    """Single tile request details."""

    url: str
    offset: int  # Byte offset in COG file
    size: int  # Size in bytes to read
    row: int  # Tile row in the grid
    col: int  # Tile column in the grid
    metadata: CogMetadata  # Full metadata including transform


def compute_tile_indices(
    geometry: Polygon,
    transform: List[float],
    tile_size: Tuple[int, int],
    image_size: Tuple[int, int],
    debug: bool = False,
) -> List[Tuple[int, int]]:
    """
    Compute tile indices that intersect with geometry.
    Using simplified direct mapping approach from tiles.py.
    """
    # Extract parameters
    scale_x, translate_x, scale_y, translate_y = transform
    tile_width, tile_height = tile_size
    image_width, image_height = image_size

    # Calculate number of tiles
    tiles_x = (image_width + tile_width - 1) // tile_width
    tiles_y = (image_height + tile_height - 1) // tile_height

    # Get geometry bounds
    minx, miny, maxx, maxy = geometry.bounds

    if debug:
        logger.info(
            f"""
        Computing tile indices:
        - Bounds: {minx}, {miny}, {maxx}, {maxy}
        - Transform: {scale_x}, {translate_x}, {scale_y}, {translate_y} 
        - Image size: {image_width}x{image_height}
        - Tile size: {tile_width}x{tile_height}
        """
        )

    # Convert to pixel coordinates, handling negative scales
    col_min = max(0, int((minx - translate_x) / abs(scale_x)))
    col_max = min(image_width - 1, int((maxx - translate_x) / abs(scale_x)))

    # Handle Y coordinate inversion in raster space
    row_min = max(0, int((translate_y - maxy) / abs(scale_y)))
    row_max = min(image_height - 1, int((translate_y - miny) / abs(scale_y)))

    if debug:
        logger.info(f"Pixel bounds: x({col_min}-{col_max}), y({row_min}-{row_max})")

    # Convert to tile indices
    tile_col_min = max(0, col_min // tile_width)
    tile_col_max = min(tiles_x - 1, col_max // tile_width)
    tile_row_min = max(0, row_min // tile_height)
    tile_row_max = min(tiles_y - 1, row_max // tile_height)

    if debug:
        logger.info(
            f"Tile indices: x({tile_col_min}-{tile_col_max}), y({tile_row_min}-{tile_row_max})"
        )

    # Validate tile ranges
    if tile_col_min > tile_col_max or tile_row_min > tile_row_max:
        if debug:
            logger.info("No valid tiles in range")
        return []

    # Find intersecting tiles
    intersecting_tiles = []
    for row in range(tile_row_min, tile_row_max + 1):
        for col in range(tile_col_min, tile_col_max + 1):
            # Calculate tile bounds in UTM coordinates
            tile_minx = translate_x + col * tile_width * scale_x
            tile_maxx = tile_minx + tile_width * scale_x
            tile_maxy = translate_y - row * tile_height * abs(scale_y)
            tile_miny = tile_maxy - tile_height * abs(scale_y)

            # Create tile box and check intersection
            tile_box = box(
                min(tile_minx, tile_maxx),
                min(tile_miny, tile_maxy),
                max(tile_minx, tile_maxx),
                max(tile_miny, tile_maxy),
            )

            if geometry.intersects(tile_box):
                intersecting_tiles.append((row, col))
                if debug:
                    logger.info(f"Added intersecting tile: ({row}, {col})")

    if debug:
        logger.info(f"Found {len(intersecting_tiles)} intersecting tiles")

    return intersecting_tiles


def merge_tiles(
    tiles: Dict[Tuple[int, int], np.ndarray],
    tile_size: Tuple[int, int],
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Merge multiple tiles into a single array.
    Returns merged array and bounds (min_row, min_col, max_row, max_col).
    """
    if not tiles:
        return np.array([], dtype=dtype), (0, 0, 0, 0)

    # Find bounds
    rows, cols = zip(*tiles.keys())
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    tile_width, tile_height = tile_size

    # Create output array
    height = (max_row - min_row + 1) * tile_height
    width = (max_col - min_col + 1) * tile_width
    merged = np.full((height, width), np.nan, dtype=dtype)

    # Place tiles with exact positioning
    for (row, col), data in tiles.items():
        if data is not None:  # Handle potentially failed tiles
            y_start = (row - min_row) * tile_height
            x_start = (col - min_col) * tile_width
            y_end = min(y_start + data.shape[0], height)
            x_end = min(x_start + data.shape[1], width)
            merged[y_start:y_end, x_start:x_end] = data[
                : y_end - y_start, : x_end - x_start
            ]

    return merged, (min_row, min_col, max_row, max_col)


def apply_mask_and_crop(
    data: np.ndarray,
    geometry: Polygon,
    transform: Affine,
) -> Tuple[np.ndarray, Affine]:
    """Apply geometry mask and crop to valid data region."""

    mask = geometry_mask(
        [geometry],
        out_shape=data.shape,
        transform=transform,
        all_touched=True,
        invert=True,
    )

    # Find valid data bounds
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Crop data and mask
    data_cropped = data[row_min : row_max + 1, col_min : col_max + 1]
    mask_cropped = mask[row_min : row_max + 1, col_min : col_max + 1]

    # Apply mask to cropped data
    masked_data = np.where(mask_cropped, data_cropped, np.nan)

    # Update transform for cropped array
    cropped_transform = Affine(
        transform.a,
        transform.b,
        transform.c + col_min * transform.a,
        transform.d,
        transform.e,
        transform.f + row_min * transform.e,
    )

    return masked_data, cropped_transform


async def read_tile(
    request: COGTileRequest,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[np.ndarray]:
    """Read a single tile using byte range request."""
    for attempt in range(retries):
        try:
            async with sem:
                headers = {
                    "Range": f"bytes={request.offset}-{request.offset+request.size-1}"
                }
                response = await client.get(request.url, headers=headers)
                if response.status_code != 206:
                    raise ValueError(f"Range request failed: {response.status_code}")

                # Simple, direct data flow like tiles.py
                decompressed = imagecodecs.zlib_decode(response.content)
                data = np.frombuffer(decompressed, dtype=np.uint16)
                data = data.reshape(
                    (request.metadata.tile_height, request.metadata.tile_width)
                )

                # Predictor handling exactly like tiles.py
                if request.metadata.predictor == 2:
                    data = data.astype(np.uint16)
                    for i in range(data.shape[0]):
                        data[i] = np.cumsum(data[i], dtype=np.uint16)

                return data.astype(np.float32)

        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to read tile: {str(e)}")
                return None
            await asyncio.sleep(retry_delay * (2**attempt))


async def read_cog_tile_data(
    url: str,
    metadata: CogMetadata,
    geometry: Optional[Polygon] = None,
    max_concurrent: int = 50,
    debug: bool = False,
) -> Tuple[np.ndarray, Optional[Affine]]:
    """Read COG data, optionally masked by geometry.

    Args:
        url: URL of the COG file
        metadata: COG metadata including transform
        geometry: Optional polygon to mask/filter data
        max_concurrent: Maximum concurrent requests
        debug: Enable debug logging

    Returns:
        Tuple of:
        - np.ndarray: The masked data array
        - Affine: Transform matrix for the masked data
          None if no transform available
    """
    if debug:
        logger.info(
            f"""
        Input Parameters:
        - CRS: {metadata.crs}
        - Transform: {metadata.transform}
        - Image Size: {metadata.width}x{metadata.height}
        - Tile Size: {metadata.tile_width}x{metadata.tile_height}
        - Geometry: {geometry.wkt if geometry else None}
        """
        )

    if metadata.transform is None:
        return np.array([]), None

    # Convert geometry to image CRS if needed
    if geometry:
        if metadata.crs != 4326:
            geometry = wgs84_to_utm_convert_poly(geom=geometry, epsg_code=metadata.crs)
            if debug:
                logger.info(f"Transformed geometry bounds: {geometry.bounds}")

        # Get tiles that intersect with geometry
        intersecting_tiles = compute_tile_indices(
            geometry=geometry,
            transform=metadata.transform,
            tile_size=(metadata.tile_width, metadata.tile_height),
            image_size=(metadata.width, metadata.height),
            debug=debug,
        )
    else:
        # Read all tiles if no geometry provided
        tiles_x = (metadata.width + metadata.tile_width - 1) // metadata.tile_width
        tiles_y = (metadata.height + metadata.tile_height - 1) // metadata.tile_height
        intersecting_tiles = [(r, c) for r in range(tiles_y) for c in range(tiles_x)]

    if not intersecting_tiles:
        return np.array([]), None

    # Set up HTTP client with connection pooling
    limits = httpx.Limits(
        max_keepalive_connections=max_concurrent, max_connections=max_concurrent
    )
    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
        sem = asyncio.Semaphore(max_concurrent)

        # Read tiles
        tiles = {}
        tasks = []
        tiles_x = (metadata.width + metadata.tile_width - 1) // metadata.tile_width

        # Create tasks for all tiles
        for row, col in intersecting_tiles:
            tile_idx = row * tiles_x + col  # Linear tile index

            if tile_idx >= len(metadata.tile_offsets):
                if debug:
                    logger.warning(f"Tile index {tile_idx} out of bounds")
                continue

            # Create tile request
            request = COGTileRequest(
                url=url,
                offset=metadata.tile_offsets[tile_idx],
                size=metadata.tile_byte_counts[tile_idx],
                row=row,
                col=col,
                metadata=metadata,
            )

            tasks.append((row, col, read_tile(request, client, sem)))

        # Gather results
        for row, col, task in tasks:
            try:
                tile_data = await task
                if tile_data is not None:
                    tiles[(row, col)] = tile_data
            except Exception as e:
                logger.error(f"Failed to read tile at ({row}, {col}): {str(e)}")

        if not tiles:
            return np.array([]), None

    # Merge tiles
    merged_data, bounds = merge_tiles(
        tiles, (metadata.tile_width, metadata.tile_height), dtype=np.float32
    )

    if debug:
        logger.info(
            f"""
        Merged Data:
        - Shape: {merged_data.shape}
        - Bounds: {bounds}
        - Data Range: {np.nanmin(merged_data)}-{np.nanmax(merged_data)}
        """
        )

    # Calculate transform for merged data
    min_row, min_col, max_row, max_col = bounds
    scale_x, translate_x, scale_y, translate_y = metadata.transform

    merged_transform = Affine(
        scale_x,
        0,
        translate_x + min_col * metadata.tile_width * scale_x,
        0,
        scale_y,
        translate_y + min_row * metadata.tile_height * scale_y,
    )

    # Apply geometry mask if provided
    if geometry is not None:
        merged_data, cropped_transform = apply_mask_and_crop(
            merged_data, geometry, merged_transform
        )

    if debug:
        logger.info(
            f"""
        Final Output:
        - Shape: {merged_data.shape}
        - Transform: {merged_transform}
        - Data Range: {np.nanmin(merged_data)}-{np.nanmax(merged_data)}
        """
        )

    return merged_data, cropped_transform
