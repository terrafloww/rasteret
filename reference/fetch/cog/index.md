# rasteret.fetch.cog

Async COG tile reader with HTTP connection pooling and byte-range merging.

## cog

### Classes

#### COGTileRequest

```python
COGTileRequest(
    url: str,
    offset: int,
    size: int,
    row: int,
    col: int,
    metadata: CogMetadata,
    band_index: int | None = None,
)
```

Single tile request details.

#### CogReadResult

```python
CogReadResult(
    data: ndarray,
    transform: Affine,
    valid_mask: ndarray,
    fill_value_used: float | int | None,
    mode: Literal["aoi", "window", "full"],
)
```

Result of a COG read operation.

`valid_mask` is True for pixels that are inside the requested AOI/window *and* inside raster coverage. When `filled=True`, pixels where `valid_mask=False` are set to *fill_value_used*.

#### COGReader

```python
COGReader(
    max_concurrent: int = 150, backend: object | None = None
)
```

Manages connection pooling and COG reading operations.

Manages Rasteret's custom async byte-range IO. Uses obstore as the HTTP transport layer for multi-cloud URL routing. Optionally accepts a custom :class:`~rasteret.cloud.StorageBackend` (e.g. a pre-configured `S3Store`).

Parameters:

| Name             | Type             | Description                                                                              | Default |
| ---------------- | ---------------- | ---------------------------------------------------------------------------------------- | ------- |
| `max_concurrent` | `int`            | Maximum number of concurrent HTTP requests / byte-range reads.                           | `150`   |
| `backend`        | `StorageBackend` | Pluggable I/O backend. When None, an obstore HTTPStore backend is created automatically. | `None`  |

Source code in `src/rasteret/fetch/cog.py`

```python
def __init__(
    self,
    max_concurrent: int = 150,
    backend: object | None = None,
):
    self.max_concurrent = max_concurrent
    self._backend = backend
    self.sem = None
    self.batch_size = 20
```

##### Functions

###### merge_ranges

```python
merge_ranges(
    requests: list[COGTileRequest],
    gap_threshold: int = 1024,
) -> list[tuple[int, int]]
```

Merge nearby byte ranges to minimize HTTP requests

Source code in `src/rasteret/fetch/cog.py`

```python
def merge_ranges(
    self, requests: list[COGTileRequest], gap_threshold: int = 1024
) -> list[tuple[int, int]]:
    """Merge nearby byte ranges to minimize HTTP requests"""
    if not requests:
        return []

    ranges = [(r.offset, r.offset + r.size) for r in requests]
    ranges.sort()
    merged = [ranges[0]]

    for curr in ranges[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1] + gap_threshold:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)

    return merged
```

###### read_merged_tiles

```python
read_merged_tiles(
    requests: list[COGTileRequest], debug: bool = False
) -> dict[tuple[int, int, int | None], ndarray]
```

Parallel tile reading with HTTP/2 multiplexing

Source code in `src/rasteret/fetch/cog.py`

```python
async def read_merged_tiles(
    self,
    requests: list[COGTileRequest],
    debug: bool = False,
) -> dict[tuple[int, int, int | None], np.ndarray]:
    """Parallel tile reading with HTTP/2 multiplexing"""
    if not requests:
        return {}

    # Group by URL for HTTP/2 connection reuse
    url_groups = {}
    for req in requests:
        url_groups.setdefault(req.url, []).append(req)

    results = {}
    for url, group_requests in url_groups.items():
        ranges = self.merge_ranges(group_requests)

        # Process ranges in batches
        for i in range(0, len(ranges), self.batch_size):
            batch = ranges[i : i + self.batch_size]
            batch_tasks = [
                self._read_and_process_range(
                    url,
                    start,
                    end,
                    [r for r in group_requests if start <= r.offset < end],
                )
                for start, end in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            for result in batch_results:
                results.update(result)

    return results
```

### Functions

#### compute_tile_indices

```python
compute_tile_indices(
    geometry_bbox: tuple[float, float, float, float],
    transform: list[float],
    tile_size: tuple[int, int],
    image_size: tuple[int, int],
    debug: bool = False,
) -> list[tuple[int, int]]
```

Compute tile indices that intersect with a geometry's bounding box.

Parameters:

| Name            | Type             | Description                                                          | Default    |
| --------------- | ---------------- | -------------------------------------------------------------------- | ---------- |
| `geometry_bbox` | `tuple of float` | (minx, miny, maxx, maxy) bounding box in the same CRS as the raster. | *required* |
| `transform`     | `list of float`  | Affine transform coefficients for the raster.                        | *required* |
| `tile_size`     | `tuple of int`   | (tile_width, tile_height) in pixels.                                 | *required* |
| `image_size`    | `tuple of int`   | (image_width, image_height) in pixels.                               | *required* |
| `debug`         | `bool`           | Enable debug logging.                                                | `False`    |

Returns:

| Type            | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| `list of tuple` | (row, col) indices of tiles that intersect the geometry bbox. |

Source code in `src/rasteret/fetch/cog.py`

```python
def compute_tile_indices(
    geometry_bbox: tuple[float, float, float, float],
    transform: list[float],
    tile_size: tuple[int, int],
    image_size: tuple[int, int],
    debug: bool = False,
) -> list[tuple[int, int]]:
    """Compute tile indices that intersect with a geometry's bounding box.

    Parameters
    ----------
    geometry_bbox : tuple of float
        ``(minx, miny, maxx, maxy)`` bounding box in the same CRS as the raster.
    transform : list of float
        Affine transform coefficients for the raster.
    tile_size : tuple of int
        ``(tile_width, tile_height)`` in pixels.
    image_size : tuple of int
        ``(image_width, image_height)`` in pixels.
    debug : bool
        Enable debug logging.

    Returns
    -------
    list of tuple
        ``(row, col)`` indices of tiles that intersect the geometry bbox.
    """
    # Extract parameters
    scale_x, translate_x, scale_y, translate_y = normalize_transform(transform)
    tile_width, tile_height = tile_size
    image_width, image_height = image_size

    # Calculate number of tiles
    tiles_x = (image_width + tile_width - 1) // tile_width
    tiles_y = (image_height + tile_height - 1) // tile_height

    minx, miny, maxx, maxy = geometry_bbox

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

    if scale_x == 0.0 or scale_y == 0.0:
        raise ValueError("Invalid transform: zero pixel scale")

    # Convert bounds to pixel coordinates using the transform's actual sign.
    # The transform maps pixel coordinates to map coordinates as:
    #   x = translate_x + col * scale_x
    #   y = translate_y + row * scale_y
    #
    # This supports both north-up (scale_y < 0) and south-up/bottom-up rasters
    # (scale_y > 0) without special-casing.
    col0 = (minx - translate_x) / scale_x
    col1 = (maxx - translate_x) / scale_x
    row0 = (miny - translate_y) / scale_y
    row1 = (maxy - translate_y) / scale_y

    col_min = int(math.floor(min(col0, col1)))
    col_max = int(math.ceil(max(col0, col1))) - 1
    row_min = int(math.floor(min(row0, row1)))
    row_max = int(math.ceil(max(row0, row1))) - 1

    # Clamp to the image extent.
    col_min = max(0, col_min)
    row_min = max(0, row_min)
    col_max = min(image_width - 1, col_max)
    row_max = min(image_height - 1, row_max)

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

    # Find intersecting tiles using bbox arithmetic (no Shapely)
    intersecting_tiles = []
    for row in range(tile_row_min, tile_row_max + 1):
        for col in range(tile_col_min, tile_col_max + 1):
            # Calculate tile bounds in map coordinates
            x0 = translate_x + col * tile_width * scale_x
            x1 = translate_x + (col + 1) * tile_width * scale_x
            y0 = translate_y + row * tile_height * scale_y
            y1 = translate_y + (row + 1) * tile_height * scale_y

            tile_bbox = (
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1),
            )

            if bbox_intersects(geometry_bbox, tile_bbox):
                intersecting_tiles.append((row, col))
                if debug:
                    logger.info(f"Added intersecting tile: ({row}, {col})")

    if debug:
        logger.info(f"Found {len(intersecting_tiles)} intersecting tiles")

    return intersecting_tiles
```

#### merge_tiles

```python
merge_tiles(
    tiles: dict[tuple[int, int], ndarray],
    tile_size: tuple[int, int],
    dtype: dtype,
    *,
    fill_value: float | int = 0,
) -> tuple[ndarray, tuple[int, int, int, int], ndarray]
```

Merge multiple tiles into a single array.

Parameters:

| Name         | Type           | Description                                           | Default    |
| ------------ | -------------- | ----------------------------------------------------- | ---------- |
| `tiles`      | `dict`         | Mapping of (row, col) to tile np.ndarray.             | *required* |
| `tile_size`  | `tuple of int` | (tile_width, tile_height) in pixels.                  | *required* |
| `dtype`      | `dtype`        | Desired output dtype.                                 | *required* |
| `fill_value` | `float or int` | Fill value for empty regions (outside tile coverage). | `0`        |

Returns:

| Type    | Description                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| `tuple` | (merged_array, (min_row, min_col, max_row, max_col), tile_mask). tile_mask is True where a tile contributed pixels. |

Source code in `src/rasteret/fetch/cog.py`

```python
def merge_tiles(
    tiles: dict[tuple[int, int], np.ndarray],
    tile_size: tuple[int, int],
    dtype: np.dtype,
    *,
    fill_value: float | int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """Merge multiple tiles into a single array.

    Parameters
    ----------
    tiles : dict
        Mapping of ``(row, col)`` to tile ``np.ndarray``.
    tile_size : tuple of int
        ``(tile_width, tile_height)`` in pixels.
    dtype : numpy.dtype
        Desired output dtype.
    fill_value : float or int
        Fill value for empty regions (outside tile coverage).

    Returns
    -------
    tuple
        ``(merged_array, (min_row, min_col, max_row, max_col), tile_mask)``.
        *tile_mask* is True where a tile contributed pixels.
    """
    if not tiles:
        return np.array([], dtype=dtype), (0, 0, 0, 0), np.zeros((0, 0), dtype=bool)

    # Find bounds
    rows, cols = zip(*tiles.keys())
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    tile_width, tile_height = tile_size

    # Create output array with appropriate fill value.
    height = (max_row - min_row + 1) * tile_height
    width = (max_col - min_col + 1) * tile_width

    out_dtype = np.dtype(dtype)
    fill = out_dtype.type(fill_value)

    merged = np.full((height, width), fill, dtype=out_dtype)
    tile_mask = np.zeros((height, width), dtype=bool)

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

            tile_mask[y_start:y_end, x_start:x_end] = True

    return merged, (min_row, min_col, max_row, max_col), tile_mask
```

#### apply_mask_and_crop

```python
apply_mask_and_crop(
    data: ndarray,
    geojson: dict,
    transform: Affine,
    nodata: float | int | None = None,
    *,
    all_touched: bool = False,
    filled: bool = True,
    fill_value: float | int | None = None,
    return_mask: bool = False,
) -> (
    tuple[ndarray, Affine] | tuple[ndarray, ndarray, Affine]
)
```

Apply geometry mask and crop to the valid data region.

Parameters:

| Name          | Type           | Description                                                                                                                                                      | Default    |
| ------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `data`        | `ndarray`      | 2-D raster array.                                                                                                                                                | *required* |
| `geojson`     | `dict`         | GeoJSON dict ({"type": "Polygon", "coordinates": ...}) in the same CRS as transform.                                                                             | *required* |
| `transform`   | `Affine`       | Affine transform for data.                                                                                                                                       | *required* |
| `nodata`      | `float or int` | Dataset nodata value, if known.                                                                                                                                  | `None`     |
| `all_touched` | `bool`         | Passed through to rasterio's geometry_mask (default False).                                                                                                      | `False`    |
| `filled`      | `bool`         | If True, fill pixels outside the geometry with fill_value. If False, do not modify values outside the geometry; return a validity mask when return_mask is True. | `True`     |
| `fill_value`  | `float or int` | Fill value used when filled=True. Defaults to nodata when provided, otherwise 0.                                                                                 | `None`     |
| `return_mask` | `bool`         | When True, return a boolean mask where True indicates pixels inside the geometry.                                                                                | `False`    |

Returns:

| Type    | Description                                                                                  |
| ------- | -------------------------------------------------------------------------------------------- |
| `tuple` | (masked_data, cropped_transform) or (masked_data, mask, cropped_transform) when return_mask. |

Source code in `src/rasteret/fetch/cog.py`

```python
def apply_mask_and_crop(
    data: np.ndarray,
    geojson: dict,
    transform: Affine,
    nodata: float | int | None = None,
    *,
    all_touched: bool = False,
    filled: bool = True,
    fill_value: float | int | None = None,
    return_mask: bool = False,
) -> tuple[np.ndarray, Affine] | tuple[np.ndarray, np.ndarray, Affine]:
    """Apply geometry mask and crop to the valid data region.

    Parameters
    ----------
    data : numpy.ndarray
        2-D raster array.
    geojson : dict
        GeoJSON dict (``{"type": "Polygon", "coordinates": ...}``)
        in the same CRS as *transform*.
    transform : Affine
        Affine transform for *data*.
    nodata : float or int, optional
        Dataset nodata value, if known.
    all_touched : bool
        Passed through to rasterio's ``geometry_mask`` (default False).
    filled : bool
        If True, fill pixels outside the geometry with *fill_value*.
        If False, do not modify values outside the geometry; return a
        validity mask when *return_mask* is True.
    fill_value : float or int, optional
        Fill value used when ``filled=True``. Defaults to *nodata* when
        provided, otherwise 0.
    return_mask : bool
        When True, return a boolean mask where True indicates pixels
        inside the geometry.

    Returns
    -------
    tuple
        ``(masked_data, cropped_transform)`` or
        ``(masked_data, mask, cropped_transform)`` when *return_mask*.
    """

    mask = geometry_mask(
        [geojson],
        out_shape=data.shape,
        transform=transform,
        all_touched=all_touched,
        invert=True,
    )

    fill = (
        data.dtype.type(fill_value)
        if fill_value is not None
        else data.dtype.type(nodata)
        if nodata is not None
        else data.dtype.type(0)
    )

    if not bool(mask.any()):
        # Geometry does not intersect any pixels; return fill-valued array.
        empty_data = np.full(data.shape, fill, dtype=data.dtype)
        empty_mask = np.zeros(data.shape, dtype=bool)
        if return_mask:
            return empty_data, empty_mask, transform
        return empty_data, transform

    # Crop to the geometry's bbox window (rasterio-aligned crop semantics).
    coords = geojson.get("coordinates")
    if not coords:
        raise ValueError("Invalid geojson: missing coordinates")
    # Polygon coordinates: [[[x,y], ...]]; MultiPolygon adds one more nesting.
    while isinstance(coords, list) and coords and isinstance(coords[0], list):
        if (
            coords
            and coords
            and isinstance(coords[0], (tuple, list))
            and len(coords[0]) == 2
        ):
            break
        coords = coords[0]
    xs = [float(x) for x, _y in coords]
    ys = [float(y) for _x, y in coords]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    data_cropped, mask_cropped, cropped_transform = _crop_to_bbox_window(
        data, mask, transform, bbox
    )

    # Apply mask to cropped data (or return unfilled)
    if filled:
        masked_data = np.where(mask_cropped, data_cropped, fill)
    else:
        masked_data = data_cropped

    if return_mask:
        return masked_data, mask_cropped, cropped_transform
    return masked_data, cropped_transform
```

#### read_cog

```python
read_cog(
    url: str,
    metadata: CogMetadata,
    *,
    band_index: int | None = None,
    geom_array: Array | None = None,
    geom_idx: int = 0,
    geometry_crs: int | None = 4326,
    max_concurrent: int = 150,
    debug: bool = False,
    reader: COGReader | None = None,
    all_touched: bool = False,
    filled: bool = True,
    fill_value: float | int | None = None,
    crop: bool = True,
    mode: Literal["aoi", "window", "full"] = "aoi",
    bounds: tuple[float, float, float, float] | None = None,
    out_shape: tuple[int, int] | None = None,
    mask_geometry: Literal["polygon", "bbox"] = "polygon",
) -> CogReadResult
```

Primary Rasteret COG read API.

This returns a :class:`CogReadResult` containing the pixel array, its affine transform, and a `valid_mask` that is True only for pixels inside the requested AOI/window *and* inside raster coverage.

Defaults are rasterio-aligned:

- `all_touched=False`
- when `filled=True`: fill with `nodata` if known, otherwise 0
- preserve native dtype by default (no NaN-driven promotion)

Source code in `src/rasteret/fetch/cog.py`

```python
async def read_cog(
    url: str,
    metadata: CogMetadata,
    *,
    band_index: int | None = None,
    geom_array: pa.Array | None = None,
    geom_idx: int = 0,
    geometry_crs: int | None = 4326,
    max_concurrent: int = 150,
    debug: bool = False,
    reader: COGReader | None = None,
    all_touched: bool = False,
    filled: bool = True,
    fill_value: float | int | None = None,
    crop: bool = True,
    mode: Literal["aoi", "window", "full"] = "aoi",
    bounds: tuple[float, float, float, float] | None = None,
    out_shape: tuple[int, int] | None = None,
    mask_geometry: Literal["polygon", "bbox"] = "polygon",
) -> CogReadResult:
    """Primary Rasteret COG read API.

    This returns a :class:`CogReadResult` containing the pixel array,
    its affine transform, and a ``valid_mask`` that is True only for
    pixels inside the requested AOI/window *and* inside raster coverage.

    Defaults are rasterio-aligned:
    - ``all_touched=False``
    - when ``filled=True``: fill with ``nodata`` if known, otherwise 0
    - preserve native dtype by default (no NaN-driven promotion)
    """
    if debug:
        logger.info(f"Reading COG data from {url}")

    if metadata.transform is None:
        empty = np.array([], dtype=np.dtype(metadata.dtype))
        return CogReadResult(
            data=empty,
            transform=Affine.identity(),
            valid_mask=np.zeros((0, 0), dtype=bool),
            fill_value_used=None,
            mode=mode,
        )

    if (
        int(getattr(metadata, "planar_configuration", 1) or 1) == 1
        and int(getattr(metadata, "samples_per_pixel", 1) or 1) > 1
        and band_index is None
    ):
        raise NotImplementedError(
            "Chunky multi-sample TIFFs (PlanarConfiguration=1, SamplesPerPixel>1) "
            "require an explicit band_index to select the requested sample."
        )

    if (
        metadata.tile_offsets is None
        or metadata.tile_byte_counts is None
        or len(metadata.tile_offsets) == 0
        or len(metadata.tile_byte_counts) == 0
    ):
        raise ValueError(
            "Rasteret's tile reader requires a tiled GeoTIFF (TileOffsets/TileByteCounts). "
            "This asset appears to be non-tiled (or missing tile metadata). "
            "Use TorchGeo/rasterio for this file, or convert it to a tiled COG."
        )

    if len(metadata.tile_offsets) != len(metadata.tile_byte_counts):
        raise ValueError(
            "Invalid tile metadata: TileOffsets/TileByteCounts length mismatch "
            f"({len(metadata.tile_offsets)} vs {len(metadata.tile_byte_counts)})."
        )

    # Derive bbox and GeoJSON from GeoArrow array (no Shapely)
    geom_bbox = None
    geom_geojson = None
    if bounds is not None:
        geom_bbox = bounds
        # bounds is always axis-aligned, used as the AOI/window region
        xmin, ymin, xmax, ymax = geom_bbox
        geom_geojson = {
            "type": "Polygon",
            "coordinates": [
                [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
            ],
        }
        intersecting_tiles = compute_tile_indices(
            geometry_bbox=geom_bbox,
            transform=metadata.transform,
            tile_size=(metadata.tile_width, metadata.tile_height),
            image_size=(metadata.width, metadata.height),
            debug=debug,
        )
    elif geom_array is not None:
        # Always compute the input bbox in the geometry CRS first. This lets us
        # provide a clear error when the record CRS is missing and the geometry
        # looks like WGS84 lon/lat.
        geom_bbox = bbox_single(geom_array, geom_idx)

        if (
            geometry_crs == 4326
            and metadata.crs is None
            and geom_bbox is not None
            and -180.0 <= geom_bbox[0] <= 180.0
            and -180.0 <= geom_bbox[2] <= 180.0
            and -90.0 <= geom_bbox[1] <= 90.0
            and -90.0 <= geom_bbox[3] <= 90.0
        ):
            raise ValueError(
                "Record CRS is missing (proj:epsg) so Rasteret cannot transform the "
                "WGS84 query geometry into raster CRS. Fix by adding a per-record "
                "`proj:epsg` column to your record table, or build with "
                "`enrich_cog=True` so Rasteret can infer CRS from GeoTIFF headers "
                "when available."
            )

        needs_crs_transform = (
            geometry_crs is not None
            and metadata.crs is not None
            and geometry_crs != metadata.crs
        )
        if needs_crs_transform:
            # CRS-transform the geometry, get GeoJSON dict in target CRS
            geom_geojson = transform_coords(
                geom_array, geom_idx, geometry_crs, metadata.crs
            )
            # Extract bbox from the transformed GeoJSON coordinates
            from rasteret.core.geometry import bbox_from_geojson_coords

            geom_bbox = bbox_from_geojson_coords(geom_geojson)

            if debug:
                logger.info(
                    f"Transformed geometry bbox "
                    f"(EPSG:{geometry_crs} -> EPSG:{metadata.crs}): "
                    f"{geom_bbox}"
                )
        else:
            geom_geojson = to_rasterio_geojson(geom_array, geom_idx)

        if geom_bbox is not None and mask_geometry == "bbox":
            xmin, ymin, xmax, ymax = geom_bbox
            geom_geojson = {
                "type": "Polygon",
                "coordinates": [
                    [
                        (xmin, ymin),
                        (xmax, ymin),
                        (xmax, ymax),
                        (xmin, ymax),
                        (xmin, ymin),
                    ]
                ],
            }

        # Get tiles that intersect with geometry bbox
        intersecting_tiles = compute_tile_indices(
            geometry_bbox=geom_bbox,
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
        empty = np.array([], dtype=np.dtype(metadata.dtype))
        return CogReadResult(
            data=empty,
            transform=Affine.identity(),
            valid_mask=np.zeros((0, 0), dtype=bool),
            fill_value_used=None,
            mode=mode,
        )

    # Create tile requests
    requests = []
    tiles_x = (metadata.width + metadata.tile_width - 1) // metadata.tile_width
    tiles_y = (metadata.height + metadata.tile_height - 1) // metadata.tile_height
    expected_tiles = tiles_x * tiles_y
    if expected_tiles > len(metadata.tile_offsets):
        raise ValueError(
            "Unsupported tiled GeoTIFF layout: tile offset table is shorter than expected "
            f"({len(metadata.tile_offsets)} < {expected_tiles})."
        )

    for row, col in intersecting_tiles:
        tile_idx = row * tiles_x + col
        if tile_idx >= len(metadata.tile_offsets):
            if debug:
                logger.warning(f"Tile index {tile_idx} out of bounds")
            continue

        requests.append(
            COGTileRequest(
                url=url,
                offset=metadata.tile_offsets[tile_idx],
                size=metadata.tile_byte_counts[tile_idx],
                row=row,
                col=col,
                metadata=metadata,
                band_index=band_index,
            )
        )

    # Use COGReader for efficient tile reading
    if reader is None:
        async with COGReader(max_concurrent=max_concurrent) as local_reader:
            raw_tiles = await local_reader.read_merged_tiles(requests, debug=debug)
    else:
        raw_tiles = await reader.read_merged_tiles(requests, debug=debug)

    tiles: dict[tuple[int, int], np.ndarray] = {}
    for (row, col, _band_index), arr in raw_tiles.items():
        key = (row, col)
        if key in tiles:
            raise RuntimeError(
                f"Duplicate tile key produced for read_cog merge: {key}. "
                "Expected one array per tile in single-band read path."
            )
        tiles[key] = arr

    if not tiles:
        empty = np.array([], dtype=np.dtype(metadata.dtype))
        return CogReadResult(
            data=empty,
            transform=Affine.identity(),
            valid_mask=np.zeros((0, 0), dtype=bool),
            fill_value_used=None,
            mode=mode,
        )

    # Merge tiles and handle transforms
    native_dtype = np.dtype(
        metadata.dtype
        if not hasattr(metadata.dtype, "to_pandas_dtype")
        else metadata.dtype.to_pandas_dtype()
    )
    fill_value_used: float | int = (
        fill_value
        if fill_value is not None
        else metadata.nodata
        if metadata.nodata is not None
        else 0
    )

    merged_data, tile_bounds, tile_mask = merge_tiles(
        tiles,
        (metadata.tile_width, metadata.tile_height),
        dtype=native_dtype,
        fill_value=fill_value_used,
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
    min_row, min_col, max_row, max_col = tile_bounds
    scale_x, translate_x, scale_y, translate_y = normalize_transform(metadata.transform)
    src_transform = Affine(scale_x, 0.0, translate_x, 0.0, scale_y, translate_y)

    merged_transform = Affine(
        scale_x,
        0,
        translate_x + min_col * metadata.tile_width * scale_x,
        0,
        scale_y,
        translate_y + min_row * metadata.tile_height * scale_y,
    )

    merged_global_row0 = min_row * metadata.tile_height
    merged_global_col0 = min_col * metadata.tile_width

    coverage = _coverage_mask_for_merged(
        merged_shape=merged_data.shape,
        raster_width=metadata.width,
        raster_height=metadata.height,
        merged_global_row0=merged_global_row0,
        merged_global_col0=merged_global_col0,
    )
    # Where tiles exist AND within raster extent.
    valid_mask = coverage & tile_mask

    if mode == "full" or (geom_geojson is None and mode == "aoi"):
        # For full reads, crop to exact raster extent (remove padded tile multiples).
        max_rows = max(
            0, min(merged_data.shape[0], metadata.height - merged_global_row0)
        )
        max_cols = max(
            0, min(merged_data.shape[1], metadata.width - merged_global_col0)
        )
        merged_data = merged_data[:max_rows, :max_cols]
        valid_mask = valid_mask[:max_rows, :max_cols]
        merged_transform = merged_transform

    elif mode == "aoi" and geom_geojson is not None:
        if crop:
            row_start, row_stop, col_start, col_stop = _geometry_window_pixels(
                geom_geojson=geom_geojson,
                transform=src_transform,
                raster_width=metadata.width,
                raster_height=metadata.height,
            )

            local_r0 = row_start - merged_global_row0
            local_c0 = col_start - merged_global_col0
            local_r1 = row_stop - merged_global_row0
            local_c1 = col_stop - merged_global_col0

            local_r0 = max(0, local_r0)
            local_c0 = max(0, local_c0)
            local_r1 = min(merged_data.shape[0], local_r1)
            local_c1 = min(merged_data.shape[1], local_c1)

            if local_r1 <= local_r0 or local_c1 <= local_c0:
                empty = merged_data[:0, :0]
                empty_mask = valid_mask[:0, :0]
                return CogReadResult(
                    data=empty,
                    transform=Affine.identity(),
                    valid_mask=empty_mask,
                    fill_value_used=fill_value_used if filled else None,
                    mode=mode,
                )

            merged_data = merged_data[local_r0:local_r1, local_c0:local_c1]
            valid_mask = valid_mask[local_r0:local_r1, local_c0:local_c1]
            merged_transform = Affine(
                src_transform.a,
                src_transform.b,
                src_transform.c + col_start * src_transform.a,
                src_transform.d,
                src_transform.e,
                src_transform.f + row_start * src_transform.e,
            )

            # Apply geometry mask in the cropped window grid (matches rasterio.mask).
            geom_mask = geometry_mask(
                [geom_geojson],
                out_shape=merged_data.shape,
                transform=merged_transform,
                all_touched=all_touched,
                invert=True,
            )
            valid_mask = valid_mask & geom_mask

        else:
            # Geometry mask in raster CRS (tile-aligned canvas).
            geom_mask = geometry_mask(
                [geom_geojson],
                out_shape=merged_data.shape,
                transform=merged_transform,
                all_touched=all_touched,
                invert=True,
            )
            valid_mask = valid_mask & geom_mask

        if filled and merged_data.size:
            merged_data = np.where(
                valid_mask,
                merged_data,
                np.array(fill_value_used, dtype=merged_data.dtype),
            )

    elif mode == "window":
        if geom_bbox is None:
            raise ValueError("window mode requires bounds or geom_array")
        if metadata.transform is None:
            raise ValueError("window mode requires a valid transform")

        xmin, ymin, xmax, ymax = geom_bbox
        window_geojson = {
            "type": "Polygon",
            "coordinates": [
                [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
            ],
        }
        row_start, row_stop, col_start, col_stop = _geometry_window_pixels(
            geom_geojson=window_geojson,
            transform=src_transform,
            raster_width=None,
            raster_height=None,
        )
        out_h = max(row_stop - row_start, 0)
        out_w = max(col_stop - col_start, 0)
        if out_shape is not None:
            out_h, out_w = out_shape

        out_data = np.full(
            (out_h, out_w), native_dtype.type(fill_value_used), dtype=native_dtype
        )
        out_mask = np.zeros((out_h, out_w), dtype=bool)

        # Overlap region in global pixel coords.
        overlap_row0 = max(row_start, merged_global_row0)
        overlap_col0 = max(col_start, merged_global_col0)
        overlap_row1 = min(row_start + out_h, merged_global_row0 + merged_data.shape[0])
        overlap_col1 = min(col_start + out_w, merged_global_col0 + merged_data.shape[1])
        if overlap_row1 > overlap_row0 and overlap_col1 > overlap_col0:
            src_r0 = overlap_row0 - merged_global_row0
            src_c0 = overlap_col0 - merged_global_col0
            src_r1 = overlap_row1 - merged_global_row0
            src_c1 = overlap_col1 - merged_global_col0
            dst_r0 = overlap_row0 - row_start
            dst_c0 = overlap_col0 - col_start
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out_data[dst_r0:dst_r1, dst_c0:dst_c1] = merged_data[
                src_r0:src_r1, src_c0:src_c1
            ]
            out_mask[dst_r0:dst_r1, dst_c0:dst_c1] = valid_mask[
                src_r0:src_r1, src_c0:src_c1
            ]

        merged_data = out_data
        valid_mask = out_mask
        merged_transform = Affine(
            src_transform.a,
            0,
            src_transform.c + col_start * src_transform.a,
            0,
            src_transform.e,
            src_transform.f + row_start * src_transform.e,
        )

    else:
        raise ValueError(f"Unknown read mode: {mode}")

    if debug:
        logger.info(
            f"""
        Final Output:
        - Shape: {merged_data.shape}
        - Transform: {merged_transform}
        - Data Range: {np.nanmin(merged_data)}-{np.nanmax(merged_data)}
        """
        )

    return CogReadResult(
        data=merged_data,
        transform=merged_transform,
        valid_mask=valid_mask,
        fill_value_used=fill_value_used if filled else None,
        mode=mode,
    )
```
