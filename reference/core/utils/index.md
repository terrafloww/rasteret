# rasteret.core.utils

Geometry, CRS, and grid computation utilities.

## utils

### Classes

### Functions

#### run_sync

```python
run_sync(coro: Any) -> Any
```

Run a coroutine from synchronous API entrypoints.

Jupyter-safe: if an event loop is already running (e.g. inside a notebook), the coroutine is dispatched to a background thread.

Source code in `src/rasteret/core/utils.py`

```python
def run_sync(coro: Any) -> Any:
    """Run a coroutine from synchronous API entrypoints.

    Jupyter-safe: if an event loop is already running (e.g. inside a notebook),
    the coroutine is dispatched to a background thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
```

#### infer_data_source

```python
infer_data_source(collection: 'Collection') -> str
```

Infer collection source for URL policy and band mapping.

Returns an empty string when no data source can be inferred.

Source code in `src/rasteret/core/utils.py`

```python
def infer_data_source(collection: "Collection") -> str:
    """Infer collection source for URL policy and band mapping.

    Returns an empty string when no data source can be inferred.
    """
    if collection.data_source:
        return str(collection.data_source)

    if getattr(collection, "_hf_streaming", None) is not None:
        try:
            head = collection.head(1, columns=["collection"])
            if head.num_rows:
                source = head.column("collection")[0].as_py()
                if isinstance(source, str) and source:
                    return source
        except Exception as exc:
            logger.debug(
                "Failed to infer data_source from HF streaming collection: %s", exc
            )

    return infer_data_source_from_dataset(collection.dataset)
```

#### infer_data_source_from_dataset

```python
infer_data_source_from_dataset(dataset: Any) -> str
```

Infer data source from a PyArrow dataset's metadata/columns.

Parameters:

| Name      | Type              | Description                   | Default    |
| --------- | ----------------- | ----------------------------- | ---------- |
| `dataset` | `Dataset or None` | Dataset backing a Collection. | *required* |

Returns:

| Type  | Description                                         |
| ----- | --------------------------------------------------- |
| `str` | Inferred data source, or empty string when unknown. |

Source code in `src/rasteret/core/utils.py`

```python
def infer_data_source_from_dataset(dataset: Any) -> str:
    """Infer data source from a PyArrow dataset's metadata/columns.

    Parameters
    ----------
    dataset : pyarrow.dataset.Dataset or None
        Dataset backing a Collection.

    Returns
    -------
    str
        Inferred data source, or empty string when unknown.
    """
    if dataset is None:
        return ""

    metadata = getattr(dataset.schema, "metadata", {}) or {}
    raw = metadata.get(b"data_source")
    if raw:
        try:
            decoded = raw.decode("utf-8").strip()
        except (UnicodeDecodeError, AttributeError):
            decoded = ""
        if decoded and decoded.lower() != "unknown":
            return decoded

    if "collection" in dataset.schema.names:
        # Avoid materializing the entire column.
        try:
            expr = ds.field("collection").is_valid() & (ds.field("collection") != "")
            scanner = dataset.scanner(columns=["collection"], filter=expr)
            head = scanner.head(1)
            if head.num_rows:
                source = head.column("collection")[0].as_py()
                if isinstance(source, str) and source:
                    return source
        except (pa.ArrowInvalid, pa.ArrowKeyError, OSError) as exc:
            # Best-effort fallback; failure here should not break reads.
            logger.debug("Failed to read 'collection' column: %s", exc)

    logger.debug("Could not infer data_source for Collection")
    return ""
```

#### transform_polygon

```python
transform_polygon(
    geom, src_crs: int | str, dst_crs: int | str
)
```

Transform a Shapely polygon between coordinate systems.

Parameters:

| Name      | Type         | Description                           | Default    |
| --------- | ------------ | ------------------------------------- | ---------- |
| `geom`    | `Geometry`   | Input Shapely polygon geometry.       | *required* |
| `src_crs` | `int or str` | Source CRS (EPSG code or WKT string). | *required* |
| `dst_crs` | `int or str` | Target CRS (EPSG code or WKT string). | *required* |

Returns:

| Type       | Description                  |
| ---------- | ---------------------------- |
| `Geometry` | Reprojected Shapely polygon. |

Source code in `src/rasteret/core/utils.py`

```python
def transform_polygon(
    geom,
    src_crs: int | str,
    dst_crs: int | str,
):
    """Transform a Shapely polygon between coordinate systems.

    Parameters
    ----------
    geom : shapely.Geometry
        Input Shapely polygon geometry.
    src_crs : int or str
        Source CRS (EPSG code or WKT string).
    dst_crs : int or str
        Target CRS (EPSG code or WKT string).

    Returns
    -------
    shapely.Geometry
        Reprojected Shapely polygon.
    """
    from shapely.ops import transform

    transformer = Transformer.from_crs(
        f"EPSG:{src_crs}" if isinstance(src_crs, int) else src_crs,
        f"EPSG:{dst_crs}" if isinstance(dst_crs, int) else dst_crs,
        always_xy=True,
    )
    return transform(transformer.transform, geom)
```

#### transform_bbox

```python
transform_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: int | str,
    dst_crs: int | str,
) -> tuple[float, float, float, float]
```

Transform bounding box between coordinate systems.

Parameters:

| Name      | Type                                | Description                           | Default    |
| --------- | ----------------------------------- | ------------------------------------- | ---------- |
| `bbox`    | `tuple[float, float, float, float]` | Input bbox (minx, miny, maxx, maxy).  | *required* |
| `src_crs` | `int or str`                        | Source CRS (EPSG code or WKT string). | *required* |
| `dst_crs` | `int or str`                        | Target CRS (EPSG code or WKT string). | *required* |

Returns:

| Type                                | Description       |
| ----------------------------------- | ----------------- |
| `tuple[float, float, float, float]` | Transformed bbox. |

Source code in `src/rasteret/core/utils.py`

```python
def transform_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: int | str,
    dst_crs: int | str,
) -> tuple[float, float, float, float]:
    """Transform bounding box between coordinate systems.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Input bbox ``(minx, miny, maxx, maxy)``.
    src_crs : int or str
        Source CRS (EPSG code or WKT string).
    dst_crs : int or str
        Target CRS (EPSG code or WKT string).

    Returns
    -------
    tuple[float, float, float, float]
        Transformed bbox.
    """
    if hasattr(bbox, "bounds"):
        minx, miny, maxx, maxy = bbox.bounds
    else:
        minx, miny, maxx, maxy = bbox

    src_str = f"EPSG:{src_crs}" if isinstance(src_crs, int) else src_crs
    dst_str = f"EPSG:{dst_crs}" if isinstance(dst_crs, int) else dst_crs
    transformer = Transformer.from_crs(src_str, dst_str, always_xy=True)
    return transformer.transform_bounds(minx, miny, maxx, maxy)
```

#### normalize_transform

```python
normalize_transform(
    transform: object,
) -> tuple[float, float, float, float]
```

Normalize common affine transform representations.

Accepts:

- 4 values: (scale_x, translate_x, scale_y, translate_y)
- 6 values: GDAL/rasterio affine (a, b, c, d, e, f) for north-up rasters

Returns: (scale_x, translate_x, scale_y, translate_y)

Source code in `src/rasteret/core/utils.py`

```python
def normalize_transform(transform: object) -> tuple[float, float, float, float]:
    """Normalize common affine transform representations.

    Accepts:
      - 4 values: (scale_x, translate_x, scale_y, translate_y)
      - 6 values: GDAL/rasterio affine (a, b, c, d, e, f) for north-up rasters

    Returns:
      (scale_x, translate_x, scale_y, translate_y)
    """
    if transform is None:
        raise ValueError("Transform is missing")

    try:
        values = list(transform)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"Transform must be iterable, got {type(transform)!r}") from exc

    if len(values) == 4:
        sx, tx, sy, ty = values
        return float(sx), float(tx), float(sy), float(ty)

    if len(values) == 6:
        a, b, c, d, e, f = values
        if not np.isclose(float(b), 0.0) or not np.isclose(float(d), 0.0):
            raise ValueError(
                "Rotated/sheared affine transforms are not supported " f"(b={b}, d={d})"
            )
        return float(a), float(c), float(e), float(f)

    raise ValueError(f"Transform must have 4 or 6 values, got {len(values)}")
```

#### reproject_array

```python
reproject_array(
    src_array: ndarray,
    src_transform: object,
    src_crs: int,
    dst_crs: int,
    dst_transform: object,
    dst_shape: tuple[int, int],
    resampling: str = "bilinear",
) -> ndarray
```

Reproject a 2-D array between coordinate reference systems.

Thin wrapper around `rasterio.warp.reproject` that works with in-memory numpy arrays and EPSG codes; no file handle required.

Parameters:

| Name            | Type              | Description                                                                                                         | Default      |
| --------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- | ------------ |
| `src_array`     | `ndarray`         | Input 2-D array (any numeric dtype; integer dtypes are promoted to float32 so that NaN fill values work correctly). | *required*   |
| `src_transform` | `Affine`          | Affine transform for the source grid.                                                                               | *required*   |
| `src_crs`       | `int`             | Source EPSG code.                                                                                                   | *required*   |
| `dst_crs`       | `int`             | Target EPSG code.                                                                                                   | *required*   |
| `dst_transform` | `Affine`          | Affine transform for the destination grid.                                                                          | *required*   |
| `dst_shape`     | `tuple[int, int]` | (height, width) of the destination array.                                                                           | *required*   |
| `resampling`    | `str`             | Resampling method name (default "bilinear").                                                                        | `'bilinear'` |

Returns:

| Type      | Description                                                       |
| --------- | ----------------------------------------------------------------- |
| `ndarray` | Reprojected 2-D float32 array with NaN where no source data maps. |

Source code in `src/rasteret/core/utils.py`

```python
def reproject_array(
    src_array: np.ndarray,
    src_transform: object,
    src_crs: int,
    dst_crs: int,
    dst_transform: object,
    dst_shape: tuple[int, int],
    resampling: str = "bilinear",
) -> np.ndarray:
    """Reproject a 2-D array between coordinate reference systems.

    Thin wrapper around ``rasterio.warp.reproject`` that works with
    in-memory numpy arrays and EPSG codes; no file handle required.

    Parameters
    ----------
    src_array : numpy.ndarray
        Input 2-D array (any numeric dtype; integer dtypes are promoted
        to ``float32`` so that NaN fill values work correctly).
    src_transform : Affine
        Affine transform for the source grid.
    src_crs : int
        Source EPSG code.
    dst_crs : int
        Target EPSG code.
    dst_transform : Affine
        Affine transform for the destination grid.
    dst_shape : tuple[int, int]
        ``(height, width)`` of the destination array.
    resampling : str
        Resampling method name (default ``"bilinear"``).

    Returns
    -------
    numpy.ndarray
        Reprojected 2-D ``float32`` array with ``NaN`` where no source
        data maps.
    """
    from rasterio.crs import CRS as RioCRS
    from rasterio.warp import Resampling, reproject

    # Always use a float dtype so NaN fill works correctly.
    # Integer dtypes (uint16, int8, etc.) silently cast NaN -> 0.
    out_dtype = src_array.dtype
    if not np.issubdtype(out_dtype, np.floating):
        out_dtype = np.float32
    dst_array = np.full(dst_shape, np.nan, dtype=out_dtype)
    src_for_warp = src_array.astype(out_dtype, copy=False)
    reproject(
        source=src_for_warp,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=RioCRS.from_epsg(src_crs),
        dst_transform=dst_transform,
        dst_crs=RioCRS.from_epsg(dst_crs),
        resampling=getattr(Resampling, resampling),
    )
    return dst_array
```

#### compute_dst_grid

```python
compute_dst_grid(
    bounds: tuple[float, float, float, float],
    res: tuple[float, float],
) -> tuple[object, tuple[int, int]]
```

Compute an Affine transform and pixel dimensions for a target grid.

The caller must supply *res* in the **destination CRS units**. When the source and destination CRS share the same linear unit (e.g. both UTM metres) the source resolution can be passed directly. For cross-unit reprojection (e.g. UTM metres -> EPSG:4326 degrees) use :func:`compute_dst_grid_from_src` instead, which delegates to `rasterio.warp.calculate_default_transform`.

Args: bounds: `(xmin, ymin, xmax, ymax)` in the target CRS. res: `(res_x, res_y)` pixel resolution **in target CRS units**.

Returns: `(affine_transform, (height, width))`

Source code in `src/rasteret/core/utils.py`

```python
def compute_dst_grid(
    bounds: tuple[float, float, float, float],
    res: tuple[float, float],
) -> tuple[object, tuple[int, int]]:
    """Compute an Affine transform and pixel dimensions for a target grid.

    The caller must supply *res* in the **destination CRS units**.  When
    the source and destination CRS share the same linear unit (e.g. both
    UTM metres) the source resolution can be passed directly.  For
    cross-unit reprojection (e.g. UTM metres -> EPSG:4326 degrees) use
    :func:`compute_dst_grid_from_src` instead, which delegates to
    ``rasterio.warp.calculate_default_transform``.

    Args:
        bounds: ``(xmin, ymin, xmax, ymax)`` in the target CRS.
        res: ``(res_x, res_y)`` pixel resolution **in target CRS units**.

    Returns:
        ``(affine_transform, (height, width))``
    """
    from affine import Affine as _Affine

    xmin, ymin, xmax, ymax = bounds
    # Match rasterio-style windowing semantics: the output grid must fully
    # cover the requested bounds (ceil, not round).
    width = int(math.ceil((xmax - xmin) / res[0]))
    height = int(math.ceil((ymax - ymin) / res[1]))
    dst_transform = _Affine(res[0], 0, xmin, 0, -res[1], ymax)
    return dst_transform, (height, width)
```

#### compute_dst_grid_from_src

```python
compute_dst_grid_from_src(
    src_crs: int,
    dst_crs: int,
    width: int,
    height: int,
    src_bounds: tuple[float, float, float, float],
) -> tuple[object, tuple[int, int]]
```

Compute destination grid via GDAL's suggested-warp-output algorithm.

Wraps `rasterio.warp.calculate_default_transform` which delegates to GDAL's `GDALSuggestedWarpOutput2`. This correctly handles cross-unit CRS conversions (e.g. UTM metres -> EPSG:4326 degrees) by sampling the source grid, transforming points, and computing an optimal destination pixel size that preserves spatial-information density.

Use this instead of :func:`compute_dst_grid` whenever the source and destination CRS may have different linear units.

Args: src_crs: Source EPSG code. dst_crs: Destination EPSG code. width: Source raster width (pixels). height: Source raster height (pixels). src_bounds: `(left, bottom, right, top)` in the **source** CRS.

Returns: `(affine_transform, (height, width))` in the destination CRS.

Source code in `src/rasteret/core/utils.py`

```python
def compute_dst_grid_from_src(
    src_crs: int,
    dst_crs: int,
    width: int,
    height: int,
    src_bounds: tuple[float, float, float, float],
) -> tuple[object, tuple[int, int]]:
    """Compute destination grid via GDAL's suggested-warp-output algorithm.

    Wraps ``rasterio.warp.calculate_default_transform`` which delegates to
    GDAL's ``GDALSuggestedWarpOutput2``.  This correctly handles cross-unit
    CRS conversions (e.g. UTM metres -> EPSG:4326 degrees) by sampling the
    source grid, transforming points, and computing an optimal destination
    pixel size that preserves spatial-information density.

    Use this instead of :func:`compute_dst_grid` whenever the source and
    destination CRS may have different linear units.

    Args:
        src_crs: Source EPSG code.
        dst_crs: Destination EPSG code.
        width: Source raster width (pixels).
        height: Source raster height (pixels).
        src_bounds: ``(left, bottom, right, top)`` in the **source** CRS.

    Returns:
        ``(affine_transform, (height, width))`` in the destination CRS.
    """
    from rasterio.crs import CRS as RioCRS
    from rasterio.warp import calculate_default_transform

    dst_transform, dst_width, dst_height = calculate_default_transform(
        RioCRS.from_epsg(src_crs),
        RioCRS.from_epsg(dst_crs),
        width,
        height,
        *src_bounds,
    )
    return dst_transform, (dst_height, dst_width)
```
