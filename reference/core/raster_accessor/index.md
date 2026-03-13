# rasteret.core.raster_accessor

Data-loading handle for a single Parquet row (record). Each record in a Collection gets a `RasterAccessor` via `Collection.iterate_rasters()`.

## raster_accessor

### Classes

#### RasterAccessor

```python
RasterAccessor(info: RasterInfo, data_source: str)
```

Data-loading handle for a single Parquet record (row) in a Collection.

Each record in a Rasteret Collection represents one raster item: typically a satellite scene, but could be a drone image, derived product, or any tiled GeoTIFF. `RasterAccessor` wraps that record's metadata and provides methods to load band data as arrays.

Handles:

- Async band data loading via cached COG metadata
- Tile management and geometry masking
- Multi-band concurrent fetching

Initialize from a record's metadata.

Parameters:

| Name          | Type         | Description                                      | Default    |
| ------------- | ------------ | ------------------------------------------------ | ---------- |
| `info`        | `RasterInfo` | Record metadata including URLs and COG metadata. | *required* |
| `data_source` | `str`        | Data source identifier for band mapping.         | *required* |

Source code in `src/rasteret/core/raster_accessor.py`

```python
def __init__(self, info: RasterInfo, data_source: str) -> None:
    """Initialize from a record's metadata.

    Parameters
    ----------
    info : RasterInfo
        Record metadata including URLs and COG metadata.
    data_source : str
        Data source identifier for band mapping.
    """
    self.id = info.id
    self.datetime = info.datetime
    self.bbox = info.bbox
    self.footprint = info.footprint
    self.crs = info.crs
    self.cloud_cover = info.cloud_cover
    self.assets = info.assets
    self.band_metadata = info.band_metadata
    self.collection = info.collection
    self.data_source = data_source
```

##### Attributes

###### geometry

```python
geometry
```

Alias for `footprint`.

###### available_bands

```python
available_bands: list[str]
```

List available band keys for this record.

##### Functions

###### try_get_band_cog_metadata

```python
try_get_band_cog_metadata(
    band_code: str,
) -> tuple[CogMetadata | None, str | None, int | None]
```

Return tiled GeoTIFF/COG metadata and URL for *band_code*.

Returns `(None, None)` when the asset or required per-band metadata is missing.

Source code in `src/rasteret/core/raster_accessor.py`

```python
def try_get_band_cog_metadata(
    self,
    band_code: str,
) -> tuple[CogMetadata | None, str | None, int | None]:
    """Return tiled GeoTIFF/COG metadata and URL for *band_code*.

    Returns ``(None, None)`` when the asset or required per-band metadata
    is missing.
    """

    # Support both legacy asset-key conventions:
    # - Old STAC-backed Collections often use STAC asset keys (e.g. "blue")
    # - Newer/normalized Collections use logical band codes (e.g. "B02")
    #
    # Resolve by trying: direct band code, registry forward map (B02->blue),
    # then registry reverse map ("blue"->B02), taking the first key that exists.
    candidates: list[str] = [band_code]
    band_map = BandRegistry.get(self.data_source)
    forward = band_map.get(band_code)
    if forward:
        candidates.append(forward)
    if band_map and band_code in band_map.values():
        reverse = {v: k for k, v in band_map.items()}
        back = reverse.get(band_code)
        if back:
            candidates.append(back)

    asset_key = next((c for c in candidates if c in self.assets), None)
    if asset_key is None:
        return None, None, None

    asset = self.assets[asset_key]

    url = self._extract_asset_href(asset)
    band_index = asset.get("band_index") if isinstance(asset, dict) else None

    # Band metadata key could be either band_code or resolved asset_key
    metadata_keys = [f"{band_code}_metadata", f"{asset_key}_metadata"]
    raw_metadata = None
    for key in metadata_keys:
        if key in self.band_metadata:
            raw_metadata = self.band_metadata[key]
            break

    if raw_metadata is None or url is None:
        return None, None, None

    try:
        cog_metadata = CogMetadata.from_dict(raw_metadata, crs=self.crs)
        idx = None
        if band_index is not None:
            try:
                idx = int(band_index)
            except (TypeError, ValueError):
                idx = None
        return cog_metadata, url, idx
    except KeyError:
        return None, None, None
```

###### intersects

```python
intersects(geometry) -> bool
```

Return `True` if this record's bbox overlaps *geometry*'s bbox.

Source code in `src/rasteret/core/raster_accessor.py`

```python
def intersects(self, geometry) -> bool:
    """Return ``True`` if this record's bbox overlaps *geometry*'s bbox."""
    from rasteret.core.geometry import (
        bbox_array,
        bbox_intersects,
        coerce_to_geoarrow,
    )

    geo_arr = coerce_to_geoarrow(geometry)
    xmin, ymin, xmax, ymax = bbox_array(geo_arr)
    geom_bbox = (xmin[0].as_py(), ymin[0].as_py(), xmax[0].as_py(), ymax[0].as_py())
    record_bbox = tuple(self.bbox) if self.bbox else None
    if record_bbox is None:
        return False
    return bbox_intersects(record_bbox, geom_bbox)
```

###### sample_points

```python
sample_points(
    *,
    points: Array,
    band_codes: list[str],
    point_indices: list[int] | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    reader: object | None = None,
    geometry_crs: int | None = 4326,
    method: str = "nearest",
) -> Table
```

Sample point values for this record.

Parameters:

| Name             | Type          | Description                                                                                                                | Default     |
| ---------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `points`         | `Array`       | GeoArrow-native point array.                                                                                               | *required*  |
| `band_codes`     | `list of str` | Band codes to sample.                                                                                                      | *required*  |
| `point_indices`  | `list of int` | Absolute point indices corresponding to points. When omitted, emitted rows use the local point positions.                  | `None`      |
| `max_concurrent` | `int`         | Maximum concurrent HTTP requests.                                                                                          | `50`        |
| `backend`        | `object`      | Pluggable I/O backend.                                                                                                     | `None`      |
| `reader`         | `COGReader`   | Active shared COG reader for connection/session reuse across records. When omitted, this method creates and owns a reader. | `None`      |
| `geometry_crs`   | `int`         | CRS EPSG code of input points. Defaults to EPSG:4326.                                                                      | `4326`      |
| `method`         | `str`         | Sampling method. Only "nearest" is currently supported.                                                                    | `'nearest'` |

Returns:

| Type    | Description                                       |
| ------- | ------------------------------------------------- |
| `Table` | One row per (point, band) sample for this record. |

Source code in `src/rasteret/core/raster_accessor.py`

```python
async def sample_points(
    self,
    *,
    points: pa.Array,
    band_codes: list[str],
    point_indices: list[int] | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    reader: object | None = None,
    geometry_crs: int | None = 4326,
    method: str = "nearest",
) -> pa.Table:
    """Sample point values for this record.

    Parameters
    ----------
    points : pa.Array
        GeoArrow-native point array.
    band_codes : list of str
        Band codes to sample.
    point_indices : list of int, optional
        Absolute point indices corresponding to *points*. When omitted,
        emitted rows use the local point positions.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    backend : object, optional
        Pluggable I/O backend.
    reader : COGReader, optional
        Active shared COG reader for connection/session reuse across
        records. When omitted, this method creates and owns a reader.
    geometry_crs : int, optional
        CRS EPSG code of input points. Defaults to EPSG:4326.
    method : str
        Sampling method. Only ``"nearest"`` is currently supported.

    Returns
    -------
    pyarrow.Table
        One row per ``(point, band)`` sample for this record.
    """
    if method != "nearest":
        raise ValueError("Only nearest point sampling is supported currently.")

    type_name = getattr(points.type, "extension_name", "") or ""
    if "geoarrow.point" not in type_name:
        from rasteret.core.geometry import UnsupportedGeometryError

        raise UnsupportedGeometryError(
            "Point sampling requires Point geometries. "
            "Use get_xarray/get_numpy/get_gdf for Polygon/MultiPolygon AOIs."
        )

    import geoarrow.pyarrow as ga

    from rasteret.fetch.cog import COGReader, COGTileRequest

    transformer = None
    if (
        geometry_crs is not None
        and self.crs is not None
        and geometry_crs != self.crs
    ):
        from pyproj import Transformer

        transformer = Transformer.from_crs(geometry_crs, self.crs, always_xy=True)

    record_datetime_us: np.datetime64 | None = None
    if self.datetime is not None:
        try:
            record_datetime_us = np.datetime64(
                pd.Timestamp(self.datetime).to_datetime64(), "us"
            )
        except (OverflowError, TypeError, ValueError):
            logger.debug("Could not normalize record datetime for %s", self.id)

    point_crs_value = int(geometry_crs) if geometry_crs is not None else None
    raster_crs_value = int(self.crs) if self.crs is not None else None
    cloud_cover_value = (
        float(self.cloud_cover) if self.cloud_cover is not None else None
    )
    point_indices_arr: np.ndarray | None = None
    point_xs_arr, point_ys_arr = ga.point_coords(points)
    point_xs = point_xs_arr.to_numpy(zero_copy_only=False)
    point_ys = point_ys_arr.to_numpy(zero_copy_only=False)
    if point_indices is not None:
        point_indices_arr = np.asarray(point_indices, dtype=np.int64)
        if len(point_indices_arr) != len(point_xs):
            raise ValueError("point_indices length must match the number of points")
    sample_xs, sample_ys = point_xs, point_ys
    if transformer is not None:
        sample_xs, sample_ys = transformer.transform(point_xs, point_ys)

    def _constant_int32_array(value: int | None, row_count: int) -> pa.Array:
        if value is None:
            return pa.nulls(row_count, type=pa.int32())
        return pa.array(np.full(row_count, value, dtype=np.int32), type=pa.int32())

    def _constant_float64_array(value: float | None, row_count: int) -> pa.Array:
        if value is None:
            return pa.nulls(row_count, type=pa.float64())
        return pa.array(
            np.full(row_count, value, dtype=np.float64), type=pa.float64()
        )

    def _constant_timestamp_array(
        value: np.datetime64 | None, row_count: int
    ) -> pa.Array:
        if value is None:
            return pa.nulls(row_count, type=pa.timestamp("us"))
        return pa.array(
            np.full(row_count, value, dtype="datetime64[us]"),
            type=pa.timestamp("us"),
        )

    def _constant_string_array(value: str, row_count: int) -> pa.Array:
        return pa.array(np.full(row_count, value, dtype=object), type=pa.string())

    band_sources: list[dict[str, object]] = []
    for band_code in band_codes:
        cog_meta, url, band_index = self.try_get_band_cog_metadata(band_code)
        if cog_meta is None or url is None or cog_meta.transform is None:
            raise ValueError(
                f"Missing band metadata or href for band '{band_code}' "
                f"in record '{self.id}'"
            )

        scale_x, trans_x, scale_y, trans_y = normalize_transform(cog_meta.transform)
        src_transform = Affine(
            float(scale_x),
            0.0,
            float(trans_x),
            0.0,
            float(scale_y),
            float(trans_y),
        )
        band_sources.append(
            {
                "band_code": band_code,
                "metadata": cog_meta,
                "url": url,
                "band_index": band_index,
                "transform": src_transform,
                "group_key": (
                    url,
                    tuple(float(value) for value in cog_meta.transform),
                    int(cog_meta.width),
                    int(cog_meta.height),
                    int(cog_meta.tile_width),
                    int(cog_meta.tile_height),
                    str(np.dtype(cog_meta.dtype)),
                    int(getattr(cog_meta, "samples_per_pixel", 1) or 1),
                    int(getattr(cog_meta, "planar_configuration", 1) or 1),
                    # Planar TIFFs store each band in its own IFD (with distinct
                    # tile offsets). Grouping across band_index would incorrectly
                    # read band 0's tiles for every band.
                    int(band_index)
                    if int(getattr(cog_meta, "planar_configuration", 1) or 1) == 2
                    else None,
                ),
            }
        )

    def _tile_window(
        metadata: CogMetadata,
        *,
        tile_row: int,
        tile_col: int,
    ) -> tuple[int, int, int, int]:
        row_start = tile_row * int(metadata.tile_height)
        col_start = tile_col * int(metadata.tile_width)
        window_height = min(
            int(metadata.tile_height), int(metadata.height) - row_start
        )
        window_width = min(
            int(metadata.tile_width), int(metadata.width) - col_start
        )
        return row_start, col_start, window_height, window_width

    async def _sample_with_reader(shared_reader: COGReader) -> pa.Table:
        record_batches: list[pa.RecordBatch] = []
        grouped_sources: dict[object, list[dict[str, object]]] = {}
        for source in band_sources:
            grouped_sources.setdefault(source["group_key"], []).append(source)

        for source_group in grouped_sources.values():
            first_source = source_group[0]
            cog_meta = first_source["metadata"]
            url = str(first_source["url"])
            src_transform = first_source["transform"]

            tile_groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            for point_index in range(len(point_xs)):
                sample_x = float(sample_xs[point_index])
                sample_y = float(sample_ys[point_index])
                col_f, row_f = (~src_transform) * (sample_x, sample_y)
                col = int(np.floor(col_f))
                row = int(np.floor(row_f))
                if (
                    row < 0
                    or col < 0
                    or row >= int(cog_meta.height)
                    or col >= int(cog_meta.width)
                ):
                    continue

                tile_row = row // int(cog_meta.tile_height)
                tile_col = col // int(cog_meta.tile_width)
                tile_groups.setdefault((tile_row, tile_col), []).append(
                    (point_index, row, col)
                )

            tiles = list(tile_groups.keys())

            tile_batch_size = 128
            for start in range(0, len(tiles), tile_batch_size):
                batch = tiles[start : start + tile_batch_size]
                sample_requests: list[COGTileRequest] = []
                for tile_row, tile_col in batch:
                    for source in source_group:
                        source_meta = source["metadata"]
                        source_band_index = source["band_index"]

                        tiles_x = (
                            int(source_meta.width) + int(source_meta.tile_width) - 1
                        ) // int(source_meta.tile_width)
                        tiles_y = (
                            int(source_meta.height)
                            + int(source_meta.tile_height)
                            - 1
                        ) // int(source_meta.tile_height)
                        if (
                            tile_row < 0
                            or tile_col < 0
                            or tile_row >= tiles_y
                            or tile_col >= tiles_x
                        ):
                            continue

                        tile_idx = tile_row * tiles_x + tile_col
                        if tile_idx >= len(source_meta.tile_offsets):
                            continue

                        sample_requests.append(
                            COGTileRequest(
                                url=url,
                                offset=int(source_meta.tile_offsets[tile_idx]),
                                size=int(source_meta.tile_byte_counts[tile_idx]),
                                row=tile_row,
                                col=tile_col,
                                metadata=source_meta,
                                band_index=source_band_index,
                            )
                        )

                tile_arrays_map = (
                    await shared_reader.read_merged_tiles(sample_requests)
                    if sample_requests
                    else {}
                )

                for tile_row, tile_col in batch:
                    tile_arrays = []
                    missing_band = False
                    for source in source_group:
                        band_index = source["band_index"]
                        tile_data = tile_arrays_map.get(
                            (tile_row, tile_col, band_index)
                        )
                        if tile_data is None:
                            missing_band = True
                            break
                        tile_arrays.append(tile_data)
                    if missing_band or not tile_arrays:
                        continue

                    samples = tile_groups.get((tile_row, tile_col), [])
                    if not samples:
                        continue

                    row_start, col_start, window_height, window_width = (
                        _tile_window(
                            metadata=cog_meta,
                            tile_row=tile_row,
                            tile_col=tile_col,
                        )
                    )

                    sample_matrix = np.asarray(samples, dtype=np.int64)
                    if sample_matrix.size == 0:
                        continue

                    local_point_indices = sample_matrix[:, 0]
                    local_rows = sample_matrix[:, 1] - row_start
                    local_cols = sample_matrix[:, 2] - col_start
                    in_window = (
                        (local_rows >= 0)
                        & (local_cols >= 0)
                        & (local_rows < window_height)
                        & (local_cols < window_width)
                    )
                    if not np.any(in_window):
                        continue

                    local_point_indices = local_point_indices[in_window]
                    local_rows = local_rows[in_window]
                    local_cols = local_cols[in_window]

                    output_point_indices = (
                        point_indices_arr[local_point_indices]
                        if point_indices_arr is not None
                        else local_point_indices
                    )
                    point_x_values = point_xs[local_point_indices]
                    point_y_values = point_ys[local_point_indices]
                    row_count = int(local_point_indices.size)
                    point_index_arr = pa.array(
                        output_point_indices, type=pa.int64()
                    )
                    point_x_arr = pa.array(
                        np.asarray(point_x_values, dtype=np.float64),
                        type=pa.float64(),
                    )
                    point_y_arr = pa.array(
                        np.asarray(point_y_values, dtype=np.float64),
                        type=pa.float64(),
                    )
                    point_crs_arr = _constant_int32_array(
                        point_crs_value, row_count
                    )
                    record_id_arr = _constant_string_array(str(self.id), row_count)
                    datetime_arr = _constant_timestamp_array(
                        record_datetime_us, row_count
                    )
                    collection_arr = _constant_string_array(
                        str(self.collection), row_count
                    )
                    cloud_cover_arr = _constant_float64_array(
                        cloud_cover_value, row_count
                    )
                    raster_crs_arr = _constant_int32_array(
                        raster_crs_value, row_count
                    )

                    if len(tile_arrays) != len(source_group):
                        raise RuntimeError(
                            "Internal point-sampling mismatch: tile arrays "
                            f"({len(tile_arrays)}) do not match source bands "
                            f"({len(source_group)}) for record '{self.id}'."
                        )

                    for source, tile_data in zip(
                        source_group, tile_arrays, strict=True
                    ):
                        values = np.asarray(
                            tile_data[local_rows, local_cols],
                            dtype=np.float64,
                        )
                        batch_columns = {
                            "point_index": point_index_arr,
                            "point_x": point_x_arr,
                            "point_y": point_y_arr,
                            "point_crs": point_crs_arr,
                            "record_id": record_id_arr,
                            "datetime": datetime_arr,
                            "collection": collection_arr,
                            "cloud_cover": cloud_cover_arr,
                            "band": _constant_string_array(
                                str(source["band_code"]), row_count
                            ),
                            "value": pa.array(values, type=pa.float64()),
                            "raster_crs": raster_crs_arr,
                        }
                        record_batches.append(
                            pa.record_batch(
                                [
                                    batch_columns[field.name]
                                    for field in POINT_SAMPLES_SCHEMA
                                ],
                                schema=POINT_SAMPLES_SCHEMA,
                            )
                        )

        if not record_batches:
            return empty_point_samples_table()
        return pa.Table.from_batches(record_batches, schema=POINT_SAMPLES_SCHEMA)

    if reader is not None:
        return await _sample_with_reader(reader)

    async with COGReader(max_concurrent=max_concurrent, backend=backend) as owned:
        return await _sample_with_reader(owned)
```

###### load_bands

```python
load_bands(
    geometries: Array,
    band_codes: list[str],
    max_concurrent: int = 50,
    for_xarray: bool = True,
    for_numpy: bool = False,
    progress: bool = False,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
)
```

Load bands for all geometries with parallel processing.

Parameters:

| Name             | Type          | Description                                                                                              | Default    |
| ---------------- | ------------- | -------------------------------------------------------------------------------------------------------- | ---------- |
| `geometries`     | `Array`       | GeoArrow native array of areas of interest.                                                              | *required* |
| `band_codes`     | `list of str` | Band codes to load.                                                                                      | *required* |
| `max_concurrent` | `int`         | Maximum concurrent HTTP requests.                                                                        | `50`       |
| `for_xarray`     | `bool`        | If True, return xr.Dataset; otherwise gpd.GeoDataFrame.                                                  | `True`     |
| `for_numpy`      | `bool`        | If True, return raw per-geometry band results for NumPy assembly without constructing GeoPandas objects. | `False`    |
| `backend`        | `object`      | Pluggable I/O backend.                                                                                   | `None`     |
| `target_crs`     | `int`         | Reproject results to this CRS.                                                                           | `None`     |
| `geometry_crs`   | `int`         | CRS of the geometries input (default EPSG:4326).                                                         | `4326`     |
| `all_touched`    | `bool`        | Passed through to polygon masking behavior. False matches rasterio default semantics.                    | `False`    |

Returns:

| Type                      | Description                                                                                                                                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dataset or GeoDataFrame` | Data is returned in the native COG dtype (e.g. uint16, int8, float32). Integer arrays promote to float32 only when geometry masking requires NaN and no nodata value is declared in the COG metadata. |

Source code in `src/rasteret/core/raster_accessor.py`

```python
async def load_bands(
    self,
    geometries: pa.Array,
    band_codes: list[str],
    max_concurrent: int = 50,
    for_xarray: bool = True,
    for_numpy: bool = False,
    progress: bool = False,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
):
    """Load bands for all geometries with parallel processing.

    Parameters
    ----------
    geometries : pa.Array
        GeoArrow native array of areas of interest.
    band_codes : list of str
        Band codes to load.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    for_xarray : bool
        If ``True``, return ``xr.Dataset``; otherwise ``gpd.GeoDataFrame``.
    for_numpy : bool
        If ``True``, return raw per-geometry band results for NumPy assembly
        without constructing GeoPandas objects.
    backend : object, optional
        Pluggable I/O backend.
    target_crs : int, optional
        Reproject results to this CRS.
    geometry_crs : int, optional
        CRS of the *geometries* input (default EPSG:4326).
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.

    Returns
    -------
    xarray.Dataset or geopandas.GeoDataFrame
        Data is returned in the native COG dtype (e.g. ``uint16``,
        ``int8``, ``float32``). Integer arrays promote to ``float32``
        only when geometry masking requires NaN and no nodata value is
        declared in the COG metadata.
    """
    from rasteret.fetch.cog import COGReader

    if for_xarray and for_numpy:
        raise ValueError(
            "load_bands() cannot request xarray and numpy outputs together"
        )

    n_geoms = len(geometries)
    logger.debug(f"Loading {len(band_codes)} bands for {n_geoms} geometries")

    geom_progress = None
    if progress:
        geom_progress = tqdm(total=n_geoms, desc=f"Record {self.id}")

    async with COGReader(max_concurrent=max_concurrent, backend=backend) as reader:

        async def process_geometry(geom_idx: int, geom_id: int):
            band_progress = None
            if progress:
                band_progress = tqdm(
                    total=len(band_codes), desc=f"Geom {geom_id}", leave=False
                )

            band_tasks = []
            for band_code in band_codes:
                task = self._load_single_band(
                    geometries,
                    geom_idx,
                    band_code,
                    max_concurrent,
                    reader=reader,
                    geometry_crs=geometry_crs,
                    all_touched=all_touched,
                )
                band_tasks.append(task)

            raw_results = await asyncio.gather(*band_tasks, return_exceptions=True)
            results = []
            first_error: BaseException | None = None
            failed_band_codes: list[str] = []
            for band_code, r in zip(band_codes, raw_results):
                if isinstance(r, Exception):
                    from rasteret.core.geometry import UnsupportedGeometryError

                    if isinstance(r, UnsupportedGeometryError):
                        # Deterministic user input issue: fail fast.
                        raise UnsupportedGeometryError(
                            "Unsupported geometry type for Rasteret sampling "
                            f"(record_id='{self.id}', geometry_index={geom_id}). "
                            "Rasteret currently supports Polygon and MultiPolygon geometries "
                            "for masking-based sampling via get_xarray/get_numpy/get_gdf. "
                            "Point sampling is not supported yet."
                        ) from r
                    failed_band_codes.append(band_code)
                    if first_error is None:
                        first_error = r
                    logger.error(
                        "Band load failed (record_id=%s, geometry_index=%s, band=%s): %s",
                        self.id,
                        geom_id,
                        band_code,
                        r,
                    )
                else:
                    results.append(r)
            if band_progress is not None:
                band_progress.update(len(band_codes))
                band_progress.close()
            if geom_progress is not None:
                geom_progress.update(1)

            valid = [r for r in results if r is not None]
            if not valid and first_error is not None:
                from rasteret.core.geometry import UnsupportedGeometryError

                if isinstance(first_error, UnsupportedGeometryError):
                    raise UnsupportedGeometryError(
                        f"Unsupported geometry type for Rasteret sampling "
                        f"(record_id='{self.id}', geometry_index={geom_id}). "
                        "Rasteret currently supports Polygon and MultiPolygon geometries "
                        "for masking-based sampling via get_xarray/get_numpy/get_gdf. "
                        "Point sampling is not supported yet."
                    ) from first_error
                raise RuntimeError(
                    "All band reads failed for the requested geometry "
                    f"(record_id='{self.id}', geometry_index={geom_id}). "
                    "See the chained exception for the first failure."
                ) from first_error
            if target_crs is not None and target_crs != self.crs and valid:
                valid = self._reproject_band_results(valid, target_crs)
            return valid, geom_id, failed_band_codes, first_error

        # Process geometries concurrently with semaphore
        sem = asyncio.Semaphore(max_concurrent)

        async def bounded_process(geom_idx: int, geom_id: int):
            async with sem:
                return await process_geometry(geom_idx, geom_id)

        tasks = [bounded_process(idx, idx + 1) for idx in range(n_geoms)]
        raw_geom_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[tuple[list[dict], int]] = []
    first_error: BaseException | None = None
    geom_failures = 0
    partial_band_failures: list[tuple[int, list[str], BaseException | None]] = []
    for r in raw_geom_results:
        if isinstance(r, Exception):
            from rasteret.core.geometry import UnsupportedGeometryError

            if isinstance(r, UnsupportedGeometryError):
                # Geometry-type errors are deterministic user input issues.
                # Fail fast so they do not become a misleading
                # "No valid data found" downstream.
                raise r
            geom_failures += 1
            if first_error is None:
                first_error = r
            logger.error("Geometry processing failed: %s", r)
        else:
            band_results, geom_id, failed_band_codes, band_first_error = r
            results.append((band_results, geom_id))
            if failed_band_codes:
                partial_band_failures.append(
                    (geom_id, failed_band_codes, band_first_error)
                )

    if geom_progress is not None:
        geom_progress.close()

    if not results and first_error is not None:
        raise RuntimeError(
            f"All geometry reads failed for record_id='{self.id}'. "
            "See the chained exception for the first failure."
        ) from first_error
    if results and (geom_failures or partial_band_failures):
        parts = []
        if geom_failures:
            parts.append(f"{geom_failures}/{n_geoms} geometry task(s) failed")
        if partial_band_failures:
            n_failed_bands = sum(
                len(bands) for _gid, bands, _err in partial_band_failures
            )
            parts.append(
                f"{n_failed_bands} band read(s) failed across {len(partial_band_failures)} geometries"
            )

        first_detail = None
        if geom_failures and first_error is not None:
            first_detail = f"first geometry failure: {first_error}"
        elif partial_band_failures:
            gid, bands, err = partial_band_failures[0]
            band = bands[0] if bands else "<unknown>"
            if err is not None:
                first_detail = f"first band failure: geometry_index={gid}, band='{band}': {err}"
            else:
                first_detail = (
                    f"first band failure: geometry_index={gid}, band='{band}'"
                )

        msg = (
            f"Partial read failures for record_id='{self.id}': "
            + "; ".join(parts)
            + (f"; {first_detail}" if first_detail else "")
            + "."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # Process results
    if for_numpy:
        return results
    if for_xarray:
        return self._merge_xarray_results(results, target_crs=target_crs)
    else:
        return self._merge_geodataframe_results(
            results,
            geometries,
            geometry_crs=geometry_crs,
            target_crs=target_crs,
        )
```

### Functions
