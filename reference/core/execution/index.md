# rasteret.core.execution

Data loading pipeline: iterate collection records, fetch tiles, and merge results into NumPy, xarray, or GeoDataFrame outputs.

This module powers `Collection.get_numpy(...)`, `Collection.get_xarray(...)`, `Collection.get_gdf(...)`, and `Collection.sample_points(...)`.

`bands` is required because each raster in the collection may contain dozens of assets (bands). Specifying which bands to load avoids fetching unnecessary data and keeps memory usage predictable.

## execution

Data loading pipeline for Collection reads.

This module orchestrates the read path:

1. Iterate records in the Collection
1. Load bands concurrently via COGReader
1. Merge results into xarray.Dataset or geopandas.GeoDataFrame

Users access this via `Collection.get_xarray()`, `Collection.get_gdf()`, and `Collection.get_numpy()`.

### Classes

### Functions

#### get_collection_xarray

```python
get_collection_xarray(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    geometry_column: str | None = None,
    all_touched: bool = False,
    progress: bool = False,
    xr_combine: str = "combine_first",
    **filters: Any,
) -> Dataset
```

Load selected bands as an `xarray.Dataset`.

Parameters:

| Name              | Type                                                        | Description                                                                                                                                                                                                                                                                                                            | Default           |
| ----------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `collection`      | `Collection`                                                | Source collection.                                                                                                                                                                                                                                                                                                     | *required*        |
| `geometries`      | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Area(s) of interest.                                                                                                                                                                                                                                                                                                   | *required*        |
| `bands`           | `list of str`                                               | Band codes to load (e.g. ["B04", "B08"]).                                                                                                                                                                                                                                                                              | *required*        |
| `data_source`     | `str`                                                       | Override the inferred data source for band mapping and URL signing.                                                                                                                                                                                                                                                    | `None`            |
| `max_concurrent`  | `int`                                                       | Maximum concurrent HTTP requests (default 50).                                                                                                                                                                                                                                                                         | `50`              |
| `backend`         | `StorageBackend`                                            | Pluggable I/O backend (e.g. ObstoreBackend).                                                                                                                                                                                                                                                                           | `None`            |
| `target_crs`      | `int`                                                       | Reproject all records to this EPSG code before merging. When None and the collection spans multiple CRS zones, auto-reprojection to the most common CRS is triggered.                                                                                                                                                  | `None`            |
| `geometry_column` | `str`                                                       | Geometry column to read when geometries is a tabular AOI input.                                                                                                                                                                                                                                                        | `None`            |
| `all_touched`     | `bool`                                                      | Passed through to polygon masking behavior. False matches rasterio default semantics.                                                                                                                                                                                                                                  | `False`           |
| `xr_combine`      | `str`                                                       | Strategy for merging per-record xarray Datasets. "combine_first" (default) preserves all data and fills NaN gaps from subsequent records. "merge" uses xr.merge(join="outer") which raises on value conflicts. "merge_override" uses xr.merge(compat="override") which silently picks one record's values in overlaps. | `'combine_first'` |
| `filters`         | `kwargs`                                                    | Additional keyword arguments forwarded to Collection.subset().                                                                                                                                                                                                                                                         | `{}`              |

Returns:

| Type      | Description                                                                                                                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dataset` | Band arrays in native COG dtype (e.g. uint16 for Sentinel-2). CRS is encoded via CF conventions (spatial_ref coordinate with WKT2, PROJJSON, and GeoTransform). Multi-CRS queries are auto-reprojected. |

Examples:

```pycon
>>> ds = get_collection_xarray(
...     collection=col,
...     geometries=(77.55, 13.01, 77.58, 13.08),
...     bands=["B04", "B08"],
... )
>>> ds.B04.dtype
dtype('uint16')
```

Source code in `src/rasteret/core/execution.py`

```python
def get_collection_xarray(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    geometry_column: str | None = None,
    all_touched: bool = False,
    progress: bool = False,
    xr_combine: str = "combine_first",
    **filters: Any,
) -> xr.Dataset:
    """Load selected bands as an ``xarray.Dataset``.

    Parameters
    ----------
    collection : Collection
        Source collection.
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
        Area(s) of interest.
    bands : list of str
        Band codes to load (e.g. ``["B04", "B08"]``).
    data_source : str, optional
        Override the inferred data source for band mapping and URL signing.
    max_concurrent : int
        Maximum concurrent HTTP requests (default 50).
    backend : StorageBackend, optional
        Pluggable I/O backend (e.g. ``ObstoreBackend``).
    target_crs : int, optional
        Reproject all records to this EPSG code before merging. When
        ``None`` and the collection spans multiple CRS zones,
        auto-reprojection to the most common CRS is triggered.
    geometry_column : str, optional
        Geometry column to read when ``geometries`` is a tabular AOI input.
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
    xr_combine : str
        Strategy for merging per-record xarray Datasets.
        ``"combine_first"`` (default) preserves all data and fills
        NaN gaps from subsequent records. ``"merge"`` uses
        ``xr.merge(join="outer")`` which raises on value conflicts.
        ``"merge_override"`` uses ``xr.merge(compat="override")``
        which silently picks one record's values in overlaps.
    filters : kwargs
        Additional keyword arguments forwarded to ``Collection.subset()``.

    Returns
    -------
    xarray.Dataset
        Band arrays in native COG dtype (e.g. ``uint16`` for Sentinel-2).
        CRS is encoded via CF conventions (``spatial_ref`` coordinate with
        WKT2, PROJJSON, and GeoTransform). Multi-CRS queries are
        auto-reprojected.

    Examples
    --------
    >>> ds = get_collection_xarray(
    ...     collection=col,
    ...     geometries=(77.55, 13.01, 77.58, 13.08),
    ...     bands=["B04", "B08"],
    ... )
    >>> ds.B04.dtype
    dtype('uint16')
    """
    import xarray as xr

    def _merge(datasets, _aoi_metadata=None):
        logger.info("Merging %s datasets", len(datasets))
        if xr_combine == "combine_first":
            merged = _combine_first_int_with_fill(datasets)
            if merged is None:
                from functools import reduce

                merged = reduce(lambda a, b: a.combine_first(b), datasets)
        elif xr_combine == "merge_override":
            merged = xr.merge(datasets, join="outer", compat="override")
        elif xr_combine == "merge":
            merged = xr.merge(datasets, join="outer")
        else:
            raise ValueError(
                f"Unknown xr_combine strategy {xr_combine!r}. "
                "Use 'combine_first', 'merge', or 'merge_override'."
            )
        if "time" in merged.coords:
            merged = merged.sortby("time")
        return _normalize_spatial_y_axis_order(merged)

    return _load_and_merge(
        collection=collection,
        geometries=geometries,
        bands=bands,
        for_xarray=True,
        for_numpy=False,
        merge_fn=_merge,
        data_source=data_source,
        max_concurrent=max_concurrent,
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        geometry_column=geometry_column,
        all_touched=all_touched,
        progress=bool(progress),
        **filters,
    )
```

#### get_collection_gdf

```python
get_collection_gdf(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    geometry_column: str | None = None,
    all_touched: bool = False,
    progress: bool = False,
    **filters: Any,
) -> GeoDataFrame
```

Load selected bands as a `geopandas.GeoDataFrame`.

Parameters:

| Name              | Type                                                        | Description                                                                                                                            | Default    |
| ----------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `collection`      | `Collection`                                                | Source collection.                                                                                                                     | *required* |
| `geometries`      | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Area(s) of interest.                                                                                                                   | *required* |
| `bands`           | `list of str`                                               | Band codes to load.                                                                                                                    | *required* |
| `data_source`     | `str`                                                       | Override the inferred data source.                                                                                                     | `None`     |
| `max_concurrent`  | `int`                                                       | Maximum concurrent HTTP requests (default 50).                                                                                         | `50`       |
| `backend`         | `StorageBackend`                                            | Pluggable I/O backend.                                                                                                                 | `None`     |
| `target_crs`      | `int`                                                       | Reproject all records to this EPSG code before building the GeoDataFrame.                                                              | `None`     |
| `geometry_column` | `str`                                                       | Geometry column to read when geometries is a tabular AOI input. Non-geometry AOI columns are joined back to the output by geometry_id. | `None`     |
| `all_touched`     | `bool`                                                      | Passed through to polygon masking behavior. False matches rasterio default semantics.                                                  | `False`    |
| `filters`         | `kwargs`                                                    | Additional keyword arguments forwarded to Collection.subset().                                                                         | `{}`       |

Returns:

| Type           | Description                                                                                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `GeoDataFrame` | Band arrays in native COG dtype. Each row is a geometry-record pair with pixel data and the read-window transform as columns. |

Source code in `src/rasteret/core/execution.py`

```python
def get_collection_gdf(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    geometry_column: str | None = None,
    all_touched: bool = False,
    progress: bool = False,
    **filters: Any,
) -> gpd.GeoDataFrame:
    """Load selected bands as a ``geopandas.GeoDataFrame``.

    Parameters
    ----------
    collection : Collection
        Source collection.
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
        Area(s) of interest.
    bands : list of str
        Band codes to load.
    data_source : str, optional
        Override the inferred data source.
    max_concurrent : int
        Maximum concurrent HTTP requests (default 50).
    backend : StorageBackend, optional
        Pluggable I/O backend.
    target_crs : int, optional
        Reproject all records to this EPSG code before building the
        GeoDataFrame.
    geometry_column : str, optional
        Geometry column to read when ``geometries`` is a tabular AOI input.
        Non-geometry AOI columns are joined back to the output by
        ``geometry_id``.
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
    filters : kwargs
        Additional keyword arguments forwarded to ``Collection.subset()``.

    Returns
    -------
    geopandas.GeoDataFrame
        Band arrays in native COG dtype. Each row is a geometry-record
        pair with pixel data and the read-window transform as columns.
    """

    def _merge_gdfs(
        dfs: list[gpd.GeoDataFrame], aoi_metadata: pa.Table | None = None
    ) -> gpd.GeoDataFrame:
        merged = pd.concat(dfs, ignore_index=True)
        if aoi_metadata is not None and "geometry_id" in merged.columns:
            fail_on_metadata_collisions(
                aoi_metadata,
                output_columns=set(str(col) for col in merged.columns),
                join_column="geometry_id",
            )
            merged = merged.merge(
                aoi_metadata.to_pandas(),
                on="geometry_id",
                how="left",
            )
        gdf = gpd.GeoDataFrame(merged, geometry="geometry")
        crs = next(
            (getattr(df, "crs", None) for df in dfs if getattr(df, "crs", None)), None
        )
        if crs is not None:
            gdf = gdf.set_crs(crs, allow_override=True)
        return gdf

    return _load_and_merge(
        collection=collection,
        geometries=geometries,
        bands=bands,
        for_xarray=False,
        for_numpy=False,
        merge_fn=_merge_gdfs,
        data_source=data_source,
        max_concurrent=max_concurrent,
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        geometry_column=geometry_column,
        all_touched=all_touched,
        progress=bool(progress),
        **filters,
    )
```

#### get_collection_numpy

```python
get_collection_numpy(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    progress: bool = False,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    geometry_column: str | None = None,
    all_touched: bool = False,
    **filters: Any,
)
```

Load selected bands as NumPy arrays without xarray merge overhead.

Parameters:

| Name              | Type                                                        | Description                                                                           | Default    |
| ----------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------- |
| `collection`      | `Collection`                                                | Source collection.                                                                    | *required* |
| `geometries`      | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Area(s) of interest.                                                                  | *required* |
| `bands`           | `list of str`                                               | Band codes to load.                                                                   | *required* |
| `data_source`     | `str`                                                       | Override the inferred data source.                                                    | `None`     |
| `max_concurrent`  | `int`                                                       | Maximum concurrent HTTP requests.                                                     | `50`       |
| `backend`         | `StorageBackend`                                            | Pluggable I/O backend.                                                                | `None`     |
| `target_crs`      | `int`                                                       | Reproject all records to this CRS before assembling arrays.                           | `None`     |
| `geometry_column` | `str`                                                       | Geometry column to read when geometries is a tabular AOI input.                       | `None`     |
| `all_touched`     | `bool`                                                      | Passed through to polygon masking behavior. False matches rasterio default semantics. | `False`    |
| `filters`         | `kwargs`                                                    | Additional keyword arguments forwarded to Collection.subset().                        | `{}`       |

Returns:

| Type      | Description                                                                                           |
| --------- | ----------------------------------------------------------------------------------------------------- |
| `ndarray` | Single-band queries return [N, H, W]. Multi-band queries return [N, C, H, W] in requested band order. |

Notes

All selected samples must resolve to a consistent shape per band. A `ValueError` is raised for ragged outputs.

Source code in `src/rasteret/core/execution.py`

```python
def get_collection_numpy(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    progress: bool = False,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: GeometryCrsInput = AUTO_CRS,
    geometry_column: str | None = None,
    all_touched: bool = False,
    **filters: Any,
):
    """Load selected bands as NumPy arrays without xarray merge overhead.

    Parameters
    ----------
    collection : Collection
        Source collection.
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
        Area(s) of interest.
    bands : list of str
        Band codes to load.
    data_source : str, optional
        Override the inferred data source.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    backend : StorageBackend, optional
        Pluggable I/O backend.
    target_crs : int, optional
        Reproject all records to this CRS before assembling arrays.
    geometry_column : str, optional
        Geometry column to read when ``geometries`` is a tabular AOI input.
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
    filters : kwargs
        Additional keyword arguments forwarded to ``Collection.subset()``.

    Returns
    -------
    numpy.ndarray
        Single-band queries return ``[N, H, W]``.
        Multi-band queries return ``[N, C, H, W]`` in requested band order.

    Notes
    -----
    All selected samples must resolve to a consistent shape per band.
    A ``ValueError`` is raised for ragged outputs.
    """
    import numpy as np

    def _merge_numpy(frames: list[list[tuple[list[dict], int]]], _aoi_metadata=None):
        if not frames:
            raise ValueError("No valid data found")

        expected_bands = set(bands)
        per_band_arrays: dict[str, list[np.ndarray]] = {band: [] for band in bands}

        for frame in frames:
            if frame is None:
                continue
            if hasattr(frame, "empty"):
                if frame.empty:
                    continue
                if "band" not in frame.columns or "data" not in frame.columns:
                    raise ValueError(
                        "Cannot assemble numpy output: missing 'band'/'data' columns."
                    )
                band_values = frame["band"].to_numpy()
                data_values = frame["data"].to_numpy()
                for band_value, data_value in zip(
                    band_values, data_values, strict=False
                ):
                    band_name = str(band_value)
                    if band_name in expected_bands:
                        per_band_arrays[band_name].append(data_value)
                continue
            if len(frame) == 0:
                continue
            for band_results, _geom_id in frame:
                for band_result in band_results:
                    band_name = str(band_result["band"])
                    if band_name in expected_bands:
                        per_band_arrays[band_name].append(band_result["data"])

        if not any(per_band_arrays.values()):
            raise ValueError("No valid data found")

        per_band: list[np.ndarray] = []
        sample_count: int | None = None

        for band in bands:
            arrays = per_band_arrays[band]
            if not arrays:
                raise ValueError(f"No data resolved for band '{band}'.")

            if sample_count is None:
                sample_count = len(arrays)
            elif len(arrays) != sample_count:
                raise ValueError(
                    f"Inconsistent sample count for band '{band}': "
                    f"expected {sample_count}, got {len(arrays)}."
                )

            shapes = {tuple(a.shape) for a in arrays}
            if len(shapes) != 1:
                raise ValueError(
                    f"Ragged shapes for band '{band}': {sorted(shapes)}. "
                    "Use get_gdf() when variable output shapes are expected."
                )
            per_band.append(np.stack(arrays, axis=0))

        if len(per_band) == 1:
            return per_band[0]

        reference_shape = per_band[0].shape
        for band, arr in zip(bands[1:], per_band[1:], strict=False):
            if arr.shape != reference_shape:
                raise ValueError(
                    f"Band '{band}' shape {arr.shape} does not match "
                    f"reference shape {reference_shape}. "
                    "Use get_gdf() or request shape-compatible bands."
                )

        return np.stack(per_band, axis=1)

    return _load_and_merge(
        collection=collection,
        geometries=geometries,
        bands=bands,
        for_xarray=False,
        for_numpy=True,
        merge_fn=_merge_numpy,
        data_source=data_source,
        max_concurrent=max_concurrent,
        progress=bool(progress),
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        geometry_column=geometry_column,
        all_touched=all_touched,
        **filters,
    )
```
