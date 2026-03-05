# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Data loading pipeline for Collection reads.

This module orchestrates the read path:

1. Iterate records in the Collection
2. Load bands concurrently via COGReader
3. Merge results into xarray.Dataset or geopandas.GeoDataFrame

Users access this via ``Collection.get_xarray()``, ``Collection.get_gdf()``,
and ``Collection.get_numpy()``.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import pandas as pd
import pyarrow as pa
from tqdm.asyncio import tqdm

from rasteret.core.utils import infer_data_source, run_sync

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

    from rasteret.core.collection import Collection

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Loading pipeline
# ------------------------------------------------------------------


def _ensure_geoarrow(geometries: Any) -> pa.Array:
    """Coerce any geometry input to a GeoArrow native array."""
    from rasteret.core.geometry import coerce_to_geoarrow

    return coerce_to_geoarrow(geometries)


def _resolve_xy_columns(
    names: list[str],
    x_column: str | None,
    y_column: str | None,
) -> tuple[str, str] | None:
    """Resolve x/y columns from explicit names or common defaults."""
    if x_column and y_column:
        if x_column in names and y_column in names:
            return x_column, y_column
        missing = [col for col in (x_column, y_column) if col not in names]
        raise ValueError(f"Missing point coordinate columns: {missing}")

    candidates = [
        ("x", "y"),
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("lng", "lat"),
    ]
    for x_name, y_name in candidates:
        if x_name in names and y_name in names:
            return x_name, y_name
    return None


def _ensure_point_geoarrow(
    points: Any,
    *,
    geometry_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
) -> pa.Array:
    """Normalize point inputs into a GeoArrow point array.

    Supports Arrow arrays, Arrow tables, pandas/GeoPandas/Polars DataFrames,
    and DuckDB/SedonaDB relations when they can export Arrow.
    """
    import geoarrow.pyarrow as ga

    if isinstance(points, pa.RecordBatchReader):
        points = points.read_all()

    if isinstance(points, pa.Table):
        names = points.schema.names
        if geometry_column is not None:
            if geometry_column not in names:
                raise ValueError(
                    f"geometry_column='{geometry_column}' not found in table columns"
                )
            return _ensure_geoarrow(points.column(geometry_column))
        xy = _resolve_xy_columns(names, x_column, y_column)
        if xy is not None:
            x_name, y_name = xy
            return ga.make_point(points.column(x_name), points.column(y_name))
        raise TypeError(
            "Unsupported table input for point sampling. Provide geometry_column "
            "(WKB/GeoArrow point column) or x_column/y_column."
        )

    to_arrow_table = getattr(points, "to_arrow_table", None)
    if callable(to_arrow_table):
        return _ensure_point_geoarrow(
            to_arrow_table(),
            geometry_column=geometry_column,
            x_column=x_column,
            y_column=y_column,
        )

    arrow_export = getattr(points, "arrow", None)
    if callable(arrow_export):
        try:
            return _ensure_point_geoarrow(
                arrow_export(),
                geometry_column=geometry_column,
                x_column=x_column,
                y_column=y_column,
            )
        except Exception:
            # Fall through to other handlers if this object has a non-standard
            # arrow() method.
            pass

    to_arrow = getattr(points, "to_arrow", None)
    if callable(to_arrow):
        try:
            return _ensure_point_geoarrow(
                to_arrow(),
                geometry_column=geometry_column,
                x_column=x_column,
                y_column=y_column,
            )
        except Exception:
            pass

    # Optional DataFrame handlers (import lazily).
    try:
        import polars as pl

        if isinstance(points, pl.DataFrame):
            names = points.columns
            if geometry_column is not None:
                if geometry_column not in names:
                    raise ValueError(
                        f"geometry_column='{geometry_column}' not found in DataFrame"
                    )
                return _ensure_geoarrow(points[geometry_column].to_arrow())
            xy = _resolve_xy_columns(names, x_column, y_column)
            if xy is not None:
                x_name, y_name = xy
                return ga.make_point(
                    points[x_name].to_arrow(),
                    points[y_name].to_arrow(),
                )
            raise TypeError(
                "Unsupported Polars DataFrame input for point sampling. Provide "
                "geometry_column or x_column/y_column."
            )
    except ImportError:
        pass

    if isinstance(points, pd.DataFrame):
        names = list(points.columns)
        if geometry_column is not None:
            if geometry_column not in names:
                raise ValueError(
                    f"geometry_column='{geometry_column}' not found in DataFrame"
                )
            return _ensure_geoarrow(points[geometry_column].tolist())

        # GeoPandas geometry convenience
        if (
            hasattr(points, "geometry")
            and getattr(points, "geometry", None) is not None
        ):
            try:
                geom_series = points.geometry
                if len(geom_series) > 0:
                    return _ensure_geoarrow(geom_series.tolist())
            except Exception:
                pass

        xy = _resolve_xy_columns(names, x_column, y_column)
        if xy is not None:
            x_name, y_name = xy
            return ga.make_point(
                pa.array(points[x_name].to_numpy()),
                pa.array(points[y_name].to_numpy()),
            )
        raise TypeError(
            "Unsupported pandas/GeoPandas DataFrame input for point sampling. "
            "Provide geometry_column or x_column/y_column."
        )

    return _ensure_geoarrow(points)


async def _load_collection_data(
    *,
    collection: "Collection",
    data_source: str,
    geometries: pa.Array,
    bands: list[str],
    max_concurrent: int,
    for_xarray: bool,
    batch_size: int = 10,
    progress: bool = False,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    **filters: Any,
) -> tuple[list[gpd.GeoDataFrame] | list, list[tuple[str, Exception]]]:
    """Core loading loop: iterate records, fetch tiles."""
    selected_collection = collection.subset(**filters) if filters else collection

    results = []
    errors: list[tuple[str, Exception]] = []
    raster_batches = []
    current_batch = []

    async for raster in selected_collection.iterate_rasters(data_source):
        current_batch.append(raster)

        if len(current_batch) == batch_size:
            raster_batches.append(current_batch)
            current_batch = []

    if current_batch:
        raster_batches.append(current_batch)

    iterable = (
        tqdm(raster_batches, desc="Loading rasters") if progress else raster_batches
    )
    for batch in iterable:
        tasks = [
            raster.load_bands(
                geometries=geometries,
                band_codes=bands,
                max_concurrent=max_concurrent,
                for_xarray=for_xarray,
                progress=progress,
                backend=backend,
                target_crs=target_crs,
                geometry_crs=geometry_crs,
                all_touched=all_touched,
            )
            for raster in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for raster, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                errors.append((getattr(raster, "id", "<unknown>"), result))
                logger.error("Record load failed (id=%s): %s", errors[-1][0], result)
            elif result is not None:
                if (
                    for_xarray
                    and hasattr(result, "data_vars")
                    and len(result.data_vars) == 0
                ):
                    continue
                if not for_xarray and hasattr(result, "empty") and result.empty:
                    continue
                results.append(result)

    return results, errors


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def _load_and_merge(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    for_xarray: bool,
    merge_fn: Callable[[list[Any]], Any],
    data_source: str | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    target_crs: int | None = None,
    progress: bool = False,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    **filters: Any,
):
    """Load collection data and merge via *merge_fn*.

    Parameters
    ----------
    collection : Collection
        Source collection.
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
        Area(s) of interest.
    bands : list of str
        Band codes to load.
    for_xarray : bool
        ``True`` for xarray output, ``False`` for GeoDataFrame.
    merge_fn : callable
        ``merge_fn(results) -> merged_output``.
    data_source, max_concurrent, backend, target_crs, filters
        Forwarded to :func:`_load_collection_data`.
    """

    async def _async_load():
        results, errors = await _load_collection_data(
            collection=collection,
            data_source=data_source or infer_data_source(collection),
            geometries=_ensure_geoarrow(geometries),
            bands=bands,
            max_concurrent=max_concurrent,
            backend=backend,
            for_xarray=for_xarray,
            target_crs=target_crs,
            progress=bool(progress),
            geometry_crs=geometry_crs,
            all_touched=all_touched,
            **filters,
        )
        if errors and results:
            record_id, first = errors[0]
            warnings.warn(
                "Some records failed to load "
                f"({len(errors)} failure(s)); first failure in record '{record_id}': {first}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not results:
            if errors:
                record_id, first = errors[0]
                msg = (
                    "No valid data found. "
                    f"{len(errors)} record(s) failed; first failure in record "
                    f"'{record_id}': {first}"
                )
                raise ValueError(msg) from first
            raise ValueError("No valid data found")
        return merge_fn(results)

    return run_sync(_async_load())


def _detect_target_crs(
    collection: "Collection",
    filters: dict[str, Any],
) -> int | None:
    """Auto-detect multi-CRS and return a target CRS if reprojection is needed.

    When the (filtered) collection contains records from more than one
    ``proj:epsg`` zone, we must reproject to a single CRS before
    ``xr.merge``.  This function picks the most-common CRS and logs a
    warning so the user knows auto-reprojection is happening.

    Returns ``None`` when no reprojection is needed: single CRS,
    ``collection`` is ``None``, the dataset is empty, or the
    ``proj:epsg`` column is absent.
    """
    if collection is None:
        return None
    selected = collection.subset(**filters) if filters else collection
    ds = selected.dataset
    if ds is None or "proj:epsg" not in ds.schema.names:
        return None

    from collections import Counter

    import pyarrow.compute as pc

    counts: Counter[int] = Counter()
    scanner = ds.scanner(columns=["proj:epsg"])
    for batch in scanner.to_batches():
        col = batch.column(batch.schema.get_field_index("proj:epsg"))
        non_null = pc.drop_null(col)
        if len(non_null) == 0:
            continue
        value_counts = pc.value_counts(non_null)
        values = value_counts.field("values")
        freqs = value_counts.field("counts")
        for idx in range(len(value_counts)):
            counts[int(values[idx].as_py())] += int(freqs[idx].as_py())

    unique = sorted(counts.keys())
    if len(unique) <= 1:
        return None
    most_common = counts.most_common(1)[0][0]
    logger.warning(
        "Query spans %d CRS zones %s; auto-reprojecting to EPSG:%d. "
        "Pass target_crs= explicitly to override.",
        len(unique),
        sorted(int(v) for v in unique),
        int(most_common),
    )
    return int(most_common)


def get_collection_xarray(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    progress: bool = False,
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
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
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

    # Auto-detect multi-CRS to prevent silent spatial data corruption
    # from merging tiles with incompatible coordinate systems.
    if target_crs is None:
        target_crs = _detect_target_crs(collection, filters)

    def _merge(datasets):
        logger.info("Merging %s datasets", len(datasets))
        merged = xr.merge(datasets, join="outer", compat="override")
        if "time" in merged.coords:
            return merged.sortby("time")
        return merged

    return _load_and_merge(
        collection=collection,
        geometries=geometries,
        bands=bands,
        for_xarray=True,
        merge_fn=_merge,
        data_source=data_source,
        max_concurrent=max_concurrent,
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        all_touched=all_touched,
        progress=bool(progress),
        **filters,
    )


def get_collection_gdf(
    *,
    collection: "Collection",
    geometries: Any,
    bands: list[str],
    data_source: str | None = None,
    max_concurrent: int = 50,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
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
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
    filters : kwargs
        Additional keyword arguments forwarded to ``Collection.subset()``.

    Returns
    -------
    geopandas.GeoDataFrame
        Band arrays in native COG dtype. Each row is a geometry-record
        pair with pixel data as columns.
    """

    def _merge_gdfs(dfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        merged = pd.concat(dfs, ignore_index=True)
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
        merge_fn=_merge_gdfs,
        data_source=data_source,
        max_concurrent=max_concurrent,
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        all_touched=all_touched,
        progress=bool(progress),
        **filters,
    )


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
    geometry_crs: int | None = 4326,
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

    def _merge_numpy(frames: list[gpd.GeoDataFrame]):
        if not frames:
            raise ValueError("No valid data found")

        expected_bands = set(bands)
        per_band_arrays: dict[str, list[np.ndarray]] = {band: [] for band in bands}

        for frame in frames:
            if frame is None or frame.empty:
                continue
            if "band" not in frame.columns or "data" not in frame.columns:
                raise ValueError(
                    "Cannot assemble numpy output: missing 'band'/'data' columns."
                )
            band_values = frame["band"].to_numpy()
            data_values = frame["data"].to_numpy()
            for band_value, data_value in zip(band_values, data_values, strict=False):
                band_name = str(band_value)
                if band_name in expected_bands:
                    per_band_arrays[band_name].append(data_value)

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
        merge_fn=_merge_numpy,
        data_source=data_source,
        max_concurrent=max_concurrent,
        progress=bool(progress),
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        all_touched=all_touched,
        **filters,
    )


def _empty_point_samples_table() -> pa.Table:
    """Return an empty point-sampling table with stable schema."""
    return pa.table(
        {
            "point_index": pa.array([], type=pa.int64()),
            "point_x": pa.array([], type=pa.float64()),
            "point_y": pa.array([], type=pa.float64()),
            "point_crs": pa.array([], type=pa.int32()),
            "record_id": pa.array([], type=pa.string()),
            "datetime": pa.array([], type=pa.timestamp("us")),
            "collection": pa.array([], type=pa.string()),
            "cloud_cover": pa.array([], type=pa.float64()),
            "band": pa.array([], type=pa.string()),
            "value": pa.array([], type=pa.float64()),
            "raster_crs": pa.array([], type=pa.int32()),
        }
    )


def get_collection_point_samples(
    *,
    collection: "Collection",
    points: Any,
    bands: list[str],
    geometry_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    data_source: str | None = None,
    max_concurrent: int = 50,
    progress: bool = False,
    backend: object | None = None,
    geometry_crs: int | None = 4326,
    match: Literal["all", "latest"] = "all",
    **filters: Any,
) -> pa.Table:
    """Sample point values across matching records.

    Parameters
    ----------
    collection : Collection
        Source collection.
    points : Any
        Point input as Arrow/GeoArrow/WKB/Shapely/GeoJSON, or tabular input
        (Arrow table, pandas/GeoPandas, Polars, DuckDB/SedonaDB relation).
    bands : list of str
        Band codes to sample.
    geometry_column : str, optional
        Geometry column name when *points* is tabular. Column may contain WKB,
        GeoArrow points, or Shapely Point objects.
    x_column, y_column : str, optional
        Coordinate column names when *points* is tabular.
    data_source : str, optional
        Override the inferred data source.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    progress : bool
        Show progress bars when ``True``.
    backend : StorageBackend, optional
        Pluggable I/O backend.
    geometry_crs : int, optional
        CRS EPSG code for input points. Defaults to EPSG:4326.
    match : {"all", "latest"}
        ``"all"`` returns every matching record for each point. ``"latest"``
        returns one row per ``(point_index, band)`` using max datetime.
    filters : kwargs
        Additional keyword arguments forwarded to ``Collection.subset()``.

    Returns
    -------
    pyarrow.Table
        Tabular point samples with one row per sampled value.
    """
    if match not in {"all", "latest"}:
        raise ValueError("match must be either 'all' or 'latest'")

    async def _async_sample() -> pa.Table:
        selected_collection = collection.subset(**filters) if filters else collection
        resolved_source = data_source or infer_data_source(collection)
        point_array = _ensure_point_geoarrow(
            points,
            geometry_column=geometry_column,
            x_column=x_column,
            y_column=y_column,
        )

        rows: list[dict[str, Any]] = []
        errors: list[tuple[str, Exception]] = []

        iterable: Any
        if progress:
            rasters = []
            async for raster in selected_collection.iterate_rasters(resolved_source):
                rasters.append(raster)
            iterable = tqdm(rasters, desc="Sampling points")
        else:
            iterable = selected_collection.iterate_rasters(resolved_source)

        if progress:
            # iterable is a materialized list in progress mode
            for raster in iterable:
                try:
                    sampled = await raster.sample_points(
                        points=point_array,
                        band_codes=bands,
                        max_concurrent=max_concurrent,
                        backend=backend,
                        geometry_crs=geometry_crs,
                    )
                    rows.extend(sampled)
                except Exception as exc:
                    errors.append((getattr(raster, "id", "<unknown>"), exc))
                    logger.error(
                        "Point sampling failed (id=%s): %s",
                        errors[-1][0],
                        exc,
                    )
        else:
            async for raster in iterable:
                try:
                    sampled = await raster.sample_points(
                        points=point_array,
                        band_codes=bands,
                        max_concurrent=max_concurrent,
                        backend=backend,
                        geometry_crs=geometry_crs,
                    )
                    rows.extend(sampled)
                except Exception as exc:
                    errors.append((getattr(raster, "id", "<unknown>"), exc))
                    logger.error(
                        "Point sampling failed (id=%s): %s",
                        errors[-1][0],
                        exc,
                    )

        if errors and rows:
            record_id, first = errors[0]
            warnings.warn(
                "Some records failed during point sampling "
                f"({len(errors)} failure(s)); first failure in record '{record_id}': {first}",
                RuntimeWarning,
                stacklevel=2,
            )

        if not rows:
            if errors:
                record_id, first = errors[0]
                msg = (
                    "No point samples resolved. "
                    f"{len(errors)} record(s) failed; first failure in record "
                    f"'{record_id}': {first}"
                )
                raise ValueError(msg) from first
            return _empty_point_samples_table()

        table = pa.Table.from_pylist(rows)
        if match == "latest":
            frame = table.to_pandas()
            frame = frame.sort_values("datetime", ascending=False, kind="stable")
            frame = frame.drop_duplicates(
                subset=["point_index", "band"],
                keep="first",
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)

        return table

    return run_sync(_async_sample())
