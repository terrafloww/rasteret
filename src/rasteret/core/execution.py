# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Data loading pipeline for Collection reads.

This module orchestrates the read path:

1. Iterate records in the Collection
2. Load bands concurrently via COGReader
3. Merge results into xarray.Dataset or geopandas.GeoDataFrame

Users access this via ``Collection.get_xarray()`` and ``Collection.get_gdf()``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
import pyarrow as pa
from tqdm.asyncio import tqdm

from rasteret.core.utils import infer_data_source, run_sync

if TYPE_CHECKING:  # pragma: no cover
    from rasteret.core.collection import Collection

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Loading pipeline
# ------------------------------------------------------------------


def _ensure_geoarrow(geometries: Any) -> pa.Array:
    """Coerce any geometry input to a GeoArrow native array."""
    from rasteret.core.geometry import coerce_to_geoarrow

    return coerce_to_geoarrow(geometries)


async def _load_collection_data(
    *,
    collection: "Collection",
    data_source: str,
    geometries: pa.Array,
    bands: list[str],
    max_concurrent: int,
    for_xarray: bool,
    batch_size: int = 10,
    backend: object | None = None,
    target_crs: int | None = None,
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

    for batch in tqdm(raster_batches, desc="Loading rasters"):
        tasks = [
            raster.load_bands(
                geometries=geometries,
                band_codes=bands,
                max_concurrent=max_concurrent,
                for_xarray=for_xarray,
                backend=backend,
                target_crs=target_crs,
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
            **filters,
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

    import pyarrow.compute as pc

    epsg_col = ds.to_table(columns=["proj:epsg"]).column("proj:epsg")
    unique = pc.drop_null(pc.unique(epsg_col)).to_pylist()
    if len(unique) <= 1:
        return None

    from collections import Counter

    counts = Counter(v for v in epsg_col.to_pylist() if v is not None)
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
    **filters: Any,
):
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
    filters : kwargs
        Additional keyword arguments forwarded to ``Collection.subset()``.

    Returns
    -------
    geopandas.GeoDataFrame
        Band arrays in native COG dtype. Each row is a geometry-record
        pair with pixel data as columns.
    """
    return _load_and_merge(
        collection=collection,
        geometries=geometries,
        bands=bands,
        for_xarray=False,
        merge_fn=lambda dfs: gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True)),
        data_source=data_source,
        max_concurrent=max_concurrent,
        backend=backend,
        target_crs=target_crs,
        **filters,
    )
