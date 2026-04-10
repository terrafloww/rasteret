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
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from tqdm.asyncio import tqdm

from rasteret.core.geometry import bbox_array, intersect_bbox
from rasteret.core.utils import infer_data_source, run_sync

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

    from rasteret.core.collection import Collection

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Loading pipeline
# ------------------------------------------------------------------


def _normalize_spatial_y_axis_order(merged: Any) -> Any:
    """Normalize y-axis ordering to match CF GeoTransform semantics.

    Xarray alignment/merge operations can change coordinate ordering when
    forming unions. For raster-like outputs, downstream consumers often
    assume that:
      - north-up rasters (GeoTransform[5] < 0) have decreasing y
      - south-up rasters (GeoTransform[5] > 0) have increasing y

    When we can infer the expected orientation from ``spatial_ref``'s
    ``GeoTransform`` attribute, we enforce it via ``sortby("y")``.
    """
    if not hasattr(merged, "coords") or "y" not in getattr(merged, "coords", {}):
        return merged

    spatial_ref = None
    try:
        spatial_ref = merged.coords.get("spatial_ref")
    except Exception:  # pragma: no cover - defensive for non-xarray types
        spatial_ref = None
    if spatial_ref is None:
        return merged

    geotransform = getattr(spatial_ref, "attrs", {}).get("GeoTransform")
    if not geotransform:
        return merged

    try:
        parts = [float(p) for p in str(geotransform).split()]
    except Exception:
        return merged
    if len(parts) != 6:
        return merged
    e = float(parts[5])
    if e == 0.0:
        return merged

    try:
        import numpy as np

        y = merged["y"].values
        if getattr(y, "ndim", 1) != 1 or y.size < 2:
            return merged

        is_non_decreasing = bool(np.all(y[1:] >= y[:-1]))
        is_non_increasing = bool(np.all(y[1:] <= y[:-1]))
    except Exception:
        return merged

    if e < 0.0:
        if is_non_increasing:
            return merged
        return merged.sortby("y", ascending=False)

    # e > 0.0 (south-up)
    if is_non_decreasing:
        return merged
    return merged.sortby("y", ascending=True)


def _ensure_geoarrow(geometries: Any) -> pa.Array:
    """Coerce any geometry input to a GeoArrow native array."""
    from rasteret.core.geometry import coerce_to_geoarrow

    return coerce_to_geoarrow(geometries)


def _combine_first_int_with_fill(datasets: list[Any]) -> Any | None:
    """Combine integer datasets via `_FillValue` without NA-driven upcast."""
    try:
        import numpy as np
        import xarray as xr
    except Exception:
        return None

    if not datasets:
        return None

    out: dict[str, Any] = {}
    names = sorted({n for ds_obj in datasets for n in ds_obj.data_vars})
    for name in names:
        arrays = [ds_obj[name] for ds_obj in datasets if name in ds_obj.data_vars]
        if not arrays or not all(np.issubdtype(a.dtype, np.integer) for a in arrays):
            return None
        fills = {
            a.attrs.get("_FillValue")
            for a in arrays
            if a.attrs.get("_FillValue") is not None
        }
        if len(fills) != 1:
            return None
        fill_value = int(next(iter(fills)))
        aligned = xr.align(*arrays, join="outer", copy=False, fill_value=fill_value)
        merged = aligned[0]
        for nxt in aligned[1:]:
            merged = xr.where(merged == fill_value, nxt, merged)
        merged.attrs.update(arrays[0].attrs)
        out[name] = merged

    return xr.Dataset(out) if out else None


def _derive_query_bbox(
    geometries: Any,
    *,
    geometry_crs: int | None,
) -> tuple[float, float, float, float] | None:
    """Derive a query bbox in EPSG:4326 from any supported geometry input."""
    if geometries is None:
        return None

    from rasteret.core.utils import transform_bbox

    geo_arr = _ensure_geoarrow(geometries)
    if len(geo_arr) == 0:
        return None

    xmin, ymin, xmax, ymax = bbox_array(geo_arr)
    bbox = (
        pc.min(xmin).as_py(),
        pc.min(ymin).as_py(),
        pc.max(xmax).as_py(),
        pc.max(ymax).as_py(),
    )
    if any(value is None for value in bbox):
        return None

    derived = tuple(float(value) for value in bbox)
    if geometry_crs not in (None, 4326):
        derived = transform_bbox(derived, geometry_crs, 4326)
    return derived


def _narrow_query_filters(
    *,
    geometries: Any,
    geometry_crs: int | None,
    filters: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return filters narrowed by any bbox derivable from *geometries*.

    The returned bool is ``True`` when user-provided spatial constraints are
    provably disjoint, allowing callers to short-circuit without any pixel I/O.
    """
    narrowed = dict(filters)
    derived_bbox = _derive_query_bbox(geometries, geometry_crs=geometry_crs)
    if derived_bbox is None:
        return narrowed, False

    merged_bbox = intersect_bbox(narrowed.get("bbox"), derived_bbox)
    if narrowed.get("bbox") is not None and merged_bbox is None:
        return narrowed, True

    narrowed["bbox"] = merged_bbox or derived_bbox
    return narrowed, False


async def _load_collection_data(
    *,
    collection: "Collection",
    data_source: str,
    geometries: pa.Array,
    bands: list[str],
    max_concurrent: int,
    for_xarray: bool,
    for_numpy: bool = False,
    batch_size: int = 10,
    progress: bool = False,
    backend: object | None = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    **filters: Any,
) -> tuple[list[gpd.GeoDataFrame] | list, list[tuple[str, Exception]]]:
    """Core loading loop: iterate records, fetch tiles."""
    narrowed_filters, is_empty = _narrow_query_filters(
        geometries=geometries,
        geometry_crs=geometry_crs,
        filters=filters,
    )
    if is_empty:
        return [], []

    selected_collection = (
        collection.subset(**narrowed_filters) if narrowed_filters else collection
    )

    results = []
    errors: list[tuple[str, Exception]] = []
    raster_batches = []
    current_batch = []

    async for raster in selected_collection.iterate_rasters(data_source, bands=bands):
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
                for_numpy=for_numpy,
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
                if for_numpy and len(result) == 0:
                    continue
                if (
                    not for_xarray
                    and not for_numpy
                    and hasattr(result, "empty")
                    and result.empty
                ):
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
    for_numpy: bool,
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
            for_numpy=for_numpy,
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
    *,
    geometries: Any = None,
    geometry_crs: int | None = 4326,
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
    narrowed_filters, is_empty = _narrow_query_filters(
        geometries=geometries,
        geometry_crs=geometry_crs,
        filters=filters,
    )
    if is_empty:
        return None

    selected = collection.subset(**narrowed_filters) if narrowed_filters else collection
    schema = selected._schema
    if schema is None or "proj:epsg" not in schema.names:
        return None

    from collections import Counter

    import pyarrow.compute as pc

    counts: Counter[int] = Counter()
    for batch in selected._iter_record_batches(columns=["proj:epsg"]):
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

    # Auto-detect multi-CRS to prevent silent spatial data corruption
    # from merging tiles with incompatible coordinate systems.
    if target_crs is None:
        target_crs = _detect_target_crs(
            collection,
            geometries=geometries,
            geometry_crs=geometry_crs,
            filters=filters,
        )

    def _merge(datasets):
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
        for_numpy=False,
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

    def _merge_numpy(frames: list[list[tuple[list[dict], int]]]):
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
        all_touched=all_touched,
        **filters,
    )


# Backwards-compatible import path (internal). Prefer `Collection.sample_points`.
