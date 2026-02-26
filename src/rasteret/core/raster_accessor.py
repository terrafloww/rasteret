# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm.asyncio import tqdm

from rasteret.constants import BandRegistry
from rasteret.fetch.cog import read_cog
from rasteret.types import CogMetadata, RasterInfo

logger = logging.getLogger(__name__)


class RasterAccessor:
    """Data-loading handle for a single Parquet record (row) in a Collection.

    Each record in a Rasteret Collection represents one raster item:
    typically a satellite scene, but could be a drone image, derived
    product, or any tiled GeoTIFF.  ``RasterAccessor`` wraps that
    record's metadata and provides methods to load band data as arrays.

    Handles:
    - Async band data loading via cached COG metadata
    - Tile management and geometry masking
    - Multi-band concurrent fetching
    """

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

    def _get_band_radiometric_params(self, band_code: str) -> dict[str, float] | None:
        """Get radiometric parameters from STAC metadata if available."""
        try:
            asset = self.assets[band_code]
            band_info = asset["raster:bands"][0]

            if "scale" in band_info and "offset" in band_info:
                return {
                    "scale": float(band_info["scale"]),
                    "offset": float(band_info["offset"]),
                }
        except (KeyError, IndexError):
            pass

        return None

    def _extract_asset_href(self, asset: dict) -> str | None:
        """Resolve the most appropriate href for a STAC asset."""

        if asset is None:
            return None

        if not isinstance(asset, dict):
            return asset

        href = asset.get("href")
        if href:
            return href

        alternates = asset.get("alternate") or {}
        if isinstance(alternates, dict):
            preferred_order = ["s3", "aws", "https", "http", "cloudfront"]
            for key in preferred_order:
                alt = alternates.get(key)
                if isinstance(alt, dict) and alt.get("href"):
                    return alt["href"]
            for alt in alternates.values():
                if isinstance(alt, dict) and alt.get("href"):
                    return alt["href"]

        links = asset.get("links") if isinstance(asset, dict) else None
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and link.get("href"):
                    return link["href"]

        return None

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

    @property
    def geometry(self):
        """Alias for ``footprint``."""
        return self.footprint

    @property
    def available_bands(self) -> list[str]:
        """List available band keys for this record."""
        return list(self.assets.keys())

    def __repr__(self) -> str:
        return (
            f"RasterAccessor(id='{self.id}', "
            f"datetime='{self.datetime}', "
            f"cloud_cover={self.cloud_cover})"
        )

    async def _load_single_band(
        self,
        geom_array: pa.Array,
        geom_idx: int,
        band_code: str,
        max_concurrent: int = 50,
        reader: object | None = None,
    ) -> dict | None:
        """Load single band data for a geometry identified by index."""
        cog_meta, url, band_index = self.try_get_band_cog_metadata(band_code)
        if cog_meta is None or url is None:
            raise ValueError(
                f"Missing band metadata or href for band '{band_code}' "
                f"in record '{self.id}'"
            )

        result = await read_cog(
            url,
            cog_meta,
            band_index=band_index,
            geom_array=geom_array,
            geom_idx=geom_idx,
            max_concurrent=max_concurrent,
            reader=reader,
        )
        if result.data.size == 0:
            return None

        # TODO: Apply radiometric correction (scale/offset from STAC
        # raster:bands) when opted in.  See _get_band_radiometric_params().
        # Needs: opt-in flag (apply_scale_offset=False default),
        # nodata masking, and dtype promotion (uint16 -> float32).

        return {"data": result.data, "transform": result.transform, "band": band_code}

    def _reproject_band_results(
        self,
        results: list[dict],
        target_crs: int,
    ) -> list[dict]:
        """Reproject band results from source CRS to *target_crs*."""
        from rasteret.core.utils import (
            compute_dst_grid_from_src,
            reproject_array,
        )

        reprojected = []
        for r in results:
            data = r["data"]
            src_tf = r["transform"]
            h, w = data.shape

            xmin = float(src_tf.c)
            ymax = float(src_tf.f)
            xmax = xmin + w * float(src_tf.a)
            ymin = ymax + h * float(src_tf.e)

            src_bounds = (
                min(xmin, xmax),
                min(ymin, ymax),
                max(xmin, xmax),
                max(ymin, ymax),
            )
            dst_tf, dst_shape = compute_dst_grid_from_src(
                self.crs,
                target_crs,
                w,
                h,
                src_bounds,
            )
            reprojected_data = reproject_array(
                data,
                src_tf,
                self.crs,
                target_crs,
                dst_tf,
                dst_shape,
            )
            reprojected.append(
                {
                    "data": reprojected_data,
                    "transform": dst_tf,
                    "band": r["band"],
                }
            )
        return reprojected

    async def load_bands(
        self,
        geometries: pa.Array,
        band_codes: list[str],
        max_concurrent: int = 50,
        for_xarray: bool = True,
        backend: object | None = None,
        target_crs: int | None = None,
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
        backend : object, optional
            Pluggable I/O backend.
        target_crs : int, optional
            Reproject results to this CRS.

        Returns
        -------
        xarray.Dataset or geopandas.GeoDataFrame
            Data is returned in the native COG dtype (e.g. ``uint16``,
            ``int8``, ``float32``). Integer arrays promote to ``float32``
            only when geometry masking requires NaN and no nodata value is
            declared in the COG metadata.
        """
        from rasteret.fetch.cog import COGReader

        n_geoms = len(geometries)
        logger.debug(f"Loading {len(band_codes)} bands for {n_geoms} geometries")

        geom_progress = tqdm(total=n_geoms, desc=f"Record {self.id}")

        async with COGReader(max_concurrent=max_concurrent, backend=backend) as reader:

            async def process_geometry(geom_idx: int, geom_id: int):
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
                    )
                    band_tasks.append(task)

                raw_results = await asyncio.gather(*band_tasks, return_exceptions=True)
                results = []
                for r in raw_results:
                    if isinstance(r, Exception):
                        logger.error("Band load failed: %s", r)
                    else:
                        results.append(r)
                band_progress.update(len(band_codes))
                band_progress.close()
                geom_progress.update(1)

                valid = [r for r in results if r is not None]
                if target_crs is not None and target_crs != self.crs and valid:
                    valid = self._reproject_band_results(valid, target_crs)
                return valid, geom_id

            # Process geometries concurrently with semaphore
            sem = asyncio.Semaphore(max_concurrent)

            async def bounded_process(geom_idx: int, geom_id: int):
                async with sem:
                    return await process_geometry(geom_idx, geom_id)

            tasks = [bounded_process(idx, idx + 1) for idx in range(n_geoms)]
            raw_geom_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for r in raw_geom_results:
            if isinstance(r, Exception):
                logger.error("Geometry processing failed: %s", r)
            else:
                results.append(r)

        geom_progress.close()

        # Process results
        if for_xarray:
            return self._merge_xarray_results(results, target_crs=target_crs)
        else:
            return self._merge_geodataframe_results(results, geometries)

    @staticmethod
    def _write_crs_cf(ds_or_da, epsg_code, transform=None):
        """Attach CRS as a CF-convention ``spatial_ref`` coordinate.

        Uses ``pyproj`` directly (no rioxarray dependency).  Writes:
        - CF grid-mapping attributes via ``CRS.to_cf()``
        - WKT2 (ISO 19162:2019) in ``crs_wkt``
        - PROJJSON for forward-compatible consumers
        - GDAL-style ``GeoTransform`` string when *transform* is given
        """
        import xarray as xr
        from pyproj import CRS

        crs_obj = CRS.from_epsg(epsg_code)
        cf_attrs = crs_obj.to_cf()
        cf_attrs["crs_wkt"] = crs_obj.to_wkt(version="WKT2_2019")
        cf_attrs["projjson"] = crs_obj.to_json()
        if transform is not None:
            cf_attrs["GeoTransform"] = (
                f"{transform.c} {transform.a} {transform.b} "
                f"{transform.f} {transform.d} {transform.e}"
            )

        spatial_ref = xr.DataArray(0, attrs=cf_attrs)
        ds_or_da = ds_or_da.assign_coords(spatial_ref=spatial_ref)

        if hasattr(ds_or_da, "data_vars"):
            for var in ds_or_da.data_vars:
                ds_or_da[var].encoding["grid_mapping"] = "spatial_ref"
        else:
            ds_or_da.encoding["grid_mapping"] = "spatial_ref"
        return ds_or_da

    def _merge_xarray_results(
        self,
        results: list[tuple[list[dict], int]],
        target_crs: int | None = None,
    ):
        """Merge results into xarray Dataset."""
        import xarray as xr

        data_arrays = []

        for band_results, geom_id in results:
            if not band_results:
                continue

            geom_arrays = []
            for band_result in band_results:
                h, w = band_result["data"].shape
                transform = band_result["transform"]
                da = xr.DataArray(
                    data=band_result["data"],
                    dims=["y", "x"],
                    coords={
                        "y": transform.f + (np.arange(h) + 0.5) * transform.e,
                        "x": transform.c + (np.arange(w) + 0.5) * transform.a,
                    },
                    name=band_result["band"],
                )
                crs_out = target_crs if target_crs is not None else self.crs
                da = self._write_crs_cf(da, crs_out, transform=transform)
                geom_arrays.append(da)

            if geom_arrays:
                ds = xr.merge(geom_arrays, compat="override")
                # Strip timezone for xarray compat (numpy can't merge tz-aware datetime64)
                ts = (
                    pd.Timestamp(self.datetime).tz_localize(None)
                    if hasattr(self.datetime, "tzinfo") and self.datetime.tzinfo
                    else pd.Timestamp(self.datetime)
                )
                ds = ds.expand_dims({"time": [ts], "geometry": [geom_id]})
                # CRS is already written per-DataArray above (line 445) and
                # propagates through xr.merge.  Do NOT re-write here: the
                # loop variable `transform` holds the *last* band's value,
                # which is wrong for mixed-resolution band sets.
                ds.attrs.update(
                    {
                        "crs": crs_out,
                        "geometry_id": geom_id,
                        "record_id": self.id,
                        "datetime": self.datetime,
                        "cloud_cover": self.cloud_cover,
                        "collection": self.collection,
                    }
                )
                data_arrays.append(ds)

        if not data_arrays:
            return xr.Dataset()

        return xr.merge(data_arrays, compat="override")

    def _merge_geodataframe_results(
        self, results: list[tuple[list[dict], int]], geometries: pa.Array
    ) -> gpd.GeoDataFrame:
        """Merge results into GeoDataFrame."""
        from rasteret.core.geometry import to_rasterio_geojson

        rows = []

        for band_results, geom_id in results:
            if not band_results:
                continue

            # Convert GeoArrow geometry to GeoJSON dict at output boundary
            geojson = to_rasterio_geojson(geometries, geom_id - 1)

            for band_result in band_results:
                rows.append(
                    {
                        "record_id": self.id,
                        "datetime": self.datetime,
                        "cloud_cover": self.cloud_cover,
                        "collection": self.collection,
                        "geometry": geojson,
                        "band": band_result["band"],
                        "data": band_result["data"],
                    }
                )

        if not rows:
            return gpd.GeoDataFrame()

        # Let GeoPandas create Shapely geometries from GeoJSON at output
        import shapely

        gdf = gpd.GeoDataFrame(rows)
        gdf["geometry"] = gdf["geometry"].apply(shapely.geometry.shape)
        return gdf

    def __dir__(self) -> list[str]:
        names = super().__dir__()
        return sorted(
            name
            for name in names
            if (name.startswith("__") and name.endswith("__"))
            or not name.startswith("_")
        )
