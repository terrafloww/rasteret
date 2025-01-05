""" Scene class for handling COG data loading and processing. """

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from shapely.geometry import Polygon
import geopandas as gpd
import rioxarray  # noqa
import asyncio
from tqdm.asyncio import tqdm

from rasteret.types import SceneInfo, CogMetadata
from rasteret.constants import STAC_COLLECTION_BAND_MAPS
from rasteret.fetch.cog import read_cog_tile_data
from rasteret.cloud import CloudProvider, CloudConfig
from rasteret.logging import setup_logger

logger = setup_logger("INFO")


class Scene:
    """
    A single scene with associated metadata and data access methods.

    Scene handles the actual data loading from COGs, including:
    - Async data loading
    - Tile management
    - Geometry masking
    """

    def __init__(self, info: SceneInfo, data_source: str) -> None:
        """Initialize Scene from metadata.

        Args:
            info: Scene metadata including urls and COG info
        """
        self.id = info.id
        self.datetime = info.datetime
        self.bbox = info.bbox
        self.scene_geometry = info.scene_geometry
        self.crs = info.crs
        self.cloud_cover = info.cloud_cover
        self.assets = info.assets
        self.scene_metadata = info.metadata
        self.collection = info.collection
        self.data_source = data_source

    def _get_band_radiometric_params(
        self, band_code: str
    ) -> Optional[Dict[str, float]]:
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

    def _get_asset_url(
        self, asset: Dict, provider: CloudProvider, cloud_config: CloudConfig
    ) -> str:
        """Get authenticated URL for asset"""
        url = asset["href"] if isinstance(asset, dict) else asset
        if provider and cloud_config:
            return provider.get_url(url, cloud_config)
        return url

    def get_band_cog_metadata(
        self,
        band_code: str,
        provider: Optional[CloudProvider] = None,
        cloud_config: Optional[CloudConfig] = None,
    ) -> Tuple[CogMetadata, str]:
        """Get COG metadata and url for a specified band."""

        actual_band_code = STAC_COLLECTION_BAND_MAPS.get(self.data_source, {}).get(
            band_code, band_code
        )

        if actual_band_code not in self.assets:
            raise ValueError(f"Band {band_code} not found in assets")

        asset = self.assets[actual_band_code]

        # Prefer S3 URL for AWS assets
        url = self._get_asset_url(asset, provider, cloud_config)

        # Band metadata key could be either band_code or actual_band_code
        metadata_keys = [f"{band_code}_metadata", f"{actual_band_code}_metadata"]
        raw_metadata = None
        for key in metadata_keys:
            if key in self.scene_metadata:
                raw_metadata = self.scene_metadata[key]
                break

        if raw_metadata is None or url is None:
            logger.error(
                f"Metadata not found for band {band_code} in scene {self.id}. Available keys: {list(self.scene_metadata.keys())}"
            )
            return None, None

        try:
            cog_metadata = CogMetadata(
                width=raw_metadata.get("image_width", raw_metadata.get("width")),
                height=raw_metadata.get("image_height", raw_metadata.get("height")),
                tile_width=raw_metadata["tile_width"],
                tile_height=raw_metadata["tile_height"],
                dtype=np.dtype(raw_metadata["dtype"]),
                transform=raw_metadata["transform"],
                crs=self.crs,
                tile_offsets=raw_metadata["tile_offsets"],
                tile_byte_counts=raw_metadata["tile_byte_counts"],
                predictor=raw_metadata.get("predictor"),
                compression=raw_metadata.get("compression"),
                pixel_scale=raw_metadata.get("pixel_scale"),
                tiepoint=raw_metadata.get("tiepoint"),
            )
            return cog_metadata, url
        except KeyError as e:
            logger.error(
                f"Missing required metadata field {e} for band {band_code} in scene {self.id}"
            )
            logger.debug(f"Available metadata: {raw_metadata}")
            return None, None

    def intersects(self, geometry: Polygon) -> bool:
        """Check if scene intersects with geometry."""
        return self.geometry.intersects(geometry)

    @property
    def available_bands(self) -> List[str]:
        """List of available bands for this scene."""
        return list(self.assets.keys())

    def __repr__(self) -> str:
        return (
            f"Scene(id='{self.id}', "
            f"datetime='{self.datetime}', "
            f"cloud_cover={self.cloud_cover})"
        )

    async def _load_single_band(
        self,
        geometry: Polygon,
        band_code: str,
        cloud_provider: Optional[CloudProvider],
        cloud_config: Optional[CloudConfig],
        max_concurrent: int = 50,
    ) -> Optional[Dict]:
        """Load single band data for geometry."""
        cog_meta, url = self.get_band_cog_metadata(
            band_code, provider=cloud_provider, cloud_config=cloud_config
        )
        if not cog_meta or not url:
            return None

        data, transform = await read_cog_tile_data(
            url, cog_meta, geometry, max_concurrent
        )
        if data is None or transform is None:
            return None

        return {"data": data, "transform": transform, "band": band_code}

    async def load_bands(
        self,
        geometries: List[Polygon],
        band_codes: List[str],
        max_concurrent: int = 50,
        cloud_provider: Optional[CloudProvider] = None,
        cloud_config: Optional[CloudConfig] = None,
        for_xarray: bool = True,
    ) -> Union[gpd.GeoDataFrame, xr.Dataset]:
        """Load bands with parallel processing and progress tracking."""

        logger.debug(
            f"Loading {len(band_codes)} bands for {len(geometries)} geometries"
        )

        geom_progress = tqdm(total=len(geometries), desc=f"Scene {self.id}", position=0)

        async def process_geometry(geometry: Polygon, geom_id: int):
            band_progress = tqdm(
                total=len(band_codes), desc=f"Geom {geom_id}", position=1, leave=False
            )

            band_tasks = []
            for band_code in band_codes:
                task = self._load_single_band(
                    geometry, band_code, cloud_provider, cloud_config, max_concurrent
                )
                band_tasks.append(task)

            results = await asyncio.gather(*band_tasks)
            band_progress.update(len(band_codes))
            band_progress.close()
            geom_progress.update(1)

            return [r for r in results if r is not None], geom_id

        # Process geometries concurrently with semaphore
        sem = asyncio.Semaphore(max_concurrent)

        async def bounded_process(geometry: Polygon, geom_id: int):
            async with sem:
                return await process_geometry(geometry, geom_id)

        tasks = [bounded_process(geom, idx + 1) for idx, geom in enumerate(geometries)]
        results = await asyncio.gather(*tasks)

        geom_progress.close()

        # Process results
        if for_xarray:
            return self._merge_xarray_results(results)
        else:
            return self._merge_geodataframe_results(results, geometries)

    def _merge_xarray_results(
        self,
        results: List[Tuple[List[Dict], int]],
    ) -> xr.Dataset:
        """Merge results into xarray Dataset."""
        data_arrays = []

        for band_results, geom_id in results:
            if not band_results:
                continue

            geom_arrays = []
            for band_result in band_results:
                da = xr.DataArray(
                    data=band_result["data"],
                    dims=["y", "x"],
                    coords={
                        "y": band_result["transform"].f
                        + np.arange(band_result["data"].shape[0])
                        * band_result["transform"].e,
                        "x": band_result["transform"].c
                        + np.arange(band_result["data"].shape[1])
                        * band_result["transform"].a,
                    },
                    name=band_result["band"],
                )
                da.rio.write_crs(self.crs, inplace=True)
                da.rio.write_transform(band_result["transform"], inplace=True)
                geom_arrays.append(da)

            if geom_arrays:
                ds = xr.merge(geom_arrays)
                ds = ds.expand_dims({"time": [self.datetime], "geometry": [geom_id]})
                ds.rio.write_crs(self.crs, inplace=True)
                ds.attrs.update(
                    {
                        "crs": self.crs,
                        "geometry_id": geom_id,
                        "scene_id": self.id,
                        "datetime": self.datetime,
                        "cloud_cover": self.cloud_cover,
                        "collection": self.collection,
                    }
                )
                data_arrays.append(ds)

        if not data_arrays:
            return None

        return xr.merge(data_arrays)

    def _merge_geodataframe_results(
        self, results: List[Tuple[List[Dict], int]], geometries: List[Polygon]
    ) -> Optional[gpd.GeoDataFrame]:
        """Merge results into GeoDataFrame."""
        rows = []

        for band_results, geom_id in results:
            if not band_results:
                continue

            for band_result in band_results:
                rows.append(
                    {
                        "scene_id": self.id,
                        "datetime": self.datetime,
                        "cloud_cover": self.cloud_cover,
                        "collection": self.collection,
                        "geometry": geometries[geom_id - 1],
                        "band": band_result["band"],
                        "data": band_result["data"],
                    }
                )

        if not rows:
            return None

        return gpd.GeoDataFrame(rows)
