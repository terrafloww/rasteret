'''
Copyright 2025 Terrafloww Labs, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

"""
Rasteret: Efficient satellite imagery retrieval and processing
==========================================================

Core Components:
---------------
- Rasteret: Main interface for querying and processing scenes
- Collection: Manages indexed satellite data
- Scene: Handles individual scene processing

Example:
-------
>>> from rasteret import Rasteret
>>> processor = Rasteret(
...     data_source="landsat-c2l2-sr",
...     workspace_dir="workspace"
... )
>>> processor.create_index(
...     bbox=[77.55, 13.01, 77.58, 13.04],
...     date_range=["2024-01-01", "2024-01-31"]
... )
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import xarray as xr
import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds
from shapely.geometry import Polygon
from tqdm.asyncio import tqdm
from datetime import datetime

from rasteret.constants import STAC_ENDPOINTS, DataSources
from rasteret.core.collection import Collection
from rasteret.stac.indexer import StacToGeoParquetIndexer
from rasteret.cloud import AWSProvider, CloudProvider, CloudConfig
from rasteret.logging import setup_logger
from rasteret.fetch.cog import COGReader

logger = setup_logger("INFO", customname="rasteret.processor")


class URLSigningCache:
    """Efficient URL signing cache with thread-safe caching."""

    def __init__(
        self,
        cloud_provider: Optional[CloudProvider] = None,
        cloud_config: Optional[CloudConfig] = None,
        max_size: int = 1024,
    ):
        """
        Initialize URL signing cache.

        Args:
            cloud_provider: Cloud provider for URL signing
            cloud_config: Cloud configuration
            max_size: Maximum number of cached signed URLs
        """
        self._cloud_provider = cloud_provider
        self._cloud_config = cloud_config
        self._cache = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get_signed_url(self, url: str) -> str:
        """
        Get a signed URL, using cache if possible.

        Args:
            url: Original URL to be signed

        Returns:
            Signed URL
        """
        async with self._lock:
            # Check cache first
            if url in self._cache:
                return self._cache[url]

            # Sign URL if provider exists
            if self._cloud_provider and self._cloud_config:
                signed_url = self._cloud_provider.get_url(url, self._cloud_config)

                # Manage cache size
                if len(self._cache) >= self._max_size:
                    # Remove oldest entry
                    self._cache.pop(next(iter(self._cache)))

                self._cache[url] = signed_url
                return signed_url

            # If no signing possible, return original URL
            return url


class Rasteret:
    """Optimized Rasteret processor with connection pooling and caching."""

    def __init__(
        self,
        data_source: Union[str, DataSources],
        workspace_dir: Optional[Union[str, Path]] = None,
        custom_name: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        max_concurrent: int = 50,
    ):
        """
        Initialize Rasteret processor with optimized async handling.

        Args:
            data_source: Source of satellite data
            workspace_dir: Directory for storing collections
            custom_name: Custom name for the collection
            date_range: Date range for collection
            max_concurrent: Maximum concurrent connections
        """
        self.data_source = data_source
        self.workspace_dir = Path(workspace_dir or Path.home() / "rasteret_workspace")
        self.custom_name = custom_name
        self.date_range = date_range
        self.max_concurrent = max_concurrent

        # Initialize cloud resources
        self.cloud_config = CloudConfig.get_config(str(data_source))
        self._cloud_provider = None if not self.cloud_config else AWSProvider()

        # URL signing cache
        self._url_cache = URLSigningCache(
            cloud_provider=self._cloud_provider, cloud_config=self.cloud_config
        )

        # Persistent COG reader
        self._cog_reader = None
        self._collection = None

    async def __aenter__(self):
        """Async context manager entry for resource management."""
        # Initialize COG reader with connection pooling
        self._cog_reader = COGReader(max_concurrent=self.max_concurrent)
        await self._cog_reader.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit for cleanup."""
        if self._cog_reader:
            await self._cog_reader.__aexit__(exc_type, exc_val, exc_tb)

    def create_collection(
        self,
        bbox: List[float],
        date_range: Optional[Tuple[str, str]] = None,
        force: bool = False,
        **filters,
    ) -> None:
        """Sync interface for collection creation"""

        def _sync_create():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._async_create_collection(bbox, date_range, force, **filters)
                )
            finally:
                loop.close()

        return _sync_create()

    async def _async_create_collection(
        self,
        bbox: List[float],
        date_range: Optional[Tuple[str, str]] = None,
        force: bool = False,
        **filters,
    ) -> None:
        """Internal async implementation for collection creation"""
        if not self.custom_name:
            raise ValueError("custom_name is required")

        collection_name = Collection.create_name(
            self.custom_name, date_range or self.date_range, str(self.data_source)
        )
        collection_path = self.workspace_dir / f"{collection_name}_stac"

        if collection_path.exists() and not force:
            logger.info(f"Collection {collection_name} already exists")
            self._collection = Collection.from_local(collection_path)
            return

        # Initialize indexer with required params
        indexer = StacToGeoParquetIndexer(
            data_source=str(self.data_source),
            stac_api=STAC_ENDPOINTS[self.data_source],
            workspace_dir=collection_path,
            name=collection_name,
            cloud_provider=self._cloud_provider,
            cloud_config=self.cloud_config,
            max_concurrent=self.max_concurrent,
        )

        # Use build_index instead of create_index
        self._collection = await indexer.build_index(
            bbox=bbox, date_range=date_range or self.date_range, query=filters
        )

        logger.info(f"Created collection: {collection_name}")

    @classmethod
    def list_collections(
        cls, workspace_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """List collections with metadata (unchanged)."""
        workspace_dir = workspace_dir or Path.home() / "rasteret_workspace"
        collections = []

        for stac_dir in workspace_dir.glob("*_stac"):
            try:
                # Parse name for data source
                name = stac_dir.name.replace("_stac", "")
                data_source = name.split("_")[-1].upper()

                # Read dataset
                dataset = ds.dataset(str(stac_dir))
                table = dataset.to_table()
                df = table.to_pandas()

                # Get date range from data
                if "datetime" in df.columns:
                    start_date = pd.to_datetime(df["datetime"].min()).strftime(
                        "%Y-%m-%d"
                    )
                    end_date = pd.to_datetime(df["datetime"].max()).strftime("%Y-%m-%d")
                    date_range = f"{start_date} to {end_date}"
                else:
                    date_range = ""

                collections.append(
                    {
                        "name": name,
                        "data_source": data_source,
                        "date_range": date_range,
                        "size": len(df),
                        "created": datetime.fromtimestamp(
                            stac_dir.stat().st_ctime
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            except Exception as e:
                logger.debug(f"Failed to read collection {stac_dir}: {e}")

        return collections

    @classmethod
    def load_collection(
        cls, collection_name: str, workspace_dir: Optional[Path] = None
    ) -> "Rasteret":
        """Load collection by name with async preparation."""
        workspace_dir = workspace_dir or Path.home() / "rasteret_workspace"

        # Remove _stac suffix if present
        clean_name = collection_name.replace("_stac", "")
        stac_path = workspace_dir / f"{clean_name}_stac"

        if not stac_path.exists():
            raise ValueError(f"Collection not found: {collection_name}")

        # Parse data source from name
        try:
            data_source = clean_name.split("_")[-1].upper()
        except IndexError:
            data_source = "UNKNOWN"

        # Create processor
        processor = cls(
            data_source=getattr(DataSources, data_source, data_source),
            workspace_dir=workspace_dir,
            custom_name=clean_name,
        )

        # Load collection
        processor._collection = Collection.from_local(stac_path)

        logger.info(f"Loaded existing collection: {collection_name}")
        return processor

    async def _sign_scene_urls(self, scene):
        """
        Sign URLs for a scene's assets.

        Args:
            scene: Scene with assets to sign

        Returns:
            Scene with signed URLs
        """
        # Create copies to avoid modifying original
        signed_assets = {}
        for band, asset in scene.assets.items():
            # Get signed URL
            signed_url = await self._url_cache.get_signed_url(asset["href"])

            # Create a copy of the asset with signed URL
            signed_asset = asset.copy()
            signed_asset["href"] = signed_url
            signed_assets[band] = signed_asset

        # Update scene assets with signed URLs
        scene.assets = signed_assets
        return scene

    async def _get_scene_data(
        self,
        geometries: List[Polygon],
        bands: List[str],
        for_xarray: bool = True,
        batch_size: int = 10,
        **filters,
    ) -> Union[List[gpd.GeoDataFrame], List[xr.Dataset]]:
        """
        Optimized async scene data retrieval with URL signing and batching.

        Args:
            geometries: List of geometries to process
            bands: Bands to retrieve
            for_xarray: Whether to return xarray or GeoDataFrame
            batch_size: Number of scenes to process in parallel
            **filters: Additional filtering parameters

        Returns:
            List of processed datasets
        """
        if not self._collection:
            raise ValueError("No collection loaded")

        # Apply filters if provided
        if filters:
            self._collection = self._collection.filter_scenes(**filters)

        results = []

        # Prepare scene batches
        scene_batches = []
        current_batch = []

        async for scene in self._collection.iterate_scenes(self.data_source):
            # Sign URLs for the scene
            scene = await self._sign_scene_urls(scene)
            current_batch.append(scene)

            if len(current_batch) == batch_size:
                scene_batches.append(current_batch)
                current_batch = []

        # Add remaining scenes
        if current_batch:
            scene_batches.append(current_batch)

        # Process scene batches in parallel
        for batch in tqdm(scene_batches, desc="Processing scenes"):
            # Create tasks for batch processing
            tasks = [
                scene.load_bands(
                    geometries=geometries,
                    band_codes=bands,
                    max_concurrent=self.max_concurrent,
                    cloud_provider=self._cloud_provider,
                    cloud_config=self.cloud_config,
                    for_xarray=for_xarray,
                )
                for scene in batch
            ]

            # Gather results from batch
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r is not None])

        return results

    def get_xarray(
        self, geometries: Union[Polygon, List[Polygon]], bands: List[str], **filters
    ) -> xr.Dataset:
        """Sync interface for xarray retrieval"""
        if isinstance(geometries, Polygon):
            geometries = [geometries]

        def _sync_get():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._async_get_xarray(geometries, bands, **filters)
                )
            finally:
                loop.close()

        return _sync_get()

    async def _async_get_xarray(
        self, geometries: List[Polygon], bands: List[str], **filters
    ) -> xr.Dataset:
        """Internal async implementation"""
        async with self:
            scene_datasets = await self._get_scene_data(
                geometries=geometries, bands=bands, for_xarray=True, **filters
            )
            if not scene_datasets:
                raise ValueError("No valid data found")
            logger.info(f"Merging {len(scene_datasets)} datasets")
            merged = xr.merge(scene_datasets)
            return merged.sortby("time")

    def get_gdf(
        self, geometries: Union[Polygon, List[Polygon]], bands: List[str], **filters
    ) -> gpd.GeoDataFrame:
        """Sync interface for GeoDataFrame retrieval"""
        if isinstance(geometries, Polygon):
            geometries = [geometries]

        def _sync_get():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._async_get_gdf(geometries, bands, **filters)
                )
            finally:
                loop.close()

        return _sync_get()

    async def _async_get_gdf(
        self, geometries: List[Polygon], bands: List[str], **filters
    ) -> gpd.GeoDataFrame:
        """Internal async implementation"""
        async with self:
            scene_dfs = await self._get_scene_data(
                geometries=geometries, bands=bands, for_xarray=False, **filters
            )
            if not scene_dfs:
                raise ValueError("No valid data found")
            return gpd.GeoDataFrame(pd.concat(scene_dfs, ignore_index=True))
