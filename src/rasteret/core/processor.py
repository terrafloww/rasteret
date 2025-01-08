"""
Rasteret: Efficient satellite imagery retrieval and processing
==========================================================

Core Components:
--------------
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

logger = setup_logger("INFO", customname="rasteret.processor")


class Rasteret:
    """Main interface for satellite data retrieval and processing.

    Attributes:
        data_source (str): Source dataset identifier
        workspace_dir (Path): Directory for storing collections
        custom_name (str, optional): Collection name prefix
        date_range (Tuple[str, str], optional): Date range for collection
        aws_profile (str, optional): AWS profile for authentication

    Examples:
        >>> processor = Rasteret("landsat-c2l2-sr", "workspace")
        >>> processor.create_index(
        ...     bbox=[77.55, 13.01, 77.58, 13.04],
        ...     date_range=["2024-01-01", "2024-01-31"]
        ... )
        >>> df = processor.query(geometries=[polygon], bands=["B4", "B5"])
    """

    def __init__(
        self,
        data_source: Union[str, DataSources],
        workspace_dir: Optional[Union[str, Path]] = None,
        custom_name: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ):
        """Initialize Rasteret processor."""
        self.data_source = data_source
        self.workspace_dir = Path(workspace_dir or Path.home() / "rasteret_workspace")
        self.custom_name = custom_name
        self.date_range = date_range

        # Initialize cloud config early
        self.cloud_config = CloudConfig.get_config(str(data_source))
        self._cloud_provider = None if not self.cloud_config else AWSProvider()
        self._collection = None

    @property
    def provider(self):
        """Get cloud provider."""
        return self._cloud_provider

    def _get_collection_path(self) -> Path:
        """Get expected collection path"""
        return self.workspace_dir / f"{self.custom_name}.parquet"

    def _get_bbox_from_geometries(self, geometries: List[Polygon]) -> List[float]:
        """Get combined bbox from geometries"""
        bounds = [geom.bounds for geom in geometries]
        return [
            min(b[0] for b in bounds),  # minx
            min(b[1] for b in bounds),  # miny
            max(b[2] for b in bounds),  # maxx
            max(b[3] for b in bounds),  # maxy
        ]

    def _ensure_collection(self) -> None:
        """Ensure collection exists and is loaded with proper partitioning."""
        if not self.custom_name:
            raise ValueError("custom_name is required")

        stac_path = self.workspace_dir / f"{self.custom_name}_stac"

        if self._collection is None:
            if stac_path.exists():
                self._collection = Collection.from_local(stac_path)
            else:
                raise ValueError(f"Collection not found: {stac_path}")

    def create_collection(
        self,
        bbox: List[float],
        date_range: Optional[Tuple[str, str]] = None,
        force: bool = False,
        **filters,
    ) -> None:
        """Create or load STAC index."""
        if not self.custom_name:
            raise ValueError("custom_name is required")

        # Create standardized collection name with date range
        collection_name = Collection.create_name(
            self.custom_name, date_range or self.date_range, str(self.data_source)
        )

        stac_path = self.workspace_dir / f"{collection_name}_stac"

        if stac_path.exists() and not force:
            self._collection = Collection.from_local(stac_path)
            logger.info(f"Loading existing collection: {collection_name}")
            return

        # Create new collection
        indexer = StacToGeoParquetIndexer(
            data_source=self.data_source,
            stac_api=STAC_ENDPOINTS[self.data_source],
            workspace_dir=stac_path,
            cloud_provider=self.provider,
            cloud_config=self.cloud_config,
            name=collection_name,
        )

        self._collection = asyncio.run(
            indexer.build_index(
                bbox=bbox, date_range=date_range or self.date_range, query=filters
            )
        )

        if self._collection is not None:
            logger.info(f"Created collection: {collection_name}")

    @classmethod
    def list_collections(
        cls, workspace_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """List collections with metadata."""
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
        """Load collection by name."""
        workspace_dir = workspace_dir or Path.home() / "rasteret_workspace"
        stac_path = workspace_dir / f"{collection_name.replace('_stac', '')}_stac"

        if not stac_path.exists():
            raise ValueError(f"Collection not found: {collection_name}")

        # Get data source from name
        data_source = collection_name.split("_")[-1].upper()

        # Create processor
        processor = cls(
            data_source=getattr(DataSources, data_source, data_source),
            workspace_dir=workspace_dir,
            custom_name=collection_name,
        )

        # Load collection
        processor._collection = Collection.from_local(stac_path)

        logger.info(f"Loaded existing collection: {collection_name}")
        return processor

    def get_gdf(
        self,
        geometries: List[Polygon],
        bands: List[str],
        max_concurrent: int = 50,
        **filters,
    ) -> gpd.GeoDataFrame:
        """Query indexed scenes matching filters for specified geometries and bands."""
        # Validate inputs
        if not geometries:
            raise ValueError("No geometries provided")
        if not bands:
            raise ValueError("No bands specified")

        # Ensure collection exists and is loaded
        self._ensure_collection()

        if filters:
            self._collection = self._collection.filter_scenes(**filters)

        return asyncio.run(self._get_gdf(geometries, bands, max_concurrent))

    async def _get_gdf(self, geometries, bands, max_concurrent):
        total_scenes = len(self._collection.dataset.to_table())
        logger.info(f"Processing {total_scenes} scenes for {len(bands)} bands")
        results = []

        async for scene in tqdm(
            self._collection.iterate_scenes(self.data_source),
            total=total_scenes,
            desc="Loading scenes",
        ):
            scene_results = await scene.load_bands(
                geometries,
                bands,
                max_concurrent,
                cloud_provider=self.provider,
                cloud_config=self.cloud_config,
                for_xarray=False,
            )
            results.append(scene_results)

        return gpd.GeoDataFrame(pd.concat(results, ignore_index=True))

    def get_xarray(
        self,
        geometries: List[Polygon],
        bands: List[str],
        max_concurrent: int = 50,
        **filters,
    ) -> xr.Dataset:
        """
        Query collection and return as xarray Dataset.

        Args:
            geometries: List of polygons to query
            bands: List of band identifiers
            max_concurrent: Maximum concurrent requests
            **filters: Additional filters (e.g. cloud_cover_lt=20)

        Returns:
            xarray Dataset with data, coordinates, and CRS
        """
        # Same validation as query()
        if not geometries:
            raise ValueError("No geometries provided")
        if not bands:
            raise ValueError("No bands specified")

        self._ensure_collection()

        if filters:
            self._collection = self._collection.filter_scenes(**filters)

        return asyncio.run(
            self._get_xarray(
                geometries=geometries, bands=bands, max_concurrent=max_concurrent
            )
        )

    async def _get_xarray(
        self,
        geometries: List[Polygon],
        bands: List[str],
        max_concurrent: int,
        for_xarray: bool = True,
    ) -> xr.Dataset:
        datasets = []
        total_scenes = len(self._collection.dataset.to_table())

        logger.info(f"Processing {total_scenes} scenes for {len(bands)} bands")

        async for scene in tqdm(
            self._collection.iterate_scenes(self.data_source),
            total=total_scenes,
            desc="Loading scenes",
        ):
            scene_ds = await scene.load_bands(
                geometries=geometries,
                band_codes=bands,
                max_concurrent=max_concurrent,
                cloud_provider=self.provider,
                cloud_config=self.cloud_config,
                for_xarray=for_xarray,
            )
            if scene_ds is not None:
                datasets.append(scene_ds)
                logger.debug(
                    f"Loaded scene {scene.id} ({len(datasets)}/{total_scenes})"
                )

        if not datasets:
            raise ValueError("No valid data found for query")

        logger.info(f"Merging {len(datasets)} datasets")
        merged = xr.merge(datasets)
        merged = merged.sortby("time")

        logger.info(f"Data retrieved for {len(geometries)} geometries")
        logger.info(f"Dataset shape: {merged.sizes}")

        return merged

    def __repr__(self):
        return (
            f"Rasteret(data_source={self.data_source}, custom_name={self.custom_name})"
        )
