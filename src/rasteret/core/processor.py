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
...     output_dir="workspace"
... )
>>> processor.create_index(
...     bbox=[77.55, 13.01, 77.58, 13.04],
...     date_range=["2024-01-01", "2024-01-31"]
... )
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from tqdm.asyncio import tqdm

from rasteret.constants import STAC_ENDPOINTS
from rasteret.core.collection import Collection
from rasteret.stac.indexer import StacToGeoParquetIndexer
from rasteret.cloud import CLOUD_CONFIG, AWSProvider, CloudProvider
from rasteret.logging import setup_logger

logger = setup_logger("INFO", customname="rasteret.processor")


class Rasteret:
    """Main interface for satellite data retrieval and processing.

    Attributes:
        data_source (str): Source dataset identifier
        output_dir (Path): Directory for storing collections
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
        data_source: str,
        output_dir: Union[str, Path],
        custom_name: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        aws_profile: Optional[str] = None,
    ):
        self.data_source = data_source
        self.output_dir = Path(output_dir)

        # Check credentials early if needed
        self.cloud_config = CLOUD_CONFIG.get(data_source)
        if self.cloud_config and self.cloud_config.requester_pays:
            if not CloudProvider.check_aws_credentials():
                raise ValueError(
                    f"Data source '{data_source}' requires valid AWS credentials"
                )

        # Generate name
        if custom_name and date_range:
            custom_name = Collection.create_name(
                custom_name=custom_name, date_range=date_range, data_source=data_source
            )

            self.custom_name = custom_name

        # Check if collection exists
        if custom_name:
            collection_path = self.output_dir / f"{custom_name}_stac"
            if collection_path.exists():
                logger.info(f"Loading existing collection: {custom_name}")
                self._collection = Collection.from_local(collection_path)
            else:
                logger.warning(
                    f"Collection '{custom_name}' not found. "
                    "Use create_index() to initialize collection."
                )
                self._collection = None
        else:
            self._collection = None

        # Initialize cloud provider if needed
        self.cloud_config = CLOUD_CONFIG.get(data_source)
        self.provider = None
        if self.cloud_config and self.cloud_config.requester_pays:
            self.provider = AWSProvider(
                profile=aws_profile, region=self.cloud_config.region
            )
            logger.info(f"Using {self.provider} as cloud provider")

    def _get_collection_path(self) -> Path:
        """Get expected collection path"""
        return self.output_dir / f"{self.custom_name}.parquet"

    def _get_bbox_from_geometries(self, geometries: List[Polygon]) -> List[float]:
        """Get combined bbox from geometries"""
        bounds = [geom.bounds for geom in geometries]
        return [
            min(b[0] for b in bounds),  # minx
            min(b[1] for b in bounds),  # miny
            max(b[2] for b in bounds),  # maxx
            max(b[3] for b in bounds),  # maxy
        ]

    def _ensure_collection(self, geometries: List[Polygon], **filters) -> None:
        """Ensure collection exists and is loaded with proper partitioning."""
        stac_path = self.output_dir / f"{self.custom_name}_stac"

        if self._collection is None:
            if stac_path.exists():
                try:
                    # Use partitioned dataset loading
                    self._collection = Collection.from_local(stac_path)
                    logger.info(f"Loaded collection from {stac_path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load collection: {e}")

            # No valid collection found
            bbox = self._get_bbox_from_geometries(geometries)
            error_msg = (
                f"\nNo valid collection found at: {stac_path}\n"
                f"\nTo create collection run:\n"
                f"processor.create_index(\n"
                f"    bbox={bbox},\n"
                f"    date_range=['YYYY-MM-DD', 'YYYY-MM-DD'],\n"
                f"    query={filters}\n"
                f")"
            )
            raise ValueError(error_msg)

    def create_collection(
        self, bbox: List[float], date_range: List[str], force: bool = False, **filters
    ) -> None:
        """
        Create or load STAC index.

        Args:
            bbox: Bounding box
            date_range: Date range
            query: Optional STAC query
            force: If True, recreate index even if exists
        """
        stac_path = self.output_dir / f"{self.custom_name}_stac"
        output_path = self.output_dir / f"{self.custom_name}_outputs"

        # Check if collection exists
        if stac_path.exists() and not force:
            logger.info(f"Collection {self.custom_name} exists, loading from disk")
            self._collection = Collection.from_local(stac_path)
            return

        # Create new collection
        indexer = StacToGeoParquetIndexer(
            data_source=self.data_source,
            stac_api=STAC_ENDPOINTS[self.data_source],
            output_dir=stac_path,
            cloud_provider=self.provider,
            cloud_config=self.cloud_config,
            name=self.custom_name,
        )

        self._collection = asyncio.run(
            indexer.build_index(bbox=bbox, date_range=date_range, query=filters)
        )

        output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def list_collections(self, dir) -> List[Dict]:
        """List available collections in directory."""

        if not Path(dir).exists():
            raise FileNotFoundError(f"Directory {dir} not found")
        if dir is None:
            logger.warning("No output directory provided, check default location")
            dir = Path.home() / "rasteret_workspace"

        return Collection.list_collections(output_dir=dir)

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
        self._ensure_collection(geometries, **filters)

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

        self._ensure_collection(geometries, **filters)

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
        return merged

    def __repr__(self):
        return (
            f"Rasteret(data_source={self.data_source}, custom_name={self.custom_name})"
        )
