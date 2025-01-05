""" Indexer for creating GeoParquet collections from STAC catalogs. """

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pystac_client
import stac_geoparquet
from shapely.geometry import shape

from rasteret.stac.parser import AsyncCOGHeaderParser
from rasteret.cloud import CloudProvider, CloudConfig
from rasteret.logging import setup_logger
from rasteret.types import BoundingBox, DateRange
from rasteret.core.collection import Collection
from rasteret.constants import STAC_COLLECTION_BAND_MAPS, COG_BAND_METADATA_STRUCT

logger = setup_logger("INFO", customname="rasteret.stac.indexer")


class StacToGeoParquetIndexer:
    """Creates searchable GeoParquet collections from STAC catalogs."""

    def __init__(
        self,
        data_source: str,
        stac_api: str,
        output_dir: Optional[Path] = None,
        name: Optional[str] = None,
        cloud_provider: Optional[CloudProvider] = None,
        cloud_config: Optional[CloudConfig] = None,
        max_concurrent: int = 50,
    ):
        self.data_source = data_source
        self.stac_api = stac_api
        self.output_dir = output_dir
        self.cloud_provider = cloud_provider
        self.cloud_config = cloud_config
        self.name = name
        self.max_concurrent = max_concurrent

    @property
    def band_map(self) -> Dict[str, str]:
        """Get band mapping for current collection."""
        return STAC_COLLECTION_BAND_MAPS.get(self.data_source, {})

    async def build_index(
        self,
        bbox: Optional[BoundingBox] = None,
        date_range: Optional[DateRange] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """
        Build GeoParquet collection from STAC search.

        Args:
            bbox: Bounding box filter
            date_range: Date range filter
            query: Additional query parameters

        Returns:
            Created Collection
        """
        logger.info("Starting STAC index creation...")
        if bbox:
            logger.info(f"Spatial filter: {bbox}")
        if date_range:
            logger.info(f"Temporal filter: {date_range[0]} to {date_range[1]}")
        if query:
            logger.info(f"Additional query parameters: {query}")

        # 1. Get STAC items
        stac_items = await self._search_stac(bbox, date_range, query)
        logger.info(f"Found {len(stac_items)} scenes in STAC catalog")

        # 2. Process in batches, adding COG metadata
        processed_items = []
        batch_size = 10
        total_batches = (len(stac_items) + batch_size - 1) // batch_size

        logger.info(
            f"Processing {len(stac_items)} scenes (each scene has multiple bands)..."
        )

        async with AsyncCOGHeaderParser(
            max_concurrent=self.max_concurrent,
            cloud_provider=self.cloud_provider,
            cloud_config=self.cloud_config,
        ) as cog_parser:

            for i in range(0, len(stac_items), batch_size):
                batch = stac_items[i : i + batch_size]
                batch_records = await self._process_batch(batch, cog_parser)
                if batch_records:
                    processed_items.extend(batch_records)
                logger.info(
                    f"Processed scene batch {(i//batch_size)+1}/{total_batches} yielding {len(batch_records)} band assets"
                )

        total_assets = sum(len(item["assets"]) for item in stac_items)
        logger.info(
            f"Completed processing {len(stac_items)} scenes with {len(processed_items)}/{total_assets} band assets"
        )

        logger.info(f"Successfully processed {len(processed_items)} items")

        try:
            logger.info("Creating GeoParquet table with metadata...")
            # First create json file with STAC items
            temp_ndjson = Path(f"/tmp/stac_items_{datetime.now().timestamp()}.ndjson")
            with open(temp_ndjson, "w") as f:
                for item in stac_items:  # Use original STAC items
                    json.dump(item, f)
                    f.write("\n")

            # Create temporary parquet with stac-geoparquet
            temp_parquet = Path(f"/tmp/temp_stac_{datetime.now().timestamp()}.parquet")
            stac_geoparquet.arrow.parse_stac_ndjson_to_parquet(
                temp_ndjson, temp_parquet
            )

            # Read and enrich parquet table
            stac_table = pq.read_table(temp_parquet)

            if not pa.types.is_timestamp(stac_table["datetime"].type):
                stac_table = stac_table.append_column(
                    "datetime",
                    pa.array(
                        [
                            datetime.fromisoformat(str(d.as_py()))
                            for d in stac_table["datetime"]
                        ],
                        type=pa.timestamp("us"),
                    ),
                )

            logger.info("Adding time columns...")
            # Add time columns
            datetime_col = stac_table.column("datetime")
            if not pa.types.is_timestamp(datetime_col.type):
                datetime_col = pa.array(
                    [datetime.fromisoformat(str(d.as_py())) for d in datetime_col],
                    type=pa.timestamp("us"),
                )

            table = stac_table.append_column(
                "year", pa.compute.year(datetime_col)
            ).append_column("month", pa.compute.month(datetime_col))

            logger.info("Adding scene bounding boxes...")
            scene_bboxes = {}
            for item in stac_items:
                polygon = shape(item["geometry"])
                scene_bboxes[item["id"]] = list(polygon.bounds)

            bbox_list = [scene_bboxes[id_] for id_ in table.column("id").to_pylist()]
            table = table.append_column(
                "scene_bbox", pa.array(bbox_list, type=pa.list_(pa.float64(), 4))
            )

            logger.info("Adding band metadata...")
            # Add band metadata columns
            scene_metadata = {}
            for scene_id in table.column("id").to_pylist():
                scene_metadata[scene_id] = {band: None for band in self.band_map.keys()}

            for item in processed_items:
                if (
                    "scene_id" not in item
                ):  # Handle case where scene_id might be missing
                    continue
                scene_id = item["scene_id"]
                band = item["band"]
                if scene_id in scene_metadata:
                    scene_metadata[scene_id][band] = {
                        "image_width": item["width"],
                        "image_height": item["height"],
                        "tile_width": item["tile_width"],
                        "tile_height": item["tile_height"],
                        "dtype": item["dtype"],
                        "transform": item.get("transform", []),
                        "predictor": item["predictor"],
                        "compression": item["compression"],
                        "tile_offsets": item["tile_offsets"],
                        "tile_byte_counts": item["tile_byte_counts"],
                        "pixel_scale": item.get("pixel_scale", []),
                        "tiepoint": item.get("tiepoint", []),
                    }
                    logger.debug(f"Added metadata for scene {scene_id} band {band}")

            for band in self.band_map.keys():
                metadata_list = [
                    scene_metadata[id_][band] for id_ in table.column("id").to_pylist()
                ]
                table = table.append_column(
                    f"{band}_metadata",
                    pa.array(metadata_list, type=COG_BAND_METADATA_STRUCT),
                )

            logger.info("Creating final collection...")
            # Create collection
            collection = Collection(
                dataset=ds.dataset(table),  # Create dataset from table
                name=self.name,
                description="STAC collection indexed from {self.data_source}",
            )

            # Optionally write to disk
            if self.output_dir:
                logger.info(f"Saving collection to {self.output_dir}")
                collection.save_to_parquet(self.output_dir)

            logger.info("Index creation completed successfully")
            return collection

        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise
        finally:
            # Cleanup temp files
            logger.debug("Cleaning up temporary files...")
            if temp_ndjson.exists():
                temp_ndjson.unlink()
            if temp_parquet.exists():
                temp_parquet.unlink()

    async def _search_stac(
        self,
        bbox: Optional[BoundingBox] = None,
        date_range: Optional[DateRange] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """
        Search STAC API for items.

        Returns:
            List of STAC items
        """

        # Build search parameters
        search_params = {"collections": [self.data_source], "limit": None}
        if bbox:
            search_params["bbox"] = bbox
        if date_range:
            search_params["datetime"] = f"{date_range[0]}/{date_range[1]}"
        if query is not None:
            search_params["query"] = query

        # Initialize STAC client and search
        client = pystac_client.Client.open(self.stac_api)
        search = client.search(**search_params)

        items = []
        for item in search.items():
            items.append(item.to_dict())

        logger.info(f"Found {len(items)} scenes")
        return items

    def _get_asset_url(self, asset: Dict) -> str:
        """Get authenticated URL for asset"""
        url = asset["href"] if isinstance(asset, dict) else asset
        if self.cloud_provider and self.cloud_config:
            return self.cloud_provider.get_url(url, self.cloud_config)
        return url

    async def _process_batch(
        self, stac_items: List[dict], cog_parser: AsyncCOGHeaderParser
    ) -> List[dict]:
        """
        Add COG metadata to STAC items.
        """
        urls_to_process = []
        url_mapping = {}  # Track which url belongs to which item/band

        for item in stac_items:
            item_id = item.get("id")
            if not item_id:
                continue

            for band_code, asset_name in self.band_map.items():
                if asset_name not in item["assets"]:
                    continue

                asset = item["assets"][asset_name]
                url = self._get_asset_url(asset)
                if url:
                    urls_to_process.append(url)
                    url_mapping[url] = (item_id, band_code, item)

        # Get COG metadata for all URLs
        metadata_results = await cog_parser.process_cog_headers_batch(urls_to_process)

        # Enrich items with metadata
        processed_items = {}

        for url, metadata in zip(urls_to_process, metadata_results):
            if not metadata:
                continue

            item_id, band_code, item = url_mapping[url]

            if item_id not in processed_items:
                processed_items[item_id] = {
                    "id": item_id,
                    "scene_id": item_id,
                    "geometry": item["geometry"],
                    "datetime": item["properties"].get("datetime"),
                    "cloud_cover": item["properties"].get("eo:cloud_cover"),
                    "bands": {},
                }

            processed_items[item_id]["bands"][band_code] = {
                "width": metadata.width,
                "height": metadata.height,
                "tile_width": metadata.tile_width,
                "tile_height": metadata.tile_height,
                "dtype": str(metadata.dtype),
                "transform": metadata.transform,
                "predictor": metadata.predictor,
                "compression": metadata.compression,
                "tile_offsets": metadata.tile_offsets,
                "tile_byte_counts": metadata.tile_byte_counts,
                "pixel_scale": metadata.pixel_scale,
                "tiepoint": metadata.tiepoint,
            }

        # Convert to enriched items list with proper band metadata structure
        enriched_items = []
        for item_id, item_data in processed_items.items():
            for band_code, band_metadata in item_data["bands"].items():
                enriched_items.append(
                    {
                        "scene_id": item_id,
                        "band": band_code,
                        "geometry": item_data["geometry"],
                        "datetime": item_data["datetime"],
                        "cloud_cover": item_data["cloud_cover"],
                        **band_metadata,
                    }
                )

        return enriched_items
