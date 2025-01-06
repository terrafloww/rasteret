""" Collection class for managing raster data collections. """

from __future__ import annotations
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon
from pathlib import Path
from typing import AsyncIterator, Optional, Union, Dict, List, Any, Tuple

from rasteret.types import SceneInfo
from rasteret.core.scene import Scene
from rasteret.logging import setup_logger

logger = setup_logger("INFO", customname="rasteret.core.collection")


class Collection:
    """
    A collection of raster data with flexible initialization.

    Collections can be  created from:
    - Local partitioned datasets
    - Single Arrow tables
    - Empty (for building gradually)

    Collections maintain efficient partitioned storage when using files.

    Examples
    --------
    # From partitioned dataset
    >>> collection = Collection.from_local("path/to/dataset")

    # Filter and process
    >>> filtered = collection.filter_scenes(cloud_cover_lt=20)
    >>> ds = filtered.get_xarray(...)
    """

    def __init__(
        self,
        dataset: Optional[ds.Dataset] = None,
        name: str = "",
        description: str = "",
        data_source: str = "",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """Initialize Collection."""
        self.dataset = dataset
        self.name = name
        self.description = description
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self._validate_parquet_dataset()

    @classmethod
    def from_local(cls, path: Union[str, Path]) -> Collection:
        """
        Create collection from local partitioned dataset.

        Parameters
        ----------
        path : str or Path
            Path to dataset directory with Hive-style partitioning
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        try:
            dataset = ds.dataset(
                str(path),
                format="parquet",
                partitioning=ds.HivePartitioning(
                    pa.schema([("year", pa.int32()), ("month", pa.int32())])
                ),
                exclude_invalid_files=True,
                filesystem=pa.fs.LocalFileSystem(),
            )
        except Exception as e:
            raise ValueError(f"Invalid dataset at {path}: {str(e)}")

        return cls(dataset=dataset, name=path.name)

    def filter_scenes(self, **kwargs) -> Collection:
        """
        Filter collection creating new view.

        Parameters
        ----------
        **kwargs :
            Supported filters:
            - cloud_cover_lt: float
            - date_range: Tuple[str, str]
            - bbox: Tuple[float, float, float, float]
        """
        filter_expr = None

        # Build filter expression
        if len(self.dataset.to_table()) == 0:
            return self

        if "cloud_cover_lt" in kwargs:
            if "eo:cloud_cover" not in self.dataset.schema.names:
                raise ValueError("Collection has no cloud cover data")

            if not isinstance(kwargs["cloud_cover_lt"], (int, float)):
                raise ValueError("Invalid cloud cover value")
            elif kwargs["cloud_cover_lt"] < 0 or kwargs["cloud_cover_lt"] > 100:
                raise ValueError("Invalid cloud cover value")

            filter_expr = ds.field("eo:cloud_cover") < kwargs["cloud_cover_lt"]

        if "date_range" in kwargs:
            if "datetime" not in self.dataset.schema.names:
                raise ValueError("Collection has no datetime data")

            start, end = kwargs["date_range"]

            if not (start and end):
                raise ValueError("Invalid date range")
            elif start > end:
                raise ValueError("Invalid date range")
            elif start == end:
                raise ValueError("Date range must be > 1 day")
            elif len(start) != 10 or len(end) != 10:
                raise ValueError("Date format must be 'YYYY-MM-DD'")

            start_ts = pd.Timestamp(start).tz_localize("UTC")
            end_ts = pd.Timestamp(end).tz_localize("UTC")

            # Convert to Arrow timestamps
            start_timestamp = pa.scalar(start_ts, type=pa.timestamp("us", tz="UTC"))
            end_timestamp = pa.scalar(end_ts, type=pa.timestamp("us", tz="UTC"))

            date_filter = (ds.field("datetime") >= start_timestamp) & (
                ds.field("datetime") <= end_timestamp
            )
            filter_expr = (
                date_filter if filter_expr is None else filter_expr & date_filter
            )

        if "bbox" in kwargs:
            if "scene_bbox" not in self.dataset.schema.names:
                raise ValueError("Collection has no bbox data")
            bbox = kwargs["bbox"]

            if len(bbox) != 4:
                raise ValueError("Invalid bbox format")
            elif bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                raise ValueError("Invalid bbox coordinates")
            elif any(not isinstance(coord, (int, float)) for coord in bbox):
                raise ValueError("Invalid bbox coordinates")

            bbox_filter = (
                (ds.field("scene_bbox").x0 >= bbox[0])
                & (ds.field("scene_bbox").y0 >= bbox[1])
                & (ds.field("scene_bbox").x1 <= bbox[2])
                & (ds.field("scene_bbox").y1 <= bbox[3])
            )
            filter_expr = (
                bbox_filter if filter_expr is None else filter_expr & bbox_filter
            )

        if "geometries" in kwargs:
            if "scene_bbox" not in self.dataset.schema.names:
                raise ValueError("Collection has no bbox data")
            geometries = kwargs["geometries"]

            if not all(isinstance(geom, Polygon) for geom in geometries):
                raise ValueError("Invalid geometry format")

            bbox_filters = [
                (ds.field("scene_bbox").x0 >= geom.bounds[0])
                & (ds.field("scene_bbox").y0 >= geom.bounds[1])
                & (ds.field("scene_bbox").x1 <= geom.bounds[2])
                & (ds.field("scene_bbox").y1 <= geom.bounds[3])
                for geom in geometries
            ]
            filter_expr = bbox_filters[0]
            for bbox_filter in bbox_filters[1:]:
                filter_expr |= bbox_filter

        if filter_expr is None:
            raise ValueError("No valid filters provided")

        filtered_dataset = self.dataset.filter(filter_expr)
        return Collection(dataset=filtered_dataset, name=self.name)

    @classmethod
    def list_collections(
        cls, workspace_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """List collections with metadata."""
        if workspace_dir is None:
            workspace_dir = Path.home() / "rasteret_workspace"

        collections = []

        # Look for _stac directories
        for stac_dir in workspace_dir.glob("*_stac"):
            try:
                dataset = ds.dataset(str(stac_dir))
                table = dataset.to_table()
                df = table.to_pandas()

                # Get collection name without _stac suffix
                name = stac_dir.name.replace("_stac", "")

                # Get date range from data
                if "datetime" in df.columns:
                    date_range = (
                        pd.to_datetime(df["datetime"].min()).strftime("%Y-%m-%d"),
                        pd.to_datetime(df["datetime"].max()).strftime("%Y-%m-%d"),
                    )
                else:
                    date_range = None

                # Get data source from name
                data_source = name.split("_")[-1] if "_" in name else "unknown"

                collections.append(
                    {
                        "name": name,
                        "data_source": data_source,
                        "date_range": date_range,
                        "size": len(df),
                        "created": stac_dir.stat().st_ctime,
                    }
                )

            except Exception as e:
                logger.debug(f"Failed to read collection {stac_dir}: {e}")
                continue

        return collections

    def save_to_parquet(
        self, path: Union[str, Path], partition_by: List[str] = ["year", "month"]
    ) -> None:
        """Save collection with enhanced metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.dataset is None:
            raise ValueError("No Pyarrow dataset provided")

        table = self.dataset.to_table()

        # Enhanced metadata with fallbacks
        custom_metadata = {
            b"description": (
                self.description.encode("utf-8") if self.description else b""
            ),
            b"created": str(datetime.now()).encode("utf-8"),
            b"custom_name": self.name.encode("utf-8") if self.name else b"",
            b"data_source": (
                self.data_source.encode("utf-8") if self.data_source else b"unknown"
            ),
            b"date_range": (
                f"{self.start_date.isoformat()},{self.end_date.isoformat()}".encode(
                    "utf-8"
                )
                if self.start_date and self.end_date
                else b""
            ),
            b"version": b"1.0.0",
        }

        # Merge with existing metadata
        merged_metadata = {**custom_metadata, **(table.schema.metadata or {})}
        table_with_metadata = table.replace_schema_metadata(merged_metadata)

        # Write dataset
        pq.write_to_dataset(
            table_with_metadata,
            root_path=str(path),
            partition_cols=partition_by,
            compression="zstd",
            compression_level=3,
            row_group_size=20 * 1024 * 1024,
            write_statistics=True,
            use_dictionary=True,
            write_batch_size=10000,
            basename_template="part-{i}.parquet",
        )

    async def iterate_scenes(self, data_source: str) -> AsyncIterator[Scene]:
        """
        Iterate through scenes.

        Args:
            data_source: Data source for the scenes

        Yields
        ------
        Scene
            Scene objects for processing
        """
        required_fields = {"id", "datetime", "geometry", "assets"}

        if len(self.dataset.to_table()) == 0:
            return

        # Check required fields
        missing = required_fields - set(self.dataset.schema.names)
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        for batch in self.dataset.to_batches():
            for row in batch.to_pylist():
                try:
                    scene_info = SceneInfo(
                        id=row["id"],
                        datetime=row["datetime"],
                        scene_geometry=row["geometry"],
                        bbox=row["scene_bbox"],
                        crs=row.get("proj:epsg", None),
                        cloud_cover=row.get("eo:cloud_cover", 0),
                        assets=row["assets"],
                        metadata=self._extract_band_metadata(row),
                        collection=row.get(
                            "collection", data_source
                        ),  # Use data_source as default collection
                    )
                    yield Scene(
                        scene_info, data_source
                    )  # Pass data_source to Scene constructor
                except Exception as e:
                    # Log error but continue with other scenes
                    logger.error(f"Error creating scene from row: {str(e)}")
                    continue

    async def get_first_scene(self) -> Scene:
        """
        Get first scene in collection.

        Returns
        -------
        Scene
            Scene object for processing
        """
        async for scene in self.iterate_scenes(data_source=self.name):
            return scene
        raise ValueError("No scenes found in collection")

    def _validate_parquet_dataset(self) -> None:
        """Basic dataset validation."""
        if not isinstance(self.dataset, ds.Dataset):
            raise TypeError("Expected pyarrow.dataset.Dataset")

    def _extract_band_metadata(self, row: Dict) -> Dict:
        """Extract band metadata from row."""
        return {k: v for k, v in row.items() if k.endswith("_metadata")}

    @classmethod
    def _format_date_range(
        cls, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> str:
        """Format date range for collection name."""
        if start_date.year == end_date.year:
            return f"{start_date.strftime('%Y%m')}-{end_date.strftime('%m')}"
        return f"{start_date.strftime('%Y%m')}-{end_date.strftime('%Y%m')}"

    @classmethod
    def create_name(
        cls, custom_name: str, date_range: Tuple[str, str], data_source: str
    ) -> str:
        """Create standardized collection name."""
        if "_" in custom_name:
            raise ValueError("Custom name cannot contain underscore (_)")

        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        name_parts = [
            custom_name.lower().replace(" ", "-"),
            cls._format_date_range(start_date, end_date),
            data_source.split("-")[0].lower(),
        ]
        return "_".join(name_parts)

    @classmethod
    def parse_name(cls, name: str) -> Dict[str, str]:
        """Parse collection name components."""
        try:
            # Remove _stac suffix if present
            name = name.replace("_stac", "")

            # Split parts
            parts = name.split("_")
            if len(parts) != 3:
                raise ValueError(f"Invalid name format: {name}")

            custom_name, date_str, source = parts

            # Parse date range
            date_parts = date_str.split("-")
            if len(date_parts) != 2:
                raise ValueError(f"Invalid date format: {date_str}")

            return {
                "custom_name": custom_name,
                "data_source": source,
                "name": name,  # Return full standardized name
            }

        except Exception as e:
            logger.debug(f"Failed to parse name {name}: {e}")
            return {"name": name, "custom_name": name, "data_source": "unknown"}
