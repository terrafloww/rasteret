"""Type definitions used throughout Rasteret."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import pyarrow as pa
import numpy as np

# Type aliases
BoundingBox = Tuple[float, float, float, float]  # minx, miny, maxx, maxy
DateRange = Tuple[str, str]  # ("YYYY-MM-DD", "YYYY-MM-DD")
Transform = List[float]  # Affine transform coefficients


@dataclass
class CogMetadata:
    """
    Metadata for a Cloud-Optimized GeoTIFF.

    Attributes:
        width (int): Image width in pixels
        height (int): Image height in pixels
        tile_width (int): Internal tile width
        tile_height (int): Internal tile height
        dtype (Union[np.dtype, pa.DataType]): Data type
        crs (int): Coordinate reference system code
        predictor (Optional[int]): Compression predictor
        transform (Optional[List[float]]): Affine transform coefficients
        compression (Optional[int]): Compression type
        tile_offsets (Optional[List[int]]): Byte offsets to tiles
        tile_byte_counts (Optional[List[int]]): Size of each tile
        pixel_scale (Optional[Tuple[float, ...]]): Resolution in CRS units
        tiepoint (Optional[Tuple[float, ...]]): Reference point coordinates
    """

    width: int
    height: int
    tile_width: int
    tile_height: int
    dtype: Union[np.dtype, pa.DataType]
    crs: int
    predictor: Optional[int] = None
    transform: Optional[List[float]] = None
    compression: Optional[int] = None
    tile_offsets: Optional[List[int]] = None
    tile_byte_counts: Optional[List[int]] = None
    pixel_scale: Optional[Tuple[float, ...]] = None
    tiepoint: Optional[Tuple[float, ...]] = None


@dataclass
class SceneInfo:
    """Metadata for a single scene.

    Attributes:
        id (str): Scene ID of Scene from STAC
        datetime (datetime): Datetime of Scene from STAC
        bbox (List[float]): Bounds of Scene from STAC
        scene_geometry (Any): Geometry of the scene (shapely geometry)
        crs (int): Coordinate reference system code
        cloud_cover (float): Cloud cover percentage
        assets (Dict[str, Any]): Assets associated with the scene
        metadata (Dict[str, Any]): Additional metadata
        collection (str): Collection to which the scene belongs
    """

    id: str
    datetime: datetime
    bbox: List[float]
    scene_geometry: Any
    crs: int
    cloud_cover: float
    assets: Dict[str, Any]
    metadata: Dict[str, Any]
    collection: str
