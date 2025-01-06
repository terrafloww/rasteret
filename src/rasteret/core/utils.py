""" Utility functions for rasteret package. """

from typing import Optional
from urllib.parse import urlparse
import boto3
from pathlib import Path
from shapely.geometry import Polygon
from pyproj import Transformer
from shapely.ops import transform
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Dict, Union, List
from rasterio.warp import transform_bounds

from rasteret.logging import setup_logger

logger = setup_logger()


def wgs84_to_utm_convert_poly(geom: Polygon, epsg_code: int) -> Polygon:
    """
    Convert scene geometry to UTM.

    Parameters
    ----------
    geom : shapely Polygon
        Scene geometry in WGS84
    epsg_code : int
        UTM zone to convert to (e.g. 32643)

    Returns
    -------
    shapely Polygon
        Scene geometry in UTM zone
    """
    wgs84_to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg_code}", always_xy=True  # eg. 32643
    )
    utm_poly = transform(wgs84_to_utm.transform, geom)

    return utm_poly


class S3URLSigner:
    """Handle S3 URL signing with AWS credential chain."""

    def __init__(self, aws_profile: Optional[str] = None, region: str = "us-west-2"):
        self.region = region
        self.aws_profile = aws_profile
        self._session = None
        self._client = None

    @property
    def session(self):
        """Lazily create boto3 session."""
        if self._session is None:
            if self.aws_profile:
                self._session = boto3.Session(profile_name=self.aws_profile)
            else:
                self._session = boto3.Session()
        return self._session

    @property
    def client(self):
        """Lazily create S3 client."""
        if self._client is None:
            self._client = self.session.client("s3", region_name=self.region)
        return self._client

    def has_valid_credentials(self) -> bool:
        """Check if we have valid AWS credentials."""
        try:
            self.session.get_credentials()
            return True
        except Exception:
            raise ValueError("Missing AWS credentials")

    def get_signed_url(self, s3_uri: str) -> Optional[str]:
        """Get signed URL if credentials available."""
        try:
            parsed = urlparse(s3_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key, "RequestPayer": "requester"},
                ExpiresIn=3600,
            )
            return url

        except Exception as e:
            logger.debug(f"Failed to sign S3 URL {s3_uri}: {str(e)}")
            return None


class CloudStorageURLHandler:
    """Handle URL signing for different cloud storage providers."""

    def __init__(
        self,
        storage_platform: str,
        aws_profile: Optional[str] = None,
        aws_region: str = "us-west-2",
    ):
        self.storage_platform = storage_platform.upper()
        self._s3_signer = None
        self.aws_profile = aws_profile
        self.aws_region = aws_region

    def get_signed_url(self, url: str) -> Optional[str]:
        """Get signed URL based on storage platform."""
        if self.storage_platform == "AWS":
            if self._s3_signer is None:
                self._s3_signer = S3URLSigner(
                    aws_profile=self.aws_profile, region=self.aws_region
                )
            return self._s3_signer.get_signed_url(url)
        elif self.storage_platform in ["AZURE", "GCS"]:
            # These platforms are not supported yet
            logger.warning(f"Unsupported storage platform: {self.storage_platform}")
            return url
        else:
            logger.warning(f"Unknown storage platform: {self.storage_platform}")
            return url


def transform_bbox(
    bbox: Union[Tuple[float, float, float, float], Polygon],
    src_crs: Union[int, str],
    dst_crs: Union[int, str],
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box between coordinate systems.

    Args:
        bbox: Input bbox (minx, miny, maxx, maxy) or Polygon
        src_crs: Source CRS (EPSG code or WKT string)
        dst_crs: Target CRS (EPSG code or WKT string)

    Returns:
        Transformed bbox
    """
    if isinstance(bbox, Polygon):
        minx, miny, maxx, maxy = bbox.bounds
    else:
        minx, miny, maxx, maxy = bbox

    return transform_bounds(
        src_crs=f"EPSG:{src_crs}" if isinstance(src_crs, int) else src_crs,
        dst_crs=f"EPSG:{dst_crs}" if isinstance(dst_crs, int) else dst_crs,
        left=minx,
        bottom=miny,
        right=maxx,
        top=maxy,
    )


def calculate_scale_offset(
    arr: np.ndarray,
    target_dtype: np.dtype,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate optimal scale/offset for data type conversion.

    Args:
        arr: Input array
        target_dtype: Target numpy dtype
        valid_min: Optional minimum valid value
        valid_max: Optional maximum valid value

    Returns:
        Dict with scale and offset values
    """
    # Get data range
    if valid_min is None:
        valid_min = np.nanmin(arr)
    if valid_max is None:
        valid_max = np.nanmax(arr)

    # Get target dtype info
    target_info = np.iinfo(target_dtype)

    # Calculate scale and offset
    scale = (valid_max - valid_min) / (target_info.max - target_info.min)
    offset = valid_min - (target_info.min * scale)

    return {"scale": float(scale), "offset": float(offset)}


def save_per_geometry(
    ds: xr.Dataset, output_dir: Path, file_prefix: str = "ndvi", data_var: str = "NDVI"
) -> Dict[int, List[Path]]:
    """Save each geometry's timeseries as separate GeoTIFFs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Get CRS info
    crs = ds.attrs.get("crs", None)

    # Process each geometry
    for g in range(ds.geometry.size):
        # Extract geometry data
        geom_ds = ds.isel(geometry=g)
        geom_id = geom_ds.geometry.values.item()

        # Create geometry subfolder
        geom_dir = output_dir / f"geometry_{geom_id}"
        geom_dir.mkdir(exist_ok=True)

        # Process each timestamp
        for t in range(len(geom_ds.time)):
            # Extract 2D array for this timestamp
            time_data = geom_ds[data_var].isel(time=t)
            timestamp = pd.Timestamp(geom_ds.time[t].values)

            # Create 2D dataset
            ds_2d = xr.Dataset(
                data_vars={"NDVI": (("y", "x"), time_data.values)},
                coords={"y": geom_ds.y, "x": geom_ds.x},
            )

            # Set spatial metadata
            ds_2d.rio.write_crs(crs, inplace=True)
            if ds.rio.transform():
                ds_2d.rio.write_transform(ds.rio.transform(), inplace=True)

            # Save as GeoTIFF
            outfile = geom_dir / f"{file_prefix}_{timestamp.strftime('%Y%m%d')}.tif"
            ds_2d.rio.to_raster(outfile)
            outputs.setdefault(geom_id, []).append(outfile)

    return outputs
