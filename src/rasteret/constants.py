"""Constants and configurations for rasteret."""

from typing import Dict
import pyarrow as pa


class DataSources:
    """Registry of supported data sources based on STAC endpoint collection names."""

    LANDSAT = "landsat-c2l2-sr"
    SENTINEL2 = "sentinel-2-l2a"

    @classmethod
    def list_sources(cls) -> Dict[str, str]:
        """List available data sources with descriptions."""
        return {
            cls.LANDSAT: "Landsat Collection 2 Level 2 Surface Reflectance",
            cls.SENTINEL2: "Sentinel-2 Level 2A",
        }


SENTINEL2_BANDS: Dict[str, str] = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}

LANDSAT9_BANDS: Dict[str, str] = {
    "B1": "coastal",
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "nir08",
    "B6": "swir16",
    "B7": "swir22",
    "qa_aerosol": "qa_aerosol",
    "qa_pixel": "qa_pixel",
    "qa_radsat": "qa_radsat",
}


STAC_ENDPOINTS = {
    DataSources.SENTINEL2: "https://earth-search.aws.element84.com/v1",
    DataSources.LANDSAT: "https://landsatlook.usgs.gov/stac-server",
}

STAC_COLLECTION_BAND_MAPS = {
    DataSources.SENTINEL2: SENTINEL2_BANDS,
    DataSources.LANDSAT: LANDSAT9_BANDS,
    "LANDSAT": LANDSAT9_BANDS,
    "SENTINEL2": SENTINEL2_BANDS,
}

# Metadata struct for COG headers
COG_BAND_METADATA_STRUCT = pa.struct(
    [
        ("image_width", pa.int32()),
        ("image_height", pa.int32()),
        ("tile_width", pa.int32()),
        ("tile_height", pa.int32()),
        ("dtype", pa.string()),
        ("transform", pa.list_(pa.float64())),
        ("predictor", pa.int32()),
        ("compression", pa.int32()),
        ("tile_offsets", pa.list_(pa.int64())),
        ("tile_byte_counts", pa.list_(pa.int64())),
        ("pixel_scale", pa.list_(pa.float64())),
        ("tiepoint", pa.list_(pa.float64())),
    ]
)

# Default partition settings
DEFAULT_GEO_PARQUET_SETTINGS = {
    "compression": "zstd",
    "compression_level": 3,
    "row_group_size": 20 * 1024 * 1024,
    "write_statistics": True,
    "use_dictionary": True,
    "write_batch_size": 10000,
    "basename_template": "part-{i}.parquet",
}
