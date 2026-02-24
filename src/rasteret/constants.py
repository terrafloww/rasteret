# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa

# ---------------------------------------------------------------------------
# Band definitions for built-in data sources
# ---------------------------------------------------------------------------

SENTINEL2_BANDS: dict[str, str] = {
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

LANDSAT_C2L2_BANDS: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# BandRegistry: extensible mapping of collection id → band names
# ---------------------------------------------------------------------------


class BandRegistry:
    """Registry of collection → band-name mappings.

    Built-in collections (Sentinel-2, Landsat) are pre-registered.
    Users can register custom collections at any time::

        from rasteret.constants import BandRegistry

        BandRegistry.register("my-collection", {
            "B1": "red",
            "B2": "green",
            "B3": "blue",
        })
    """

    _maps: ClassVar[dict[str, dict[str, str]]] = {}

    @classmethod
    def register(cls, collection_id: str, band_map: dict[str, str]) -> None:
        """Register a band mapping for a collection."""
        cls._maps[collection_id] = band_map

    @classmethod
    def get(
        cls,
        collection_id: str,
        default: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Look up band mapping for *collection_id*, returning *default* if missing."""
        return cls._maps.get(collection_id, default or {})

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return all registered data-source ids."""
        return list(cls._maps.keys())


# Pre-register built-in data sources.
BandRegistry.register("sentinel-2-l2a", SENTINEL2_BANDS)
BandRegistry.register("landsat-c2-l2", LANDSAT_C2L2_BANDS)


# ---------------------------------------------------------------------------
# DataSources: namespace with well-known collection ids
# ---------------------------------------------------------------------------


class DataSources:
    """Well-known data-source identifiers."""

    LANDSAT = "landsat-c2-l2"
    SENTINEL2 = "sentinel-2-l2a"

    @classmethod
    def list_sources(cls) -> list[str]:
        """List all registered data sources (built-in + user-added)."""
        return BandRegistry.list_registered()


# ---------------------------------------------------------------------------
# Parquet / COG metadata schema constants
# ---------------------------------------------------------------------------

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
        # Extended metadata (added for correctness -- older Parquet files
        # missing these fields load with safe defaults via CogMetadata.from_dict).
        ("nodata", pa.float64()),
        ("samples_per_pixel", pa.int32()),
        ("planar_configuration", pa.int32()),
        ("photometric", pa.int32()),
        ("extra_samples", pa.list_(pa.int32())),
    ]
)
