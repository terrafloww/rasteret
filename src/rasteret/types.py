# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa

# Type aliases
BoundingBox = tuple[float, float, float, float]  # minx, miny, maxx, maxy
DateRange = tuple[str, str]  # ("YYYY-MM-DD", "YYYY-MM-DD")

POINT_SAMPLES_SCHEMA = pa.schema(
    [
        pa.field("point_index", pa.int64(), nullable=False),
        pa.field("point_x", pa.float64(), nullable=False),
        pa.field("point_y", pa.float64(), nullable=False),
        pa.field("point_crs", pa.int32(), nullable=True),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("datetime", pa.timestamp("us"), nullable=True),
        pa.field("collection", pa.string(), nullable=False),
        pa.field("cloud_cover", pa.float64(), nullable=True),
        pa.field("band", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("raster_crs", pa.int32(), nullable=True),
    ]
)


# Optional extension schema used by `sample_points(return_neighbourhood=True)`.
# This keeps the scalar sample (the `value` column) while also returning the
# full pixel neighborhood around each point as a 1D list in row-major order.
POINT_SAMPLES_NEIGHBORHOOD_SCHEMA = POINT_SAMPLES_SCHEMA.append(
    pa.field("neighborhood_values", pa.list_(pa.float64()), nullable=False)
)


def empty_point_samples_table() -> pa.Table:
    """Return an empty point-sampling table with stable schema."""
    return pa.table(
        {field.name: pa.array([], type=field.type) for field in POINT_SAMPLES_SCHEMA}
    )


def empty_point_samples_neighborhood_table() -> pa.Table:
    """Return an empty point-sampling table with neighborhood schema."""
    return pa.table(
        {
            field.name: pa.array([], type=field.type)
            for field in POINT_SAMPLES_NEIGHBORHOOD_SCHEMA
        }
    )


#: Accepted geometry inputs for read operations.
#:
#: - ``(minx, miny, maxx, maxy)`` bbox tuple
#: - ``pa.Array`` / ``pa.ChunkedArray`` (WKB or native GeoArrow)
#: - ``bytes`` (single WKB) or ``list[bytes]``
#: - ``dict`` with ``"type"`` and ``"coordinates"`` (GeoJSON)
#: - Shapely geometries (detected at runtime without import)
GeometryInput = (
    BoundingBox
    | pa.Array
    | pa.ChunkedArray
    | bytes
    | list[bytes]
    | dict
    | tuple[float, float, float, float]
)


@dataclass
class CogMetadata:
    """Metadata for a tiled GeoTIFF (including COGs).

    New fields added for correctness (nodata, samples_per_pixel,
    planar_configuration, photometric, extra_samples) have defaults
    so existing Parquet files missing them load without error.
    """

    width: int
    height: int
    tile_width: int
    tile_height: int
    dtype: np.dtype | pa.DataType
    crs: int | None
    predictor: int | None = None
    transform: list[float] | None = None
    compression: int | None = None
    tile_offsets: list[int] | None = None
    tile_byte_counts: list[int] | None = None
    pixel_scale: tuple[float, ...] | None = None
    tiepoint: tuple[float, ...] | None = None
    nodata: float | int | None = None
    samples_per_pixel: int = 1
    planar_configuration: int = 1  # 1=chunky, 2=planar separate
    photometric: int | None = None
    extra_samples: tuple[int, ...] | None = None

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        crs: int | None = None,
        transform_override: list[float] | None = None,
    ) -> CogMetadata:
        """Build from a raw metadata dict (as stored in the Parquet table).

        Parameters
        ----------
        raw : dict
            The ``{band}_metadata`` dict from the collection schema.
        crs : int, optional
            EPSG code to use (overrides any value in *raw*).
        transform_override : list[float], optional
            Pre-normalized transform.  When ``None``, ``raw["transform"]``
            is used as-is.
        """
        # Parse nodata: stored as float64 in Parquet, may be NaN or None.
        raw_nodata = raw.get("nodata")
        nodata = None
        if raw_nodata is not None:
            fval = float(raw_nodata)
            if fval != fval:  # NaN sentinel
                nodata = float("nan")
            else:
                int_val = int(fval)
                nodata = int_val if float(int_val) == fval else fval

        extra = raw.get("extra_samples")

        return cls(
            width=raw.get("image_width", raw.get("width")),
            height=raw.get("image_height", raw.get("height")),
            tile_width=raw["tile_width"],
            tile_height=raw["tile_height"],
            dtype=np.dtype(raw["dtype"]),
            transform=transform_override
            if transform_override is not None
            else raw.get("transform"),
            crs=crs,
            tile_offsets=raw["tile_offsets"],
            tile_byte_counts=raw["tile_byte_counts"],
            predictor=raw.get("predictor"),
            compression=raw.get("compression"),
            pixel_scale=raw.get("pixel_scale"),
            tiepoint=raw.get("tiepoint"),
            nodata=nodata,
            samples_per_pixel=raw.get("samples_per_pixel", 1),
            planar_configuration=raw.get("planar_configuration", 1),
            photometric=raw.get("photometric"),
            extra_samples=tuple(extra) if extra is not None else None,
        )


@dataclass
class RasterInfo:
    """Metadata for a single raster record (Parquet row).

    Each row in a Rasteret Collection Parquet becomes a ``RasterInfo``
    that is then used to construct a :class:`~rasteret.core.raster_accessor.RasterAccessor`.

    Parameters
    ----------
    id : str
        Unique record identifier.
    datetime : datetime
        Acquisition / observation time.
    bbox : list of float
        Bounding box ``[minx, miny, maxx, maxy]``.
    footprint : object
        Footprint geometry (Shapely Polygon or WKB bytes).
    crs : int or None
        EPSG code for the record's native CRS.
    cloud_cover : float
        Cloud cover percentage (0--100).
    assets : dict
        Mapping of band code to asset dict (href, raster:bands, etc.).
    band_metadata : dict
        Per-band COG metadata dicts keyed by ``{band}_metadata``.
    collection : str
        Parent collection identifier.
    """

    id: str
    datetime: datetime
    bbox: list[float]
    footprint: Any
    crs: int | None
    cloud_cover: float
    assets: dict[str, Any]
    band_metadata: dict[str, Any]
    collection: str
