# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pytest

from rasteret.ingest.normalize import build_collection_from_table


def _collection_with_res(res_x: float, res_y: float, *, band: str = "B04"):
    """A minimal offline collection whose {band}_metadata encodes a known res."""
    # GDAL/rasterio 6-value affine: (a, b, c, d, e, f) north-up => a=res_x, e=-res_y
    meta = {
        "transform": [res_x, 0.0, 399960.0, 0.0, -res_y, 4500000.0],
        "image_width": 10980,
        "image_height": 10980,
        "tile_width": 512,
        "tile_height": 512,
        "dtype": "uint16",
        "compress": 8,
        "predictor": 1,
        "tile_offsets": [0],
        "tile_byte_counts": [1],
    }
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2"]),
            "datetime": pa.array(
                [datetime(2024, 1, 15), datetime(2024, 1, 16)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {band: {"href": "https://example.com/s1.tif"}},
                    {band: {"href": "https://example.com/s2.tif"}},
                ]
            ),
            "proj:epsg": pa.array([32632, 32632], type=pa.int32()),
            f"{band}_metadata": pa.array([meta, meta]),
        }
    )
    return build_collection_from_table(table, name="native-res-demo")


def test_native_res_reads_pixel_size_from_metadata() -> None:
    collection = _collection_with_res(10.0, 10.0)
    assert collection.native_res() == (10.0, 10.0)


def test_native_res_returns_positive_magnitudes_for_anisotropic_grid() -> None:
    collection = _collection_with_res(20.0, 60.0)
    # y scale is stored negative (north-up); native_res must return magnitudes.
    assert collection.native_res("B04") == (20.0, 60.0)


def test_native_res_defaults_to_first_band() -> None:
    collection = _collection_with_res(30.0, 30.0, band="SR_B4")
    assert collection.native_res() == (30.0, 30.0)


def test_native_res_rejects_unknown_band() -> None:
    collection = _collection_with_res(10.0, 10.0)
    with pytest.raises(ValueError):
        collection.native_res("B99")
