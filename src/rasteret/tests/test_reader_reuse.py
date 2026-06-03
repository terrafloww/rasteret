# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from datetime import datetime

import numpy as np
import pyarrow as pa

from rasteret.core.execution import get_collection_numpy
from rasteret.ingest.normalize import build_collection_from_table


def _collection_for_reader_reuse():
    meta = {
        "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
        "image_width": 2,
        "image_height": 2,
        "tile_width": 2,
        "tile_height": 2,
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
                    {"B04": {"href": "https://example.com/s1.tif"}},
                    {"B04": {"href": "https://example.com/s2.tif"}},
                ]
            ),
            "bbox": pa.array(
                [
                    {"xmin": 0.0, "ymin": 0.0, "xmax": 2.0, "ymax": 2.0},
                    {"xmin": 0.0, "ymin": 0.0, "xmax": 2.0, "ymax": 2.0},
                ],
                type=pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
            "proj:epsg": pa.array([4326, 4326], type=pa.int32()),
            "B04_metadata": pa.array([meta, meta]),
        }
    )
    return build_collection_from_table(table, name="reader-reuse-demo")


def test_get_collection_numpy_reuses_shared_reader(monkeypatch) -> None:
    """All rasters in a single get_collection_numpy call share the same reader."""
    collection = _collection_for_reader_reuse()
    seen_readers: list[object] = []
    fake_reader = object()

    class FakeReaderPool:
        reader = fake_reader

        def run(self, coro):
            import asyncio

            return asyncio.run(coro)

        def close(self):
            pass

    class FakeRaster:
        def __init__(self, raster_id: str):
            self.id = raster_id

        async def load_bands(self, *args, **kwargs):
            seen_readers.append(kwargs["reader"])
            return [([{"band": "B04", "data": np.ones((1, 1), dtype=np.uint16)}], 1)]

    async def fake_iterate_rasters(self, data_source=None, bands=None):
        _ = self, data_source, bands
        yield FakeRaster("scene-1")
        yield FakeRaster("scene-2")

    monkeypatch.setattr(
        "rasteret.core.collection.Collection.iterate_rasters",
        fake_iterate_rasters,
    )

    get_collection_numpy(
        collection=collection,
        geometries=(0.0, 0.0, 1.0, 1.0),
        bands=["B04"],
        reader_pool=FakeReaderPool(),
    )

    assert len(seen_readers) == 2
    assert seen_readers[0] is seen_readers[1]
    assert seen_readers[0] is fake_reader
