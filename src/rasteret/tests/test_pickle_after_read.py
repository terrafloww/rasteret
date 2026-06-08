# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""A Collection must stay picklable after a read has created a live reader pool.

This is the DataLoader(num_workers>0) path: a user does a read in the main
process (which lazily creates the background reader pool), then hands the
dataset to a DataLoader, which pickles it to ship to worker processes. The
pool owns a thread + asyncio loop and cannot be pickled, so Collection drops
it in ``__getstate__`` and each worker rebuilds its own on first read.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from affine import Affine

from rasteret.core.collection import Collection
from rasteret.core.reader_pool import AsyncCOGReaderPool
from rasteret.ingest.normalize import build_collection_from_table

_META = {
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
    "nodata": 0.0,
}


def _file_backed_collection(tmp_path: Path) -> Collection:
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 15)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/scene-1.tif"}}]),
            "proj:epsg": pa.array([32615], type=pa.int32()),
            "B04_metadata": pa.array([_META]),
        }
    )
    # Normalize to the full schema, then persist so the dataset itself is
    # picklable (in-memory datasets are not). This mirrors how build()/load()
    # produce file-backed collections in real use.
    normalized = build_collection_from_table(table, name="pickle-after-read")
    path = tmp_path / "collection.parquet"
    pq.write_table(normalized.dataset.to_table(), path)
    return Collection(dataset=pads.dataset(str(path)), data_source="pickle-after-read")


def test_collection_picklable_after_real_pool_created(monkeypatch, tmp_path) -> None:
    collection = _file_backed_collection(tmp_path)

    # Fake the COG byte read only — the real AsyncCOGReaderPool is still
    # created, so we exercise the actual unpicklable object, not a stub.
    async def fake_read_cog(*args, **kwargs):
        return type(
            "Result",
            (),
            {
                "data": np.array([[1, 1], [1, 1]], dtype=np.uint16),
                "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
            },
        )()

    monkeypatch.setattr("rasteret.core.window_read.read_cog", fake_read_cog)

    read_kwargs = dict(
        record_ids=["scene-1"],
        bounds=(0.0, 0.0, 2.0, 2.0),
        res=(1.0, 1.0),
        bands=["B04"],
    )

    restored: Collection | None = None
    try:
        arr = collection.read_window(**read_kwargs)
        # A real, live, unpicklable pool now exists on the collection.
        assert isinstance(collection._reader_pool, AsyncCOGReaderPool)

        # Without __getstate__ this raises "cannot pickle '_thread.lock'".
        restored = pickle.loads(pickle.dumps(collection))
        assert restored._reader_pool is None

        # The restored collection rebuilds its pool lazily and reads correctly.
        arr2 = restored.read_window(**read_kwargs)
        np.testing.assert_array_equal(arr, arr2)
        assert isinstance(restored._reader_pool, AsyncCOGReaderPool)
    finally:
        collection._close_reader_pool()
        if restored is not None:
            restored._close_reader_pool()


def test_fork_path_does_not_close_inherited_pool(monkeypatch, tmp_path) -> None:
    """_close_reader_pool must not call pool.close() when called from a child process.

    After fork() the background thread is dead, so calling pool.close() would
    deadlock waiting on a loop lock the absent thread still holds. The pid-check
    in _close_reader_pool guards against this by only closing pools the current
    process created.
    """
    collection = _file_backed_collection(tmp_path)

    close_calls: list[str] = []

    class _TrackingPool:
        """Minimal stand-in that records whether close() was called."""

        def close(self) -> None:
            close_calls.append("close")

    pool = _TrackingPool()
    collection._reader_pool = pool  # type: ignore[assignment]
    collection._reader_pool_pid = os.getpid() + 1  # simulate a different (parent) PID

    collection._close_reader_pool()

    assert not close_calls, "pool.close() must not be called for an inherited pool"
    assert collection._reader_pool is None
