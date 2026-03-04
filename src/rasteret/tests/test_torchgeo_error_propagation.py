# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from unittest.mock import patch

import pyarrow as pa
import pytest


def test_torchgeo_fetch_arrays_raises_first_error_when_all_fail() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    torchgeo = pytest.importorskip("rasteret.integrations.torchgeo")
    RasteretGeoDataset = getattr(torchgeo, "RasteretGeoDataset", None)
    if RasteretGeoDataset is None or not hasattr(RasteretGeoDataset, "_fetch_arrays"):
        pytest.skip("RasteretGeoDataset is not available in this environment")

    class _DummyPool:
        def __init__(self) -> None:
            self.reader = object()

        def run(self, coro):
            import asyncio

            return asyncio.run(coro)

    ds = object.__new__(RasteretGeoDataset)
    ds.epsg = 4326
    ds.max_concurrent = 5

    requests = [
        ("https://example.com/a.tif", object(), None),
        ("https://example.com/b.tif", object(), 0),
    ]
    patch_array = pa.array([None])
    boom = RuntimeError("boom")

    with patch.object(torchgeo, "read_cog", side_effect=boom):
        with pytest.raises(
            ValueError, match="All 2 COG read request\\(s\\) failed"
        ) as excinfo:
            ds._fetch_arrays(requests, patch_array, _DummyPool(), out_shape=None)
    assert excinfo.value.__cause__ is boom
