# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import shapely
from affine import Affine


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


def test_torchgeo_fetch_arrays_skips_empty_results_when_some_succeed() -> None:
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
    ds._warned_cog_read_failures = False

    requests = [
        ("https://example.com/a.tif", object(), None),
        ("https://example.com/b.tif", object(), 0),
    ]
    patch_array = pa.array([None])

    empty = SimpleNamespace(data=np.array([], dtype=np.float32))
    good = SimpleNamespace(
        data=np.ones((4, 4), dtype=np.float32), transform=Affine.identity()
    )
    with patch.object(torchgeo, "read_cog", side_effect=[empty, good]):
        out = ds._fetch_arrays(requests, patch_array, _DummyPool(), out_shape=None)
    assert len(out) == 1
    assert out[0][0].data.shape == (4, 4)


def test_torchgeo_filter_positive_overlap_drops_touching_only_rows() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    torchgeo = pytest.importorskip("rasteret.integrations.torchgeo")
    RasteretGeoDataset = getattr(torchgeo, "RasteretGeoDataset", None)
    if RasteretGeoDataset is None or not hasattr(
        RasteretGeoDataset, "_filter_positive_overlap"
    ):
        pytest.skip("RasteretGeoDataset is not available in this environment")

    ds = object.__new__(RasteretGeoDataset)
    interval = pd.Interval(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    index = pd.IntervalIndex([interval, interval], name="datetime")
    df = gpd.GeoDataFrame(
        {"rid": [0, 1]},
        geometry=[
            shapely.box(0.0, 0.0, 1.0, 1.0),
            shapely.box(1.0, 0.0, 2.0, 1.0),
        ],
        index=index,
        crs="EPSG:4326",
    )

    filtered = ds._filter_positive_overlap(
        df, slice(1.0, 2.0, 1.0), slice(0.0, 1.0, 1.0)
    )
    assert filtered["rid"].tolist() == [1]


def test_torchgeo_getitem_falls_back_to_next_readable_record() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    torchgeo = pytest.importorskip("rasteret.integrations.torchgeo")
    RasteretGeoDataset = getattr(torchgeo, "RasteretGeoDataset", None)
    if RasteretGeoDataset is None:
        pytest.skip("RasteretGeoDataset is not available in this environment")

    ds = object.__new__(RasteretGeoDataset)
    ds.time_series = False
    ds.epsg = 4326
    ds._multi_crs = False
    ds.max_concurrent = 5
    ds.bands = ("B01",)
    ds.is_image = True
    ds.transforms = None
    ds.label_field = None
    ds._warned_image_dtype_casts = set()
    ds._warned_cog_read_failures = False

    interval = pd.Interval(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    index = pd.IntervalIndex([interval, interval], name="datetime")
    ds.index = gpd.GeoDataFrame(
        {"rid": [0, 1]},
        geometry=[
            shapely.box(0.0, 0.0, 1.0, 1.0),  # larger overlap, but unreadable
            shapely.box(0.0, 0.0, 0.6, 1.0),  # smaller overlap, readable
        ],
        index=index,
        crs="EPSG:4326",
    )
    ds._payload = pd.DataFrame([{"rid": 0}, {"rid": 1}])

    class _DummyPool:
        pass

    ds._ensure_pool = lambda: _DummyPool()
    ds._disambiguate_slice = lambda idx: idx
    ds._slice_to_tensor = lambda idx: torch.zeros(9, dtype=torch.float32)
    ds._build_band_requests = lambda row: [(f"url-{int(row['rid'])}", object(), None)]
    calls: list[str] = []

    def _fetch(requests, patch_array, pool, *, out_shape):
        url = requests[0][0]
        calls.append(url)
        if url == "url-0":
            raise ValueError("empty tile")
        result = SimpleNamespace(
            data=np.ones((8, 8), dtype=np.float32), transform=Affine.identity()
        )
        return [(result, SimpleNamespace(nodata=None))]

    ds._fetch_arrays = _fetch
    ds._merge_resample_to_query_grid = lambda data, *_args, **_kwargs: data
    ds._image_tensor_from_numpy = torch.from_numpy

    sample = ds[
        slice(0.0, 1.0, 1.0),
        slice(0.0, 1.0, 1.0),
        slice(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), 1),
    ]

    assert calls == ["url-0", "url-1"]
    assert "image" in sample
    assert sample["image"].shape == (1, 8, 8)


def test_torchgeo_time_series_raises_on_partial_cog_failures() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    torchgeo = pytest.importorskip("rasteret.integrations.torchgeo")
    RasteretGeoDataset = getattr(torchgeo, "RasteretGeoDataset", None)
    if RasteretGeoDataset is None:
        pytest.skip("RasteretGeoDataset is not available in this environment")

    ds = object.__new__(RasteretGeoDataset)
    ds.time_series = True
    ds.epsg = 4326
    ds._multi_crs = False
    ds.max_concurrent = 5
    ds.bands = ("B01",)
    ds.is_image = True
    ds.transforms = None
    ds.label_field = None
    ds._warned_image_dtype_casts = set()
    ds._warned_cog_read_failures = False

    interval = pd.Interval(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    index = pd.IntervalIndex([interval, interval], name="datetime")
    ds.index = gpd.GeoDataFrame(
        {"rid": [0, 1]},
        geometry=[
            shapely.box(0.0, 0.0, 1.0, 1.0),
            shapely.box(0.0, 0.0, 1.0, 1.0),
        ],
        index=index,
        crs="EPSG:4326",
    )
    ds._payload = pd.DataFrame([{"rid": 0}, {"rid": 1}])
    ds._ensure_pool = lambda: object()
    ds._disambiguate_slice = lambda idx: idx
    ds._filter_positive_overlap = lambda df, *_args: df
    ds._payload_row = lambda rid: ds._payload.iloc[int(rid)]
    ds._build_band_requests = lambda _row: [("https://example.com/x.tif", object(), 0)]
    ds._fetch_arrays = lambda *_args, **_kwargs: [
        (
            SimpleNamespace(
                data=np.ones((8, 8), dtype=np.float32), transform=Affine.identity()
            ),
            SimpleNamespace(nodata=None),
        )
    ]

    with pytest.raises(ValueError, match="COG reads failed for time series sample"):
        ds[
            slice(0.0, 1.0, 1.0),
            slice(0.0, 1.0, 1.0),
            slice(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), 1),
        ]


def test_torchgeo_time_series_respects_query_temporal_slice() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    torchgeo = pytest.importorskip("rasteret.integrations.torchgeo")
    RasteretGeoDataset = getattr(torchgeo, "RasteretGeoDataset", None)
    if RasteretGeoDataset is None:
        pytest.skip("RasteretGeoDataset is not available in this environment")

    ds = object.__new__(RasteretGeoDataset)
    ds.time_series = True
    ds.epsg = 4326
    ds._multi_crs = False
    ds.max_concurrent = 5
    ds.bands = ("B01",)
    ds.is_image = True
    ds.transforms = None
    ds.label_field = None
    ds._warned_image_dtype_casts = set()
    ds._warned_cog_read_failures = False

    t0 = pd.Timestamp("2024-01-01")
    t1 = pd.Timestamp("2024-02-01")
    t2 = pd.Timestamp("2024-03-01")
    t3 = pd.Timestamp("2024-04-01")

    index = pd.IntervalIndex(
        [pd.Interval(t0, t1, closed="both"), pd.Interval(t2, t3, closed="both")],
        name="datetime",
    )
    ds.index = gpd.GeoDataFrame(
        {"rid": [0, 1]},
        geometry=[shapely.box(0.0, 0.0, 1.0, 1.0), shapely.box(0.0, 0.0, 1.0, 1.0)],
        index=index,
        crs="EPSG:4326",
    )
    ds._payload = pd.DataFrame([{"rid": 0}, {"rid": 1}])
    ds._ensure_pool = lambda: object()
    ds._disambiguate_slice = lambda idx: idx
    ds._filter_positive_overlap = lambda df, *_args: df
    ds._slice_to_tensor = lambda _idx: torch.zeros(9, dtype=torch.float32)
    ds._payload_row = lambda rid: ds._payload.iloc[int(rid)]
    ds._build_band_requests = lambda row: [(f"url-{int(row['rid'])}", object(), 0)]

    def _fetch(requests, *_args, **_kwargs):
        url = requests[0][0]
        rid = int(url.rsplit("-", maxsplit=1)[1])
        result = SimpleNamespace(
            data=np.full((4, 4), rid + 1, dtype=np.float32),
            transform=Affine.identity(),
        )
        return [(result, SimpleNamespace(nodata=None))]

    ds._fetch_arrays = _fetch
    ds._merge_resample_to_query_grid = lambda data, *_args, **_kwargs: data
    ds._image_tensor_from_numpy = torch.from_numpy

    sample = ds[
        slice(0.0, 1.0, 1.0),
        slice(0.0, 1.0, 1.0),
        slice(pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-20"), 1),
    ]
    image = sample["image"]
    assert image.shape == (1, 1, 4, 4)
    assert torch.all(image[0, 0] == 1.0)


def test_torchgeo_non_time_series_mosaics_overlapping_records() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    torchgeo = pytest.importorskip("rasteret.integrations.torchgeo")
    RasteretGeoDataset = getattr(torchgeo, "RasteretGeoDataset", None)
    if RasteretGeoDataset is None:
        pytest.skip("RasteretGeoDataset is not available in this environment")

    ds = object.__new__(RasteretGeoDataset)
    ds.time_series = False
    ds.epsg = 4326
    ds._multi_crs = False
    ds.max_concurrent = 5
    ds.bands = ("B01",)
    ds.is_image = True
    ds.transforms = None
    ds.label_field = None
    ds._warned_image_dtype_casts = set()
    ds._warned_cog_read_failures = False

    interval = pd.Interval(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
    index = pd.IntervalIndex([interval, interval], name="datetime")
    ds.index = gpd.GeoDataFrame(
        {"rid": [0, 1]},
        geometry=[shapely.box(0.0, 0.0, 1.0, 1.0), shapely.box(0.0, 0.0, 1.0, 1.0)],
        index=index,
        crs="EPSG:4326",
    )
    ds._payload = pd.DataFrame([{"rid": 0}, {"rid": 1}])

    class _DummyPool:
        pass

    ds._ensure_pool = lambda: _DummyPool()
    ds._disambiguate_slice = lambda idx: idx
    ds._slice_to_tensor = lambda idx: torch.zeros(9, dtype=torch.float32)
    ds._build_band_requests = lambda row: [(f"url-{int(row['rid'])}", object(), None)]

    def _fetch(requests, *_args, **_kwargs):
        url = requests[0][0]
        if url == "url-0":
            arr = np.array(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ],
                dtype=np.float32,
            )
        else:
            arr = np.full((4, 4), 2.0, dtype=np.float32)
        return [
            (
                SimpleNamespace(data=arr, transform=Affine.identity()),
                SimpleNamespace(nodata=0.0),
            )
        ]

    ds._fetch_arrays = _fetch
    ds._merge_resample_to_query_grid = lambda data, *_args, **_kwargs: data
    ds._image_tensor_from_numpy = torch.from_numpy

    sample = ds[
        slice(0.0, 1.0, 1.0),
        slice(0.0, 1.0, 1.0),
        slice(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), 1),
    ]
    image = sample["image"]

    expected = torch.tensor(
        [
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
            ]
        ]
    )
    assert image.shape == (1, 4, 4)
    assert torch.equal(image, expected)
