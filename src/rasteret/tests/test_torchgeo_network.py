# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Network tests for Rasteret's TorchGeo adapter.

Each test exercises a specific GeoDataset contract surface or pitch claim
against real satellite data. Module-scoped fixtures share collection builds
so the full suite runs in ~1 min, not ~5 min.

Run with:
    uv run pytest --network -k test_torchgeo_network -v

Coverage map (pitch table row -> test):
    1. [T, C, H, W] time_series      -> test_time_series_shape
    2. IntervalIndex contract         -> test_geodataset_index_contract
    3. Spatiotemporal stack behavior  -> test_spatial_only_intersection
    4. Concurrent async reads         -> test_multi_band_multi_scene_correctness
    5. Multi-CRS reprojection         -> test_cross_crs_*, test_crs_drop_warning
    6. collection.subset() filtering  -> test_subset_to_dataset
    7. STAC / GeoParquet sources      -> test_export_reload_roundtrip
    8. Samplers / collation / comp.   -> test_*_sampler, test_*_dataset, test_mask_chip

Pixel-level accuracy is separately verified by test_dataset_pixel_comparison.py
across all 12 catalog datasets.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

import rasteret
from rasteret.catalog import DatasetRegistry

pytestmark = pytest.mark.network

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTINEL2_ID = "earthsearch/sentinel-2-l2a"
WORLDCOVER_ID = "pc/esa-worldcover"

# Bangalore, single UTM zone (EPSG:32643)
BBOX_SINGLE = (77.55, 13.01, 77.58, 13.08)
DATE_RANGE = ("2024-03-01", "2024-06-30")

# Near Hyderabad, UTM zone boundary (EPSG:32643 / 32644)
BBOX_CROSS_CRS = (78.25, 17.30, 78.45, 17.40)
DATE_RANGE_CROSS_CRS = ("2024-04-01", "2024-04-30")


# ---------------------------------------------------------------------------
# Module-scoped fixtures: each collection builds once, reused across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def s2_collection(tmp_path_factory: pytest.TempPathFactory) -> rasteret.Collection:
    """Sentinel-2 collection (Bangalore, 6 scenes)."""
    tmp = tmp_path_factory.mktemp("s2")
    return rasteret.build(
        SENTINEL2_ID,
        name="tg-s2",
        workspace_dir=tmp / "tg-s2",
        force=True,
        max_concurrent=8,
        bbox=BBOX_SINGLE,
        date_range=DATE_RANGE,
        query={"max_items": 6},
    )


@pytest.fixture(scope="module")
def worldcover_collection(
    tmp_path_factory: pytest.TempPathFactory,
) -> rasteret.Collection:
    """ESA WorldCover collection (1 scene, Planetary Computer)."""
    d = DatasetRegistry.get(WORLDCOVER_ID)
    assert d is not None
    bbox, date_range = d.example_bbox, d.example_date_range
    assert bbox is not None and date_range is not None

    tmp = tmp_path_factory.mktemp("wc")
    return rasteret.build(
        WORLDCOVER_ID,
        name="tg-wc",
        workspace_dir=tmp / "tg-wc",
        force=True,
        max_concurrent=8,
        bbox=bbox,
        date_range=date_range,
    )


@pytest.fixture(scope="module")
def cross_crs_collection(
    tmp_path_factory: pytest.TempPathFactory,
) -> rasteret.Collection:
    """Sentinel-2 spanning two UTM zones (Hyderabad, 10 scenes)."""
    tmp = tmp_path_factory.mktemp("crs")
    return rasteret.build(
        SENTINEL2_ID,
        name="tg-crs",
        workspace_dir=tmp / "tg-crs",
        force=True,
        max_concurrent=8,
        bbox=BBOX_CROSS_CRS,
        date_range=DATE_RANGE_CROSS_CRS,
        query={"max_items": 10},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wide_time_slice():
    """Return (t0, now) spanning 10 years for guaranteed overlap."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    return now - pd.Timedelta(days=3650), now


def _read_chip(ds, *, chip_size: int = 64):
    """Read a single chip from a TorchGeo dataset using a wide time slice."""
    t0, now = _wide_time_slice()
    x, y, _t = ds.bounds
    return ds[
        slice(x.start, x.start + chip_size * x.step, x.step),
        slice(y.start, y.start + chip_size * y.step, y.step),
        slice(t0, now, 1),
    ]


def _has_multi_crs(collection: rasteret.Collection) -> bool:
    table = collection.dataset.to_table(columns=["proj:epsg"])
    return len(set(table.column("proj:epsg").to_pylist())) >= 2


# ---------------------------------------------------------------------------
# GeoDataset contract (Claim 2)
# ---------------------------------------------------------------------------


def test_geodataset_index_contract(s2_collection: rasteret.Collection) -> None:
    """Index is a GeoDataFrame with IntervalIndex named 'datetime' + geometry."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    import geopandas as gpd

    ds = s2_collection.to_torchgeo_dataset(bands=["B04"], chip_size=64)

    # GeoDataFrame
    assert isinstance(ds.index, gpd.GeoDataFrame)
    # IntervalIndex named "datetime"
    assert isinstance(ds.index.index, pd.IntervalIndex)
    assert ds.index.index.name == "datetime"
    # Geometry column with non-null footprints
    assert ds.index.geometry is not None
    assert all(g is not None for g in ds.index.geometry)
    # CRS and res set
    assert ds.crs is not None
    assert ds._res[0] > 0 and ds._res[1] > 0


# ---------------------------------------------------------------------------
# Image chip via DataLoader (Claim 8: RandomGeoSampler)
# ---------------------------------------------------------------------------


def test_image_chip_random_sampler(s2_collection: rasteret.Collection) -> None:
    """RandomGeoSampler -> DataLoader -> batch [B, C, H, W]."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.datasets.utils import stack_samples
    from torchgeo.samplers import RandomGeoSampler

    ds = s2_collection.to_torchgeo_dataset(bands=["B04", "B03", "B02"], chip_size=64)
    sampler = RandomGeoSampler(ds, size=64, length=4)
    loader = DataLoader(
        ds, sampler=sampler, batch_size=2, num_workers=0, collate_fn=stack_samples
    )
    batch = next(iter(loader))

    assert "image" in batch
    assert "bounds" in batch
    assert "transform" in batch
    assert batch["image"].shape == (2, 3, 64, 64)


# ---------------------------------------------------------------------------
# Mask chip on WorldCover (Claim 8: is_image=False, different provider)
# ---------------------------------------------------------------------------


def test_mask_chip(worldcover_collection: rasteret.Collection) -> None:
    """is_image=False returns 'mask' key with squeezed channel dim."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.samplers import RandomGeoSampler

    d = DatasetRegistry.get(WORLDCOVER_ID)
    band = sorted(d.band_map.keys())[0] if d.band_map else None
    if band is None:
        pytest.skip("No band_map for WorldCover")

    ds = worldcover_collection.to_torchgeo_dataset(
        bands=[band], chip_size=64, is_image=False
    )
    sampler = RandomGeoSampler(ds, size=64, length=2)
    loader = DataLoader(ds, sampler=sampler, batch_size=2, num_workers=0)
    batch = next(iter(loader))

    assert "mask" in batch
    assert "image" not in batch
    assert batch["mask"].ndim == 3  # [B, H, W] (C squeezed)
    assert batch["mask"].shape[-2:] == (64, 64)


# ---------------------------------------------------------------------------
# Time series (Claim 1)
# ---------------------------------------------------------------------------


def test_time_series_shape(s2_collection: rasteret.Collection) -> None:
    """time_series=True returns [T, C, H, W] with T >= 2."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")

    ds = s2_collection.to_torchgeo_dataset(
        bands=["B02", "B03"], chip_size=64, time_series=True
    )
    sample = _read_chip(ds)

    img = sample["image"]
    assert img.ndim == 4, f"Expected 4D [T, C, H, W], got ndim={img.ndim}"
    T, C, H, W = img.shape
    assert T >= 2, f"Expected T >= 2, got T={T}"
    assert C == 2
    assert (H, W) == (64, 64)


# ---------------------------------------------------------------------------
# Multi-band correctness (Claim 4)
# ---------------------------------------------------------------------------


def test_multi_band_multi_scene_correctness(
    s2_collection: rasteret.Collection,
) -> None:
    """4 bands x multiple timesteps: each band returns non-zero data."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")

    ds = s2_collection.to_torchgeo_dataset(
        bands=["B02", "B03", "B04", "B08"], chip_size=64, time_series=True
    )
    sample = _read_chip(ds)

    img = sample["image"]
    assert img.ndim == 4
    T, C, H, W = img.shape
    assert C == 4
    for c in range(C):
        assert img[:, c, :, :].sum() > 0, f"Band {c} is all zeros"


# ---------------------------------------------------------------------------
# Spatial-only intersection (Claim 3)
# ---------------------------------------------------------------------------


def test_spatial_only_intersection(s2_collection: rasteret.Collection) -> None:
    """time_series=True stacks timesteps selected by the query slice."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")

    ds = s2_collection.to_torchgeo_dataset(
        bands=["B04"], chip_size=64, time_series=True
    )
    sample = _read_chip(ds)
    T = sample["image"].shape[0]

    assert T >= 2, (
        f"Expected >= 2 timesteps stacked, got T={T} "
        f"(collection has {len(s2_collection)} records)"
    )


# ---------------------------------------------------------------------------
# GridGeoSampler (Claim 8)
# ---------------------------------------------------------------------------


def test_grid_geo_sampler(s2_collection: rasteret.Collection) -> None:
    """GridGeoSampler works through DataLoader."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.datasets.utils import stack_samples
    from torchgeo.samplers import GridGeoSampler

    ds = s2_collection.to_torchgeo_dataset(bands=["B04"], chip_size=64)
    sampler = GridGeoSampler(ds, size=64, stride=64)
    loader = DataLoader(
        ds, sampler=sampler, batch_size=2, num_workers=0, collate_fn=stack_samples
    )
    batch = next(iter(loader))

    assert batch["image"].ndim == 4  # [B, C, H, W]
    assert batch["image"].shape[-2:] == (64, 64)


# ---------------------------------------------------------------------------
# IntersectionDataset (Claim 8)
# ---------------------------------------------------------------------------


def test_intersection_dataset(s2_collection: rasteret.Collection) -> None:
    """images & masks composition produces batch with both keys."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.datasets.utils import stack_samples
    from torchgeo.samplers import RandomGeoSampler

    images = s2_collection.to_torchgeo_dataset(
        bands=["B04", "B03", "B02"], chip_size=64
    )
    masks = s2_collection.to_torchgeo_dataset(
        bands=["B04"], chip_size=64, is_image=False
    )

    merged = images & masks
    sampler = RandomGeoSampler(merged, size=64, length=2)
    loader = DataLoader(
        merged, sampler=sampler, batch_size=2, num_workers=0, collate_fn=stack_samples
    )
    batch = next(iter(loader))

    assert "image" in batch
    assert "mask" in batch
    assert batch["image"].shape[-2:] == (64, 64)
    assert batch["mask"].shape[-2:] == (64, 64)


# ---------------------------------------------------------------------------
# UnionDataset (Claim 8)
# ---------------------------------------------------------------------------


def test_union_dataset(s2_collection: rasteret.Collection) -> None:
    """UnionDataset is constructable and sampler produces slices."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torchgeo.samplers import RandomGeoSampler

    ds1 = s2_collection.to_torchgeo_dataset(bands=["B04"], chip_size=64)
    ds2 = s2_collection.to_torchgeo_dataset(bands=["B04"], chip_size=64)

    union = ds1 | ds2
    sampler = RandomGeoSampler(union, size=64, length=2)
    assert len(list(sampler)) == 2


# ---------------------------------------------------------------------------
# Cross-CRS reprojection (Claim 5)
# ---------------------------------------------------------------------------


def test_cross_crs_build_and_read(
    cross_crs_collection: rasteret.Collection,
) -> None:
    """target_crs keeps all CRS zones and reads chips successfully."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.datasets.utils import concat_samples
    from torchgeo.samplers import RandomGeoSampler

    if not _has_multi_crs(cross_crs_collection):
        pytest.skip("AOI does not span multiple CRS zones for these dates")

    ds = cross_crs_collection.to_torchgeo_dataset(
        bands=["B04"], chip_size=64, time_series=True, target_crs=32643
    )

    assert ds.epsg == 32643
    assert ds._multi_crs is True

    # Use concat_samples: variable T per chip is expected (known variable-T)
    sampler = RandomGeoSampler(ds, size=64, length=2)
    loader = DataLoader(
        ds, sampler=sampler, batch_size=2, num_workers=0, collate_fn=concat_samples
    )
    batch = next(iter(loader))

    assert "image" in batch
    assert batch["image"].shape[-2:] == (64, 64)
    assert batch["image"].sum() > 0, "Cross-CRS chip is all zeros"


def test_crs_drop_warning(cross_crs_collection: rasteret.Collection) -> None:
    """Without target_crs, non-majority CRS records are dropped with a warning."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")

    if not _has_multi_crs(cross_crs_collection):
        pytest.skip("AOI does not span multiple CRS zones for these dates")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cross_crs_collection.to_torchgeo_dataset(bands=["B04"], chip_size=64)

    crs_warnings = [
        x for x in w if "CRS" in str(x.message) and "dropped" in str(x.message)
    ]
    assert len(crs_warnings) > 0, "Expected a warning about dropped CRS records"


# ---------------------------------------------------------------------------
# Subset filtering (Claim 6)
# ---------------------------------------------------------------------------


def test_subset_to_dataset(s2_collection: rasteret.Collection) -> None:
    """subset(cloud_cover_lt=, date_range=) feeds into to_torchgeo_dataset."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")

    original_count = len(s2_collection)

    # Cloud cover filter
    filtered = s2_collection.subset(cloud_cover_lt=80)
    assert len(filtered) <= original_count

    if len(filtered) > 0:
        ds = filtered.to_torchgeo_dataset(bands=["B04"], chip_size=64)
        assert len(ds.index) > 0

    # Date range filter
    filtered = s2_collection.subset(date_range=("2024-04-01", "2024-04-30"))
    assert len(filtered) <= original_count

    if len(filtered) > 0:
        ds = filtered.to_torchgeo_dataset(bands=["B04"], chip_size=64)
        assert len(ds.index) > 0


# ---------------------------------------------------------------------------
# Export -> reload -> read chip (Claim 7)
# ---------------------------------------------------------------------------


def test_export_reload_roundtrip(
    s2_collection: rasteret.Collection,
    tmp_path: Path,
) -> None:
    """Export collection, reload from Parquet, build TorchGeo dataset, read chip."""
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")

    n_original = len(s2_collection)

    # Export
    export_dir = tmp_path / "exported"
    s2_collection.export(export_dir)

    # Reload
    reloaded = rasteret.load(export_dir)
    assert len(reloaded) == n_original

    # Build dataset and read a chip
    ds = reloaded.to_torchgeo_dataset(bands=["B04"], chip_size=64)
    assert len(ds.index) >= 1

    sample = _read_chip(ds)
    assert "image" in sample
    assert sample["image"].sum() > 0
