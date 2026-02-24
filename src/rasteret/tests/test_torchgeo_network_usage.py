# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Real TorchGeo usage tests (network).

These tests ensure Rasteret's TorchGeo adapter works with:
- TorchGeo samplers (RandomGeoSampler)
- PyTorch DataLoader collation
- Both image-style and mask-style samples
- time_series stacking

Run with:
  uv run pytest --network -k torchgeo_network_usage -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rasteret
from rasteret.catalog import DatasetRegistry

pytestmark = pytest.mark.network


def _build(
    tmp_path: Path,
    dataset_id: str,
    *,
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    max_items: int,
) -> rasteret.Collection:
    descriptor = DatasetRegistry.get(dataset_id)
    assert descriptor is not None
    return rasteret.build(
        dataset_id,
        name=f"tg-net-{dataset_id.split('/', 1)[-1]}",
        workspace_dir=tmp_path / dataset_id.replace("/", "-"),
        force=True,
        max_concurrent=8,
        bbox=bbox,
        date_range=date_range,
        query={"max_items": max_items} if not descriptor.static_catalog else None,
    )


def test_torchgeo_dataloader_image_chip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.samplers.single import RandomGeoSampler

    dataset_id = "earthsearch/sentinel-2-l2a"
    d = DatasetRegistry.get(dataset_id)
    assert d is not None
    bbox, date_range = d.example_bbox, d.example_date_range
    assert bbox is not None and date_range is not None

    collection = _build(
        tmp_path, dataset_id, bbox=bbox, date_range=date_range, max_items=3
    )
    ds = collection.to_torchgeo_dataset(bands=["B02", "B03", "B04"], chip_size=64)

    sampler = RandomGeoSampler(ds, size=64, length=4)
    loader = DataLoader(ds, sampler=sampler, batch_size=2, num_workers=0)

    batch = next(iter(loader))
    assert "image" in batch
    assert batch["image"].ndim == 4  # [B, C, H, W]
    assert batch["image"].shape[0] == 2
    assert batch["image"].shape[1] == 3
    assert batch["image"].shape[-2:] == (64, 64)


def test_torchgeo_dataloader_mask_chip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.samplers.single import RandomGeoSampler

    dataset_id = "pc/esa-worldcover"
    d = DatasetRegistry.get(dataset_id)
    assert d is not None
    bbox, date_range = d.example_bbox, d.example_date_range
    assert bbox is not None and date_range is not None

    collection = _build(
        tmp_path, dataset_id, bbox=bbox, date_range=date_range, max_items=1
    )
    band = None
    if d.band_map:
        band = sorted(d.band_map.keys())[0]
    if band is None:
        pytest.skip(f"{dataset_id} has no configured band_map to pick a band code")
    ds = collection.to_torchgeo_dataset(bands=[band], chip_size=64, is_image=False)

    sampler = RandomGeoSampler(ds, size=64, length=2)
    loader = DataLoader(ds, sampler=sampler, batch_size=2, num_workers=0)
    batch = next(iter(loader))

    assert "mask" in batch
    assert "image" not in batch
    assert batch["mask"].ndim == 3  # [B, H, W] (C squeezed)
    assert batch["mask"].shape[-2:] == (64, 64)


def test_torchgeo_time_series_manual_slice(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")

    dataset_id = "earthsearch/sentinel-2-l2a"
    d = DatasetRegistry.get(dataset_id)
    assert d is not None
    bbox, date_range = d.example_bbox, d.example_date_range
    assert bbox is not None and date_range is not None

    # Build with multiple items so time_series stacking is meaningful.
    collection = _build(
        tmp_path, dataset_id, bbox=bbox, date_range=date_range, max_items=6
    )
    ds = collection.to_torchgeo_dataset(
        bands=["B02", "B03"],
        chip_size=64,
        time_series=True,
    )

    # A wide time slice that should overlap at least one record.
    import pandas as pd

    # TorchGeo uses tz-naive datetimes in its index.
    now = pd.Timestamp.utcnow().tz_localize(None)
    t0 = now - pd.Timedelta(days=3650)
    t1 = now

    x, y, _t = ds.bounds
    sample = ds[
        slice(x.start, x.start + 64 * x.step, x.step),
        slice(y.start, y.start + 64 * y.step, y.step),
        slice(t0, t1, 1),
    ]

    assert "image" in sample
    assert sample["image"].ndim == 4  # [T, C, H, W]
    assert sample["image"].shape[1] == 2
    assert sample["image"].shape[-2:] == (64, 64)


def test_torchgeo_intersection_dataset_builds(tmp_path: Path) -> None:
    """TorchGeo dataset composition (`&`) should work with RasteretGeoDataset.

    This catches index column collisions during `gpd.overlay(index1, index2, ...)`.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchgeo")
    from torch.utils.data import DataLoader
    from torchgeo.samplers.single import RandomGeoSampler

    dataset_id = "earthsearch/sentinel-2-l2a"
    d = DatasetRegistry.get(dataset_id)
    assert d is not None
    bbox, date_range = d.example_bbox, d.example_date_range
    assert bbox is not None and date_range is not None

    collection = _build(
        tmp_path, dataset_id, bbox=bbox, date_range=date_range, max_items=4
    )
    images = collection.to_torchgeo_dataset(bands=["B02", "B03", "B04"], chip_size=64)
    masks = collection.to_torchgeo_dataset(bands=["B02"], chip_size=64, is_image=False)

    merged = images & masks
    sampler = RandomGeoSampler(merged, size=64, length=2)
    loader = DataLoader(merged, sampler=sampler, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
