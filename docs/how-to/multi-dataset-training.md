# Multi-Dataset Training

Rasteret Collections produce standard TorchGeo `GeoDataset` objects, so
TorchGeo's dataset composition operators (`&`, `|`) work directly.

---

## Combining datasets with `&` (intersection)

Use `&` when you need aligned data from two sources, e.g. Sentinel-2
imagery paired with AEF embeddings, or imagery paired with a label mask.
The sampler only draws chips where **both** datasets have spatial and
temporal coverage.

```python
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets.utils import stack_samples

s2 = s2_collection.to_torchgeo_dataset(bands=["B04", "B03", "B02"], chip_size=256)
aef = aef_collection.to_torchgeo_dataset(bands=["emb_0", "emb_1"], chip_size=256)

combined = s2 & aef  # IntersectionDataset: channels concatenated

sampler = RandomGeoSampler(combined, size=256, length=100)
loader = DataLoader(combined, sampler=sampler, batch_size=4,
                    collate_fn=stack_samples)

for batch in loader:
    # batch["image"] shape: [B, C_s2 + C_aef, H, W]
    ...
```

---

## Combining datasets with `|` (union)

Use `|` when you want to train on whichever source has coverage, e.g.
fall back to Landsat when Sentinel-2 is cloudy, or mosaic adjacent
coverage from different providers.

```python
ds1 = col1.to_torchgeo_dataset(bands=["B04", "B03", "B02"], chip_size=256)
ds2 = col2.to_torchgeo_dataset(bands=["B04", "B03", "B02"], chip_size=256)

combined = ds1 | ds2  # UnionDataset
```

---

## How alignment works

- CRS and resolution are aligned to the first dataset at init time
- `&` computes spatial intersection via GeoPandas overlay; only areas
  where both datasets exist are valid for sampling
- `|` concatenates indices; the sampler can draw from either dataset's
  coverage area
- Band tensors are concatenated (for `&`) or merged (for `|`) automatically

---

## xarray path

For analysis workflows, load each Collection separately and combine
with standard xarray:

```python
import xarray as xr

ds_s2 = s2_collection.get_xarray(geometries=aoi, bands=["B04", "B08"])
ds_aef = aef_collection.get_xarray(geometries=aoi, bands=["emb_0"])
combined = xr.merge([ds_s2, ds_aef])
```

Both must be in the same CRS. Use `target_crs=` if they differ.
