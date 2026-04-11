# Multi-Dataset Training

Use this page when you want to combine multiple Rasteret-backed TorchGeo
datasets in one training workflow.

Each collection can become a standard TorchGeo `GeoDataset`:

```python
s2 = s2_collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02"],
    chip_size=256,
)

mask = mask_collection.to_torchgeo_dataset(
    bands=["mask"],
    chip_size=256,
    is_image=False,
)
```

TorchGeo handles dataset composition with `&` and `|`.

## Intersection With `&`

Use `&` when a sample should come from areas where both datasets have coverage.
This is common for imagery plus masks, imagery plus embeddings, or imagery plus
another aligned raster source.

```python
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import RandomGeoSampler

training = s2 & mask

sampler = RandomGeoSampler(training, size=256, length=100)
loader = DataLoader(
    training,
    sampler=sampler,
    batch_size=4,
    collate_fn=stack_samples,
)

for batch in loader:
    image = batch["image"]
    target = batch["mask"]
    break
```

TorchGeo's `IntersectionDataset` computes the spatial and temporal overlap. By
default, when both datasets return the same key such as `image`, TorchGeo stacks
the arrays along the channel dimension. If you want separate keys, create one
dataset with `is_image=False` so it returns `sample["mask"]`.

## Union With `|`

Use `|` when a sample can come from either dataset's coverage area:

```python
s2 = s2_collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02"],
    chip_size=256,
)
landsat = landsat_collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02"],
    chip_size=256,
)

training = s2 | landsat
```

TorchGeo's `UnionDataset` concatenates the spatial index and tries each dataset
for a requested sample. When multiple datasets can satisfy the same sample, its
default collation merges the returned sample dictionaries.

## CRS And Resolution

TorchGeo composition aligns the second dataset to the first dataset's CRS and
resolution metadata. If your Rasteret collections span multiple raster CRS
zones, pass `target_crs=...` when creating each dataset:

```python
s2 = s2_collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02"],
    chip_size=256,
    target_crs=32610,
)

aef = aef_collection.to_torchgeo_dataset(
    bands=["A00", "A01"],
    chip_size=256,
    target_crs=32610,
)
```

## xarray Path

For analysis workflows, read each collection separately and combine with xarray:

```python
import xarray as xr

ds_s2 = s2_collection.get_xarray(geometries=aoi, bands=["B04", "B08"])
ds_aef = aef_collection.get_xarray(geometries=aoi, bands=["A00"])
combined = xr.merge([ds_s2, ds_aef])
```

Use `target_crs=...` on the read calls when the collections are in different
CRS zones.
