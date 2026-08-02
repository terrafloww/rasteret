# Multi-Dataset Training

Use this page when you want to combine multiple Rasteret-backed TorchGeo datasets in one training workflow.

Each collection becomes a standard TorchGeo `GeoDataset` via `RasteretDataset`:

```python
from torchgeo.datasets import RasteretDataset

s2 = RasteretDataset(collection=s2_collection, bands=["B04", "B03", "B02"])
mask = RasteretDataset(collection=mask_collection, bands=["mask"], is_image=False)
```

TorchGeo handles dataset composition with `&` and `|`.

## Intersection With `&`

Use `&` when a sample should come from areas where both datasets have coverage. This is common for imagery plus masks, imagery plus embeddings, or imagery plus another aligned raster source.

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

TorchGeo's `IntersectionDataset` computes the spatial and temporal overlap. By default, when both datasets return the same key such as `image`, TorchGeo stacks the arrays along the channel dimension. If you want separate keys, create one dataset with `is_image=False` so it returns `sample["mask"]`.

## Union With `|`

Use `|` when a sample can come from either dataset's coverage area:

```python
s2 = RasteretDataset(collection=s2_collection, bands=["B04", "B03", "B02"])
landsat = RasteretDataset(collection=landsat_collection, bands=["B04", "B03", "B02"])

training = s2 | landsat
```

TorchGeo's `UnionDataset` concatenates the spatial index and tries each dataset for a requested sample. When multiple datasets can satisfy the same sample, its default collation merges the returned sample dictionaries.

## CRS And Resolution

TorchGeo composition aligns the second dataset to the first dataset's CRS and resolution metadata. If your Rasteret collections span multiple raster CRS zones, pass `crs=...` when creating each dataset:

```python
from pyproj import CRS

s2 = RasteretDataset(
    collection=s2_collection,
    bands=["B04", "B03", "B02"],
    crs=CRS.from_epsg(32610),
)

aef = RasteretDataset(
    collection=aef_collection,
    bands=["A00", "A01"],
    crs=CRS.from_epsg(32610),
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

Use `target_crs=...` on the read calls when the collections are in different CRS zones.
