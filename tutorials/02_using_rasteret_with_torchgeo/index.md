# Using Rasteret with TorchGeo[¶](#using-rasteret-with-torchgeo)

Use a Rasteret Collection as a TorchGeo GeoDataset while keeping the standard TorchGeo sampler and DataLoader flow.

For the full technical narrative, see the [**Concepts**](https://terrafloww.github.io/rasteret/explanation/concepts/).

In \[ \]:

Copied!

```
from pathlib import Path

from shapely.geometry import Polygon

import rasteret
```

from pathlib import Path from shapely.geometry import Polygon import rasteret

## Build a Collection[¶](#build-a-collection)

Same `build()` call as notebook 01. If the cache already exists, this returns instantly.

In \[ \]:

Copied!

```
aoi = Polygon(
    [
        (77.55, 13.01),
        (77.58, 13.01),
        (77.58, 13.08),
        (77.55, 13.08),
        (77.55, 13.01),
    ]
)

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="bangalore",
    bbox=aoi.bounds,
    date_range=("2024-01-01", "2024-01-31"),
    workspace_dir=Path.home() / "rasteret_workspace",
)

print(f"Scenes: {collection.dataset.count_rows()}")
```

aoi = Polygon( [ (77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01), ] ) collection = rasteret.build( "earthsearch/sentinel-2-l2a", name="bangalore", bbox=aoi.bounds, date_range=("2024-01-01", "2024-01-31"), workspace_dir=Path.home() / "rasteret_workspace", ) print(f"Scenes: {collection.dataset.count_rows()}")

## Create a TorchGeo GeoDataset[¶](#create-a-torchgeo-geodataset)

`to_torchgeo_dataset()` returns a `RasteretGeoDataset(GeoDataset)`. It builds a GeoDataFrame spatial index from the Collection's cached metadata, no file discovery or GDAL VRT creation needed.

Optional parameters:

- `split` / `split_column`: filter to train/val/test before creating the dataset
- `label_field`: include a column value as `sample["label"]`
- `geometries`: restrict to a spatial region of interest

In \[ \]:

Copied!

```
dataset = collection.to_torchgeo_dataset(
    bands=["B02", "B03", "B04", "B08"],
    geometries=[aoi],
    chip_size=256,
)

print(f"Type: {type(dataset).__mro__[:3]}")
print(f"Bounds: {dataset.bounds}")
print(f"CRS: {dataset.crs}")
```

dataset = collection.to_torchgeo_dataset( bands=["B02", "B03", "B04", "B08"], geometries=[aoi], chip_size=256, ) print(f"Type: {type(dataset).__mro__[:3]}") print(f"Bounds: {dataset.bounds}") print(f"CRS: {dataset.crs}")

## Standard TorchGeo sampling + DataLoader[¶](#standard-torchgeo-sampling-dataloader)

Everything below is pure TorchGeo: `RandomGeoSampler`, `stack_samples`, `DataLoader`. Rasteret is invisible at this point.

In \[ \]:

Copied!

```
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import RandomGeoSampler

sampler = RandomGeoSampler(dataset, size=256, length=16)
loader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=4,
    num_workers=0,
    collate_fn=stack_samples,
)

batch = next(iter(loader))
print(f"image shape:     {batch['image'].shape}")  # (B, C, H, W)
print(f"bounds shape:    {batch['bounds'].shape}")  # (B, 9): TorchGeo 0.9 tensor
print(f"transform shape: {batch['transform'].shape}")  # (B, 6): affine coefficients
```

from torch.utils.data import DataLoader from torchgeo.datasets.utils import stack_samples from torchgeo.samplers import RandomGeoSampler sampler = RandomGeoSampler(dataset, size=256, length=16) loader = DataLoader( dataset, sampler=sampler, batch_size=4, num_workers=0, collate_fn=stack_samples, ) batch = next(iter(loader)) print(f"image shape: {batch['image'].shape}") # (B, C, H, W) print(f"bounds shape: {batch['bounds'].shape}") # (B, 9): TorchGeo 0.9 tensor print(f"transform shape: {batch['transform'].shape}") # (B, 6): affine coefficients

## What's in each sample?[¶](#whats-in-each-sample)

| Key         | Shape            | Description                                                                   |
| ----------- | ---------------- | ----------------------------------------------------------------------------- |
| `image`     | `(C, H, W)`      | Float32 pixel values, one channel per requested band                          |
| `bounds`    | `(9,)`           | `[xmin, xmax, xres, ymin, ymax, yres, tmin, tmax, tres]`: TorchGeo 0.9 format |
| `transform` | `(6,)`           | Affine coefficients `[a, b, c, d, e, f]`                                      |
| `label`     | scalar or tensor | Present only when `label_field` is set                                        |

No `crs` key. TorchGeo 0.9 removed it from samples. Access CRS via `dataset.crs`.

## Split-based workflow[¶](#split-based-workflow)

If your Collection has a `split` column (e.g. from a Parquet record table), pass `split="train"` to `to_torchgeo_dataset()` and it filters before building the spatial index.

```
train_ds = collection.to_torchgeo_dataset(
    bands=["B04", "B08"],
    split="train",
    label_field="class",
)
val_ds = collection.to_torchgeo_dataset(
    bands=["B04", "B08"],
    split="val",
)
```

Next: [Build from Parquet and Arrow Tables](https://terrafloww.github.io/rasteret/tutorials/06_build_from_parquet_and_arrow_tables/index.md)
