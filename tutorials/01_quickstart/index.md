# Quickstart[¶](#quickstart)

This notebook shows the minimal Rasteret workflow:

1. **Build** a `Collection` from the dataset catalog (one-time, cached)
1. **Fetch** pixels for a small AOI + time window as xarray, NumPy, and GeoPandas outputs
1. **Compute** NDVI from both xarray and NumPy arrays

In \[ \]:

Copied!

```
from pathlib import Path

import xarray as xr
from shapely.geometry import Polygon

import rasteret
```

from pathlib import Path import xarray as xr from shapely.geometry import Polygon import rasteret

## Define area of interest[¶](#define-area-of-interest)

We use a small polygon over Bengaluru, India. The STAC query uses the polygon's bounding box to find matching Sentinel-2 L2A scenes.

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
```

aoi = Polygon( [ (77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01), ] )

## Build the Collection[¶](#build-the-collection)

`build()` picks a dataset from the catalog (here Sentinel-2 on Earth Search), queries the STAC API, parses COG headers for every matching scene, and writes the result to a local Parquet index. On the next run, the cache is loaded in milliseconds.

In \[ \]:

Copied!

```
workspace = Path.home() / "rasteret_workspace"

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="bangalore",
    bbox=aoi.bounds,
    date_range=("2024-01-01", "2024-01-31"),
    workspace_dir=workspace,
)

print(f"Collection: {collection.name}")
print(f"Scenes: {collection.dataset.count_rows()}")
print(f"Columns: {collection.dataset.schema.names[:8]}...")
```

workspace = Path.home() / "rasteret_workspace" collection = rasteret.build( "earthsearch/sentinel-2-l2a", name="bangalore", bbox=aoi.bounds, date_range=("2024-01-01", "2024-01-31"), workspace_dir=workspace, ) print(f"Collection: {collection.name}") print(f"Scenes: {collection.dataset.count_rows()}") print(f"Columns: {collection.dataset.schema.names[:8]}...")

## Fetch pixels as xarray[¶](#fetch-pixels-as-xarray)

`get_xarray()` reads only tiles intersecting the AOI and returns a standard `xarray.Dataset` with spatial coordinates and CRS metadata.

In \[ \]:

Copied!

```
ds = collection.get_xarray(
    geometries=[aoi],
    bands=["B04", "B08"],
    cloud_cover_lt=20,
    date_range=("2024-01-10", "2024-01-30"),
)

ds
```

ds = collection.get_xarray( geometries=[aoi], bands=["B04", "B08"], cloud_cover_lt=20, date_range=("2024-01-10", "2024-01-30"), ) ds

## Fetch pixels as NumPy[¶](#fetch-pixels-as-numpy)

Use `get_numpy()` when you want plain arrays for custom ML or analysis code. Multi-band output shape is `[N, C, H, W]`.

In \[ \]:

Copied!

```
arr = collection.get_numpy(
    geometries=[aoi],
    bands=["B04", "B08"],
    cloud_cover_lt=20,
    date_range=("2024-01-10", "2024-01-30"),
)

arr.shape, arr.dtype
```

arr = collection.get_numpy( geometries=[aoi], bands=["B04", "B08"], cloud_cover_lt=20, date_range=("2024-01-10", "2024-01-30"), ) arr.shape, arr.dtype

## Fetch pixels as GeoPandas[¶](#fetch-pixels-as-geopandas)

Use `get_gdf()` when you want one row per geometry-scene pair with band arrays attached as columns.

In \[ \]:

Copied!

```
gdf = collection.get_gdf(
    geometries=[aoi],
    bands=["B04", "B08"],
    cloud_cover_lt=20,
    date_range=("2024-01-10", "2024-01-30"),
)

gdf[["id", "datetime", "B04", "B08"]].head()
```

gdf = collection.get_gdf( geometries=[aoi], bands=["B04", "B08"], cloud_cover_lt=20, date_range=("2024-01-10", "2024-01-30"), ) gdf\[["id", "datetime", "B04", "B08"]\].head()

## Compute NDVI (xarray + NumPy)[¶](#compute-ndvi-xarray-numpy)

Both outputs represent the same pixel values. Use whichever API fits your pipeline.

In \[ \]:

Copied!

```
# xarray NDVI
nir_xr = ds["B08"].astype(float)
red_xr = ds["B04"].astype(float)
ndvi_xr = (nir_xr - red_xr) / (nir_xr + red_xr)

# NumPy NDVI (arr channels follow requested band order: B04, B08)
red_np = arr[:, 0].astype("float32")
nir_np = arr[:, 1].astype("float32")
ndvi_np = (nir_np - red_np) / (nir_np + red_np + 1e-6)

xr.Dataset({"ndvi": ndvi_xr}, coords=ds.coords, attrs=ds.attrs), ndvi_np.shape
```

# xarray NDVI

nir_xr = ds["B08"].astype(float) red_xr = ds["B04"].astype(float) ndvi_xr = (nir_xr - red_xr) / (nir_xr + red_xr)

# NumPy NDVI (arr channels follow requested band order: B04, B08)

red_np = arr[:, 0].astype("float32") nir_np = arr[:, 1].astype("float32") ndvi_np = (nir_np - red_np) / (nir_np + red_np + 1e-6) xr.Dataset({"ndvi": ndvi_xr}, coords=ds.coords, attrs=ds.attrs), ndvi_np.shape

## What happened[¶](#what-happened)

1. `build()` queried STAC once, parsed COG headers, and cached a Parquet index.
1. `get_xarray()`, `get_numpy()`, and `get_gdf()` all reused the same cached index.
1. Rasteret fetched only intersecting tiles in parallel and assembled the output format you requested.

Next: [TorchGeo Integration](https://terrafloww.github.io/rasteret/tutorials/02_torchgeo_09_accelerator/index.md)
