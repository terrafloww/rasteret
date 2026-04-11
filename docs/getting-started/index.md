# Getting Started

This guide builds a small Sentinel-2 collection and reads pixels from it. The
goal is to show the Rasteret loop you will use in larger projects:

```text
build a collection -> inspect/filter it -> read pixels in the shape you need
```

## Installation

Rasteret requires Python 3.12 or later.

```bash
uv pip install rasteret
```

Install extras for the output surfaces you plan to use:

```bash
uv pip install "rasteret[xarray]"    # Collection.get_xarray()
uv pip install "rasteret[torchgeo]"  # Collection.to_torchgeo_dataset()
uv pip install "rasteret[all]"       # explore the optional integrations
```

## 1. Build A Collection

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2-bangalore",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
)
```

This works because `earthsearch/sentinel-2-l2a` is in Rasteret's built-in
catalog. The build step searches the source STAC API, parses the COG header
metadata once, and creates a Rasteret collection.

The important part: Rasteret does **not** move all pixels into Parquet. It keeps
the imagery in the source COGs and stores a queryable Arrow/Parquet index with
metadata such as IDs, footprints, assets, bounding boxes, CRS sidecars, and
per-band COG header metadata.

!!! tip "Why build first?"
    Opening many remote TIFFs repeatedly is expensive because each environment
    has to rediscover header metadata such as tile offsets, transforms, dtype,
    nodata, and CRS. Rasteret pays that setup cost during build, then reuses the
    collection for later reads.

## 2. Inspect The Collection

```python
collection.describe()
collection.bands
collection.bounds
collection.epsg
```

If the collection came from the built-in catalog, compare what you built against
the catalog descriptor:

```python
collection.compare_to_catalog()
```

Useful mental model:

- `geometry` is the scene footprint, stored as WKB footprint geometry.
- `crs` / `proj:epsg` describe the native raster CRS for each row.
- `*_metadata` columns store per-band COG header metadata used by the read path.
- Extra columns such as splits, labels, AOI IDs, or experiment metadata can live
  beside the raster metadata.

## 3. Filter Before Reading

```python
filtered = collection.subset(
    cloud_cover_lt=20,
    bbox=(77.55, 13.01, 77.58, 13.08),
)
```

Filtering works on metadata first. That keeps pixel reads focused on candidate
records instead of opening every raster and checking later.

## 4. Read Pixels

Choose the output surface that matches your task.

For xarray analysis:

```python
ds = filtered.get_xarray(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For array-first workflows:

```python
arr = filtered.get_numpy(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For GeoPandas output:

```python
gdf = filtered.get_gdf(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For TorchGeo pipelines:

```python
dataset = filtered.to_torchgeo_dataset(
    bands=["B04", "B03", "B02", "B08"],
    chip_size=256,
)
```

Everything after `to_torchgeo_dataset()` is standard TorchGeo sampler and
DataLoader code.

## 5. Sample Point Values

```python
import pyarrow as pa

points = pa.table({"lon": [77.56, 77.57], "lat": [13.03, 13.04]})

samples = filtered.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["B04", "B08"],
    geometry_crs=4326,
)
```

`samples` is a `pyarrow.Table`, so it can move into Arrow-native tools without
first becoming a pandas dataframe.

## What To Use When

Use `build()` when the dataset is already in Rasteret's catalog.

If it is not in the catalog:

- custom STAC API: use `build_from_stac(...)`
- existing Parquet or GeoParquet record table: use `build_from_table(...)`
- read-ready Arrow table in memory: use `as_collection(...)`
- exported Rasteret collection: use `load(...)`

## Practical Notes

!!! info "Native dtype is preserved"
    Rasteret preserves the native COG dtype. If you need floating-point math,
    cast explicitly or compute with an output surface that fits your workflow.

!!! tip "Share the collection, not a notebook full of setup code"
    `collection.export("path/")` writes a portable collection artifact. Another
    user can reopen it with `rasteret.load("path/")`.

!!! warning "CRS still matters"
    Query geometries default to EPSG:4326 (`geometry_crs=4326`). If your input
    points or polygons are in another CRS, pass the correct `geometry_crs`.
    Rasteret uses the raster CRS metadata in the collection to transform reads
    safely.

## Next

Next: [Concepts](../explanation/concepts.md)
