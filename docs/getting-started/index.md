# Getting Started

This guide walks you through your first installation and your first automated "Build once, reuse often" workflow.

## Installation

Rasteret requires Python 3.12 or later.

```bash
uv pip install rasteret
```

Add extras for specific integrations:
```bash
uv pip install "rasteret[xarray]"    # For get_xarray()
uv pip install "rasteret[torchgeo]"  # For ML training
uv pip install "rasteret[all]" # Incase you want to explore the whole thing anyway
```

## First Workflow

This section shows you how to move from a remote catalog and into your first pixel-ready collection.

### 1. Build a collection

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2-bangalore",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
)
```

This works because `earthsearch/sentinel-2-l2a` is already in Rasteret's
built-in catalog.

!!! tip "Why two steps?"
    By separating the **Build** (Control Plane) from the **Read** (Data Plane), we remove the network latency of re-parsing TIFF headers every time you run your script. For the full technical story behind this shift, see the [**Conceptual Roadmap**](../explanation/conceptual-roadmap.md).

Useful inspection calls:

```python
collection.bands
collection.bounds
collection.epsg
len(collection)
```

If you want to compare the collection you built against the source dataset:

```python
collection.compare_to_catalog()
```

### 2. Read pixels

All of these read paths operate from the same collection.

For analysis:

```python
ds = collection.get_xarray(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For array-first workflows:

```python
arr = collection.get_numpy(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For TorchGeo pipelines:

```python
dataset = collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02", "B08"],
    chip_size=256,
)
```

Everything after `to_torchgeo_dataset()` is standard TorchGeo.

### 3. Sample point values

```python
import pyarrow as pa

points = pa.table({"lon": [77.56, 77.57], "lat": [13.03, 13.04]})

samples = collection.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["B04", "B08"],
    geometry_crs=4326,
)
```

`samples` is a `pyarrow.Table`, so it stays easy to use with Arrow-native tools.

That is the same pattern again: one collection, different output surfaces.

## Browse the catalog first, if needed

If you do not already know which dataset ID you want, inspect the catalog:

```bash
rasteret datasets list
```

```python
import rasteret

for d in rasteret.DatasetRegistry.list():
    print(d.id, d.name)
```

## What to use when

Use `build()` when the dataset is already in Rasteret's catalog.

If not:

- custom STAC API:
  [Build from STAC via `build_from_stac()`](../reference/rasteret.md)
- existing Parquet or GeoParquet:
  [Build from Parquet via `build_from_table()`](../how-to/build-from-parquet.md)
- already have a read-ready collection table in memory:
  use `as_collection(...)`
- already have an exported collection:
  use `load(...)`

## A few practical notes

!!! info "Native dtypes are preserved"
    Rasteret preserves the native COG dtype. If you need floating-point values
    for downstream math, cast explicitly.

!!! tip "Sharing collections"
    `collection.export("path/")` writes a portable artifact that another user
    can reopen with `rasteret.load("path/")`.

!!! tip "Keep the collection at the center"
    When you are unsure what to do next, come back to the collection-first
    pattern:
    build or load a collection, filter it, then choose the output surface you need.

## Next pages

After this page, go where your task takes you:

- hands-on notebook walkthroughs:
  [Tutorials](../tutorials/index.md)
- existing Parquet or GeoParquet source:
  [Build from Parquet](../how-to/build-from-parquet.md)
- point sampling and masking behavior:
  [Point Sampling and Masking](../how-to/point-sampling-and-masking.md)
- collection enrichment, splits, labels, experiment metadata:
  [Enriched Parquet Workflows](../how-to/enriched-parquet-workflows.md)
- local collection lifecycle:
  [Collection Management](../how-to/collection-management.md)
- built-in dataset IDs and local dataset registration:
  [Dataset Catalog](../how-to/dataset-catalog.md)
- auth and requester-pays reads:
  [Custom Cloud Provider](../how-to/custom-cloud-provider.md)
