# TorchGeo Integration

`RasteretDataset` in `torchgeo.datasets` is the supported integration point for
using Rasteret collections in TorchGeo training pipelines. Install both packages
and pass a collection directly:

```python
from torchgeo.datasets import RasteretDataset
from torchgeo.samplers import RandomGeoSampler

dataset = RasteretDataset(collection=collection, bands=["B04", "B03", "B02"])
sampler = RandomGeoSampler(dataset, size=256, length=100)
```

This page covers the underlying public boundary that `RasteretDataset` (and any
custom `GeoDataset` subclass) relies on:

- `collection.to_table(...)` — for building the spatial/temporal sampling index
- `collection.read_window(...)` — for chip-level pixel reads

That split keeps TorchGeo dataset semantics on the TorchGeo side while Rasteret
owns collection metadata, COG planning, and byte-range pixel reads.

## Build A Sampling Table

```python
import rasteret

collection = rasteret.load("my_experiment")

table = collection.to_table(
    columns=[
        "id",
        "datetime",
        "geometry",
        "proj:epsg",
        "label",
        "B08_metadata",
        "B04_metadata",
        "B03_metadata",
    ],
)
```

TorchGeo can turn this Arrow-native collection metadata into its own GeoPandas
index, choose a sampling CRS/resolution, and keep Rasteret focused on pixel
reads.

## Read A Fixed Grid Window

```python
window = collection.read_window(
    record_ids=table.column("id").to_pylist()[:1],
    bounds=(500000.0, 999000.0, 501280.0, 1000000.0),
    res=(10.0, 10.0),
    bands=["B08", "B04", "B03"],
)
```

`read_window(...)` returns a NumPy array on the exact query grid. Overlapping
records are mosaicked internally with fixed-grid semantics.

## Filter Before Indexing

You can filter the collection first:

```python
train = collection.subset(cloud_cover_lt=20)
table = train.to_table(
    columns=[
        "id",
        "datetime",
        "geometry",
        "proj:epsg",
        "biomass_value",
        "B04_metadata",
        "B03_metadata",
        "B02_metadata",
        "B08_metadata",
    ],
)
```

For adding split and label columns before this step, see
[Bring Your Own AOIs, Points, And Metadata](enriched-collection-workflows.md).
For benchmark methodology and current numbers, see
[Benchmarks](../explanation/benchmark.md) and the
[TorchGeo Benchmark: Rasteret vs Native Rasterio notebook](../tutorials/05_torchgeo_benchmark_rasteret_vs_rasterio.ipynb).
