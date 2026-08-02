# TorchGeo Integration

`RasteretDataset` in `torchgeo.datasets` is the supported integration point for using Rasteret collections in TorchGeo training pipelines. Install both packages and pass a collection directly:

```python
from torchgeo.datasets import RasteretDataset
from torchgeo.samplers import RandomGeoSampler

dataset = RasteretDataset(collection=collection, bands=["B04", "B03", "B02"])
sampler = RandomGeoSampler(dataset, size=256, length=100)
```

This page covers the underlying public boundary that `RasteretDataset` (and any custom `GeoDataset` subclass) relies on:

- `collection.footprints(crs=..., band=...)` — per-record COG pixel-grid bbox as a `GeoDataFrame` in the target CRS, derived from band metadata. Use this to build a precise spatial index without reprojecting the WGS84 `geometry` column (reprojection inflates edge bounds by projection curvature).
- `collection.to_table(...)` — Arrow/GeoArrow access to the rest of the collection metadata (datetimes, custom columns, raster CRS sidecars).
- `collection.read_window(...)` — fixed-grid window read for chip-level pixel data.

That split keeps TorchGeo dataset semantics on the TorchGeo side while Rasteret owns collection metadata, COG planning, and byte-range pixel reads.

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

TorchGeo can turn this Arrow-native collection metadata into its own GeoPandas index, choose a sampling CRS/resolution, and keep Rasteret focused on pixel reads.

## Read A Fixed Grid Window

```python
window = collection.read_window(
    record_ids=table.column("id").to_pylist()[:1],
    bounds=(500000.0, 999000.0, 501280.0, 1000000.0),
    res=(10.0, 10.0),
    bands=["B08", "B04", "B03"],
)
```

`read_window(...)` returns a NumPy array on the exact query grid. Overlapping records are mosaicked internally with fixed-grid semantics.

## Filtering

Metadata and attribute filtering (cloud cover, date range, custom columns) belongs on the collection before construction. Spatial ROI is a sampler concern — pass `roi=` to `RandomGeoSampler`. `dataset.index` is a public GeoDataFrame and can be sliced after construction if needed.

`subset(...)` covers the common cases. For anything more complex — joins, custom expressions, multi-step transforms — the collection is Arrow-native, so you can work with it in DuckDB, Polars, or pandas and wrap the result back with `rasteret.as_collection(table, data_source=collection.data_source)`.

Use `collection.subset(...)` for common filters:

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

For adding split and label columns before this step, see [Bring Your Own AOIs, Points, And Metadata](https://terrafloww.github.io/rasteret/how-to/enriched-collection-workflows/index.md). For benchmark methodology and current numbers, see [Benchmarks](https://terrafloww.github.io/rasteret/explanation/benchmark/index.md) and the [TorchGeo Benchmark: Rasteret vs Native Rasterio notebook](https://terrafloww.github.io/rasteret/tutorials/05_torchgeo_benchmark_rasteret_vs_rasterio/index.md).
