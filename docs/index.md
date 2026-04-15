# Rasteret

### Build a queryable raster collection once. Read pixels up to 20x faster.

Rasteret helps you work with cloud-hosted tiled GeoTIFFs and COGs without
rewriting the same STAC, GDAL, threading, CRS, and header-parsing code for
every experiment.

The core idea is simple:

- keep **metadata** in a queryable Arrow/Parquet collection
- keep **pixels** in the original COGs
- cache the expensive raster header metadata once
- read only the byte ranges needed for your AOIs, chips, or points

That gives you a Collection-centered workflow for ML training, xarray analysis,
NumPy arrays, point sampling, and GeoPandas-friendly outputs.

```text
STAC / record table  ->  Rasteret Collection  ->  get_numpy / get_xarray / TorchGeo
       build once          query/filter/share        read pixels on demand
```

## Why This Helps

Traditional raster workflows often start by opening many remote TIFFs just to
discover their tile layout, transforms, dtype, nodata, and CRS. That is painfully slow
in notebooks, training jobs, and new docker containers because the setup cost repeats.

Rasteret moves that setup into a collection build step. The collection stores
image metadata, footprints, assets, splits or labels, and parsed COG header
metadata in Arrow/Parquet. Later reads can go straight from a filtered table of
records to byte-range reads against the original COGs.

This is most useful when you:

- train or evaluate models over many satellite/drone scenes
- repeatedly sample the same raster collection with different AOIs
- need TorchGeo/xarray/NumPy outputs from the same source
- want dataset splits, labels, IDs, and scene metadata next to the imagery index
- want Arrow-native metadata that works with tools like PyArrow, DuckDB,
  GeoPandas, LanceDB, and related dataframe systems

## First Path Through The Docs

If you are new, read these in order:

1. [Getting Started](getting-started/index.md): install Rasteret and build your first collection.
2. [Concepts](explanation/concepts.md): understand the collection model and why Rasteret is not just another raster reader.
3. [Migrating from Rasterio](how-to/migrating-from-rasterio.md): see the side-by-side workflow shift if you already know rasterio or GDAL.
4. [How-To Guides](how-to/index.md): choose a task guide for collection ingest, AOI/point tables, point sampling, TorchGeo, catalog, or cloud access.
5. [Tutorials](tutorials/index.md): follow notebook walkthroughs after the core ideas are familiar.

## A Minimal Workflow

```python
import pyarrow as pa
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2-demo",
    bbox=(-122.55, 37.65, -122.30, 37.90),
    date_range=("2024-01-01", "2024-02-01"),
)

filtered = collection.subset(cloud_cover_lt=20)

arr = filtered.get_numpy(
    geometries=(-122.50, 37.70, -122.40, 37.80),
    bands=["B04", "B08"],
)

points = pa.table(
    {
        "site_id": ["site-a", "site-b"],
        "lon": [-122.40, -122.39],
        "lat": [37.79, 37.80],
    }
)
samples = filtered.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["B04"],
    geometry_crs=4326,
)
```

The pattern stays the same:

```text
build/load/as_collection -> subset/where -> choose an output surface
```

Common output surfaces:

| Method | Output | Use it when |
| --- | --- | --- |
| `get_numpy()` | `numpy.ndarray` | You want raw arrays for ML or custom processing. |
| `get_xarray()` | `xarray.Dataset` | You want labeled raster data, coordinates, and CRS metadata. |
| `get_gdf()` | `geopandas.GeoDataFrame` | You want one row per geometry/record result, with AOI metadata preserved. |
| `sample_points()` | `pyarrow.Table` | You want pixel values at points, with point metadata preserved. |
| `to_torchgeo_dataset()` | TorchGeo `GeoDataset` | You want a TorchGeo-compatible dataset backed by Rasteret reads. |

## Dataset Catalog

Rasteret ships with a small built-in catalog for known cloud-native raster
sources such as Earth Search, Planetary Computer, and AlphaEarth Foundation.

```bash
rasteret datasets list
```

```python
import rasteret

for descriptor in rasteret.DatasetRegistry.list():
    print(descriptor.id, descriptor.name)
```

Use `rasteret.build("catalog/id", ...)` when a dataset is already registered.
Use `build_from_stac()` or `build_from_table()` when you have your own source.
Either way, the result is a `Collection`.

## What Rasteret Is Not

Rasteret is not a replacement for every raster tool.

- Best fit: remote or local **tiled GeoTIFFs / COGs**.
- Best workflow: build a collection once, then reuse it for many reads.
- Not the best fit: one-off reads of a single local raster, non-tiled TIFFs, or
  arbitrary non-TIFF raster formats.

It works alongside rasterio, TorchGeo, xarray, DuckDB, GeoPandas, and Arrow
tools. The goal is to remove repeated cloud-raster plumbing, not hide the
geospatial model.

## Reference Material

- [Schema Contract](explanation/schema-contract.md): what a Rasteret collection stores.
- [Design Decisions](explanation/design-decisions.md): why Parquet + COGs instead of moving pixels into Parquet.
- [Benchmarks](explanation/benchmark.md): cold-start and cloud-read measurements.
- [API Reference](reference/index.md): exact method signatures and docstrings.

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    First install, first collection, first reads.

    [:octicons-arrow-right-24: Get started](getting-started/index.md)

-   :material-book-open-variant:{ .lg .middle } **How-To Guides**

    ---

    Task-focused guides for common Rasteret workflows.

    [:octicons-arrow-right-24: How-to guides](how-to/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Hands-on notebook walkthroughs.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Public API surface and docstrings.

    [:octicons-arrow-right-24: API reference](reference/index.md)

-   :material-lightbulb-on:{ .lg .middle } **Explanation**

    ---

    Architecture, guarantees, and ecosystem context.

    [:octicons-arrow-right-24: Explanation](explanation/index.md)

-   :material-heart:{ .lg .middle } **Contributing**

    ---

    Contributor and maintainer guidance.

    [:octicons-arrow-right-24: Contributing](contributing.md)

</div>

Next: [Getting Started](getting-started/index.md)
