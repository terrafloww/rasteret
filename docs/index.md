# Rasteret

### A high-performance rasterio/GDAL alternative for AI-first geospatial workflows.

Rasteret is an index-first I/O engine designed to eliminate the friction of training ML models on massive satellite imagery collections. By delivering up to **20x faster reads** from remote COGs compared to traditional sequential approaches, Rasteret lets you spend less time on "plumbing" and more time on science.

By separating the **Control Plane** (your metadata) from the **Data Plane** (the pixels), Rasteret removes the friction of cold-start header parsing and fragile multithreaded plumbing.

---

## Technical Evidence

If you are a skeptical engineer (we hope you are!), explore the data behind our architecture:

- [**Benchmarks**](explanation/benchmark.md): See how we compare against **GDAL cold starts** and **HuggingFace MajorTOM** streaming.
- [**Design Decisions**](explanation/design-decisions.md): Why we chose Parquet over Zarr, JSON, or SQLite.

---

## Where to start

If you are new, follow the story:

1. [**The Mental Model**](explanation/conceptual-roadmap.md): The "Verbs to Nouns" shift.
2. [**The LoC Win**](how-to/transitioning-from-rasterio.md): See why researchers are switching.
3. [**Getting Started**](getting-started/index.md): Install and build your first collection.

## What the workflow looks like

```python
import pyarrow as pa
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2",
    bbox=(-122.55, 37.65, -122.30, 37.90),
    date_range=("2024-01-01", "2024-02-01"),
)

sub = collection.subset(cloud_cover_lt=20)

arr = sub.get_numpy(
    geometries=(-122.50, 37.70, -122.40, 37.80),
    bands=["B04", "B08"],
)

points = pa.table({"lon": [-122.40, -122.39], "lat": [37.79, 37.80]})
samples = sub.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["B04"],
    geometry_crs=4326,
)
```

The collection stays at the center:

`build/load/as_collection -> subset/where -> get_xarray/get_numpy/sample_points/to_torchgeo_dataset`

## Dataset catalog

Rasteret ships with a growing built-in dataset catalog across sources such as
Earth Search, Planetary Computer, and AlphaEarth Foundation.

```bash
rasteret datasets list
```

```python
for d in rasteret.DatasetRegistry.list():
    print(d.id, d.name)
```

If the dataset is in the catalog, use `build()`. If not, use
`build_from_stac()` or `build_from_table()` and still end up with the same
collection-centered workflow.

## Where to start

If you are new:

1. Read the [**Conceptual Roadmap**](explanation/conceptual-roadmap.md) (Why Rasteret?)
2. View the [**Transitioning from Rasterio**](how-to/transitioning-from-rasterio.md) comparison (The LOC Win)
3. Follow the [Getting Started](getting-started/index.md) guide
4. Run the [Quickstart tutorial](tutorials/01_quickstart.ipynb)

If you already know your task:

- existing Parquet or GeoParquet source:
  [Build from Parquet](how-to/build-from-parquet.md)
- many point samples or masking behavior:
  [Point Sampling and Masking](how-to/point-sampling-and-masking.md)
- collection enrichment, splits, labels, workflow metadata:
  [Enriched Parquet Workflows](how-to/enriched-parquet-workflows.md)
- built-in datasets and local dataset IDs:
  [Dataset Catalog](how-to/dataset-catalog.md)

## Why people keep the collection around

The collection is where Rasteret becomes more than a one-off reader:

- cached header metadata avoids repeated setup cost
- table-native filtering fits DuckDB, PyArrow, Polars, pandas, and GeoPandas workflows
- extra columns like splits, AOIs, IDs, and labels can live with the same artifact
- exported collections are easy to share and reload later

## Scope

- Best fit: remote, tiled GeoTIFFs / COGs
- Also works with local tiled GeoTIFFs
- Non-tiled TIFFs and non-TIFF formats are better handled by TorchGeo or rasterio directly

Rasteret is an opt-in accelerator. It works alongside TorchGeo, xarray,
rasterio, DuckDB, and Arrow-native tooling rather than replacing them.

For architecture and design rationale, see [Explanation](explanation/index.md).
For exact APIs, see [API Reference](reference/index.md).

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    First install, first collection, first reads.

    [:octicons-arrow-right-24: Get started](getting-started/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Hands-on notebook walkthroughs.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-book-open-variant:{ .lg .middle } **How-To Guides**

    ---

    Task-focused guides for common Rasteret workflows.

    [:octicons-arrow-right-24: How-to guides](how-to/index.md)

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
