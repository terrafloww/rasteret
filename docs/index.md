# Rasteret

Rasteret is a Python library for 20x+ faster reads of GeoTIFF than Rasterio/GDAL
---

!!! failure "The cold-start tax"

    Your colleague read those Sentinel-2 scenes last Tuesday. Rasterio/GDAL
    re-parsed every TIFF metadata - per scene, per band. Even in CI.
    In the intern's notebook. PyTorch respawns DataLoader workers every
    epoch, so your ML training run re-parses TIFF metadata hundreds of times over.

    A single project repeats **millions of redundant requests** across a
    team - zero pixels delivered.

!!! success "What Rasteret does"

    Parse headers **once**, cache in Parquet, read pixels concurrently
    with no GDAL in the path.

    ```text
    STAC API / GeoParquet  -->  Parquet Index  -->  Tile-level byte reads
           (once)                 (queryable)          (no GDAL, custom I/O)
    ```

!!! info "Category: index-first geospatial image retrieval"

    Rasteret treats Parquet as the **control plane** (scene metadata + COG
    header metadata + user-enriched columns), and COG object storage as the
    **data plane** (pixel bytes fetched on demand). Metadata stays table-native;
    imagery bytes stay in source COGs.

---

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **No cold-start penalty**

    ---

    Header metadata lives in Parquet, not behind HTTP.
    New kernel? New pod? Same speed as the tenth run.

-   :material-cloud-off-outline:{ .lg .middle } **Zero downloads**

    ---

    Work with terabytes of cloud imagery while storing
    only megabytes of metadata locally.

-   :material-api-off:{ .lg .middle } **No STAC at training time**

    ---

    Query STAC once at setup. Zero API calls during
    training loops, no rate-limiting risk.

-   :material-share-variant:{ .lg .middle } **Reproducible and shareable**

    ---

    Same Parquet index = same records = same results.
    Share a few MB file and collaborators skip re-indexing.

</div>

---

## Dataset catalog

Rasteret ships with **12 built-in datasets**: Sentinel-2, Landsat,
NAIP, Copernicus DEM, ESRI Land Cover, ESA WorldCover, USDA CDL,
ALOS DEM, NASADEM, and AlphaEarth Foundation embeddings across Earth Search,
Planetary Computer, and AEF.

```bash
rasteret datasets list          # CLI
```

```python
for d in rasteret.DatasetRegistry.list():
    print(d.id, d.name)         # Python
```

The catalog is open and community-driven. Each entry is ~20 lines of Python.
One PR adds a dataset and every user gets access on the next release. No proprietary APIs, no platform lock-in. See
[Dataset Catalog](how-to/dataset-catalog.md) for details, or
[Design Decisions](explanation/design-decisions.md) for the thinking behind it.

Pick any ID and pass it to `build()`. For datasets not in the catalog, use
`build_from_stac()` or `build_from_table()`. Reopen persisted collections with
`load()`, or re-wrap read-ready Arrow tables with `as_collection()`.

---

!!! note "New here?"

    Start with [Getting Started](getting-started/index.md), then run
    the [Quickstart tutorial](tutorials/01_quickstart.ipynb).

## How it works

```python
import pyarrow as pa
import rasteret

# 1) Build an index (one-time, cached as GeoParquet)
collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2",
    bbox=(-122.55, 37.65, -122.30, 37.90),
    date_range=("2024-01-01", "2024-02-01"),
)

# 2) Filter metadata (in-memory, instant)
sub = collection.subset(cloud_cover_lt=20)

# 3) Read pixels on demand (bbox, Arrow column, Shapely geometry, or WKB)
arr = sub.get_numpy(geometries=(-122.50, 37.70, -122.40, 37.80), bands=["B04", "B08"])

# 4) Turn rasters into a point feature table (Arrow-native output)
points = pa.table({"lon": [-122.40, -122.39], "lat": [37.79, 37.80]})
samples = sub.sample_points(points=points, x_column="lon", y_column="lat", bands=["B04"], geometry_crs=4326)
```

Any Arrow-compatible table/column source works for `geometries=` and `points=`.

The workflow stays collection-centric:
`build/load/as_collection -> subset/where -> get_xarray/get_numpy/sample_points/to_torchgeo_dataset`.

`build()` picks from a growing catalog of pre-registered datasets across
Earth Search, Planetary Computer, and AlphaEarth Foundation. For existing Parquet
files - [Source Cooperative](https://source.coop) exports, STAC GeoParquet,
or your own catalog - use `build_from_table()`.
For multi-band COGs like [AlphaEarth Foundation embeddings](how-to/aef-embeddings.md),
use `band_index` in the asset dict to select individual bands from a shared file.
For custom STAC APIs not in the catalog, use `build_from_stac()`.
See the [API Reference](reference/index.md) for full method signatures.

## Benchmarks

![Rasteret vs TorchGeo benchmark](assets/benchmark_results.png)

These are cold-start numbers: no HTTP cache, no OS page cache. Every new
notebook kernel, VM, k8s pod, or CI runner starts cold. That is the
real-world scenario, and where Rasteret's Parquet index matters most.

For full methodology and numbers, see [Benchmarks](explanation/benchmark.md).

??? note "HF `datasets` baseline (Major TOM keyed patches)"
    Separate benchmark against Hugging Face payload-Parquet workflows using
    `datasets.load_dataset(..., streaming=True, filters=...)` (PyArrow-backed predicate pushdown):

    | Patches | HF `datasets` parquet filters | Rasteret index+COG | Speedup |
    |---:|---:|---:|---:|
    | 120 | 46.83 s | 12.09 s | **3.88x** |
    | 1000 | 771.59 s | 118.69 s | **6.50x** |

    ![HF vs Rasteret processing time](assets/benchmark_hf_results.png)
    ![HF vs Rasteret speedup](assets/benchmark_hf_speedup.png)

    Full details: [Benchmarks](explanation/benchmark.md).

!!! tip "Share your speed-ups"

    Running Rasteret on your own data? We'd love to hear your numbers.
    Post in [Show and Tell](https://github.com/terrafloww/rasteret/discussions/categories/show-and-tell)
    or drop them in [Discord](https://discord.gg/V5vvuEBc).

## Scope

- Optimized for **remote, tiled GeoTIFFs** (COGs), where the biggest speedups happen.
- Works with **local tiled GeoTIFFs** too. Speedups are smaller without network overhead, but the Parquet index is still useful for organizing, filtering, and sharing collections.
- Non-tiled GeoTIFFs and non-TIFF formats (NetCDF, HDF5) are best handled by TorchGeo or rasterio directly.
- CRS is encoded via CF conventions (pyproj); no rioxarray dependency.

Rasteret is an **opt-in accelerator**: `RasteretGeoDataset` is a standard
TorchGeo `GeoDataset` subclass that honors the full contract (`index`, `crs`,
`res`, `__getitem__`). Samplers, DataLoader, transforms, and dataset composition
(`IntersectionDataset`, `UnionDataset`) all work unchanged. Rasteret replaces
the I/O backend, not the training interface.
For how Rasteret relates to other tools, see
[Ecosystem Comparison](explanation/interop.md).

---

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Installation and first steps.

    [:octicons-arrow-right-24: Get started](getting-started/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Hands-on notebooks for learning Rasteret.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-book-open-variant:{ .lg .middle } **How-To Guides**

    ---

    Task-oriented recipes for common workflows.

    [:octicons-arrow-right-24: How-to guides](how-to/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Auto-generated from source code docstrings.

    [:octicons-arrow-right-24: API reference](reference/index.md)

-   :material-lightbulb-on:{ .lg .middle } **Explanation**

    ---

    Architecture, design decisions, and ecosystem context.

    [:octicons-arrow-right-24: Explanation](explanation/index.md)

-   :material-heart:{ .lg .middle } **Contributing**

    ---

    Add a dataset, improve docs, or build something new.
    All contributions are welcome.

    [:octicons-arrow-right-24: Contributing](contributing.md)

</div>
