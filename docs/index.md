# Rasteret

**Made to beat cold starts.** Index-first access to cloud-native GeoTIFF collections for ML and geospatial analysis.

---

!!! failure "The problem no one fixed"

    Your colleague read those Sentinel-2 scenes last Tuesday. The tools
    re-parsed every file header over HTTP - per scene, per band. So did CI.
    So did the intern's notebook. PyTorch respawns DataLoader workers every
    epoch, so your own training run re-parses them hundreds of times over.

    A single project racks up **millions of redundant requests** across a
    team - zero pixels delivered.

!!! success "What Rasteret does"

    Parse headers **once**, cache in Parquet, read pixels concurrently
    with no GDAL in the path.

    ```text
    STAC API / GeoParquet  -->  Parquet Index  -->  Tile-level byte reads
           (once)                 (queryable)          (no GDAL, no headers)
    ```

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
    Share a 5 MB file and collaborators skip re-indexing.

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
One PR adds a dataset and every user gets accelerated access on the next
release. No proprietary APIs, no platform lock-in. See
[Dataset Catalog](how-to/dataset-catalog.md) for details, or
[Design Decisions](explanation/design-decisions.md) for the thinking behind it.

Pick any ID and pass it to `build()`. For datasets not in the catalog, use
`build_from_stac()` or `build_from_table()`.

---

!!! note "New here?"

    Start with [Getting Started](getting-started/index.md), then run
    [Tutorial 01 - Quickstart: xarray](tutorials/01_quickstart_xarray.ipynb).

## How it works

```python
import rasteret

# 1. Build index (one-time, cached)
collection = rasteret.build("earthsearch/sentinel-2-l2a", name="s2", bbox=(...), date_range=(...))
collection.bands   # ['B01', 'B02', ..., 'B12', 'SCL']

# 2. Filter metadata (in-memory, instant)
sub = collection.subset(cloud_cover_lt=20, date_range=("2024-03-01", "2024-06-01"))

# 3. Read pixels - pass a bbox, Arrow column, Shapely geometry, or WKB
ds = sub.get_xarray(geometries=(-122.5, 37.7, -122.3, 37.9), bands=["B04", "B08"])
dataset = sub.to_torchgeo_dataset(bands=["B04", "B03", "B02"])    # TorchGeo
```

`build()` picks from a growing catalog of pre-registered datasets across
Earth Search and Planetary Computer. For existing Parquet
files - [Source Cooperative](https://source.coop) exports, STAC GeoParquet,
or your own catalog - use `build_from_table()`.
For multi-band COGs like [AlphaEarth Foundation embeddings](how-to/aef-embeddings.md),
use `band_index` in the asset dict to select individual bands from a shared file.
For custom STAC APIs not in the catalog, use `build_from_stac()`.
See the [API Reference](reference/index.md) for full method signatures.

## Benchmarks

![Rasteret vs TorchGeo benchmark](assets/benchmark_breakdown.png)

These are cold-start numbers: no HTTP cache, no OS page cache. Every new
notebook kernel, VM, k8s pod, or CI runner starts cold. That is the
real-world scenario, and where Rasteret's Parquet index matters most.

For full methodology and numbers, see [Benchmarks](explanation/benchmark.md).

!!! tip "Share your speed-ups"

    Running Rasteret on your own data? We'd love to hear your numbers.
    Post in [Show and Tell](https://github.com/terrafloww/rasteret/discussions/categories/show-and-tell)
    or drop them in [Discord](https://discord.gg/V5vvuEBc).

## Scope

- Optimized for **remote, tiled GeoTIFFs** (COGs), where the biggest speedups happen.
- Works with **local tiled GeoTIFFs** too. Speedups are smaller without network overhead, but the Parquet index is still useful for organizing, filtering, and sharing collections.
- Non-tiled GeoTIFFs and non-TIFF formats (NetCDF, HDF5) are best handled by TorchGeo or rasterio directly.
- CRS is encoded via CF conventions (pyproj); no rioxarray dependency.

Rasteret is an **opt-in accelerator**: it integrates cleanly with TorchGeo by
returning a standard `GeoDataset`, and it does not try to replace TorchGeo's
samplers, DataLoader patterns, or transforms.
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
    Every contribution benefits the whole ecosystem.

    [:octicons-arrow-right-24: Contributing](contributing.md)

</div>
