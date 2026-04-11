<h1 align="center">🛰️ Rasteret</h1>

<p align="center">
  <strong>The AI practitioner's multiplier for cloud-native satellite data.</strong><br>
  <em>A high-performance rasterio/GDAL alternative for scaleable ML workflows.</em>
</p>
<p align="center">
Rasteret helps you manage and read massive satellite imagery collections with zero friction. <br>
It provides a high-performance "drop-in" backend for **TorchGeo**, **xarray**, and **NumPy** that is up to 20x faster than traditional GDAL-based workflows.
</p>

<p align="center">
  <a href="https://terrafloww.github.io/rasteret"><img src="https://img.shields.io/badge/docs-terrafloww.github.io%2Frasteret-009DD1" alt="Documentation"></a>
  <a href="https://discord.gg/86NgTB3Xa"><img src="https://img.shields.io/badge/Discord-chat-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://pypi.org/project/rasteret/"><img src="https://img.shields.io/pypi/v/rasteret?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/rasteret/"><img src="https://img.shields.io/pypi/pyversions/rasteret" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

---

## Why Rasteret?

Geospatial data science is often 80% "plumbing." You spend hours writing `pystac-client` loops, manual `ThreadPoolExecutor` code, and fragile CRS-alignment logic just to get a batch of pixels for your model.

**Rasteret turns those 80% into a single line of code.**

It separates the **Control Plane** (managing your scenes, labels, and splits in a local Parquet index) from the **Data Plane** (streaming pixels directly from cloud COGs).

### The "Friction" vs. "Flow" Comparison

**The Old Way (25+ lines of fragile plumbing)**:
1. Search STAC catalog ✅
2. Loop over items ✅
3. Handle pagination ✅
4. Filter by cloud cover ✅
5. **Wait 500ms per file** to parse remote TIFF headers (GDAL cold start) ❌
6. Manage `ThreadPoolExecutor` manually ❌
7. Manually stack results and align CRS ❌

**The Rasteret Way (3 lines of robust code)**:
```python
import rasteret

# 1. Load or Build your collection (Index is local, metadata is relational)
collection = rasteret.load("my_s2_experiment")

# 2. Query like a Table: "Give me the training scenes with <10% clouds"
filtered = collection.subset(split="train", cloud_cover_lt=10)

# 3. Batch Read: "Fetch aligned pixels for these 1000 polygons"
data = filtered.get_numpy(geometries=my_polygons, bands=["B04", "B08"])
```

---

## Key Features

- **🚀 20x Faster Cold Starts**: By caching tile-layout metadata locally, Rasteret jumps straight to the pixels, skipping expensive remote header parsing, which happens in every new environment.
- **📦 Seamless "Drop-in" Backends**: Boost **TorchGeo** or **xarray** performance by simply swapping the reader. No need to rewrite your training code.
- **🧬 Relational Imagery**: Store your labels, `train/val/test` splits, and custom metadata directly in the imagery index. No more separate CSVs.
- **🛠️ Zero-Config Throughput**: Automatic cloud storage presigning with `Obstore`, and custom async I/O handles the networking so you don't have to.

## Performance

Rasteret's claims are backed by rigorous, reproducible benchmarks. We measure across three dimensions: cold-start latency, cloud-native scale, and comparison against legacy "data-inside-parquet" patterns.

### 1. Cold-start comparison with TorchGeo
Same AOIs, same scenes, same sampler, same DataLoader. Rasteret eliminates the "cold start tax" by caching IFD headers in the local Parquet index.

| Scenario | rasterio/GDAL (Standard) | Rasteret (Index-First) | Speedup |
|---|---|---|---|
| Single AOI, 15 scenes | 9.08 s | 1.14 s | **8x** |
| Multi-AOI, 30 scenes | 42.05 s | 2.25 s | **19x** |
| Cross-CRS boundary | 12.47 s | 0.59 s | **21x** |

![Processing time comparison](./assets/benchmark_results.png)
![Speedup breakdown](./assets/benchmark_breakdown.png)

### 2. The Cloud vs. Edge Comparison
How does Rasteret stack up against **Google Earth Engine (GEE)** or a highly parallelized Rasterio setup for time-series extraction?

| Library | First Run (Cold) | Subsequent Runs (Hot) |
|---------|-----------------|-----------------------|
| **Rasterio** + ThreadPool | 32 s | 24 s |
| **Google Earth Engine** | 10–30 s | 3–5 s |
| **Rasteret** | **3 s** | **3 s** |

![Single request performance](./assets/single_timeseries_request.png)

### 3. HuggingFace `MajorTOM` vs. Rasteret
Recent "images-inside-Parquet" approaches (like MajorTOM) try to store image bytes in Parquet files. Rasteret keeps imagery in cloud COGs while using Parquet as a high-performance index—delivering better throughput without the data movement overhead.

| Patches | HF `datasets` (streaming) | Rasteret index+COGs | Speedup |
|---:|---:|---:|---:|
| 120 | 46.83 s | 12.09 s | **3.88x** |
| 1000 | 771.59 s | 118.69 s | **6.50x** |

![HF vs Rasteret speedup](./assets/benchmark_hf_speedup.png)

*All numbers measured on AWS us-west-2 4CPU machine (same region as data) vs. cold-start GDAL.*

---

## Technical Deep Dives

For the full architectural rationale, methodology, and reproducibility scripts, see:

- [**Full Benchmarks Guide**](https://terrafloww.github.io/rasteret/explanation/benchmark/): Methodology and results.
- [**Design Decisions**](https://terrafloww.github.io/rasteret/explanation/design-decisions.md): Why we chose Parquet + COGs
- [**Schema Contract**](https://terrafloww.github.io/rasteret/explanation/schema-contract/): The internal anatomy of a Collection.

```text
STAC API / GeoParquet  -->  Parquet Collection  -->  Tile-level byte reads
       (once)                  (queryable)             (no GDAL hot path)
```

## Quick Start

### 1. Build a Collection
```python
import rasteret

# Build from any STAC API or Parquet Metadata table
collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2_training",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30")
)
```

### 2. Turbocharge your ML (TorchGeo)
Rasteret provides a high-performance backend that honors the `GeoDataset` contract.

```python
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler

# Same API as TorchGeo, much faster pixel pipe
dataset = collection.to_torchgeo_dataset(bands=["B04", "B08"], chip_size=256)

sampler = RandomGeoSampler(dataset, size=256, length=100)
loader  = DataLoader(dataset, sampler=sampler, batch_size=4)
```

### 3. Fast Xarray creation
```python
ds = collection.get_xarray(geometries=my_aoi, bands=["B04", "B08"])
ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04)
```

## Key Entry Points

Rasteret is built for flexibility. Choose the output format that fits your existing workflow:

| Method | Output | Purpose |
|---|---|---|
| [**`to_torchgeo_dataset()`**](https://terrafloww.github.io/rasteret/reference/integrations/torchgeo/) | `RasteretGeoDataset` | Drop-in high-performance backend for **TorchGeo** training. |
| [**`get_xarray()`**](https://terrafloww.github.io/rasteret/reference/core/collection/#rasteret.core.collection.Collection.get_xarray) | `xarray.Dataset` | Quick create Xarray for analysis. |
| [**`get_numpy()`**](https://terrafloww.github.io/rasteret/reference/core/collection/#rasteret.core.collection.Collection.get_numpy) | `numpy.ndarray` | Raw pixel arrays (`[N, C, H, W]`) directly. |
| [**`get_gdf()`**](https://terrafloww.github.io/rasteret/reference/core/collection/#rasteret.core.collection.Collection.get_gdf) | `GeoDataFrame` | Metadata and pixel arrays as a standard geopandas dataframe. |
| [**`sample_points()`**](https://terrafloww.github.io/rasteret/reference/core/collection/#rasteret.core.collection.Collection.sample_points) | `DataFrame` | Exact pixel values at points geometries with intuitive configurable fallback for nodata pixels |

---

Full documentation at **[terrafloww.github.io/rasteret](https://terrafloww.github.io/rasteret)**:

- [**Concepts**](https://terrafloww.github.io/rasteret/explanation/concepts/): Why Rasteret?
- [**Transitioning from Rasterio**](https://terrafloww.github.io/rasteret/how-to/transitioning-from-rasterio/): Side-by-side patterns.
- [**Turbocharging TorchGeo**](https://terrafloww.github.io/rasteret/how-to/turbocharging-torchgeo/): Scaling your DL loaders.
- [**Tutorials**](https://terrafloww.github.io/rasteret/tutorials/): Hands-on examples.

## License

Code: [Apache-2.0](LICENSE)
