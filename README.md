<h1 align="center">🛰️ Rasteret</h1>

<p align="center">
  <strong>Made to beat cold starts.</strong><br>
  Index-first access to cloud-native GeoTIFF collections for ML and analysis.
</p>

<p align="center">
  <a href="https://terrafloww.github.io/rasteret"><img src="https://img.shields.io/badge/docs-terrafloww.github.io%2Frasteret-009DD1" alt="Documentation"></a>
  <a href="https://discord.gg/V5vvuEBc"><img src="https://img.shields.io/badge/Discord-chat-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://pypi.org/project/rasteret/"><img src="https://img.shields.io/pypi/v/rasteret?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/rasteret/"><img src="https://img.shields.io/pypi/pyversions/rasteret" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

---

Every cold start re-parses satellite image metadata over HTTP - per
scene, per band. Sentinel-2, Landsat, NAIP, every time. Your colleague
did it last Tuesday, CI did it overnight, PyTorch respawns DataLoader
workers every epoch. A single project racks up **millions of redundant
requests** before a pixel moves.

Rasteret parses those headers **once**, caches them in Parquet, and its
own reader fetches pixels concurrently with no GDAL in the path.
**Over 20x faster** on cold starts.

- **Easy** - three lines from STAC search or Parquet file to TorchGeo dataset
- **Zero downloads** - work with terabytes of imagery while storing only megabytes of metadata
- **No STAC at training time** - query once at setup; zero API calls during training
- **Reproducible** - same Parquet index = same records = same results
- **Native dtypes** - uint16 stays uint16; no silent float32 promotion in the read path
- **Shareable** - a 5 MB index captures scene selection, band metadata, and split assignments

Rasteret is an **opt-in accelerator**. Your TorchGeo samplers, DataLoader,
xarray workflows, and analysis tools stay the same - Rasteret handles the
async tile I/O underneath.

---

## Installation

```bash
pip install "rasteret[all]"    # recommended, includes all extras
```

> [!TIP]
> We recommend [uv](https://docs.astral.sh/uv/) for faster installs: `uv pip install "rasteret[all]"`

<details>
<summary><strong>More install options</strong></summary>

```bash
pip install rasteret                 # core only
pip install "rasteret[xarray]"       # + xarray output
pip install "rasteret[torchgeo]"     # + TorchGeo for ML
```

Available extras: `xarray`, `torchgeo`, `aws`, `azure`, `earthdata`, `examples`, `dev`, `docs`, `all`.
See [Getting Started](https://terrafloww.github.io/rasteret/getting-started/) for details.

> [!NOTE]
> **Requester-pays data (Landsat, etc.):** Install the `aws` extra and
> configure AWS credentials (`aws configure` or environment variables).
> Free public collections like Sentinel-2 on Element84 work without credentials.

</details>

---

## Built-in datasets

Rasteret ships with a growing catalog of datasets, no STAC URLs to memorize:

```
$ rasteret datasets list
ID                          Name                                       Coverage       License        Auth
earthsearch/sentinel-2-l2a  Sentinel-2 Level-2A                        global         proprietary    none
earthsearch/landsat-c2-l2   Landsat Collection 2 Level-2               global         proprietary    required
earthsearch/naip            NAIP                                       north-america  proprietary    required
earthsearch/cop-dem-glo-30  Copernicus DEM 30m                         global         proprietary    none
earthsearch/cop-dem-glo-90  Copernicus DEM 90m                         global         proprietary    none
pc/sentinel-2-l2a           Sentinel-2 Level-2A (Planetary Computer)   global         proprietary    required
pc/io-lulc-annual-v02       ESRI 10m Land Use/Land Cover               global         CC-BY-4.0      required
pc/alos-dem                 ALOS World 3D 30m DEM                      global         proprietary    required
pc/nasadem                  NASADEM                                    global         proprietary    required
pc/esa-worldcover           ESA WorldCover                             global         CC-BY-4.0      required
pc/usda-cdl                 USDA Cropland Data Layer                   conus          proprietary    required
aef/v1-annual               AlphaEarth Foundation Embeddings (Annual)  global         CC-BY-4.0      none
```

Each entry includes license metadata sourced from the authoritative STAC API,
and a `commercial_use` flag for quick filtering.

The catalog is open and community-driven. Each dataset entry is ~20 lines of
Python: One PR adds a dataset; every user gets accelerated access on the next release.

Pick any ID and pass it to `build()`. Don't see your dataset? Use
`build_from_stac()` for any STAC API, `build_from_table()` for existing
Parquet, or [add it to the catalog](https://terrafloww.github.io/rasteret/how-to/dataset-catalog/#add-your-own-catalog-entries-advanced)
so everyone benefits.

---

## Quick start

### Build a Collection

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2_training",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
)
```

`build()` picks the dataset from the catalog, queries the STAC API, parses
COG headers, and caches everything as Parquet. The next run loads in
milliseconds.

### ML training (TorchGeo)

```python
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets.utils import stack_samples

dataset = collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02", "B08"],
    chip_size=256,
)

sampler = RandomGeoSampler(dataset, size=256, length=100)
loader = DataLoader(dataset, sampler=sampler, batch_size=4, collate_fn=stack_samples)
```

### Analysis (xarray)

```python
ds = collection.get_xarray(
    geometries=(77.55, 13.01, 77.58, 13.08),  # bbox, Arrow array, Shapely, or WKB
    bands=["B04", "B08"],
)
ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04)
```

<details>
<summary><strong>Going further</strong></summary>

| What | Where |
|---|---|
| Datasets not in the catalog | [`build_from_stac()`](https://terrafloww.github.io/rasteret/how-to/collection-management/) |
| Parquet with COG URLs (Source Cooperative, STAC GeoParquet, custom) | [`build_from_table(path, name=...)`](https://terrafloww.github.io/rasteret/how-to/build-from-parquet/) |
| Multi-band COGs (AEF embeddings, etc.) | [AEF Embeddings guide](https://terrafloww.github.io/rasteret/how-to/aef-embeddings/) |
| Authenticated sources (PC, requester-pays, Earthdata, etc.) | [Custom Cloud Provider](https://terrafloww.github.io/rasteret/how-to/custom-cloud-provider/) |
| Share a Collection | `collection.export("path/")` then `rasteret.load("path/")` |
| Filter by cloud cover, date, bbox | [`collection.subset()`](https://terrafloww.github.io/rasteret/tutorials/) |

</details>

---

## Benchmarks

### Single request performance

Processing pipeline: Filter 450,000 scenes -> 22 matches -> Read 44 COG files

![Single request performance](./assets/single_timeseries_request.png)

### TorchGeo comparison (cold start)

Apples-to-apples: same AOIs, same scenes, same sampler, same DataLoader.
Both paths output identical `[batch, T, C, H, W]` tensors.
Cold-start numbers: no HTTP cache, no OS page cache, no pre-opened file handles.

| Scenario | TorchGeo | Rasteret | Speedup |
|---|---|---|---|
| Single AOI, 15 scenes | 9.08 s | 1.14 s | **8.0x** |
| Multi-AOI, 30 scenes | 42.05 s | 2.25 s | **18.7x** |
| Cross-CRS boundary, 12 scenes | 12.47 s | 0.59 s | **21.3x** |

![Processing time comparison](./assets/benchmark_results.png)
![Speedup breakdown](./assets/benchmark_breakdown.png)

Full methodology: [Benchmarks](https://terrafloww.github.io/rasteret/explanation/benchmark/)
 ·  Notebook: [`05_torchgeo_comparison.ipynb`](docs/tutorials/05_torchgeo_comparison.ipynb)
 ·  Blog: [blog.terrafloww.com](https://blog.terrafloww.com/rasteret-a-library-for-faster-and-cheaper-open-satellite-data-access/)

> [!IMPORTANT]
> Measured on 12-30 Sentinel-2 scenes. The speedup grows with scene count.
> If you run Rasteret on larger workloads, share your numbers on
> [GitHub Discussions](https://github.com/terrafloww/rasteret/discussions/categories/show-and-tell)
> or [Discord](https://discord.gg/V5vvuEBc).

---

## Scope and stability

| Area | Status |
|---|---|
| STAC + COG scene workflows | Stable |
| Parquet-first workflows (`build_from_table()`) | Stable |
| Multi-band / planar-separate COGs (`band_index`) | Stable |
| Multi-cloud (S3, Azure Blob, GCS) | Stable |
| Dataset catalog | Stable |
| TorchGeo adapter | Stable |

Rasteret is optimized for **remote, tiled GeoTIFFs** (COGs). It also works
with local tiled GeoTIFFs for indexing, filtering, and sharing collections.
Non-tiled TIFFs and non-TIFF formats are best handled by TorchGeo or rasterio.

---

## Documentation

Full docs at **[terrafloww.github.io/rasteret](https://terrafloww.github.io/rasteret)**:

| | |
|---|---|
| [Getting Started](https://terrafloww.github.io/rasteret/getting-started/) | Installation and first steps |
| [Tutorials](https://terrafloww.github.io/rasteret/tutorials/) | Six hands-on notebooks |
| [How-To Guides](https://terrafloww.github.io/rasteret/how-to/) | Task-oriented recipes |
| [API Reference](https://terrafloww.github.io/rasteret/reference/) | Auto-generated from source |
| [Architecture](https://terrafloww.github.io/rasteret/explanation/architecture/) | Design decisions |
| [Ecosystem Comparison](https://terrafloww.github.io/rasteret/explanation/interop/) | Rasteret vs TACO, async-geotiff, virtual-tiff |

## Contributing

The catalog grows fastest with community help:

- **Add a dataset**: write a ~20 line descriptor in `catalog.py`, open a PR. See [prerequisites](https://terrafloww.github.io/rasteret/how-to/dataset-catalog/#prerequisites-for-contributing-a-built-in-dataset) and [guide](https://terrafloww.github.io/rasteret/how-to/dataset-catalog/#add-your-own-catalog-entries-advanced)
- **Improve docs**: fix a typo, add an example, clarify a section
- **Build something new**: ingest drivers, cloud backends, readers. See [Architecture](https://terrafloww.github.io/rasteret/explanation/architecture/)

Every contribution benefits the whole community.
See [Contributing](https://terrafloww.github.io/rasteret/contributing/) for dev setup and we are happy to discuss all aspects of library.
Ideas welcome on [GitHub Discussions](https://github.com/terrafloww/rasteret/discussions) or join our [Discord](https://discord.gg/V5vvuEBc) to just chat.

## Technical notes

<details>
<summary><strong>GeoParquet and Parquet Raster</strong></summary>

Rasteret Collections are written as **GeoParquet 1.1** (WKB footprint geometry
+ `geo` metadata; coordinates in CRS84). Parquet is adding native
`GEOMETRY`/`GEOGRAPHY` logical types and GeoParquet 2.0 is evolving alongside
that; Rasteret tracks this and plans to adopt when ecosystem support stabilizes.

GeoParquet also has an **alpha "Parquet Raster"** draft for storing raster
payloads in Parquet. Rasteret does **not** write Parquet Raster files: pixels
stay in GeoTIFF/COGs, and Parquet stays the index.

</details>

<details>
<summary><strong>TorchGeo interop</strong></summary>

`to_torchgeo_dataset()` returns a standard TorchGeo `GeoDataset`, so you keep
using TorchGeo for sampling and training while Rasteret handles async tile I/O
underneath. This is pipeline-level interop, not a replacement for TorchGeo's
rasterio/GDAL-backed `RasterDataset` path (which is still the right choice for
non-tiled TIFFs and non-TIFF formats).

</details>

## License

Code: [Apache-2.0](LICENSE)
