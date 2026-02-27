# Getting Started

## Installation

Rasteret requires Python 3.12 or later.

```bash
uv pip install rasteret
```

Add extras as needed:

```bash
uv pip install "rasteret[xarray]"       # + xarray output
uv pip install "rasteret[torchgeo]"     # + TorchGeo for ML pipelines
uv pip install "rasteret[aws]"          # + requester-pays buckets (Landsat, NAIP)
```

Combine extras: `uv pip install "rasteret[xarray,aws]"`

??? note "All extras"

    | Extra | What it adds | When you need it |
    |-------|-------------|-----------------|
    | `xarray` | xarray | `get_xarray()` output |
    | `torchgeo` | TorchGeo `>=0.9.0` | ML training pipelines |
    | `aws` | boto3 | Requester-pays buckets (Landsat, NAIP) - credential resolution via boto3 |
    | `azure` | planetary-computer | Planetary Computer signed URLs |
    | `earthdata` | requests | Earthdata / DAAC auth (temporary S3 credentials) |
    | `dev` | pytest, ruff, pre-commit | Running tests and linting |
    | `docs` | mkdocs + plugins | Building documentation locally |

`get_numpy()` and `get_gdf()` are available in the base install (no extra needed).

Verify the installation:

```bash
python -c "import rasteret; print(rasteret.version())"
```

### Running notebooks

=== "Jupyter / JupyterLab"

    Jupyter runs each notebook in a **kernel**, a separate Python process.
    To use your Rasteret environment as a kernel:

    ```bash
    uv pip install ipykernel
    python -m ipykernel install --user --name rasteret --display-name "Rasteret"
    ```

    Then select the **Rasteret** kernel when you open a notebook.

=== "marimo"

    [marimo](https://marimo.io) manages dependencies inline, no kernel
    registration needed.  Just `uv pip install marimo` alongside Rasteret
    and run `marimo edit notebook.py`.

=== "VS Code"

    VS Code auto-detects virtual environments.  Open a `.ipynb` file,
    click the kernel picker (top-right), and select the Python interpreter
    from your Rasteret venv.

---

## Key concepts

- **Collection** - a local Parquet index of raster scenes + cached COG headers. Build, filter, read, share.
- **Catalog** - the registry of known data sources (Sentinel-2, Landsat, NAIP, ...). What `build()` looks up.
- **Workspace** - the directory where Collections are cached (`~/rasteret_workspace/` by default).

---

## Browse the catalog

Before building, see what's available:

=== "CLI"

    ```bash
    $ rasteret datasets list
    ID                          Name                                     Coverage       Auth
    earthsearch/sentinel-2-l2a  Sentinel-2 Level-2A                      global         none
    earthsearch/cop-dem-glo-90  Copernicus DEM 90m                       global         none
    earthsearch/landsat-c2-l2   Landsat Collection 2 Level-2             global         required
    earthsearch/naip            NAIP                                     north-america  required
    earthsearch/cop-dem-glo-30  Copernicus DEM 30m                       global         none
    pc/io-lulc-annual-v02       ESRI 10m Land Use/Land Cover             global         required
    pc/esa-worldcover           ESA WorldCover                           global         required
    aef/v1-annual               AlphaEarth Foundation Embeddings         global         none
    ...
    ```

=== "Python"

    ```python
    import rasteret
    for d in rasteret.DatasetRegistry.list():
        print(d.id, d.name)
    ```

Pick any dataset ID and pass it to `build()` in the next section. For data not
in the catalog, use `build_from_stac()` or `build_from_table()`, or
[add it to the catalog](../how-to/dataset-catalog.md#add-your-own-catalog-entries-advanced)
so the whole community benefits.

---

## Quick start

### 1. Build a Collection

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2-bangalore",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
)
collection
# Collection('s2-bangalore', source='sentinel-2-l2a', bands=13, records=42, crs=32643, 2024-01-01..2024-06-30)

collection.bands     # ['B01', 'B02', ..., 'B12', 'SCL']
collection.bounds    # (77.50, 12.90, 77.70, 13.10)
collection.epsg      # [32643]
len(collection)      # 42
```

!!! info "What just happened?"
    Rasteret searched the STAC catalog, parsed the COG tile-layout headers for
    every matching scene, and cached everything in a local Parquet index
    (in `~/rasteret_workspace/` by default). The next call with the same
    parameters reuses the cached index instantly; no network requests,
    no header parsing. This is the core idea: **build once, read many times**.

!!! tip "Collection vs catalog"
    A Collection contains only the scenes you indexed (your bbox, date range,
    filters).  Use `compare_to_catalog()` to see what you built vs what the
    source offers:

    ```python
    collection.compare_to_catalog()
    # Collection: s2-bangalore
    #
    #   Property  Value
    #   ────────  ─────────────────────────────────────────────────────────
    #   Records   42
    #   Bands     B01, B02, B03, B04, B05 (+8 more) (13/13)
    #   CRS       EPSG:32643
    #   Bounds    (77.5000, 12.9000, 77.7000, 13.1000)
    #   Dates     2024-01-01 .. 2024-06-30  (source: 2015-06-23 .. present)
    #   Source    earthsearch/sentinel-2-l2a (Sentinel-2 Level-2A)
    #   Coverage  global
    #   Auth      none
    ```

    In Jupyter/marimo notebooks this renders as a styled HTML table.

This example uses Sentinel-2 from Earth Search, which is public; no
credentials needed. For requester-pays or authenticated sources, see
[Custom Cloud Provider](../how-to/custom-cloud-provider.md).

### 2. Read pixels

Pick whichever output fits your workflow:

=== "ML training (TorchGeo)"

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

    Everything after `to_torchgeo_dataset()` is standard TorchGeo - your
    samplers, transforms, and training loop stay the same.

=== "Analysis (xarray)"

    ```python
    ds = collection.get_xarray(
        geometries=(77.55, 13.01, 77.58, 13.08),  # bbox tuple
        bands=["B04", "B08"],
    )
    ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04)
    ```

    `geometries` also accepts Arrow arrays (from GeoParquet), Shapely
    objects, or raw WKB. See [Build from Parquet](../how-to/build-from-parquet.md)
    for the Arrow-native path.

=== "Fast arrays (NumPy)"

    ```python
    arr = collection.get_numpy(
        geometries=(77.55, 13.01, 77.58, 13.08),
        bands=["B04", "B08"],
    )
    # shape: [N, C, H, W] for multi-band, [N, H, W] for single-band
    ```

    Use this when you want direct NumPy output for custom ML/data pipelines
    without xarray coordinates.

That's it for the basics. Two calls: `build()` to index, then read pixels.

!!! info "Data types"

    Rasteret preserves the native COG dtype. Sentinel-2 data returns as
    `uint16`, not `float32`. If you need float for computation (e.g., NDVI),
    cast explicitly: `ds.B04.astype("float32")`.

!!! tip "Which function do I use?"

    | Situation | Use |
    |---|---|
    | Dataset in the catalog (Sentinel-2, Landsat, NAIP, ...) | `rasteret.build("earthsearch/sentinel-2-l2a", ...)` |
    | Custom STAC API not in the catalog | `rasteret.build_from_stac(stac_api="...", ...)` |
    | Existing Parquet with COG URLs ([Source Cooperative](https://source.coop), STAC GeoParquet, custom) | `rasteret.build_from_table("s3://...parquet", ...)` |
    | Raw local/S3 COG files (no STAC/Parquet index yet) | First create a Parquet record table (`id`, `datetime`, `geometry`, `assets`), then `build_from_table(..., enrich_cog=True)` |
    | You already have a read-ready Arrow table from an existing Collection | `rasteret.as_collection(table, data_source=collection.data_source)` |
    | Someone shared a Collection with you | `rasteret.load("path/to/collection/")` |

**Sharing**: `collection.export("path/")` writes a portable copy. Your teammate runs `rasteret.load("path/")`.

`build*` functions ingest/normalize external data, `as_collection()` re-wraps read-ready in-memory Arrow objects, and `load()` reopens persisted artifacts.

---

## Going further

Once the quick start works, explore these as you need them:

| I want to... | Go to |
|---|---|
| Follow a hands-on notebook | [Tutorials](../tutorials/index.md) |
| Build from existing Parquet (Source Cooperative, STAC GeoParquet, custom) | [Build from Parquet](../how-to/build-from-parquet.md) |
| Enrich a collection with AOIs and labels | [Enriched Parquet Workflows](../how-to/enriched-parquet-workflows.md) |
| Use train/val/test splits and labels | [ML Training with Splits](../how-to/ml-training-splits.md) |
| Manage cached collections (build/import/list/info/delete) | [Collection Management](../how-to/collection-management.md) |
| Browse the dataset catalog and reuse local Collections | [Dataset Catalog](../how-to/dataset-catalog.md) |
| Read authenticated/requester-pays data (Landsat, PC, Earthdata) | [Custom Cloud Provider](../how-to/custom-cloud-provider.md) |
| Understand how Rasteret works (Explanation) | [Explanation](../explanation/index.md) |
| See how Rasteret fits with TorchGeo/xarray/rasterio | [Ecosystem Comparison](../explanation/interop.md) |
| See the full API | [API Reference](../reference/index.md) |

??? note "Development install"

    ```bash
    git clone https://github.com/terrafloww/rasteret.git
    cd rasteret
    uv sync --all-extras
    uv run pytest -q
    ```
