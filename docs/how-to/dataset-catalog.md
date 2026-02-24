# Dataset Catalog

Rasteret ships with a built-in **dataset catalog**: a registry of known
datasets so you can build a Collection without remembering STAC API URLs or
endpoint details. Most users only need the `build()` function shown below;
the later sections cover browsing, local registration, and advanced
customisation.

The built-in catalog includes 12 datasets: Sentinel-2, Landsat,
NAIP, Copernicus DEM, ESRI Land Cover, ESA WorldCover, USDA CDL,
ALOS DEM, NASADEM, and AlphaEarth Foundation embeddings. Run
`rasteret datasets list` to see them all.

Each catalog entry includes **license metadata** sourced from the
authoritative STAC API: a license identifier, a URL to the full license
text, and a `commercial_use` flag so you can quickly tell whether the
data can be used commercially.

In short:

- A **Collection** is the Parquet index you build and reuse for fast reads.
- A **catalog entry** is lightweight metadata that tells Rasteret *how to build or load* a Collection.
- Catalog entries also include **coverage/temporal hints**, **license info**, and (when available) a small **example bbox + date range** you can use to sanity-check access.

---

## Build a Collection from the catalog (CLI)

```bash
# Build a local Collection from a catalog entry
rasteret datasets build earthsearch/sentinel-2-l2a bangalore \
  --bbox 77.55,13.01,77.58,13.08 \
  --date-range 2024-01-01,2024-06-30
```

The resulting Collection is written under `~/rasteret_workspace/` by default.
You can override this with `--workspace-dir`.

To browse the catalog before building:

```bash
# List every known dataset
rasteret datasets list

# Inspect a single catalog entry
rasteret datasets info earthsearch/sentinel-2-l2a
```

Many entries include an **Example bbox** and **Example time**. These are known-good values used by Rasteret's live smoke tests, and are a good first check when debugging auth or rate limits.

---

## Build a Collection from the catalog (Python)

```python
import rasteret

# Build from a catalog entry (STAC-backed datasets require bbox + date_range)
collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="bangalore",
    bbox=(77.55, 13.01, 77.58, 13.08),
    date_range=("2024-01-01", "2024-06-30"),
)
```

If you want explicit control over STAC endpoints and collection IDs, use
[`build_from_stac()`](../reference/rasteret.md).

!!! note "Authenticated datasets"

    Some catalog entries require credentials:

    - **Requester-pays S3 (Landsat, NAIP):** install `rasteret[aws]` and configure AWS creds.
    - **Planetary Computer (Azure):** install `rasteret[azure]` for SAS signing (signing can be rate-limited; see Custom Cloud Provider).
    - **Authenticated STAC catalogs (Earthdata, etc.):** install `rasteret[earthdata]`
      and configure credentials via `~/.netrc` or environment variables when needed.

    For details and custom backends, see
    [Custom Cloud Provider](../how-to/custom-cloud-provider.md).

!!! note "Testing new catalog entries"

    Built-in catalog entries should be exercised by Rasteret's network smoke
    tests (they auto-skip when credentials/extras are missing). When you add a
    dataset, document which extras and credentials are required to run it
    locally, and keep those prerequisites explicit in the PR description.

    See the [Compatibility Matrix](../explanation/compatibility-matrix.md) and
    [Contributing](../contributing.md) for how the test suite is structured.

??? tip "Browsing the catalog programmatically"

    You can list or search catalog entries via the `DatasetRegistry` class:

    ```python
    for d in rasteret.DatasetRegistry.list():
        print(d.id, d.name)
    ```

    This is useful for interactive exploration in a notebook, but most
    scripts can simply pass a known dataset ID straight to `build()`.

!!! info "How this differs from TorchGeo's dataset classes"

    TorchGeo ships per-dataset Python classes that download data locally
    and read from disk via rasterio/GDAL. Rasteret's catalog points at
    cloud-hosted data (STAC APIs, GeoParquet): no downloads, no custom
    code per dataset. You get a standard `Collection` from any catalog
    entry, then read pixels on demand from the cloud.

---

!!! info "Everything below is optional"

    The two sections above (CLI and Python) cover the common workflow.
    The remaining sections explain how to register your own datasets and
    customise the catalog. You can safely skip them until you need them.

---

## Make local collections reusable (optional)

Local collections are shareable as Parquet directories/files. The catalog layer
is what makes them *discoverable* across sessions and usable via `build()`.

To register a local Collection as a catalog entry:

```bash
rasteret datasets register-local local/bangalore ./bangalore_202401-06_sentinel_stac
```

By default, Rasteret persists local catalog entries to:

- `~/.rasteret/datasets.local.json` (or `RASTERET_LOCAL_DATASETS_PATH`)

This is a simple JSON list of entries. It is editable and
friendly to version control if you choose to track it.

### Exporting and removing local entries

When you want to hand off a Collection, exporting a single catalog entry
file makes review and sharing easier than copying an entire registry file:

```bash
rasteret datasets export-local local/bangalore ./bangalore.dataset.json
```

To remove a local entry later:

```bash
rasteret datasets unregister-local local/bangalore
```

---

## Add your own catalog entries (advanced)

!!! note "Advanced"

    This section is for users who need to programmatically register custom
    datasets at runtime or contribute new built-in entries to Rasteret.

There are two supported patterns:

1. **Runtime registration** in Python (process-local):

   ```python
   import rasteret
   from rasteret.catalog import DatasetDescriptor

   rasteret.register(
        DatasetDescriptor(
            id="acme/field-survey-2024",
            name="ACME Field Survey",
            geoparquet_uri="/path/to/collection_parquet",
            description="Drone mosaics for 2024 survey.",
            band_map={"R": "red", "G": "green", "B": "blue"},
            license="proprietary",
            license_url="https://acme.example.com/license",
        )
   )
   ```

   The `band_map` maps user-facing band codes to STAC asset keys.
   It is auto-registered so downstream code resolves band names
   without users needing to touch `BandRegistry` directly.

2. **Contribute a built-in entry** via PR: add a `DatasetDescriptor` to
   `src/rasteret/catalog.py`. See the prerequisites checklist below
   and existing entries in `catalog.py` for the pattern.

For full field documentation, see the [`rasteret.catalog`](../reference/catalog.md)
API reference.

---

## Prerequisites for contributing a built-in dataset

Before adding a new dataset to the built-in catalog, verify these
requirements. Every built-in entry must actually work with Rasteret's
pipeline. Listing a dataset that can't be ingested is worse than not
listing it at all.

### 1. STAC access works

The dataset must be reachable via either a **STAC API** (with a `/search`
endpoint) or a **static STAC catalog** (`catalog.json` on S3). Verify
with:

```python
# STAC API
import pystac_client
client = pystac_client.Client.open("<stac_api_url>")
col = client.get_collection("<collection_id>")

# Static catalog
import pystac
cat = pystac.Catalog.from_file("<catalog_json_url>")
items = list(cat.get_all_items())  # should return items
```

### 2. Band map has at least one working COG asset

The `band_map` must map at least one band code to a STAC asset key that
points to a Cloud-Optimized GeoTIFF (COG). Rasteret parses COG headers
during `build()`. If no assets can be parsed, Rasteret can't index or read
the dataset.

Check a sample item's asset keys:

```python
item = items[0]
for key, asset in item.assets.items():
    print(f"{key}: {asset.media_type}")
# Look for "image/tiff" or "application=geotiff" entries
```

### 3. End-to-end `build()` succeeds

Run a real build with a small item limit:

```python
import rasteret
col = rasteret.build(
    "<dataset_id>",
    name="smoke-test",
    query={"max_items": 2},
    force=True,
)
print(col.dataset.count_rows())  # should be > 0
```

For STAC API datasets (non-static catalogs), `bbox` and `date_range` are required.

### 4. License is verified from the authoritative source

Pull the license from the STAC API or catalog metadata. Do not guess:

```python
# STAC API
col = client.get_collection("<collection_id>")
print(col.license)  # "CC-BY-4.0", "proprietary", etc.
license_links = [l.href for l in col.links if l.rel == "license"]

# Static catalog - check item-level properties
item = items[0]
print(item.properties.get("license"))
```

Set `commercial_use=False` when the license prohibits it (e.g.
`CC-BY-NC-4.0`).

### 5. Descriptor includes required metadata

Include at minimum: `id`, `name`, `description`, `stac_api` (or
`geoparquet_uri`), `band_map`, `license`, `license_url`, `spatial_coverage`,
`temporal_range`. For static catalogs, set `static_catalog=True`.

---

## Catalog spec direction

The underlying `DatasetDescriptor` class is intentionally **spec-aligned**: a
pragmatic working format that can evolve into portable YAML/JSON. The goal is
for catalog entries to be shareable across tools, not just Rasteret.
