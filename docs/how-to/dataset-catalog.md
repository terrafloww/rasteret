# Dataset Catalog

Use the dataset catalog when you want to build a Rasteret Collection from a
known dataset ID instead of manually wiring a STAC endpoint, GeoParquet path,
band map, and cloud configuration.

A catalog entry is usually not the collection itself. It is a lightweight
descriptor that tells Rasteret how to discover or load raster records. The
result of a build is still a normal `Collection`.

```text
catalog entry -> rasteret.build(...) -> Collection
```

The exception is `aef/v1-annual`, the maintained AlphaEarth Foundation
Embeddings Collection. That ID points at an existing read-ready Rasteret
Collection on Source Cooperative and should be opened with:

```python
collection = rasteret.load("aef/v1-annual")
```

All other built-in satellite data IDs are build recipes and should use
`rasteret.build(...)`.

## Browse The Catalog

From the CLI:

```bash
rasteret datasets list
rasteret datasets list --search sentinel
rasteret datasets info earthsearch/sentinel-2-l2a
```

Use `--json` when scripting:

```bash
rasteret datasets list --json
```

From Python:

```python
import rasteret

for dataset in rasteret.DatasetRegistry.list():
    print(dataset.id, dataset.name)
```

Catalog entries include practical metadata such as source type, available bands,
coverage hints, auth requirements, license fields, and example query values when
available. Treat license fields as a starting point and check the authoritative
provider license for production or commercial use.

## Build From A Catalog ID

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="bangalore",
    bbox=(77.55, 13.01, 77.58, 13.08),
    date_range=("2024-01-01", "2024-06-30"),
)
```

STAC-backed entries usually need `bbox` and `date_range`. GeoParquet-backed
entries may not, because the record table is already the build source.

The CLI equivalent is:

```bash
rasteret datasets build earthsearch/sentinel-2-l2a bangalore \
  --bbox 77.55,13.01,77.58,13.08 \
  --date-range 2024-01-01,2024-06-30
```

Use `--force` to rebuild an existing cache and `--workspace-dir` to choose a
different local workspace.

## Auth And Cloud Access

Some catalog entries point at public data. Others need cloud credentials or URL
signing. Inspect the entry before building:

```bash
rasteret datasets info earthsearch/landsat-c2-l2
```

Common cases:

| Source | Typical requirement |
| --- | --- |
| Earth Search public Sentinel-2 | Usually no credentials |
| Earth Search Landsat / NAIP requester-pays S3 | AWS credentials and requester-pays config |
| Planetary Computer | Azure/SAS signing support via `rasteret[azure]` |
| Earthdata-backed sources | Earthdata credentials |

For requester-pays buckets, private stores, Planetary Computer signing, and
custom credential providers, see
[Custom Cloud Provider](custom-cloud-provider.md).

## Compare A Built Collection To Its Catalog Entry

After building from the catalog, use `compare_to_catalog()` to see how your
subset relates to the source descriptor:

```python
collection.compare_to_catalog()
```

Use this when you want to check bands, date range, source coverage, or auth
metadata. If a collection was built from an unregistered table, use
`collection.describe()` instead.

## Register A Local Collection into Dataset Catalog

If you have a local exported collection or Parquet record table that you want to
reuse by dataset ID, register it:

```bash
rasteret datasets register-local local/bangalore ./bangalore_collection \
  --name "Bangalore Sentinel-2 subset" \
  --description "Reusable local collection for examples"
```

After registration:

```bash
rasteret datasets list --search bangalore
rasteret datasets info local/bangalore
```

You can then build/load through the catalog ID:

```python
collection = rasteret.build("local/bangalore", name="bangalore-copy")
```

By default, local registry entries are persisted to:

```text
~/.rasteret/datasets.local.json
```

Set `RASTERET_LOCAL_DATASETS_PATH` or pass `--registry-path` when you need a
different registry file.

## Export Or Remove A Local Entry

Export one local descriptor as JSON for review or sharing:

```bash
rasteret datasets export-local local/bangalore ./bangalore.dataset.json
```

Remove a local descriptor:

```bash
rasteret datasets unregister-local local/bangalore
```

This removes the catalog registration. It does not delete the underlying
collection artifact.

## Runtime Registration

For process-local registration in Python:

```python
import rasteret
from rasteret.catalog import DatasetDescriptor

rasteret.register(
    DatasetDescriptor(
        id="acme/field-survey-2024",
        name="ACME Field Survey",
        geoparquet_uri="/path/to/collection_or_record_table",
        description="Drone mosaics for 2024 survey.",
        band_map={"R": "red", "G": "green", "B": "blue"},
        license="proprietary",
        license_url="https://acme.example.com/license",
    )
)
```

This is useful in notebooks, tests, or applications that want a dataset ID for
the current process without writing to the local registry file.

For contributor guidance on adding built-in catalog entries, see
[Contributing](../contributing.md) and the
[`rasteret.catalog` API reference](../reference/catalog.md).
