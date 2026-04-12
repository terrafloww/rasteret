# Collection Management

Learn how to manage a Rasteret Collection and its local lifecycle:
build/import, inspect, export, reload, filter, or delete.

For first-time data ingest details, use:

- [Build from Parquet and Arrow Tables](build-from-tables.md) for external Parquet, GeoParquet,
  DuckDB, Polars, or Arrow record tables.
- [Dataset Catalog](dataset-catalog.md) for built-in dataset IDs and local
  catalog registration.
- [Custom Cloud Provider](custom-cloud-provider.md) for requester-pays,
  authenticated buckets, URL rewriting, and custom backends.

## What Gets Managed

A Rasteret Collection is an Arrow/Parquet table of raster records. It stores the
metadata Rasteret needs for fast reads, including record IDs, footprints, assets,
CRS sidecars, and optional COG header metadata. Pixel bytes stay in the source
COGs.

The default local workspace is:

```text
~/rasteret_workspace
```

Rasteret uses two suffixes for cached collection directories:

| Suffix | Created by |
| --- | --- |
| `_stac` | `build()`, `build_from_stac()`, `rasteret collections build` |
| `_records` | `build_from_table()`, `rasteret collections import` |

You can pass `workspace_dir=...` in Python or `--workspace-dir ...` in the CLI
when you want a different location.

## CLI Workflow

Build or refresh a STAC-backed collection cache:

```bash
rasteret collections build bangalore \
  --stac-api https://earth-search.aws.element84.com/v1 \
  --collection sentinel-2-l2a \
  --bbox 77.55,13.01,77.58,13.08 \
  --date-range 2024-01-01,2024-06-30
```

Import an existing Parquet or GeoParquet record table:

```bash
rasteret collections import my-collection \
  --record-table s3://my-bucket/stac-items.parquet
```

`collections import` materializes a local collection from the record table. If
your source table still needs COG header enrichment for pixel reads, use the
Python `build_from_table(..., enrich_cog=True)` path instead.

Inspect local caches:

```bash
rasteret collections list
rasteret collections info bangalore
```

Delete a local cache:

```bash
rasteret collections delete bangalore
```

Use `--json` on `build`, `import`, `list`, and `info` when scripting. Use
`--yes` with `delete` to skip the confirmation prompt.

## Python Entry Points

Choose the entry point based on what you already have:

| Situation | Use |
| --- | --- |
| Registered catalog dataset | `rasteret.build("catalog/id", ...)` |
| Custom STAC API | `rasteret.build_from_stac(...)` |
| External Parquet/GeoParquet/Arrow record table | `rasteret.build_from_table(...)` |
| Existing exported collection artifact | `rasteret.load(path)` |
| In-memory read-ready Arrow object | `rasteret.as_collection(obj)` |

Example:

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="bangalore",
    bbox=(77.55, 13.01, 77.58, 13.08),
    date_range=("2024-01-01", "2024-06-30"),
)
```

When `name` is provided, Rasteret can cache the built collection in the default
workspace with the given name as the folder name.

Pass `force=True` when you want to rebuild an existing cache.

## Inspect A Collection

Use `describe()` for a human-readable summary:

```python
collection.describe()
```

The returned object is also programmatic:

```python
summary = collection.describe()
summary["bands"]
summary.data
```

Common quick checks:

```python
collection.bands
collection.bounds
collection.epsg
len(collection)
```

If the collection came from a registered catalog dataset, compare it to the
source descriptor:

```python
collection.compare_to_catalog()
```

This is useful when you built a date or AOI subset and want to see how it relates
to the full catalog entry. If there is no matching catalog descriptor, use
`collection.describe()` instead.

## Export And Reload

Export a portable collection artifact:

```python
collection.export("./bangalore_collection")
```

Reload it later:

```python
reloaded = rasteret.load("./bangalore_collection")
```

The export contains the collection metadata table, labels/splits you added, COG
header metadata, and asset references. It does not copy the source raster pixel
bytes.

## List Cached Collections

```python
from pathlib import Path

from rasteret import Collection

for item in Collection.list_collections(Path.home() / "rasteret_workspace"):
    print(item["name"], item["kind"], item["size"])
```

or

```bash
rasteret collections list
```

Each item includes fields such as `name`, `kind`, `data_source`, `date_range`,
`size`, and `created`.

## Filter Metadata Before Reading

Use `subset()` for common filters:

```python
train = collection.subset(
    cloud_cover_lt=15,
    date_range=("2024-03-01", "2024-04-30"),
    bbox=(77.55, 13.03, 77.57, 13.06),
    split="train",
)
```

Use `where()` for custom Arrow dataset expressions:

```python
import pyarrow.dataset as ds

clear = collection.where(ds.field("eo:cloud_cover") < 5.0)
```

Then read pixels from the filtered collection:

```python
arr = clear.get_numpy(
    geometries=(77.55, 13.03, 77.57, 13.06),
    bands=["B04", "B08"],
)
```

For labels, splits, AOI joins, and other experiment metadata, see
[Enriched Collection Workflows](enriched-collection-workflows.md).
