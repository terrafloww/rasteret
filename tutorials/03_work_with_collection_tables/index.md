# Work With Collection Tables[¶](#work-with-collection-tables)

Use the Collection table for filtering, export/reload workflows, and Arrow-native metadata queries before reading pixels.

For the full technical narrative, see the [**Concepts**](https://terrafloww.github.io/rasteret/explanation/concepts/).

In \[ \]:

Copied!

```
from pathlib import Path

import pyarrow.dataset as ds

import rasteret
from rasteret import Collection
```

from pathlib import Path import pyarrow.dataset as ds import rasteret from rasteret import Collection

## 1. Build and cache a Collection[¶](#1-build-and-cache-a-collection)

`build()` looks up the dataset in the catalog and creates a cached Parquet index.

In \[ \]:

Copied!

```
import shutil

workspace = (
    Path.cwd() / ".cache" / "notebooks" / "03_work_with_collection_tables"
).resolve()
shutil.rmtree(workspace, ignore_errors=True)
workspace.mkdir(parents=True, exist_ok=True)

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="bangalore-s2-demo-fresh",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
    workspace_dir=workspace,
)

print(f"Scenes: {collection.dataset.count_rows()}")
```

import shutil workspace = ( Path.cwd() / ".cache" / "notebooks" / "03_work_with_collection_tables" ).resolve() shutil.rmtree(workspace, ignore_errors=True) workspace.mkdir(parents=True, exist_ok=True) collection = rasteret.build( "earthsearch/sentinel-2-l2a", name="bangalore-s2-demo-fresh", bbox=(77.5, 12.9, 77.7, 13.1), date_range=("2024-01-01", "2024-06-30"), workspace_dir=workspace, ) print(f"Scenes: {collection.dataset.count_rows()}")

## 2. Export and reload[¶](#2-export-and-reload)

`build()` already caches the Collection as Parquet in your workspace. Use `export()` when you want a copy at a specific path - for sharing with a teammate, versioning, or persisting a filtered view. `load()` reloads any Parquet that follows the Collection schema.

In \[ \]:

Copied!

```
cache_path = workspace / f"{collection.name}_stac"
cached = rasteret.load(cache_path)
print(f"Reloaded from cache: {cached.dataset.count_rows()} rows")

export_path = workspace / "exports" / "demo_exported"
collection.export(export_path)

reloaded = rasteret.load(export_path, name="demo_exported")
print(f"Reloaded from export: {reloaded.dataset.count_rows()} rows")
```

cache_path = workspace / f"{collection.name}\_stac" cached = rasteret.load(cache_path) print(f"Reloaded from cache: {cached.dataset.count_rows()} rows") export_path = workspace / "exports" / "demo_exported" collection.export(export_path) reloaded = rasteret.load(export_path, name="demo_exported") print(f"Reloaded from export: {reloaded.dataset.count_rows()} rows")

## 3. Discover cached collections[¶](#3-discover-cached-collections)

### CLI[¶](#cli)

```
rasteret collections list                   # list all local collections
rasteret collections info bangalore-s2-demo # details for one collection
rasteret datasets list                      # browse built-in catalog entries
```

See [Collection Management](https://terrafloww.github.io/rasteret/tutorials/how-to/collection-management.md) for the full CLI reference (build, import, delete, JSON output, etc.).

### Python API[¶](#python-api)

`Collection.list_collections()` scans the workspace for `*_stac` and `*_records` directories.

In \[ \]:

Copied!

```
cached = Collection.list_collections(workspace)
for c in cached:
    print(f"  {c['name']:30s}  {c['size']:>5} rows  ({c['kind']})")
```

cached = Collection.list_collections(workspace) for c in cached: print(f" {c\['name'\]:30s} {c\['size'\]:>5} rows ({c['kind']})")

## 4. Filter with `subset()`[¶](#4-filter-with-subset)

`subset()` applies Arrow pushdown predicates, no Python-side scanning. Filters compose with AND.

In \[ \]:

Copied!

```
# Cloud cover filter
clear = collection.subset(cloud_cover_lt=10)
print(f"Cloud < 10%: {clear.dataset.count_rows()} rows")

# Date range filter
spring = collection.subset(date_range=("2024-03-01", "2024-05-31"))
print(f"Mar-May: {spring.dataset.count_rows()} rows")

# Combined
clear_spring = collection.subset(
    cloud_cover_lt=10,
    date_range=("2024-03-01", "2024-05-31"),
)
print(f"Cloud < 10% + Mar-May: {clear_spring.dataset.count_rows()} rows")

# Bbox filter (uses scalar bbox columns from the normalize layer)
try:
    spatial = collection.subset(bbox=(77.55, 13.0, 77.60, 13.05))
    print(f"Bbox filter: {spatial.dataset.count_rows()} rows")
except ValueError as e:
    print(f"Bbox: {e}")
```

# Cloud cover filter

clear = collection.subset(cloud_cover_lt=10) print(f"Cloud < 10%: {clear.dataset.count_rows()} rows")

# Date range filter

spring = collection.subset(date_range=("2024-03-01", "2024-05-31")) print(f"Mar-May: {spring.dataset.count_rows()} rows")

# Combined

clear_spring = collection.subset( cloud_cover_lt=10, date_range=("2024-03-01", "2024-05-31"), ) print(f"Cloud < 10% + Mar-May: {clear_spring.dataset.count_rows()} rows")

# Bbox filter (uses scalar bbox columns from the normalize layer)

try: spatial = collection.subset(bbox=(77.55, 13.0, 77.60, 13.05)) print(f"Bbox filter: {spatial.dataset.count_rows()} rows") except ValueError as e: print(f"Bbox: {e}")

## 5. Raw Arrow expression with `where()`[¶](#5-raw-arrow-expression-with-where)

For filters not covered by `subset()`, pass any `pyarrow.dataset.Expression`.

In \[ \]:

Copied!

```
expr = ds.field("eo:cloud_cover") < 5.0
very_clear = collection.where(expr)
print(f"where(cloud < 5): {very_clear.dataset.count_rows()} rows")
```

expr = ds.field("eo:cloud_cover") < 5.0 very_clear = collection.where(expr) print(f"where(cloud < 5): {very_clear.dataset.count_rows()} rows")

## 6. Load from an existing Parquet file[¶](#6-load-from-an-existing-parquet-file)

`build_from_table()` loads any Parquet file with `id`, `datetime`, `geometry`, `assets` columns. Supports `column_map` for renaming and `filter_expr` for scan-time pushdown.

```
# Example with column remapping and filter
collection = rasteret.build_from_table(
    "s3://my-bucket/record-table.parquet",
    column_map={"scene_id": "id", "timestamp": "datetime"},
    filter_expr=ds.field("eo:cloud_cover") < 20,
)
```

## Summary[¶](#summary)

- `export()`: write a portable copy of the Collection (for sharing/versioning)
- `load()`: reload any Collection Parquet (build cache or exported copy)
- `list_collections()`: discover what's cached
- `subset()`: cloud cover, date range, bbox, split filters (Arrow pushdown)
- `where()`: raw Arrow expressions
- `build_from_table()`: load any Parquet file with optional column mapping

Next: [Custom Cloud and Bands](https://terrafloww.github.io/rasteret/tutorials/04_custom_datasets_with_rasteret/index.md)
