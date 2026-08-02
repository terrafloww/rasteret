# Build From Parquet And Arrow Tables[¶](#build-from-parquet-and-arrow-tables)

Turn a Parquet or Arrow table of COG records into a read-ready Rasteret Collection.

For the full written guide, see [Build from Parquet and Arrow Tables](https://terrafloww.github.io/rasteret/how-to/build-from-tables/).

## 1. Build a Collection from a remote Parquet[¶](#1-build-a-collection-from-a-remote-parquet)

`build_from_table()` reads the Parquet from S3 via PyArrow, validates the four required columns (`id`, `datetime`, `geometry`, `assets`), and produces a Collection.

Source Cooperative data lives in a public S3 bucket. Set `AWS_NO_SIGN_REQUEST` so PyArrow skips credential lookup.

In \[ \]:

Copied!

```
import os

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

import rasteret

SOURCE_COOP_URI = (
    "s3://us-west-2.opendata.source.coop" "/maxar/maxar-opendata/maxar-opendata.parquet"
)

collection = rasteret.build_from_table(SOURCE_COOP_URI, name="maxar-opendata")

print(f"Collection: {collection.name}")
print(f"Scenes: {collection.dataset.count_rows()}")
print(f"Columns: {collection.dataset.schema.names[:8]}...")
```

import os os.environ["AWS_NO_SIGN_REQUEST"] = "YES" import rasteret SOURCE_COOP_URI = ( "s3://us-west-2.opendata.source.coop" "/maxar/maxar-opendata/maxar-opendata.parquet" ) collection = rasteret.build_from_table(SOURCE_COOP_URI, name="maxar-opendata") print(f"Collection: {collection.name}") print(f"Scenes: {collection.dataset.count_rows()}") print(f"Columns: {collection.dataset.schema.names[:8]}...")

## 2. Explore the Collection with DuckDB[¶](#2-explore-the-collection-with-duckdb)

The Collection is backed by Arrow. DuckDB reads Arrow tables with zero copy - pass the Python variable directly, no file I/O.

> Requires `duckdb`: `pip install duckdb`

In \[ \]:

Copied!

```
import duckdb

# Arrow table from the Collection - this is the variable DuckDB reads
maxar = collection.dataset.to_table()

con = duckdb.connect()

# What disaster events are in this catalog?
con.sql("""
    SELECT
        replace(
            split_part(split_part(assets.visual.href, '/events/', 2), '/ard/', 1),
            '-', ' '
        ) AS event,
        count(*) AS scenes,
        min(datetime)::date AS earliest,
        max(datetime)::date AS latest,
        round(avg(gsd), 2) AS avg_gsd_m
    FROM maxar
    WHERE assets.visual IS NOT NULL
    GROUP BY event
    ORDER BY scenes DESC
""").show()
```

import duckdb

# Arrow table from the Collection - this is the variable DuckDB reads

maxar = collection.dataset.to_table() con = duckdb.connect()

# What disaster events are in this catalog?

con.sql(""" SELECT replace( split_part(split_part(assets.visual.href, '/events/', 2), '/ard/', 1), '-', ' ' ) AS event, count(\*) AS scenes, min(datetime)::date AS earliest, max(datetime)::date AS latest, round(avg(gsd), 2) AS avg_gsd_m FROM maxar WHERE assets.visual IS NOT NULL GROUP BY event ORDER BY scenes DESC """).show()

## 3. Filter[¶](#3-filter)

Rasteret's `subset()` and `where()` are convenience methods for common filters. You can also filter the Arrow table directly with DuckDB, PyArrow, or pandas - whichever fits your workflow.

In \[ \]:

Copied!

```
from datetime import datetime, timezone

import pyarrow.dataset as ds

# --- Option A: Rasteret convenience filter ---
aug_scenes = collection.where(
    (ds.field("datetime") >= datetime(2023, 8, 9, tzinfo=timezone.utc))
    & (ds.field("datetime") < datetime(2023, 8, 13, tzinfo=timezone.utc))
)
print(f"Rasteret where(): {aug_scenes.dataset.count_rows()} scenes")

# --- Option B: DuckDB on the Arrow table ---
result = con.sql("""
    SELECT count(*) AS scenes
    FROM maxar
    WHERE datetime >= '2023-08-09' AND datetime < '2023-08-13'
""").fetchone()
print(f"DuckDB filter:    {result[0]} scenes")

# --- Option C: PyArrow compute ---
import pyarrow.compute as pc

mask = pc.and_(
    pc.greater_equal(
        maxar.column("datetime"),
        pc.assume_timezone(pc.strptime("2023-08-09", "%Y-%m-%d", "us"), timezone="UTC"),
    ),
    pc.less(
        maxar.column("datetime"),
        pc.assume_timezone(pc.strptime("2023-08-13", "%Y-%m-%d", "us"), timezone="UTC"),
    ),
)
print(f"PyArrow filter:   {pc.sum(mask).as_py()} scenes")

print("\nAll three query the same Arrow data. Use whichever fits your workflow.")
```

from datetime import datetime, timezone import pyarrow.dataset as ds

# --- Option A: Rasteret convenience filter ---

aug_scenes = collection.where( (ds.field("datetime") >= datetime(2023, 8, 9, tzinfo=timezone.utc)) & (ds.field("datetime") < datetime(2023, 8, 13, tzinfo=timezone.utc)) ) print(f"Rasteret where(): {aug_scenes.dataset.count_rows()} scenes")

# --- Option B: DuckDB on the Arrow table ---

result = con.sql(""" SELECT count(\*) AS scenes FROM maxar WHERE datetime >= '2023-08-09' AND datetime < '2023-08-13' """).fetchone() print(f"DuckDB filter: {result[0]} scenes")

# --- Option C: PyArrow compute ---

import pyarrow.compute as pc mask = pc.and\_( pc.greater_equal( maxar.column("datetime"), pc.assume_timezone(pc.strptime("2023-08-09", "%Y-%m-%d", "us"), timezone="UTC"), ), pc.less( maxar.column("datetime"), pc.assume_timezone(pc.strptime("2023-08-13", "%Y-%m-%d", "us"), timezone="UTC"), ), ) print(f"PyArrow filter: {pc.sum(mask).as_py()} scenes") print("\\nAll three query the same Arrow data. Use whichever fits your workflow.")

## 4. Export and share[¶](#4-export-and-share)

Export the Collection so a teammate can load it; no S3 access or Source Cooperative account needed on their end.

In \[ \]:

Copied!

```
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    export_path = Path(tmpdir) / "maxar_collection"
    collection.export(export_path)

    # Teammate loads it in one line:
    reloaded = rasteret.load(export_path)
    print(f"Loaded: {reloaded.name}, {reloaded.dataset.count_rows()} scenes")
```

import tempfile from pathlib import Path with tempfile.TemporaryDirectory() as tmpdir: export_path = Path(tmpdir) / "maxar_collection" collection.export(export_path)

# Teammate loads it in one line:

reloaded = rasteret.load(export_path) print(f"Loaded: {reloaded.name}, {reloaded.dataset.count_rows()} scenes")

## 5. Column mapping (non-standard schemas)[¶](#5-column-mapping-non-standard-schemas)

Not every Parquet file uses STAC column names. If your source uses different names, provide a `column_map`:

```
collection = rasteret.build_from_table(
    "s3://my-bucket/my-catalog.parquet",
    name="custom",
    column_map={"scene_id": "id", "timestamp": "datetime"},
)
```

Rasteret requires four columns: `id`, `datetime`, `geometry`, `assets`. Everything else is passed through as-is. See the [Schema Contract](https://terrafloww.github.io/rasteret/explanation/schema-contract/index.md) for details.

## Summary[¶](#summary)

| Step                                       | What happens                                                          |
| ------------------------------------------ | --------------------------------------------------------------------- |
| `build_from_table(s3_uri)`                 | Reads Parquet from S3/GCS/local, validates schema, creates Collection |
| `collection.dataset.to_table()`            | Arrow table - pass directly to DuckDB, PyArrow, pandas                |
| `collection.where()` / `subset()`          | Convenience filters (Arrow pushdown)                                  |
| `collection.export()` -> `rasteret.load()` | Share a portable Collection                                           |

**When to use which build function:**

| Situation                                               | Use                           |
| ------------------------------------------------------- | ----------------------------- |
| Dataset in the catalog (Sentinel-2, Landsat, NAIP, ...) | `rasteret.build()`            |
| Custom STAC API not in the catalog                      | `rasteret.build_from_stac()`  |
| Existing Parquet with GeoTIFF URLs (this notebook)      | `rasteret.build_from_table()` |
| Someone shared a Collection with you                    | `rasteret.load()`             |

Next: [Work With Collection Tables](https://terrafloww.github.io/rasteret/tutorials/03_work_with_collection_tables/index.md)
