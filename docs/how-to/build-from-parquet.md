# Build a Collection from Parquet Files

`build_from_table()` creates a Collection from **any Parquet file that
contains GeoTIFF URLs**: STAC GeoParquet, Source Cooperative exports,
or your own custom catalog. No STAC API needed.

PyArrow reads the file from local paths, `s3://`, or `gs://` URIs.
Rasteret validates the schema, derives per-record bounding boxes from the
GeoParquet `geometry` column, and produces a standard Collection backed by
Arrow.

---

## Supported sources

| Source | Example URI |
|--------|-------------|
| [Source Cooperative](https://source.coop) | `s3://us-west-2.opendata.source.coop/maxar/maxar-opendata/maxar-opendata.parquet` |
| STAC GeoParquet exports (Planetary Computer, Element84, ...) | `s3://sentinel-cogs/sentinel-s2-l2a-cogs/items.parquet` |
| Your own Parquet with GeoTIFF URLs | `s3://my-bucket/my-catalog.parquet` or `/local/path.parquet` |

Any Parquet file works as long as it has the four required columns:
`id`, `datetime`, `geometry`, `assets` (where `assets` contains
GeoTIFF/COG URLs). See the [Schema Contract](../explanation/schema-contract.md)
for details.

---

## Build from a remote Parquet

```python
import os

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"  # for public S3 buckets

import rasteret

# Source Cooperative - reads directly from S3 via PyArrow
collection = rasteret.build_from_table(
    "s3://us-west-2.opendata.source.coop/maxar/maxar-opendata/maxar-opendata.parquet",
    name="maxar-opendata",
)

print(f"Rows: {collection.dataset.count_rows()}")
```

When `name` is provided, the collection is cached to
`~/rasteret_workspace/{name}_records/` and discoverable via
`rasteret cache list`. Subsequent calls with the same name load
from the cache instantly. Pass `force=True` to rebuild.

See [`build_from_table()`](../reference/rasteret.md) API reference.

!!! note
    `build_from_table()` uses PyArrow's dataset API internally, which supports
    local paths, `s3://`, and `gs://` URIs. HTTPS URLs are **not** supported
    by PyArrow's scanner. Download HTTPS files locally first, or use an S3/GCS
    URI when available.

---

## Filter during scan

For large remote files, pass `filter_expr` and `columns` to push
filtering and projection down to the scan layer (only matching row
groups are transferred):

```python
import pyarrow.dataset as ds

collection = rasteret.build_from_table(
    "s3://my-bucket/stac-items.parquet",
    name="filtered",
    filter_expr=ds.field("eo:cloud_cover") < 20.0,
    columns=["id", "datetime", "geometry", "assets", "eo:cloud_cover"],
)
```

---

## Column mapping

If the source Parquet uses different column names, remap them:

```python
collection = rasteret.build_from_table(
    "path/to/records.parquet",
    name="custom",
    column_map={"scene_id": "id", "timestamp": "datetime"},
)
```

Rasteret requires four columns: `id`, `datetime`, `geometry`, `assets`.
Everything else is passed through as-is.

The `assets` column is a mapping from band key -> asset dict. Each asset
dict must contain a resolvable `href`. For multi-sample planar-separate
GeoTIFFs (multiple bands in one file), you can also include `band_index`
to select which sample/band the asset refers to.

---

## Enrich with COG headers

By default, `build_from_table()` imports the Parquet as-is. The resulting
Collection works for filtering and metadata queries, but cannot do fast
tiled reads (`get_xarray()`, `get_gdf()`, `to_torchgeo_dataset()`) because
it has no cached tile offsets.

Pass `enrich_cog=True` to parse COG headers from the asset URLs during
the build. This adds `{band}_metadata` struct columns to the Parquet
index (tile offsets, byte counts, image dimensions, etc.) that enable
Rasteret's accelerated reads:

```python
collection = rasteret.build_from_table(
    "s3://my-bucket/my-catalog.parquet",
    name="my-enriched-collection",
    enrich_cog=True,
    band_codes=["B04", "B08"],       # which bands to enrich (optional)
    max_concurrent=300,               # concurrent header fetches
)
```

`band_codes` specifies which asset keys to parse. When omitted, Rasteret
enriches every asset found in the `assets` column. For large datasets,
specifying only the bands you need saves time and storage.

!!! tip "When do I need enrichment?"

    | Use case | `enrich_cog` needed? |
    |----------|---------------------|
    | Filtering by time, location, cloud cover | No |
    | Exporting / sharing the Collection | No |
    | `get_xarray()`, `get_gdf()` - reading pixels | **Yes** |
    | `to_torchgeo_dataset()` - ML training | **Yes** |

    If you built from a STAC API via `build()` or `build_from_stac()`,
    enrichment already happened automatically. You only need
    `enrich_cog=True` when using `build_from_table()`.

Once enriched, use the Collection like any other. `geometries` accepts
Arrow arrays, bbox tuples, Shapely objects, or raw WKB - Arrow columns
from GeoParquet are the fastest path (no Python-object conversion):

```python
import pyarrow.parquet as pq

# Arrow geometry column - passed directly, no conversion
parcels = pq.read_table("field_boundaries.geoparquet", columns=["geometry"])
ds = collection.get_xarray(
    geometries=parcels.column("geometry"),
    bands=["B04", "B08"],
)

# Bbox tuple also works for a single area of interest
ds = collection.get_xarray(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For more on what the enrichment columns contain, see
[Schema Contract - Tier 2: COG acceleration columns](../explanation/schema-contract.md#tier-2-cog-acceleration-columns).

---

## CLI

```bash
rasteret cache import maxar-opendata \
  --record-table "s3://us-west-2.opendata.source.coop/maxar/maxar-opendata/maxar-opendata.parquet"
```

The full runnable script is at
`examples/build_collection_from_parquet.py`.
