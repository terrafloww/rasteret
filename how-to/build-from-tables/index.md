# Build From Parquet And Arrow Tables

Use `build_from_table()` when you already have a table of raster records and want Rasteret to turn it into a reusable collection.

This is the right path for:

- STAC GeoParquet exports
- a Parquet/Arrow record table you created yourself
- a DuckDB relation or PyArrow table in memory
- private drone, satellite, or derived COG catalogs

The table describes the rasters. The pixels stay in the original COGs.

## The Record Table Contract

At minimum, Rasteret needs these record fields:

| Column     | Meaning                                                                |
| ---------- | ---------------------------------------------------------------------- |
| `id`       | Stable record id.                                                      |
| `datetime` | Acquisition or record time. Integer years are accepted and normalized. |
| `geometry` | Footprint geometry, usually WKB/GeoArrow or GeoPandas geometry.        |
| `assets`   | Mapping from band key to asset info, including `href`.                 |

Rasteret can add `bbox`, `year`, and `month` during normalization. Pass `enrich_cog=True` so that Rasteret also parses COG headers and adds `{band}_metadata` columns used by pixel reads.

## Build From An In-Memory Table

```python
from datetime import datetime

import geopandas as gpd
import rasteret
from shapely.geometry import box

records = [
    {
        "id": "scene-001",
        "datetime": datetime(2024, 1, 15),
        "geometry": box(77.50, 12.90, 77.70, 13.10),
        "assets": {
            "B04": {"href": "s3://my-bucket/scenes/scene-001_B04.tif"},
            "B08": {"href": "s3://my-bucket/scenes/scene-001_B08.tif"},
        },
    }
]

gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="OGC:CRS84")

collection = rasteret.build_from_table(
    gdf.to_arrow(geometry_encoding="WKB"),
    name="my-scenes",
    enrich_cog=True,
)
```

`build_from_table()` accepts direct file/cloud-storage paths to Parquet/Arrow tables and Arrow-compatible in-memory objects.

Rasteret treats the collection footprint `geometry` as CRS84. If the input geometry is a GeoArrow field with CRS metadata, Rasteret can reproject the footprint before deriving `bbox`. The raster CRS still belongs in row-level sidecar columns such as `crs` and `proj:epsg`.

## Build From DuckDB

DuckDB is useful when your record table needs SQL filtering or joins before Rasteret builds the collection.

```python
import duckdb
import rasteret

con = duckdb.connect()

records = con.sql("""
    SELECT *
    FROM 'records.parquet'
    WHERE "eo:cloud_cover" < 20
""")

collection = rasteret.build_from_table(
    records,
    name="filtered-scenes",
    enrich_cog=True,
)
```

No intermediate file is required. Rasteret consumes the DuckDB relation through Arrow.

## Build From A Parquet Path

```python
import rasteret

collection = rasteret.build_from_table(
    "s3://my-bucket/my-cog-records.parquet",
    name="my-cog-records",
    enrich_cog=True,
    band_codes=["B04", "B08"],
    max_concurrent=300,
)
```

When `name` is provided, Rasteret caches the collection under `~/rasteret_workspace/{name}_records/`.

Note

PyArrow dataset scanning supports local paths and many cloud URIs such as `s3://` and `gs://`. For plain HTTPS Parquet files, download the file first or use the cloud URI when available.

## Column Mapping

If your table uses different names, map source columns to Rasteret's canonical fields:

```python
collection = rasteret.build_from_table(
    "records.parquet",
    name="custom",
    column_map={
        "scene_id": "id",
        "geom": "geometry",
        "year": "datetime",
    },
    href_column="url",
    band_index_map={"B04": 0, "B08": 1},
    enrich_cog=True,
)
```

In this example, `records.parquet` has a `url` column. Each row's URL points to one multi-band GeoTIFF/COG, and `band_index_map` tells Rasteret which 0-based sample index inside that file should be treated as each logical band:

| Logical band | Source file      | Sample index |
| ------------ | ---------------- | ------------ |
| `B04`        | value from `url` | `0`          |
| `B08`        | value from `url` | `1`          |

Rasteret uses that to build an `assets` value like:

```python
{
    "B04": {"href": "<row url>", "band_index": 0},
    "B08": {"href": "<row url>", "band_index": 1},
}
```

If your table already has separate URLs for separate single-band COGs, build an `assets` column instead. For example, `B04` and `B08` should point to different `href` values when they live in different files. The `href` values can point to cloud files or local tiled GeoTIFF/COG files.

## When Enrichment Is Needed

By default, `build_from_table()` normalizes the table. It does not parse every COG header unless you ask it to.

| Use case                                                 | Need `enrich_cog=True`? |
| -------------------------------------------------------- | ----------------------- |
| Filter metadata only                                     | No                      |
| Add labels/splits and export a metadata table            | No                      |
| `get_numpy()` / `get_xarray()` / `get_gdf()` pixel reads | Yes                     |
| `sample_points()` pixel reads                            | Yes                     |
| `to_torchgeo_dataset()`                                  | Yes                     |

If you built from a STAC API with `build()` or `build_from_stac()`, COG header enrichment already happens during build.

## Read From The Collection

Once the table is enriched, it behaves like any Rasteret collection:

```python
ds = collection.get_xarray(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

You can also pass AOI tables directly to pixel reads. Use this when the table describes user geometries such as parcels, plots, or labels rather than raster records:

```python
import geopandas as gpd
from shapely.geometry import box

parcels = gpd.GeoDataFrame(
    {
        "parcel_id": ["parcel-001"],
        "crop": ["rice"],
    },
    geometry=[box(77.55, 13.01, 77.58, 13.08)],
    crs="OGC:CRS84",
).to_arrow(geometry_encoding="WKB")

gdf = collection.get_gdf(
    geometries=parcels,
    geometry_column="geometry",
    bands=["B04", "B08"],
)
```

`get_gdf(...)` preserves non-geometry AOI columns such as `parcel_id` and `crop`. For array-only outputs such as `get_numpy(...)` and `get_xarray(...)`, the same table geometry input works, but the returned object does not carry AOI metadata.

## Troubleshooting

If a requested band cannot be resolved, check:

- the keys inside `assets`
- the available `{band}_metadata` columns
- whether you need `data_source=...` for a registered band mapping
- whether the table was built with `enrich_cog=True`

If COG header parsing fails for all assets, check whether the URLs require credentials, requester-pays configuration, URL rewriting, or a custom backend. For local files, a normal path like `/data/scene_B04.tif` is enough.

For the exact runtime schema, see [Schema Contract](https://terrafloww.github.io/rasteret/explanation/schema-contract/index.md).
