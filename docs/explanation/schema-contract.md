# Schema Contract

This page describes the table shape Rasteret expects and produces. It is useful
when you are building an ingest path, preparing a Parquet/Arrow record table, or
debugging why a collection can filter but cannot read pixels yet.

The short version:

```text
required record columns + optional COG metadata + your own workflow columns
```

Pixels stay in the original GeoTIFF/COG files. The collection stores the record
metadata and the COG header metadata Rasteret needs to read byte ranges later.

## Required Record Columns

Every Rasteret collection starts from four record fields:

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | `string` | Stable record identifier. |
| `datetime` | `timestamp` | Acquisition or record time. Integer years can be normalized by `build_from_table()`. |
| `geometry` | WKB / GeoArrow-compatible geometry | Footprint geometry for the raster record. |
| `assets` | struct-like mapping | Band key to asset metadata, including a resolvable `href`. |

The normalisation layer validates these fields and raises `ValueError` when any
are missing.

An `assets` value usually looks like:

```python
{
    "B04": {"href": "s3://bucket/scene_B04.tif"},
    "B08": {"href": "s3://bucket/scene_B08.tif"},
}
```

For multi-band GeoTIFFs where several logical bands live in the same file, each
band can point to the same `href` with a different 0-based `band_index`:

```python
{
    "R": {"href": "s3://bucket/naip.tif", "band_index": 0},
    "G": {"href": "s3://bucket/naip.tif", "band_index": 1},
    "B": {"href": "s3://bucket/naip.tif", "band_index": 2},
    "NIR": {"href": "s3://bucket/naip.tif", "band_index": 3},
}
```

## Columns Rasteret Can Add

During normalization, Rasteret adds these when missing:

| Column | Meaning |
| --- | --- |
| `bbox` | Struct with `xmin`, `ymin`, `xmax`, `ymax`, derived from `geometry`. |
| `year` | Partition column derived from `datetime`. |
| `month` | Partition column derived from `datetime`. |

During COG enrichment, Rasteret can add CRS sidecars:

| Column | Meaning |
| --- | --- |
| `proj:epsg` | Integer EPSG code for the native raster CRS. |
| `crs` | String CRS code such as `EPSG:32632`. |

These CRS sidecars describe the raster asset CRS for each row. They do not
describe the footprint `geometry` column, which Rasteret treats as CRS84
footprint geometry for Arrow/GeoArrow export.

## COG Metadata Columns

Pixel reads need per-band COG header metadata. These columns are added when you
build from STAC/catalog sources, or when you call:

```python
rasteret.build_from_table(..., enrich_cog=True)
```

Column names follow this pattern:

```text
{band}_metadata
```

Examples:

```text
B04_metadata
B08_metadata
red_metadata
```

Each metadata struct stores the header data Rasteret needs for tiled reads:

| Field | Meaning |
| --- | --- |
| `image_width`, `image_height` | Full raster dimensions. |
| `tile_width`, `tile_height` | Tile dimensions. |
| `dtype` | Source NumPy dtype string. |
| `transform` | Affine transform parameters. |
| `tile_offsets`, `tile_byte_counts` | Byte ranges for each tile. |
| `compression`, `predictor`, `photometric` | TIFF decode metadata. |
| `pixel_scale`, `tiepoint` | GeoTIFF georeferencing tags when present. |
| `nodata` | GDAL nodata value when present. |
| `samples_per_pixel`, `planar_configuration`, `extra_samples` | Multi-sample TIFF layout metadata. |

A null `{band}_metadata` value means that record was not enriched for that band,
the band was missing for that row, or header parsing failed for that asset.
Rasteret skips records with null metadata for requested pixel reads.

## User Columns

You can add columns beside Rasteret's columns. Common examples:

| Column | Purpose |
| --- | --- |
| `split` | Train/validation/test assignment. |
| `label` | Classification or regression target. |
| `plot_id`, `aoi_id`, `fold` | Experiment grouping keys. |
| `eo:cloud_cover` | Scene-level cloud percentage from STAC or your own metadata. |
| `quality_flag` | Custom filtering or audit value. |

For examples, see
[Bring Your Own AOIs, Points, And Metadata](../how-to/enriched-collection-workflows.md).

## Arrow And GeoArrow Interop

Rasteret collections can be passed to Arrow-aware tools. On Arrow export,
Rasteret marks the `geometry` field as `geoarrow.wkb` so GeoPandas and other
GeoArrow-aware consumers can detect the footprint geometry.

Important CRS distinction:

- `geometry` is the raster footprint and is exported with CRS84 metadata.
- `crs` and `proj:epsg` are row-level raster CRS sidecars used for pixel reads.

That means a collection can have footprints in CRS84 while the rasters
themselves are in UTM or another projected CRS.

AOI and point tables are separate from the collection table. They can carry
their own geometry column, CRS, and business columns such as `plot_id`,
`sensor_id`, `split`, or `label`. Rasteret keeps those business columns in
`get_gdf(...)` and `sample_points(...)` outputs unless a column name collides
with a Rasteret output field.

## Entry Points

Use the entry point that matches your table state:

| Situation | Use |
| --- | --- |
| External record table that needs normalization or enrichment | `build_from_table(...)` |
| Read-ready Rasteret Arrow table already in memory | `as_collection(...)` |
| Previously exported collection artifact | `load(...)` |

`as_collection(...)` expects the table to already have Rasteret's read-ready
columns, including `{band}_metadata` by default. Use `build_from_table(...)` for
first-time external record tables.

## Layer Requirements

For filtering and metadata work:

- required record columns
- `bbox` for spatial filtering

For `get_numpy()`, `get_xarray()`, `get_gdf()`, and `sample_points()`:

- required record columns
- `bbox`
- `proj:epsg` or enough COG header CRS metadata to backfill it during enrichment
- `{band}_metadata` for each requested band

For `to_torchgeo_dataset(...)`:

- required record columns
- `proj:epsg`
- `{band}_metadata` for each requested band
- usable `datetime` or `start_datetime` / `end_datetime` values for temporal indexing

## Data Source Resolution

When Rasteret needs a data source for band mapping or cloud configuration, it
checks:

1. An explicit `data_source=...` argument.
2. `Collection.data_source`.
3. Parquet schema metadata written by `Collection.export()`.
4. The first non-empty value from a `collection` column.
5. Otherwise, no data source is assumed.

When a table engine drops schema metadata or changes string types during an
Arrow round trip, pass `data_source=collection.data_source` explicitly when
calling `as_collection(...)`.
