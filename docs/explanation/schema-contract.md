# Schema Contract

## Why three tiers?

A Rasteret Parquet index serves two purposes at once: **I/O acceleration
cache** and **experiment metadata store**. Most systems keep these separate
(a manifest for discovery, a cache for byte offsets, a spreadsheet for
labels). Rasteret unifies them in one file.

This works because the schema has three tiers of columns, each with
different guarantees:

1. **Required columns** (4): the contract every ingest path must satisfy.
   These are what make a Parquet file a valid Rasteret Collection.
2. **COG acceleration columns** (per band): cached TIFF header data
   that eliminates HTTP round-trips at read time. This is what makes
   Rasteret fast.
3. **User-extensible columns**: splits, labels, quality flags, custom
   metadata. This is what makes the same Parquet file useful for
   experiment tracking and reproducibility.

The three tiers are independent. A Collection without COG acceleration
columns still works for filtering and metadata queries (just no fast
tile reads). A Collection without user columns still works for reading
pixels (just no split/label support). You can build a
Collection in stages: ingest first, enrich with COG metadata later, add
splits and labels when you need them.

### Why Parquet?

Parquet is queryable, schema-evolvable, ecosystem-native, and portable.
For the full rationale (including alternatives considered), see
[Design Decisions](design-decisions.md#why-parquet-indexes).

### GeoParquet (and Parquet native geometry types)

Rasteret Collections are written as **GeoParquet**: the footprint geometry is
stored as **WKB** (in **CRS84**) and `Collection.export()` writes the GeoParquet
1.1 `geo` file metadata so other tools can interpret the geometry column.

Parquet is also adding native `GEOMETRY`/`GEOGRAPHY` logical types. The GeoParquet
community is planning GeoParquet 2.0 to align with that evolution as tool support
matures. Rasteret tracks this and plans to adopt newer encodings when ecosystem
support stabilizes.

GeoParquet is also incubating an **alpha "Parquet Raster"** proposal
for representing raster pixels (and/or external raster references) *inside* Parquet.
See the [draft spec](https://github.com/opengeospatial/geoparquet/blob/main/format-specs/parquet-raster.md).
Rasteret's Collections are different: they are **record tables** (GeoParquet) that
reference existing GeoTIFF/COG assets and add acceleration metadata for fast
byte-range tile reads, rather than storing raster payloads in Parquet.

### Dataset descriptors

Rasteret uses `DatasetDescriptor` objects to describe how a dataset is discovered
(STAC API vs GeoParquet), accessed (cloud auth / URL rewriting), and mapped
(band codes -> asset keys + optional `band_index` for multi-sample GeoTIFFs).

See:

- [Architecture](architecture.md) for the end-to-end flow
- [`rasteret.types`](../reference/types.md) for the concrete descriptor fields

---

## Tier 1: Required columns

These columns are required for `Collection` to function:

| Column | Type | Description |
|--------|------|-------------|
| `id` | `string` | Unique record identifier |
| `datetime` | `timestamp` | Acquisition / observation time (may be null when `start_datetime`/`end_datetime` present) |
| `geometry` | `binary` (WKB) | Record footprint geometry |
| `assets` | `struct` | Band key -> asset dict with resolvable `href` |

The normalisation layer (`build_collection_from_table`) validates these
exist and raises `ValueError` if any are missing.

The asset dict supports:

- `href` (required): URL/path to a tiled GeoTIFF/COG.
- `band_index` (optional, default `0`): 0-based sample/band index within a
  multi-sample GeoTIFF asset. This is required when multiple logical bands
  live in one file (e.g. NAIP `image` contains R/G/B/NIR), and is also used
  to slice planar-separate tile tables at enrichment time.

### Derived columns (added automatically)

The normalisation layer adds these when missing:

| Column | Type | Description |
|--------|------|-------------|
| `scene_bbox` | `list<float64>[4]` | `[minx, miny, maxx, maxy]` from geometry |
| `bbox_minx`, `bbox_miny`, `bbox_maxx`, `bbox_maxy` | `float64` | Scalar bbox for Arrow predicate pushdown |
| `year` | `int64` | Partition column from datetime |
| `month` | `int64` | Partition column from datetime |

---

## Tier 2: COG acceleration columns

Per-band metadata struct columns enable fast tiled reads by caching IFD
data from COG headers. These are added by COG enrichment
(`enrich_table_with_cog_metadata`) or by `StacCollectionBuilder`.

Column name pattern: `{band}_metadata` (e.g. `B04_metadata`, `red_metadata`).

Each struct contains:

| Field | Type | Description |
|-------|------|-------------|
| `image_width` | `int32` | Full image width in pixels |
| `image_height` | `int32` | Full image height in pixels |
| `tile_width` | `int32` | Tile width in pixels |
| `tile_height` | `int32` | Tile height in pixels |
| `dtype` | `string` | NumPy dtype string (e.g. `"uint16"`) |
| `transform` | `list<float64>` | Affine transform parameters |
| `predictor` | `int32` | TIFF predictor tag |
| `compression` | `int32` | TIFF compression tag |
| `tile_offsets` | `list<int64>` | Byte offsets of each tile in the file |
| `tile_byte_counts` | `list<int64>` | Byte sizes of each tile |
| `pixel_scale` | `list<float64>` | GeoTIFF ModelPixelScaleTag |
| `tiepoint` | `list<float64>` | GeoTIFF ModelTiepointTag |
| `nodata` | `float64` | GDAL nodata value (null when tag absent) |
| `samples_per_pixel` | `int32` | Bands per IFD (TIFF SamplesPerPixel) |
| `planar_configuration` | `int32` | 1 = chunky, 2 = planar separate |
| `photometric` | `int32` | TIFF PhotometricInterpretation |
| `extra_samples` | `list<int32>` | Extra sample types (alpha, unspecified) |

### Null values in COG metadata

A `null` value in a `{band}_metadata` column means that record was not
enriched for that band. This happens when:

- The record was ingested without COG enrichment (e.g. [`build_from_table()`](../reference/rasteret.md)
  without `enrich_cog=True`)
- The COG header fetch failed for that specific record during enrichment
- The band does not exist for that record

The execution layer skips records with null metadata for the requested
bands. Partial enrichment is valid; some records can have metadata while
others don't.

---

## Tier 3: User-extensible columns

Any column can be added to a Collection's Parquet file. Common examples:

| Column | Type | Purpose |
|--------|------|---------|
| `split` | `string` | `"train"` / `"val"` / `"test"` assignment |
| `label` (or custom name) | varies | Classification target, regression value |
| `cloud_cover` / `eo:cloud_cover` | `float64` | Scene-level cloud percentage |
| `quality_flag` | `int32` | Custom quality score |
| `collection` | `string` | Parent collection identifier |
| `proj:epsg` | `int32` | EPSG code for native CRS |

### Adding custom columns

Load the Collection's table, add columns with PyArrow, rebuild with
`build_collection_from_table()`, and save. For a complete walkthrough
(splits, labels, TorchGeo integration), see
[ML Training with Splits](../how-to/ml-training-splits.md).

---

## Layer requirements

### TorchGeo adapter

`Collection.to_torchgeo_dataset(...)` requires:

- `id`, `datetime`, `assets`, `proj:epsg`
- `{band}_metadata` for each requested band
- `collection` is optional (data source resolved via `Collection.data_source`)
- Requires either a usable `datetime` or `start_datetime`/`end_datetime` for temporal indexing; rows without a resolved timestamp are dropped

### Execution layer

`Collection.get_numpy(...)`, `Collection.get_xarray(...)`, and `Collection.get_gdf(...)` iterate
rasters via `Collection.iterate_rasters()`, which needs:

- All four required columns
- `scene_bbox` (for spatial filtering)
- `{band}_metadata` for each requested band (for tile window calculation)

### Data source resolution

When resolving which band mapping to use, the execution layer checks:

1. Explicit `data_source=...` parameter
2. `Collection.data_source` attribute
3. Parquet schema metadata `data_source` (written by `Collection.export`)
4. First non-empty value from the `collection` column
5. Otherwise: no data source is assumed (identity band mapping, no URL signing)

---

## Validation expectations for new ingest paths

When introducing a new ingestion source:

1. Emit the required core columns exactly as above.
2. Use `build_collection_from_table()` for normalisation.
3. Call `enrich_table_with_cog_metadata()` if the source has COG assets.
4. Add a smoke test that creates a Collection and runs [`get_numpy()`](../reference/core/collection.md) or [`get_xarray()`](../reference/core/collection.md)
   on a small geometry.
