# Design Decisions

This page documents the key design choices behind Rasteret and the reasoning
that drives them. It is aimed at contributors and advanced users who want to
understand *why* things work the way they do.

## Category framing

Rasteret follows an **index-first geospatial retrieval** architecture:

- **Control plane (tables/Parquet)**: discovery, filtering, splits/labels, and cached COG header metadata.
- **Data plane (COG object storage)**: on-demand tile byte reads from source GeoTIFFs.

This separation is intentional. It preserves table interoperability for metadata
workflows while avoiding payload-in-Parquet duplication for routine pixel reads.

## Why Parquet indexes?

Remote COGs require an HTTP HEAD + IFD range read per file just to discover
tile offsets. For a 30-scene time series with 4 bands, that is 120 HTTP
round-trips *before any pixel data is fetched*.

Rasteret pre-parses COG headers at index time and stores everything in a local
Parquet dataset. Subsequent reads skip all header fetching and jump straight to
pixel byte ranges.

Why Parquet specifically (not Zarr manifests, JSON, or SQLite)?

- **Queryable**: Arrow predicate/projection pushdown enables subsetting
  without reading the full index.
- **Schema evolution**: new columns can be added without breaking existing
  files.
- **Ecosystem native**: Parquet is readable by DuckDB, DataFusion, Polars,
  pandas, Spark, and every major data tool.
- **Portable**: a single file (or partitioned directory) that travels with
  the data; no running service needed.

## Why a custom COG reader?

No existing TIFF reader exposes a Python API to inject pre-cached tile offsets
and byte counts. Rasteret needs to skip the IFD fetch entirely because tile
layout is already cached in Parquet, so it needs its own read path.

The reader is minimal: HTTP range requests, TIFF tag parsing, and tile
decompression in a thread pool. It is intentionally **not** a general-purpose
TIFF library.

## Why COGs stay COGs

Rasteret does not convert COGs to Zarr, HDF5, or any other format. The data
stays where it is, in the format it was published in. Concretely:

- GIS tools (QGIS, GDAL, rasterio) still work on the same files.
- No data duplication or format lock-in.
- Index can be rebuilt at any time from the original source.

The Parquet index is metadata only. It tells Rasteret *where* to read, not
*what* to read.

## No GDAL/rasterio in the hot path

Rasterio is a dependency and is used for reprojection and geometry masking.
CRS and coordinate transforms use pyproj directly. CRS encoding in xarray
output uses pyproj's CF conventions (WKT2, PROJJSON, GeoTransform), not
rioxarray. The tile-read hot path uses Rasteret's own async HTTP +
decompression pipeline.

## Native dtype preservation

Rasteret preserves the native COG dtype through the entire read pipeline.
A uint16 Sentinel-2 tile comes back as uint16; an int8 embedding comes
back as int8. TorchGeo's `RasterDataset` converts images to `float32` by
default; Rasteret keeps native dtypes unless you opt into casting/resampling.

When does promotion happen?

- **NaN fill is explicit**: Rasteret does not auto-promote integer arrays
  just to use NaN. If a caller requests NaN fill, the output dtype must be
  floating (or the call will fail when casting the fill value).
- **Cross-CRS reprojection**: rasterio's `reproject()` operates on the
  chosen output dtype; if a workflow uses NaN fill, float output is
  generally required.

In both cases, the promotion is explicit at the call site (not an implicit
side effect of masking).

## obstore as the HTTP backend

Rasteret uses [obstore](https://github.com/developmentseed/obstore) for all
remote byte-range reads. obstore wraps the
[`object_store`](https://docs.rs/object_store/) Rust crate. It is Rust-native,
multi-cloud, and does not depend on GDAL.

Why obstore as a hard dependency rather than optional?

- **Multi-cloud + auth**: Rasteret previously used a Python HTTP client
  (aiohttp) for range reads. obstore gives a single, well-tested interface for
  S3/GCS/Azure/HTTPS plus credential providers (requester-pays, SAS signing,
  Earthdata, etc.).
- **Single code path**: one backend means fewer branches to test and maintain.
- **Multi-cloud**: Rasteret auto-routes URLs to native cloud stores -- S3Store
  for `s3://` and `*.s3.*.amazonaws.com`, GCSStore for `gs://` and
  `storage.googleapis.com`, AzureStore for `*.blob.core.windows.net`, and
  HTTPStore for everything else. Authenticated reads use obstore credential
  providers via `create_backend()`.
- **Well-maintained lineage**: obstore is from Development Seed and it wraps
  the same Rust crate [`object_store`](https://docs.rs/object_store/) that Databricks, InfluxDB, and many in Arrow
  ecosystem depend on.

## Decoupled index and read layers

The Parquet index is the stable contract. The read layer ([COGReader](../reference/fetch/cog.md), obstore)
is swappable via the [`StorageBackend`](../reference/cloud.md) protocol. Concretely:

- Custom backends (e.g. a pre-configured `S3Store`) can be plugged in without
  changing the index format.
- The index can be consumed by tools other than Rasteret (DuckDB, DataFusion).
- Contributors can work on ingest drivers and readers independently.

## Schema flexibility

The Collection Parquet has three tiers of columns:

1. **Required** (4 columns): `id`, `datetime`, `geometry`, `assets`.
   Every ingest path must produce these.
2. **COG acceleration** (per-band): `{band}_metadata` struct columns with
   tile offsets, byte counts, transform, etc. These enable fast tiled
   reads and are added by COG enrichment phase methods.
3. **User-extensible**: `split`, `label`, `cloud_cover`, custom metadata.
   Anything the user's workflow needs.

This layered schema means the same Parquet file serves as both an I/O
acceleration cache and an experiment metadata store.

See [Schema Contract](schema-contract.md) for the full column specification.
See also the [`Collection`](../reference/core/collection.md) API reference.

## The index is shareable

The Parquet index is small (megabytes) while the data it describes is
large (terabytes). This makes the Collection practical to share:

- **Team workflows**: one person indexes 10,000 scenes, team members
  load the same index file and get identical accelerated reads.
- **Academic Paper reproduction**: share a few MB Parquet file instead of
  terabytes of imagery. A collaborator loads it, reads pixels from the
  same remote COGs, and gets the same results.
- **CI/CD**: commit the index to version control. The training pipeline
  loads it directly, no STAC queries, no re-indexing.
- **Experiment management**: different Parquet files for different
  experiments (different AOIs, different split assignments, different
  label columns) over the same underlying data.

**Important caveat**: The index is a snapshot of metadata at build time.
If the underlying TIFF files change (reprocessed, moved to a different
URL, deleted), the index becomes stale and needs to be rebuilt. Signed
URLs also expire. Stable public URLs (S3, GCS, Azure public buckets)
work best for long-lived indexes.

## Tiled GeoTIFFs only

Rasteret's tile execution path expects **tiled** GeoTIFFs (COGs recommended
for remote access). Striped/untiled TIFFs and non-TIFF formats are not
supported in the read path. Use TorchGeo or rasterio directly for those.

This is an intentional scope constraint, not a limitation to fix. Tiled
GeoTIFFs are the format where Rasteret's index-first approach provides the
most value.
