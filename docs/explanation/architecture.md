# Architecture

This page explains how Rasteret's components work together.

## The Index-First Architecture

Rasteret separates raster metadata work from pixel reads.

- **Control plane**: an Arrow/Parquet collection for discovery, filtering, and
  experiment metadata such as train/validation/test splits or more.
- **Data plane**: on-demand pixel reads from source GeoTIFF/COG assets.

For the technical reasoning behind this split and the remote-header cold-start
cost it avoids, see [Design Decisions](design-decisions.md).

### Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Rasteret control plane (Collection lifecycle + index schema)                 │
│ build* / load / as_collection / export / subset / where                      │
│ outputs: Parquet-backed collection rows (scene metadata + COG header cache)  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ Parquet rows + geometry + user columns
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Arrow ecosystem (optional compute + enrichment)                              │
│ DuckDB / Polars / GeoPandas / pandas / PyArrow                               │
│ operations: add split/label/patch/AOI columns, filter, joins, aggregations   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ filtered rows + geometry column
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Rasteret IO engine (custom byte range fetches)                               │
│ get_numpy() / get_xarray() / sample_points() / to_torchgeo_dataset()         │
│ consumes filtered rows + geometry + cached tile metadata                     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ async byte-range tile requests
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ Data plane: object storage with GeoTIFF/COG files                            │
│ S3 / GCS / Azure / *.tif                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Data flow

```text
DatasetRegistry  (stores DatasetDescriptor entries)
    |
    |-- auto-populates --> BandRegistry  (band code -> asset key lookup)
    |-- auto-populates --> CloudConfig   (auth, requester-pays, URL rewriting)
    |
    v
build() / build_from_stac() / build_from_table()
    |
    v
Ingest Drivers (stac_indexer, parquet_record_table)
    |
    v
Normalize Layer (build_collection_from_table)
    |
    v
Collection  (Arrow dataset wrapper)
    |
    +--> to_torchgeo_dataset()  --> RasteretGeoDataset  --> DataLoader
    |
    +--> get_xarray()           --> xr.Dataset
    |
    +--> get_numpy()            --> np.ndarray
    |
    +--> get_gdf()              --> gpd.GeoDataFrame
    |
    +--> sample_points(points=...) --> pyarrow.Table
    |
    +--> iterate_rasters()      --> async RasterAccessor stream
    |
    v
custom IO engine (async byte-range reads, tile decode, geometry mask) — obstore as HTTP transport (auto-routes to S3Store / AzureStore / GCSStore / HTTPStore)
```

`DatasetRegistry` is the in-code dataset catalog. It powers `build()` and the
`rasteret datasets ...` CLI commands. Each `DatasetDescriptor` is a single place
to declare:

- how to find the data (STAC API / static catalog / GeoParquet URI)
- how to resolve bands (band code -> asset key, optional `band_index_map`)
- how to access it (auth/requester-pays, URL signing/rewrites)

At runtime, Rasteret stores band-resolution and cloud-access settings in two
global lookups: `BandRegistry` and `CloudConfig`. These are keyed by the
Collection's `data_source` string.

For built-in catalog datasets, `data_source` is the dataset ID (e.g.
`earthsearch/sentinel-2-l2a`, `pc/sentinel-2-l2a`). This avoids collisions when
multiple providers use the same STAC collection id but different asset keys or
auth rules.

For BYO STAC/Parquet workflows, `data_source` defaults to the STAC collection id
when available. Override `data_source=` when you need to select a specific band
mapping or cloud/auth behavior (for example, when two providers share a
collection id).

Most users only touch `DatasetDescriptor`/`build()`. The registries are internal
implementation details used to keep read-time code simple and fast.

## Key components

### Collection

[`Collection`][rasteret.core.collection.Collection] is the central user-facing
object. It wraps a `pyarrow.dataset.Dataset` and provides:

- **Filtering**: [`subset()`](../reference/core/collection.md), [`where()`](../reference/core/collection.md), `select_split()`
- **Output adapters**: [`to_torchgeo_dataset()`](../reference/integrations/torchgeo.md), [`get_numpy()`](../reference/core/collection.md), [`get_xarray()`](../reference/core/collection.md), [`get_gdf()`](../reference/core/collection.md), [`sample_points()`](../reference/core/collection.md)
- **Export**: [`export()`](../reference/core/collection.md)
- **Discovery**: [`list_collections()`](../reference/core/collection.md)

Collections are immutable; filtering returns a new `Collection` with a
filtered view of the same underlying dataset.

### RasterAccessor

[`RasterAccessor`][rasteret.core.raster_accessor.RasterAccessor] is the data-loading
handle for a single Parquet row (record) in a Collection. Each accessor
loads bands concurrently using `asyncio.gather`, which is the main source of
Rasteret's speedup over sequential approaches.

### COGReader

[`COGReader`][rasteret.fetch.cog.COGReader] manages connection pooling and
tile-level byte-range reads. It:

1. Merges nearby byte ranges to minimize HTTP round-trips.
2. Issues async byte-range requests via obstore (HTTP transport layer), with
   automatic URL routing to native cloud stores (S3Store, AzureStore, GCSStore,
   or HTTPStore).
3. Decompresses tiles (deflate, LZW, zstd, LERC) in a thread pool.
   (TIFF JPEG is currently rejected with a hard error until implemented.)

For authenticated sources, pass a `backend` created with
[`create_backend()`](../reference/rasteret.md) and an obstore credential provider.

### Native dtype preservation

Tiles are decompressed and returned in their native COG dtype (uint16, int8,
float32, etc.).

By default, AOI reads fill outside-AOI / outside-coverage pixels with the
COG `nodata` value when present, otherwise `0`. Rasteret also computes a
`valid_mask` during COG reads; point sampling uses it to skip filled pixels.
TorchGeo samples remain TorchGeo-standard and do not include `valid_mask`
(see [Ecosystem Comparison](interop.md#torchgeo)).

### Ingest drivers

Each driver knows how to read one source type:

- [`StacCollectionBuilder`][rasteret.ingest.stac_indexer.StacCollectionBuilder]: STAC API
  search, COG header parsing, GeoParquet output.
- [`RecordTableBuilder`][rasteret.ingest.parquet_record_table.RecordTableBuilder]:
  reads an existing Parquet table with column mapping and optional COG header
  enrichment.

Both converge on [`build_collection_from_table()`][rasteret.ingest.normalize.build_collection_from_table],
which validates the collection contract, derives `bbox`, and adds partition
columns.

## Why Parquet indexes?

Traditional geospatial workflows often pay a cold-start cost every time they
open remote COGs. GDAL/rasterio needs TIFF header metadata before it can know
where tile bytes live.

Rasteret moves that work into collection build/enrichment by caching header
metadata in the Parquet collection.

For a full discussion of why Parquet over Zarr manifests, JSON, or SQLite, see [Design Decisions](design-decisions.md#why-parquet-indexes).

### High-Throughput I/O: Request Coalescing

Rasteret's speed doesn't just come from skipping header parsing; it comes from **Request Coalescing**.

When you request multiple bands or a large AOI, the I/O engine analyzes the underlying tile layout and merges nearby byte-range requests into a single, larger HTTP request. This drastically reduces the overhead of network round-trips for multi-spectral data.

The concurrency model is built on Python `asyncio` and `obstore` (Rust):
- **Record-level**: Concurrent reads across multiple rasters in a batch.
- **Band-level**: Concurrent fetches for multiple bands within a single scene.
- **Tile-level**: Coalesced byte-range reads with intelligent semaphores to cap network pressure.
- **Decompression**: Thread-pool offloading for CPU-bound tile decoding.
