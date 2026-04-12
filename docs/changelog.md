# Changelog

## v0.3.11

### Added

- **Native Arrow / GeoArrow Collection export**: `Collection` now implements Arrow
  PyCapsule protocol surfaces for stream, array, and schema export, allowing
  Arrow-compatible consumers to read collection metadata directly.
- **GeoArrow WKB footprint metadata**: Rasteret now exports the `geometry`
  footprint column as `geoarrow.wkb` with CRS metadata set to `OGC:CRS84`.
- **Row-level raster CRS sidecar**: enriched and normalized collections now carry
  an Arrow-friendly `crs` string sidecar such as `EPSG:32632`, while preserving
  existing `proj:epsg` for Rasteret runtime compatibility.
- **`Collection.crs`**: returns unique row-level raster CRS codes as strings,
  using `crs` when present and falling back to legacy `proj:epsg`.
- **Arrow inputs for table builders**: `build_from_table(...)` now accepts
  Arrow-compatible in-memory objects and PyCapsule producers, including
  DuckDB/Polars-style Arrow stream exporters.

### Changed

- **Footprint CRS and raster CRS are now separated in Arrow exports**:
  GeoArrow metadata on `geometry` describes the footprint geometry CRS
  (`OGC:CRS84`), while raster-native CRS remains row-level metadata in `crs`
  and `proj:epsg`.
- **COG enrichment writes consistent CRS sidecars**: when parsed GeoTIFF header
  CRS is used to backfill CRS metadata, Rasteret now keeps `crs` and `proj:epsg`
  aligned.
- **Hugging Face streaming collections participate in Arrow export**:
  HF-backed collections now expose enriched Arrow schemas and batches instead
  of appearing empty to Arrow consumers.
- **Band/cloud read errors are more actionable**: unresolved band requests and
  cloud read failures now include available asset/metadata keys, data-source
  mapping context, and provider-specific credential/requester-pays hints.
- **Point sampling output schema is leaner**: `sample_points()` no longer emits
  `cloud_cover` by default, keeping the result focused on point, record, band,
  value, and CRS fields.
- **Docs now follow a reader-first path**: the MkDocs nav now keeps
  Getting Started, Concepts, and Transitioning from Rasterio as the first
  linear path, with task-focused How-To Guides before notebook Tutorials.
- **How-to and explanation docs were consolidated**: table/Arrow workflows,
  collection management, catalog/cloud access, point sampling, TorchGeo,
  AEF, schema, interop, and benchmark pages were tightened around the current
  Collection APIs and CRS model.
- **Benchmark docs now include the README evidence set**: the benchmark page
  now includes the Google Earth Engine time-series table and the supporting
  benchmark/cost/scaling images from the docs asset set.

### Fixed

- Fixed Arrow single-batch export for empty collections with non-empty schemas.
- Fixed requested-schema PyCapsule handling by importing requested Arrow schema
  capsules before applying compatible PyArrow casts.
- Fixed mixed-raster-CRS Arrow export so raster CRS is not incorrectly written
  as geometry column CRS.

### Tested

- Full test suite during Arrow interop validation: `364 passed, 42 skipped`.
- Verified Arrow interop with PyArrow table/reader/capsule paths.
- Verified `GeoDataFrame.from_arrow(collection)` reads footprint geometry as
  `OGC:CRS84`.
- Verified DuckDB 1.5.1 imports geometry as `GEOMETRY('OGC:CRS84')` and keeps
  `crs` as `VARCHAR`.
- Verified LanceDB 0.30.2 preserves `geoarrow.wkb` and the `crs` sidecar.
- Verified docs with `uv run --extra docs mkdocs build --strict -d /tmp/rasteret-mkdocs-check`.

## v0.3.10

### Changed

- **AEF load alias routing**: `rasteret.load("aef/v1-annual")` now opens
  the published Source Cooperative `/data` collection while keeping
  `index.parquet` attached as the runtime sidecar index for pushdown.
  The public `load()` implementation did not change; this is a catalog
  descriptor routing fix.
- **Examples cleanup**: removed the older `ml_training_with_splits.py` script;
  the split/label pattern remains documented in the ML training how-to and
  TorchGeo tutorial notebooks.
- **AEF how-to refresh**: rewrote the AEF embeddings guide around the
  maintained `rasteret.load("aef/v1-annual")` flow and moved build/DuckDB
  guidance to custom-data use cases.

### Fixed

- **Collection export bbox normalization**: `Collection.export()` now
  normalizes legacy list-style `bbox` columns through a chunk-safe struct
  conversion, avoiding `ChunkedArray`/duplicate `bbox` failures on
  export-reload workflows.


## v0.3.9

### Changed

- **Filtered CRS auto-detection for xarray reads**: internal batch iteration
  now respects active collection filters when scanning `proj:epsg` for
  `get_xarray(...)` auto-CRS selection. This avoids unintended reprojection
  on filtered subsets.
- **Nodata-aware integer xarray merge path**: for
  `xr_combine="combine_first"`, Rasteret now prefers an integer-preserving
  merge path when `_FillValue` metadata is present and consistent across
  overlapping datasets.
- **Per-band nodata propagation into xarray assembly**: COG read nodata is
  carried through RasterAccessor band results and attached as `_FillValue`
  during xarray construction.

### Fixed

- **AEF `get_xarray` memory blow-up regression**: fixed a path where filtered
  reads could trigger unnecessary reprojection and `float32` upcast, causing
  significantly higher peak RAM than expected.
- **TorchGeo instant-timestamp slice regression**:
  `RasteretGeoDataset.__getitem__` now handles zero-length temporal slices
  (`t.start == t.stop`) with a closed interval so overlap checks do not drop
  valid records/chips.

### Docs

- Updated `notebooks/07_aef_similarity_search.ipynb` with explicit memory
  notes for Franklin County / AEF 64-band runs:
  - `get_xarray`: highest peak memory (~20-24 GB)
  - `get_gdf`: medium peak memory (~3 GB)
  - `to_torchgeo_dataset`: lower bounded-memory path (~3-5 GB)
- Updated the notebook summary table to include these concrete memory ranges.
- Updated AEF notebooks to clarify that the AEF collection is already prebuilt
  for 2018–2024 (Source Cooperative + Hugging Face), and that
  `rasteret.build()` can be skipped in these workflows.
- Updated parquet how-to/examples to use **record table** terminology and a
  first-run-friendly default SourceCoop Maxar record table.

### Tested

- Verified `_detect_target_crs(...)` on filtered subsets now returns `None`
  when all selected records are same-CRS.
- Verified `sub.get_xarray(...)` on AEF Franklin subset preserves `int8`
  dtype after merge.
- Reproduced and fixed TorchGeo chip sampling regression:
  from `used=0, skipped=16` to `used=16, skipped=0` on the same Franklin
  sampler workload.

## v0.3.8 - Hotfix

- Fix catalog.py to use sourcecoop and HF AEF index.parquet instead of reading large collection parquets

## v0.3.7

### Changed

- **TorchGeo adapter semantic alignment**: `Collection.to_torchgeo_dataset(...)`
  now follows TorchGeo `RasterDataset` query behavior more closely.
  - `time_series=True`: applies temporal overlap from the sampler/query slice,
    then stacks selected records into `[T, C, H, W]`.
  - `time_series=False`: mosaics overlapping records on the query grid with
    first-record precedence and nodata-aware filling.
- **TorchGeo docs clarity**: interop and reference docs now describe the
  updated time-series and non-time-series slice behavior.

### Tested

- Expanded `test_torchgeo_error_propagation` with semantic coverage for:
  - query-temporal filtering in `time_series=True`
  - overlapping-record mosaicking in `time_series=False`
- Updated `test_torchgeo_network` wording to match the query-slice semantics.

## v0.3.6

### Added

- **Bounded nodata fallback for `sample_points()`**: point sampling can now
  search outward from the base pixel under the input point using
  `max_distance_pixels`, measured in Chebyshev distance (square rings).
- **Neighbourhood window output for `sample_points()`**:
  `return_neighbourhood` now supports:
  - `"off"`: no neighbourhood column
  - `"always"`: always return the full searched window
  - `"if_center_nodata"`: return the window only when the base pixel is
    nodata/NaN

### Changed

- **Point sampling semantics are now explicit and bounded**:
  `sample_points()` still samples the pixel containing the point first, but
  when that base pixel is nodata/NaN and `max_distance_pixels > 0`, Rasteret
  searches outward ring by ring and chooses the closest valid candidate by
  exact point-to-pixel-rectangle distance.
- **Neighbourhood output schema**: when `return_neighbourhood != "off"`,
  `sample_points()` returns a nullable `neighbourhood_values` list column in
  row-major order. With `"if_center_nodata"`, rows whose base pixel is valid
  keep `neighbourhood_values = NULL`.
- **Neighbourhood mode validation**: neighbourhood output now requires
  `max_distance_pixels > 0` so the requested window size is always explicit.

### Tested

- Expanded `test_execution` for bounded nodata fallback, exact-distance winner
  selection, cross-tile fallback, neighbourhood window output, NULL-window
  behavior for `"if_center_nodata"`, and empty-result schema stability.
- Expanded `test_public_api_surface` for the new `sample_points()` arguments.

## v0.3.5

### Added

- **Descriptor-backed dual-surface planning**: catalog entries can now
  describe separate record-table, index, and collection surfaces via
  `record_table_uri`, `index_uri`, and `collection_uri`. This lets Rasteret
  keep build-time metadata sources separate from published runtime
  collections.
- **`Collection.head()`**: first-class metadata preview API that uses the
  narrow record index when available instead of forcing a wide collection
  scan.
- **Internal Parquet read planner**: `ParquetReadPlanner` keeps index-side and
  wide-scan filter state coherent across `subset()`, `where()`, `len()`,
  `head()`, pixel reads, and TorchGeo entry points.
- **Hugging Face streaming runtime**: `HFStreamingSource` and batch iterators
  now provide a stable metadata path for `hf://datasets/...` collections
  without routing through the older `datasets` streaming shutdown path.

### Changed

- **Index-first runtime filtering**: when a collection has both a narrow index
  and a wide read-ready surface, Rasteret now plans metadata filters against
  the index first, then carries compatible predicates into the wide scan.
- **GeoParquet bbox contract**: bbox filtering now targets the canonical
  `bbox` struct (`xmin`, `ymin`, `xmax`, `ymax`) instead of older scalar bbox
  columns. Catalog tests, CLI fixtures, and filter tests were updated to
  match the GeoParquet 1.1 shape.
- **TorchGeo filter propagation**: `to_torchgeo_dataset()` now respects
  collection filters and geometry / bbox narrowing before chip sampling
  starts, reducing unnecessary raster candidates for gridded reads.
- **Local Parquet metadata reuse**: local datasets prefer a `_metadata`
  sidecar when present, and Rasteret now keeps an in-process Parquet dataset
  cache to reduce repeated footer/schema setup during interactive sessions.
- **Catalog surface reporting**: CLI output and descriptor helpers now surface
  record-table, index, and collection paths explicitly instead of collapsing
  everything into `geoparquet_uri`.

### Fixed

- **Hugging Face filter execution**: Arrow expression handling on HF-backed
  metadata batches now fails more clearly and avoids the unstable runtime path
  that could crash during shutdown or long scans.
- **AEF runtime aliasing**: the built-in AEF descriptor now points at the
  published Terrafloww collection/index surfaces with explicit field-role and
  filter-capability hints.
- **Read-path correctness on filtered collections**: `len()`, `head()`,
  `sample_points()`, TorchGeo reads, and other collection-backed reads now
  stay aligned when filters are staged across both record-index and wide-data
  surfaces.

### Tested

- Expanded `test_collection_filters` for bbox-struct filtering and TorchGeo
  prefilter behavior.
- Expanded `test_catalog`, `test_cli`, and `test_public_api_surface` for the
  new descriptor surface roles and runtime load/build paths.
- Expanded `test_huggingface`, `test_execution`, and `test_torchgeo_adapter`
  for the new HF runtime path, filter propagation, and collection-read
  behavior.

---

## v0.3.4

### Added

- **HuggingFace integration**: `rasteret.load("hf://datasets/<org>/<repo>")`
  resolves remote Parquet shards via `huggingface_hub` and loads them as a
  Collection. No local clone or download step needed.
- **AEF v1 Annual Rasteret Collection**: prebuilt read-ready collection
  published on [HuggingFace](https://huggingface.co/datasets/terrafloww/aef-v1-annual-rasteret)
  and [Source Cooperative](https://source.coop/terrafloww/aef-v1-annual-rasteret).
  235K tiles, 64-band int8 embeddings, global coverage 2017–2024.
- **AEF similarity search tutorial** (`notebooks/07_aef_similarity_search`):
  end-to-end embedding similarity search using `sample_points`, `get_xarray`,
  `get_gdf`, and TorchGeo `GridGeoSampler`. Demonstrates DuckDB Arrow-native
  pivot for reference vectors and lonboard GPU-accelerated visualization.

### Changed

- **Unified tile engine**: `RasterAccessor` consolidated into a single
  `_read_tile()` code path shared by `get_xarray`, `get_numpy`, `get_gdf`,
  and TorchGeo `__getitem__`. Four divergent tile-read implementations
  replaced with one.
- **COGReader simplified**: removed duplicate decode paths, tightened the
  fetch → decompress → crop pipeline, reduced code surface by ~200 lines.
- **TorchGeo edge-chip hardening**: empty-read validation returns
  nodata-filled tensors for chips outside coverage; positive-overlap
  filtering skips false bbox-only candidates; fallback loop fills chips
  with nodata when all candidate tiles fail instead of crashing the
  DataLoader.
- **Error surfacing**: `get_gdf` and `get_numpy` warn on partial
  band/geometry/record failures instead of returning silent empty results.
  Point geometry AOIs raise `UnsupportedGeometryError` pointing to
  `sample_points()`. Exception chaining (`raise ... from e`) throughout
  the read pipeline.

### Fixed

- `sample_points` COG read path: tile metadata validation was skipping
  valid tiles when the source raster had multiple matching records.
- `xr_combine` parameter now correctly plumbed through `get_xarray()`.

### Tested

- `test_torchgeo_error_propagation`: edge-chip, empty-read, and fallback
  loop coverage.
- `test_huggingface`: URI resolution and table loading mocks.
- `test_public_api_surface`: validates all public imports from `rasteret`.
- `test_public_network_smoke`: live AEF reads and point sampling parity
  against `rasterio.sample()`.
- Extended `test_execution` with `get_gdf` error path coverage.

---

## v0.3.3
### Performance

- **Arrow-batch native point sampling**: `sample_points()` internals now use
  vectorized NumPy gathers and `Table.from_batches` instead of per-sample
  Python append + `pa.concat_tables`. Eliminates row materialization in the
  hot loop.
- **COGReader session reuse**: `read_cog()` accepts a shared `reader=`
  parameter. Point sampling across multiple rasters now reuses a single
  `COGReader` (and its HTTP/2 connection pool) instead of creating one per
  raster, reducing connection overhead for multi-scene workloads.

### Refactored

- **Dedicated `point_sampling` module**: point sampling ownership moved from
  `execution.py` to `rasteret.core.point_sampling`. `execution.py` stays
  focused on area/chip reads (`get_xarray`, `get_numpy`, `get_gdf`).
- **`POINT_SAMPLES_SCHEMA`** defined in `types.py` as the single source of
  truth for point sample output columns. Nullable fields explicitly modeled.
- **Point input helpers consolidated**: `ensure_point_geoarrow`,
  `candidate_point_indices_for_raster`, and related helpers moved into
  `geometry.py`.
- **Strict tile/source alignment guard** in `RasterAccessor.sample_points`:
  validates that tile metadata matches the source raster before sampling.

### Changed

- Landing page (`docs/index.md`) code example simplified: single PyArrow
  import, cleaner `sample_points` call, HF benchmark collapsed into
  admonition.

---

## v0.3.2

### Added

- **`Collection.sample_points()`**: first-class point sampling API returning a
  `pyarrow.Table` (`point_index`, `record_id`, `datetime`, `band`, `value`,
  CRS columns, and metadata fields). Supports `match="all"` and
  `match="latest"`.
- New guide: **[Point Sampling and Masking](how-to/point-sampling-and-masking.md)**.

### Changed

- **Masking controls surfaced at Collection level**:
  `get_xarray()`, `get_gdf()`, and `get_numpy()` now accept
  `all_touched=...` directly.
- **Table-native point sampling inputs**:
  `Collection.sample_points(...)` now accepts Arrow tables and common
  dataframe/relation inputs with `x_column`/`y_column` or
  `geometry_column` (WKB/GeoArrow/shapely point column).
- Point sampling aligns to **rasterio `sample()` semantics** (nearest-pixel
  index math) for deterministic parity on real COGs.
- Core read internals are more Arrow-native:
  - `iterate_rasters()` now scans columnar batches (no `batch.to_pylist()` in
    the core read iterator),
  - `infer_data_source()` uses filtered `scanner.head(1)`,
  - multi-CRS detection in xarray path streams `proj:epsg` counts per batch,
  - `sample_points()` uses vectorized `geoarrow.point_coords`.
- Error messaging for non-OGC binary geometry input now explicitly points
  DuckDB users to `ST_AsWKB(geom)` when needed.
- Network parity coverage is tighter:
  - AOI windowing now matches rasterio `geometry_window()` edge semantics
    exactly (fixes the WorldCover 1-pixel mismatch),
  - transient STAC API timeouts are retried during live builds,
  - the AEF south-up TorchGeo oracle path is corrected for manual/explicit
    parity runs via `WarpedVRT`.

### Tested

- Public network smoke: `test_public_network_smoke.py` passes with `--network`,
  including point sampling parity against `rasterio.sample()`.
- Network smoke (`test_network_smoke.py`) passes for available providers
  (expected skips remain for missing optional extras/credentials).

### Packaging

- `sedonadb>=0.2.0` added to the `examples` extra for Arrow-native point
  workflow examples.

---

## v0.3.1

### Added

- **`as_collection()`**: lightweight re-entry from an in-memory Arrow
  table or dataset back into a Collection. Validates contract columns
  and COG band metadata structs without re-running ingest, enrichment,
  or persistence. Use this after enriching a Collection's table with
  DuckDB, Polars, PyArrow, or any other tool.
  See [Enriched Collection Workflows](how-to/enriched-collection-workflows.md).

### Changed

- **Lifecycle docs**: all entry points (`build*`, `load`, `as_collection`)
  now cross-reference each other in docstrings and how-to guides.
  Contributing guide updated with four-layer architecture
  (Build → Query → Read → Re-entry).
- Major TOM on-the-fly example uses `as_collection()` + explicit
  `export()` instead of `build_from_table(enrich_cog=False)`.
  `enrich_major_tom_columns()` now preserves `year`/`month` partition
  columns from the base Collection.

---

## v0.3.0

### Highlights

- License changed from AGPL-3.0-only to **Apache-2.0**.
- **Dataset catalog**: `build()` with 12 pre-registered datasets across
  Earth Search, Planetary Computer, and AlphaEarth Foundation.
  Catalog entries can point to a STAC API or a GeoParquet file.
  `register_local()` for adding your own.
- **`build_from_stac()`** and **`build_from_table()`**: build a Collection
  from any STAC API or any Parquet/GeoParquet file with COG URLs (Source
  Cooperative exports, STAC GeoParquet, custom catalogs). No STAC API
  required for the table path. Optional `enrich_cog=True` parses COG
  headers for accelerated reads.
- **Multi-cloud support**: S3, Azure Blob, and GCS routing via URL
  auto-detection, with automatic fallback to anonymous access. obstore replaces
  direct aiohttp as the HTTP transport, adding unified multi-cloud routing.
- **`create_backend()`** for authenticated reads with credential
  providers (e.g., Planetary Computer SAS tokens).
- **TorchGeo adapter**: `collection.to_torchgeo_dataset()` returns a
  `GeoDataset` backed by Rasteret's async COG reader. Supports
  `time_series=True` (`[T, C, H, W]` output), `label_field` for
  per-sample labels, `target_crs` for cross-CRS reprojection,
  `allow_resample=True` for mixed-resolution bands, and `is_image=False`
  for mask-style datasets. Works with all TorchGeo samplers, collation
  helpers, and dataset composition (`IntersectionDataset`, `UnionDataset`).
- **Native dtype preservation**: COG tiles return in their source dtype
  (uint16, int8, float32, etc.) instead of forcing float32.
- **Rasterio-aligned masking**: AOI reads default to `all_touched=False`
  and fill outside-coverage pixels with `nodata` when present, otherwise `0`.
  `read_cog` returns a `valid_mask`.
- **rioxarray removed**: CRS encoding uses pyproj CF conventions directly.
  The `xarray` extra no longer pulls rioxarray.
- **Extended TIFF header parsing**: nodata, SamplesPerPixel,
  PlanarConfiguration, PhotometricInterpretation, ExtraSamples,
  GeoDoubleParams CRS support.
- **Multi-CRS auto-reprojection**: queries spanning multiple UTM zones
  reproject to the most common CRS using GDAL's
  `calculate_default_transform`.
- **`get_numpy()`**: lightweight NumPy output path returning `[N, H, W]`
  (single band) or `[N, C, H, W]` (multi-band) arrays. No extra
  dependencies beyond NumPy. Accepts bbox tuples, Arrow arrays, Shapely
  objects, or raw WKB.
- **`get_gdf()`**: GeoDataFrame output path for analysis workflows.
- **Enriched Parquet workflows**: append arbitrary columns (splits, labels,
  AOI polygons, model scores) to a Collection's Parquet, query with
  DuckDB/PyArrow, and fetch pixels for matching rows on demand. See
  [Enriched Collection Workflows](how-to/enriched-collection-workflows.md).
- **Major TOM on-the-fly**: example workflow rebuilding Major TOM-style
  patch-grid semantics from source Sentinel-2 COGs instead of
  payload-in-Parquet. Benchmarked 3.9-6.5x faster than HF `datasets`
  Parquet-filter reads.
- **`earthdata` optional extra**: `pip install rasteret[earthdata]` for
  NASA Earthdata auto-credential detection.

### Collection API

- **Output paths**: `get_xarray()`, `get_numpy()`, `get_gdf()`,
  `to_torchgeo_dataset()`. All share the same async tile I/O underneath.
- **Inspection**: `.bands`, `.bounds`, `.epsg`, `len()`, `__repr__()`,
  `.describe()`, `.compare_to_catalog()`.
- **Filtering**: `collection.subset(cloud_cover_lt=..., date_range=...,
  bbox=..., geometries=..., split=...)` and `collection.where(expr)` for
  raw Arrow expressions.
- **Sharing**: `collection.export("path/")` writes a portable copy;
  `rasteret.load("path/")` reloads it. `list_collections()` discovers
  cached collections in the workspace.
- **Three-tier schema**: required columns (`id`, `datetime`, `geometry`,
  `assets`), COG acceleration columns (per-band tile offsets and metadata),
  and user-extensible columns (`split`, `label`, `cloud_cover`, custom
  metadata). See [Schema Contract](explanation/schema-contract.md).
- **Public exports**: `Collection`, `CloudConfig`, `BandRegistry`,
  `DatasetDescriptor`, `DatasetRegistry` are all importable from
  `rasteret`.

### Other changes

- Arrow-native geometry internals (GeoArrow replaces Shapely in hot paths).
- obstore as base dependency (multi-cloud support).
- CLI: `rasteret collections build|list|info|delete|import`,
  `rasteret datasets list|info|build|register-local|export-local|unregister-local`.
- TorchGeo `time_series=True` uses spatial-only intersection, matching
  TorchGeo's own `RasterDataset` behaviour where all spatially overlapping
  records are stacked regardless of the sampler's time slice.
- Cloud workspace URIs (e.g. `s3://bucket/path`) are preserved correctly
  in `CollectionBuilder` base class.

### Tested

- All four output paths (xarray, GeoDataFrame, NumPy, TorchGeo) tested
  against direct rasterio reads across 12 datasets (Sentinel-2, Landsat,
  NAIP, Copernicus DEM, ESA WorldCover, AEF, and more).
- TorchGeo adapter verified against the full GeoDataset contract:
  `IntervalIndex`, samplers, collation, dataset composition, cross-CRS
  reprojection, and export/reload roundtrips.

### Requirements

- **Python 3.12+** required.
- `rasterio>=1.4.3,<1.5.0` is a core dependency (used for geometry
  masking, CRS reprojection, and TorchGeo query-grid placement; not in
  the tile-read path).

### Breaking changes

- `get_xarray()` returns data in native COG dtype instead of always float32.
  Code that assumed float32 output may need adjustment.
- The `xarray` extra no longer installs rioxarray. If you depend on
  `ds.rio.*` methods, install rioxarray separately.
