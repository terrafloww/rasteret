# Changelog

## v0.3.1

### Added

- **`as_collection()`**: lightweight re-entry from an in-memory Arrow
  table or dataset back into a Collection. Validates contract columns
  and COG band metadata structs without re-running ingest, enrichment,
  or persistence. Use this after enriching a Collection's table with
  DuckDB, Polars, PyArrow, or any other tool.
  See [Enriched Parquet Workflows](how-to/enriched-parquet-workflows.md).

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
- **Multi-cloud obstore backend**: S3, Azure Blob, and GCS routing via URL
  auto-detection, with automatic fallback to anonymous access.
- **`create_backend()`** for authenticated reads with obstore credential
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
  [Enriched Parquet Workflows](how-to/enriched-parquet-workflows.md).
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
- obstore as base dependency (Rust-native async HTTP).
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
