# Changelog

## v0.3.0

### Highlights

- License changed from AGPL-3.0-only to **Apache-2.0**.
- **Dataset catalog**: `build()` with 13 pre-registered datasets across
  Earth Search, Planetary Computer, and AlphaEarth Foundation.
  `register_local()` for adding your own.
- **Multi-cloud obstore backend**: S3, Azure Blob, and GCS routing via URL
  auto-detection, with automatic fallback to anonymous access.
- **`create_backend()`** for authenticated reads with obstore credential
  providers (e.g., Planetary Computer SAS tokens).
- **TorchGeo adapter**: `collection.to_torchgeo_dataset()` returns a
  `GeoDataset` backed by Rasteret's async COG reader. Supports
  `time_series=True` (`[T, C, H, W]` output), multi-CRS reprojection,
  and works with all TorchGeo samplers and collation helpers.
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

### Collection API

- **Inspection**: `.bands`, `.bounds`, `.epsg`, `len()`, `__repr__()`,
  `.describe()`, `.compare_to_catalog()`.
- **Filtering**: `collection.subset(cloud_cover_lt=..., date_range=...,
  bbox=..., geometries=..., split=...)` and `collection.where(expr)` for
  raw Arrow expressions.
- **Sharing**: `collection.export("path/")` writes a portable copy;
  `rasteret.load("path/")` reloads it.

### Other changes

- Arrow-native geometry internals (GeoArrow replaces Shapely in hot paths).
- obstore as base dependency (Rust-native async HTTP).
- CLI: `rasteret collections build|list|info|delete|import`,
  `rasteret datasets list|info|build|register-local|export-local|unregister-local`.
- TorchGeo `time_series=True` uses spatial-only intersection, matching
  TorchGeo's own `RasterDataset` behaviour where all spatially overlapping
  records are stacked regardless of the sampler's time slice.

### Tested

- All three output paths (xarray, GeoDataFrame, TorchGeo) tested against
  direct rasterio reads across 12 datasets (Sentinel-2, Landsat, NAIP,
  Copernicus DEM, ESA WorldCover, AEF, and more).

### Breaking changes

- `get_xarray()` returns data in native COG dtype instead of always float32.
  Code that assumed float32 output may need adjustment.
- The `xarray` extra no longer installs rioxarray. If you depend on
  `ds.rio.*` methods, install rioxarray separately.
