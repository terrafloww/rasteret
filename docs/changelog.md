# Changelog

## v0.3.0

### Highlights

- License changed from AGPL-3.0-only to **Apache-2.0**.
- **Dataset catalog**: `build()`, `register()`, `register_local()` with a
  growing catalog of pre-registered datasets across Earth Search, Planetary
  Computer.
- **Multi-cloud obstore backend**: native S3, Azure Blob, and GCS routing
  via URL auto-detection. Cross-cloud credential provider guard with
  automatic fallback to anonymous access.
- **`create_backend()`** for authenticated reads with obstore credential
  providers (e.g., Planetary Computer SAS).
- **Local catalog persistence**: `register_local()` persists to
  `~/.rasteret/datasets.local.json`; `export_local_descriptor()` for
  sharing catalog entries alongside Collections.
- **Native dtype preservation**: COG tiles return in their source dtype (uint16, int8,
  float32, etc.). No forced float32 conversion.
- **Rasterio-aligned masking defaults**: AOI reads now default to `all_touched=False`
  and fill masked/outside-coverage pixels with `nodata` when present, otherwise `0`.
  The primary read API (`read_cog`) returns a `valid_mask` so ML pipelines can avoid
  learning from filled pixels.
- **rioxarray removed**: CRS encoding uses pyproj CF conventions directly (WKT2, PROJJSON,
  GeoTransform). The `xarray` extra no longer pulls rioxarray.
- **Extended TIFF header parsing**: nodata, SamplesPerPixel, PlanarConfiguration,
  PhotometricInterpretation, ExtraSamples, GeoDoubleParams CRS support.
- **Cross-CRS masking**: by default, uses the exact transformed polygon (rasterio-aligned).
  Optional bbox masking remains available for bbox-style workflows.
- **Multi-CRS auto-reprojection**: queries spanning multiple UTM zones automatically
  reproject to the most common CRS. Cross-CRS reprojection uses GDAL's
  `calculate_default_transform` for correct resolution handling.

### Other changes

- Arrow-native geometry internals (GeoArrow replaces Shapely in hot paths).
- obstore as base dependency for Rust-native HTTP backend.
- CLI: `rasteret cache build|list|info|delete|import`.
- CLI: `rasteret datasets list|info|build|register-local|export-local|unregister-local`.
- Polished documentation, tutorials, and example scripts.
- CI workflow fixes and public repo cleanup.

### Tested

- All three output paths (xarray, GDF, TorchGeo) are tested against direct
  rasterio reads across 12 datasets (Sentinel-2, Landsat, NAIP, Copernicus DEM,
  ESA WorldCover, AEF, and more). The TorchGeo path uses `rasterio.merge.merge`
  as the oracle, matching TorchGeo's own read semantics. See
  `test_dataset_pixel_comparison.py` and `test_network_smoke.py`.

### Stability

- STAC + COG scene workflows: **stable**
- Multi-cloud (S3, Azure Blob, GCS): stable
- Dataset catalog: stable
- TorchGeo adapter: **stable** (upgraded from experimental)
- Non-STAC / record-table ingestion (build_from_table): stable

### Breaking changes

- `get_xarray()` returns data in native COG dtype instead of always float32. Code that
  assumed float32 output may need adjustment (e.g., `ds.B04.values.dtype` is now `uint16`
  for Sentinel-2 instead of `float32`).
- The `xarray` extra no longer installs rioxarray. If you depend on `ds.rio.*` methods,
  install rioxarray separately.
