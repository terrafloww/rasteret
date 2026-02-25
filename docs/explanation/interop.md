# Ecosystem Comparison

Rasteret accelerates reads from tiled GeoTIFF collections by caching tile
layout metadata in a Parquet index. It works alongside TorchGeo, xarray,
and rasterio, not instead of them.

Collections are written as **GeoParquet 1.1** today (WKB + `geo` metadata).
Parquet-native `GEOMETRY`/`GEOGRAPHY` logical types and GeoParquet 2.0 are
emerging; Rasteret tracks this and plans to adopt newer encodings when ecosystem
support stabilizes.

## Interop

### TorchGeo

`Collection.to_torchgeo_dataset()` returns a standard TorchGeo
[`GeoDataset`](../reference/integrations/torchgeo.md). Your samplers,
DataLoader, and training loop do not change.

This is **pipeline-level interop**: Rasteret provides a TorchGeo dataset object
that plugs into TorchGeo's samplers and transforms, while Rasteret remains the
pixel I/O backend. TorchGeo's own `RasterDataset` still reads via rasterio/GDAL
and remains the right tool when Rasteret's COG/tile constraints don't apply.

```python
dataset = collection.to_torchgeo_dataset(bands=["B04", "B03", "B02"], chip_size=256)
sampler = RandomGeoSampler(dataset, size=256, length=100)
loader  = DataLoader(dataset, sampler=sampler, batch_size=4, collate_fn=stack_samples)
```

Rasteret preserves the native COG dtype (e.g., `uint16` for Sentinel-2)
whereas TorchGeo converts to `float32` by default (via its `dtype` property).

Multi-CRS scenes are auto-reprojected to a common CRS using GDAL's
`calculate_default_transform` for correct resolution handling.

Rasteret's read pipeline can produce a `valid_mask` (boolean) so ML workflows
can distinguish filled pixels from real source data. The TorchGeo adapter keeps
samples TorchGeo-standard by default and does not include `valid_mask`.

For mask-style datasets, pass `is_image=False` to return `sample["mask"]`
instead of `sample["image"]` (single-band data squeezes the channel
dimension, matching TorchGeo `RasterDataset` conventions).

If requested bands have different resolutions, Rasteret fails fast by default.
To opt into resampling bands onto a common grid in the TorchGeo adapter, pass
`allow_resample=True` to `Collection.to_torchgeo_dataset(...)`.

See [Tutorial 02](../tutorials/02_torchgeo_09_accelerator.ipynb) and
[Tutorial 05](../tutorials/05_torchgeo_comparison.ipynb).

### xarray / GeoPandas

Rasteret handles the I/O (async byte-range reads via obstore), then hands
off to standard xarray and GeoPandas objects for analysis:

- [`Collection.get_xarray(...)`](../reference/core/execution.md) returns an `xr.Dataset`
- [`Collection.get_gdf(...)`](../reference/core/execution.md) returns a `gpd.GeoDataFrame`

See [Tutorial 01](../tutorials/01_quickstart_xarray.ipynb).

#### CRS encoding

xarray output uses CF conventions via pyproj (no rioxarray dependency):

- `spatial_ref` coordinate with WKT2 (ISO 19162:2019), PROJJSON, and
  CF grid-mapping attributes
- `GeoTransform` attribute for GDAL-compatible tools
- Pixel-center coordinates (half-pixel offset from tile origin)

Code that uses `ds.rio.*` methods will need to `pip install rioxarray`
separately. The `spatial_ref` coordinate written by Rasteret is compatible
with rioxarray if installed.

#### Data types

Band arrays return in the native COG dtype. For Sentinel-2 L2A, that is
`uint16` (surface reflectance values 0-10000). Geometry masking fills
outside-AOI / outside-coverage pixels with the COG `nodata` value when
present, otherwise `0`, preserving native dtype. For ML workloads that
should avoid learning from filled pixels, use the `valid_mask` returned
by Rasteret reads.

#### Multi-CRS

When a query spans records from multiple CRS zones (e.g., adjacent UTM
zones), Rasteret auto-detects this and reprojects all tiles to the most
common CRS before merging. A warning is logged. Pass `target_crs=` to
`get_xarray()` or `get_gdf()` to override.

### rasterio

Rasteret uses rasterio for geometry masking (`rasterio.features.geometry_mask`),
multi-CRS reprojection (`rasterio.warp.reproject`), and TorchGeo query-grid
placement (`rasterio.merge.merge` via `rio_semantics.py`). CRS transforms and
coordinate operations use pyproj directly. Tile reads go through Rasteret's
own async pipeline backed by obstore. No GDAL in the tile-read path.

CRS encoding in xarray output uses pyproj's CF conventions (`CRS.to_cf()`,
`CRS.to_wkt()`, `CRS.to_json()`), not rioxarray.

## Alternative approaches

These libraries solve related problems with different designs:

**GeoParquet "Parquet Raster" (alpha/WIP)**: a draft specification for storing
raster payloads (and/or external raster references) in Parquet
([draft spec](https://github.com/opengeospatial/geoparquet/blob/main/format-specs/parquet-raster.md)).
Rasteret is different: it uses GeoParquet as a **record table/index** and reads
pixel tiles from existing GeoTIFF/COG assets via byte-range I/O. If Parquet Raster
stabilizes, it may become an interop/export target, but it is not what Rasteret
writes today.

**TACO / tacoTIFF**: packaging-first (materializes data into a TACO layout
with a `level0.parquet` manifest). Rasteret is indexing-first (indexes
existing tiled GeoTIFFs in place, no data copying). The approaches are
complementary: Rasteret's
`DatasetDescriptor` can point to a TACO `level0.parquet` via `geoparquet_uri`, and
`build_from_table()` can ingest it like any other Parquet source. As TACO
matures, deeper interop (e.g. layout-aware reads) is a natural extension.

**async-geotiff / async-tiff**: fast low-level async GeoTIFF readers.
Interop with Rasteret is possible by replacing the tile-reading layer, but
they don't yet support passing pre-parsed IFD metadata.

**virtual-tiff**: oriented towards making TIFF data accessible to the Zarr
ecosystem by exposing tiles as Zarr-compatible chunks. Rasteret reads tiles
directly via byte-range requests using a Parquet index of tile-layout
metadata.

## When to use what

| Your data | Recommendation |
|-----------|---------------|
| Cloud-hosted tiled GeoTIFFs (Sentinel-2, Landsat, etc.) | Rasteret (over 20x faster) |
| Local tiled GeoTIFFs | Rasteret works; speedup is smaller, but the index is still useful for filtering and sharing |
| Non-tiled GeoTIFFs (striped layout) | TorchGeo / rasterio |
| Non-TIFF formats (NetCDF, HDF5, GRIB) | TorchGeo / rasterio |

## Testing

The test suite includes pixel-level comparisons against direct rasterio
reads for the xarray, GeoDataFrame, and TorchGeo output paths. The TorchGeo
comparison uses `rasterio.merge.merge` as the oracle, matching what TorchGeo's
own `_merge_or_stack` calls. Coverage spans 12 datasets including Sentinel-2,
Landsat, NAIP, Copernicus DEM, ESA WorldCover, and AEF (south-up). See
`test_dataset_pixel_comparison.py` (requires `--network`), plus
`test_public_network_smoke.py`, `test_torchgeo_network_usage.py`, and
`test_network_smoke.py`.

If you encounter edge cases where output differs from rasterio, please
[file an issue](https://github.com/terrafloww/rasteret/issues).
