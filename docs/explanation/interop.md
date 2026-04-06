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

!!! note "Version requirement"
    Rasteret requires **TorchGeo >= 0.9.0**. Earlier versions use a different
    `GeoDataset` index structure that is incompatible with the adapter.

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

#### GeoDataset contract

`RasteretGeoDataset` subclasses TorchGeo's `GeoDataset` and honors the
full contract that samplers and dataset composition rely on:

| Surface | What Rasteret does |
|---|---|
| `__getitem__(GeoSlice) -> Sample` | Returns `{"image": Tensor, "bounds": Tensor, "transform": Tensor}` (or `"mask"` when `is_image=False`) |
| `index` | GeoPandas GeoDataFrame with `IntervalIndex` named `"datetime"` and Shapely footprint geometry |
| `crs` | Set from the collection's EPSG code via `CRS.from_epsg()` |
| `res` | Derived from the first record's COG metadata transform |
| Samplers | Works with `RandomGeoSampler`, `GridGeoSampler`, and any sampler that reads `bounds`, `index`, and `res` |
| Dataset composition | Works with `IntersectionDataset` and `UnionDataset`; the index is designed so `reset_index()` does not conflict |

Rasteret replaces the I/O backend (custom IO instead of rasterio/GDAL)
but speaks the same interface. Nothing downstream of the dataset object
needs to change.

#### Rasteret additions

These are features Rasteret adds on top of the GeoDataset contract. They
do not break interop because TorchGeo ignores unknown sample keys, and
constructor parameters are Rasteret-specific.

| Feature | What it does | Interop impact |
|---|---|---|
| `label_field` | Adds `sample["label"]` from a metadata column | None: extra key, ignored by TorchGeo trainers |
| `time_series=True` | Stacks records that overlap the sampler/query spatiotemporal slice into `[T, C, H, W]` | None: standard tensor shape, works with TorchGeo transforms |
| `target_crs=` | Reprojects scenes from different CRS zones on the fly | None: result has uniform CRS, transparent to samplers |
| `cloud_config=` | Configures authenticated cloud reads (requester-pays, signed URLs) | None: constructor-level, transparent to samplers |
| `allow_resample=True` | Resamples bands with different native resolutions onto a common grid | None: output tensor has uniform resolution |

#### Behavior details

Rasteret preserves the native COG dtype (e.g., `uint16` for Sentinel-2)
whereas TorchGeo converts to `float32` by default (via its `dtype` property).

Multi-CRS scenes are auto-reprojected to a common CRS using GDAL's
`calculate_default_transform` for correct resolution handling.

Rasteret computes a `valid_mask` (boolean) during COG reads to identify valid
pixels. Point sampling uses this mask to skip filled pixels. The TorchGeo
adapter keeps samples TorchGeo-standard by default and does not include
`valid_mask`.

For mask-style datasets, pass `is_image=False` to return `sample["mask"]`
instead of `sample["image"]` (single-band data squeezes the channel
dimension, matching TorchGeo `RasterDataset` conventions).

If requested bands have different resolutions, Rasteret fails fast by default.
To opt into resampling bands onto a common grid in the TorchGeo adapter, pass
`allow_resample=True` to `Collection.to_torchgeo_dataset(...)`.

When records in a collection have different native resolutions, Rasteret warns
at dataset creation time. The read path resamples each tile to the query grid
correctly regardless.

See [TorchGeo Integration](../tutorials/02_torchgeo_09_accelerator.ipynb) and
[TorchGeo Benchmark](../tutorials/05_torchgeo_comparison.ipynb).

### xarray / GeoPandas / NumPy

Rasteret handles the I/O (async byte-range reads), then hands
off to standard xarray, GeoPandas, or NumPy outputs:

- [`Collection.get_xarray(...)`](../reference/core/collection.md) returns an `xr.Dataset`
- [`Collection.get_gdf(...)`](../reference/core/collection.md) returns a `gpd.GeoDataFrame`
- [`Collection.get_numpy(...)`](../reference/core/collection.md) returns NumPy arrays (`[N, H, W]` or `[N, C, H, W]`)
- [`Collection.sample_points(...)`](../reference/core/collection.md) returns an Arrow table for point features (`pyarrow.Table`)

See [Quickstart](../tutorials/01_quickstart.ipynb).

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
present, otherwise `0`, preserving native dtype.

#### Multi-CRS

When a query spans records from multiple CRS zones (e.g., adjacent UTM
zones), Rasteret auto-detects this and reprojects all tiles to the most
common CRS before merging. A warning is logged. Pass `target_crs=` to
`get_xarray()`, `get_numpy()`, or `get_gdf()` to override.

### rasterio

Rasteret uses rasterio for geometry masking (`rasterio.features.geometry_mask`),
multi-CRS reprojection (`rasterio.warp.reproject`), and TorchGeo query-grid
placement (`rasterio.merge.merge` via `rio_semantics.py`). CRS transforms and
coordinate operations use pyproj directly. Tile reads go through Rasteret's
own async IO. No GDAL in the tile-read path.

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
| Cloud-hosted tiled GeoTIFFs (Sentinel-2, Landsat, etc.) | Rasteret (up to 20x faster) |
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
`test_public_network_smoke.py`, `test_torchgeo_network.py`, and
`test_network_smoke.py`.

If you encounter edge cases where output differs from rasterio, please
[file an issue](https://github.com/terrafloww/rasteret/issues).
