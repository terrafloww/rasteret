# Ecosystem Interop

Rasteret is designed to work with the geospatial and Arrow tools you already use. It owns the COG indexing and byte-range read path, then hands metadata or pixels to standard libraries.

The main boundary is:

```text
Rasteret Collection metadata: Arrow / Parquet
Rasteret pixel reads: COG byte ranges
Outputs: NumPy / xarray / GeoPandas / TorchGeo / Arrow tables
```

## Arrow And GeoArrow

A collection can be passed to Arrow-aware tools through the Arrow protocol. Rasteret exports the footprint `geometry` column as `geoarrow.wkb` and marks that footprint geometry as CRS84.

```python
import polars as pl

frame = pl.from_arrow(collection)
```

For read-ready tables coming back from Arrow-native tools:

```python
experiment = rasteret.as_collection(
    frame,
    data_source=collection.data_source,
)
```

Use `data_source=...` when a table engine may have dropped schema metadata or changed Arrow string types during a round trip.

CRS distinction:

- `geometry` is the raster footprint and is exported as CRS84 GeoArrow WKB.
- `crs` and `proj:epsg` are row-level raster CRS sidecars used by Rasteret's read path.

This lets GeoPandas, DuckDB, Polars, and PyArrow inspect or join collection metadata.

## TorchGeo

`Collection.to_torchgeo_dataset(...)` returns a TorchGeo `GeoDataset`. TorchGeo samplers, transforms, composition, and dataloaders stay standard; the pixel reads are served by Rasteret.

```python
dataset = collection.to_torchgeo_dataset(
    bands=["B04", "B03", "B02"],
    chip_size=256,
)
```

Rasteret-specific options for TorchGeo include:

| Option                | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| `label_field="label"` | Include a collection column as `sample["label"]`.            |
| `split="train"`       | Filter the collection before dataset construction.           |
| `target_crs=...`      | Read multi-zone data into a chosen CRS.                      |
| `time_series=True`    | Stack matching records as `[T, C, H, W]`.                    |
| `allow_resample=True` | Opt into resampling bands with different native resolutions. |
| `is_image=False`      | Return `sample["mask"]` for mask-style datasets.             |

For workflow examples, see [TorchGeo Integration](https://terrafloww.github.io/rasteret/how-to/torchgeo-integration/index.md), [Bring Your Own AOIs, Points, And Metadata](https://terrafloww.github.io/rasteret/how-to/enriched-collection-workflows/index.md), and [Multi-Dataset Training](https://terrafloww.github.io/rasteret/how-to/multi-dataset-training/index.md).

## xarray, GeoPandas, NumPy, And Point Tables

Rasteret's main pixel output methods are:

| Method               | Output                                             |
| -------------------- | -------------------------------------------------- |
| `get_numpy(...)`     | NumPy array.                                       |
| `get_xarray(...)`    | xarray Dataset with coordinates and CRS metadata.  |
| `get_gdf(...)`       | GeoPandas GeoDataFrame with pixel arrays attached. |
| `sample_points(...)` | PyArrow table with point sample rows.              |

`get_gdf(...)` and `sample_points(...)` keep useful columns from your AOI or point tables, such as IDs, labels, or splits. `get_numpy(...)` and `get_xarray(...)` can read table geometries too, but they return array-style objects rather than business tables.

xarray output uses pyproj/CF-style CRS metadata, including a `spatial_ref` coordinate and GDAL-compatible `GeoTransform` attribute. rioxarray is not required, but tools that understand CF grid mapping metadata can use those fields.

Band arrays preserve the native COG dtype. For example, Sentinel-2 L2A data is typically `uint16`; AEF embeddings are `int8`. Cast or de-quantize when your analysis needs floats.

Point sampling returns a table:

```python
samples = collection.sample_points(
    points=points_table,
    x_column="lon",
    y_column="lat",
    bands=["B04"],
    geometry_crs=4326,
)
```

The result can continue into PyArrow, DuckDB, Polars, pandas, or another table tool. Point-table columns are preserved unless they collide with Rasteret output column names.

## rasterio And pyproj

Rasteret uses rasterio and pyproj where their semantics matter:

- pyproj for CRS transforms and xarray CRS metadata.
- rasterio geometry masking for polygon reads.
- rasterio reprojection/merge semantics for multi-CRS and query-grid behavior.

Tile byte-range reads go through Rasteret's custom async COG reader, not through GDAL's file-opening path.

## Related Formats And Tools

**GeoParquet**: Rasteret collection metadata is table-shaped and can be exported with GeoParquet metadata for footprint geometry. Rasteret does not store raster pixel payloads inside GeoParquet.

**Parquet Raster**: GeoParquet has an alpha raster proposal for representing raster payloads "images-inside-Parquet". Rasteret is a COG index/read engine, not a Parquet-raster writer.

**TACO / tacoTIFF**: TACO packages imagery into a layout with a Parquet manifest. Rasteret can ingest suitable manifests through `build_from_table(...)`; deeper layout-aware reads would be a future integration point.

**async-geotiff / async-tiff**: these are lower-level async TIFF readers. They could replace parts of Rasteret's tile reader in the future if they support the metadata and layout behavior Rasteret needs.

**virtual-tiff**: exposes TIFF tiles as Zarr-compatible chunks. Rasteret instead reads COG tiles directly from byte ranges using metadata stored in the collection.

## When To Use What

| Data / task                                        | Usually use                                  |
| -------------------------------------------------- | -------------------------------------------- |
| Remote tiled GeoTIFFs / COGs with repeated reads   | Rasteret                                     |
| One-off local TIFF inspection                      | rasterio                                     |
| Non-TIFF raster formats such as NetCDF, HDF5, GRIB | rasterio/xarray/TorchGeo-native paths        |
| TorchGeo training over COG collections             | Rasteret `to_torchgeo_dataset(...)`          |
| Metadata joins, splits, labels, and filtering      | Arrow-native tools plus `as_collection(...)` |

## Verification

Rasteret's tests compare key output paths against rasterio-style behavior and exercise Arrow/GeoArrow, GeoPandas, TorchGeo, COG reading, cloud routing, and catalog builds. Network-dependent tests are separated from the unit suite and auto-skip when optional credentials or extras are missing.

If you find an interop case where Rasteret output differs from the source tool semantics, please [open an issue](https://github.com/terrafloww/rasteret/issues).
