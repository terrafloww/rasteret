# Concepts

Rasteret is an index-first raster engine for cloud-native tiled GeoTIFFs and COGs. It helps you turn a pile of remote raster assets into a queryable collection that can feed NumPy, xarray, GeoPandas, TorchGeo, and Arrow-native tools.

The collection is the center of the workflow.

## Metadata And Pixels Are Separate

Most raster workflows mix two jobs together:

- discover which files exist and what they contain
- read the pixels needed for a task

Rasteret separates those jobs.

**The collection** is the metadata side. It is an Arrow/Parquet table with rows for raster records and columns such as IDs, timestamps, footprints, bounding boxes, asset URLs, raster CRS sidecars, split labels, and cached COG header metadata.

**The pixels** stay where they already are: in the original cloud or local COGs. Rasteret only reads pixel byte ranges when you call methods such as `get_numpy()`, `get_xarray()`, `get_gdf()`, `sample_points()`, or `to_torchgeo_dataset()`.

That split is why Rasteret can be fast without copying a raster archive into a new format.

## Why Header Metadata Matters

To read a remote tiled TIFF efficiently, a program needs to know things like:

- tile offsets and byte counts
- image width and height
- dtype and nodata
- transform and pixel size
- raster CRS

Traditional code often rediscovers this by opening remote TIFFs during the read loop. That is the cold-start cost Rasteret tries to remove.

During build/enrichment, Rasteret parses the COG headers once and stores the useful header metadata in the collection. Later, when you ask for pixels, Rasteret can plan byte-range reads directly.

## The Collection Is A Table

Because a collection is a table, you can treat metadata as data:

- filter scenes before reading pixels
- keep train/validation/test splits beside the asset metadata
- keep labels, AOI IDs, or quality flags beside the imagery index
- share an exported collection with another environment
- pass collection metadata through tools such as DuckDB, Polars, PyArrow, and GeoPandas

Rasteret still owns the raster read path. Other tools can help with metadata work, joins, and inspection.

You will usually see three kinds of tables:

| Table              | What it describes                                    | Common use                                                  |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------------------- |
| Collection table   | Raster records, assets, footprints, and COG metadata | Build once, filter, share, and reuse.                       |
| AOI or point table | User geometries plus business columns                | Pass plots, sensors, labels, or splits into reads.          |
| Output table       | Pixel results plus preserved metadata                | Continue analysis in GeoPandas, PyArrow, DuckDB, or Polars. |

## CRS In Rasteret

Rasteret stores two CRS ideas that should not be confused:

- `geometry` is the footprint geometry. It is exported as GeoArrow WKB in `OGC:CRS84`.
- `crs` and `proj:epsg` describe the native raster CRS for each row.

The raster CRS is used when Rasteret transforms query geometries into raster space. The footprint CRS is what GeoPandas, DuckDB, and other Arrow consumers should use when reading the `geometry` column.

User AOIs and points may be in another CRS. If Rasteret cannot read the CRS from the geometry column, pass `geometry_crs=...`.

## Read Planning

When you request pixels, Rasteret builds a read plan:

1. Filter the collection to likely records.
1. Convert the requested AOI or points into the needed raster CRS.
1. Find the COG tiles touched by the request.
1. Group tile byte ranges where possible.
1. Fetch and decode those tiles concurrently.
1. Return the result in the output surface you asked for.

You do not need to write the STAC loop, TIFF-header loop, thread pool, and stacking logic yourself.

## Lifecycle

1. **Build**: Start from a catalog entry, STAC search, Parquet/GeoParquet record table, or Arrow table. Rasteret normalizes the record metadata and can parse COG headers.
1. **Inspect and filter**: Use the collection as a table of raster records.
1. **Read**: Choose `get_numpy()`, `get_xarray()`, `get_gdf()`, `sample_points()`, or `to_torchgeo_dataset()`.
1. **Export and reload**: Save the collection so future runs skip the build work.

## Comparison At A Glance

| Concept            | Manual rasterio/GDAL workflow                 | Rasteret workflow         |
| ------------------ | --------------------------------------------- | ------------------------- |
| Discovery          | STAC and file-opening loops in user code      | collection build step     |
| Header parsing     | repeated during reads and in new environments | cached in the collection  |
| Metadata filtering | custom pandas/GeoPandas/STAC glue             | table-first filtering     |
| Labels and splits  | separate files joined by convention           | columns in the collection |
| Pixel reads        | manual windows, threads, stacking             | planned byte-range reads  |

Next: [Migrating from Rasterio](https://terrafloww.github.io/rasteret/how-to/migrating-from-rasterio/index.md)
