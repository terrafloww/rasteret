# Bring Your Own AOIs, Points, And Metadata

Use this page when you already have business data and want imagery for it: farm plots, store locations, field visits, labels, train/validation/test splits, model outputs, or any other table with rows you care about.

AOI means **area of interest**. In practice, an AOI is usually a field boundary, parcel, fire perimeter, district, grid cell, or other polygon.

The practical rule is:

```text
Rasteret collection = raster records and COG metadata
Your AOI/point table = geometries, labels, IDs, splits, and business columns
```

Use the tool that already fits your data: GeoPandas, DuckDB, Polars, PyArrow, SedonaDB, or another table tool. When it is time to read pixels, pass that table to Rasteret.

## Start With A Collection

The examples below use the public AlphaEarth Foundation collection because it is already read-ready:

```python
import rasteret

collection = rasteret.load("aef/v1-annual").subset(
    bbox=(11.3, -0.002, 11.5, 0.001),
    date_range=("2023-01-01", "2023-12-31"),
)
```

For your own data, use any collection built with `rasteret.build(...)`, `rasteret.build_from_table(...)`, or reopened with `rasteret.load(...)`.

## Polygon AOIs From GeoPandas

If your polygons are in GeoPandas, call `to_arrow()` and pass the result to Rasteret. Columns such as `plot_id`, `crop`, and `split` come back in the `get_gdf(...)` result.

```python
import geopandas as gpd
from shapely.geometry import box

plots = gpd.GeoDataFrame(
    {
        "plot_id": ["plot-a", "plot-b"],
        "crop": ["cassava", "maize"],
        "split": ["train", "val"],
    },
    geometry=[
        box(11.35, -0.0018, 11.38, -0.0008),
        box(11.42, -0.0012, 11.46, -0.0002),
    ],
    crs="OGC:CRS84",
)

aoi_table = plots.to_arrow(geometry_encoding="WKB")

gdf = collection.get_gdf(
    geometries=aoi_table,
    geometry_column="geometry",
    bands=["A00", "A01"],
)
```

The returned GeoDataFrame includes Rasteret result columns plus the AOI metadata columns such as `plot_id`, `crop`, and `split`.

## Point Tables

For point sampling, pass a table with coordinate columns and the columns you want to keep. Plain `x/y` or `lon/lat` columns do not say which CRS they use, so pass `geometry_crs`.

```python
import pyarrow as pa

points = pa.table(
    {
        "sensor_id": ["s-001", "s-002"],
        "plot_id": ["plot-a", "plot-b"],
        "lon": [11.36, 11.44],
        "lat": [-0.001, -0.0005],
    }
)

samples = collection.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["A00", "A01"],
    geometry_crs=4326,
)
```

The result includes the sample values, raster metadata, and the point metadata columns such as `sensor_id`, `plot_id`, `lon`, and `lat`.

## DuckDB Spatial Joins

Use DuckDB when you want SQL joins before pixel reads. Copy the `ST_GeomFromWKB(...)` pattern below when joining Rasteret footprint geometry with a geometry column from another table.

```python
import duckdb

con = duckdb.connect()
con.sql("INSTALL spatial; LOAD spatial;")
con.register("records", collection)
con.register("plots", aoi_table)

matched_plots = con.sql("""
    SELECT
        plots.plot_id,
        plots.crop,
        plots.split,
        plots.geometry AS plot_geometry
    FROM records, plots
    WHERE ST_Intersects(
        ST_GeomFromWKB(records.geometry),
        ST_GeomFromWKB(plots.geometry)
    )
""")

gdf = collection.get_gdf(
    geometries=matched_plots,
    geometry_column="plot_geometry",
    geometry_crs=4326,
    bands=["A00"],
)
```

The DuckDB query can be passed directly to Rasteret. `geometry_crs=4326` is included because the SQL result no longer says which CRS the geometry uses.

## Add Metadata To Collection Rows

Sometimes your new columns describe the image rows themselves rather than the plots or points. Examples include scene labels, split assignments, quality flags, and model version IDs. In that case, add columns to the collection table and wrap it back as a collection.

```python
import pyarrow as pa

table = collection.to_table()
table = table.append_column("split", pa.array(["train"] * table.num_rows))
table = table.append_column(
    "experiment_id",
    pa.array(["aef-demo"] * table.num_rows),
)

experiment = rasteret.as_collection(
    table,
    name="aef-demo-experiment",
    data_source=collection.data_source,
)

train = experiment.subset(split="train")
```

Use `as_collection(...)` only for tables that still describe Rasteret image records. For ordinary plot, parcel, sensor, or label tables, pass the table directly to `get_gdf(...)` or `sample_points(...)`.

## Choose The Output Surface

| Method                     | What happens to your columns                                          |
| -------------------------- | --------------------------------------------------------------------- |
| `get_gdf(...)`             | Keeps AOI columns such as `plot_id`, `crop`, and `split`.             |
| `sample_points(...)`       | Keeps point columns such as `sensor_id`, `plot_id`, `lon`, and `lat`. |
| `get_numpy(...)`           | Reads the pixels, but returns only arrays.                            |
| `get_xarray(...)`          | Reads the pixels, but returns an xarray Dataset.                      |
| `to_torchgeo_dataset(...)` | Uses collection columns such as `split` and `label_field`.            |

Use `get_gdf(...)` or `sample_points(...)` when labels, IDs, folds, or audit columns need to stay attached to the pixel results.

## CRS Rules

- Rasteret collection footprints use longitude/latitude.
- Rasters can still have their own native CRS. Rasteret handles that internally.
- Your AOIs and points can be in longitude/latitude, UTM, or another CRS.
- If the geometry column includes CRS information, Rasteret uses it.
- If your table only has `lon` / `lat` columns, or a geometry column with no CRS information, pass `geometry_crs=...`.

Rasteret fails when it cannot know the input CRS. That error is useful: a silent CRS guess can return pixels from the wrong place.

## Column Names

Some column names are used by Rasteret outputs, such as `record_id`, `datetime`, `band`, `data`, `value`, `geometry_id`, and `point_index`. If your input table uses one of those names for something else, rename it before reading.

For example:

```python
safe_points = points.rename_columns(
    ["sensor_id", "input_plot_id", "lon", "lat"]
)
```

Rasteret raises an error for these collisions instead of silently renaming or overwriting your columns.
