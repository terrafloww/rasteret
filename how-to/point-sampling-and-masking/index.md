# Point Sampling and Masking

Use this guide when you need either:

- per-point values (tabular output), or
- polygon/bbox chip reads with explicit mask behavior.

## Sample values at points (standard Arrow-native workflow)

`Collection.sample_points(...)` returns a `pyarrow.Table` with one row per sampled value.

Recommended pattern:

1. Keep points in your current engine (DuckDB / Polars / pandas).
1. Select `x`/`y` columns.
1. Pass the table directly to `sample_points(...)` with column names.

If your columns are named `x/y`, `lon/lat`, `longitude/latitude`, or `lng/lat`, you can omit `x_column` and `y_column`.

Collection-centric workflow reminder: `build/load/as_collection -> subset/where -> sample_points`.

Why `points=`? `sample_points()` is a point-value query API, so it uses `points=...`. Area/chip read APIs (`get_xarray()`, `get_numpy()`, `get_gdf()`) use `geometries=...`.

### DuckDB -> Rasteret

```python
import duckdb
tbl = duckdb.sql("""
    SELECT lon, lat
    FROM my_points
    WHERE lon BETWEEN -122.5 AND -122.3
      AND lat BETWEEN 37.7 AND 37.9
""").arrow().read_all()

samples = collection.sample_points(
    points=tbl,
    x_column="lon",
    y_column="lat",
    bands=["B04", "B08"],
    geometry_crs=4326,
)
```

### SedonaDB -> Rasteret

```python
import pandas as pd
import sedonadb

ctx = sedonadb.connect()
ctx.create_data_frame(
    pd.DataFrame({"lon": [-122.39, -122.395], "lat": [37.79, 37.795]})
).to_view("pts")

tbl = ctx.sql("SELECT lon, lat FROM pts").to_arrow_table()
samples = collection.sample_points(
    points=tbl,
    x_column="lon",
    y_column="lat",
    bands=["B04"],
    geometry_crs=4326,
)
```

### Polars -> Rasteret

```python
import polars as pl
df = pl.read_parquet("points.parquet").select(["lon", "lat"])
samples = collection.sample_points(
    points=df,
    x_column="lon",
    y_column="lat",
    bands=["B04"],
    geometry_crs=4326,
)
```

### pandas -> Rasteret

```python
import pandas as pd
df = pd.read_parquet("points.parquet", columns=["lon", "lat"])
samples = collection.sample_points(
    points=df,
    x_column="lon",
    y_column="lat",
    bands=["B04"],
    geometry_crs=4326,
)
```

This is the standard easy way: use it as part of Arrow-native tools, then pass the point column to Rasteret.

If your upstream tool already has a WKB/GeoArrow point column, pass the table with `geometry_column="..."` (no conversion step required). For DuckDB, use `ST_AsWKB(geom)` in SQL when starting from a `GEOMETRY` column.

### Output columns

`sample_points()` returns:

- `point_index`, `point_x`, `point_y`, `point_crs`
- `record_id`, `datetime`, `collection`, `cloud_cover`
- `band`, `value`, `raster_crs`

This is intentionally Arrow-first, so you can keep processing in PyArrow, DuckDB, Polars, SedonaDB and geopandas.

### `match="all"` vs `match="latest"`

- `match="all"`: keep all matching records per point/band (time series style).
- `match="latest"`: keep only the latest datetime per `(point_index, band)`.

## Convenience inputs (quick scripts)

`sample_points()` also accepts convenience formats (Shapely Point, WKB, GeoJSON Point), which are useful for quick scripts. For large-scale point jobs, prefer Arrow-native point arrays (example above).

## Control polygon masking (`all_touched`)

For `get_xarray()`, `get_gdf()`, and `get_numpy()`, you can now pass `all_touched` directly.

```python
arr = collection.get_numpy(
    geometries=(-122.42, 37.76, -122.36, 37.82),
    bands=["B04"],
    all_touched=False,  # rasterio default
)
```

Set `all_touched=True` when you want edge pixels included whenever the geometry touches them.

## CRS notes

- `geometry_crs` must describe your input points.
- Output includes both `point_crs` and `raster_crs` for explicit downstream handling.
- For multi-zone/time-series workloads, keep results in Arrow and group/filter before converting to other formats.
