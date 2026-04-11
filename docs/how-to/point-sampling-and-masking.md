# Point Sampling And Masking

Use this page when you want pixel values at point locations, or when you want to
control how polygon edges are masked for chip/area reads.

`Collection.sample_points(...)` returns a `pyarrow.Table` with one row per
sampled point, record, and band. The usual workflow is:

```text
collection -> filter metadata -> pass points -> receive Arrow samples
```

## Point Tables

For large point jobs, keep your points in the table engine you already use and
pass that object directly to Rasteret. Rasteret accepts Arrow tables, pandas or
GeoPandas dataframes, Polars dataframes, and DuckDB/SedonaDB-style relations.

If your point table has `x/y`, `lon/lat`, `longitude/latitude`, or `lng/lat`
columns, Rasteret can infer the coordinate columns. You can still pass
`x_column` and `y_column` explicitly when that is clearer.

### Polars

```python
import polars as pl

points = pl.DataFrame(
    {
        "plot_id": ["plot-a", "plot-b"],
        "lon": [-122.40, -122.39],
        "lat": [37.79, 37.80],
    }
)

samples = collection.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["B04", "B08"],
    geometry_crs=4326,
)
```

### DuckDB

```python
import duckdb

con = duckdb.connect()

points = con.sql("""
    SELECT plot_id, lon, lat
    FROM plot_points
    WHERE lon BETWEEN -122.5 AND -122.3
      AND lat BETWEEN 37.7 AND 37.9
""")

samples = collection.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["B04", "B08"],
    geometry_crs=4326,
)
```

No CSV, pandas dataframe, or temporary file writes are required for this pattern.

### Geometry Columns

If your upstream tool already has a point geometry column, pass
`geometry_column=...` instead of `x_column` and `y_column`.

```python
samples = collection.sample_points(
    points=point_table,
    geometry_column="geometry",
    bands=["B04"],
    geometry_crs=4326,
)
```

The geometry column should be WKB or GeoArrow points. For DuckDB `GEOMETRY`
values, convert them in SQL first:

```sql
SELECT plot_id, ST_AsWKB(geom) AS geometry
FROM plot_points
```

### Output Columns

`sample_points()` returns an Arrow table with:

- `point_index`, `point_x`, `point_y`, `point_crs`
- `record_id`, `datetime`, `collection`, `cloud_cover`
- `band`, `value`, `raster_crs`
- `neighbourhood_values` when `return_neighbourhood!="off"`

`cloud_cover` may be null when the source collection does not provide it.

Use `match="all"` for time-series style output with every matching record per
point and band:

```python
samples = collection.sample_points(
    points=points,
    bands=["B04"],
    match="all",
)
```

Use `match="latest"` to keep one row per `(point_index, band)`:

```python
latest = collection.sample_points(
    points=points,
    bands=["B04"],
    match="latest",
)
```

### Nodata Fallback

By default, `sample_points()` samples the base pixel containing the point:

- If the base pixel is valid, Rasteret returns it.
- If the base pixel is nodata/NaN, Rasteret returns that nodata value.

Set `max_distance_pixels > 0` to search nearby pixels when the base pixel is
nodata:

```python
samples = collection.sample_points(
    points=points,
    bands=["B04"],
    geometry_crs=4326,
    max_distance_pixels=2,  # search up to a 5x5 window
)
```

The distance is measured in square rings around the base pixel. Rasteret uses
Chebyshev distance for the ring limit and picks the closest valid candidate by
point-to-pixel-rectangle distance.

If you also want the searched window in the result, use
`return_neighbourhood`:

```python
samples = collection.sample_points(
    points=points,
    bands=["B04"],
    geometry_crs=4326,
    max_distance_pixels=2,
    return_neighbourhood="if_center_nodata",
)
```

Options:

| Value | Behavior |
| --- | --- |
| `"off"` | Do not include `neighbourhood_values`. |
| `"always"` | Include the searched window for every sampled row. |
| `"if_center_nodata"` | Include the searched window only when the base pixel is nodata/NaN. |

`neighbourhood_values` is a row-major list with length `(2r + 1)^2`, where `r`
is `max_distance_pixels`.

### Convenience Inputs

For quick scripts, `sample_points()` can also accept Shapely points, WKB point
bytes, GeoJSON points, or lists of those values. For larger jobs, prefer Arrow inputs.

## Polygon Masking

For area/chip reads, `all_touched` controls whether pixels touched by polygon
edges are included:

```python
arr = collection.get_numpy(
    geometries=(-122.42, 37.76, -122.36, 37.82),
    bands=["B04"],
    all_touched=False,  # rasterio default
)
```

Set `all_touched=True` when edge pixels should be included whenever the polygon
touches them.

## CRS Notes

- `geometry_crs` describes the input points, not the raster CRS.
- `geometry_crs` defaults to EPSG:4326.
- Output includes both `point_crs` and `raster_crs`.
- For multi-zone or time-series workloads, keep results in Arrow and group or
  filter before converting to formats that only support one CRS at a time.
