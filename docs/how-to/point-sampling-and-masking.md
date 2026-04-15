# Point Sampling And Masking

Use this page when you want pixel values at point locations, or when you want to
control how polygon edges are masked for chip/area reads.

`Collection.sample_points(...)` returns a table with one row per sampled point,
record, and band. The usual workflow is:

```text
collection -> filter metadata -> pass points -> receive a sample table
```

The examples below use the public AlphaEarth Foundation collection:

```python
import rasteret

collection = rasteret.load("aef/v1-annual").subset(
    bbox=(11.3, -0.002, 11.5, 0.001),
    date_range=("2023-01-01", "2023-12-31"),
)
```

## Point Tables

For large point jobs, keep your points in the table tool you already use.
GeoPandas, DuckDB, Polars, PyArrow, and SedonaDB can all hand tables to
Rasteret.

If your point table has `x/y`, `lon/lat`, `longitude/latitude`, or `lng/lat`
columns, Rasteret can infer the coordinate columns. You can still pass
`x_column` and `y_column` explicitly when that is clearer.

### Coordinate Columns

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

Plain coordinate columns do not say which CRS they use, so pass `geometry_crs`.

### DuckDB

DuckDB query results can be passed directly:

```python
import duckdb

con = duckdb.connect()
con.register("point_source", points)

points = con.sql("""
    SELECT sensor_id, plot_id, lon, lat
    FROM point_source
    WHERE lon BETWEEN 11.3 AND 11.5
      AND lat BETWEEN -0.002 AND 0.001
""")

samples = collection.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["A00", "A01"],
    geometry_crs=4326,
)
```

No CSV or temporary file writes are required for this pattern.

### Geometry Columns

If your upstream tool already has a point geometry column, pass
`geometry_column=...` instead of `x_column` and `y_column`.

```python
import geopandas as gpd
from shapely.geometry import Point

point_gdf = gpd.GeoDataFrame(
    {
        "plot_id": ["plot-a", "plot-b"],
        "sensor_id": ["s-001", "s-002"],
    },
    geometry=[Point(11.36, -0.001), Point(11.44, -0.0005)],
    crs="OGC:CRS84",
)

point_table = point_gdf.to_arrow(geometry_encoding="WKB")

samples = collection.sample_points(
    points=point_table,
    geometry_column="geometry",
    bands=["A00"],
)
```

If the geometry column includes CRS information, Rasteret uses it. If it does
not, pass `geometry_crs=...`.

For DuckDB `GEOMETRY` values, convert them before passing the result to
Rasteret:

```sql
SELECT plot_id, ST_AsWKB(geom) AS geometry
FROM plot_points
```

### Output Columns

`sample_points()` returns a table with:

- `point_index`, `point_x`, `point_y`, `point_crs`
- `record_id`, `datetime`, `collection`
- `band`, `value`, `raster_crs`
- `neighbourhood_values` when `return_neighbourhood!="off"`

It also keeps point-table columns such as `plot_id`, `sensor_id`, or `split`,
unless they collide with Rasteret output column names.

Use `match="all"` for time-series style output with every matching record per
point and band:

```python
samples = collection.sample_points(
    points=points,
    bands=["A00"],
    geometry_crs=4326,
    match="all",
)
```

Use `match="latest"` to keep one row per `(point_index, band)`:

```python
latest = collection.sample_points(
    points=points,
    bands=["A00"],
    geometry_crs=4326,
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
    bands=["A00"],
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
    bands=["A00"],
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
bytes, GeoJSON points, or lists of those values. For larger jobs, prefer table
inputs.

## Polygon Masking

For area/chip reads, `all_touched` controls whether pixels touched by polygon
edges are included:

```python
arr = collection.get_numpy(
    geometries=(11.35, -0.0018, 11.38, -0.0008),
    bands=["A00"],
    all_touched=False,  # rasterio default
)
```

Set `all_touched=True` when edge pixels should be included whenever the polygon
touches them.

## CRS Notes

- `geometry_crs` describes your input points, not the raster CRS.
- Some geometry columns include CRS information, and Rasteret uses it.
- Plain coordinate columns and geometry columns with no CRS need
  `geometry_crs=...`.
- Output includes both `point_crs` and `raster_crs`.
- For multi-zone or time-series workloads, keep results as a table while you
  group or filter them.
