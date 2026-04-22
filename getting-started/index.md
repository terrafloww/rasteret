# Getting Started

This guide builds a small Sentinel-2 collection and reads pixels from it. The goal is to show the Rasteret loop you will use in larger projects:

```text
build a collection -> inspect/filter it -> read pixels in the shape you need
```

## Installation

Rasteret requires Python 3.12 or later.

```bash
uv pip install rasteret
```

The base install covers NumPy, xarray, GeoPandas, DuckDB, and point sampling. Install extras only when you need optional integrations:

```bash
uv pip install "rasteret[torchgeo]"  # Collection.to_torchgeo_dataset()
uv pip install "rasteret[aws]"       # AWS helper dependencies
uv pip install "rasteret[azure]"     # Planetary Computer helpers
uv pip install "rasteret[all]"       # all optional integrations for exploration
```

## 1. Build A Collection

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2-bangalore",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-01-31"),
)
```

This works because `earthsearch/sentinel-2-l2a` is in Rasteret's built-in catalog. The build step searches the source STAC API, parses the COG header metadata once, and creates a Rasteret collection.

The important part: Rasteret does **not** move all pixels into Parquet. It keeps the imagery in the source COGs and stores a queryable Arrow/Parquet index with metadata such as IDs, footprints, assets, bounding boxes, CRS sidecars, and per-band COG header metadata.

Why build first?

Opening many remote TIFFs repeatedly is expensive because each environment has to rediscover header metadata such as tile offsets, transforms, dtype, nodata, and CRS. Rasteret pays that setup cost during build, then reuses the collection for later reads.

## 2. Inspect The Collection

```python
collection.describe()
collection.bands
collection.bounds
collection.epsg
```

If the collection came from the built-in catalog, compare what you built against the catalog descriptor:

```python
collection.compare_to_catalog()
```

Useful mental model:

- `geometry` is the scene footprint.
- `crs` / `proj:epsg` describe the native raster CRS for each row.
- `*_metadata` columns store per-band COG header metadata used by the read path.
- Extra columns such as splits, labels, AOI IDs, or experiment metadata can live beside the raster metadata.

## 3. Filter Before Reading

```python
filtered = collection.subset(
    cloud_cover_lt=50,
    bbox=(77.55, 13.01, 77.58, 13.08),
)
```

Filtering works on metadata first. That keeps pixel reads focused on candidate records instead of opening every raster and checking later.

## 4. Read Pixels

Choose the output surface that matches your task.

For xarray analysis:

```python
ds = filtered.get_xarray(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For array-first workflows:

```python
arr = filtered.get_numpy(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

For GeoPandas output:

```python
gdf = filtered.get_gdf(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

If your AOIs already live in GeoPandas, pass them with `to_arrow()` and select the geometry column. Rasteret keeps columns such as `plot_id` and `crop` in `get_gdf(...)` output:

```python
import geopandas as gpd
from shapely.geometry import box

plots = gpd.GeoDataFrame(
    {
        "plot_id": ["plot-a"],
        "crop": ["rice"],
    },
    geometry=[box(77.55, 13.01, 77.58, 13.08)],
    crs="OGC:CRS84",
)

gdf = filtered.get_gdf(
    geometries=plots.to_arrow(geometry_encoding="WKB"),
    geometry_column="geometry",
    bands=["B04", "B08"],
)
```

For TorchGeo pipelines:

```python
dataset = filtered.to_torchgeo_dataset(
    bands=["B04", "B03", "B02", "B08"],
    chip_size=256,
)
```

Everything after `to_torchgeo_dataset()` is standard TorchGeo sampler and DataLoader code.

## 5. Sample Point Values

```python
import pyarrow as pa

points = pa.table(
    {
        "plot_id": ["plot-a", "plot-b"],
        "lon": [77.56, 77.57],
        "lat": [13.03, 13.04],
    }
)

samples = filtered.sample_points(
    points=points,
    x_column="lon",
    y_column="lat",
    bands=["B04", "B08"],
    geometry_crs=4326,
)
```

`samples` is a table with the point values and input columns such as `plot_id`.

## What To Use When

Use `build()` when the dataset is already in Rasteret's catalog.

If it is not in the catalog:

- custom STAC API: use `build_from_stac(...)`
- existing Parquet or GeoParquet record table: use `build_from_table(...)`
- read-ready Arrow table in memory: use `as_collection(...)`
- exported Rasteret collection: use `load(...)`

## Practical Notes

Native dtype is preserved

Rasteret preserves the native COG dtype. If you need floating-point math, cast explicitly or compute with an output surface that fits your workflow.

Share the collection, not a notebook full of setup code

`collection.export("path/")` writes a portable collection artifact. Another user can reopen it with `rasteret.load("path/")`.

CRS still matters

Some geometry columns tell Rasteret their CRS. Plain coordinate columns do not, so pass `geometry_crs=...` for those inputs. Rasteret uses the raster CRS metadata in the collection to transform reads safely.

## Next

Next: [Concepts](https://terrafloww.github.io/rasteret/explanation/concepts/index.md)
