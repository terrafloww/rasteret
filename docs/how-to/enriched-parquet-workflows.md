# Enriched Parquet for Reproducible Experiments

Rasteret's Parquet index is extensible. You can add columns (AOI polygons,
train/val/test splits, labels, quality flags) and query them later with
standard Arrow-compatible tools. Everything stays in one file, making
experiments reproducible and shareable.

The pattern:

1. **Rasteret builds** the index (scene metadata + COG header cache)
2. **You enrich** it with experiment-specific columns
3. **Arrow tools filter** the enriched Parquet (DuckDB, PyArrow, GeoPandas)
4. **Rasteret fetches** COG pixels for the filtered subset

All data flows through Arrow tables. No conversion to Python lists.

## 1. Build and enrich

```python
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely.geometry import Polygon

import rasteret

# Build collection from STAC
collection = rasteret.build_from_stac(
    name="bangalore",
    stac_api="https://earth-search.aws.element84.com/v1",
    collection="sentinel-2-l2a",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
    workspace_dir=Path.home() / "rasteret_workspace",
)

# Get the Arrow table
enriched = collection.dataset.to_table()
n = enriched.num_rows
```

### Add an AOI column

Store the AOI polygon as WKB binary. Parquet dictionary-encodes repeated
values, so storing the same AOI on every row costs almost nothing.

```python
aoi = Polygon([
    (77.55, 13.01), (77.58, 13.01),
    (77.58, 13.08), (77.55, 13.08),
    (77.55, 13.01),
])

# shapely -> WKB bytes -> Arrow binary column (no Python loop)
aoi_wkb = shapely.to_wkb(aoi)
enriched = enriched.append_column(
    "aoi", pa.array([aoi_wkb] * n, type=pa.binary())
)
```

### Add splits and labels

```python
rng = np.random.default_rng(42)
splits = rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.15, 0.15])
enriched = enriched.append_column("split", pa.array(splits))

labels = rng.integers(0, 5, size=n)
enriched = enriched.append_column("label", pa.array(labels, type=pa.int32()))
```

### Save

```python
pq.write_table(enriched, "./experiment_v1.parquet")
collection = rasteret.load("./experiment_v1.parquet")
```

The enriched Parquet now contains scene metadata, COG tile cache, AOI
geometry, splits, and labels in a single portable Collection.

---

## 2. Query with DuckDB

Reload the collection later and query with DuckDB. DuckDB reads Arrow
tables directly with zero copy.

```python
import duckdb

import rasteret

collection = rasteret.load("./experiment_v1.parquet")
enriched = collection.dataset.to_table()

con = duckdb.connect()
result = con.sql("""
    SELECT DISTINCT aoi
    FROM enriched
    WHERE split = 'train' AND "eo:cloud_cover" < 15
""").fetch_arrow_table()

# Pass the Arrow WKB column directly to Rasteret (no Shapely needed)
aoi_col = result.column("aoi")
```

The Arrow column goes straight to Rasteret -- no conversion required:

```python
train = collection.subset(split="train", cloud_cover_lt=15)
ds = train.get_xarray(geometries=aoi_col, bands=["B04", "B08"])
ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04)
```

### More DuckDB queries

```python
# Monthly breakdown by split
con.sql("""
    SELECT split,
           month(datetime) AS mo,
           count(*) AS scenes,
           round(avg("eo:cloud_cover"), 1) AS avg_cloud
    FROM enriched
    GROUP BY split, mo
    ORDER BY split, mo
""").show()

# Scenes per AOI (when using multiple AOIs)
con.sql("""
    SELECT aoi, split, count(*) AS scenes
    FROM enriched
    GROUP BY aoi, split
""").show()
```

---

## 3. Query with PyArrow only

If you prefer no extra dependencies, PyArrow compute works on the same
Arrow table:

```python
import pyarrow.compute as pc

import rasteret

collection = rasteret.load("./experiment_v1.parquet")
enriched = collection.dataset.to_table()

# Filter: train split, low cloud
mask = pc.and_(
    pc.equal(enriched.column("split"), "train"),
    pc.less(enriched.column("eo:cloud_cover"), 15.0),
)
filtered = enriched.filter(mask)

# Deduplicate AOIs in Arrow, pass directly to Rasteret
unique_wkb = filtered.column("aoi").unique()

# Fetch -- Arrow WKB column goes directly, no Shapely conversion
train = collection.subset(split="train", cloud_cover_lt=15)
ds = train.get_xarray(geometries=unique_wkb, bands=["B04", "B03", "B02"])
```

---

## 4. Query with GeoPandas

GeoPandas reads WKB columns from Arrow and gives you spatial operations
(intersection, buffer, distance) for free:

```python
import geopandas as gpd

import rasteret

collection = rasteret.load("./experiment_v1.parquet")
enriched = collection.dataset.to_table()

gdf = gpd.GeoDataFrame(
    enriched.to_pandas(),
    geometry=gpd.GeoSeries.from_wkb(enriched.column("aoi").to_pandas()),
)

# Spatial query: scenes whose AOI intersects a new region
from shapely.geometry import box
new_region = box(77.55, 13.01, 77.60, 13.05)
hits = gdf[gdf.intersects(new_region)]
aois = list(hits.geometry.unique())
```

---

## Multiple AOIs per experiment

When an experiment uses several AOIs, assign each scene to the AOI it
belongs to. Parquet compresses repeated WKB values efficiently.

```python
import shapely
from shapely.geometry import Polygon

aoi_north = Polygon([(77.55, 13.05), (77.58, 13.05), (77.58, 13.08), (77.55, 13.08), (77.55, 13.05)])
aoi_south = Polygon([(77.55, 12.95), (77.58, 12.95), (77.58, 13.00), (77.55, 13.00), (77.55, 12.95)])

# Scene footprints from the existing geometry column
footprints = shapely.from_wkb(
    enriched.column("geometry").to_numpy(zero_copy_only=False)
)

# Assign AOI based on intersection
aoi_assignments = []
for fp in footprints:
    if shapely.intersects(fp, aoi_north):
        aoi_assignments.append(shapely.to_wkb(aoi_north))
    elif shapely.intersects(fp, aoi_south):
        aoi_assignments.append(shapely.to_wkb(aoi_south))
    else:
        aoi_assignments.append(None)

enriched = enriched.append_column(
    "aoi", pa.array(aoi_assignments, type=pa.binary())
)
```

---

## Summary

| Step | Tool | Arrow-native? |
|------|------|---------------|
| Build index | `rasteret.build_from_stac()` | Arrow dataset output |
| Add columns | `pa.Table.append_column()` | Yes |
| Export | `collection.export()` | Yes |
| Query | DuckDB / `pyarrow.compute` / GeoPandas | Yes (zero-copy reads) |
| Fetch pixels | `collection.get_xarray(...)` or `collection.get_numpy(...)` | Yes -- Arrow WKB/GeoArrow direct |

Rasteret accepts Arrow columns, WKB bytes, Shapely geometries, bbox tuples,
and GeoJSON dicts. Arrow columns are the zero-copy preferred path.

Rasteret builds the index. You own the enrichment. Arrow tools query it.
Rasteret fetches the pixels.

## Major TOM-style enrichment

Use this pattern when you want Major TOM-like metadata ergonomics but keep
image bytes in source Sentinel-2 COGs. Parquet stays the control plane;
Rasteret reads pixels directly from remote COGs.

This gives you:

- Major TOM-style keys in your Rasteret index (`major_tom_product_id`, `major_tom_grid_cell`)
- Deterministic `train/val/test` split column at index level
- Arrow-native geometry -> `get_numpy()` retrieval path (no per-row Python geometry conversion)

### Build a Major TOM-style index from Sentinel-2

Install the helper once:

```bash
pip install git+https://github.com/ESA-PhiLab/Major-TOM
```

Run the example script:

```bash
python examples/major_tom_on_the_fly_collection.py \
  --name major-tom-on-the-fly \
  --bbox -122.55 37.65 -122.30 37.90 \
  --date-range 2024-01-01 2024-02-01 \
  --samples 24 \
  --bands B02 B08
```

What it does:

1. Builds a Sentinel-2 Collection from the catalog
2. Uses `majortom.grid.Grid` on scene centroids to derive `major_tom_grid_cell`
3. Uses STAC `s2:product_uri` (without `.SAFE`) as `major_tom_product_id`
4. Adds deterministic split labels from `grid_cell`
5. Constructs per-scene patch geometries from scene centers, chip size, and resolution
6. Fetches chips with `Collection.get_numpy()` scene-batched via Arrow WKB geometry

### Retrieve pixels from Arrow geometry

Retrieval is scene-batched: patches are grouped by `major_tom_product_id`,
and each scene's patches are fetched in one `get_numpy()` call with an
Arrow WKB geometry array:

```python
subset = collection.subset(split="train")
scene_view = subset.where(ds.field("major_tom_product_id") == product_id)
arr = scene_view.get_numpy(geometries=patch_wkb_array, bands=["B02", "B08"])
```

### Throughput vs HF best-practice baseline

Measured on **February 26, 2026** with matched Major TOM keys:

| Patches | HF `datasets` parquet filters (best-practice) | Rasteret index+COG | Speedup |
|---:|---:|---:|---:|
| 120 | 46.83 s | 12.09 s | 3.88x |
| 1000 | 771.59 s | 118.69 s | 6.50x |

Baseline method: Hugging Face `datasets.load_dataset(...)` with Parquet filters
(PyArrow-backed) for keyed retrieval. This is a stronger baseline than the
streaming-generator style commonly used in Major TOM exploration notebooks.
