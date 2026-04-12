# Enriched Collection Workflows

Rasteret Collections are Arrow/Parquet metadata tables. That means you can add
experiment metadata into Collection and do lots of planning/filtering on the raster records without moving the pixels
of the original COGs.

Use this pattern for:

- train/validation/test splits
- labels and quality flags
- AI model output IDs
- audit columns for reproducibility
or anything else custom to you and related to your images!

The key pattern: keep the read-ready Rasteret columns intact. Add your own columns next to them.

## Start From A Read-Ready Collection

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="bangalore",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
)
```

The collection can be handed directly to Arrow-aware tools. Polars can consume
Rasteret's Arrow interface without an intermediate file or pandas dataframe.

## Add Splits And Labels With Polars

```python
import numpy as np
import polars as pl

frame = pl.from_arrow(collection)
n = frame.height
rng = np.random.default_rng(42)

frame = frame.with_columns(
    pl.Series(
        "split",
        rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.15, 0.15]),
    ),
    pl.Series("label", rng.integers(0, 5, size=n), dtype=pl.Int32),
)
```

Wrap the enriched table back as a collection:

```python
experiment = rasteret.as_collection(
    frame,
    name="bangalore-experiment-v1",
    data_source=collection.data_source,
)
```

Passing `data_source` preserves source-specific band mapping and avoids relying
on schema metadata that table engines may not round-trip exactly.

## Join External GIS Data With DuckDB

A common workflow starts with labels or plots in a GeoJSON, Shapefile,
FlatGeoBuf, or GeoPackage. Load it with GeoPandas, pass it to DuckDB through
Arrow, join it with Rasteret collection metadata, and wrap the result back as a
collection.

```python
import duckdb
import geopandas as gpd
import rasteret

# Example custom data: known farm plots, field visits, labels, or AOIs.
plots_gdf = gpd.read_file("path/to/plots.geojson").to_crs("OGC:CRS84")
plots = plots_gdf.to_arrow(geometry_encoding="WKB")

con = duckdb.connect()
con.sql("INSTALL spatial; LOAD spatial;")
con.register("my_cog_collection", experiment)
con.register("plots", plots)

joined = con.sql("""
    SELECT
        my_cog_collection.*,
        plots.plot_id,
        plots.crop,
        plots.geometry AS plot_geometry
    FROM my_cog_collection, plots
    WHERE my_cog_collection."eo:cloud_cover" < 10
      AND ST_Intersects(
          my_cog_collection.geometry,
          plots.geometry
      )
""")

plot_experiment = rasteret.as_collection(
    joined,
    data_source=experiment.data_source,
)
```

The `plots` file is your own external GIS data. GeoPandas reads it, `to_arrow()`
hands it to DuckDB without a CSV/export step, and the SQL query keeps the
Rasteret collection columns while adding label and plot geometry columns next
to them. The result stays read-ready because the query selected
`my_cog_collection.*`.

## Read Pixels From The Enriched Collection

Use the added columns for filtering or AOIs:

```python
train = plot_experiment.subset(split="train")
plot_geometries = train.to_table(columns=["plot_geometry"]).column("plot_geometry")

arr = train.get_numpy(
    geometries=plot_geometries,
    bands=["B04", "B08"],
)
```

## Use Splits And Labels With TorchGeo

If the enriched collection has `split` and `label` columns, pass them into the
TorchGeo adapter:

```python
train_dataset = experiment.to_torchgeo_dataset(
    bands=["B02", "B03", "B04", "B08"],
    split="train",
    label_field="label",
    chip_size=256,
)

val_dataset = experiment.to_torchgeo_dataset(
    bands=["B02", "B03", "B04", "B08"],
    split="val",
    label_field="label",
    chip_size=256,
)
```

Everything after dataset creation is standard TorchGeo and PyTorch.

Point sampling works the same way, use a geometry column that contains points:

```python
samples = train.sample_points(
    points=train.to_table(columns=["plot_center_point"]),
    geometry_column="plot_center_point",
    bands=["B04"],
    geometry_crs=4326,
)
```

## Export The Enriched Collection

```python
experiment.export("./bangalore_experiment_v1")
reloaded = rasteret.load("./bangalore_experiment_v1")
```

The exported collection contains metadata, labels, splits, assets, and COG
header metadata. Pixel bytes still live in the source COGs.

## Query With PyArrow Only

You do not need DuckDB for simple filters, you can use the subset method:

```python
import rasteret

# Load the collection
collection = rasteret.load("bangalore-experiment-v1")

# Filter the collection
train = collection.subset(split="train", cloud_cover_lt=15.0)
```

## Notes

- `as_collection(...)` is for tables that are already read-ready Rasteret
  Collections.
- `build_from_table(...)` is for first-time external record tables that still
  need normalization or COG enrichment.
- Keep Rasteret's required columns when you use SQL or dataframe tools.
- Keep geometry columns in WKB/GeoArrow form when possible.
- If your SQL engine drops schema metadata, pass `data_source=...` explicitly
  when wrapping the table back with `as_collection(...)`.
