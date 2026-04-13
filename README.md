<h1 align="center">Rasteret</h1>

<p align="center">
  <strong>Build a collection once. Query it like a table. Read pixels 20x faster from cloud COGs.</strong>
</p>

<p align="center">
  <a href="https://terrafloww.github.io/rasteret"><img src="https://img.shields.io/badge/docs-terrafloww.github.io%2Frasteret-009DD1" alt="Documentation"></a>
  <a href="https://discord.gg/86NgTB3Xa"><img src="https://img.shields.io/badge/Discord-chat-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://pypi.org/project/rasteret/"><img src="https://img.shields.io/pypi/v/rasteret?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/rasteret/"><img src="https://img.shields.io/pypi/pyversions/rasteret" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

Rasteret is an index-first reader for cloud-hosted tiled GeoTIFFs and COGs. It
builds a queryable Arrow/Parquet collection with scene metadata, asset URLs, CRS
sidecars, and parsed COG header metadata. Pixels stay in the original COGs.

After that, you can filter, join, and enrich the collection as a table, then
read only the pixels you need into NumPy, xarray, GeoPandas, TorchGeo, or Arrow
point-sample tables.

```text
STAC / Parquet / Arrow table -> Rasteret Collection -> NumPy / xarray / GeoPandas / TorchGeo
external labels / plots / points    filter/join/share          read pixels on demand
```

## Why Rasteret

Remote raster workflows often repeat the same setup work: STAC loops, COG header
parsing, tile byte-range planning, CRS transforms, retries, and output assembly.

Rasteret moves the expensive raster metadata discovery into a `Collection` build
step and reuses that metadata for later reads.

That helps when you:

- train or evaluate models over many remote COG scenes
- repeatedly sample the same imagery with different AOIs, points, labels, or splits
- avoid rediscovering raster header metadata in new notebooks, containers, or machines
- want one source collection to feed TorchGeo, xarray, NumPy, GeoPandas, and Arrow tools
- need DuckDB, Polars, PyArrow, or GeoPandas to work on metadata and external
  geometries before pixel reads

## Quick Example

```python
import rasteret

sentinel2_collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2_bangalore",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
)

clear = sentinel2_collection.subset(cloud_cover_lt=20)

arr = clear.get_numpy(
    geometries=(77.55, 13.01, 77.58, 13.08),
    bands=["B04", "B08"],
)
```

The same collection can feed TorchGeo:

```python
dataset = clear.to_torchgeo_dataset(
    bands=["B04", "B03", "B02", "B08"],
    chip_size=256,
)
```

## Bring Your Own Geometry And Metadata

Rasteret collections are Arrow tables. That means external labels, farm plots,
asset locations, fire boundaries, or point samples can be joined to the
collection before any pixels are read. Rasteret then uses the enriched table and
the requested geometries to find the right COGs and fetch the needed pixels.

```python
import duckdb
import geopandas as gpd
import rasteret

plots = gpd.read_file("plots.geojson").to_crs("OGC:CRS84")
plots_arrow = plots.to_arrow(geometry_encoding="WKB")

con = duckdb.connect()
con.sql("INSTALL spatial; LOAD spatial;")
con.register("sen2_rasteret", sentinel2_collection)
con.register("plots", plots_arrow)

#bring your own geometries
plot_scenes = con.sql("""
    SELECT
        sen2_rasteret.*,
        plots.plot_id,
        plots.crop,
        ST_AsWKB(plots.geometry) AS plot_geometry
    FROM sen2_rasteret, plots
    WHERE sen2_rasteret."eo:cloud_cover" < 10
      AND ST_Intersects(sen2_rasteret.geometry, plots.geometry)
""")

# convert enriched table to rasteret collection
plot_collection = rasteret.as_collection(
    plot_scenes,
)

plot_geometries = plot_collection.to_table(columns=["plot_geometry"])["plot_geometry"]
patches = plot_collection.get_numpy(
    geometries=plot_geometries,
    bands=["B04", "B08"],
)
```

The same pattern works with Polars or PyArrow for split/label columns, and with
`sample_points(...)` when your external data is point-based.

## What You Can Do

| Task | Rasteret surface |
| --- | --- |
| Build from a registered dataset | `rasteret.build("catalog/id", ...)` |
| Build from your own Parquet, GeoParquet, DuckDB, Polars, or Arrow record table | `rasteret.build_from_table(...)` |
| Reopen a saved Collection | `rasteret.load(path_or_dataset_id)` |
| Re-wrap a read-ready Arrow object | `rasteret.as_collection(...)` |
| Get numpy arrays | `Collection.get_numpy(...)` |
| Get xarray dataset | `Collection.get_xarray(...)` |
| Get GeoPandas rows with pixel arrays | `Collection.get_gdf(...)` |
| Sample pixels at points | `Collection.sample_points(...)` |
| Train/infer with TorchGeo | `Collection.to_torchgeo_dataset(...)` |

## Performance

Rasteret is 10x to 20x faster than rasterio/GDAL

| Scenario | TorchGeo/rasterio | Rasteret | Speedup |
| --- | ---: | ---: | ---: |
| Single AOI, 15 scenes | 9.08 s | 1.14 s | 8.0x |
| Multi-AOI, 30 scenes | 42.05 s | 2.25 s | 18.7x |
| Cross-CRS, 12 scenes | 12.47 s | 0.59 s | 21.3x |

![Processing time comparison](./assets/benchmark_results.png)

Rasteret also compares well against time-series workflows that use Google Earth
Engine or thread-pooled rasterio for the measured setup:

| Library | First run (cold) | Subsequent runs (hot) |
| --- | ---: | ---: |
| Rasterio + ThreadPool | 32 s | 24 s |
| Google Earth Engine | 10-30 s | 3-5 s |
| Rasteret | 3 s | 3 s |

![Single request performance](./assets/single_timeseries_request.png)

See the [Benchmarks guide](https://terrafloww.github.io/rasteret/explanation/benchmark/)
for methodology, environment details, and additional Hugging Face `datasets`
comparisons.

## Install

```bash
uv pip install rasteret
```

Optional integrations:

```bash
uv pip install "rasteret[torchgeo]"
uv pip install "rasteret[aws]"
uv pip install "rasteret[azure]"
uv pip install "rasteret[all]"
```

Rasteret requires Python 3.12 or later.

## Learn More

- [Getting Started](https://terrafloww.github.io/rasteret/getting-started/)
- [Build from Parquet and Arrow Tables](https://terrafloww.github.io/rasteret/how-to/build-from-tables/)
- [Enriched Collection Workflows](https://terrafloww.github.io/rasteret/how-to/enriched-collection-workflows/)
- [TorchGeo Integration](https://terrafloww.github.io/rasteret/how-to/torchgeo-integration/)
- [Benchmarks](https://terrafloww.github.io/rasteret/explanation/benchmark/)
- [API Reference](https://terrafloww.github.io/rasteret/reference/)

## License

Code: [Apache-2.0](LICENSE)
