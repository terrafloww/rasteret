# Tutorials

Use the tutorials when you want to learn Rasteret by following a notebook
workflow end to end.

If you want the canonical task guide instead of a walkthrough, use the
[How-To Guides](../how-to/index.md).

Recommended sequence:

**Quickstart** -> **Using Rasteret with TorchGeo** -> **Build from Parquet and Arrow Tables** ->
**Work with Collection Tables** -> **Custom Datasets with Rasteret**

| Tutorial | What it walks through |
|---------|------------------------|
| [Quickstart](01_quickstart.ipynb) | The first collection build and the basic read paths |
| [Using Rasteret with TorchGeo](02_using_rasteret_with_torchgeo.ipynb) | Use a Rasteret collection inside a TorchGeo workflow |
| [Build from Parquet and Arrow Tables](06_build_from_parquet_and_arrow_tables.ipynb) | A notebook walkthrough of `build_from_table()` with real Parquet data |
| [Work with Collection Tables](03_work_with_collection_tables.ipynb) | Table-first filtering before raster reads |
| [Custom Datasets with Rasteret](04_custom_datasets_with_rasteret.ipynb) | Advanced dataset registration, band mapping, URL rewriting, and cloud access |

Benchmark notebook:

- [TorchGeo Benchmark: Rasteret vs Native Rasterio](05_torchgeo_benchmark_rasteret_vs_rasterio.ipynb): benchmark evidence for the [Benchmarks](../explanation/benchmark.md) page.

Useful pairing:

- [Quickstart](01_quickstart.ipynb) pairs well with
  [Getting Started](../getting-started/index.md)
- [Build from Parquet and Arrow Tables](06_build_from_parquet_and_arrow_tables.ipynb) pairs well with
  [Build from Parquet and Arrow Tables](../how-to/build-from-tables.md)

Some notebooks use live STAC queries and cloud reads. To execute one locally:

```bash
uv run python -m nbconvert --execute docs/tutorials/01_quickstart.ipynb
```
