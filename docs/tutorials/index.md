# Tutorials

Use the tutorials when you want to learn Rasteret by following a notebook
workflow end to end.

If you want the canonical task guide instead of a walkthrough, use the
[How-To Guides](../how-to/index.md).

Recommended sequence:

**Quickstart** -> **TorchGeo Integration** -> **Building from Parquet** ->
**Parquet-first Filtering** -> **Configuring Custom Collections** ->
**TorchGeo Benchmark**

| Tutorial | What it walks through |
|---------|------------------------|
| [Quickstart](01_quickstart.ipynb) | The first collection build and the basic read paths |
| [TorchGeo Integration](02_torchgeo_09_accelerator.ipynb) | Using a Rasteret collection inside a TorchGeo workflow |
| [Building from Parquet](06_non_stac_cog_collections.ipynb) | A notebook walkthrough of `build_from_table()` with real Parquet data |
| [Parquet-first Filtering](03_parquet_first_filtering.ipynb) | Table-first filtering before raster reads |
| [Configuring Custom Collections](04_custom_cloud_and_bands.ipynb) | Cloud config, band mapping, and custom collection setup |
| [TorchGeo Benchmark](05_torchgeo_comparison.ipynb) | Benchmarking Rasteret against a native TorchGeo path |

Useful pairing:

- [Quickstart](01_quickstart.ipynb) pairs well with
  [Getting Started](../getting-started/index.md)
- [Building from Parquet](06_non_stac_cog_collections.ipynb) pairs well with
  [Build from Parquet](../how-to/build-from-parquet.md)

Some notebooks use live STAC queries and cloud reads. To execute one locally:

```bash
uv run python -m nbconvert --execute docs/tutorials/01_quickstart.ipynb
```
