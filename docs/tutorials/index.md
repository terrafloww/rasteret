# Tutorials

Hands-on notebooks for learning Rasteret step by step.

Recommended sequence (matches sidebar order and notebook “Next” links):
**Quickstart** -> **TorchGeo Integration** -> **Building from Parquet** ->
**Parquet-first Filtering** -> **Configuring Custom Collections** ->
**TorchGeo Benchmark**.

| Tutorial | What you'll learn |
|---------|------------------|
| [Quickstart](01_quickstart.ipynb) | `build()` a Collection, fetch pixels as xarray/NumPy/GeoPandas, compute NDVI |
| [TorchGeo Integration](02_torchgeo_09_accelerator.ipynb) | Plug a Rasteret Collection into a TorchGeo training pipeline |
| [Building from Parquet](06_non_stac_cog_collections.ipynb) | `build_from_table()` with real data from Source Cooperative + DuckDB exploration |
| [Parquet-first Filtering](03_parquet_first_filtering.ipynb) | Cache once, filter anywhere with Arrow predicates |
| [Configuring Custom Collections](04_custom_cloud_and_bands.ipynb) | Cloud configs, band mappings, and storage backends for non-built-in datasets |
| [TorchGeo Benchmark](05_torchgeo_comparison.ipynb) | Side-by-side performance comparison with native TorchGeo |

!!! tip "Which build function?"
    **Quickstart** and **TorchGeo Integration** use `build()`, which looks up
    STAC API details from the [dataset catalog](../how-to/dataset-catalog.md).
    **Building from Parquet** uses `build_from_table()` for existing Parquet files.
    For custom STAC APIs not in the catalog, use `build_from_stac()`; see
    [Collection Management](../how-to/collection-management.md).

!!! note
    Some notebooks use live STAC queries and cloud reads. Re-execute locally with:

    ```bash
    uv run python -m nbconvert --execute docs/tutorials/01_quickstart.ipynb
    ```
