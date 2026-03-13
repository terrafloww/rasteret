# How-To Guides

Task-oriented recipes for common Rasteret workflows. Each guide focuses on a single task and assumes you have Rasteret installed. Parquet and non-STAC workflows come first, followed by ML training, then collection management and advanced configuration.

Recommended learning path for new users:

1. build or load a Collection, 2) filter with `subset()` / `where()`,
1. read via `get_xarray()` / `get_numpy()` / `sample_points()`.

| Guide                                                                                                          | Description                                                                                                  |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [Build from Parquet](https://terrafloww.github.io/rasteret/how-to/build-from-parquet/index.md)                 | Build a Collection from any Parquet with GeoTIFF URLs (Source Cooperative, STAC GeoParquet, custom catalogs) |
| [Point Sampling and Masking](https://terrafloww.github.io/rasteret/how-to/point-sampling-and-masking/index.md) | Sample per-point values as Arrow tables; control polygon masking with `all_touched`                          |
| [AEF Embeddings](https://terrafloww.github.io/rasteret/how-to/aef-embeddings/index.md)                         | Read AlphaEarth Foundation 64-band embeddings (built-in catalog dataset)                                     |
| [Enriched Parquet Workflows](https://terrafloww.github.io/rasteret/how-to/enriched-parquet-workflows/index.md) | Add AOIs, splits, labels; query with DuckDB/PyArrow; Major TOM-style enrichment                              |
| [ML Training with Splits](https://terrafloww.github.io/rasteret/how-to/ml-training-splits/index.md)            | Train/val/test splits with TorchGeo integration                                                              |
| [Multi-Dataset Training](https://terrafloww.github.io/rasteret/how-to/multi-dataset-training/index.md)         | Combine Collections with TorchGeo `&` / \`                                                                   |
| [Collection Management](https://terrafloww.github.io/rasteret/how-to/collection-management/index.md)           | Build, persist, discover, filter, and share collections                                                      |
| [Dataset Catalog](https://terrafloww.github.io/rasteret/how-to/dataset-catalog/index.md)                       | Browse built-ins, register locals, and build by ID                                                           |
| [Custom Cloud Provider](https://terrafloww.github.io/rasteret/how-to/custom-cloud-provider/index.md)           | Multi-cloud auth, credential providers, custom storage configs                                               |
