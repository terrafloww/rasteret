# How-To Guides

Use the how-to pages when you already know the task you want to do and want the canonical written guide for it.

If you are still learning the basic workflow, start with [Getting Started](https://terrafloww.github.io/rasteret/getting-started/index.md), then [Concepts](https://terrafloww.github.io/rasteret/explanation/concepts/index.md), then [Migrating from Rasterio](https://terrafloww.github.io/rasteret/how-to/migrating-from-rasterio/index.md).

How-to pages are grouped by the kinds of tasks people usually do next:

- bringing data into Rasteret
- reading from collections
- enriching collections for downstream workflows
- training and multi-dataset use
- managing collections and cloud access

| Guide                                                                                                                            | Use it when you want to...                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [Migrating from Rasterio](https://terrafloww.github.io/rasteret/how-to/migrating-from-rasterio/index.md)                         | Compare Rasteret's collection-first shape with manual rasterio/STAC loops                  |
| [TorchGeo Integration](https://terrafloww.github.io/rasteret/how-to/torchgeo-integration/index.md)                               | Use a Rasteret collection as a TorchGeo `GeoDataset`                                       |
| [Build from Parquet and Arrow Tables](https://terrafloww.github.io/rasteret/how-to/build-from-tables/index.md)                   | Create a collection from an existing Parquet, GeoParquet, DuckDB, Polars, or Arrow source  |
| [Bring Your Own AOIs, Points, And Metadata](https://terrafloww.github.io/rasteret/how-to/enriched-collection-workflows/index.md) | Pass business tables from GeoPandas, DuckDB, Polars, PyArrow, or SedonaDB into pixel reads |
| [Point Sampling and Masking](https://terrafloww.github.io/rasteret/how-to/point-sampling-and-masking/index.md)                   | Sample point values or control polygon masking behavior                                    |
| [AEF Embeddings](https://terrafloww.github.io/rasteret/how-to/aef-embeddings/index.md)                                           | Work with AlphaEarth Foundation embeddings through Rasteret                                |
| [Multi-Dataset Training](https://terrafloww.github.io/rasteret/how-to/multi-dataset-training/index.md)                           | Compose Rasteret-backed TorchGeo datasets or merge xarray outputs                          |
| [Collection Management](https://terrafloww.github.io/rasteret/how-to/collection-management/index.md)                             | Build, import, inspect, export, reload, and filter collections                             |
| [Dataset Catalog](https://terrafloww.github.io/rasteret/how-to/dataset-catalog/index.md)                                         | Browse built-in datasets and register local dataset IDs                                    |
| [Custom Cloud Provider](https://terrafloww.github.io/rasteret/how-to/custom-cloud-provider/index.md)                             | Configure requester-pays or authenticated cloud access                                     |

When a notebook and a how-to page cover similar ground, the how-to page is the canonical written reference and the notebook is the guided walkthrough.
