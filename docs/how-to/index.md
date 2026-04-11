# How-To Guides

Use the how-to pages when you already know the task you want to do and want the
canonical written guide for it.

If you are still learning the basic workflow, start with
[Getting Started](../getting-started/index.md), then
[Concepts](../explanation/concepts.md), then
[Transitioning from Rasterio](transitioning-from-rasterio.md).

How-to pages are grouped by the kinds of tasks people usually do next:

- bringing data into Rasteret
- reading from collections
- enriching collections for downstream workflows
- training and multi-dataset use
- managing collections and cloud access

| Guide | Use it when you want to... |
|-------|----------------------------|
| [Build from Parquet](build-from-parquet.md) | Create a collection from an existing local or S3 Parquet/GeoParquet source |
| [Point Sampling and Masking](point-sampling-and-masking.md) | Sample point values or control polygon masking behavior |
| [AEF Embeddings](aef-embeddings.md) | Work with AlphaEarth Foundation embeddings through Rasteret |
| [Enriched Parquet Workflows](enriched-parquet-workflows.md) | Enrich a collection with workflow metadata such as splits, labels, AOIs, grids, or grouping keys |
| [ML Training with Splits](ml-training-splits.md) | Add split/label columns and pass them into TorchGeo |
| [Multi-Dataset Training](multi-dataset-training.md) | Compose Rasteret-backed TorchGeo datasets or merge xarray outputs |
| [Collection Management](collection-management.md) | Build, import, inspect, export, reload, and filter collections |
| [Dataset Catalog](dataset-catalog.md) | Browse built-in datasets and register local dataset IDs |
| [Custom Cloud Provider](custom-cloud-provider.md) | Configure requester-pays or authenticated cloud access |

When a notebook and a how-to page cover similar ground, the how-to page is the
canonical written reference and the notebook is the guided walkthrough.
