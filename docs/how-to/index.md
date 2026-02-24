# How-To Guides

Task-oriented recipes for common Rasteret workflows. Each guide focuses on
a single task and assumes you have Rasteret installed. Parquet and
non-STAC workflows come first, followed by ML training, then collection
management and advanced configuration.

| Guide | Description |
|-------|-------------|
| [Build from Parquet](build-from-parquet.md) | Build a Collection from any Parquet with GeoTIFF URLs (Source Cooperative, STAC GeoParquet, custom catalogs) |
| [AEF Embeddings](aef-embeddings.md) | Read AlphaEarth Foundation 64-band embeddings (built-in catalog dataset) |
| [Enriched Parquet Workflows](enriched-parquet-workflows.md) | Add AOIs, splits, labels; query with DuckDB or PyArrow |
| [ML Training with Splits](ml-training-splits.md) | Train/val/test splits with TorchGeo integration |
| [Multi-Dataset Training](multi-dataset-training.md) | Combine Collections with TorchGeo `&` / `|` operators or xarray merge |
| [Collection Management](collection-management.md) | Build, persist, discover, filter, and share collections |
| [Dataset Catalog](dataset-catalog.md) | Browse built-ins, register locals, and build by ID |
| [Custom Cloud Provider](custom-cloud-provider.md) | Multi-cloud auth, credential providers, custom storage configs |
