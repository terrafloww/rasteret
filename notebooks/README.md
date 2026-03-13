# Notebooks

This folder is a **01-07 series** intended for humans (and AI agents) who want to learn Rasteret
progressively. Notebooks 01-04, 06, and 07 are tutorials; notebook 05 is a benchmark
that lives under **Explanation** in the docs.

Notebooks are designed to:

- start simple and grow into TorchGeo / advanced workflows,
- be easy to adapt to your own AOIs and collections,
- avoid Rasteret-specific "magic" and hand off to TorchGeo in a clean way.

Prerequisites:
- Install notebook and optional ML tooling: `uv sync --extra dev --extra torchgeo`
- `obstore` is Rasteret's HTTP transport dependency for multi-cloud URL routing (S3/GCS/Azure); no separate extra needed
- Notebooks that call live STAC/cloud endpoints require network access; requester-pays
  examples also require cloud credentials.

## Notebooks

| # | Notebook | What it covers |
|---|----------|---------------|
| 01 | `01_quickstart.ipynb` | STAC -> Collection -> `get_xarray()`/`get_numpy()`/`get_gdf()` -> NDVI |
| 02 | `02_torchgeo_09_accelerator.ipynb` | Collection -> TorchGeo 0.9.0 `GeoDataset` -> sampler -> DataLoader |
| 03 | `03_parquet_first_filtering.ipynb` | Save/load Parquet, `subset()`, `where()`, `build_from_table()` |
| 04 | `04_custom_cloud_and_bands.ipynb` | `CloudConfig.register()`, `BandRegistry.register()`, `ObstoreBackend` |
| 05 | `05_torchgeo_comparison.ipynb` | **Benchmark** (Explanation section in docs): side-by-side standard TorchGeo vs Rasteret-accelerated workflow |
| 06 | `06_non_stac_cog_collections.ipynb` | Building from Parquet: `build_from_table()` with Source Cooperative Maxar data, DuckDB exploration, export/share |
| 07 | `07_aef_similarity_search.ipynb` | Similarity search notebook derived from GeoPython and GEE community tutorials, uses HuggingFace published Rasteret AEF collection |
| 08 | `08_aef_fire_lancedb_torchgeo.ipynb` | AEF fire-patch retrieval with `sample_points()`, `to_torchgeo_dataset()`, using arrow compatible tools LanceDB, Lonboard for viz and Source Coop published Rasteret AEF collection |

## Running

Rasteret itself uses `uv`, but notebook servers are a user choice. One option:

```bash
uv sync --extra dev --extra torchgeo
uv pip install jupyter
uv run jupyter lab
```
