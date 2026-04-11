# Examples

Production-oriented scripts for Rasteret workflows. Each script uses
`argparse` or is fully runnable with default arguments.

For step-by-step learning, see the [tutorial notebooks](../notebooks/).

## Collection management

- `collection_management.py`: Build, persist, reload, discover, and filter
  collections. Covers `export`, `load`, `list_collections`,
  `subset()`, `where()`, `CloudConfig.register()`, `BandRegistry.register()`.

## Build from Parquet

- `build_collection_from_parquet.py`: Build a Collection from any Parquet
  with COG URLs: Source Cooperative exports, STAC GeoParquet, or custom
  catalogs. Supports Arrow predicate/projection pushdown.
  CLI-scriptable with `--manifest-url`.

## Requester-pays / Landsat

- `landsat_xarray.py`: Landsat xarray workflow via the USGS STAC endpoint.
  Requires AWS credentials (`aws configure`) for the requester-pays bucket.

## Benchmarks

- For the full TorchGeo comparison and benchmark charts, see [`05_torchgeo_comparison.ipynb`](../docs/tutorials/05_torchgeo_comparison.ipynb).

## Major TOM-style workflows

- `major_tom_benchmark/`: A three-script workflow to build a scene-level
  collection cache with Major TOM-style columns and run HF vs Rasteret
  throughput benchmarks. Requires `pip install git+https://github.com/ESA-PhiLab/Major-TOM`.
