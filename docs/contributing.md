# Contributing

Thanks for your interest in Rasteret! Bug reports, feature ideas, docs
improvements, and code changes are all welcome. Open an
[issue](https://github.com/terrafloww/rasteret/issues) or
[discussion](https://github.com/terrafloww/rasteret/discussions) if you want
to talk through an idea first.

## Development setup

```bash
git clone https://github.com/terrafloww/rasteret.git
cd rasteret
uv sync --extra dev
uv run pre-commit install
uv run pytest -q
```

Optional extras for specific test suites:

```bash
# TorchGeo adapter tests
uv sync --extra dev --extra torchgeo

# COG header parsing tests (requires async-tiff fixture data)
export ASYNC_TIFF_FIXTURES=/path/to/async-tiff/fixtures/image-tiff
```

## Architecture overview

Rasteret has four layers. Every user interaction flows through them
top-to-bottom:

```text
BUILD                       QUERY                     READ                     RE-ENTRY
─────                       ─────                     ────                     ────────
build()                     Collection.subset()       COGReader                load()
build_from_stac()           Collection.where()        RasterAccessor           as_collection()
build_from_table()          Collection.select_split() header_parser
rasteret collections build

ingest/                     core/collection.py        fetch/cog.py             __init__.py
  stac_indexer.py                                     fetch/header_parser.py
  parquet_record_table.py                             core/raster_accessor.py
  normalize.py                                        core/execution.py
  enrich.py
  base.py
```

**BUILD** acquires data from an external source (STAC API, Parquet table)
and produces a Collection backed by Rasteret's Parquet schema.

**QUERY** filters the Collection in-memory using Arrow predicate pushdown.
No network access, no pixel reads.

**READ** uses cached COG metadata from the Parquet index to fetch only the
exact tiles needed. This is where the up to 20x speedup comes from.

**RE-ENTRY** reuses already-ingested data. `load()` reopens persisted
artifacts; `as_collection()` wraps read-ready Arrow tables/datasets without
rebuilding.

## Correctness contract

Rasteret's user-visible correctness guarantees are documented in
[Correctness Contract](explanation/correctness.md). If a change would alter the
meaning of `transform`, masking/fill behavior, dtype preservation, or `valid_mask`,
it should be treated as a correctness change and validated carefully.

## Project structure

```text
src/rasteret/
├── __init__.py              Top-level public API (build_from_stac, load, etc.)
├── cli.py                   CLI entry point (rasteret collections build/list/info/delete/import, rasteret build)
├── cloud.py                 [CloudConfig](reference/cloud.md), [ObstoreBackend](reference/cloud.md), StorageBackend protocol, rewrite_url()
├── catalog.py               DatasetRegistry, DatasetDescriptor, built-in + local catalog
├── constants.py             BandRegistry, built-in band mappings
├── types.py                 Shared type aliases and dataclasses
│
├── core/
│   ├── collection.py        [Collection](reference/core/collection.md) class, Parquet dataset wrapper + filtering + adapters
│   ├── execution.py         Read pipeline: data loading, xarray/GDF output
│   ├── geometry.py          Geometry helpers: WKB ↔ bbox, spatial filtering
│   ├── raster_accessor.py   Per-record data handle: async band loading, tile management
│   └── utils.py             Sync/async bridging (run_sync), CRS transforms, data source inference
│
├── fetch/
│   ├── cog.py               [COGReader](reference/fetch/cog.md): HTTP range reads, tile decompression, obstore backend
│   └── header_parser.py     TIFF/COG header parsing: IFD extraction, tile offset discovery
│
├── ingest/
│   ├── base.py              CollectionBuilder ABC: the contract all builders follow
│   ├── normalize.py         build_collection_from_table(): validation, bbox derivation, partitioning
│   ├── stac_indexer.py      [StacCollectionBuilder](reference/ingest/stac_indexer.md): STAC API search + COG header enrichment
│   ├── parquet_record_table.py  [RecordTableBuilder](reference/ingest/parquet_record_table.md): Parquet/GeoParquet ingestion with column mapping
│   ├── enrich.py            COG enrichment: parse headers, add {band}_metadata struct columns
│
├── integrations/
│   └── torchgeo.py          [RasteretGeoDataset](reference/integrations/torchgeo.md): standard TorchGeo GeoDataset adapter
│
└── tests/                   Test suite (see Testing section below)
```

## Testing

### Test groups

| Test file | What it covers | Dependencies |
|-----------|---------------|--------------|
| `test_ingest.py` | Ingest pipeline, normalization, COG enrichment | None (unit) |
| `test_collection_filters.py` | Collection.subset(), where(), select_split() | None (unit) |
| `test_execution.py` | run_sync, infer_data_source, xarray merge | None (unit) |
| `test_utils.py` | CRS transforms, grid computation | None (unit) |
| `test_cog_reader.py` | COGReader tile decompression, range merging | None (unit) |
| `test_obstore_routing.py` | obstore backend URL routing, cloud config passthrough | None (unit) |
| `test_registry.py` | BandRegistry, CloudConfig registry | None (unit) |
| `test_catalog.py` | DatasetRegistry, DatasetDescriptor, `build()` routing | None (unit) |
| `test_collection_naming.py` | Collection.create_name() short names and normalisation | None (unit) |
| `test_compat_matrix.py` | End-to-end wiring of build-time enrichment across catalog/router/builder/parser | None (unit) |
| `test_cli.py` | CLI argument parsing and handlers | None (unit) |
| `test_geoparquet_conformance.py` | GeoParquet schema and metadata conformance | None (unit) |
| `test_public_api_surface.py` | Top-level `rasteret.*` exports | None (unit) |
| `test_local_tiff_support.py` | Local tiled GeoTIFF reads via COGReader | None (unit) |
| `test_torchgeo_adapter.py` | RasteretGeoDataset | `torchgeo` extra |
| `test_stac_indexer.py` | StacCollectionBuilder | Network (mocked in CI) |
| `test_public_network_smoke.py` | Public no-auth endpoints (S3/GCS/AEF) | Internet access (no creds) |
| `test_network_smoke.py` | Catalog builds (Earth Search, PC, Landsat) | Network + optional deps/creds (auto-skips) |
| `test_header_parser_local.py` | TIFF header parsing against real files | `ASYNC_TIFF_FIXTURES` env var |

### Running tests

```bash
# All tests (unit + public no-auth smoke)
uv run pytest -q

# The default suite hits a few public endpoints (anonymous range reads + AEF).
# For fully offline runs, exclude the public smoke file:
#   uv run pytest -q --ignore=src/rasteret/tests/test_public_network_smoke.py

# Only network tests
uv run pytest -m network -v

# Specific test file
uv run pytest src/rasteret/tests/test_ingest.py -v

# TorchGeo adapter tests (requires extra)
uv sync --extra dev --extra torchgeo
uv run pytest src/rasteret/tests/test_torchgeo_adapter.py -v

# With coverage
uv run pytest --cov=rasteret --cov-report=term-missing
```

### When to write tests

- **New ingest driver**: Smoke test that produces a valid Collection from
  minimal input. Test column remapping and missing-column errors.
- **New filtering method**: Unit test with a fixture Collection, verify
  row counts and column values after filtering.
- **Bug fix**: Add a regression test that fails without the fix.

## Writing a new ingest driver

All ingest paths follow the same pattern:

1. Subclass `CollectionBuilder` from `rasteret.ingest.base`
2. Implement `build()` to acquire and return a `Collection`
3. Call `build_collection_from_table()` from `rasteret.ingest.normalize`
   to validate and normalize the Arrow table

The normalize layer enforces the schema contract: it checks for the 4
required columns (`id`, `datetime`, `geometry`, `assets`), derives
`scene_bbox` and scalar bbox columns, and adds `year`/`month` partition
columns.

```python
from rasteret.ingest.base import CollectionBuilder
from rasteret.ingest.normalize import build_collection_from_table

class MyCustomBuilder(CollectionBuilder):
    """Build a Collection from my custom data source."""

    def __init__(self, source_path: str, **kwargs):
        super().__init__(**kwargs)
        self.source_path = source_path

    def build(self, **kwargs) -> Collection:
        # 1. Acquire data as a PyArrow table
        table = self._read_my_source()

        # 2. Normalize and return a Collection
        return build_collection_from_table(
            table,
            name=self.name,
            data_source=self.data_source,
            workspace_dir=self.workspace_dir,
        )
```

For COG-backed data, call `enrich_table_with_cog_metadata()` from
`rasteret.ingest.enrich` to parse TIFF headers and add per-band
`{band}_metadata` struct columns. Without these columns, the Collection
works for filtering and metadata queries but cannot do accelerated
tile reads.

## Adding a dataset to the catalog

The built-in catalog lives in `src/rasteret/catalog.py`. Each entry is a
`DatasetDescriptor`: roughly 20 lines of Python declaring a data source
(STAC API, static STAC catalog, or GeoParquet URI), band map, license
info, and coverage hints.

Before adding a dataset, work through the
[prerequisites checklist](how-to/dataset-catalog.md#prerequisites-for-contributing-a-built-in-dataset).
The short version:

1. **Data source is reachable**: STAC API, static catalog, or GeoParquet
   file. Verify you can query it or read it with PyArrow.
2. **Band map has at least one working COG**: Rasteret parses COG headers
   during `build()`. If no asset can be parsed, Rasteret can't index or read
   the dataset.
3. **End-to-end `build()` succeeds**: run `rasteret.build()` with a small
   scope and verify `len(col) > 0`.
4. **License is verified from the authoritative source**: pull `license`,
   `license_url`, and `commercial_use` from the data provider. Do not guess.
5. **Descriptor includes required metadata**: include `id`, `name`, `description`,
   `stac_api` (or `geoparquet_uri`), `band_map`, `license`, `license_url`,
   `spatial_coverage`, `temporal_range`. For static catalogs, set
   `static_catalog=True`. For GeoParquet sources, include `column_map` if
   columns need renaming.

For static STAC catalogs (no `/search` endpoint), set `static_catalog=True`
on the descriptor. Rasteret uses `pystac.Catalog.from_file()` to traverse
these catalogs with client-side bbox/date filtering.

## Public API discipline

- Keep the top-level `rasteret` surface small and intentional
  (`build`, `build_from_stac`, `build_from_table`, `load`, `as_collection`, `register`,
  `register_local`, `create_backend`, `version`, `Collection`,
  `CloudConfig`, `BandRegistry`, `DatasetDescriptor`, `DatasetRegistry`).
- New user-facing APIs need a docstring and a smoke test.

## TorchGeo interop expectations

The adapter in `src/rasteret/integrations/torchgeo.py` follows TorchGeo
conventions and works alongside TorchGeo:

- **NumPy-style docstrings** on all public methods.
- **Full type annotations** (no untyped public signatures).
- **Standard sample format**: `{"image": Tensor, "bounds": Tensor, ...}`
, no custom keys unless documented.
- **Sampler compatibility**: shapes compatible with TorchGeo samplers
  and `stack_samples` collation.
- Output is a standard `GeoDataset`. Users do not need to learn
  Rasteret internals to use it.

## DCO sign-off

All commits must include a `Signed-off-by` line to certify the
[Developer Certificate of Origin](https://developercertificate.org/).
Use `git commit -s` to add it automatically.

```text
Signed-off-by: Your Name <your@email.com>
```

Unsigned commits will be rejected by CI.

## Code style

- **Linting**: `uv run ruff check src/`
- **Formatting**: `uv run ruff format --check`
- **Pre-commit**: Install hooks with `uv run pre-commit install`
- **Docstrings**: NumPy style on all public methods. Private methods
  get a docstring when the logic is not obvious from the name.
- **Type annotations**: Required on public signatures. Use `from __future__
  import annotations` for forward references.
