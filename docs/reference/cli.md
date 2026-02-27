# rasteret.cli

Command-line interface for building collections and browsing the dataset catalog.

If you are starting fresh, the shortest path is:
`datasets list` -> `datasets info` -> `datasets build`.
Use `register-local` / `export-local` / `unregister-local` after you want
reusable local catalog entries.

For a deeper walkthrough of the dataset catalog, see
[Dataset Catalog](../how-to/dataset-catalog.md).

Common commands:

```bash
# Browse the catalog
rasteret datasets list
rasteret datasets info earthsearch/sentinel-2-l2a
rasteret datasets info pc/sentinel-2-l2a

# Build a collection from the catalog
#   datasets build <dataset_id> <name> --bbox <minx,miny,maxx,maxy> --date-range <start,end>
rasteret datasets build earthsearch/sentinel-2-l2a my_s2 \
  --bbox 77.5,12.9,77.7,13.1 --date-range 2024-01-01,2024-06-30

# Manage local collections
rasteret collections list
rasteret collections info my_s2
rasteret collections delete my_s2 --yes

# Build from a catalog entry (top-level shortcut)
rasteret build earthsearch/sentinel-2-l2a my_s2 \
  --bbox 77.5,12.9,77.7,13.1 --date-range 2024-01-01,2024-06-30

# Build from a custom STAC API (not in the catalog)
#   collections build <name> --stac-api <url> --collection <id> --bbox ... --date-range ...
rasteret collections build my_custom \
  --stac-api https://example.com/stac --collection my-collection \
  --bbox 77.5,12.9,77.7,13.1 --date-range 2024-01-01,2024-06-30

# Import a Parquet/GeoParquet table into a local collection
rasteret collections import my_parquet --record-table /path/or/uri.parquet

# Register a local Collection as a reusable catalog entry
rasteret datasets register-local local/my-collection /path/to/collection_parquet
rasteret datasets build local/my-collection my_local_build

# Export or remove local catalog entries
rasteret datasets export-local local/my-collection ./my-collection.dataset.json
rasteret datasets unregister-local local/my-collection
```

!!! note
    `rasteret collections import` materializes the record table as a local
    collection. If you need accelerated pixel reads, ensure the source table
    already contains enriched COG metadata, or build via Python
    `build_from_table(..., enrich_cog=True)`.

!!! note "Authenticated datasets"

    - **Planetary Computer (`pc/*`)**: install `rasteret[azure]` for SAS signing.
    - **Requester-pays Landsat**: install `rasteret[aws]` and configure AWS credentials.

`datasets register-local` persists catalog entries to
`~/.rasteret/datasets.local.json` unless `--no-persist` is passed.
`datasets export-local` writes one catalog entry as a JSON file (for sharing/reuse).
`datasets unregister-local` removes local entries from the persisted registry
and current runtime registry.

Use `--json` on most commands for script-friendly output.

!!! info "Where do collections end up?"

    Collections are stored as Parquet datasets under
    `~/rasteret_workspace/` (or the directory set by `--workspace-dir`).
    The directory name depends on the build path:

    | Build method | Directory name |
    |---|---|
    | `datasets build` / `build()` (STAC-backed descriptor) | `{name}_{daterange}_{source}_stac/` (or `{name}_{source}_stac/` when no date range) |
    | `datasets build` / `build()` (GeoParquet-backed descriptor) | `{name}_records/` |
    | `collections import` / `build_from_table()` | `{name}_records/` |
    | `collections build` / `build_from_stac()` | `{name}_{daterange}_{source}_stac/` (or `{name}_{source}_stac/` when no date range) |

    For example, `datasets build earthsearch/sentinel-2-l2a my_s2 --date-range 2024-01-01,2024-06-30`
    creates `~/rasteret_workspace/my-s2_202401-06_sentinel_stac/`.

    Use `rasteret collections list` to see all local collections and their paths.

::: rasteret.cli
