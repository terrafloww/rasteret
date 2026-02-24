# rasteret

Top-level entry points for building, loading, and extending collections.

Most users need only a few of these:

- **`build()`** - build a Collection from the [catalog](../how-to/dataset-catalog.md) by ID.
- **`build_from_stac()`** / **`build_from_table()`** - full-control builders for STAC APIs or existing Parquet.
- **`load()`** - reload a previously built Collection from Parquet.
- **`register()`** - add a custom catalog entry to the in-memory registry.
- **`register_local()`** - register a local Collection as a catalog entry (persists to `~/.rasteret/datasets.local.json`).
- **`create_backend()`** - create an authenticated I/O backend for [multi-cloud reads](../how-to/custom-cloud-provider.md).

See [Getting Started](../getting-started/index.md) for usage examples.

::: rasteret
    options:
      members:
        - build
        - build_from_stac
        - build_from_table
        - create_backend
        - load
        - register
        - register_local
        - version
