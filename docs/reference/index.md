# API Reference

Full API documentation auto-generated from source code docstrings using
[mkdocstrings](https://mkdocstrings.github.io/).

Start with **[Entry Points](rasteret.md)** (`rasteret` module) for the
top-level functions most users need. **[Core](core/collection.md)** covers
the `Collection` class and execution layer (`get_numpy()`, `get_xarray()`,
`get_gdf()`, `to_torchgeo_dataset()`). **[Ingest](ingest/index.md)**
has the builders for STAC and Parquet sources. **[Fetch](fetch/cog.md)**
has the COG reader and obstore backend internals. **[Integrations](integrations/torchgeo.md)**
covers TorchGeo. **[Configuration](types.md)** has types and
the [CLI](cli.md).

Browse by navigating the module tree in the sidebar, or use the search box.

!!! note
    Only public API is documented here. Private methods (prefixed with `_`)
    and test modules are excluded.
