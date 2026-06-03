# TorchGeo Boundary

Rasteret no longer ships a built-in TorchGeo `GeoDataset` adapter.

The TorchGeo integration boundary now lives at the `Collection` public API:

- `to_table(...)`
- `read_window(...)`

These methods expose the two pieces a downstream TorchGeo dataset needs:

1. Arrow/GeoArrow collection metadata with footprints, datetimes, raster CRS sidecars, and band metadata.
2. A fixed-grid window read that renders selected records onto exact query bounds.

This keeps TorchGeo-specific dataset logic out of Rasteret while preserving the
collection-first read path and byte-range I/O engine.

Until the downstream TorchGeo dataset lands, use Rasteret's existing output
surfaces directly:

- `get_numpy(...)`
- `get_xarray(...)`
- `get_gdf(...)`
- `sample_points(...)`
