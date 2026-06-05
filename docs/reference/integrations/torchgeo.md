# TorchGeo Boundary

Rasteret does not ship a built-in TorchGeo `GeoDataset` adapter.
The supported integration is `RasteretDataset` in `torchgeo.datasets` (TorchGeo 0.10+).

The TorchGeo integration boundary on the Rasteret side is the `Collection` public API:

- `to_table(...)` — Arrow/GeoArrow collection metadata: footprints, datetimes, raster CRS sidecars, and band metadata
- `read_window(...)` — fixed-grid window read that renders selected records onto exact query bounds

This keeps TorchGeo-specific dataset logic in TorchGeo while Rasteret owns the
collection-first read path and byte-range I/O engine.

For other output surfaces:

- `get_numpy(...)`
- `get_xarray(...)`
- `get_gdf(...)`
- `sample_points(...)`
