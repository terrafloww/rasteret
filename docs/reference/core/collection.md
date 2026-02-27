# rasteret.core.collection

The central `Collection` class: Arrow dataset wrapper with filtering, output adapters, and persistence.

Most-used read APIs on `Collection`:

- `get_numpy(...)` -> NumPy arrays (`[N, H, W]` single-band, `[N, C, H, W]` multi-band)
- `get_xarray(...)` -> `xarray.Dataset`
- `get_gdf(...)` -> `geopandas.GeoDataFrame`
- `to_torchgeo_dataset(...)` -> TorchGeo-compatible dataset

::: rasteret.core.collection
