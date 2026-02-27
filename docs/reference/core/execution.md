# rasteret.core.execution

Data loading pipeline: iterate collection records, fetch tiles,
and merge results into NumPy, xarray, or GeoDataFrame outputs.

This module powers `Collection.get_numpy(...)`, `Collection.get_xarray(...)`,
and `Collection.get_gdf(...)`.

`bands` is required because each raster in the collection may contain dozens
of assets (bands). Specifying which bands to load avoids fetching unnecessary
data and keeps memory usage predictable.

::: rasteret.core.execution
