# The Correctness Contract

This page defines Rasteret's user-visible accuracy contract. The goal is to
match rasterio/GDAL semantics for supported tiled GeoTIFF/COG reads.

When Rasteret cannot safely match those semantics, it should fail with an
actionable error rather than silently guess.

## Scope (what Rasteret supports)

- **Remote, tiled GeoTIFFs / COGs** that support byte-range reads (HTTP/S3/GCS/Azure).
- **Local tiled GeoTIFFs** are supported for development/testing, but Rasteret is
  optimized for object stores.

Rasteret does **not** try to read everything GDAL can:

- **Non-tiled (striped) TIFFs** are rejected (no `TileOffsets/TileByteCounts`).
- Some TIFF encodings/layouts are intentionally unsupported and raise
  `NotImplementedError`.

## Georeferencing

- `transform` semantics are **PixelIsArea-normalized** (GDAL-style). GeoTIFFs
  using PixelIsPoint conventions are corrected so pixel grids align with rasterio.
- Rotated/sheared transforms are not supported in the core reader.

## Read semantics (AOI/window)

Rasteret's defaults are rasterio-aligned:

- `all_touched=False` by default for polygon masking.
- When `filled=True`, pixels outside the requested AOI/window or outside raster
  coverage are filled with:
  - the COG `nodata` value when present, otherwise
  - `0` (preserving native dtype).

`all_touched` can be controlled at Collection API level on:

- `get_xarray(..., all_touched=...)`
- `get_gdf(..., all_touched=...)`
- `get_numpy(..., all_touched=...)`

## `valid_mask` semantics

Rasteret computes a boolean `valid_mask` during COG reads that is
**True** only where a pixel is both:

1) inside the requested AOI/window, **and**\
2) inside actual raster coverage (not padded/fill pixels).

Point sampling uses this mask to exclude invalid pixels. Public high-level
outputs (`get_xarray`, `get_numpy`, `get_gdf`)
currently return filled arrays/frames and do not expose `valid_mask` as a
separate output.

## Data types

- Rasteret preserves the **native COG dtype** by default (e.g., Sentinel-2
  `uint16`, AEF `int8`).
- Masking/filling does not silently promote to a floating dtype.

## TorchGeo interop

`Collection.to_torchgeo_dataset()` provides **pipeline-level interop**:
it returns a standard TorchGeo `GeoDataset` so samplers/DataLoader/training code
stay in TorchGeo, while Rasteret provides fast pixel I/O underneath.

Pixel placement uses `rasterio.merge.merge(bounds=..., res=...)` semantics,
matching what TorchGeo's own `RasterDataset._merge_or_stack()` calls. This is
handled by `rio_semantics.py`, which delegates placement entirely to rasterio
and does not reimplement merge/warp logic. South-up rasters (e.g. AEF, where
`transform.e > 0`) are normalised to north-up before merge, consistent with
TorchGeo's `WarpedVRT` approach in `_load_warp_file()`.

If requested bands have different resolutions, Rasteret fails fast by default.
To opt into resampling bands onto a common grid in the TorchGeo adapter, pass
`allow_resample=True` to `Collection.to_torchgeo_dataset(...)`.

## Point sampling semantics

- `Collection.sample_points(...)` samples the pixel containing the point first,
  aligned with `rasterio.sample(...)` nearest sampling semantics.
- When that base pixel is nodata/NaN, Rasteret can search neighboring pixels
  out to a bounded Chebyshev distance via `max_distance_pixels` (square rings)
  and choose the closest candidate by exact point-to-pixel-rectangle distance.
- When `return_neighbourhood!="off"`, Rasteret also returns a neighbourhood
  window in row-major order so downstream tools can compute their own stats
  without extra raster reads. With `return_neighbourhood="if_center_nodata"`,
  rows with a valid center pixel have a NULL window.
- Point outputs are tabular (`pyarrow.Table`) with explicit CRS columns
  (`point_crs`, `raster_crs`) for downstream joins and multi-CRS workflows.
- Non-geometry point metadata columns are preserved unless they collide with
  Rasteret output column names.

## What "fail loudly" means

For unsupported inputs, Rasteret raises explicit errors like:

- "requires a tiled GeoTIFF"
- "unsupported TIFF compression"
- "chunky multi-sample TIFFs require an explicit band_index"
- "Tabular Arrow geometry input is missing GeoArrow CRS metadata"

These errors are preferred over partial reads or heuristic fallbacks that could
silently produce wrong pixels.

## Contributor validation checklist

When you change anything in the read pipeline (`fetch/`, `core/`, TorchGeo adapter):

- `uv run pytest -q`
- `uv run pytest --network -q` (when the change affects real cloud reads)
- `uv run mkdocs build --strict`
