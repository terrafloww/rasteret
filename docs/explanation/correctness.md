# Correctness Contract

This page describes Rasteret's **user-visible correctness guarantees**.
It is the contract that contributors must preserve.

Rasteret aims to be **GDAL/rasterio-aligned** for supported inputs. When Rasteret
cannot safely match that behavior, it fails loudly with actionable errors
instead of guessing.

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

## `valid_mask` (ML-safe outputs)

All primary reads return a boolean `valid_mask` that is **True** only where a
pixel is both:

1) inside the requested AOI/window, **and**\
2) inside actual raster coverage (not padded/fill pixels).

This lets ML pipelines avoid training on filled pixels without changing the
read dtype/values.

## Data types

- Rasteret preserves the **native COG dtype** by default (e.g., Sentinel-2
  `uint16`, AEF `int8`).
- Masking/filling does not silently promote to a floating dtype;
  use `valid_mask` to distinguish fill from real data.

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

## What "fail loudly" means

For unsupported inputs, Rasteret raises explicit errors like:

- "requires a tiled GeoTIFF"
- "unsupported TIFF compression"
- "chunky multi-sample TIFFs require an explicit band_index"

These errors are preferred over partial reads or heuristic fallbacks that could
silently produce wrong pixels.

## Contributor validation checklist

When you change anything in the read pipeline (`fetch/`, `core/`, TorchGeo adapter):

- `uv run pytest -q`
- `uv run pytest --network -q` (when the change affects real cloud reads)
- `uv run mkdocs build --strict`
