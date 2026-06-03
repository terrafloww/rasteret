# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from affine import Affine

from rasteret.constants import BandRegistry
from rasteret.core.geometry import coerce_to_geoarrow
from rasteret.core.reader_pool import AsyncCOGReaderPool
from rasteret.core.rio_semantics import MergeGrid, merge_semantic_resample_single_source
from rasteret.core.utils import normalize_transform, reproject_array
from rasteret.fetch.cog import read_cog
from rasteret.types import CogMetadata

logger = logging.getLogger(__name__)


def _valid_data_mask(
    band_array: np.ndarray,
    nodata: float | int | None,
) -> np.ndarray:
    if nodata is None:
        if np.issubdtype(band_array.dtype, np.floating):
            return np.isfinite(band_array)
        return np.ones(band_array.shape, dtype=bool)
    if isinstance(nodata, float) and np.isnan(nodata):
        return ~np.isnan(band_array)
    valid = band_array != nodata
    if np.issubdtype(band_array.dtype, np.floating):
        valid &= np.isfinite(band_array)
    return valid


def _mosaic_window_records(
    records: list[np.ndarray],
    nodata_values: list[list[float | int | None]],
) -> np.ndarray:
    if not records:
        raise ValueError("No records available for mosaicking")
    out = np.zeros_like(records[0])
    filled = np.zeros_like(out, dtype=bool)
    for record, record_nodata in zip(records, nodata_values, strict=False):
        if record.shape != out.shape:
            raise ValueError(
                f"Mosaic shape mismatch: expected {out.shape}, got {record.shape}"
            )
        for band_idx in range(out.shape[0]):
            valid = _valid_data_mask(record[band_idx], record_nodata[band_idx])
            write_mask = valid & ~filled[band_idx]
            if np.any(write_mask):
                out[band_idx, write_mask] = record[band_idx, write_mask]
                filled[band_idx, write_mask] = True
    return out


def _resampling_for_dtype(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.floating):
        return "bilinear"
    return "nearest"


def _full_transform(meta: Any) -> Affine:
    sx, tx, sy, ty = normalize_transform(meta.transform)
    return Affine(float(sx), 0.0, float(tx), 0.0, float(sy), float(ty))


def _extract_asset_href(asset: dict | None) -> str | None:
    """Resolve the most appropriate href from a STAC asset dict."""
    if not asset:
        return None
    if not isinstance(asset, dict):
        return asset  # type: ignore[return-value]
    href = asset.get("href")
    if href:
        return href
    alternates = asset.get("alternate") or {}
    if isinstance(alternates, dict):
        for key in ("s3", "aws", "https", "http", "cloudfront"):
            alt = alternates.get(key)
            if isinstance(alt, dict) and alt.get("href"):
                return alt["href"]
        for alt in alternates.values():
            if isinstance(alt, dict) and alt.get("href"):
                return alt["href"]
    links = asset.get("links")
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and link.get("href"):
                return link["href"]
    return None


def read_collection_window(
    *,
    collection: Any,
    record_ids: Sequence[str] | pa.Array,
    bounds: tuple[float, float, float, float],
    res: tuple[float, float],
    bands: list[str],
    target_crs: int | None = None,
    max_concurrent: int = 50,
    backend: Any = None,
    reader_pool: AsyncCOGReaderPool | None = None,
) -> np.ndarray:
    """Read selected records onto a fixed output grid and mosaic overlaps."""
    if reader_pool is None:
        raise ValueError("read_collection_window requires a reader_pool")

    # Normalize record_ids → Python list for ordering + Arrow array for dataset filter.
    if isinstance(record_ids, pa.Array):
        if len(record_ids) == 0:
            raise ValueError("record_ids cannot be empty")
        ids_array = record_ids.cast(pa.string())
        ordered_ids = [str(v.as_py()) for v in ids_array]
    else:
        if not record_ids:
            raise ValueError("record_ids cannot be empty")
        ordered_ids = [str(rid) for rid in record_ids]
        ids_array = pa.array(ordered_ids, type=pa.string())

    scan_dataset = collection._filtered_data_dataset()
    if scan_dataset is None:
        raise ValueError(
            "read_window requires a dataset-backed collection scan. "
            "Streaming-only collection backends are not supported here."
        )

    # Resolve band alias candidates once (BandRegistry forward + reverse maps).
    # This is the same logic as RasterAccessor.try_get_band_cog_metadata but done
    # once per read_window call rather than once per (row × band).
    data_source = collection.data_source or ""
    band_map = BandRegistry.get(data_source)

    schema_names = set(scan_dataset.schema.names)

    # For each requested band, determine:
    #   - ordered list of asset key candidates to try (in priority order)
    #   - which metadata column name exists in the schema
    band_asset_candidates: dict[str, list[str]] = {}
    band_meta_col: dict[str, str] = {}  # band_code -> actual column name in schema

    for band in bands:
        candidates: list[str] = [band]
        forward = band_map.get(band)
        if forward:
            candidates.append(forward)
        if band_map and band in band_map.values():
            reverse = {v: k for k, v in band_map.items()}
            back = reverse.get(band)
            if back:
                candidates.append(back)
        band_asset_candidates[band] = candidates

        for candidate in candidates:
            col = f"{candidate}_metadata"
            if col in schema_names:
                band_meta_col[band] = col
                break

    # Read only the three columns we actually need (plus resolved metadata columns).
    # This skips datetime, geometry, bbox, cloud_cover — everything iterate_rasters
    # would have materialized but window reads never use.
    needed_meta_cols = list(dict.fromkeys(band_meta_col.values()))  # deduped, ordered
    read_cols = [
        c for c in ["id", "assets", "proj:epsg", *needed_meta_cols] if c in schema_names
    ]

    selected_dataset = scan_dataset.filter(ds.field("id").isin(ids_array))
    table = selected_dataset.to_table(columns=read_cols)

    if table.num_rows == 0:
        raise ValueError("No collection rows matched the requested record_ids")

    # Build row-index lookup, preserving duplicate IDs (uncommon but possible).
    id_to_rows: dict[str, list[int]] = {}
    id_col = table.column("id")
    for row_idx in range(table.num_rows):
        row_id = str(id_col[row_idx].as_py())
        id_to_rows.setdefault(row_id, []).append(row_idx)

    # ordered_row_indices preserves the caller-specified priority order.
    ordered_row_indices: list[int] = []
    for rid in ordered_ids:
        ordered_row_indices.extend(id_to_rows.get(rid, []))

    assets_col = table.column("assets")
    crs_col = table.column("proj:epsg") if "proj:epsg" in table.schema.names else None
    meta_cols = {col: table.column(col) for col in needed_meta_cols}

    async def _read() -> np.ndarray:
        patch = coerce_to_geoarrow(bounds)

        if target_crs is not None:
            output_crs = int(target_crs)
        else:
            output_crs = None
            for row_idx in ordered_row_indices:
                if crs_col is not None and crs_col[row_idx].is_valid:
                    crs_val = crs_col[row_idx].as_py()
                    if crs_val is not None:
                        output_crs = int(crs_val)
                        break
            if output_crs is None:
                raise ValueError(
                    "read_window requires record CRS metadata (`proj:epsg`) when "
                    "target_crs is not provided."
                )

        query_grid = MergeGrid(
            bounds=bounds,
            res=(abs(float(res[0])), abs(float(res[1]))),
        )

        reader = reader_pool.reader  # type: ignore[union-attr]

        # Plan all band reads across all records before firing any IO.
        # record_requests: (row_id, row_crs, per_band_request_list, slice_start, slice_stop)
        record_requests: list[
            tuple[str, int | None, list[tuple[str, CogMetadata, int | None]], int, int]
        ] = []
        all_requests: list[tuple[str, CogMetadata, int | None]] = []
        skipped_records = 0
        first_error: BaseException | None = None

        for row_idx in ordered_row_indices:
            row_id = str(id_col[row_idx].as_py())
            row_crs: int | None = None
            if crs_col is not None and crs_col[row_idx].is_valid:
                crs_val = crs_col[row_idx].as_py()
                if crs_val is not None:
                    row_crs = int(crs_val)

            assets_dict: dict = assets_col[row_idx].as_py() or {}

            requests: list[tuple[str, CogMetadata, int | None]] = []
            failed = False

            for band in bands:
                # Find the asset key for this band in this row's assets dict.
                asset_key: str | None = None
                for candidate in band_asset_candidates[band]:
                    if candidate in assets_dict:
                        asset_key = candidate
                        break

                if asset_key is None:
                    err = ValueError(
                        f"Band {band!r} not found in assets for record {row_id!r}"
                    )
                    if first_error is None:
                        first_error = err
                    logger.warning(
                        "Skipping record %s: band %s not found in assets", row_id, band
                    )
                    failed = True
                    break

                asset = assets_dict[asset_key]
                url = _extract_asset_href(asset)
                if url is None:
                    err = ValueError(
                        f"No href for band {band!r} asset in record {row_id!r}"
                    )
                    if first_error is None:
                        first_error = err
                    logger.warning(
                        "Skipping record %s: no href for band %s", row_id, band
                    )
                    failed = True
                    break

                band_index: int | None = None
                if isinstance(asset, dict):
                    raw_idx = asset.get("band_index")
                    if raw_idx is not None:
                        try:
                            band_index = int(raw_idx)
                        except (TypeError, ValueError):
                            pass

                meta_col_name = band_meta_col.get(band)
                if meta_col_name is None:
                    err = ValueError(f"No metadata column found for band {band!r}")
                    if first_error is None:
                        first_error = err
                    logger.warning(
                        "Skipping record %s: no metadata column for band %s",
                        row_id,
                        band,
                    )
                    failed = True
                    break

                raw_meta_val = meta_cols[meta_col_name][row_idx]
                if not raw_meta_val.is_valid:
                    err = ValueError(
                        f"Null metadata for band {band!r} in record {row_id!r}"
                    )
                    if first_error is None:
                        first_error = err
                    logger.warning(
                        "Skipping record %s: null metadata for band %s", row_id, band
                    )
                    failed = True
                    break

                raw_meta = raw_meta_val.as_py()
                cog_meta = CogMetadata.from_dict(raw_meta, crs=row_crs)
                requests.append((url, cog_meta, band_index))

            if failed:
                skipped_records += 1
                continue

            start = len(all_requests)
            all_requests.extend(requests)
            record_requests.append(
                (row_id, row_crs, requests, start, len(all_requests))
            )

        if not record_requests:
            raise ValueError(
                "No readable records were available for the requested window."
            ) from first_error

        raw_results = await asyncio.gather(
            *[
                read_cog(
                    url,
                    meta,
                    band_index=band_index,
                    geom_array=patch,
                    geom_idx=0,
                    geometry_crs=output_crs,
                    max_concurrent=max_concurrent,
                    reader=reader,
                    mode="window",
                )
                for url, meta, band_index in all_requests
            ],
            return_exceptions=True,
        )

        per_record_arrays: list[np.ndarray] = []
        per_record_nodata: list[list[float | int | None]] = []

        for row_id, row_crs, requests, start, stop in record_requests:
            record_results = raw_results[start:stop]
            arrays: list[np.ndarray] = []
            nodata_values: list[float | int | None] = []
            record_failed = False

            for result, (_url, meta, _band_index) in zip(
                record_results, requests, strict=False
            ):
                if isinstance(result, Exception):
                    skipped_records += 1
                    if first_error is None:
                        first_error = result
                    logger.warning(
                        "Skipping record %s after COG read failure: %s", row_id, result
                    )
                    record_failed = True
                    break

                data = getattr(result, "data", None)
                if not isinstance(data, np.ndarray) or data.ndim != 2 or data.size == 0:
                    skipped_records += 1
                    empty_error = ValueError(
                        "COG read returned empty/non-2D data "
                        f"(shape={getattr(data, 'shape', None)})"
                    )
                    if first_error is None:
                        first_error = empty_error
                    logger.warning(
                        "Skipping record %s after empty COG read result", row_id
                    )
                    record_failed = True
                    break

                nodata_values.append(meta.nodata)

                if row_crs is not None and row_crs != output_crs:
                    arrays.append(
                        reproject_array(
                            data,
                            result.transform,
                            row_crs,
                            output_crs,
                            query_grid.transform,
                            query_grid.shape,
                            resampling=_resampling_for_dtype(data.dtype),
                        )
                    )
                else:
                    arrays.append(
                        merge_semantic_resample_single_source(
                            data,
                            src_crop_transform=result.transform,
                            src_full_transform=_full_transform(meta),
                            src_full_width=int(meta.width),
                            src_full_height=int(meta.height),
                            src_crs=int(meta.crs or output_crs),
                            grid=query_grid,
                            resampling=_resampling_for_dtype(data.dtype),
                            src_nodata=meta.nodata,
                        )
                    )

            if record_failed or len(arrays) != len(bands):
                continue

            per_record_arrays.append(np.stack(arrays, axis=0))
            per_record_nodata.append(nodata_values)

        if skipped_records and per_record_arrays and first_error is not None:
            warnings.warn(
                f"read_window skipped unreadable records "
                f"({skipped_records}/{len(ordered_row_indices)} failure(s)); "
                f"first failure: {first_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not per_record_arrays:
            raise ValueError(
                "No readable records were available for the requested window."
            ) from first_error

        return _mosaic_window_records(per_record_arrays, per_record_nodata)

    return reader_pool.run(_read())
