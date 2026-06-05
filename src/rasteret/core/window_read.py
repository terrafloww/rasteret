# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging
import warnings
from collections import defaultdict
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

# (row_id, row_crs, per-band read specs, start-index into flat request list, stop-index)
RecordRequest = tuple[str, Any, list[tuple[str, CogMetadata, Any]], int, int]


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
    group_by: str | None = None,
) -> np.ndarray:
    """Read selected records onto a fixed output grid and mosaic overlaps.

    Parameters
    ----------
    group_by : str, optional
        When ``"datetime"``, records are grouped by acquisition date and each
        group is mosaicked independently.  All groups are read concurrently in
        a single pool submission, returning ``[T, C, H, W]`` instead of
        ``[C, H, W]``.  This is the correct mode for time-series chip reads
        (e.g. TorchGeo ``time_series=True``).
    """
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

    # Read only the columns we actually need (plus resolved metadata columns).
    # This skips geometry, bbox, cloud_cover — everything iterate_rasters would
    # have materialized but window reads never use.
    # datetime is included when group_by="datetime" for time-series stacking.
    needed_meta_cols = list(dict.fromkeys(band_meta_col.values()))  # deduped, ordered
    base_cols = ["id", "assets", "proj:epsg"]
    if group_by == "datetime" and "datetime" in schema_names:
        base_cols = ["id", "datetime", "assets", "proj:epsg"]
    read_cols = [c for c in [*base_cols, *needed_meta_cols] if c in schema_names]

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
    datetime_col = (
        table.column("datetime")
        if group_by == "datetime" and "datetime" in table.schema.names
        else None
    )

    # ---------------------------------------------------------------------------
    # Inner helpers shared by the single-mosaic and time-series paths.
    # ---------------------------------------------------------------------------

    def _plan_rows(
        row_indices: list[int],
        all_requests: list[tuple[str, CogMetadata, Any]],
        first_error_ref: list[BaseException | None],
    ) -> list[RecordRequest]:
        """Populate *all_requests* with band read specs for *row_indices*.

        Returns a list of ``(row_id, row_crs, requests, start, stop)`` tuples
        for the rows that could be planned.  Unreadable rows are skipped with a
        warning.
        """
        record_requests: list[RecordRequest] = []
        for row_idx in row_indices:
            row_id = str(id_col[row_idx].as_py())
            row_crs: int | None = None
            if crs_col is not None and crs_col[row_idx].is_valid:
                v = crs_col[row_idx].as_py()
                if v is not None:
                    row_crs = int(v)

            assets_dict: dict = assets_col[row_idx].as_py() or {}
            requests: list[tuple[str, CogMetadata, int | None]] = []
            failed = False

            for band in bands:
                asset_key: str | None = next(
                    (c for c in band_asset_candidates[band] if c in assets_dict), None
                )
                if asset_key is None:
                    err: BaseException = ValueError(
                        f"Band {band!r} not found in assets for record {row_id!r}"
                    )
                    if first_error_ref[0] is None:
                        first_error_ref[0] = err
                    logger.warning(
                        "Skipping record %s: band %s not found", row_id, band
                    )
                    failed = True
                    break

                asset = assets_dict[asset_key]
                url = _extract_asset_href(asset)
                if url is None:
                    err = ValueError(f"No href for band {band!r} in record {row_id!r}")
                    if first_error_ref[0] is None:
                        first_error_ref[0] = err
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
                    err = ValueError(f"No metadata column for band {band!r}")
                    if first_error_ref[0] is None:
                        first_error_ref[0] = err
                    logger.warning(
                        "Skipping record %s: no metadata col for %s", row_id, band
                    )
                    failed = True
                    break

                raw_meta_val = meta_cols[meta_col_name][row_idx]
                if not raw_meta_val.is_valid:
                    err = ValueError(f"Null metadata for band {band!r} in {row_id!r}")
                    if first_error_ref[0] is None:
                        first_error_ref[0] = err
                    logger.warning(
                        "Skipping record %s: null metadata for %s", row_id, band
                    )
                    failed = True
                    break

                cog_meta = CogMetadata.from_dict(raw_meta_val.as_py(), crs=row_crs)
                requests.append((url, cog_meta, band_index))

            if failed:
                continue

            start = len(all_requests)
            all_requests.extend(requests)
            record_requests.append(
                (row_id, row_crs, requests, start, len(all_requests))
            )

        return record_requests

    def _assemble_mosaic(
        raw_results: list[Any],
        record_requests: list[RecordRequest],
        output_crs: int,
        query_grid: MergeGrid,
        skipped_ref: list[int],
        first_error_ref: list[BaseException | None],
    ) -> tuple[list[np.ndarray], list[list[float | int | None]]]:
        """Turn raw read_cog results into per-record band arrays ready for mosaicking."""
        per_record_arrays: list[np.ndarray] = []
        per_record_nodata: list[list[float | int | None]] = []

        for row_id, row_crs, requests, start, stop in record_requests:
            record_results = raw_results[start:stop]
            arrays: list[np.ndarray] = []
            nodata_values: list[float | int | None] = []
            record_failed = False

            for result, (_url, meta, _bi) in zip(
                record_results, requests, strict=False
            ):
                if isinstance(result, Exception):
                    skipped_ref[0] += 1
                    if first_error_ref[0] is None:
                        first_error_ref[0] = result
                    logger.warning(
                        "Skipping record %s after read failure: %s", row_id, result
                    )
                    record_failed = True
                    break

                data = getattr(result, "data", None)
                if not isinstance(data, np.ndarray) or data.ndim != 2 or data.size == 0:
                    skipped_ref[0] += 1
                    empty_err: BaseException = ValueError(
                        f"COG read returned empty/non-2D data (shape={getattr(data,'shape',None)})"
                    )
                    if first_error_ref[0] is None:
                        first_error_ref[0] = empty_err
                    logger.warning("Skipping record %s after empty COG result", row_id)
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

        return per_record_arrays, per_record_nodata

    # ---------------------------------------------------------------------------
    # Main coroutine.
    # ---------------------------------------------------------------------------

    async def _read() -> np.ndarray:
        patch = coerce_to_geoarrow(bounds)

        if target_crs is not None:
            output_crs = int(target_crs)
        else:
            output_crs = None
            for row_idx in ordered_row_indices:
                if crs_col is not None and crs_col[row_idx].is_valid:
                    v = crs_col[row_idx].as_py()
                    if v is not None:
                        output_crs = int(v)
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

        first_error_ref: list[BaseException | None] = [None]
        skipped_ref: list[int] = [0]

        if group_by == "datetime" and datetime_col is not None:
            # Time-series path: group rows by acquisition datetime, plan all
            # requests across all timesteps, fire ONE asyncio.gather for
            # maximum concurrency, then assemble [T, C, H, W].
            dt_to_rows: dict[Any, list[int]] = defaultdict(list)
            for row_idx in ordered_row_indices:
                dt_to_rows[datetime_col[row_idx].as_py()].append(row_idx)

            sorted_groups = sorted(dt_to_rows.items())  # ascending datetime

            all_requests: list[tuple[str, CogMetadata, Any]] = []
            per_group_record_requests: list[list[RecordRequest]] = []
            for _, grp_rows in sorted_groups:
                per_group_record_requests.append(
                    _plan_rows(grp_rows, all_requests, first_error_ref)
                )

            if not any(per_group_record_requests):
                raise ValueError(
                    "No readable records were available for any timestep."
                ) from first_error_ref[0]

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

            group_arrays: list[np.ndarray] = []
            for grp_record_requests in per_group_record_requests:
                if not grp_record_requests:
                    continue
                per_rec, per_nd = _assemble_mosaic(
                    raw_results,
                    grp_record_requests,
                    output_crs,
                    query_grid,
                    skipped_ref,
                    first_error_ref,
                )
                if per_rec:
                    group_arrays.append(_mosaic_window_records(per_rec, per_nd))

            if not group_arrays:
                raise ValueError(
                    "No valid timestep data after reading."
                ) from first_error_ref[0]
            return np.stack(group_arrays, axis=0)

        # Single-mosaic path (no group_by): mosaic all records into [C, H, W].
        all_requests_single: list[tuple[str, CogMetadata, Any]] = []
        record_requests = _plan_rows(
            ordered_row_indices, all_requests_single, first_error_ref
        )

        if not record_requests:
            raise ValueError(
                "No readable records were available for the requested window."
            ) from first_error_ref[0]

        raw_results_single = await asyncio.gather(
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
                for url, meta, band_index in all_requests_single
            ],
            return_exceptions=True,
        )

        per_record_arrays, per_record_nodata = _assemble_mosaic(
            raw_results_single,
            record_requests,
            output_crs,
            query_grid,
            skipped_ref,
            first_error_ref,
        )

        if skipped_ref[0] and per_record_arrays and first_error_ref[0] is not None:
            warnings.warn(
                f"read_window skipped unreadable records "
                f"({skipped_ref[0]}/{len(ordered_row_indices)} failure(s)); "
                f"first failure: {first_error_ref[0]}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not per_record_arrays:
            raise ValueError(
                "No readable records were available for the requested window."
            ) from first_error_ref[0]

        return _mosaic_window_records(per_record_arrays, per_record_nodata)

    return reader_pool.run(_read())
