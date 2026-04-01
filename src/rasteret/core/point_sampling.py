# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from tqdm.asyncio import tqdm

from rasteret.core.geometry import (
    ensure_point_geoarrow,
    intersect_bbox,
)
from rasteret.core.point_sample_helpers import (
    candidate_point_indices_for_raster,
    point_bounds_4326,
)
from rasteret.core.utils import infer_data_source, run_sync
from rasteret.fetch.cog import COGReader
from rasteret.types import (
    POINT_SAMPLES_NEIGHBORHOOD_SCHEMA,
    POINT_SAMPLES_SCHEMA,
    empty_point_samples_neighborhood_table,
    empty_point_samples_table,
)

logger = logging.getLogger(__name__)


def _ensure_point_samples_table(
    sampled: Any, *, return_neighbourhood: bool = False
) -> pa.Table:
    """Normalize internal point-sample results to a typed Arrow table."""
    schema = (
        POINT_SAMPLES_NEIGHBORHOOD_SCHEMA
        if return_neighbourhood
        else POINT_SAMPLES_SCHEMA
    )
    empty_table = (
        empty_point_samples_neighborhood_table
        if return_neighbourhood
        else empty_point_samples_table
    )

    table: pa.Table
    if isinstance(sampled, pa.Table):
        table = sampled
    elif sampled is None:
        return empty_table()
    elif isinstance(sampled, list):
        if not sampled:
            return empty_table()
        table = pa.Table.from_pylist(sampled)
    else:
        raise TypeError(f"Unsupported point sample result type: {type(sampled)!r}")

    if table.num_rows == 0:
        return empty_table()
    if table.schema == schema:
        return table

    missing = [name for name in schema.names if name not in table.schema.names]
    if missing:
        raise ValueError(
            "Point sample table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    aligned = table.select(schema.names)
    return aligned.cast(schema, safe=False)


def get_collection_point_samples(
    *,
    collection: Any,
    points: Any,
    bands: list[str],
    geometry_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    data_source: str | None = None,
    max_concurrent: int = 50,
    progress: bool = False,
    backend: object | None = None,
    geometry_crs: int | None = 4326,
    match: Literal["all", "latest"] = "all",
    max_distance_pixels: int = 0,
    return_neighbourhood: bool = False,
    **filters: Any,
) -> pa.Table:
    """Sample point values across matching records.

    This is the public entrypoint used by ``Collection.sample_points``.
    """
    if match not in {"all", "latest"}:
        raise ValueError("match must be either 'all' or 'latest'")
    if max_distance_pixels < 0:
        raise ValueError("max_distance_pixels must be >= 0")

    async def _async_sample() -> pa.Table:
        expected_sample_errors: tuple[type[Exception], ...] = (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            pa.ArrowException,
        )
        empty_samples = (
            empty_point_samples_neighborhood_table
            if return_neighbourhood
            else empty_point_samples_table
        )

        point_array = ensure_point_geoarrow(
            points,
            geometry_column=geometry_column,
            x_column=x_column,
            y_column=y_column,
        )
        planned_point_xs, planned_point_ys, point_bbox = point_bounds_4326(
            point_array,
            geometry_crs=geometry_crs,
        )

        sample_filters = dict(filters)
        if point_bbox is not None:
            intersected_bbox = intersect_bbox(sample_filters.get("bbox"), point_bbox)
            if sample_filters.get("bbox") is not None and intersected_bbox is None:
                return empty_samples()
            sample_filters["bbox"] = intersected_bbox or point_bbox

        selected_collection = (
            collection.subset(**sample_filters) if sample_filters else collection
        )
        resolved_source = data_source or infer_data_source(collection)

        sample_batches: list[pa.RecordBatch] = []
        errors: list[tuple[str, Exception]] = []
        total_points = len(point_array)
        all_point_indices = list(range(total_points))

        def _raster_datetime_key(raster: Any):
            value = getattr(raster, "datetime", None)
            if value is None:
                from datetime import datetime

                return datetime.min

            try:
                from datetime import datetime, timedelta

                if isinstance(value, datetime):
                    return value.replace(tzinfo=None)
                # Support numpy datetime64 (used in unit tests / Arrow roundtrips).
                import numpy as np

                if isinstance(value, np.datetime64):
                    micros = int(value.astype("datetime64[us]").astype("int64"))
                    return datetime(1970, 1, 1) + timedelta(microseconds=micros)
            except (ImportError, TypeError, ValueError, OverflowError):
                logger.debug("Could not normalize raster datetime value: %r", value)

            from datetime import datetime

            try:
                return datetime.fromisoformat(
                    str(value).replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except ValueError:
                return datetime.min

        def _subset_points(
            point_indices: list[int],
        ) -> tuple[pa.Array, list[int]] | None:
            if point_indices == []:
                return None
            if len(point_indices) == len(point_array):
                return point_array, point_indices
            return (
                point_array.take(pa.array(point_indices, type=pa.int64())),
                point_indices,
            )

        def _append_sample_table(sampled: pa.Table) -> None:
            if sampled.num_rows == 0:
                return
            sample_batches.extend(sampled.to_batches())

        def _prepare_raster_points(
            raster: Any,
            *,
            point_indices_override: list[int] | None = None,
        ) -> tuple[pa.Array, list[int]] | None:
            point_indices = candidate_point_indices_for_raster(
                raster_bbox=getattr(raster, "bbox", None),
                point_xs=planned_point_xs,
                point_ys=planned_point_ys,
            )
            if point_indices is None:
                point_indices = all_point_indices
            if point_indices_override is not None:
                allowed = set(point_indices_override)
                point_indices = [idx for idx in point_indices if idx in allowed]
            if point_indices == []:
                return None
            if len(point_indices) == len(point_array):
                return point_array, point_indices
            return (
                point_array.take(pa.array(point_indices, type=pa.int64())),
                point_indices,
            )

        def _available_raster_bands(
            raster: Any, requested_bands: list[str]
        ) -> list[str]:
            resolver = getattr(raster, "try_get_band_cog_metadata", None)
            if not callable(resolver):
                return requested_bands
            available = []
            for band in requested_bands:
                cog_meta, url, _band_index = resolver(band)
                if (
                    cog_meta is not None
                    and url is not None
                    and getattr(cog_meta, "transform", None) is not None
                ):
                    available.append(band)
            return available

        if match == "latest":
            # Select the latest record per (point_index, band) using only metadata.
            # This avoids reading pixels from older records that will be discarded.
            rasters: list[Any] = []
            async for raster in selected_collection.iterate_rasters(
                resolved_source,
                bands=bands,
            ):
                rasters.append(raster)
            rasters.sort(key=_raster_datetime_key, reverse=True)

            target_pairs = total_points * len(bands)
            resolved = 0
            winners: dict[tuple[int, str], int] = {}

            for raster_idx, raster in enumerate(rasters):
                if resolved >= target_pairs:
                    break

                requested_bands = _available_raster_bands(raster, bands)
                if not requested_bands:
                    continue

                candidate_indices = candidate_point_indices_for_raster(
                    raster_bbox=getattr(raster, "bbox", None),
                    point_xs=planned_point_xs,
                    point_ys=planned_point_ys,
                )
                if candidate_indices is None:
                    candidate_indices = all_point_indices
                if not candidate_indices:
                    continue

                for point_index in candidate_indices:
                    for band in requested_bands:
                        key = (int(point_index), str(band))
                        if key in winners:
                            continue
                        winners[key] = raster_idx
                        resolved += 1
                        if resolved >= target_pairs:
                            break
                    if resolved >= target_pairs:
                        break

            if not winners:
                return empty_samples()

            points_by_raster: dict[int, set[int]] = {}
            bands_by_raster: dict[int, set[str]] = {}
            winner_key_lists_by_raster: dict[int, list[str]] = {}
            key_sep = "\x1f"
            for (point_index, band), raster_idx in winners.items():
                points_by_raster.setdefault(raster_idx, set()).add(point_index)
                bands_by_raster.setdefault(raster_idx, set()).add(band)
                winner_key_lists_by_raster.setdefault(raster_idx, []).append(
                    f"{point_index}{key_sep}{band}"
                )
            winner_keys_by_raster = {
                raster_idx: pa.array(keys, type=pa.string())
                for raster_idx, keys in winner_key_lists_by_raster.items()
            }

            async def _sample_latest_raster(
                *,
                raster_idx: int,
                raster: Any,
                point_indices_needed: list[int],
                band_codes: list[str],
                shared_reader: COGReader,
            ) -> pa.Table:
                subset = _subset_points(point_indices_needed)
                if subset is None:
                    return empty_samples()
                raster_points, point_indices = subset
                sampled = _ensure_point_samples_table(
                    await raster.sample_points(
                        points=raster_points,
                        band_codes=band_codes,
                        point_indices=point_indices,
                        max_concurrent=max_concurrent,
                        reader=shared_reader,
                        geometry_crs=geometry_crs,
                        max_distance_pixels=max_distance_pixels,
                        return_neighbourhood=return_neighbourhood,
                    ),
                    return_neighbourhood=return_neighbourhood,
                )
                if sampled.num_rows == 0:
                    return sampled
                winner_keys = winner_keys_by_raster.get(raster_idx)
                if winner_keys is None or len(winner_keys) == 0:
                    return (
                        empty_point_samples_neighborhood_table()
                        if return_neighbourhood
                        else empty_point_samples_table()
                    )
                sampled_keys = pc.binary_join_element_wise(
                    pc.cast(sampled.column("point_index"), pa.string()),
                    sampled.column("band"),
                    key_sep,
                )
                keep_mask = pc.is_in(sampled_keys, value_set=winner_keys)
                return sampled.filter(keep_mask)

            async with COGReader(
                max_concurrent=max_concurrent, backend=backend
            ) as shared_reader:
                raster_indices = sorted(points_by_raster.keys())
                raster_iterable: Any = (
                    tqdm(raster_indices, desc="Sampling points")
                    if progress
                    else raster_indices
                )
                raster_batch_size = 64
                batch: list[tuple[int, Any, list[int], list[str]]] = []

                for raster_idx in raster_iterable:
                    raster = rasters[raster_idx]
                    point_indices_needed = sorted(points_by_raster[raster_idx])
                    band_codes = [b for b in bands if b in bands_by_raster[raster_idx]]
                    if not band_codes:
                        continue

                    batch.append((raster_idx, raster, point_indices_needed, band_codes))
                    if len(batch) < raster_batch_size:
                        continue

                    results = await asyncio.gather(
                        *[
                            _sample_latest_raster(
                                raster_idx=ridx,
                                raster=rr,
                                point_indices_needed=pidx,
                                band_codes=bcodes,
                                shared_reader=shared_reader,
                            )
                            for ridx, rr, pidx, bcodes in batch
                        ],
                        return_exceptions=True,
                    )
                    for (ridx, rr, _pidx, _bcodes), result in zip(
                        batch, results, strict=False
                    ):
                        if isinstance(result, Exception):
                            if not isinstance(result, expected_sample_errors):
                                raise result
                            errors.append((getattr(rr, "id", "<unknown>"), result))
                            logger.error(
                                "Point sampling failed (id=%s): %s",
                                errors[-1][0],
                                result,
                            )
                            continue
                        _append_sample_table(result)
                    batch = []

                if batch:
                    results = await asyncio.gather(
                        *[
                            _sample_latest_raster(
                                raster_idx=ridx,
                                raster=rr,
                                point_indices_needed=pidx,
                                band_codes=bcodes,
                                shared_reader=shared_reader,
                            )
                            for ridx, rr, pidx, bcodes in batch
                        ],
                        return_exceptions=True,
                    )
                    for (ridx, rr, _pidx, _bcodes), result in zip(
                        batch, results, strict=False
                    ):
                        if isinstance(result, Exception):
                            if not isinstance(result, expected_sample_errors):
                                raise result
                            errors.append((getattr(rr, "id", "<unknown>"), result))
                            logger.error(
                                "Point sampling failed (id=%s): %s",
                                errors[-1][0],
                                result,
                            )
                            continue
                        _append_sample_table(result)
        else:
            iterable: Any
            if progress:
                rasters = []
                async for raster in selected_collection.iterate_rasters(
                    resolved_source,
                    bands=bands,
                ):
                    rasters.append(raster)
                iterable = tqdm(rasters, desc="Sampling points")
            else:
                iterable = selected_collection.iterate_rasters(
                    resolved_source,
                    bands=bands,
                )

            if progress:
                async with COGReader(
                    max_concurrent=max_concurrent, backend=backend
                ) as shared_reader:
                    for raster in iterable:
                        prepared = _prepare_raster_points(raster)
                        if prepared is None:
                            continue
                        raster_points, point_indices = prepared
                        try:
                            sampled = _ensure_point_samples_table(
                                await raster.sample_points(
                                    points=raster_points,
                                    band_codes=bands,
                                    point_indices=point_indices,
                                    max_concurrent=max_concurrent,
                                    reader=shared_reader,
                                    geometry_crs=geometry_crs,
                                    max_distance_pixels=max_distance_pixels,
                                    return_neighbourhood=return_neighbourhood,
                                ),
                                return_neighbourhood=return_neighbourhood,
                            )
                            _append_sample_table(sampled)
                        except expected_sample_errors as exc:
                            errors.append((getattr(raster, "id", "<unknown>"), exc))
                            logger.error(
                                "Point sampling failed (id=%s): %s",
                                errors[-1][0],
                                exc,
                            )
            else:
                async with COGReader(
                    max_concurrent=max_concurrent, backend=backend
                ) as shared_reader:
                    async for raster in iterable:
                        prepared = _prepare_raster_points(raster)
                        if prepared is None:
                            continue
                        raster_points, point_indices = prepared
                        try:
                            sampled = _ensure_point_samples_table(
                                await raster.sample_points(
                                    points=raster_points,
                                    band_codes=bands,
                                    point_indices=point_indices,
                                    max_concurrent=max_concurrent,
                                    reader=shared_reader,
                                    geometry_crs=geometry_crs,
                                    max_distance_pixels=max_distance_pixels,
                                    return_neighbourhood=return_neighbourhood,
                                ),
                                return_neighbourhood=return_neighbourhood,
                            )
                            _append_sample_table(sampled)
                        except expected_sample_errors as exc:
                            errors.append((getattr(raster, "id", "<unknown>"), exc))
                            logger.error(
                                "Point sampling failed (id=%s): %s",
                                errors[-1][0],
                                exc,
                            )

        if errors and sample_batches:
            record_id, first = errors[0]
            warnings.warn(
                "Some records failed during point sampling "
                f"({len(errors)} failure(s)); first failure in record '{record_id}': {first}",
                RuntimeWarning,
                stacklevel=2,
            )

        if not sample_batches:
            if errors:
                record_id, first = errors[0]
                msg = (
                    "No point samples resolved. "
                    f"{len(errors)} record(s) failed; first failure in record "
                    f"'{record_id}': {first}"
                )
                raise ValueError(msg) from first
            return (
                empty_point_samples_neighborhood_table()
                if return_neighbourhood
                else empty_point_samples_table()
            )

        schema = (
            POINT_SAMPLES_NEIGHBORHOOD_SCHEMA
            if return_neighbourhood
            else POINT_SAMPLES_SCHEMA
        )
        return pa.Table.from_batches(sample_batches, schema=schema)

    return run_sync(_async_sample())
