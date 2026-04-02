# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from affine import Affine
from tqdm.asyncio import tqdm

from rasteret.constants import BandRegistry
from rasteret.core.point_sample_helpers import (
    candidate_values_from_tile_cache,
    chebyshev_ring_offsets,
    nodata_mask,
    pixel_rect_distance_sq,
    plan_point_tile_groups,
    ring_candidate_grid,
    square_neighborhood_offsets,
    tile_grid_shape,
    tile_request_specs_for_pairs,
    unique_tile_pairs_for_candidates,
)
from rasteret.core.utils import normalize_transform
from rasteret.fetch.cog import read_cog
from rasteret.types import (
    POINT_SAMPLES_NEIGHBORHOOD_SCHEMA,
    POINT_SAMPLES_SCHEMA,
    CogMetadata,
    RasterInfo,
    empty_point_samples_neighborhood_table,
    empty_point_samples_table,
)

logger = logging.getLogger(__name__)


class RasterAccessor:
    """Data-loading handle for a single Parquet record (row) in a Collection.

    Each record in a Rasteret Collection represents one raster item:
    typically a satellite scene, but could be a drone image, derived
    product, or any tiled GeoTIFF.  ``RasterAccessor`` wraps that
    record's metadata and provides methods to load band data as arrays.

    Handles:
    - Async band data loading via cached COG metadata
    - Tile management and geometry masking
    - Multi-band concurrent fetching
    """

    def __init__(self, info: RasterInfo, data_source: str) -> None:
        """Initialize from a record's metadata.

        Parameters
        ----------
        info : RasterInfo
            Record metadata including URLs and COG metadata.
        data_source : str
            Data source identifier for band mapping.
        """
        self.id = info.id
        self.datetime = info.datetime
        self.bbox = info.bbox
        self.footprint = info.footprint
        self.crs = info.crs
        self.cloud_cover = info.cloud_cover
        self.assets = info.assets
        self.band_metadata = info.band_metadata
        self.collection = info.collection
        self.data_source = data_source

    def _get_band_radiometric_params(self, band_code: str) -> dict[str, float] | None:
        """Get radiometric parameters from STAC metadata if available."""
        try:
            asset = self.assets[band_code]
            band_info = asset["raster:bands"][0]

            if "scale" in band_info and "offset" in band_info:
                return {
                    "scale": float(band_info["scale"]),
                    "offset": float(band_info["offset"]),
                }
        except (KeyError, IndexError):
            pass

        return None

    def _extract_asset_href(self, asset: dict) -> str | None:
        """Resolve the most appropriate href for a STAC asset."""

        if asset is None:
            return None

        if not isinstance(asset, dict):
            return asset

        href = asset.get("href")
        if href:
            return href

        alternates = asset.get("alternate") or {}
        if isinstance(alternates, dict):
            preferred_order = ["s3", "aws", "https", "http", "cloudfront"]
            for key in preferred_order:
                alt = alternates.get(key)
                if isinstance(alt, dict) and alt.get("href"):
                    return alt["href"]
            for alt in alternates.values():
                if isinstance(alt, dict) and alt.get("href"):
                    return alt["href"]

        links = asset.get("links") if isinstance(asset, dict) else None
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and link.get("href"):
                    return link["href"]

        return None

    def try_get_band_cog_metadata(
        self,
        band_code: str,
    ) -> tuple[CogMetadata | None, str | None, int | None]:
        """Return tiled GeoTIFF/COG metadata and URL for *band_code*.

        Returns ``(None, None)`` when the asset or required per-band metadata
        is missing.
        """

        # Support both legacy asset-key conventions:
        # - Old STAC-backed Collections often use STAC asset keys (e.g. "blue")
        # - Newer/normalized Collections use logical band codes (e.g. "B02")
        #
        # Resolve by trying: direct band code, registry forward map (B02->blue),
        # then registry reverse map ("blue"->B02), taking the first key that exists.
        candidates: list[str] = [band_code]
        band_map = BandRegistry.get(self.data_source)
        forward = band_map.get(band_code)
        if forward:
            candidates.append(forward)
        if band_map and band_code in band_map.values():
            reverse = {v: k for k, v in band_map.items()}
            back = reverse.get(band_code)
            if back:
                candidates.append(back)

        asset_key = next((c for c in candidates if c in self.assets), None)
        if asset_key is None:
            return None, None, None

        asset = self.assets[asset_key]

        url = self._extract_asset_href(asset)
        band_index = asset.get("band_index") if isinstance(asset, dict) else None

        # Band metadata key could be either band_code or resolved asset_key
        metadata_keys = [f"{band_code}_metadata", f"{asset_key}_metadata"]
        raw_metadata = None
        for key in metadata_keys:
            if key in self.band_metadata:
                raw_metadata = self.band_metadata[key]
                break

        if raw_metadata is None or url is None:
            return None, None, None

        try:
            cog_metadata = CogMetadata.from_dict(raw_metadata, crs=self.crs)
            idx = None
            if band_index is not None:
                try:
                    idx = int(band_index)
                except (TypeError, ValueError):
                    idx = None
            return cog_metadata, url, idx
        except KeyError:
            return None, None, None

    def intersects(self, geometry) -> bool:
        """Return ``True`` if this record's bbox overlaps *geometry*'s bbox."""
        from rasteret.core.geometry import (
            bbox_array,
            bbox_intersects,
            coerce_to_geoarrow,
        )

        geo_arr = coerce_to_geoarrow(geometry)
        xmin, ymin, xmax, ymax = bbox_array(geo_arr)
        geom_bbox = (xmin[0].as_py(), ymin[0].as_py(), xmax[0].as_py(), ymax[0].as_py())
        record_bbox = tuple(self.bbox) if self.bbox else None
        if record_bbox is None:
            return False
        return bbox_intersects(record_bbox, geom_bbox)

    @property
    def geometry(self):
        """Alias for ``footprint``."""
        return self.footprint

    @property
    def available_bands(self) -> list[str]:
        """List available band keys for this record."""
        return list(self.assets.keys())

    def __repr__(self) -> str:
        return (
            f"RasterAccessor(id='{self.id}', "
            f"datetime='{self.datetime}', "
            f"cloud_cover={self.cloud_cover})"
        )

    async def _load_single_band(
        self,
        geom_array: pa.Array,
        geom_idx: int,
        band_code: str,
        max_concurrent: int = 50,
        reader: object | None = None,
        geometry_crs: int | None = 4326,
        all_touched: bool = False,
    ) -> dict | None:
        """Load single band data for a geometry identified by index."""
        cog_meta, url, band_index = self.try_get_band_cog_metadata(band_code)
        if cog_meta is None or url is None:
            raise ValueError(
                f"Missing band metadata or href for band '{band_code}' "
                f"in record '{self.id}'"
            )

        result = await read_cog(
            url,
            cog_meta,
            band_index=band_index,
            geom_array=geom_array,
            geom_idx=geom_idx,
            geometry_crs=geometry_crs,
            max_concurrent=max_concurrent,
            reader=reader,
            all_touched=all_touched,
        )
        if result.data.size == 0:
            return None

        # TODO: Apply radiometric correction (scale/offset from STAC
        # raster:bands) when opted in.  See _get_band_radiometric_params().
        # Needs: opt-in flag (apply_scale_offset=False default),
        # nodata masking, and dtype promotion (uint16 -> float32).

        return {"data": result.data, "transform": result.transform, "band": band_code}

    async def sample_points(
        self,
        *,
        points: pa.Array,
        band_codes: list[str],
        point_indices: list[int] | None = None,
        max_concurrent: int = 50,
        backend: object | None = None,
        reader: object | None = None,
        geometry_crs: int | None = 4326,
        method: str = "nearest",
        max_distance_pixels: int = 0,
        return_neighbourhood: bool = False,
    ) -> pa.Table:
        """Sample point values for this record.

        Parameters
        ----------
        points : pa.Array
            GeoArrow-native point array.
        band_codes : list of str
            Band codes to sample.
        point_indices : list of int, optional
            Absolute point indices corresponding to *points*. When omitted,
            emitted rows use the local point positions.
        max_concurrent : int
            Maximum concurrent HTTP requests.
        backend : object, optional
            Pluggable I/O backend.
        reader : COGReader, optional
            Active shared COG reader for connection/session reuse across
            records. When omitted, this method creates and owns a reader.
        geometry_crs : int, optional
            CRS EPSG code of input points. Defaults to EPSG:4326.
        method : str
            Sampling method. Only ``"nearest"`` is currently supported.
        max_distance_pixels : int
            Maximum pixel distance for nodata fallback search, measured in
            Chebyshev distance (square rings). When the nearest pixel is nodata
            and this is > 0, Rasteret searches outward in square rings up to
            this distance and picks the closest candidate by exact
            point-to-pixel-rectangle distance. ``0`` disables fallback and
            returns the nearest pixel value as-is.
        return_neighbourhood : bool
            If ``True``, include a ``neighbourhood_values`` column with the full
            square neighborhood centered on the base pixel under each point.
            The list is row-major and has length
            ``(2 * max_distance_pixels + 1) ** 2``.

        Returns
        -------
        pyarrow.Table
            One row per ``(point, band)`` sample for this record.
        """
        if method != "nearest":
            raise ValueError("Only nearest point sampling is supported currently.")
        if max_distance_pixels < 0:
            raise ValueError("max_distance_pixels must be >= 0")

        output_schema = (
            POINT_SAMPLES_NEIGHBORHOOD_SCHEMA
            if return_neighbourhood
            else POINT_SAMPLES_SCHEMA
        )

        type_name = getattr(points.type, "extension_name", "") or ""
        if "geoarrow.point" not in type_name:
            from rasteret.core.geometry import UnsupportedGeometryError

            raise UnsupportedGeometryError(
                "Point sampling requires Point geometries. "
                "Use get_xarray/get_numpy/get_gdf for Polygon/MultiPolygon AOIs."
            )

        import geoarrow.pyarrow as ga

        from rasteret.fetch.cog import COGReader, COGTileRequest

        transformer = None
        if (
            geometry_crs is not None
            and self.crs is not None
            and geometry_crs != self.crs
        ):
            from pyproj import Transformer

            transformer = Transformer.from_crs(geometry_crs, self.crs, always_xy=True)

        record_datetime_us: np.datetime64 | None = None
        if self.datetime is not None:
            try:
                record_datetime_us = np.datetime64(
                    pd.Timestamp(self.datetime).to_datetime64(), "us"
                )
            except (OverflowError, TypeError, ValueError):
                logger.debug("Could not normalize record datetime for %s", self.id)

        point_crs_value = int(geometry_crs) if geometry_crs is not None else None
        raster_crs_value = int(self.crs) if self.crs is not None else None
        cloud_cover_value = (
            float(self.cloud_cover) if self.cloud_cover is not None else None
        )
        point_indices_arr: np.ndarray | None = None
        point_xs_arr, point_ys_arr = ga.point_coords(points)
        point_xs = point_xs_arr.to_numpy(zero_copy_only=False)
        point_ys = point_ys_arr.to_numpy(zero_copy_only=False)
        if point_indices is not None:
            point_indices_arr = np.asarray(point_indices, dtype=np.int64)
            if len(point_indices_arr) != len(point_xs):
                raise ValueError("point_indices length must match the number of points")
        sample_xs, sample_ys = point_xs, point_ys
        if transformer is not None:
            sample_xs, sample_ys = transformer.transform(point_xs, point_ys)

        def _constant_int32_array(value: int | None, row_count: int) -> pa.Array:
            if value is None:
                return pa.nulls(row_count, type=pa.int32())
            return pa.array(np.full(row_count, value, dtype=np.int32), type=pa.int32())

        def _constant_float64_array(value: float | None, row_count: int) -> pa.Array:
            if value is None:
                return pa.nulls(row_count, type=pa.float64())
            return pa.array(
                np.full(row_count, value, dtype=np.float64), type=pa.float64()
            )

        def _constant_timestamp_array(
            value: np.datetime64 | None, row_count: int
        ) -> pa.Array:
            if value is None:
                return pa.nulls(row_count, type=pa.timestamp("us"))
            return pa.array(
                np.full(row_count, value, dtype="datetime64[us]"),
                type=pa.timestamp("us"),
            )

        def _constant_string_array(value: str, row_count: int) -> pa.Array:
            return pa.array(np.full(row_count, value, dtype=object), type=pa.string())

        band_sources: list[dict[str, object]] = []
        for band_code in band_codes:
            cog_meta, url, band_index = self.try_get_band_cog_metadata(band_code)
            if cog_meta is None or url is None or cog_meta.transform is None:
                raise ValueError(
                    f"Missing band metadata or href for band '{band_code}' "
                    f"in record '{self.id}'"
                )

            scale_x, trans_x, scale_y, trans_y = normalize_transform(cog_meta.transform)
            src_transform = Affine(
                float(scale_x),
                0.0,
                float(trans_x),
                0.0,
                float(scale_y),
                float(trans_y),
            )
            band_sources.append(
                {
                    "band_code": band_code,
                    "metadata": cog_meta,
                    "url": url,
                    "band_index": band_index,
                    "transform": src_transform,
                    "group_key": (
                        url,
                        tuple(float(value) for value in cog_meta.transform),
                        int(cog_meta.width),
                        int(cog_meta.height),
                        int(cog_meta.tile_width),
                        int(cog_meta.tile_height),
                        str(np.dtype(cog_meta.dtype)),
                        int(getattr(cog_meta, "samples_per_pixel", 1) or 1),
                        int(getattr(cog_meta, "planar_configuration", 1) or 1),
                        # Planar TIFFs store each band in its own IFD (with distinct
                        # tile offsets). Grouping across band_index would incorrectly
                        # read band 0's tiles for every band.
                        int(band_index)
                        if int(getattr(cog_meta, "planar_configuration", 1) or 1) == 2
                        else None,
                    ),
                }
            )

        async def _sample_with_reader(shared_reader: COGReader) -> pa.Table:
            record_batches: list[pa.RecordBatch] = []
            grouped_sources: dict[object, list[dict[str, object]]] = {}
            for source in band_sources:
                grouped_sources.setdefault(source["group_key"], []).append(source)

            neighborhood_row_offsets: np.ndarray | None = None
            neighborhood_col_offsets: np.ndarray | None = None
            neighborhood_size = 0
            if return_neighbourhood:
                neighborhood_row_offsets, neighborhood_col_offsets = (
                    square_neighborhood_offsets(max_distance_pixels)
                )
                neighborhood_size = int(neighborhood_row_offsets.size)

            for source_group in grouped_sources.values():
                # Phase 1: map -> pixel -> tile grouping.
                # This is the key point-sampling-specific planning step that lets us
                # fetch each tile once and then do fast local indexing.
                first_source = source_group[0]
                cog_meta = first_source["metadata"]
                url = str(first_source["url"])
                src_transform = first_source["transform"]

                (
                    point_rows,
                    point_cols,
                    point_row_fs,
                    point_col_fs,
                    tile_groups,
                    tiles,
                ) = plan_point_tile_groups(
                    np.asarray(sample_xs, dtype=np.float64),
                    np.asarray(sample_ys, dtype=np.float64),
                    scale_x=float(src_transform.a),
                    translate_x=float(src_transform.c),
                    scale_y=float(src_transform.e),
                    translate_y=float(src_transform.f),
                    raster_height=int(cog_meta.height),
                    raster_width=int(cog_meta.width),
                    tile_height=int(cog_meta.tile_height),
                    tile_width=int(cog_meta.tile_width),
                )
                if not tiles:
                    continue

                # Phase 2: fetch base tiles and sample nearest pixel per point.
                source_values: list[np.ndarray] = [
                    np.full(len(point_xs), np.nan, dtype=np.float64)
                    for _ in source_group
                ]
                source_sampled_masks: list[np.ndarray] = [
                    np.zeros(len(point_xs), dtype=bool) for _ in source_group
                ]
                source_tile_cache: list[dict[tuple[int, int], np.ndarray]] = [
                    {} for _ in source_group
                ]
                source_neighborhood_values: list[np.ndarray | None] = [
                    (
                        np.full(
                            (len(point_xs), neighborhood_size),
                            np.nan,
                            dtype=np.float64,
                        )
                        if return_neighbourhood
                        else None
                    )
                    for _ in source_group
                ]

                async def _cache_candidate_tiles(
                    *,
                    candidate_rows: np.ndarray,
                    candidate_cols: np.ndarray,
                    in_bounds: np.ndarray,
                    tile_cache: dict[tuple[int, int], np.ndarray],
                    source_meta: CogMetadata,
                    source_band_index: int,
                    tiles_x: int,
                    tiles_y: int,
                ) -> None:
                    unique_tile_pairs = unique_tile_pairs_for_candidates(
                        candidate_rows,
                        candidate_cols,
                        in_bounds,
                        tile_height=int(source_meta.tile_height),
                        tile_width=int(source_meta.tile_width),
                    )
                    if not unique_tile_pairs.size:
                        return

                    extra_requests: list[COGTileRequest] = []
                    for (
                        tile_row,
                        tile_col,
                        offset,
                        size,
                    ) in tile_request_specs_for_pairs(
                        unique_tile_pairs,
                        source_meta.tile_offsets,
                        source_meta.tile_byte_counts,
                        tiles_x=tiles_x,
                        tiles_y=tiles_y,
                    ):
                        if (tile_row, tile_col) in tile_cache:
                            continue
                        extra_requests.append(
                            COGTileRequest(
                                url=url,
                                offset=offset,
                                size=size,
                                row=tile_row,
                                col=tile_col,
                                metadata=source_meta,
                                band_index=source_band_index,
                            )
                        )

                    if not extra_requests:
                        return

                    extra_map = await shared_reader.read_merged_tiles(extra_requests)
                    for (
                        tile_row,
                        tile_col,
                        band_index,
                    ), tile_data in extra_map.items():
                        if band_index != source_band_index:
                            continue
                        tile_cache[(tile_row, tile_col)] = tile_data

                # Batch tile reads to keep request lists bounded. `read_merged_tiles()`
                # works on (tile-offset, byte-count) ranges and materializes whole
                # tiles, so the main limiter here is the size of the request list,
                # not per-pixel reads.
                #
                # Use `max_concurrent` as a proxy for how much outstanding I/O we
                # can efficiently schedule, and cap to avoid huge batches that
                # increase range-merging overhead.
                requests_per_tile = max(1, len(source_group))
                max_requests_per_batch = max(64, min(4096, int(max_concurrent) * 8))
                tiles_per_batch = max(1, max_requests_per_batch // requests_per_tile)
                for start in range(0, len(tiles), tiles_per_batch):
                    batch = tiles[start : start + tiles_per_batch]
                    sample_requests: list[COGTileRequest] = []
                    for source in source_group:
                        source_meta = source["metadata"]
                        source_band_index = source["band_index"]
                        tile_offsets = source_meta.tile_offsets
                        tile_byte_counts = source_meta.tile_byte_counts
                        if tile_offsets is None or tile_byte_counts is None:
                            continue

                        tiles_x, tiles_y = tile_grid_shape(
                            raster_width=int(source_meta.width),
                            raster_height=int(source_meta.height),
                            tile_width=int(source_meta.tile_width),
                            tile_height=int(source_meta.tile_height),
                        )

                        for (
                            tile_row,
                            tile_col,
                            offset,
                            size,
                        ) in tile_request_specs_for_pairs(
                            batch,
                            tile_offsets,
                            tile_byte_counts,
                            tiles_x=tiles_x,
                            tiles_y=tiles_y,
                        ):
                            sample_requests.append(
                                COGTileRequest(
                                    url=url,
                                    offset=offset,
                                    size=size,
                                    row=tile_row,
                                    col=tile_col,
                                    metadata=source_meta,
                                    band_index=source_band_index,
                                )
                            )

                    tile_arrays_map = (
                        await shared_reader.read_merged_tiles(sample_requests)
                        if sample_requests
                        else {}
                    )

                    for tile_row, tile_col in batch:
                        tile_arrays: list[np.ndarray] = []
                        missing_band = False
                        for source_idx, source in enumerate(source_group):
                            band_index = source["band_index"]
                            tile_data = tile_arrays_map.get(
                                (tile_row, tile_col, band_index)
                            )
                            if tile_data is None:
                                missing_band = True
                                break
                            tile_arrays.append(tile_data)
                            source_tile_cache[source_idx][(tile_row, tile_col)] = (
                                tile_data
                            )
                        if missing_band or not tile_arrays:
                            continue

                        sample_matrix = tile_groups.get((tile_row, tile_col))
                        if sample_matrix is None or sample_matrix.size == 0:
                            continue
                        row_start = tile_row * int(cog_meta.tile_height)
                        col_start = tile_col * int(cog_meta.tile_width)

                        local_point_indices = sample_matrix[:, 0]
                        local_rows = sample_matrix[:, 1] - row_start
                        local_cols = sample_matrix[:, 2] - col_start

                        for source_idx, tile_data in enumerate(tile_arrays):
                            values = np.asarray(
                                tile_data[local_rows, local_cols], dtype=np.float64
                            )
                            source_values[source_idx][local_point_indices] = values
                            source_sampled_masks[source_idx][local_point_indices] = True

                if max_distance_pixels > 0:
                    # Phase 3: optional nodata fallback (nearest non-nodata pixel).
                    # For points whose nearest sample is nodata, search outward in
                    # Chebyshev rings up to `max_distance_pixels`. Within the first ring
                    # that yields valid candidates, pick the candidate with minimum
                    # exact distance from the query point to the candidate pixel
                    # footprint (scaled by pixel width/height).
                    #
                    # This mirrors PostGIS' ST_NearestValue behavior (perimeter scan
                    # of expanding extents + minimum point-to-pixel distance), but
                    # we cap the search radius to avoid unbounded remote I/O.
                    for source_idx, source in enumerate(source_group):
                        source_meta = source["metadata"]
                        source_band_index = source["band_index"]
                        nodata_value = getattr(source_meta, "nodata", None)
                        if (
                            source_meta.tile_offsets is None
                            or source_meta.tile_byte_counts is None
                        ):
                            continue

                        unresolved = source_sampled_masks[source_idx] & nodata_mask(
                            source_values[source_idx], nodata_value
                        )
                        if not np.any(unresolved):
                            continue

                        tile_height = int(source_meta.tile_height)
                        tile_width = int(source_meta.tile_width)
                        raster_height = int(source_meta.height)
                        raster_width = int(source_meta.width)
                        pixel_width = float(src_transform.a)
                        pixel_height = float(src_transform.e)
                        tiles_x, tiles_y = tile_grid_shape(
                            raster_width=raster_width,
                            raster_height=raster_height,
                            tile_width=tile_width,
                            tile_height=tile_height,
                        )
                        tile_cache = source_tile_cache[source_idx]

                        for radius in range(1, max_distance_pixels + 1):
                            if not np.any(unresolved):
                                break
                            row_offsets, col_offsets = chebyshev_ring_offsets(radius)
                            if row_offsets.size == 0:
                                continue

                            unresolved_indices = np.nonzero(unresolved)[0]
                            candidate_rows, candidate_cols, in_bounds = (
                                ring_candidate_grid(
                                    point_rows[unresolved_indices],
                                    point_cols[unresolved_indices],
                                    row_offsets,
                                    col_offsets,
                                    raster_height=raster_height,
                                    raster_width=raster_width,
                                )
                            )
                            if not np.any(in_bounds):
                                continue

                            await _cache_candidate_tiles(
                                candidate_rows=candidate_rows,
                                candidate_cols=candidate_cols,
                                in_bounds=in_bounds,
                                tile_cache=tile_cache,
                                source_meta=source_meta,
                                source_band_index=source_band_index,
                                tiles_x=tiles_x,
                                tiles_y=tiles_y,
                            )

                            candidate_values, candidate_sampled = (
                                candidate_values_from_tile_cache(
                                    candidate_rows,
                                    candidate_cols,
                                    in_bounds,
                                    tile_cache=tile_cache,
                                    tile_height=tile_height,
                                    tile_width=tile_width,
                                    tiles_x=tiles_x,
                                )
                            )
                            valid_candidates = candidate_sampled & ~nodata_mask(
                                candidate_values, nodata_value
                            )
                            if not np.any(valid_candidates):
                                continue

                            distance_sq = pixel_rect_distance_sq(
                                point_row_fs[unresolved_indices],
                                point_col_fs[unresolved_indices],
                                candidate_rows,
                                candidate_cols,
                                pixel_width=pixel_width,
                                pixel_height=pixel_height,
                            )
                            distance_sq = np.where(
                                valid_candidates, distance_sq, np.inf
                            )
                            best_offset = np.argmin(distance_sq, axis=1)
                            best_distance_sq = distance_sq[
                                np.arange(unresolved_indices.size),
                                best_offset,
                            ]
                            resolved_here = np.isfinite(best_distance_sq)
                            if not np.any(resolved_here):
                                continue

                            accepted_indices = unresolved_indices[resolved_here]
                            accepted_offsets = best_offset[resolved_here]
                            source_values[source_idx][accepted_indices] = (
                                candidate_values[
                                    resolved_here,
                                    accepted_offsets,
                                ]
                            )
                            unresolved[accepted_indices] = False

                if return_neighbourhood:
                    assert neighborhood_row_offsets is not None
                    assert neighborhood_col_offsets is not None
                    for source_idx, source in enumerate(source_group):
                        local_point_indices = np.nonzero(
                            source_sampled_masks[source_idx]
                        )[0]
                        if local_point_indices.size == 0:
                            continue

                        source_meta = source["metadata"]
                        source_band_index = source["band_index"]
                        if (
                            source_meta.tile_offsets is None
                            or source_meta.tile_byte_counts is None
                        ):
                            continue

                        tile_height = int(source_meta.tile_height)
                        tile_width = int(source_meta.tile_width)
                        raster_height = int(source_meta.height)
                        raster_width = int(source_meta.width)
                        tiles_x, tiles_y = tile_grid_shape(
                            raster_width=raster_width,
                            raster_height=raster_height,
                            tile_width=tile_width,
                            tile_height=tile_height,
                        )
                        tile_cache = source_tile_cache[source_idx]

                        candidate_rows, candidate_cols, in_bounds = ring_candidate_grid(
                            point_rows[local_point_indices],
                            point_cols[local_point_indices],
                            neighborhood_row_offsets,
                            neighborhood_col_offsets,
                            raster_height=raster_height,
                            raster_width=raster_width,
                        )

                        await _cache_candidate_tiles(
                            candidate_rows=candidate_rows,
                            candidate_cols=candidate_cols,
                            in_bounds=in_bounds,
                            tile_cache=tile_cache,
                            source_meta=source_meta,
                            source_band_index=source_band_index,
                            tiles_x=tiles_x,
                            tiles_y=tiles_y,
                        )

                        neighborhood_values, _ = candidate_values_from_tile_cache(
                            candidate_rows,
                            candidate_cols,
                            in_bounds,
                            tile_cache=tile_cache,
                            tile_height=tile_height,
                            tile_width=tile_width,
                            tiles_x=tiles_x,
                        )
                        neighborhood_store = source_neighborhood_values[source_idx]
                        assert neighborhood_store is not None
                        neighborhood_store[local_point_indices] = neighborhood_values

                # Phase 4: build Arrow record batches: one row per (point, band).
                for source_idx, source in enumerate(source_group):
                    local_point_indices = np.nonzero(source_sampled_masks[source_idx])[
                        0
                    ]
                    if local_point_indices.size == 0:
                        continue
                    output_point_indices = (
                        point_indices_arr[local_point_indices]
                        if point_indices_arr is not None
                        else local_point_indices
                    )
                    point_x_values = point_xs[local_point_indices]
                    point_y_values = point_ys[local_point_indices]
                    row_count = int(local_point_indices.size)
                    point_index_arr = pa.array(output_point_indices, type=pa.int64())
                    point_x_arr = pa.array(
                        np.asarray(point_x_values, dtype=np.float64),
                        type=pa.float64(),
                    )
                    point_y_arr = pa.array(
                        np.asarray(point_y_values, dtype=np.float64),
                        type=pa.float64(),
                    )
                    point_crs_arr = _constant_int32_array(point_crs_value, row_count)
                    record_id_arr = _constant_string_array(str(self.id), row_count)
                    datetime_arr = _constant_timestamp_array(
                        record_datetime_us, row_count
                    )
                    collection_arr = _constant_string_array(
                        str(self.collection), row_count
                    )
                    cloud_cover_arr = _constant_float64_array(
                        cloud_cover_value, row_count
                    )
                    raster_crs_arr = _constant_int32_array(raster_crs_value, row_count)
                    batch_columns = {
                        "point_index": point_index_arr,
                        "point_x": point_x_arr,
                        "point_y": point_y_arr,
                        "point_crs": point_crs_arr,
                        "record_id": record_id_arr,
                        "datetime": datetime_arr,
                        "collection": collection_arr,
                        "cloud_cover": cloud_cover_arr,
                        "band": _constant_string_array(
                            str(source["band_code"]), row_count
                        ),
                        "value": pa.array(
                            source_values[source_idx][local_point_indices],
                            type=pa.float64(),
                        ),
                        "raster_crs": raster_crs_arr,
                    }
                    if return_neighbourhood:
                        neighborhood_store = source_neighborhood_values[source_idx]
                        assert neighborhood_store is not None
                        batch_columns["neighbourhood_values"] = pa.array(
                            neighborhood_store[local_point_indices].tolist(),
                            type=pa.list_(pa.float64()),
                        )
                    record_batches.append(
                        pa.record_batch(
                            [batch_columns[field.name] for field in output_schema],
                            schema=output_schema,
                        )
                    )

            if not record_batches:
                return (
                    empty_point_samples_neighborhood_table()
                    if return_neighbourhood
                    else empty_point_samples_table()
                )
            return pa.Table.from_batches(record_batches, schema=output_schema)

        if reader is not None:
            return await _sample_with_reader(reader)

        async with COGReader(max_concurrent=max_concurrent, backend=backend) as owned:
            return await _sample_with_reader(owned)

    def _reproject_band_results(
        self,
        results: list[dict],
        target_crs: int,
    ) -> list[dict]:
        """Reproject band results from source CRS to *target_crs*."""
        from rasteret.core.utils import (
            compute_dst_grid_from_src,
            reproject_array,
        )

        reprojected = []
        for r in results:
            data = r["data"]
            src_tf = r["transform"]
            h, w = data.shape

            xmin = float(src_tf.c)
            ymax = float(src_tf.f)
            xmax = xmin + w * float(src_tf.a)
            ymin = ymax + h * float(src_tf.e)

            src_bounds = (
                min(xmin, xmax),
                min(ymin, ymax),
                max(xmin, xmax),
                max(ymin, ymax),
            )
            dst_tf, dst_shape = compute_dst_grid_from_src(
                self.crs,
                target_crs,
                w,
                h,
                src_bounds,
            )
            reprojected_data = reproject_array(
                data,
                src_tf,
                self.crs,
                target_crs,
                dst_tf,
                dst_shape,
            )
            reprojected.append(
                {
                    "data": reprojected_data,
                    "transform": dst_tf,
                    "band": r["band"],
                }
            )
        return reprojected

    async def load_bands(
        self,
        geometries: pa.Array,
        band_codes: list[str],
        max_concurrent: int = 50,
        for_xarray: bool = True,
        for_numpy: bool = False,
        progress: bool = False,
        backend: object | None = None,
        target_crs: int | None = None,
        geometry_crs: int | None = 4326,
        all_touched: bool = False,
    ):
        """Load bands for all geometries with parallel processing.

        Parameters
        ----------
        geometries : pa.Array
            GeoArrow native array of areas of interest.
        band_codes : list of str
            Band codes to load.
        max_concurrent : int
            Maximum concurrent HTTP requests.
        for_xarray : bool
            If ``True``, return ``xr.Dataset``; otherwise ``gpd.GeoDataFrame``.
        for_numpy : bool
            If ``True``, return raw per-geometry band results for NumPy assembly
            without constructing GeoPandas objects.
        backend : object, optional
            Pluggable I/O backend.
        target_crs : int, optional
            Reproject results to this CRS.
        geometry_crs : int, optional
            CRS of the *geometries* input (default EPSG:4326).
        all_touched : bool
            Passed through to polygon masking behavior. ``False`` matches
            rasterio default semantics.

        Returns
        -------
        xarray.Dataset or geopandas.GeoDataFrame
            Data is returned in the native COG dtype (e.g. ``uint16``,
            ``int8``, ``float32``). Integer arrays promote to ``float32``
            only when geometry masking requires NaN and no nodata value is
            declared in the COG metadata.
        """
        from rasteret.fetch.cog import COGReader

        if for_xarray and for_numpy:
            raise ValueError(
                "load_bands() cannot request xarray and numpy outputs together"
            )

        n_geoms = len(geometries)
        logger.debug(f"Loading {len(band_codes)} bands for {n_geoms} geometries")

        geom_progress = None
        if progress:
            geom_progress = tqdm(total=n_geoms, desc=f"Record {self.id}")

        async with COGReader(max_concurrent=max_concurrent, backend=backend) as reader:

            async def process_geometry(geom_idx: int, geom_id: int):
                band_progress = None
                if progress:
                    band_progress = tqdm(
                        total=len(band_codes), desc=f"Geom {geom_id}", leave=False
                    )

                band_tasks = []
                for band_code in band_codes:
                    task = self._load_single_band(
                        geometries,
                        geom_idx,
                        band_code,
                        max_concurrent,
                        reader=reader,
                        geometry_crs=geometry_crs,
                        all_touched=all_touched,
                    )
                    band_tasks.append(task)

                raw_results = await asyncio.gather(*band_tasks, return_exceptions=True)
                results = []
                first_error: BaseException | None = None
                failed_band_codes: list[str] = []
                for band_code, r in zip(band_codes, raw_results):
                    if isinstance(r, Exception):
                        from rasteret.core.geometry import UnsupportedGeometryError

                        if isinstance(r, UnsupportedGeometryError):
                            # Deterministic user input issue: fail fast.
                            raise UnsupportedGeometryError(
                                "Unsupported geometry type for Rasteret sampling "
                                f"(record_id='{self.id}', geometry_index={geom_id}). "
                                "Rasteret currently supports Polygon and MultiPolygon geometries "
                                "for masking-based sampling via get_xarray/get_numpy/get_gdf. "
                                "Point sampling is not supported yet."
                            ) from r
                        failed_band_codes.append(band_code)
                        if first_error is None:
                            first_error = r
                        logger.error(
                            "Band load failed (record_id=%s, geometry_index=%s, band=%s): %s",
                            self.id,
                            geom_id,
                            band_code,
                            r,
                        )
                    else:
                        results.append(r)
                if band_progress is not None:
                    band_progress.update(len(band_codes))
                    band_progress.close()
                if geom_progress is not None:
                    geom_progress.update(1)

                valid = [r for r in results if r is not None]
                if not valid and first_error is not None:
                    from rasteret.core.geometry import UnsupportedGeometryError

                    if isinstance(first_error, UnsupportedGeometryError):
                        raise UnsupportedGeometryError(
                            f"Unsupported geometry type for Rasteret sampling "
                            f"(record_id='{self.id}', geometry_index={geom_id}). "
                            "Rasteret currently supports Polygon and MultiPolygon geometries "
                            "for masking-based sampling via get_xarray/get_numpy/get_gdf. "
                            "Point sampling is not supported yet."
                        ) from first_error
                    raise RuntimeError(
                        "All band reads failed for the requested geometry "
                        f"(record_id='{self.id}', geometry_index={geom_id}). "
                        "See the chained exception for the first failure."
                    ) from first_error
                if target_crs is not None and target_crs != self.crs and valid:
                    valid = self._reproject_band_results(valid, target_crs)
                return valid, geom_id, failed_band_codes, first_error

            # Process geometries concurrently with semaphore
            sem = asyncio.Semaphore(max_concurrent)

            async def bounded_process(geom_idx: int, geom_id: int):
                async with sem:
                    return await process_geometry(geom_idx, geom_id)

            tasks = [bounded_process(idx, idx + 1) for idx in range(n_geoms)]
            raw_geom_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[tuple[list[dict], int]] = []
        first_error: BaseException | None = None
        geom_failures = 0
        partial_band_failures: list[tuple[int, list[str], BaseException | None]] = []
        for r in raw_geom_results:
            if isinstance(r, Exception):
                from rasteret.core.geometry import UnsupportedGeometryError

                if isinstance(r, UnsupportedGeometryError):
                    # Geometry-type errors are deterministic user input issues.
                    # Fail fast so they do not become a misleading
                    # "No valid data found" downstream.
                    raise r
                geom_failures += 1
                if first_error is None:
                    first_error = r
                logger.error("Geometry processing failed: %s", r)
            else:
                band_results, geom_id, failed_band_codes, band_first_error = r
                results.append((band_results, geom_id))
                if failed_band_codes:
                    partial_band_failures.append(
                        (geom_id, failed_band_codes, band_first_error)
                    )

        if geom_progress is not None:
            geom_progress.close()

        if not results and first_error is not None:
            raise RuntimeError(
                f"All geometry reads failed for record_id='{self.id}'. "
                "See the chained exception for the first failure."
            ) from first_error
        if results and (geom_failures or partial_band_failures):
            parts = []
            if geom_failures:
                parts.append(f"{geom_failures}/{n_geoms} geometry task(s) failed")
            if partial_band_failures:
                n_failed_bands = sum(
                    len(bands) for _gid, bands, _err in partial_band_failures
                )
                parts.append(
                    f"{n_failed_bands} band read(s) failed across {len(partial_band_failures)} geometries"
                )

            first_detail = None
            if geom_failures and first_error is not None:
                first_detail = f"first geometry failure: {first_error}"
            elif partial_band_failures:
                gid, bands, err = partial_band_failures[0]
                band = bands[0] if bands else "<unknown>"
                if err is not None:
                    first_detail = f"first band failure: geometry_index={gid}, band='{band}': {err}"
                else:
                    first_detail = (
                        f"first band failure: geometry_index={gid}, band='{band}'"
                    )

            msg = (
                f"Partial read failures for record_id='{self.id}': "
                + "; ".join(parts)
                + (f"; {first_detail}" if first_detail else "")
                + "."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        # Process results
        if for_numpy:
            return results
        if for_xarray:
            return self._merge_xarray_results(results, target_crs=target_crs)
        else:
            return self._merge_geodataframe_results(
                results,
                geometries,
                geometry_crs=geometry_crs,
                target_crs=target_crs,
            )

    @staticmethod
    def _write_crs_cf(ds_or_da, epsg_code, transform=None):
        """Attach CRS as a CF-convention ``spatial_ref`` coordinate.

        Uses ``pyproj`` directly (no rioxarray dependency).  Writes:
        - CF grid-mapping attributes via ``CRS.to_cf()``
        - WKT2 (ISO 19162:2019) in ``crs_wkt``
        - PROJJSON for forward-compatible consumers
        - GDAL-style ``GeoTransform`` string when *transform* is given
        """
        import xarray as xr
        from pyproj import CRS

        crs_obj = CRS.from_epsg(epsg_code)
        cf_attrs = crs_obj.to_cf()
        cf_attrs["crs_wkt"] = crs_obj.to_wkt(version="WKT2_2019")
        cf_attrs["projjson"] = crs_obj.to_json()
        if transform is not None:
            cf_attrs["GeoTransform"] = (
                f"{transform.c} {transform.a} {transform.b} "
                f"{transform.f} {transform.d} {transform.e}"
            )

        spatial_ref = xr.DataArray(0, attrs=cf_attrs)
        ds_or_da = ds_or_da.assign_coords(spatial_ref=spatial_ref)

        if hasattr(ds_or_da, "data_vars"):
            for var in ds_or_da.data_vars:
                ds_or_da[var].encoding["grid_mapping"] = "spatial_ref"
        else:
            ds_or_da.encoding["grid_mapping"] = "spatial_ref"
        return ds_or_da

    def _merge_xarray_results(
        self,
        results: list[tuple[list[dict], int]],
        target_crs: int | None = None,
    ):
        """Merge results into xarray Dataset."""
        import xarray as xr

        data_arrays = []

        for band_results, geom_id in results:
            if not band_results:
                continue

            geom_arrays = []
            for band_result in band_results:
                h, w = band_result["data"].shape
                transform = band_result["transform"]
                da = xr.DataArray(
                    data=band_result["data"],
                    dims=["y", "x"],
                    coords={
                        "y": transform.f + (np.arange(h) + 0.5) * transform.e,
                        "x": transform.c + (np.arange(w) + 0.5) * transform.a,
                    },
                    name=band_result["band"],
                )
                crs_out = target_crs if target_crs is not None else self.crs
                da = self._write_crs_cf(da, crs_out, transform=transform)
                geom_arrays.append(da)

            if geom_arrays:
                ds = xr.merge(geom_arrays, compat="override")
                # Strip timezone for xarray compat (numpy can't merge tz-aware datetime64)
                ts = (
                    pd.Timestamp(self.datetime).tz_localize(None)
                    if hasattr(self.datetime, "tzinfo") and self.datetime.tzinfo
                    else pd.Timestamp(self.datetime)
                )
                ds = ds.expand_dims({"time": [ts], "geometry": [geom_id]})
                # CRS is already written per-DataArray above (line 445) and
                # propagates through xr.merge.  Do NOT re-write here: the
                # loop variable `transform` holds the *last* band's value,
                # which is wrong for mixed-resolution band sets.
                ds.attrs.update(
                    {
                        "crs": crs_out,
                        "geometry_id": geom_id,
                        "record_id": self.id,
                        "datetime": self.datetime,
                        "cloud_cover": self.cloud_cover,
                        "collection": self.collection,
                    }
                )
                data_arrays.append(ds)

        if not data_arrays:
            return xr.Dataset()

        return xr.merge(data_arrays, compat="override")

    def _merge_geodataframe_results(
        self,
        results: list[tuple[list[dict], int]],
        geometries: pa.Array,
        *,
        geometry_crs: int | None,
        target_crs: int | None,
    ) -> gpd.GeoDataFrame:
        """Merge results into GeoDataFrame."""
        import shapely

        from rasteret.core.geometry import to_rasterio_geojson, transform_coords

        out_crs = target_crs if target_crs is not None else geometry_crs
        if out_crs is None:
            out_crs = 4326

        rows: list[dict] = []

        for band_results, geom_id in results:
            if not band_results:
                continue

            # Convert GeoArrow geometry to GeoJSON dict at output boundary
            if geometry_crs is not None and out_crs != geometry_crs:
                geojson = transform_coords(
                    geometries, geom_id - 1, geometry_crs, out_crs
                )
            else:
                geojson = to_rasterio_geojson(geometries, geom_id - 1)
            geom_obj = shapely.geometry.shape(geojson)

            for band_result in band_results:
                rows.append(
                    {
                        "record_id": self.id,
                        "datetime": self.datetime,
                        "cloud_cover": self.cloud_cover,
                        "collection": self.collection,
                        "geometry": geom_obj,
                        "band": band_result["band"],
                        "data": band_result["data"],
                    }
                )

        if not rows:
            empty_geometry = gpd.GeoSeries([], name="geometry", crs=f"EPSG:{out_crs}")
            return gpd.GeoDataFrame(geometry=empty_geometry, crs=f"EPSG:{out_crs}")

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{out_crs}")

    def __dir__(self) -> list[str]:
        names = super().__dir__()
        return sorted(
            name
            for name in names
            if (name.startswith("__") and name.endswith("__"))
            or not name.startswith("_")
        )
