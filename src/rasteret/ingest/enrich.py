# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""COG header enrichment: shared logic for adding ``{band}_metadata`` columns.

Both :class:`StacCollectionBuilder` and :class:`RecordTableBuilder`
converge here after extracting per-band URLs from their source-specific
formats.  The two entry points are:

- :func:`add_band_metadata_columns`: append struct columns from
  already-parsed results (used when the caller drives parsing itself).
- :func:`enrich_table_with_cog_metadata`, full async pipeline:
  extract URLs, parse COG headers, append columns.
"""

from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa

from rasteret.cloud import StorageBackend
from rasteret.constants import COG_BAND_METADATA_STRUCT
from rasteret.fetch.header_parser import AsyncCOGHeaderParser
from rasteret.ingest.normalize import crs_code_from_epsg

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# URL extraction helpers
# ------------------------------------------------------------------


def slice_tile_tables_for_band(
    *,
    metadata: Any,
    band_index: int,
) -> tuple[list[int] | None, list[int] | None]:
    """Select per-band tile tables for planar separate multi-sample TIFFs.

    For tiled GeoTIFFs with PlanarConfiguration=2, GeoTIFF encodes
    TileOffsets/TileByteCounts as:

      [all tiles for sample 0] + [all tiles for sample 1] + ...

    Rasteret's reader expects one tile table per band. During enrichment we
    slice the shared table based on *band_index* so each ``{band}_metadata``
    column contains only the offsets for that band.
    """
    offsets = getattr(metadata, "tile_offsets", None)
    counts = getattr(metadata, "tile_byte_counts", None)
    if not offsets or not counts:
        return offsets, counts
    if len(offsets) != len(counts):
        raise ValueError(
            "Invalid tile metadata: TileOffsets/TileByteCounts length mismatch "
            f"({len(offsets)} vs {len(counts)})."
        )

    try:
        width = int(metadata.width)
        height = int(metadata.height)
        tile_width = int(metadata.tile_width)
        tile_height = int(metadata.tile_height)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tile grid metadata: {exc}") from exc

    if tile_width <= 0 or tile_height <= 0:
        return offsets, counts

    tiles_x = (width + tile_width - 1) // tile_width
    tiles_y = (height + tile_height - 1) // tile_height
    tiles_per_plane = tiles_x * tiles_y
    if tiles_per_plane <= 0:
        return offsets, counts

    if len(offsets) == tiles_per_plane:
        # Single-sample tiled GeoTIFF: nothing to slice.
        return offsets, counts

    if len(offsets) % tiles_per_plane != 0:
        # Unknown layout: don't guess.
        return offsets, counts

    samples = len(offsets) // tiles_per_plane
    if band_index < 0 or band_index >= samples:
        raise ValueError(f"band_index {band_index} out of range for {samples} samples")

    start = band_index * tiles_per_plane
    end = start + tiles_per_plane
    return offsets[start:end], counts[start:end]


def build_url_index_from_assets(
    table: pa.Table,
    band_codes: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build ``{record_id: {band_code: {url, band_index}}}`` from ``assets``.

    Parameters
    ----------
    table : pa.Table
        Must contain ``id`` and ``assets`` columns.
    band_codes : list of str, optional
        If given, only include these bands.  Otherwise include all.

    Returns
    -------
    dict
        Nested mapping of record ID -> band code -> asset reference dict.

        The asset reference dict contains:

        - ``url``: asset href (string)
        - ``band_index``: optional band/sample index within a multi-sample
          tiled GeoTIFF (defaults to 0). This is used during enrichment to
          select the correct TileOffsets/TileByteCounts segment for planar
          separate assets.
    """
    ids = table.column("id").to_pylist()
    assets_list = table.column("assets").to_pylist()
    url_index: dict[str, dict[str, dict[str, Any]]] = {}

    for record_id, assets in zip(ids, assets_list):
        if not assets or not isinstance(assets, dict):
            continue
        band_urls: dict[str, dict[str, Any]] = {}
        for key, asset in assets.items():
            if band_codes and key not in band_codes:
                continue
            if isinstance(asset, dict):
                href = asset.get("href")
                band_index = asset.get("band_index", 0)
            elif isinstance(asset, str):
                href = asset
                band_index = 0
            else:
                continue
            if href:
                try:
                    idx = int(band_index)
                except (TypeError, ValueError):
                    idx = 0
                band_urls[key] = {"url": href, "band_index": idx}
        if band_urls:
            url_index[record_id] = band_urls

    return url_index


# ------------------------------------------------------------------
# Column construction (sync, no I/O)
# ------------------------------------------------------------------


def add_band_metadata_columns(
    table: pa.Table,
    band_codes: list[str],
    processed_items: list[dict],
) -> pa.Table:
    """Append ``{band}_metadata`` struct columns from parsed COG headers.

    Parameters
    ----------
    table : pa.Table
        Arrow table with an ``id`` column.
    band_codes : list of str
        Band codes to create columns for.
    processed_items : list of dict
        Each dict must have ``record_id`` (record identifier), ``band``,
        and the twelve COG metadata fields (``width``, ``height``,
        ``tile_width``, etc.).

    Returns
    -------
    pa.Table
        Input table with ``{band}_metadata`` columns appended.
    """
    record_metadata: dict[str, dict[str, Any]] = {}
    for record_id in table.column("id").to_pylist():
        record_metadata[record_id] = {band: None for band in band_codes}

    for item in processed_items:
        record_id = item.get("record_id")
        if not record_id:
            continue
        band = item["band"]
        if record_id in record_metadata and band in record_metadata[record_id]:
            record_metadata[record_id][band] = {
                "image_width": item["width"],
                "image_height": item["height"],
                "tile_width": item["tile_width"],
                "tile_height": item["tile_height"],
                "dtype": item["dtype"],
                "transform": item.get("transform", []),
                "predictor": item["predictor"],
                "compression": item["compression"],
                "tile_offsets": item["tile_offsets"],
                "tile_byte_counts": item["tile_byte_counts"],
                "pixel_scale": item.get("pixel_scale", []),
                "tiepoint": item.get("tiepoint", []),
                "nodata": item.get("nodata"),
                "samples_per_pixel": item.get("samples_per_pixel", 1),
                "planar_configuration": item.get("planar_configuration", 1),
                "photometric": item.get("photometric"),
                "extra_samples": item.get("extra_samples"),
            }

    for band in band_codes:
        metadata_list = [
            record_metadata[id_][band] for id_ in table.column("id").to_pylist()
        ]
        table = table.append_column(
            f"{band}_metadata",
            pa.array(metadata_list, type=COG_BAND_METADATA_STRUCT),
        )

    return table


# ------------------------------------------------------------------
# Full enrichment pipeline (async, network I/O)
# ------------------------------------------------------------------


async def enrich_table_with_cog_metadata(
    table: pa.Table,
    url_index: dict[str, dict[str, dict[str, Any]]],
    band_codes: list[str],
    *,
    max_concurrent: int = 300,
    batch_size: int = 100,
    backend: StorageBackend | None = None,
) -> pa.Table:
    """Parse COG headers and add ``{band}_metadata`` columns.

    This is the high-level entry point for builders that have a
    ``url_index`` but have not yet parsed COG headers.

    Parameters
    ----------
    table : pa.Table
        Arrow table with an ``id`` column.
    url_index : dict
        Mapping of ``record_id`` -> ``band_code`` -> asset reference dict:
        ``{record_id: {band_code: {"url": str, "band_index": int}}}``.
    band_codes : list of str
        Band codes to create metadata columns for.
    max_concurrent : int
        Maximum concurrent HTTP connections.
    batch_size : int
        Batch size for COG header parsing.
    backend : StorageBackend, optional
        I/O backend for authenticated range reads during COG header
        parsing.  When omitted, uses the default auto-detecting backend.

    Returns
    -------
    pa.Table
        Table with ``{band}_metadata`` struct columns appended.
    """

    def _slice_tile_tables(
        *,
        metadata: Any,
        band_index: int,
    ) -> tuple[list[int] | None, list[int] | None]:
        return slice_tile_tables_for_band(metadata=metadata, band_index=band_index)

    # Flatten url_index for batch processing, deduping URLs while preserving all
    # (record_id, band_code) pairs that share the same asset URL.
    urls_to_process: list[str] = []
    url_mapping: dict[str, list[tuple[str, str, int]]] = {}

    for record_id, bands in url_index.items():
        for band_code, asset_ref in bands.items():
            url = asset_ref.get("url")
            if not url:
                continue
            band_index = asset_ref.get("band_index", 0)
            try:
                band_index_int = int(band_index)
            except (TypeError, ValueError):
                band_index_int = 0

            if url not in url_mapping:
                urls_to_process.append(url)
                url_mapping[url] = [(record_id, band_code, band_index_int)]
            else:
                url_mapping[url].append((record_id, band_code, band_index_int))

    if not urls_to_process:
        logger.warning("No URLs to process for COG enrichment")
        return table

    logger.info(
        "Parsing COG headers for %d band assets across %d records...",
        len(urls_to_process),
        len(url_index),
    )

    async with AsyncCOGHeaderParser(
        max_concurrent=max_concurrent,
        batch_size=batch_size,
        backend=backend,
    ) as cog_parser:
        metadata_results = await cog_parser.process_cog_headers_batch(urls_to_process)

    processed_items: list[dict] = []
    record_crs: dict[str, int] = {}
    for url, metadata in zip(urls_to_process, metadata_results):
        if not metadata:
            continue
        for record_id, band_code, band_index in url_mapping[url]:
            if getattr(metadata, "crs", None) is not None:
                crs_val = int(metadata.crs)  # type: ignore[arg-type]
                prev = record_crs.get(record_id)
                if prev is None:
                    record_crs[record_id] = crs_val
                elif prev != crs_val:
                    raise ValueError(
                        "Conflicting CRS values detected during enrichment for "
                        f"record '{record_id}' ({prev} vs {crs_val}). "
                        "Ensure all assets in a record share the same proj:epsg."
                    )
            offsets, counts = _slice_tile_tables(
                metadata=metadata, band_index=band_index
            )
            nodata_val = metadata.nodata
            if nodata_val is not None and nodata_val != nodata_val:
                nodata_val = float("nan")
            elif nodata_val is not None:
                nodata_val = float(nodata_val)
            processed_items.append(
                {
                    "record_id": record_id,
                    "band": band_code,
                    "width": metadata.width,
                    "height": metadata.height,
                    "tile_width": metadata.tile_width,
                    "tile_height": metadata.tile_height,
                    "dtype": str(metadata.dtype),
                    "transform": metadata.transform,
                    "predictor": metadata.predictor,
                    "compression": metadata.compression,
                    "tile_offsets": offsets,
                    "tile_byte_counts": counts,
                    "pixel_scale": metadata.pixel_scale,
                    "tiepoint": metadata.tiepoint,
                    "nodata": nodata_val,
                    "samples_per_pixel": metadata.samples_per_pixel,
                    "planar_configuration": metadata.planar_configuration,
                    "photometric": metadata.photometric,
                    "extra_samples": list(metadata.extra_samples)
                    if metadata.extra_samples
                    else None,
                }
            )

    logger.info(
        "Parsed %d/%d band assets successfully",
        len(processed_items),
        len(urls_to_process),
    )

    if not processed_items:
        url_sample = urls_to_process[0] if urls_to_process else ""
        hints: list[str] = [
            "verify the assets are tiled COGs (tiled TIFF with TileOffsets/TileByteCounts)",
            "reduce concurrency (max_concurrent) and retry if the host is throttling",
        ]
        if "blob.core.windows.net" in url_sample:
            hints.insert(
                0,
                "for Planetary Computer, ensure SAS signing is working (install rasteret[azure], consider PC_SDK_SUBSCRIPTION_KEY)",
            )
        if url_sample.startswith("s3://"):
            hints.insert(
                0,
                "for S3 requester-pays buckets, ensure AWS credentials are configured",
            )

        joined = "; ".join(hints)
        raise ValueError(
            "COG header enrichment failed for all assets in this build. "
            f"Common fixes: {joined}."
        )

    # Ensure CRS sidecars exist for read-time transforms and Arrow interop.
    #
    # Many record tables omit per-record CRS, but Rasteret's read path needs a
    # record CRS to transform WGS84 query geometries into raster CRS. When the
    # header parser extracted an EPSG code, we backfill legacy ``proj:epsg`` and
    # the Arrow-friendly row-level ``crs`` code for any null/missing values.
    if record_crs:
        ids = table.column("id").to_pylist()
        existing_epsg = (
            table.column("proj:epsg").to_pylist()
            if "proj:epsg" in table.schema.names
            else None
        )
        existing_crs = (
            table.column("crs").to_pylist() if "crs" in table.schema.names else None
        )
        epsg_values: list[int | None] = []
        crs_values: list[str | None] = []
        for i, record_id in enumerate(ids):
            header_epsg = record_crs.get(record_id)
            current_epsg = existing_epsg[i] if existing_epsg is not None else None
            if current_epsg is None:
                resolved_epsg = header_epsg
            else:
                try:
                    resolved_epsg = int(current_epsg)
                except (TypeError, ValueError):
                    resolved_epsg = header_epsg

            current_crs = existing_crs[i] if existing_crs is not None else None
            epsg_values.append(resolved_epsg)
            crs_values.append(
                crs_code_from_epsg(resolved_epsg)
                or (str(current_crs).strip() if current_crs is not None else None)
            )

        epsg_col = pa.array(epsg_values, type=pa.int32())
        if "proj:epsg" in table.schema.names:
            idx = table.schema.get_field_index("proj:epsg")
            table = table.set_column(idx, "proj:epsg", epsg_col)
        else:
            table = table.append_column("proj:epsg", epsg_col)

        crs_col = pa.array(crs_values, type=pa.string())
        if "crs" in table.schema.names:
            idx = table.schema.get_field_index("crs")
            table = table.set_column(idx, "crs", crs_col)
        else:
            table = table.append_column("crs", crs_col)

    return add_band_metadata_columns(table, band_codes, processed_items)
