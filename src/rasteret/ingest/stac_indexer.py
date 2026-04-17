# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""STAC collection indexer: STAC API / static catalog -> Collection.

Searches a STAC API (or traverses a static STAC catalog), parses COG
headers for tile metadata, and normalises results into the Collection
contract via the shared
:func:`~rasteret.ingest.normalize.build_collection_from_table` layer.

Static catalogs (``catalog.json`` files on S3 with no ``/search``
endpoint) are supported via ``pystac.Catalog.from_file()`` with
client-side bbox and date filtering.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa
import pystac_client
from pystac_client.exceptions import APIError

from rasteret.cloud import CloudConfig, StorageBackend, rewrite_url
from rasteret.constants import BandRegistry
from rasteret.core.geometry import geojson_dicts_to_wkb
from rasteret.fetch.header_parser import AsyncCOGHeaderParser
from rasteret.ingest.base import CollectionBuilder
from rasteret.ingest.enrich import add_band_metadata_columns, slice_tile_tables_for_band
from rasteret.ingest.normalize import build_collection_from_table
from rasteret.types import BoundingBox, DateRange

logger = logging.getLogger(__name__)


def _is_retryable_stac_api_error(exc: Exception) -> bool:
    """Return ``True`` for transient STAC API failures worth retrying."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    transient_markers = (
        "maximum allowed time",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "try again",
        "service unavailable",
    )
    return any(marker in message for marker in transient_markers)


class StacCollectionBuilder(CollectionBuilder):
    """Build a Collection from a STAC API search or static catalog.

    Searches a STAC API (or traverses a static STAC catalog when
    ``static_catalog=True``), parses COG headers for tile metadata,
    and produces a Parquet-backed Collection with per-band acceleration
    columns.
    """

    def __init__(
        self,
        data_source: str,
        stac_api: str,
        stac_collection: str | None = None,
        workspace_dir: Path | None = None,
        name: str | None = None,
        band_map: dict[str, str] | None = None,
        band_index_map: dict[str, int] | None = None,
        cloud_config: CloudConfig | None = None,
        max_concurrent: int = 300,
        backend: StorageBackend | None = None,
        static_catalog: bool = False,
    ):
        super().__init__(
            name=name or "",
            data_source=data_source,
            workspace_dir=workspace_dir,
        )
        self.stac_collection = stac_collection or data_source
        self.stac_api = stac_api
        self._band_map = band_map
        self._band_index_map = band_index_map or {}
        self.cloud_config = cloud_config
        self.max_concurrent = max_concurrent
        self.batch_size = 100
        self._backend = backend
        self.static_catalog = static_catalog

    @property
    def band_map(self) -> dict[str, str]:
        """Get band mapping for current collection."""
        if self._band_map is not None:
            return self._band_map
        return BandRegistry.get(self.data_source)

    def build(self, **kwargs: Any):
        """Build a Collection from STAC (sync wrapper).

        Accepts ``bbox``, ``date_range``, ``query`` keyword arguments.
        Delegates to the async :meth:`build_index`.
        """
        from rasteret.core.utils import run_sync

        return run_sync(self.build_index(**kwargs))

    async def build_index(
        self,
        bbox: BoundingBox | None = None,
        date_range: DateRange | None = None,
        query: dict[str, Any] | None = None,
    ):
        """Build GeoParquet collection from STAC search (async).

        Returns a :class:`~rasteret.core.collection.Collection`.
        """
        logger.info("Starting STAC index creation...")
        if bbox:
            logger.info("Spatial filter: %s", bbox)
        if date_range:
            logger.info("Temporal filter: %s to %s", date_range[0], date_range[1])
        if query:
            logger.info("Additional query parameters: %s", query)

        # 1. Search STAC
        stac_items = await self._search_stac(bbox, date_range, query)
        logger.info("Found %d scenes in STAC catalog", len(stac_items))
        if not stac_items:
            raise ValueError(
                "No STAC scenes matched the request "
                f"(bbox={bbox}, date_range={date_range}, query={query})."
            )
        self._ensure_band_map_matches_assets(stac_items)

        # 2. Process in batches, adding COG metadata
        processed_items = await self._enrich_with_cog_metadata(stac_items)

        logger.info("Successfully processed %d items", len(processed_items))
        if not processed_items:
            raise ValueError(
                "COG header enrichment produced no band metadata. "
                "Verify your band_map points to tiled COG assets and that the "
                "STAC items include those assets."
            )

        # 3. Build Arrow table from STAC items + enrichment
        table = self._build_stac_table(stac_items, processed_items)

        # 4. Normalise to Collection via shared layer
        return build_collection_from_table(
            table,
            name=self.name or "",
            description=f"STAC collection indexed from {self.data_source}",
            data_source=self.data_source,
            date_range=date_range,
            workspace_dir=self.workspace_dir,
        )

    async def _enrich_with_cog_metadata(self, stac_items: list[dict]) -> list[dict]:
        """Parse COG headers for all items in batches."""
        processed_items: list[dict] = []
        total_batches = (len(stac_items) + self.batch_size - 1) // self.batch_size

        logger.info(
            "Processing %d scenes (each scene has multiple bands)...",
            len(stac_items),
        )

        async with AsyncCOGHeaderParser(
            max_concurrent=self.max_concurrent,
            batch_size=self.batch_size,
            backend=self._backend,
        ) as cog_parser:
            for i in range(0, len(stac_items), self.batch_size):
                batch = stac_items[i : i + self.batch_size]
                batch_records = await self._process_batch(batch, cog_parser)
                if batch_records:
                    processed_items.extend(batch_records)
                logger.info(
                    "Processed scene batch %d/%d yielding %d band assets",
                    (i // self.batch_size) + 1,
                    total_batches,
                    len(batch_records),
                )

        total_assets = sum(len(item["assets"]) for item in stac_items)
        logger.info(
            "Completed processing %d scenes with %d/%d band assets",
            len(stac_items),
            len(processed_items),
            total_assets,
        )
        return processed_items

    def _build_stac_table(
        self,
        stac_items: list[dict],
        processed_items: list[dict],
    ) -> pa.Table:
        """Convert STAC items + COG enrichment into an Arrow table.

        Builds the Arrow table directly from STAC item dicts using native
        pyarrow, avoiding the stac-geoparquet ndjson/parquet roundtrip.

        Generic normalisation (year/month/bbox/validation) is
        delegated to :func:`build_collection_from_table`.
        """
        from datetime import datetime as dt

        logger.info("Creating GeoParquet table with metadata...")

        # Build a normalized assets mapping from processed_items:
        # {record_id: {band_code: {href, band_index}}}
        assets_by_id: dict[str, dict[str, dict[str, object]]] = {}
        for item in processed_items:
            record_id = item.get("record_id")
            band = item.get("band")
            href = item.get("href")
            if not record_id or not band or not href:
                continue
            try:
                band_index = int(item.get("band_index", 0))
            except (TypeError, ValueError):
                band_index = 0
            assets_by_id.setdefault(str(record_id), {})[str(band)] = {
                "href": str(href),
                "band_index": band_index,
            }

        rows = []
        geojson_geoms = []
        for item in stac_items:
            props = item.get("properties", {})
            row = {
                "id": item["id"],
                "assets": assets_by_id.get(item["id"], {}),
            }
            geojson_geoms.append(item["geometry"])
            # Flatten STAC properties to top-level columns
            for key, value in props.items():
                row[key] = value
            row["collection"] = item.get("collection") or self.stac_collection
            rows.append(row)

        table = pa.Table.from_pylist(rows)

        # Build geometry column as WKB via shapely's vectorized API
        geom_wkb = geojson_dicts_to_wkb(geojson_geoms)
        table = table.append_column("geometry", geom_wkb)

        # Ensure datetime is timestamp type
        if "datetime" in table.schema.names and not pa.types.is_timestamp(
            table["datetime"].type
        ):
            dt_idx = table.schema.get_field_index("datetime")
            dt_vals = []
            for d in table.column("datetime"):
                raw = d.as_py()
                if raw is None:
                    dt_vals.append(None)
                    continue
                if isinstance(raw, dt):
                    dt_vals.append(raw)
                    continue
                s = str(raw)
                # Common STAC form: "2024-01-10T00:00:00Z"
                if s.endswith("Z"):
                    s = f"{s[:-1]}+00:00"
                dt_vals.append(dt.fromisoformat(s))
            table = table.set_column(
                dt_idx, "datetime", pa.array(dt_vals, type=pa.timestamp("us"))
            )

        # Add canonical bbox struct from original STAC geometries
        item_bboxes = {}
        for item in stac_items:
            if "bbox" in item and item["bbox"]:
                b = item["bbox"]
                item_bboxes[item["id"]] = {
                    "xmin": b[0],
                    "ymin": b[1],
                    "xmax": b[2],
                    "ymax": b[3],
                }
            else:
                from rasteret.core.geometry import bbox_from_geojson_coords

                b = bbox_from_geojson_coords(item["geometry"])
                item_bboxes[item["id"]] = {
                    "xmin": b[0],
                    "ymin": b[1],
                    "xmax": b[2],
                    "ymax": b[3],
                }
        bbox_list = [item_bboxes[id_] for id_ in table.column("id").to_pylist()]
        table = table.append_column(
            "bbox",
            pa.array(
                bbox_list,
                type=pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        )

        # Add per-band metadata columns from COG header enrichment
        table = add_band_metadata_columns(
            table, list(self.band_map.keys()), processed_items
        )

        return table

    # ------------------------------------------------------------------
    # STAC search + URL signing
    # ------------------------------------------------------------------

    async def _search_stac(self, bbox, date_range, query) -> list[dict]:
        """Search STAC catalog and return items as dicts."""
        if self.static_catalog:
            max_items: int | None = None
            if isinstance(query, dict) and "max_items" in query:
                try:
                    max_items = int(query["max_items"])
                except (TypeError, ValueError):
                    max_items = None
            return self._crawl_static_catalog(bbox, date_range, max_items)

        max_items: int | None = None
        resolved_query = query
        if isinstance(query, dict):
            # `max_items` is a local control knob (not part of the STAC
            # `query` filter); extract it before forwarding to the STAC API.
            if "max_items" in query:
                try:
                    max_items = int(query["max_items"])
                except (TypeError, ValueError):
                    max_items = None
                resolved_query = {k: v for k, v in query.items() if k != "max_items"}

        limit = 1000
        if max_items is not None:
            # Minimize payloads for smoke tests / quick builds.
            limit = max(1, min(1000, max_items))

        search_params = {
            "collections": [self.stac_collection],
            "limit": limit,
            **({"bbox": bbox} if bbox else {}),
            **({"datetime": f"{date_range[0]}/{date_range[1]}"} if date_range else {}),
            **({"query": resolved_query} if resolved_query else {}),
        }

        items: list[dict] = []
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                client = pystac_client.Client.open(self.stac_api)
                search = client.search(**search_params)
                items = []
                for item in search.items_as_dicts():
                    items.append(item)
                    if max_items is not None and len(items) >= max_items:
                        break
                break
            except APIError as exc:
                if attempt >= max_attempts or not _is_retryable_stac_api_error(exc):
                    raise
                sleep_s = float(2 ** (attempt - 1))
                logger.warning(
                    "Transient STAC API error from %s (attempt %d/%d): %s. Retrying in %.1fs",
                    self.stac_api,
                    attempt,
                    max_attempts,
                    exc,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)

        host = urlparse(self.stac_api).netloc.lower()
        if "planetarycomputer.microsoft.com" in host:
            # If an obstore backend is provided, prefer native AzureStore reads
            # via the credential provider (avoids embedding SAS tokens in the
            # cached Parquet index).
            if self._backend is None:
                try:
                    import planetary_computer
                except Exception as exc:
                    raise ValueError(
                        "Planetary Computer datasets require SAS signing. "
                        'Install "rasteret[azure]" (planetary-computer) or pass '
                        "backend= created with PlanetaryComputerCredentialProvider."
                    ) from exc
                for item in items:
                    try:
                        planetary_computer.sign_inplace(item)
                    except Exception as exc:
                        response = getattr(exc, "response", None)
                        status_code = getattr(response, "status_code", None)
                        retry_after = None
                        if response is not None:
                            headers = getattr(response, "headers", None) or {}
                            retry_after = headers.get("Retry-After") or headers.get(
                                "retry-after"
                            )

                        if status_code == 429 or " 429" in f" {exc}":
                            retry_suffix = (
                                f" (Retry-After: {retry_after}s)"
                                if retry_after is not None
                                else ""
                            )
                            raise ValueError(
                                "Planetary Computer SAS signing was rate-limited (HTTP 429)"
                                f"{retry_suffix}. "
                                "This happens when the signing service is called too frequently. "
                                "Mitigations: (1) reduce STAC query size (e.g. `query={'max_items': ...}`), "
                                "(2) retry later, (3) set `PC_SDK_SUBSCRIPTION_KEY` (or run `planetarycomputer configure`) "
                                "for less restrictive rate limits, or (4) pass `backend=` created with "
                                "`obstore.auth.planetary_computer.PlanetaryComputerCredentialProvider` to avoid "
                                "embedding SAS URLs in the cached index."
                            ) from exc

                        raise ValueError(
                            "Planetary Computer SAS signing failed. "
                            'Install "rasteret[azure]" and ensure the Planetary Computer STAC API is reachable, '
                            "or pass backend= created with PlanetaryComputerCredentialProvider."
                        ) from exc

        if self.cloud_config:
            for item in items:
                for asset in item["assets"].values():
                    self._rewrite_asset_url(asset)

        return items

    def _crawl_static_catalog(
        self,
        bbox: Any | None,
        date_range: Any | None,
        max_items: int | None = None,
    ) -> list[dict]:
        """Traverse a static STAC catalog and return items as dicts."""
        import pystac

        catalog = pystac.Catalog.from_file(self.stac_api)

        # If data_source names a child collection, narrow scope
        if self.data_source:
            try:
                child = catalog.get_child(self.data_source)
                if child is not None:
                    catalog = child
            except Exception as exc:
                logger.warning(
                    "Could not narrow static STAC catalog to child '%s': %s",
                    self.data_source,
                    exc,
                )

        items: list[dict] = []
        for item in catalog.get_all_items():
            # Resolve relative hrefs to absolute URLs
            item.make_asset_hrefs_absolute()
            item_dict = item.to_dict()

            # Client-side bbox filter
            if bbox:
                ib = item_dict.get("bbox")
                if ib and not (
                    ib[0] <= bbox[2]
                    and ib[2] >= bbox[0]
                    and ib[1] <= bbox[3]
                    and ib[3] >= bbox[1]
                ):
                    continue

            # Client-side date filter
            if date_range:
                dt_str = (item_dict.get("properties") or {}).get("datetime")
                if dt_str:
                    if dt_str < date_range[0] or dt_str > f"{date_range[1]}T23:59:59Z":
                        continue

            items.append(item_dict)
            if max_items and len(items) >= max_items:
                break

        if items:
            self._ensure_band_map_matches_assets(items)

        return items

    def _ensure_band_map_matches_assets(self, stac_items: list[dict]) -> None:
        """Resolve or validate band mapping before expensive COG header reads."""
        asset_keys = self._collect_asset_keys(stac_items)
        explicit_map = self._band_map is not None
        resolved = self.band_map

        if resolved:
            if explicit_map:
                missing = set(resolved.values()) - asset_keys
                if not missing:
                    return
                detected = self._format_asset_keys(asset_keys)
                expected = self._format_asset_keys(set(resolved.values()))
                missing_text = self._format_asset_keys(missing)
                raise ValueError(
                    "STAC band_map references asset keys not present in the "
                    "selected STAC items. "
                    f"Missing asset keys: {missing_text}. "
                    f"Configured asset keys: {expected}. "
                    f"Detected asset keys: {detected}. "
                    "Individual items may omit valid bands, but every configured "
                    "asset key must appear at least once in the selected result."
                )
            if self._mapped_asset_keys(resolved, asset_keys):
                return

        if not explicit_map:
            inferred = self._infer_band_map_from_assets(asset_keys)
            if inferred:
                self._band_map = inferred
                logger.info(
                    "Inferred STAC band map for %s from asset keys: %s",
                    self.data_source,
                    inferred,
                )
                return

        detected = self._format_asset_keys(asset_keys)
        if resolved:
            expected = self._format_asset_keys(set(resolved.values()))
            raise ValueError(
                "No STAC asset keys matched the resolved band_map. "
                f"Expected one of: {expected}. Detected asset keys: {detected}. "
                "Pass band_map={band: asset_key} with asset keys present in the "
                "STAC items, or use a registered data_source with matching asset "
                "conventions."
            )

        raise ValueError(
            "No band_map is configured for this STAC source, and Rasteret could "
            f"not infer one from detected asset keys: {detected}. Pass "
            "band_map={band: asset_key} explicitly or register a BandRegistry "
            "mapping for this data_source."
        )

    @staticmethod
    def _collect_asset_keys(stac_items: list[dict]) -> set[str]:
        keys: set[str] = set()
        for item in stac_items:
            assets = item.get("assets") or {}
            if isinstance(assets, dict):
                keys.update(str(key) for key in assets.keys())
        return keys

    @staticmethod
    def _mapped_asset_keys(band_map: dict[str, str], asset_keys: set[str]) -> set[str]:
        return {asset for asset in band_map.values() if asset in asset_keys}

    def _infer_band_map_from_assets(self, asset_keys: set[str]) -> dict[str, str]:
        """Infer provider asset conventions from registered dataset band maps."""
        if not asset_keys:
            return {}

        candidate_sources = [
            self.data_source,
            self.stac_collection,
            *BandRegistry.list_registered(),
        ]
        seen: set[str] = set()
        best: dict[str, str] = {}
        for source in candidate_sources:
            if not source or source in seen:
                continue
            seen.add(source)
            registered = BandRegistry.get(source)
            if not registered:
                continue

            provider_map = {
                band: asset for band, asset in registered.items() if asset in asset_keys
            }
            if len(provider_map) > len(best):
                best = provider_map

            identity_map = {band: band for band in registered if band in asset_keys}
            if len(identity_map) > len(best):
                best = identity_map

        return best

    @staticmethod
    def _format_asset_keys(asset_keys: set[str], limit: int = 20) -> str:
        if not asset_keys:
            return "<none>"
        ordered = sorted(asset_keys)
        shown = ", ".join(ordered[:limit])
        if len(ordered) > limit:
            shown = f"{shown}, ... (+{len(ordered) - limit} more)"
        return shown

    def _rewrite_asset_url(self, asset: dict) -> None:
        """Rewrite a single asset URL in-place using cloud_config patterns."""
        if "href" in asset:
            asset["href"] = rewrite_url(asset["href"], self.cloud_config)

    def _get_asset_url(self, asset: dict) -> str:
        """Get URL for asset, applying URL rewrites."""
        url = asset["href"] if isinstance(asset, dict) else asset
        return rewrite_url(url, self.cloud_config)

    async def _process_batch(
        self, stac_items: list[dict], cog_parser: AsyncCOGHeaderParser
    ) -> list[dict]:
        """Add COG metadata to a batch of STAC items."""
        urls_to_process: list[str] = []
        # url -> list of (item_id, item, band_code, href, band_index)
        url_mapping: dict[str, list[tuple[str, dict, str, str, int]]] = {}

        # Multi-band assets (multiple band codes mapping to one STAC asset key)
        # require a band_index_map to disambiguate samples.
        asset_to_codes: dict[str, list[str]] = {}
        for code, asset_name in self.band_map.items():
            asset_to_codes.setdefault(asset_name, []).append(code)
        if (
            any(len(codes) > 1 for codes in asset_to_codes.values())
            and not self._band_index_map
        ):
            dupes = {k: v for k, v in asset_to_codes.items() if len(v) > 1}
            raise NotImplementedError(
                "Multiple requested bands map to the same STAC asset key, but no "
                "band_index_map was provided. This is required for multi-band "
                "GeoTIFF assets (e.g. NAIP 'image' contains R/G/B/NIR). "
                f"Duplicates: {dupes}"
            )

        for item in stac_items:
            item_id = item.get("id")
            if not item_id:
                continue
            for band_code, asset_name in self.band_map.items():
                if asset_name not in item["assets"]:
                    continue
                asset = item["assets"][asset_name]
                url = self._get_asset_url(asset)
                band_index = int(self._band_index_map.get(band_code, 0))

                # When a backend is available, prefer S3 URLs (s3_* sibling
                # assets) because the backend can route them to native S3Store
                # with credential providers.
                if self._backend is not None:
                    s3_key = f"s3_{asset_name}"
                    s3_asset = item["assets"].get(s3_key)
                    if isinstance(s3_asset, dict):
                        s3_href = s3_asset.get("href", "")
                        if s3_href.startswith("s3://"):
                            url = s3_href

                if url:
                    if url not in url_mapping:
                        urls_to_process.append(url)
                        url_mapping[url] = [(item_id, item, band_code, url, band_index)]
                    else:
                        url_mapping[url].append(
                            (item_id, item, band_code, url, band_index)
                        )

        metadata_results = await cog_parser.process_cog_headers_batch(urls_to_process)

        processed_items: dict[str, dict] = {}
        for url, metadata in zip(urls_to_process, metadata_results):
            if not metadata:
                continue
            for item_id, item, band_code, href, band_index in url_mapping[url]:
                if item_id not in processed_items:
                    processed_items[item_id] = {
                        "id": item_id,
                        "record_id": item_id,
                        "geometry": item["geometry"],
                        "datetime": item["properties"].get("datetime"),
                        "cloud_cover": item["properties"].get("eo:cloud_cover"),
                        "bands": {},
                    }
                offsets, counts = slice_tile_tables_for_band(
                    metadata=metadata,
                    band_index=band_index,
                )
                nodata_val = metadata.nodata
                if nodata_val is not None and nodata_val != nodata_val:
                    nodata_val = float("nan")  # ensure NaN is float for Arrow
                elif nodata_val is not None:
                    nodata_val = float(nodata_val)
                band_payload = {
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
                    "href": href,
                    "band_index": band_index,
                }
                processed_items[item_id]["bands"][band_code] = band_payload

        enriched_items = []
        for item_data in processed_items.values():
            for band_code, band_metadata in item_data["bands"].items():
                enriched_items.append(
                    {
                        "record_id": item_data["id"],
                        "band": band_code,
                        **band_metadata,
                    }
                )
        return enriched_items

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
