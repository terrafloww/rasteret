# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.


from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
from dataclasses import dataclass
from datetime import timedelta as _timedelta
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import imagecodecs
import numpy as np
import pyarrow as pa
from affine import Affine
from rasterio.mask import geometry_mask

try:
    from obstore.exceptions import GenericError as _ObstoreGenericError
    from obstore.exceptions import UnknownConfigurationKeyError
except ImportError:  # pragma: no cover - obstore is a runtime dependency
    _ObstoreGenericError = Exception
    UnknownConfigurationKeyError = TypeError

from rasteret.core.geometry import (
    bbox_intersects,
    bbox_single,
    to_rasterio_geojson,
    transform_coords,
)
from rasteret.core.utils import normalize_transform
from rasteret.types import CogMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# obstore auto-detection
# ---------------------------------------------------------------------------

# Default ``ClientConfig`` values tuned for Rasteret's COG workload.
# These are passed to ``HTTPStore.from_url`` when obstore is auto-detected.
_OBSTORE_CLIENT_OPTIONS: dict[str, str] = {
    "connect_timeout": "5s",
    "pool_idle_timeout": "90s",
    "pool_max_idle_per_host": "50",
    "http2_keep_alive_interval": "30s",
    "http2_keep_alive_while_idle": "true",
    "timeout": "30s",
}

# Retry configuration for obstore stores.  obstore retries 5xx errors,
# connection resets, and timeouts for safe (read-only) requests.
_OBSTORE_RETRY_CONFIG: dict[str, object] = {
    "max_retries": 5,
    "retry_timeout": _timedelta(seconds=60),
    "backoff": {
        "init_backoff": _timedelta(milliseconds=200),
        "max_backoff": _timedelta(seconds=10),
        "base": 2,
    },
}


_S3_HOST_RE = re.compile(
    r"^([a-z0-9][a-z0-9.\-]+)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com$"
)
_AZURE_BLOB_RE = re.compile(r"^([a-z0-9]+)\.blob\.core\.windows\.net$")
_CREDENTIAL_PROVIDER_EXCEPTIONS = (TypeError, UnknownConfigurationKeyError)


def _extract_s3_bucket(netloc: str) -> str | None:
    """Extract bucket name from an S3 virtual-hosted-style hostname.

    Returns ``None`` if *netloc* is not an S3-style host.
    """
    m = _S3_HOST_RE.match(netloc)
    return m.group(1) if m else None


def _extract_azure_account(netloc: str) -> str | None:
    """Extract account name from an Azure Blob hostname.

    Matches ``<account>.blob.core.windows.net``.
    Returns ``None`` if *netloc* is not an Azure Blob host.
    """
    m = _AZURE_BLOB_RE.match(netloc)
    return m.group(1) if m else None


def _try_boto3_credential_provider() -> object | None:
    """Try to create a ``Boto3CredentialProvider`` for S3 authentication.

    Returns ``None`` if boto3 is not installed or no credentials are
    available (e.g. no ``~/.aws/credentials``, no env vars, no instance
    profile).  This is used as an automatic fallback for requester-pays
    buckets when no explicit credential provider was passed.
    """
    try:
        from obstore.auth.boto3 import Boto3CredentialProvider
    except ModuleNotFoundError:
        # obstore was installed without its boto3 integration.
        return None

    try:
        provider = Boto3CredentialProvider()
    except Exception as exc:
        logger.warning("Failed to create Boto3CredentialProvider: %s", exc)
        return None

    # Validate that credentials actually exist - Boto3CredentialProvider
    # defers the lookup, but the underlying session check is cheap.
    if provider.credentials is None or provider.credentials.access_key is None:
        return None
    return provider


def _create_obstore_backend(
    s3_overrides: dict[str, dict[str, str]] | None = None,
    credential_provider: object | None = None,
    default_s3_config: dict[str, str] | None = None,
    url_patterns: dict[str, str] | None = None,
) -> _AutoObstoreBackend:
    """Create the default obstore-backed storage client.

    The returned backend lazily creates one store per origin.  For S3,
    Azure Blob, and GCS URLs it uses native stores; for everything
    else it falls back to ``HTTPStore``.

    Parameters
    ----------
    s3_overrides : dict, optional
        Per-bucket S3Store config, e.g.
        ``{"usgs-landsat": {"request_payer": "true"}}``.
    credential_provider : object, optional
        An obstore credential provider (e.g.
        ``PlanetaryComputerCredentialProvider``,
        ``NasaEarthdataCredentialProvider``).  Passed through to the
        appropriate store constructor.
    default_s3_config : dict, optional
        Default S3Store config applied to all buckets that don't have
        per-bucket overrides (e.g. ``{"region": "us-west-2"}``).
    url_patterns : dict, optional
        URL prefix rewrites applied before routing, e.g.
        ``{"https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"}``.
    """
    return _AutoObstoreBackend(
        s3_overrides=s3_overrides,
        credential_provider=credential_provider,
        default_s3_config=default_s3_config,
        url_patterns=url_patterns,
    )


def _discover_s3_bucket_region(
    bucket: str, *, timeout_seconds: float = 3.0
) -> str | None:
    """Best-effort region discovery for public S3 buckets.

    S3 responds to global-endpoint requests with ``301`` and an
    ``x-amz-bucket-region`` header, but sometimes no ``Location`` header.
    Some HTTP clients treat that as an error. We use a simple ``HEAD`` request
    to extract the region hint.
    """
    urls = [
        f"https://s3.amazonaws.com/{bucket}",
        f"https://{bucket}.s3.amazonaws.com/",
    ]

    for url in urls:
        req = Request(url, method="HEAD", headers={"User-Agent": "rasteret"})
        try:
            with contextlib.closing(urlopen(req, timeout=timeout_seconds)) as resp:
                region = resp.headers.get("x-amz-bucket-region")
                if region:
                    return region
        except HTTPError as exc:
            region = exc.headers.get("x-amz-bucket-region")
            if region:
                return region
        except URLError:
            continue

    return None


class _AutoObstoreBackend:
    """StorageBackend that lazily creates per-origin store instances.

    Routes URLs to the appropriate native store:

    - ``s3://`` and ``*.s3.*.amazonaws.com`` -> ``S3Store``
    - ``gs://`` and ``storage.googleapis.com`` -> ``GCSStore``
    - ``*.blob.core.windows.net`` -> ``AzureStore``
    - Pre-signed / SAS-signed URLs (query params) -> ``HTTPStore``
    - Other HTTPS -> ``HTTPStore``

    Each store holds a Rust ``reqwest`` connection pool, so
    one-per-origin is the correct granularity.
    """

    def __init__(
        self,
        s3_overrides: dict[str, dict[str, str]] | None = None,
        credential_provider: object | None = None,
        default_s3_config: dict[str, str] | None = None,
        url_patterns: dict[str, str] | None = None,
    ) -> None:
        self._stores: dict[str, object] = {}
        self._s3_overrides = s3_overrides or {}
        self._credential_provider = credential_provider
        self._default_s3_config = default_s3_config or {}
        self._s3_regions: dict[str, str] = {}
        self._url_patterns = url_patterns or {}

    def _create_store_with_provider_fallback(
        self,
        *,
        store_name: str,
        store_target: str,
        store_constructor,
        base_kwargs: dict,
        provider_kwargs: dict,
    ) -> object:
        """Build store with provider, falling back to anonymous when incompatible."""
        if self._credential_provider is None:
            return store_constructor(**base_kwargs)

        kwargs = dict(provider_kwargs)
        kwargs["credential_provider"] = self._credential_provider
        try:
            return store_constructor(**kwargs)
        except _CREDENTIAL_PROVIDER_EXCEPTIONS as exc:
            logger.debug(
                "credential_provider incompatible with %s for %s (%s); "
                "falling back to anonymous access",
                store_name,
                store_target,
                exc.__class__.__name__,
            )
            return store_constructor(**base_kwargs)

    def _store_for(self, url: str) -> tuple[object, str]:
        """Return (store, relative_path) for *url*."""
        for pattern, replacement in self._url_patterns.items():
            if url.startswith(pattern):
                url = url.replace(pattern, replacement, 1)
                break
        parsed = urlparse(url)

        # --- s3:// scheme -> S3Store ---
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            path = parsed.path.lstrip("/")
            return self._get_s3_store(bucket), path

        # --- gs:// scheme -> GCSStore ---
        if parsed.scheme == "gs":
            bucket = parsed.netloc
            return self._get_gcs_store(bucket), parsed.path.lstrip("/")

        # --- Pre-signed / SAS-signed URLs: HTTPStore per full URL ---
        if parsed.query:
            from obstore.store import HTTPStore

            store = self._stores.get(url)
            if store is None:
                store = HTTPStore.from_url(
                    url,
                    client_options=_OBSTORE_CLIENT_OPTIONS,
                    retry_config=_OBSTORE_RETRY_CONFIG,
                )
                self._stores[url] = store
            return store, ""

        # --- S3 virtual-hosted HTTPS -> S3Store ---
        bucket = _extract_s3_bucket(parsed.netloc)
        if bucket:
            return self._get_s3_store(bucket), parsed.path.lstrip("/")

        # --- Azure Blob HTTPS -> AzureStore ---
        azure_account = _extract_azure_account(parsed.netloc)
        if azure_account:
            parts = parsed.path.lstrip("/").split("/", 1)
            container = parts[0]
            path = parts[1] if len(parts) > 1 else ""
            store = self._get_azure_store(azure_account, container)
            # When a credential provider sets a prefix on the store (e.g.
            # PlanetaryComputerCredentialProvider), obstore prepends that
            # prefix to every path automatically.  Strip it from *our* path
            # to avoid doubling (v002/v002/...).
            store_prefix = getattr(store, "prefix", None)
            if isinstance(store_prefix, str) and path.startswith(store_prefix):
                path = path[len(store_prefix) :].lstrip("/")
            return store, path

        # --- GCS HTTPS -> GCSStore ---
        if parsed.netloc == "storage.googleapis.com":
            parts = parsed.path.lstrip("/").split("/", 1)
            bucket = parts[0]
            path = parts[1] if len(parts) > 1 else ""
            return self._get_gcs_store(bucket), path

        # --- Default: HTTPStore per origin ---
        from obstore.store import HTTPStore

        origin = f"{parsed.scheme}://{parsed.netloc}/"
        store = self._stores.get(origin)
        if store is None:
            store = HTTPStore.from_url(
                origin,
                client_options=_OBSTORE_CLIENT_OPTIONS,
                retry_config=_OBSTORE_RETRY_CONFIG,
            )
            self._stores[origin] = store
        return store, parsed.path.lstrip("/")

    def _get_s3_store(self, bucket: str) -> object:
        """Return a cached ``S3Store`` for *bucket*."""
        cache_key = f"s3://{bucket}"
        store = self._stores.get(cache_key)
        if store is None:
            from obstore.store import S3Store

            base_config: dict[str, str] = {"skip_signature": "true"}
            base_config.update(self._default_s3_config)
            base_config.update(self._s3_overrides.get(bucket, {}))
            if "region" not in base_config and bucket in self._s3_regions:
                base_config["region"] = self._s3_regions[bucket]
            # Requester-pays buckets require authenticated requests. If we have
            # requester-pays config, ensure we don't force anonymous access.
            needs_signing = base_config.get("request_payer") == "true"
            if needs_signing:
                base_config.pop("skip_signature", None)
            provider_config = dict(base_config)
            provider_config.pop("skip_signature", None)

            # When requester-pays requires signing but no credential provider
            # was explicitly passed, try Boto3CredentialProvider as a fallback.
            # obstore's S3Store doesn't auto-discover ~/.aws/credentials, so
            # this bridges the gap for users with standard AWS CLI credentials.
            effective_provider = self._credential_provider
            if needs_signing and effective_provider is None:
                effective_provider = _try_boto3_credential_provider()

            base_kwargs: dict = {
                "bucket": bucket,
                "config": base_config,
                "client_options": _OBSTORE_CLIENT_OPTIONS,
                "retry_config": _OBSTORE_RETRY_CONFIG,
            }
            provider_kwargs: dict = {
                "bucket": bucket,
                "config": provider_config,
                "client_options": _OBSTORE_CLIENT_OPTIONS,
                "retry_config": _OBSTORE_RETRY_CONFIG,
            }
            if effective_provider is not None:
                provider_kwargs["credential_provider"] = effective_provider
                try:
                    store = S3Store(**provider_kwargs)
                except _CREDENTIAL_PROVIDER_EXCEPTIONS as exc:
                    logger.debug(
                        "credential_provider incompatible with S3Store for %s (%s); "
                        "falling back to base config",
                        bucket,
                        exc.__class__.__name__,
                    )
                    store = S3Store(**base_kwargs)
            else:
                store = self._create_store_with_provider_fallback(
                    store_name="S3Store",
                    store_target=bucket,
                    store_constructor=S3Store,
                    base_kwargs=base_kwargs,
                    provider_kwargs=provider_kwargs,
                )
            self._stores[cache_key] = store
        return store

    def _get_azure_store(self, account: str, container: str) -> object:
        """Return a cached ``AzureStore`` for *account*/*container*."""
        cache_key = f"azure://{account}/{container}"
        store = self._stores.get(cache_key)
        if store is None:
            from obstore.store import AzureStore

            base_kwargs: dict = {
                "container_name": container,
                "config": {"account_name": account},
                "client_options": _OBSTORE_CLIENT_OPTIONS,
                "retry_config": _OBSTORE_RETRY_CONFIG,
            }
            provider_kwargs = dict(base_kwargs)
            store = self._create_store_with_provider_fallback(
                store_name="AzureStore",
                store_target=f"{account}/{container}",
                store_constructor=AzureStore,
                base_kwargs=base_kwargs,
                provider_kwargs=provider_kwargs,
            )
            self._stores[cache_key] = store
        return store

    def _get_gcs_store(self, bucket: str) -> object:
        """Return a cached ``GCSStore`` for *bucket*."""
        cache_key = f"gs://{bucket}"
        store = self._stores.get(cache_key)
        if store is None:
            from obstore.store import GCSStore

            base_kwargs: dict = {
                "bucket": bucket,
                "config": {"skip_signature": "true"},
                "client_options": _OBSTORE_CLIENT_OPTIONS,
                "retry_config": _OBSTORE_RETRY_CONFIG,
            }
            provider_kwargs: dict = {
                "bucket": bucket,
                "config": {},
                "client_options": _OBSTORE_CLIENT_OPTIONS,
                "retry_config": _OBSTORE_RETRY_CONFIG,
            }
            store = self._create_store_with_provider_fallback(
                store_name="GCSStore",
                store_target=bucket,
                store_constructor=GCSStore,
                base_kwargs=base_kwargs,
                provider_kwargs=provider_kwargs,
            )
            self._stores[cache_key] = store
        return store

    async def get_range(self, url: str, start: int, length: int) -> bytes:
        import obstore as obs

        store, path = self._store_for(url)
        try:
            buf = await obs.get_range_async(store, path, start=start, length=length)
            return bytes(buf)
        except _ObstoreGenericError as exc:
            # obstore raises GenericError for S3 region redirects.  There is
            # no typed redirect exception, so we check the message string.
            parsed = urlparse(url)
            bucket = (
                parsed.netloc
                if parsed.scheme == "s3"
                else _extract_s3_bucket(parsed.netloc)
            )
            if not bucket or "Received redirect without LOCATION" not in str(exc):
                raise

            logger.debug(
                "S3 region redirect for bucket '%s', discovering region", bucket
            )
            region = self._s3_regions.get(bucket) or _discover_s3_bucket_region(bucket)
            if not region:
                raise RuntimeError(
                    f"S3 bucket region auto-detection failed for '{bucket}'. "
                    "Provide a region explicitly (e.g. via "
                    "`rasteret.create_backend(default_s3_config={'region': '...'})` "
                    "or a per-bucket override) and retry."
                ) from exc

            self._s3_regions[bucket] = region
            self._stores.pop(f"s3://{bucket}", None)
            store = self._get_s3_store(bucket)
            buf = await obs.get_range_async(store, path, start=start, length=length)
            return bytes(buf)

    async def get_ranges(self, url: str, ranges: list[tuple[int, int]]) -> list[bytes]:
        if not ranges:
            return []

        import obstore as obs

        store, path = self._store_for(url)
        starts, lengths = zip(*ranges)
        try:
            buffers = await obs.get_ranges_async(
                store, path, starts=list(starts), lengths=list(lengths)
            )
            return [bytes(b) for b in buffers]
        except _ObstoreGenericError as exc:
            parsed = urlparse(url)
            bucket = (
                parsed.netloc
                if parsed.scheme == "s3"
                else _extract_s3_bucket(parsed.netloc)
            )
            if not bucket or "Received redirect without LOCATION" not in str(exc):
                raise

            logger.debug(
                "S3 region redirect for bucket '%s', discovering region", bucket
            )
            region = self._s3_regions.get(bucket) or _discover_s3_bucket_region(bucket)
            if not region:
                raise RuntimeError(
                    f"S3 bucket region auto-detection failed for '{bucket}'. "
                    "Provide a region explicitly and retry."
                ) from exc

            self._s3_regions[bucket] = region
            self._stores.pop(f"s3://{bucket}", None)
            store = self._get_s3_store(bucket)
            buffers = await obs.get_ranges_async(
                store, path, starts=list(starts), lengths=list(lengths)
            )
            return [bytes(b) for b in buffers]


@dataclass
class COGTileRequest:
    """Single tile request details."""

    url: str
    offset: int  # Byte offset in COG file
    size: int  # Size in bytes to read
    row: int  # Tile row in the grid
    col: int  # Tile column in the grid
    metadata: CogMetadata  # Full metadata including transform
    band_index: int | None = None  # Optional sample index for chunky multi-sample TIFFs


@dataclass(frozen=True)
class CogReadResult:
    """Result of a COG read operation.

    ``valid_mask`` is True for pixels that are inside the requested AOI/window
    *and* inside raster coverage. When ``filled=True``, pixels where
    ``valid_mask=False`` are set to *fill_value_used*.
    """

    data: np.ndarray
    transform: Affine
    valid_mask: np.ndarray
    fill_value_used: float | int | None
    mode: Literal["aoi", "window", "full"]


class COGReader:
    """Manages connection pooling and COG reading operations.

    Manages Rasteret's custom async byte-range IO. Uses obstore as the HTTP
    transport layer for multi-cloud URL routing.  Optionally accepts a custom
    :class:`~rasteret.cloud.StorageBackend` (e.g. a pre-configured ``S3Store``).

    Parameters
    ----------
    max_concurrent : int
        Maximum number of concurrent HTTP requests / byte-range reads.
    backend : StorageBackend, optional
        Pluggable I/O backend. When ``None``, an obstore ``HTTPStore``
        backend is created automatically.
    """

    def __init__(
        self,
        max_concurrent: int = 150,
        backend: object | None = None,
    ):
        self.max_concurrent = max_concurrent
        self._backend = backend
        self.sem = None
        self.batch_size = 20

    async def __aenter__(self) -> COGReader:
        self.sem = asyncio.Semaphore(self.max_concurrent)
        if self._backend is None:
            self._backend = _create_obstore_backend()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def merge_ranges(
        self, requests: list[COGTileRequest], gap_threshold: int = 1024
    ) -> list[tuple[int, int]]:
        """Merge nearby byte ranges to minimize HTTP requests"""
        if not requests:
            return []

        ranges = [(r.offset, r.offset + r.size) for r in requests]
        ranges.sort()
        merged = [ranges[0]]

        for curr in ranges[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1] + gap_threshold:
                merged[-1] = (prev[0], max(prev[1], curr[1]))
            else:
                merged.append(curr)

        return merged

    async def read_merged_tiles(
        self,
        requests: list[COGTileRequest],
        debug: bool = False,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Parallel tile reading with HTTP/2 multiplexing"""
        if not requests:
            return {}

        # Group by URL for HTTP/2 connection reuse
        url_groups = {}
        for req in requests:
            url_groups.setdefault(req.url, []).append(req)

        results = {}
        for url, group_requests in url_groups.items():
            ranges = self.merge_ranges(group_requests)

            # Process ranges in batches
            for i in range(0, len(ranges), self.batch_size):
                batch = ranges[i : i + self.batch_size]
                batch_tasks = [
                    self._read_and_process_range(
                        url,
                        start,
                        end,
                        [r for r in group_requests if start <= r.offset < end],
                    )
                    for start, end in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks)
                for result in batch_results:
                    results.update(result)

        return results

    async def _read_and_process_range(
        self,
        url: str,
        start: int,
        end: int,
        requests: list[COGTileRequest],
    ) -> dict[tuple[int, int], np.ndarray]:
        """Read and process a byte range with retries"""
        data = await self._read_range(url, start, end)

        # Process tiles in parallel
        tasks = []
        for req in requests:
            offset = req.offset - start
            tile_data = data[offset : offset + req.size]
            tasks.append(
                self._process_tile(tile_data, req.metadata, band_index=req.band_index)
            )

        tiles = await asyncio.gather(*tasks)
        return {(req.row, req.col): tile for req, tile in zip(requests, tiles)}

    async def _read_range(self, url: str, start: int, end: int) -> bytes:
        """Read a byte range via the storage backend."""
        # Local file paths are handled directly, regardless of backend.
        parsed = urlparse(url)
        if parsed.scheme in {"", "file"}:
            path = unquote(parsed.path) if parsed.scheme == "file" else url

            def _read_local() -> bytes:
                with open(path, "rb") as f:
                    f.seek(start)
                    return f.read(end - start)

            async with self.sem:
                return await asyncio.to_thread(_read_local)

        async with self.sem:
            return await self._backend.get_range(url, start, end - start)

    async def _process_tile(
        self,
        data: bytes,
        metadata: CogMetadata,
        *,
        band_index: int | None = None,
    ) -> np.ndarray:
        """Process tile data in thread pool"""
        loop = asyncio.get_running_loop()

        decompressed = await loop.run_in_executor(
            None, self._decompress_tile_sync, data, metadata
        )
        return await loop.run_in_executor(
            None, self._process_tile_sync, decompressed, metadata, band_index
        )

    def _decompress_tile_sync(
        self, data: bytes, metadata: CogMetadata
    ) -> bytes | np.ndarray:
        """Decompress a single tile based on TIFF compression."""
        compression = metadata.compression or 1

        # TIFF Compression tag (259) values we explicitly support.
        # Reference values are commonly used across libtiff/GDAL-derived GeoTIFFs.
        if compression == 1:  # no compression
            return data
        if compression in {8, 32946}:  # deflate (Adobe) / deflate
            return imagecodecs.zlib_decode(data)
        if compression == 5:  # LZW
            return imagecodecs.lzw_decode(data)
        if compression == 32773:  # PackBits
            return imagecodecs.packbits_decode(data)
        if compression == 34925:  # LZMA
            return imagecodecs.lzma_decode(data)
        if compression == 50000:  # ZSTD (GeoTIFF/GDAL extension)
            return imagecodecs.zstd_decode(data)
        if compression == 7:  # JPEG
            raise NotImplementedError(
                "TIFF JPEG compression (tag 259 = 7) is not supported yet. "
                "Many TIFFs require JPEGTables handling (tag 347) during tile decode."
            )
        if compression == 34887:  # LERC
            return imagecodecs.lerc_decode(data)

        raise NotImplementedError(f"Unsupported TIFF compression: {compression}")

    def _process_tile_sync(  # noqa: PLR0912
        self,
        data: bytes | np.ndarray,
        metadata: CogMetadata,
        band_index: int | None = None,
    ) -> np.ndarray:
        """Synchronous tile processing (dtype + predictor)."""
        dtype = (
            np.dtype(metadata.dtype)
            if not hasattr(metadata.dtype, "to_pandas_dtype")
            else np.dtype(metadata.dtype.to_pandas_dtype())
        )

        predictor = metadata.predictor or 1
        samples_per_pixel = int(getattr(metadata, "samples_per_pixel", 1) or 1)
        planar_configuration = int(getattr(metadata, "planar_configuration", 1) or 1)
        if planar_configuration not in (1, 2):
            raise NotImplementedError(
                f"Unsupported PlanarConfiguration: {planar_configuration}"
            )

        if planar_configuration == 1 and samples_per_pixel > 1:
            tile_shape: tuple[int, ...] = (
                metadata.tile_height,
                metadata.tile_width,
                samples_per_pixel,
            )
        else:
            tile_shape = (metadata.tile_height, metadata.tile_width)

        axis_x = -1 if len(tile_shape) == 2 else -2

        # Floating-point predictor (3) must run on the raw byte buffer *before*
        # normal reshape, because it includes a byte-shuffle step.
        # We create an ndarray *view* of the raw bytes with the target dtype/shape,
        # then floatpred_decode unshuffles + undifferences in-place.
        if predictor == 3:
            if dtype.kind != "f":
                raise ValueError(
                    f"TIFF floating predictor requires float dtype, got {dtype}"
                )
            raw = bytes(data) if isinstance(data, np.ndarray) else data
            tile = np.frombuffer(bytearray(raw), dtype=dtype).reshape(tile_shape)
            tile = imagecodecs.floatpred_decode(tile, axis=axis_x)
        elif isinstance(data, np.ndarray):
            tile = data.astype(dtype, copy=False)
            if tile.shape != tile_shape:
                tile = tile.reshape(tile_shape)
            tile = np.ascontiguousarray(tile)
        else:
            tile = np.frombuffer(data, dtype=dtype).reshape(tile_shape).copy()

        if predictor == 2:
            # Horizontal differencing (works for integer dtypes).
            imagecodecs.delta_decode(tile, axis=axis_x, out=tile)
        elif predictor not in (1, 3):
            raise NotImplementedError(f"Unsupported TIFF predictor: {predictor}")

        if planar_configuration == 1 and samples_per_pixel > 1:
            if band_index is None:
                raise NotImplementedError(
                    "Chunky multi-sample TIFF tiles require an explicit band_index to "
                    "select the requested sample (SamplesPerPixel > 1)."
                )
            if band_index < 0 or band_index >= samples_per_pixel:
                raise ValueError(
                    f"band_index {band_index} out of range for {samples_per_pixel} samples"
                )
            tile = tile[:, :, band_index]

        return tile


def compute_tile_indices(
    geometry_bbox: tuple[float, float, float, float],
    transform: list[float],
    tile_size: tuple[int, int],
    image_size: tuple[int, int],
    debug: bool = False,
) -> list[tuple[int, int]]:
    """Compute tile indices that intersect with a geometry's bounding box.

    Parameters
    ----------
    geometry_bbox : tuple of float
        ``(minx, miny, maxx, maxy)`` bounding box in the same CRS as the raster.
    transform : list of float
        Affine transform coefficients for the raster.
    tile_size : tuple of int
        ``(tile_width, tile_height)`` in pixels.
    image_size : tuple of int
        ``(image_width, image_height)`` in pixels.
    debug : bool
        Enable debug logging.

    Returns
    -------
    list of tuple
        ``(row, col)`` indices of tiles that intersect the geometry bbox.
    """
    # Extract parameters
    scale_x, translate_x, scale_y, translate_y = normalize_transform(transform)
    tile_width, tile_height = tile_size
    image_width, image_height = image_size

    # Calculate number of tiles
    tiles_x = (image_width + tile_width - 1) // tile_width
    tiles_y = (image_height + tile_height - 1) // tile_height

    minx, miny, maxx, maxy = geometry_bbox

    if debug:
        logger.info(
            f"""
        Computing tile indices:
        - Bounds: {minx}, {miny}, {maxx}, {maxy}
        - Transform: {scale_x}, {translate_x}, {scale_y}, {translate_y}
        - Image size: {image_width}x{image_height}
        - Tile size: {tile_width}x{tile_height}
        """
        )

    if scale_x == 0.0 or scale_y == 0.0:
        raise ValueError("Invalid transform: zero pixel scale")

    # Convert bounds to pixel coordinates using the transform's actual sign.
    # The transform maps pixel coordinates to map coordinates as:
    #   x = translate_x + col * scale_x
    #   y = translate_y + row * scale_y
    #
    # This supports both north-up (scale_y < 0) and south-up/bottom-up rasters
    # (scale_y > 0) without special-casing.
    col0 = (minx - translate_x) / scale_x
    col1 = (maxx - translate_x) / scale_x
    row0 = (miny - translate_y) / scale_y
    row1 = (maxy - translate_y) / scale_y

    col_min = int(math.floor(min(col0, col1)))
    col_max = int(math.ceil(max(col0, col1))) - 1
    row_min = int(math.floor(min(row0, row1)))
    row_max = int(math.ceil(max(row0, row1))) - 1

    # Clamp to the image extent.
    col_min = max(0, col_min)
    row_min = max(0, row_min)
    col_max = min(image_width - 1, col_max)
    row_max = min(image_height - 1, row_max)

    if debug:
        logger.info(f"Pixel bounds: x({col_min}-{col_max}), y({row_min}-{row_max})")

    # Convert to tile indices
    tile_col_min = max(0, col_min // tile_width)
    tile_col_max = min(tiles_x - 1, col_max // tile_width)
    tile_row_min = max(0, row_min // tile_height)
    tile_row_max = min(tiles_y - 1, row_max // tile_height)

    if debug:
        logger.info(
            f"Tile indices: x({tile_col_min}-{tile_col_max}), y({tile_row_min}-{tile_row_max})"
        )

    # Validate tile ranges
    if tile_col_min > tile_col_max or tile_row_min > tile_row_max:
        if debug:
            logger.info("No valid tiles in range")
        return []

    # Find intersecting tiles using bbox arithmetic (no Shapely)
    intersecting_tiles = []
    for row in range(tile_row_min, tile_row_max + 1):
        for col in range(tile_col_min, tile_col_max + 1):
            # Calculate tile bounds in map coordinates
            x0 = translate_x + col * tile_width * scale_x
            x1 = translate_x + (col + 1) * tile_width * scale_x
            y0 = translate_y + row * tile_height * scale_y
            y1 = translate_y + (row + 1) * tile_height * scale_y

            tile_bbox = (
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1),
            )

            if bbox_intersects(geometry_bbox, tile_bbox):
                intersecting_tiles.append((row, col))
                if debug:
                    logger.info(f"Added intersecting tile: ({row}, {col})")

    if debug:
        logger.info(f"Found {len(intersecting_tiles)} intersecting tiles")

    return intersecting_tiles


def merge_tiles(
    tiles: dict[tuple[int, int], np.ndarray],
    tile_size: tuple[int, int],
    dtype: np.dtype,
    *,
    fill_value: float | int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """Merge multiple tiles into a single array.

    Parameters
    ----------
    tiles : dict
        Mapping of ``(row, col)`` to tile ``np.ndarray``.
    tile_size : tuple of int
        ``(tile_width, tile_height)`` in pixels.
    dtype : numpy.dtype
        Desired output dtype.
    fill_value : float or int
        Fill value for empty regions (outside tile coverage).

    Returns
    -------
    tuple
        ``(merged_array, (min_row, min_col, max_row, max_col), tile_mask)``.
        *tile_mask* is True where a tile contributed pixels.
    """
    if not tiles:
        return np.array([], dtype=dtype), (0, 0, 0, 0), np.zeros((0, 0), dtype=bool)

    # Find bounds
    rows, cols = zip(*tiles.keys())
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    tile_width, tile_height = tile_size

    # Create output array with appropriate fill value.
    height = (max_row - min_row + 1) * tile_height
    width = (max_col - min_col + 1) * tile_width

    out_dtype = np.dtype(dtype)
    fill = out_dtype.type(fill_value)

    merged = np.full((height, width), fill, dtype=out_dtype)
    tile_mask = np.zeros((height, width), dtype=bool)

    # Place tiles with exact positioning
    for (row, col), data in tiles.items():
        if data is not None:  # Handle potentially failed tiles
            y_start = (row - min_row) * tile_height
            x_start = (col - min_col) * tile_width
            y_end = min(y_start + data.shape[0], height)
            x_end = min(x_start + data.shape[1], width)
            merged[y_start:y_end, x_start:x_end] = data[
                : y_end - y_start, : x_end - x_start
            ]

            tile_mask[y_start:y_end, x_start:x_end] = True

    return merged, (min_row, min_col, max_row, max_col), tile_mask


def apply_mask_and_crop(
    data: np.ndarray,
    geojson: dict,
    transform: Affine,
    nodata: float | int | None = None,
    *,
    all_touched: bool = False,
    filled: bool = True,
    fill_value: float | int | None = None,
    return_mask: bool = False,
) -> tuple[np.ndarray, Affine] | tuple[np.ndarray, np.ndarray, Affine]:
    """Apply geometry mask and crop to the valid data region.

    Parameters
    ----------
    data : numpy.ndarray
        2-D raster array.
    geojson : dict
        GeoJSON dict (``{"type": "Polygon", "coordinates": ...}``)
        in the same CRS as *transform*.
    transform : Affine
        Affine transform for *data*.
    nodata : float or int, optional
        Dataset nodata value, if known.
    all_touched : bool
        Passed through to rasterio's ``geometry_mask`` (default False).
    filled : bool
        If True, fill pixels outside the geometry with *fill_value*.
        If False, do not modify values outside the geometry; return a
        validity mask when *return_mask* is True.
    fill_value : float or int, optional
        Fill value used when ``filled=True``. Defaults to *nodata* when
        provided, otherwise 0.
    return_mask : bool
        When True, return a boolean mask where True indicates pixels
        inside the geometry.

    Returns
    -------
    tuple
        ``(masked_data, cropped_transform)`` or
        ``(masked_data, mask, cropped_transform)`` when *return_mask*.
    """

    mask = geometry_mask(
        [geojson],
        out_shape=data.shape,
        transform=transform,
        all_touched=all_touched,
        invert=True,
    )

    fill = (
        data.dtype.type(fill_value)
        if fill_value is not None
        else data.dtype.type(nodata)
        if nodata is not None
        else data.dtype.type(0)
    )

    if not bool(mask.any()):
        # Geometry does not intersect any pixels; return fill-valued array.
        empty_data = np.full(data.shape, fill, dtype=data.dtype)
        empty_mask = np.zeros(data.shape, dtype=bool)
        if return_mask:
            return empty_data, empty_mask, transform
        return empty_data, transform

    # Crop to the geometry's bbox window (rasterio-aligned crop semantics).
    coords = geojson.get("coordinates")
    if not coords:
        raise ValueError("Invalid geojson: missing coordinates")
    # Polygon coordinates: [[[x,y], ...]]; MultiPolygon adds one more nesting.
    while isinstance(coords, list) and coords and isinstance(coords[0], list):
        if (
            coords
            and coords
            and isinstance(coords[0], (tuple, list))
            and len(coords[0]) == 2
        ):
            break
        coords = coords[0]
    xs = [float(x) for x, _y in coords]
    ys = [float(y) for _x, y in coords]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    data_cropped, mask_cropped, cropped_transform = _crop_to_bbox_window(
        data, mask, transform, bbox
    )

    # Apply mask to cropped data (or return unfilled)
    if filled:
        masked_data = np.where(mask_cropped, data_cropped, fill)
    else:
        masked_data = data_cropped

    if return_mask:
        return masked_data, mask_cropped, cropped_transform
    return masked_data, cropped_transform


def _crop_to_mask_bounds(
    data: np.ndarray,
    mask: np.ndarray,
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray, Affine]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    if row_indices.size == 0 or col_indices.size == 0:
        empty = data[:0, :0]
        empty_mask = mask[:0, :0]
        return empty, empty_mask, transform

    row_min, row_max = row_indices[[0, -1]]
    col_min, col_max = col_indices[[0, -1]]
    data_cropped = data[row_min : row_max + 1, col_min : col_max + 1]
    mask_cropped = mask[row_min : row_max + 1, col_min : col_max + 1]
    cropped_transform = Affine(
        transform.a,
        transform.b,
        transform.c + col_min * transform.a,
        transform.d,
        transform.e,
        transform.f + row_min * transform.e,
    )
    return data_cropped, mask_cropped, cropped_transform


def _crop_to_bbox_window(
    data: np.ndarray,
    mask: np.ndarray,
    transform: Affine,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Crop to the outermost pixel window containing *bbox*.

    This matches the windowing semantics used by rasterio's mask() when
    crop=True: floor offsets and ceil width/height of the geometry bounds.
    """
    xmin, ymin, xmax, ymax = bbox
    sx = float(transform.a)
    sy = float(transform.e)
    if sx == 0.0 or sy == 0.0:
        raise ValueError("Invalid transform: zero pixel scale")

    # Match rasterio.features.geometry_window(): compute pixel bounds, then
    # floor offsets and ceil stop.
    tx = float(transform.c)
    ty = float(transform.f)
    col0 = (xmin - tx) / sx
    col1 = (xmax - tx) / sx
    row0 = (ymin - ty) / sy
    row1 = (ymax - ty) / sy
    col_start = int(math.floor(min(col0, col1)))
    col_stop = int(math.ceil(max(col0, col1)))
    row_start = int(math.floor(min(row0, row1)))
    row_stop = int(math.ceil(max(row0, row1)))

    # Clamp to array bounds.
    col_start = max(0, col_start)
    row_start = max(0, row_start)
    col_stop = min(data.shape[1], col_stop)
    row_stop = min(data.shape[0], row_stop)

    if row_stop <= row_start or col_stop <= col_start:
        empty = data[:0, :0]
        empty_mask = mask[:0, :0]
        return empty, empty_mask, transform

    data_cropped = data[row_start:row_stop, col_start:col_stop]
    mask_cropped = mask[row_start:row_stop, col_start:col_stop]
    cropped_transform = Affine(
        transform.a,
        transform.b,
        transform.c + col_start * transform.a,
        transform.d,
        transform.e,
        transform.f + row_start * transform.e,
    )
    return data_cropped, mask_cropped, cropped_transform


def _geometry_window_pixels(
    *,
    geom_geojson: dict,
    transform: Affine,
    raster_width: int | None,
    raster_height: int | None,
) -> tuple[int, int, int, int]:
    """Compute a rasterio-style geometry window in pixel coordinates.

    Returns (row_start, row_stop, col_start, col_stop) where stop indices are
    exclusive.

    This mirrors rasterio.features.geometry_window():
    - bounds() computed in pixel space via transform inversion
    - floor offsets, ceil stop
    - optional intersection with raster extent
    """
    from rasterio.features import bounds as rio_bounds

    left, bottom, right, top = rio_bounds(geom_geojson, transform=~transform)
    cols = [left, right, right, left]
    rows = [top, top, bottom, bottom]
    # Guard against floating-point drift at exact pixel edges (e.g. 1068.0
    # becoming 1068.0000000002 and expanding the window by 1 pixel).
    eps = 1e-9
    row_start = int(math.floor(min(rows) + eps))
    row_stop = int(math.ceil(max(rows) - eps))
    col_start = int(math.floor(min(cols) + eps))
    col_stop = int(math.ceil(max(cols) - eps))

    if raster_width is not None and raster_height is not None:
        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_stop = min(row_stop, raster_height)
        col_stop = min(col_stop, raster_width)

    return row_start, row_stop, col_start, col_stop


def _coverage_mask_for_merged(
    *,
    merged_shape: tuple[int, int],
    raster_width: int,
    raster_height: int,
    merged_global_row0: int,
    merged_global_col0: int,
) -> np.ndarray:
    """Compute raster-coverage mask for a merged tile canvas.

    The merge canvas is tile-aligned, so it may extend beyond the raster
    extent at the bottom/right edges. This mask is True only for pixels
    that correspond to real raster pixels.
    """
    height, width = merged_shape
    max_valid_rows = max(0, min(height, raster_height - merged_global_row0))
    max_valid_cols = max(0, min(width, raster_width - merged_global_col0))
    mask = np.zeros((height, width), dtype=bool)
    if max_valid_rows > 0 and max_valid_cols > 0:
        mask[:max_valid_rows, :max_valid_cols] = True
    return mask


async def read_cog(
    url: str,
    metadata: CogMetadata,
    *,
    band_index: int | None = None,
    geom_array: pa.Array | None = None,
    geom_idx: int = 0,
    geometry_crs: int | None = 4326,
    max_concurrent: int = 150,
    debug: bool = False,
    reader: COGReader | None = None,
    all_touched: bool = False,
    filled: bool = True,
    fill_value: float | int | None = None,
    crop: bool = True,
    mode: Literal["aoi", "window", "full"] = "aoi",
    bounds: tuple[float, float, float, float] | None = None,
    out_shape: tuple[int, int] | None = None,
    mask_geometry: Literal["polygon", "bbox"] = "polygon",
) -> CogReadResult:
    """Primary Rasteret COG read API.

    This returns a :class:`CogReadResult` containing the pixel array,
    its affine transform, and a ``valid_mask`` that is True only for
    pixels inside the requested AOI/window *and* inside raster coverage.

    Defaults are rasterio-aligned:
    - ``all_touched=False``
    - when ``filled=True``: fill with ``nodata`` if known, otherwise 0
    - preserve native dtype by default (no NaN-driven promotion)
    """
    if debug:
        logger.info(f"Reading COG data from {url}")

    if metadata.transform is None:
        empty = np.array([], dtype=np.dtype(metadata.dtype))
        return CogReadResult(
            data=empty,
            transform=Affine.identity(),
            valid_mask=np.zeros((0, 0), dtype=bool),
            fill_value_used=None,
            mode=mode,
        )

    if (
        int(getattr(metadata, "planar_configuration", 1) or 1) == 1
        and int(getattr(metadata, "samples_per_pixel", 1) or 1) > 1
        and band_index is None
    ):
        raise NotImplementedError(
            "Chunky multi-sample TIFFs (PlanarConfiguration=1, SamplesPerPixel>1) "
            "require an explicit band_index to select the requested sample."
        )

    if (
        metadata.tile_offsets is None
        or metadata.tile_byte_counts is None
        or len(metadata.tile_offsets) == 0
        or len(metadata.tile_byte_counts) == 0
    ):
        raise ValueError(
            "Rasteret's tile reader requires a tiled GeoTIFF (TileOffsets/TileByteCounts). "
            "This asset appears to be non-tiled (or missing tile metadata). "
            "Use TorchGeo/rasterio for this file, or convert it to a tiled COG."
        )

    if len(metadata.tile_offsets) != len(metadata.tile_byte_counts):
        raise ValueError(
            "Invalid tile metadata: TileOffsets/TileByteCounts length mismatch "
            f"({len(metadata.tile_offsets)} vs {len(metadata.tile_byte_counts)})."
        )

    # Derive bbox and GeoJSON from GeoArrow array (no Shapely)
    geom_bbox = None
    geom_geojson = None
    if bounds is not None:
        geom_bbox = bounds
        # bounds is always axis-aligned, used as the AOI/window region
        xmin, ymin, xmax, ymax = geom_bbox
        geom_geojson = {
            "type": "Polygon",
            "coordinates": [
                [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
            ],
        }
        intersecting_tiles = compute_tile_indices(
            geometry_bbox=geom_bbox,
            transform=metadata.transform,
            tile_size=(metadata.tile_width, metadata.tile_height),
            image_size=(metadata.width, metadata.height),
            debug=debug,
        )
    elif geom_array is not None:
        # Always compute the input bbox in the geometry CRS first. This lets us
        # provide a clear error when the record CRS is missing and the geometry
        # looks like WGS84 lon/lat.
        geom_bbox = bbox_single(geom_array, geom_idx)

        if (
            geometry_crs == 4326
            and metadata.crs is None
            and geom_bbox is not None
            and -180.0 <= geom_bbox[0] <= 180.0
            and -180.0 <= geom_bbox[2] <= 180.0
            and -90.0 <= geom_bbox[1] <= 90.0
            and -90.0 <= geom_bbox[3] <= 90.0
        ):
            raise ValueError(
                "Record CRS is missing (proj:epsg) so Rasteret cannot transform the "
                "WGS84 query geometry into raster CRS. Fix by adding a per-record "
                "`proj:epsg` column to your record table, or build with "
                "`enrich_cog=True` so Rasteret can infer CRS from GeoTIFF headers "
                "when available."
            )

        needs_crs_transform = (
            geometry_crs is not None
            and metadata.crs is not None
            and geometry_crs != metadata.crs
        )
        if needs_crs_transform:
            # CRS-transform the geometry, get GeoJSON dict in target CRS
            geom_geojson = transform_coords(
                geom_array, geom_idx, geometry_crs, metadata.crs
            )
            # Extract bbox from the transformed GeoJSON coordinates
            from rasteret.core.geometry import bbox_from_geojson_coords

            geom_bbox = bbox_from_geojson_coords(geom_geojson)

            if debug:
                logger.info(
                    f"Transformed geometry bbox "
                    f"(EPSG:{geometry_crs} -> EPSG:{metadata.crs}): "
                    f"{geom_bbox}"
                )
        else:
            geom_geojson = to_rasterio_geojson(geom_array, geom_idx)

        if geom_bbox is not None and mask_geometry == "bbox":
            xmin, ymin, xmax, ymax = geom_bbox
            geom_geojson = {
                "type": "Polygon",
                "coordinates": [
                    [
                        (xmin, ymin),
                        (xmax, ymin),
                        (xmax, ymax),
                        (xmin, ymax),
                        (xmin, ymin),
                    ]
                ],
            }

        # Get tiles that intersect with geometry bbox
        intersecting_tiles = compute_tile_indices(
            geometry_bbox=geom_bbox,
            transform=metadata.transform,
            tile_size=(metadata.tile_width, metadata.tile_height),
            image_size=(metadata.width, metadata.height),
            debug=debug,
        )
    else:
        # Read all tiles if no geometry provided
        tiles_x = (metadata.width + metadata.tile_width - 1) // metadata.tile_width
        tiles_y = (metadata.height + metadata.tile_height - 1) // metadata.tile_height
        intersecting_tiles = [(r, c) for r in range(tiles_y) for c in range(tiles_x)]

    if not intersecting_tiles:
        empty = np.array([], dtype=np.dtype(metadata.dtype))
        return CogReadResult(
            data=empty,
            transform=Affine.identity(),
            valid_mask=np.zeros((0, 0), dtype=bool),
            fill_value_used=None,
            mode=mode,
        )

    # Create tile requests
    requests = []
    tiles_x = (metadata.width + metadata.tile_width - 1) // metadata.tile_width
    tiles_y = (metadata.height + metadata.tile_height - 1) // metadata.tile_height
    expected_tiles = tiles_x * tiles_y
    if expected_tiles > len(metadata.tile_offsets):
        raise ValueError(
            "Unsupported tiled GeoTIFF layout: tile offset table is shorter than expected "
            f"({len(metadata.tile_offsets)} < {expected_tiles})."
        )

    for row, col in intersecting_tiles:
        tile_idx = row * tiles_x + col
        if tile_idx >= len(metadata.tile_offsets):
            if debug:
                logger.warning(f"Tile index {tile_idx} out of bounds")
            continue

        requests.append(
            COGTileRequest(
                url=url,
                offset=metadata.tile_offsets[tile_idx],
                size=metadata.tile_byte_counts[tile_idx],
                row=row,
                col=col,
                metadata=metadata,
                band_index=band_index,
            )
        )

    # Use COGReader for efficient tile reading
    if reader is None:
        async with COGReader(max_concurrent=max_concurrent) as local_reader:
            tiles = await local_reader.read_merged_tiles(requests, debug=debug)
    else:
        tiles = await reader.read_merged_tiles(requests, debug=debug)

    if not tiles:
        empty = np.array([], dtype=np.dtype(metadata.dtype))
        return CogReadResult(
            data=empty,
            transform=Affine.identity(),
            valid_mask=np.zeros((0, 0), dtype=bool),
            fill_value_used=None,
            mode=mode,
        )

    # Merge tiles and handle transforms
    native_dtype = np.dtype(
        metadata.dtype
        if not hasattr(metadata.dtype, "to_pandas_dtype")
        else metadata.dtype.to_pandas_dtype()
    )
    fill_value_used: float | int = (
        fill_value
        if fill_value is not None
        else metadata.nodata
        if metadata.nodata is not None
        else 0
    )

    merged_data, tile_bounds, tile_mask = merge_tiles(
        tiles,
        (metadata.tile_width, metadata.tile_height),
        dtype=native_dtype,
        fill_value=fill_value_used,
    )

    if debug:
        logger.info(
            f"""
        Merged Data:
        - Shape: {merged_data.shape}
        - Bounds: {bounds}
        - Data Range: {np.nanmin(merged_data)}-{np.nanmax(merged_data)}
        """
        )

    # Calculate transform for merged data
    min_row, min_col, max_row, max_col = tile_bounds
    scale_x, translate_x, scale_y, translate_y = normalize_transform(metadata.transform)
    src_transform = Affine(scale_x, 0.0, translate_x, 0.0, scale_y, translate_y)

    merged_transform = Affine(
        scale_x,
        0,
        translate_x + min_col * metadata.tile_width * scale_x,
        0,
        scale_y,
        translate_y + min_row * metadata.tile_height * scale_y,
    )

    merged_global_row0 = min_row * metadata.tile_height
    merged_global_col0 = min_col * metadata.tile_width

    coverage = _coverage_mask_for_merged(
        merged_shape=merged_data.shape,
        raster_width=metadata.width,
        raster_height=metadata.height,
        merged_global_row0=merged_global_row0,
        merged_global_col0=merged_global_col0,
    )
    # Where tiles exist AND within raster extent.
    valid_mask = coverage & tile_mask

    if mode == "full" or (geom_geojson is None and mode == "aoi"):
        # For full reads, crop to exact raster extent (remove padded tile multiples).
        max_rows = max(
            0, min(merged_data.shape[0], metadata.height - merged_global_row0)
        )
        max_cols = max(
            0, min(merged_data.shape[1], metadata.width - merged_global_col0)
        )
        merged_data = merged_data[:max_rows, :max_cols]
        valid_mask = valid_mask[:max_rows, :max_cols]
        merged_transform = merged_transform

    elif mode == "aoi" and geom_geojson is not None:
        if crop:
            row_start, row_stop, col_start, col_stop = _geometry_window_pixels(
                geom_geojson=geom_geojson,
                transform=src_transform,
                raster_width=metadata.width,
                raster_height=metadata.height,
            )

            local_r0 = row_start - merged_global_row0
            local_c0 = col_start - merged_global_col0
            local_r1 = row_stop - merged_global_row0
            local_c1 = col_stop - merged_global_col0

            local_r0 = max(0, local_r0)
            local_c0 = max(0, local_c0)
            local_r1 = min(merged_data.shape[0], local_r1)
            local_c1 = min(merged_data.shape[1], local_c1)

            if local_r1 <= local_r0 or local_c1 <= local_c0:
                empty = merged_data[:0, :0]
                empty_mask = valid_mask[:0, :0]
                return CogReadResult(
                    data=empty,
                    transform=Affine.identity(),
                    valid_mask=empty_mask,
                    fill_value_used=fill_value_used if filled else None,
                    mode=mode,
                )

            merged_data = merged_data[local_r0:local_r1, local_c0:local_c1]
            valid_mask = valid_mask[local_r0:local_r1, local_c0:local_c1]
            merged_transform = Affine(
                src_transform.a,
                src_transform.b,
                src_transform.c + col_start * src_transform.a,
                src_transform.d,
                src_transform.e,
                src_transform.f + row_start * src_transform.e,
            )

            # Apply geometry mask in the cropped window grid (matches rasterio.mask).
            geom_mask = geometry_mask(
                [geom_geojson],
                out_shape=merged_data.shape,
                transform=merged_transform,
                all_touched=all_touched,
                invert=True,
            )
            valid_mask = valid_mask & geom_mask

        else:
            # Geometry mask in raster CRS (tile-aligned canvas).
            geom_mask = geometry_mask(
                [geom_geojson],
                out_shape=merged_data.shape,
                transform=merged_transform,
                all_touched=all_touched,
                invert=True,
            )
            valid_mask = valid_mask & geom_mask

        if filled and merged_data.size:
            merged_data = np.where(
                valid_mask,
                merged_data,
                np.array(fill_value_used, dtype=merged_data.dtype),
            )

    elif mode == "window":
        if geom_bbox is None:
            raise ValueError("window mode requires bounds or geom_array")
        if metadata.transform is None:
            raise ValueError("window mode requires a valid transform")

        xmin, ymin, xmax, ymax = geom_bbox
        window_geojson = {
            "type": "Polygon",
            "coordinates": [
                [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
            ],
        }
        row_start, row_stop, col_start, col_stop = _geometry_window_pixels(
            geom_geojson=window_geojson,
            transform=src_transform,
            raster_width=None,
            raster_height=None,
        )
        out_h = max(row_stop - row_start, 0)
        out_w = max(col_stop - col_start, 0)
        if out_shape is not None:
            out_h, out_w = out_shape

        out_data = np.full(
            (out_h, out_w), native_dtype.type(fill_value_used), dtype=native_dtype
        )
        out_mask = np.zeros((out_h, out_w), dtype=bool)

        # Overlap region in global pixel coords.
        overlap_row0 = max(row_start, merged_global_row0)
        overlap_col0 = max(col_start, merged_global_col0)
        overlap_row1 = min(row_start + out_h, merged_global_row0 + merged_data.shape[0])
        overlap_col1 = min(col_start + out_w, merged_global_col0 + merged_data.shape[1])
        if overlap_row1 > overlap_row0 and overlap_col1 > overlap_col0:
            src_r0 = overlap_row0 - merged_global_row0
            src_c0 = overlap_col0 - merged_global_col0
            src_r1 = overlap_row1 - merged_global_row0
            src_c1 = overlap_col1 - merged_global_col0
            dst_r0 = overlap_row0 - row_start
            dst_c0 = overlap_col0 - col_start
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out_data[dst_r0:dst_r1, dst_c0:dst_c1] = merged_data[
                src_r0:src_r1, src_c0:src_c1
            ]
            out_mask[dst_r0:dst_r1, dst_c0:dst_c1] = valid_mask[
                src_r0:src_r1, src_c0:src_c1
            ]

        merged_data = out_data
        valid_mask = out_mask
        merged_transform = Affine(
            src_transform.a,
            0,
            src_transform.c + col_start * src_transform.a,
            0,
            src_transform.e,
            src_transform.f + row_start * src_transform.e,
        )

    else:
        raise ValueError(f"Unknown read mode: {mode}")

    if debug:
        logger.info(
            f"""
        Final Output:
        - Shape: {merged_data.shape}
        - Transform: {merged_transform}
        - Data Range: {np.nanmin(merged_data)}-{np.nanmax(merged_data)}
        """
        )

    return CogReadResult(
        data=merged_data,
        transform=merged_transform,
        valid_mask=valid_mask,
        fill_value_used=fill_value_used if filled else None,
        mode=mode,
    )
