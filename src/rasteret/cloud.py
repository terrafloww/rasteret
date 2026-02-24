# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Storage configuration for a data source.

    Built-in configs may be pre-registered for some well-known data sources that
    need URL rewriting or requester-pays access.
    Register your own via :meth:`register`::

        CloudConfig.register("my-collection", CloudConfig(
            provider="aws", requester_pays=True, region="eu-west-1",
        ))
    """

    provider: str
    requester_pays: bool = False
    region: str = "us-west-2"
    url_patterns: dict[str, str] = field(default_factory=dict)

    # Class-level registry keyed by lower-cased collection id.
    _configs: ClassVar[dict[str, CloudConfig]] = {}

    @classmethod
    def register(cls, collection_id: str, config: CloudConfig) -> None:
        """Register a cloud config for a collection id."""
        cls._configs[collection_id.lower()] = config

    @classmethod
    def get_config(cls, data_source: str) -> CloudConfig | None:
        """Look up cloud config for *data_source*."""
        return cls._configs.get(data_source.lower())


# Pre-register built-in sources.
CloudConfig.register(
    "sentinel-2-l2a",
    CloudConfig(provider="aws", requester_pays=False, region="us-west-2"),
)

CloudConfig.register(
    "landsat-c2-l2",
    CloudConfig(provider="aws", requester_pays=True, region="us-west-2"),
)


# ---------------------------------------------------------------------------
# StorageBackend: pluggable byte-range fetchers for COGReader
# ---------------------------------------------------------------------------


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal protocol for range-based reads from cloud storage.

    Implement this to plug in a custom I/O backend (e.g. obstore,
    fsspec, or a mocked reader for tests).
    """

    async def get_range(self, url: str, start: int, length: int) -> bytes: ...

    async def get_ranges(
        self, url: str, ranges: list[tuple[int, int]]
    ) -> list[bytes]: ...


class ObstoreBackend:
    """StorageBackend backed by the ``obstore`` library.

    Wraps ``obstore.get_range_async`` / ``obstore.get_ranges_async``.
    Pass any ``obstore.store.*Store`` instance.

    Parameters
    ----------
    store : obstore Store
        Any obstore store (``S3Store``, ``HTTPStore``, etc.).
    url_prefix : str, optional
        URL prefix to strip before passing paths to obstore.  obstore
        expects paths relative to the store root, but rasteret works with
        full URLs.  For example, if COG URLs look like
        ``https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/...``
        and the store is ``HTTPStore.from_url("https://sentinel-cogs.s3.us-west-2.amazonaws.com/")``,
        set ``url_prefix="https://sentinel-cogs.s3.us-west-2.amazonaws.com/"``.
    client_options : dict, optional
        ``ClientConfig`` options forwarded to ``HTTPStore.from_url`` when
        constructing the store.  Ignored if a pre-built *store* is provided.
    """

    def __init__(
        self,
        store: object,
        url_prefix: str = "",
        client_options: dict[str, object] | None = None,
    ) -> None:
        self._store = store
        self._url_prefix = url_prefix
        self._client_options = client_options

    def _resolve_path(self, url: str) -> str:
        if self._url_prefix and url.startswith(self._url_prefix):
            return url[len(self._url_prefix) :]
        return url

    async def get_range(self, url: str, start: int, length: int) -> bytes:
        import obstore as obs

        path = self._resolve_path(url)
        buf = await obs.get_range_async(self._store, path, start=start, length=length)
        return bytes(buf)

    async def get_ranges(self, url: str, ranges: list[tuple[int, int]]) -> list[bytes]:
        import obstore as obs

        path = self._resolve_path(url)
        starts, lengths = zip(*ranges)
        buffers = await obs.get_ranges_async(
            self._store, path, starts=list(starts), lengths=list(lengths)
        )
        return [bytes(b) for b in buffers]


def rewrite_url(url: str, config: CloudConfig | None) -> str:
    """Apply URL pattern rewrites without presigning.

    Pure string transformation.  Returns the original URL if no
    patterns match or *config* is ``None``.
    """
    if config and config.url_patterns and isinstance(url, str):
        for http_pattern, s3_pattern in config.url_patterns.items():
            if url.startswith(http_pattern):
                return url.replace(http_pattern, s3_pattern)
    return url


def s3_overrides_from_config(config: CloudConfig | None) -> dict[str, dict[str, str]]:
    """Derive per-bucket S3Store config from a :class:`CloudConfig`.

    Returns a mapping of ``{bucket: {config_key: value}}`` suitable for
    passing to :class:`~rasteret.fetch.cog._AutoObstoreBackend`.

    Example: ``CloudConfig(requester_pays=True, url_patterns={...})``
    yields ``{"usgs-landsat": {"request_payer": "true", "region": "us-west-2"}}``.
    """
    if config is None:
        return {}
    overrides: dict[str, dict[str, str]] = {}
    for s3_pattern in (config.url_patterns or {}).values():
        # Extract bucket from s3://bucket/ pattern
        if s3_pattern.startswith("s3://"):
            bucket = s3_pattern.split("/")[2]
            bucket_config: dict[str, str] = {"region": config.region}
            if config.requester_pays:
                bucket_config["request_payer"] = "true"
            else:
                bucket_config["skip_signature"] = "true"
            overrides[bucket] = bucket_config
    return overrides


def backend_config_from_cloud_config(config: CloudConfig | None) -> dict:
    """Extract :func:`~rasteret.fetch.cog._create_obstore_backend` kwargs from a CloudConfig."""
    if config is None:
        return {}
    result: dict = {}
    overrides = s3_overrides_from_config(config)
    if overrides:
        result["s3_overrides"] = overrides
    if config.url_patterns:
        result["url_patterns"] = config.url_patterns
    # When requester_pays is True, set default_s3_config so that buckets
    # discovered at runtime (e.g. from s3:// sibling assets in STAC)
    # are configured correctly even if they don't appear in url_patterns.
    if config.requester_pays:
        result["default_s3_config"] = {
            "request_payer": "true",
            "region": config.region,
        }
    return result


__all__ = [
    "CloudConfig",
    "ObstoreBackend",
    "StorageBackend",
    "backend_config_from_cloud_config",
    "rewrite_url",
    "s3_overrides_from_config",
]
