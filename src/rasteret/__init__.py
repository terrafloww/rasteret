# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa
    import pyarrow.dataset as pads

    from rasteret.catalog import DatasetDescriptor
    from rasteret.cloud import StorageBackend
    from rasteret.core.collection import Collection

# Library-level NullHandler: application code controls logging output.
logging.getLogger("rasteret").addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)


def version() -> str:
    """Return the installed rasteret package version."""
    try:
        return get_version("rasteret")
    except PackageNotFoundError:
        return "0.0.0+local"


__version__ = version()


def _validate_bbox(bbox: tuple[float, float, float, float] | None) -> None:
    """Raise ``ValueError`` if *bbox* coordinates are inverted."""
    if bbox is None:
        return
    west, south, east, north = bbox
    problems: list[str] = []
    if west >= east:
        problems.append(f"west ({west}) must be less than east ({east})")
    if south >= north:
        problems.append(f"south ({south}) must be less than north ({north})")
    if problems:
        raise ValueError(
            f"Invalid bbox: {' and '.join(problems)}. "
            "Expected format: (west, south, east, north)."
        )


def _validate_date_range(date_range: tuple[str, str] | None) -> None:
    """Raise ``ValueError`` if *date_range* start is after end."""
    if date_range is None:
        return
    start_str, end_str = date_range
    from datetime import date as _date

    try:
        start = _date.fromisoformat(start_str[:10])
        end = _date.fromisoformat(end_str[:10])
    except (ValueError, TypeError):
        return
    if start > end:
        raise ValueError(
            f"Invalid date_range: start '{start_str}' is after end '{end_str}'."
        )


def build_from_stac(
    *,
    name: str,
    stac_api: str,
    collection: str,
    data_source: str | None = None,
    band_map: dict[str, str] | None = None,
    band_index_map: dict[str, int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    date_range: tuple[str, str] | None = None,
    workspace_dir: str | Path | None = None,
    force: bool = False,
    max_concurrent: int = 50,
    cloud_config: Any = None,
    query: dict[str, Any] | None = None,
    backend: "StorageBackend | None" = None,
    static_catalog: bool = False,
) -> "Collection":
    """Build or load a local Parquet-backed collection from a STAC API.

    Searches STAC, parses COG headers for tile metadata, and stores
    everything in a local Parquet index. On subsequent calls with the same
    parameters, Rasteret reuses the cached index (no STAC query / header
    parsing), but pixel reads still fetch remote tiles unless assets are local.

    Parameters
    ----------
    name : str
        Logical name for the collection.
    stac_api : str
        STAC API endpoint URL.
    collection : str
        STAC collection ID (e.g. ``"sentinel-2-l2a"``).
    data_source : str, optional
        Optional Rasteret data-source key used for band mapping and cloud config
        lookup. Defaults to *collection*. Use this to namespace provider-specific
        conventions (e.g. ``"earthsearch/sentinel-2-l2a"`` vs
        ``"pc/sentinel-2-l2a"``) and avoid collisions.
    band_map : dict, optional
        Optional mapping of band code to STAC asset key. When omitted,
        Rasteret falls back to built-in mappings for known collections.
    band_index_map : dict, optional
        Optional mapping of band code to the 0-based band/sample index within a
        multi-sample GeoTIFF asset. Required when multiple requested bands map
        to the same STAC asset key (e.g. a single multi-band ``"image"`` COG).
    bbox : tuple of float
        ``(minx, miny, maxx, maxy)`` bounding box for the search.
    date_range : tuple of str
        ``(start, end)`` ISO date strings.
    workspace_dir : str or Path, optional
        Directory for the local cache. Defaults to ``~/rasteret_workspace``.
    force : bool
        Rebuild even if a cache already exists.
    max_concurrent : int
        Maximum concurrent COG header fetch operations.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    query : dict, optional
        Additional STAC search query parameters.
    backend : StorageBackend, optional
        I/O backend for authenticated range reads during COG header
        parsing.  See :func:`create_backend`.

    Returns
    -------
    Collection
    """
    _validate_bbox(bbox)
    _validate_date_range(date_range)

    from rasteret.core.collection import Collection

    workspace_dir_path = Path(workspace_dir or Path.home() / "rasteret_workspace")
    resolved_source = data_source or str(collection)

    if date_range is not None:
        collection_name = Collection.create_name(name, date_range, resolved_source)
    else:
        # Static catalogs may not have a date range; use name + collection id.
        safe = name.lower().replace(" ", "-").replace("/", "-")
        collection_name = f"{safe}_{resolved_source.replace('/', '-')}"
    collection_path = workspace_dir_path / f"{collection_name}_stac"

    if collection_path.exists() and not force:
        return Collection._load_cached(collection_path)

    from rasteret.cloud import CloudConfig, backend_config_from_cloud_config
    from rasteret.ingest.stac_indexer import StacCollectionBuilder

    resolved_cloud_config = cloud_config or CloudConfig.get_config(resolved_source)

    if backend is None and resolved_cloud_config:
        from rasteret.fetch.cog import _create_obstore_backend

        cfg = backend_config_from_cloud_config(resolved_cloud_config)
        if cfg:
            backend = _create_obstore_backend(**cfg)

    builder = StacCollectionBuilder(
        data_source=resolved_source,
        stac_collection=str(collection),
        stac_api=stac_api,
        workspace_dir=collection_path,
        name=collection_name,
        band_map=band_map,
        band_index_map=band_index_map,
        cloud_config=resolved_cloud_config,
        max_concurrent=max_concurrent,
        backend=backend,
        static_catalog=static_catalog,
    )

    async def _build() -> Collection:
        return await builder.build_index(
            bbox=list(bbox) if bbox else None,
            date_range=date_range,
            query=query,
        )

    from rasteret.core.utils import run_sync

    return run_sync(_build())


def _auto_backend_for_descriptor(descriptor: "DatasetDescriptor") -> Any:
    """Create an I/O backend from descriptor auth hints, or return ``None``.

    Uses ``s3_credentials_url`` to pick the right obstore credential
    provider.  Returns ``None`` when no
    auto-detection is possible so the caller can fall back gracefully.
    """
    if not descriptor.s3_credentials_url:
        return None

    url = descriptor.s3_credentials_url.lower()
    try:
        if "earthdata" in url or "lpdaac" in url:
            from obstore.auth.earthdata import NasaEarthdataCredentialProvider

            return create_backend(
                credential_provider=NasaEarthdataCredentialProvider(
                    credentials_url=descriptor.s3_credentials_url,
                ),
                # NASA LP DAAC data is in us-west-2; S3 direct access
                # requires requests from the same region.
                region="us-west-2",
            )
    except Exception as exc:
        logger.warning(
            "Could not auto-create backend for %s: %s",
            descriptor.id,
            exc,
        )
    return None


def build(
    dataset: str,
    *,
    name: str,
    bbox: tuple[float, float, float, float] | None = None,
    date_range: tuple[str, str] | None = None,
    workspace_dir: str | Path | None = None,
    force: bool = False,
    max_concurrent: int = 50,
    query: dict[str, Any] | None = None,
    prefer_geoparquet: bool = False,
    stac_api: str | None = None,
    cloud_config: Any = None,
    backend: "StorageBackend | None" = None,
) -> "Collection":
    """Build a Collection from a registered dataset.

    Looks up *dataset* in the :class:`~rasteret.catalog.DatasetRegistry`
    and routes to :func:`build_from_stac` or :func:`build_from_table`
    based on the descriptor's access fields.

    For descriptors backed only by ``geoparquet_uri`` (for example local
    collections registered with :func:`register_local`), ``bbox`` and
    ``date_range`` are optional and ignored.

    For auth-required datasets, Rasteret can auto-create a backend from a
    descriptor's ``s3_credentials_url`` when no explicit *backend* is passed.
    This requires valid credentials in the environment (or ``~/.netrc``)
    for the relevant provider.

    Parameters
    ----------
    dataset : str
        Registry ID (e.g. ``"earthsearch/sentinel-2-l2a"``).
    name : str
        Logical name for the collection.
    bbox : tuple of float, optional
        ``(minx, miny, maxx, maxy)`` bounding box.
        Required for STAC-backed descriptors.
    date_range : tuple of str, optional
        ``(start, end)`` ISO date strings.
        Required for STAC-backed descriptors.
    workspace_dir : str or Path, optional
        Cache directory. Defaults to ``~/rasteret_workspace``.
    force : bool
        Rebuild even if a cache already exists.
    max_concurrent : int
        Maximum concurrent COG header fetches.
    query : dict, optional
        Additional STAC search parameters.
    prefer_geoparquet : bool
        Use the GeoParquet path when available.
    stac_api : str, optional
        Override the descriptor's default STAC API endpoint.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    backend : StorageBackend, optional
        I/O backend for authenticated range reads.  When omitted,
        Rasteret auto-creates one for known auth-required datasets.
        See :func:`create_backend`.

    Returns
    -------
    Collection

    Raises
    ------
    KeyError
        If *dataset* is not in the registry.
    ValueError
        If the descriptor has no configured access method, or if
        auth is required but no backend could be created.
    """
    _validate_bbox(bbox)
    _validate_date_range(date_range)

    from rasteret.catalog import DatasetRegistry

    descriptor = DatasetRegistry.get(dataset)
    if descriptor is None:
        available = [d.id for d in DatasetRegistry.list()]
        raise KeyError(
            f"Dataset '{dataset}' not found in registry. " f"Available: {available}"
        )

    # Auto-create backend for datasets that provide an explicit backend hint.
    #
    # Important: `requires_auth` alone is not enough to decide whether a backend
    # is mandatory. Some datasets work via URL signing or requester-pays config
    # (cloud_config) without needing an obstore credential
    # provider. We only fail fast when the descriptor provides a concrete hint
    # that build-time enrichment needs a backend (currently `s3_credentials_url`).
    resolved_backend = backend
    if resolved_backend is None and descriptor.s3_credentials_url:
        resolved_backend = _auto_backend_for_descriptor(descriptor)
        if resolved_backend is None and descriptor.requires_auth:
            extra_hint = ""
            url = (descriptor.s3_credentials_url or "").lower()
            if "earthdata" in url or "lpdaac" in url:
                extra_hint = (
                    " If you're using an Earthdata-backed dataset, install "
                    '"rasteret[earthdata]" to enable the Earthdata credential provider.'
                )
            raise ValueError(
                f"Dataset '{descriptor.id}' requires authentication for build-time "
                f"COG header enrichment but no backend could be created. Either pass "
                f"backend= explicitly (see rasteret.create_backend()), or configure "
                f"credentials via environment variables / ~/.netrc.{extra_hint}"
            )

    resolved_workspace: str | Path | None = workspace_dir
    if workspace_dir is not None:
        workspace = Path(workspace_dir)
        if workspace.name.endswith(("_stac", "_records")):
            resolved_workspace = workspace
        else:
            resolved_workspace = workspace / f"{name}_records"

    # Resolve band_codes from descriptor for GeoParquet enrichment.
    descriptor_band_codes = (
        list(descriptor.band_map.keys()) if descriptor.band_map else None
    )

    # Local descriptors are already-built Collections. Prefer loading them as-is
    # rather than re-running enrichment/build logic.
    if descriptor.spatial_coverage == "local" and descriptor.geoparquet_uri:
        local_path = Path(descriptor.geoparquet_uri).expanduser()
        if local_path.exists():
            return load(local_path, name=name)

    def _build_from_geoparquet() -> "Collection":
        """Route a GeoParquet-backed descriptor through build_from_table.

        Constructs filter_expr from bbox_columns + bbox/date_range,
        filesystem for anonymous S3, and passes descriptor normalisation
        fields through.
        """
        import pyarrow.dataset as pads

        geoparquet_uri = descriptor.geoparquet_uri or ""

        # --- Construct filter_expr from bbox_columns + date_range ---
        filter_parts: list[pads.Expression] = []

        if bbox and descriptor.bbox_columns:
            bc = descriptor.bbox_columns
            if "minx" in bc and "maxx" in bc and "miny" in bc and "maxy" in bc:
                filter_parts.append(pads.field(bc["minx"]) <= bbox[2])
                filter_parts.append(pads.field(bc["maxx"]) >= bbox[0])
                filter_parts.append(pads.field(bc["miny"]) <= bbox[3])
                filter_parts.append(pads.field(bc["maxy"]) >= bbox[1])

        # Date range filter: find the source column that maps to "datetime".
        if date_range and descriptor.column_map:
            datetime_source = None
            for src, dst in descriptor.column_map.items():
                if dst == "datetime":
                    datetime_source = src
                    break
            if datetime_source and datetime_source != "datetime":
                start_year = int(date_range[0][:4])
                end_year = int(date_range[1][:4])
                filter_parts.append(pads.field(datetime_source) >= start_year)
                filter_parts.append(pads.field(datetime_source) <= end_year)

        filter_expr = None
        if filter_parts:
            filter_expr = filter_parts[0]
            for part in filter_parts[1:]:
                filter_expr = filter_expr & part

        # --- Construct filesystem for anonymous S3 URIs ---
        fs = None
        if geoparquet_uri.startswith("s3://") and not descriptor.requires_auth:
            try:
                import pyarrow.fs as pafs

                cloud_region = "us-west-2"
                if descriptor.cloud_config:
                    cloud_region = descriptor.cloud_config.get("region", cloud_region)
                fs = pafs.S3FileSystem(anonymous=True, region=cloud_region)
            except Exception as exc:
                logger.warning(
                    "Could not create anonymous S3 filesystem for %s: %s",
                    descriptor.id,
                    exc,
                )

        # --- Strip scheme when filesystem is provided ---
        # PyArrow expects bare "bucket/key" paths with an explicit filesystem,
        # not full "s3://bucket/key" URIs.
        read_path = geoparquet_uri
        if fs is not None and geoparquet_uri.startswith("s3://"):
            read_path = geoparquet_uri[len("s3://") :]

        # --- URL rewrite patterns from cloud_config ---
        url_rewrite_patterns = None
        if descriptor.cloud_config:
            url_rewrite_patterns = descriptor.cloud_config.get("url_patterns")

        return build_from_table(
            read_path,
            name=name,
            data_source=descriptor.stac_collection or descriptor.id,
            workspace_dir=resolved_workspace,
            column_map=descriptor.column_map,
            href_column=descriptor.href_column,
            band_index_map=descriptor.band_index_map,
            url_rewrite_patterns=url_rewrite_patterns,
            filesystem=fs,
            filter_expr=filter_expr,
            enrich_cog=True,
            band_codes=descriptor_band_codes,
            cloud_config=cloud_config,
            max_concurrent=max_concurrent,
            force=force,
            backend=resolved_backend,
        )

    # GeoParquet-first path: descriptors that have no STAC API, or user
    # explicitly prefers GeoParquet.
    if descriptor.geoparquet_uri and (
        prefer_geoparquet or not (descriptor.stac_api and descriptor.stac_collection)
    ):
        return _build_from_geoparquet()

    # STAC path (default for STAC-backed descriptors)
    api = stac_api or descriptor.stac_api
    if api and (descriptor.stac_collection or descriptor.static_catalog):
        if not descriptor.static_catalog and (bbox is None or date_range is None):
            raise ValueError(
                f"Dataset '{dataset}' requires bbox and date_range for STAC queries."
            )
        return build_from_stac(
            name=name,
            stac_api=api,
            collection=descriptor.stac_collection or descriptor.id,
            data_source=descriptor.id,
            band_map=descriptor.band_map,
            band_index_map=descriptor.band_index_map,
            bbox=bbox,
            date_range=date_range,
            workspace_dir=workspace_dir,
            force=force,
            max_concurrent=max_concurrent,
            cloud_config=cloud_config,
            query=query,
            backend=resolved_backend,
            static_catalog=descriptor.static_catalog,
        )

    # GeoParquet fallback
    if descriptor.geoparquet_uri:
        return _build_from_geoparquet()

    raise ValueError(
        f"Dataset '{dataset}' has no STAC API or GeoParquet URI configured."
    )


def register(descriptor: "DatasetDescriptor") -> None:
    """Register a dataset descriptor in the global registry.

    Parameters
    ----------
    descriptor : DatasetDescriptor
        The descriptor to register.  See :class:`rasteret.catalog.DatasetDescriptor`.
    """
    from rasteret.catalog import DatasetRegistry

    DatasetRegistry.register(descriptor)


def register_local(
    dataset_id: str,
    path: str | Path,
    *,
    name: str | None = None,
    description: str = "",
    data_source: str = "",
    persist: bool = True,
    registry_path: str | Path | None = None,
) -> "DatasetDescriptor":
    """Register a local Parquet collection as a dataset descriptor.

    This is useful when you want local/shared Collection Parquet artifacts
    to appear in ``DatasetRegistry`` and CLI dataset commands.

    Parameters
    ----------
    dataset_id : str
        Descriptor id (e.g. ``"local/my-collection"``).
    path : str or Path
        Path to a local Collection Parquet file or directory.
    name : str, optional
        Human-readable dataset name. Defaults to the loaded collection name.
    description : str
        Optional one-line description.
    data_source : str
        Optional data source id. If omitted, inferred from collection metadata.
    persist : bool
        When ``True`` (default), save descriptor to local registry JSON so it
        is auto-loaded in future sessions.
    registry_path : str or Path, optional
        Override local registry JSON path (defaults to
        ``~/.rasteret/datasets.local.json`` or ``RASTERET_LOCAL_DATASETS_PATH``).

    Returns
    -------
    DatasetDescriptor
    """
    from rasteret.catalog import (
        DatasetDescriptor,
        DatasetRegistry,
        save_local_descriptor,
    )
    from rasteret.constants import BandRegistry
    from rasteret.core.utils import infer_data_source

    local_path = Path(path).expanduser()
    collection = load(local_path)
    resolved_source = data_source or infer_data_source(collection)
    temporal_range: tuple[str, str] | None = None

    if collection.dataset is not None and "datetime" in collection.dataset.schema.names:
        values = collection.dataset.to_table(columns=["datetime"]).column("datetime")
        datetimes = [value for value in values.to_pylist() if value is not None]
        if datetimes:
            start = min(datetimes).date().isoformat()
            end = max(datetimes).date().isoformat()
            temporal_range = (start, end)

    band_map = BandRegistry.get(resolved_source) if resolved_source else {}
    descriptor = DatasetDescriptor(
        id=dataset_id,
        name=name or collection.name or dataset_id,
        description=description or collection.description,
        geoparquet_uri=str(local_path),
        stac_collection=resolved_source or None,
        band_map=band_map or None,
        separate_files=True,
        spatial_coverage="local",
        temporal_range=temporal_range,
        requires_auth=False,
    )
    DatasetRegistry.register(descriptor)
    if persist:
        save_local_descriptor(descriptor, registry_path)
    return descriptor


def load(path: str | Path, name: str = "") -> "Collection":
    """Load an existing Rasteret Collection from Parquet.

    Use this when you've previously built a collection with
    :func:`build_from_stac` or :func:`build_from_table` and want
    to reload it.

    Parameters
    ----------
    path : str or Path
        Path to the Parquet file or dataset directory.
    name : str
        Optional name override.

    Returns
    -------
    Collection
    """
    from rasteret.core.collection import Collection

    return Collection.from_parquet(path, name=name)


def build_from_table(
    path: "str | Path | pa.Table | pads.Dataset",
    *,
    name: str = "",
    data_source: str = "",
    workspace_dir: str | Path | None = None,
    column_map: dict[str, str] | None = None,
    href_column: str | None = None,
    band_index_map: dict[str, int] | None = None,
    url_rewrite_patterns: dict[str, str] | None = None,
    filesystem: Any | None = None,
    columns: list[str] | None = None,
    filter_expr: Any | None = None,
    enrich_cog: bool = False,
    band_codes: list[str] | None = None,
    cloud_config: Any = None,
    max_concurrent: int = 300,
    force: bool = False,
    backend: "StorageBackend | None" = None,
) -> "Collection":
    """Build a Collection from a Parquet/GeoParquet record table.

    A record table is a Parquet dataset where each row is a raster item
    (satellite scene, drone image, derived product, etc.) with at minimum
    ``id``, ``datetime``, ``geometry``, ``assets``, or columns that can
    be normalised into them via ``column_map`` and ``href_column``.

    When ``enrich_cog=True``, COG headers are parsed from the asset URLs
    and cached as ``{band}_metadata`` struct columns in the Parquet index,
    enabling fast tiled reads and TorchGeo integration.

    When *name* is provided and *workspace_dir* is omitted, the collection
    is persisted to ``~/rasteret_workspace/{name}_records/`` so that it is
    discoverable via :meth:`Collection.list_collections` and the CLI.

    Parameters
    ----------
    path : str, Path, or pyarrow object
        Path/URI to a Parquet/GeoParquet file or dataset directory, **or**
        an in-memory Arrow object (``pyarrow.Table`` or ``pyarrow.dataset.Dataset``).
    name : str
        Optional collection name.  When given without *workspace_dir*,
        the collection is cached in the default workspace.
    data_source : str
        Data source identifier for band mapping and URL policy.
    workspace_dir : str or Path, optional
        Persist the collection as partitioned Parquet at this path.
        Defaults to ``~/rasteret_workspace/{name}_records/`` when
        *name* is provided.
    column_map : dict, optional
        ``{source_name: contract_name}`` alias map.  Source columns are
        preserved; contract-name columns are added as zero-copy aliases.
    href_column : str, optional
        Column containing COG URLs.  When set and ``assets`` is absent
        after aliasing, the normalisation layer constructs the ``assets``
        struct from this column and ``band_index_map``.
    band_index_map : dict, optional
        ``{band_code: sample_index}`` for multi-band COGs.
    url_rewrite_patterns : dict, optional
        ``{source_prefix: target_prefix}`` for URL rewriting during
        assets construction.
    filesystem : pyarrow.fs.FileSystem, optional
        PyArrow filesystem for reading remote URIs (e.g.
        ``S3FileSystem(anonymous=True)``).
    columns : list of str, optional
        Scan-time column projection.
    filter_expr : pyarrow.dataset.Expression, optional
        Scan-time predicate pushdown.
    enrich_cog : bool
        Parse COG headers and add per-band metadata columns.
    band_codes : list of str, optional
        Bands to enrich.  Defaults to all bands in ``assets``.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    max_concurrent : int
        Maximum concurrent HTTP connections for COG header parsing.
    force : bool
        Rebuild even if a cached collection already exists at the
        resolved workspace path.
    backend : StorageBackend, optional
        I/O backend for authenticated range reads during COG header
        parsing.  See :func:`create_backend`.

    Returns
    -------
    Collection
    """
    from rasteret.core.collection import Collection
    from rasteret.ingest.normalize import build_collection_from_table
    from rasteret.ingest.parquet_record_table import (
        RecordTableBuilder,
        _apply_column_map_aliases,
    )

    # Resolve workspace path with _records suffix convention so that
    # list_collections() and the CLI can discover this collection.
    resolved_workspace: str | Path | None = workspace_dir
    if workspace_dir is not None:
        ws = Path(workspace_dir)
        if not ws.name.endswith(("_stac", "_records")):
            resolved_workspace = ws / f"{name}_records" if name else ws
    elif name:
        resolved_workspace = Path.home() / "rasteret_workspace" / f"{name}_records"

    # Cache hit: reuse existing collection.
    if resolved_workspace is not None:
        rw = Path(resolved_workspace)
        if rw.exists() and not force:
            return Collection._load_cached(rw)

    # Arrow-native path: accept an in-memory Arrow table / dataset.
    import pyarrow as pa
    import pyarrow.dataset as pads

    if isinstance(path, (pa.Table, pads.Dataset)):
        if isinstance(path, pads.Dataset):
            table = path.to_table(columns=columns, filter=filter_expr)
        else:
            table = path
            if columns or filter_expr:
                dataset = pads.dataset(path)
                table = dataset.to_table(columns=columns, filter=filter_expr)

        table = _apply_column_map_aliases(table, column_map)

        # Run the same normalisation that RecordTableBuilder._prepare_table does.
        _builder = RecordTableBuilder.__new__(RecordTableBuilder)
        _builder.href_column = href_column
        _builder.band_index_map = band_index_map
        _builder.url_rewrite_patterns = url_rewrite_patterns or {}
        table = _builder._prepare_table(table)

        if enrich_cog:
            from rasteret.core.utils import run_sync
            from rasteret.ingest.enrich import (
                build_url_index_from_assets,
                enrich_table_with_cog_metadata,
            )

            url_index = build_url_index_from_assets(table, band_codes)
            resolved_band_codes = band_codes or sorted(
                {band for bands in url_index.values() for band in bands}
            )
            if url_index and resolved_band_codes:
                table = run_sync(
                    enrich_table_with_cog_metadata(
                        table,
                        url_index,
                        resolved_band_codes,
                        max_concurrent=max_concurrent,
                        backend=backend,
                    )
                )

        return build_collection_from_table(
            table,
            name=name or "record_table",
            data_source=data_source,
            workspace_dir=resolved_workspace,
        )

    builder = RecordTableBuilder(
        path,
        data_source=data_source,
        workspace_dir=resolved_workspace,
        column_map=column_map,
        href_column=href_column,
        band_index_map=band_index_map,
        url_rewrite_patterns=url_rewrite_patterns,
        filesystem=filesystem,
        columns=columns,
        filter_expr=filter_expr,
        enrich_cog=enrich_cog,
        band_codes=band_codes,
        max_concurrent=max_concurrent,
        backend=backend,
    )
    return builder.build(name=name, workspace_dir=resolved_workspace)


def create_backend(
    credential_provider: Any = None,
    cloud_config: Any = None,
    region: str | None = None,
    default_s3_config: dict[str, str] | None = None,
) -> Any:
    """Create an I/O backend for authenticated cloud reads.

    Pass the result as ``backend=`` to
    :meth:`~rasteret.core.collection.Collection.get_xarray` or
    :meth:`~rasteret.core.collection.Collection.get_gdf`.

    Parameters
    ----------
    credential_provider : object, optional
        An obstore credential provider, e.g.
        ``PlanetaryComputerCredentialProvider``,
        ``NasaEarthdataCredentialProvider``.
    cloud_config : CloudConfig, optional
        Cloud configuration for S3 URL rewriting and per-bucket overrides.
    region : str, optional
        Convenience alias for ``default_s3_config={"region": region}``.
    default_s3_config : dict, optional
        Default S3Store config applied to all buckets that don't have
        per-bucket overrides (e.g. ``{"region": "us-west-2"}``).

    Examples
    --------
    >>> from obstore.auth.planetary_computer import PlanetaryComputerCredentialProvider
    >>> pc_asset_url = "https://naipeuwest.blob.core.windows.net/naip/v002/"
    >>> backend = rasteret.create_backend(
    ...     credential_provider=PlanetaryComputerCredentialProvider(pc_asset_url)
    ... )
    >>> ds = collection.get_xarray(geometries=aoi, bands=["B04"], backend=backend)
    """
    from rasteret.cloud import s3_overrides_from_config
    from rasteret.fetch.cog import _create_obstore_backend

    resolved_default_s3_config = default_s3_config
    if resolved_default_s3_config is None and region is not None:
        resolved_default_s3_config = {"region": region}

    s3_overrides = s3_overrides_from_config(cloud_config) if cloud_config else None
    url_patterns = cloud_config.url_patterns if cloud_config else None
    return _create_obstore_backend(
        s3_overrides=s3_overrides,
        credential_provider=credential_provider,
        default_s3_config=resolved_default_s3_config,
        url_patterns=url_patterns,
    )


def __getattr__(name: str) -> Any:
    """Lazy-load heavy modules to keep import-time coupling low."""
    if name == "Collection":
        from rasteret.core.collection import Collection

        return Collection
    if name == "CloudConfig":
        from rasteret.cloud import CloudConfig

        return CloudConfig
    if name == "BandRegistry":
        from rasteret.constants import BandRegistry

        return BandRegistry
    if name == "DatasetDescriptor":
        from rasteret.catalog import DatasetDescriptor

        return DatasetDescriptor
    if name == "DatasetRegistry":
        from rasteret.catalog import DatasetRegistry

        return DatasetRegistry
    raise AttributeError(f"module 'rasteret' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose a curated top-level surface for REPL/autocomplete."""
    return sorted(set(__all__))


__all__ = [
    "BandRegistry",
    "CloudConfig",
    "Collection",
    "DatasetDescriptor",
    "DatasetRegistry",
    "__version__",
    "build",
    "build_from_stac",
    "build_from_table",
    "create_backend",
    "load",
    "register",
    "register_local",
    "version",
]
