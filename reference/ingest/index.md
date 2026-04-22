# rasteret.ingest

Ingest drivers: source-specific logic that feeds into the Collection contract.

## ingest

Ingest builders: source-specific logic that feeds into the Collection contract.

Each builder knows how to read records from one source type (STAC API, Parquet record tables, etc.) and normalise them into an Arrow table that satisfies the Collection contract columns (`id`, `datetime`, `geometry`, `assets`, `bbox`, plus optional `proj:epsg`, `{band}_metadata`, `year`, `month`).

The shared normalisation layer lives in :mod:`rasteret.ingest.normalize`.

### Classes

#### CollectionBuilder

```python
CollectionBuilder(
    *,
    name: str = "",
    data_source: str = "",
    workspace_dir: str | Path | None = None,
)
```

Bases: `ABC`

Abstract base class for all collection builders.

Subclasses implement :meth:`build` to acquire data from their specific source, normalise it, and return a `Collection`.

Parameters:

| Name            | Type   | Description                                             | Default |
| --------------- | ------ | ------------------------------------------------------- | ------- |
| `name`          | `str`  | Human-readable collection name.                         | `''`    |
| `data_source`   | `str`  | Data source identifier for band mapping and URL policy. | `''`    |
| `workspace_dir` | `Path` | If set, persist the collection as partitioned Parquet.  | `None`  |

Source code in `src/rasteret/ingest/base.py`

```python
def __init__(
    self,
    *,
    name: str = "",
    data_source: str = "",
    workspace_dir: str | Path | None = None,
) -> None:
    self.name = name
    self.data_source = data_source
    if workspace_dir is None:
        self.workspace_dir: str | Path | None = None
    elif isinstance(workspace_dir, Path):
        self.workspace_dir = workspace_dir
    else:
        ws = str(workspace_dir)
        if "://" in ws and not ws.startswith("file://"):
            self.workspace_dir = ws
        else:
            self.workspace_dir = Path(ws)
```

##### Functions

###### build

```python
build(**kwargs: Any) -> 'Collection'
```

Build and return a Collection.

Each subclass decides how to acquire data and normalise it into the Collection contract.

Source code in `src/rasteret/ingest/base.py`

```python
@abstractmethod
def build(self, **kwargs: Any) -> "Collection":
    """Build and return a Collection.

    Each subclass decides how to acquire data and normalise it
    into the Collection contract.
    """
    ...
```

#### RecordTableBuilder

```python
RecordTableBuilder(
    path: str | Path,
    *,
    data_source: str = "",
    column_map: dict[str, str] | None = None,
    href_column: str | None = None,
    band_index_map: dict[str, int] | None = None,
    url_rewrite_patterns: dict[str, str] | None = None,
    filesystem: Any | None = None,
    columns: list[str] | None = None,
    filter_expr: Expression | None = None,
    name: str = "",
    workspace_dir: str | Path | None = None,
    enrich_cog: bool = False,
    band_codes: list[str] | None = None,
    max_concurrent: int = 300,
    backend: StorageBackend | None = None,
)
```

Bases: `CollectionBuilder`

Build a Collection from an existing Parquet/GeoParquet table.

Reads a Parquet record table where each row is a raster item with at minimum the four contract columns (`id`, `datetime`, `geometry`, `assets`), or columns that can be normalised into them via `column_map`, `href_column`, and `band_index_map`.

When `enrich_cog=True`, the builder parses COG headers from the asset URLs and adds `{band}_metadata` struct columns, making the resulting Collection suitable for fast tiled reads and TorchGeo integration.

Parameters:

| Name                   | Type             | Description                                                                                                                                                      | Default    |
| ---------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `path`                 | `str or Path`    | Path/URI to the Parquet/GeoParquet file or dataset directory.                                                                                                    | *required* |
| `data_source`          | `str`            | Data-source identifier for the resulting Collection.                                                                                                             | `''`       |
| `column_map`           | `dict`           | {source_name: contract_name} alias map applied before normalisation. Source columns are preserved; Rasteret adds the contract-name aliases without copying data. | `None`     |
| `href_column`          | `str`            | Column containing COG URLs. When set and assets is absent after aliasing, the builder constructs the assets struct from this column and band_index_map.          | `None`     |
| `band_index_map`       | `dict`           | {band_code: sample_index} for multi-band COGs. Used with href_column to build per-band asset references.                                                         | `None`     |
| `url_rewrite_patterns` | `dict`           | {source_prefix: target_prefix} patterns applied to URLs during assets construction (e.g. S3 -> HTTPS rewriting).                                                 | `None`     |
| `filesystem`           | `FileSystem`     | PyArrow filesystem for reading remote URIs (e.g. S3FileSystem(anonymous=True)).                                                                                  | `None`     |
| `columns`              | `list of str`    | Scan-time column projection.                                                                                                                                     | `None`     |
| `filter_expr`          | `Expression`     | Scan-time predicate pushdown.                                                                                                                                    | `None`     |
| `enrich_cog`           | `bool`           | If True, parse COG headers from asset URLs and add per-band metadata columns. Default False.                                                                     | `False`    |
| `band_codes`           | `list of str`    | Bands to enrich. If omitted, all bands found in the assets column are enriched.                                                                                  | `None`     |
| `max_concurrent`       | `int`            | Maximum concurrent HTTP connections for COG header parsing.                                                                                                      | `300`      |
| `name`                 | `str`            | Collection name. Passed through to the normalisation layer.                                                                                                      | `''`       |
| `workspace_dir`        | `str or Path`    | If provided, persist the resulting Collection as Parquet here.                                                                                                   | `None`     |
| `backend`              | `StorageBackend` | I/O backend for authenticated range reads during COG header parsing.                                                                                             | `None`     |

Source code in `src/rasteret/ingest/parquet_record_table.py`

```python
def __init__(
    self,
    path: str | Path,
    *,
    data_source: str = "",
    column_map: dict[str, str] | None = None,
    href_column: str | None = None,
    band_index_map: dict[str, int] | None = None,
    url_rewrite_patterns: dict[str, str] | None = None,
    filesystem: Any | None = None,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
    name: str = "",
    workspace_dir: str | Path | None = None,
    enrich_cog: bool = False,
    band_codes: list[str] | None = None,
    max_concurrent: int = 300,
    backend: StorageBackend | None = None,
) -> None:
    super().__init__(
        name=name,
        data_source=data_source,
        workspace_dir=workspace_dir,
    )
    self.path = str(path)
    self.column_map = column_map or {}
    self.href_column = href_column
    self.band_index_map = band_index_map
    self.url_rewrite_patterns = url_rewrite_patterns or {}
    self._filesystem = filesystem
    self.columns = columns
    self.filter_expr = filter_expr
    self.enrich_cog = enrich_cog
    self.band_codes = band_codes
    self.max_concurrent = max_concurrent
    self._backend = backend
```

##### Functions

###### build

```python
build(**kwargs: Any) -> 'Collection'
```

Read the record table and return a normalized Collection.

Pipeline: read -> alias -> prepare -> enrich -> normalize.

Parameters:

| Name       | Type  | Description                                                                                | Default |
| ---------- | ----- | ------------------------------------------------------------------------------------------ | ------- |
| `**kwargs` | `Any` | name and workspace_dir can be passed here to override the values set at construction time. | `{}`    |

Returns:

| Type         | Description |
| ------------ | ----------- |
| `Collection` |             |

Source code in `src/rasteret/ingest/parquet_record_table.py`

```python
def build(self, **kwargs: Any) -> "Collection":
    """Read the record table and return a normalized Collection.

    Pipeline: read -> alias -> prepare -> enrich -> normalize.

    Parameters
    ----------
    **kwargs
        ``name`` and ``workspace_dir`` can be passed here to override
        the values set at construction time.

    Returns
    -------
    Collection
    """
    name = kwargs.get("name", self.name)
    workspace_dir = kwargs.get("workspace_dir", self.workspace_dir)

    table = self._read_table()
    table = _apply_column_map_aliases(table, self.column_map)
    table = self._prepare_table(table)

    if self.enrich_cog:
        table = self._enrich(table)

    return build_collection_from_table(
        table,
        name=name or self._default_name(),
        data_source=self.data_source,
        workspace_dir=workspace_dir,
    )
```

#### StacCollectionBuilder

```python
StacCollectionBuilder(
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
    strict_band_map_validation: bool = False,
)
```

Bases: `CollectionBuilder`

Build a Collection from a STAC API search or static catalog.

Searches a STAC API (or traverses a static STAC catalog when `static_catalog=True`), parses COG headers for tile metadata, and produces a Parquet-backed Collection with per-band acceleration columns.

Source code in `src/rasteret/ingest/stac_indexer.py`

```python
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
    strict_band_map_validation: bool = False,
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
    self.strict_band_map_validation = strict_band_map_validation
```

##### Attributes

###### band_map

```python
band_map: dict[str, str]
```

Get band mapping for current collection.

##### Functions

###### build

```python
build(**kwargs: Any)
```

Build a Collection from STAC (sync wrapper).

Accepts `bbox`, `date_range`, `query` keyword arguments. Delegates to the async :meth:`build_index`.

Source code in `src/rasteret/ingest/stac_indexer.py`

```python
def build(self, **kwargs: Any):
    """Build a Collection from STAC (sync wrapper).

    Accepts ``bbox``, ``date_range``, ``query`` keyword arguments.
    Delegates to the async :meth:`build_index`.
    """
    from rasteret.core.utils import run_sync

    return run_sync(self.build_index(**kwargs))
```

###### build_index

```python
build_index(
    bbox: BoundingBox | None = None,
    date_range: DateRange | None = None,
    query: dict[str, Any] | None = None,
)
```

Build GeoParquet collection from STAC search (async).

Returns a :class:`~rasteret.core.collection.Collection`.

Source code in `src/rasteret/ingest/stac_indexer.py`

```python
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
```

### Functions

#### add_band_metadata_columns

```python
add_band_metadata_columns(
    table: Table,
    band_codes: list[str],
    processed_items: list[dict],
) -> Table
```

Append `{band}_metadata` struct columns from parsed COG headers.

Parameters:

| Name              | Type           | Description                                                                                                                    | Default    |
| ----------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `table`           | `Table`        | Arrow table with an id column.                                                                                                 | *required* |
| `band_codes`      | `list of str`  | Band codes to create columns for.                                                                                              | *required* |
| `processed_items` | `list of dict` | Each dict must have record_id (record identifier), band, and the twelve COG metadata fields (width, height, tile_width, etc.). | *required* |

Returns:

| Type    | Description                                         |
| ------- | --------------------------------------------------- |
| `Table` | Input table with {band}\_metadata columns appended. |

Source code in `src/rasteret/ingest/enrich.py`

```python
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
```

#### build_url_index_from_assets

```python
build_url_index_from_assets(
    table: Table, band_codes: list[str] | None = None
) -> dict[str, dict[str, dict[str, Any]]]
```

Build `{record_id: {band_code: {url, band_index}}}` from `assets`.

Parameters:

| Name         | Type          | Description                                                | Default    |
| ------------ | ------------- | ---------------------------------------------------------- | ---------- |
| `table`      | `Table`       | Must contain id and assets columns.                        | *required* |
| `band_codes` | `list of str` | If given, only include these bands. Otherwise include all. | `None`     |

Returns:

| Type   | Description                                                                                                                                                                                                                                                                                                                                   |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dict` | Nested mapping of record ID -> band code -> asset reference dict. The asset reference dict contains: url: asset href (string) band_index: optional band/sample index within a multi-sample tiled GeoTIFF (defaults to 0). This is used during enrichment to select the correct TileOffsets/TileByteCounts segment for planar separate assets. |

Source code in `src/rasteret/ingest/enrich.py`

```python
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
```

#### enrich_table_with_cog_metadata

```python
enrich_table_with_cog_metadata(
    table: Table,
    url_index: dict[str, dict[str, dict[str, Any]]],
    band_codes: list[str],
    *,
    max_concurrent: int = 300,
    batch_size: int = 100,
    backend: StorageBackend | None = None,
) -> Table
```

Parse COG headers and add `{band}_metadata` columns.

This is the high-level entry point for builders that have a `url_index` but have not yet parsed COG headers.

Parameters:

| Name             | Type             | Description                                                                                                                 | Default    |
| ---------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `table`          | `Table`          | Arrow table with an id column.                                                                                              | *required* |
| `url_index`      | `dict`           | Mapping of record_id -> band_code -> asset reference dict: {record_id: {band_code: {"url": str, "band_index": int}}}.       | *required* |
| `band_codes`     | `list of str`    | Band codes to create metadata columns for.                                                                                  | *required* |
| `max_concurrent` | `int`            | Maximum concurrent HTTP connections.                                                                                        | `300`      |
| `batch_size`     | `int`            | Batch size for COG header parsing.                                                                                          | `100`      |
| `backend`        | `StorageBackend` | I/O backend for authenticated range reads during COG header parsing. When omitted, uses the default auto-detecting backend. | `None`     |

Returns:

| Type    | Description                                          |
| ------- | ---------------------------------------------------- |
| `Table` | Table with {band}\_metadata struct columns appended. |

Source code in `src/rasteret/ingest/enrich.py`

```python
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
```

#### build_collection_from_table

```python
build_collection_from_table(
    table: Table,
    *,
    name: str = "",
    description: str = "",
    data_source: str = "",
    date_range: tuple[str, str] | None = None,
    workspace_dir: str | Path | None = None,
    partition_cols: Sequence[str] = ("year", "month"),
) -> Any
```

Normalise an Arrow table into a Collection.

Validates the Collection contract columns, adds `bbox` and partition columns when missing, and optionally materialises to Parquet.

Parameters:

| Name             | Type              | Description                                     | Default                                                      |
| ---------------- | ----------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| `table`          | `Table`           | Arrow table with at least the required columns. | *required*                                                   |
| `name`           | `str`             | Human-readable collection name.                 | `''`                                                         |
| `description`    | `str`             | Free-text description.                          | `''`                                                         |
| `data_source`    | `str`             | Data source identifier (e.g. "sentinel-2-l2a"). | `''`                                                         |
| `date_range`     | \`tuple[str, str] | None\`                                          | (start, end) ISO date strings. Used for collection metadata. |
| `workspace_dir`  | \`str             | Path                                            | None\`                                                       |
| `partition_cols` | `Sequence[str]`   | Columns to partition by when writing Parquet.   | `('year', 'month')`                                          |

Returns:

| Type         | Description |
| ------------ | ----------- |
| `Collection` |             |

Source code in `src/rasteret/ingest/normalize.py`

```python
def build_collection_from_table(
    table: pa.Table,
    *,
    name: str = "",
    description: str = "",
    data_source: str = "",
    date_range: tuple[str, str] | None = None,
    workspace_dir: str | Path | None = None,
    partition_cols: Sequence[str] = ("year", "month"),
) -> Any:
    """Normalise an Arrow table into a Collection.

    Validates the Collection contract columns, adds ``bbox``
    and partition columns when missing, and optionally materialises
    to Parquet.

    Parameters
    ----------
    table:
        Arrow table with at least the required columns.
    name:
        Human-readable collection name.
    description:
        Free-text description.
    data_source:
        Data source identifier (e.g. ``"sentinel-2-l2a"``).
    date_range:
        ``(start, end)`` ISO date strings.  Used for collection metadata.
    workspace_dir:
        If provided, persist the collection as partitioned Parquet here.
    partition_cols:
        Columns to partition by when writing Parquet.

    Returns
    -------
    Collection
    """
    from rasteret.core.collection import Collection

    missing = REQUIRED_COLUMNS - set(table.schema.names)
    if missing:
        raise ValueError(f"Table is missing required columns: {missing}")

    table, transformed_geometry = _ensure_footprint_geometry_crs84(table)
    if transformed_geometry:
        table = _drop_column_if_present(table, "bbox")

    table = normalize_raster_crs_sidecars(table)

    # Add canonical bbox if absent.
    if "bbox" not in table.schema.names:
        table = _add_bbox_struct(table)

    # Add year/month partition columns if absent.
    datetime_col = table.column("datetime")
    if "year" not in table.schema.names:
        # Ensure the column is a timestamp type.
        if not pa.types.is_timestamp(datetime_col.type):
            datetime_col = pc.cast(datetime_col, pa.timestamp("us"))
        table = table.append_column("year", pc.year(datetime_col))
    if "month" not in table.schema.names:
        if not pa.types.is_timestamp(datetime_col.type):
            datetime_col = pc.cast(datetime_col, pa.timestamp("us"))
        table = table.append_column("month", pc.month(datetime_col))

    start_date = datetime.fromisoformat(date_range[0]) if date_range else None
    end_date = datetime.fromisoformat(date_range[1]) if date_range else None

    collection = Collection(
        dataset=ds.dataset(table),
        name=name,
        description=description,
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
    )

    if workspace_dir:
        collection.export(workspace_dir)

    return collection
```
