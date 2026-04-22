# rasteret.ingest.stac_indexer

STAC ingest driver: STAC API search to Collection via GeoParquet.

## stac_indexer

STAC collection indexer: STAC API / static catalog -> Collection.

Searches a STAC API (or traverses a static STAC catalog), parses COG headers for tile metadata, and normalises results into the Collection contract via the shared :func:`~rasteret.ingest.normalize.build_collection_from_table` layer.

Static catalogs (`catalog.json` files on S3 with no `/search` endpoint) are supported via `pystac.Catalog.from_file()` with client-side bbox and date filtering.

### Classes

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
