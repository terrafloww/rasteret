# rasteret.ingest.parquet_record_table

Parquet/Arrow record-table driver for building Collections from tabular raster indexes. For user-facing examples, see [Build from Parquet and Arrow Tables](https://terrafloww.github.io/rasteret/how-to/build-from-tables/index.md).

## parquet_record_table

Record-table collection builder.

Reads a Parquet/GeoParquet **record table** (one row per raster item) and normalizes it into a :class:`~rasteret.core.collection.Collection` via :func:`~rasteret.ingest.normalize.build_collection_from_table`.

Terminology

- *Record table* -- a tabular index that enumerates raster items (satellite scenes, drone images, derived products, grid cells, etc.). It may come from stac-geoparquet, a lab-specific registry, or a custom export.
- *Collection Parquet* -- Rasteret's normalized, runtime-ready Parquet dataset that follows the `Schema Contract <explanation/schema-contract>`\_ docs page.

### Classes

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

### Functions

#### prepare_record_table

```python
prepare_record_table(
    table: Table,
    *,
    href_column: str | None = None,
    band_index_map: dict[str, int] | None = None,
    url_rewrite_patterns: dict[str, str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> Table
```

Normalise column types and construct `assets` when absent.

This is a pure function (no instance state) so it can be used from both :class:`RecordTableBuilder` and the in-memory `build_from_table()` path without constructing a builder object.

Steps:

1. Auto-coerce `id`: integer -> string.
1. Auto-coerce `datetime`: integer year -> timestamp.
1. Construct `assets` from *href_column* + *band_index_map*.
1. Derive legacy `proj:epsg` from a `crs` column when present.

Source code in `src/rasteret/ingest/parquet_record_table.py`

```python
def prepare_record_table(
    table: pa.Table,
    *,
    href_column: str | None = None,
    band_index_map: dict[str, int] | None = None,
    url_rewrite_patterns: dict[str, str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> pa.Table:
    """Normalise column types and construct ``assets`` when absent.

    This is a pure function (no instance state) so it can be used from both
    :class:`RecordTableBuilder` and the in-memory ``build_from_table()`` path
    without constructing a builder object.

    Steps:

    1. Auto-coerce ``id``: integer -> string.
    2. Auto-coerce ``datetime``: integer year -> timestamp.
    3. Construct ``assets`` from *href_column* + *band_index_map*.
    4. Derive legacy ``proj:epsg`` from a ``crs`` column when present.
    """
    names = set(table.schema.names)
    rewrites = url_rewrite_patterns or {}
    required = set(required_columns) if required_columns is not None else None

    # --- id: int -> string ---
    if (
        "id" in names
        and pa.types.is_integer(table.schema.field("id").type)
        and (required is None or "id" in required)
    ):
        table = table.set_column(
            table.schema.get_field_index("id"),
            "id",
            pc.cast(table.column("id"), pa.string()),
        )

    # --- datetime: int year -> timestamp ---
    if (
        "datetime" in names
        and pa.types.is_integer(table.schema.field("datetime").type)
        and (required is None or "datetime" in required)
    ):
        years = table.column("datetime").to_pylist()
        timestamps = pa.array(
            [datetime(int(y), 1, 1) if y is not None else None for y in years],
            type=pa.timestamp("us"),
        )
        table = table.set_column(
            table.schema.get_field_index("datetime"),
            "datetime",
            timestamps,
        )

    # --- assets: construct from href_column + band_index_map ---
    if (
        "assets" not in names
        and href_column
        and band_index_map
        and (required is None or "assets" in required)
    ):
        if href_column not in names:
            raise ValueError(
                f"href_column '{href_column}' not found in table. "
                f"Available: {sorted(names)}"
            )
        urls = table.column(href_column).to_pylist()
        assets_list: list[dict[str, dict[str, object]]] = []
        for url in urls:
            if url is None:
                assets_list.append({})
                continue
            rewritten = _rewrite_url_simple(str(url), rewrites)
            assets_list.append(
                {
                    band: {"href": rewritten, "band_index": idx}
                    for band, idx in band_index_map.items()
                }
            )
        table = table.append_column("assets", pa.array(assets_list))

    return normalize_raster_crs_sidecars(table, required_columns=required_columns)
```
