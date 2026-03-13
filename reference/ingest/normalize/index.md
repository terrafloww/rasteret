# rasteret.ingest.normalize

Shared normalization layer: raw Arrow table to Collection.

## normalize

Shared normalisation layer: raw Arrow table -> Collection.

Every ingest driver calls :func:`build_collection_from_table` as its final step to validate column contract, add partition columns, and construct a :class:`~rasteret.core.collection.Collection`.

### Functions

#### parse_epsg

```python
parse_epsg(crs_value: object) -> int | None
```

Extract an integer EPSG code from a CRS value.

Accepts `int` (returned as-is), `"EPSG:32632"`-style strings, or `None`. Returns `None` when parsing fails.

Source code in `src/rasteret/ingest/normalize.py`

```python
def parse_epsg(crs_value: object) -> int | None:
    """Extract an integer EPSG code from a CRS value.

    Accepts ``int`` (returned as-is), ``"EPSG:32632"``-style strings,
    or ``None``.  Returns ``None`` when parsing fails.
    """
    if crs_value is None:
        return None
    if isinstance(crs_value, int):
        return crs_value
    if isinstance(crs_value, str):
        s = crs_value.strip().upper()
        if s.startswith("EPSG:"):
            try:
                return int(s.split(":", 1)[1])
            except ValueError:
                return None
    return None
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
