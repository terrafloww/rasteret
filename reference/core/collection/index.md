# rasteret.core.collection

The central `Collection` class: Arrow dataset wrapper with filtering, output adapters, and persistence.

Most-used read APIs on `Collection`:

- `get_numpy(...)` -> NumPy arrays (`[N, H, W]` single-band, `[N, C, H, W]` multi-band)
- `get_xarray(...)` -> `xarray.Dataset`
- `get_gdf(...)` -> `geopandas.GeoDataFrame`
- `sample_points(...)` -> `pyarrow.Table` (point-value table)
- `to_torchgeo_dataset(...)` -> TorchGeo-compatible dataset

## collection

### Classes

#### Collection

```python
Collection(
    dataset: Dataset | None = None,
    hf_streaming: HFStreamingSource | None = None,
    collection_path: str | None = None,
    record_index_path: str | None = None,
    record_index_field_roles: dict[str, str] | None = None,
    record_index_column_map: dict[str, str] | None = None,
    record_index_href_column: str | None = None,
    record_index_band_index_map: dict[str, int]
    | None = None,
    record_index_url_rewrite_patterns: dict[str, str]
    | None = None,
    record_index_filesystem: Any | None = None,
    surface_fields: dict[str, list[str]] | None = None,
    filter_capabilities: dict[str, list[str]] | None = None,
    record_index_filter_expr: Expression | None = None,
    wide_filter_expr: Expression | None = None,
    name: str = "",
    description: str = "",
    data_source: str = "",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
)
```

A collection of raster data with flexible initialization.

Collections can be created from:

- Local partitioned datasets
- Single Arrow tables

Collections maintain efficient partitioned storage when using files.

Examples:

##### From partitioned dataset

```pycon
>>> collection = Collection.from_parquet("path/to/dataset")
```

##### Filter and process

```pycon
>>> filtered = collection.subset(cloud_cover_lt=20)
>>> ds = filtered.get_xarray(...)
```

Initialize a Collection.

Parameters:

| Name           | Type                | Description                                                                    | Default |
| -------------- | ------------------- | ------------------------------------------------------------------------------ | ------- |
| `dataset`      | `Dataset`           | Backing Arrow dataset. None creates an empty or non-Dataset-backed collection. | `None`  |
| `hf_streaming` | `HFStreamingSource` | Hugging Face streaming-backed metadata source.                                 | `None`  |
| `name`         | `str`               | Human-readable collection name.                                                | `''`    |
| `description`  | `str`               | Free-text description.                                                         | `''`    |
| `data_source`  | `str`               | Data source identifier (e.g. "sentinel-2-l2a").                                | `''`    |
| `start_date`   | `datetime`          | Collection temporal start.                                                     | `None`  |
| `end_date`     | `datetime`          | Collection temporal end.                                                       | `None`  |

Source code in `src/rasteret/core/collection.py`

```python
def __init__(
    self,
    dataset: ds.Dataset | None = None,
    hf_streaming: HFStreamingSource | None = None,
    collection_path: str | None = None,
    record_index_path: str | None = None,
    record_index_field_roles: dict[str, str] | None = None,
    record_index_column_map: dict[str, str] | None = None,
    record_index_href_column: str | None = None,
    record_index_band_index_map: dict[str, int] | None = None,
    record_index_url_rewrite_patterns: dict[str, str] | None = None,
    record_index_filesystem: Any | None = None,
    surface_fields: dict[str, list[str]] | None = None,
    filter_capabilities: dict[str, list[str]] | None = None,
    record_index_filter_expr: ds.Expression | None = None,
    wide_filter_expr: ds.Expression | None = None,
    name: str = "",
    description: str = "",
    data_source: str = "",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    """Initialize a Collection.

    Parameters
    ----------
    dataset : pyarrow.dataset.Dataset, optional
        Backing Arrow dataset. ``None`` creates an empty or non-Dataset-backed
        collection.
    hf_streaming : HFStreamingSource, optional
        Hugging Face streaming-backed metadata source.
    name : str
        Human-readable collection name.
    description : str
        Free-text description.
    data_source : str
        Data source identifier (e.g. ``"sentinel-2-l2a"``).
    start_date : datetime, optional
        Collection temporal start.
    end_date : datetime, optional
        Collection temporal end.
    """
    self.dataset = dataset
    self._hf_streaming = hf_streaming
    self.name = name
    self.description = description
    self.data_source = data_source
    self.start_date = start_date
    self.end_date = end_date
    self._planner = ParquetReadPlanner(
        collection_path=collection_path,
        record_index_path=record_index_path,
        record_index_field_roles=record_index_field_roles or {},
        record_index_column_map=record_index_column_map or {},
        record_index_href_column=record_index_href_column,
        record_index_band_index_map=record_index_band_index_map,
        record_index_url_rewrite_patterns=record_index_url_rewrite_patterns or {},
        record_index_filesystem=record_index_filesystem,
        surface_fields=(
            {
                surface: tuple(fields)
                for surface, fields in (surface_fields or {}).items()
            }
            or None
        ),
        filter_capabilities=(
            {
                surface: tuple(fields)
                for surface, fields in (filter_capabilities or {}).items()
            }
            or None
        ),
        record_index_filter_expr=record_index_filter_expr,
        wide_filter_expr=wide_filter_expr,
    )
    self._record_index_dataset: ds.Dataset | None = None
    if self.dataset is not None and self._hf_streaming is not None:
        raise ValueError(
            "Collection cannot use both Dataset and HF streaming backends"
        )
    if self.dataset is not None:
        self._validate_parquet_dataset()
```

##### Attributes

###### bands

```python
bands: list[str]
```

Available band codes in this collection.

###### bounds

```python
bounds: tuple[float, float, float, float] | None
```

Spatial extent as `(minx, miny, maxx, maxy)` or `None`.

###### epsg

```python
epsg: list[int]
```

Unique EPSG codes in this collection.

##### Functions

###### from_parquet

```python
from_parquet(
    path: str | Path,
    name: str = "",
    *,
    data_source: str = "",
    defer_dataset_open: bool = False,
    record_index_path: str | None = None,
    record_index_field_roles: dict[str, str] | None = None,
    record_index_column_map: dict[str, str] | None = None,
    record_index_href_column: str | None = None,
    record_index_band_index_map: dict[str, int]
    | None = None,
    record_index_url_rewrite_patterns: dict[str, str]
    | None = None,
    record_index_filesystem: Any | None = None,
    surface_fields: dict[str, list[str]] | None = None,
    filter_capabilities: dict[str, list[str]] | None = None,
) -> Collection
```

Load a Collection from any Parquet file or directory.

Accepts local paths **and** cloud URIs (`s3://`, `gs://`). Tries Hive-style partitioning first (year/month), falls back to plain Parquet. Validates that the core contract columns are present.

See the `Schema Contract <../explanation/schema-contract/>`\_ docs page.

Source code in `src/rasteret/core/collection.py`

```python
@classmethod
def from_parquet(
    cls,
    path: str | Path,
    name: str = "",
    *,
    data_source: str = "",
    defer_dataset_open: bool = False,
    record_index_path: str | None = None,
    record_index_field_roles: dict[str, str] | None = None,
    record_index_column_map: dict[str, str] | None = None,
    record_index_href_column: str | None = None,
    record_index_band_index_map: dict[str, int] | None = None,
    record_index_url_rewrite_patterns: dict[str, str] | None = None,
    record_index_filesystem: Any | None = None,
    surface_fields: dict[str, list[str]] | None = None,
    filter_capabilities: dict[str, list[str]] | None = None,
) -> Collection:
    """Load a Collection from any Parquet file or directory.

    Accepts local paths **and** cloud URIs (``s3://``, ``gs://``).
    Tries Hive-style partitioning first (year/month), falls back to
    plain Parquet.  Validates that the core contract columns are present.

    See the `Schema Contract <../explanation/schema-contract/>`_ docs page.
    """
    path_str = str(path)
    if not _is_cloud_uri(path_str):
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found at {path_str}")

    if is_hf_dataset_uri(path_str):
        try:
            hf_streaming = open_hf_streaming_source(path_str)
        except Exception as exc:
            raise FileNotFoundError(f"Cannot open Parquet at {path_str}") from exc

        required = {"id", "datetime", "geometry", "assets"}
        missing = required - set(hf_streaming.schema.names)
        if missing or _bbox_struct_field(hf_streaming.schema) is None:
            raise ValueError(
                f"Parquet is missing required columns: {missing or {'bbox'}}. "
                "See the Schema Contract page in docs for the expected schema."
            )

        return cls(
            hf_streaming=hf_streaming,
            name=name or _stem_from_path(path_str),
            data_source=data_source,
            record_index_path=record_index_path,
            record_index_field_roles=record_index_field_roles,
            record_index_column_map=record_index_column_map,
            record_index_href_column=record_index_href_column,
            record_index_band_index_map=record_index_band_index_map,
            record_index_url_rewrite_patterns=record_index_url_rewrite_patterns,
            record_index_filesystem=record_index_filesystem,
            surface_fields=surface_fields,
            filter_capabilities=filter_capabilities,
        )

    dataset = None
    meta: dict[str, str] = {}
    if not defer_dataset_open:
        try:
            dataset = _open_parquet_dataset(path_str)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise FileNotFoundError(f"Cannot open Parquet at {path_str}") from exc

        required = {"id", "datetime", "geometry", "assets"}
        missing = required - set(dataset.schema.names)
        if missing or _bbox_struct_field(dataset.schema) is None:
            raise ValueError(
                f"Parquet is missing required columns: {missing or {'bbox'}}. "
                "See the Schema Contract page in docs for the expected schema."
            )

        meta = cls._metadata_from_schema(dataset)
    resolved_name = name or meta.get("name") or _stem_from_path(path_str)

    start_date = None
    end_date = None
    dr = meta.get("date_range", "")
    if "," in dr:
        s, e = dr.split(",", 1)
        start_date = datetime.fromisoformat(s)
        end_date = datetime.fromisoformat(e)

    return cls(
        dataset=dataset,
        collection_path=path_str if defer_dataset_open else None,
        record_index_path=record_index_path,
        record_index_field_roles=record_index_field_roles,
        record_index_column_map=record_index_column_map,
        record_index_href_column=record_index_href_column,
        record_index_band_index_map=record_index_band_index_map,
        record_index_url_rewrite_patterns=record_index_url_rewrite_patterns,
        record_index_filesystem=record_index_filesystem,
        surface_fields=surface_fields,
        filter_capabilities=filter_capabilities,
        name=resolved_name,
        data_source=data_source or meta.get("data_source", ""),
        description=meta.get("description", ""),
        start_date=start_date,
        end_date=end_date,
    )
```

###### subset

```python
subset(
    *,
    cloud_cover_lt: float | None = None,
    date_range: tuple[str, str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geometries: Any = None,
    split: str | Sequence[str] | None = None,
    split_column: str = "split",
) -> Collection
```

Return a filtered view of this Collection.

All provided criteria are combined with AND.

Parameters:

| Name             | Type                                                        | Description                                                                                                                                                                                                                    | Default   |
| ---------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| `cloud_cover_lt` | `float`                                                     | Keep records with eo:cloud_cover below this value (0--100).                                                                                                                                                                    | `None`    |
| `date_range`     | `tuple of str`                                              | (start, end) ISO date strings for temporal filtering.                                                                                                                                                                          | `None`    |
| `bbox`           | `tuple of float`                                            | (minx, miny, maxx, maxy) bounding box filter.                                                                                                                                                                                  | `None`    |
| `geometries`     | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Spatial filter; records whose bbox overlaps any geometry are kept. Accepts (minx, miny, maxx, maxy) bbox tuples, Arrow arrays (e.g. a geometry column read from GeoParquet), Shapely objects, raw WKB bytes, or GeoJSON dicts. | `None`    |
| `split`          | `str or sequence of str`                                    | Keep only rows matching the given split value(s).                                                                                                                                                                              | `None`    |
| `split_column`   | `str`                                                       | Column name holding split labels. Defaults to "split".                                                                                                                                                                         | `'split'` |

Returns:

| Type         | Description                                      |
| ------------ | ------------------------------------------------ |
| `Collection` | A new Collection with the filtered dataset view. |

Source code in `src/rasteret/core/collection.py`

```python
def subset(
    self,
    *,
    cloud_cover_lt: float | None = None,
    date_range: tuple[str, str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geometries: Any = None,
    split: str | Sequence[str] | None = None,
    split_column: str = "split",
) -> Collection:
    """Return a filtered view of this Collection.

    All provided criteria are combined with AND.

    Parameters
    ----------
    cloud_cover_lt : float, optional
        Keep records with ``eo:cloud_cover`` below this value (0--100).
    date_range : tuple of str, optional
        ``(start, end)`` ISO date strings for temporal filtering.
    bbox : tuple of float, optional
        ``(minx, miny, maxx, maxy)`` bounding box filter.
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict, optional
        Spatial filter; records whose bbox overlaps any geometry are kept.
        Accepts ``(minx, miny, maxx, maxy)`` bbox tuples, Arrow arrays
        (e.g. a geometry column read from GeoParquet), Shapely objects,
        raw WKB bytes, or GeoJSON dicts.
    split : str or sequence of str, optional
        Keep only rows matching the given split value(s).
    split_column : str
        Column name holding split labels. Defaults to ``"split"``.

    Returns
    -------
    Collection
        A new Collection with the filtered dataset view.
    """
    if self._hf_streaming is not None:
        if all(
            value is None
            for value in (
                cloud_cover_lt,
                date_range,
                bbox,
                geometries,
                split,
            )
        ):
            raise ValueError("No filters provided")
        return self._view(
            hf_streaming=subset_hf_streaming_source(
                self._hf_streaming,
                cloud_cover_lt=cloud_cover_lt,
                date_range=date_range,
                bbox=bbox,
                geometries=geometries,
                split=split,
                split_column=split_column,
            )
        )

    if self._has_record_index():
        filter_expr = self._record_index_filter_expr
        wide_filter_expr = self._wide_filter_expr
        index_dataset = self._open_record_index_dataset()
        wide_dataset = self.dataset
        index_schema = index_dataset.schema
        wide_schema = wide_dataset.schema if wide_dataset is not None else None

        if all(
            value is None
            for value in (
                cloud_cover_lt,
                date_range,
                bbox,
                geometries,
                split,
            )
        ):
            raise ValueError("No filters provided")

        if cloud_cover_lt is not None:
            if not self._surface_supports_filter(
                "index",
                "eo:cloud_cover",
                schema=index_schema,
            ):
                filtered_dataset = self._filtered_data_dataset()
                return self._view(
                    filtered_dataset.filter(
                        ds.field("eo:cloud_cover") < float(cloud_cover_lt)
                    )
                    if filtered_dataset is not None
                    else None,
                    record_index_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                    wide_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                    drop_record_index=True,
                )
            if not isinstance(cloud_cover_lt, (int, float)) or not (
                0 <= cloud_cover_lt <= 100
            ):
                raise ValueError(
                    f"Invalid cloud_cover_lt={cloud_cover_lt!r}: must be between 0 and 100."
                )
            filter_expr = _and_filters(
                filter_expr, ds.field("eo:cloud_cover") < float(cloud_cover_lt)
            )
            if self._surface_supports_filter(
                "collection",
                "eo:cloud_cover",
                schema=wide_schema,
            ):
                wide_filter_expr = _and_filters(
                    wide_filter_expr,
                    ds.field("eo:cloud_cover") < float(cloud_cover_lt),
                )

        if date_range is not None:
            start_raw, end_raw = date_range
            if not start_raw or not end_raw:
                raise ValueError("Invalid date range")
            start = pd.Timestamp(start_raw)
            end = pd.Timestamp(end_raw)
            if start > end:
                raise ValueError("Invalid date range")
            datetime_source = self._record_index_source_column("datetime")
            if datetime_source not in index_schema.names:
                raise ValueError("Collection has no datetime data")
            dt_type = index_schema.field(datetime_source).type
            if pa.types.is_integer(dt_type):
                filter_expr = _and_filters(
                    filter_expr, ds.field(datetime_source) >= int(start.year)
                )
                filter_expr = _and_filters(
                    filter_expr, ds.field(datetime_source) <= int(end.year)
                )
            else:
                start_scalar = pa.scalar(start.to_pydatetime(), type=dt_type)
                end_scalar = pa.scalar(end.to_pydatetime(), type=dt_type)
                filter_expr = _and_filters(
                    filter_expr,
                    (ds.field(datetime_source) >= start_scalar)
                    & (ds.field(datetime_source) <= end_scalar),
                )
            if (
                self._surface_supports_filter(
                    "collection",
                    "datetime",
                    schema=wide_schema,
                )
                and wide_schema is not None
                and "datetime" in wide_schema.names
            ):
                wide_ts_type = wide_schema.field("datetime").type
                start_scalar = pa.scalar(start.to_pydatetime(), type=wide_ts_type)
                end_scalar = pa.scalar(end.to_pydatetime(), type=wide_ts_type)
                wide_filter_expr = _and_filters(
                    wide_filter_expr,
                    (ds.field("datetime") >= start_scalar)
                    & (ds.field("datetime") <= end_scalar),
                )
            if self._surface_has_field("collection", "year", schema=wide_schema):
                wide_filter_expr = _and_filters(
                    wide_filter_expr, ds.field("year") >= int(start.year)
                )
                wide_filter_expr = _and_filters(
                    wide_filter_expr, ds.field("year") <= int(end.year)
                )

        if bbox is not None:
            if not self._surface_supports_filter(
                "index", "bbox", schema=index_schema
            ):
                raise ValueError(
                    "bbox filtering requires a root-level 'bbox' struct with "
                    "xmin/ymin/xmax/ymax children."
                )
            if len(bbox) != 4:
                raise ValueError("Invalid bbox format")
            minx, miny, maxx, maxy = bbox
            if minx > maxx or miny > maxy:
                raise ValueError("Invalid bbox coordinates")
            filter_expr = _and_filters(
                filter_expr,
                _bbox_overlap_expr(
                    minx,
                    miny,
                    maxx,
                    maxy,
                    field_name=self._record_index_source_column("bbox"),
                ),
            )
            if self._surface_supports_filter(
                "collection",
                "bbox",
                schema=wide_schema,
            ):
                wide_filter_expr = _and_filters(
                    wide_filter_expr, _bbox_overlap_expr(minx, miny, maxx, maxy)
                )

        if geometries is not None:
            if not self._surface_supports_filter(
                "index", "bbox", schema=index_schema
            ):
                raise ValueError(
                    "geometry filtering requires a root-level 'bbox' struct with "
                    "xmin/ymin/xmax/ymax children."
                )
            from rasteret.core.geometry import bbox_array, coerce_to_geoarrow

            geo_arr = coerce_to_geoarrow(geometries)
            xmin, ymin, xmax, ymax = bbox_array(geo_arr)
            geometry_filter: ds.Expression | None = None
            for i in range(len(xmin)):
                geom_expr = _bbox_overlap_expr(
                    xmin[i].as_py(),
                    ymin[i].as_py(),
                    xmax[i].as_py(),
                    ymax[i].as_py(),
                    field_name=self._record_index_source_column("bbox"),
                )
                geometry_filter = (
                    geom_expr
                    if geometry_filter is None
                    else (geometry_filter | geom_expr)
                )
            filter_expr = _and_filters(filter_expr, geometry_filter)
            if geometry_filter is not None and self._surface_supports_filter(
                "collection",
                "bbox",
                schema=wide_schema,
            ):
                wide_filter_expr = _and_filters(wide_filter_expr, geometry_filter)

        if split is not None:
            if split_column not in index_schema.names:
                filtered_dataset = self._filtered_data_dataset()
                return self._view(
                    Collection(dataset=filtered_dataset)
                    .subset(split=split, split_column=split_column)
                    .dataset
                    if filtered_dataset is not None
                    else None,
                    record_index_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                    wide_filter_expr=_UNSET_RECORD_INDEX_FILTER,
                    drop_record_index=True,
                )
            if isinstance(split, str):
                split_expr = ds.field(split_column) == split
            elif (
                isinstance(split, Sequence)
                and not isinstance(split, (str, bytes))
                and split
                and all(isinstance(value, str) for value in split)
            ):
                split_expr = ds.field(split_column).isin(list(split))
            else:
                raise ValueError(
                    "Invalid split filter. Use a split name or sequence of split names."
                )
            filter_expr = _and_filters(filter_expr, split_expr)
            if self._surface_supports_filter(
                "collection",
                split_column,
                schema=wide_schema,
            ):
                wide_filter_expr = _and_filters(wide_filter_expr, split_expr)

        return self._view(
            self.dataset,
            record_index_filter_expr=filter_expr,
            wide_filter_expr=wide_filter_expr,
        )

    if self.dataset is None:
        return self

    filter_expr: ds.Expression | None = None

    def _and(current: ds.Expression | None, new: ds.Expression) -> ds.Expression:
        return new if current is None else current & new

    if cloud_cover_lt is not None:
        if "eo:cloud_cover" not in self.dataset.schema.names:
            raise ValueError("Collection has no cloud cover data")
        if not isinstance(cloud_cover_lt, (int, float)) or not (
            0 <= cloud_cover_lt <= 100
        ):
            raise ValueError(
                f"Invalid cloud_cover_lt={cloud_cover_lt!r}: must be between 0 and 100."
            )
        filter_expr = _and(
            filter_expr, ds.field("eo:cloud_cover") < float(cloud_cover_lt)
        )

    if date_range is not None:
        if "datetime" not in self.dataset.schema.names:
            raise ValueError("Collection has no datetime data")
        start_raw, end_raw = date_range
        if not start_raw or not end_raw:
            raise ValueError("Invalid date range")
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        if start > end:
            raise ValueError("Invalid date range")

        ts_type = self.dataset.schema.field("datetime").type
        if not pa.types.is_timestamp(ts_type):
            raise ValueError("Collection datetime column is not a timestamp")
        start_scalar = pa.scalar(start.to_pydatetime(), type=ts_type)
        end_scalar = pa.scalar(end.to_pydatetime(), type=ts_type)
        date_filter = (ds.field("datetime") >= start_scalar) & (
            ds.field("datetime") <= end_scalar
        )
        filter_expr = _and(filter_expr, date_filter)

    if bbox is not None:
        if _bbox_struct_field(self.dataset.schema) is None:
            raise ValueError(
                "bbox filtering requires a root-level 'bbox' struct with "
                "xmin/ymin/xmax/ymax children. "
                "Rebuild or re-normalize the collection with rasteret>=1.0.0."
            )
        if len(bbox) != 4:
            raise ValueError("Invalid bbox format")
        minx, miny, maxx, maxy = bbox
        if minx > maxx or miny > maxy:
            raise ValueError("Invalid bbox coordinates")
        filter_expr = _and(filter_expr, _bbox_overlap_expr(minx, miny, maxx, maxy))

    if geometries is not None:
        if _bbox_struct_field(self.dataset.schema) is None:
            raise ValueError(
                "geometry filtering requires a root-level 'bbox' struct with "
                "xmin/ymin/xmax/ymax children. "
                "Rebuild or re-normalize the collection with rasteret>=1.0.0."
            )
        from rasteret.core.geometry import bbox_array, coerce_to_geoarrow

        geo_arr = coerce_to_geoarrow(geometries)
        xmin, ymin, xmax, ymax = bbox_array(geo_arr)

        geometry_filter: ds.Expression | None = None
        for i in range(len(xmin)):
            geom_expr = _bbox_overlap_expr(
                xmin[i].as_py(),
                ymin[i].as_py(),
                xmax[i].as_py(),
                ymax[i].as_py(),
            )
            geometry_filter = (
                geom_expr
                if geometry_filter is None
                else (geometry_filter | geom_expr)
            )
        if geometry_filter is not None:
            filter_expr = _and(filter_expr, geometry_filter)

    if split is not None:
        if split_column not in self.dataset.schema.names:
            raise ValueError(f"Collection has no split column: '{split_column}'")
        if isinstance(split, str):
            split_expr = ds.field(split_column) == split
        elif (
            isinstance(split, Sequence)
            and not isinstance(split, (str, bytes))
            and split
            and all(isinstance(value, str) for value in split)
        ):
            split_expr = ds.field(split_column).isin(list(split))
        else:
            raise ValueError(
                "Invalid split filter. Use a split name or sequence of split names."
            )
        filter_expr = _and(filter_expr, split_expr)

    if filter_expr is None:
        raise ValueError("No filters provided")

    return self._view(self.dataset.filter(filter_expr))
```

###### select_split

```python
select_split(
    split: str | Sequence[str],
    *,
    split_column: str = "split",
) -> Collection
```

Return a split-filtered view of this Collection.

This is a convenience wrapper around `subset(split=...)` to keep the intent obvious in training code.

Source code in `src/rasteret/core/collection.py`

```python
def select_split(
    self,
    split: str | Sequence[str],
    *,
    split_column: str = "split",
) -> Collection:
    """Return a split-filtered view of this Collection.

    This is a convenience wrapper around ``subset(split=...)`` to keep the
    intent obvious in training code.
    """
    return self.subset(split=split, split_column=split_column)
```

###### where

```python
where(expr: Expression) -> Collection
```

Return a filtered view using a raw Arrow dataset expression.

Source code in `src/rasteret/core/collection.py`

```python
def where(self, expr: ds.Expression) -> Collection:
    """Return a filtered view using a raw Arrow dataset expression."""
    if self._hf_streaming is not None:
        raise NotImplementedError(
            "where(expr) is not supported for HF streaming collections. "
            "Use subset(...) with managed filters instead."
        )
    if self._has_record_index():
        index_expr = expr if self._record_index_supports_expr(expr) else None
        wide_expr = (
            expr if self._dataset_supports_expr(self.dataset, expr) else None
        )
        if index_expr is None and wide_expr is None:
            raise ValueError("where(expr) could not be applied to the collection")
        if index_expr is not None:
            return self._view(
                self.dataset,
                record_index_filter_expr=_and_filters(
                    self._record_index_filter_expr, index_expr
                ),
                wide_filter_expr=_and_filters(self._wide_filter_expr, wide_expr),
            )
        filtered_dataset = self._filtered_data_dataset()
        if filtered_dataset is None:
            return self
        return self._view(
            filtered_dataset.filter(expr),
            record_index_filter_expr=_UNSET_RECORD_INDEX_FILTER,
            wide_filter_expr=_UNSET_RECORD_INDEX_FILTER,
            drop_record_index=True,
        )
    if self.dataset is None:
        return self
    return self._view(self.dataset.filter(expr))
```

###### head

```python
head(n: int = 5, columns: list[str] | None = None) -> Table
```

Return the first *n* metadata rows as a PyArrow table.

Source code in `src/rasteret/core/collection.py`

```python
def head(self, n: int = 5, columns: list[str] | None = None) -> pa.Table:
    """Return the first *n* metadata rows as a PyArrow table."""
    if n < 0:
        raise ValueError("head() requires n >= 0")
    if self._has_record_index():
        return self._prepare_record_index_table(columns=columns, limit=n)
    if self.dataset is not None:
        return self.dataset.head(n, columns=columns)
    if self._hf_streaming is not None:
        return head_hf_streaming_source(self._hf_streaming, n=n, columns=columns)
    schema = (
        pa.schema([])
        if columns is None
        else pa.schema([pa.field(name, pa.null()) for name in columns])
    )
    return schema.empty_table()
```

###### list_collections

```python
list_collections(
    workspace_dir: Path | None = None,
) -> list[dict[str, Any]]
```

List cached collections with summary metadata.

Parameters:

| Name            | Type   | Description                                                                 | Default |
| --------------- | ------ | --------------------------------------------------------------------------- | ------- |
| `workspace_dir` | `Path` | Directory to scan for cached collections. Defaults to ~/rasteret_workspace. | `None`  |

Returns:

| Type           | Description                                                                |
| -------------- | -------------------------------------------------------------------------- |
| `list of dict` | Each dict contains name, kind, data_source, date_range, size, and created. |

Source code in `src/rasteret/core/collection.py`

```python
@classmethod
def list_collections(
    cls, workspace_dir: Path | None = None
) -> list[dict[str, Any]]:
    """List cached collections with summary metadata.

    Parameters
    ----------
    workspace_dir : Path, optional
        Directory to scan for cached collections. Defaults to
        ``~/rasteret_workspace``.

    Returns
    -------
    list of dict
        Each dict contains ``name``, ``kind``, ``data_source``,
        ``date_range``, ``size``, and ``created``.
    """
    if workspace_dir is None:
        workspace_dir = Path.home() / "rasteret_workspace"

    def _date_range(dataset: ds.Dataset) -> tuple[str, str] | None:
        if "datetime" not in dataset.schema.names:
            return None
        scanner = dataset.scanner(columns=["datetime"])
        min_value = None
        max_value = None
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            column = batch.column(0)
            batch_min = pc.min(column).as_py()
            batch_max = pc.max(column).as_py()
            if batch_min is not None:
                min_value = (
                    batch_min if min_value is None else min(min_value, batch_min)
                )
            if batch_max is not None:
                max_value = (
                    batch_max if max_value is None else max(max_value, batch_max)
                )
        if min_value is None or max_value is None:
            return None
        return (min_value.date().isoformat(), max_value.date().isoformat())

    collections: list[dict[str, Any]] = []

    def _data_source_from_metadata(dataset: ds.Dataset) -> str | None:
        metadata = dataset.schema.metadata or {}
        value = metadata.get(b"data_source")
        if not value:
            return None
        try:
            decoded = value.decode("utf-8").strip()
        except (UnicodeDecodeError, AttributeError):
            return None
        return decoded or None

    # Look for cached directories
    for suffix in ("_stac", "_records"):
        dirs = workspace_dir.glob(f"*{suffix}")
        for cache_dir in dirs:
            try:
                try:
                    dataset = ds.dataset(
                        str(cache_dir), format="parquet", partitioning="hive"
                    )
                except pa.ArrowInvalid:
                    dataset = ds.dataset(str(cache_dir), format="parquet")
                name = cache_dir.name.removesuffix(suffix)
                date_range = _date_range(dataset)
                data_source = _data_source_from_metadata(dataset) or (
                    name.split("_")[-1] if "_" in name else "unknown"
                )

                collections.append(
                    {
                        "name": name,
                        "kind": suffix.removeprefix("_"),
                        "data_source": data_source,
                        "date_range": date_range,
                        "size": dataset.count_rows(),
                        "created": cache_dir.stat().st_ctime,
                    }
                )

            except (pa.ArrowInvalid, OSError) as exc:
                logger.debug("Failed to read collection %s: %s", cache_dir, exc)
                continue

    return collections
```

###### export

```python
export(
    path: str | Path,
    partition_by: Sequence[str] = ("year", "month"),
) -> None
```

Export the collection as a partitioned Parquet dataset.

Use this to produce a portable copy of the collection that can be shared with teammates via :func:`rasteret.load`.

Parameters:

| Name           | Type              | Description                                                          | Default             |
| -------------- | ----------------- | -------------------------------------------------------------------- | ------------------- |
| `path`         | `str or Path`     | Output directory. Accepts local paths and cloud URIs (s3://, gs://). | *required*          |
| `partition_by` | `sequence of str` | Columns to partition by. Defaults to ("year", "month").              | `('year', 'month')` |

Source code in `src/rasteret/core/collection.py`

```python
def export(
    self,
    path: str | Path,
    partition_by: Sequence[str] = ("year", "month"),
) -> None:
    """Export the collection as a partitioned Parquet dataset.

    Use this to produce a portable copy of the collection that can
    be shared with teammates via :func:`rasteret.load`.

    Parameters
    ----------
    path : str or Path
        Output directory.  Accepts local paths and cloud URIs
        (``s3://``, ``gs://``).
    partition_by : sequence of str
        Columns to partition by. Defaults to ``("year", "month")``.
    """
    path_str = str(path)
    if not _is_cloud_uri(path_str):
        Path(path_str).mkdir(parents=True, exist_ok=True)

    if self.dataset is None:
        raise ValueError("No Pyarrow dataset provided")

    table = self.dataset.to_table()

    # Enhanced metadata with fallbacks
    custom_metadata = {
        b"description": (
            self.description.encode("utf-8") if self.description else b""
        ),
        b"created": datetime.now().isoformat().encode("utf-8"),
        b"name": self.name.encode("utf-8") if self.name else b"",
        b"data_source": (
            self.data_source.encode("utf-8") if self.data_source else b""
        ),
        b"date_range": (
            f"{self.start_date.isoformat()},{self.end_date.isoformat()}".encode(
                "utf-8"
            )
            if self.start_date and self.end_date
            else b""
        ),
        b"rasteret_collection_version": b"1",
    }

    # Merge with existing metadata
    merged_metadata = {**custom_metadata, **(table.schema.metadata or {})}

    # GeoParquet metadata: declare the geometry column as WKB.
    #
    # Rasteret stores footprint geometries in CRS84 (lon/lat) for portability.
    # GeoParquet 1.1 treats missing `crs` as CRS84 by default.
    if "geometry" in table.schema.names and b"geo" not in merged_metadata:
        geom_types = _geometry_types_from_wkb(table.column("geometry"))
        geo = {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": geom_types,
                }
            },
        }
        if _bbox_struct_field(table.schema) is not None:
            geo["columns"]["geometry"]["covering"] = {
                "bbox": {
                    "xmin": ["bbox", "xmin"],
                    "ymin": ["bbox", "ymin"],
                    "xmax": ["bbox", "xmax"],
                    "ymax": ["bbox", "ymax"],
                }
            }
        merged_metadata[b"geo"] = json.dumps(
            geo, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    table_with_metadata = table.replace_schema_metadata(merged_metadata)

    # Write dataset
    pq.write_to_dataset(
        table_with_metadata,
        root_path=path_str,
        partition_cols=partition_by,
        compression="zstd",
        compression_level=3,
        row_group_size=50_000,
        write_statistics=True,
        use_dictionary=True,
        write_batch_size=10000,
        basename_template="part-{i}.parquet",
    )
```

###### iterate_rasters

```python
iterate_rasters(
    data_source: str | None = None,
    bands: list[str] | None = None,
) -> AsyncIterator[RasterAccessor]
```

Iterate through raster records in this Collection.

Each Parquet row becomes a :class:`RasterAccessor` that provides async band-loading methods.

Parameters:

| Name          | Type  | Description                                                                                         | Default |
| ------------- | ----- | --------------------------------------------------------------------------------------------------- | ------- |
| `data_source` | `str` | Data source identifier for band mapping. Defaults to self.data_source or inferred from the dataset. | `None`  |

Yields:

| Type             | Description |
| ---------------- | ----------- |
| `RasterAccessor` |             |

Source code in `src/rasteret/core/collection.py`

```python
async def iterate_rasters(
    self,
    data_source: str | None = None,
    bands: list[str] | None = None,
) -> AsyncIterator[RasterAccessor]:
    """Iterate through raster records in this Collection.

    Each Parquet row becomes a :class:`RasterAccessor` that provides
    async band-loading methods.

    Parameters
    ----------
    data_source : str, optional
        Data source identifier for band mapping. Defaults to
        ``self.data_source`` or inferred from the dataset.

    Yields
    ------
    RasterAccessor
    """
    required_fields = {"id", "datetime", "geometry", "assets", "bbox"}

    schema = self._schema
    if schema is None:
        return

    # Check required fields
    missing = required_fields - set(schema.names)
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    resolved_source = data_source or self.data_source or ""
    schema_names = set(schema.names)
    band_metadata_cols = [
        name for name in schema.names if name.endswith("_metadata")
    ]
    optional_cols = [
        name
        for name in ("proj:epsg", "eo:cloud_cover", "collection")
        if name in schema_names
    ]
    requested_band_metadata_cols: list[str] | None = None
    if bands:
        requested_band_metadata_cols = [
            f"{band}_metadata"
            for band in bands
            if f"{band}_metadata" in schema_names
        ]
    scan_cols = [
        "id",
        "datetime",
        "geometry",
        "assets",
        "bbox",
        *optional_cols,
        *(
            requested_band_metadata_cols
            if requested_band_metadata_cols is not None
            else band_metadata_cols
        ),
    ]

    batch_source: Collection = self
    if self.dataset is not None:
        scan_dataset = self._filtered_data_dataset()
        if scan_dataset is None:
            return
        batch_source = self._view(scan_dataset)

    for batch in batch_source._iter_record_batches(columns=scan_cols):
        ids = batch.column(batch.schema.get_field_index("id"))
        datetimes = batch.column(batch.schema.get_field_index("datetime"))
        geometries = batch.column(batch.schema.get_field_index("geometry"))
        assets = batch.column(batch.schema.get_field_index("assets"))
        bbox_col = batch.column(batch.schema.get_field_index("bbox"))

        crs_col = (
            batch.column(batch.schema.get_field_index("proj:epsg"))
            if "proj:epsg" in batch.schema.names
            else None
        )
        cloud_col = (
            batch.column(batch.schema.get_field_index("eo:cloud_cover"))
            if "eo:cloud_cover" in batch.schema.names
            else None
        )
        collection_col = (
            batch.column(batch.schema.get_field_index("collection"))
            if "collection" in batch.schema.names
            else None
        )
        band_cols = {
            name: batch.column(batch.schema.get_field_index(name))
            for name in (
                requested_band_metadata_cols
                if requested_band_metadata_cols is not None
                else band_metadata_cols
            )
            if name in batch.schema.names
        }

        for idx in range(batch.num_rows):
            try:
                band_metadata: dict[str, Any] = {}
                for key, col in band_cols.items():
                    val = col[idx]
                    if val.is_valid:
                        py_val = val.as_py()
                        if py_val is not None:
                            band_metadata[key] = py_val

                info = RasterInfo(
                    id=ids[idx].as_py(),
                    datetime=datetimes[idx].as_py(),
                    footprint=geometries[idx].as_py(),
                    bbox=_bbox_value_to_list(bbox_col[idx].as_py()) or [],
                    crs=crs_col[idx].as_py() if crs_col is not None else None,
                    cloud_cover=(
                        cloud_col[idx].as_py()
                        if cloud_col is not None and cloud_col[idx].is_valid
                        else 0
                    ),
                    assets=assets[idx].as_py(),
                    band_metadata=band_metadata,
                    collection=(
                        collection_col[idx].as_py()
                        if collection_col is not None
                        and collection_col[idx].is_valid
                        else resolved_source
                    ),
                )
                yield RasterAccessor(info, resolved_source)
            except (KeyError, TypeError, ValueError):
                logger.exception(
                    "Failed to create RasterAccessor from collection row"
                )
                continue
```

###### get_first_raster

```python
get_first_raster() -> RasterAccessor
```

Return the first raster record in the collection.

Returns:

| Type             | Description |
| ---------------- | ----------- |
| `RasterAccessor` |             |

Source code in `src/rasteret/core/collection.py`

```python
async def get_first_raster(self) -> RasterAccessor:
    """Return the first raster record in the collection.

    Returns
    -------
    RasterAccessor
    """
    async for raster in self.iterate_rasters():
        return raster
    raise ValueError("No raster records found in collection")
```

###### describe

```python
describe() -> DescribeResult
```

Summary of this collection.

Returns a :class:`~rasteret.core.display.DescribeResult` that renders as a clean table in terminals and as styled HTML in notebooks (Jupyter, marimo, Colab).

The underlying data is accessible via `.data` or `["key"]`.

Examples:

```pycon
>>> collection.describe()           # pretty table in REPL
>>> collection.describe()["bands"]  # programmatic access
>>> collection.describe().data      # full dict
```

Source code in `src/rasteret/core/collection.py`

```python
def describe(self) -> DescribeResult:
    """Summary of this collection.

    Returns a :class:`~rasteret.core.display.DescribeResult` that renders
    as a clean table in terminals and as styled HTML in notebooks
    (Jupyter, marimo, Colab).

    The underlying data is accessible via ``.data`` or ``["key"]``.

    Examples
    --------
    >>> collection.describe()           # pretty table in REPL
    >>> collection.describe()["bands"]  # programmatic access
    >>> collection.describe().data      # full dict
    """
    from rasteret.core.display import build_describe_result

    dates = None
    if self.start_date and self.end_date:
        dates = (str(self.start_date)[:10], str(self.end_date)[:10])
    try:
        records = len(self)
    except Exception:
        records = "?"
    return build_describe_result(
        name=self.name,
        records=records,
        bands=self.bands,
        bounds=self.bounds,
        crs=self.epsg,
        dates=dates,
        source=self.data_source,
    )
```

###### compare_to_catalog

```python
compare_to_catalog() -> DescribeResult
```

Compare this collection against its catalog source.

Shows collection properties side-by-side with the catalog entry (bands coverage, date range vs source range, spatial coverage, auth requirements).

Raises :class:`ValueError` if the collection has no catalog match.

Returns a :class:`~rasteret.core.display.DescribeResult` that renders as a table in terminals and styled HTML in notebooks.

Examples:

```pycon
>>> collection.compare_to_catalog()        # pretty comparison table
>>> collection.compare_to_catalog().data    # full dict with catalog info
```

Source code in `src/rasteret/core/collection.py`

```python
def compare_to_catalog(self) -> DescribeResult:
    """Compare this collection against its catalog source.

    Shows collection properties side-by-side with the catalog entry
    (bands coverage, date range vs source range, spatial coverage,
    auth requirements).

    Raises :class:`ValueError` if the collection has no catalog match.

    Returns a :class:`~rasteret.core.display.DescribeResult` that renders
    as a table in terminals and styled HTML in notebooks.

    Examples
    --------
    >>> collection.compare_to_catalog()        # pretty comparison table
    >>> collection.compare_to_catalog().data    # full dict with catalog info
    """
    from rasteret.core.display import build_catalog_comparison

    desc = self._resolve_catalog_descriptor()
    if desc is None:
        raise ValueError(
            f"No catalog entry found for data_source={self.data_source!r}. "
            "Use describe() for collection-only summary."
        )

    dates = None
    if self.start_date and self.end_date:
        dates = (str(self.start_date)[:10], str(self.end_date)[:10])

    return build_catalog_comparison(
        name=self.name,
        records=self.describe()["records"],
        bands=self.bands,
        bounds=self.bounds,
        crs=self.epsg,
        dates=dates,
        source=self.data_source,
        catalog_name=desc.name,
        catalog_bands=list(desc.band_map) if desc.band_map else [],
        catalog_temporal=desc.temporal_range,
        catalog_coverage=desc.spatial_coverage,
        catalog_auth=desc.requires_auth,
        catalog_license=desc.license,
    )
```

###### create_name

```python
create_name(
    custom_name: str,
    date_range: tuple[str, str],
    data_source: str,
) -> str
```

Create a standardized collection name.

Parameters:

| Name          | Type           | Description                                                       | Default    |
| ------------- | -------------- | ----------------------------------------------------------------- | ---------- |
| `custom_name` | `str`          | User-chosen name component. Underscores are normalised to dashes. | *required* |
| `date_range`  | `tuple of str` | (start, end) ISO date strings.                                    | *required* |
| `data_source` | `str`          | Data source identifier (e.g. "sentinel-2-l2a").                   | *required* |

Returns:

| Type  | Description                                       |
| ----- | ------------------------------------------------- |
| `str` | Name in the format {custom}_{daterange}_{source}. |

Source code in `src/rasteret/core/collection.py`

```python
@classmethod
def create_name(
    cls, custom_name: str, date_range: tuple[str, str], data_source: str
) -> str:
    """Create a standardized collection name.

    Parameters
    ----------
    custom_name : str
        User-chosen name component. Underscores are normalised to dashes.
    date_range : tuple of str
        ``(start, end)`` ISO date strings.
    data_source : str
        Data source identifier (e.g. ``"sentinel-2-l2a"``).

    Returns
    -------
    str
        Name in the format ``{custom}_{daterange}_{source}``.
    """
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    custom_token = custom_name.lower().replace(" ", "-").replace("_", "-")
    custom_token = re.sub(r"[^a-z0-9-]+", "-", custom_token)
    custom_token = re.sub(r"-{2,}", "-", custom_token).strip("-")
    if not custom_token:
        custom_token = "collection"

    name_parts = [
        custom_token,
        cls._format_date_range(start_date, end_date),
        cls._source_token(data_source),
    ]
    return "_".join(name_parts)
```

###### parse_name

```python
parse_name(name: str) -> dict[str, str | None]
```

Parse a standardized collection name into its components.

Parameters:

| Name   | Type  | Description                                   | Default    |
| ------ | ----- | --------------------------------------------- | ---------- |
| `name` | `str` | Collection name created by :meth:create_name. | *required* |

Returns:

| Type   | Description                                                 |
| ------ | ----------------------------------------------------------- |
| `dict` | Keys: custom_name, data_source (None if unparseable), name. |

Source code in `src/rasteret/core/collection.py`

```python
@classmethod
def parse_name(cls, name: str) -> dict[str, str | None]:
    """Parse a standardized collection name into its components.

    Parameters
    ----------
    name : str
        Collection name created by :meth:`create_name`.

    Returns
    -------
    dict
        Keys: ``custom_name``, ``data_source`` (``None`` if unparseable),
        ``name``.
    """
    try:
        # Remove _stac suffix if present
        clean = name.replace("_stac", "")

        # Split parts
        parts = clean.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid name format: {clean}")

        custom_name, date_str, source = parts

        # Parse date range
        date_parts = date_str.split("-")
        if len(date_parts) != 2:
            raise ValueError(f"Invalid date format: {date_str}")

        return {
            "custom_name": custom_name,
            "data_source": source,
            "name": clean,
        }

    except ValueError as e:
        logger.debug("Failed to parse collection name %r: %s", name, e)
        return {"name": name, "custom_name": name, "data_source": None}
```

###### to_torchgeo_dataset

```python
to_torchgeo_dataset(
    *,
    bands: list[str],
    chip_size: int | None = None,
    is_image: bool = True,
    allow_resample: bool = False,
    cloud_cover_lt: float | None = None,
    date_range: tuple[str, str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    split: str | Sequence[str] | None = None,
    split_column: str = "split",
    label_field: str | None = None,
    geometries: Any = None,
    geometries_crs: int = 4326,
    transforms: Any = None,
    max_concurrent: int = 50,
    cloud_config: Any = None,
    backend: Any = None,
    time_series: bool = False,
    target_crs: int | None = None,
) -> RasteretGeoDataset
```

Create a TorchGeo GeoDataset backed by this Collection.

This integration is optional and requires `torchgeo` and its dependencies.

Parameters:

| Name             | Type                                                        | Description                                                                                                                                                                                        | Default    |
| ---------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `bands`          | `list of str`                                               | Band codes to load (e.g. ["B04", "B03", "B02"]).                                                                                                                                                   | *required* |
| `chip_size`      | `int`                                                       | Spatial extent of each chip in pixels.                                                                                                                                                             | `None`     |
| `is_image`       | `bool`                                                      | If True (default), return chips as sample["image"]. If False, return chips as sample["mask"] (single-band data will have its channel dimension squeezed to match TorchGeo RasterDataset behavior). | `True`     |
| `allow_resample` | `bool`                                                      | If True, Rasteret will resample bands to the dataset grid when requested bands have different resolutions. This is opt-in because it may change pixel values (resampling) and can be slow.         | `False`    |
| `cloud_cover_lt` | `float`                                                     | Keep only records with eo:cloud_cover below this value before constructing the TorchGeo dataset.                                                                                                   | `None`     |
| `date_range`     | `tuple of str`                                              | Keep only records whose datetime falls within (start, end) before constructing the TorchGeo dataset.                                                                                               | `None`     |
| `bbox`           | `tuple of float`                                            | Spatial bbox filter applied before constructing the TorchGeo dataset.                                                                                                                              | `None`     |
| `split`          | `str or sequence of str`                                    | Filter to the given split(s) before creating the dataset.                                                                                                                                          | `None`     |
| `split_column`   | `str`                                                       | Column holding split labels. Defaults to "split".                                                                                                                                                  | `'split'`  |
| `label_field`    | `str`                                                       | Column name to include as sample["label"].                                                                                                                                                         | `None`     |
| `geometries`     | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Spatial extent for the dataset. Accepts (minx, miny, maxx, maxy) bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects, raw WKB bytes, or GeoJSON dicts.                               | `None`     |
| `geometries_crs` | `int`                                                       | EPSG code for geometries. Defaults to 4326.                                                                                                                                                        | `4326`     |
| `transforms`     | `callable`                                                  | TorchGeo-compatible transforms applied to each sample.                                                                                                                                             | `None`     |
| `max_concurrent` | `int`                                                       | Maximum concurrent HTTP requests.                                                                                                                                                                  | `50`       |
| `cloud_config`   | `CloudConfig`                                               | Cloud configuration for URL rewriting.                                                                                                                                                             | `None`     |
| `backend`        | `StorageBackend`                                            | Pluggable I/O backend (e.g. ObstoreBackend).                                                                                                                                                       | `None`     |
| `time_series`    | `bool`                                                      | When True, stack all timesteps as [T, C, H, W].                                                                                                                                                    | `False`    |
| `target_crs`     | `int`                                                       | Reproject all records to this EPSG code at read time.                                                                                                                                              | `None`     |

Returns:

| Type                 | Description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| `RasteretGeoDataset` | A standard TorchGeo GeoDataset. Pixel data is in the native COG dtype (e.g. uint16 for Sentinel-2). |

Source code in `src/rasteret/core/collection.py`

```python
def to_torchgeo_dataset(
    self,
    *,
    bands: list[str],
    chip_size: int | None = None,
    is_image: bool = True,
    allow_resample: bool = False,
    cloud_cover_lt: float | None = None,
    date_range: tuple[str, str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    split: str | Sequence[str] | None = None,
    split_column: str = "split",
    label_field: str | None = None,
    geometries: Any = None,
    geometries_crs: int = 4326,
    transforms: Any = None,
    max_concurrent: int = 50,
    cloud_config: Any = None,
    backend: Any = None,
    time_series: bool = False,
    target_crs: int | None = None,
) -> RasteretGeoDataset:
    """Create a TorchGeo GeoDataset backed by this Collection.

    This integration is optional and requires ``torchgeo`` and its
    dependencies.

    Parameters
    ----------
    bands : list of str
        Band codes to load (e.g. ``["B04", "B03", "B02"]``).
    chip_size : int, optional
        Spatial extent of each chip in pixels.
    is_image : bool
        If ``True`` (default), return chips as ``sample[\"image\"]``.
        If ``False``, return chips as ``sample[\"mask\"]`` (single-band data
        will have its channel dimension squeezed to match TorchGeo
        ``RasterDataset`` behavior).
    allow_resample : bool
        If ``True``, Rasteret will resample bands to the dataset grid when
        requested bands have different resolutions. This is opt-in because
        it may change pixel values (resampling) and can be slow.
    cloud_cover_lt : float, optional
        Keep only records with ``eo:cloud_cover`` below this value before
        constructing the TorchGeo dataset.
    date_range : tuple of str, optional
        Keep only records whose ``datetime`` falls within
        ``(start, end)`` before constructing the TorchGeo dataset.
    bbox : tuple of float, optional
        Spatial bbox filter applied before constructing the TorchGeo
        dataset.
    split : str or sequence of str, optional
        Filter to the given split(s) before creating the dataset.
    split_column : str
        Column holding split labels. Defaults to ``"split"``.
    label_field : str, optional
        Column name to include as ``sample["label"]``.
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict, optional
        Spatial extent for the dataset. Accepts ``(minx, miny, maxx, maxy)``
        bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects,
        raw WKB bytes, or GeoJSON dicts.
    geometries_crs : int
        EPSG code for *geometries*. Defaults to ``4326``.
    transforms : callable, optional
        TorchGeo-compatible transforms applied to each sample.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    backend : StorageBackend, optional
        Pluggable I/O backend (e.g. ``ObstoreBackend``).
    time_series : bool
        When ``True``, stack all timesteps as ``[T, C, H, W]``.
    target_crs : int, optional
        Reproject all records to this EPSG code at read time.

    Returns
    -------
    RasteretGeoDataset
        A standard TorchGeo ``GeoDataset``. Pixel data is in the
        native COG dtype (e.g. ``uint16`` for Sentinel-2).
    """
    from rasteret.integrations.torchgeo import RasteretGeoDataset

    self._validate_bands(bands)

    selected_collection = self
    explicit_prefilter_kwargs: dict[str, Any] = {}
    if cloud_cover_lt is not None:
        explicit_prefilter_kwargs["cloud_cover_lt"] = cloud_cover_lt
    if date_range is not None:
        explicit_prefilter_kwargs["date_range"] = date_range
    if bbox is not None:
        explicit_prefilter_kwargs["bbox"] = bbox
    if split is not None:
        explicit_prefilter_kwargs["split"] = split
        explicit_prefilter_kwargs["split_column"] = split_column

    if explicit_prefilter_kwargs:
        selected_collection = self.subset(**explicit_prefilter_kwargs)

    if geometries is not None:
        derived_bbox = _derive_query_bbox(geometries, geometry_crs=geometries_crs)
        if derived_bbox is not None:
            merged_bbox = intersect_bbox(bbox, derived_bbox)
            if bbox is not None and merged_bbox is None:
                selected_collection = selected_collection._view(
                    selected_collection.dataset.filter(ds.scalar(False))
                )
            else:
                try:
                    selected_collection = selected_collection.subset(
                        bbox=merged_bbox or derived_bbox
                    )
                except ValueError as exc:
                    logger.debug(
                        "TorchGeo prefilter could not apply derived bbox %s: %s",
                        merged_bbox or derived_bbox,
                        exc,
                    )

    return RasteretGeoDataset(
        collection=selected_collection,
        bands=bands,
        chip_size=chip_size,
        is_image=is_image,
        allow_resample=allow_resample,
        label_field=label_field,
        geometries=geometries,
        geometries_crs=geometries_crs,
        transforms=transforms,
        cloud_config=cloud_config,
        max_concurrent=max_concurrent,
        backend=backend,
        time_series=time_series,
        target_crs=target_crs,
    )
```

###### get_xarray

```python
get_xarray(
    geometries: Any,
    bands: list[str],
    *,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    xr_combine: str = "combine_first",
    **filters: Any,
) -> Dataset
```

Load selected bands into an xarray Dataset.

Parameters:

| Name             | Type                                                        | Description                                                                                                                                                                                                                                                                                                            | Default           |
| ---------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `geometries`     | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Area(s) of interest to load. Accepts (minx, miny, maxx, maxy) bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects, raw WKB bytes, or GeoJSON dicts.                                                                                                                                                      | *required*        |
| `bands`          | `list of str`                                               | Band codes to load.                                                                                                                                                                                                                                                                                                    | *required*        |
| `max_concurrent` | `int`                                                       | Maximum concurrent HTTP requests.                                                                                                                                                                                                                                                                                      | `50`              |
| `cloud_config`   | `CloudConfig`                                               | Cloud configuration for URL rewriting.                                                                                                                                                                                                                                                                                 | `None`            |
| `data_source`    | `str`                                                       | Override the inferred data source.                                                                                                                                                                                                                                                                                     | `None`            |
| `backend`        | `StorageBackend`                                            | Pluggable I/O backend.                                                                                                                                                                                                                                                                                                 | `None`            |
| `target_crs`     | `int`                                                       | Reproject all records to this CRS before merging.                                                                                                                                                                                                                                                                      | `None`            |
| `all_touched`    | `bool`                                                      | Passed through to polygon masking behavior. False matches rasterio default semantics.                                                                                                                                                                                                                                  | `False`           |
| `xr_combine`     | `str`                                                       | Strategy for merging per-record xarray Datasets. "combine_first" (default) preserves all data and fills NaN gaps from subsequent records. "merge" uses xr.merge(join="outer") which raises on value conflicts. "merge_override" uses xr.merge(compat="override") which silently picks one record's values in overlaps. | `'combine_first'` |
| `progress`       | `bool`                                                      | If True, show progress bars during remote reads. If None, uses the global default set by :func:rasteret.set_options.                                                                                                                                                                                                   | `None`            |
| `filters`        | `kwargs`                                                    | Additional keyword arguments passed to :meth:subset.                                                                                                                                                                                                                                                                   | `{}`              |

Returns:

| Type      | Description                                                                                                                                                                                                             |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dataset` | Band arrays in native COG dtype (e.g. uint16 for Sentinel-2). CRS encoded via CF conventions (spatial_ref coordinate with WKT2, PROJJSON, GeoTransform). Multi-CRS queries are auto-reprojected to the most common CRS. |

Source code in `src/rasteret/core/collection.py`

```python
def get_xarray(
    self,
    geometries: Any,
    bands: list[str],
    *,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    xr_combine: str = "combine_first",
    **filters: Any,
) -> xr.Dataset:
    """Load selected bands into an xarray Dataset.

    Parameters
    ----------
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
        Area(s) of interest to load. Accepts ``(minx, miny, maxx, maxy)``
        bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects,
        raw WKB bytes, or GeoJSON dicts.
    bands : list of str
        Band codes to load.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    data_source : str, optional
        Override the inferred data source.
    backend : StorageBackend, optional
        Pluggable I/O backend.
    target_crs : int, optional
        Reproject all records to this CRS before merging.
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
    xr_combine : str
        Strategy for merging per-record xarray Datasets.
        ``"combine_first"`` (default) preserves all data and fills
        NaN gaps from subsequent records. ``"merge"`` uses
        ``xr.merge(join="outer")`` which raises on value conflicts.
        ``"merge_override"`` uses ``xr.merge(compat="override")``
        which silently picks one record's values in overlaps.
    progress : bool, optional
        If ``True``, show progress bars during remote reads. If ``None``,
        uses the global default set by :func:`rasteret.set_options`.
    filters : kwargs
        Additional keyword arguments passed to :meth:`subset`.

    Returns
    -------
    xarray.Dataset
        Band arrays in native COG dtype (e.g. ``uint16`` for
        Sentinel-2). CRS encoded via CF conventions (``spatial_ref``
        coordinate with WKT2, PROJJSON, GeoTransform). Multi-CRS
        queries are auto-reprojected to the most common CRS.
    """
    self._validate_bands(bands)
    if backend is None:
        backend = self._auto_backend(cloud_config, data_source)
    if progress is None:
        from rasteret.options import get_options

        progress = get_options().progress
    return get_collection_xarray(
        collection=self,
        geometries=geometries,
        bands=bands,
        data_source=data_source,
        max_concurrent=max_concurrent,
        progress=bool(progress),
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        all_touched=all_touched,
        xr_combine=xr_combine,
        **filters,
    )
```

###### get_gdf

```python
get_gdf(
    geometries: Any,
    bands: list[str],
    *,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    **filters: Any,
) -> GeoDataFrame
```

Load selected bands into a GeoDataFrame.

Parameters:

| Name             | Type                                                        | Description                                                                                                                                                       | Default    |
| ---------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `geometries`     | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Area(s) of interest to load. Accepts (minx, miny, maxx, maxy) bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects, raw WKB bytes, or GeoJSON dicts. | *required* |
| `bands`          | `list of str`                                               | Band codes to load.                                                                                                                                               | *required* |
| `max_concurrent` | `int`                                                       | Maximum concurrent HTTP requests.                                                                                                                                 | `50`       |
| `cloud_config`   | `CloudConfig`                                               | Cloud configuration for URL rewriting.                                                                                                                            | `None`     |
| `data_source`    | `str`                                                       | Override the inferred data source.                                                                                                                                | `None`     |
| `backend`        | `StorageBackend`                                            | Pluggable I/O backend.                                                                                                                                            | `None`     |
| `target_crs`     | `int`                                                       | Reproject all records to this CRS before building the GeoDataFrame.                                                                                               | `None`     |
| `all_touched`    | `bool`                                                      | Passed through to polygon masking behavior. False matches rasterio default semantics.                                                                             | `False`    |
| `progress`       | `bool`                                                      | If True, show progress bars during remote reads. If None, uses the global default set by :func:rasteret.set_options.                                              | `None`     |
| `filters`        | `kwargs`                                                    | Additional keyword arguments passed to :meth:subset.                                                                                                              | `{}`       |

Returns:

| Type           | Description                                                                                     |
| -------------- | ----------------------------------------------------------------------------------------------- |
| `GeoDataFrame` | Band arrays in native COG dtype. Each row is a geometry-record pair with pixel data as columns. |

Source code in `src/rasteret/core/collection.py`

```python
def get_gdf(
    self,
    geometries: Any,
    bands: list[str],
    *,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    **filters: Any,
) -> gpd.GeoDataFrame:
    """Load selected bands into a GeoDataFrame.

    Parameters
    ----------
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
        Area(s) of interest to load. Accepts ``(minx, miny, maxx, maxy)``
        bbox tuples, Arrow arrays (e.g. from GeoParquet), Shapely objects,
        raw WKB bytes, or GeoJSON dicts.
    bands : list of str
        Band codes to load.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    data_source : str, optional
        Override the inferred data source.
    backend : StorageBackend, optional
        Pluggable I/O backend.
    target_crs : int, optional
        Reproject all records to this CRS before building the GeoDataFrame.
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
    progress : bool, optional
        If ``True``, show progress bars during remote reads. If ``None``,
        uses the global default set by :func:`rasteret.set_options`.
    filters : kwargs
        Additional keyword arguments passed to :meth:`subset`.

    Returns
    -------
    geopandas.GeoDataFrame
        Band arrays in native COG dtype. Each row is a
        geometry-record pair with pixel data as columns.
    """
    self._validate_bands(bands)
    if backend is None:
        backend = self._auto_backend(cloud_config, data_source)
    if progress is None:
        from rasteret.options import get_options

        progress = get_options().progress
    return get_collection_gdf(
        collection=self,
        geometries=geometries,
        bands=bands,
        data_source=data_source,
        max_concurrent=max_concurrent,
        progress=bool(progress),
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        all_touched=all_touched,
        **filters,
    )
```

###### get_numpy

```python
get_numpy(
    geometries: Any,
    bands: list[str],
    *,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    **filters: Any,
)
```

Load selected bands into NumPy arrays.

Parameters:

| Name             | Type                                                        | Description                                                                                                          | Default    |
| ---------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------- |
| `geometries`     | `bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict` | Area(s) of interest to load.                                                                                         | *required* |
| `bands`          | `list of str`                                               | Band codes to load.                                                                                                  | *required* |
| `max_concurrent` | `int`                                                       | Maximum concurrent HTTP requests.                                                                                    | `50`       |
| `cloud_config`   | `CloudConfig`                                               | Cloud configuration for URL rewriting.                                                                               | `None`     |
| `data_source`    | `str`                                                       | Override the inferred data source.                                                                                   | `None`     |
| `backend`        | `StorageBackend`                                            | Pluggable I/O backend.                                                                                               | `None`     |
| `target_crs`     | `int`                                                       | Reproject all records to this CRS before assembly.                                                                   | `None`     |
| `all_touched`    | `bool`                                                      | Passed through to polygon masking behavior. False matches rasterio default semantics.                                | `False`    |
| `progress`       | `bool`                                                      | If True, show progress bars during remote reads. If None, uses the global default set by :func:rasteret.set_options. | `None`     |
| `filters`        | `kwargs`                                                    | Additional keyword arguments passed to :meth:subset.                                                                 | `{}`       |

Returns:

| Type      | Description                                                                                           |
| --------- | ----------------------------------------------------------------------------------------------------- |
| `ndarray` | Single-band queries return [N, H, W]. Multi-band queries return [N, C, H, W] in requested band order. |

Source code in `src/rasteret/core/collection.py`

```python
def get_numpy(
    self,
    geometries: Any,
    bands: list[str],
    *,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    target_crs: int | None = None,
    geometry_crs: int | None = 4326,
    all_touched: bool = False,
    **filters: Any,
):
    """Load selected bands into NumPy arrays.

    Parameters
    ----------
    geometries : bbox tuple, pa.Array, Shapely, WKB bytes, or GeoJSON dict
        Area(s) of interest to load.
    bands : list of str
        Band codes to load.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    data_source : str, optional
        Override the inferred data source.
    backend : StorageBackend, optional
        Pluggable I/O backend.
    target_crs : int, optional
        Reproject all records to this CRS before assembly.
    all_touched : bool
        Passed through to polygon masking behavior. ``False`` matches
        rasterio default semantics.
    progress : bool, optional
        If ``True``, show progress bars during remote reads. If ``None``,
        uses the global default set by :func:`rasteret.set_options`.
    filters : kwargs
        Additional keyword arguments passed to :meth:`subset`.

    Returns
    -------
    numpy.ndarray
        Single-band queries return ``[N, H, W]``.
        Multi-band queries return ``[N, C, H, W]`` in requested band order.
    """
    self._validate_bands(bands)
    if backend is None:
        backend = self._auto_backend(cloud_config, data_source)
    if progress is None:
        from rasteret.options import get_options

        progress = get_options().progress
    return get_collection_numpy(
        collection=self,
        geometries=geometries,
        bands=bands,
        data_source=data_source,
        max_concurrent=max_concurrent,
        progress=bool(progress),
        backend=backend,
        target_crs=target_crs,
        geometry_crs=geometry_crs,
        all_touched=all_touched,
        **filters,
    )
```

###### sample_points

```python
sample_points(
    points: Any,
    bands: list[str],
    *,
    geometry_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    geometry_crs: int | None = 4326,
    match: str = "all",
    **filters: Any,
) -> Table
```

Sample point values into an Arrow table.

Parameters:

| Name              | Type                | Description                                                                                                                            | Default    |
| ----------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `points`          | `Any`               | Point input as Arrow/GeoArrow/WKB/Shapely/GeoJSON, or tabular input (Arrow table, pandas/GeoPandas, Polars, DuckDB/SedonaDB relation). | *required* |
| `bands`           | `list of str`       | Band codes to sample.                                                                                                                  | *required* |
| `geometry_column` | `str`               | Geometry column name when points is tabular. Column may contain WKB, GeoArrow points, or Shapely Point objects.                        | `None`     |
| `x_column`        | `str`               | Coordinate column names when points is tabular.                                                                                        | `None`     |
| `y_column`        | `str`               | Coordinate column names when points is tabular.                                                                                        | `None`     |
| `max_concurrent`  | `int`               | Maximum concurrent HTTP requests.                                                                                                      | `50`       |
| `progress`        | `bool`              | If True, show progress bars during remote reads. If None, uses the global default set by :func:rasteret.set_options.                   | `None`     |
| `cloud_config`    | `CloudConfig`       | Cloud configuration for URL rewriting.                                                                                                 | `None`     |
| `data_source`     | `str`               | Override the inferred data source.                                                                                                     | `None`     |
| `backend`         | `StorageBackend`    | Pluggable I/O backend.                                                                                                                 | `None`     |
| `geometry_crs`    | `int`               | CRS EPSG code of input points. Defaults to EPSG:4326.                                                                                  | `4326`     |
| `match`           | `('all', 'latest')` | "all" returns every matching record for each point. "latest" returns one row per (point_index, band).                                  | `"all"`    |
| `filters`         | `kwargs`            | Additional keyword arguments passed to :meth:subset.                                                                                   | `{}`       |

Returns:

| Type    | Description                                     |
| ------- | ----------------------------------------------- |
| `Table` | Table with sampled values and metadata columns. |

Source code in `src/rasteret/core/collection.py`

```python
def sample_points(
    self,
    points: Any,
    bands: list[str],
    *,
    geometry_column: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    max_concurrent: int = 50,
    progress: bool | None = None,
    cloud_config: Any = None,
    data_source: str | None = None,
    backend: Any = None,
    geometry_crs: int | None = 4326,
    match: str = "all",
    **filters: Any,
) -> pa.Table:
    """Sample point values into an Arrow table.

    Parameters
    ----------
    points : Any
        Point input as Arrow/GeoArrow/WKB/Shapely/GeoJSON, or tabular input
        (Arrow table, pandas/GeoPandas, Polars, DuckDB/SedonaDB relation).
    bands : list of str
        Band codes to sample.
    geometry_column : str, optional
        Geometry column name when *points* is tabular. Column may contain WKB,
        GeoArrow points, or Shapely Point objects.
    x_column, y_column : str, optional
        Coordinate column names when *points* is tabular.
    max_concurrent : int
        Maximum concurrent HTTP requests.
    progress : bool, optional
        If ``True``, show progress bars during remote reads. If ``None``,
        uses the global default set by :func:`rasteret.set_options`.
    cloud_config : CloudConfig, optional
        Cloud configuration for URL rewriting.
    data_source : str, optional
        Override the inferred data source.
    backend : StorageBackend, optional
        Pluggable I/O backend.
    geometry_crs : int, optional
        CRS EPSG code of input points. Defaults to EPSG:4326.
    match : {"all", "latest"}
        ``"all"`` returns every matching record for each point.
        ``"latest"`` returns one row per ``(point_index, band)``.
    filters : kwargs
        Additional keyword arguments passed to :meth:`subset`.

    Returns
    -------
    pyarrow.Table
        Table with sampled values and metadata columns.
    """
    self._validate_bands(bands)
    if backend is None:
        backend = self._auto_backend(cloud_config, data_source)
    if progress is None:
        from rasteret.options import get_options

        progress = get_options().progress
    return get_collection_point_samples(
        collection=self,
        points=points,
        bands=bands,
        geometry_column=geometry_column,
        x_column=x_column,
        y_column=y_column,
        data_source=data_source,
        max_concurrent=max_concurrent,
        progress=bool(progress),
        backend=backend,
        geometry_crs=geometry_crs,
        match=match,
        **filters,
    )
```

### Functions
