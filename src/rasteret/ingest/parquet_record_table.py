# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Record-table collection builder.

Reads a Parquet/GeoParquet **record table** (one row per raster item) and
normalizes it into a :class:`~rasteret.core.collection.Collection` via
:func:`~rasteret.ingest.normalize.build_collection_from_table`.

Terminology
-----------
- *Record table* -- a tabular index that enumerates raster items (satellite
  scenes, drone images, derived products, grid cells, etc.). It may come from
  stac-geoparquet, a lab-specific registry, or a custom export.
- *Collection Parquet* -- Rasteret's normalized, runtime-ready Parquet dataset
  that follows the `Schema Contract <explanation/schema-contract>`_ docs page.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from rasteret.cloud import StorageBackend
from rasteret.ingest.base import CollectionBuilder
from rasteret.ingest.normalize import build_collection_from_table, parse_epsg

if TYPE_CHECKING:  # pragma: no cover
    from rasteret.core.collection import Collection

logger = logging.getLogger(__name__)


def _apply_column_map_aliases(
    table: pa.Table, column_map: dict[str, str] | None
) -> pa.Table:
    """Apply a column map without dropping source columns.

    Rasteret's record-table contract requires ``id``, ``datetime``,
    ``geometry``, and ``assets``.  Many upstream tables use different names
    (e.g. ``fid``, ``geom``).  We treat ``column_map`` as an *alias map*:
    when a source column exists and the destination column does not, we add
    a new destination column that references the same Arrow array buffers
    (no data copy), preserving the original name.
    """
    if not column_map:
        return table

    current = set(table.schema.names)
    for src, dst in column_map.items():
        if not src or not dst:
            continue
        if src in current and dst not in current:
            table = table.append_column(dst, table.column(src))
            current.add(dst)
    return table


class RecordTableBuilder(CollectionBuilder):
    """Build a Collection from an existing Parquet/GeoParquet table.

    Reads a Parquet record table where each row is a raster item
    with at minimum the four contract columns (``id``, ``datetime``,
    ``geometry``, ``assets``), or columns that can be normalised into
    them via ``column_map``, ``href_column``, and ``band_index_map``.

    When ``enrich_cog=True``, the builder parses COG headers from the
    asset URLs and adds ``{band}_metadata`` struct columns, making
    the resulting Collection suitable for fast tiled reads and TorchGeo
    integration.

    Parameters
    ----------
    path : str or Path
        Path/URI to the Parquet/GeoParquet file or dataset directory.
    data_source : str
        Data-source identifier for the resulting Collection.
    column_map : dict, optional
        ``{source_name: contract_name}`` alias map applied before
        normalisation.  Source columns are preserved; Rasteret adds
        the contract-name aliases without copying data.
    href_column : str, optional
        Column containing COG URLs.  When set and ``assets`` is absent
        after aliasing, the builder constructs the ``assets`` struct
        from this column and ``band_index_map``.
    band_index_map : dict, optional
        ``{band_code: sample_index}`` for multi-band COGs.  Used with
        ``href_column`` to build per-band asset references.
    url_rewrite_patterns : dict, optional
        ``{source_prefix: target_prefix}`` patterns applied to URLs
        during assets construction (e.g. S3 → HTTPS rewriting).
    filesystem : pyarrow.fs.FileSystem, optional
        PyArrow filesystem for reading remote URIs (e.g.
        ``S3FileSystem(anonymous=True)``).
    columns : list of str, optional
        Scan-time column projection.
    filter_expr : pyarrow.dataset.Expression, optional
        Scan-time predicate pushdown.
    enrich_cog : bool
        If ``True``, parse COG headers from asset URLs and add per-band
        metadata columns.  Default ``False``.
    band_codes : list of str, optional
        Bands to enrich.  If omitted, all bands found in the ``assets``
        column are enriched.
    max_concurrent : int
        Maximum concurrent HTTP connections for COG header parsing.
    name : str, optional
        Collection name.  Passed through to the normalisation layer.
    workspace_dir : str or Path, optional
        If provided, persist the resulting Collection as Parquet here.
    backend : StorageBackend, optional
        I/O backend for authenticated range reads during COG header
        parsing.
    """

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

    def _default_name(self) -> str:
        parsed = urlparse(self.path)
        if parsed.scheme and parsed.path:
            stem = Path(parsed.path).stem
            return stem or "record_table"
        stem = Path(self.path).stem
        return stem or "record_table"

    def _read_table(self) -> pa.Table:
        """Read the source Parquet, using *filesystem* for remote URIs."""
        if self._filesystem is not None:
            dataset = ds.dataset(
                self.path, format="parquet", filesystem=self._filesystem
            )
        else:
            dataset = ds.dataset(self.path, format="parquet")
        return dataset.to_table(columns=self.columns, filter=self.filter_expr)

    def _prepare_table(self, table: pa.Table) -> pa.Table:
        """Normalise column types and construct ``assets`` when absent.

        Called after ``_apply_column_map_aliases`` but before enrichment.

        1. Auto-coerce ``id``: integer → string.
        2. Auto-coerce ``datetime``: integer year → timestamp.
        3. Construct ``assets`` from ``href_column`` + ``band_index_map``.
        4. Derive ``proj:epsg`` from a ``crs`` column when present.
        """
        names = set(table.schema.names)

        # --- id: int → string ---
        if "id" in names and pa.types.is_integer(table.schema.field("id").type):
            table = table.set_column(
                table.schema.get_field_index("id"),
                "id",
                pc.cast(table.column("id"), pa.string()),
            )

        # --- datetime: int year → timestamp ---
        if "datetime" in names and pa.types.is_integer(
            table.schema.field("datetime").type
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
        if "assets" not in names and self.href_column and self.band_index_map:
            if self.href_column not in names:
                raise ValueError(
                    f"href_column '{self.href_column}' not found in table. "
                    f"Available: {sorted(names)}"
                )
            urls = table.column(self.href_column).to_pylist()
            assets_list: list[dict[str, dict[str, object]]] = []
            for url in urls:
                if url is None:
                    assets_list.append({})
                    continue
                rewritten = self._rewrite_url(str(url))
                assets_list.append(
                    {
                        band: {"href": rewritten, "band_index": idx}
                        for band, idx in self.band_index_map.items()
                    }
                )
            table = table.append_column("assets", pa.array(assets_list))

        # --- proj:epsg: derive from crs column ---
        if "proj:epsg" not in names and "crs" in names:
            crs_values = table.column("crs").to_pylist()
            epsg_array = pa.array([parse_epsg(v) for v in crs_values], type=pa.int32())
            table = table.append_column("proj:epsg", epsg_array)

        return table

    def _rewrite_url(self, url: str) -> str:
        """Apply URL rewrite patterns (e.g. S3 → HTTPS)."""
        for src_prefix, dst_prefix in self.url_rewrite_patterns.items():
            if url.startswith(src_prefix):
                return url.replace(src_prefix, dst_prefix, 1)
        return url

    def build(self, **kwargs: Any) -> "Collection":
        """Read the record table and return a normalized Collection.

        Pipeline: read → alias → prepare → enrich → normalize.

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

    def _enrich(self, table: pa.Table) -> pa.Table:
        """Parse COG headers and add band metadata columns."""
        from rasteret.core.utils import run_sync
        from rasteret.ingest.enrich import (
            build_url_index_from_assets,
            enrich_table_with_cog_metadata,
        )

        url_index = build_url_index_from_assets(table, self.band_codes)
        band_codes = self.band_codes or sorted(
            {band for bands in url_index.values() for band in bands}
        )

        if not url_index:
            logger.warning("No asset URLs found for COG enrichment")
            return table

        return run_sync(
            enrich_table_with_cog_metadata(
                table,
                url_index,
                band_codes,
                max_concurrent=self.max_concurrent,
                backend=self._backend,
            )
        )
