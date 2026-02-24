# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Ingest builders: source-specific logic that feeds into the Collection contract.

Each builder knows how to read records from one source type (STAC API,
Parquet record tables, etc.) and normalise them into an Arrow table that
satisfies the Collection contract columns
(``id``, ``datetime``, ``geometry``, ``assets``, ``scene_bbox``,
plus optional ``proj:epsg``, ``{band}_metadata``, ``year``, ``month``).

The shared normalisation layer lives in :mod:`rasteret.ingest.normalize`.
"""

from rasteret.ingest.base import CollectionBuilder
from rasteret.ingest.enrich import (
    add_band_metadata_columns,
    build_url_index_from_assets,
    enrich_table_with_cog_metadata,
)
from rasteret.ingest.normalize import build_collection_from_table
from rasteret.ingest.parquet_record_table import RecordTableBuilder
from rasteret.ingest.stac_indexer import StacCollectionBuilder

__all__ = [
    "CollectionBuilder",
    "RecordTableBuilder",
    "StacCollectionBuilder",
    "add_band_metadata_columns",
    "build_collection_from_table",
    "build_url_index_from_assets",
    "enrich_table_with_cog_metadata",
]
