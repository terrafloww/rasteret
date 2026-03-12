# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pyarrow.dataset as ds


@dataclass(frozen=True)
class ParquetReadPlanner:
    """Internal descriptor-backed parquet read state.

    This is intentionally small: it stores the two parquet surfaces and the
    accumulated filter state that Rasteret compiles into index and wide-data
    scans. Collection methods still own execution; this object keeps the state
    coherent.
    """

    collection_path: str | None = None
    record_index_path: str | None = None
    record_index_field_roles: dict[str, str] | None = None
    record_index_column_map: dict[str, str] | None = None
    record_index_href_column: str | None = None
    record_index_band_index_map: dict[str, int] | None = None
    record_index_url_rewrite_patterns: dict[str, str] | None = None
    record_index_filesystem: Any | None = None
    surface_fields: dict[str, tuple[str, ...]] | None = None
    filter_capabilities: dict[str, tuple[str, ...]] | None = None
    record_index_filter_expr: ds.Expression | None = None
    wide_filter_expr: ds.Expression | None = None

    def with_updates(self, **changes: Any) -> "ParquetReadPlanner":
        return replace(self, **changes)

    @property
    def has_record_index(self) -> bool:
        return bool(self.record_index_path)

    def source_field(self, canonical: str) -> str:
        if self.record_index_field_roles and canonical in self.record_index_field_roles:
            return self.record_index_field_roles[canonical]
        return canonical

    def surface_has_field(self, surface: str, canonical: str) -> bool:
        if self.surface_fields and surface in self.surface_fields:
            return canonical in set(self.surface_fields[surface])
        return False

    def surface_supports_filter(self, surface: str, capability: str) -> bool:
        if self.filter_capabilities and surface in self.filter_capabilities:
            return capability in set(self.filter_capabilities[surface])
        return self.surface_has_field(surface, capability)
