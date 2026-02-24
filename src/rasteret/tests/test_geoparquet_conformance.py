# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import json
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from rasteret.ingest.normalize import build_collection_from_table


def test_export_writes_geoparquet_geo_metadata(tmp_path) -> None:
    import shapely

    geom = shapely.box(0.0, 0.0, 1.0, 1.0)
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([shapely.to_wkb(geom)], type=pa.binary()),
            "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
        }
    )

    collection = build_collection_from_table(
        table, date_range=("2024-01-01", "2024-01-02")
    )
    out = tmp_path / "collection"
    collection.export(out)

    part = next(out.rglob("*.parquet"))
    md = pq.ParquetFile(part).metadata.metadata or {}

    assert b"geo" in md
    geo = json.loads(md[b"geo"].decode("utf-8"))
    assert geo["version"] == "1.1.0"
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"
    assert "geometry_types" in geo["columns"]["geometry"]
