"""AEF + DuckDB + Arrow zero-copy → Rasteret.

Advanced workflow for users who need full SQL control over the AEF index.
For most use cases, ``rasteret.build("aef/v1-annual", ...)`` is simpler.

This script demonstrates Arrow zero-copy interop:

1) DuckDB queries the published AEF GeoParquet index over HTTPS
2) ``fetch_arrow_table()`` returns a PyArrow Table (zero-copy)
3) The Arrow table is passed directly to ``build_from_table()``
   - no file I/O, no serialization, no intermediate copies
4) Rasteret normalizes columns, enriches COG headers, and reads pixels

Example::

    uv run python examples/aef_duckdb_query.py
"""

from __future__ import annotations

import duckdb
import numpy as np

import rasteret

# ---------------------------------------------------------------------------
# 1. Query the AEF index with DuckDB (full SQL control)
# ---------------------------------------------------------------------------
print("Querying AEF index from Source Cooperative...")
INDEX_URI = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
con = duckdb.connect()

# DuckDB reads GeoParquet over HTTPS natively. You can use any SQL
# predicate here - UTM zone, year, spatial bounds, LIMIT, etc.
filtered = con.execute(
    """
    SELECT *
    FROM read_parquet(?)
    WHERE year = 2023
      AND utm_zone = '32N'
      AND wgs84_east >= 11.3 AND wgs84_west <= 11.5
      AND wgs84_north >= -0.002 AND wgs84_south <= 0.001
    LIMIT 3
    """,
    [INDEX_URI],
).fetch_arrow_table()  # <-- zero-copy: DuckDB → PyArrow, no serialization

if filtered.num_rows == 0:
    raise SystemExit("No tiles matched the filter; try a different bbox/year.")

print(
    f"  Filtered to {filtered.num_rows} tiles (Arrow table, {filtered.nbytes:,} bytes)"
)

# ---------------------------------------------------------------------------
# 2. Build + enrich via Arrow zero-copy (no file I/O)
# ---------------------------------------------------------------------------
# The Arrow table from step 1 is passed directly to build_from_table().
# Rasteret handles the schema normalization declaratively:
#
#   column_map     → aliases source columns to contract names
#   href_column    → identifies the COG URL column
#   band_index_map → maps band codes to sample indices in the multi-band COG
#   proj:epsg      → derived automatically from the "crs" column
#
# No manual column construction needed.
bands = ["A00", "A01", "A31", "A63"]

print("Building enriched collection (zero-copy Arrow → Rasteret)...")
collection = rasteret.build_from_table(
    filtered,  # Arrow table, no disk round-trip
    name="aef-duckdb-example",
    column_map={"fid": "id", "geom": "geometry", "year": "datetime"},
    href_column="path",
    band_index_map={f"A{i:02d}": i for i in range(64)},
    url_rewrite_patterns={
        "s3://us-west-2.opendata.source.coop/": "https://data.source.coop/",
    },
    enrich_cog=True,
    band_codes=bands,
    force=True,
)
print(f"  Collection rows: {collection.dataset.count_rows()}")

# Verify the schema: source columns preserved alongside contract columns
schema_names = collection.dataset.schema.names
assert "fid" in schema_names, "Source column 'fid' should be preserved"
assert "id" in schema_names, "Contract column 'id' should be added"
assert "proj:epsg" in schema_names, "proj:epsg should be derived from 'crs'"
print("  Schema: source columns preserved, contract columns added")

# ---------------------------------------------------------------------------
# 3. Read embeddings
# ---------------------------------------------------------------------------
print("Reading embeddings...")
ds = collection.get_xarray(
    geometries=(11.3, -0.002, 11.5, 0.001),
    bands=bands,
)

for band in bands:
    arr = ds[band].values.squeeze()
    valid = arr[(~np.isnan(arr)) & (arr != -128)]
    if len(valid) > 0:
        print(
            f"  {band}: shape={arr.shape}, valid={len(valid)}, range=[{valid.min():.0f}, {valid.max():.0f}]"
        )
    else:
        print(f"  {band}: shape={arr.shape}, no valid pixels in bbox")

# De-quantize one band as an example.
# AEF: int8 with nodata=-128, divide by 127 for [-1.0, 1.0].
raw = ds["A00"].values.squeeze()
embedding = np.where(raw == -128, np.nan, raw.astype(np.float32) / 127.0)
valid_emb = embedding[~np.isnan(embedding)]
if len(valid_emb) > 0:
    print(f"\n  De-quantized A00: range=[{valid_emb.min():.3f}, {valid_emb.max():.3f}]")

print("\nDone.")
