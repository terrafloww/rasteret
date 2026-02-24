# AlphaEarth Foundation Embeddings (AEF)

[AlphaEarth Foundation (AEF)](https://source.coop/repositories/tge-labs/aef)
embeddings are 64-band, 10 m resolution foundation-model features published as
tiled GeoTIFFs. Rasteret reads them efficiently by caching COG tile headers once
and then doing fast byte-range reads for each query.

---

## Quick start

AEF is a built-in catalog dataset. Three lines gets you from zero to pixels:

```python
import rasteret

collection = rasteret.build(
    "aef/v1-annual",
    name="aef-demo",
    bbox=(11.3, -0.002, 11.5, 0.001),
    date_range=("2023-01-01", "2023-12-31"),
)

ds = collection.get_xarray(
    geometries=(11.3, -0.002, 11.5, 0.001),
    bands=["A00", "A01", "A31", "A63"],
)
```

`build()` reads the published AEF GeoParquet index from Source Cooperative,
filters by bbox and year, constructs the assets from each tile's COG URL,
parses COG headers, and caches everything as a local Parquet index.
Subsequent calls reuse the cache.

---

## Selecting bands

AEF COGs contain 64 bands (`A00`--`A63`). Enrichment parses the full
tile layout once per COG (a single HTTP request regardless of band count),
so you pay no extra cost at build time.

Select which bands to read at query time:

```python
# Read all 64 bands
ds = collection.get_xarray(geometries=bbox, bands=[f"A{i:02d}" for i in range(64)])

# Read a subset
ds = collection.get_xarray(geometries=bbox, bands=["A00", "A15", "A31"])
```

---

## De-quantization and nodata

AEF embeddings are stored as signed 8-bit integers. Nodata is `-128`.
To convert to floats in `[-1.0, 1.0]`:

```python
import numpy as np

raw = ds["A00"].values.squeeze()
embedding = np.where(raw == -128, np.nan, raw.astype(np.float32) / 127.0)
```

---

## DuckDB + Arrow workflow (advanced)

If you want full control over filtering (e.g. UTM zone, custom SQL
predicates), use DuckDB to query the AEF GeoParquet index, then pass
the filtered Arrow table directly into Rasteret via zero-copy:

```python
import duckdb
import rasteret


con = duckdb.connect()
filtered = con.execute("""
    SELECT *
    FROM read_parquet('https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet')
    WHERE year = 2023
      AND utm_zone = '32N'
      AND wgs84_east >= 11.3 AND wgs84_west <= 11.5
    LIMIT 3
""").fetch_arrow_table()  # zero-copy Arrow table

collection = rasteret.build_from_table(
    filtered,  # Arrow table passed directly, no serialization
    name="aef-duckdb",
    column_map={"fid": "id", "geom": "geometry", "year": "datetime"},
    href_column="path",
    band_index_map={f"A{i:02d}": i for i in range(64)},
    url_rewrite_patterns={
        "s3://us-west-2.opendata.source.coop/": "https://data.source.coop/",
    },
    enrich_cog=True,
    band_codes=["A00", "A01", "A31", "A63"],
    force=True,
)
```

The key: `fetch_arrow_table()` returns a PyArrow Table, and
`build_from_table()` accepts it directly. No CSV export, no file I/O,
no intermediate copies. Rasteret's `column_map` aliases the DuckDB
column names to the contract schema, `href_column` + `band_index_map`
construct the assets struct, and `proj:epsg` is derived automatically
from the `crs` column.

See the runnable script at `examples/aef_duckdb_query.py`.

---

## How it works internally

Rasteret reads AEF COGs with the same reader used for all datasets.
The AEF-specific piece is the [catalog descriptor](../reference/catalog.md)
which declares how the GeoParquet index maps to Rasteret's schema:

| Descriptor field | Purpose |
|---|---|
| `column_map` | `{"fid": "id", "geom": "geometry", "year": "datetime"}` - alias source columns to the schema contract |
| `href_column` | `"path"` - column containing COG URLs |
| `band_index_map` | `{"A00": 0, "A01": 1, ..., "A63": 63}` - maps band codes to sample indices in the multi-band COG |
| `bbox_columns` | `{"minx": "wgs84_west", ...}` - enables predicate pushdown on the 235k-row index |

Three general-purpose COG features combine to make the GeoTIFFs work:

- **ModelTransformationTag**: AEF COGs store their geotransform as a 4x4
  matrix (TIFF tag 34264) instead of the more common PixelScale + Tiepoint
  tags. Rasteret's header parser extracts axis-aligned transforms from
  either representation.

- **South-up orientation**: AEF tiles have a positive Y pixel scale
  (`scale_y > 0`), meaning row 0 is the southern edge. Rasteret's tile
  math is sign-agnostic and works with both north-up and south-up rasters.

- **Planar-separate multi-band**: The 64 bands are stored with TIFF
  PlanarConfiguration=2. The `band_index` field in the asset dict tells
  the enrichment pipeline to slice the TileOffsets/TileByteCounts arrays
  per band, so the tile reader handles each band independently.

For the full `band_index` specification, see
[Schema Contract -- asset dict](../explanation/schema-contract.md).
