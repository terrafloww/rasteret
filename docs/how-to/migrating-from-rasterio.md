# Migrating From Rasterio

Rasteret changes the shape of remote COG workflows: build a reusable
collection, filter metadata first, then read only the pixels you need. For the
measured TorchGeo/rasterio comparison, see [Benchmarks](../explanation/benchmark.md).

## The Mental Shift

In a rasterio + STAC API workflow, you manage files:

- find STAC items
- pick asset URLs
- open each raster
- compute windows
- handle CRS and resolution differences
- schedule concurrent reads
- stack the results

In Rasteret, you manage a `Collection`.

The collection is a table of raster records. It stores metadata and COG header
information; the pixels stay in the original COGs. You filter the table first,
then choose the output surface you need.

## A Common Task

Suppose you want to read geospatial images for an AOI and stack the result for
analysis or model input.

### Manual rasterio shape

This code is intentionally incomplete, because real code also needs retries,
provider auth, nodata handling, and edge cases. The shape is the point.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pystac_client
import rasterio
from rasterio.windows import from_bounds

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=(-122.5, 37.7, -122.4, 37.8),
    datetime="2024-01-01/2024-06-30",
)
items = list(search.items())


def read_band(item, bounds):
    href = item.assets["red"].href
    with rasterio.open(href) as src:
        # Real code also needs CRS alignment and bounds checks here.
        window = from_bounds(*bounds, transform=src.transform)
        return src.read(1, window=window)


with ThreadPoolExecutor(max_workers=10) as pool:
    arrays = list(pool.map(lambda item: read_band(item, my_bounds), items))

data = np.stack(arrays)
```

### Rasteret shape

```python
import rasteret

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2-training",
    bbox=(-122.5, 37.7, -122.4, 37.8),
    date_range=("2024-01-01", "2024-06-30"),
)

# OR if you have your own COGs
collection = rasteret.build_from_table(
    "path/to/table_with_cogs_metadata.parquet",
    name="my-cogs",
    enrich_cog=True,
)

filtered = collection.subset(cloud_cover_lt=20)

data = filtered.get_numpy(
    geometries=(-122.5, 37.7, -122.4, 37.8),
    bands=["B04", "B08"],
)
```

Rasteret handles the repeated setup:

- catalog/record normalization
- COG header parsing during build
- raster CRS sidecars
- tile and byte-range planning
- async cloud reads
- output assembly

## Build Once, Read Many Times

The build step creates a reusable collection. It stores metadata such as:

- record IDs and timestamps
- footprint geometry
- bounding boxes
- asset hrefs
- native raster CRS sidecars
- per-band COG header metadata
- any extra columns you keep, such as labels or splits

It does not move pixels into Parquet. Pixel data remains in the original
COGs and is read on demand.

## Add Your Own Metadata

If your workflow has labels, split assignments, AOI IDs, or quality flags, add
them as columns in the collection table. Rasteret keeps pixels in the COGs, so
metadata joins can happen in Arrow-native tools without rewriting raster data.

For the full GeoPandas -> DuckDB -> Rasteret pattern, see
[Enriched Collection Workflows](enriched-collection-workflows.md).

## When To Use Which Tool

| Task | Rasterio is a good fit | Rasteret is a good fit |
| --- | --- | --- |
| Inspect one local TIFF | yes | not necessary |
| Debug TIFF tags or profiles | yes | not the focus |
| Read 1 or 2 files just once | yes | maybe not worth building |
| Repeated TIFF reads from cloud storage | possible, but you write the plumbing | yes |
| ML training over many scenes | possible, but setup grows quickly | yes |
| Metadata joins, splits, and labels | external glue code | yes, these can be extra columns in the collection |
| TorchGeo/xarray/NumPy outputs from one source | custom wrappers | yes, via built-in output surfaces |

Next: [How-To Guides](index.md)
