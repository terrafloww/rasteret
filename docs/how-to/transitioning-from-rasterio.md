# Transitioning from Rasterio & STAC

If you are looking for a **high-performance rasterio alternative** that scales beyond the limitations of sequential GDAL reads, you’ve come to the right place.

Rasteret is designed to **complement** traditional geospatial patterns while drastically simplifying the "plumbing" involved in building scaleable AI/ML training data. We entregue up to **20x faster reads** from remote COGs compared to manual `rasterio.open()` loops.

## The Mental Shift: From Files to Collections

In the traditional workflow, you manage **individual assets**. You are responsible for the loop, the threading, the coordinate alignment, and keeping your labels (like a CSV of crop types) in sync with the scene IDs.

In Rasteret, you manage a **Collection**. A Collection is a single, relational table where pixels and metadata live together.

## The "Plumbing" Comparison

Let's look at a common task: **Load 20 Sentinel-2 patches for a set of polygons and stack them into a NumPy array.**

### The Traditional Way (25+ Lines)
This script is powerful but fragile. It requires manual handling of search pagination, threading, and window calculations.

```python
import pystac_client
import rasterio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# 1. Search and resolve assets
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(collections=["sentinel-2-l2a"], bbox=..., datetime=...)
items = list(search.items())

# 2. Manual tiling and alignment logic
def read_window(item, geom):
    # You have to calculate the window relative to each scene's unique transform
    with rasterio.open(item.assets["B04"].href) as src:
        # Handle CRS alignment, resolution mismatches, and OOB reads manually...
        win = rasterio.windows.from_bounds(*geom.bounds, src.transform)
        return src.read(1, window=win)

# 3. Manage concurrency manually
with ThreadPoolExecutor(max_workers=10) as executor:
    # Need to keep track of which result belongs to which geometry
    results = list(executor.map(lambda i: read_one(i, my_geom), items))

# 4. Final stack
data = np.stack(results)
```

### The Rasteret Way (3 Lines)
Rasteret hides the plumbing. The "loop" is handled by the index, the "threading" is handled by the async IO engine, and the "alignment" is handled by the execution layer.

```python
import rasteret

# 1. Build or Load your collection (index is local and instant)
collection = rasteret.load("my_s2_experiment")

# 2. One call to fetch aligned pixels across N scenes and N geometries
data = collection.get_numpy(geometries=my_polygons, bands=["B04"])
```

## Why this matters for ML

### 1. No "Cold Start" re-parsing
Every time you run `rasterio.open("s3://...")`, GDAL has to fetch the TIFF header (IFD) over the network. If you have 100 scenes in your training batch, you just made 100 blocking network calls before a single pixel moved.

**Rasteret parses these headers once** during `build()` and caches them in your local Parquet index. Subsequent reads are pure pixel-range fetches.

### 2. Relational Metadata
Instead of trying to parse dates or IDs out of filenames, you can treat your Collection like a Dataframe.

```python
# Add your training labels directly to the image index
collection = collection.where(pc.field("crop_type") == "maize")
```

### 3. Native Concurrency
You don't need `ThreadPoolExecutor`. Rasteret's IO engine is built on `asyncio` and `obstore` (Rust), which can pull hundreds of byte-ranges simultaneously without the Python Global Interpreter Lock (GIL) getting in the way of your data prep.

## When to use which tool?

| Task | Use Rasterio when... | Use Rasteret when... |
|---|---|---|
| **Inspection** | You need to `src.tags()` on a single local TIFF. | - |
| **Simple Scripts** | You're reading 1-5 files once. | - |
| **ML Training** | - | You're iterating over thousands of cloud scenes repeatedly. |
| **Large Catalogs** | - | You need to filter 100k scenes instantly without hitting a STAC API. |
| **Sharing Data** | You share a folder of `.tif` files. | You share a 2MB Parquet index that points to the cloud data. |

## Next Step: Take Action

Now that you've seen the "Why" and the "Shift," it's time to build your first collection.

👉 [**First Steps: Getting Started**](../getting-started/index.md)
