# TorchGeo Benchmark: Rasteret vs Native Rasterio[¶](#torchgeo-benchmark-rasteret-vs-native-rasterio)

Compare a Rasteret-backed TorchGeo dataset with a native TorchGeo/rasterio workflow for the measured remote COG setup.

For the full technical narrative, see the [**Concepts**](https://terrafloww.github.io/rasteret/explanation/concepts/).

In \[ \]:

Copied!

```
import os
import time
from pathlib import Path

from shapely.geometry import Polygon

# TorchGeo-recommended GDAL settings (from Pangeo COG best practices).
# These give TorchGeo its best-case scenario for remote COG reads.
# Source: torchgeo.main.rasterio_best_practices
# Reference: https://github.com/pangeo-data/cog-best-practices
rasterio_best_practices = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",  # 200 MB
    "GDAL_SWATH_SIZE": "200000000",  # 200 MB
    "VSI_CURL_CACHE_SIZE": "200000000",  # 200 MB
}
os.environ.update(rasterio_best_practices)

# --- Shared parameters (identical for all paths) ---
aoi = Polygon(
    [
        (77.55, 13.01),
        (77.58, 13.01),
        (77.58, 13.08),
        (77.55, 13.08),
        (77.55, 13.01),
    ]
)
DATE_RANGE = "2024-03-01/2024-08-31"
BAND = "B04"  # single band for fair comparison
STAC_BAND = "red"  # Earth Search v1 common name for B04
CHIP_SIZE = 256
BATCH_SIZE = 2
N_SAMPLES = 8
MAX_SCENES = 15

# --- Shared STAC search (same scenes for all paths) ---
from pystac_client import Client

catalog = Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=aoi.bounds,
    datetime=DATE_RANGE,
    max_items=MAX_SCENES,
)
items = list(search.items())
scene_urls = [item.assets[STAC_BAND].href for item in items if STAC_BAND in item.assets]
scene_dates = sorted({item.datetime.isoformat()[:10] for item in items})

# Derive exact date window from returned scenes (for Rasteret path).
SCENE_DATE_MIN = scene_dates[0]
SCENE_DATE_MAX = scene_dates[-1]

print(f"Shared STAC search: {len(scene_urls)} scenes")
print(f"Date range: {DATE_RANGE}")
print(f"Actual scene dates: {SCENE_DATE_MIN} → {SCENE_DATE_MAX}")
print(f"Band: {BAND} ('{STAC_BAND}' in Earth Search)")
print(f"AOI: {aoi.bounds}")
print("\nGDAL settings (TorchGeo recommended):")
for k, v in rasterio_best_practices.items():
    print(f"  {k}={v}")
```

import os import time from pathlib import Path from shapely.geometry import Polygon

# TorchGeo-recommended GDAL settings (from Pangeo COG best practices).

# These give TorchGeo its best-case scenario for remote COG reads.

# Source: torchgeo.main.rasterio_best_practices

# Reference: https://github.com/pangeo-data/cog-best-practices

rasterio_best_practices = { "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "AWS_NO_SIGN_REQUEST": "YES", "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000", # 200 MB "GDAL_SWATH_SIZE": "200000000", # 200 MB "VSI_CURL_CACHE_SIZE": "200000000", # 200 MB } os.environ.update(rasterio_best_practices)

# --- Shared parameters (identical for all paths) ---

aoi = Polygon( [ (77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01), ] ) DATE_RANGE = "2024-03-01/2024-08-31" BAND = "B04" # single band for fair comparison STAC_BAND = "red" # Earth Search v1 common name for B04 CHIP_SIZE = 256 BATCH_SIZE = 2 N_SAMPLES = 8 MAX_SCENES = 15

# --- Shared STAC search (same scenes for all paths) ---

from pystac_client import Client catalog = Client.open("https://earth-search.aws.element84.com/v1") search = catalog.search( collections=["sentinel-2-l2a"], bbox=aoi.bounds, datetime=DATE_RANGE, max_items=MAX_SCENES, ) items = list(search.items()) scene_urls = \[item.assets[STAC_BAND].href for item in items if STAC_BAND in item.assets\] scene_dates = sorted({item.datetime.isoformat()[:10] for item in items})

# Derive exact date window from returned scenes (for Rasteret path).

SCENE_DATE_MIN = scene_dates[0] SCENE_DATE_MAX = scene_dates[-1] print(f"Shared STAC search: {len(scene_urls)} scenes") print(f"Date range: {DATE_RANGE}") print(f"Actual scene dates: {SCENE_DATE_MIN} → {SCENE_DATE_MAX}") print(f"Band: {BAND} ('{STAC_BAND}' in Earth Search)") print(f"AOI: {aoi.bounds}") print("\\nGDAL settings (TorchGeo recommended):") for k, v in rasterio_best_practices.items(): print(f" {k}={v}")

______________________________________________________________________

## Path A: TorchGeo `time_series=True`[¶](#path-a-torchgeo-time_seriestrue)

TorchGeo reads remote COGs via GDAL's `/vsicurl/`. Each file triggers `rasterio.open()` → HTTP HEAD + IFD range reads for header parsing. With `time_series=True`, every timestep is read **sequentially** then stacked.

In \[ \]:

Copied!

```
# Index build: each /vsicurl/ path triggers rasterio.open() over HTTP
from torchgeo.datasets import RasterDataset

vsicurl_paths = [f"/vsicurl/{url}" for url in scene_urls]

t0 = time.perf_counter()
torchgeo_ds = RasterDataset(
    paths=vsicurl_paths,
    crs="epsg:32643",
    res=10,
    time_series=True,
)
t_index_a = time.perf_counter() - t0

print(
    f"Index build: {t_index_a:.2f}s ({len(vsicurl_paths)} rasterio.open() calls over HTTP)"
)
print(f"Bounds: {torchgeo_ds.bounds}")
```

# Index build: each /vsicurl/ path triggers rasterio.open() over HTTP

from torchgeo.datasets import RasterDataset vsicurl_paths = [f"/vsicurl/{url}" for url in scene_urls] t0 = time.perf_counter() torchgeo_ds = RasterDataset( paths=vsicurl_paths, crs="epsg:32643", res=10, time_series=True, ) t_index_a = time.perf_counter() - t0 print( f"Index build: {t_index_a:.2f}s ({len(vsicurl_paths)} rasterio.open() calls over HTTP)" ) print(f"Bounds: {torchgeo_ds.bounds}")

In \[ \]:

Copied!

```
# Sample chips, each sample reads ALL timesteps sequentially
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import RandomGeoSampler

sampler_a = RandomGeoSampler(torchgeo_ds, size=CHIP_SIZE, length=N_SAMPLES)
loader_a = DataLoader(
    torchgeo_ds,
    sampler=sampler_a,
    batch_size=BATCH_SIZE,
    num_workers=0,
    collate_fn=stack_samples,
)

t0 = time.perf_counter()
batch_a = next(iter(loader_a))
t_read_a = time.perf_counter() - t0

print(f"Batch read: {t_read_a:.2f}s for batch of {BATCH_SIZE} chips")
print(f"image shape: {batch_a['image'].shape}")
print("\n--- Path A totals ---")
print(f"  Index:  {t_index_a:.2f}s")
print(f"  Read:   {t_read_a:.2f}s")
print(f"  Total:  {t_index_a + t_read_a:.2f}s")
```

# Sample chips, each sample reads ALL timesteps sequentially

from torch.utils.data import DataLoader from torchgeo.datasets.utils import stack_samples from torchgeo.samplers import RandomGeoSampler sampler_a = RandomGeoSampler(torchgeo_ds, size=CHIP_SIZE, length=N_SAMPLES) loader_a = DataLoader( torchgeo_ds, sampler=sampler_a, batch_size=BATCH_SIZE, num_workers=0, collate_fn=stack_samples, ) t0 = time.perf_counter() batch_a = next(iter(loader_a)) t_read_a = time.perf_counter() - t0 print(f"Batch read: {t_read_a:.2f}s for batch of {BATCH_SIZE} chips") print(f"image shape: {batch_a['image'].shape}") print("\\n--- Path A totals ---") print(f" Index: {t_index_a:.2f}s") print(f" Read: {t_read_a:.2f}s") print(f" Total: {t_index_a + t_read_a:.2f}s")

______________________________________________________________________

## Path B: Rasteret `time_series=True`[¶](#path-b-rasteret-time_seriestrue)

Rasteret caches COG header metadata (IFD offsets, byte counts, transforms) in a local GeoParquet index. At read time, it skips IFD parsing entirely and fires **concurrent** HTTP range requests for pixel data across ALL timesteps.

`to_torchgeo_dataset(time_series=True)` returns a standard `GeoDataset`, same samplers, same DataLoader, same `stack_samples` collate function as Path A. Each sample returns `[T, C, H, W]` with all timesteps stacked.

In \[ \]:

Copied!

```
import rasteret
from rasteret.constants import DataSources

# Use the exact date window derived from the shared STAC search.
t0 = time.perf_counter()
collection = rasteret.build_from_stac(
    name="bangalorets",
    stac_api="https://earth-search.aws.element84.com/v1",
    collection=DataSources.SENTINEL2,
    bbox=aoi.bounds,
    date_range=(SCENE_DATE_MIN, SCENE_DATE_MAX),
    workspace_dir=Path.home() / "rasteret_workspace",
)
t_stac = time.perf_counter() - t0

n_scenes = collection.dataset.count_rows()
print(f"build_from_stac: {t_stac:.1f}s (cached after first run)")
print(f"Scenes in index: {n_scenes} (target: {len(scene_urls)} from Path A)")
```

import rasteret from rasteret.constants import DataSources

# Use the exact date window derived from the shared STAC search.

t0 = time.perf_counter() collection = rasteret.build_from_stac( name="bangalorets", stac_api="https://earth-search.aws.element84.com/v1", collection=DataSources.SENTINEL2, bbox=aoi.bounds, date_range=(SCENE_DATE_MIN, SCENE_DATE_MAX), workspace_dir=Path.home() / "rasteret_workspace", ) t_stac = time.perf_counter() - t0 n_scenes = collection.dataset.count_rows() print(f"build_from_stac: {t_stac:.1f}s (cached after first run)") print(f"Scenes in index: {n_scenes} (target: {len(scene_urls)} from Path A)")

In \[ \]:

Copied!

```
# Build TorchGeo dataset from Rasteret collection + sample chips
# Same sampler, same DataLoader, same collate_fn as Path A
# time_series=True → all timesteps stacked as [T, C, H, W]

t0 = time.perf_counter()
rasteret_dataset = collection.to_torchgeo_dataset(
    bands=[BAND],
    geometries=[aoi],
    chip_size=CHIP_SIZE,
    time_series=True,
)
t_index_b = time.perf_counter() - t0

sampler_b = RandomGeoSampler(rasteret_dataset, size=CHIP_SIZE, length=N_SAMPLES)
loader_b = DataLoader(
    rasteret_dataset,
    sampler=sampler_b,
    batch_size=BATCH_SIZE,
    num_workers=0,
    collate_fn=stack_samples,
)

t0 = time.perf_counter()
batch_b = next(iter(loader_b))
t_read_b = time.perf_counter() - t0

print(f"Index build: {t_index_b:.2f}s (from Parquet, no HTTP)")
print(f"Batch read:  {t_read_b:.2f}s for batch of {BATCH_SIZE} chips")
print(f"image shape: {batch_b['image'].shape}")
print("\n--- Path B totals ---")
print(f"  STAC:   {t_stac:.2f}s (one-time, cached)")
print(f"  Index:  {t_index_b:.2f}s")
print(f"  Read:   {t_read_b:.2f}s")
print(f"  Total:  {t_index_b + t_read_b:.2f}s (excl. one-time STAC cache)")
```

# Build TorchGeo dataset from Rasteret collection + sample chips

# Same sampler, same DataLoader, same collate_fn as Path A

# time_series=True → all timesteps stacked as [T, C, H, W]

t0 = time.perf_counter() rasteret_dataset = collection.to_torchgeo_dataset( bands=[BAND], geometries=[aoi], chip_size=CHIP_SIZE, time_series=True, ) t_index_b = time.perf_counter() - t0 sampler_b = RandomGeoSampler(rasteret_dataset, size=CHIP_SIZE, length=N_SAMPLES) loader_b = DataLoader( rasteret_dataset, sampler=sampler_b, batch_size=BATCH_SIZE, num_workers=0, collate_fn=stack_samples, ) t0 = time.perf_counter() batch_b = next(iter(loader_b)) t_read_b = time.perf_counter() - t0 print(f"Index build: {t_index_b:.2f}s (from Parquet, no HTTP)") print(f"Batch read: {t_read_b:.2f}s for batch of {BATCH_SIZE} chips") print(f"image shape: {batch_b['image'].shape}") print("\\n--- Path B totals ---") print(f" STAC: {t_stac:.2f}s (one-time, cached)") print(f" Index: {t_index_b:.2f}s") print(f" Read: {t_read_b:.2f}s") print(f" Total: {t_index_b + t_read_b:.2f}s (excl. one-time STAC cache)")

In \[ \]:

Copied!

```
# --- Timing summary ---
print("=" * 62)
print(f"{'':20} {'TorchGeo':>12} {'Rasteret':>12} {'Speedup':>10}")
print("-" * 62)
print(f"{'Index/header':20} {t_index_a:>11.2f}s {t_index_b:>11.2f}s {'':>10}")
print(f"{'Batch read':20} {t_read_a:>11.2f}s {t_read_b:>11.2f}s {'':>10}")
total_a = t_index_a + t_read_a
total_b = t_index_b + t_read_b
print(f"{'Total':20} {total_a:>11.2f}s {total_b:>11.2f}s {'':>10}")
print(
    f"{'vs TorchGeo':20} {'1.0x':>12} {total_a / max(total_b, 0.001):>11.1f}x {'':>10}"
)
print("=" * 62)
print("\nControlled variables:")
print(f"  Scenes: {len(scene_urls)}, Band: {BAND}, Chip: {CHIP_SIZE}x{CHIP_SIZE}")
print(f"  Batch: {BATCH_SIZE}, Samples: {N_SAMPLES}")
print(f"\nPath A shape: {batch_a['image'].shape}")
print(f"Path B shape: {batch_b['image'].shape}")

section1_speedup = total_a / max(total_b, 0.001)
```

# --- Timing summary ---

print("=" * 62) print(f"{'':20} {'TorchGeo':>12} {'Rasteret':>12} {'Speedup':>10}") print("-" * 62) print(f"{'Index/header':20} {t_index_a:>11.2f}s {t_index_b:>11.2f}s {'':>10}") print(f"{'Batch read':20} {t_read_a:>11.2f}s {t_read_b:>11.2f}s {'':>10}") total_a = t_index_a + t_read_a total_b = t_index_b + t_read_b print(f"{'Total':20} {total_a:>11.2f}s {total_b:>11.2f}s {'':>10}") print( f"{'vs TorchGeo':20} {'1.0x':>12} {total_a / max(total_b, 0.001):>11.1f}x {'':>10}" ) print("=" * 62) print("\\nControlled variables:") print(f" Scenes: {len(scene_urls)}, Band: {BAND}, Chip: {CHIP_SIZE}x{CHIP_SIZE}") print(f" Batch: {BATCH_SIZE}, Samples: {N_SAMPLES}") print(f"\\nPath A shape: {batch_a['image'].shape}") print(f"Path B shape: {batch_b['image'].shape}") section1_speedup = total_a / max(total_b, 0.001)

______________________________________________________________________

## What's different under the hood[¶](#whats-different-under-the-hood)

Both paths produce `[batch, T, C, H, W]`, all timesteps stacked per chip.

|                                | Path A: TorchGeo                                | Path B: Rasteret                      |
| ------------------------------ | ----------------------------------------------- | ------------------------------------- |
| **Index build**                | `rasterio.open()` per COG over HTTP             | Pre-built GeoParquet (read from disk) |
| **Time series read**           | Sequential: one `rasterio.merge()` per timestep | All T timesteps fired concurrently    |
| **HTTP overhead per timestep** | HEAD + IFD ranges + pixel ranges                | Pixel ranges only (headers cached)    |
| **Concurrency**                | None; GDAL reads are serial                     | asyncio.gather across all T × C reads |

### Where the bottleneck is[¶](#where-the-bottleneck-is)

TorchGeo's `_merge_or_stack` with `time_series=True`:

```
dest = np.stack([rasterio.merge.merge([fh], **kwargs)[0] for fh in vrt_fhs])
```

Each `fh` is a `WarpedVRT` wrapping a `rasterio.open("/vsicurl/...")`. For cloud COGs, each `rasterio.open()` triggers HTTP HEAD + 1-3 range requests for IFD headers, **all sequential, no concurrency**.

For T=15 timesteps × ~3 HTTP requests each = **45 round trips at ~100ms = 4.5s of pure header overhead**, before any pixel data flows.

Rasteret pre-caches all IFD metadata in the GeoParquet index, then fires T × C `read_cog()` calls via `asyncio.gather`, all concurrent.

## When to use which[¶](#when-to-use-which)

| Scenario                              | Recommendation                                                                              |
| ------------------------------------- | ------------------------------------------------------------------------------------------- |
| Cloud-hosted tiled GeoTIFFs (COGs)    | **Rasteret** for repeated collection reads; see measured results above                      |
| Local tiled GeoTIFFs                  | Rasteret works; speedup is smaller, but the index is still useful for filtering and sharing |
| Non-tiled GeoTIFFs (striped layout)   | TorchGeo / rasterio                                                                         |
| Non-TIFF formats (NetCDF, HDF5, GRIB) | TorchGeo / rasterio                                                                         |

Rasteret does not replace TorchGeo - it accelerates the data loading underneath. For the full ecosystem picture, see [Ecosystem Comparison](https://terrafloww.github.io/rasteret/explanation/interop/).

______________________________________________________________________

## Section 2: Multi-AOI Scaling[¶](#section-2-multi-aoi-scaling)

The single-AOI comparison above uses 1 region and 15 scenes. Real training pipelines cover **multiple regions** across a full year of imagery.

This section tests: **does the speedup hold (or grow) when we scale up?**

- 5 AOIs across southern India (~180 km spread)
- Full-year date range → 30 scenes
- Larger batch (4 chips × 16 samples)
- CRS auto-detected from the data (no hardcoded EPSG)

Both paths use the same `RandomGeoSampler`, no `roi` constraint, so the sampler weights by scene area and draws chips from anywhere in the index.

In \[ \]:

Copied!

```
# --- Multi-AOI setup ---
from shapely.ops import unary_union

# 5 AOIs across Bangalore metro (~50 km spread).
# All within the same Sentinel-2 tile footprint so every chip
# overlaps the same set of scenes (consistent T dimension).
aois = [
    Polygon(
        [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)]
    ),  # Bangalore center
    Polygon(
        [(77.65, 12.95), (77.68, 12.95), (77.68, 13.02), (77.65, 13.02), (77.65, 12.95)]
    ),  # Whitefield (east)
    Polygon(
        [(77.50, 13.15), (77.53, 13.15), (77.53, 13.22), (77.50, 13.22), (77.50, 13.15)]
    ),  # Yelahanka (north)
    Polygon(
        [(77.58, 12.85), (77.61, 12.85), (77.61, 12.92), (77.58, 12.92), (77.58, 12.85)]
    ),  # Electronics City (south)
    Polygon(
        [(77.40, 12.96), (77.43, 12.96), (77.43, 13.03), (77.40, 13.03), (77.40, 12.96)]
    ),  # Kengeri (west)
]
covering_bbox = unary_union(aois).bounds

DATE_RANGE_MULTI = "2024-01-01/2024-06-30"
MAX_SCENES_MULTI = 30
N_SAMPLES_MULTI = 16
BATCH_SIZE_MULTI = 4

# Shared STAC search, same scenes for both paths
search_multi = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=covering_bbox,
    datetime=DATE_RANGE_MULTI,
    max_items=MAX_SCENES_MULTI,
)
items_multi = list(search_multi.items())
scene_urls_multi = [
    item.assets[STAC_BAND].href for item in items_multi if STAC_BAND in item.assets
]
scene_dates_multi = sorted({item.datetime.isoformat()[:10] for item in items_multi})

print(f"Multi-AOI STAC search: {len(scene_urls_multi)} scenes")
print(f"Date range: {DATE_RANGE_MULTI}")
print(f"Actual scene dates: {scene_dates_multi[0]} → {scene_dates_multi[-1]}")
print(f"Covering bbox: {covering_bbox}")
print(f"AOIs: {len(aois)} regions across Bangalore metro")
```

# --- Multi-AOI setup ---

from shapely.ops import unary_union

# 5 AOIs across Bangalore metro (~50 km spread).

# All within the same Sentinel-2 tile footprint so every chip

# overlaps the same set of scenes (consistent T dimension).

aois = \[ Polygon( [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)] ), # Bangalore center Polygon( [(77.65, 12.95), (77.68, 12.95), (77.68, 13.02), (77.65, 13.02), (77.65, 12.95)] ), # Whitefield (east) Polygon( [(77.50, 13.15), (77.53, 13.15), (77.53, 13.22), (77.50, 13.22), (77.50, 13.15)] ), # Yelahanka (north) Polygon( [(77.58, 12.85), (77.61, 12.85), (77.61, 12.92), (77.58, 12.92), (77.58, 12.85)] ), # Electronics City (south) Polygon( [(77.40, 12.96), (77.43, 12.96), (77.43, 13.03), (77.40, 13.03), (77.40, 12.96)] ), # Kengeri (west) \] covering_bbox = unary_union(aois).bounds DATE_RANGE_MULTI = "2024-01-01/2024-06-30" MAX_SCENES_MULTI = 30 N_SAMPLES_MULTI = 16 BATCH_SIZE_MULTI = 4

# Shared STAC search, same scenes for both paths

search_multi = catalog.search( collections=["sentinel-2-l2a"], bbox=covering_bbox, datetime=DATE_RANGE_MULTI, max_items=MAX_SCENES_MULTI, ) items_multi = list(search_multi.items()) scene_urls_multi = \[ item.assets[STAC_BAND].href for item in items_multi if STAC_BAND in item.assets \] scene_dates_multi = sorted({item.datetime.isoformat()[:10] for item in items_multi}) print(f"Multi-AOI STAC search: {len(scene_urls_multi)} scenes") print(f"Date range: {DATE_RANGE_MULTI}") print(f"Actual scene dates: {scene_dates_multi[0]} → {scene_dates_multi[-1]}") print(f"Covering bbox: {covering_bbox}") print(f"AOIs: {len(aois)} regions across Bangalore metro")

In \[ \]:

Copied!

```
# --- Path A: TorchGeo (multi-AOI) ---
# CRS auto-detected from first file (no hardcoded EPSG).
vsicurl_multi = [f"/vsicurl/{url}" for url in scene_urls_multi]

t0 = time.perf_counter()
torchgeo_multi = RasterDataset(
    paths=vsicurl_multi,
    crs=None,  # auto-detect CRS from first scene
    res=10,
    time_series=True,
)
t_index_multi_a = time.perf_counter() - t0

print(f"CRS (auto-detected): {torchgeo_multi.crs}")
print(
    f"Index build: {t_index_multi_a:.2f}s ({len(vsicurl_multi)} rasterio.open() calls)"
)

# RandomGeoSampler, no roi, samples across all scenes weighted by area
sampler_multi_a = RandomGeoSampler(
    torchgeo_multi, size=CHIP_SIZE, length=N_SAMPLES_MULTI
)
loader_multi_a = DataLoader(
    torchgeo_multi,
    sampler=sampler_multi_a,
    batch_size=BATCH_SIZE_MULTI,
    num_workers=0,
    collate_fn=stack_samples,
)

t0 = time.perf_counter()
batch_multi_a = next(iter(loader_multi_a))
t_read_multi_a = time.perf_counter() - t0

print(f"Batch read: {t_read_multi_a:.2f}s for batch of {BATCH_SIZE_MULTI} chips")
print(f"image shape: {batch_multi_a['image'].shape}")
print("\n--- Multi-AOI Path A totals ---")
print(f"  Index:  {t_index_multi_a:.2f}s")
print(f"  Read:   {t_read_multi_a:.2f}s")
print(f"  Total:  {t_index_multi_a + t_read_multi_a:.2f}s")
```

# --- Path A: TorchGeo (multi-AOI) ---

# CRS auto-detected from first file (no hardcoded EPSG).

vsicurl_multi = [f"/vsicurl/{url}" for url in scene_urls_multi] t0 = time.perf_counter() torchgeo_multi = RasterDataset( paths=vsicurl_multi, crs=None, # auto-detect CRS from first scene res=10, time_series=True, ) t_index_multi_a = time.perf_counter() - t0 print(f"CRS (auto-detected): {torchgeo_multi.crs}") print( f"Index build: {t_index_multi_a:.2f}s ({len(vsicurl_multi)} rasterio.open() calls)" )

# RandomGeoSampler, no roi, samples across all scenes weighted by area

sampler_multi_a = RandomGeoSampler( torchgeo_multi, size=CHIP_SIZE, length=N_SAMPLES_MULTI ) loader_multi_a = DataLoader( torchgeo_multi, sampler=sampler_multi_a, batch_size=BATCH_SIZE_MULTI, num_workers=0, collate_fn=stack_samples, ) t0 = time.perf_counter() batch_multi_a = next(iter(loader_multi_a)) t_read_multi_a = time.perf_counter() - t0 print(f"Batch read: {t_read_multi_a:.2f}s for batch of {BATCH_SIZE_MULTI} chips") print(f"image shape: {batch_multi_a['image'].shape}") print("\\n--- Multi-AOI Path A totals ---") print(f" Index: {t_index_multi_a:.2f}s") print(f" Read: {t_read_multi_a:.2f}s") print(f" Total: {t_index_multi_a + t_read_multi_a:.2f}s")

In \[ \]:

Copied!

```
# --- Path B: Rasteret (multi-AOI) ---
# CRS derived from collection metadata (proj:epsg).
# geometries= accepts an array, each AOI is reprojected internally.

t0 = time.perf_counter()
collection_multi = rasteret.build_from_stac(
    name="southindiamulti",
    stac_api="https://earth-search.aws.element84.com/v1",
    collection=DataSources.SENTINEL2,
    bbox=covering_bbox,
    date_range=(scene_dates_multi[0], scene_dates_multi[-1]),
    workspace_dir=Path.home() / "rasteret_workspace",
)
t_stac_multi = time.perf_counter() - t0

t0 = time.perf_counter()
rasteret_multi = collection_multi.to_torchgeo_dataset(
    bands=[BAND],
    geometries=aois,  # all 5 AOIs as array
    chip_size=CHIP_SIZE,
    time_series=True,
)
t_index_multi_b = time.perf_counter() - t0

print(f"CRS (from collection): {rasteret_multi.crs}")
print(f"STAC ingest: {t_stac_multi:.2f}s (cached after first run)")
print(f"Index build: {t_index_multi_b:.2f}s (from Parquet, no HTTP)")
print(f"Scenes in dataset: {len(rasteret_multi.index)}")

sampler_multi_b = RandomGeoSampler(
    rasteret_multi, size=CHIP_SIZE, length=N_SAMPLES_MULTI
)
loader_multi_b = DataLoader(
    rasteret_multi,
    sampler=sampler_multi_b,
    batch_size=BATCH_SIZE_MULTI,
    num_workers=0,
    collate_fn=stack_samples,
)

t0 = time.perf_counter()
batch_multi_b = next(iter(loader_multi_b))
t_read_multi_b = time.perf_counter() - t0

print(f"Batch read:  {t_read_multi_b:.2f}s for batch of {BATCH_SIZE_MULTI} chips")
print(f"image shape: {batch_multi_b['image'].shape}")
print("\n--- Multi-AOI Path B totals ---")
print(f"  Index:  {t_index_multi_b:.2f}s")
print(f"  Read:   {t_read_multi_b:.2f}s")
print(f"  Total:  {t_index_multi_b + t_read_multi_b:.2f}s")
```

# --- Path B: Rasteret (multi-AOI) ---

# CRS derived from collection metadata (proj:epsg).

# geometries= accepts an array, each AOI is reprojected internally.

t0 = time.perf_counter() collection_multi = rasteret.build_from_stac( name="southindiamulti", stac_api="https://earth-search.aws.element84.com/v1", collection=DataSources.SENTINEL2, bbox=covering_bbox, date_range=(scene_dates_multi[0], scene_dates_multi[-1]), workspace_dir=Path.home() / "rasteret_workspace", ) t_stac_multi = time.perf_counter() - t0 t0 = time.perf_counter() rasteret_multi = collection_multi.to_torchgeo_dataset( bands=[BAND], geometries=aois, # all 5 AOIs as array chip_size=CHIP_SIZE, time_series=True, ) t_index_multi_b = time.perf_counter() - t0 print(f"CRS (from collection): {rasteret_multi.crs}") print(f"STAC ingest: {t_stac_multi:.2f}s (cached after first run)") print(f"Index build: {t_index_multi_b:.2f}s (from Parquet, no HTTP)") print(f"Scenes in dataset: {len(rasteret_multi.index)}") sampler_multi_b = RandomGeoSampler( rasteret_multi, size=CHIP_SIZE, length=N_SAMPLES_MULTI ) loader_multi_b = DataLoader( rasteret_multi, sampler=sampler_multi_b, batch_size=BATCH_SIZE_MULTI, num_workers=0, collate_fn=stack_samples, ) t0 = time.perf_counter() batch_multi_b = next(iter(loader_multi_b)) t_read_multi_b = time.perf_counter() - t0 print(f"Batch read: {t_read_multi_b:.2f}s for batch of {BATCH_SIZE_MULTI} chips") print(f"image shape: {batch_multi_b['image'].shape}") print("\\n--- Multi-AOI Path B totals ---") print(f" Index: {t_index_multi_b:.2f}s") print(f" Read: {t_read_multi_b:.2f}s") print(f" Total: {t_index_multi_b + t_read_multi_b:.2f}s")

In \[ \]:

Copied!

```
# --- Multi-AOI timing summary ---
print("=" * 68)
print(f"{'Multi-AOI (5 regions)':24} {'TorchGeo':>12} {'Rasteret':>12} {'Speedup':>10}")
print("-" * 68)
print(
    f"{'Index/header':24} {t_index_multi_a:>11.2f}s {t_index_multi_b:>11.2f}s {'':>10}"
)
print(f"{'Batch read':24} {t_read_multi_a:>11.2f}s {t_read_multi_b:>11.2f}s {'':>10}")
total_multi_a = t_index_multi_a + t_read_multi_a
total_multi_b = t_index_multi_b + t_read_multi_b
print(f"{'Total':24} {total_multi_a:>11.2f}s {total_multi_b:>11.2f}s {'':>10}")
print(
    f"{'vs TorchGeo':24} {'1.0x':>12} {total_multi_a / max(total_multi_b, 0.001):>11.1f}x {'':>10}"
)
print("=" * 68)
print("\nControlled variables:")
print(f"  AOIs: {len(aois)}, Scenes: {len(scene_urls_multi)}, Band: {BAND}")
print(
    f"  Chip: {CHIP_SIZE}x{CHIP_SIZE}, Batch: {BATCH_SIZE_MULTI}, Samples: {N_SAMPLES_MULTI}"
)
print(f"\nPath A shape: {batch_multi_a['image'].shape}")
print(f"Path B shape: {batch_multi_b['image'].shape}")

section2_speedup = total_multi_a / max(total_multi_b, 0.001)
```

# --- Multi-AOI timing summary ---

print("=" * 68) print(f"{'Multi-AOI (5 regions)':24} {'TorchGeo':>12} {'Rasteret':>12} {'Speedup':>10}") print("-" * 68) print( f"{'Index/header':24} {t_index_multi_a:>11.2f}s {t_index_multi_b:>11.2f}s {'':>10}" ) print(f"{'Batch read':24} {t_read_multi_a:>11.2f}s {t_read_multi_b:>11.2f}s {'':>10}") total_multi_a = t_index_multi_a + t_read_multi_a total_multi_b = t_index_multi_b + t_read_multi_b print(f"{'Total':24} {total_multi_a:>11.2f}s {total_multi_b:>11.2f}s {'':>10}") print( f"{'vs TorchGeo':24} {'1.0x':>12} {total_multi_a / max(total_multi_b, 0.001):>11.1f}x {'':>10}" ) print("=" * 68) print("\\nControlled variables:") print(f" AOIs: {len(aois)}, Scenes: {len(scene_urls_multi)}, Band: {BAND}") print( f" Chip: {CHIP_SIZE}x{CHIP_SIZE}, Batch: {BATCH_SIZE_MULTI}, Samples: {N_SAMPLES_MULTI}" ) print(f"\\nPath A shape: {batch_multi_a['image'].shape}") print(f"Path B shape: {batch_multi_b['image'].shape}") section2_speedup = total_multi_a / max(total_multi_b, 0.001)

### Multi-AOI takeaways[¶](#multi-aoi-takeaways)

- **CRS auto-detection**: TorchGeo infers CRS from the first file (`crs=None`). Rasteret derives it from the collection's `proj:epsg` metadata. Both expose the result via `dataset.crs`: standard TorchGeo interop.
- **Geometries as array**: Rasteret's `geometries=[aoi1, aoi2, ...]` accepts multiple polygons in WGS84 (or any CRS via `geometries_crs=`). Internally each polygon is reprojected to the dataset's native CRS and unioned for spatial filtering. TorchGeo's `roi=` only accepts a single polygon.
- **Scaling**: As T (timesteps) and the number of scenes grow, TorchGeo's sequential `rasterio.open()` + `rasterio.merge()` loop scales linearly. Rasteret's `asyncio.gather` fires all reads concurrently, bounded only by network bandwidth and `max_concurrent`.
- **AOI-only sampling**: `geometries=[...]` filters which records/tiles are included in the dataset, but samplers still sample over the dataset index bounds. To restrict chips to an AOI (for example a county polygon), pass `roi=<AOI polygon in dataset CRS>` to `GridGeoSampler` / `RandomGeoSampler`.

______________________________________________________________________

## Section 3: Cross-CRS Boundary (Multi-Zone Reprojection)[¶](#section-3-cross-crs-boundary-multi-zone-reprojection)

Sections 1-2 stayed within a single UTM zone (EPSG:32643). Real workflows often span **UTM zone boundaries**, the 78°E meridian separates zone 43N from 44N, and Sentinel-2 tiles from each zone use different CRS.

This section places an AOI in the **overlap zone** east of Hyderabad where tiles `43QHV` (EPSG:32643) and `44QKE` (EPSG:32644) both provide coverage.

- **TorchGeo**: Uses `WarpedVRT` to reproject each file to a common CRS on read
- **Rasteret**: Uses `target_crs=32643` to keep all scenes and reprojects via `rasterio.warp.reproject()` after its concurrent fetch

Both paths end at the same `[batch, T, C, H, W]` tensor in EPSG:32643.

In \[ \]:

Copied!

```
# --- Cross-CRS setup ---
from pyproj import Transformer
from shapely.ops import transform

# AOI in the overlap zone where tiles 43QHV (EPSG:32643) and 44QKE (EPSG:32644)
# both have coverage.  ~20 km × ~10 km east of Hyderabad.
aoi_boundary = Polygon(
    [
        (78.25, 17.30),
        (78.45, 17.30),
        (78.45, 17.40),
        (78.25, 17.40),
        (78.25, 17.30),
    ]
)

DATE_RANGE_CRS = "2024-04-01/2024-04-30"
N_SAMPLES_CRS = 8
BATCH_SIZE_CRS = 2
TARGET_CRS = 32643  # reproject zone 44N scenes into zone 43N

# Shared STAC search, same scenes for both paths
search_crs = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=(78.2, 17.25, 78.5, 17.45),
    datetime=DATE_RANGE_CRS,
    max_items=50,
)
items_crs = list(search_crs.items())
scene_urls_crs = [
    item.assets[STAC_BAND].href for item in items_crs if STAC_BAND in item.assets
]
scene_dates_crs = sorted({item.datetime.isoformat()[:10] for item in items_crs})

# Show CRS distribution
from collections import Counter

crs_dist = Counter(item.properties.get("proj:code", "?") for item in items_crs)

# ROI in dataset CRS for sampler, restrict to overlap zone so every chip
# sees scenes from BOTH CRS zones (consistent T dimension for stack_samples)
proj = Transformer.from_crs(4326, TARGET_CRS, always_xy=True)
roi_overlap = transform(proj.transform, aoi_boundary)

print(f"Cross-CRS STAC search: {len(scene_urls_crs)} scenes")
print(f"CRS distribution: {dict(crs_dist)}")
print(f"Date range: {DATE_RANGE_CRS}")
print(f"Overlap AOI: {aoi_boundary.bounds}")
print(f"Target CRS: EPSG:{TARGET_CRS}")
```

# --- Cross-CRS setup ---

from pyproj import Transformer from shapely.ops import transform

# AOI in the overlap zone where tiles 43QHV (EPSG:32643) and 44QKE (EPSG:32644)

# both have coverage. ~20 km × ~10 km east of Hyderabad.

aoi_boundary = Polygon( [ (78.25, 17.30), (78.45, 17.30), (78.45, 17.40), (78.25, 17.40), (78.25, 17.30), ] ) DATE_RANGE_CRS = "2024-04-01/2024-04-30" N_SAMPLES_CRS = 8 BATCH_SIZE_CRS = 2 TARGET_CRS = 32643 # reproject zone 44N scenes into zone 43N

# Shared STAC search, same scenes for both paths

search_crs = catalog.search( collections=["sentinel-2-l2a"], bbox=(78.2, 17.25, 78.5, 17.45), datetime=DATE_RANGE_CRS, max_items=50, ) items_crs = list(search_crs.items()) scene_urls_crs = \[ item.assets[STAC_BAND].href for item in items_crs if STAC_BAND in item.assets \] scene_dates_crs = sorted({item.datetime.isoformat()[:10] for item in items_crs})

# Show CRS distribution

from collections import Counter crs_dist = Counter(item.properties.get("proj:code", "?") for item in items_crs)

# ROI in dataset CRS for sampler, restrict to overlap zone so every chip

# sees scenes from BOTH CRS zones (consistent T dimension for stack_samples)

proj = Transformer.from_crs(4326, TARGET_CRS, always_xy=True) roi_overlap = transform(proj.transform, aoi_boundary) print(f"Cross-CRS STAC search: {len(scene_urls_crs)} scenes") print(f"CRS distribution: {dict(crs_dist)}") print(f"Date range: {DATE_RANGE_CRS}") print(f"Overlap AOI: {aoi_boundary.bounds}") print(f"Target CRS: EPSG:{TARGET_CRS}")

In \[ \]:

Copied!

```
# --- Path A: TorchGeo (cross-CRS) ---
# TorchGeo uses WarpedVRT to reproject zone 44N scenes into 32643 on read.
vsicurl_crs = [f"/vsicurl/{url}" for url in scene_urls_crs]

t0 = time.perf_counter()
torchgeo_crs = RasterDataset(
    paths=vsicurl_crs,
    crs="epsg:32643",  # target CRS, WarpedVRT reprojects zone 44N files
    res=10,
    time_series=True,
)
t_index_crs_a = time.perf_counter() - t0

print(f"CRS: {torchgeo_crs.crs}")
print(f"Index build: {t_index_crs_a:.2f}s ({len(vsicurl_crs)} scenes, 2 CRS zones)")

# ROI restricts sampling to the overlap zone (both tiles cover)
sampler_crs_a = RandomGeoSampler(
    torchgeo_crs,
    size=CHIP_SIZE,
    length=N_SAMPLES_CRS,
    roi=roi_overlap,
)
loader_crs_a = DataLoader(
    torchgeo_crs,
    sampler=sampler_crs_a,
    batch_size=BATCH_SIZE_CRS,
    num_workers=0,
    collate_fn=stack_samples,
)

t0 = time.perf_counter()
batch_crs_a = next(iter(loader_crs_a))
t_read_crs_a = time.perf_counter() - t0

print(f"Batch read: {t_read_crs_a:.2f}s")
print(f"image shape: {batch_crs_a['image'].shape}")
print("\n--- Cross-CRS Path A totals ---")
print(f"  Index:  {t_index_crs_a:.2f}s")
print(f"  Read:   {t_read_crs_a:.2f}s")
print(f"  Total:  {t_index_crs_a + t_read_crs_a:.2f}s")
```

# --- Path A: TorchGeo (cross-CRS) ---

# TorchGeo uses WarpedVRT to reproject zone 44N scenes into 32643 on read.

vsicurl_crs = [f"/vsicurl/{url}" for url in scene_urls_crs] t0 = time.perf_counter() torchgeo_crs = RasterDataset( paths=vsicurl_crs, crs="epsg:32643", # target CRS, WarpedVRT reprojects zone 44N files res=10, time_series=True, ) t_index_crs_a = time.perf_counter() - t0 print(f"CRS: {torchgeo_crs.crs}") print(f"Index build: {t_index_crs_a:.2f}s ({len(vsicurl_crs)} scenes, 2 CRS zones)")

# ROI restricts sampling to the overlap zone (both tiles cover)

sampler_crs_a = RandomGeoSampler( torchgeo_crs, size=CHIP_SIZE, length=N_SAMPLES_CRS, roi=roi_overlap, ) loader_crs_a = DataLoader( torchgeo_crs, sampler=sampler_crs_a, batch_size=BATCH_SIZE_CRS, num_workers=0, collate_fn=stack_samples, ) t0 = time.perf_counter() batch_crs_a = next(iter(loader_crs_a)) t_read_crs_a = time.perf_counter() - t0 print(f"Batch read: {t_read_crs_a:.2f}s") print(f"image shape: {batch_crs_a['image'].shape}") print("\\n--- Cross-CRS Path A totals ---") print(f" Index: {t_index_crs_a:.2f}s") print(f" Read: {t_read_crs_a:.2f}s") print(f" Total: {t_index_crs_a + t_read_crs_a:.2f}s")

In \[ \]:

Copied!

```
# --- Path B: Rasteret (cross-CRS) ---
# target_crs=32643 keeps scenes from BOTH zones and reprojects at read time
# via rasterio.warp.reproject() after the concurrent fetch.

t0 = time.perf_counter()
collection_crs = rasteret.build_from_stac(
    name="hydboundary",
    stac_api="https://earth-search.aws.element84.com/v1",
    collection=DataSources.SENTINEL2,
    bbox=(78.2, 17.25, 78.5, 17.45),
    date_range=(scene_dates_crs[0], scene_dates_crs[-1]),
    workspace_dir=Path.home() / "rasteret_workspace",
)
t_stac_crs = time.perf_counter() - t0

t0 = time.perf_counter()
rasteret_crs = collection_crs.to_torchgeo_dataset(
    bands=[BAND],
    geometries=[aoi_boundary],
    chip_size=CHIP_SIZE,
    time_series=True,
    target_crs=TARGET_CRS,  # keep all CRS zones, reproject to 32643
)
t_index_crs_b = time.perf_counter() - t0

# Show CRS mix in the index (column presence can vary by adapter version)
zone_col = (
    "proj:epsg" if "proj:epsg" in getattr(rasteret_crs.index, "columns", []) else None
)
zone_counts = rasteret_crs.index[zone_col].value_counts().to_dict() if zone_col else {}
print(f"CRS (target): EPSG:{TARGET_CRS}")
print(f"Scenes in index: {len(rasteret_crs.index)} (zones: {zone_counts})")
print(f"STAC ingest: {t_stac_crs:.2f}s (cached)")
print(f"Index build: {t_index_crs_b:.2f}s")

sampler_crs_b = RandomGeoSampler(
    rasteret_crs,
    size=CHIP_SIZE,
    length=N_SAMPLES_CRS,
    roi=roi_overlap,
)
loader_crs_b = DataLoader(
    rasteret_crs,
    sampler=sampler_crs_b,
    batch_size=BATCH_SIZE_CRS,
    num_workers=0,
    collate_fn=stack_samples,
)

t0 = time.perf_counter()
batch_crs_b = next(iter(loader_crs_b))
t_read_crs_b = time.perf_counter() - t0

print(f"Batch read:  {t_read_crs_b:.2f}s")
print(f"image shape: {batch_crs_b['image'].shape}")
print("\n--- Cross-CRS Path B totals ---")
print(f"  Index:  {t_index_crs_b:.2f}s")
print(f"  Read:   {t_read_crs_b:.2f}s")
print(f"  Total:  {t_index_crs_b + t_read_crs_b:.2f}s")
```

# --- Path B: Rasteret (cross-CRS) ---

# target_crs=32643 keeps scenes from BOTH zones and reprojects at read time

# via rasterio.warp.reproject() after the concurrent fetch.

t0 = time.perf_counter() collection_crs = rasteret.build_from_stac( name="hydboundary", stac_api="https://earth-search.aws.element84.com/v1", collection=DataSources.SENTINEL2, bbox=(78.2, 17.25, 78.5, 17.45), date_range=(scene_dates_crs[0], scene_dates_crs[-1]), workspace_dir=Path.home() / "rasteret_workspace", ) t_stac_crs = time.perf_counter() - t0 t0 = time.perf_counter() rasteret_crs = collection_crs.to_torchgeo_dataset( bands=[BAND], geometries=[aoi_boundary], chip_size=CHIP_SIZE, time_series=True, target_crs=TARGET_CRS, # keep all CRS zones, reproject to 32643 ) t_index_crs_b = time.perf_counter() - t0

# Show CRS mix in the index (column presence can vary by adapter version)

zone_col = ( "proj:epsg" if "proj:epsg" in getattr(rasteret_crs.index, "columns", []) else None ) zone_counts = rasteret_crs.index[zone_col].value_counts().to_dict() if zone_col else {} print(f"CRS (target): EPSG:{TARGET_CRS}") print(f"Scenes in index: {len(rasteret_crs.index)} (zones: {zone_counts})") print(f"STAC ingest: {t_stac_crs:.2f}s (cached)") print(f"Index build: {t_index_crs_b:.2f}s") sampler_crs_b = RandomGeoSampler( rasteret_crs, size=CHIP_SIZE, length=N_SAMPLES_CRS, roi=roi_overlap, ) loader_crs_b = DataLoader( rasteret_crs, sampler=sampler_crs_b, batch_size=BATCH_SIZE_CRS, num_workers=0, collate_fn=stack_samples, ) t0 = time.perf_counter() batch_crs_b = next(iter(loader_crs_b)) t_read_crs_b = time.perf_counter() - t0 print(f"Batch read: {t_read_crs_b:.2f}s") print(f"image shape: {batch_crs_b['image'].shape}") print("\\n--- Cross-CRS Path B totals ---") print(f" Index: {t_index_crs_b:.2f}s") print(f" Read: {t_read_crs_b:.2f}s") print(f" Total: {t_index_crs_b + t_read_crs_b:.2f}s")

In \[ \]:

Copied!

```
# --- Cross-CRS timing summary ---
print("=" * 68)
print(f"{'Cross-CRS (2 zones)':24} {'TorchGeo':>12} {'Rasteret':>12} {'Speedup':>10}")
print("-" * 68)
print(f"{'Index/header':24} {t_index_crs_a:>11.2f}s {t_index_crs_b:>11.2f}s {'':>10}")
print(f"{'Batch read':24} {t_read_crs_a:>11.2f}s {t_read_crs_b:>11.2f}s {'':>10}")
total_crs_a = t_index_crs_a + t_read_crs_a
total_crs_b = t_index_crs_b + t_read_crs_b
print(f"{'Total':24} {total_crs_a:>11.2f}s {total_crs_b:>11.2f}s {'':>10}")
print(
    f"{'vs TorchGeo':24} {'1.0x':>12} {total_crs_a / max(total_crs_b, 0.001):>11.1f}x {'':>10}"
)
print("=" * 68)

print("\nCRS zones: EPSG:32643 (UTM 43N) + EPSG:32644 (UTM 44N)")
print(f"Scenes: {len(scene_urls_crs)} ({dict(crs_dist)})")
print(f"Target CRS: EPSG:{TARGET_CRS}")
print(f"\nPath A shape: {batch_crs_a['image'].shape}")
print(f"Path B shape: {batch_crs_b['image'].shape}")

# Compare across all three sections
section3_speedup = total_crs_a / max(total_crs_b, 0.001)
print("\n--- Speedup across sections (Rasteret vs TorchGeo) ---")
print(
    f"  Section 1 (1 AOI, 1 CRS, {len(scene_urls)} scenes):    {section1_speedup:.1f}x"
)
print(
    f"  Section 2 (5 AOIs, 1 CRS, {len(scene_urls_multi)} scenes):  {section2_speedup:.1f}x"
)
print(
    f"  Section 3 (1 AOI, 2 CRS, {len(scene_urls_crs)} scenes):  {section3_speedup:.1f}x"
)
```

# --- Cross-CRS timing summary ---

print("=" * 68) print(f"{'Cross-CRS (2 zones)':24} {'TorchGeo':>12} {'Rasteret':>12} {'Speedup':>10}") print("-" * 68) print(f"{'Index/header':24} {t_index_crs_a:>11.2f}s {t_index_crs_b:>11.2f}s {'':>10}") print(f"{'Batch read':24} {t_read_crs_a:>11.2f}s {t_read_crs_b:>11.2f}s {'':>10}") total_crs_a = t_index_crs_a + t_read_crs_a total_crs_b = t_index_crs_b + t_read_crs_b print(f"{'Total':24} {total_crs_a:>11.2f}s {total_crs_b:>11.2f}s {'':>10}") print( f"{'vs TorchGeo':24} {'1.0x':>12} {total_crs_a / max(total_crs_b, 0.001):>11.1f}x {'':>10}" ) print("=" * 68) print("\\nCRS zones: EPSG:32643 (UTM 43N) + EPSG:32644 (UTM 44N)") print(f"Scenes: {len(scene_urls_crs)} ({dict(crs_dist)})") print(f"Target CRS: EPSG:{TARGET_CRS}") print(f"\\nPath A shape: {batch_crs_a['image'].shape}") print(f"Path B shape: {batch_crs_b['image'].shape}")

# Compare across all three sections

section3_speedup = total_crs_a / max(total_crs_b, 0.001) print("\\n--- Speedup across sections (Rasteret vs TorchGeo) ---") print( f" Section 1 (1 AOI, 1 CRS, {len(scene_urls)} scenes): {section1_speedup:.1f}x" ) print( f" Section 2 (5 AOIs, 1 CRS, {len(scene_urls_multi)} scenes): {section2_speedup:.1f}x" ) print( f" Section 3 (1 AOI, 2 CRS, {len(scene_urls_crs)} scenes): {section3_speedup:.1f}x" )
