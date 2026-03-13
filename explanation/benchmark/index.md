# Benchmarks

Apples-to-apples time-series comparison using `docs/tutorials/05_torchgeo_comparison.ipynb`.

**Environment:** Ubuntu Linux, Python 3.13, us-west-2 EC2 instance.

**Data source:** Sentinel-2 L2A via Earth Search v1 (COGs on S3 us-west-2).

**Controlled variables:** Same scenes, same sampler, same DataLoader, same `stack_samples` collate. Both paths output identical `[batch, T, C, H, W]` tensors.

DataLoader uses default settings (no custom `num_workers`, `prefetch_factor`, or `persistent_workers`) for both paths.

TorchGeo runs with its recommended GDAL settings (`rasterio_best_practices` from Pangeo COG best practices) for best-case remote COG performance:

```python
GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR"
AWS_NO_SIGN_REQUEST = "YES"
GDAL_MAX_RAW_BLOCK_CACHE_SIZE = "200000000"  # 200 MB
GDAL_SWATH_SIZE = "200000000"
VSI_CURL_CACHE_SIZE = "200000000"
```

This gives TorchGeo its best-case scenario for remote COG reads.

### What "Index/header" time measures

- **TorchGeo**: Time spent in `rasterio.open("/vsicurl/...")` calls. Each call triggers HTTP HEAD + 1-3 range requests to parse the IFD (Image File Directory) that contains tile offsets and byte counts. For T timesteps with ~3 HTTP round-trips each at ~100 ms, this adds up to seconds of pure header overhead before any pixel data flows.
- **Rasteret**: Time to read the pre-built Parquet index from local disk. Tile offsets and byte counts are already cached, so no network I/O is needed.

Both are measuring the same logical step: making tile layout metadata available before pixel reads begin.

## Backends tested

| Backend      | Transport                                                               | Install                   |
| ------------ | ----------------------------------------------------------------------- | ------------------------- |
| **TorchGeo** | GDAL `/vsicurl/` (sequential)                                           | `uv pip install torchgeo` |
| **Rasteret** | Custom async IO engine (concurrent byte-range reads, obstore transport) | `uv pip install rasteret` |

## Cold start (first run)

| Section                  | TorchGeo | Rasteret | Speedup   | Shape                  |
| ------------------------ | -------- | -------- | --------- | ---------------------- |
| 1: Single AOI, 15 scenes | 9.08 s   | 1.14 s   | **8.0x**  | `[2, 15, 1, 256, 256]` |
| 2: Multi-AOI, 30 scenes  | 42.05 s  | 2.25 s   | **18.7x** | `[4, 30, 1, 256, 256]` |
| 3: Cross-CRS, 12 scenes  | 12.47 s  | 0.59 s   | **21.3x** | `[2, 12, 1, 256, 256]` |

## Warm cache (immediate re-run)

| Section                  | TorchGeo | Rasteret | Speedup   | Shape                  |
| ------------------------ | -------- | -------- | --------- | ---------------------- |
| 1: Single AOI, 15 scenes | 9.14 s   | 0.81 s   | **11.3x** | `[2, 15, 1, 256, 256]` |
| 2: Multi-AOI, 30 scenes  | 29.68 s  | 2.60 s   | **11.4x** | `[4, 30, 1, 256, 256]` |
| 3: Cross-CRS, 12 scenes  | 3.61 s   | 1.06 s   | **3.4x**  | `[2, 12, 1, 256, 256]` |

## Key observations

1. The difference grows with scene count: the rasterio/GDAL path re-parses headers over HTTP per file (sequential), while Rasteret reads cached headers from disk and fetches pixels concurrently.
1. Cross-CRS adds reprojection overhead (~0.1-0.3 s) to both paths.

## HF `datasets` baseline (Major TOM keyed patches)

Separate benchmark against Hugging Face payload-Parquet workflows using `datasets.load_dataset(..., streaming=True, filters=...)` (PyArrow-backed predicate pushdown):

| Patches | HF `datasets` parquet filters | Rasteret index+COG | Speedup   |
| ------- | ----------------------------- | ------------------ | --------- |
| 120     | 46.83 s                       | 12.09 s            | **3.88x** |
| 1000    | 771.59 s                      | 118.69 s           | **6.50x** |

Major TOM notebooks often use HF streaming generators for exploration; the table above uses `filters=...` keyed retrieval for fairness.

## Where the difference comes from

|                       | rasterio/GDAL path                         | Rasteret path                        |
| --------------------- | ------------------------------------------ | ------------------------------------ |
| **Index**             | `rasterio.open()` per COG over HTTP        | Pre-built GeoParquet (disk read)     |
| **Time-series read**  | Sequential `rasterio.merge()` per timestep | All T timesteps via `asyncio.gather` |
| **HTTP per timestep** | HEAD + IFD + pixel ranges                  | Pixel ranges only (headers cached)   |
| **Concurrency**       | Sequential                                 | `asyncio.gather` across T x C reads  |

## Reproducibility

```bash
# Fresh run
uv run python -m nbconvert --execute docs/tutorials/05_torchgeo_comparison.ipynb

# Immediate re-run (warm cache)
uv run python -m nbconvert --execute docs/tutorials/05_torchgeo_comparison.ipynb
```

Results vary with network conditions and EC2 instance placement.

## Why cold-start numbers matter

The "cold start" row above is the number that matters most. Every new notebook kernel, every new VM, every Kubernetes pod restart, every CI runner, every colleague cloning your repo starts cold. There is no warm HTTP cache, no OS page cache, no pre-opened file handles.

TorchGeo re-parses every COG header over HTTP on each cold start. Rasteret reads the same headers from a local Parquet file in milliseconds. The gap widens with scene count: 30 scenes means 120+ HTTP round-trips that Rasteret skips entirely.

## Share your numbers

These benchmarks cover 12-30 Sentinel-2 scenes. The speedup grows with scene count, so larger workloads should widen the gap. If you're running Rasteret on bigger collections, different sensors, or production pipelines, share your timings on [GitHub Discussions](https://github.com/terrafloww/rasteret/discussions/categories/show-and-tell) or [Discord](https://discord.gg/V5vvuEBc) - it helps the community understand where the real ceiling is.
