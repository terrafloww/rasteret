# Turbocharging TorchGeo

If you are using **TorchGeo** for deep learning on geospatial data, you know that the I/O bottleneck is often the slowest part of your training loop. Traditionally, TorchGeo uses `rasterio` and GDAL to fetch pixels, which can be slow on cloud-hosted COGs due to repeated header parsing and sequential reads.

**Rasteret provides a "drop-in" high-performance I/O backend for TorchGeo.** You keep your samplers, transforms, and training logic—you just swap the dataset reader for a **20x speedup** in time-to-first-pixel.

## The "Drop-In" Performance Swap

### The Standard TorchGeo way
TorchGeo’s `RasterDataset` is versatile but reads every scene from scratch using GDAL.

```python
from torchgeo.datasets import RasterDataset

# Uses rasterio/GDAL backend (Sequential, Cold-Start heavy)
dataset = RasterDataset(root="s3://my-bucket/cogs")
```

### The Rasteret way
`RasteretGeoDataset` honors the exact same `GeoDataset` contract but uses the Rasteret async I/O engine.

```python
import rasteret

collection = rasteret.load("my_experiment")

# Drop-in replacement for any TorchGeo sampler
dataset = collection.to_torchgeo_dataset(
    bands=["B08", "B04", "B03"],
    chip_size=256
)
# Now use standard TorchGeo samplers (RandomGeoSampler, etc.)
```

## Why swap the backend?

### 1. Same API, Higher Throughput
Because `RasteretGeoDataset` is a first-class subclass of TorchGeo's `GeoDataset`, it works out-of-the-box with all TorchGeo samplers. You gain performance without rewriting your training logic.

### 2. Eliminating GDAL Cold-Starts
DataLoader workers in PyTorch are often spawned fresh every epoch. In a standard GDAL setup, each worker must re-negotiate the HTTP connection and re-parse the COG headers for every scene.

**Rasteret workers use the local Parquet index.** They skip the "negotiation" phase and jump straight to the pixels. This creates a **multiplier effect** across multiple workers, often leading to **10x - 20x speedups** in overall training throughput.

### 3. Integrated Labels (Relational Training)
One of the biggest frictions in TorchGeo is aligning tabular labels (like a CSV of tree heights) with raster shapes. Rasteret solves this at the table level.

```python
# Pass the label column name from your collection table
dataset = collection.to_torchgeo_dataset(
    bands=["B04"],
    label_field="biomass_value"
)

# Every sample contains the label automatically!
# sample = {"image": Tensor, "label": Tensor, ...}
```

## When to stick with standard TorchGeo?

While Rasteret is faster for cloud COGs, you should continue using TorchGeo's native `RasterDataset` if:
1.  **Non-TIFF Formats**: You are reading NetCDF, HDF5, or GRIB files.
2.  **Striped TIFFs**: Your files are not internally tiled (COGs).
3.  **Local Development**: You have a small folder of 5-10 files where the indexing overhead isn't worth it.

## Quick Benchmark

| Setup | Load Time (32 scenes) |
|---|---|
| TorchGeo + Rasterio/GDAL | ~42.0s |
| **TorchGeo + Rasteret** | **~2.2s** |

*Measured on AWS us-west-2 using Sentinel-2 L2A data.*
