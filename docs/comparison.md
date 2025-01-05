# Performance Comparison

## Overview

Rasteret has been benchmarked against traditional rasterio-based approaches, showing significant performance improvements for Sentinel-2 data access and processing.

## Benchmark Results

### Time Series Analysis Test Case
Processing a year of Sentinel-2 data (20 scenes) for NDVI calculation:

| Implementation | Processing Time | Memory Usage |
|----------------|----------------|--------------|
| Rasteret | ~8 seconds | Optimized |
| Rasterio Direct | ~24 seconds | Higher |

### Key Differences

1. **Data Access Strategy**
   - Rasteret: Optimized tile-based access with parallel processing
   - Rasterio: Full window reads with masking

2. **Memory Management**
   - Rasteret: Efficient tile caching and memory usage
   - Rasterio: Larger memory footprint due to full window reads

3. **Processing Pipeline**
   - Rasteret: Integrated STAC metadata and optimized band access
   - Rasterio: Direct band access and processing

## Implementation Comparison

Both implementations produce identical NDVI values, validating the accuracy of Rasteret's approach while delivering superior performance.

## Detailed Performance Analysis

### Time Series Analysis (20 scenes)

#### Rasteret
- STAC filtering: ~0.5s
- Data processing: ~7.4s
- Total time: ~8.0s
- Average per scene: ~0.4s
- Memory usage: Optimized (~200MB)

#### Traditional Rasterio
- STAC search: ~2s
- Data processing: ~23.1s
- Total time: ~24.6s (with GDAL configs set for ideal performance)
- Total time: ~44.6s (without GDAL configs set)
- Average per scene: ~1.2s to ~2.2s
- Memory usage: Higher (~500MB)

### Performance Breakdown

1. **Metadata Phase**
   - Rasteret's one-time metadata creation enables faster subsequent queries
   - STAC filtering is 3x faster with pre-cached metadata

2. **Data Processing**
   - Rasteret: ~0.4s per scene
   - Rasterio: ~1.2s per scene
   - 3x improvement in per-scene processing