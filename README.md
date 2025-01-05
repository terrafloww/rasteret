# 🛰️ Rasteret

Fast and efficient access to Cloud-Optimized GeoTIFFs (COGs), optimized for Sentinel-2 and Landsat data.

> [!WARNING]  
> Work-in-progress library. The APIs are subject to change, and as such, documentation is not yet available.

## 🚀 Features
- Fast byte-range based COG access
- STAC Geoparquet creation with extra metadata columns 
- Paid public data support (AWS S3 Landsat)
- Xarray and GeoDataFrame outputs
- Parallel data loading
- Simple high-level API

## 📋 Prerequisites
- Python ≥3.10
- AWS credentials (for accessing paid AWS buckets)

### ⚙️ AWS Credentials Setup
For accessing paid AWS buckets:
```bash
export AWS_ACCESS_KEY_ID='your_access_key'
export AWS_SECRET_ACCESS_KEY='your_secret_key'
```

## 📦 Installation
```bash
pip install rasteret
```

## 🏃‍♂️ Quick Start

```python
from pathlib import Path
from shapely.geometry import Polygon
from rasteret import Rasteret

# 1. Define areas and time of interest
aoi1_polygon = Polygon([
    (77.55, 13.01),
    (77.58, 13.01),
    (77.58, 13.08),
    (77.55, 13.08),
    (77.55, 13.01)
])

aoi2_polygon = Polygon([
    (77.56, 13.02),
    (77.59, 13.02),
    (77.59, 13.09),
    (77.56, 13.09),
    (77.56, 13.02)
])

#get total bounds of all polygons above
bbox = aoi1_polygon.union(aoi2_polygon).bounds

date_range = ("2024-01-01", "2024-01-31")

# 2. Setup workspace
workspace_dir = Path.home() / "rasteret_workspace"
workspace_dir.mkdir(exist_ok=True)

# 3. Initialize processor and create index
processor = Rasteret(
    data_source="landsat-c2l2-sr",  # or "sentinel-2-l2a"
    output_dir=workspace_dir,
    name="bangalore_jan2024_ls9"
)

# Create index with cloud cover filter
processor.create_index(
    bbox=bbox,
    date_range=date_range,
    cloud_cover_lt=90
)

# 4. Process data using xarray
ds = processor.query_xarray(
    geometries=[aoi1_polygon,aoi2_polygon],
    bands=["B4", "B5"],  # Red and NIR bands
    cloud_cover_lt=20
)

# 5. Calculate NDVI
ndvi_ds = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)

# 6. Save each date as geotiff file from xarray
output_files = save_per_geometry(ndvi_ds, output_dir, prefix="ndvi")

for geom_id, filepath in output_files.items():
    print(f"Geometry {geom_id}: {filepath}")

```

## 🌍 Supported Data Sources
- Sentinel-2 Level 2A
- Landsat Collection 2 Level 2 SR

## 📝 License
Apache 2.0 License

## 🤝 Contributing
Contributions welcome! Please read our contributing guidelines.

## ⚠️ Known Limitations
- High memory usage for large areas
- AWS requester-pays buckets need valid credentials
- Limited to Sentinel-2 and Landsat data formats
