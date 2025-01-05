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
from rasteret.constants import DataSources
from rasteret.core.utils import save_per_geometry

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

# get total bounds of all polygons above
bbox = aoi1_polygon.union(aoi2_polygon).bounds


# 2. Setup variables for Rasteret

# give your own custom name for STAC collection
custom_name = "bangalore"

# choose date range for which STAC Items are required
date_range = ("2024-01-01", "2024-01-31")

# choose either LANDSAT or SENTINEL2 from DataSources
data_source = DataSources.LANDSAT

# choose a workspace directory where all STAC related data will be saved
workspace_dir = Path.home() / "rasteret_workspace"
workspace_dir.mkdir(exist_ok=True)

# 3. Initialize Rasteret
processor = Rasteret(
    data_source=data_source
    output_dir=workspace_dir,
    name=custom_name,
    date_range=date_range
)

# Create a local geoparquet based cache that contains COG metadata columns
# and all STAC columns too. pystac_client filters can be provided here as well
# such as cloud_cover, platform name etc.
processor.create_index(
    bbox=bbox,
    date_range=date_range,
    cloud_cover_lt=90, # Filter scenes with < 90% cloud cover
    platform={"in": ["LANDSAT_8"]} 
    )

# 4. Query for bands in scenes that pass the filters on local geoparquet
# and return images as xarray dataset
ds = processor.get_xarray(
    # passing multiple geometries will return xarray dataset with multiple dimensions
    # and each geometry/AOIs respective timeseries
    geometries=[aoi1_polygon,aoi2_polygon],
    bands=["B4", "B5"],  # Choose multiple bands if required

    # extra filters 
    cloud_cover_lt=20,  # Filter scenes with < 20% cloud cover
    date_range=["2024-01-01", "2024-01-31"],  # Date range filter [start date , end date]
    bbox=bbox,  # Spatial filter using a bbox is possible
    )
)

# 5. Calculate NDVI
ndvi_ds = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)

# 6. Save each date as geotiff file from xarray
# each geometry gets a separate folder if there are multiple geometries
output_files = save_per_geometry(ndvi_ds, output_dir, prefix="ndvi")

for geom_id, filepath in output_files.items():
    print(f"Geometry {geom_id}: {filepath}")

```

## Why this library?

Details on why this library was made, and how it reads multiple COGs fast and efficient -
[Read the blog post here](https://blog.terrafloww.com/efficient-cloud-native-raster-data-access-an-alternative-to-rasterio-gdal/)

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
