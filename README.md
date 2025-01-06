# 🛰️ Rasteret

Fast and efficient access to Cloud-Optimized GeoTIFFs (COGs), optimized for Sentinel-2 and Landsat data.

> [!WARNING]  
> Work-in-progress library. The APIs are subject to change, and as such, documentation is not yet available.

## 🚀 Features
- Fast byte-range based COG access
- STAC Geoparquet creation with COG internal metadata columns 
- Paid public data support (AWS S3 Landsat)
- Xarray and GeoDataFrame outputs
- Parallel data loading
- Simple high-level API

## 📋 Prerequisites
- Python 3.10.x,3.11.x
- AWS credentials (for accessing paid AWS buckets)

### ⚙️ AWS Credentials Setup
For accessing paid AWS buckets:

(Prefferable) You can set up your AWS credentials by creating a `~/.aws/credentials` file with the following content:

```
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

Alternatively, you can set the credentials as environment variables:
```bash
export AWS_ACCESS_KEY_ID='your_access_key'
export AWS_SECRET_ACCESS_KEY='your_secret_key'
```

## 📦 Installation
```bash
pip install rasteret
```

## 🏃‍♂️ Quick Start

1. Define Areas of Interest

Create polygons for your regions of interest:

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
```

2. Configure Rasteret

Set up basic parameters for data collection, and check for existing collection
in your workspace directory, if they were created earlier.

```python
# Collection configuration
custom_name = "bangalore"
date_range = ("2024-01-01", "2024-12-31")
data_source = DataSources.LANDSAT

# Set up workspace
workspace_dir = Path.home() / "rasteret_workspace"
workspace_dir.mkdir(exist_ok=True)
)

# List existing collections
collections = Rasteret.list_collections()
for c in collections:
    print(f"- {c['name']}: {c['data_source']}, {c['date_range']}, {c['size']} scenes")

```
3. Initialize and Create Collection

Create or Load a local collection:
Containing internal COG metadata of scenes, and its STAC metadata

```python
# Try loading existing collection
try:
    # example name
    processor = Rasteret.load_collection("bangalore_202401-12_landsat")
except ValueError:
    # Create new collection
    processor = Rasteret(
        custom_name="bangalore",
        data_source=DataSources.LANDSAT,
        date_range=("2024-01-01", "2024-01-31")
    )
    processor.create_collection(
        bbox=bbox,
        cloud_cover_lt=20,
        platform={"in": ["LANDSAT_8"]}
    )
```

4. Query collection and Process Data


```python
# Query collection with filters to get the data you want
ds = processor.get_xarray(
    geometries=[aoi1_polygon,aoi2_polygon],
    bands=["B4", "B5"],
    cloud_cover_lt=20,
    date_range=["2024-01-10", "2024-01-30"]
)

# returns an xarray dataset with the data for the geometries and bands specified

# Calculate NDVI
ndvi_ds = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)
ndvi_ds = xr.Dataset(
    {"NDVI": ndvi},
    coords=ds.coords,
    attrs=ds.attrs,
)

# Save results from xarray to geotiff files, each geometry's data will be stored in
# its own folder
output_files = save_per_geometry(ndvi_ds, output_dir, file_prefix="ndvi", data_var="NDVI")

for geom_id, filepath in output_files.items():
    print(f"Geometry {geom_id}: {filepath}")
```


## Why this library?

Details on why this library was made, and how it reads multiple COGs efficiently and fast -
[Read the blog post here](https://blog.terrafloww.com/efficient-cloud-native-raster-data-access-an-alternative-to-rasterio-gdal/)

The aim of this library is to reduce the number of API calls to S3 objects (COGs), which
will result in lesser time consumed for random file access and hence faster time series analysis without needing to convert COGs to other formats like Zarr or NetCDF.

It also reduces the cost incurred by readers of paid data sources like Landsat on AWS where GET and LIST requests are significantly reduced due to local collection of COG internal metadata.

Benchmarks -

- Rasteret vs Zarr coming soon
- [Rasteret vs Rasterio](https://blog.terrafloww.com/efficient-cloud-native-raster-data-access-an-alternative-to-rasterio-gdal/)

## 🌍 Supported Data Sources
- Sentinel-2 Level 2A
- Landsat Collection 2 Level 2 SR

## 📝 License
Apache 2.0 License

## 🤝 Contributing
Contributions welcome! 

## ⚠️ Known Limitations
- High memory usage for large areas
- AWS requester-pays buckets need valid credentials
- Limited to Sentinel-2 and Landsat data formats
