# 🛰️ Rasteret

Faster querying of Cloud-Optimized GeoTIFFs (COGs) with lower HTTP requests in your workflows, currently tested for Sentinel-2 and Landsat COG files.

> [!WARNING]  
> Work-in-progress library. The APIs are subject to change, and as such, documentation is not yet available.

## Table of Contents
- [Features](#-features)
- [Why Rasteret?](#why-this-library)
- [Built-in Data Sources](#-built-in-data-sources)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [License](#-license)
- [Contributing](#-contributing)

---

## 🚀 Features
- Fast byte-range based COG access
- STAC Geoparquet creation with COG header metadata 
- Paid S3 bucket support (AWS S3 Landsat)
- Xarray and GeoDataFrame outputs
- Parallel data loading
- Simple high-level API

---

## Why this library?

### 💡 The Problem

Currently satellite image access requires multiple HTTP requests:
- Initial request to read COG headers 
- Additional requests if headers are split
- Final requests for actual data tiles
- These requests repeat in new environments:
  - New Python environments (like inside parallel Lambdas/ parallel Docker Containers in k8s)
  - Or in local environment when GDAL cache is cleared (like a Jupyter kernel restart / Laptop restart)

### ✨ Rasteret's Solution 

Rasteret reimagines how we access cloud-hosted satellite imagery by:
- Creating local 'collections' with pre-cached COG file headers along with STAC metadata
- Calculating exact byte ranges for image tiles needed, without header requests
- Making single optimized HTTP request per required tile
- Ensuring COG file headers are never re-read across new Python environments

### 📊 Performance Benchmarks

<details>
<summary><b>Speed Benchmarks</b></summary>

Test setup: Filter 1 year of STAC (100+ scenes), process 20 Sentinel-2 filtered scenes over an agricultural area, accessing RED and NIR bands (40 COG files total)

| Operation | Component | Rasterio | Rasteret | Notes |
|-----------|-----------|----------|-----------|--------|
| STAC Query | Metadata Search | 2.0s | 0.5s | Finding available scenes (STAC API vs Geoparquet) |
| Data Access | Header Reading | 12s | - | ~0.3s per file (Rasterio) vs Not required (Rasteret) |
| | Tile Reading | 32s | 8s | Actual data access |
| **Total Time** | | **44s** | **8s** | **5.5x faster** |

The speed improvement comes from:
- Querying local GeoParquet instead of STAC API endpoints
- Eliminating repeated header requests
- Optimized parallel data loading
</details>


<details>
<summary><b>Cost Analysis</b></summary>

Example: 1000 Landsat scenes (4 bands each) across 50 parallel environments

#### First Run Setup
| Operation | Rasterio | Rasteret | Calculation |
|-----------|----------|-----------|-------------|
| Header Requests | $0.0032 | $0.0032 | 1000 scenes × 4 bands × 2 requests × $0.0004/1000 |
| Data Tile Requests | $0.00032 | $0.00032 | 100 farms × 2 tiles × 4 bands × $0.0004/1000 |
| **Total Per Environment** | **$0.00352** | **$0.00352** | One-time setup for Rasteret |

#### Subsequent Runs (50 runs in new environments)
| Operation | Rasterio | Rasteret | Notes |
|-----------|----------|-----------|--------|
| Header Requests | $0.176 | $0 | 50 × $0.00352 (Rasterio) vs Cached headers (Rasteret)|
| Data Tile Requests | $0.016 | $0.016 | 50 × $0.00032 |
| **Total** | **$0.192** | **$0.016** | **91% savings** |

#### Alternative: Full Images Download
| Cost Type | Amount | Notes |
|-----------|---------|--------|
| Data Transfer | $576 | 6.4TB (1.6GB * 4000 files) × $0.09/GB |
| Monthly Storage | $150 | Varies by provider |
| GET Requests | Still needed | For company S3 access |
| **Total** | **$726+** | Plus ongoing storage costs |

The cost breakdown:
- Each COG file typically needs 2 requests to read its headers
- With Rasteret, headers are read once during Collection creation
- Subsequent access only requires data tile requests
    - In the above cases we assume 2 COG tiles are needed per farm
- Cost savings compound with distributed (in new dockers and python envs) and with repeated processing, like in ML training and inference workloads
</details>


### 🎯 Key Benefits


This makes Rasteret particularly effective for:
- Time series analysis requiring many scenes
- ML pipelines with multiple training runs
- Production systems using serverless/container deployments
- Multi-tenant applications accessing same data
- Not needing convert COG to Zarr for most usecases


---

## 🌍 Built-in Data Sources
- Sentinel-2 Level 2A
    - Earthsearch v1 [STAC Endpoint](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/) (AWS S3 US-West2 bucket)
- Landsat Collection 2 Level 2 SR
    - USGS Landsatlook STAC Server [Endpoint](https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/) (AWS S3 US-West2 bucket)

## ⚠️ Known Limitations
- Currently tested only with Sentinel-2 and Landsat 8,9 platform's data
- S3 based Rasteret Collection creation and loading is not yet supported, right now they need to be in local disk

---

## 📋 Prerequisites
- Python 3.10.x,3.11.x
- AWS credentials (for accessing paid AWS buckets)

### ⚙️ AWS Credentials Setup
For accessing paid AWS buckets:

<details>
<summary><b>Setting up AWS credentials</b></summary>

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
</details>

---

## 📦 Installation
```bash
pip install rasteret
```

---

## 🏃‍♂️ Quick Start

### 1. Define Areas of Interest

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

# Use the total bounds of all polygons above
# OR give an even larger AOI that covers all your future analysis areas
# like AOI of a State or a Country
bbox = aoi1_polygon.union(aoi2_polygon).bounds
```

### 2. Configure Rasteret

Set up basic parameters for data collection, and check for existing collection
in your workspace directory, if they were created earlier.

```python
# Collection configuration

# give your custom name for local collection, it will attached to the
# beginning of the collection name for eg., bangalore_202401-12_landsat
# date range and data source name is added automatically while rasteret creates a collection
custom_name = "bangalore"

# pay time and cost upfront for COG headers and STAC metadata
# here we are writing 1 year worth of STAC metadata and COG file headers to local disk
date_range = ("2024-01-01", "2024-12-31")

# choose from LANDSAT / SENTINEL2
data_source = DataSources.LANDSAT

# Set up workspace folder as you wish
workspace_dir = Path.home() / "rasteret_workspace"
workspace_dir.mkdir(exist_ok=True)

# List existing collections if there are any in the workspace folder (by default is /home/user/rasteret_workspace)
collections = Rasteret.list_collections()
for c in collections:
    print(f"- {c['name']}: {c['data_source']}, {c['date_range']}, {c['size']} scenes")
```

### 3. Initialize and Create Collection

```python
# Try loading existing collection
try:
    # example name given here
    processor = Rasteret.load_collection("bangalore_202401-12_landsat")
except ValueError:

    # Instantiate the Class
    processor = Rasteret(
        custom_name="bangalore",
        data_source=DataSources.LANDSAT,
        date_range=("2024-01-01", "2024-01-31")
    )

    # and create a new collection
    # here we are giving the BBOX for which STAC items and thier COG headers will be
    # downloaded to local. and also filtering using PySTAC filters for LANDSAT 8 platform
    # specifically from LANDSAT USGS STAC, and giving a scene level cloud cover filter
    processor.create_collection(
        bbox=bbox,
        cloud_cover_lt=20,
        platform={"in": ["LANDSAT_8"]}
    )
```

### 4. Query the Collection and Process Data

```python
# Query collection created above with filters to get the data you want
# in this case 2 geometries, 2 bands, and a few PySTAC search filters are provided
ds = processor.get_xarray(
    geometries=[aoi1_polygon,aoi2_polygon],
    bands=["B4", "B5"],
    cloud_cover_lt=20,
    date_range=["2024-01-10", "2024-01-30"]
)
# this returns an xarray dataset variable "ds" with the data for the geometries and bands specified
# behind the scenes, the library is efficiently filtering the local STAC geoparquet,
# for the LANDSAT scenes that pass the filters and dates provided
# then its getting the tif urls of the requested bands
# then grabbing COG tiles only for the geometries from those tif files
# and creating a xarray dataset for each geom and its time series data

# Calculate NDVI
ndvi_ds = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)

# give a data variable name for NDVI array
ndvi_ds = xr.Dataset(
    {"NDVI": ndvi},
    coords=ds.coords,
    attrs=ds.attrs,
)

# create a output folder if you wish to
output_dir = Path(f"ndvi_results_{custom_name}")
output_dir.mkdir(exist_ok=True)

# Save results from xarray to geotiff files, each geometry's data will be stored in
# its own folder. Here we are giving the file name prefix and also mentioning 
# which Xarray varible to save
# each geometry in xarray will get its own folder
output_files = save_per_geometry(ndvi_ds, output_dir, file_prefix="ndvi", data_var="NDVI")

for geom_id, filepath in output_files.items():
    print(f"Geometry {geom_id}: {filepath}")

# example print    
# geometry_1 : ndvi_results_bangalore/geometry_1/ndvi_20241207.tif
```

---

## 📝 License
Apache 2.0 License

## 🤝 Contributing
Contributions welcome!