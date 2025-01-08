# 🛰️ Rasteret

This library gives you faster querying of Cloud-Optimized GeoTIFFs (COGs), and lowers S3 HTTP requests in your workflows.

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
- Fast byte-range based COG data retrieval
- STAC Geoparquet creation with COG header metadata 
- Paid data source support (Landsat on AWS S3)
- Xarray and GeoDataFrame outputs
- Parallel data loading
- Simple high-level API

---

## Why this library?

### 💡 The Problem

Currently satellite image access requires multiple HTTP requests in the backend:

- Initial request to read COG file headers 
- Additional requests for headers if they are split in COG file
- Final requests for actual data tiles, which returns the numpy arrays

- These requests are repeated in various situations such as -
  - New Python envs, inside AWS Lambdas or in Docker Containers
  - New Local machine env when GDAL cache is cleared
    - when a Jupyter kernel is restarted
    - or your laptop itself is restarted

### ✨ Rasteret's Solution 

Rasteret reimagines how we access cloud-hosted satellite imagery by:
- Creating local 'Collections', which are STAC-geoparquet with additional columns for COG headers for each STAC item
- Calculating exact byte-ranges of image tiles needed using the local cache, and avoiding the extra HTTP requests for COG headers that most libraries always do.
- Making 1 range-request per required image tile to create the numpy arrays
- Ensuring COG file headers are never re-read across new Python environments

### 📊 Performance Benchmarks

<details>
<summary><b>Speed Benchmarks</b></summary>

Test setup: Filter 1 year of STAC items (100+ scenes), process 20 Sentinel-2 filtered scenes, over an agricultural area, by reading RED and NIR bands, which is 40 COG files in total. (2 CPU, 4 threads machine)

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

Example: 1000 Landsat scenes (4 bands each) being queried across 50 new environments

#### First Run Setup
| Operation | Rasterio | Rasteret | Calculation |
|-----------|----------|-----------|-------------|
| Header Requests | $0.0032 | $0.0032 | 1000 scenes × 4 bands × 2 requests × $0.0004/1000 |
| Data Tile Requests | $0.00032 | $0.00032 | 100 farms × 2 tiles × 4 bands × $0.0004/1000 |
| **Total Per Environment** | **$0.00352** | **$0.00352** | One-time setup for Rasteret |

#### Subsequent Runs (50 runs in new environments)
| Operation | Rasterio | Rasteret | Notes |
|-----------|----------|-----------|--------|
| Header Requests | $0.16 | $0 | 50 × $0.0032 (Rasterio) vs Cached headers (Rasteret)|
| Data Tile Requests | $0.016 | $0.016 | 50 × $0.00032 |
| **Total** | **$0.176** | **$0.016** | **90% savings** |

#### Alternative: Full Images Download
| Cost Type | Amount | Notes |
|-----------|---------|--------|
| Data Transfer | $576 | 6.4TB (1.6GB * 4000 files) × $0.09/GB |
| Monthly Storage | $150 | Varies by provider |
| GET Requests | $0.01+ | Still incurs cost within the same AWS account |
| **Total** | **$726+** | Plus ongoing storage costs |

The cost breakdown:
- Each COG file typically needs 2 requests to read its headers
- With Rasteret, headers are read once during 'Collection' creation
- Subsequent analysis only requires data tile requests
    - In the above cases we assume 2 COG tiles are needed per farm polygon

- Cost savings compound with distributed (in new dockers and python envs) and repeated processing, like in ML training and inference workloads
</details>


### 🎯 Key Benefits


This makes Rasteret particularly effective for:
- Time series analysis requiring many scenes
- ML pipelines with multiple training runs
- Production systems using serverless/container deployments
- Multi-tenant applications accessing same data
- Not needing to convert COG to Zarr for timeseries analysis


---

## 🌍 Built-in Data Sources
- Sentinel-2 Level 2A
    - Earthsearch v1 [STAC Endpoint](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/) (AWS S3 US-West2 bucket)
- Landsat Collection 2 Level 2 SR
    - USGS Landsatlook STAC Server [Endpoint](https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-sr/) (AWS S3 US-West2 bucket)

## ⚠️ Known Limitations
- Currently tested only with Sentinel-2 and Landsat 8,9 platform's data
- Creating or Loading a 'Rasteret Collection' from S3 is not yet supported

---

## 📋 Prerequisites
- Python 3.10.x,3.11.x
- AWS credentials (for accessing paid data like Landsat on AWS)

### ⚙️ AWS Credentials Setup

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

# Get the total bounds of all polygons above
bbox = aoi1_polygon.union(aoi2_polygon).bounds
# OR
# give even larger AOI bounds that covers all your future analysis areas
# eg., Polygon of a State or a Country
# bbox = country_polygon.bounds
```

### 2. Configure Rasteret

Set up basic parameters for data collection, and check for existing collection
in your workspace directory, if they were created earlier.

```python
# Collection configuration

# give your custom name for local collection, it will be attached to the
# beginning of the collection name for eg., bangalore_202401-12_landsat
custom_name = "bangalore"

# here we are aiming to write 1 year worth of STAC metadata and COG file headers to local disk
date_range = ("2024-01-01", "2024-12-31")

# choose from LANDSAT / SENTINEL2
data_source = DataSources.LANDSAT

# Set up workspace folder as you wish
workspace_dir = Path.home() / "rasteret_workspace"
workspace_dir.mkdir(exist_ok=True)

# List existing collections if there are any in the workspace folder
collections = Rasteret.list_collections(workspace_dir=workspace_dir)
for c in collections:
    print(f"- {c['name']}: {c['data_source']}, {c['date_range']}, {c['size']} scenes")
```

### 3. Initialize and Create Collection

```python
# Try loading existing collection
try:
    # example name given here
    processor = Rasteret.load_collection("bangalore_202401-12_landsat",workspace_dir=workspace_dir)
except ValueError:

    # Instantiate the Class
    processor = Rasteret(
        workspace_dir=workspace_dir,
        custom_name="bangalore",
        data_source=DataSources.LANDSAT,
        date_range=("2024-01-01", "2024-01-31")
    )

    # and create a new collection

    # we are giving the BBOX for which STAC items and their COG headers will be fetched
    # and also filtering using PySTAC filters for LANDSAT 8 platform specifically
    # from LANDSAT USGS STAC, and giving a scene level cloud-cover filter
    processor.create_collection(
        bbox=bbox,
        cloud_cover_lt=20,
        platform={"in": ["LANDSAT_8"]}
    )
```

### 4. Query the Collection and Process Data

```python
# Now we can query the collection created above, to get the data we want
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
# and creating a xarray dataset for each geometry and its time series data

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
# its own folder. We can also give file-name prefix
# and also mention which Xarray varible to save as geotiffs
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