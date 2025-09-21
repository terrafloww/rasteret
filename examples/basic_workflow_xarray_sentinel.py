# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2025 Terrafloww Labs, Inc.

from pathlib import Path
from shapely.geometry import Polygon

from rasteret import Rasteret
from rasteret.constants import DataSources
from rasteret.core.utils import save_per_geometry

import xarray as xr

aoi1_polygon = Polygon(
    [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)]
)


# Get the total bounds of all polygons above
bbox = aoi1_polygon.bounds
# OR
# give even larger AOI bounds that covers all your future analysis areas
# eg., Polygon of a State or a Country
# bbox = country_polygon.bounds

# Collection configuration

# give your custom name for local collection, it will be attached to the
# beginning of the collection name for eg., bangalore_202401-12_landsat
custom_name = "bangalore"

# here we are aiming to write 1 year worth of STAC metadata and COG file headers to local disk
date_range = ("2024-01-01", "2024-01-31")

# choose from LANDSAT / SENTINEL2
data_source = DataSources.SENTINEL2

# Set up workspace folder as you wish
workspace_dir = Path.home() / "rasteret_workspace"
workspace_dir.mkdir(exist_ok=True)

# List existing collections if there are any in the workspace folder
collections = Rasteret.list_collections(workspace_dir=workspace_dir)
for c in collections:
    print(f"\nExisting Collection in workspace dir {workspace_dir}:")
    print(f"- {c['name']}: {c['data_source']}, {c['date_range']}, {c['size']} scenes")

# Try loading existing collection
try:
    # example name given here
    processor = Rasteret.load_collection(
        "bangalore_202401-12_landsat", workspace_dir=workspace_dir
    )
except ValueError:

    # Instantiate the Class
    processor = Rasteret(
        workspace_dir=workspace_dir,
        custom_name=custom_name,
        data_source=data_source,
        date_range=date_range,
    )

    # and create a new collection

    # we are giving the BBOX for which STAC items and their COG headers will be fetched
    # and also filtering using PySTAC filters for LANDSAT 8 platform specifically
    # from LANDSAT USGS STAC, and giving a scene level cloud-cover filter
    processor.create_collection(
        bbox=bbox,
        cloud_cover_lt=20,
    )

# Now we can query the collection created above, to get the data we want
# in this case 2 geometries, 2 bands, and a few PySTAC search filters are provided
ds = processor.get_xarray(
    geometries=[aoi1_polygon],
    bands=["B04", "B08"],
    cloud_cover_lt=20,
    date_range=["2024-01-10", "2024-01-30"],
)
# this returns an xarray dataset variable "ds" with the data for the geometries and bands specified
# behind the scenes, the library is efficiently filtering the local STAC geoparquet,
# for the LANDSAT scenes that pass the filters and dates provided
# then its getting the tif urls of the requested bands
# then grabbing COG tiles only for the geometries from those tif files
# and creating a xarray dataset for each geometry and its time series data

# Calculate NDVI
ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04)

# for LANDSAT satellite
# ndvi_ds = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)

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
output_files = save_per_geometry(
    ndvi_ds, output_dir, file_prefix="ndvi", data_var="NDVI"
)

for geom_id, filepath in output_files.items():
    print(f"Geometry {geom_id}: {filepath}")