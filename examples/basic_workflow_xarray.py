# examples/basic_workflow.py
from pathlib import Path
from shapely.geometry import Polygon
import xarray as xr

from rasteret import Rasteret
from rasteret.constants import DataSources
from rasteret.core.utils import save_per_geometry


def main():

    # 1. Define parameters
    custom_name = "bangalore"
    date_range = ("2024-01-01", "2024-01-31")
    data_source = DataSources.LANDSAT  # or SENTINEL2

    workspace_dir = Path.home() / "rasteret_workspace"
    workspace_dir.mkdir(exist_ok=True)

    print("1. Defining Area of Interest")
    print("--------------------------")

    # Define area and time of interest
    aoi1_polygon = Polygon(
        [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)]
    )

    aoi2_polygon = Polygon(
        [(77.56, 13.02), (77.59, 13.02), (77.59, 13.09), (77.56, 13.09), (77.56, 13.02)]
    )

    # get total bounds of all polygons above for stac search and stac index creation
    bbox = aoi1_polygon.union(aoi2_polygon).bounds

    print("\n2. Creating and Loading Collection")
    print("--------------------------")

    # 2. Initialize processor - name generated automatically
    processor = Rasteret(
        custom_name=custom_name,
        data_source=data_source,
        output_dir=workspace_dir,
        date_range=date_range,
    )

    # Create index if collection is not present
    if processor._collection is None:
        processor.create_collection(
            bbox=bbox,
            date_range=date_range,
            cloud_cover_lt=20,
            # add platform filter for Landsat 9, 8, 7, 5, 4 if needed,
            # else remove it for all platforms
            # This is unique to Landsat STAC endpoint
            platform={"in": ["LANDSAT_8"]},
        )

    # List existing collections
    collections = Rasteret.list_collections(dir=workspace_dir)
    print("Available collections:")
    for c in collections:
        print(f"- {c['name']}: {c['size']} scenes")

    print("\n3. Processing Data")
    print("----------------")

    # Calculate NDVI using xarray operations
    ds = processor.get_xarray(
        # pass multiple geometries not its union bounds
        # for separate processing of each geometry
        geometries=[aoi1_polygon, aoi2_polygon],
        bands=["B4", "B5"],
        cloud_cover_lt=20,
    )

    print("\nInput dataset:")
    print(ds)

    # Calculate NDVI and preserve metadata
    ndvi = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)
    ndvi_ds = xr.Dataset(
        {"NDVI": ndvi},
        coords=ds.coords,  # Preserve coordinates including CRS
        attrs=ds.attrs,  # Preserve metadata
    )

    print("\nNDVI dataset:")
    print(ndvi_ds)

    # Create output directory
    output_dir = Path("ndvi_results")
    output_dir.mkdir(exist_ok=True)

    # Save per geometry, give prefix for output files in this case "ndvi"
    output_files = save_per_geometry(ndvi_ds, output_dir, prefix="ndvi")

    print("\nProcessed NDVI files:")
    for geom_id, filepath in output_files.items():
        print(f"Geometry {geom_id}: {filepath}")


if __name__ == "__main__":
    main()
