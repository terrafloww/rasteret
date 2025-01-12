# examples/basic_workflow.py
from pathlib import Path
from shapely.geometry import Polygon
import xarray as xr

from rasteret import Rasteret
from rasteret.constants import DataSources
from rasteret.core.utils import save_per_geometry


def main():
    """Example of Rasteret workflow with xarray output."""
    # 1. Setup workspace and parameters
    workspace_dir = Path.home() / "rasteret_workspace"
    workspace_dir.mkdir(exist_ok=True)

    custom_name = "bangalore"
    date_range = ("2024-01-01", "2024-03-31")
    data_source = DataSources.LANDSAT

    # Define area and time of interest
    aoi1_polygon = Polygon(
        [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)]
    )

    bbox = aoi1_polygon.bounds

    print("1. Available Collections")
    print("----------------------")

    # Here were are Collection_names, which are unique identifiers for a collection
    # based on custom_name, date_range and data_source
    # example collection name for above custom_name and date_range will be
    # "bangalore_202401-03_landsat", if date_range spans across years, it will be "bangalore_202401-202503_landsat"
    collections = Rasteret.list_collections(workspace_dir=workspace_dir)
    for c in collections:
        print(
            f"- {c['name']}: {c['data_source']}, {c['date_range']}, {c['size']} scenes"
        )

    try:

        # if you want to load a specific collection, you must pass the full collection_name
        # collection_names can be obtained from Rasteret.list_collections(workspace_dir=workspace_dir), shown above

        # here im passing a non existent collection name to raise ValueError on purpose
        processor = Rasteret.load_collection(
            collection_name="california_202401-03_landsat", workspace_dir=workspace_dir
        )
    except ValueError:
        print("\nCollection not found. Creating new collection...")
        print("-------------------------")

        # initialize Rasteret processor
        processor = Rasteret(
            custom_name=custom_name,
            data_source=data_source,
            workspace_dir=workspace_dir,
            date_range=date_range,
        )

        # Create a new collection
        # this will load an existing collection if it matches the
        # custom_name, date_range and data_source
        # so you can safely call this method without worrying about duplicates
        processor.create_collection(
            bbox=bbox,
            date_range=date_range,
            cloud_cover_lt=20,
            platform={"in": ["LANDSAT_8"]},
        )

    print("\n3. Retrieving Data")
    print("----------------")

    # Retrieve data for the area of interests and bands
    # provide as list of shapely geometries and bands
    # returns an xarray dataset with 4 dimensions: time, geometry, y, x
    ds = processor.get_xarray(
        geometries=[aoi1_polygon],
        bands=["B4", "B5"],
        cloud_cover_lt=20,
    )

    print("\nCreated dataset:")
    print(ds)

    # Calculate NDVI
    ndvi = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)

    # Create a new dataset with NDVI arrays
    # Add 'NDVI' as a new data variable
    ndvi_ds = xr.Dataset(
        {"NDVI": ndvi},
        coords=ds.coords,
        attrs=ds.attrs,
    )

    print("\nNDVI dataset:")
    print(ndvi_ds)

    # Create output directory if you wish
    output_dir = Path("ndvi_results")
    output_dir.mkdir(exist_ok=True)

    # Save NDVI results to separate files per geometry
    # here we are choosing the xarray data variable as "NDVI"
    # giving a file prefix "ndvi" and saving the files in output_dir
    output_files = save_per_geometry(
        ndvi_ds, output_dir, file_prefix="ndvi", data_var="NDVI"
    )

    print("\nProcessed NDVI files:")
    for geom_id, filepath in output_files.items():
        print(f"Geometry {geom_id}: {filepath}")


if __name__ == "__main__":
    main()
