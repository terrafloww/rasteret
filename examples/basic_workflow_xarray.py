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
    date_range = ("2024-03-01", "2024-03-31")
    data_source = DataSources.LANDSAT

    # Define area and time of interest
    aoi1_polygon = Polygon(
        [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)]
    )

    bbox = aoi1_polygon.bounds

    print("1. Available Collections")
    print("----------------------")
    collections = Rasteret.list_collections(workspace_dir=workspace_dir)
    for c in collections:
        print(
            f"- {c['name']}: {c['data_source']}, {c['date_range']}, {c['size']} scenes"
        )

    try:
        processor = Rasteret.load_collection(f"{custom_name}_202403-03_landsat")
    except ValueError:
        print("\n2. Creating New Collection")
        print("-------------------------")
        processor = Rasteret(
            custom_name=custom_name,
            data_source=data_source,
            workspace_dir=workspace_dir,
            date_range=date_range,
        )
        processor.create_collection(
            bbox=bbox,
            date_range=date_range,
            cloud_cover_lt=20,
            platform={"in": ["LANDSAT_8"]},
        )

    print("\n3. Processing Data")
    print("----------------")

    # Calculate NDVI using xarray operations
    ds = processor.get_xarray(
        geometries=[aoi1_polygon],
        bands=["B4", "B5"],
        cloud_cover_lt=20,
    )

    print("\nInput dataset:")
    print(ds)

    # Calculate NDVI and preserve metadata
    ndvi = (ds.B5 - ds.B4) / (ds.B5 + ds.B4)
    ndvi_ds = xr.Dataset(
        {"NDVI": ndvi},
        coords=ds.coords,
        attrs=ds.attrs,
    )

    print("\nNDVI dataset:")
    print(ndvi_ds)

    output_dir = Path("ndvi_results")
    output_dir.mkdir(exist_ok=True)

    output_files = save_per_geometry(
        ndvi_ds, output_dir, file_prefix="ndvi", data_var="NDVI"
    )

    print("\nProcessed NDVI files:")
    for geom_id, filepath in output_files.items():
        print(f"Geometry {geom_id}: {filepath}")


if __name__ == "__main__":
    main()
