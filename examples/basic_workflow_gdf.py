# examples/basic_workflow.py
from pathlib import Path
from shapely.geometry import Polygon

from rasteret import Rasteret
from rasteret.constants import DataSources


def main():
    """Demonstrate core workflows with Rasteret."""
    # 1. Setup workspace and parameters

    workspace_dir = Path.home() / "rasteret_workspace"
    workspace_dir.mkdir(exist_ok=True)

    custom_name = "bangalore"
    date_range = ("2024-01-01", "2024-01-31")
    data_source = DataSources.LANDSAT

    # 2. List existing collections
    print("1. Available Collections")
    print("----------------------")
    collections = Rasteret.list_collections(workspace_dir=workspace_dir)
    for c in collections:
        print(
            f"- {c['name']}: {c['data_source']}, {c['date_range']}, {c['size']} scenes"
        )

    # 3. Define areas of interest
    print("\n2. Defining Areas of Interest")
    print("---------------------------")
    aoi1_polygon = Polygon(
        [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)]
    )
    aoi2_polygon = Polygon(
        [(77.56, 13.02), (77.59, 13.02), (77.59, 13.09), (77.56, 13.09), (77.56, 13.02)]
    )
    bbox = aoi1_polygon.union(aoi2_polygon).bounds

    # 4. Load or create collection
    print("\n3. Loading/Creating Collection")
    print("---------------------------")
    try:
        processor = Rasteret.load_collection(f"{custom_name}_202401_landsat")
    except ValueError:
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

    # 5. Process data
    print("\n4. Processing Data")
    print("----------------")

    df = processor.get_gdf(
        geometries=[aoi1_polygon, aoi2_polygon], bands=["B4", "B5"], cloud_cover_lt=20
    )

    print(f"Columns: {df.columns}")
    print(f"Unique dates: {df.datetime.dt.date.unique()}")
    print(f"Unique geometries: {df.geometry.unique()}")


if __name__ == "__main__":
    main()
