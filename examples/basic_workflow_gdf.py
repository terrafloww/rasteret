# examples/basic_workflow.py
from pathlib import Path
from shapely.geometry import Polygon

from rasteret import Rasteret


def main():
    """Demonstrate core workflows with Rasteret."""
    # 1. Define parameters

    custom_name = "bangalore3"
    date_range = ("2024-01-01", "2024-01-31")
    data_source = "landsat-c2l2-sr"

    workspace_dir = Path.home() / "rasteret_workspace"
    workspace_dir.mkdir(exist_ok=True)

    print("1. Defining Area of Interest")
    print("--------------------------")

    # Define area and time of interest
    aoi_polygon = Polygon(
        [(77.55, 13.01), (77.58, 13.01), (77.58, 13.08), (77.55, 13.08), (77.55, 13.01)]
    )

    aoi_polygon2 = Polygon(
        [(77.56, 13.02), (77.59, 13.02), (77.59, 13.09), (77.56, 13.09), (77.56, 13.02)]
    )

    # get total bounds of all polygons above
    bbox = aoi_polygon.union(aoi_polygon2).bounds

    print("\n2. Creating and Loading Collection")
    print("--------------------------")

    # 2. Initialize processor - name generated automatically
    processor = Rasteret(
        custom_name=custom_name,
        data_source=data_source,
        output_dir=workspace_dir,
        date_range=date_range,
    )

    # Create index if needed
    if processor._collection is None:
        processor.create_index(
            bbox=bbox, date_range=date_range, query={"cloud_cover_lt": 20}
        )

    # List existing collections
    collections = Rasteret.list_collections(dir=workspace_dir)
    print("Available collections:")
    for c in collections:
        print(f"- {c['name']}: {c['size']} scenes")

    print("\n3. Processing Data")
    print("----------------")

    df = processor.get_gdf(
        geometries=[aoi_polygon, aoi_polygon2], bands=["B4", "B5"], cloud_cover_lt=20
    )

    print(f"Columns: {df.columns}")
    print(f"Unique dates: {df.datetime.dt.date.unique()}")
    print(f"Unique geometries: {df.geometry.unique()}")


if __name__ == "__main__":
    main()
