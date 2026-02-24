from __future__ import annotations

import pyarrow as pa
from shapely.geometry import MultiPolygon, Polygon

from rasteret.core.geometry import (
    bbox_from_geojson_coords,
    coerce_to_geoarrow,
    to_rasterio_geojson,
    transform_coords,
)


def test_to_rasterio_geojson_supports_multipolygon() -> None:
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
    mp = MultiPolygon([poly1, poly2])

    arr = coerce_to_geoarrow(mp)
    geojson = to_rasterio_geojson(arr, 0)

    assert geojson["type"] == "MultiPolygon"
    bbox = bbox_from_geojson_coords(geojson)
    assert bbox == (0.0, 0.0, 3.0, 3.0)


def test_transform_coords_supports_multipolygon_identity_crs() -> None:
    poly1 = Polygon([(-1, -1), (0, -1), (0, 0), (-1, 0), (-1, -1)])
    poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)])
    mp = MultiPolygon([poly1, poly2])

    arr = coerce_to_geoarrow(mp)
    out = transform_coords(arr, 0, src_crs=4326, dst_crs=4326)

    assert out["type"] == "MultiPolygon"
    bbox = bbox_from_geojson_coords(out)
    assert bbox == (-1.0, -1.0, 2.0, 2.0)


def test_bbox_from_geojson_coords_polygon_and_multipolygon() -> None:
    poly_geojson = {
        "type": "Polygon",
        "coordinates": [[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]],
    }
    assert bbox_from_geojson_coords(poly_geojson) == (0.0, 0.0, 2.0, 1.0)

    mp_geojson = {
        "type": "MultiPolygon",
        "coordinates": [
            [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]],
            [[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)]],
        ],
    }
    assert bbox_from_geojson_coords(mp_geojson) == (0.0, 0.0, 11.0, 11.0)


def test_coerce_to_geoarrow_accepts_geojson_multipolygon() -> None:
    mp_geojson = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
        ],
    }
    arr = coerce_to_geoarrow(mp_geojson)
    assert isinstance(arr, pa.Array)
