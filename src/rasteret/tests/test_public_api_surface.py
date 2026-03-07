# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rasteret
from rasteret.core.collection import Collection


def _write_minimal_partitioned_collection(dataset_path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
            "collection": pa.array(["sentinel-2-l2a"]),
            "year": pa.array([2024], type=pa.int32()),
            "month": pa.array([1], type=pa.int32()),
        }
    )
    pq.write_to_dataset(
        table, root_path=str(dataset_path), partition_cols=["year", "month"]
    )


def test_build_from_stac_returns_cached_collection_without_network() -> None:
    with TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)
        collection_name = Collection.create_name(
            "demo", ("2024-01-01", "2024-01-31"), "sentinel-2-l2a"
        )
        dataset_path = workspace / f"{collection_name}_stac"
        _write_minimal_partitioned_collection(dataset_path)

        collection = rasteret.build_from_stac(
            name="demo",
            stac_api="https://example-stac.invalid/v1",
            collection="sentinel-2-l2a",
            bbox=(0.0, 0.0, 1.0, 1.0),
            date_range=("2024-01-01", "2024-01-31"),
            workspace_dir=workspace,
        )

        assert isinstance(collection, Collection)
        assert collection.name == collection_name


def test_collection_analysis_methods_delegate_to_execution_layer() -> None:
    with TemporaryDirectory() as tmp_dir:
        dataset_path = Path(tmp_dir) / "example_stac"
        _write_minimal_partitioned_collection(dataset_path)
        collection = Collection._load_cached(dataset_path)

    with (
        patch(
            "rasteret.core.collection.get_collection_xarray",
            return_value="xarray-result",
        ) as mocked_xarray,
        patch(
            "rasteret.core.collection.get_collection_gdf", return_value="gdf-result"
        ) as mocked_gdf,
        patch(
            "rasteret.core.collection.get_collection_numpy",
            return_value="numpy-result",
        ) as mocked_numpy,
        patch(
            "rasteret.core.collection.get_collection_point_samples",
            return_value=pa.table({"value": pa.array([1.0])}),
        ) as mocked_points,
    ):
        xarray_result = collection.get_xarray(
            geometries=[], bands=["B04"], xr_combine="merge_override"
        )
        gdf_result = collection.get_gdf(geometries=[], bands=["B04"])
        numpy_result = collection.get_numpy(geometries=[], bands=["B04"])
        points_result = collection.sample_points(points=[], bands=["B04"])

    mocked_xarray.assert_called_once()
    assert mocked_xarray.call_args.kwargs["xr_combine"] == "merge_override"
    mocked_gdf.assert_called_once()
    mocked_numpy.assert_called_once()
    mocked_points.assert_called_once()
    assert xarray_result == "xarray-result"
    assert gdf_result == "gdf-result"
    assert numpy_result == "numpy-result"
    assert points_result.num_rows == 1


def test_get_numpy_forwards_all_touched_flag() -> None:
    with TemporaryDirectory() as tmp_dir:
        dataset_path = Path(tmp_dir) / "example_stac"
        _write_minimal_partitioned_collection(dataset_path)
        collection = Collection._load_cached(dataset_path)

    with patch(
        "rasteret.core.collection.get_collection_numpy",
        return_value="numpy-result",
    ) as mocked_numpy:
        _ = collection.get_numpy(geometries=[], bands=["B04"], all_touched=True)

    assert mocked_numpy.call_count == 1
    assert mocked_numpy.call_args.kwargs["all_touched"] is True


def test_public_api_surface_is_collection_first() -> None:
    exported = set(dir(rasteret))
    assert "build_from_stac" in exported
    assert "load" in exported
    assert "build_from_table" in exported
    assert "as_collection" in exported
    assert "Collection" in exported
    assert "BandRegistry" in exported

    assert "Rasteret" not in exported


def test_top_level_rasteret_class_is_not_exposed() -> None:
    with pytest.raises(AttributeError):
        _ = rasteret.Rasteret


def _write_minimal_flat_collection(path: Path) -> None:
    """Write a flat (non-Hive) parquet with required columns."""
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
            "scene_bbox": pa.array(
                [[0.0, 0.0, 1.0, 1.0]], type=pa.list_(pa.float64(), 4)
            ),
        }
    )
    pq.write_table(table, str(path))


def test_load_valid_file() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test.parquet"
        _write_minimal_flat_collection(path)
        collection = rasteret.load(path)
        assert isinstance(collection, Collection)
        assert collection.name == "test"


def test_load_hf_uri_uses_hf_reader(monkeypatch) -> None:
    import pyarrow.dataset as pads

    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
            "scene_bbox": pa.array(
                [[0.0, 0.0, 1.0, 1.0]], type=pa.list_(pa.float64(), 4)
            ),
        }
    )
    dataset = pads.dataset(table)

    with patch(
        "rasteret.core.collection.open_hf_parquet_dataset",
        return_value=dataset,
    ) as mocked:
        collection = rasteret.load("hf://datasets/terrafloww/example/records.parquet")

    mocked.assert_called_once_with("hf://datasets/terrafloww/example/records.parquet")
    assert isinstance(collection, Collection)
    assert collection.dataset is not None
    assert collection.dataset.count_rows() == 1


def test_load_rejects_missing_columns() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bad.parquet"
        table = pa.table({"id": pa.array(["scene-1"]), "foo": pa.array([1])})
        pq.write_table(table, str(path))
        with pytest.raises(ValueError, match="missing required columns"):
            rasteret.load(path)


def test_load_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        rasteret.load("/nonexistent/path.parquet")


def test_from_parquet_fallback_to_non_hive() -> None:
    """from_parquet should work on non-Hive partitioned parquet."""
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "flat.parquet"
        _write_minimal_flat_collection(path)
        collection = Collection.from_parquet(path)
        assert isinstance(collection, Collection)


def _minimal_read_ready_table() -> pa.Table:
    metadata_type = pa.struct(
        [
            ("image_width", pa.int32()),
            ("image_height", pa.int32()),
            ("tile_width", pa.int32()),
            ("tile_height", pa.int32()),
            ("dtype", pa.string()),
            ("transform", pa.list_(pa.float64())),
            ("predictor", pa.int32()),
            ("compression", pa.int32()),
            ("tile_offsets", pa.list_(pa.int64())),
            ("tile_byte_counts", pa.list_(pa.int64())),
            ("pixel_scale", pa.list_(pa.float64())),
            ("tiepoint", pa.list_(pa.float64())),
            ("nodata", pa.float64()),
            ("samples_per_pixel", pa.int32()),
            ("planar_configuration", pa.int32()),
            ("photometric", pa.int32()),
            ("extra_samples", pa.list_(pa.int32())),
        ]
    )
    return pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/test.tif"}}]),
            "scene_bbox": pa.array(
                [[0.0, 0.0, 1.0, 1.0]], type=pa.list_(pa.float64(), 4)
            ),
            "collection": pa.array(["sentinel-2-l2a"]),
            "B04_metadata": pa.array(
                [
                    {
                        "image_width": 512,
                        "image_height": 512,
                        "tile_width": 256,
                        "tile_height": 256,
                        "dtype": "uint16",
                        "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 0.0],
                        "predictor": 2,
                        "compression": 8,
                        "tile_offsets": [0, 1024, 2048, 3072],
                        "tile_byte_counts": [1024, 1024, 1024, 1024],
                        "pixel_scale": [10.0, 10.0, 0.0],
                        "tiepoint": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "nodata": None,
                        "samples_per_pixel": 1,
                        "planar_configuration": 1,
                        "photometric": 1,
                        "extra_samples": None,
                    }
                ],
                type=metadata_type,
            ),
        }
    )


def test_as_collection_wraps_read_ready_table() -> None:
    table = _minimal_read_ready_table()
    collection = rasteret.as_collection(table, name="wrapped")
    assert isinstance(collection, Collection)
    assert collection.name == "wrapped"
    assert collection.data_source == "sentinel-2-l2a"


def test_as_collection_requires_scene_bbox() -> None:
    table = _minimal_read_ready_table().drop_columns(["scene_bbox"])
    with pytest.raises(ValueError, match="missing required columns"):
        rasteret.as_collection(table)


def test_as_collection_requires_metadata_by_default() -> None:
    table = _minimal_read_ready_table().drop_columns(["B04_metadata"])
    with pytest.raises(ValueError, match="No '\\*_metadata' columns found"):
        rasteret.as_collection(table)


def test_as_collection_allows_metadata_optional_mode() -> None:
    table = _minimal_read_ready_table().drop_columns(["B04_metadata"])
    collection = rasteret.as_collection(table, require_band_metadata=False)
    assert isinstance(collection, Collection)


def test_as_collection_warns_for_large_in_memory_table(monkeypatch) -> None:
    table = _minimal_read_ready_table()
    monkeypatch.setattr(rasteret, "_total_ram_bytes", lambda: 512)
    monkeypatch.setattr(rasteret, "_AS_COLLECTION_MEMORY_WARNING_EMITTED", False)
    with pytest.warns(UserWarning, match="large in-memory pyarrow.Table"):
        rasteret.as_collection(table, require_band_metadata=False)
