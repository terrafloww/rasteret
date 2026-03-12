# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import rasteret
import rasteret.core.collection as collection_module
from rasteret.catalog import DatasetDescriptor, DatasetRegistry
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
            "bbox": pa.array(
                [{"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}],
                type=pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        }
    )
    pq.write_table(table, str(path))


def _write_minimal_aef_index(path: Path) -> None:
    bbox_type = pa.struct(
        [
            pa.field("xmin", pa.float64()),
            pa.field("ymin", pa.float64()),
            pa.field("xmax", pa.float64()),
            pa.field("ymax", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "fid": pa.array(["scene-1", "scene-2"]),
            "year": pa.array([2024, 2023], type=pa.int32()),
            "utm_zone": pa.array(["15N", "14N"]),
            "geom": pa.array([None, None], type=pa.null()),
            "bbox": pa.array(
                [
                    {"xmin": -95.4, "ymin": 38.45, "xmax": -95.1, "ymax": 38.7},
                    {"xmin": -100.0, "ymin": 40.0, "xmax": -99.0, "ymax": 41.0},
                ],
                type=bbox_type,
            ),
            "path": pa.array(
                [
                    "s3://example-bucket/aef/scene-1.tif",
                    "s3://example-bucket/aef/scene-2.tif",
                ]
            ),
            "crs": pa.array(["EPSG:32615", "EPSG:32614"]),
        }
    )
    pq.write_table(table, str(path))


def _write_minimal_aef_collection(path: Path) -> None:
    bbox_type = pa.struct(
        [
            pa.field("xmin", pa.float64()),
            pa.field("ymin", pa.float64()),
            pa.field("xmax", pa.float64()),
            pa.field("ymax", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2"]),
            "datetime": pa.array(
                [datetime(2024, 1, 1), datetime(2023, 1, 1)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {
                        "A00": {
                            "href": "s3://example-bucket/aef/scene-1.tif",
                            "band_index": 0,
                        }
                    },
                    {
                        "A00": {
                            "href": "s3://example-bucket/aef/scene-2.tif",
                            "band_index": 0,
                        }
                    },
                ]
            ),
            "bbox": pa.array(
                [
                    {"xmin": -95.4, "ymin": 38.45, "xmax": -95.1, "ymax": 38.7},
                    {"xmin": -100.0, "ymin": 40.0, "xmax": -99.0, "ymax": 41.0},
                ],
                type=bbox_type,
            ),
            "year": pa.array([2024, 2023], type=pa.int32()),
            "utm_zone": pa.array(["15N", "14N"]),
        }
    )
    pq.write_table(table, str(path))


def _write_partitioned_aef_collection(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    bbox_type = pa.struct(
        [
            pa.field("xmin", pa.float64()),
            pa.field("ymin", pa.float64()),
            pa.field("xmax", pa.float64()),
            pa.field("ymax", pa.float64()),
        ]
    )
    first = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array(
                [
                    {
                        "A00": {
                            "href": "s3://example-bucket/aef/scene-1.tif",
                            "band_index": 0,
                        }
                    }
                ]
            ),
            "bbox": pa.array(
                [{"xmin": -95.4, "ymin": 38.45, "xmax": -95.1, "ymax": 38.7}],
                type=bbox_type,
            ),
            "year": pa.array([2024], type=pa.int32()),
            "utm_zone": pa.array(["15N"]),
        }
    )
    second = pa.table(
        {
            "id": pa.array(["scene-2"]),
            "datetime": pa.array([datetime(2023, 1, 1)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array(
                [
                    {
                        "A00": {
                            "href": "s3://example-bucket/aef/scene-2.tif",
                            "band_index": 0,
                        }
                    }
                ]
            ),
            "bbox": pa.array(
                [{"xmin": -100.0, "ymin": 40.0, "xmax": -99.0, "ymax": 41.0}],
                type=bbox_type,
            ),
            "year": pa.array([2023], type=pa.int32()),
            "utm_zone": pa.array(["14N"]),
        }
    )
    pq.write_table(first, str(root / "part-00000.parquet"))
    pq.write_table(second, str(root / "part-00001.parquet"))


def test_load_valid_file() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test.parquet"
        _write_minimal_flat_collection(path)
        collection = rasteret.load(path)
        assert isinstance(collection, Collection)
        assert collection.name == "test"


def test_load_hf_uri_uses_hf_reader(monkeypatch) -> None:
    from rasteret.integrations.huggingface import HFStreamingSource

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("datetime", pa.timestamp("us")),
            pa.field("geometry", pa.null()),
            pa.field(
                "assets",
                pa.struct(
                    [pa.field("B04", pa.struct([pa.field("href", pa.string())]))]
                ),
            ),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )
    source = HFStreamingSource(
        path="hf://datasets/terrafloww/example/records.parquet",
        parquet_paths=("hf://datasets/terrafloww/example/records.parquet",),
        schema=schema,
    )

    with patch(
        "rasteret.core.collection.open_hf_streaming_source",
        return_value=source,
    ) as mocked:
        collection = rasteret.load("hf://datasets/terrafloww/example/records.parquet")

    mocked.assert_called_once_with("hf://datasets/terrafloww/example/records.parquet")
    assert isinstance(collection, Collection)
    assert collection.dataset is None
    assert collection._hf_streaming is source


def test_load_dataset_id_uses_descriptor_collection_uri() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "published_collection.parquet"
        _write_minimal_flat_collection(path)
        descriptor = DatasetDescriptor(
            id="demo/runtime-collection",
            name="Demo Runtime Collection",
            collection_uri=str(path),
        )
        DatasetRegistry.register(descriptor)
        try:
            collection = rasteret.load("demo/runtime-collection")
        finally:
            DatasetRegistry.unregister("demo/runtime-collection")

    assert isinstance(collection, Collection)
    assert collection.name == "Demo Runtime Collection"


def test_load_dataset_id_preserves_data_source_for_auto_backend() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "published_collection.parquet"
        _write_minimal_flat_collection(path)
        descriptor = DatasetDescriptor(
            id="demo/runtime-cloud",
            name="Demo Runtime Cloud",
            collection_uri=str(path),
            cloud_config={
                "provider": "aws",
                "region": "us-west-2",
                "url_patterns": {
                    "https://example.com/": "s3://example-bucket/",
                },
            },
        )
        DatasetRegistry.register(descriptor)
        try:
            collection = rasteret.load("demo/runtime-cloud")
        finally:
            DatasetRegistry.unregister("demo/runtime-cloud")

    assert collection.data_source == "demo/runtime-cloud"
    assert collection._auto_backend() is not None


def test_open_parquet_dataset_prefers_metadata_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    from rasteret.core.collection import _open_parquet_dataset

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    bbox_type = pa.struct(
        [
            ("xmin", pa.float64()),
            ("ymin", pa.float64()),
            ("xmax", pa.float64()),
            ("ymax", pa.float64()),
        ]
    )
    table = pa.table(
        {
            "id": pa.array(["a"], type=pa.string()),
            "bbox": pa.array(
                [{"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}],
                type=bbox_type,
            ),
        }
    )
    pq.write_table(table, dataset_dir / "part-00000.parquet")
    pq.write_metadata(table.schema, dataset_dir / "_common_metadata")
    pq.ParquetFile(dataset_dir / "part-00000.parquet").metadata.write_metadata_file(
        dataset_dir / "_metadata"
    )

    original = pads.parquet_dataset
    called: dict[str, str] = {}

    def _wrapped(path, *args, **kwargs):
        called["path"] = str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr("rasteret.core.collection.ds.parquet_dataset", _wrapped)
    dataset = _open_parquet_dataset(str(dataset_dir))
    assert dataset is not None
    assert called["path"].endswith("/_metadata")


def test_open_parquet_dataset_caches_local_dataset_until_file_changes(
    tmp_path: Path, monkeypatch
) -> None:
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    from rasteret.core.collection import _PARQUET_DATASET_CACHE, _open_parquet_dataset

    dataset_path = tmp_path / "single.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a"],
                "bbox": [{"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}],
            }
        ),
        dataset_path,
    )
    _PARQUET_DATASET_CACHE.clear()

    original = pads.dataset
    call_count = {"n": 0}

    def _wrapped(path, *args, **kwargs):
        call_count["n"] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr("rasteret.core.collection.ds.dataset", _wrapped)

    first = _open_parquet_dataset(str(dataset_path), try_hive=False)
    second = _open_parquet_dataset(str(dataset_path), try_hive=False)
    assert first is second
    assert call_count["n"] == 1

    pq.write_table(
        pa.table(
            {
                "id": ["b"],
                "bbox": [{"xmin": 2.0, "ymin": 2.0, "xmax": 3.0, "ymax": 3.0}],
            }
        ),
        dataset_path,
    )
    third = _open_parquet_dataset(str(dataset_path), try_hive=False)
    assert call_count["n"] == 2
    assert third is not first


def test_load_dataset_id_uses_record_index_for_head_and_subset() -> None:
    with TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "aef-index.parquet"
        data_path = Path(tmp_dir) / "aef-data.parquet"
        _write_minimal_aef_index(index_path)
        _write_minimal_aef_collection(data_path)

        descriptor = DatasetDescriptor(
            id="demo/aef-indexed",
            name="Demo AEF Indexed",
            record_table_uri=str(index_path),
            index_uri=str(index_path),
            collection_uri=str(data_path),
            column_map={"fid": "id", "geom": "geometry", "year": "datetime"},
            href_column="path",
            band_index_map={"A00": 0},
        )
        DatasetRegistry.register(descriptor)
        try:
            collection = rasteret.load("demo/aef-indexed")
            head = (
                collection.where(ds.field("year") == 2024)
                .subset(bbox=(-95.5, 38.4, -95.0, 38.8))
                .head(5, columns=["id", "year", "utm_zone", "bbox"])
            )
        finally:
            DatasetRegistry.unregister("demo/aef-indexed")

    assert head.column("id").to_pylist() == ["scene-1"]
    assert head.column("year").to_pylist() == [2024]
    assert head.column("utm_zone").to_pylist() == ["15N"]


def test_load_dataset_id_supports_field_roles_without_column_map() -> None:
    with TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "aef-index.parquet"
        data_path = Path(tmp_dir) / "aef-data.parquet"
        _write_minimal_aef_index(index_path)
        _write_minimal_aef_collection(data_path)

        descriptor = DatasetDescriptor(
            id="demo/aef-field-roles",
            name="Demo AEF Field Roles",
            record_table_uri=str(index_path),
            index_uri=str(index_path),
            collection_uri=str(data_path),
            field_roles={
                "id": "fid",
                "geometry": "geom",
                "datetime": "year",
                "bbox": "bbox",
                "href": "path",
            },
            band_index_map={"A00": 0},
        )
        DatasetRegistry.register(descriptor)
        try:
            collection = rasteret.load("demo/aef-field-roles")
            head = (
                collection.where(ds.field("year") == 2024)
                .subset(bbox=(-95.5, 38.4, -95.0, 38.8))
                .head(5, columns=["id", "year", "utm_zone", "bbox"])
            )
        finally:
            DatasetRegistry.unregister("demo/aef-field-roles")

    assert head.column("id").to_pylist() == ["scene-1"]
    assert head.column("year").to_pylist() == [2024]
    assert head.column("utm_zone").to_pylist() == ["15N"]


def test_load_dataset_id_head_does_not_open_wide_dataset(monkeypatch) -> None:
    with TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "aef-index.parquet"
        data_root = Path(tmp_dir) / "aef-data"
        _write_minimal_aef_index(index_path)
        _write_partitioned_aef_collection(data_root)

        descriptor = DatasetDescriptor(
            id="demo/aef-lazy-wide",
            name="Demo AEF Lazy Wide",
            record_table_uri=str(index_path),
            index_uri=str(index_path),
            collection_uri=str(data_root),
            column_map={"fid": "id", "geom": "geometry", "year": "datetime"},
            href_column="path",
            band_index_map={"A00": 0},
        )
        DatasetRegistry.register(descriptor)

        original_open = collection_module._open_parquet_dataset

        def _fail_on_wide(path: str, *args, **kwargs):
            if str(path) == str(data_root):
                raise AssertionError("wide dataset should stay unopened for head()")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(
            "rasteret.core.collection._open_parquet_dataset", _fail_on_wide
        )
        try:
            collection = rasteret.load("demo/aef-lazy-wide")
            head = (
                collection.where(ds.field("year") == 2024)
                .subset(bbox=(-95.5, 38.4, -95.0, 38.8))
                .head(1, columns=["id", "bbox"])
            )
        finally:
            DatasetRegistry.unregister("demo/aef-lazy-wide")

    assert head.column("id").to_pylist() == ["scene-1"]


def test_dual_surface_bands_do_not_open_wide_dataset(monkeypatch) -> None:
    with TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "aef-index.parquet"
        data_root = Path(tmp_dir) / "aef-data"
        _write_minimal_aef_index(index_path)
        _write_partitioned_aef_collection(data_root)

        descriptor = DatasetDescriptor(
            id="demo/aef-bands-lazy",
            name="Demo AEF Bands Lazy",
            record_table_uri=str(index_path),
            index_uri=str(index_path),
            collection_uri=str(data_root),
            field_roles={
                "id": "fid",
                "geometry": "geom",
                "datetime": "year",
                "bbox": "bbox",
                "href": "path",
                "proj:epsg": "crs",
            },
            band_index_map={"A00": 0, "A01": 1},
        )
        DatasetRegistry.register(descriptor)

        original_open = collection_module._open_parquet_dataset

        def _fail_on_wide(path: str, *args, **kwargs):
            if str(path) == str(data_root):
                raise AssertionError("wide dataset should stay unopened for bands")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(
            "rasteret.core.collection._open_parquet_dataset", _fail_on_wide
        )
        try:
            collection = rasteret.load("demo/aef-bands-lazy")
            bands = collection.bands
        finally:
            DatasetRegistry.unregister("demo/aef-bands-lazy")

    assert bands == ["A00", "A01"]


def test_dual_surface_epsg_uses_index_without_opening_wide_dataset(monkeypatch) -> None:
    with TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "aef-index.parquet"
        data_root = Path(tmp_dir) / "aef-data"
        _write_minimal_aef_index(index_path)
        _write_partitioned_aef_collection(data_root)

        descriptor = DatasetDescriptor(
            id="demo/aef-epsg-lazy",
            name="Demo AEF EPSG Lazy",
            record_table_uri=str(index_path),
            index_uri=str(index_path),
            collection_uri=str(data_root),
            field_roles={
                "id": "fid",
                "geometry": "geom",
                "datetime": "year",
                "bbox": "bbox",
                "href": "path",
                "proj:epsg": "crs",
            },
            band_index_map={"A00": 0},
        )
        DatasetRegistry.register(descriptor)

        original_open = collection_module._open_parquet_dataset

        def _fail_on_wide(path: str, *args, **kwargs):
            if str(path) == str(data_root):
                raise AssertionError("wide dataset should stay unopened for epsg")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(
            "rasteret.core.collection._open_parquet_dataset", _fail_on_wide
        )
        try:
            collection = rasteret.load("demo/aef-epsg-lazy")
            epsg = collection.epsg
        finally:
            DatasetRegistry.unregister("demo/aef-epsg-lazy")

    assert sorted(epsg) == [32614, 32615]


def test_dual_surface_subset_keeps_wide_filter_plan_without_opening_data() -> None:
    with TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "aef-index.parquet"
        data_root = Path(tmp_dir) / "aef-data"
        _write_minimal_aef_index(index_path)
        _write_partitioned_aef_collection(data_root)

        descriptor = DatasetDescriptor(
            id="demo/aef-wide-plan",
            name="Demo AEF Wide Plan",
            record_table_uri=str(index_path),
            index_uri=str(index_path),
            collection_uri=str(data_root),
            field_roles={
                "id": "fid",
                "geometry": "geom",
                "datetime": "year",
                "bbox": "bbox",
                "href": "path",
            },
            surface_fields={
                "index": [
                    "id",
                    "geometry",
                    "bbox",
                    "datetime",
                    "href",
                    "year",
                    "utm_zone",
                ],
                "collection": [
                    "id",
                    "geometry",
                    "bbox",
                    "datetime",
                    "assets",
                    "year",
                    "utm_zone",
                ],
            },
            filter_capabilities={
                "index": ["id", "bbox", "datetime", "year", "utm_zone"],
                "collection": ["id", "bbox", "datetime", "year", "utm_zone"],
            },
            band_index_map={"A00": 0},
        )
        DatasetRegistry.register(descriptor)
        try:
            collection = rasteret.load("demo/aef-wide-plan")
            assert collection.dataset is None
            filtered = collection.where(ds.field("year") == 2024).subset(
                bbox=(-95.5, 38.4, -95.0, 38.8)
            )
        finally:
            DatasetRegistry.unregister("demo/aef-wide-plan")

    assert filtered.dataset is None
    assert filtered._wide_filter_expr is not None


def test_load_dataset_id_uses_source_part_to_narrow_data_fragments(
    monkeypatch,
) -> None:
    with TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "aef-index.parquet"
        data_root = Path(tmp_dir) / "aef-data"
        _write_partitioned_aef_collection(data_root)
        bbox_type = pa.struct(
            [
                pa.field("xmin", pa.float64()),
                pa.field("ymin", pa.float64()),
                pa.field("xmax", pa.float64()),
                pa.field("ymax", pa.float64()),
            ]
        )
        index_table = pa.table(
            {
                "fid": pa.array(["scene-1", "scene-2"]),
                "year": pa.array([2024, 2023], type=pa.int32()),
                "utm_zone": pa.array(["15N", "14N"]),
                "geom": pa.array([None, None], type=pa.null()),
                "bbox": pa.array(
                    [
                        {"xmin": -95.4, "ymin": 38.45, "xmax": -95.1, "ymax": 38.7},
                        {"xmin": -100.0, "ymin": 40.0, "xmax": -99.0, "ymax": 41.0},
                    ],
                    type=bbox_type,
                ),
                "path": pa.array(
                    [
                        "s3://example-bucket/aef/scene-1.tif",
                        "s3://example-bucket/aef/scene-2.tif",
                    ]
                ),
                "crs": pa.array(["EPSG:32615", "EPSG:32614"]),
                "source_part": pa.array(["part-00000.parquet", "part-00001.parquet"]),
            }
        )
        pq.write_table(index_table, str(index_path))

        descriptor = DatasetDescriptor(
            id="demo/aef-parts",
            name="Demo AEF Parts",
            record_table_uri=str(index_path),
            index_uri=str(index_path),
            collection_uri=str(data_root),
            column_map={"fid": "id", "geom": "geometry", "year": "datetime"},
            href_column="path",
            band_index_map={"A00": 0},
        )
        DatasetRegistry.register(descriptor)
        captured_paths: list[str] = []
        original_dataset = ds.dataset

        def _capture_dataset(source, *args, **kwargs):
            if isinstance(source, list):
                captured_paths[:] = list(source)
            return original_dataset(source, *args, **kwargs)

        monkeypatch.setattr("rasteret.core.collection.ds.dataset", _capture_dataset)
        try:
            collection = rasteret.load("demo/aef-parts")
            filtered = collection.where(ds.field("year") == 2024).subset(
                bbox=(-95.5, 38.4, -95.0, 38.8)
            )
            narrowed = filtered._filtered_data_dataset()
            assert narrowed is not None
            assert narrowed.count_rows() == 1
        finally:
            DatasetRegistry.unregister("demo/aef-parts")

    assert len(captured_paths) == 1
    assert captured_paths[0].endswith("part-00000.parquet")


def test_build_geoparquet_descriptor_uses_struct_bbox_when_no_bbox_columns(
    monkeypatch,
) -> None:
    descriptor = DatasetDescriptor(
        id="demo/indexed-build",
        name="Demo Indexed Build",
        record_table_uri="s3://example-bucket/demo/index.parquet",
        field_roles={
            "id": "fid",
            "geometry": "geom",
            "datetime": "year",
            "bbox": "bbox",
            "href": "path",
        },
        band_index_map={"A00": 0},
        requires_auth=False,
        cloud_config={"provider": "aws", "region": "us-west-2"},
    )
    DatasetRegistry.register(descriptor)
    captured: dict[str, object] = {}

    def _fake_build_from_table(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    fake_schema = pa.schema(
        [
            pa.field("year", pa.int32()),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        "pyarrow.dataset.dataset",
        lambda *args, **kwargs: SimpleNamespace(schema=fake_schema),
    )
    monkeypatch.setattr(rasteret, "build_from_table", _fake_build_from_table)
    try:
        result = rasteret.build(
            "demo/indexed-build",
            name="demo",
            bbox=(-95.5, 38.4, -95.0, 38.8),
            date_range=("2024-01-01", "2024-12-31"),
        )
    finally:
        DatasetRegistry.unregister("demo/indexed-build")

    assert result == "ok"
    kwargs = captured["kwargs"]
    assert captured["args"][0] == "example-bucket/demo/index.parquet"
    expr_str = str(kwargs["filter_expr"])
    assert "FieldRef.Name(bbox)" in expr_str
    assert "FieldRef.Name(xmax)" in expr_str
    assert "FieldRef.Name(xmin)" in expr_str
    assert "FieldRef.Name(ymax)" in expr_str
    assert "FieldRef.Name(ymin)" in expr_str
    assert "(year >= 2024)" in expr_str
    assert "(year <= 2024)" in expr_str


def test_build_geoparquet_descriptor_uses_field_role_bbox_source(monkeypatch) -> None:
    descriptor = DatasetDescriptor(
        id="demo/indexed-build-bounds",
        name="Demo Indexed Build Bounds",
        record_table_uri="s3://example-bucket/demo/index.parquet",
        field_roles={
            "id": "fid",
            "geometry": "geom",
            "datetime": "year",
            "bbox": "scene_bounds",
            "href": "path",
        },
        band_index_map={"A00": 0},
        requires_auth=False,
        cloud_config={"provider": "aws", "region": "us-west-2"},
    )
    DatasetRegistry.register(descriptor)
    captured: dict[str, object] = {}

    def _fake_build_from_table(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    fake_schema = pa.schema(
        [
            pa.field("year", pa.int32()),
            pa.field(
                "scene_bounds",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        "pyarrow.dataset.dataset",
        lambda *args, **kwargs: SimpleNamespace(schema=fake_schema),
    )
    monkeypatch.setattr(rasteret, "build_from_table", _fake_build_from_table)
    try:
        result = rasteret.build(
            "demo/indexed-build-bounds",
            name="demo",
            bbox=(-95.5, 38.4, -95.0, 38.8),
            date_range=("2024-01-01", "2024-12-31"),
        )
    finally:
        DatasetRegistry.unregister("demo/indexed-build-bounds")

    assert result == "ok"
    expr_str = str(captured["kwargs"]["filter_expr"])
    assert "FieldRef.Name(scene_bounds)" in expr_str
    assert "FieldRef.Name(xmax)" in expr_str
    assert "FieldRef.Name(xmin)" in expr_str


def test_build_geoparquet_descriptor_uses_datetime_field_role_even_when_named_datetime(
    monkeypatch,
) -> None:
    descriptor = DatasetDescriptor(
        id="demo/indexed-build-datetime",
        name="Demo Indexed Build Datetime",
        record_table_uri="s3://example-bucket/demo/index.parquet",
        field_roles={
            "id": "fid",
            "geometry": "geom",
            "datetime": "datetime",
            "bbox": "bbox",
            "href": "path",
        },
        band_index_map={"A00": 0},
        requires_auth=False,
        cloud_config={"provider": "aws", "region": "us-west-2"},
    )
    DatasetRegistry.register(descriptor)
    captured: dict[str, object] = {}

    def _fake_build_from_table(*args, **kwargs):
        captured["kwargs"] = kwargs
        return "ok"

    fake_schema = pa.schema(
        [
            pa.field("datetime", pa.timestamp("us")),
            pa.field(
                "bbox",
                pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        "pyarrow.dataset.dataset",
        lambda *args, **kwargs: SimpleNamespace(schema=fake_schema),
    )
    monkeypatch.setattr(rasteret, "build_from_table", _fake_build_from_table)
    try:
        result = rasteret.build(
            "demo/indexed-build-datetime",
            name="demo",
            bbox=(-95.5, 38.4, -95.0, 38.8),
            date_range=("2024-01-01", "2024-12-31"),
        )
    finally:
        DatasetRegistry.unregister("demo/indexed-build-datetime")

    assert result == "ok"
    expr_str = str(captured["kwargs"]["filter_expr"])
    assert "datetime >=" in expr_str
    assert "datetime <=" in expr_str
    assert "2024-01-01" in expr_str
    assert "2024-12-31" in expr_str


def test_collection_head_returns_table() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test.parquet"
        _write_minimal_flat_collection(path)
        collection = rasteret.load(path)
        head = collection.head(1, columns=["id"])

    assert isinstance(head, pa.Table)
    assert head.column_names == ["id"]
    assert head.column("id").to_pylist() == ["scene-1"]


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
            "bbox": pa.array(
                [{"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}],
                type=pa.struct(
                    [
                        pa.field("xmin", pa.float64()),
                        pa.field("ymin", pa.float64()),
                        pa.field("xmax", pa.float64()),
                        pa.field("ymax", pa.float64()),
                    ]
                ),
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


def test_as_collection_wraps_record_batch() -> None:
    batch = _minimal_read_ready_table().to_batches()[0]
    collection = rasteret.as_collection(batch, name="wrapped-batch")
    assert isinstance(collection, Collection)
    assert collection.name == "wrapped-batch"
    assert collection.dataset is not None
    assert collection.dataset.count_rows() == 1


def test_as_collection_wraps_record_batch_reader() -> None:
    reader = _minimal_read_ready_table().to_reader()
    collection = rasteret.as_collection(reader, name="wrapped-reader")
    assert isinstance(collection, Collection)
    assert collection.name == "wrapped-reader"
    assert collection.dataset is not None
    assert collection.dataset.count_rows() == 1


def test_as_collection_wraps_arrow_capsule_stream_object() -> None:
    reader = _minimal_read_ready_table().to_reader()

    class _StreamObj:
        def __arrow_c_stream__(self, requested_schema=None):
            return reader.__arrow_c_stream__(requested_schema)

    collection = rasteret.as_collection(_StreamObj(), name="wrapped-stream")
    assert isinstance(collection, Collection)
    assert collection.name == "wrapped-stream"
    assert collection.dataset is not None
    assert collection.dataset.count_rows() == 1


def test_as_collection_wraps_arrow_capsule_array_object() -> None:
    batch = _minimal_read_ready_table().to_batches()[0]

    class _ArrayObj:
        def __arrow_c_array__(self, requested_schema=None):
            return batch.__arrow_c_array__(requested_schema)

    collection = rasteret.as_collection(_ArrayObj(), name="wrapped-array")
    assert isinstance(collection, Collection)
    assert collection.name == "wrapped-array"
    assert collection.dataset is not None
    assert collection.dataset.count_rows() == 1


def test_as_collection_requires_bbox() -> None:
    table = _minimal_read_ready_table().drop_columns(["bbox"])
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
