# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for DatasetRegistry, DatasetDescriptor, and build() API."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rasteret.catalog import (
    DatasetDescriptor,
    DatasetRegistry,
    export_local_descriptor,
    unregister_local_descriptor,
)
from rasteret.cloud import CloudConfig
from rasteret.constants import BandRegistry


def _write_minimal_collection(path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 10)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/s1.tif"}}]),
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
            "year": pa.array([2024], type=pa.int32()),
            "month": pa.array([1], type=pa.int32()),
        }
    )
    pq.write_to_dataset(table, root_path=str(path), partition_cols=["year", "month"])


class TestDatasetDescriptor:
    def test_frozen(self):
        d = DatasetDescriptor(id="test/x", name="X")
        with pytest.raises(AttributeError):
            d.id = "changed"

    def test_defaults(self):
        d = DatasetDescriptor(id="test/x", name="X")
        assert d.stac_api is None
        assert d.band_map is None
        assert d.separate_files is True
        assert d.requires_auth is False
        assert d.torchgeo_verified is False
        assert d.license == ""
        assert d.license_url == ""
        assert d.commercial_use is True
        assert d.static_catalog is False

    def test_field_roles_infer_compatibility_hints(self):
        d = DatasetDescriptor(
            id="test/roles",
            name="Roles",
            field_roles={
                "id": "fid",
                "geometry": "geom",
                "datetime": "year",
                "href": "path",
            },
        )
        assert d.column_map == {"fid": "id", "geom": "geometry", "year": "datetime"}
        assert d.href_column == "path"
        assert d.source_field("id") == "fid"
        assert d.source_field("datetime") == "year"
        assert d.source_field("href") == "path"

    def test_field_roles_infer_proj_epsg_alias(self):
        d = DatasetDescriptor(
            id="test/epsg-role",
            name="EPSG Role",
            field_roles={"proj:epsg": "crs"},
        )
        assert d.column_map == {"crs": "proj:epsg"}
        assert d.source_field("proj:epsg") == "crs"

    def test_invalid_surface_fields_raise_clear_error(self):
        with pytest.raises(ValueError, match="surface_fields contains unsupported"):
            DatasetDescriptor(
                id="test/bad-surface",
                name="Bad Surface",
                surface_fields={"wide_data": ["id"]},
            )

    def test_invalid_field_roles_raise_clear_error(self):
        with pytest.raises(ValueError, match="field_roles must map non-empty"):
            DatasetDescriptor(
                id="test/bad-roles",
                name="Bad Roles",
                field_roles={"id": ""},
            )


class TestDatasetRegistry:
    def test_builtin_sentinel2_registered(self):
        d = DatasetRegistry.get("earthsearch/sentinel-2-l2a")
        assert d is not None
        assert d.name == "Sentinel-2 Level-2A"
        assert d.stac_collection == "sentinel-2-l2a"
        assert d.license == "proprietary"
        assert d.license_url != ""

    def test_builtin_landsat_registered(self):
        d = DatasetRegistry.get("earthsearch/landsat-c2-l2")
        assert d is not None
        assert d.requires_auth is True

    def test_get_missing_returns_none(self):
        assert DatasetRegistry.get("nonexistent/dataset") is None

    def test_list_returns_all_builtins(self):
        descriptors = DatasetRegistry.list()
        ids = {d.id for d in descriptors}
        assert "earthsearch/sentinel-2-l2a" in ids
        assert "earthsearch/landsat-c2-l2" in ids
        assert len(descriptors) >= 12

    def test_search_by_keyword(self):
        results = DatasetRegistry.search("sentinel")
        ids = {d.id for d in results}
        assert "earthsearch/sentinel-2-l2a" in ids

    def test_search_case_insensitive(self):
        results = DatasetRegistry.search("LANDSAT")
        assert len(results) >= 1

    def test_search_no_match(self):
        assert DatasetRegistry.search("zzz_nonexistent_zzz") == []

    def test_register_custom(self):
        DatasetRegistry.register(
            DatasetDescriptor(
                id="custom/test-dataset",
                name="Test Dataset",
                stac_api="https://example.com/stac",
                stac_collection="test-col",
                band_map={"R": "red", "G": "green"},
            )
        )
        d = DatasetRegistry.get("custom/test-dataset")
        assert d is not None
        assert d.name == "Test Dataset"

    def test_register_populates_band_registry(self):
        DatasetRegistry.register(
            DatasetDescriptor(
                id="custom/band-test",
                name="Band Test",
                stac_collection="band-test-col",
                band_map={"X1": "xband"},
            )
        )
        bands = BandRegistry.get("custom/band-test")
        assert bands.get("X1") == "xband"

    def test_static_catalog_field(self):
        DatasetRegistry.register(
            DatasetDescriptor(
                id="custom/static-test",
                name="Static Test",
                stac_api="https://example.com/catalog.json",
                static_catalog=True,
                license="CC-BY-4.0",
            )
        )
        d = DatasetRegistry.get("custom/static-test")
        assert d is not None
        assert d.static_catalog is True
        assert d.license == "CC-BY-4.0"

    def test_register_populates_cloud_config(self):
        DatasetRegistry.register(
            DatasetDescriptor(
                id="custom/cloud-test",
                name="Cloud Test",
                stac_collection="cloud-test-col",
                cloud_config={
                    "provider": "aws",
                    "requester_pays": True,
                    "region": "eu-west-1",
                },
            )
        )
        config = CloudConfig.get_config("custom/cloud-test")
        assert config is not None
        assert config.requester_pays is True
        assert config.region == "eu-west-1"


class TestBuildAPI:
    def test_build_rejects_unknown_dataset(self):
        import rasteret

        with pytest.raises(KeyError, match="nonexistent"):
            rasteret.build(
                "nonexistent/dataset",
                name="x",
                bbox=(0, 0, 1, 1),
                date_range=("2024-01-01", "2024-01-31"),
            )

    def test_register_and_retrieve(self):
        import rasteret

        rasteret.register(
            DatasetDescriptor(
                id="custom/reg-test",
                name="Reg Test",
                stac_collection="reg-test-col",
                band_map={"A": "alpha"},
            )
        )
        d = DatasetRegistry.get("custom/reg-test")
        assert d is not None
        assert d.band_map == {"A": "alpha"}

    def test_register_local_persists_descriptor(self, tmp_path):
        import rasteret

        collection_path = tmp_path / "shared_collection"
        registry_path = tmp_path / "datasets.local.json"
        _write_minimal_collection(collection_path)

        descriptor = rasteret.register_local(
            "local/shared",
            collection_path,
            persist=True,
            registry_path=registry_path,
        )
        assert descriptor.id == "local/shared"
        assert descriptor.collection_uri == str(collection_path.resolve())
        assert DatasetRegistry.get("local/shared") is not None

        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        assert any(entry.get("id") == "local/shared" for entry in payload)

    def test_build_geoparquet_descriptor_without_stac_args(self, tmp_path):
        import rasteret

        collection_path = tmp_path / "local_build"
        _write_minimal_collection(collection_path)
        rasteret.register_local("local/build-test", collection_path, persist=False)

        collection = rasteret.build("local/build-test", name="local-build")
        assert collection.dataset is not None
        assert collection.dataset.count_rows() == 1

    def test_unregister_local_descriptor_removes_runtime_and_persisted(
        self, tmp_path
    ) -> None:
        import rasteret

        collection_path = tmp_path / "local_unregister"
        registry_path = tmp_path / "datasets.local.json"
        _write_minimal_collection(collection_path)
        rasteret.register_local(
            "local/unregister-test",
            collection_path,
            persist=True,
            registry_path=registry_path,
        )

        removed = unregister_local_descriptor(
            "local/unregister-test",
            path=registry_path,
        )
        assert removed is not None
        assert removed.id == "local/unregister-test"
        assert DatasetRegistry.get("local/unregister-test") is None

        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        assert payload == []

    def test_export_local_descriptor_writes_json(self, tmp_path) -> None:
        import rasteret

        collection_path = tmp_path / "local_export"
        export_path = tmp_path / "local-export.json"
        _write_minimal_collection(collection_path)
        rasteret.register_local("local/export-test", collection_path, persist=False)

        output = export_local_descriptor("local/export-test", export_path)
        assert output == export_path

        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert payload["id"] == "local/export-test"
        assert payload["collection_uri"] == str(collection_path.resolve())


class TestPublicAPISurface:
    def test_build_in_dir(self):
        import rasteret

        exported = set(dir(rasteret))
        assert "build" in exported
        assert "register" in exported
        assert "register_local" in exported

    def test_descriptor_accessible(self):
        import rasteret

        assert rasteret.DatasetDescriptor is DatasetDescriptor
        assert rasteret.DatasetRegistry is DatasetRegistry
