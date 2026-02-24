# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rasteret.cli import main
from rasteret.core.collection import Collection


def _write_cached_collection(path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array(["scene-1", "scene-2"]),
            "datetime": pa.array(
                [datetime(2024, 1, 10), datetime(2024, 1, 11)],
                type=pa.timestamp("us"),
            ),
            "geometry": pa.array([None, None], type=pa.null()),
            "assets": pa.array(
                [
                    {"B04": {"href": "https://example.com/s1.tif"}},
                    {"B04": {"href": "https://example.com/s2.tif"}},
                ]
            ),
            "scene_bbox": pa.array(
                [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                type=pa.list_(pa.float64(), 4),
            ),
            "collection": pa.array(["sentinel-2-l2a", "sentinel-2-l2a"]),
            "year": pa.array([2024, 2024], type=pa.int32()),
            "month": pa.array([1, 1], type=pa.int32()),
            "split": pa.array(["train", "val"]),
        }
    )
    pq.write_to_dataset(table, root_path=str(path), partition_cols=["year", "month"])


def test_cache_list_outputs_cached_collection(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "demo_202401-01_sentinel_stac"
    _write_cached_collection(cache_dir)

    exit_code = main(["cache", "list", "--workspace-dir", str(tmp_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "demo_202401-01_sentinel" in output


def test_cache_info_json(tmp_path, capsys) -> None:
    cache_name = "demo_202401-01_sentinel"
    cache_dir = tmp_path / f"{cache_name}_stac"
    _write_cached_collection(cache_dir)

    exit_code = main(
        ["cache", "info", cache_name, "--workspace-dir", str(tmp_path), "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == cache_name
    assert payload["scene_count"] == 2
    assert payload["has_split_column"] is True


def test_cache_delete_yes_removes_collection(tmp_path, capsys) -> None:
    cache_name = "demo_202401-01_sentinel"
    cache_dir = tmp_path / f"{cache_name}_stac"
    _write_cached_collection(cache_dir)
    assert cache_dir.exists()

    exit_code = main(
        ["cache", "delete", cache_name, "--workspace-dir", str(tmp_path), "--yes"]
    )
    assert exit_code == 0
    assert not cache_dir.exists()
    assert "Deleted:" in capsys.readouterr().out


def test_cache_build_passes_args_and_returns_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}
    cache_name = "demo_202401-01_sentinel"
    cache_dir = tmp_path / f"{cache_name}_stac"

    def fake_build_from_stac(**kwargs):
        captured.update(kwargs)
        _write_cached_collection(cache_dir)
        return Collection.from_local(cache_dir)

    monkeypatch.setattr("rasteret.cli.build_from_stac", fake_build_from_stac)

    exit_code = main(
        [
            "cache",
            "build",
            "demo",
            "--stac-api",
            "https://example.invalid/stac",
            "--collection",
            "sentinel-2-l2a",
            "--bbox",
            "0,0,1,1",
            "--date-range",
            "2024-01-01,2024-01-31",
            "--workspace-dir",
            str(tmp_path),
            "--query",
            '{"eo:cloud_cover":{"lt":20}}',
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scene_count"] == 2
    assert captured["name"] == "demo"
    assert captured["collection"] == "sentinel-2-l2a"
    assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
    assert captured["date_range"] == ("2024-01-01", "2024-01-31")
    assert captured["query"] == {"eo:cloud_cover": {"lt": 20}}


def test_cache_import_materializes_record_table(tmp_path, capsys) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    record_table_path = input_dir / "items.parquet"
    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([datetime(2024, 1, 10)], type=pa.timestamp("us")),
            "geometry": pa.array([None], type=pa.null()),
            "assets": pa.array([{"B04": {"href": "https://example.com/s1.tif"}}]),
            "scene_bbox": pa.array(
                [[0.0, 0.0, 1.0, 1.0]], type=pa.list_(pa.float64(), 4)
            ),
        }
    )
    pq.write_table(table, record_table_path)

    exit_code = main(
        [
            "cache",
            "import",
            "demo",
            "--record-table",
            str(record_table_path),
            "--workspace-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scene_count"] == 1
    assert (tmp_path / "demo_records").exists()


def test_datasets_register_local_persists_descriptor(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "demo_202401-01_sentinel_stac"
    registry_path = tmp_path / "datasets.local.json"
    _write_cached_collection(cache_dir)

    exit_code = main(
        [
            "datasets",
            "register-local",
            "local/demo",
            str(cache_dir),
            "--registry-path",
            str(registry_path),
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "local/demo"
    assert payload["geoparquet_uri"] == str(cache_dir.resolve())
    assert registry_path.exists()


def test_datasets_build_local_without_bbox_or_date(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "demo_202401-01_sentinel_stac"
    _write_cached_collection(cache_dir)

    register_exit = main(
        [
            "datasets",
            "register-local",
            "local/demo-no-bbox",
            str(cache_dir),
            "--no-persist",
            "--json",
        ]
    )
    assert register_exit == 0
    _ = json.loads(capsys.readouterr().out)

    build_exit = main(
        [
            "datasets",
            "build",
            "local/demo-no-bbox",
            "copied-demo",
            "--workspace-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert build_exit == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scene_count"] == 2


def test_datasets_export_local_writes_descriptor_json(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "demo_202401-01_sentinel_stac"
    registry_path = tmp_path / "datasets.local.json"
    export_path = tmp_path / "exported-descriptor.json"
    _write_cached_collection(cache_dir)

    register_exit = main(
        [
            "datasets",
            "register-local",
            "local/export-cli",
            str(cache_dir),
            "--registry-path",
            str(registry_path),
            "--json",
        ]
    )
    assert register_exit == 0
    _ = json.loads(capsys.readouterr().out)

    export_exit = main(
        [
            "datasets",
            "export-local",
            "local/export-cli",
            str(export_path),
            "--registry-path",
            str(registry_path),
            "--json",
        ]
    )
    assert export_exit == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "local/export-cli"
    assert payload["path"] == str(export_path)

    descriptor_payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert descriptor_payload["id"] == "local/export-cli"


def test_datasets_unregister_local_removes_descriptor(tmp_path, capsys) -> None:
    from rasteret.catalog import DatasetRegistry

    cache_dir = tmp_path / "demo_202401-01_sentinel_stac"
    registry_path = tmp_path / "datasets.local.json"
    _write_cached_collection(cache_dir)

    register_exit = main(
        [
            "datasets",
            "register-local",
            "local/unregister-cli",
            str(cache_dir),
            "--registry-path",
            str(registry_path),
        ]
    )
    assert register_exit == 0
    _ = capsys.readouterr().out
    assert DatasetRegistry.get("local/unregister-cli") is not None

    unregister_exit = main(
        [
            "datasets",
            "unregister-local",
            "local/unregister-cli",
            "--registry-path",
            str(registry_path),
            "--json",
        ]
    )
    assert unregister_exit == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "local/unregister-cli"
    assert DatasetRegistry.get("local/unregister-cli") is None

    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry_payload == []
