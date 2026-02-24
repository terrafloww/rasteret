# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Network smoke tests: verify real cloud reads work end-to-end.

These tests are **skipped by default**. Pass ``--network`` to include them::

    uv run pytest --network        # unit + network tests
    uv run pytest --network -v     # verbose
    uv run pytest -m network       # network tests only (with --network implied)

Tests that require credentials or DNS access to specific endpoints
will ``pytest.skip()`` automatically when the prerequisites are missing.
Use ``--network-strict`` to treat those skips as failures (useful in CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rasteret
from rasteret.catalog import DatasetRegistry

pytestmark = pytest.mark.network


class _AlarmTimeout(Exception):
    pass


def _timeout(seconds: int):
    """Best-effort wall-clock timeout for live network tests (POSIX only)."""
    import contextlib
    import signal

    @contextlib.contextmanager
    def _ctx():
        if not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(_signum, _frame):  # type: ignore[no-untyped-def]
            raise _AlarmTimeout(f"Timed out after {seconds}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    return _ctx()


def _assert_has_tiled_band_metadata(
    collection: rasteret.Collection, band_code: str
) -> None:
    col_name = f"{band_code}_metadata"
    assert collection.dataset is not None
    assert col_name in collection.dataset.schema.names
    table = collection.dataset.to_table(columns=[col_name])
    assert table.num_rows >= 1
    values = table.column(col_name).to_pylist()
    first = next((v for v in values if v is not None), None)
    assert first is not None, f"{col_name} is all-null (enrichment failed)"
    assert (first.get("tile_width") or 0) > 0
    assert (first.get("tile_height") or 0) > 0
    assert first.get("tile_offsets"), "tile_offsets empty (not a tiled TIFF?)"


class TestLiveCollections:
    """Live collection builds covering Earth Search, Planetary Computer, and Landsat."""

    _SAN_FRAN_BBOX = (-122.45, 37.74, -122.35, 37.84)
    _DATE_RANGE = ("2024-06-01", "2024-07-15")
    _SINGLE_SCENE_QUERY = {"max_items": 1}

    def test_earthsearch_sentinel2(self, tmp_path: Path) -> None:
        with _timeout(90):
            collection = rasteret.build(
                "earthsearch/sentinel-2-l2a",
                name="network-earthsearch-s2",
                bbox=self._SAN_FRAN_BBOX,
                date_range=self._DATE_RANGE,
                workspace_dir=tmp_path / "network-earthsearch-s2",
                force=True,
                query=self._SINGLE_SCENE_QUERY,
            )
        assert collection.dataset is not None
        assert collection.dataset.count_rows() >= 1
        _assert_has_tiled_band_metadata(collection, "B04")

    def test_planetary_computer_sentinel2(self, tmp_path: Path) -> None:
        pytest.importorskip(
            "planetary_computer",
            reason="planetary-computer extra is required to sign Planetary Computer assets",
        )
        with _timeout(120):
            collection = rasteret.build(
                "pc/sentinel-2-l2a",
                name="network-pc-s2",
                bbox=self._SAN_FRAN_BBOX,
                date_range=self._DATE_RANGE,
                workspace_dir=tmp_path / "network-pc-s2",
                force=True,
                max_concurrent=8,
                query=self._SINGLE_SCENE_QUERY,
            )
        assert collection.dataset is not None
        assert collection.dataset.count_rows() >= 1
        _assert_has_tiled_band_metadata(collection, "B04")

    def test_landsat_requester_pays(self, tmp_path: Path) -> None:
        try:
            import boto3

            has_creds = boto3.Session().get_credentials() is not None
        except Exception:
            has_creds = False
        if not has_creds:
            pytest.skip("AWS credentials required for Landsat requester-pays assets.")
        with _timeout(180):
            collection = rasteret.build(
                "earthsearch/landsat-c2-l2",
                name="network-landsat-requester",
                bbox=self._SAN_FRAN_BBOX,
                date_range=self._DATE_RANGE,
                workspace_dir=tmp_path / "network-landsat",
                force=True,
                max_concurrent=8,
                query=self._SINGLE_SCENE_QUERY,
            )
        assert collection.dataset is not None
        assert collection.dataset.count_rows() >= 1
        _assert_has_tiled_band_metadata(collection, "B4")


@pytest.mark.parametrize(
    "dataset_id",
    sorted(d.id for d in DatasetRegistry.list()),
)
def test_live_all_catalog_datasets_build_and_enrich(
    tmp_path: Path, dataset_id: str
) -> None:
    """Build every catalog dataset and verify at least one band metadata column is populated.

    This is intentionally a smoke check: it uses `max_items=1` to keep requests small.
    """
    descriptor = DatasetRegistry.get(dataset_id)
    assert descriptor is not None

    # Dependency / credential guards.
    if dataset_id.startswith("pc/"):
        pytest.importorskip(
            "planetary_computer",
            reason="planetary-computer extra is required to sign Planetary Computer assets",
        )
    if descriptor.cloud_config and descriptor.cloud_config.get("requester_pays"):
        try:
            import boto3

            has_creds = boto3.Session().get_credentials() is not None
        except Exception:
            has_creds = False
        if not has_creds:
            pytest.skip("AWS credentials required for requester-pays S3 access.")

    band_code = next(
        iter((descriptor.band_map or descriptor.band_index_map or {}).keys()), None
    )
    if not band_code:
        pytest.skip("Descriptor has no band configuration; nothing to enrich.")

    if descriptor.example_bbox is None or descriptor.example_date_range is None:
        pytest.skip("Descriptor missing example bbox/date_range for live smoke test.")

    bbox = descriptor.example_bbox
    date_range = descriptor.example_date_range

    with _timeout(180):
        collection = rasteret.build(
            dataset_id,
            name=f"network-{dataset_id.split('/', 1)[-1]}",
            bbox=bbox,
            date_range=date_range,
            workspace_dir=tmp_path / dataset_id.replace("/", "-"),
            force=True,
            max_concurrent=8,
            query={"max_items": 1},
        )

    _assert_has_tiled_band_metadata(collection, band_code)
