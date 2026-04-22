# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Compatibility matrix smoke tests (no network).

These tests are intentionally higher level than most unit tests: they
validate the end-to-end wiring of build-time enrichment across the
catalog/router/builder/parser stack for the most common dataset/auth
combinations.

They avoid real network I/O by:
- Stubbing STAC search results (pystac_client)
- Stubbing COG header parsing results (AsyncCOGHeaderParser)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@dataclass
class _DummyMeta:
    width: int = 512
    height: int = 512
    tile_width: int = 256
    tile_height: int = 256
    dtype: str = "uint16"
    transform: list[float] | None = None
    predictor: int = 1
    compression: int = 1
    tile_offsets: list[int] = None  # type: ignore[assignment]
    tile_byte_counts: list[int] = None  # type: ignore[assignment]
    pixel_scale: tuple[float, ...] | None = None
    tiepoint: tuple[float, ...] | None = None
    crs: int | None = None
    nodata: float | int | None = None
    samples_per_pixel: int = 1
    planar_configuration: int = 1
    photometric: int | None = None
    extra_samples: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.tile_offsets is None:
            self.tile_offsets = [0, 1, 2]
        if self.tile_byte_counts is None:
            self.tile_byte_counts = [10, 10, 10]


class _FakeSearch:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def items_as_dicts(self) -> list[dict[str, Any]]:
        return list(self._items)


class _FakeClient:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def search(self, **_kwargs: Any) -> _FakeSearch:
        return _FakeSearch(self._items)


def _stub_stac(monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]]) -> None:
    import pystac_client

    monkeypatch.setattr(
        pystac_client.Client, "open", lambda *_a, **_k: _FakeClient(items)
    )


def _stub_header_parser(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub AsyncCOGHeaderParser.process_cog_headers_batch to avoid I/O.

    Returns a dict capturing init kwargs for assertions (e.g. backend threading).
    """
    captured: dict[str, Any] = {}

    from rasteret.fetch import header_parser as hp

    orig_init = hp.AsyncCOGHeaderParser.__init__

    def _init(self, *args: Any, **kwargs: Any) -> None:
        captured["init_kwargs"] = dict(kwargs)
        orig_init(self, *args, **kwargs)

    async def _process(self, urls: list[str]) -> list[_DummyMeta | None]:
        captured["urls"] = list(urls)
        return [_DummyMeta() for _ in urls]

    monkeypatch.setattr(hp.AsyncCOGHeaderParser, "__init__", _init, raising=True)
    monkeypatch.setattr(
        hp.AsyncCOGHeaderParser, "process_cog_headers_batch", _process, raising=True
    )
    return captured


def _minimal_item(
    *,
    item_id: str = "scene-1",
    collection: str = "sentinel-2-l2a",
    hrefs: dict[str, str],
) -> dict[str, Any]:
    return {
        "id": item_id,
        "collection": collection,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "properties": {
            "datetime": "2024-01-10T00:00:00Z",
            "eo:cloud_cover": 0.0,
            "proj:code": "EPSG:4326",
        },
        "assets": {k: {"href": v} for k, v in hrefs.items()},
    }


def test_matrix_public_stac_https_build_wires_enrichment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public STAC dataset: no auth, HTTPS hrefs, no backend needed."""
    import rasteret

    captured = _stub_header_parser(monkeypatch)
    _stub_stac(
        monkeypatch,
        [
            _minimal_item(
                hrefs={
                    # BandRegistry for sentinel-2-l2a maps B04->red and B08->nir.
                    "red": "https://example.com/B04.tif",
                    "nir": "https://example.com/B08.tif",
                }
            )
        ],
    )

    collection = rasteret.build_from_stac(
        name="public",
        stac_api="https://earth-search.aws.element84.com/v1",
        collection="sentinel-2-l2a",
        bbox=(0, 0, 1, 1),
        date_range=("2024-01-01", "2024-01-31"),
        workspace_dir=tmp_path,
        force=True,
    )

    assert collection.dataset is not None
    assert "B04_metadata" in collection.dataset.schema.names
    assert captured["init_kwargs"].get("backend") is None
    assert captured["urls"] == [
        "https://example.com/B04.tif",
        "https://example.com/B08.tif",
    ]


def test_matrix_planetary_computer_presigned_https_does_not_require_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planetary Computer: SAS-signed HTTPS hrefs should work without backend."""
    pytest.importorskip("planetary_computer")
    import rasteret

    captured = _stub_header_parser(monkeypatch)
    _stub_stac(
        monkeypatch,
        [
            _minimal_item(
                collection="sentinel-2-l2a",
                hrefs={
                    # Planetary Computer Sentinel-2 uses band-code asset keys.
                    "B04": "https://pc.example.com/B04.tif?sig=abc",
                    "B08": "https://pc.example.com/B08.tif?sig=def",
                },
            )
        ],
    )

    collection = rasteret.build(
        "pc/sentinel-2-l2a",
        name="pc",
        bbox=(0, 0, 1, 1),
        date_range=("2024-01-01", "2024-01-31"),
        workspace_dir=tmp_path,
        force=True,
    )

    assert collection.dataset is not None
    assert captured["init_kwargs"].get("backend") is None
    # Only assets present in the STAC item are parsed.
    assert captured["urls"] == [
        "https://pc.example.com/B04.tif?sig=abc",
        "https://pc.example.com/B08.tif?sig=def",
    ]


def test_matrix_geoparquet_band_codes_prevent_non_band_enrichment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GeoParquet: descriptor band_codes should gate which assets are enriched."""
    import shapely
    from shapely.geometry import Polygon

    import rasteret
    from rasteret.catalog import DatasetDescriptor

    captured = _stub_header_parser(monkeypatch)

    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    geom_wkb = shapely.to_wkb(geom)

    table = pa.table(
        {
            "id": pa.array(["scene-1"]),
            "datetime": pa.array([None], type=pa.null()),
            "geometry": pa.array([geom_wkb], type=pa.binary()),
            "assets": pa.array(
                [
                    {
                        "B04": {"href": "https://example.com/B04.tif"},
                        "thumbnail": {"href": "https://example.com/thumb.jpg"},
                    }
                ]
            ),
        }
    )
    record_path = tmp_path / "records.parquet"
    pq.write_table(table, record_path)

    rasteret.register(
        DatasetDescriptor(
            id="custom/geoparquet",
            name="Custom",
            geoparquet_uri=str(record_path),
            stac_collection="custom",
            band_map={"B04": "B04"},
            requires_auth=False,
        )
    )

    collection = rasteret.build(
        "custom/geoparquet",
        name="gpq",
        workspace_dir=tmp_path,
        force=True,
        prefer_geoparquet=True,
    )

    assert collection.dataset is not None
    # Only the band asset should be parsed (not the thumbnail).
    assert captured["urls"] == ["https://example.com/B04.tif"]
