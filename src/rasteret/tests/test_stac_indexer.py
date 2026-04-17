# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for the STAC collection builder."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pystac
import pytest
import tifffile as tf
from pyproj import CRS
from pystac_client.exceptions import APIError

from rasteret.cloud import CloudConfig
from rasteret.ingest.stac_indexer import (
    StacCollectionBuilder,
    _is_retryable_stac_api_error,
)
from rasteret.types import CogMetadata

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_stac_items():
    return [
        pystac.Item(
            id="test_scene_1",
            datetime=datetime(2023, 1, 1, 0, 0, 0),
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=[0, 0, 1, 1],
            properties={
                "datetime": datetime(2023, 1, 1, 0, 0, 0),
                "eo:cloud_cover": 10.5,
            },
            assets={
                "B1": pystac.Asset(href="s3://test-bucket/test1_B1.tif"),
                "B2": pystac.Asset(href="s3://test-bucket/test1_B2.tif"),
            },
        )
    ]


@pytest.fixture()
def mock_cog_metadata():
    return CogMetadata(
        width=1000,
        height=1000,
        tile_width=256,
        tile_height=256,
        dtype="uint16",
        crs=4326,
        transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        predictor=2,
        compression=8,
        tile_offsets=[1000],
        tile_byte_counts=[10000],
        pixel_scale=[1.0, 1.0, 0.0],
        tiepoint=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.fixture()
def cloud_config():
    return CloudConfig(
        provider="aws",
        requester_pays=True,
        region="us-west-2",
        url_patterns={"https://test.com/": "s3://test-bucket/"},
    )


def _write_static_catalog_with_tiff(
    tmp_path,
    *,
    asset_key: str = "B01",
    collection_id: str = "S2_L2A_catalog",
) -> tuple[Path, Path]:
    root = tmp_path / "catalog"
    item_dir = root / "tile"
    item_dir.mkdir(parents=True)
    cog_path = item_dir / f"{asset_key}.tif"
    data = np.zeros((128, 128), dtype=np.uint16)
    extratags = [
        (33550, "d", 3, (1.0, 1.0, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), False),
    ]
    tf.imwrite(cog_path, data, tile=(64, 64), extratags=extratags)

    item = pystac.Item(
        id="scene-1",
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 1, 1, 0, 0, 0),
        properties={"proj:code": "EPSG:4326"},
        collection=collection_id,
    )
    item.add_asset(asset_key, pystac.Asset(href=f"{asset_key}.tif"))
    item.set_self_href(str(item_dir / "scene-1.json"))
    item.save_object()

    catalog = pystac.Catalog(id=collection_id, description="local static")
    catalog.add_item(item)
    catalog.normalize_hrefs(str(root))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return root / "catalog.json", cog_path


def _write_static_catalog_with_asset_sequence(
    tmp_path,
    asset_keys: list[str],
    *,
    collection_id: str = "mixed_catalog",
) -> Path:
    root = tmp_path / "catalog"
    root.mkdir()
    catalog = pystac.Catalog(id=collection_id, description="mixed static")

    for idx, asset_key in enumerate(asset_keys):
        item_dir = root / f"tile-{idx}"
        item_dir.mkdir()
        item = pystac.Item(
            id=f"scene-{idx}",
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 1, 1, 0, 0, 0),
            properties={},
            collection=collection_id,
        )
        item.add_asset(asset_key, pystac.Asset(href=f"{asset_key}.tif"))
        item.set_self_href(str(item_dir / f"scene-{idx}.json"))
        item.save_object()
        catalog.add_item(item)

    catalog.normalize_hrefs(str(root))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return root / "catalog.json"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStacCollectionBuilder:
    @patch("rasteret.ingest.stac_indexer.pystac_client")
    def test_stac_search(self, mock_pystac, mock_stac_items):
        mock_client = MagicMock()
        mock_search = MagicMock()

        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items_as_dicts.return_value = [
            item.to_dict() for item in mock_stac_items
        ]

        builder = StacCollectionBuilder(
            data_source="test-source", stac_api="https://test-stac.com"
        )

        items = asyncio.run(
            builder._search_stac(
                bbox=[-180, -90, 180, 90],
                date_range=["2023-01-01", "2023-12-31"],
                query=None,
            )
        )

        assert len(items) == 1
        assert items[0]["id"] == "test_scene_1"

        mock_pystac.Client.open.assert_called_once()
        mock_client.search.assert_called_once()
        mock_search.items_as_dicts.assert_called_once()

    @patch("rasteret.ingest.stac_indexer.asyncio.sleep", new_callable=AsyncMock)
    @patch("rasteret.ingest.stac_indexer.pystac_client")
    def test_stac_search_retries_transient_api_errors(
        self,
        mock_pystac,
        mock_sleep,
        mock_stac_items,
    ):
        mock_client = MagicMock()
        mock_search = MagicMock()

        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items_as_dicts.side_effect = [
            APIError(
                "The request exceeded the maximum allowed time, please try again."
            ),
            [item.to_dict() for item in mock_stac_items],
        ]

        builder = StacCollectionBuilder(
            data_source="test-source", stac_api="https://test-stac.com"
        )

        items = asyncio.run(
            builder._search_stac(
                bbox=[-180, -90, 180, 90],
                date_range=["2023-01-01", "2023-12-31"],
                query=None,
            )
        )

        assert len(items) == 1
        assert items[0]["id"] == "test_scene_1"
        assert mock_client.search.call_count == 2
        mock_sleep.assert_awaited_once_with(1.0)

    @patch("rasteret.ingest.stac_indexer.AsyncCOGHeaderParser")
    @patch("rasteret.ingest.stac_indexer.pystac_client")
    def test_index_creation(
        self, mock_pystac, mock_parser, mock_stac_items, mock_cog_metadata, tmp_path
    ):
        mock_client = MagicMock()
        mock_search = MagicMock()

        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items_as_dicts.return_value = [
            item.to_dict() for item in mock_stac_items
        ]

        mock_parser_instance = AsyncMock()
        mock_parser.return_value.__aenter__.return_value = mock_parser_instance
        mock_parser_instance.process_cog_headers_batch.return_value = [
            mock_cog_metadata
        ]

        builder = StacCollectionBuilder(
            data_source="test-source",
            stac_api="https://test-stac.com",
            workspace_dir=tmp_path / "test_output",
            band_map={"B1": "B1"},
        )

        collection = asyncio.run(
            builder.build_index(
                bbox=[-180, -90, 180, 90], date_range=["2023-01-01", "2023-12-31"]
            )
        )

        assert collection is not None
        mock_parser_instance.process_cog_headers_batch.assert_called_once()

    def test_url_rewriting(self, cloud_config):
        builder = StacCollectionBuilder(
            data_source="test-source",
            stac_api="https://test-stac.com",
            cloud_config=cloud_config,
        )

        url = builder._get_asset_url({"href": "https://test.com/asset.tif"})
        assert url == "s3://test-bucket/asset.tif"

    def test_url_rewriting_no_match(self, cloud_config):
        builder = StacCollectionBuilder(
            data_source="test-source",
            stac_api="https://test-stac.com",
            cloud_config=cloud_config,
        )

        url = builder._get_asset_url({"href": "https://other.com/asset.tif"})
        assert url == "https://other.com/asset.tif"

    def test_static_catalog_build_enriches_relative_local_asset(self, tmp_path):
        import rasteret

        catalog_path, _ = _write_static_catalog_with_tiff(tmp_path)

        collection = rasteret.build_from_stac(
            name="local-static",
            stac_api=str(catalog_path),
            collection="S2_L2A_catalog",
            static_catalog=True,
            workspace_dir=tmp_path / "workspace",
            max_concurrent=1,
            force=True,
        )

        out = collection.dataset.to_table(columns=["B01_metadata"])
        metadata = out.column("B01_metadata").to_pylist()[0]
        assert metadata is not None
        assert metadata["image_width"] == 128
        assert metadata["tile_offsets"]

    def test_static_catalog_explicit_band_map_subset(self, tmp_path):
        import rasteret

        catalog_path, _ = _write_static_catalog_with_tiff(tmp_path)

        collection = rasteret.build_from_stac(
            name="local-static-subset",
            stac_api=str(catalog_path),
            collection="S2_L2A_catalog",
            static_catalog=True,
            band_map={"B01": "B01"},
            workspace_dir=tmp_path / "workspace",
            max_concurrent=1,
            force=True,
        )

        assert "B01_metadata" in collection.dataset.schema.names

    def test_static_catalog_fails_fast_when_band_map_matches_no_assets(self, tmp_path):
        import rasteret

        catalog_path, _ = _write_static_catalog_with_tiff(tmp_path, asset_key="B01")

        with patch("rasteret.ingest.stac_indexer.AsyncCOGHeaderParser") as parser:
            with pytest.raises(ValueError, match="Missing asset keys: red"):
                rasteret.build_from_stac(
                    name="local-static-mismatch",
                    stac_api=str(catalog_path),
                    collection="S2_L2A_catalog",
                    static_catalog=True,
                    band_map={"B04": "red"},
                    workspace_dir=tmp_path / "workspace",
                    max_concurrent=1,
                    force=True,
                )
            parser.assert_not_called()

    def test_static_catalog_fails_fast_when_band_map_has_partial_typo(self, tmp_path):
        import rasteret

        catalog_path, _ = _write_static_catalog_with_tiff(tmp_path, asset_key="B01")

        with patch("rasteret.ingest.stac_indexer.AsyncCOGHeaderParser") as parser:
            with pytest.raises(ValueError, match="Missing asset keys: B02"):
                rasteret.build_from_stac(
                    name="local-static-partial-mismatch",
                    stac_api=str(catalog_path),
                    collection="S2_L2A_catalog",
                    static_catalog=True,
                    band_map={"B01": "B01", "B02": "B02"},
                    workspace_dir=tmp_path / "workspace",
                    max_concurrent=1,
                    force=True,
                )
            parser.assert_not_called()

    def test_static_catalog_validates_against_full_selected_asset_union(self, tmp_path):
        catalog_path = _write_static_catalog_with_asset_sequence(
            tmp_path,
            ["B01"] * 20 + ["B02"],
        )
        builder = StacCollectionBuilder(
            data_source="mixed_catalog",
            stac_api=str(catalog_path),
            band_map={"B01": "B01", "B02": "B02"},
            static_catalog=True,
        )

        items = builder._crawl_static_catalog(None, None)

        assert len(items) == 21

    def test_sentinel2_registry_common_name_map_is_preserved(self):
        builder = StacCollectionBuilder(
            data_source="sentinel-2-l2a",
            stac_api="https://example.com",
            stac_collection="sentinel-2-l2a",
        )

        assert builder.band_map["B02"] == "blue"

    @patch("rasteret.ingest.stac_indexer.AsyncCOGHeaderParser")
    @patch("rasteret.ingest.stac_indexer.pystac_client")
    def test_build_uses_proj_code_for_record_crs(
        self, mock_pystac, mock_parser, mock_cog_metadata, tmp_path
    ):
        item = pystac.Item(
            id="test_scene_1",
            datetime=datetime(2023, 1, 1, 0, 0, 0),
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=[0, 0, 1, 1],
            properties={
                "datetime": datetime(2023, 1, 1, 0, 0, 0),
                "proj:code": "EPSG:32632",
                "proj:epsg": 32631,
            },
            assets={"B1": pystac.Asset(href="s3://test-bucket/test1_B1.tif")},
        )
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items_as_dicts.return_value = [item.to_dict()]

        mock_parser_instance = AsyncMock()
        mock_parser.return_value.__aenter__.return_value = mock_parser_instance
        meta = CogMetadata(**{**mock_cog_metadata.__dict__, "crs": None})
        mock_parser_instance.process_cog_headers_batch.return_value = [meta]

        builder = StacCollectionBuilder(
            data_source="test-source",
            stac_api="https://test-stac.com",
            workspace_dir=tmp_path / "test_output",
            band_map={"B1": "B1"},
        )

        collection = asyncio.run(builder.build_index())
        out = collection.dataset.to_table(columns=["proj:epsg", "crs"])
        assert out.column("proj:epsg").to_pylist() == [32632]
        assert out.column("crs").to_pylist() == ["EPSG:32632"]

    @patch("rasteret.ingest.stac_indexer.AsyncCOGHeaderParser")
    @patch("rasteret.ingest.stac_indexer.pystac_client")
    def test_build_falls_back_to_cog_header_crs_when_projection_property_missing(
        self, mock_pystac, mock_parser, mock_cog_metadata, tmp_path
    ):
        item = pystac.Item(
            id="test_scene_1",
            datetime=datetime(2023, 1, 1, 0, 0, 0),
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=[0, 0, 1, 1],
            properties={"datetime": datetime(2023, 1, 1, 0, 0, 0)},
            assets={"B1": pystac.Asset(href="s3://test-bucket/test1_B1.tif")},
        )
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items_as_dicts.return_value = [item.to_dict()]

        mock_parser_instance = AsyncMock()
        mock_parser.return_value.__aenter__.return_value = mock_parser_instance
        mock_parser_instance.process_cog_headers_batch.return_value = [
            mock_cog_metadata
        ]

        builder = StacCollectionBuilder(
            data_source="test-source",
            stac_api="https://test-stac.com",
            workspace_dir=tmp_path / "test_output",
            band_map={"B1": "B1"},
        )

        collection = asyncio.run(builder.build_index())
        out = collection.dataset.to_table(columns=["proj:epsg", "crs"])
        assert out.column("proj:epsg").to_pylist() == [4326]
        assert out.column("crs").to_pylist() == ["EPSG:4326"]

    @patch("rasteret.ingest.stac_indexer.AsyncCOGHeaderParser")
    @patch("rasteret.ingest.stac_indexer.pystac_client")
    def test_build_uses_proj_wkt2_for_record_crs(
        self, mock_pystac, mock_parser, mock_cog_metadata, tmp_path
    ):
        item = pystac.Item(
            id="test_scene_1",
            datetime=datetime(2023, 1, 1, 0, 0, 0),
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=[0, 0, 1, 1],
            properties={
                "datetime": datetime(2023, 1, 1, 0, 0, 0),
                "proj:wkt2": CRS.from_epsg(32632).to_wkt(),
            },
            assets={"B1": pystac.Asset(href="s3://test-bucket/test1_B1.tif")},
        )
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items_as_dicts.return_value = [item.to_dict()]

        mock_parser_instance = AsyncMock()
        mock_parser.return_value.__aenter__.return_value = mock_parser_instance
        meta = CogMetadata(**{**mock_cog_metadata.__dict__, "crs": None})
        mock_parser_instance.process_cog_headers_batch.return_value = [meta]

        builder = StacCollectionBuilder(
            data_source="test-source",
            stac_api="https://test-stac.com",
            workspace_dir=tmp_path / "test_output",
            band_map={"B1": "B1"},
        )

        collection = asyncio.run(builder.build_index())
        out = collection.dataset.to_table(columns=["proj:epsg", "crs"])
        assert out.column("proj:epsg").to_pylist() == [32632]
        assert out.column("crs").to_pylist() == ["EPSG:32632"]

    @patch("rasteret.ingest.stac_indexer.AsyncCOGHeaderParser")
    @patch("rasteret.ingest.stac_indexer.pystac_client")
    def test_build_raises_early_when_record_crs_cannot_be_resolved(
        self, mock_pystac, mock_parser, mock_cog_metadata, tmp_path
    ):
        item = pystac.Item(
            id="test_scene_1",
            datetime=datetime(2023, 1, 1, 0, 0, 0),
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=[0, 0, 1, 1],
            properties={"datetime": datetime(2023, 1, 1, 0, 0, 0)},
            assets={"B1": pystac.Asset(href="s3://test-bucket/test1_B1.tif")},
        )
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items_as_dicts.return_value = [item.to_dict()]

        mock_parser_instance = AsyncMock()
        mock_parser.return_value.__aenter__.return_value = mock_parser_instance
        meta = CogMetadata(**{**mock_cog_metadata.__dict__, "crs": None})
        mock_parser_instance.process_cog_headers_batch.return_value = [meta]

        builder = StacCollectionBuilder(
            data_source="test-source",
            stac_api="https://test-stac.com",
            workspace_dir=tmp_path / "test_output",
            band_map={"B1": "B1"},
        )

        with pytest.raises(ValueError, match="Raster CRS could not be resolved"):
            asyncio.run(builder.build_index())


def test_retryable_stac_api_error_detection() -> None:
    assert _is_retryable_stac_api_error(
        APIError("The request exceeded the maximum allowed time, please try again.")
    )
    assert _is_retryable_stac_api_error(APIError("HTTP 503 Service Unavailable"))
    assert not _is_retryable_stac_api_error(APIError("Unauthorized"))
