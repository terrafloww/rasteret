# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for the STAC collection builder."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pystac
import pytest

from rasteret.cloud import CloudConfig
from rasteret.ingest.stac_indexer import StacCollectionBuilder
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
