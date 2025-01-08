import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import pystac
from datetime import datetime

from rasteret.stac.indexer import StacToGeoParquetIndexer
from rasteret.cloud import CloudConfig
from rasteret.types import CogMetadata


class TestStacIndexer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_stac_items = [
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

        self.mock_cog_metadata = CogMetadata(
            width=1000,
            height=1000,
            tile_width=256,
            tile_height=256,
            dtype="uint16",
            crs=4326,
            transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            predictor=2,
            compression="deflate",
            tile_offsets=[1000],
            tile_byte_counts=[10000],
            pixel_scale=[1.0, 1.0, 0.0],
            tiepoint=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        self.cloud_config = CloudConfig(
            provider="aws",
            requester_pays=True,
            region="us-west-2",
            url_patterns={"https://test.com/": "s3://test-bucket/"},
        )

    @patch("rasteret.stac.indexer.pystac_client")
    async def test_stac_search(self, mock_pystac):
        # Setup mock STAC client
        mock_client = MagicMock()
        mock_search = MagicMock()

        # Configure mock chain
        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items.return_value = iter(self.mock_stac_items)

        # Create indexer
        indexer = StacToGeoParquetIndexer(
            data_source="test-source", stac_api="https://test-stac.com"
        )

        # Test search
        items = await indexer._search_stac(
            bbox=[-180, -90, 180, 90], date_range=["2023-01-01", "2023-12-31"]
        )

        # Verify results
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "test_scene_1")

        # Verify mock calls
        mock_pystac.Client.open.assert_called_once()
        mock_client.search.assert_called_once()
        mock_search.items.assert_called_once()

    @patch("rasteret.stac.indexer.AsyncCOGHeaderParser")
    @patch("rasteret.stac.indexer.pystac_client")
    async def test_index_creation(self, mock_pystac, mock_parser):
        # Setup STAC client mock chain
        mock_client = MagicMock()
        mock_search = MagicMock()

        mock_pystac.Client.open.return_value = mock_client
        mock_client.search.return_value = mock_search
        mock_search.items.return_value = iter(self.mock_stac_items)

        # Setup COG parser mock
        mock_parser_instance = AsyncMock()
        mock_parser.return_value.__aenter__.return_value = mock_parser_instance
        mock_parser_instance.process_cog_headers_batch.return_value = [
            self.mock_cog_metadata
        ]

        indexer = StacToGeoParquetIndexer(
            data_source="test-source",
            stac_api="https://test-stac.com",
            workspace_dir=Path("/tmp/test_output"),
        )

        collection = await indexer.build_index(
            bbox=[-180, -90, 180, 90], date_range=["2023-01-01", "2023-12-31"]
        )

        self.assertIsNotNone(collection)
        mock_parser_instance.process_cog_headers_batch.assert_called_once()

    def test_url_signing(self):
        mock_provider = MagicMock()
        mock_provider.get_url.return_value = "https://signed-url.test.com"

        indexer = StacToGeoParquetIndexer(
            data_source="test-source",
            stac_api="https://test-stac.com",
            cloud_provider=mock_provider,
            cloud_config=self.cloud_config,
        )

        url = indexer._get_asset_url({"href": "https://test.com/asset.tif"})
        self.assertEqual(url, "https://signed-url.test.com")


if __name__ == "__main__":
    unittest.main()
