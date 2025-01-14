'''
Copyright 2025 Terrafloww Labs, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import unittest
from unittest.mock import patch, MagicMock
from rasteret.cloud import CloudConfig, AWSProvider


class TestCloudProvider(unittest.TestCase):

    @patch("rasteret.cloud.boto3")
    def test_s3_url_signing_handler(self, mock_boto3):
        # Mock AWS session and client
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_s3_client
        mock_s3_client.generate_presigned_url.return_value = (
            "https://signed-url.example.com"
        )

        # Mock credentials check
        mock_credentials = MagicMock()
        mock_session.get_credentials.return_value = mock_credentials

        # Create test configuration
        cloud_config = CloudConfig(
            provider="aws",
            requester_pays=True,
            region="us-west-2",
            url_patterns={"https://example.com/": "s3://example-bucket/"},
        )

        # Initialize provider
        provider = AWSProvider(region="us-west-2")

        # Test URL pattern conversion
        test_url = "https://example.com/test.tif"
        provider.get_url(test_url, cloud_config)

        # Verify S3 client calls
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={
                "Bucket": "example-bucket",
                "Key": "test.tif",
                "RequestPayer": "requester",
            },
            ExpiresIn=3600,
        )


if __name__ == "__main__":
    unittest.main()
