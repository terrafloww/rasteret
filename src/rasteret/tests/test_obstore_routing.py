# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for obstore URL routing, rewrite helpers, and S3-aware backend."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasteret.cloud import CloudConfig, rewrite_url, s3_overrides_from_config
from rasteret.fetch.cog import (
    _create_obstore_backend,
    _extract_azure_account,
    _extract_s3_bucket,
)

# ---------------------------------------------------------------------------
# _extract_s3_bucket
# ---------------------------------------------------------------------------


class TestExtractS3Bucket:
    def test_virtual_hosted_us_west_2(self):
        assert (
            _extract_s3_bucket("sentinel-cogs.s3.us-west-2.amazonaws.com")
            == "sentinel-cogs"
        )

    def test_virtual_hosted_no_region(self):
        assert _extract_s3_bucket("my-bucket.s3.amazonaws.com") == "my-bucket"

    def test_virtual_hosted_dashed_region(self):
        assert _extract_s3_bucket("data.s3.eu-central-1.amazonaws.com") == "data"

    def test_non_s3_host_returns_none(self):
        assert _extract_s3_bucket("example.com") is None

    def test_presigned_host_still_extracts(self):
        # The netloc doesn't contain query params, just the host
        assert (
            _extract_s3_bucket("usgs-landsat.s3.us-west-2.amazonaws.com")
            == "usgs-landsat"
        )

    def test_bucket_with_dots(self):
        assert (
            _extract_s3_bucket("my.dotted.bucket.s3.amazonaws.com")
            == "my.dotted.bucket"
        )


# ---------------------------------------------------------------------------
# rewrite_url
# ---------------------------------------------------------------------------


class TestRewriteUrl:
    def test_rewrites_matching_pattern(self):
        config = CloudConfig(
            provider="aws",
            url_patterns={"https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"},
        )
        result = rewrite_url(
            "https://landsatlook.usgs.gov/data/collection02/test.tif", config
        )
        assert result == "s3://usgs-landsat/collection02/test.tif"

    def test_no_match_returns_original(self):
        config = CloudConfig(
            provider="aws",
            url_patterns={"https://other.com/": "s3://other/"},
        )
        url = "https://example.com/file.tif"
        assert rewrite_url(url, config) == url

    def test_none_config_returns_original(self):
        assert (
            rewrite_url("https://example.com/file.tif", None)
            == "https://example.com/file.tif"
        )

    def test_empty_patterns_returns_original(self):
        config = CloudConfig(provider="aws", url_patterns={})
        assert (
            rewrite_url("https://example.com/f.tif", config)
            == "https://example.com/f.tif"
        )


# ---------------------------------------------------------------------------
# s3_overrides_from_config
# ---------------------------------------------------------------------------


class TestS3OverridesFromConfig:
    def test_requester_pays_config(self):
        config = CloudConfig(
            provider="aws",
            requester_pays=True,
            region="us-west-2",
            url_patterns={"https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"},
        )
        overrides = s3_overrides_from_config(config)
        assert "usgs-landsat" in overrides
        assert overrides["usgs-landsat"]["request_payer"] == "true"
        assert overrides["usgs-landsat"]["region"] == "us-west-2"
        assert "skip_signature" not in overrides["usgs-landsat"]

    def test_public_bucket_config(self):
        config = CloudConfig(
            provider="aws",
            requester_pays=False,
            region="us-west-2",
            url_patterns={"https://example.com/": "s3://public-bucket/"},
        )
        overrides = s3_overrides_from_config(config)
        assert "public-bucket" in overrides
        assert overrides["public-bucket"]["skip_signature"] == "true"
        assert "request_payer" not in overrides["public-bucket"]

    def test_none_config_returns_empty(self):
        assert s3_overrides_from_config(None) == {}

    def test_no_s3_patterns_returns_empty(self):
        config = CloudConfig(provider="aws", url_patterns={})
        assert s3_overrides_from_config(config) == {}


# ---------------------------------------------------------------------------
# _AutoObstoreBackend URL routing
# ---------------------------------------------------------------------------


class TestAutoObstoreBackendRouting:
    """Test that _store_for() routes URLs to the correct store type."""

    def test_s3_scheme_creates_s3_store_for_public_bucket(self):
        backend = _create_obstore_backend()
        store, path = backend._store_for("s3://sentinel-cogs/path/to/file.tif")
        assert path == "path/to/file.tif"
        assert "s3://sentinel-cogs" in backend._stores

    def test_s3_virtual_hosted_creates_s3_store(self):
        backend = _create_obstore_backend()
        store, path = backend._store_for(
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/tile.tif"
        )
        assert path == "sentinel-s2-l2a-cogs/tile.tif"
        assert "s3://sentinel-cogs" in backend._stores

    def test_presigned_url_creates_http_store(self):
        backend = _create_obstore_backend()
        url = "https://bucket.s3.us-west-2.amazonaws.com/key.tif?X-Amz-Signature=abc123"
        store, path = backend._store_for(url)
        # Presigned URL should be cached by full URL, path should be empty
        assert path == ""
        assert url in backend._stores

    def test_non_s3_https_creates_http_store(self):
        backend = _create_obstore_backend()
        store, path = backend._store_for("https://example.com/data/file.tif")
        assert path == "data/file.tif"
        assert "https://example.com/" in backend._stores

    def test_s3_overrides_applied(self):
        backend = _create_obstore_backend(
            s3_overrides={
                "usgs-landsat": {"request_payer": "true", "region": "us-west-2"}
            }
        )
        store, path = backend._store_for("s3://usgs-landsat/collection02/test.tif")
        assert path == "collection02/test.tif"
        assert "s3://usgs-landsat" in backend._stores

    def test_requester_pays_config_requires_signing(self):
        """Requester-pays buckets must not force anonymous (skip_signature) access."""
        backend = _create_obstore_backend(
            s3_overrides={
                "usgs-landsat": {"request_payer": "true", "region": "us-west-2"}
            }
        )

        with patch("obstore.store.S3Store") as MockS3:
            MockS3.return_value = MagicMock()
            backend._get_s3_store("usgs-landsat")
            kwargs = MockS3.call_args[1]
            config = kwargs.get("config", {})
            assert config.get("request_payer") == "true"
            assert "skip_signature" not in config

    def test_store_caching(self):
        backend = _create_obstore_backend()
        store1, _ = backend._store_for("s3://my-bucket/a.tif")
        store2, _ = backend._store_for("s3://my-bucket/b.tif")
        assert store1 is store2

    def test_s3_region_redirect_is_retried_with_discovered_region(self):
        from obstore.exceptions import GenericError

        backend = _create_obstore_backend()

        redirect_error = GenericError(
            "Generic S3 error: Received redirect without LOCATION, "
            "this normally indicates an incorrectly configured region"
        )

        with patch(
            "rasteret.fetch.cog._discover_s3_bucket_region", return_value="eu-central-1"
        ):
            with patch("obstore.get_range_async", new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = [redirect_error, b"x"]
                result = asyncio.run(
                    backend.get_range("s3://copernicus-dem-30m/foo", 0, 1)
                )
                assert result == b"x"
                assert backend._s3_regions["copernicus-dem-30m"] == "eu-central-1"
                assert mock_get.call_count == 2

    def test_get_range_retries_truncated_responses(self):
        backend = _create_obstore_backend()

        with patch("obstore.get_range_async", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [b"", b"x"]
            result = asyncio.run(
                backend.get_range("https://example.com/file.tif", 0, 1)
            )
            assert result == b"x"
            assert mock_get.call_count == 2

    def test_get_range_raises_after_repeated_truncated_responses(self):
        backend = _create_obstore_backend()

        with patch("obstore.get_range_async", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [b"", b"", b""]
            with pytest.raises(IOError, match="Truncated range response"):
                asyncio.run(backend.get_range("https://example.com/file.tif", 0, 1))
            assert mock_get.call_count == 3

    def test_get_range_retries_unexpected_range_generic_error(self):
        from obstore.exceptions import GenericError

        backend = _create_obstore_backend()

        err = GenericError(
            "Generic HTTP error: Requested 10..20, got 10..12 "
            "source: UnexpectedRange"
        )
        with patch("obstore.get_range_async", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [err, b"abcdef"]
            result = asyncio.run(
                backend.get_range("https://example.com/file.tif", 0, 6)
            )
            assert result == b"abcdef"
            assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Backend URL pattern rewriting
# ---------------------------------------------------------------------------


class TestBackendURLPatternRewriting:
    """Test that url_patterns are applied inside _store_for() before routing."""

    def test_landsat_pattern_routes_to_s3_store(self):
        backend = _create_obstore_backend(
            url_patterns={"https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"},
            s3_overrides={
                "usgs-landsat": {"request_payer": "true", "region": "us-west-2"}
            },
        )
        store, path = backend._store_for(
            "https://landsatlook.usgs.gov/data/collection02/test.tif"
        )
        assert path == "collection02/test.tif"
        assert "s3://usgs-landsat" in backend._stores

    def test_no_match_falls_through_to_https(self):
        backend = _create_obstore_backend(
            url_patterns={"https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"},
        )
        store, path = backend._store_for("https://example.com/file.tif")
        assert path == "file.tif"
        assert "https://example.com/" in backend._stores

    def test_parity_with_rewrite_url(self):
        """Backend url_patterns produce the same rewrite as rewrite_url()."""
        config = CloudConfig(
            provider="aws",
            url_patterns={"https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"},
        )
        original = "https://landsatlook.usgs.gov/data/collection02/test.tif"
        expected = rewrite_url(original, config)
        assert expected == "s3://usgs-landsat/collection02/test.tif"

        backend = _create_obstore_backend(url_patterns=config.url_patterns)
        store, path = backend._store_for(original)
        # The backend should route to the same bucket
        assert "s3://usgs-landsat" in backend._stores
        assert path == "collection02/test.tif"


# ---------------------------------------------------------------------------
# _extract_azure_account
# ---------------------------------------------------------------------------


class TestExtractAzureAccount:
    def test_standard_blob_host(self):
        assert _extract_azure_account("myaccount.blob.core.windows.net") == "myaccount"

    def test_planetary_computer_host(self):
        assert (
            _extract_azure_account("landsateuwest.blob.core.windows.net")
            == "landsateuwest"
        )

    def test_non_azure_host_returns_none(self):
        assert _extract_azure_account("example.com") is None

    def test_s3_host_returns_none(self):
        assert _extract_azure_account("bucket.s3.us-west-2.amazonaws.com") is None


# ---------------------------------------------------------------------------
# GCS URL extraction (inline in _store_for, no standalone extractor)
# ---------------------------------------------------------------------------


class TestGcsUrlRouting:
    def test_gs_scheme_routes_to_gcs_store(self):
        backend = _create_obstore_backend()
        store, path = backend._store_for(
            "gs://gcp-public-data-landsat/LC08/01/044/034/file.tif"
        )
        assert path == "LC08/01/044/034/file.tif"
        assert "gs://gcp-public-data-landsat" in backend._stores

    def test_gcs_https_routes_to_gcs_store(self):
        backend = _create_obstore_backend()
        store, path = backend._store_for(
            "https://storage.googleapis.com/gcp-public-data-landsat/LC08/file.tif"
        )
        assert path == "LC08/file.tif"
        assert "gs://gcp-public-data-landsat" in backend._stores

    def test_gcs_store_caching(self):
        backend = _create_obstore_backend()
        store1, _ = backend._store_for("gs://my-bucket/a.tif")
        store2, _ = backend._store_for("gs://my-bucket/b.tif")
        assert store1 is store2


# ---------------------------------------------------------------------------
# Multi-cloud URL routing
# ---------------------------------------------------------------------------


class TestMultiCloudRouting:
    def test_azure_blob_routes_to_azure_store(self):
        backend = _create_obstore_backend()
        store, path = backend._store_for(
            "https://landsateuwest.blob.core.windows.net/landsat-c2/path/file.tif"
        )
        assert path == "path/file.tif"
        assert "azure://landsateuwest/landsat-c2" in backend._stores

    def test_azure_sas_routes_to_http_store(self):
        """SAS-signed URLs have query params -> HTTPStore (self-authenticating)."""
        backend = _create_obstore_backend()
        url = (
            "https://landsateuwest.blob.core.windows.net/landsat-c2/file.tif"
            "?sv=2023-11-03&se=2025-01-01&sig=abc123"
        )
        store, path = backend._store_for(url)
        assert path == ""
        assert url in backend._stores

    def test_mixed_cloud_backends(self):
        """Different URLs create different store types in the same backend."""
        backend = _create_obstore_backend()
        backend._store_for("s3://sentinel-cogs/tile.tif")
        backend._store_for("gs://gcp-public-data-landsat/tile.tif")
        backend._store_for(
            "https://landsateuwest.blob.core.windows.net/landsat-c2/tile.tif"
        )
        backend._store_for("https://example.com/tile.tif")

        assert "s3://sentinel-cogs" in backend._stores
        assert "gs://gcp-public-data-landsat" in backend._stores
        assert "azure://landsateuwest/landsat-c2" in backend._stores
        assert "https://example.com/" in backend._stores
        assert len(backend._stores) == 4

    def test_azure_container_extraction(self):
        """Container is extracted from the first path segment."""
        backend = _create_obstore_backend()
        store, path = backend._store_for(
            "https://myaccount.blob.core.windows.net/mycontainer/deep/nested/file.tif"
        )
        assert path == "deep/nested/file.tif"
        assert "azure://myaccount/mycontainer" in backend._stores

    def test_gcs_bucket_from_https_url(self):
        """Bucket extracted from first path segment of GCS HTTPS URL."""
        backend = _create_obstore_backend()
        store, path = backend._store_for(
            "https://storage.googleapis.com/my-gcs-bucket/subdir/data.tif"
        )
        assert path == "subdir/data.tif"
        assert "gs://my-gcs-bucket" in backend._stores


# ---------------------------------------------------------------------------
# Credential provider passthrough
# ---------------------------------------------------------------------------


class TestCredentialProviderPassthrough:
    """Verify credential_provider is stored and affects store config logic.

    We cannot pass a MagicMock into obstore's Rust-backed store constructors
    (they validate types at the boundary), so we patch the store classes
    and inspect the kwargs they receive.
    """

    def test_s3_store_drops_skip_signature(self):
        """When credential_provider is set, skip_signature is removed from S3 config."""
        sentinel = object()
        backend = _create_obstore_backend(credential_provider=sentinel)
        assert backend._credential_provider is sentinel

        with patch("obstore.store.S3Store") as MockS3:
            MockS3.return_value = MagicMock()
            backend._get_s3_store("test-bucket")
            MockS3.assert_called_once()
            kwargs = MockS3.call_args[1]
            assert kwargs["credential_provider"] is sentinel
            assert "skip_signature" not in kwargs.get("config", {})

    def test_azure_store_receives_credential_provider(self):
        sentinel = object()
        backend = _create_obstore_backend(credential_provider=sentinel)

        with patch("obstore.store.AzureStore") as MockAzure:
            MockAzure.return_value = MagicMock()
            backend._store_for(
                "https://myaccount.blob.core.windows.net/container/file.tif"
            )
            MockAzure.assert_called_once()
            kwargs = MockAzure.call_args[1]
            assert kwargs["credential_provider"] is sentinel
            assert kwargs["container_name"] == "container"

    def test_gcs_store_receives_credential_provider(self):
        sentinel = object()
        backend = _create_obstore_backend(credential_provider=sentinel)

        with patch("obstore.store.GCSStore") as MockGCS:
            MockGCS.return_value = MagicMock()
            backend._store_for("gs://my-gcs-bucket/file.tif")
            MockGCS.assert_called_once()
            kwargs = MockGCS.call_args[1]
            assert kwargs["credential_provider"] is sentinel
            # skip_signature should be removed when cred provider is set
            assert "skip_signature" not in kwargs.get("config", {})

    def test_create_backend_public_api(self):
        """rasteret.create_backend() creates a working backend."""
        import rasteret

        backend = rasteret.create_backend()
        assert hasattr(backend, "get_range")
        assert hasattr(backend, "get_ranges")

    def test_create_backend_cloud_config_passthrough(self):
        """create_backend(cloud_config=...) converts config to s3_overrides and url_patterns."""
        import rasteret
        from rasteret.cloud import CloudConfig

        config = CloudConfig(
            provider="aws",
            requester_pays=True,
            region="us-west-2",
            url_patterns={"https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"},
        )
        backend = rasteret.create_backend(cloud_config=config)

        # The overrides should be baked into the backend
        assert backend._s3_overrides == {
            "usgs-landsat": {"region": "us-west-2", "request_payer": "true"}
        }

        # URL patterns should be baked into the backend
        assert backend._url_patterns == {
            "https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"
        }

        # Routing an S3 URL for that bucket should apply the overrides
        store, path = backend._store_for("s3://usgs-landsat/collection02/test.tif")
        assert path == "collection02/test.tif"
        assert "s3://usgs-landsat" in backend._stores

    def test_incompatible_provider_falls_back_gracefully(self):
        """An Azure credential provider should not break S3 store creation."""
        with patch("obstore.store.S3Store") as MockS3:
            mock_store = MagicMock()
            MockS3.side_effect = [TypeError("incompatible provider"), mock_store]

            backend = _create_obstore_backend(credential_provider=object())
            store = backend._get_s3_store("sentinel-cogs")
            assert MockS3.call_count == 2
            second_kwargs = MockS3.call_args_list[1][1]
            assert "credential_provider" not in second_kwargs
            assert second_kwargs["config"]["skip_signature"] == "true"
            assert store is mock_store
            assert "s3://sentinel-cogs" in backend._stores

    def test_non_compat_error_is_not_swallowed(self):
        """Errors unrelated to provider compatibility should propagate."""
        with patch("obstore.store.S3Store") as MockS3:
            MockS3.side_effect = RuntimeError("unexpected constructor failure")
            backend = _create_obstore_backend(credential_provider=object())
            with pytest.raises(RuntimeError, match="unexpected constructor failure"):
                backend._get_s3_store("my-bucket")

    def test_azure_prefix_not_doubled(self):
        """Credential provider prefix must be stripped to avoid v002/v002/... doubling."""
        from obstore.auth.planetary_computer import PlanetaryComputerCredentialProvider

        cp = PlanetaryComputerCredentialProvider(
            "https://naipeuwest.blob.core.windows.net/naip/v002/"
        )
        backend = _create_obstore_backend(credential_provider=cp)

        url = (
            "https://naipeuwest.blob.core.windows.net/naip/"
            "v002/mt/2023/mt_060cm_2023/44106/file.tif"
        )
        store, path = backend._store_for(url)

        # The store should have the prefix set by the credential provider
        assert store.prefix == "v002"
        # The returned path must NOT include the prefix (obstore prepends it)
        assert path == "mt/2023/mt_060cm_2023/44106/file.tif"
        assert not path.startswith("v002")
