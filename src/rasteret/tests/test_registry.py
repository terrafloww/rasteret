# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from rasteret.cloud import CloudConfig
from rasteret.constants import BandRegistry, DataSources


class TestBandRegistry:
    def test_builtin_sentinel2_registered(self):
        bands = BandRegistry.get("sentinel-2-l2a")
        assert "B02" in bands
        assert bands["B02"] == "blue"

    def test_builtin_landsat_registered(self):
        bands = BandRegistry.get("landsat-c2-l2")
        assert "B4" in bands
        assert bands["B4"] == "red"

    def test_get_missing_returns_empty(self):
        result = BandRegistry.get("nonexistent-collection")
        assert result == {}

    def test_get_missing_with_default(self):
        default = {"X1": "custom"}
        result = BandRegistry.get("nonexistent-collection", default=default)
        assert result is default

    def test_register_and_get(self):
        BandRegistry.register("test-custom", {"R": "red", "G": "green"})
        assert BandRegistry.get("test-custom") == {"R": "red", "G": "green"}

    def test_list_registered_includes_builtins(self):
        collections = BandRegistry.list_registered()
        assert "sentinel-2-l2a" in collections
        assert "landsat-c2-l2" in collections

    def test_register_overwrites(self):
        BandRegistry.register("overwrite-test", {"A": "alpha"})
        BandRegistry.register("overwrite-test", {"B": "beta"})
        assert BandRegistry.get("overwrite-test") == {"B": "beta"}


class TestCloudConfigRegistry:
    def test_builtin_landsat_registered(self):
        config = CloudConfig.get_config("landsat-c2-l2")
        assert config is not None
        assert config.requester_pays is True
        assert config.region == "us-west-2"

    def test_builtin_sentinel2_registered(self):
        config = CloudConfig.get_config("sentinel-2-l2a")
        assert config is not None
        assert config.requester_pays is False

    def test_get_config_case_insensitive(self):
        config = CloudConfig.get_config("Sentinel-2-L2A")
        assert config is not None

    def test_get_config_missing_returns_none(self):
        assert CloudConfig.get_config("nonexistent") is None

    def test_register_custom(self):
        CloudConfig.register(
            "test-collection",
            CloudConfig(provider="gcs", requester_pays=False, region="eu-west-1"),
        )
        config = CloudConfig.get_config("test-collection")
        assert config is not None
        assert config.provider == "gcs"
        assert config.region == "eu-west-1"


class TestDataSources:
    def test_sentinel2_constant(self):
        assert DataSources.SENTINEL2 == "sentinel-2-l2a"

    def test_landsat_constant(self):
        assert DataSources.LANDSAT == "landsat-c2-l2"

    def test_list_sources_delegates_to_band_registry(self):
        sources = DataSources.list_sources()
        assert "sentinel-2-l2a" in sources
        assert "landsat-c2-l2" in sources
