# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rasteret.catalog import DatasetRegistry
from rasteret.cloud import CloudConfig
from rasteret.constants import BandRegistry

# ---------------------------------------------------------------------------
# async-tiff fixture discovery
# ---------------------------------------------------------------------------
# Priority:
#   1. ASYNC_TIFF_FIXTURES env var (CI or custom setups)
#   2. fixtures/ directory at repo root

_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/rasteret/tests -> repo root
_FIXTURES_PATH = _REPO_ROOT / "fixtures"


def _find_async_tiff_fixtures() -> Path | None:
    if os.environ.get("ASYNC_TIFF_FIXTURES"):
        p = Path(os.environ["ASYNC_TIFF_FIXTURES"])
        if p.is_dir():
            return p
    if _FIXTURES_PATH.is_dir():
        return _FIXTURES_PATH
    return None


ASYNC_TIFF_FIXTURES: Path | None = _find_async_tiff_fixtures()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.network (skipped by default).",
    )
    parser.addoption(
        "--network-strict",
        action="store_true",
        default=False,
        help="Fail if any `-m network` test is skipped.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--network") or config.getoption("--network-strict"):
        return  # run everything
    skip_network = pytest.mark.skip(reason="pass --network to run network tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    rep = outcome.get_result()
    if (
        rep.when == "setup"
        and rep.skipped
        and item.config.getoption("--network-strict")
        and "network" in item.keywords
    ):
        rep.outcome = "failed"
        rep.longrepr = (
            "Network test was skipped in --network-strict mode. "
            "This usually means missing credentials, optional dependencies, "
            "or blocked network/DNS in the current environment."
        )


@pytest.fixture
def async_tiff_fixtures() -> Path:
    """Return path to async-tiff image-tiff fixtures, or skip."""
    if ASYNC_TIFF_FIXTURES is None:
        pytest.skip("Test fixtures not found at fixtures/ directory in repo root.")
    return ASYNC_TIFF_FIXTURES


@pytest.fixture
def async_tiff_bigtiff_fixtures(async_tiff_fixtures: Path) -> Path:
    """Return path to async-tiff BigTIFF fixtures, or skip."""
    p = async_tiff_fixtures / "bigtiff"
    if not p.is_dir():
        pytest.skip("async-tiff BigTIFF fixtures not available")
    return p


@pytest.fixture(autouse=True)
def _clean_registries():
    """Snapshot and restore global registries after each test."""
    band_snap = dict(BandRegistry._maps)
    config_snap = dict(CloudConfig._configs)
    catalog_snap = dict(DatasetRegistry._descriptors)
    yield
    BandRegistry._maps = band_snap
    CloudConfig._configs = config_snap
    DatasetRegistry._descriptors = catalog_snap
