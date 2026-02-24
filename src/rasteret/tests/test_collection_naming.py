# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from rasteret.core.collection import Collection


def test_create_name_uses_short_sentinel_token() -> None:
    name = Collection.create_name(
        "demo", ("2024-01-01", "2024-01-31"), "sentinel-2-l2a"
    )
    assert name == "demo_202401-01_sentinel"


def test_create_name_keeps_specific_non_builtin_source() -> None:
    name = Collection.create_name(
        "demo", ("2024-01-01", "2024-01-31"), "field-survey-2024"
    )
    assert name == "demo_202401-01_field-survey-2024"


def test_create_name_normalizes_custom_source() -> None:
    name = Collection.create_name(
        "demo",
        ("2024-01-01", "2024-01-31"),
        "custom/my source@v1",
    )
    assert name == "demo_202401-01_custom-my-source-v1"
