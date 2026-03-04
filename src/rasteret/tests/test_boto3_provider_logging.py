# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_try_boto3_credential_provider_logs_on_unexpected_error(caplog) -> None:
    cog = pytest.importorskip("rasteret.fetch.cog")
    pytest.importorskip("obstore.auth.boto3")

    with patch(
        "obstore.auth.boto3.Boto3CredentialProvider", side_effect=RuntimeError("boom")
    ):
        provider = cog._try_boto3_credential_provider()

    assert provider is None
    assert "Failed to create Boto3CredentialProvider" in caplog.text
