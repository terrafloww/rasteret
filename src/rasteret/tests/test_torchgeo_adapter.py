# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for the TorchGeo adapter: sample format, bounds tensor, no crs key."""

from __future__ import annotations

import numpy as np
import pytest

from rasteret.integrations.torchgeo import (
    _array_to_image_tensor_torchgeo_compatible,
    _coerce_label_value,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchgeo")
from torchgeo.datasets import GeoDataset  # noqa: E402


class TestTorchGeo09Compat:
    """Verify forward-compatibility with TorchGeo 0.9.0 API."""

    def test_slice_to_tensor_available(self):
        """TorchGeo 0.9.0 adds _slice_to_tensor to GeoDataset."""
        assert hasattr(GeoDataset, "_slice_to_tensor")

    def test_sample_dict_shape(self):
        """Sample dict should have image, bounds, transform, no crs key."""
        # Simulate the sample dict our adapter builds
        sample = {
            "image": torch.zeros(3, 64, 64, dtype=torch.float32),
            "bounds": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "transform": torch.tensor([10.0, 0.0, 500000.0, 0.0, -10.0, 1000000.0]),
        }
        assert "crs" not in sample
        assert "image" in sample
        assert "bounds" in sample
        assert "transform" in sample
        assert sample["image"].shape[0] == 3  # C, H, W

    def test_bounds_is_tensor(self):
        """In TorchGeo 0.9+, bounds must be a Tensor, not a GeoSlice."""
        bounds = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert isinstance(bounds, torch.Tensor)


class TestLabelCoercion:
    def test_integer_label_to_long_tensor(self):
        label = _coerce_label_value(3)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert int(label.item()) == 3

    def test_float_label_to_float_tensor(self):
        label = _coerce_label_value(0.25)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.float32
        assert float(label.item()) == 0.25

    def test_list_label_to_tensor(self):
        label = _coerce_label_value([1, 2, 3])
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert label.tolist() == [1, 2, 3]

    def test_string_label_passthrough(self):
        label = _coerce_label_value("forest")
        assert label == "forest"

    def test_nan_label_becomes_none(self):
        assert _coerce_label_value(np.nan) is None


class TestImageTensorConversion:
    def test_uint16_casts_to_int32_with_cast_info(self):
        arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
        tensor, cast_info = _array_to_image_tensor_torchgeo_compatible(arr)
        assert tensor.dtype == torch.int32
        assert cast_info == (np.dtype(np.uint16), np.dtype(np.int32))
        np.testing.assert_array_equal(tensor.numpy(), arr.astype(np.int32))

    def test_uint32_casts_to_int64_with_cast_info(self):
        arr = np.arange(6, dtype=np.uint32).reshape(2, 3)
        tensor, cast_info = _array_to_image_tensor_torchgeo_compatible(arr)
        assert tensor.dtype == torch.int64
        assert cast_info == (np.dtype(np.uint32), np.dtype(np.int64))
        np.testing.assert_array_equal(tensor.numpy(), arr.astype(np.int64))

    def test_uint8_uses_zero_copy_when_contiguous(self):
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        tensor, cast_info = _array_to_image_tensor_torchgeo_compatible(arr)
        assert cast_info is None
        assert tensor.dtype == torch.uint8
        arr[0, 0] = 255
        assert int(tensor[0, 0].item()) == 255
