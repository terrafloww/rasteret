# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Tests for core utility functions."""

from __future__ import annotations

import pytest

from rasteret.core.utils import (
    compute_dst_grid_from_src,
    normalize_transform,
    transform_bbox,
)


class TestNormalizeTransform:
    def test_four_element(self):
        sx, tx, sy, ty = normalize_transform([10.0, 100.0, -10.0, 200.0])
        assert sx == 10.0
        assert tx == 100.0
        assert sy == -10.0
        assert ty == 200.0

    def test_six_element_north_up(self):
        # GDAL affine (a, b, c, d, e, f) for north-up raster
        sx, tx, sy, ty = normalize_transform([10.0, 0.0, 100.0, 0.0, -10.0, 200.0])
        assert sx == 10.0
        assert tx == 100.0
        assert sy == -10.0
        assert ty == 200.0

    def test_six_element_rotated_raises(self):
        with pytest.raises(ValueError, match="Rotated"):
            normalize_transform([10.0, 0.5, 100.0, 0.5, -10.0, 200.0])

    def test_none_raises(self):
        with pytest.raises(ValueError, match="missing"):
            normalize_transform(None)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="4 or 6"):
            normalize_transform([1.0, 2.0, 3.0])

    def test_non_iterable_raises(self):
        with pytest.raises(TypeError, match="iterable"):
            normalize_transform(42)


class TestComputeDstGridFromSrc:
    """Tests for compute_dst_grid_from_src (GDAL-backed grid computation)."""

    def test_identity_same_crs(self):
        """Same CRS → output dimensions should match input."""
        tf, shape = compute_dst_grid_from_src(
            32632,
            32632,
            100,
            100,
            (500000, 5500000, 501000, 5501000),
        )
        assert shape[0] == 100
        assert shape[1] == 100
        assert abs(tf.a) == pytest.approx(10.0, rel=0.01)

    def test_utm_to_utm(self):
        """UTM zone 32N → 33N: both metres, dimensions similar."""
        tf, shape = compute_dst_grid_from_src(
            32632,
            32633,
            100,
            100,
            (500000, 5500000, 501000, 5501000),
        )
        # Output should have similar pixel count (same linear units)
        assert shape[0] > 50
        assert shape[1] > 50
        # Pixel size should still be ~10m
        assert abs(tf.a) == pytest.approx(10.0, rel=0.5)
        assert abs(tf.e) == pytest.approx(10.0, rel=0.5)

    def test_utm_to_4326_cross_unit(self):
        """UTM metres → EPSG:4326 degrees: the critical cross-unit case.

        This is the bug that compute_dst_grid would produce 0×0 pixels for,
        because it would try round(~0.01° / 10m) = 0.
        """
        tf, shape = compute_dst_grid_from_src(
            32632,
            4326,
            100,
            100,
            (500000, 5500000, 501000, 5501000),
        )
        # Must produce non-zero dimensions
        assert shape[0] > 0, "Cross-unit CRS produced 0-height output"
        assert shape[1] > 0, "Cross-unit CRS produced 0-width output"
        # Pixel size should be in degrees (~0.0001°), NOT metres
        assert abs(tf.a) < 1.0, f"Pixel size {abs(tf.a)} looks like metres, not degrees"
        assert abs(tf.e) < 1.0, f"Pixel size {abs(tf.e)} looks like metres, not degrees"
        # Dimensions should be >50 (preserving spatial information)
        assert shape[0] > 50
        assert shape[1] > 50

    def test_4326_to_utm(self):
        """EPSG:4326 degrees → UTM metres: reverse cross-unit case."""
        tf, shape = compute_dst_grid_from_src(
            4326,
            32632,
            100,
            100,
            (9.0, 48.0, 9.01, 48.01),
        )
        assert shape[0] > 0
        assert shape[1] > 0
        # Pixel size should be in metres
        assert abs(tf.a) > 1.0, f"Pixel size {abs(tf.a)} looks like degrees, not metres"


class TestTransformBbox:
    def test_basic_crs_conversion(self):
        # WGS84 bbox for a small area → UTM zone 32N (EPSG:32632)
        bbox_4326 = (9.0, 48.0, 10.0, 49.0)
        result = transform_bbox(bbox_4326, 4326, 32632)
        assert len(result) == 4
        minx, miny, maxx, maxy = result
        assert minx < maxx
        assert miny < maxy
        # UTM coordinates should be large numbers (~500k easting, ~5M northing)
        assert minx > 100_000
        assert miny > 1_000_000
