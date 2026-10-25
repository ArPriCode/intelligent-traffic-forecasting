"""Tests for top-level traffic_prediction package exports and lazy loading."""

import pytest

import traffic_prediction


def test_package_exports():
    for name in traffic_prediction.__all__:
        attr = getattr(traffic_prediction, name)
        assert attr is not None, f"Expected {name} to be exported by traffic_prediction"


def test_getattr_invalid():
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = traffic_prediction.non_existent_attribute_xyz
