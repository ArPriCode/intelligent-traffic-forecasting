"""Tests for causal lag, cyclical, spatial neighbor, and rolling features."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.features import (
    make_enhanced_sensor_features,
    make_panel_features,
    make_rolling_features,
    make_sensor_features,
    make_spatial_lag_features,
    time_features,
)


@pytest.fixture
def sample_traffic_series():
    """Create synthetic speed series of 20 timestamps for 3 sensors."""
    np.random.seed(42)
    timestamps = pd.date_range("2026-01-01 08:00", periods=20, freq="5min")
    speed = np.zeros((20, 3), dtype=np.float32)
    for s in range(3):
        speed[:, s] = 60.0 + s * 5.0 + np.arange(20) * 0.5
    clock = time_features(timestamps)
    return timestamps, speed, clock


def test_time_features(sample_traffic_series):
    timestamps, _, clock = sample_traffic_series
    assert clock.shape == (20, 4)
    assert np.all(clock >= -1.0) and np.all(clock <= 1.0)
    tod_norm = clock[:, 0] ** 2 + clock[:, 1] ** 2
    dow_norm = clock[:, 2] ** 2 + clock[:, 3] ** 2
    np.testing.assert_allclose(tod_norm, 1.0, atol=1e-5)
    np.testing.assert_allclose(dow_norm, 1.0, atol=1e-5)


def test_make_sensor_features_causality(sample_traffic_series):
    _, speed, clock = sample_traffic_series
    time_rows = np.array([5, 10, 15], dtype=np.int64)
    feats = make_sensor_features(speed, clock, time_rows, sensor_index=0)
    assert feats.shape == (3, 8)  # 4 speed lags + 4 clock features

    # For row 5: lags are 5, 4, 3, 0 (t, t-1, t-2, t-5)
    np.testing.assert_allclose(feats[0, 0], speed[5, 0])
    np.testing.assert_allclose(feats[0, 1], speed[4, 0])
    np.testing.assert_allclose(feats[0, 2], speed[3, 0])
    np.testing.assert_allclose(feats[0, 3], speed[0, 0])


def test_make_spatial_lag_features(sample_traffic_series):
    _, speed, _ = sample_traffic_series
    adj = np.array(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.0],
            [0.2, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    time_rows = np.array([5, 6], dtype=np.int64)
    spatial_feats = make_spatial_lag_features(
        speed, adj, time_rows, sensor_index=0
    )
    assert spatial_feats.shape == (2, 2)

    expected_now_5 = 0.8 * speed[5, 1] + 0.2 * speed[5, 2]
    expected_lag1_5 = 0.8 * speed[4, 1] + 0.2 * speed[4, 2]
    np.testing.assert_allclose(spatial_feats[0, 0], expected_now_5, atol=1e-5)
    np.testing.assert_allclose(spatial_feats[0, 1], expected_lag1_5, atol=1e-5)


def test_make_rolling_features(sample_traffic_series):
    _, speed, _ = sample_traffic_series
    time_rows = np.array([6, 7], dtype=np.int64)
    rolling = make_rolling_features(
        speed, time_rows, sensor_index=1, window_sizes=(3, 6)
    )
    assert rolling.shape == (2, 4)

    expected_mean_w3 = np.mean([speed[6, 1], speed[5, 1], speed[4, 1]])
    np.testing.assert_allclose(rolling[0, 0], expected_mean_w3, atol=1e-5)


def test_make_enhanced_sensor_features(sample_traffic_series):
    _, speed, clock = sample_traffic_series
    adj = np.eye(3, dtype=np.float32)
    adj[0, 1] = adj[1, 0] = 0.5
    time_rows = np.array([6, 8, 10], dtype=np.int64)
    enhanced = make_enhanced_sensor_features(
        speed, clock, adj, time_rows, sensor_index=0
    )
    # 8 (base) + 2 (spatial) + 4 (rolling) = 14 features
    assert enhanced.shape == (3, 14)
    assert np.isfinite(enhanced).all()


def test_make_panel_features(sample_traffic_series):
    _, speed, clock = sample_traffic_series
    sensor_ids = ["s0", "s1", "s2"]
    time_rows = np.array([5, 5, 6, 6], dtype=np.int64)
    sensor_rows = np.array([0, 1, 0, 1], dtype=np.int64)
    panel = make_panel_features(
        speed, clock, sensor_ids, time_rows, sensor_rows
    )
    assert isinstance(panel, pd.DataFrame)
    assert len(panel) == 4
    assert "speed_now" in panel.columns
    assert "sensor_id" in panel.columns
    assert list(panel["sensor_id"]) == ["s0", "s1", "s0", "s1"]
