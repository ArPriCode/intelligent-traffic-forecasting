"""Unit and regression tests for traffic_prediction components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from traffic_prediction.data import (
    observed_speeds,
    prepare_speed_inputs,
    time_split,
)
from traffic_prediction.features import (
    make_panel_features,
    make_sensor_features,
    time_features,
)
from traffic_prediction.graph import build_distance_adjacency
from traffic_prediction.metrics import peak_mask, regression_metrics
from traffic_prediction.models import GraphAttentionLayer, GraphAttentionNetwork


def test_time_split() -> None:
    split = time_split(100, train_fraction=0.70, validation_fraction=0.10)
    assert split.train_end == 70
    assert split.validation_end == 80
    assert split.size == 100

    with pytest.raises(ValueError):
        time_split(2)
    with pytest.raises(ValueError):
        time_split(100, train_fraction=0.8, validation_fraction=0.25)


def test_observed_speeds_and_imputation() -> None:
    raw = pd.DataFrame(
        {
            "s1": [50.0, 0.0, 55.0, np.nan, 60.0],
            "s2": [0.0, 30.0, 35.0, 40.0, 45.0],
        }
    )
    observed = observed_speeds(raw)
    assert np.isnan(observed[1, 0])  # 0.0 should be converted to NaN
    assert np.isnan(observed[3, 0])  # NaN remains NaN

    filled = prepare_speed_inputs(observed, train_end=3, statistic="mean")
    assert np.isfinite(filled).all()
    assert filled.shape == (5, 2)


def test_features() -> None:
    dates = pd.date_range("2024-01-01 00:00", periods=10, freq="5min")
    calendar = time_features(dates)
    assert calendar.shape == (10, 4)
    assert np.all(calendar >= -1.0) and np.all(calendar <= 1.0)

    speeds = np.full((10, 2), 50.0, dtype=np.float32)
    time_rows = np.array([5, 6, 7])
    sensor_features = make_sensor_features(speeds, calendar, time_rows, sensor_index=0)
    assert sensor_features.shape == (3, 8)

    panel = make_panel_features(
        speeds, calendar, ["s1", "s2"], time_rows, np.array([0, 1, 0])
    )
    assert len(panel) == 3
    assert "sensor_id" in panel.columns
    assert "speed_now" in panel.columns


def test_graph_construction() -> None:
    distances = pd.DataFrame(
        {
            "from": ["1", "2", "1", "3"],
            "to": ["2", "1", "3", "1"],
            "cost": [10.0, 10.0, 25.0, 25.0],
        }
    )
    sensor_ids = ["1", "2", "3"]
    adj = build_distance_adjacency(distances, sensor_ids, threshold=0.01)
    assert adj.shape == (3, 3)
    assert np.allclose(np.diag(adj), 1.0)
    assert np.allclose(adj, adj.T)


def test_regression_metrics() -> None:
    actual = np.array([50.0, 60.0, 0.0, np.nan, 70.0])
    pred = np.array([48.0, 62.0, 10.0, 50.0, 70.0])
    res = regression_metrics(actual, pred)
    assert res["valid_test_values"] == 3
    assert np.isclose(res["MAE"], (2.0 + 2.0 + 0.0) / 3)


def test_peak_mask() -> None:
    # Monday 08:00 (peak) vs Monday 12:00 (offpeak) vs Sunday 08:00 (offpeak)
    dates = pd.to_datetime(
        ["2024-01-01 08:00", "2024-01-01 12:00", "2024-01-07 08:00"]
    )
    mask = peak_mask(dates)
    assert mask[0] == True
    assert mask[1] == False
    assert mask[2] == False


def test_gat_model() -> None:
    adj = torch.eye(5)
    model = GraphAttentionNetwork(adjacency=adj, input_size=10, hidden_size=16, output_size=4)
    x = torch.randn(2, 5, 10)
    out = model(x)
    assert out.shape == (2, 5, 4)


def test_sensor_analysis_artifacts() -> None:
    from pathlib import Path
    metrics_path = Path("results/metrics/metr-la_sample_sensor_analysis.csv")
    fig_path = Path("results/figures/29_metr-la_sample_sensor_analysis.png")
    assert metrics_path.exists(), "Sensor analysis metrics CSV should exist."
    assert fig_path.exists(), "Sensor analysis figure PNG should exist."
    df = pd.read_csv(metrics_path)
    assert len(df) >= 1
    assert "mean_speed_mph" in df.columns
    assert "Persistence_MAE_5m" in df.columns

