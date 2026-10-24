"""Tests for masked forecast metrics and peak/off-peak masks."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.metrics import peak_mask, regression_metrics


def test_regression_metrics_masked_calculation():
    # Valid targets: index 0 (60.0 vs 55.0) and index 3 (70.0 vs 80.0)
    # Outage targets (NaN and 0.0) must be masked out
    y_true = np.array([60.0, np.nan, 0.0, 70.0])
    y_pred = np.array([55.0, 50.0, 40.0, 80.0])

    metrics = regression_metrics(y_true, y_pred)
    assert metrics["valid_test_values"] == 2
    assert pytest.approx(metrics["MAE"], 0.001) == 7.5
    assert pytest.approx(metrics["RMSE"], 0.001) == np.sqrt(62.5)
    assert pytest.approx(metrics["MAPE_percent"], 0.01) == 11.3095


def test_regression_metrics_no_valid():
    y_true = np.array([np.nan, 0.0, -5.0])
    y_pred = np.array([50.0, 50.0, 50.0])
    with pytest.raises(ValueError):
        regression_metrics(y_true, y_pred)


def test_peak_mask_weekdays():
    # Monday 08:30 (Peak), Monday 12:00 (Off-peak), Monday 17:15 (Peak)
    # Saturday 08:30 (Off-peak: weekend), Saturday 17:15 (Off-peak: weekend)
    timestamps = pd.to_datetime(
        [
            "2026-01-05 08:30",  # Mon Peak
            "2026-01-05 12:00",  # Mon Off-peak
            "2026-01-05 17:15",  # Mon Peak
            "2026-01-10 08:30",  # Sat Off-peak
            "2026-01-10 17:15",  # Sat Off-peak
        ]
    )
    mask = peak_mask(timestamps)
    assert list(mask) == [True, False, True, False, False]
