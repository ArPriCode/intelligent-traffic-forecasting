"""Tests for scripts (data verification and sensor analysis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_sensor_analysis import analyze_sensors
from scripts.verify_data import check_graph_files, check_traffic_csv


def test_check_graph_files():
    # Graph directory is checked in repository
    assert check_graph_files() is True


def test_check_traffic_csv_missing(tmp_path):
    # Non-existent file
    assert check_traffic_csv("non_existent_file.csv") is False


def test_analyze_sensors_with_custom_dataframe():
    dates = pd.date_range("2026-01-01 00:00", periods=36, freq="5min")
    np.random.seed(42)
    speeds = 55.0 + np.random.randn(36, 2) * 5.0
    speeds[speeds <= 0] = 50.0
    df = pd.DataFrame(speeds, index=dates, columns=["s1", "s2"])

    res = analyze_sensors(
        sensor_ids=["s1", "s2"],
        dataset="metr-la",
        plot_hours=2,
        custom_df=df,
        save_artifacts=False,
    )
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert "mean_speed_mph" in res.columns
    assert "Persistence_MAE_5m" in res.columns
    assert "Persistence_MAE_60m" in res.columns
