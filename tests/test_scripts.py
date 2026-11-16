"""Tests for scripts (data verification and sensor analysis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_predictions import evaluate_prediction_file
from scripts.generate_benchmark_summary import generate_benchmark_summary
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


def test_evaluate_prediction_file(tmp_path):
    dates = pd.date_range("2026-01-05 08:00", periods=20, freq="5min")
    actual = np.full(20, 60.0)
    pred = np.full(20, 58.0)
    pred[0] = 50.0  # single larger error

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "actual_speed": actual,
            "predicted_speed": pred,
        }
    )
    csv_file = tmp_path / "test_preds.csv"
    df.to_csv(csv_file, index=False)

    summary = evaluate_prediction_file(csv_file)
    assert summary["file"] == "test_preds.csv"
    assert summary["valid_samples"] == 20
    assert summary["MAE"] > 0
    assert summary["RMSE"] > 0
    assert "p50_MAE" in summary
    assert "p90_MAE" in summary
    assert "peak_MAE" in summary
    assert "offpeak_MAE" in summary


def test_generate_benchmark_summary(tmp_path):
    df = pd.DataFrame(
        {
            "dataset": ["METR-LA"] * 4,
            "model": ["Persistence", "Random Forest", "Persistence", "Random Forest"],
            "horizon_minutes": [15, 15, 15, 15],
            "MAE": [3.5, 3.1, 4.5, 4.3],
            "RMSE": [6.4, 5.7, 8.2, 7.5],
            "MAPE_percent": [8.6, 8.4, 14.3, 15.1],
            "period": ["All", "All", "Peak", "Peak"],
        }
    )
    csv_file = tmp_path / "model_comp.csv"
    df.to_csv(csv_file, index=False)
    out_md = tmp_path / "summary.md"

    text = generate_benchmark_summary(csv_file, output_md=out_md)
    assert "Benchmark Executive Summary" in text
    assert "Persistence" in text
    assert "Random Forest" in text
    assert out_md.is_file()
