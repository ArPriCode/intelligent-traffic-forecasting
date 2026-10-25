"""Tests for data loading, validation, and chronological splitting."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data import (
    TimeSplit,
    observed_speeds,
    prepare_speed_inputs,
    time_split,
)


def test_time_split_boundaries():
    split = time_split(100, train_fraction=0.70, validation_fraction=0.10)
    assert isinstance(split, TimeSplit)
    assert split.train_end == 70
    assert split.validation_end == 80
    assert split.size == 100
    assert split.train_end < split.validation_end < split.size


def test_time_split_invalid_fractions():
    with pytest.raises(ValueError):
        time_split(100, train_fraction=0.80, validation_fraction=0.25)
    with pytest.raises(ValueError):
        time_split(2)


def test_observed_speeds_masking():
    # 0 and negative speeds must be converted to NaN (sensor outages)
    df = pd.DataFrame(
        {
            "sensor_1": [65.0, 0.0, -1.0, 55.0],
            "sensor_2": [np.nan, 70.0, 60.0, 0.0],
        }
    )
    observed = observed_speeds(df)
    assert observed.dtype == np.float32
    assert observed.shape == (4, 2)
    assert observed[0, 0] == 65.0
    assert np.isnan(observed[1, 0])  # zero speed -> outage
    assert np.isnan(observed[2, 0])  # negative -> outage
    assert observed[3, 0] == 55.0
    assert np.isnan(observed[0, 1])
    assert observed[1, 1] == 70.0
    assert np.isnan(observed[3, 1])


def test_prepare_speed_inputs_causal_forward_fill():
    # Test that causal forward fill works and fallback uses train-only median
    observed = np.array(
        [
            [np.nan, 50.0],  # initial NaN filled with train median
            [60.0, np.nan],  # sensor 2 forward fills 50.0
            [np.nan, 70.0],  # sensor 1 forward fills 60.0
            [65.0, np.nan],  # sensor 2 forward fills 70.0
        ],
        dtype=np.float32,
    )
    # train_end = 2: train medians are sensor 1: 60.0, sensor 2: 50.0
    filled = prepare_speed_inputs(observed, train_end=2, statistic="median")
    assert filled.shape == (4, 2)
    assert filled[0, 0] == 60.0  # fell back to train median
    assert filled[1, 0] == 60.0
    assert filled[2, 0] == 60.0  # forward filled from row 1
    assert filled[3, 0] == 65.0

    assert filled[0, 1] == 50.0
    assert filled[1, 1] == 50.0  # forward filled from row 0
    assert filled[2, 1] == 70.0
    assert filled[3, 1] == 70.0  # forward filled from row 2


def test_prepare_speed_inputs_invalid_inputs():
    with pytest.raises(ValueError, match="statistic must be"):
        prepare_speed_inputs(np.ones((5, 2)), train_end=3, statistic="mode")

    with pytest.raises(ValueError, match="valid training boundary"):
        prepare_speed_inputs(np.ones((5, 2)), train_end=0)

    # All NaNs in train
    all_nan = np.full((5, 2), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Every sensor needs at least one"):
        prepare_speed_inputs(all_nan, train_end=3)


def test_load_speed_csv_valid_and_invalid(tmp_path):
    from traffic_prediction.data import load_speed_csv

    # 1. Valid CSV
    dates = pd.date_range("2026-01-01 00:00", periods=5, freq="5min")
    df = pd.DataFrame({"s1": [50.0, 52.0, 54.0, 55.0, 56.0]}, index=dates)
    csv_path = tmp_path / "valid.csv"
    df.to_csv(csv_path)

    loaded = load_speed_csv(csv_path, expected_sensor_ids=["s1"])
    assert loaded.shape == (5, 1)

    # 2. Sensor mismatch
    with pytest.raises(ValueError, match="expected sensor order"):
        load_speed_csv(csv_path, expected_sensor_ids=["s2"])

    # 3. Irregular intervals
    dates_bad = pd.date_range("2026-01-01 00:00", periods=5, freq="10min")
    df_bad = pd.DataFrame({"s1": [50.0, 52.0, 54.0, 55.0, 56.0]}, index=dates_bad)
    bad_csv = tmp_path / "bad_interval.csv"
    df_bad.to_csv(bad_csv)
    with pytest.raises(ValueError, match="five-minute intervals"):
        load_speed_csv(bad_csv)
