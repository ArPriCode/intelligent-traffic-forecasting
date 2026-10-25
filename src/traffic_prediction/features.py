"""Causal lag and cyclic calendar features for traffic-speed models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Encode time-of-day and weekday as cyclic sine/cosine features."""
    minute = (index.hour * 60 + index.minute).to_numpy()
    dow = getattr(index, "day_of_week", None)
    if dow is None:
        dow = index.dayofweek
    weekday = np.asarray(dow)
    time_angle = 2 * np.pi * minute / 1440
    weekday_angle = 2 * np.pi * weekday / 7
    return np.column_stack(
        [
            np.sin(time_angle),
            np.cos(time_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
        ]
    ).astype(np.float32)


def _validate_rows(
    speed_inputs: np.ndarray,
    time_rows: np.ndarray,
    sensor_rows: np.ndarray,
    maximum_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.asarray(speed_inputs)
    times = np.asarray(time_rows, dtype=np.int64)
    sensors = np.asarray(sensor_rows, dtype=np.int64)
    if speed.ndim != 2 or times.ndim != 1 or sensors.ndim != 1:
        raise ValueError(
            "Speed inputs must be 2D and row/sensor indices must be 1D."
        )
    if len(times) != len(sensors):
        raise ValueError("Each time row must have a corresponding sensor row.")
    if len(times) and (
        times.min() < maximum_lag
        or times.max() >= speed.shape[0]
        or sensors.min() < 0
        or sensors.max() >= speed.shape[1]
    ):
        raise IndexError(
            "Time or sensor index falls outside the available speed data."
        )
    return times, sensors


def make_sensor_features(
    speed_inputs: np.ndarray,
    calendar_features: np.ndarray,
    time_rows: np.ndarray,
    sensor_index: int,
) -> np.ndarray:
    """Create RF features using current speed, lags 1/2/5, and clock."""
    speed = np.asarray(speed_inputs, dtype=np.float32)
    clock = np.asarray(calendar_features, dtype=np.float32)
    times = np.asarray(time_rows, dtype=np.int64)
    if not (0 <= sensor_index < speed.shape[1]):
        raise IndexError("sensor_index falls outside the available sensor dimension.")
    _validate_rows(speed, times, np.full(len(times), sensor_index), 5)
    if clock.shape != (speed.shape[0], 4):
        raise ValueError("calendar_features must have shape (timestamps, 4).")
    return np.column_stack(
        [
            speed[times, sensor_index],
            speed[times - 1, sensor_index],
            speed[times - 2, sensor_index],
            speed[times - 5, sensor_index],
            clock[times],
        ]
    ).astype(np.float32, copy=False)


def make_panel_features(
    speed_inputs: np.ndarray,
    calendar_features: np.ndarray,
    sensor_ids: list[str],
    time_rows: np.ndarray,
    sensor_rows: np.ndarray,
) -> pd.DataFrame:
    """Create pooled XGBoost features, including sensor identity."""
    speed = np.asarray(speed_inputs, dtype=np.float32)
    clock = np.asarray(calendar_features, dtype=np.float32)
    times, sensors = _validate_rows(speed, time_rows, sensor_rows, 5)
    if clock.shape != (speed.shape[0], 4):
        raise ValueError("calendar_features must have shape (timestamps, 4).")
    if len(sensor_ids) != speed.shape[1]:
        raise ValueError(
            "sensor_ids must match the sensor dimension of speed_inputs."
        )
    if len(sensors) == 0:
        raise ValueError("At least one row is required for panel features.")

    features = pd.DataFrame(
        {
            "speed_now": speed[times, sensors],
            "speed_lag_1": speed[times - 1, sensors],
            "speed_lag_2": speed[times - 2, sensors],
            "speed_lag_5": speed[times - 5, sensors],
            "time_sin": clock[times, 0],
            "time_cos": clock[times, 1],
            "weekday_sin": clock[times, 2],
            "weekday_cos": clock[times, 3],
            "sensor_id": pd.Categorical(
                np.asarray(sensor_ids, dtype=str)[sensors],
                categories=np.asarray(sensor_ids, dtype=str),
            ),
        }
    )
    return features


def make_spatial_lag_features(
    speed_inputs: np.ndarray,
    adjacency: np.ndarray,
    time_rows: np.ndarray,
    sensor_index: int,
) -> np.ndarray:
    """Compute 1-hop spatial neighbor speeds (current and lag 1)."""
    speed = np.asarray(speed_inputs, dtype=np.float32)
    adj = np.asarray(adjacency, dtype=np.float32)
    times = np.asarray(time_rows, dtype=np.int64)
    if (
        adj.ndim != 2
        or adj.shape[0] != adj.shape[1]
        or adj.shape[0] != speed.shape[1]
    ):
        raise ValueError("adjacency must match square sensor dimension.")
    if not (0 <= sensor_index < speed.shape[1]):
        raise IndexError("sensor_index falls outside the available sensor dimension.")
    _validate_rows(speed, times, np.full(len(times), sensor_index), 5)

    weights = adj[sensor_index].copy()
    weights[sensor_index] = 0.0  # exclude self-loop
    weight_sum = weights.sum()
    if weight_sum > 0:
        norm_weights = weights / weight_sum
        neighbor_now = speed[times] @ norm_weights
        neighbor_lag1 = speed[times - 1] @ norm_weights
    else:
        neighbor_now = speed[times, sensor_index]
        neighbor_lag1 = speed[times - 1, sensor_index]

    return np.column_stack([neighbor_now, neighbor_lag1]).astype(
        np.float32, copy=False
    )


def make_rolling_features(
    speed_inputs: np.ndarray,
    time_rows: np.ndarray,
    sensor_index: int,
    window_sizes: tuple[int, ...] = (3, 6),
) -> np.ndarray:
    """Compute causal rolling window mean and standard deviation."""
    speed = np.asarray(speed_inputs, dtype=np.float32)
    times = np.asarray(time_rows, dtype=np.int64)
    if not window_sizes or any(w <= 0 for w in window_sizes):
        raise ValueError("window_sizes must be non-empty and contain positive integers.")
    if not (0 <= sensor_index < speed.shape[1]):
        raise IndexError("sensor_index falls outside the available sensor dimension.")
    max_window = max(window_sizes)
    _validate_rows(speed, times, np.full(len(times), sensor_index), max_window)

    cols = []
    sensor_series = speed[:, sensor_index]
    for w in window_sizes:
        window_matrix = np.stack(
            [sensor_series[times - k] for k in range(w)], axis=1
        )
        mean_w = np.mean(window_matrix, axis=1)
        std_w = np.std(window_matrix, axis=1)
        cols.extend([mean_w, std_w])

    return np.column_stack(cols).astype(np.float32, copy=False)


def make_enhanced_sensor_features(
    speed_inputs: np.ndarray,
    calendar_features: np.ndarray,
    adjacency: np.ndarray,
    time_rows: np.ndarray,
    sensor_index: int,
) -> np.ndarray:
    """Combine own lags, clock, spatial neighbors, and rolling stats."""
    base_feats = make_sensor_features(
        speed_inputs, calendar_features, time_rows, sensor_index
    )
    spatial_feats = make_spatial_lag_features(
        speed_inputs, adjacency, time_rows, sensor_index
    )
    rolling_feats = make_rolling_features(
        speed_inputs, time_rows, sensor_index, window_sizes=(3, 6)
    )
    return np.column_stack(
        [base_feats, spatial_feats, rolling_feats]
    ).astype(np.float32, copy=False)
