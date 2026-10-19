"""Masked forecast metrics and time-period labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    """Compute MAE, MAPE (%), and RMSE over finite positive targets."""
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    valid = np.isfinite(y_true) & (y_true > 0) & np.isfinite(y_pred)
    if not valid.any():
        raise ValueError("No finite positive targets are available for scoring.")

    errors = y_true[valid] - y_pred[valid]
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "MAPE_percent": float(np.mean(np.abs(errors) / y_true[valid]) * 100),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "valid_test_values": int(valid.sum()),
    }


def peak_mask(
    timestamps: pd.DatetimeIndex,
    target_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Mark weekday targets in 07:00–09:59 or 16:00–18:59 as peak."""
    target_index = timestamps
    if target_rows is not None:
        rows = np.asarray(target_rows, dtype=np.int64)
        if rows.ndim != 1 or (len(rows) and (rows.min() < 0 or rows.max() >= len(timestamps))):
            raise IndexError("Target row falls outside the timestamp index.")
        target_index = timestamps[rows]

    hours = target_index.hour.to_numpy()
    weekdays = target_index.dayofweek.to_numpy()
    return (weekdays < 5) & (
        ((hours >= 7) & (hours < 10)) | ((hours >= 16) & (hours < 19))
    )
