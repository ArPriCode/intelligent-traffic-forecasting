"""Loading and preparing chronological traffic-speed observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

HORIZON_STEPS = {5: 1, 15: 3, 30: 6, 60: 12}
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class TimeSplit:
    """Chronological train, validation, and test boundary indices."""

    train_end: int
    validation_end: int
    size: int


def time_split(
    size: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
) -> TimeSplit:
    """Return boundaries for a chronological split; validation is post-train."""
    if size < 3:
        raise ValueError("At least three timestamps are required.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must sum to less than one.")

    train_end = int(size * train_fraction)
    validation_end = int(size * (train_fraction + validation_fraction))
    if train_end == 0 or validation_end <= train_end or validation_end >= size:
        raise ValueError("Split fractions do not leave non-empty train, validation, and test sets.")
    return TimeSplit(train_end=train_end, validation_end=validation_end, size=size)


def load_speed_csv(
    path: str | Path,
    expected_sensor_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load a timestamp-indexed, five-minute speed CSV and validate its timeline."""
    traffic = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(traffic.index, pd.DatetimeIndex):
        raise ValueError("The first CSV column must contain parseable timestamps.")
    traffic.index.name = "timestamp"
    traffic.columns = traffic.columns.astype(str)
    if traffic.empty or traffic.shape[1] == 0:
        raise ValueError("Traffic CSV must contain timestamps and sensor columns.")
    if not traffic.index.is_monotonic_increasing or traffic.index.has_duplicates:
        raise ValueError("Traffic timestamps must be strictly increasing and unique.")
    intervals = traffic.index.to_series().diff().dropna()
    if not intervals.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("Traffic timestamps must be spaced at five-minute intervals.")
    if expected_sensor_ids is not None and traffic.columns.tolist() != expected_sensor_ids:
        raise ValueError("Traffic sensor columns do not match the expected sensor order.")

    converted = traffic.apply(pd.to_numeric, errors="coerce").astype("float32")
    traffic = pd.DataFrame(converted)
    if not np.isfinite(traffic.to_numpy()).any():
        raise ValueError("Traffic CSV contains no finite speed readings.")
    return traffic


def observed_speeds(traffic: pd.DataFrame) -> FloatArray:
    """Return observed positive speeds; zero and invalid readings are missing."""
    values = traffic.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    return values


def prepare_speed_inputs(
    observed: FloatArray,
    train_end: int,
    *,
    statistic: str = "median",
) -> FloatArray:
    """Causally forward-fill inputs and use training-only sensor statistics initially."""
    values = np.asarray(observed, dtype=np.float32)
    if values.ndim != 2 or not 0 < train_end <= len(values):
        raise ValueError("Expected a 2D speed array and a valid training boundary.")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN.*")
        fill_values = (
            np.nanmean(values[:train_end], axis=0)
            if statistic == "mean"
            else np.nanmedian(values[:train_end], axis=0)
        )
    if not np.isfinite(fill_values).all():
        raise ValueError("Every sensor needs at least one observed training speed.")

    filled = pd.DataFrame(values).ffill().to_numpy(dtype=np.float32)
    filled = np.where(np.isfinite(filled), filled, fill_values[None, :])
    return filled.astype(np.float32, copy=False)
