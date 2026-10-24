"""Chronological validation and tuning for AML traffic models."""

from __future__ import annotations

import itertools
from typing import Any, Generator

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def chronological_cv_splits(
    n_samples: int,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Generate forward-chaining train and validation indices.

    Ensures that validation sets always succeed training sets,
    preventing temporal autocorrelation and sequence leakage.
    """
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")

    min_train = int(n_samples * min_train_fraction)
    remaining = n_samples - min_train
    step = remaining // n_splits
    if step == 0:
        raise ValueError(
            "n_samples is too small for the requested number of splits."
        )

    for i in range(n_splits):
        train_end = min_train + i * step
        val_end = min_train + (i + 1) * step if i < n_splits - 1 else n_samples
        yield np.arange(0, train_end), np.arange(train_end, val_end)


def evaluate_masked_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE over positive finite observations."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_t) & (y_t > 0) & np.isfinite(y_p)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(y_t[mask] - y_p[mask])))


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform chronological grid search for Random Forest regressor."""
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows.")

    default_grid = {
        "n_estimators": [10, 25],
        "max_depth": [6, 10],
        "min_samples_leaf": [4, 8],
    }
    grid = param_grid or default_grid
    keys = list(grid.keys())
    combinations = [
        dict(zip(keys, prod)) for prod in itertools.product(*grid.values())
    ]

    best_score = float("inf")
    best_params = combinations[0]
    results = []

    splits = list(chronological_cv_splits(len(X_arr), n_splits=n_splits))

    for params in combinations:
        fold_scores = []
        for train_idx, val_idx in splits:
            model = RandomForestRegressor(
                **params,
                n_jobs=-1,
                random_state=random_state,
            )
            model.fit(X_arr[train_idx], y_arr[train_idx])
            preds = model.predict(X_arr[val_idx])
            score = evaluate_masked_mae(y_arr[val_idx], preds)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        results.append(
            {"params": params, "mean_mae": mean_score, "folds": fold_scores}
        )
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return {
        "best_params": best_params,
        "best_mae": best_score,
        "results": results,
    }


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform chronological grid search for XGBoost regressor."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed.")

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same number of rows.")

    default_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    grid = param_grid or default_grid
    keys = list(grid.keys())
    combinations = [
        dict(zip(keys, prod)) for prod in itertools.product(*grid.values())
    ]

    best_score = float("inf")
    best_params = combinations[0]
    results = []

    splits = list(chronological_cv_splits(len(X_arr), n_splits=n_splits))

    for params in combinations:
        fold_scores = []
        for train_idx, val_idx in splits:
            model = XGBRegressor(
                **params,
                n_jobs=-1,
                random_state=random_state,
            )
            model.fit(X_arr[train_idx], y_arr[train_idx])
            preds = model.predict(X_arr[val_idx])
            score = evaluate_masked_mae(y_arr[val_idx], preds)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        results.append(
            {"params": params, "mean_mae": mean_score, "folds": fold_scores}
        )
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return {
        "best_params": best_params,
        "best_mae": best_score,
        "results": results,
    }
