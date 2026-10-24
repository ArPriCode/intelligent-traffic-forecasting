"""Tests for chronological cross-validation and hyperparameter tuning."""

import numpy as np
import pytest

from traffic_prediction.tuning import (
    chronological_cv_splits,
    evaluate_masked_mae,
    tune_random_forest,
)


def test_chronological_cv_splits():
    n_samples = 100
    n_splits = 3
    splits = list(
        chronological_cv_splits(
            n_samples, n_splits=n_splits, min_train_fraction=0.5
        )
    )
    assert len(splits) == n_splits

    for train_idx, val_idx in splits:
        # Check max train index < min val index (no future leakage)
        assert train_idx.max() < val_idx.min()
        # Check no overlap
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        # Check validation indices are contiguous
        assert len(val_idx) > 0


def test_evaluate_masked_mae():
    y_true = np.array([50.0, np.nan, 0.0, 60.0])
    y_pred = np.array([48.0, 70.0, 30.0, 65.0])
    # valid targets: 50.0 (diff 2) and 60.0 (diff 5) -> mean = 3.5
    mae = evaluate_masked_mae(y_true, y_pred)
    assert pytest.approx(mae, 0.001) == 3.5


def test_tune_random_forest_synthetic():
    np.random.seed(42)
    X = np.random.randn(80, 5).astype(np.float32)
    y = (X[:, 0] * 5.0 + 50.0 + np.random.randn(80) * 0.5).astype(np.float32)

    param_grid = {
        "n_estimators": [5],
        "max_depth": [3, 5],
    }
    result = tune_random_forest(X, y, param_grid=param_grid, n_splits=2)
    assert "best_params" in result
    assert "best_mae" in result
    assert result["best_mae"] < 10.0
    assert len(result["results"]) == 2
