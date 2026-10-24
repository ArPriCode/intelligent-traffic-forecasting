"""Tests for distance graph and adjacency matrix construction."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.graph import build_distance_adjacency


def test_build_distance_adjacency_structure():
    sensor_ids = ["s1", "s2", "s3"]
    distances = pd.DataFrame(
        {
            "from": ["s1", "s2", "s2", "s3"],
            "to": ["s2", "s1", "s3", "s2"],
            "cost": [1000.0, 1000.0, 2500.0, 2500.0],
        }
    )
    adj = build_distance_adjacency(distances, sensor_ids, threshold=0.1)

    assert adj.shape == (3, 3)
    assert adj.dtype == np.float32

    # Diagonal must be 1.0 (self-loops)
    np.testing.assert_allclose(np.diag(adj), 1.0)

    # Matrix must be strictly symmetric
    np.testing.assert_allclose(adj, adj.T)

    # All values must be in [0, 1]
    assert np.all(adj >= 0.0) and np.all(adj <= 1.0)


def test_build_distance_adjacency_missing_columns():
    sensor_ids = ["s1", "s2"]
    distances = pd.DataFrame(
        {"source": ["s1"], "target": ["s2"], "cost": [10.0]}
    )
    with pytest.raises(ValueError, match="Distance table must include"):
        build_distance_adjacency(distances, sensor_ids)
