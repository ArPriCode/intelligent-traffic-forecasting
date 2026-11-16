"""Tests for distance graph and adjacency matrix construction."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.graph import (
    build_directed_distance_adjacency,
    build_distance_adjacency,
    compute_dual_diffusion_matrices,
    to_networkx_graph,
)


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


def test_build_distance_adjacency_invalid_inputs():
    distances = pd.DataFrame(
        {"from": ["s1"], "to": ["s2"], "cost": [10.0]}
    )
    # Empty sensor_ids
    with pytest.raises(ValueError, match="non-empty and unique"):
        build_distance_adjacency(distances, [])

    # Duplicate sensor_ids
    with pytest.raises(ValueError, match="non-empty and unique"):
        build_distance_adjacency(distances, ["s1", "s1"])

    # Invalid threshold
    with pytest.raises(ValueError, match="threshold must be between zero and one"):
        build_distance_adjacency(distances, ["s1", "s2"], threshold=0.0)

    # Negative distance
    neg_dist = pd.DataFrame({"from": ["s1"], "to": ["s2"], "cost": [-5.0]})
    with pytest.raises(ValueError, match="finite non-negative"):
        build_distance_adjacency(neg_dist, ["s1", "s2"])


def test_build_directed_distance_adjacency():
    sensor_ids = ["s1", "s2", "s3"]
    # Only s1 -> s2, no s2 -> s1
    distances = pd.DataFrame(
        {
            "from": ["s1", "s2"],
            "to": ["s2", "s3"],
            "cost": [100.0, 200.0],
        }
    )
    adj = build_directed_distance_adjacency(distances, sensor_ids, threshold=0.01)
    assert adj.shape == (3, 3)
    np.testing.assert_allclose(np.diag(adj), 1.0)
    # Directed: s1 -> s2 should be positive, s2 -> s1 should be 0.0
    assert adj[0, 1] > 0.0
    assert adj[1, 0] == 0.0


def test_compute_dual_diffusion_matrices():
    adj = np.array(
        [
            [1.0, 0.8, 0.0],
            [0.0, 1.0, 0.5],
            [0.2, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    P_f, P_b = compute_dual_diffusion_matrices(adj)
    assert P_f.shape == (3, 3)
    assert P_b.shape == (3, 3)
    # Row sums of forward transition must be approx 1.0
    np.testing.assert_allclose(P_f.sum(axis=1), np.ones(3), rtol=1e-5)
    np.testing.assert_allclose(P_b.sum(axis=1), np.ones(3), rtol=1e-5)


def test_to_networkx_graph():
    """Verify conversion of adjacency matrix to NetworkX Graph and DiGraph."""
    import networkx as nx

    adj = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.8],
            [0.0, 0.8, 1.0],
        ],
        dtype=np.float32,
    )
    sensor_ids = ["s1", "s2", "s3"]
    locations = pd.DataFrame(
        {
            "sensor_id": ["s1", "s2", "s3"],
            "latitude": [34.0, 34.1, 34.2],
            "longitude": [-118.0, -118.1, -118.2],
        }
    )

    # Test undirected graph
    G = to_networkx_graph(adj, sensor_ids, directed=False, locations=locations)
    assert isinstance(G, nx.Graph)
    assert not G.is_directed()
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2  # (s1, s2) and (s2, s3)
    assert G.nodes["s1"]["latitude"] == 34.0
    assert G.nodes["s1"]["longitude"] == -118.0
    assert G.edges["s1", "s2"]["weight"] == 0.5

    # Test directed graph
    directed_adj = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.8],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    DG = to_networkx_graph(directed_adj, sensor_ids, directed=True)
    assert isinstance(DG, nx.DiGraph)
    assert DG.is_directed()
    assert DG.number_of_edges() == 2
    assert DG.has_edge("s1", "s2")
    assert not DG.has_edge("s2", "s1")


