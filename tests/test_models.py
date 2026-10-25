"""Unit tests for spatial GraphAttentionLayer and GraphAttentionNetwork models."""

from __future__ import annotations

import pytest
import torch

from traffic_prediction import GraphAttentionLayer, GraphAttentionNetwork


def test_graph_attention_layer_forward():
    adj = torch.eye(4)
    adj[0, 1] = adj[1, 0] = 0.5
    layer = GraphAttentionLayer(input_size=8, hidden_size=16, adjacency=adj)
    x = torch.randn(2, 4, 8)
    out = layer(x)
    assert out.shape == (2, 4, 16)
    assert torch.isfinite(out).all()


def test_graph_attention_layer_validation():
    adj = torch.eye(4)

    # Invalid input/hidden sizes
    with pytest.raises(ValueError, match="positive integers"):
        GraphAttentionLayer(input_size=0, hidden_size=16, adjacency=adj)
    with pytest.raises(ValueError, match="positive integers"):
        GraphAttentionLayer(input_size=8, hidden_size=-1, adjacency=adj)

    # Non-square adjacency
    with pytest.raises(ValueError, match="square"):
        GraphAttentionLayer(input_size=8, hidden_size=16, adjacency=torch.eye(4)[:3])

    # Negative adjacency
    bad_adj = torch.eye(4)
    bad_adj[0, 1] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        GraphAttentionLayer(input_size=8, hidden_size=16, adjacency=bad_adj)

    # Disconnected sensor (zero row sum)
    zero_row_adj = torch.eye(4)
    zero_row_adj[0, 0] = 0.0
    with pytest.raises(ValueError, match="at least one graph connection"):
        GraphAttentionLayer(input_size=8, hidden_size=16, adjacency=zero_row_adj)


def test_graph_attention_network_validation():
    adj = torch.eye(4)
    with pytest.raises(ValueError, match="Model dimensions must be positive"):
        GraphAttentionNetwork(adjacency=adj, input_size=0)
    with pytest.raises(ValueError, match="dropout must be in"):
        GraphAttentionNetwork(adjacency=adj, dropout=1.5)


def test_graph_attention_network_forward():
    adj = torch.eye(3)
    adj[0, 1] = 0.7
    adj[1, 0] = 0.7
    model = GraphAttentionNetwork(adjacency=adj, input_size=6, hidden_size=12, output_size=4)
    x = torch.randn(3, 3, 6)
    out = model(x)
    assert out.shape == (3, 3, 4)
