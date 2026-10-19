"""Graph-attention model used for spatial traffic-speed forecasting."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    """Single spatial attention layer constrained by a fixed adjacency matrix."""

    def __init__(self, input_size: int, hidden_size: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square (sensors, sensors) matrix.")
        if not torch.isfinite(adjacency).all() or torch.any(adjacency < 0):
            raise ValueError("adjacency must contain finite non-negative weights.")
        if torch.any(adjacency.sum(dim=1) <= 0):
            raise ValueError("Every sensor must have at least one graph connection.")

        self.projection = nn.Linear(input_size, hidden_size, bias=False)
        self.source_attention = nn.Parameter(torch.empty(hidden_size))
        self.target_attention = nn.Parameter(torch.empty(hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.normal_(self.source_attention, std=0.1)
        nn.init.normal_(self.target_attention, std=0.1)
        edge_bias = torch.where(
            adjacency > 0,
            adjacency.clamp_min(1e-8).log(),
            torch.full_like(adjacency, -1e9),
        )
        self.register_buffer("edge_bias", edge_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.edge_bias.shape[0]:
            raise ValueError("features must have shape (batch, sensors, features).")
        hidden = F.elu(self.projection(features))
        source = (hidden * self.source_attention).sum(dim=-1, keepdim=True)
        target = (hidden * self.target_attention).sum(dim=-1).unsqueeze(1)
        logits = F.leaky_relu(source + target, negative_slope=0.2) + self.edge_bias
        attention = torch.softmax(logits, dim=-1)
        messages = torch.bmm(attention, hidden)
        return F.elu(self.norm(hidden + messages))


class GraphAttentionNetwork(nn.Module):
    """Predict multiple future speeds for every sensor in a fixed graph."""

    def __init__(
        self,
        adjacency: torch.Tensor,
        input_size: int = 10,
        hidden_size: int = 32,
        output_size: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("Model dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.gat = GraphAttentionLayer(input_size, hidden_size, adjacency)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.gat(features)))
