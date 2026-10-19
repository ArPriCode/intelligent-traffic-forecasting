"""Distance-based METR-LA adjacency construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_distance_adjacency(
    distances: pd.DataFrame,
    sensor_ids: list[str],
    *,
    threshold: float = 0.1,
) -> np.ndarray:
    """Build a symmetric Gaussian distance graph in the requested sensor order."""
    required = {"from", "to", "cost"}
    if not required.issubset(distances.columns):
        raise ValueError(f"Distance table must include columns: {sorted(required)}.")
    if not sensor_ids or len(sensor_ids) != len(set(sensor_ids)):
        raise ValueError("sensor_ids must be non-empty and unique.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one.")

    sensor_index = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    matrix = np.full((len(sensor_ids), len(sensor_ids)), np.inf, dtype=np.float64)
    edges = distances.loc[
        distances["from"].astype(str).isin(sensor_index)
        & distances["to"].astype(str).isin(sensor_index),
        ["from", "to", "cost"],
    ]
    for source, target, cost in edges.itertuples(index=False, name=None):
        distance = float(cost)
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Road distances must be finite non-negative values.")
        matrix[sensor_index[str(source)], sensor_index[str(target)]] = distance

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("No road-distance rows match the provided sensor IDs.")
    distance_std = float(finite.std())
    if distance_std <= 0:
        raise ValueError("Matching road distances must have a non-zero standard deviation.")

    adjacency = np.zeros_like(matrix, dtype=np.float32)
    valid = np.isfinite(matrix)
    adjacency[valid] = np.exp(-np.square(matrix[valid] / distance_std))
    adjacency[adjacency < threshold] = 0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency
