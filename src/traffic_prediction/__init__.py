"""Reusable components for traffic-speed forecasting."""

from typing import TYPE_CHECKING

from traffic_prediction.data import (
    HORIZON_STEPS,
    TimeSplit,
    load_speed_csv,
    observed_speeds,
    prepare_speed_inputs,
    time_split,
)
from traffic_prediction.features import (
    make_panel_features,
    make_sensor_features,
    time_features,
)
from traffic_prediction.graph import build_distance_adjacency
from traffic_prediction.metrics import peak_mask, regression_metrics

if TYPE_CHECKING:
    from traffic_prediction.models import GraphAttentionNetwork

__all__ = [
    "HORIZON_STEPS",
    "GraphAttentionNetwork",
    "TimeSplit",
    "build_distance_adjacency",
    "load_speed_csv",
    "make_panel_features",
    "make_sensor_features",
    "observed_speeds",
    "peak_mask",
    "prepare_speed_inputs",
    "regression_metrics",
    "time_features",
    "time_split",
]


def __getattr__(name: str) -> object:
    if name == "GraphAttentionNetwork":
        from traffic_prediction.models import GraphAttentionNetwork

        return GraphAttentionNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
