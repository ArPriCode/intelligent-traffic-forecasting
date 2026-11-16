"""Generate network spatial congestion maps and evaluate GAT forecasts across METR-LA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from traffic_prediction.features import time_features
from traffic_prediction.graph import to_networkx_graph
from traffic_prediction.models import GraphAttentionNetwork

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "METR-LA.csv"
DEFAULT_LOCATIONS_PATH = PROJECT_ROOT / "data" / "sensor_graph" / "graph_sensor_locations.csv"
DEFAULT_ADJACENCY_PATH = PROJECT_ROOT / "results" / "metrics" / "metr_la_distance_adjacency.csv"
DEFAULT_STATE_DICT_PATH = PROJECT_ROOT / "results" / "metrics" / "metr_la_gat_state_dict.pt"
DEFAULT_OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "31_metr_la_network_congestion_maps.png"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "metrics" / "metr_la_spatial_congestion_snapshot.csv"

# Target representative peak rush-hour timestamp (evening peak traffic collapse)
DEFAULT_PEAK_ORIGIN = "2012-06-14 17:30:00"


def generate_congestion_maps(
    data_path: Path = DEFAULT_DATA_PATH,
    locations_path: Path = DEFAULT_LOCATIONS_PATH,
    adjacency_path: Path = DEFAULT_ADJACENCY_PATH,
    state_dict_path: Path = DEFAULT_STATE_DICT_PATH,
    origin_timestamp: str = DEFAULT_PEAK_ORIGIN,
    horizon_minutes: int = 15,
    output_fig: Path = DEFAULT_OUTPUT_FIG,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """Generate spatial congestion maps and snapshot metrics for METR-LA highway network.

    Parameters
    ----------
    data_path : Path
        Path to METR-LA speed CSV.
    locations_path : Path
        Path to graph_sensor_locations.csv with sensor_id, latitude, longitude.
    adjacency_path : Path
        Path to precomputed distance adjacency CSV.
    state_dict_path : Path
        Path to PyTorch GAT model weights.
    origin_timestamp : str
        Origin time for forecast generation.
    horizon_minutes : int
        Forecast horizon (5, 15, 30, or 60 minutes).
    output_fig : Path
        Destination for output PNG figure.
    output_csv : Path
        Destination for output snapshot CSV.

    Returns
    -------
    pd.DataFrame
        Per-sensor spatial predictions, ground truth, and absolute errors.
    """
    horizon_map = {5: (1, 0), 15: (3, 1), 30: (6, 2), 60: (12, 3)}
    if horizon_minutes not in horizon_map:
        raise ValueError(f"horizon_minutes must be one of {list(horizon_map.keys())}")
    step_offset, head_index = horizon_map[horizon_minutes]

    # 1. Load sensor metadata and locations
    if not locations_path.is_file():
        raise FileNotFoundError(f"Locations file not found: {locations_path}")
    raw_locs = pd.read_csv(locations_path)
    raw_locs["sensor_id"] = raw_locs["sensor_id"].astype(str)

    # 2. Load traffic speed dataset
    if not data_path.is_file():
        raise FileNotFoundError(f"Speed dataset not found: {data_path}")
    traffic = pd.read_csv(data_path, index_col=0)
    traffic.index = pd.to_datetime(traffic.index)
    sensor_ids = [str(c) for c in traffic.columns]

    locs = raw_locs.set_index("sensor_id").loc[sensor_ids].reset_index()

    # 3. Fit scaling parameters strictly on training split (first 70% timestamps)
    n_train = int(len(traffic) * 0.7)
    train_traffic = traffic.iloc[:n_train].replace(0.0, np.nan)
    train_mean = train_traffic.mean().to_numpy(dtype=np.float32)
    train_std = train_traffic.std().to_numpy(dtype=np.float32)
    train_mean[~np.isfinite(train_mean)] = 0.0
    train_std[~np.isfinite(train_std) | (train_std < 1e-4)] = 1.0

    observed = traffic.to_numpy(dtype=np.float32)
    observed[observed == 0.0] = np.nan
    speed_scaled = np.where(
        np.isfinite(observed), (observed - train_mean) / train_std, 0.0
    ).astype(np.float32)

    # 4. Load adjacency and build NetworkX graph for edge connectivity
    if not adjacency_path.is_file():
        raise FileNotFoundError(f"Adjacency file not found: {adjacency_path}")
    adj = np.loadtxt(adjacency_path, delimiter=",", dtype=np.float32)
    graph = to_networkx_graph(adj, sensor_ids, directed=False, locations=locs)

    # 5. Load trained GAT model
    if not state_dict_path.is_file():
        raise FileNotFoundError(f"GAT model weights not found: {state_dict_path}")
    model = GraphAttentionNetwork(torch.tensor(adj), input_size=10, hidden_size=32, output_size=4)
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    model.eval()

    # 6. Locate origin timestamp and prepare causal input features
    origin_dt = pd.Timestamp(origin_timestamp)
    if origin_dt not in traffic.index:
        raise ValueError(f"Origin timestamp {origin_timestamp} not in dataset timeline.")
    t = traffic.index.get_loc(origin_dt)
    target_dt = traffic.index[t + step_offset]

    # Input: 6 lags [t-5, t-4, t-3, t-2, t-1, t] + 4 cyclic calendar features
    history = np.stack([speed_scaled[t - lag] for lag in range(5, -1, -1)], axis=-1)
    clock = time_features(pd.DatetimeIndex([origin_dt]))[0]
    clock_exp = np.broadcast_to(clock[None, :], (len(sensor_ids), 4))
    x_input = np.concatenate([history, clock_exp], axis=-1)[None, ...]

    # 7. Run model inference
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_input, dtype=torch.float32)).numpy()[0]

    pred_unscaled = pred_scaled * train_std[:, None] + train_mean[:, None]
    pred_horizon = pred_unscaled[:, head_index]
    target_horizon = observed[t + step_offset]
    pers_horizon = observed[t]

    abs_error = np.abs(target_horizon - pred_horizon)
    pers_error = np.abs(target_horizon - pers_horizon)

    # Construct snapshot dataframe
    snapshot_df = pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "latitude": locs["latitude"].to_numpy(),
            "longitude": locs["longitude"].to_numpy(),
            "origin_timestamp": str(origin_dt),
            "target_timestamp": str(target_dt),
            "horizon_minutes": horizon_minutes,
            "ground_truth_speed_mph": target_horizon,
            "gat_predicted_speed_mph": pred_horizon,
            "gat_abs_error_mph": abs_error,
            "persistence_speed_mph": pers_horizon,
            "persistence_abs_error_mph": pers_error,
        }
    )

    # Classify congestion state
    conditions = [
        snapshot_df["ground_truth_speed_mph"] < 35.0,
        snapshot_df["ground_truth_speed_mph"].between(35.0, 55.0),
        snapshot_df["ground_truth_speed_mph"] >= 55.0,
    ]
    labels = ["Severe Congestion (<35 mph)", "Moderate (35-55 mph)", "Free Flow (>55 mph)"]
    snapshot_df["congestion_state"] = np.select(conditions, labels, default="Sensor Outage")

    # 8. Render publication-quality 4-panel spatial figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)

    # Color scaling: 15 to 70 mph for traffic speed (Red -> Yellow -> Green)
    speed_cmap = plt.cm.RdYlGn
    speed_norm = matplotlib.colors.Normalize(vmin=15.0, vmax=68.0)

    # Error scaling: 0 to 16 mph (Yellow -> Orange -> Red)
    error_cmap = plt.cm.YlOrRd
    error_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=15.0)

    # Subplot A: Ground Truth Speed Map
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#0f172a")  # Dark slate background for clean contrast
    # Draw road network edges
    for u, v in graph.edges():
        u_node, v_node = graph.nodes[u], graph.nodes[v]
        ax_a.plot(
            [u_node["longitude"], v_node["longitude"]],
            [u_node["latitude"], v_node["latitude"]],
            color="#475569",
            lw=0.7,
            alpha=0.5,
            zorder=1,
        )
    sc_a = ax_a.scatter(
        snapshot_df["longitude"],
        snapshot_df["latitude"],
        c=snapshot_df["ground_truth_speed_mph"],
        cmap=speed_cmap,
        norm=speed_norm,
        s=36,
        edgecolor="#020617",
        linewidth=0.5,
        zorder=2,
    )
    cb_a = fig.colorbar(sc_a, ax=ax_a, fraction=0.046, pad=0.03)
    cb_a.set_label("Speed (mph)", color="#1e293b", fontsize=10, weight="bold")
    ax_a.set_title(
        f"(A) Ground Truth Congestion Map\nTarget: {target_dt} (Peak Rush Hour)",
        fontsize=12,
        weight="bold",
        pad=10,
    )
    ax_a.set_xlabel("Longitude")
    ax_a.set_ylabel("Latitude")
    ax_a.set_aspect("equal")
    ax_a.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")

    # Subplot B: GAT Forecast Speed Map
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#0f172a")
    for u, v in graph.edges():
        u_node, v_node = graph.nodes[u], graph.nodes[v]
        ax_b.plot(
            [u_node["longitude"], v_node["longitude"]],
            [u_node["latitude"], v_node["latitude"]],
            color="#475569",
            lw=0.7,
            alpha=0.5,
            zorder=1,
        )
    sc_b = ax_b.scatter(
        snapshot_df["longitude"],
        snapshot_df["latitude"],
        c=snapshot_df["gat_predicted_speed_mph"],
        cmap=speed_cmap,
        norm=speed_norm,
        s=36,
        edgecolor="#020617",
        linewidth=0.5,
        zorder=2,
    )
    cb_b = fig.colorbar(sc_b, ax=ax_b, fraction=0.046, pad=0.03)
    cb_b.set_label("Predicted Speed (mph)", color="#1e293b", fontsize=10, weight="bold")
    ax_b.set_title(
        f"(B) Graph Attention Network (GAT) Forecast\nHorizon: {horizon_minutes} min (Origin: {origin_dt})",
        fontsize=12,
        weight="bold",
        pad=10,
    )
    ax_b.set_xlabel("Longitude")
    ax_b.set_ylabel("Latitude")
    ax_b.set_aspect("equal")
    ax_b.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")

    # Subplot C: Spatial Forecast Absolute Error Map
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#0f172a")
    for u, v in graph.edges():
        u_node, v_node = graph.nodes[u], graph.nodes[v]
        ax_c.plot(
            [u_node["longitude"], v_node["longitude"]],
            [u_node["latitude"], v_node["latitude"]],
            color="#475569",
            lw=0.7,
            alpha=0.5,
            zorder=1,
        )
    sc_c = ax_c.scatter(
        snapshot_df["longitude"],
        snapshot_df["latitude"],
        c=snapshot_df["gat_abs_error_mph"],
        cmap=error_cmap,
        norm=error_norm,
        s=36,
        edgecolor="#020617",
        linewidth=0.5,
        zorder=2,
    )
    cb_c = fig.colorbar(sc_c, ax=ax_c, fraction=0.046, pad=0.03)
    cb_c.set_label("Absolute Error |y - ŷ| (mph)", color="#1e293b", fontsize=10, weight="bold")
    valid_mask = np.isfinite(target_horizon)
    mean_err = abs_error[valid_mask].mean()
    med_err = np.median(abs_error[valid_mask])
    ax_c.set_title(
        f"(C) Spatial Forecast Error Distribution\nNetwork Mean AE: {mean_err:.2f} mph | Median AE: {med_err:.2f} mph",
        fontsize=12,
        weight="bold",
        pad=10,
    )
    ax_c.set_xlabel("Longitude")
    ax_c.set_ylabel("Latitude")
    ax_c.set_aspect("equal")
    ax_c.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")

    # Subplot D: Calibration Scatter & Error Histogram
    ax_d = axes[1, 1]
    valid_gt = target_horizon[valid_mask]
    valid_pred = pred_horizon[valid_mask]
    r2 = 1.0 - (np.sum((valid_gt - valid_pred) ** 2) / np.sum((valid_gt - valid_gt.mean()) ** 2))

    ax_d.scatter(
        valid_gt,
        valid_pred,
        color="#2563eb",
        alpha=0.6,
        s=30,
        edgecolor="none",
        label=f"Sensors (N={len(valid_gt)})",
    )
    # Ideal 1-to-1 line
    diag_min = min(valid_gt.min(), valid_pred.min()) - 2
    diag_max = max(valid_gt.max(), valid_pred.max()) + 2
    ax_d.plot([diag_min, diag_max], [diag_min, diag_max], "r--", lw=1.5, label="Perfect Agreement (y = x)")

    ax_d.set_title(
        f"(D) Congestion Calibration & Correlation\nNetwork R²: {r2:.3f} | Pearson r: {np.corrcoef(valid_gt, valid_pred)[0, 1]:.3f}",
        fontsize=12,
        weight="bold",
        pad=10,
    )
    ax_d.set_xlabel("Ground Truth Observed Speed (mph)", weight="bold")
    ax_d.set_ylabel("GAT Predicted Speed (mph)", weight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(frameon=True, facecolor="#f8fafc", loc="upper left")

    fig.suptitle(
        f"METR-LA Spatio-Temporal Congestion Mapping & Network-Aware Forecast Validation ({horizon_minutes}m Ahead)",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save artifacts
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    snapshot_df.to_csv(output_csv, index=False)

    print(f"Spatial congestion maps saved to: {output_fig}")
    print(f"Spatial snapshot table saved to: {output_csv}")
    print(f"Network {horizon_minutes}m MAE: {mean_err:.3f} mph across {len(valid_gt)} valid sensors.")

    return snapshot_df


def main() -> None:
    """CLI entrypoint for spatial congestion mapping."""
    parser = argparse.ArgumentParser(
        description="Generate METR-LA spatial network congestion maps and forecast evaluation."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--locations-path", type=Path, default=DEFAULT_LOCATIONS_PATH)
    parser.add_argument("--adjacency-path", type=Path, default=DEFAULT_ADJACENCY_PATH)
    parser.add_argument("--state-dict-path", type=Path, default=DEFAULT_STATE_DICT_PATH)
    parser.add_argument("--origin-timestamp", type=str, default=DEFAULT_PEAK_ORIGIN)
    parser.add_argument("--horizon", type=int, default=15, choices=[5, 15, 30, 60])
    parser.add_argument("--output-fig", type=Path, default=DEFAULT_OUTPUT_FIG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)

    args = parser.parse_args()

    try:
        generate_congestion_maps(
            data_path=args.data_path,
            locations_path=args.locations_path,
            adjacency_path=args.adjacency_path,
            state_dict_path=args.state_dict_path,
            origin_timestamp=args.origin_timestamp,
            horizon_minutes=args.horizon,
            output_fig=args.output_fig,
            output_csv=args.output_csv,
        )
    except Exception as e:
        print(f"Error generating congestion maps: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
