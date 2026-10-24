"""Executable script for arbitrary sensor speed analysis, profiling, and baseline evaluation.

Usage:
    python scripts/run_sensor_analysis.py --dataset metr-la
    python scripts/run_sensor_analysis.py --dataset pems-bay
    python scripts/run_sensor_analysis.py --sensors 773869 767541 767542 717447 --dataset metr-la
    python scripts/run_sensor_analysis.py --sensors 400001 400017 400030 400040 --dataset pems-bay
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_FIG = ROOT / "results" / "figures"
RESULTS_METRICS = ROOT / "results" / "metrics"

RESULTS_FIG.mkdir(parents=True, exist_ok=True)
RESULTS_METRICS.mkdir(parents=True, exist_ok=True)


def analyze_sensors(
    sensor_ids: list[str] | None = None,
    dataset: str = "metr-la",
    plot_hours: int = 48,
) -> pd.DataFrame:
    """Analyze speed distributions, persistence forecasting, and temporal patterns."""
    if dataset.lower() in ("pems-bay", "bay"):
        csv_file = DATA_DIR / "PEMS-BAY.csv"
        meta_file = DATA_DIR / "PEMS-BAY-META.csv"
        default_sensors = ["400001", "400017", "400030", "400040"]
        ds_name = "PEMS-BAY"
    else:
        csv_file = DATA_DIR / "METR-LA.csv"
        meta_file = DATA_DIR / "sensor_graph" / "graph_sensor_locations.csv"
        default_sensors = ["773869", "767541", "767542", "717447"]
        ds_name = "METR-LA"

    if not csv_file.exists():
        raise FileNotFoundError(f"Traffic dataset not found at {csv_file}")

    print(f"Loading {ds_name} dataset from {csv_file}...")
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True).astype("float32")
    df.columns = df.columns.astype(str)
    # Mask non-positive speeds as NaN
    df = df.where(df > 0)

    if sensor_ids is None or len(sensor_ids) == 0:
        sensor_ids = default_sensors

    # Filter to sensors present in dataset
    valid_sensors = [str(s) for s in sensor_ids if str(s) in df.columns]
    if not valid_sensors:
        # Fallback to the first available sensors
        valid_sensors = list(df.columns[:4])
        print(f"Provided sensors not found in {ds_name}. Falling back to: {valid_sensors}")

    print(f"Analyzing {len(valid_sensors)} sensors in {ds_name}: {valid_sensors}...")

    # Load metadata if available
    meta_df = None
    if meta_file and meta_file.exists():
        try:
            m = pd.read_csv(meta_file)
            if "sensor_id" in m.columns:
                meta_df = m.set_index(m["sensor_id"].astype(str))
            elif "id" in m.columns:
                meta_df = m.set_index(m["id"].astype(str))
        except Exception as e:
            print(f"Notice: Could not parse metadata {meta_file}: {e}")

    records = []
    # Horizons: 5m (1 step), 15m (3 steps), 30m (6 steps), 60m (12 steps)
    horizons = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}

    for sid in valid_sensors:
        series = df[sid].dropna()
        n_obs = len(series)
        mean_spd = float(series.mean())
        std_spd = float(series.std())
        min_spd = float(series.min())
        p05_spd = float(series.quantile(0.05))
        p50_spd = float(series.median())
        p95_spd = float(series.quantile(0.95))
        max_spd = float(series.max())
        missing_rate = float(df[sid].isna().mean() * 100.0)

        # Baseline persistence MAE for each horizon
        pers_maes = {}
        for h_name, h_step in horizons.items():
            actual = series.iloc[h_step:].values
            pred = series.iloc[:-h_step].values
            mae = float(np.mean(np.abs(actual - pred)))
            pers_maes[f"Persistence_MAE_{h_name}"] = round(mae, 3)

        rec = {
            "dataset": ds_name,
            "sensor_id": sid,
            "total_observations": n_obs,
            "missing_pct": round(missing_rate, 2),
            "mean_speed_mph": round(mean_spd, 2),
            "std_speed_mph": round(std_spd, 2),
            "min_speed_mph": round(min_spd, 2),
            "p05_speed_mph": round(p05_spd, 2),
            "median_speed_mph": round(p50_spd, 2),
            "p95_speed_mph": round(p95_spd, 2),
            "max_speed_mph": round(max_spd, 2),
            **pers_maes,
        }

        if meta_df is not None and sid in meta_df.index:
            row = meta_df.loc[sid]
            for col in ["Fwy", "Dir", "Lanes", "Latitude", "latitude", "Longitude", "longitude"]:
                if col in row:
                    rec[col.lower()] = row[col]

        records.append(rec)

    metrics_df = pd.DataFrame(records)
    metrics_path = RESULTS_METRICS / f"{dataset.lower()}_sample_sensor_analysis.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved sensor metrics table to {metrics_path}")

    # Plot 4-panel analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)

    # Panel 1: Time series over window
    ax1 = axes[0, 0]
    steps = min(plot_hours * 12, len(df))
    sub_df = df[valid_sensors].iloc[:steps]
    for sid in valid_sensors:
        ax1.plot(sub_df.index, sub_df[sid], label=f"Sensor {sid}", lw=1.5, alpha=0.85)
    ax1.set_title(f"Observed Speed Profile ({ds_name} - First {plot_hours} Hours)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Speed (mph)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8)

    # Panel 2: Hourly Diurnal Profile
    ax2 = axes[0, 1]
    hours = df.index.hour
    for sid in valid_sensors:
        hourly_means = df[sid].groupby(hours).mean()
        ax2.plot(hourly_means.index, hourly_means.values, marker="o", label=f"Sensor {sid}", lw=1.8)
    ax2.set_title(f"Diurnal 24-Hour Mean Speed ({ds_name})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Hour of Day (0 - 23)")
    ax2.set_ylabel("Mean Speed (mph)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8)

    # Panel 3: Speed Distribution (Histogram/KDE)
    ax3 = axes[1, 0]
    for sid in valid_sensors:
        s_data = df[sid].dropna()
        ax3.hist(s_data, bins=35, density=True, alpha=0.4, label=f"Sensor {sid}")
    ax3.set_title("Speed Density Distribution", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Speed (mph)")
    ax3.set_ylabel("Probability Density")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Persistence Error Growth Across Horizons
    ax4 = axes[1, 1]
    h_labels = ["5m", "15m", "30m", "60m"]
    for sid in valid_sensors:
        sid_row = metrics_df[metrics_df["sensor_id"] == sid].iloc[0]
        y_vals = [sid_row[f"Persistence_MAE_{h}"] for h in h_labels]
        ax4.plot(h_labels, y_vals, marker="s", lw=2.0, label=f"Sensor {sid}")
    ax4.set_title("Persistence Forecast Degradation (MAE vs Horizon)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Forecast Horizon")
    ax4.set_ylabel("MAE (mph)")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig_name = f"29_{dataset.lower()}_sample_sensor_analysis.png"
    fig_path = RESULTS_FIG / fig_name
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved sensor analysis plot to {fig_path}")

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect arbitrary sensors and generate analysis.")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        help="List of sensor IDs to inspect",
    )
    parser.add_argument(
        "--dataset",
        choices=["metr-la", "pems-bay"],
        default="metr-la",
        help="Target dataset (default: metr-la)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="Number of hours to display in profile",
    )
    args = parser.parse_args()

    df = analyze_sensors(sensor_ids=args.sensors, dataset=args.dataset, plot_hours=args.hours)
    print("\n--- Summary Results ---")
    cols_to_print = [
        "sensor_id",
        "mean_speed_mph",
        "std_speed_mph",
        "p05_speed_mph",
        "p95_speed_mph",
        "Persistence_MAE_5m",
        "Persistence_MAE_60m",
    ]
    print(df[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
