#!/usr/bin/env python3
"""CLI utility to evaluate and profile traffic forecast predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

# Ensure src/ is in sys.path for direct script execution
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from traffic_prediction.metrics import error_distribution, peak_mask, regression_metrics


def find_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    """Identify actual, predicted, and timestamp columns from dataframe."""
    actual_col = None
    pred_col = None
    time_col = None

    for col in df.columns:
        c_lower = str(col).lower()
        if "time" in c_lower or "date" in c_lower:
            time_col = col
        elif any(k in c_lower for k in ("actual", "ground_truth", "y_true")):
            actual_col = col
        elif any(k in c_lower for k in ("pred", "forecast", "y_pred")):
            pred_col = col

    if actual_col is None or pred_col is None:
        raise ValueError(
            f"Could not automatically detect actual and predicted columns in: {list(df.columns)}"
        )

    return actual_col, pred_col, time_col


def evaluate_prediction_file(
    filepath: Path,
    percentiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0),
) -> dict[str, Any]:
    """Evaluate a single prediction CSV file and compute stratified metrics."""
    df = pd.read_csv(filepath)
    actual_col, pred_col, time_col = find_columns(df)

    actual = df[actual_col].to_numpy(dtype=np.float64)
    predicted = df[pred_col].to_numpy(dtype=np.float64)

    reg_metrics = regression_metrics(actual, predicted)
    dist_metrics = error_distribution(actual, predicted, percentiles=percentiles)

    summary: dict[str, Any] = {
        "file": filepath.name,
        "total_rows": len(df),
        "valid_samples": reg_metrics["valid_test_values"],
        "MAE": reg_metrics["MAE"],
        "RMSE": reg_metrics["RMSE"],
        "MAPE_percent": reg_metrics["MAPE_percent"],
        "mean_bias": dist_metrics["mean_bias"],
        "max_error": dist_metrics["max_error"],
        "p50_MAE": dist_metrics.get("p50_MAE", np.nan),
        "p90_MAE": dist_metrics.get("p90_MAE", np.nan),
        "p95_MAE": dist_metrics.get("p95_MAE", np.nan),
        "p99_MAE": dist_metrics.get("p99_MAE", np.nan),
    }

    # Stratified peak vs off-peak evaluation
    if time_col is not None:
        try:
            ts_series = pd.to_datetime(df[time_col])
            is_peak = peak_mask(pd.DatetimeIndex(ts_series))

            peak_actual, peak_pred = actual[is_peak], predicted[is_peak]
            offpeak_actual, offpeak_pred = actual[~is_peak], predicted[~is_peak]

            peak_valid = np.isfinite(peak_actual) & (peak_actual > 0) & np.isfinite(peak_pred)
            if peak_valid.any():
                peak_m = regression_metrics(peak_actual, peak_pred)
                summary["peak_MAE"] = peak_m["MAE"]
                summary["peak_RMSE"] = peak_m["RMSE"]
                summary["peak_MAPE_percent"] = peak_m["MAPE_percent"]
                summary["peak_samples"] = peak_m["valid_test_values"]

            offpeak_valid = np.isfinite(offpeak_actual) & (offpeak_actual > 0) & np.isfinite(offpeak_pred)
            if offpeak_valid.any():
                offpeak_m = regression_metrics(offpeak_actual, offpeak_pred)
                summary["offpeak_MAE"] = offpeak_m["MAE"]
                summary["offpeak_RMSE"] = offpeak_m["RMSE"]
                summary["offpeak_MAPE_percent"] = offpeak_m["MAPE_percent"]
                summary["offpeak_samples"] = offpeak_m["valid_test_values"]
        except Exception:
            pass

    return summary


def print_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Print formatted evaluation report."""
    print("=" * 80)
    print("Traffic Speed Forecast Evaluation & Diagnostic Report")
    print("=" * 80)

    for item in results:
        print(f"\nModel File: {item['file']}")
        print(f"  Samples:    {item['valid_samples']} valid (total {item['total_rows']})")
        print(f"  Overall:    MAE: {item['MAE']:.3f} mph | RMSE: {item['RMSE']:.3f} mph | MAPE: {item['MAPE_percent']:.2f}%")
        print(f"  Bias:       {item['mean_bias']:+.3f} mph (Mean Signed Error)")
        print(f"  Tail Errors: Median (p50): {item['p50_MAE']:.2f} mph | p90: {item['p90_MAE']:.2f} mph | p95: {item['p95_MAE']:.2f} mph | Max: {item['max_error']:.2f} mph")

        if "peak_MAE" in item and "offpeak_MAE" in item:
            deg_ratio = item["peak_MAE"] / item["offpeak_MAE"] if item["offpeak_MAE"] > 0 else 1.0
            print(f"  Periods:    Peak MAE: {item['peak_MAE']:.3f} mph | Off-Peak MAE: {item['offpeak_MAE']:.3f} mph (Degradation: {deg_ratio:.2f}x)")

    print("\n" + "=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate traffic forecast predictions and output diagnostic metrics."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to a single prediction CSV file.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default="results/predictions",
        help="Directory of prediction CSV files (used if --input is not provided).",
    )
    parser.add_argument(
        "--output-summary",
        "-o",
        type=str,
        default=None,
        help="Optional path to save summary table as CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files_to_eval: list[Path] = []
    if args.input:
        p = Path(args.input)
        if not p.is_file():
            print(f"Error: Input file not found: {p}", file=sys.stderr)
            return 1
        files_to_eval.append(p)
    else:
        p_dir = Path(args.dir)
        if not p_dir.is_dir():
            print(f"Error: Directory not found: {p_dir}", file=sys.stderr)
            return 1
        # Find prediction CSVs excluding feature samples
        files_to_eval = sorted(
            [f for f in p_dir.glob("*.csv") if "sample_sensor" in f.name or "pred" in f.name]
        )
        if not files_to_eval:
            # Fall back to any csv in directory
            files_to_eval = sorted([f for f in p_dir.glob("*.csv") if "feature" not in f.name])

    if not files_to_eval:
        print("No prediction files found to evaluate.", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    for f in files_to_eval:
        try:
            summary = evaluate_prediction_file(f)
            results.append(summary)
        except Exception as exc:
            print(f"Warning: Failed to evaluate {f.name}: {exc}", file=sys.stderr)

    if not results:
        print("Could not evaluate any prediction files.", file=sys.stderr)
        return 1

    print_evaluation_report(results)

    if args.output_summary:
        out_path = Path(args.output_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(out_path, index=False)
        print(f"Saved evaluation summary to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
