#!/usr/bin/env python3
"""Generate executive benchmark comparison summaries from model evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def generate_benchmark_summary(
    comparison_csv: Path,
    output_md: Path | None = None,
) -> str:
    """Read model comparison metrics and construct an executive markdown report."""
    if not comparison_csv.is_file():
        raise FileNotFoundError(f"Model comparison CSV not found at: {comparison_csv}")

    df = pd.read_csv(comparison_csv)
    required_cols = {"model", "horizon_minutes", "MAE", "RMSE", "MAPE_percent", "period"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns. Expected {required_cols}, found {set(df.columns)}")

    lines: list[str] = []
    lines.append("# Intelligent Traffic Forecasting — Benchmark Executive Summary\n")
    lines.append(f"**Dataset**: {df['dataset'].iloc[0] if 'dataset' in df.columns else 'METR-LA'}")
    lines.append("**Evaluation Metrics**: Masked MAE, RMSE, and MAPE (%) over held-out chronological test split.\n")
    lines.append("---\n")

    # 1. Overall Comparison Across Horizons (Period == 'All')
    lines.append("## 1. Overall Performance Across Forecast Horizons\n")
    lines.append("| Horizon | Model | MAE (mph) | RMSE (mph) | MAPE (%) | vs. Persistence (MAE Δ) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    df_all = df[df["period"].str.lower() == "all"].copy()
    horizons = sorted(df_all["horizon_minutes"].unique())

    for h in horizons:
        h_df = df_all[df_all["horizon_minutes"] == h]
        persist_row = h_df[h_df["model"].str.lower() == "persistence"]
        persist_mae = persist_row["MAE"].iloc[0] if not persist_row.empty else None

        for _, row in h_df.iterrows():
            model = row["model"]
            mae = row["MAE"]
            rmse = row["RMSE"]
            mape = row["MAPE_percent"]

            if persist_mae is not None and model.lower() != "persistence":
                delta = ((persist_mae - mae) / persist_mae) * 100
                delta_str = f"**-{delta:.1f}%**" if delta > 0 else f"+{abs(delta):.1f}%"
            elif model.lower() == "persistence":
                delta_str = "Baseline"
            else:
                delta_str = "—"

            lines.append(f"| {h} min | **{model}** | {mae:.3f} | {rmse:.3f} | {mape:.2f}% | {delta_str} |")

    # 2. Peak vs. Off-Peak Congestion Degradation Analysis
    lines.append("\n---\n")
    lines.append("## 2. Rush Hour Congestion Robustness (Peak vs. Off-Peak)\n")
    lines.append("> [!NOTE]")
    lines.append("> **Peak Hours**: Weekdays 07:00–09:59 & 16:00–18:59 PST. **Off-Peak**: All remaining hours.\n")
    lines.append("| Horizon | Model | Peak MAE | Off-Peak MAE | Peak Degradation Ratio |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")

    df_peak = df[df["period"].str.lower() == "peak"]
    df_offpeak = df[df["period"].str.lower() == "off-peak"]

    if not df_peak.empty and not df_offpeak.empty:
        for h in horizons:
            p_h = df_peak[df_peak["horizon_minutes"] == h]
            op_h = df_offpeak[df_offpeak["horizon_minutes"] == h]
            common_models = sorted(set(p_h["model"]).intersection(set(op_h["model"])))

            for m in common_models:
                p_mae = p_h[p_h["model"] == m]["MAE"].iloc[0]
                op_mae = op_h[op_h["model"] == m]["MAE"].iloc[0]
                ratio = p_mae / op_mae if op_mae > 0 else np.nan
                lines.append(f"| {h} min | **{m}** | {p_mae:.3f} | {op_mae:.3f} | {ratio:.2f}x |")

    report_text = "\n".join(lines) + "\n"

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(report_text, encoding="utf-8")

    return report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison executive report from metrics CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="results/metrics/metr_la_model_comparison.csv",
        help="Path to the model comparison metrics CSV.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/metrics/benchmark_executive_summary.md",
        help="Path to output markdown report file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress printing report to console.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.input)
    out_path = Path(args.output) if args.output else None

    try:
        report = generate_benchmark_summary(csv_path, output_md=out_path)
        if not args.quiet:
            print(report)
        if out_path:
            print(f"Benchmark summary written to: {out_path}")
        return 0
    except Exception as exc:
        print(f"Error generating benchmark summary: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
