"""Interactive Streamlit dashboard for intelligent traffic forecasting and signal control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    """Read and cache a project result table."""
    return pd.read_csv(path)


def require_result(filename: str, directory: Path) -> Path | None:
    """Display a visible app error when a required result file is unavailable."""
    path = directory / filename
    if not path.is_file():
        st.error(f"Required project result is missing: `{path.relative_to(PROJECT_ROOT)}`")
        return None
    return path


def run_dashboard() -> None:
    """Render the full Streamlit traffic dashboard."""
    st.set_page_config(
        page_title="Intelligent Traffic Forecasting & Control",
        page_icon="🚦",
        layout="wide",
    )

    st.title("🚦 Intelligent Traffic Speed Forecasting & Signal Control")
    st.caption("Advanced Machine Learning Benchmark & Adaptive Signal Queuing Simulation")

    quality_path = require_result("traffic_data_quality.csv", METRICS_PATH)
    comparison_path = require_result("metr_la_model_comparison.csv", METRICS_PATH)
    if quality_path is None or comparison_path is None:
        st.stop()

    quality = read_csv(str(quality_path))
    comparison = read_csv(str(comparison_path))
    la_quality = quality.loc[quality["dataset"] == "METR-LA"]
    if la_quality.empty:
        st.error("METR-LA data-quality results are not present in the result table.")
        st.stop()
    la_quality = la_quality.iloc[0]

    metric_columns = {"MAE", "MAPE_percent", "RMSE", "valid_test_values"}
    required_comparison_columns = {
        "dataset",
        "model",
        "period",
        "horizon_minutes",
        *metric_columns,
    }
    missing_columns = required_comparison_columns.difference(comparison.columns)
    if missing_columns:
        st.error(f"Model comparison table is missing columns: {sorted(missing_columns)}")
        st.stop()

    # Overview Metrics Row
    overview = st.columns(4)
    overview[0].metric("Sensors (METR-LA)", f"{int(la_quality['sensors']):,}")
    overview[1].metric("Five-minute Records", f"{int(la_quality['time_steps']):,}")
    overview[2].metric(
        "Observed Timeline",
        f"{la_quality['start']} – {str(la_quality['end'])[:10]}",
    )
    overview[3].metric("Sensor Missing / Zero", f"{la_quality['missing_or_zero_percent']:.2f}%")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Model Comparison",
            "📈 Forecast Trajectories",
            "🛣️ Sensor Diagnostics",
            "🚦 Adaptive Signal Simulation",
        ]
    )

    with tab1:
        st.subheader("Held-Out Benchmark Performance (METR-LA)")
        st.markdown(
            "Compare the four evaluated paradigms across forecast horizons and traffic periods."
        )
        controls = st.columns(3)
        available_periods = ["All", "Peak", "Off-peak"]
        selected_period = controls[0].selectbox("Traffic Period", available_periods)
        available_horizons = sorted(
            comparison["horizon_minutes"].dropna().astype(int).unique()
        )
        selected_horizon = controls[1].selectbox(
            "Forecast Horizon (minutes ahead)",
            available_horizons,
            index=available_horizons.index(15) if 15 in available_horizons else 0,
        )
        metric_name = controls[2].selectbox("Evaluation Metric", ["MAE", "RMSE", "MAPE_percent"])

        filtered = comparison.loc[
            (comparison["dataset"] == "METR-LA")
            & (comparison["period"] == selected_period)
            & (comparison["horizon_minutes"] == selected_horizon)
        ].copy()

        if filtered.empty:
            st.warning("No saved metrics match the selected period and forecast horizon.")
        else:
            filtered = filtered.sort_values(metric_name)
            chart_col, table_col = st.columns([1, 1])
            with chart_col:
                st.markdown(f"**{metric_name} Comparison ({selected_horizon}m, {selected_period})**")
                st.bar_chart(filtered.set_index("model")[[metric_name]])
            with table_col:
                st.markdown("**Metric Summary Table**")
                display_metrics = filtered[
                    ["model", "MAE", "MAPE_percent", "RMSE", "valid_test_values"]
                ].rename(
                    columns={
                        "MAPE_percent": "MAPE (%)",
                        "valid_test_values": "Valid targets",
                    }
                )
                st.dataframe(display_metrics.round(3), hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Held-Out 15-Minute Sample Forecasts")
        st.caption(
            "Observed ground truth vs. multi-model predictions for sample sensor 773869."
        )

        prediction_files = {
            "Random Forest": ("metr_la_random_forest_sample_sensor.csv", "predicted_speed"),
            "XGBoost": ("metr_la_xgboost_sample_sensor.csv", "predicted_speed_15min"),
            "GAT": ("metr_la_gat_sample_sensor.csv", "predicted_speed_15min"),
        }
        actual_series: pd.Series | None = None
        model_series: dict[str, pd.Series] = {}
        missing_prediction_files: list[str] = []

        for model_name, (filename, prediction_column) in prediction_files.items():
            path = PREDICTIONS_PATH / filename
            if not path.is_file():
                missing_prediction_files.append(filename)
                continue
            frame = read_csv(str(path))
            if not {"timestamp", "actual_speed", prediction_column}.issubset(frame.columns):
                st.error(f"Saved forecast file `{filename}` has an unexpected column format.")
                st.stop()
            ts = pd.to_datetime(frame["timestamp"], errors="coerce")
            valid = pd.notna(ts)
            ts_valid = pd.DatetimeIndex(ts[valid])

            if actual_series is None:
                actual_series = pd.Series(
                    frame.loc[valid, "actual_speed"].to_numpy(dtype=np.float32),
                    index=ts_valid,
                    name="actual_speed",
                )
            model_series[model_name] = pd.Series(
                frame.loc[valid, prediction_column].to_numpy(dtype=np.float32),
                index=ts_valid,
                name=model_name,
            )

        if missing_prediction_files:
            st.error(f"Missing saved forecast files: {', '.join(missing_prediction_files)}")
        elif actual_series is not None and model_series:
            combined = (
                pd.DataFrame({"actual_speed": actual_series, **model_series})
                .dropna()
                .sort_index()
            )
            forecast_models = list(model_series.keys())

            f_col1, f_col2 = st.columns([2, 1])
            with f_col1:
                selected_models = st.multiselect(
                    "Forecast Models to Display",
                    forecast_models,
                    default=forecast_models,
                )
            with f_col2:
                max_points = max(24, len(combined))
                window_size = st.slider(
                    "Visible Timeline (5-min steps)",
                    min_value=24,
                    max_value=min(288, max_points),
                    value=min(144, max_points),
                    step=12,
                )

            columns_to_plot = ["actual_speed", *selected_models]
            st.line_chart(combined[columns_to_plot].tail(window_size), height=420)

    with tab3:
        st.subheader("Sensor Profiling & Baseline Error Growth")
        st.markdown(
            "Empirical speed distributions and persistence error degradation across forecast horizons."
        )

        sensor_ds = st.radio("Select Network", ["METR-LA", "PEMS-BAY"], horizontal=True)
        fname_prefix = "metr-la" if sensor_ds == "METR-LA" else "pems-bay"
        analysis_csv = METRICS_PATH / f"{fname_prefix}_sample_sensor_analysis.csv"
        analysis_fig = FIGURES_PATH / f"29_{fname_prefix}_sample_sensor_analysis.png"

        if analysis_csv.is_file():
            sensor_df = read_csv(str(analysis_csv))
            st.dataframe(sensor_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"Sensor analysis CSV not found: {analysis_csv.name}")

        if analysis_fig.is_file():
            st.image(str(analysis_fig), caption=f"{sensor_ds} Sample Sensor Diagnostics & Persistence Degradation")

    with tab4:
        st.subheader("Simulated Signal Policy: Fixed vs. Adaptive Control")
        st.markdown(
            "Dynamic signal control simulation demonstrating Greenshields demand ingestion and queue dissipation."
        )

        sim_csv = METRICS_PATH / "simulated_signal_policy_metrics.csv"
        sim_fig = FIGURES_PATH / "28_simulated_signal_policy_experiment.png"

        if sim_csv.is_file():
            sim_df = read_csv(str(sim_csv))
            sim_df["timestamp"] = pd.to_datetime(sim_df["timestamp"])

            m1, m2, m3 = st.columns(3)
            fixed_vals = np.asarray(sim_df["delay_fixed_veh_min"], dtype=np.float64)
            adapt_vals = np.asarray(sim_df["delay_adapt_veh_min"], dtype=np.float64)
            tot_fixed_delay = float(np.sum(fixed_vals))
            tot_adapt_delay = float(np.sum(adapt_vals))
            delay_reduction = (
                (tot_fixed_delay - tot_adapt_delay) / tot_fixed_delay * 100
                if tot_fixed_delay > 0
                else 0.0
            )

            m1.metric("Fixed Total Delay", f"{tot_fixed_delay:.1f} veh-min")
            m2.metric("Adaptive Total Delay", f"{tot_adapt_delay:.1f} veh-min")
            m3.metric("Delay Reduction", f"{delay_reduction:.1f}%")

            st.markdown("#### Queue Comparison over Time")
            st.line_chart(
                sim_df.set_index("timestamp")[["queue_fixed_total", "queue_adapt_total"]],
                height=320,
            )

        if sim_fig.is_file():
            st.image(str(sim_fig), caption="Signal Policy Simulation: Queues, Delays, and Adaptive Green Splits")

    st.divider()
    st.info(
        "💡 Speed is measured directly via loop detectors. Flow/occupancy and signal control "
        "dynamics are simulated using macroscopic traffic flow theory (Greenshields model)."
    )


if __name__ == "__main__":
    run_dashboard()
else:
    # When imported by Streamlit runner
    run_dashboard()
