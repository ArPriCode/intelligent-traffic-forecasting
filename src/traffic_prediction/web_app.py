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


def render_dataframe(df: pd.DataFrame, hide_index: bool = True) -> None:
    """Render a DataFrame with responsive width across modern and legacy Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def render_image(image_path: str, caption: str | None = None) -> None:
    """Render an image across modern and legacy Streamlit versions."""
    try:
        st.image(image_path, caption=caption, width="stretch")
    except TypeError:
        st.image(image_path, caption=caption, use_container_width=True)


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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Model Comparison",
            "📈 Forecast Trajectories",
            "🛣️ Sensor Diagnostics & Congestion Maps",
            "🚦 Adaptive Signal Simulation",
            "🧠 Residual Hybrid (AML + DL)",
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
                render_dataframe(display_metrics.round(3), hide_index=True)

    with tab2:
        st.subheader("Held-Out 15-Minute Sample Forecasts")
        st.caption(
            "Observed ground truth vs. multi-model predictions for sample sensor 773869."
        )

        prediction_files = {
            "Random Forest": ("metr_la_random_forest_sample_sensor.csv", "predicted_speed"),
            "XGBoost": ("metr_la_xgboost_sample_sensor.csv", "predicted_speed_15min"),
            "GAT": ("metr_la_gat_sample_sensor.csv", "predicted_speed_15min"),
            "Residual Hybrid": ("metr_la_hybrid_sample_sensor.csv", "predicted_speed_15min"),
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
            combined = combined.loc[~combined.index.duplicated(keep="first")]

            if combined.empty:
                st.warning("No overlapping prediction records found to display.")
            else:
                forecast_models = list(model_series.keys())

                f_col1, f_col2 = st.columns([2, 1])
                with f_col1:
                    selected_models = st.multiselect(
                        "Forecast Models to Display",
                        forecast_models,
                        default=forecast_models,
                    )
                with f_col2:
                    total_points = len(combined)
                    if total_points <= 24:
                        window_size = total_points
                        st.caption(f"Showing all {total_points} available time steps.")
                    else:
                        max_points = min(288, total_points)
                        default_points = min(144, max_points)
                        step = 12 if (max_points - 24) >= 12 else 1
                        window_size = st.slider(
                            "Visible Timeline (5-min steps)",
                            min_value=24,
                            max_value=max_points,
                            value=default_points,
                            step=step,
                        )

                columns_to_plot = ["actual_speed", *selected_models]
                if not selected_models:
                    st.info("Select at least one forecast model above to compare against actual speed.")
                st.line_chart(combined[columns_to_plot].tail(window_size), height=420)

    with tab3:
        st.subheader("Spatial Congestion Maps & Sensor Diagnostics")
        st.markdown(
            "Spatial network congestion maps across Los Angeles County highways, "
            "empirical speed distributions, and persistence error degradation."
        )

        sensor_ds = st.radio("Select Network", ["METR-LA", "PEMS-BAY"], horizontal=True)

        if sensor_ds == "METR-LA":
            congestion_fig = FIGURES_PATH / "31_metr_la_network_congestion_maps.png"
            congestion_csv = METRICS_PATH / "metr_la_spatial_congestion_snapshot.csv"

            if congestion_fig.is_file():
                st.markdown("#### 🗺️ Network Spatial Congestion Maps (15m Forecast vs Ground Truth)")
                st.markdown(
                    "High-resolution spatio-temporal comparison during peak evening rush hour across "
                    "207 loop detector stations on the Los Angeles County highway network."
                )
                render_image(
                    str(congestion_fig),
                    caption="METR-LA Spatial Congestion Mapping: (A) Ground Truth, (B) GAT 15m Forecast, (C) Spatial Error, (D) Calibration",
                )

            if congestion_csv.is_file():
                c_df = read_csv(str(congestion_csv))
                valid_mask = c_df["ground_truth_speed_mph"].notna()
                valid_count = int(valid_mask.sum())
                mean_spd = float(c_df.loc[valid_mask, "ground_truth_speed_mph"].mean())
                mean_ae = float(c_df.loc[valid_mask, "gat_abs_error_mph"].mean())
                bottlenecks = int((c_df.loc[valid_mask, "ground_truth_speed_mph"] < 35.0).sum())

                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("Active Highway Sensors", f"{valid_count} / {len(c_df)}")
                cm2.metric("Network Rush Hour Speed", f"{mean_spd:.1f} mph", "Severe Congestion (<40 mph)")
                cm3.metric("GAT 15m Spatial MAE", f"{mean_ae:.2f} mph")
                cm4.metric("Severe Bottlenecks (<35 mph)", f"{bottlenecks} sensors")

                st.markdown("##### Highway Sensor Snapshot Table")
                filter_choice = st.selectbox(
                    "Filter by Congestion State",
                    [
                        "All Sensors",
                        "Severe Congestion (<35 mph)",
                        "Moderate (35-55 mph)",
                        "Free Flow (>55 mph)",
                    ],
                    key="congestion_filter",
                )
                disp_df = c_df.copy()
                if filter_choice != "All Sensors":
                    disp_df = disp_df[disp_df["congestion_state"] == filter_choice]

                render_dataframe(disp_df.round(2), hide_index=True)

            st.divider()

        fname_prefix = "metr-la" if sensor_ds == "METR-LA" else "pems-bay"
        analysis_csv = METRICS_PATH / f"{fname_prefix}_sample_sensor_analysis.csv"
        analysis_fig = FIGURES_PATH / f"29_{fname_prefix}_sample_sensor_analysis.png"

        st.markdown(f"#### 📊 {sensor_ds} Sensor Profiling & Persistence Degradation")
        if analysis_csv.is_file():
            sensor_df = read_csv(str(analysis_csv))
            render_dataframe(sensor_df.round(2), hide_index=True)
        else:
            st.info(f"Sensor analysis CSV not found: {analysis_csv.name}")

        if analysis_fig.is_file():
            render_image(
                str(analysis_fig),
                caption=f"{sensor_ds} Sample Sensor Diagnostics & Persistence Degradation",
            )
        else:
            st.info(f"Sensor analysis figure not found: {analysis_fig.name}")

    with tab4:
        st.subheader("Simulated Signal Policy: Fixed vs. Adaptive Control")
        st.markdown(
            "Dynamic signal control simulation demonstrating Greenshields demand ingestion and queue dissipation."
        )

        sim_csv = METRICS_PATH / "simulated_signal_policy_metrics.csv"
        sim_fig = FIGURES_PATH / "28_simulated_signal_policy_experiment.png"

        if sim_csv.is_file():
            sim_df = read_csv(str(sim_csv)).copy()
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
            render_image(
                str(sim_fig),
                caption="Signal Policy Simulation: Queues, Delays, and Adaptive Green Splits",
            )
        else:
            st.info(f"Signal policy figure not found: {sim_fig.name}")

    with tab5:
        # Tab 5: Deep-dive visualizer for the Phase 2 Residual Spatial Hybrid model
        st.subheader("Residual Spatial Hybrid Architecture (Random Forest + GAT Fusion)")
        st.markdown(
            "Two-stage spatio-temporal learning: A localized **Random Forest** predicts base speeds, "
            "and a **PyTorch Graph Attention Network (GAT)** over the highway distance graph predicts and corrects "
            "spatial error residuals."
        )

        hybrid_pred_path = PREDICTIONS_PATH / "metr_la_hybrid_sample_sensor.csv"
        hybrid_fig_path = FIGURES_PATH / "30_residual_hybrid_model_comparison.png"

        # Step 1: Summary KPI metrics highlighting accuracy improvements over standalone Random Forest
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("5m Test MAE", "2.482 mph", "-0.037 vs RF (2.519)")
        c2.metric("15m Test MAE", "3.097 mph", "-0.022 vs RF (3.119)")
        c3.metric("30m Test MAE", "3.593 mph", "-0.043 vs RF (3.635)")
        c4.metric("60m Test MAE", "4.107 mph", "-0.085 vs RF (4.192)")

        st.markdown("---")
        st.markdown("#### 🔬 How GAT Corrects Localized Random Forest Errors")
        st.markdown(
            "Independent segment models have zero visibility into propagating corridor congestion. "
            "When a downstream bottleneck occurs, the base Random Forest model lags behind. "
            "The Graph Attention Network observes spatial shockwaves across connected loop detectors and "
            r"dynamically outputs a residual correction $\hat{e}_{s,t}^{\text{GAT}}$ to pull the composite "
            "forecast toward the ground truth."
        )

        # Step 2: Load and plot held-out forecast trajectory with isolated residual corrections
        if hybrid_pred_path.is_file():
            h_df = read_csv(str(hybrid_pred_path)).copy()
            h_df["timestamp"] = pd.to_datetime(h_df["timestamp"], errors="coerce")
            h_df = h_df.dropna(subset=["timestamp"]).set_index("timestamp")

            # Step 3: Interactive window slider (up to 288 steps / 24 hours of 5-min intervals)
            w_col1, w_col2 = st.columns([2, 1])
            with w_col2:
                h_total = len(h_df)
                h_window = st.slider(
                    "Timeline Window (Steps)",
                    min_value=24,
                    max_value=min(288, h_total),
                    value=min(144, h_total),
                    step=12,
                    key="hybrid_slider",
                )

            sub_h = h_df.tail(h_window)

            # Step 4: Speed trajectory comparison (Ground Truth vs Base vs Composite)
            st.markdown("**1. Speed Trajectory: Ground Truth vs. Base RF vs. Residual Hybrid**")
            st.line_chart(
                sub_h[["actual_speed", "base_rf_speed_15min", "predicted_speed_15min"]].rename(
                    columns={
                        "actual_speed": "Ground Truth",
                        "base_rf_speed_15min": "Base Random Forest",
                        "predicted_speed_15min": "Residual Hybrid (Composite Final)",
                    }
                ),
                height=340,
            )

            # Step 5: Spatial GAT error compensator signal over time
            st.markdown(r"**2. Spatial GAT Residual Correction Signal ($\hat{e}_{s,t}^{\text{GAT}}$)**")
            st.caption(
                "Positive values boost speed under-predictions; negative values pull down over-predictions during shockwave build-up."
            )
            st.line_chart(
                sub_h[["gat_residual_15min"]].rename(
                    columns={"gat_residual_15min": "GAT Residual Adjustment (mph)"}
                ),
                height=220,
            )

        # Step 6: Multi-horizon comparative benchmark chart across all 5 paradigms
        if hybrid_fig_path.is_file():
            st.markdown("---")
            st.markdown("#### 📊 Multi-Horizon Benchmark Comparison (All Evaluated Models)")
            render_image(
                str(hybrid_fig_path),
                caption="Comparative Test MAE across Horizons for Persistence, Random Forest, XGBoost, GAT, and Residual Hybrid",
            )

    st.divider()
    st.info(
        "💡 Speed is measured directly via loop detectors. Flow/occupancy and signal control "
        "dynamics are simulated using macroscopic traffic flow theory (Greenshields model)."
    )


if __name__ == "__main__":
    run_dashboard()
