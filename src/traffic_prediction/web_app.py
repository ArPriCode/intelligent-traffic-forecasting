"""Local Streamlit dashboard for saved METR-LA forecast results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions"

st.set_page_config(
    page_title="Traffic Forecasting Dashboard",
    page_icon="🚦",
    layout="wide",
)


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


st.title("Traffic Speed Forecasting")
st.caption("METR-LA model comparison and held-out sample forecasts")

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

overview = st.columns(4)
overview[0].metric("Sensors", f"{int(la_quality['sensors']):,}")
overview[1].metric("Five-minute records", f"{int(la_quality['time_steps']):,}")
overview[2].metric(
    "Observed date range",
    f"{la_quality['start']} – {str(la_quality['end'])[:10]}",
)
overview[3].metric("Missing / zero readings", f"{la_quality['missing_or_zero_percent']:.2f}%")

st.divider()
st.subheader("Held-out model performance")
controls = st.columns(3)
available_periods = ["All", "Peak", "Off-peak"]
selected_period = controls[0].selectbox("Period", available_periods)
available_horizons = sorted(comparison["horizon_minutes"].dropna().astype(int).unique())
selected_horizon = controls[1].selectbox(
    "Forecast horizon (minutes)",
    available_horizons,
    index=available_horizons.index(15) if 15 in available_horizons else 0,
)
metric_name = controls[2].selectbox("Metric", ["MAE", "RMSE", "MAPE_percent"])

filtered = comparison.loc[
    (comparison["dataset"] == "METR-LA")
    & (comparison["period"] == selected_period)
    & (comparison["horizon_minutes"] == selected_horizon)
].copy()
if filtered.empty:
    st.warning("No saved metrics match the selected period and forecast horizon.")
else:
    filtered = filtered.sort_values(metric_name)
    st.bar_chart(filtered.set_index("model")[[metric_name]])
    display_metrics = filtered[
        ["model", "MAE", "MAPE_percent", "RMSE", "valid_test_values"]
    ].rename(columns={"MAPE_percent": "MAPE (%)", "valid_test_values": "Valid targets"})
    st.dataframe(display_metrics.round(3), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Sample 15-minute forecasts")
st.caption(
    "These are saved held-out predictions for the same sample sensor, not a live retraining run."
)

prediction_files = {
    "Random Forest": ("metr_la_random_forest_sample_sensor.csv", "predicted_speed"),
    "XGBoost": ("metr_la_xgboost_sample_sensor.csv", "predicted_speed_15min"),
    "GAT": ("metr_la_gat_sample_sensor.csv", "predicted_speed_15min"),
}
prediction_tables: dict[str, pd.DataFrame] = {}
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
    frame = frame[["timestamp", "actual_speed", prediction_column]].rename(
        columns={prediction_column: model_name}
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    prediction_tables[model_name] = frame.dropna(subset=["timestamp"]).set_index("timestamp")

if missing_prediction_files:
    st.error(f"Missing saved forecast files: {', '.join(missing_prediction_files)}")
    st.stop()

if prediction_tables:
    combined = pd.concat(prediction_tables.values(), axis=1, join="inner")
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index()
    forecast_models = list(prediction_tables)
    selected_models = st.multiselect(
        "Forecasts to show",
        forecast_models,
        default=forecast_models,
    )
    window_size = st.slider(
        "Visible forecast points (five minutes each)",
        min_value=24,
        max_value=min(288, len(combined)),
        value=min(144, len(combined)),
        step=12,
    )
    columns_to_plot = ["actual_speed", *selected_models]
    st.line_chart(combined[columns_to_plot].tail(window_size), height=420)

st.info(
    "Only speed is available in the supplied traffic data. Flow/occupancy and traffic-signal "
    "control are not modeled in this dashboard."
)
