# Traffic Flow Control and Signal Control Simulation

Notebook-first speed forecasting project using five-minute METR-LA and PEMS-BAY observations. It covers data quality, EDA, AML baselines, graph construction, a trained Graph Attention Network (GAT), and a documented optional signal-control extension.

## Current data

- `data/METR-LA.csv`: 34,272 timestamps × 207 sensor speed series.
- `data/PEMS-BAY.csv`: 52,116 timestamps × 325 sensor speed series.
- `data/PEMS-BAY-META.csv`: Bay Area sensor coordinates and road attributes.
- `data/sensor_graph/`: METR-LA sensor IDs, locations, and official road-distance table from DCRNN commit `602afd9d767d3aa1c9b3eac51710d6aeee12c227`. All 207 graph IDs align with the METR-LA speed columns.

The traffic files contain speed, not flow or occupancy. Those variables are not fabricated. The METR-LA distance graph is symmetrized and does not provide verified upstream/downstream travel direction.

## Quick start

```bash
cd Traffic-Flow-Prediction
python -m pip install -r requirements.txt
```

Run notebooks 01–09 in numeric order. Notebook 02 contains short, step-by-step EDA cells and plots. Notebooks 05–08 fit and evaluate models; notebook 09 documents the requirements for optional signal simulation.

## Project layout

```text
Traffic-Flow-Prediction/
├── data/
│   ├── METR-LA.csv
│   ├── PEMS-BAY.csv
│   ├── PEMS-BAY-META.csv
│   └── sensor_graph/
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_protocol.ipynb
│   ├── 04_features.ipynb
│   ├── 05_baselines.ipynb
│   ├── 06_xgboost.ipynb
│   ├── 07_backtest.ipynb
│   ├── 08_gat_model.ipynb
│   └── 09_signal_simulation.ipynb
├── src/
│   └── traffic_prediction/
│       ├── data.py
│       ├── features.py
│       ├── graph.py
│       ├── metrics.py
│       └── models.py
├── report/
└── results/
    ├── figures/
    ├── metrics/
    └── predictions/
```

## AML / DL completion matrix

| Requirement | Status | Implementation / caveat |
|---|---|---|
| Data validation and EDA | **Done** | Timestamp/sensor quality checks and time-series, sensor, metadata and coordinate visualizations |
| AML lagged speed features | **Done** | Causal speed lags, cyclic time-of-day and weekday, plus sensor/segment identity |
| Flow and occupancy features | **Unavailable in data** | The provided traffic CSVs contain speed only; flow/occupancy features are therefore omitted |
| AML multi-output Random Forest | **Done** | One four-horizon model per METR-LA sensor segment |
| AML XGBoost | **Done** | Pooled sensor-aware model evaluated at 5/15/30/60 minutes |
| Missing/outage handling | **Done** | Zero/unobserved readings are forward-filled only for causal inputs and masked from training targets and metrics |
| Chronological split / scaling | **Done** | 70% train / 10% validation / 20% test; GAT scaler fitted on training timestamps only; no random sequence split |
| Fixed graph construction | **Done** | Official METR-LA distances aligned to all 207 sensors; distance-kernel adjacency, threshold 0.1, 1,313 undirected links |
| Graph Attention Network | **Done** | PyTorch GAT trained on MPS; masked loss, validation-based early stopping, forecast and weights saved |
| Fair horizon and period comparison | **Done** | Persistence, RF, XGBoost and GAT share test origins/valid target counts; all, peak and off-peak MAE/RMSE/MAPE are saved |
| Directed upstream/downstream propagation | **Partial** | Spatial attention uses a symmetrized distance graph; direction of travel is not verified in the supplied graph |
| SUMO signal-policy experiment | **Optional / not done** | Requires flow/queue observations, signal phases and timing plans not supplied here |

The GAT successfully trains, but the current run does **not** beat Random Forest or XGBoost on overall MAE. This is reported as a result, not hidden as a successful improvement.

## METR-LA held-out MAE (speed units)

| Model | 5 min | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|
| Persistence | 2.814 | 3.499 | 4.207 | 5.381 |
| Random Forest | **2.519** | **3.119** | **3.635** | **4.192** |
| XGBoost | 2.554 | 3.239 | 3.854 | 4.542 |
| GAT | 2.879 | 3.466 | 4.087 | 4.898 |

The full comparison includes MAE, RMSE and MAPE for all/peak/off-peak periods in [`report/main.md`](./report/main.md) and `results/metrics/metr_la_model_comparison.csv`. Figures, metrics and sample forecasts are saved under `results/`.

## Reproduction and attribution

Install `requirements.txt`, then execute the notebooks in numeric order. Model notebooks save results to `results/metrics/`, `results/predictions/` and `results/figures/`.

The reusable source package can be installed in editable mode with `python -m pip install -e .`. It provides traffic CSV validation and chronological splitting, causal speed-input preparation, lag/calendar feature builders, masked forecast metrics, distance-graph construction, and the PyTorch graph-attention model. The notebooks are unchanged and remain the existing executed workflow.

METR-LA/PEMS-BAY data are distributed by the DCRNN authors; follow the dataset terms and citation on the [DCRNN project page](https://github.com/liyaguang/DCRNN). Traffic CSVs and PEMS-BAY metadata are user-provided.
