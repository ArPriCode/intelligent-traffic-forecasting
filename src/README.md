# Reusable Traffic Prediction Source Package (`src/traffic_prediction`)

The `traffic_prediction` package provides production-grade, modular Python utilities for data validation, chronological splitting, causal feature engineering, Gaussian distance graph construction, masked evaluation metrics, and PyTorch Graph Attention Networks.

---

## Package Architecture

```text
src/
└── traffic_prediction/
    ├── __init__.py           # Package exports and semantic versioning
    ├── data.py               # Timeline validation, chronological splitting, causal imputation
    ├── features.py           # Autoregressive speed lags, cyclic calendar embeddings
    ├── graph.py              # Thresholded Gaussian distance adjacency builder
    ├── metrics.py            # Masked MAE, RMSE, MAPE over valid positive observations
    ├── models.py             # PyTorch GraphAttentionLayer & GraphAttentionNetwork
    └── web_app.py            # Streamlit dashboard for interactive forecast analysis
```

---

## Module Documentation & APIs

### 1. `data.py`
Handles rigorous data ingestion, timestamp alignment verification, chronological train/validation/test splitting, and causal forward-fill imputation.

* `time_split(size: int, train_fraction: float = 0.70, validation_fraction: float = 0.10) -> TimeSplit`:
  Computes index boundaries ensuring strictly chronological partitions (70% train, 10% validation, 20% test). Rejects non-chronological or invalid fractions.
* `load_speed_csv(path: str | Path, expected_sensor_ids: list[str] | None = None) -> pd.DataFrame`:
  Loads a five-minute speed CSV, verifies monotonic timestamp progression, validates 5-minute interval spacing, and confirms sensor column alignment.
* `observed_speeds(traffic: pd.DataFrame) -> np.ndarray`:
  Extracts positive speed values; non-positive, zero, and infinite values are set to `np.nan`.
* `prepare_speed_inputs(observed: np.ndarray, train_end: int, statistic: str = "median") -> np.ndarray`:
  Causally forward-fills missing observations. Any initial missing entries prior to the first valid observation are filled using training-set-only sensor statistics (`median` or `mean`).

---

### 2. `features.py`
Generates non-leaking temporal representations for tree-based baselines and deep learning pipelines.

* `time_features(index: pd.DatetimeIndex) -> np.ndarray`:
  Encodes continuous diurnal and weekly cycles using 4 trigonometric features:
  $$[\sin(2\pi m / 1440), \cos(2\pi m / 1440), \sin(2\pi d / 7), \cos(2\pi d / 7)]$$
* `make_sensor_features(speed_inputs, calendar_features, time_rows, sensor_index) -> np.ndarray`:
  Generates feature matrices for segment-wise Random Forest models using speed lags $t, t-1, t-2, t-5$ (covering 25 minutes of history) combined with the 4 cyclic time features.
* `make_panel_features(speed_inputs, calendar_features, sensor_ids, time_rows, sensor_rows) -> pd.DataFrame`:
  Builds panel feature matrices for pooled XGBoost models, including categorical sensor identities.

---

### 3. `graph.py`
Constructs spatial adjacency matrices from road network distance tables.

* `build_distance_adjacency(distances: pd.DataFrame, sensor_ids: list[str], threshold: float = 0.1) -> np.ndarray`:
  Converts pairwise road distance measurements into a symmetric Gaussian distance matrix:
  $$W_{ij} = \exp\left(-\left(\frac{\text{dist}(v_i, v_j)}{\sigma}\right)^2\right)$$
  Filters weak spatial connections ($W_{ij} < 0.1 \rightarrow 0$), symmetrizes via $\max(W, W^T)$, and inserts identity self-loops ($W_{ii} = 1.0$).

---

### 4. `metrics.py`
Computes evaluation statistics over strictly observed, valid traffic targets.

* `regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]`:
  Calculates Masked Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE %):
  $$\text{MAE} = \frac{1}{|V|} \sum_{i \in V} |y_i - \hat{y}_i|$$
  $$\text{RMSE} = \sqrt{\frac{1}{|V|} \sum_{i \in V} (y_i - \hat{y}_i)^2}$$
  $$\text{MAPE} = \frac{1}{|V|} \sum_{i \in V} \frac{|y_i - \hat{y}_i|}{y_i} \times 100\%$$
  where $V = \{i : y_i > 0 \land \text{isfinite}(y_i) \land \text{isfinite}(\hat{y}_i)\}$.
* `peak_mask(timestamps: pd.DatetimeIndex, target_rows: np.ndarray | None = None) -> np.ndarray`:
  Identifies morning peak (07:00–09:59) and evening peak (16:00–18:59) on weekdays (Monday through Friday).

---

### 5. `models.py`
Implements PyTorch spatial graph attention architectures.

* `GraphAttentionLayer(input_size: int, hidden_size: int, adjacency: torch.Tensor)`:
  Computes spatial self-attention constrained by the road network topology. Features are projected via linear transformation, attention logits are calculated using LeakyReLU activations with edge weight biases, normalized across neighbors via softmax, and aggregated with LayerNorm and ELU activations.
* `GraphAttentionNetwork(adjacency: torch.Tensor, input_size: int = 10, hidden_size: int = 32, output_size: int = 4, dropout: float = 0.15)`:
  Full multi-horizon network that ingests 6 lags of speed and time features across all 207 sensors and simultaneously predicts speeds across 4 horizons (5m, 15m, 30m, 60m).

---

### 6. `web_app.py`
Streamlit application for interactive model evaluation and held-out trajectory visual analytics:
* Visualizes dataset-level summary statistics and missing rate metrics.
* Filterable model benchmark tables and dynamic bar charts by period (All, Peak, Off-Peak) and horizon (5, 15, 30, 60 min).
* Interactive time-series forecast comparison against ground truth for sample sensors.

---

## Installation & Running Tests

Install the package in editable mode:
```bash
python -m pip install -e .
```

Run the automated test suite:
```bash
python -m pytest tests/
```

Launch the Streamlit web dashboard:
```bash
streamlit run src/traffic_prediction/web_app.py
```
