# Traffic Speed Forecasting: AML Baselines and Graph Learning

### Team Quanta
**Course**: Applied Machine Learning (AML)  
**Institution**: Newton School of Technology, Rishihood University  

| Student Name | Enrollment Number | Institutional Email |
| :--- | :---: | :--- |
| **Arun Kumar Giri** | `2401010099` | [arun.giri2024@nst.rishihood.edu.in](mailto:arun.giri2024@nst.rishihood.edu.in) |
| **Gaurav Meena** | `2401010169` | [gaurav.meena2024@nst.rishihood.edu.in](mailto:gaurav.meena2024@nst.rishihood.edu.in) |
| **Mausam Kumar Dwivedi** | `2401010266` | [mausam.dwivedi2024@nst.rishihood.edu.in](mailto:mausam.dwivedi2024@nst.rishihood.edu.in) |
| **Ritik Atri** | `2401010386` | [ritik.attri2024@nst.rishihood.edu.in](mailto:ritik.attri2024@nst.rishihood.edu.in) |
| **Daniel Tayal** | `2401010140` | [daniel.tayal2024@nst.rishihood.edu.in](mailto:daniel.tayal2024@nst.rishihood.edu.in) |

---

## Summary

This project evaluates speed forecasts from METR-LA using persistence, a segment-wise multi-output Random Forest, pooled XGBoost, and a trained Graph Attention Network (GAT). Splits are chronological, missing/zero target speeds are masked, and the test set is evaluated at 5, 15, 30, and 60 minutes. The GAT and fixed graph are implemented and trained. In this run the GAT does not beat the AML tree models on overall MAE; its results and limitations are reported below rather than framed as an improvement.

## Data quality and exploratory analysis

| Dataset | Time steps | Sensors | Date range | Missing or zero |
|---|---:|---:|---|---:|
| METR-LA | 34,272 | 207 | 2012-03-01 to 2012-06-27 | 8.11% |
| PEMS-BAY | 52,116 | 325 | 2017-01-01 to 2017-06-30 | 0.0031% |

The traffic CSVs contain speed only; flow and occupancy are unavailable and are not imputed. PEMS-BAY metadata matches the Bay Area sensors and supplies road attributes and coordinates. Notebook 02 includes time-series, hourly/weekday patterns, sensor heatmaps and distributions, coordinate maps, road metadata, missingness, and correlation views. Zero speeds are treated as missing observations, forward-filled only when building causal inputs, and excluded from target fitting and metrics.

The official METR-LA graph ID order matches all 207 speed-series columns. Graph IDs, locations, and road distances were obtained from DCRNN commit `602afd9d767d3aa1c9b3eac51710d6aeee12c227`. The distance table is converted to a Gaussian distance-weighted adjacency using the DCRNN thresholding rule (`normalized_k=0.1`); the matrix is symmetrized and self-loops are added. It has 207 nodes and 1,313 undirected links (mean degree 12.7). This is a spatial distance graph, not a verified directed upstream/downstream network.

## Forecast protocol and models

The METR-LA time axis is split chronologically: earliest 70% for training, next 10% for validation, final 20% for test. Forecast horizons are 5, 15, 30, and 60 minutes. Peak is weekdays 07:00–09:59 and 16:00–18:59; off-peak is all remaining periods, including weekends. MAE, RMSE, and MAPE use observed nonzero target readings only.

- **Persistence:** predicts the last observed speed.
- **Random Forest:** one multi-output model per sensor, forecasting all four horizons from causal speed lags and cyclic time features; up to 2,000 training timestamps per sensor.
- **XGBoost:** pooled sensor-aware models with speed lags, cyclic time features, and sensor identity; up to 50,000 observed sensor-time targets per horizon.
- **GAT:** spatial attention over the fixed 207-sensor graph with six consecutive five-minute speed observations and cyclic time/weekday features. It uses the same chronological split and 25-minute maximum history. Per-sensor scaling is fitted on training timestamps only; missing inputs are causally forward-filled, while missing targets are masked. A custom PyTorch implementation was trained on MPS for six epochs with validation-based early stopping; best validation masked MSE was 0.4220.

The comparison uses identical test origins and valid target counts for every model, horizon, and period. The exact feature representation and training volume differ by model: the RF is capped at 2,000 timestamps per sensor, XGBoost at 50,000 targets per horizon, and GAT uses all eligible training windows. Therefore, this is a controlled chronological evaluation, but not an equal-compute benchmark.

## Overall METR-LA held-out metrics

| Horizon | Model | MAE | MAPE (%) | RMSE | Valid targets |
|---|---|---:|---:|---:|---:|
| 5 min | Persistence | 2.814 | 6.489 | 4.627 | 1,244,846 |
| 5 min | Random Forest | **2.519** | 6.327 | **4.281** | 1,244,846 |
| 5 min | XGBoost | 2.554 | 6.435 | 4.326 | 1,244,846 |
| 5 min | GAT | 2.879 | 7.206 | 4.511 | 1,244,846 |
| 15 min | Persistence | 3.499 | 8.612 | 6.418 | 1,244,834 |
| 15 min | Random Forest | **3.119** | **8.431** | **5.702** | 1,244,834 |
| 15 min | XGBoost | 3.239 | 8.846 | 5.873 | 1,244,834 |
| 15 min | GAT | 3.466 | 9.168 | 5.830 | 1,244,834 |
| 30 min | Persistence | 4.207 | 10.858 | 8.052 | 1,244,816 |
| 30 min | Random Forest | **3.635** | **10.401** | **6.706** | 1,244,816 |
| 30 min | XGBoost | 3.854 | 11.106 | 6.988 | 1,244,816 |
| 30 min | GAT | 4.087 | 11.434 | 6.965 | 1,244,816 |
| 60 min | Persistence | 5.381 | 14.535 | 10.345 | 1,244,780 |
| 60 min | Random Forest | **4.192** | **12.623** | **7.657** | 1,244,780 |
| 60 min | XGBoost | 4.542 | 13.696 | 7.982 | 1,244,780 |
| 60 min | GAT | 4.898 | 14.328 | 8.227 | 1,244,780 |

Random Forest has the lowest overall MAE at each horizon. GAT improves on persistence MAE at 15, 30, and 60 minutes, but not at 5 minutes, and remains behind both tree models overall.

## Peak and off-peak MAE

| Period | Model | 5 min | 15 min | 30 min | 60 min |
|---|---|---:|---:|---:|---:|
| Peak | Persistence | 3.287 | 4.565 | 5.831 | 7.804 |
| Peak | Random Forest | **3.189** | **4.341** | **5.324** | **6.374** |
| Peak | XGBoost | 3.243 | 4.535 | 5.808 | 7.277 |
| Peak | GAT | 3.601 | 4.737 | 6.001 | 7.940 |
| Off-peak | Persistence | 2.695 | 3.230 | 3.797 | 4.770 |
| Off-peak | Random Forest | **2.350** | **2.812** | **3.210** | **3.643** |
| Off-peak | XGBoost | 2.381 | 2.913 | 3.362 | 3.853 |
| Off-peak | GAT | 2.696 | 3.146 | 3.605 | 4.131 |

Complete MAE, MAPE, RMSE and valid-target counts for all periods are in `results/metrics/metr_la_model_comparison.csv`. RF/XGBoost and GAT period-specific source tables are also saved separately in `results/metrics/`.

![Peak/off-peak comparison](../results/figures/27_metr_la_peak_offpeak_model_comparison.png)

## Simulated Signal Policy Experiment (Optional Deliverable)

An intersection queuing simulation was executed in `notebooks/09_signal_simulation.ipynb` to evaluate dynamic signal control without claiming live traffic control. Using the Greenshields fundamental diagram ($q = k \cdot v$), arrival demands for an arterial corridor were inferred from the 15-minute speed forecasts and matched against a cross-arterial. 

Comparing a **Fixed-Time Signal Policy** (static 90s cycle, 53% / 36% split) against a **Forecast-Aware Adaptive (Max-Pressure) Signal Policy** (dynamically adjusting green splits based on anticipated queue buildup), the adaptive policy achieved a **42.1% reduction in total intersection vehicle delay** during peak congestion periods.

![Simulated Signal Policy Experiment](../results/figures/28_simulated_signal_policy_experiment.png)

## Deliverables and limitations

- AML baselines: `notebooks/05_baselines.ipynb`, `notebooks/06_xgboost.ipynb`.
- Chronological comparison: `notebooks/07_backtest.ipynb`.
- Graph construction and trained GAT: `notebooks/08_gat_model.ipynb`.
- Simulated signal policy experiment: `notebooks/09_signal_simulation.ipynb`.
- Model metrics, generated adjacency, sample predictions, and trained state are under `results/metrics/` and `results/predictions/`.
- Figures include the fixed adjacency, GAT training/forecasts, peak/off-peak comparison, and simulated signal dynamics under `results/figures/`.

Although spatial attention is implemented, the symmetrized distance graph has no verified travel direction. Real-world live traffic control is strictly not claimed; full field control requires microscopic simulators (SUMO / CityFlow) with calibrated origin-destination volumes, phase sequences, and detector actuations. The project-run scores are reported transparently and not claimed to reproduce separate external reference scores.
