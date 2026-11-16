# Intelligent Traffic Forecasting — Benchmark Executive Summary

**Dataset**: METR-LA
**Evaluation Metrics**: Masked MAE, RMSE, and MAPE (%) over held-out chronological test split.

---

## 1. Overall Performance Across Forecast Horizons

| Horizon | Model | MAE (mph) | RMSE (mph) | MAPE (%) | vs. Persistence (MAE Δ) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 5 min | **Persistence** | 2.814 | 4.627 | 6.49% | Baseline |
| 5 min | **Random Forest** | 2.519 | 4.281 | 6.33% | **-10.5%** |
| 5 min | **XGBoost** | 2.554 | 4.326 | 6.44% | **-9.2%** |
| 5 min | **GAT** | 2.879 | 4.511 | 7.21% | +2.3% |
| 5 min | **Residual Hybrid** | 2.478 | 4.239 | 6.14% | **-11.9%** |
| 15 min | **Persistence** | 3.499 | 6.418 | 8.61% | Baseline |
| 15 min | **Random Forest** | 3.119 | 5.702 | 8.43% | **-10.8%** |
| 15 min | **XGBoost** | 3.239 | 5.873 | 8.85% | **-7.4%** |
| 15 min | **GAT** | 3.466 | 5.830 | 9.17% | **-0.9%** |
| 15 min | **Residual Hybrid** | 3.079 | 5.654 | 8.20% | **-12.0%** |
| 30 min | **Persistence** | 4.207 | 8.052 | 10.86% | Baseline |
| 30 min | **Random Forest** | 3.635 | 6.706 | 10.40% | **-13.6%** |
| 30 min | **XGBoost** | 3.854 | 6.988 | 11.11% | **-8.4%** |
| 30 min | **GAT** | 4.087 | 6.965 | 11.43% | **-2.8%** |
| 30 min | **Residual Hybrid** | 3.592 | 6.634 | 10.11% | **-14.6%** |
| 60 min | **Persistence** | 5.381 | 10.345 | 14.54% | Baseline |
| 60 min | **Random Forest** | 4.192 | 7.657 | 12.62% | **-22.1%** |
| 60 min | **XGBoost** | 4.542 | 7.982 | 13.70% | **-15.6%** |
| 60 min | **GAT** | 4.898 | 8.227 | 14.33% | **-9.0%** |
| 60 min | **Residual Hybrid** | 4.132 | 7.547 | 12.28% | **-23.2%** |

---

## 2. Rush Hour Congestion Robustness (Peak vs. Off-Peak)

> [!NOTE]
> **Peak Hours**: Weekdays 07:00–09:59 & 16:00–18:59 PST. **Off-Peak**: All remaining hours.

| Horizon | Model | Peak MAE | Off-Peak MAE | Peak Degradation Ratio |
| :--- | :--- | :---: | :---: | :---: |
| 5 min | **GAT** | 3.601 | 2.696 | 1.34x |
| 5 min | **Persistence** | 3.287 | 2.695 | 1.22x |
| 5 min | **Random Forest** | 3.189 | 2.350 | 1.36x |
| 5 min | **Residual Hybrid** | 3.143 | 2.310 | 1.36x |
| 5 min | **XGBoost** | 3.243 | 2.381 | 1.36x |
| 15 min | **GAT** | 4.737 | 3.146 | 1.51x |
| 15 min | **Persistence** | 4.565 | 3.230 | 1.41x |
| 15 min | **Random Forest** | 4.341 | 2.812 | 1.54x |
| 15 min | **Residual Hybrid** | 4.294 | 2.773 | 1.55x |
| 15 min | **XGBoost** | 4.535 | 2.913 | 1.56x |
| 30 min | **GAT** | 6.001 | 3.605 | 1.66x |
| 30 min | **Persistence** | 5.831 | 3.797 | 1.54x |
| 30 min | **Random Forest** | 5.324 | 3.210 | 1.66x |
| 30 min | **Residual Hybrid** | 5.274 | 3.169 | 1.66x |
| 30 min | **XGBoost** | 5.808 | 3.362 | 1.73x |
| 60 min | **GAT** | 7.940 | 4.131 | 1.92x |
| 60 min | **Persistence** | 7.804 | 4.770 | 1.64x |
| 60 min | **Random Forest** | 6.374 | 3.643 | 1.75x |
| 60 min | **Residual Hybrid** | 6.319 | 3.581 | 1.76x |
| 60 min | **XGBoost** | 7.277 | 3.853 | 1.89x |

---

## 3. Spatial Network Congestion Mapping & Shockwave Verification

> [!TIP]
> Spatial snapshot evaluated during peak evening rush hour across 207 highway loop detectors.

* **Peak Congestion Timestamp**: `2012-06-14 17:45:00`
* **Active Highway Detectors**: 186 / 207 stations reporting
* **Network-Wide Rush Hour Speed**: **38.06 mph**
* **GAT 15m Network MAE**: **4.855 mph** (Median: **3.387 mph**)
* **Severe Highway Bottlenecks (<35 mph)**: **104** sensor stations
* **High-Resolution Spatial Map Artifact**: `results/figures/31_metr_la_network_congestion_maps.png`

