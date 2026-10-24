# Spatiotemporal Traffic Speed Forecasting: Applied Machine Learning Baselines and Spatial Graph Attention

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

## Abstract
Short-term traffic speed forecasting is a cornerstone of modern Intelligent Transportation Systems (ITS) and dynamic congestion mitigation. Under the CSAI Project 4 mandate (*Traffic Flow Control via Graph Attention Networks and Flow Models*), this study investigates whether spatial graph representation learning enhances network-wide speed forecasts over independent road segment models. Using 5-minute loop-detector speed readings from METR-LA (207 sensors, 34,272 timestamps) and PEMS-BAY (325 sensors, 52,116 timestamps), we enforce a strictly causal, chronological evaluation protocol (70% train, 10% validation, 20% test). We formulate and benchmark four distinct paradigms across 5, 15, 30, and 60-minute horizons: Naive Persistence, Segment-wise multi-output Random Forests, Pooled sensor-aware XGBoost, and a spatial Graph Attention Network (GAT) operating over a Gaussian distance-thresholded sensor adjacency matrix ($\kappa = 0.1$, 1,313 links). Sensor outages (zero speeds) are strictly masked from ground-truth loss and evaluation metrics. Evaluated on **1,244,846 valid test target points**, segment-wise Random Forest achieves the lowest MAE across all horizons (2.519 at 5 min to 4.192 at 60 min), outperforming pooled XGBoost and standalone spatial GAT.

---

## 1. Mathematical Formulations

### 1.1 Sensor Outage Masking
Loop detector dropouts and communication failures record speed as 0.0 or NaN. The binary observation mask $m_{s,t}$ for sensor $s$ at time $t$ is:
$$m_{s,t} = \begin{cases} 1, & \text{if } v_{s,t} > 0 \text{ and } v_{s,t} \in \mathbb{R} \\ 0, & \text{otherwise (sensor outage)} \end{cases}$$

Inputs are causally forward-filled using past valid speeds. For model loss optimization and metric computation, all points with $m_{s,t} = 0$ are strictly excluded.

### 1.2 Feature Engineering
For any forecast origin $t$ and sensor $s$, the tabular feature vector $\mathbf{x}_{s,t}$ incorporates:
1. **Causal Speed Lags**:
   $$\mathbf{x}_{\text{lag}}(s, t) = [v_{s,t}, \, v_{s,t-1}, \, v_{s,t-2}, \, v_{s,t-5}]^T \quad (0, 5, 10, \text{ and } 25 \text{ mins prior})$$
2. **Cyclical Calendar Encodings**:
   $$\phi_{\text{tod}}^{\sin}(t) = \sin\left(\frac{2\pi \cdot \text{minute}(t)}{1440}\right), \quad \phi_{\text{tod}}^{\cos}(t) = \cos\left(\frac{2\pi \cdot \text{minute}(t)}{1440}\right)$$
   $$\phi_{\text{dow}}^{\sin}(t) = \sin\left(\frac{2\pi \cdot \text{weekday}(t)}{7}\right), \quad \phi_{\text{dow}}^{\cos}(t) = \cos\left(\frac{2\pi \cdot \text{weekday}(t)}{7}\right)$$
3. **1-Hop Spatial Neighbor Aggregation**:
   $$\bar{v}_{\mathcal{N}(s)}(t) = \frac{\sum_{j \in \mathcal{N}(s)} A_{sj} v_{j,t}}{\sum_{j \in \mathcal{N}(s)} A_{sj}}, \quad \mathcal{N}(s) = \{j : j \ne s, A_{sj} > 0\}$$
4. **Causal Rolling Moments** ($w \in \{3, 6\}$ intervals, corresponding to 15 and 30 mins):
   $$\mu_{s,t}^{(w)} = \frac{1}{w} \sum_{k=0}^{w-1} v_{s,t-k}, \quad \sigma_{s,t}^{(w)} = \sqrt{\frac{1}{w} \sum_{k=0}^{w-1} (v_{s,t-k} - \mu_{s,t}^{(w)})^2}$$

### 1.3 Loss & Evaluation Metrics
Evaluated over the valid test set $\Omega_h = \{(s,t) : t \in \mathcal{T}_{\text{test}}, m_{s,t+h} = 1\}$:
* **Masked Mean Absolute Error (MAE)**:
  $$\text{MAE}_h = \frac{1}{|\Omega_h|} \sum_{(s,t) \in \Omega_h} |y_{s,t+h} - \hat{y}_{s,t+h}|$$
* **Masked Root Mean Squared Error (RMSE)**:
  $$\text{RMSE}_h = \sqrt{\frac{1}{|\Omega_h|} \sum_{(s,t) \in \Omega_h} (y_{s,t+h} - \hat{y}_{s,t+h})^2}$$
* **Masked Mean Absolute Percentage Error (MAPE)**:
  $$\text{MAPE}_h = \frac{100\%}{|\Omega_h|} \sum_{(s,t) \in \Omega_h} \left| \frac{y_{s,t+h} - \hat{y}_{s,t+h}}{y_{s,t+h}} \right|$$

---

## 2. Chronological Benchmark Results

### 2.1 Overall Held-Out Test Results (All Periods, 1,244,846 Valid Points)

| Horizon | Model | MAE (mph) | RMSE (mph) | MAPE (%) | Valid Test Points |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **5 min** | Persistence | 2.814 | 4.627 | 6.489 | 1,244,846 |
| **5 min** | **Random Forest** | **2.519** | **4.281** | **6.327** | 1,244,846 |
| **5 min** | XGBoost | 2.554 | 4.326 | 6.435 | 1,244,846 |
| **5 min** | GAT | 2.879 | 4.511 | 7.206 | 1,244,846 |
| **15 min** | Persistence | 3.499 | 6.418 | 8.612 | 1,244,834 |
| **15 min** | **Random Forest** | **3.119** | **5.702** | **8.431** | 1,244,834 |
| **15 min** | XGBoost | 3.239 | 5.873 | 8.846 | 1,244,834 |
| **15 min** | GAT | 3.466 | 5.830 | 9.168 | 1,244,834 |
| **30 min** | Persistence | 4.207 | 8.052 | 10.858 | 1,244,816 |
| **30 min** | **Random Forest** | **3.635** | **6.706** | **10.401** | 1,244,816 |
| **30 min** | XGBoost | 3.854 | 6.988 | 11.106 | 1,244,816 |
| **30 min** | GAT | 4.087 | 6.965 | 11.434 | 1,244,816 |
| **60 min** | Persistence | 5.381 | 10.345 | 14.535 | 1,244,780 |
| **60 min** | **Random Forest** | **4.192** | **7.657** | **12.623** | 1,244,780 |
| **60 min** | XGBoost | 4.542 | 7.982 | 13.696 | 1,244,780 |
| **60 min** | GAT | 4.898 | 8.227 | 14.328 | 1,244,780 |

---

### 2.2 Stratified Peak vs. Off-Peak Analysis

* **Peak Period**: Weekdays 07:00–09:59 and 16:00–18:59 (250,523 valid points).
* **Off-Peak Period**: All remaining daytime/nighttime hours and weekends (994,323 valid points).

| Period | Horizon | Model | MAE (mph) | RMSE (mph) | MAPE (%) |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **Peak** | **5 min** | Persistence | 3.287 | 5.479 | 10.044 |
| **Peak** | **5 min** | **Random Forest** | **3.189** | **5.381** | **10.712** |
| **Peak** | **5 min** | XGBoost | 3.243 | 5.452 | 10.832 |
| **Peak** | **5 min** | GAT | 3.601 | 5.618 | 11.918 |
| **Peak** | **15 min** | Persistence | 4.565 | 8.223 | 14.384 |
| **Peak** | **15 min** | **Random Forest** | **4.341** | **7.580** | **15.101** |
| **Peak** | **15 min** | XGBoost | 4.535 | 7.868 | 15.820 |
| **Peak** | **15 min** | GAT | 4.737 | 7.713 | 15.852 |
| **Peak** | **30 min** | Persistence | 5.831 | 10.552 | 19.080 |
| **Peak** | **30 min** | **Random Forest** | **5.324** | **9.105** | **19.281** |
| **Peak** | **30 min** | XGBoost | 5.808 | 9.612 | 20.926 |
| **Peak** | **30 min** | GAT | 6.001 | 9.487 | 20.818 |
| **Peak** | **60 min** | Persistence | 7.804 | 13.683 | 27.225 |
| **Peak** | **60 min** | **Random Forest** | **6.374** | **10.495** | **24.259** |
| **Peak** | **60 min** | XGBoost | 7.277 | 11.231 | 26.955 |
| **Peak** | **60 min** | GAT | 7.940 | 11.619 | 27.820 |
| \midrule | \midrule | \midrule | \midrule | \midrule | \midrule |
| **Off-peak** | **5 min** | Persistence | 2.695 | 4.386 | 5.593 |
| **Off-peak** | **5 min** | **Random Forest** | **2.350** | **3.955** | **5.222** |
| **Off-peak** | **5 min** | XGBoost | 2.381 | 3.992 | 5.328 |
| **Off-peak** | **5 min** | GAT | 2.696 | 4.187 | 6.019 |
| **Off-peak** | **15 min** | Persistence | 3.230 | 5.877 | 7.157 |
| **Off-peak** | **15 min** | **Random Forest** | **2.812** | **5.121** | **6.750** |
| **Off-peak** | **15 min** | XGBoost | 2.913 | 5.253 | 7.089 |
| **Off-peak** | **15 min** | GAT | 3.146 | 5.249 | 7.484 |
| **Off-peak** | **30 min** | Persistence | 3.797 | 7.288 | 8.787 |
| **Off-peak** | **30 min** | **Random Forest** | **3.210** | **5.951** | **8.163** |
| **Off-peak** | **30 min** | XGBoost | 3.362 | 6.152 | 8.632 |
| **Off-peak** | **30 min** | GAT | 3.605 | 6.169 | 9.070 |
| **Off-peak** | **60 min** | Persistence | 4.770 | 9.318 | 11.338 |
| **Off-peak** | **60 min** | **Random Forest** | **3.643** | **6.756** | **9.691** |
| **Off-peak** | **60 min** | XGBoost | 3.853 | 6.927 | 10.356 |
| **Off-peak** | **60 min** | GAT | 4.131 | 7.121 | 10.929 |

---

## 3. Key Findings & Discussion
1. **Dominance of Independent Random Forests**:
   * Segment-wise Random Forests achieved the lowest MAE across every horizon. Because each sensor has an independent model, trees can optimize thresholds tailored strictly to that specific freeway stretch (e.g., carpool lanes, bottleneck merges) without negative cross-corridor gradient interference.
2. **XGBoost Scaling**:
   * Pooled XGBoost learned robust global relationships while conditioning on categorical sensor IDs, beating Persistence across all horizons (e.g. 4.542 vs 5.381 MAE at 60 min).
3. **Analysis of the Standalone GAT Gap**:
   * GAT beat persistence at 15, 30, and 60 minutes, but underperformed tree models overall.
   * Root Cause: Spatial message passing occurred over flattened static lags without a recurrent (GRU) or 1D-temporal convolution backbone, losing sequential dynamics.
   * Furthermore, the distance graph is undirected, which spreads spatial attention symmetrically upstream and downstream regardless of physical traffic flow direction.

---

## 4. Simulated Signal Policy Experiment (Optional Deliverable)

An intersection queuing simulation was executed in `notebooks/09_signal_simulation.ipynb` to evaluate dynamic signal control without claiming live traffic control. Using the Greenshields fundamental diagram ($q = k \cdot v$), arrival demands for an arterial corridor were inferred from the 15-minute speed forecasts and matched against a cross-arterial. 

Comparing a **Fixed-Time Signal Policy** (static 90s cycle, 53% / 36% split) against a **Forecast-Aware Adaptive (Max-Pressure) Signal Policy** (dynamically adjusting green splits based on anticipated queue buildup), the adaptive policy achieved a **42.1% reduction in total intersection vehicle delay** during peak congestion periods.

![Simulated Signal Policy Experiment](../results/figures/28_simulated_signal_policy_experiment.png)

## 5. Deliverables and Limitations

- AML baselines: `notebooks/05_baselines.ipynb`, `notebooks/06_xgboost.ipynb`.
- Chronological comparison: `notebooks/07_backtest.ipynb`.
- Graph construction and trained GAT: `notebooks/08_gat_model.ipynb`.
- Simulated signal policy experiment: `notebooks/09_signal_simulation.ipynb`.
- Model metrics, generated adjacency, sample predictions, and trained state are under `results/metrics/` and `results/predictions/`.
- Figures include the fixed adjacency, GAT training/forecasts, peak/off-peak comparison, and simulated signal dynamics under `results/figures/`.

Although spatial attention is implemented, the symmetrized distance graph has no verified travel direction. Real-world live traffic control is strictly not claimed; full field control requires microscopic simulators (SUMO / CityFlow) with calibrated origin-destination volumes, phase sequences, and detector actuations. The project-run scores are reported transparently and not claimed to reproduce separate external reference scores.

---

## 6. Phase 2 Innovation Roadmap: Residual Hybrid Model
To satisfy the Phase 2 Hybrid Approach and unlock additional marks:
* **Residual Spatial Hybrid**:
  1. Fit Random Forest / XGBoost to predict baseline localized traffic: $\hat{y}_{s,t}^{\text{AML}}$.
  2. Compute training residuals: $e_{s,t} = y_{s,t} - \hat{y}_{s,t}^{\text{AML}}$.
  3. Train a recurrent Graph Neural Network on the spatial network residuals to model cross-sensor error propagation.
  4. Final prediction: $\hat{y}_{s,t}^{\text{Hybrid}} = \hat{y}_{s,t}^{\text{AML}} + \hat{e}_{s,t}^{\text{GNN}}$.
