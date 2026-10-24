# Traffic Datasets & Network Graph Directory

This directory stores the traffic observation time-series, road attributes, sensor metadata, and highway distance graphs used for model training, feature extraction, and graph attention learning.

---

## Directory Inventory

```text
data/
├── METR-LA.csv                      # Primary speed benchmark time-series (Los Angeles)
├── PEMS-BAY.csv                     # Extended speed benchmark time-series (Bay Area)
├── PEMS-BAY-META.csv                # Highway attributes, lane counts, and GPS coordinates
└── sensor_graph/
    ├── distances_la_2012.csv        # Pairwise road distances between sensors
    ├── graph_sensor_ids.txt         # Official ordered list of 207 METR-LA sensor IDs
    └── graph_sensor_locations.csv   # Latitude and longitude coordinates for METR-LA sensors
```

---

## 1. Dataset Descriptions & Schema

### `METR-LA.csv` (Primary Forecasting Benchmark)
* **Domain:** Highway system of Los Angeles County (loop detector network).
* **Temporal Horizon:** March 1, 2012 to June 27, 2012 (approx. 4 months).
* **Sampling Rate:** Continuous 5-minute intervals ($12 \text{ records/hour} \times 24 \text{ hours/day} = 288 \text{ steps/day}$).
* **Dimensions:** 34,272 timestamps × 207 road sensors ($7,094,304$ total speed readings).
* **Variable:** Speed in miles per hour (mph).
* **Missing & Zero Rate:** $8.11\%$ of total values are zero or unobserved (treated as sensor outages, causally forward-filled for inputs, and masked out from target loss and evaluation metrics).

### `PEMS-BAY.csv` (Extended California Network)
* **Domain:** California Department of Transportation (Caltrans) Performance Measurement System (PeMS) in the San Francisco Bay Area.
* **Temporal Horizon:** January 1, 2017 to June 30, 2017 (6 months).
* **Sampling Rate:** Continuous 5-minute intervals ($288 \text{ steps/day}$).
* **Dimensions:** 52,116 timestamps × 325 road sensors ($16,937,700$ total speed readings).
* **Missing & Zero Rate:** $0.0031\%$ missing readings.

### `PEMS-BAY-META.csv` (Sensor Metadata)
Contains spatial and structural road characteristics for Bay Area loop detectors:
* `sensor_id`: Unique detector identifier (aligned with `PEMS-BAY.csv` columns).
* `Fwy`: Freeway number (e.g., US-101, I-880, I-280, SR-85).
* `Dir`: Direction of travel (`N`, `S`, `E`, `W`).
* `Lanes`: Number of travel lanes at sensor location (ranging from 1 to 5).
* `Latitude` & `Longitude`: Exact WGS84 geographic coordinates.

### `sensor_graph/` (METR-LA Highway Network)
Sourced from the canonical DCRNN (Diffusion Convolutional Recurrent Neural Network) benchmark:
* **`distances_la_2012.csv`**: Contains `[from, to, cost]` triples representing network road distance (in meters) between sensors.
* **`graph_sensor_ids.txt`**: List of all 207 sensor IDs matching the exact column order of `METR-LA.csv`.
* **`graph_sensor_locations.csv`**: GPS coordinates for network mapping and distance verification.

---

## 2. Dataset Attribution & Download Links

The METR-LA and PEMS-BAY benchmark datasets are distributed in public academic repositories:
* **Zenodo Repository:** [Zenodo Traffic Benchmark Record 5146275](https://zenodo.org/records/5146275)
* **DCRNN Canonical Repository:** [GitHub DCRNN (commit 602afd9d767d3aa1c9b3eac51710d6aeee12c227)](https://github.com/liyaguang/DCRNN)

If downloading fresh copies, extract `METR-LA.csv` and `PEMS-BAY.csv` directly into this `data/` directory.

---

## 3. Important Methodological Caveats

1. **Speed Only (No Flow or Occupancy):** The public benchmark CSVs record traffic speed exclusively. Traffic flow (vehicles/hour) and lane occupancy (%) are not present in these files and are strictly not fabricated.
2. **Symmetrized Distance Graph:** The distance table between highway loop detectors is converted to an undirected adjacency using a thresholded Gaussian distance kernel. Verified upstream vs. downstream travel direction is not explicitly encoded in the raw distance table.
3. **Outage Handling:** Zero speed measurements indicate hardware dropouts or transmission failures rather than gridlock at 0.00 mph. They are causally imputed for lag generation, but masked from all evaluation metrics.
