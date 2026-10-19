# Data directory

The large traffic observation CSVs are kept out of GitHub and must be supplied locally:

```text
data/
├── METR-LA.csv
├── PEMS-BAY.csv
├── PEMS-BAY-META.csv
└── sensor_graph/
    ├── distances_la_2012.csv
    ├── graph_sensor_ids.txt
    └── graph_sensor_locations.csv
```

The three traffic CSVs are ignored by Git to avoid committing large dataset files. Place them directly in this `data/` directory before running the notebooks or browser dashboard.

`sensor_graph/` contains the METR-LA road-distance data and sensor metadata used by notebook 08. Its sensor ID order matches the columns in `METR-LA.csv`. The graph files are from the pinned DCRNN revision documented in the project README.
