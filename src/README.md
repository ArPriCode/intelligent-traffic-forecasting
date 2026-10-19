# Source package

Reusable application code is organized under `traffic_prediction/`:

```text
src/
├── README.md
└── traffic_prediction/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── graph.py
    ├── metrics.py
    ├── models.py
    └── web_app.py
```

The package contains data preparation, feature engineering, graph construction, metrics, the graph-attention model, and the local Streamlit dashboard. Install it from the project root with `python -m pip install -e .`; launch the dashboard with `streamlit run src/traffic_prediction/web_app.py`.
