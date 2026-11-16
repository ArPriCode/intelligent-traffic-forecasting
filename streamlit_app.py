"""Entrypoint for Streamlit Community Cloud deployment."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ package is discoverable on Streamlit Community Cloud without editable install
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_prediction.web_app import run_dashboard

if __name__ == "__main__":
    run_dashboard()
