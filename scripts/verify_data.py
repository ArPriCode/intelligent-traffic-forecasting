"""Dataset verification script for METR-LA and PEMS-BAY observations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GRAPH_DIR = DATA_DIR / "sensor_graph"


def check_graph_files() -> bool:
    """Verify official sensor graph files."""
    print("=" * 60)
    print("Checking Sensor Graph Directory:")
    print("=" * 60)
    required_files = [
        "distances_la_2012.csv",
        "graph_sensor_ids.txt",
        "graph_sensor_locations.csv",
    ]
    all_ok = True
    for fname in required_files:
        path = GRAPH_DIR / fname
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"  [OK] {fname:<30} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {fname:<30}")
            all_ok = False
    return all_ok


def check_traffic_csv(
    filename: str, expected_sensors: int | None = None
) -> bool:
    """Check and inspect a traffic CSV file."""
    path = DATA_DIR / filename
    if not path.is_file():
        print(f"  [MISSING] {filename}")
        print(f"            Expected path: {path}")
        print("            Please provide this file in the data/ folder.")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [FOUND]   {filename} ({size_mb:.1f} MB)")
    try:
        df = pd.read_csv(path, index_col=0, nrows=5)
        n_sensors = df.shape[1]
        print(f"            Found {n_sensors} sensor columns in sample.")
        if expected_sensors and n_sensors != expected_sensors:
            print(
                f"            [WARNING] Expected {expected_sensors} sensors, "
                f"found {n_sensors}."
            )
        return True
    except Exception as exc:
        print(f"            [ERROR] Failed to read CSV: {exc}")
        return False


def main() -> int:
    print("\nIntelligent Traffic Forecasting — Dataset Verification\n")
    graph_ok = check_graph_files()

    print("\n" + "=" * 60)
    print("Checking Traffic CSV Files:")
    print("=" * 60)
    la_ok = check_traffic_csv("METR-LA.csv", expected_sensors=207)
    bay_ok = check_traffic_csv("PEMS-BAY.csv", expected_sensors=325)
    meta_ok = check_traffic_csv("PEMS-BAY-META.csv")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    if graph_ok and la_ok and bay_ok and meta_ok:
        print("All required datasets and graph files are VERIFIED.")
        return 0
    else:
        print("Some traffic CSV files are not present in data/ folder.")
        print("As per instructions, these large CSVs are kept out of git.")
        print("Place METR-LA.csv, PEMS-BAY.csv, and PEMS-BAY-META.csv")
        print("into the 'data/' folder.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
