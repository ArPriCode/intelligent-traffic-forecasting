"""Tests for Streamlit dashboard rendering and interaction."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "traffic_prediction" / "web_app.py")


def test_web_app_full_render() -> None:
    """Test that the Streamlit dashboard renders all tabs and metrics without error."""
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    assert not app.exception, f"App crashed with exceptions: {app.exception}"
    assert not app.error, f"App rendered with errors: {app.error}"

    # Verify header and title
    assert len(app.title) >= 1
    assert "Intelligent Traffic Speed Forecasting" in app.title[0].value

    # Verify metrics
    metric_labels = [m.label for m in app.metric]
    assert "Sensors (METR-LA)" in metric_labels
    assert "Fixed Total Delay" in metric_labels
    assert "Adaptive Total Delay" in metric_labels
    assert "Delay Reduction" in metric_labels

    # Verify interactive components
    assert len(app.tabs) == 4
    assert len(app.radio) >= 1
    assert len(app.slider) >= 1
    assert len(app.multiselect) >= 1


def test_web_app_interactions() -> None:
    """Test switching tabs, radio buttons, and timeline slider."""
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    # Toggle radio to PEMS-BAY
    if app.radio:
        app.radio[0].set_value("PEMS-BAY").run()
        assert not app.exception
        assert not app.error

    # Change timeline slider
    if app.slider:
        app.slider[0].set_value(72).run()
        assert not app.exception
        assert not app.error
