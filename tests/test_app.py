"""Streamlit smoke tests for safe empty-state rendering."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_env_and_keeps_start_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    page = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert list(page.exception) == []
    assert page.title[0].value == "📈 Trend Bot — Alpaca Paper Trading"
    visible = [element.value for element in page.error]
    assert visible.count("PAPER TRADING — NO REAL MONEY") >= 2
    assert any("paper credentials are not configured" in w.value.lower() for w in page.warning)
    start = next(button for button in page.button if button.label == "Start Bot")
    assert start.disabled is True
