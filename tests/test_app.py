"""Streamlit smoke tests for safe empty-state rendering."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_env_and_keeps_start_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET_KEY", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    page = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert list(page.exception) == []
    assert page.title[0].value == "📈 Trend & Crypto Bot — Alpaca"
    visible = [element.value for element in page.error]
    assert visible.count("PAPER TRADING — NO REAL MONEY") >= 2
    assert any("paper credentials are not configured" in w.value.lower() for w in page.warning)
    start = next(button for button in page.button if button.label == "Start Bot")
    assert start.disabled is True
    crypto_start = next(button for button in page.button if button.label == "Start Crypto Bot")
    assert crypto_start.disabled is True
    account_mode = next(radio for radio in page.radio if radio.label == "Alpaca crypto account")
    assert account_mode.value == "Paper"
    assert account_mode.options == ["Paper", "Live"]


def test_live_switch_is_fail_closed_when_deployment_flag_is_absent(tmp_path, monkeypatch):
    for name in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_PAPER_API_KEY",
        "ALPACA_PAPER_SECRET_KEY",
        "ALPACA_LIVE_API_KEY",
        "ALPACA_LIVE_SECRET_KEY",
        "ENABLE_LIVE_TRADING",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    page = AppTest.from_file(str(app_path), default_timeout=20).run()
    page.radio[0].set_value("Live").run()
    assert list(page.exception) == []
    assert any("LIVE ACCOUNT" in error.value for error in page.error)
    unlock = next(button for button in page.button if button.label == "Unlock LIVE Trading")
    start = next(button for button in page.button if button.label == "Start Crypto Bot")
    verify = next(
        button for button in page.button if button.label == "Verify Crypto Account & Assets"
    )
    assert unlock.disabled is True
    assert verify.disabled is True
    assert start.disabled is True
