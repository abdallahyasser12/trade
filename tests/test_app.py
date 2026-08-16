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
    engine_radio = next(radio for radio in page.radio if radio.label == "Decision engine")
    # Cross-sectional is the default: it is the only engine with positive
    # out-of-sample expectancy across eleven walk-forward experiments, while
    # the regime ensemble it replaced as default is the worst-measured of the
    # three (Sharpe -1.80).
    assert engine_radio.value == "Cross-sectional momentum (default)"


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
    account_mode = next(
        radio for radio in page.radio if radio.label == "Alpaca crypto account"
    )
    account_mode.set_value("Live").run()
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


def test_the_crypto_decision_engine_radio_offers_all_three_engines():
    """Cross-sectional must be reachable from the dashboard, not research-only."""

    import app as app_module

    source = Path(app_module.__file__).read_text()
    for label in (
        "Cross-sectional momentum (default)",
        "Regime ensemble",
        "Base breakout/trend",
    ):
        assert label in source, f"decision-engine radio is missing {label!r}"


def test_each_engine_selection_reaches_its_own_live_factory():
    """A radio that does not change the engine is worse than no radio."""

    import app as app_module

    source = Path(app_module.__file__).read_text()
    # Live/paper trading
    assert "make_regime_aware_crypto_engine(" in source
    assert "make_cross_sectional_crypto_engine(" in source
    # Backtest
    assert "run_regime_ensemble_backtest(" in source
    assert "run_cross_sectional_backtest(" in source
    # Walk-forward research
    assert "run_regime_ensemble_walk_forward(" in source
    assert "run_cross_sectional_walk_forward(" in source


def test_the_engine_choice_is_part_of_the_cache_fingerprint():
    """Otherwise cached research silently mixes engines."""

    import app as app_module
    from crypto_config import (
        CryptoBacktestConfig,
        CryptoEngineConfig,
        CryptoRiskConfig,
        CryptoStrategyConfig,
    )

    creds = type("C", (), {"fingerprint": "abc"})()
    args = (
        creds,
        CryptoStrategyConfig(),
        CryptoRiskConfig(),
        CryptoEngineConfig(),
        CryptoBacktestConfig(),
    )
    prints = {
        app_module._crypto_context_fingerprint(*args, decision_engine=name)
        for name in ("base", "ensemble", "cross")
    }
    assert len(prints) == 3, "engines share a cache key; results would be mixed"


def test_the_cross_engine_research_grid_searches_what_it_reads():
    """The same no-op-search defect the CLI had must not exist in the app."""

    import app as app_module

    source = Path(app_module.__file__).read_text()
    marker = source.index('grid = {')
    block = source[marker - 600:marker + 900]
    assert '"cross": [' in block
    assert "CrossSectionalConfig(" in block
    assert "require_benchmark_bull=gated" in block
    # And the breakout keys the cross engine never reads stay on the other branch.
    breakout_grid = source[source.index('"breakout_period": sorted('):]
    assert "min_adx" in breakout_grid[:200]
