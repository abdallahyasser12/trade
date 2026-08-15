"""Credential isolation and immutable crypto configuration tests."""

from dataclasses import FrozenInstanceError

import pytest

from config import AccountMode, live_trading_enabled, load_credentials
from crypto_config import (
    CryptoEngineConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
    crypto_bars_per_year,
    validate_crypto_symbol,
)


def test_live_credentials_never_fall_back_to_paper(monkeypatch):
    # Blank rather than delete: the namespaced names take precedence over the
    # legacy ones, and `load_credentials` calls `load_dotenv`, which would
    # repopulate a deleted variable from a local .env. Setting them empty keeps
    # this hermetic on a machine that has real credentials exported.
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "")
    monkeypatch.setenv("ALPACA_API_KEY", "legacy-paper")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "legacy-secret")
    monkeypatch.setenv("ALPACA_LIVE_API_KEY", "")
    monkeypatch.setenv("ALPACA_LIVE_SECRET_KEY", "")
    live = load_credentials(account_mode=AccountMode.LIVE)
    paper = load_credentials(account_mode=AccountMode.PAPER)
    assert live.is_configured is False
    assert paper.api_key == "legacy-paper"
    assert live.fingerprint != paper.fingerprint


@pytest.mark.parametrize("raw, enabled", [("true", True), ("YES", True), ("1", True), ("on", True), ("typo", False), ("", False)])
def test_live_feature_flag_is_strict(monkeypatch, raw, enabled):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", raw)
    assert live_trading_enabled() is enabled


@pytest.mark.parametrize(
    "symbol, valid",
    [("BTC/USD", True), ("ETH/USD", True), ("btc/usd", False), ("BTCUSDT", False), ("USD/USD", False)],
)
def test_crypto_symbol_validation(symbol, valid):
    assert (validate_crypto_symbol(symbol) is None) is valid


def test_crypto_config_is_frozen_and_benchmark_must_be_in_universe():
    config = CryptoStrategyConfig(symbols=("ETH/USD",))
    assert any("Benchmark symbol" in error for error in config.validate())
    with pytest.raises(FrozenInstanceError):
        config.timeframe = "1Day"


def test_live_exposure_cap_cannot_exceed_paper_cap():
    config = CryptoRiskConfig(
        max_total_exposure_pct=5.0,
        max_position_pct=5.0,
        max_live_total_exposure_pct=10.0,
    )
    assert any("Live total exposure" in error for error in config.validate())


def test_live_caps_have_absolute_code_ceilings():
    config = CryptoRiskConfig(
        max_live_order_notional=10_001,
        max_live_total_exposure_pct=21,
        max_total_exposure_pct=30,
    )
    errors = config.validate()
    assert any("$10,000" in error for error in errors)
    assert any("20%" in error for error in errors)


def test_crypto_timeframes_annualize_continuous_market():
    assert crypto_bars_per_year("1Hour") == pytest.approx(365.25 * 24)
    assert CryptoEngineConfig(account_mode=AccountMode.PAPER).validate() == []
