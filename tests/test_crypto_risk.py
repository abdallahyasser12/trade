"""Fractional risk sizing, concentration, and irreversible loss-latch tests."""

import pandas as pd
import pytest

from config import AccountMode
from crypto_config import CryptoRiskConfig
from crypto_risk import CryptoRiskManager, floor_to_increment
from models import AssetInfo


ASSET = AssetInfo(
    symbol="BTC/USD",
    name="Bitcoin",
    tradable=True,
    active=True,
    fractionable=True,
    asset_class="crypto",
    min_order_size=0.000001,
    min_trade_increment=0.000001,
    price_increment=0.01,
    marginable=False,
    shortable=False,
)


def size(manager, **overrides):
    values = dict(
        equity=100_000.0,
        cash=100_000.0,
        buying_power=100_000.0,
        entry_price=100.0,
        stop_price=95.0,
        asset=ASSET,
        current_crypto_exposure=0.0,
        current_position_count=0,
        peak_equity=100_000.0,
        atr_percentile=50.0,
        account_mode=AccountMode.PAPER,
    )
    values.update(overrides)
    return manager.size_position(**values)


def test_fractional_floor_never_rounds_risk_up():
    assert floor_to_increment(1.23456789, 0.001) == pytest.approx(1.234)
    manager = CryptoRiskManager(CryptoRiskConfig())
    result = size(manager)
    assert result.approved
    assert result.qty * result.risk_per_unit <= result.risk_budget + 1e-6


@pytest.mark.parametrize(
    "overrides",
    [
        {"entry_price": 0.0},
        {"entry_price": -1.0},
        {"stop_price": 100.0},
        {"stop_price": 101.0},
        {"cash": 0.0},
        {"buying_power": 0.0},
        {"entry_price": float("nan")},
    ],
)
def test_invalid_or_unfunded_sizing_never_approves(overrides):
    result = size(CryptoRiskManager(CryptoRiskConfig()), **overrides)
    assert not result.approved
    assert result.qty == 0


def test_enormous_atr_and_expensive_asset_can_produce_safe_fractional_size():
    result = size(
        CryptoRiskManager(CryptoRiskConfig(min_order_notional=1.0)),
        entry_price=100_000.0,
        stop_price=50_000.0,
    )
    assert result.approved
    assert result.qty < 0.01


def test_remaining_portfolio_exposure_is_a_hard_cap():
    manager = CryptoRiskManager(CryptoRiskConfig())
    result = size(manager, current_crypto_exposure=30_000.0)
    assert not result.approved
    assert "zero quantity" in result.rejection_reason


def test_live_mode_applies_stricter_order_and_total_exposure_caps():
    manager = CryptoRiskManager(
        CryptoRiskConfig(max_live_order_notional=100.0, max_live_total_exposure_pct=5.0)
    )
    result = size(manager, account_mode=AccountMode.LIVE)
    assert result.approved
    assert result.position_notional <= 100.0 + 1e-6


def test_session_loss_halts_at_exact_threshold_and_never_reactivates():
    manager = CryptoRiskManager(CryptoRiskConfig(max_session_loss_pct=3.0))
    assert manager.check_session_loss(100_000.0, 97_000.0) is True
    assert manager.session_halted
    assert manager.check_session_loss(100_000.0, 101_000.0) is True
    assert not size(manager).approved


def test_correlation_gate_rejects_highly_correlated_addition():
    manager = CryptoRiskManager(CryptoRiskConfig(correlation_lookback=20))
    index = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    candidate = pd.Series([index % 3 / 100 for index in range(30)], index=index)
    held = candidate * 1.1
    allowed, reason, correlation = manager.correlation_gate(candidate, {"ETH/USD": held})
    assert not allowed
    assert correlation == pytest.approx(1.0)
    assert "exceeds" in reason


def test_correlation_gate_fails_closed_on_insufficient_history():
    manager = CryptoRiskManager(CryptoRiskConfig(correlation_lookback=120))
    index = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    returns = pd.Series(range(5), index=index, dtype=float)
    allowed, reason, correlation = manager.correlation_gate(returns, {"ETH/USD": returns})
    assert not allowed
    assert correlation is None
    assert "Insufficient" in reason
