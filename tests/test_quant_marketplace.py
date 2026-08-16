"""Marketplace rolling-statistics tests over persisted attributions."""

from datetime import datetime, timedelta, timezone

import pytest

from quant.contracts import AttributionRecord, Market, StrategyState
from quant.marketplace import (
    MIN_SAMPLE,
    compute_strategy_health,
    regime_compatibility,
)
from quant.store import QuantStore

NOW = datetime(2026, 1, 10, tzinfo=timezone.utc)
SID = "donchian_trend"


@pytest.fixture()
def store():
    s = QuantStore(":memory:")
    yield s
    s.close()


def record(store, r, regime="STRONG_BULL", shadow=False, net=None):
    store.record_attribution(
        AttributionRecord(
            strategy_id=SID,
            strategy_version="1.0.0",
            symbol="BTC/USD",
            market=Market.CRYPTO,
            timeframe="1Hour",
            session_id=None,
            signal_timestamp=NOW,
            entry_timestamp=NOW,
            exit_timestamp=NOW + timedelta(hours=4),
            qty=1.0,
            entry_price=100.0,
            exit_price=100.0 + r * 5,
            initial_stop_price=95.0,
            fees=0.5,
            expected_cost_bps=None,
            realized_slippage_bps=None,
            gross_pnl=r * 5,
            net_pnl=net if net is not None else r * 5,
            r_multiple=r,
            max_favorable_excursion_pct=max(0.0, r * 5),
            max_adverse_excursion_pct=1.0,
            entry_reason="breakout",
            exit_reason="stop",
            regime_at_entry=regime,
            shadow=shadow,
        )
    )


def test_below_min_sample_reports_none(store):
    for r in (1.0, -1.0, 0.5):
        record(store, r)
    health = compute_strategy_health(store, SID, StrategyState.SHADOW, include_shadow=True)
    assert health.trades == 3
    assert health.hit_rate is None
    assert health.expectancy_r is None
    assert health.profit_factor is None
    assert health.net_pnl == pytest.approx(0.5 * 5)


def test_statistics_with_adequate_sample(store):
    rs = [1.0, -1.0, 2.0, -1.0, 0.5, 1.5, -1.0, 1.0, -0.5, 2.5, 1.0, -1.0]
    for r in rs:
        record(store, r)
    health = compute_strategy_health(store, SID, StrategyState.SHADOW, include_shadow=True)
    assert health.trades == len(rs)
    assert health.hit_rate == pytest.approx(sum(1 for r in rs if r > 0) / len(rs))
    assert health.expectancy_r == pytest.approx(sum(rs) / len(rs))
    positive = sum(r for r in rs if r > 0)
    negative = abs(sum(r for r in rs if r <= 0))
    assert health.profit_factor == pytest.approx(positive / negative)
    assert health.max_drawdown_pct is not None and health.max_drawdown_pct > 0
    assert health.tail_loss_r is not None
    assert health.sharpe is not None and health.sortino is not None


def test_shadow_rows_excluded_when_requested(store):
    for _ in range(MIN_SAMPLE):
        record(store, 1.0, shadow=True)
    real_only = compute_strategy_health(store, SID, StrategyState.PAPER, include_shadow=False)
    assert real_only.trades == 0
    with_shadow = compute_strategy_health(store, SID, StrategyState.PAPER, include_shadow=True)
    assert with_shadow.trades == MIN_SAMPLE


def test_regime_buckets_respect_floor(store):
    for _ in range(6):
        record(store, 1.0, regime="STRONG_BULL")
    for _ in range(3):
        record(store, -1.0, regime="RANGE")
    for _ in range(4):
        record(store, 0.5, regime="STRONG_BULL")
    health = compute_strategy_health(store, SID, StrategyState.SHADOW, include_shadow=True)
    assert "STRONG_BULL" in health.regime_performance
    # RANGE has only 3 trades: below the 5-trade floor, excluded not shown.
    assert "RANGE" not in health.regime_performance


def test_regime_compatibility_monotone(store):
    for _ in range(6):
        record(store, 1.5, regime="STRONG_BULL")
    for _ in range(6):
        record(store, -1.0, regime="WEAK_BEAR")
    health = compute_strategy_health(store, SID, StrategyState.SHADOW, include_shadow=True)
    good = regime_compatibility(health, "STRONG_BULL")
    bad = regime_compatibility(health, "WEAK_BEAR")
    assert good is not None and bad is not None
    assert good > 0.9
    assert bad < 0.1
    assert regime_compatibility(health, "RANGE") is None
