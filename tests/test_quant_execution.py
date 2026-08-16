"""Execution-quality gate tests: measured-cost folding and fail-closed rules."""

import math

import pytest

from quant.contracts import Market
from quant.execution import ExecutionQualityConfig, ExecutionQualityEngine
from state_store import StateStore


@pytest.fixture()
def store_with_bad_fills():
    store = StateStore(":memory:")
    # 25 sell fills each 30 bps worse than reference (100 -> 99.7).
    for i in range(25):
        store.record_execution(
            "s1",
            f"cid-{i}",
            "BTC/USD",
            "exit",
            "sell",
            reference_price=100.0,
            fill_price=99.7,
            filled_qty=1.0,
            event_key=f"evt-{i}",
        )
    yield store
    store.close()


def test_measured_worse_than_assumed_tightens(store_with_bad_fills):
    engine = ExecutionQualityEngine(
        ExecutionQualityConfig(assumed_slippage_bps=10.0, min_measured_samples=20),
        state_store=store_with_bad_fills,
    )
    slippage, source, samples = engine.slippage_estimate()
    assert source == "measured"
    assert samples == 25
    assert slippage == pytest.approx(30.0, abs=0.1)


def test_measured_better_than_assumed_never_loosens():
    store = StateStore(":memory:")
    for i in range(25):
        store.record_execution(
            "s1", f"cid-{i}", "BTC/USD", "exit", "sell",
            reference_price=100.0, fill_price=99.999, filled_qty=1.0,
            event_key=f"evt-{i}",
        )
    engine = ExecutionQualityEngine(
        ExecutionQualityConfig(assumed_slippage_bps=10.0, min_measured_samples=20),
        state_store=store,
    )
    slippage, source, _ = engine.slippage_estimate()
    # Measured ~0.1 bps is better than assumed 10; the gate stays at 10.
    assert slippage == pytest.approx(10.0)
    assert source == "measured"
    store.close()


def test_sample_threshold_falls_back_to_assumed():
    store = StateStore(":memory:")
    store.record_execution(
        "s1", "cid", "BTC/USD", "exit", "sell",
        reference_price=100.0, fill_price=99.0, filled_qty=1.0, event_key="e",
    )
    engine = ExecutionQualityEngine(
        ExecutionQualityConfig(min_measured_samples=20), state_store=store
    )
    slippage, source, samples = engine.slippage_estimate()
    assert source == "assumed"
    assert samples == 1
    assert slippage == pytest.approx(10.0)
    store.close()


def test_wide_spread_rejected():
    engine = ExecutionQualityEngine(ExecutionQualityConfig(max_spread_bps=40.0))
    estimate = engine.estimate(
        "AAPL", Market.US_EQUITY, bid=100.0, ask=100.5, quote_age_seconds=1.0
    )
    assert not estimate.viable
    assert estimate.reason == "spread_above_maximum"


def test_high_participation_rejected():
    engine = ExecutionQualityEngine(ExecutionQualityConfig(max_participation_pct=2.0))
    estimate = engine.estimate(
        "AAPL",
        Market.US_EQUITY,
        bid=100.0,
        ask=100.02,
        quote_age_seconds=1.0,
        notional=50_000.0,
        avg_bar_dollar_volume=1_000_000.0,
    )
    assert estimate.participation_pct == pytest.approx(5.0)
    assert not estimate.viable
    assert estimate.reason == "participation_above_maximum"


def test_stale_quote_falls_back_to_assumed_spread():
    engine = ExecutionQualityEngine(
        ExecutionQualityConfig(quote_max_age_seconds=120.0, equity_assumed_spread_bps=5.0)
    )
    estimate = engine.estimate(
        "AAPL", Market.US_EQUITY, bid=100.0, ask=100.5, quote_age_seconds=999.0
    )
    assert estimate.spread_bps == pytest.approx(5.0)
    assert "assumed_spread" in estimate.cost_source


def test_market_specific_fees():
    engine = ExecutionQualityEngine()
    crypto = engine.estimate("BTC/USD", Market.CRYPTO)
    equity = engine.estimate("AAPL", Market.US_EQUITY)
    assert crypto.fee_bps == pytest.approx(25.0)
    assert equity.fee_bps == pytest.approx(0.0)
    assert crypto.round_trip_bps > equity.round_trip_bps


def test_net_alpha_gate_arithmetic():
    engine = ExecutionQualityEngine(
        ExecutionQualityConfig(uncertainty_margin_bps=5.0)
    )
    estimate = engine.estimate(
        "AAPL", Market.US_EQUITY, bid=100.0, ask=100.02, quote_age_seconds=1.0
    )
    tradeable, net = engine.net_alpha_after_costs(50.0, estimate)
    assert net == pytest.approx(50.0 - estimate.round_trip_bps - 5.0)
    assert tradeable == (net > 0)
    blocked, worse = engine.net_alpha_after_costs(estimate.round_trip_bps, estimate)
    assert not blocked
    assert worse < 0


def test_nan_gross_alpha_fails_closed():
    engine = ExecutionQualityEngine()
    estimate = engine.estimate("AAPL", Market.US_EQUITY)
    tradeable, net = engine.net_alpha_after_costs(float("nan"), estimate)
    assert not tradeable
    assert net == -math.inf


def test_config_validation():
    assert ExecutionQualityConfig(max_spread_bps=0.0).validate()
    assert ExecutionQualityConfig(min_measured_samples=0).validate()
    assert not ExecutionQualityConfig().validate()
