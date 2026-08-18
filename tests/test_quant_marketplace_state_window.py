from datetime import datetime, timedelta, timezone

from quant.contracts import AttributionRecord, Market, StrategyState
from quant.marketplace import compute_strategy_health
from quant.store import QuantStore


def record(store, when, r):
    # record_attribution sets recorded_at to current time, so use direct SQL only in
    # this storage-level regression to make two distinct lifecycle windows.
    store.record_attribution(
        AttributionRecord(
            strategy_id="s",
            strategy_version="1",
            symbol="BTC/USD",
            market=Market.CRYPTO,
            timeframe="5Min",
            session_id=None,
            signal_timestamp=when,
            entry_timestamp=when,
            exit_timestamp=when,
            qty=1,
            entry_price=100,
            exit_price=101,
            initial_stop_price=99,
            fees=0,
            expected_cost_bps=None,
            realized_slippage_bps=None,
            gross_pnl=r,
            net_pnl=r,
            r_multiple=r,
            max_favorable_excursion_pct=1,
            max_adverse_excursion_pct=0.5,
            shadow=False,
        )
    )


def test_since_window_excludes_prior_lifecycle_rows():
    store = QuantStore(":memory:")
    try:
        now = datetime.now(timezone.utc)
        for _ in range(20):
            record(store, now - timedelta(days=1), 0.5)
        # The rows were physically recorded just now. A future transition boundary
        # must therefore exclude all of them from the next lifecycle rung.
        boundary = datetime.now(timezone.utc) + timedelta(seconds=1)
        health = compute_strategy_health(
            store,
            "s",
            StrategyState.LIVE_CANARY,
            include_shadow=False,
            since=boundary,
        )
        assert health.trades == 0
        assert health.expectancy_r is None
    finally:
        store.close()
