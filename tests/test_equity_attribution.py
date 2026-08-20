from datetime import datetime, timezone

from equity_marketwide_deployment import ManifestLockedMarketWideEquityTradeEngine
from models import OrderRecord
from quant.contracts import AttributionRecord, Market
from quant.equity_attribution import record_equity_attribution_once
from quant.store import QuantStore

NOW = datetime(2026, 8, 18, 15, 0, tzinfo=timezone.utc)


def _attribution():
    return AttributionRecord(
        strategy_id="opening_range_breakout",
        strategy_version="1.0.0",
        symbol="AAPL",
        market=Market.US_EQUITY,
        timeframe="15Min",
        session_id="session-1",
        signal_timestamp=NOW,
        entry_timestamp=NOW,
        exit_timestamp=NOW,
        qty=5.0,
        entry_price=100.0,
        exit_price=102.0,
        initial_stop_price=98.0,
        fees=0.0,
        expected_cost_bps=None,
        realized_slippage_bps=1.0,
        gross_pnl=10.0,
        net_pnl=10.0,
        r_multiple=1.0,
        max_favorable_excursion_pct=2.5,
        max_adverse_excursion_pct=0.5,
        entry_reason="opening_range_breakout",
        exit_reason="strategy_exit",
        shadow=False,
    )


def test_real_attribution_is_exactly_once_by_fill_event():
    store = QuantStore(":memory:")
    try:
        record = _attribution()
        assert record_equity_attribution_once(store, record, event_key="order-1:5") is True
        assert record_equity_attribution_once(store, record, event_key="order-1:5") is False
        rows = store.get_attributions(strategy_id="opening_range_breakout")
        assert len(rows) == 1
        assert rows[0]["r_multiple"] == 1.0
        assert rows[0]["net_pnl"] == 10.0
    finally:
        store.close()


def test_partial_fill_price_is_reconstructed_from_cumulative_averages():
    previous = OrderRecord(
        broker_order_id="1",
        client_order_id="entry-1",
        symbol="AAPL",
        purpose="entry",
        side="buy",
        requested_qty=10.0,
        filled_qty=4.0,
        status="partially_filled",
        fill_price=100.0,
        submitted_at=NOW,
        updated_at=NOW,
    )
    current = OrderRecord(
        broker_order_id="1",
        client_order_id="entry-1",
        symbol="AAPL",
        purpose="entry",
        side="buy",
        requested_qty=10.0,
        filled_qty=10.0,
        status="filled",
        fill_price=101.2,
        submitted_at=NOW,
        updated_at=NOW,
    )
    # cumulative notional = 1012; previous notional = 400; remaining 6 shares = 102
    assert ManifestLockedMarketWideEquityTradeEngine._incremental_fill_price(
        current, previous
    ) == 102.0
