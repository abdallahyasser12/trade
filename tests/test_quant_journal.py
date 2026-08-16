"""Decision-journal adapter tests: engine decisions -> desk journal rows."""

from datetime import datetime, timezone

import pytest

from crypto_models import CryptoRegime, CryptoSignalDecision
from models import BotAction, SignalDecision
from quant.contracts import Market
from quant.journal import DecisionJournal, strategy_id_for
from quant.store import QuantStore

NOW = datetime(2026, 1, 12, tzinfo=timezone.utc)


@pytest.fixture()
def store():
    s = QuantStore(":memory:")
    yield s
    s.close()


def make_journal(store, warnings=None):
    return DecisionJournal(
        store,
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy_id="crypto_regime_breakout",
        warn=(warnings.append if warnings is not None else None),
    )


def crypto_decision(action=BotAction.BUY, **overrides):
    base = dict(
        symbol="BTC/USD",
        action=action,
        timestamp=NOW,
        reason="confirmed_regime_breakout",
        regime=CryptoRegime.BULL_TREND,
        score=1.2,
        close=50_000.0,
        atr=800.0,
        stop_price=48_000.0,
        filters={"trend": True, "volume": True},
    )
    base.update(overrides)
    return CryptoSignalDecision(**base)


def test_crypto_buy_and_hold_decisions_journal_with_reasons(store):
    journal = make_journal(store)
    hold = crypto_decision(
        action=BotAction.HOLD,
        reason="entry_filters_failed",
        filters={"trend": True, "volume": False},
        details={"failed_filters": ["volume"]},
        stop_price=None,
    )
    journal.record_crypto_decisions([crypto_decision(), hold], "session-1")
    rows = store.get_decisions()
    assert len(rows) == 2
    by_outcome = {row["outcome"]: row for row in rows}
    assert by_outcome["EXECUTE"]["action"] == "BUY"
    assert "volume" in by_outcome["REJECT"]["rejection_reasons"]
    assert by_outcome["EXECUTE"]["session_id"] == "session-1"


def test_stock_decision_with_block_reason(store):
    journal = DecisionJournal(
        store, market=Market.US_EQUITY, timeframe="1Day", strategy_id="stock_trend"
    )
    decision = SignalDecision(
        action=BotAction.BUY,
        timestamp=NOW,
        reason="bullish_crossover_confirmed",
        filters={"trend": True, "rsi": True},
        details={"close": 450.0},
    )
    journal.record_stock_decision("SPY", decision, "session-2", blocked_reason="market_closed")
    rows = store.get_decisions()
    assert rows[0]["outcome"] == "REJECT"
    assert "market_closed" in rows[0]["rejection_reasons"]


def test_exit_fill_attribution_math(store):
    journal = make_journal(store)
    journal.record_crypto_exit_fill(
        symbol="BTC/USD",
        session_id="session-1",
        qty=0.5,
        exit_price=54_000.0,
        total_fees=50.0,
        exit_reason="stop",
        entry_price=50_000.0,
        entry_time_iso=NOW.isoformat(),
        initial_stop_price=48_000.0,
        highest_close=55_000.0,
        lowest_close=49_500.0,
        entry_regime="BULL_TREND",
    )
    rows = store.get_attributions()
    assert len(rows) == 1
    row = rows[0]
    assert row["gross_pnl"] == pytest.approx((54_000 - 50_000) * 0.5)
    assert row["net_pnl"] == pytest.approx(2_000 - 50.0)
    # R = net / (risk per unit * qty) = 1950 / (2000 * 0.5)
    assert row["r_multiple"] == pytest.approx(1950.0 / 1000.0)
    assert row["mfe_pct"] == pytest.approx(10.0)
    assert row["mae_pct"] == pytest.approx(1.0)
    assert row["regime_at_entry"] == "BULL_TREND"
    assert row["shadow"] == 0


def test_journal_failures_never_raise(store):
    warnings: list[str] = []
    journal = make_journal(store, warnings)
    store.close()  # force write failures
    journal.record_crypto_decisions([crypto_decision()], "session-1")
    journal.record_crypto_exit_fill(
        symbol="BTC/USD",
        session_id=None,
        qty=1.0,
        exit_price=1.0,
        total_fees=0.0,
        exit_reason="stop",
        entry_price=1.0,
        entry_time_iso=None,
        initial_stop_price=None,
        highest_close=None,
        lowest_close=None,
        entry_regime=None,
    )
    assert len(warnings) == 2


def test_strategy_id_mapping():
    class CryptoRegimeStrategy:  # same class NAME as the production one
        pass

    assert strategy_id_for(CryptoRegimeStrategy()) == "crypto_regime_breakout"

    class SomethingNew:
        pass

    assert strategy_id_for(SomethingNew()) == "SomethingNew"
