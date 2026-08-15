"""Order idempotency, partial-fill protection, restart, and watchdog tests."""

from datetime import timedelta

import pytest

from broker import BrokerError
from config import AccountMode
from crypto_config import CryptoEngineConfig, CryptoRiskConfig
from crypto_engine import CRYPTO_KILL_SWITCH_KEY, CryptoTradeEngine
from crypto_models import CryptoPositionState, CryptoQuote, CryptoSignalDecision
from models import (
    AccountInfo,
    AssetInfo,
    BotAction,
    BrokerPosition,
    OrderRecord,
)
from state_store import StateStore
from tests.crypto_helpers import breakout_bars, compact_crypto_config


def asset(symbol="BTC/USD"):
    return AssetInfo(
        symbol=symbol,
        name=symbol,
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


class FakeBroker:
    account_mode = AccountMode.PAPER

    def __init__(self):
        self.account = AccountInfo(100_000, 100_000, 100_000, "ACTIVE", "USD", False, 0, False)
        self.positions = {}
        self.orders = {}
        self.market_calls = []
        self.stop_calls = []
        self.cancel_calls = []
        self.timeout_after_acceptance = False
        self.hidden_lookups_after_timeout = 0
        self._hidden_lookups = {}
        self.cancel_position_qty = None
        self._counter = 0

    def verify_account(self):
        return self.account

    def get_account(self):
        return self.account

    def verify_crypto_assets(self, symbols):
        return {symbol: asset(symbol) for symbol in symbols}

    def list_positions(self):
        return list(self.positions.values())

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def list_open_orders(self, symbol=None):
        return [
            order
            for order in self.orders.values()
            if order.is_open and (symbol is None or order.symbol == symbol)
        ]

    def get_order_by_client_id(self, client_order_id):
        remaining = self._hidden_lookups.get(client_order_id, 0)
        if remaining > 0:
            self._hidden_lookups[client_order_id] = remaining - 1
            return None
        return self.orders.get(client_order_id)

    def get_order(self, broker_order_id):
        return next(order for order in self.orders.values() if order.broker_order_id == broker_order_id)

    def submit_crypto_market_order(self, symbol, qty, side, client_order_id, **kwargs):
        self._counter += 1
        purpose = client_order_id.split("-")[-2]
        order = OrderRecord(
            f"market-{self._counter}",
            client_order_id,
            symbol,
            purpose,
            side,
            qty,
            0.0,
            "new",
        )
        self.orders[client_order_id] = order
        self.market_calls.append((symbol, qty, side, client_order_id))
        if self.timeout_after_acceptance:
            self.timeout_after_acceptance = False
            if self.hidden_lookups_after_timeout:
                self._hidden_lookups[client_order_id] = self.hidden_lookups_after_timeout
            raise BrokerError("simulated timeout after acceptance")
        return order

    def submit_crypto_stop_limit_order(
        self, symbol, qty, stop_price, limit_price, client_order_id, **kwargs
    ):
        self._counter += 1
        order = OrderRecord(
            f"stop-{self._counter}",
            client_order_id,
            symbol,
            "stop",
            "sell",
            qty,
            0.0,
            "new",
            stop_price=stop_price,
            limit_price=limit_price,
        )
        self.orders[client_order_id] = order
        self.stop_calls.append((symbol, qty, stop_price, limit_price, client_order_id))
        return order

    def cancel_order(self, broker_order_id):
        order = self.get_order(broker_order_id)
        self.cancel_calls.append(broker_order_id)
        order.status = "canceled"
        if self.cancel_position_qty is not None:
            current = self.positions.get(order.symbol)
            if self.cancel_position_qty <= 0:
                self.positions.pop(order.symbol, None)
            elif current is not None:
                current.qty = self.cancel_position_qty


class FakeFeed:
    def __init__(self, bars, now):
        self.bars = bars
        self.now = now
        self.event = None

    def set_interrupt_event(self, event):
        self.event = event

    def get_recent_bars(self, symbol, *args, **kwargs):
        return self.bars[symbol]

    def get_latest_quotes(self, symbols):
        return {
            symbol: CryptoQuote(symbol, self.now, 112.4, 112.6)
            for symbol in symbols
        }


def make_engine(tmp_path, bars=None, broker=None, engine_config=None):
    frame = bars if bars is not None else breakout_bars(periods=31)
    now = frame.index[-1].to_pydatetime() + timedelta(hours=1, seconds=10)
    broker = broker or FakeBroker()
    store = StateStore(str(tmp_path / "crypto.db"))
    feed = FakeFeed({"BTC/USD": frame}, now)
    engine = CryptoTradeEngine(
        broker,
        feed,
        store,
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        engine_config or CryptoEngineConfig(poll_interval_seconds=5),
        now_fn=lambda: now,
    )
    return engine, broker, store, now


def test_duplicate_bar_and_timeout_after_acceptance_submit_one_economic_entry(tmp_path):
    engine, broker, store, _ = make_engine(tmp_path)
    broker.timeout_after_acceptance = True
    engine._startup()
    engine.run_iteration()
    engine.run_iteration()
    buys = [call for call in broker.market_calls if call[2] == "buy"]
    assert len(buys) == 1
    assert store.is_bar_processed("BTC/USD", "1Hour", breakout_bars(periods=31).index[-1])
    stored = store.get_order_by_client_id(buys[0][3])
    assert stored.status == "new"


def test_partial_entry_fill_is_protected_then_stop_resizes_to_new_broker_qty(tmp_path):
    engine, broker, store, now = make_engine(tmp_path)
    engine._assets = {"BTC/USD": asset()}
    broker.positions["BTC/USD"] = BrokerPosition("BTC/USD", 40.0, 100.0, 101.0, 4_040, 40, 0.01)
    local = CryptoPositionState("BTC/USD", 40, 100, now.isoformat(), 2, 95, 101)
    store.set_bot_position("BTC/USD", local.to_dict())
    engine._ensure_protection("BTC/USD", broker.positions["BTC/USD"], now)
    first_stop_id = store.get_bot_position("BTC/USD")["stop_client_order_id"]
    assert broker.stop_calls[-1][1] == 40

    broker.positions["BTC/USD"].qty = 70.0
    engine._ensure_protection("BTC/USD", broker.positions["BTC/USD"], now + timedelta(minutes=1))
    assert first_stop_id != store.get_bot_position("BTC/USD")["stop_client_order_id"]
    assert broker.stop_calls[-1][1] == 70
    assert broker.orders[first_stop_id].status == "canceled"


def test_strategy_exit_refetches_qty_after_stop_partial_fill_and_cannot_oversell(tmp_path):
    engine, broker, store, now = make_engine(tmp_path)
    engine._assets = {"BTC/USD": asset()}
    broker.positions["BTC/USD"] = BrokerPosition("BTC/USD", 100, 100, 100, 10_000, 0, 0)
    stop = OrderRecord("stop-1", "trendbot-old-BTCUSD-stop-1", "BTC/USD", "stop", "sell", 100, 0, "new", stop_price=95)
    broker.orders[stop.client_order_id] = stop
    local = CryptoPositionState(
        "BTC/USD", 100, 100, now.isoformat(), 2, 95, 100,
        stop_client_order_id=stop.client_order_id,
        stop_broker_order_id=stop.broker_order_id,
        stop_qty=100,
    )
    store.set_bot_position("BTC/USD", local.to_dict())
    broker.cancel_position_qty = 60.0
    decision = CryptoSignalDecision("BTC/USD", BotAction.EXIT, now, "trend_exit", close=99)
    engine._initiate_exit("BTC/USD", decision, breakout_bars().index[-1])
    assert broker.market_calls[-1][1] == 60.0
    assert broker.market_calls[-1][2] == "sell"


def test_stop_limit_watchdog_cancels_owned_stop_then_submits_one_market_exit(tmp_path):
    config = CryptoEngineConfig(poll_interval_seconds=5, stop_watchdog_grace_seconds=0)
    engine, broker, store, now = make_engine(tmp_path, engine_config=config)
    engine._assets = {"BTC/USD": asset()}
    position = BrokerPosition("BTC/USD", 1, 100, 90, 90, -10, -0.1)
    broker.positions["BTC/USD"] = position
    stop = OrderRecord("stop-1", "trendbot-old-BTCUSD-stop-1", "BTC/USD", "stop", "sell", 1, 0, "new", stop_price=95)
    broker.orders[stop.client_order_id] = stop
    local = CryptoPositionState(
        "BTC/USD", 1, 100, now.isoformat(), 2, 95, 100,
        stop_client_order_id=stop.client_order_id,
        stop_broker_order_id=stop.broker_order_id,
        stop_qty=1,
        breached_at=(now - timedelta(seconds=1)).isoformat(),
    )
    store.set_bot_position("BTC/USD", local.to_dict())
    quote = CryptoQuote("BTC/USD", now, 90, 90.1)
    # A sustained breach needs stop_watchdog_min_confirmations observations;
    # the first pass only records the breach.
    engine._run_stop_watchdogs(now, {"BTC/USD": quote}, {"BTC/USD": position})
    assert broker.market_calls == []
    later = now + timedelta(seconds=5)
    engine._run_stop_watchdogs(
        later, {"BTC/USD": CryptoQuote("BTC/USD", later, 90, 90.1)}, {"BTC/USD": position}
    )
    assert stop.status == "canceled"
    assert broker.market_calls[-1][1:3] == (1, "sell")
    assert broker.market_calls[-1][3].split("-")[-2] == "guard"
    # Exactly one exit, never a second on a later pass.
    engine._run_stop_watchdogs(
        later + timedelta(seconds=5),
        {"BTC/USD": CryptoQuote("BTC/USD", later, 90, 90.1)},
        {"BTC/USD": position},
    )
    assert len(broker.market_calls) == 1


def test_stop_watchdog_never_trades_from_a_stale_quote(tmp_path):
    config = CryptoEngineConfig(poll_interval_seconds=5, stop_watchdog_grace_seconds=0)
    engine, broker, store, now = make_engine(tmp_path, engine_config=config)
    engine._assets = {"BTC/USD": asset()}
    position = BrokerPosition("BTC/USD", 1, 100, 90, 90, -10, -0.1)
    broker.positions["BTC/USD"] = position
    local = CryptoPositionState(
        "BTC/USD", 1, 100, now.isoformat(), 2, 95, 100,
        breached_at=(now - timedelta(seconds=1)).isoformat(),
    )
    store.set_bot_position("BTC/USD", local.to_dict())
    stale = CryptoQuote("BTC/USD", now - timedelta(minutes=5), 90, 90.1)
    engine._run_stop_watchdogs(now, {"BTC/USD": stale}, {"BTC/USD": position})
    assert broker.market_calls == []


def test_recovered_bid_clears_the_breach_instead_of_liquidating(tmp_path):
    """A wick below the stop must not arm a liquidation that outlives it."""

    # Production 1Hour cadence: the grace window is shorter than one poll, so
    # a latch with no un-latch fires on the very next pass.
    config = CryptoEngineConfig(poll_interval_seconds=120, stop_watchdog_grace_seconds=15)
    engine, broker, store, now = make_engine(tmp_path, engine_config=config)
    engine._assets = {"BTC/USD": asset()}
    position = BrokerPosition("BTC/USD", 1, 100, 99, 99, -1, -0.01)
    broker.positions["BTC/USD"] = position
    stop = OrderRecord(
        "stop-1", "trendbot-s-BTCUSD-stop-1", "BTC/USD", "stop", "sell", 1, 0, "new",
        stop_price=95,
    )
    broker.orders[stop.client_order_id] = stop
    local = CryptoPositionState(
        "BTC/USD", 1, 100, now.isoformat(), 2, 95, 100,
        stop_client_order_id=stop.client_order_id,
        stop_broker_order_id=stop.broker_order_id,
        stop_qty=1,
    )
    store.set_bot_position("BTC/USD", local.to_dict())

    engine._run_stop_watchdogs(
        now, {"BTC/USD": CryptoQuote("BTC/USD", now, 94.9, 95.0)}, {"BTC/USD": position}
    )
    assert broker.market_calls == []
    assert store.get_bot_position("BTC/USD")["breached_at"] is not None

    later = now + timedelta(seconds=120)
    engine._run_stop_watchdogs(
        later, {"BTC/USD": CryptoQuote("BTC/USD", later, 99.0, 99.1)}, {"BTC/USD": position}
    )
    assert broker.market_calls == []
    assert stop.status == "new"
    cleared = store.get_bot_position("BTC/USD")
    assert cleared["breached_at"] is None
    assert cleared["breach_confirmations"] == 0


def test_sustained_breach_still_escalates_after_a_recovery(tmp_path):
    """Clearing the latch must not disarm a genuine, sustained breach."""

    config = CryptoEngineConfig(poll_interval_seconds=5, stop_watchdog_grace_seconds=0)
    engine, broker, store, now = make_engine(tmp_path, engine_config=config)
    engine._assets = {"BTC/USD": asset()}
    position = BrokerPosition("BTC/USD", 1, 100, 90, 90, -10, -0.1)
    broker.positions["BTC/USD"] = position
    local = CryptoPositionState("BTC/USD", 1, 100, now.isoformat(), 2, 95, 100)
    store.set_bot_position("BTC/USD", local.to_dict())

    # Dip, recover (latch clears), then a real breach that persists.
    engine._run_stop_watchdogs(
        now, {"BTC/USD": CryptoQuote("BTC/USD", now, 94.5, 94.6)}, {"BTC/USD": position}
    )
    t1 = now + timedelta(seconds=5)
    engine._run_stop_watchdogs(
        t1, {"BTC/USD": CryptoQuote("BTC/USD", t1, 96.0, 96.1)}, {"BTC/USD": position}
    )
    assert broker.market_calls == []
    for step in (2, 3):
        ts = now + timedelta(seconds=5 * step + 5)
        engine._run_stop_watchdogs(
            ts, {"BTC/USD": CryptoQuote("BTC/USD", ts, 93.0, 93.1)}, {"BTC/USD": position}
        )
    assert len(broker.market_calls) == 1
    assert broker.market_calls[-1][1:3] == (1, "sell")


def test_engine_polls_faster_while_a_position_is_open(tmp_path):
    config = CryptoEngineConfig(
        poll_interval_seconds=120, position_open_poll_interval_seconds=5
    )
    engine, _broker, store, now = make_engine(tmp_path, engine_config=config)
    assert not engine._has_exposure()
    local = CryptoPositionState("BTC/USD", 1, 100, now.isoformat(), 2, 95, 100)
    store.set_bot_position("BTC/USD", local.to_dict())
    assert engine._has_exposure()
    # Confirmation-based escalation is only safe with the faster exposed cadence.
    assert config.exposed_poll_interval("1Hour") == 5
    assert config.resolved_poll_interval("1Hour") == 120


def test_rejected_guard_exit_advances_to_a_fresh_client_id(tmp_path):
    """A rejected guard must not freeze the id and leave the position naked."""

    config = CryptoEngineConfig(poll_interval_seconds=5, stop_watchdog_grace_seconds=0)
    engine, broker, store, now = make_engine(tmp_path, engine_config=config)
    engine._assets = {"BTC/USD": asset()}
    position = BrokerPosition("BTC/USD", 1, 100, 90, 90, -10, -0.1)
    broker.positions["BTC/USD"] = position
    local = CryptoPositionState("BTC/USD", 1, 100, now.isoformat(), 2, 95, 100)
    store.set_bot_position("BTC/USD", local.to_dict())

    quote = CryptoQuote("BTC/USD", now, 90, 90.1)
    engine._run_stop_watchdogs(now, {"BTC/USD": quote}, {"BTC/USD": position})
    t1 = now + timedelta(seconds=5)
    engine._run_stop_watchdogs(
        t1, {"BTC/USD": CryptoQuote("BTC/USD", t1, 90, 90.1)}, {"BTC/USD": position}
    )
    first_id = broker.market_calls[-1][3]

    # The broker rejects it; the engine clears the pending marker and retries.
    broker.orders[first_id].status = "rejected"
    reloaded = CryptoPositionState.from_dict(store.get_bot_position("BTC/USD"))
    reloaded.pending_exit_client_order_id = None
    store.set_bot_position("BTC/USD", reloaded.to_dict())

    t2 = t1 + timedelta(seconds=5)
    engine._run_stop_watchdogs(
        t2, {"BTC/USD": CryptoQuote("BTC/USD", t2, 90, 90.1)}, {"BTC/USD": position}
    )
    assert len(broker.market_calls) == 2
    assert broker.market_calls[-1][3] != first_id


def test_restart_with_owned_position_and_exact_stop_does_not_duplicate_stop(tmp_path):
    engine, broker, store, now = make_engine(tmp_path)
    position = BrokerPosition("BTC/USD", 1, 100, 101, 101, 1, 0.01)
    broker.positions["BTC/USD"] = position
    entry = OrderRecord(
        "entry-1", "trendbot-old-BTCUSD-entry-1", "BTC/USD", "entry", "buy",
        1, 1, "filled", fill_price=100,
    )
    broker.orders[entry.client_order_id] = entry
    store.record_order(entry, "old-session")
    stop = OrderRecord("stop-1", "trendbot-old-BTCUSD-stop-1", "BTC/USD", "stop", "sell", 1, 0, "new", stop_price=95)
    broker.orders[stop.client_order_id] = stop
    store.record_order(stop, "old-session")
    local = CryptoPositionState(
        "BTC/USD", 1, 100, now.isoformat(), 2, 95, 101,
        stop_client_order_id=stop.client_order_id,
        stop_broker_order_id=stop.broker_order_id,
        stop_qty=1,
    )
    store.set_bot_position("BTC/USD", local.to_dict())
    engine._startup()
    assert broker.stop_calls == []
    assert store.get_bot_position("BTC/USD")["stop_client_order_id"] == stop.client_order_id


def test_startup_refuses_unowned_manual_crypto_position(tmp_path):
    engine, broker, _, _ = make_engine(tmp_path)
    broker.positions["BTC/USD"] = BrokerPosition("BTC/USD", 1, 100, 100, 100, 0, 0)
    with pytest.raises(BrokerError, match="manual or foreign exposure"):
        engine._startup()


def test_manual_position_increase_is_not_absorbed_into_bot_stop(tmp_path):
    engine, broker, store, now = make_engine(tmp_path)
    engine._assets = {"BTC/USD": asset()}
    entry = OrderRecord(
        "entry-1", "trendbot-old-BTCUSD-entry-1", "BTC/USD", "entry", "buy",
        1, 1, "filled", fill_price=100,
    )
    broker.orders[entry.client_order_id] = entry
    store.record_order(entry, "old-session")
    local = CryptoPositionState("BTC/USD", 1, 100, now.isoformat(), 2, 95, 100)
    store.set_bot_position("BTC/USD", local.to_dict())
    broker.positions["BTC/USD"] = BrokerPosition("BTC/USD", 2, 100, 100, 200, 0, 0)
    with pytest.raises(BrokerError, match="manual or foreign exposure"):
        engine._reconcile_position_truth(now)


def test_session_loss_latch_persists_kill_switch_and_blocks_new_entry(tmp_path):
    engine, broker, store, _ = make_engine(tmp_path)
    engine._startup()
    engine._starting_equity = 100_000
    broker.account = AccountInfo(97_000, 97_000, 97_000, "ACTIVE", "USD", False, 0, False)
    engine.run_iteration()
    assert engine.risk.session_halted
    assert store.get_kv(CRYPTO_KILL_SWITCH_KEY) is not None
    assert broker.market_calls == []


def test_delayed_gtc_entry_is_canceled_before_it_can_fill_late(tmp_path):
    engine, broker, store, now = make_engine(tmp_path)
    order = OrderRecord(
        "entry-1",
        "trendbot-old-BTCUSD-entry-1",
        "BTC/USD",
        "entry",
        "buy",
        1,
        0,
        "new",
        submitted_at=now - timedelta(minutes=3),
        updated_at=now - timedelta(minutes=3),
    )
    broker.orders[order.client_order_id] = order
    store.record_order(order, "old-session")
    engine._cancel_stale_entry_orders(now)
    assert order.status == "canceled"
    assert broker.cancel_calls == ["entry-1"]


def test_unknown_exit_outcome_keeps_intent_and_reuses_same_id_without_rearming_stop(tmp_path):
    engine, broker, store, now = make_engine(tmp_path)
    engine._assets = {"BTC/USD": asset()}
    position = BrokerPosition("BTC/USD", 1, 100, 99, 99, -1, -0.01)
    broker.positions["BTC/USD"] = position
    entry = OrderRecord(
        "entry-1", "trendbot-old-BTCUSD-entry-1", "BTC/USD", "entry", "buy",
        1, 1, "filled", fill_price=100,
    )
    broker.orders[entry.client_order_id] = entry
    store.record_order(entry, "old-session")
    stop = OrderRecord(
        "stop-1", "trendbot-old-BTCUSD-stop-1", "BTC/USD", "stop", "sell",
        1, 0, "new", stop_price=95,
    )
    broker.orders[stop.client_order_id] = stop
    local = CryptoPositionState(
        "BTC/USD", 1, 100, now.isoformat(), 2, 95, 100,
        stop_client_order_id=stop.client_order_id,
        stop_broker_order_id=stop.broker_order_id,
        stop_qty=1,
    )
    store.set_bot_position("BTC/USD", local.to_dict())
    broker.timeout_after_acceptance = True
    broker.hidden_lookups_after_timeout = 2
    decision = CryptoSignalDecision("BTC/USD", BotAction.EXIT, now, "trend_exit", close=99)
    with pytest.raises(BrokerError, match="timeout"):
        engine._initiate_exit("BTC/USD", decision, breakout_bars().index[-1])
    pending_id = store.get_bot_position("BTC/USD")["pending_exit_client_order_id"]
    assert pending_id
    assert stop.status == "canceled"

    reconciled = engine._reconcile_position_truth(now)
    assert "BTC/USD" in reconciled
    assert store.get_bot_position("BTC/USD")["pending_exit_client_order_id"] == pending_id
    assert len([call for call in broker.market_calls if call[2] == "sell"]) == 1
    engine._ensure_all_protection(now, reconciled)
    assert broker.stop_calls == []


def test_entry_blocked_when_round_trip_cost_eats_the_risk_unit(tmp_path):
    """A stop so tight that friction exceeds it must never be entered."""

    from crypto_models import CryptoSignalDecision as Decision

    engine, broker, store, now = make_engine(tmp_path)
    engine._assets = {"BTC/USD": asset()}
    account = broker.get_account()
    bar_ts = breakout_bars(periods=31).index[-1]
    prepared = engine.strategy.prepare_universe(
        {"BTC/USD": breakout_bars(periods=31)}
    )

    # ATR of 0.001 on a $100 price gives a 2.5bp stop distance against a
    # 75bp round trip: cost is many multiples of 1R.
    decision = Decision("BTC/USD", BotAction.BUY, bar_ts, "breakout", close=100, atr=0.001)
    quote = CryptoQuote("BTC/USD", now, 99.99, 100.0)
    engine._handle_entries(
        [decision], prepared, bar_ts, {"BTC/USD": quote}, account, {}, now
    )
    assert [call for call in broker.market_calls if call[2] == "buy"] == []

    # A normal ATR keeps friction well inside 1R and the entry proceeds.
    decision = Decision("BTC/USD", BotAction.BUY, bar_ts, "breakout", close=100, atr=2.0)
    engine._handle_entries(
        [decision], prepared, bar_ts, {"BTC/USD": quote}, account, {}, now
    )
    assert [call for call in broker.market_calls if call[2] == "buy"]


def test_stock_positions_do_not_consume_the_crypto_exposure_cap(tmp_path):
    """A shared account's equity holdings must not throttle crypto sizing."""

    engine, broker, store, _ = make_engine(tmp_path)
    # A large stock position: under the old unfiltered exposure sum this
    # consumed the entire 30% crypto cap and blocked every entry.
    broker.positions["AAPL"] = BrokerPosition(
        "AAPL", 100, 500, 500, 50_000, 0, 0, asset_class="us_equity"
    )
    engine._startup()
    engine.run_iteration()
    buys = [call for call in broker.market_calls if call[2] == "buy"]
    assert len(buys) == 1
