"""Behavior tests for trade_engine.TradeEngine and EngineController.

All collaborators are replaced with deterministic in-file fakes:

* ``FakeBroker`` — the exact AlpacaPaperBroker surface the engine uses.
  Market orders fill immediately at a settable price, stop orders rest,
  positions are maintained from fills, and every submission/cancel is
  recorded so tests can assert exactly what the engine did.
* ``FakeDataFeed`` — returns preset OHLCV frames (optionally one frame
  per call, for incremental delivery), or raises on demand.

No network access ever happens; alpaca-py is never exercised.

The synthetic frames use a deliberately tiny strategy configuration
(short_ma=2, long_ma=3, rsi=2, atr=2, volume=2, volatility filter off)
so a confirmed BUY / bearish EXIT can be engineered in a handful of
bars.  The frame shapes were verified against TrendStrategy directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import pytest

import trade_engine
from backtest import run_backtest
from broker import BrokerError, make_client_order_id
from config import (
    BacktestConfig,
    CLIENT_ORDER_PREFIX,
    EngineConfig,
    RiskConfig,
    StrategyConfig,
    timeframe_to_timedelta,
)
from data_feed import DataFeedError
from models import (
    AccountInfo,
    AssetInfo,
    BotAction,
    BrokerPosition,
    EngineState,
    MarketClock,
    OrderPurpose,
    OrderRecord,
    SignalDecision,
    StopReason,
)
from risk_manager import RiskManager
from state_store import StateStore
from strategy import TrendStrategy
from trade_engine import TradeEngine, get_controller

SYMBOL = "TEST"

# Fixed, timezone-aware "wall clock" for deterministic tests.
NOW = datetime(2026, 8, 10, 15, 30, 0, tzinfo=timezone.utc)

# Close sequences engineered for the tiny config below (verified against
# TrendStrategy while developing):
# * CLOSES_BUY ends in a confirmed BUY (bullish 2/3-MA crossover, RSI(2)
#   ~56.5 < 70, volume ratio 1.0 >= 0.1).
# * CLOSES_EXIT extends it to end in a bearish crossover (EXIT).
# * CLOSES_PARITY (~40 bars) contains an early positionless EXIT, a
#   confirmed BUY, a later EXIT that closes the position, then a quiet tail.
CLOSES_BUY = [100.0, 101.0, 102.0, 103.0, 104.0, 98.0, 96.0, 97.0, 98.0]
CLOSES_EXIT = CLOSES_BUY + [99.0, 97.0, 94.0]
CLOSES_PARITY = (
    CLOSES_BUY
    + [99.5, 101.0, 102.5, 104.0, 105.5, 107.0, 108.0]
    + [107.5, 105.0, 102.0, 99.0, 96.0]
    + [95.5, 95.0, 94.5, 94.0, 93.5, 93.0, 92.5, 92.0, 91.5,
       91.0, 90.5, 90.0, 89.5, 89.0, 88.5, 88.0, 87.5, 87.0]
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_bars(now: datetime, closes: list[float], timeframe: str = "1Day") -> pd.DataFrame:
    """Deterministic tz-aware OHLCV frame ending shortly before ``now``.

    1Day: the last bar starts 25h before ``now`` (completed one hour ago,
    comfortably inside the 5-day staleness allowance).  Intraday: the last
    bar starts one bar-duration plus one minute before ``now``.
    """
    n = len(closes)
    if timeframe == "1Day":
        last_start = now - timedelta(hours=25)
        freq = "D"
    else:
        last_start = now - timeframe_to_timedelta(timeframe) - timedelta(minutes=1)
        freq = timeframe.replace("Min", "min")
    idx = pd.date_range(end=last_start, periods=n, freq=freq)
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + 1.0 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 1.0 for o, c in zip(opens, closes)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * n,
        },
        index=idx,
    )


def make_buy_bars(now: datetime, timeframe: str = "1Day") -> pd.DataFrame:
    """Frame whose LAST completed bar is a confirmed BUY signal."""
    return _make_bars(now, CLOSES_BUY, timeframe)


def make_exit_bars(now: datetime) -> pd.DataFrame:
    """Frame whose LAST completed bar is a bearish-crossover EXIT signal."""
    return _make_bars(now, CLOSES_EXIT)


def make_parity_bars() -> pd.DataFrame:
    """~40-bar daily frame with a confirmed BUY then a later EXIT."""
    return _make_bars(NOW, CLOSES_PARITY)


def small_strategy_config(timeframe: str = "1Day") -> StrategyConfig:
    return StrategyConfig(
        symbol=SYMBOL,
        timeframe=timeframe,
        short_ma_period=2,
        long_ma_period=3,
        rsi_period=2,
        atr_period=2,
        volume_period=2,
        min_volume_ratio=0.1,
        enable_volatility_filter=False,
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _purpose_from_client_id(client_order_id: str) -> str:
    """Derive purpose exactly like broker.order_to_record: for the id
    format ``trendbot-<sess8>-<SYMBOL>-<purpose>-<barEpoch>`` the purpose
    is the second-to-last dash segment."""
    if client_order_id and client_order_id.startswith(f"{CLIENT_ORDER_PREFIX}-"):
        parts = client_order_id.split("-")
        if len(parts) >= 4:
            return parts[-2]
    return "unknown"


class FakeBroker:
    """In-memory stand-in for AlpacaPaperBroker.

    * market orders fill immediately at ``fill_price`` (or are rejected
      when ``reject_submits`` is set, or raise ``submit_error``);
    * stop orders rest with status "new"; ``cancel_order`` flips an order
      to "canceled";
    * positions are maintained from fills (a buy fill creates/increases
      the position, a sell fill reduces/clears it);
    * every submit/cancel is recorded (``submitted``, ``cancel_calls``,
      and the ordered ``events`` log) for assertions.
    """

    def __init__(self, now: datetime = NOW):
        self.account = AccountInfo(
            equity=100_000.0,
            cash=100_000.0,
            buying_power=100_000.0,
            status="ACTIVE",
            currency="USD",
            pattern_day_trader=False,
            daytrade_count=0,
            trading_blocked=False,
        )
        self.clock = MarketClock(
            timestamp=now,
            is_open=True,
            next_open=now + timedelta(hours=18),
            next_close=now + timedelta(hours=3),
        )
        self.asset_tradable = True
        self.fill_price = 100.0
        self.reject_submits = False
        self.submit_error: Optional[Exception] = None
        self.verify_error: Optional[Exception] = None
        self.get_account_error: Optional[Exception] = None
        self.orders: dict[str, OrderRecord] = {}  # keyed by client_order_id
        self.positions: dict[str, BrokerPosition] = {}
        self.submitted: list[OrderRecord] = []
        self.cancel_calls: list[str] = []
        self.events: list[tuple[str, str]] = []
        self._seq = 0

    # -- account / environment -----------------------------------------

    def verify_paper_account(self) -> AccountInfo:
        if self.verify_error is not None:
            raise self.verify_error
        return self.account

    def get_account(self) -> AccountInfo:
        if self.get_account_error is not None:
            raise self.get_account_error
        return self.account

    def get_clock(self) -> MarketClock:
        return self.clock

    def get_asset(self, symbol: str) -> AssetInfo:
        return AssetInfo(
            symbol=symbol,
            name=symbol,
            tradable=self.asset_tradable,
            active=True,
            fractionable=False,
            asset_class="us_equity",
        )

    # -- positions -------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        pos = self.positions.get(symbol)
        return replace(pos) if pos is not None else None

    def list_positions(self) -> list[BrokerPosition]:
        return [replace(p) for p in self.positions.values()]

    def _apply_fill(self, symbol: str, qty: float, side: str, price: float) -> None:
        if side == "buy":
            existing = self.positions.get(symbol)
            if existing is None:
                self.positions[symbol] = BrokerPosition(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=price,
                    current_price=price,
                    market_value=qty * price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                )
            else:
                total = existing.qty + qty
                avg = (existing.qty * existing.avg_entry_price + qty * price) / total
                existing.qty = total
                existing.avg_entry_price = avg
        else:
            existing = self.positions.get(symbol)
            if existing is not None:
                existing.qty -= qty
                if existing.qty <= 1e-9:
                    del self.positions[symbol]

    # -- orders ----------------------------------------------------------

    def submit_market_order(
        self, symbol: str, qty: int, side: str, client_order_id: str
    ) -> OrderRecord:
        self.events.append(("submit_market", client_order_id))
        if self.submit_error is not None:
            raise self.submit_error
        self._seq += 1
        ts = self.clock.timestamp
        if self.reject_submits:
            record = OrderRecord(
                broker_order_id=f"fake-{self._seq}",
                client_order_id=client_order_id,
                symbol=symbol,
                purpose=_purpose_from_client_id(client_order_id),
                side=side,
                requested_qty=float(qty),
                filled_qty=0.0,
                status="rejected",
                fill_price=None,
                stop_price=None,
                submitted_at=ts,
                updated_at=ts,
            )
        else:
            record = OrderRecord(
                broker_order_id=f"fake-{self._seq}",
                client_order_id=client_order_id,
                symbol=symbol,
                purpose=_purpose_from_client_id(client_order_id),
                side=side,
                requested_qty=float(qty),
                filled_qty=float(qty),
                status="filled",
                fill_price=self.fill_price,
                stop_price=None,
                submitted_at=ts,
                updated_at=ts,
            )
            self._apply_fill(symbol, float(qty), side, self.fill_price)
        self.orders[client_order_id] = record
        self.submitted.append(record)
        return replace(record)

    def submit_stop_order(
        self, symbol: str, qty: int, stop_price: float, client_order_id: str
    ) -> OrderRecord:
        self.events.append(("submit_stop", client_order_id))
        if self.submit_error is not None:
            raise self.submit_error
        self._seq += 1
        ts = self.clock.timestamp
        status = "rejected" if self.reject_submits else "new"
        record = OrderRecord(
            broker_order_id=f"fake-{self._seq}",
            client_order_id=client_order_id,
            symbol=symbol,
            purpose=_purpose_from_client_id(client_order_id),
            side="sell",
            requested_qty=float(qty),
            filled_qty=0.0,
            status=status,
            fill_price=None,
            stop_price=stop_price,
            submitted_at=ts,
            updated_at=ts,
        )
        self.orders[client_order_id] = record
        self.submitted.append(record)
        return replace(record)

    def get_order(self, broker_order_id: str) -> OrderRecord:
        for record in self.orders.values():
            if record.broker_order_id == broker_order_id:
                return replace(record)
        raise BrokerError(f"Unknown broker order id {broker_order_id!r}")

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderRecord]:
        record = self.orders.get(client_order_id)
        return replace(record) if record is not None else None

    def list_open_orders(self, symbol: Optional[str] = None) -> list[OrderRecord]:
        return [
            replace(o)
            for o in self.orders.values()
            if o.is_open
            and o.client_order_id.startswith(f"{CLIENT_ORDER_PREFIX}-")
            and (symbol is None or o.symbol == symbol)
        ]

    def cancel_order(self, broker_order_id: str) -> None:
        for record in self.orders.values():
            if record.broker_order_id == broker_order_id:
                record.status = "canceled"
                record.updated_at = self.clock.timestamp
                self.cancel_calls.append(record.client_order_id)
                self.events.append(("cancel", record.client_order_id))
                return
        raise BrokerError(f"Unknown broker order id {broker_order_id!r}")


class FakeDataFeed:
    """Returns preset frames; ``queue`` serves one frame per call before
    falling back to ``frame``; ``error`` raises instead."""

    def __init__(self, frame: Optional[pd.DataFrame] = None):
        self.frame = frame
        self.queue: list[pd.DataFrame] = []
        self.error: Optional[Exception] = None
        self.calls = 0

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
        now: Optional[datetime] = None,
        tolerance_seconds: float = 5.0,
    ) -> Optional[pd.DataFrame]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.queue:
            return self.queue.pop(0)
        return self.frame


class MutableNow:
    """Callable now_fn whose value tests can move forward."""

    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class Harness:
    engine: TradeEngine
    broker: FakeBroker
    feed: FakeDataFeed
    store: StateStore
    now: MutableNow
    strategy_cfg: StrategyConfig
    risk_cfg: RiskConfig
    engine_cfg: EngineConfig


def build_harness(
    tmp_path,
    timeframe: str = "1Day",
    now: datetime = NOW,
    engine_cfg: Optional[EngineConfig] = None,
    account: Optional[AccountInfo] = None,
    market_open: bool = True,
) -> Harness:
    """Wire a TradeEngine to fakes with a fixed now_fn equal to the fake
    clock timestamp (no skew) and next_close 3h away (no near-close block)."""
    broker = FakeBroker(now=now)
    if account is not None:
        broker.account = account
    broker.clock = replace(broker.clock, is_open=market_open)
    feed = FakeDataFeed()
    strategy_cfg = small_strategy_config(timeframe)
    risk_cfg = RiskConfig()
    resolved_engine_cfg = engine_cfg or EngineConfig(poll_interval_seconds=5)
    store = StateStore(str(tmp_path / "state.db"))
    holder = MutableNow(now)
    engine = TradeEngine(
        broker=broker,
        data_feed=feed,
        strategy=TrendStrategy(strategy_cfg),
        risk_manager=RiskManager(risk_cfg),
        state_store=store,
        strategy_config=strategy_cfg,
        risk_config=risk_cfg,
        engine_config=resolved_engine_cfg,
        now_fn=holder,
    )
    return Harness(
        engine=engine,
        broker=broker,
        feed=feed,
        store=store,
        now=holder,
        strategy_cfg=strategy_cfg,
        risk_cfg=risk_cfg,
        engine_cfg=resolved_engine_cfg,
    )


def advance_time(h: Harness, new_now: datetime, is_open: bool = True) -> None:
    """Move the engine's wall clock AND the broker clock together so the
    clock-skew guard never fires accidentally."""
    h.now.now = new_now
    h.broker.clock = MarketClock(
        timestamp=new_now,
        is_open=is_open,
        next_open=new_now + timedelta(hours=18),
        next_close=new_now + timedelta(hours=3),
    )


def inject_resting_order(
    h: Harness,
    purpose: str,
    qty: int = 10,
    stop_price: Optional[float] = None,
    bar_ts: Optional[datetime] = None,
) -> str:
    """Place a resting bot-namespace order directly into the fake broker
    (NOT via submit, so it never appears in broker.submitted)."""
    bar_ts = bar_ts or (h.now.now - timedelta(days=2))
    cid = make_client_order_id(h.engine.session_id, purpose, bar_ts, symbol=SYMBOL)
    h.broker._seq += 1
    h.broker.orders[cid] = OrderRecord(
        broker_order_id=f"inj-{h.broker._seq}",
        client_order_id=cid,
        symbol=SYMBOL,
        purpose=purpose,
        side="sell" if purpose == OrderPurpose.STOP.value else "buy",
        requested_qty=float(qty),
        filled_qty=0.0,
        status="new",
        fill_price=None,
        stop_price=stop_price,
        submitted_at=h.broker.clock.timestamp,
        updated_at=h.broker.clock.timestamp,
    )
    return cid


def preset_bot_position(
    h: Harness, qty: int = 10, entry_price: float = 100.0, atr: float = 2.0,
    stop_price: float = 96.0,
) -> str:
    """Give the engine a bot-owned position (store record + matching broker
    position + resting protective stop).  Returns the stop client id."""
    stop_cid = inject_resting_order(h, OrderPurpose.STOP.value, qty=qty, stop_price=stop_price)
    h.broker.positions[SYMBOL] = BrokerPosition(
        symbol=SYMBOL,
        qty=float(qty),
        avg_entry_price=entry_price,
        current_price=entry_price,
        market_value=qty * entry_price,
        unrealized_pl=0.0,
        unrealized_plpc=0.0,
    )
    h.store.set_bot_position(
        SYMBOL,
        {
            "qty": qty,
            "entry_price": entry_price,
            "atr_at_entry": atr,
            "stop_price": stop_price,
            "stop_client_order_id": stop_cid,
            "session_id": h.engine.session_id,
            "entry_time": (h.now.now - timedelta(days=1)).isoformat(),
        },
    )
    return stop_cid


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def submitted_by_purpose(h: Harness, purpose: str) -> list[OrderRecord]:
    return [o for o in h.broker.submitted if o.purpose == purpose]


def last_signal(h: Harness) -> dict:
    raw = h.store.get_kv(trade_engine.LAST_SIGNAL_KEY)
    assert raw is not None
    return json.loads(raw)


@pytest.fixture(autouse=True)
def _reset_controller_singleton():
    """trade_engine keeps a process-wide EngineController singleton (it
    must survive Streamlit reruns in production).  Tests all run in one
    process, so without this reset every test would inherit the controller
    — and possibly a still-running engine — from a previous test.  Clear
    it before each test and stop any leftover engine afterwards."""
    trade_engine._controller = None
    yield
    controller = trade_engine._controller
    if controller is not None and controller.engine is not None:
        controller.engine.stop()
        controller.engine.join(timeout=10)
    trade_engine._controller = None


# ---------------------------------------------------------------------------
# 1-3: lifecycle via the controller / emergency stop
# ---------------------------------------------------------------------------


def test_controller_start_is_idempotent(tmp_path):
    """Starting twice must not create a second engine: the second call
    returns (False, ...) and the factory runs exactly once."""
    controller = get_controller()
    created: list[Harness] = []

    def factory() -> TradeEngine:
        h = build_harness(tmp_path, market_open=False)
        created.append(h)
        return h.engine

    ok1, _ = controller.start_engine(factory)
    try:
        assert ok1
        assert wait_for(lambda: controller.is_running)
        ok2, msg2 = controller.start_engine(factory)
        assert not ok2
        assert "already running" in msg2
        assert len(created) == 1
        assert controller.engine is created[0].engine
    finally:
        controller.stop_engine()
        created[0].engine.join(10)
    assert not controller.is_running


def test_controller_stop_ends_thread_and_persists_user_stop(tmp_path):
    h = build_harness(tmp_path, market_open=False)
    controller = get_controller()
    ok, _ = controller.start_engine(lambda: h.engine)
    assert ok
    assert wait_for(lambda: h.engine.status.state == EngineState.RUNNING)
    ok, _ = controller.stop_engine()
    assert ok
    h.engine.join(10)
    assert not h.engine.is_running
    assert h.engine.status.state == EngineState.STOPPED
    session = h.store.get_session(h.engine.session_id)
    assert session is not None
    assert session["stop_reason"] == StopReason.USER_STOP.value
    # A clean user stop must NOT arm the kill switch.
    assert h.store.get_kv(trade_engine.KILL_SWITCH_KEY) is None


def test_emergency_stop_sets_kill_switch_and_preserves_protective_stop(tmp_path):
    """Emergency Stop persists EMERGENCY_STOP + the kill switch, and
    deliberately leaves the broker-side protective stop resting so the
    open position keeps its downside protection after the engine dies."""
    h = build_harness(tmp_path, market_open=False)
    stop_cid = preset_bot_position(h)
    h.engine.start()
    try:
        assert wait_for(lambda: h.engine.status.state == EngineState.RUNNING)
        h.engine.emergency_stop()
        h.engine.join(10)
    finally:
        h.engine.stop()
        h.engine.join(5)
    assert not h.engine.is_running
    session = h.store.get_session(h.engine.session_id)
    assert session["stop_reason"] == StopReason.EMERGENCY_STOP.value
    raw = h.store.get_kv(trade_engine.KILL_SWITCH_KEY)
    assert raw is not None
    assert json.loads(raw)["reason"] == StopReason.EMERGENCY_STOP.value
    # The resting protective stop was not touched by shutdown.
    assert h.broker.orders[stop_cid].status == "new"
    assert h.broker.cancel_calls == []


# ---------------------------------------------------------------------------
# 4-5: duplicate-execution protection
# ---------------------------------------------------------------------------


def test_same_bar_is_processed_only_once(tmp_path):
    """Two iterations over the same completed bar submit exactly one
    entry order and record the bar in processed_bars once."""
    h = build_harness(tmp_path)
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()
    h.engine.run_iteration()
    assert len(submitted_by_purpose(h, OrderPurpose.ENTRY.value)) == 1
    # entry + its protective stop, nothing more on the repeat iteration
    assert len(h.broker.submitted) == 2
    assert len(h.store.get_processed_bars()) == 1


def test_client_order_id_guard_prevents_duplicate_entry(tmp_path):
    """Even with the in-memory pending marker wiped, the persisted
    deterministic client order id blocks a resubmission of the same
    (session, symbol, purpose, bar) order."""
    h = build_harness(tmp_path)
    h.broker.reject_submits = True  # keep the entry terminal but unfilled
    bars = make_buy_bars(NOW)
    h.feed.frame = bars
    h.engine._startup()
    h.engine.run_iteration()
    entries = submitted_by_purpose(h, OrderPurpose.ENTRY.value)
    assert len(entries) == 1
    entry_cid = entries[0].client_order_id
    assert h.store.get_order_by_client_id(entry_cid) is not None

    # Simulate losing the in-memory marker (e.g. a restart mid-cycle).
    h.engine._pending_entry = None
    h.engine.run_iteration()  # same bars: the bar dedup fires first
    assert len(submitted_by_purpose(h, OrderPurpose.ENTRY.value)) == 1
    assert len(h.store.get_processed_bars()) == 1

    # Bypass the bar dedup entirely: the store client-id guard must hold.
    decision = TrendStrategy(h.strategy_cfg).generate_signal(bars)
    assert decision.action == BotAction.BUY
    bar_ts = bars.index[-1].to_pydatetime()
    h.engine._handle_buy_signal(
        decision, bar_ts, h.broker.clock, h.broker.account, h.now.now
    )
    assert len(submitted_by_purpose(h, OrderPurpose.ENTRY.value)) == 1
    assert any(
        "not resubmitting" in (row["message"] or "")
        for row in h.store.get_logs(limit=200)
    )


# ---------------------------------------------------------------------------
# 6-8: entry guards and rejected orders
# ---------------------------------------------------------------------------


def test_existing_bot_position_blocks_new_entry(tmp_path):
    h = build_harness(tmp_path)
    preset_bot_position(h)
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()
    assert submitted_by_purpose(h, OrderPurpose.ENTRY.value) == []
    assert last_signal(h)["details"]["entry_block"] == "position_already_open"
    assert any(
        "position_already_open" in (row["message"] or "")
        for row in h.store.get_logs(limit=200)
    )


def test_pending_entry_blocks_new_entry(tmp_path):
    h = build_harness(tmp_path)
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    cid = inject_resting_order(h, OrderPurpose.ENTRY.value, qty=5)
    h.engine._pending_entry = {"client_order_id": cid, "atr": 2.0, "signal": None}
    h.engine.run_iteration()
    assert h.broker.submitted == []
    assert last_signal(h)["details"]["entry_block"] == "entry_order_pending"
    assert h.engine._pending_entry is not None  # still waiting on the fill


def test_rejected_entry_creates_no_position_and_no_stop(tmp_path):
    h = build_harness(tmp_path)
    h.broker.reject_submits = True
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()
    assert len(submitted_by_purpose(h, OrderPurpose.ENTRY.value)) == 1
    assert h.store.get_bot_position(SYMBOL) is None
    assert submitted_by_purpose(h, OrderPurpose.STOP.value) == []
    assert h.engine._pending_entry is None  # rejection was reconciled away
    assert any(
        "ended rejected with no fill" in (row["message"] or "")
        for row in h.store.get_logs(limit=200)
    )


# ---------------------------------------------------------------------------
# 9-10: fill confirmation and protective stop
# ---------------------------------------------------------------------------


def test_entry_fill_recorded_at_fill_price_not_signal_close(tmp_path):
    """The bot position must book the broker's ACTUAL fill price, not the
    signal bar's close."""
    h = build_harness(tmp_path)
    h.broker.fill_price = 101.37
    bars = make_buy_bars(NOW)
    h.feed.frame = bars
    h.engine._startup()
    h.engine.run_iteration()

    entries = submitted_by_purpose(h, OrderPurpose.ENTRY.value)
    assert len(entries) == 1
    assert entries[0].side == "buy"

    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None
    assert pos["qty"] == entries[0].requested_qty
    assert pos["entry_price"] == pytest.approx(101.37)
    signal_close = float(bars["close"].iloc[-1])
    assert abs(pos["entry_price"] - signal_close) > 1.0  # clearly not the close

    trades = [t for t in h.store.get_trades() if t["action"] == "BUY"]
    assert len(trades) == 1
    assert trades[0]["fill_price"] == pytest.approx(101.37)
    assert trades[0]["qty"] == entries[0].requested_qty


def test_protective_stop_placed_from_fill_price_and_signal_atr(tmp_path):
    """Stop price = actual fill price - ATR-at-entry * multiplier, where
    the ATR comes from the entry signal."""
    h = build_harness(tmp_path)
    h.broker.fill_price = 101.37
    bars = make_buy_bars(NOW)
    h.feed.frame = bars
    decision = TrendStrategy(h.strategy_cfg).generate_signal(bars)
    assert decision.action == BotAction.BUY
    expected_stop = 101.37 - decision.atr * h.strategy_cfg.atr_multiplier

    h.engine._startup()
    h.engine.run_iteration()

    stops = submitted_by_purpose(h, OrderPurpose.STOP.value)
    entries = submitted_by_purpose(h, OrderPurpose.ENTRY.value)
    assert len(stops) == 1
    assert stops[0].side == "sell"
    assert stops[0].stop_price == pytest.approx(expected_stop)
    assert stops[0].requested_qty == entries[0].requested_qty

    pos = h.store.get_bot_position(SYMBOL)
    assert pos["stop_price"] == pytest.approx(expected_stop)
    assert pos["atr_at_entry"] == pytest.approx(decision.atr)
    assert pos["stop_client_order_id"] == stops[0].client_order_id


# ---------------------------------------------------------------------------
# 11-15: halts, guards and failure modes
# ---------------------------------------------------------------------------


def test_daily_loss_halts_session_and_arms_kill_switch(tmp_path):
    """A session loss beyond max_daily_loss_pct halts the session before
    any trading and leaves the kill switch armed after finalize."""
    h = build_harness(tmp_path)
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    loss_fraction = 1.0 - h.risk_cfg.max_daily_loss_pct / 100.0
    breach_equity = 100_000.0 * loss_fraction - 500.0  # clearly past the limit
    h.broker.account = replace(h.broker.account, equity=breach_equity)
    h.engine.run_iteration()

    assert h.engine._stop_reason == StopReason.DAILY_LOSS_HALT
    assert h.broker.submitted == []  # halted before any trading decision
    session = h.store.get_session(h.engine.session_id)
    assert session["halted"] is True
    assert session["halt_reason"]

    h.engine._finalize()
    assert h.store.get_kv(trade_engine.KILL_SWITCH_KEY) is not None
    session = h.store.get_session(h.engine.session_id)
    assert session["stop_reason"] == StopReason.DAILY_LOSS_HALT.value
    assert h.engine.status.state == EngineState.HALTED


def test_market_closed_skips_data_and_orders(tmp_path):
    h = build_harness(tmp_path, market_open=False)
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()
    assert h.feed.calls == 0
    assert h.broker.submitted == []
    assert h.engine.status.market_open is False


def test_runtime_expiration_stops_engine(tmp_path):
    h = build_harness(
        tmp_path,
        engine_cfg=EngineConfig(poll_interval_seconds=5, max_runtime_seconds=10),
    )
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.now.now = NOW + timedelta(seconds=60)  # well past the 10s budget
    h.engine.run_iteration()
    assert h.engine._stop_reason == StopReason.RUNTIME_EXPIRED
    assert h.engine._stop_event.is_set()
    assert h.broker.submitted == []
    h.engine._finalize()
    session = h.store.get_session(h.engine.session_id)
    assert session["stop_reason"] == StopReason.RUNTIME_EXPIRED.value
    assert h.engine.status.state == EngineState.STOPPED
    # Runtime expiry is a clean stop: no kill switch.
    assert h.store.get_kv(trade_engine.KILL_SWITCH_KEY) is None


def test_transient_data_failure_skips_cycle_then_recovers(tmp_path):
    """DataFeedError inside get_recent_bars is absorbed by _fetch_bars;
    the iteration completes and a later good cycle still trades."""
    h = build_harness(tmp_path)
    h.feed.error = DataFeedError("transient blip")
    h.engine._startup()
    h.engine.run_iteration()  # must not raise
    assert h.feed.calls == 1
    assert h.broker.submitted == []

    h.feed.error = None
    h.feed.frame = make_buy_bars(NOW)
    h.engine.run_iteration()
    assert len(submitted_by_purpose(h, OrderPurpose.ENTRY.value)) == 1


def test_fatal_error_in_loop_sets_error_state_and_kill_switch(tmp_path):
    """A non-BrokerError exception in the threaded loop is fatal: the
    engine ends in ERROR with FATAL_ERROR persisted and the kill switch set."""
    h = build_harness(tmp_path, market_open=False)
    h.broker.get_account_error = RuntimeError("boom: internal invariant broken")
    h.engine.start()
    h.engine.join(10)
    assert not h.engine.is_running
    status = h.engine.status
    assert status.state == EngineState.ERROR
    assert status.stop_reason == StopReason.FATAL_ERROR
    session = h.store.get_session(h.engine.session_id)
    assert session["stop_reason"] == StopReason.FATAL_ERROR.value
    raw = h.store.get_kv(trade_engine.KILL_SWITCH_KEY)
    assert raw is not None
    assert json.loads(raw)["reason"] == StopReason.FATAL_ERROR.value


# ---------------------------------------------------------------------------
# 16-17: startup safety gates
# ---------------------------------------------------------------------------


def test_paper_verification_failure_blocks_startup(tmp_path):
    h = build_harness(tmp_path)
    h.broker.verify_error = BrokerError("not pointed at the paper endpoint")
    h.feed.frame = make_buy_bars(NOW)
    h.engine.start()
    h.engine.join(10)
    assert not h.engine.is_running
    assert h.engine.status.state == EngineState.ERROR
    assert h.engine.status.stop_reason == StopReason.FATAL_ERROR
    assert h.broker.submitted == []


def test_unknown_preexisting_position_blocks_startup(tmp_path):
    """A broker position the bot did not open (no bot_position record)
    must fail startup — the bot only manages positions it created."""
    h = build_harness(tmp_path)
    h.broker.positions[SYMBOL] = BrokerPosition(
        symbol=SYMBOL,
        qty=25.0,
        avg_entry_price=90.0,
        current_price=91.0,
        market_value=2275.0,
        unrealized_pl=25.0,
        unrealized_plpc=0.011,
    )
    h.feed.frame = make_buy_bars(NOW)
    h.engine.start()
    h.engine.join(10)
    assert h.engine.status.state == EngineState.ERROR
    assert h.broker.submitted == []
    assert "did not open" in (h.engine.status.error or "")


# ---------------------------------------------------------------------------
# 18-20: PDT, near-close and clock-skew protections
# ---------------------------------------------------------------------------


def test_pdt_restriction_blocks_intraday_entry(tmp_path):
    """Under $25k equity with 3 day trades already used, an intraday BUY
    must be blocked with reason pdt_restriction."""
    account = AccountInfo(
        equity=10_000.0,
        cash=10_000.0,
        buying_power=10_000.0,
        status="ACTIVE",
        currency="USD",
        pattern_day_trader=False,
        daytrade_count=3,
        trading_blocked=False,
    )
    assert account.equity < trade_engine.PDT_EQUITY_THRESHOLD
    h = build_harness(tmp_path, timeframe="5Min", account=account)
    h.feed.frame = make_buy_bars(NOW, timeframe="5Min")
    h.engine._startup()
    h.engine.run_iteration()
    assert h.broker.submitted == []
    signal = last_signal(h)
    assert signal["action"] == BotAction.BUY.value
    assert signal["details"]["entry_block"] == "pdt_restriction"


def test_entry_blocked_near_market_close(tmp_path):
    """With 10 minutes to the close and the default 30-minute buffer, a
    BUY signal is blocked as too_close_to_close."""
    h = build_harness(tmp_path)
    h.broker.clock = MarketClock(
        timestamp=NOW,
        is_open=True,
        next_open=NOW + timedelta(hours=18),
        next_close=NOW + timedelta(minutes=10),
    )
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()
    assert h.broker.submitted == []
    assert last_signal(h)["details"]["entry_block"] == "too_close_to_close"


def test_clock_skew_blocks_trading_with_warning(tmp_path):
    """A 120s disagreement with the broker clock (threshold 30s) blocks
    the trading decision entirely and surfaces a status warning."""
    h = build_harness(tmp_path)
    h.broker.clock = MarketClock(
        timestamp=NOW + timedelta(seconds=120),
        is_open=True,
        next_open=NOW + timedelta(hours=18),
        next_close=NOW + timedelta(hours=3),
    )
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()
    assert "clock_skew_detected" in h.engine.status.warnings
    assert h.feed.calls == 0  # blocked before ever fetching data
    assert h.broker.submitted == []


# ---------------------------------------------------------------------------
# 21: backtest/live decision parity (spec D5)
# ---------------------------------------------------------------------------


def test_backtest_and_live_engine_share_identical_decisions(tmp_path, monkeypatch):
    """The exact same strategy fed the same completed bars must produce
    the identical (timestamp, action) sequence in run_backtest and in the
    live engine consuming the series one bar at a time.  The backtest
    stops evaluating one bar earlier (it needs bar N+1 to execute), so
    the engine's final decision is trimmed before comparing."""
    bars = make_parity_bars()
    records: list[tuple] = []
    original = TrendStrategy.generate_signal

    def recording(self, price_data):
        decision = original(self, price_data)
        records.append((price_data.index[-1], decision.action))
        return decision

    monkeypatch.setattr(TrendStrategy, "generate_signal", recording)

    h = build_harness(tmp_path)
    warmup = h.strategy_cfg.warmup_bars()

    result = run_backtest(bars, h.strategy_cfg, h.risk_cfg, BacktestConfig())
    backtest_seq = list(records)
    records.clear()
    assert result.num_trades == 1  # the BUY/EXIT pair closed one round-trip

    # Live engine: deliver growing prefixes, advancing the wall clock so
    # each prefix's last bar is freshly completed (never stale) and the
    # broker clock always agrees (no skew).
    advance_time(h, bars.index[warmup - 1].to_pydatetime() + timedelta(hours=25))
    h.engine._startup()
    for k in range(warmup, len(bars) + 1):
        prefix = bars.iloc[:k]
        advance_time(h, prefix.index[-1].to_pydatetime() + timedelta(hours=25))
        h.feed.frame = prefix
        h.engine.run_iteration()
    engine_seq = list(records)

    assert len(backtest_seq) == len(bars) - warmup
    assert len(engine_seq) == len(bars) - warmup + 1
    assert backtest_seq == engine_seq[: len(backtest_seq)]

    actions = [action for _, action in backtest_seq]
    buy_index = actions.index(BotAction.BUY)
    assert BotAction.EXIT in actions[buy_index + 1:]
    # The engine really traded the sequence, not just observed it.
    assert len(submitted_by_purpose(h, OrderPurpose.ENTRY.value)) == 1
    assert len(submitted_by_purpose(h, OrderPurpose.EXIT.value)) == 1
    assert h.store.get_bot_position(SYMBOL) is None  # exited by the end


# ---------------------------------------------------------------------------
# 22: exit flow safety
# ---------------------------------------------------------------------------


def test_exit_cancels_stop_before_selling_and_books_pnl(tmp_path):
    """On a bearish crossover the engine must cancel its protective stop
    FIRST (so the exit can never combine with the stop into a short),
    then market-sell; the fill clears the bot position and records a
    realized-P&L SELL trade."""
    h = build_harness(tmp_path)
    h.broker.fill_price = 100.0
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()

    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None
    stop_cid = pos["stop_client_order_id"]
    assert h.broker.orders[stop_cid].status == "new"
    qty = pos["qty"]

    exit_now = NOW + timedelta(days=3)
    advance_time(h, exit_now)
    h.broker.fill_price = 97.0
    h.feed.frame = make_exit_bars(exit_now)
    h.engine.run_iteration()

    exits = submitted_by_purpose(h, OrderPurpose.EXIT.value)
    assert len(exits) == 1
    assert exits[0].side == "sell"
    assert exits[0].requested_qty == qty

    cancel_index = h.broker.events.index(("cancel", stop_cid))
    sell_index = h.broker.events.index(("submit_market", exits[0].client_order_id))
    assert cancel_index < sell_index  # stop canceled BEFORE the exit sell
    assert h.broker.orders[stop_cid].status == "canceled"

    assert h.store.get_bot_position(SYMBOL) is None
    sells = [t for t in h.store.get_trades() if t["action"] == "SELL"]
    assert len(sells) == 1
    assert sells[0]["fill_price"] == pytest.approx(97.0)
    assert sells[0]["realized_pnl"] == pytest.approx((97.0 - 100.0) * qty)


# ---------------------------------------------------------------------------
# Regression tests for crash-window findings from the adversarial review:
# lost HTTP responses, deferred exits, and fills that complete while the
# process is down must never produce duplicate live orders, untracked
# sells, dropped exits, or a refused restart of the bot's own position.
# ---------------------------------------------------------------------------


class LostResponseBroker(FakeBroker):
    """FakeBroker where a submit can succeed AT THE BROKER but the HTTP
    response is lost (the call raises after the order was accepted)."""

    def __init__(self, now: datetime = NOW):
        super().__init__(now=now)
        self.lose_next_stop_response = False
        self.lose_next_sell_response = False

    def submit_stop_order(self, symbol, qty, stop_price, client_order_id):
        record = super().submit_stop_order(symbol, qty, stop_price, client_order_id)
        if self.lose_next_stop_response:
            self.lose_next_stop_response = False
            raise BrokerError("connection dropped after the stop was accepted")
        return record

    def submit_market_order(self, symbol, qty, side, client_order_id):
        record = super().submit_market_order(symbol, qty, side, client_order_id)
        if self.lose_next_sell_response and side == "sell":
            self.lose_next_sell_response = False
            raise BrokerError("connection dropped after the sell was accepted")
        return record


class FlakyCancelBroker(FakeBroker):
    """FakeBroker whose cancel_order fails a set number of times first."""

    def __init__(self, now: datetime = NOW):
        super().__init__(now=now)
        self.cancel_failures_remaining = 0

    def cancel_order(self, broker_order_id):
        if self.cancel_failures_remaining > 0:
            self.cancel_failures_remaining -= 1
            raise BrokerError("cancel temporarily unavailable")
        super().cancel_order(broker_order_id)


def _swap_broker(h: Harness, broker_cls):
    """Replace the harness broker with a stateful subclass, preserving clock."""
    new_broker = broker_cls(now=h.now.now)
    new_broker.clock = h.broker.clock
    h.broker = new_broker
    h.engine.broker = new_broker
    return new_broker


def test_lost_stop_submit_response_never_arms_second_stop(tmp_path):
    """Finding: a stop submit whose response is lost must be adopted from
    broker truth on the next cycle, not duplicated (two resting sells for
    2x the position could turn the account short)."""
    h = build_harness(tmp_path)
    broker = _swap_broker(h, LostResponseBroker)
    h.feed.frame = make_buy_bars(NOW)
    broker.lose_next_stop_response = True

    h.engine._startup()
    h.engine.run_iteration()  # entry fills; stop submit "fails" after acceptance

    # The stop exists at the broker even though the engine saw an error.
    stops_at_broker = [o for o in broker.orders.values() if o.purpose == "stop"]
    assert len(stops_at_broker) == 1

    # Next cycles reconcile the persisted intent against broker truth.
    h.engine.run_iteration()
    h.engine.run_iteration()

    assert len(submitted_by_purpose(h, OrderPurpose.STOP.value)) == 1
    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None
    assert pos["stop_client_order_id"] == stops_at_broker[0].client_order_id


def test_lost_exit_submit_response_is_healed_not_shadowed_by_new_stop(tmp_path):
    """Finding: a sell whose response is lost must be re-tracked from the
    persisted order id; meanwhile no new protective stop may be armed
    under the live sell (stop + filled sell would go short)."""
    h = build_harness(tmp_path)
    broker = _swap_broker(h, LostResponseBroker)
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()  # position + stop armed
    assert len(submitted_by_purpose(h, OrderPurpose.STOP.value)) == 1

    exit_now = NOW + timedelta(days=3)
    advance_time(h, exit_now)
    broker.clock = h.broker.clock
    broker.fill_price = 97.0
    broker.lose_next_sell_response = True
    h.feed.frame = make_exit_bars(exit_now)
    with pytest.raises(BrokerError):
        h.engine.run_iteration()  # sell accepted+filled at broker, response lost

    # The engine lost track in memory but persisted the order id first.
    assert h.engine._pending_exit is None
    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None and pos["exit_client_order_id"]

    h.engine.run_iteration()  # heals from broker truth

    assert h.store.get_bot_position(SYMBOL) is None
    assert len(submitted_by_purpose(h, OrderPurpose.STOP.value)) == 1  # no new stop
    sells = [t for t in h.store.get_trades() if t["action"] == "SELL"]
    assert len(sells) == 1
    assert sells[0]["realized_pnl"] is not None


def test_deferred_exit_is_retried_until_it_completes(tmp_path):
    """Finding: an EXIT whose stop-cancel fails must not be dropped just
    because the bar was already consumed — the persisted intent retries it
    on later cycles, and no stop is re-armed while it is pending."""
    h = build_harness(tmp_path)
    broker = _swap_broker(h, FlakyCancelBroker)
    h.feed.frame = make_buy_bars(NOW)
    h.engine._startup()
    h.engine.run_iteration()  # position + stop armed
    stops_before = len(submitted_by_purpose(h, OrderPurpose.STOP.value))

    exit_now = NOW + timedelta(days=3)
    advance_time(h, exit_now)
    broker.clock = h.broker.clock
    broker.fill_price = 97.0
    broker.cancel_failures_remaining = 1
    h.feed.frame = make_exit_bars(exit_now)
    h.engine.run_iteration()  # cancel fails; exit deferred, intent persisted

    assert submitted_by_purpose(h, OrderPurpose.EXIT.value) == []
    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None and pos["exit_pending_reason"] == "bearish_crossover"

    h.engine.run_iteration()  # same bar (already processed) — retry fires anyway

    exits = submitted_by_purpose(h, OrderPurpose.EXIT.value)
    assert len(exits) == 1
    assert h.store.get_bot_position(SYMBOL) is None
    assert len(submitted_by_purpose(h, OrderPurpose.STOP.value)) == stops_before
    sells = [t for t in h.store.get_trades() if t["action"] == "SELL"]
    assert len(sells) == 1
    assert sells[0]["reason"] == "bearish_crossover"


def test_restart_adopts_position_from_own_entry_filled_while_down(tmp_path):
    """Finding: an entry that fills while the process is down must be
    adopted at restart (the orders table proves ownership) instead of
    blocking the session and leaving the position unprotected."""
    h = build_harness(tmp_path)
    entry_ts = NOW - timedelta(days=1)
    cid = make_client_order_id("deadbeefcafe", OrderPurpose.ENTRY.value, entry_ts, symbol=SYMBOL)
    filled_entry = OrderRecord(
        broker_order_id="prior-1",
        client_order_id=cid,
        symbol=SYMBOL,
        purpose=OrderPurpose.ENTRY.value,
        side="buy",
        requested_qty=5.0,
        filled_qty=5.0,
        status="filled",
        fill_price=101.0,
        stop_price=None,
        submitted_at=entry_ts,
        updated_at=entry_ts,
    )
    h.broker.orders[cid] = filled_entry
    h.broker.positions[SYMBOL] = BrokerPosition(
        symbol=SYMBOL, qty=5.0, avg_entry_price=101.0, current_price=101.0,
        market_value=505.0, unrealized_pl=0.0, unrealized_plpc=0.0,
    )
    h.store.record_order(filled_entry, "prior-session")
    h.store.set_kv(
        trade_engine.PENDING_ENTRY_KEY_PREFIX + SYMBOL,
        json.dumps({"client_order_id": cid, "atr": 2.5, "signal": None}),
    )

    h.engine._startup()  # must adopt, not raise

    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None
    assert pos["qty"] == pytest.approx(5.0)
    assert pos["entry_price"] == pytest.approx(101.0)
    assert pos["atr_at_entry"] == pytest.approx(2.5)
    buys = [t for t in h.store.get_trades() if t["action"] == "BUY"]
    assert any(t["reason"] == "entry_fill_recovered_at_restart" for t in buys)

    h.feed.frame = make_buy_bars(NOW)
    h.engine.run_iteration()  # arms the protective stop for the adopted position

    stops = submitted_by_purpose(h, OrderPurpose.STOP.value)
    assert len(stops) == 1
    expected_stop = 101.0 - 2.5 * h.strategy_cfg.atr_multiplier
    assert stops[0].stop_price == pytest.approx(expected_stop)
    assert stops[0].requested_qty == pytest.approx(5.0)


def test_missing_atr_at_entry_falls_back_to_latest_signal_atr(tmp_path):
    """Finding: a recovered position with no persisted ATR context must
    still get a protective stop (from the latest computed ATR) instead of
    running unprotected."""
    h = build_harness(tmp_path)
    h.broker.positions[SYMBOL] = BrokerPosition(
        symbol=SYMBOL, qty=4.0, avg_entry_price=100.0, current_price=100.0,
        market_value=400.0, unrealized_pl=0.0, unrealized_plpc=0.0,
    )
    h.store.set_bot_position(
        SYMBOL,
        {
            "qty": 4,
            "entry_price": 100.0,
            "atr_at_entry": None,
            "stop_price": None,
            "stop_client_order_id": None,
            "session_id": h.engine.session_id,
            "entry_time": (NOW - timedelta(days=1)).isoformat(),
        },
    )
    with h.engine._lock:
        h.engine._status.last_signal = SignalDecision(
            action=BotAction.HOLD, timestamp=NOW, reason="no_crossover", atr=3.0
        )

    h.engine._ensure_protective_stop(NOW)

    stops = submitted_by_purpose(h, OrderPurpose.STOP.value)
    assert len(stops) == 1
    assert stops[0].stop_price == pytest.approx(100.0 - 3.0 * h.strategy_cfg.atr_multiplier)


# ---------------------------------------------------------------------------
# Independent audit regressions: partial fills and broker-truth drift
# ---------------------------------------------------------------------------


def _inject_partial_entry(h: Harness, filled_qty: float = 40.0) -> OrderRecord:
    cid = make_client_order_id(
        h.engine.session_id, OrderPurpose.ENTRY.value, NOW, symbol=SYMBOL
    )
    order = OrderRecord(
        broker_order_id="partial-entry",
        client_order_id=cid,
        symbol=SYMBOL,
        purpose=OrderPurpose.ENTRY.value,
        side="buy",
        requested_qty=100.0,
        filled_qty=filled_qty,
        status="partially_filled",
        fill_price=100.0,
        stop_price=None,
        submitted_at=NOW,
        updated_at=NOW,
    )
    h.broker.orders[cid] = order
    h.broker.positions[SYMBOL] = BrokerPosition(
        symbol=SYMBOL, qty=filled_qty, avg_entry_price=100.0, current_price=100.0,
        market_value=filled_qty * 100.0, unrealized_pl=0.0, unrealized_plpc=0.0,
    )
    h.store.record_order(order, h.engine.session_id)
    context = {"client_order_id": cid, "atr": 2.0, "signal": {"reason": "audit"}}
    h.store.set_kv(trade_engine.PENDING_ENTRY_KEY_PREFIX + SYMBOL, json.dumps(context))
    h.engine._pending_entry = context
    return order


def test_partial_entry_is_recorded_protected_and_remainder_canceled(tmp_path):
    h = build_harness(tmp_path)
    h.engine._startup()
    entry = _inject_partial_entry(h)

    h.engine._reconcile_entry_order(NOW)

    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None and pos["qty"] == pytest.approx(40.0)
    assert h.broker.orders[entry.client_order_id].status == "canceled"
    stops = submitted_by_purpose(h, OrderPurpose.STOP.value)
    assert len(stops) == 1
    assert stops[0].requested_qty == pytest.approx(40.0)
    assert len([t for t in h.store.get_trades() if t["action"] == "BUY"]) == 1


def test_additional_fill_during_entry_cancel_resizes_stop_to_final_quantity(
    tmp_path, monkeypatch
):
    h = build_harness(tmp_path)
    h.engine._startup()
    entry = _inject_partial_entry(h)
    original_cancel = h.broker.cancel_order

    def fill_then_cancel(broker_order_id: str):
        record = next(
            o for o in h.broker.orders.values() if o.broker_order_id == broker_order_id
        )
        if record.client_order_id == entry.client_order_id:
            record.filled_qty = 70.0
            h.broker.positions[SYMBOL].qty = 70.0
            h.broker.positions[SYMBOL].market_value = 7_000.0
        original_cancel(broker_order_id)

    monkeypatch.setattr(h.broker, "cancel_order", fill_then_cancel)
    h.engine._reconcile_entry_order(NOW)

    assert h.store.get_bot_position(SYMBOL)["qty"] == pytest.approx(70.0)
    open_stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(open_stops) == 1
    assert open_stops[0].requested_qty == pytest.approx(70.0)


def test_partial_exit_cancels_remainder_and_rearms_only_remaining_quantity(tmp_path):
    h = build_harness(tmp_path)
    stop_id = preset_bot_position(h, qty=100)
    h.engine._startup()
    h.broker.orders[stop_id].status = "canceled"
    pos = h.store.get_bot_position(SYMBOL)
    pos["stop_client_order_id"] = None
    pos["stop_price"] = None

    cid = make_client_order_id(h.engine.session_id, "exit", NOW, symbol=SYMBOL)
    exit_order = OrderRecord(
        "partial-exit", cid, SYMBOL, "exit", "sell", 100.0, 40.0,
        "partially_filled", 99.0, submitted_at=NOW, updated_at=NOW,
    )
    h.broker.orders[cid] = exit_order
    h.broker.positions[SYMBOL].qty = 60.0
    h.broker.positions[SYMBOL].market_value = 5_940.0
    pos.update(
        exit_client_order_id=cid,
        exit_pending_reason="bearish_crossover",
        exit_bar_epoch=int(NOW.timestamp()),
        exit_attempts=1,
    )
    h.store.set_bot_position(SYMBOL, pos)
    h.engine._pending_exit = {"client_order_id": cid, "reason": "bearish_crossover"}

    h.engine._reconcile_exit_order(NOW)

    remaining = h.store.get_bot_position(SYMBOL)
    assert remaining is not None and remaining["qty"] == pytest.approx(60.0)
    open_stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(open_stops) == 1
    assert open_stops[0].requested_qty == pytest.approx(60.0)


def test_partial_stop_fill_keeps_one_exact_remaining_stop(tmp_path):
    h = build_harness(tmp_path)
    stop_id = preset_bot_position(h, qty=100)
    h.engine._startup()
    stop = h.broker.orders[stop_id]
    stop.status = "partially_filled"
    stop.filled_qty = 40.0
    stop.fill_price = 96.0
    h.broker.positions[SYMBOL].qty = 60.0

    h.engine._reconcile_stop_order(NOW)

    assert h.store.get_bot_position(SYMBOL)["qty"] == pytest.approx(60.0)
    assert h.store.get_bot_position(SYMBOL)["stop_client_order_id"] == stop_id
    assert len([o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]) == 1


def test_stop_partial_fill_during_strategy_exit_sells_only_remaining_shares(
    tmp_path, monkeypatch
):
    h = build_harness(tmp_path)
    stop_id = preset_bot_position(h, qty=100)
    h.engine._startup()
    original_cancel = h.broker.cancel_order

    def partial_fill_then_cancel(broker_order_id: str):
        stop = h.broker.orders[stop_id]
        if broker_order_id == stop.broker_order_id:
            stop.filled_qty = 40.0
            stop.fill_price = 96.0
            h.broker._apply_fill(SYMBOL, 40.0, "sell", 96.0)
        original_cancel(broker_order_id)

    monkeypatch.setattr(h.broker, "cancel_order", partial_fill_then_cancel)
    h.broker.fill_price = 95.0
    h.engine._initiate_exit("bearish_crossover", NOW)

    exits = submitted_by_purpose(h, OrderPurpose.EXIT.value)
    assert len(exits) == 1 and exits[0].requested_qty == pytest.approx(60.0)
    assert h.broker.get_position(SYMBOL) is None
    assert h.store.get_bot_position(SYMBOL) is None
    assert not [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]


def test_rejected_exit_uses_new_bounded_retry_ids_then_rearms_stop(tmp_path):
    h = build_harness(tmp_path)
    preset_bot_position(h, qty=10)
    h.engine._startup()
    h.broker.reject_submits = True

    h.engine._initiate_exit("bearish_crossover", NOW)
    h.engine._retry_exit_intent(NOW)
    h.engine._retry_exit_intent(NOW)

    exits = submitted_by_purpose(h, OrderPurpose.EXIT.value)
    assert len(exits) == 3
    assert len({o.client_order_id for o in exits}) == 3
    assert all(len(o.client_order_id) <= 48 for o in exits)
    h.broker.reject_submits = False
    h.engine._retry_exit_intent(NOW)
    h.engine._reconcile_stop_order(NOW)
    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None and pos["exit_pending_reason"] is None
    assert any(o.is_open and o.purpose == "stop" for o in h.broker.orders.values())


def test_daily_loss_halt_reconciles_fill_and_stop_first(tmp_path):
    h = build_harness(tmp_path, market_open=False)
    h.engine._startup()
    entry = _inject_partial_entry(h, filled_qty=40.0)
    entry.status = "filled"
    entry.requested_qty = 40.0
    h.broker.orders[entry.client_order_id] = entry
    h.broker.account = replace(h.broker.account, equity=97_000.0)

    h.engine.run_iteration()

    assert h.engine._stop_reason == StopReason.DAILY_LOSS_HALT
    assert h.store.get_bot_position(SYMBOL)["qty"] == pytest.approx(40.0)
    assert any(o.is_open and o.purpose == "stop" for o in h.broker.orders.values())


def test_startup_cancels_oversized_stop_and_rearms_exact_quantity(tmp_path):
    h = build_harness(tmp_path, market_open=False)
    stop_id = preset_bot_position(h, qty=10)
    h.broker.orders[stop_id].requested_qty = 100.0

    h.engine._startup()
    h.engine.run_iteration()

    assert h.broker.orders[stop_id].status == "canceled"
    open_stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(open_stops) == 1
    assert open_stops[0].requested_qty == pytest.approx(10.0)


def test_runtime_manual_reduction_resizes_oversized_stop(tmp_path):
    h = build_harness(tmp_path, market_open=False)
    old_stop = preset_bot_position(h, qty=10)
    h.engine._startup()
    h.broker.positions[SYMBOL].qty = 4.0

    h.engine.run_iteration()

    assert h.broker.orders[old_stop].status == "canceled"
    assert h.store.get_bot_position(SYMBOL)["qty"] == pytest.approx(4.0)
    open_stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(open_stops) == 1 and open_stops[0].requested_qty == pytest.approx(4.0)


def test_stale_historical_fill_without_exact_pending_intent_is_not_adopted(tmp_path):
    h = build_harness(tmp_path)
    old_cid = make_client_order_id("oldsession", "entry", NOW, symbol=SYMBOL)
    old = OrderRecord(
        "old", old_cid, SYMBOL, "entry", "buy", 5.0, 5.0, "filled", 90.0,
        submitted_at=NOW, updated_at=NOW,
    )
    h.broker.orders[old_cid] = old
    h.store.record_order(old, "old-session")
    h.broker.positions[SYMBOL] = BrokerPosition(
        SYMBOL, 5.0, 110.0, 110.0, 550.0, 0.0, 0.0
    )

    with pytest.raises(BrokerError, match="did not open"):
        h.engine._startup()
    assert h.store.get_bot_position(SYMBOL) is None


def test_running_engine_owns_immutable_configuration_snapshot(tmp_path):
    h = build_harness(tmp_path)
    original_symbol = h.strategy_cfg.symbol
    original_risk = h.risk_cfg.risk_per_trade_pct
    h.strategy_cfg.symbol = "CHANGED"
    h.risk_cfg.risk_per_trade_pct = 9.0
    h.engine_cfg.close_buffer_minutes = 1

    assert h.engine.symbol == original_symbol
    assert h.engine.strategy.config.symbol == original_symbol
    assert h.engine.risk_manager.config.risk_per_trade_pct == original_risk
    assert h.engine.engine_config.close_buffer_minutes == 30


# ---------------------------------------------------------------------------
# Third-audit regressions: coverage holes found by mutation testing, plus a
# newly discovered pending-entry/position-truth race.
# ---------------------------------------------------------------------------


def test_position_visible_before_entry_order_reports_fill_is_not_fatal(tmp_path):
    """A market entry can show up as a broker position before the order
    object reports the fill.  With a live pending-entry intent the engine
    must defer to order reconciliation (and then protect the fill), never
    fatal-halt with an unprotected position."""
    h = build_harness(tmp_path)
    h.engine._startup()
    cid = make_client_order_id(
        h.engine.session_id, OrderPurpose.ENTRY.value, NOW, symbol=SYMBOL
    )
    order = OrderRecord(
        broker_order_id="race-entry", client_order_id=cid, symbol=SYMBOL,
        purpose="entry", side="buy", requested_qty=10.0, filled_qty=0.0,
        status="accepted", submitted_at=NOW, updated_at=NOW,
    )
    h.broker.orders[cid] = order
    h.store.record_order(order, h.engine.session_id)
    context = {"client_order_id": cid, "atr": 2.0, "signal": {"reason": "race"}}
    h.store.set_kv(trade_engine.PENDING_ENTRY_KEY_PREFIX + SYMBOL, json.dumps(context))
    h.engine._pending_entry = context
    h.broker.positions[SYMBOL] = BrokerPosition(SYMBOL, 10.0, 100.0, 100.0, 1_000.0, 0.0, 0.0)

    h.engine.run_iteration()

    assert h.engine.status.state != EngineState.ERROR
    assert h.engine._stop_reason is None

    # Broker truth catches up; the next cycle adopts and protects the fill.
    order.filled_qty = 10.0
    order.status = "filled"
    order.fill_price = 100.0
    h.engine.run_iteration()

    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None and pos["qty"] == pytest.approx(10.0)
    stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(stops) == 1 and stops[0].requested_qty == pytest.approx(10.0)
    assert len([t for t in h.store.get_trades() if t["action"] == "BUY"]) == 1


def test_foreign_position_without_pending_entry_still_fatals(tmp_path):
    """The in-flight-entry deferral must not weaken the ownership boundary:
    a position with no pending entry intent remains a fatal halt."""
    h = build_harness(tmp_path)
    h.engine._startup()
    h.broker.positions[SYMBOL] = BrokerPosition(SYMBOL, 5.0, 90.0, 90.0, 450.0, 0.0, 0.0)

    h.engine.run_iteration()

    assert h.engine.status.state == EngineState.ERROR
    assert h.engine._stop_reason == StopReason.FATAL_ERROR
    assert h.store.get_bot_position(SYMBOL) is None


def test_linked_stop_with_wrong_quantity_is_resized_to_position(tmp_path):
    """_ensure_protective_stop must verify the linked stop's remaining
    quantity against the position instead of blindly trusting the link
    (mutation 'ensure-stop-blind-trust' survived the prior suite)."""
    h = build_harness(tmp_path, market_open=False)
    preset_bot_position(h, qty=70, stop_price=96.0)
    h.engine._startup()
    # AFTER startup, the link degrades to a 40-share stop (e.g. an
    # unconfirmed cancel during a partial-fill resize left the old order):
    # only the runtime linked-stop verification can catch this.
    small_cid = inject_resting_order(
        h, OrderPurpose.STOP.value, qty=40, stop_price=96.0,
        bar_ts=h.now.now - timedelta(days=3),
    )
    pos = h.store.get_bot_position(SYMBOL)
    old_cid = pos["stop_client_order_id"]
    del h.broker.orders[old_cid]
    pos["stop_client_order_id"] = small_cid
    h.store.set_bot_position(SYMBOL, pos)

    h.engine.run_iteration()

    assert h.broker.orders[small_cid].status == "canceled"
    open_stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(open_stops) == 1
    assert open_stops[0].requested_qty == pytest.approx(70.0)


def test_duplicate_resting_stops_collapse_to_one_exact_stop(tmp_path):
    """Two resting bot sell stops (e.g. a lost-response resubmit race) must
    collapse to exactly one stop of exactly the position quantity — two
    resting sells could otherwise oversell the account short."""
    h = build_harness(tmp_path, market_open=False)
    preset_bot_position(h, qty=70, stop_price=96.0)
    pos = h.store.get_bot_position(SYMBOL)
    exact_cid = pos["stop_client_order_id"]
    dup_cid = inject_resting_order(
        h, OrderPurpose.STOP.value, qty=40, stop_price=95.0,
        bar_ts=h.now.now - timedelta(days=3),
    )
    pos["stop_client_order_id"] = None
    pos["stop_price"] = None
    h.store.set_bot_position(SYMBOL, pos)
    h.engine._startup()

    h.engine.run_iteration()

    open_stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(open_stops) == 1
    assert open_stops[0].requested_qty == pytest.approx(70.0)
    assert h.broker.orders[dup_cid].status == "canceled"
    assert h.store.get_bot_position(SYMBOL)["stop_client_order_id"] == exact_cid


def test_startup_refuses_to_clear_position_when_orphan_stop_cancel_fails(tmp_path, monkeypatch):
    """If the broker position vanished but the leftover bot sell stop cannot
    be confirmed canceled, startup must refuse to proceed (a resting sell
    with no position could fill short).  Local ownership is preserved for
    the retry (mutation 'orphan-stop-ignore-failure' survived the suite)."""
    h = build_harness(tmp_path)
    stop_cid = preset_bot_position(h, qty=10)
    del h.broker.positions[SYMBOL]

    def failing_cancel(broker_order_id: str):
        raise BrokerError("simulated cancel outage")

    monkeypatch.setattr(h.broker, "cancel_order", failing_cancel)
    with pytest.raises(BrokerError, match="could not be confirmed canceled"):
        h.engine._startup()

    assert h.store.get_bot_position(SYMBOL) is not None
    assert h.broker.orders[stop_cid].is_open


def test_cancel_and_confirm_requires_terminal_state(tmp_path, monkeypatch):
    """_cancel_and_confirm must poll until the order can no longer fill;
    a cancel request alone (pending_cancel) is not confirmation (mutation
    'cancel-without-confirm' survived the suite)."""
    h = build_harness(tmp_path)
    h.engine._startup()
    cid = inject_resting_order(h, OrderPurpose.STOP.value, qty=10, stop_price=96.0)
    record = h.broker.orders[cid]

    def async_cancel(broker_order_id: str):
        assert broker_order_id == record.broker_order_id
        record.status = "pending_cancel"  # request accepted, not yet dead

    monkeypatch.setattr(h.broker, "cancel_order", async_cancel)
    assert h.engine._cancel_and_confirm(replace(record)) is False

    refreshes = {"n": 0}
    original_get = h.broker.get_order_by_client_id

    def eventually_canceled(client_order_id: str):
        refreshes["n"] += 1
        if refreshes["n"] >= 2:
            record.status = "canceled"
        return original_get(client_order_id)

    record.status = "new"
    monkeypatch.setattr(h.broker, "get_order_by_client_id", eventually_canceled)
    assert h.engine._cancel_and_confirm(replace(record)) is True
    assert record.status == "canceled"


def test_restart_adopts_entry_fill_even_when_crash_preceded_record_order(tmp_path):
    """Crash window: pending-entry KV is persisted, the submit reached
    Alpaca and filled, but the process died before record_order wrote the
    local orders row.  Restart must still adopt the fill via the exact
    pending client id and protect it — refusing to start would leave a
    filled position permanently stopless."""
    h = build_harness(tmp_path)
    cid = make_client_order_id(
        h.engine.session_id, OrderPurpose.ENTRY.value, NOW, symbol=SYMBOL
    )
    h.broker.orders[cid] = OrderRecord(
        broker_order_id="crash-entry", client_order_id=cid, symbol=SYMBOL,
        purpose="entry", side="buy", requested_qty=10.0, filled_qty=10.0,
        status="filled", fill_price=100.0, submitted_at=NOW, updated_at=NOW,
    )
    h.broker.positions[SYMBOL] = BrokerPosition(SYMBOL, 10.0, 100.0, 100.0, 1_000.0, 0.0, 0.0)
    h.store.set_kv(
        trade_engine.PENDING_ENTRY_KEY_PREFIX + SYMBOL,
        json.dumps({"client_order_id": cid, "atr": 2.0, "signal": {"reason": "crash"}}),
    )
    # NOTE: no h.store.record_order(...) — that is the crash window.

    h.engine._startup()

    pos = h.store.get_bot_position(SYMBOL)
    assert pos is not None and pos["qty"] == pytest.approx(10.0)
    assert len([t for t in h.store.get_trades() if t["action"] == "BUY"]) == 1

    h.engine.run_iteration()
    stops = [o for o in h.broker.list_open_orders(SYMBOL) if o.purpose == "stop"]
    assert len(stops) == 1 and stops[0].requested_qty == pytest.approx(10.0)
