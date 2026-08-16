"""Shared typed data structures for the trading bot.

Every structure that crosses a module boundary (strategy -> engine,
engine -> dashboard, backtest -> dashboard, broker -> engine) is defined
here so the rest of the codebase can rely on one set of shapes.

This module intentionally has no imports from other project modules to
avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd


class BotAction(str, Enum):
    """The only actions the long-only strategy can request."""

    BUY = "BUY"
    EXIT = "EXIT"
    HOLD = "HOLD"


class StopReason(str, Enum):
    """Why a live engine session ended."""

    USER_STOP = "USER_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RUNTIME_EXPIRED = "RUNTIME_EXPIRED"
    DAILY_LOSS_HALT = "DAILY_LOSS_HALT"
    FATAL_ERROR = "FATAL_ERROR"


# Stop reasons that leave the kill switch armed: after one of these the UI
# must require an explicit user acknowledgement before the bot can start again.
UNCLEAN_STOP_REASONS = frozenset(
    {StopReason.EMERGENCY_STOP, StopReason.DAILY_LOSS_HALT, StopReason.FATAL_ERROR}
)


class EngineState(str, Enum):
    """Lifecycle state of the background trading engine."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    HALTED = "HALTED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class OrderPurpose(str, Enum):
    """What a bot-owned order is for.  Encoded into the client_order_id."""

    ENTRY = "entry"
    EXIT = "exit"
    STOP = "stop"


# Order statuses the engine treats as terminal (no further fills possible).
# ``done_for_day`` and ``calculated`` are deliberately not terminal: a GTC
# order completed for the current day can resume on the next trading day.
# A replaced order cannot receive new fills under its original order id.
TERMINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "replaced"}
)

# Order statuses that mean the order is still working at the broker.
OPEN_ORDER_STATUSES = frozenset(
    {
        "new",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "partially_filled",
        "pending_cancel",
        "pending_replace",
        "pending_review",
        "held",
        "calculated",
        "stopped",
        "suspended",
        "done_for_day",
    }
)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


@dataclass
class SignalDecision:
    """Full, serializable explanation of one strategy evaluation.

    ``filters`` maps filter name -> passed?, e.g.
    ``{"trend": True, "rsi": True, "volume": False, "volatility": True}``.
    ``details`` carries any extra context (blocking reasons, raw values).
    """

    action: BotAction
    timestamp: Optional[datetime]
    reason: str
    bullish_crossover: bool = False
    bearish_crossover: bool = False
    short_ma: Optional[float] = None
    long_ma: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    volume: Optional[float] = None
    average_volume: Optional[float] = None
    volume_ratio: Optional[float] = None
    atr_percentile: Optional[float] = None
    adx: Optional[float] = None
    efficiency_ratio: Optional[float] = None
    market_regime_bullish: Optional[bool] = None
    regime_close: Optional[float] = None
    regime_ma: Optional[float] = None
    relative_strength: Optional[float] = None
    risk_scale: Optional[float] = None
    filters: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        d["timestamp"] = _iso(self.timestamp)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SignalDecision":
        data = dict(d)
        data["action"] = BotAction(data["action"])
        ts = data.get("timestamp")
        data["timestamp"] = datetime.fromisoformat(ts) if ts else None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TradeRecord:
    """One completed (or still-open) round-trip trade in a backtest."""

    symbol: str
    qty: int
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    entry_reason: str = ""

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None and self.exit_price is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = _iso(self.entry_time)
        d["exit_time"] = _iso(self.exit_time)
        return d


@dataclass
class BacktestResult:
    """Typed result of a historical simulation.

    ``equity_curve`` and ``benchmark_curve`` are pandas Series indexed by
    bar timestamp, both normalized to start at ``starting_equity``.
    ``sharpe_ratio`` is None when there were too few return observations
    to be meaningful (see backtest.MIN_SHARPE_OBSERVATIONS).
    ``max_drawdown_pct`` is a positive magnitude, e.g. 12.5 means the
    equity fell at most 12.5% below its running peak.
    """

    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[TradeRecord]
    total_return_pct: float
    benchmark_return_pct: float
    win_rate: Optional[float]
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    sharpe_annualization_factor: float
    risk_free_rate: float
    num_trades: int
    starting_equity: float
    ending_equity: float
    realized_pnl: float
    avg_win: Optional[float]
    avg_loss: Optional[float]
    profit_factor: Optional[float]
    best_trade: Optional[float]
    worst_trade: Optional[float]
    config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSizeResult:
    """Outcome of risk-based position sizing.

    ``qty`` is 0 and ``rejection_reason`` is set when the entry must be
    rejected (zero size, below minimums, no buying power, ...).
    """

    qty: int
    shares_by_risk: int
    shares_by_exposure: int
    shares_by_buying_power: int
    position_value: float
    risk_amount: float
    rejection_reason: Optional[str] = None

    @property
    def approved(self) -> bool:
        return self.qty > 0 and self.rejection_reason is None


@dataclass
class OrderRecord:
    """Normalized view of one broker order the bot owns."""

    broker_order_id: str
    client_order_id: str
    symbol: str
    purpose: str
    side: str
    requested_qty: float
    filled_qty: float
    status: str
    fill_price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Price the decision was made against. Set on the locally-created intent;
    # broker-returned records leave it None and the store preserves the
    # original, so realized slippage stays measurable.
    reference_price: Optional[float] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_ORDER_STATUSES

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_ORDER_STATUSES

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["submitted_at"] = _iso(self.submitted_at)
        d["updated_at"] = _iso(self.updated_at)
        return d


@dataclass
class BrokerPosition:
    """Normalized view of one broker-side position."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: Optional[float]
    market_value: Optional[float]
    unrealized_pl: Optional[float]
    unrealized_plpc: Optional[float]
    # None means unknown; exposure gates treat unknown as crypto so they can
    # only fail conservative on a shared stock-and-crypto account.
    asset_class: Optional[str] = None


@dataclass
class AccountInfo:
    """Normalized view of the Alpaca paper account."""

    equity: float
    cash: float
    buying_power: float
    status: str
    currency: str
    pattern_day_trader: bool
    daytrade_count: int
    trading_blocked: bool

    @property
    def can_trade(self) -> bool:
        return self.status == "ACTIVE" and not self.trading_blocked


@dataclass
class MarketClock:
    """Alpaca market clock snapshot (all timestamps timezone-aware)."""

    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass
class AssetInfo:
    """Normalized view of an Alpaca asset record."""

    symbol: str
    name: str
    tradable: bool
    active: bool
    fractionable: bool
    asset_class: str
    min_order_size: Optional[float] = None
    min_trade_increment: Optional[float] = None
    price_increment: Optional[float] = None
    marginable: bool = False
    shortable: bool = False

    @property
    def is_supported(self) -> bool:
        return self.tradable and self.active and self.asset_class == "us_equity"

    @property
    def is_supported_crypto(self) -> bool:
        return (
            self.tradable
            and self.active
            and self.fractionable
            and self.asset_class == "crypto"
            and not self.marginable
            and not self.shortable
        )


@dataclass
class PositionSnapshot:
    """Dashboard-facing snapshot of the bot-managed position."""

    symbol: str
    qty: int
    avg_entry_price: float
    current_price: Optional[float]
    market_value: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pnl_pct: Optional[float]
    stop_price: Optional[float]
    risk_amount: Optional[float]
    risk_pct: Optional[float]
    bot_owned: bool = True


@dataclass
class EngineStatus:
    """Live engine status the dashboard renders.  Serializable."""

    state: EngineState = EngineState.STOPPED
    session_id: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    started_at: Optional[datetime] = None
    intended_stop_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    last_bar_time: Optional[datetime] = None
    last_signal: Optional[SignalDecision] = None
    last_order_status: Optional[str] = None
    stop_reason: Optional[StopReason] = None
    halt_reason: Optional[str] = None
    error: Optional[str] = None
    market_open: Optional[bool] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "started_at": _iso(self.started_at),
            "intended_stop_at": _iso(self.intended_stop_at),
            "last_heartbeat": _iso(self.last_heartbeat),
            "last_bar_time": _iso(self.last_bar_time),
            "last_signal": self.last_signal.to_dict() if self.last_signal else None,
            "last_order_status": self.last_order_status,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "halt_reason": self.halt_reason,
            "error": self.error,
            "market_open": self.market_open,
            "warnings": list(self.warnings),
        }
