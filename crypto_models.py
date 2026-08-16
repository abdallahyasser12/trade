"""Typed structures crossing crypto strategy, risk, backtest, engine, and UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd

from models import BotAction, EngineState, StopReason


class CryptoRegime(str, Enum):
    BULL_TREND = "BULL_TREND"
    RANGE = "RANGE"
    STRESS = "STRESS"
    UNKNOWN = "UNKNOWN"


@dataclass
class CryptoSignalDecision:
    symbol: str
    action: BotAction
    timestamp: Optional[datetime]
    reason: str
    regime: CryptoRegime = CryptoRegime.UNKNOWN
    score: Optional[float] = None
    close: Optional[float] = None
    atr: Optional[float] = None
    stop_price: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    efficiency_ratio: Optional[float] = None
    volume_ratio: Optional[float] = None
    atr_percentile: Optional[float] = None
    momentum: Optional[float] = None
    relative_strength: Optional[float] = None
    filters: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        data["regime"] = self.regime.value
        data["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return data


@dataclass(frozen=True)
class CryptoQuote:
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    @property
    def midpoint(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread_bps(self) -> float:
        midpoint = self.midpoint
        return (self.ask_price - self.bid_price) / midpoint * 10_000.0


@dataclass
class CryptoPositionSizeResult:
    qty: float
    risk_budget: float
    risk_per_unit: float
    quantity_by_risk: float
    quantity_by_position: float
    quantity_by_portfolio: float
    quantity_by_buying_power: float
    position_notional: float
    risk_scale: float
    quantity_by_heat: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def approved(self) -> bool:
        return self.qty > 0 and self.rejection_reason is None


@dataclass
class CryptoTradeRecord:
    symbol: str
    qty: float
    entry_time: datetime
    entry_price: float
    entry_fee: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_fee: float = 0.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    entry_reason: str = ""

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None and self.exit_price is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entry_time"] = self.entry_time.isoformat()
        data["exit_time"] = self.exit_time.isoformat() if self.exit_time else None
        return data


@dataclass
class CryptoBacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[CryptoTradeRecord]
    total_return_pct: float
    benchmark_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    profit_factor: Optional[float]
    win_rate: Optional[float]
    num_trades: int
    starting_equity: float
    ending_equity: float
    realized_pnl: float
    total_fees: float
    exposure_time_pct: float
    round_trip_cost_bps: float = 0.0
    median_cost_to_risk: Optional[float] = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoPositionState:
    symbol: str
    qty: float
    entry_price: float
    entry_time: str
    atr_at_entry: float
    stop_price: float
    highest_close: float
    stop_client_order_id: Optional[str] = None
    stop_broker_order_id: Optional[str] = None
    stop_qty: float = 0.0
    stop_attempt: int = 1
    pending_exit_client_order_id: Optional[str] = None
    breached_at: Optional[str] = None
    breach_confirmations: int = 0
    guard_attempt: int = 0
    # Attribution context, added after the fields above shipped. Optional
    # with defaults so state persisted by earlier builds loads unchanged
    # (from_dict drops unknown keys, missing keys take these defaults).
    # ``lowest_close`` tracks completed closes only, mirroring
    # ``highest_close`` — the adverse-excursion counterpart.
    initial_stop_price: Optional[float] = None
    lowest_close: Optional[float] = None
    entry_regime: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CryptoPositionState":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass
class CryptoEngineStatus:
    state: EngineState = EngineState.STOPPED
    session_id: Optional[str] = None
    account_mode: str = "paper"
    symbols: list[str] = field(default_factory=list)
    timeframe: Optional[str] = None
    started_at: Optional[datetime] = None
    intended_stop_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    last_bar_time: Optional[datetime] = None
    last_signals: dict[str, CryptoSignalDecision] = field(default_factory=dict)
    stop_reason: Optional[StopReason] = None
    halt_reason: Optional[str] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        iso = lambda value: value.isoformat() if value else None
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "account_mode": self.account_mode,
            "symbols": list(self.symbols),
            "timeframe": self.timeframe,
            "started_at": iso(self.started_at),
            "intended_stop_at": iso(self.intended_stop_at),
            "last_heartbeat": iso(self.last_heartbeat),
            "last_bar_time": iso(self.last_bar_time),
            "last_signals": {
                symbol: decision.to_dict() for symbol, decision in self.last_signals.items()
            },
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "halt_reason": self.halt_reason,
            "error": self.error,
            "warnings": list(self.warnings),
        }
