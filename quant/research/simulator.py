"""Event-driven portfolio simulator for ``UniverseStrategy`` strategies.

This is the research twin of the production engines' execution semantics,
extended with the exit-intelligence layer (partials, break-even, time
stops, and the NORMAL -> STRONG_WINNER -> RUNNER machine) so every exit
feature can be measured before it is ever enabled in a deployed config.

Execution model — identical in spirit to ``crypto_backtest`` and audited
against the same conventions:

* decisions are made on COMPLETED bar N and execute at bar N+1's open;
* protective stops are evaluated against the next bar's open/low BEFORE
  any close-derived logic; a gap below the stop fills at the adverse open;
* an entry is protected on its own fill bar (a same-bar collapse through
  the stop exits at the stop, or at the open when the open itself gapped
  under it);
* management instructions produced at bar N's close (stop ratchets,
  partial exits, time stops) act no earlier than bar N+1;
* costs: both legs pay slippage plus half the spread as price adjustment,
  and a percentage fee on notional; the same ``cost_model`` arithmetic the
  live path uses.

Every rejected entry is counted with its reason — research inherits the
platform's "no silent rejections" rule.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

import pandas as pd

from cost_model import round_trip_cost_bps
from models import BotAction
from quant.exits import (
    ExitInstruction,
    ExitInstructionKind,
    ExitPolicyConfig,
    ManagedPosition,
    PositionPhase,
    TradeManagementPolicy,
)
from quant.strategies.base import UniverseStrategy

BPS = 10_000.0


@dataclass(frozen=True)
class SimulationConfig:
    """Cost, sizing, and portfolio assumptions for one simulation run."""

    initial_capital: float = 100_000.0
    risk_per_trade_pct: float = 0.5
    max_positions: int = 5
    max_total_exposure_pct: float = 60.0
    slippage_bps: float = 10.0
    spread_bps: float = 5.0
    fee_bps: float = 0.0
    allow_fractional: bool = True
    # Bars per year for Sharpe annualization — the caller sets this from
    # config.bars_per_year / crypto_config.crypto_bars_per_year so equity
    # and 24/7 markets are annualized honestly.
    annualization_factor: float = 252.0
    min_sharpe_observations: int = 30

    def validate(self) -> list[str]:
        errors: list[str] = []
        for label, value in {
            "initial capital": self.initial_capital,
            "risk per trade": self.risk_per_trade_pct,
            "annualization factor": self.annualization_factor,
        }.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and greater than zero.")
        if self.risk_per_trade_pct > 2.0:
            errors.append("Simulated risk per trade above 2% is not allowed.")
        if not (1 <= self.max_positions <= 64):
            errors.append("Maximum positions must be between 1 and 64.")
        if not (0 < self.max_total_exposure_pct <= 100):
            errors.append("Maximum total exposure must be in (0, 100]%.")
        for label, value in {
            "slippage": self.slippage_bps,
            "spread": self.spread_bps,
            "fee": self.fee_bps,
        }.items():
            if not math.isfinite(value) or value < 0 or value > 2_000:
                errors.append(f"{label} must be within [0, 2000] bps.")
        if self.min_sharpe_observations < 2:
            errors.append("Sharpe requires at least two observations.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimExitLeg:
    timestamp: datetime
    qty: float
    price: float
    fee: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class SimTradeRecord:
    """One round trip, possibly closed across multiple exit legs."""

    symbol: str
    entry_time: datetime
    entry_price: float
    qty: float
    initial_stop_price: float
    entry_fee: float
    entry_reason: str
    exit_legs: list[SimExitLeg] = field(default_factory=list)
    phase_reached: str = PositionPhase.NORMAL.value
    bars_held: int = 0
    max_high_while_open: float = 0.0
    min_low_while_open: float = math.inf

    @property
    def closed_qty(self) -> float:
        return sum(leg.qty for leg in self.exit_legs)

    @property
    def is_closed(self) -> bool:
        return self.closed_qty >= self.qty - 1e-12

    @property
    def fees_total(self) -> float:
        return self.entry_fee + sum(leg.fee for leg in self.exit_legs)

    @property
    def gross_pnl(self) -> Optional[float]:
        if not self.is_closed:
            return None
        return sum(leg.qty * (leg.price - self.entry_price) for leg in self.exit_legs)

    @property
    def net_pnl(self) -> Optional[float]:
        gross = self.gross_pnl
        return None if gross is None else gross - self.fees_total

    @property
    def r_multiple(self) -> Optional[float]:
        net = self.net_pnl
        if net is None:
            return None
        risk = self.qty * (self.entry_price - self.initial_stop_price)
        return net / risk if risk > 0 else None

    @property
    def mfe_pct(self) -> Optional[float]:
        if self.max_high_while_open <= 0 or self.entry_price <= 0:
            return None
        return max(0.0, (self.max_high_while_open - self.entry_price) / self.entry_price * 100.0)

    @property
    def mae_pct(self) -> Optional[float]:
        if not math.isfinite(self.min_low_while_open) or self.entry_price <= 0:
            return None
        return max(0.0, (self.entry_price - self.min_low_while_open) / self.entry_price * 100.0)

    @property
    def return_pct(self) -> Optional[float]:
        if not self.is_closed or self.qty <= 0 or self.entry_price <= 0:
            return None
        weighted_exit = sum(leg.qty * leg.price for leg in self.exit_legs) / self.closed_qty
        return (weighted_exit / self.entry_price - 1.0) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "qty": self.qty,
            "initial_stop_price": self.initial_stop_price,
            "entry_reason": self.entry_reason,
            "exit_legs": [leg.to_dict() for leg in self.exit_legs],
            "phase_reached": self.phase_reached,
            "bars_held": self.bars_held,
            "fees_total": self.fees_total,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "r_multiple": self.r_multiple,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "return_pct": self.return_pct,
        }


@dataclass
class SimulationResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[SimTradeRecord]
    total_return_pct: float
    benchmark_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    profit_factor: Optional[float]
    win_rate: Optional[float]
    num_trades: int
    starting_equity: float
    ending_equity: float
    total_fees: float
    exposure_time_pct: float
    round_trip_cost_bps: float
    signals_seen: int
    entries_taken: int
    entries_rejected: dict[str, int]
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def r_multiples(self) -> list[float]:
        return [
            t.r_multiple
            for t in self.trades
            if t.r_multiple is not None and math.isfinite(t.r_multiple)
        ]


@dataclass
class _OpenPosition:
    symbol: str
    trade: SimTradeRecord
    managed: ManagedPosition
    pending: list[ExitInstruction] = field(default_factory=list)
    pending_full_exit_reason: Optional[str] = None

    @property
    def open_qty(self) -> float:
        return self.trade.qty - self.trade.closed_qty


def _validate_prepared(prepared: Mapping[str, pd.DataFrame]) -> None:
    for symbol, frame in prepared.items():
        if "atr" not in frame.columns:
            raise ValueError(
                f"Prepared frame for {symbol} lacks an 'atr' column; every "
                "UniverseStrategy must compute ATR in prepare_symbol."
            )
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"Prepared frame for {symbol} needs a tz-aware index.")


def run_simulation(
    strategy: UniverseStrategy,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    config: SimulationConfig,
    exit_policy: Optional[ExitPolicyConfig] = None,
    benchmark_symbol: Optional[str] = None,
) -> SimulationResult:
    """Simulate ``strategy`` over the common history of ``bars_by_symbol``.

    ``benchmark_symbol`` defaults to the strategy config's benchmark when
    reachable, else the first symbol; buy-and-hold of that symbol over the
    same evaluated window is the comparison curve.
    """
    errors = config.validate()
    if errors:
        raise ValueError("Invalid simulation configuration: " + "; ".join(errors))
    policy = TradeManagementPolicy(exit_policy or ExitPolicyConfig())

    prepared = strategy.prepare_universe(bars_by_symbol)
    _validate_prepared(prepared)

    common: Optional[pd.DatetimeIndex] = None
    for frame in prepared.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) < strategy.required_bars() + 2:
        raise ValueError(
            "Not enough aligned history to simulate: "
            f"{0 if common is None else len(common)} common bars, "
            f"{strategy.required_bars() + 2} required."
        )
    common = common.sort_values()

    bench = benchmark_symbol
    if bench is None:
        bench = getattr(getattr(strategy, "config", None), "benchmark_symbol", "") or ""
        if not bench or bench not in prepared:
            bench = sorted(prepared)[0]
    if bench not in prepared:
        raise ValueError(f"Benchmark symbol {bench!r} is not in the universe.")

    slippage_fraction = config.slippage_bps / BPS
    half_spread_fraction = config.spread_bps / (2.0 * BPS)
    fee_rate = config.fee_bps / BPS
    buy_adjust = 1.0 + slippage_fraction + half_spread_fraction
    sell_adjust = 1.0 - slippage_fraction - half_spread_fraction
    rt_cost_bps = round_trip_cost_bps(
        slippage_bps=config.slippage_bps,
        spread_bps=config.spread_bps,
        taker_fee_bps=config.fee_bps,
    )

    cash = config.initial_capital
    open_positions: dict[str, _OpenPosition] = {}
    trades: list[SimTradeRecord] = []
    pending_entries: dict[str, dict[str, Any]] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    exposure_bars = 0
    signals_seen = 0
    entries_taken = 0
    entries_rejected: dict[str, int] = {}
    total_fees = 0.0

    def reject(reason: str) -> None:
        entries_rejected[reason] = entries_rejected.get(reason, 0) + 1

    def sell(
        position: _OpenPosition, timestamp: pd.Timestamp, qty: float, price: float, reason: str
    ) -> None:
        nonlocal cash, total_fees
        qty = min(qty, position.open_qty)
        if qty <= 0:
            return
        notional = qty * price
        fee = notional * fee_rate
        cash += notional - fee
        total_fees += fee
        position.trade.exit_legs.append(
            SimExitLeg(
                timestamp=timestamp.to_pydatetime(),
                qty=qty,
                price=price,
                fee=fee,
                reason=reason,
            )
        )

    start_position = strategy.required_bars() - 1
    evaluated_window = common[start_position:]

    for position_index in range(start_position, len(common)):
        timestamp = common[position_index]
        rows = {symbol: prepared[symbol].loc[timestamp] for symbol in prepared}

        # 1) queued management/exit instructions from the previous close
        #    execute at this bar's open, before anything else.
        for symbol in list(open_positions):
            position = open_positions[symbol]
            open_price = float(rows[symbol]["open"])
            if position.pending_full_exit_reason is not None:
                sell(
                    position,
                    timestamp,
                    position.open_qty,
                    open_price * sell_adjust,
                    position.pending_full_exit_reason,
                )
                position.pending_full_exit_reason = None
            else:
                for instruction in position.pending:
                    if instruction.kind is ExitInstructionKind.PARTIAL_EXIT:
                        take_qty = position.trade.qty * (instruction.exit_fraction or 0.0)
                        sell(
                            position,
                            timestamp,
                            take_qty,
                            open_price * sell_adjust,
                            instruction.reason,
                        )
                    elif instruction.kind is ExitInstructionKind.FULL_EXIT:
                        sell(
                            position,
                            timestamp,
                            position.open_qty,
                            open_price * sell_adjust,
                            instruction.reason,
                        )
            position.pending = []
            if position.trade.is_closed:
                trades.append(position.trade)
                del open_positions[symbol]

        # 2) protective stops against this bar's open/low.
        for symbol in list(open_positions):
            position = open_positions[symbol]
            row = rows[symbol]
            open_price = float(row["open"])
            low_price = float(row["low"])
            stop = position.managed.current_stop_price
            if open_price <= stop:
                sell(position, timestamp, position.open_qty, open_price * sell_adjust, "stop_gap")
            elif low_price <= stop:
                sell(position, timestamp, position.open_qty, stop * sell_adjust, "stop_hit")
            if position.trade.is_closed:
                trades.append(position.trade)
                del open_positions[symbol]

        # 3) queued entries fill at this bar's open; same-bar protection.
        for symbol, intent in list(pending_entries.items()):
            del pending_entries[symbol]
            if symbol in open_positions:
                reject("already_held")
                continue
            if len(open_positions) >= config.max_positions:
                reject("max_positions")
                continue
            row = rows[symbol]
            raw_open = float(row["open"])
            if not math.isfinite(raw_open) or raw_open <= 0:
                reject("invalid_open_price")
                continue
            fill = raw_open * buy_adjust
            stop_distance = intent["stop_distance"]
            stop = fill - stop_distance
            if stop <= 0:
                reject("stop_not_positive")
                continue
            held_value = sum(
                pos.open_qty * float(rows[pos.symbol]["open"]) for pos in open_positions.values()
            )
            equity_now = cash + held_value
            if equity_now <= 0:
                reject("no_equity")
                continue
            risk_budget = equity_now * config.risk_per_trade_pct / 100.0
            qty = risk_budget / stop_distance
            if not config.allow_fractional:
                qty = float(math.floor(qty))
            max_notional = equity_now * config.max_total_exposure_pct / 100.0 - held_value
            if max_notional <= 0:
                reject("exposure_cap")
                continue
            qty = min(qty, max_notional / fill)
            affordable = cash / (fill * (1.0 + fee_rate))
            qty = min(qty, affordable)
            if not config.allow_fractional:
                qty = float(math.floor(qty))
            if qty <= 0:
                reject("zero_quantity")
                continue
            notional = qty * fill
            fee = notional * fee_rate
            cash -= notional + fee
            total_fees += fee
            trade = SimTradeRecord(
                symbol=symbol,
                entry_time=timestamp.to_pydatetime(),
                entry_price=fill,
                qty=qty,
                initial_stop_price=stop,
                entry_fee=fee,
                entry_reason=intent["reason"],
            )
            managed = ManagedPosition(
                entry_price=fill,
                initial_stop_price=stop,
                current_stop_price=stop,
                highest_close=fill,
                bars_held=0,
            )
            position = _OpenPosition(symbol=symbol, trade=trade, managed=managed)
            open_positions[symbol] = position
            entries_taken += 1
            # Same-bar protection: an immediate collapse exits on this bar.
            low_price = float(row["low"])
            if low_price <= stop:
                sell(position, timestamp, position.open_qty, stop * sell_adjust, "stop_hit_entry_bar")
                trades.append(position.trade)
                del open_positions[symbol]

        # 4) completed-bar accounting: excursions, management, signals.
        for symbol, position in open_positions.items():
            row = rows[symbol]
            position.trade.max_high_while_open = max(
                position.trade.max_high_while_open, float(row["high"])
            )
            position.trade.min_low_while_open = min(
                position.trade.min_low_while_open, float(row["low"])
            )
            position.trade.bars_held += 1

        for symbol in list(open_positions):
            position = open_positions[symbol]
            row = rows[symbol]
            close = float(row["close"])
            atr_value = float(row["atr"])
            if not math.isfinite(close) or close <= 0 or not math.isfinite(atr_value) or atr_value <= 0:
                continue  # degraded bar: keep the existing stop, take no action
            efficiency = row.get("efficiency_ratio")
            efficiency_value = (
                float(efficiency)
                if efficiency is not None and math.isfinite(float(efficiency))
                else None
            )
            update = policy.on_completed_bar(
                position.managed,
                close=close,
                atr_value=atr_value,
                efficiency_ratio=efficiency_value,
            )
            position.managed = update.position
            if position.managed.phase.value != position.trade.phase_reached:
                order = [p.value for p in PositionPhase]
                if order.index(position.managed.phase.value) > order.index(
                    position.trade.phase_reached
                ):
                    position.trade.phase_reached = position.managed.phase.value
            position.pending = [
                instruction
                for instruction in update.instructions
                if instruction.kind is not ExitInstructionKind.UPDATE_STOP
            ]

        held_symbols = frozenset(open_positions)
        decisions = strategy.evaluate_prepared(prepared, timestamp, held_symbols)
        for decision in decisions:
            if decision.action is BotAction.BUY:
                signals_seen += 1
                if decision.symbol in open_positions or decision.symbol in pending_entries:
                    reject("already_held")
                    continue
                close = decision.close
                stop_price = decision.stop_price
                if (
                    close is None
                    or stop_price is None
                    or not math.isfinite(close)
                    or not math.isfinite(stop_price)
                    or stop_price <= 0
                    or stop_price >= close
                ):
                    reject("invalid_signal_stop")
                    continue
                pending_entries[decision.symbol] = {
                    "stop_distance": close - stop_price,
                    "reason": decision.reason,
                }
            elif decision.action is BotAction.EXIT and decision.symbol in open_positions:
                open_positions[decision.symbol].pending_full_exit_reason = decision.reason

        closes = {symbol: float(rows[symbol]["close"]) for symbol in prepared}
        held_value = sum(
            pos.open_qty * closes[pos.symbol] for pos in open_positions.values()
        )
        equity_points.append((timestamp, cash + held_value))
        if open_positions:
            exposure_bars += 1

    # Liquidate whatever is still open at the final close so results are
    # comparable across runs; the leg is labeled so nobody mistakes it for
    # a signal exit.
    if open_positions:
        final_timestamp = common[-1]
        for symbol in list(open_positions):
            position = open_positions[symbol]
            close = float(prepared[symbol].loc[final_timestamp]["close"])
            sell(position, final_timestamp, position.open_qty, close * sell_adjust, "end_of_simulation")
            trades.append(position.trade)
            del open_positions[symbol]
        last_ts, _ = equity_points[-1]
        held_value = 0.0
        equity_points[-1] = (last_ts, cash + held_value)

    equity_curve = pd.Series(
        [value for _, value in equity_points],
        index=pd.DatetimeIndex([ts for ts, _ in equity_points]),
        dtype=float,
    )
    bench_closes = prepared[bench]["close"].reindex(evaluated_window).astype(float)
    if bench_closes.isna().any() or (bench_closes <= 0).any():
        raise ValueError(f"Benchmark {bench!r} has unusable closes in the window.")
    benchmark_curve = bench_closes / bench_closes.iloc[0] * config.initial_capital

    ending_equity = float(equity_curve.iloc[-1])
    total_return_pct = (ending_equity / config.initial_capital - 1.0) * 100.0
    benchmark_return_pct = float(
        (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1.0) * 100.0
    )

    running_peak = equity_curve.cummax()
    drawdown = ((running_peak - equity_curve) / running_peak).max()
    max_drawdown_pct = float(drawdown * 100.0) if math.isfinite(drawdown) else 0.0

    bar_returns = equity_curve.pct_change().dropna()
    sharpe: Optional[float] = None
    if len(bar_returns) >= config.min_sharpe_observations:
        std = float(bar_returns.std(ddof=1))
        if std > 0:
            sharpe = float(
                bar_returns.mean() / std * math.sqrt(config.annualization_factor)
            )

    closed = [t for t in trades if t.net_pnl is not None]
    wins = [t for t in closed if t.net_pnl > 0]
    losses = [t for t in closed if t.net_pnl <= 0]
    win_rate = len(wins) / len(closed) if closed else None
    gross_wins = sum(t.net_pnl for t in wins)
    gross_losses = abs(sum(t.net_pnl for t in losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else None

    return SimulationResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=trades,
        total_return_pct=total_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        win_rate=win_rate,
        num_trades=len(trades),
        starting_equity=config.initial_capital,
        ending_equity=ending_equity,
        total_fees=total_fees,
        exposure_time_pct=(
            exposure_bars / len(evaluated_window) * 100.0 if len(evaluated_window) else 0.0
        ),
        round_trip_cost_bps=rt_cost_bps,
        signals_seen=signals_seen,
        entries_taken=entries_taken,
        entries_rejected=dict(sorted(entries_rejected.items())),
        config_snapshot={
            "simulation": config.to_dict(),
            "exit_policy": asdict(exit_policy or ExitPolicyConfig()),
            "benchmark_symbol": bench,
        },
    )
