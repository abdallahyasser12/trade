"""Causal multi-asset crypto simulation using shared strategy and risk code."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd

from config import AccountMode
from crypto_config import (
    CryptoBacktestConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
    crypto_bars_per_year,
)
from crypto_models import CryptoBacktestResult, CryptoSignalDecision, CryptoTradeRecord
from crypto_risk import CryptoRiskManager, floor_to_increment
from crypto_strategy import CryptoRegimeStrategy
from models import AssetInfo, BotAction


MIN_SHARPE_OBSERVATIONS = 20


@dataclass
class _Position:
    symbol: str
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    entry_fee: float
    entry_reason: str
    atr_at_entry: float
    stop_price: float
    highest_close: float


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return max(0.0, -float(drawdown.min()) * 100.0)


def _sharpe(
    equity: pd.Series, timeframe: str, risk_free_rate: float
) -> Optional[float]:
    returns = equity.pct_change().dropna()
    if len(returns) < MIN_SHARPE_OBSERVATIONS:
        return None
    factor = crypto_bars_per_year(timeframe)
    excess = returns - risk_free_rate / factor
    std = float(excess.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    return float(excess.mean() / std * math.sqrt(factor))


def _validate_bars(symbol: str, bars: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    if not isinstance(bars, pd.DataFrame):
        raise ValueError(f"{symbol} bars must be a pandas DataFrame.")
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"{symbol} bars are missing {sorted(missing)}.")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError(f"{symbol} bars need a timezone-aware DatetimeIndex.")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError(f"{symbol} bars must be chronological and de-duplicated.")
    values = bars[list(required)].astype(float)
    if not values.map(math.isfinite).all(axis=None):
        raise ValueError(f"{symbol} bars contain NaN or infinite OHLCV values.")
    coherent = (
        values[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & values["volume"].ge(0)
        & values["high"].ge(values[["open", "close"]].max(axis=1))
        & values["low"].le(values[["open", "close"]].min(axis=1))
        & values["high"].ge(values["low"])
    )
    if not bool(coherent.all()):
        raise ValueError(f"{symbol} bars contain invalid or inconsistent OHLCV values.")


def _default_asset(symbol: str) -> AssetInfo:
    return AssetInfo(
        symbol=symbol,
        name=symbol,
        tradable=True,
        active=True,
        fractionable=True,
        asset_class="crypto",
        min_order_size=1e-9,
        min_trade_increment=1e-9,
        price_increment=1e-9,
        marginable=False,
        shortable=False,
    )


def run_crypto_backtest(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    strategy_config: CryptoStrategyConfig,
    risk_config: CryptoRiskConfig,
    backtest_config: CryptoBacktestConfig,
    *,
    assets: Optional[Mapping[str, AssetInfo]] = None,
    active_start: Optional[pd.Timestamp] = None,
) -> CryptoBacktestResult:
    """Simulate completed-bar decisions and strictly next-bar execution.

    Protective stops are evaluated against the next bar's open/low before any
    close-derived signal. A gap below the stop fills at the adverse gap open,
    never at the unavailable stop price.
    """

    strategy = CryptoRegimeStrategy(strategy_config)
    risk = CryptoRiskManager(risk_config)
    errors = backtest_config.validate()
    if errors:
        raise ValueError("Invalid crypto backtest configuration: " + "; ".join(errors))
    for symbol in strategy_config.symbols:
        if symbol not in bars_by_symbol:
            raise ValueError(f"Missing bars for {symbol}.")
        _validate_bars(symbol, bars_by_symbol[symbol])

    reference_symbol = strategy_config.symbols[0]
    common = bars_by_symbol[reference_symbol].index
    for symbol in strategy_config.symbols[1:]:
        if not bars_by_symbol[symbol].index.equals(common):
            raise ValueError(
                "Crypto portfolio bars must be exactly timestamp-aligned; "
                f"{symbol} differs from {reference_symbol}."
            )
    if len(common) < strategy.required_bars() + 2:
        count = len(common)
        raise ValueError(
            f"Aligned crypto history has {count} bars; at least "
            f"{strategy.required_bars() + 2} are required."
        )
    common = common.sort_values()
    aligned = {symbol: bars_by_symbol[symbol].loc[common].copy() for symbol in strategy_config.symbols}
    prepared = strategy.prepare_universe(aligned)

    if active_start is None:
        activation = common[0]
    else:
        activation = pd.Timestamp(active_start)
        if activation.tzinfo is None:
            raise ValueError("active_start must be timezone-aware.")
        activation = activation.tz_convert(common.tz)
        if int((common >= activation).sum()) < 2:
            raise ValueError("active_start leaves fewer than two executable bars.")

    resolved_assets = {
        symbol: (assets[symbol] if assets and symbol in assets else _default_asset(symbol))
        for symbol in strategy_config.symbols
    }
    for symbol, asset in resolved_assets.items():
        if not asset.is_supported_crypto:
            raise ValueError(f"{symbol} asset metadata is not supported spot crypto.")

    slippage = backtest_config.slippage_bps / 10_000.0
    half_spread = backtest_config.spread_bps / 20_000.0
    fee_rate = backtest_config.taker_fee_bps / 10_000.0
    buy_cost = slippage + half_spread
    sell_cost = slippage + half_spread

    cash = float(backtest_config.initial_capital)
    peak_equity = cash
    positions: dict[str, _Position] = {}
    pending_entries: dict[str, CryptoSignalDecision] = {}
    pending_exits: dict[str, str] = {}
    trades: list[CryptoTradeRecord] = []
    equity_points: list[float] = []
    equity_index: list[pd.Timestamp] = []
    exposure_bars = 0
    total_fees = 0.0

    def close_position(symbol: str, timestamp: pd.Timestamp, fill: float, reason: str) -> None:
        nonlocal cash, total_fees
        position = positions.pop(symbol)
        exit_notional = position.qty * fill
        exit_fee = exit_notional * fee_rate
        cash += exit_notional - exit_fee
        total_fees += exit_fee
        pnl = (
            (fill - position.entry_price) * position.qty
            - position.entry_fee
            - exit_fee
        )
        invested = position.entry_price * position.qty + position.entry_fee
        trades.append(
            CryptoTradeRecord(
                symbol=symbol,
                qty=position.qty,
                entry_time=position.entry_time.to_pydatetime(),
                entry_price=position.entry_price,
                entry_fee=position.entry_fee,
                exit_time=timestamp.to_pydatetime(),
                exit_price=fill,
                exit_fee=exit_fee,
                pnl=pnl,
                return_pct=pnl / invested * 100.0 if invested > 0 else 0.0,
                exit_reason=reason,
                entry_reason=position.entry_reason,
            )
        )

    for timestamp in common:
        rows = {symbol: aligned[symbol].loc[timestamp] for symbol in strategy_config.symbols}
        active = timestamp >= activation

        # Signals from N-1 execute at N open. Exits go first so protective
        # orders and portfolio caps cannot oversell or count a closed position.
        if active:
            for symbol, reason in list(pending_exits.items()):
                if symbol in positions:
                    fill = float(rows[symbol]["open"]) * (1.0 - sell_cost)
                    close_position(symbol, timestamp, fill, reason)
                pending_exits.pop(symbol, None)

            for symbol, decision in list(pending_entries.items()):
                pending_entries.pop(symbol, None)
                if symbol in positions or len(positions) >= risk_config.max_positions:
                    continue
                raw_open = float(rows[symbol]["open"])
                fill = raw_open * (1.0 + buy_cost)
                atr_value = float(decision.atr or 0.0)
                stop = fill - atr_value * strategy_config.atr_stop_multiplier
                open_equity = cash + sum(
                    position.qty * float(rows[held]["open"])
                    for held, position in positions.items()
                )
                exposure = sum(
                    position.qty * float(rows[held]["open"])
                    for held, position in positions.items()
                )
                size = risk.size_position(
                    equity=open_equity,
                    cash=cash,
                    buying_power=cash,
                    entry_price=fill,
                    stop_price=stop,
                    asset=resolved_assets[symbol],
                    current_crypto_exposure=exposure,
                    current_position_count=len(positions),
                    peak_equity=peak_equity,
                    atr_percentile=float(decision.atr_percentile or 100.0),
                    account_mode=AccountMode.PAPER,
                )
                if not size.approved:
                    continue
                increment = float(resolved_assets[symbol].min_trade_increment or 0.0)
                affordable = floor_to_increment(cash / (fill * (1.0 + fee_rate)), increment)
                quantity = floor_to_increment(min(size.qty, affordable), increment)
                minimum = float(resolved_assets[symbol].min_order_size or math.inf)
                if quantity <= 0 or quantity + increment / 10.0 < minimum:
                    continue
                notional = quantity * fill
                entry_fee = notional * fee_rate
                if notional + entry_fee > cash + 1e-8:
                    continue
                cash -= notional + entry_fee
                total_fees += entry_fee
                positions[symbol] = _Position(
                    symbol=symbol,
                    qty=quantity,
                    entry_time=timestamp,
                    entry_price=fill,
                    entry_fee=entry_fee,
                    entry_reason=decision.reason,
                    atr_at_entry=atr_value,
                    stop_price=stop,
                    highest_close=float(rows[symbol]["close"]),
                )

        # Broker-side protection exists immediately after an entry fill. A
        # stop may therefore trigger on the entry bar in a pathological range.
        for symbol, position in list(positions.items()):
            row = rows[symbol]
            raw_open = float(row["open"])
            if raw_open <= position.stop_price:
                fill = raw_open * (1.0 - sell_cost)
                close_position(symbol, timestamp, fill, "atr_stop_gap")
                pending_exits.pop(symbol, None)
            elif float(row["low"]) <= position.stop_price:
                fill = position.stop_price * (1.0 - sell_cost)
                close_position(symbol, timestamp, fill, "atr_stop")
                pending_exits.pop(symbol, None)

        close_equity = cash + sum(
            position.qty * float(rows[symbol]["close"])
            for symbol, position in positions.items()
        )
        peak_equity = max(peak_equity, close_equity)
        if active:
            risk.check_session_loss(backtest_config.initial_capital, close_equity)
            held = set(positions)
            decisions = strategy.evaluate_prepared(prepared, timestamp, held)
            for decision in decisions:
                if decision.action is BotAction.EXIT and decision.symbol in positions:
                    pending_exits[decision.symbol] = decision.reason

            if not risk.session_halted:
                available_slots = max(
                    0,
                    risk_config.max_positions
                    - len(positions)
                    - len(pending_entries),
                )
                held_returns = {
                    symbol: prepared[symbol]["close"].pct_change().loc[:timestamp]
                    for symbol in positions
                }
                for decision in decisions:
                    if available_slots <= 0:
                        break
                    symbol = decision.symbol
                    if (
                        decision.action is not BotAction.BUY
                        or symbol in positions
                        or symbol in pending_entries
                        or symbol in pending_exits
                    ):
                        continue
                    candidate_returns = prepared[symbol]["close"].pct_change().loc[:timestamp]
                    allowed, _, _ = risk.correlation_gate(candidate_returns, held_returns)
                    if not allowed:
                        continue
                    pending_entries[symbol] = decision
                    held_returns[symbol] = candidate_returns
                    available_slots -= 1

            # Completed-bar trailing updates happen after this bar's stop
            # simulation and therefore cannot affect the bar that created them.
            for symbol, position in positions.items():
                current_atr = float(prepared[symbol].loc[timestamp, "atr"])
                new_stop, highest = strategy.ratchet_stop(
                    position.stop_price,
                    position.highest_close,
                    float(rows[symbol]["close"]),
                    current_atr,
                )
                position.stop_price = min(new_stop, float(rows[symbol]["close"]) * (1 - 1e-12))
                position.highest_close = highest

            equity_points.append(close_equity)
            equity_index.append(timestamp)
            if positions:
                exposure_bars += 1

    if not equity_points:
        raise ValueError("No active crypto bars were simulated.")

    if backtest_config.liquidate_at_end and positions:
        final_timestamp = common[-1]
        for symbol in list(positions):
            raw_close = float(aligned[symbol].loc[final_timestamp, "close"])
            close_position(
                symbol,
                final_timestamp,
                raw_close * (1.0 - sell_cost),
                "end_of_backtest",
            )
        equity_points[-1] = cash
    elif positions:
        # Preserve open-lot visibility when the caller explicitly declines
        # liquidation; these records are excluded from closed-trade statistics.
        for position in positions.values():
            trades.append(
                CryptoTradeRecord(
                    symbol=position.symbol,
                    qty=position.qty,
                    entry_time=position.entry_time.to_pydatetime(),
                    entry_price=position.entry_price,
                    entry_fee=position.entry_fee,
                    entry_reason=position.entry_reason,
                )
            )

    equity = pd.Series(equity_points, index=pd.DatetimeIndex(equity_index), dtype=float)
    equity.name = "crypto_portfolio_equity"
    benchmark_close = aligned[strategy_config.benchmark_symbol]["close"].reindex(equity.index)
    benchmark = benchmark_close / float(benchmark_close.iloc[0]) * backtest_config.initial_capital
    benchmark.name = f"{strategy_config.benchmark_symbol}_buy_hold"
    ending_equity = float(equity.iloc[-1])
    total_return = (ending_equity / backtest_config.initial_capital - 1.0) * 100.0
    benchmark_return = (float(benchmark.iloc[-1]) / float(benchmark.iloc[0]) - 1.0) * 100.0
    closed = [trade for trade in trades if trade.is_closed and trade.pnl is not None]
    wins = [float(trade.pnl) for trade in closed if float(trade.pnl) > 0]
    losses = [-float(trade.pnl) for trade in closed if float(trade.pnl) < 0]
    profit_factor = (
        sum(wins) / sum(losses)
        if losses
        else math.inf
        if wins
        else None
    )
    return CryptoBacktestResult(
        equity_curve=equity,
        benchmark_curve=benchmark,
        trades=trades,
        total_return_pct=total_return,
        benchmark_return_pct=benchmark_return,
        max_drawdown_pct=_max_drawdown(equity),
        sharpe_ratio=_sharpe(equity, strategy_config.timeframe, backtest_config.risk_free_rate),
        profit_factor=profit_factor,
        win_rate=(len(wins) / len(closed) * 100.0) if closed else None,
        num_trades=len(closed),
        starting_equity=backtest_config.initial_capital,
        ending_equity=ending_equity,
        realized_pnl=sum(float(trade.pnl or 0.0) for trade in closed),
        total_fees=total_fees,
        exposure_time_pct=exposure_bars / len(equity_points) * 100.0,
        config_snapshot={
            "strategy": strategy_config.to_dict(),
            "risk": risk_config.to_dict(),
            "backtest": backtest_config.to_dict(),
            "execution": {
                "signal": "completed bar N",
                "fill": "bar N+1 open",
                "stop_gap": "adverse gap open",
                "costs": "half-spread + directional slippage + percentage taker fee",
            },
        },
    )
