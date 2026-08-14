"""Historical simulation using the exact same TrendStrategy and
RiskManager as live trading.

Execution model (documented, tested):

* A signal is generated on completed bar N using data through bar N only.
* The resulting order fills at bar N+1's OPEN (never bar N's close) —
  next-bar execution, no look-ahead.
* Protective stop: for an open long with stop S, on each subsequent bar
  (including the entry bar, after the fill): if the bar OPENS at/below S
  the fill is the open price (gap through the stop — you do not get S);
  otherwise if the bar's LOW touches S the fill is min(S, bar open) — a
  stop exit is never booked at a better price than the bar's open.  The
  min() matters on the entry bar with slippage: the entry fills at
  open×(1+s), so with a small ATR distance S can sit above the raw open,
  and the honest fill there is the open, not S.
* Slippage is directionally realistic: BUY fills at price*(1+bps/10000),
  SELL fills at price*(1-bps/10000).
* ``commission_per_trade`` is charged per executed order (entry and exit
  each count once).
* The backtest models a cash account: buying power == cash, whole shares.
* Daily loss protection uses the same RiskManager rule as live trading,
  with the baseline reset to the previous close equity at the start of
  each UTC calendar day; once breached, new entries are blocked for the
  rest of that day (exits and stops still run).
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from config import BacktestConfig, RiskConfig, StrategyConfig, bars_per_year
from indicators import compute_indicators
from models import BacktestResult, BotAction, TradeRecord
from risk_manager import RiskManager
from strategy import TrendStrategy

# Below this many completed trades, win rate / Sharpe are statistically
# meaningless and the UI shows a small-sample warning.
SMALL_SAMPLE_TRADE_THRESHOLD = 30

# Minimum number of return observations for a Sharpe ratio to be reported.
MIN_SHARPE_OBSERVATIONS = 20


def compute_max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from the running peak, as a positive percentage.

    drawdown_t = equity_t / running_peak_t - 1; the result is
    ``-min(drawdown) * 100`` — e.g. 12.5 means at worst the equity sat
    12.5% below its prior peak.
    """
    if equity is None or len(equity) == 0:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    worst = float(drawdown.min())
    return max(0.0, -worst * 100.0)


def compute_sharpe(
    equity: pd.Series,
    timeframe: str,
    risk_free_rate: float = 0.0,
    min_observations: int = MIN_SHARPE_OBSERVATIONS,
) -> tuple[Optional[float], float]:
    """Annualized Sharpe ratio of per-bar equity returns.

    Assumptions (displayed in the UI next to the number):
    * returns are simple per-bar returns of the equity curve;
    * the annualization factor is the approximate number of completed
      regular-session bars per year for ``timeframe`` (config.bars_per_year),
      not a hardcoded 252 for every timeframe;
    * ``risk_free_rate`` is an annual decimal rate (default 0.0), converted
      to a per-bar rate by simple division.

    Returns (sharpe_or_None, annualization_factor).  None when there are
    fewer than ``min_observations`` returns or volatility is zero.
    """
    factor = bars_per_year(timeframe)
    if equity is None or len(equity) < 2:
        return None, factor
    returns = equity.pct_change().dropna()
    if len(returns) < min_observations:
        return None, factor
    rf_per_bar = risk_free_rate / factor
    excess = returns - rf_per_bar
    std = float(excess.std())
    if std == 0.0 or math.isnan(std):
        return None, factor
    sharpe = float(excess.mean()) / std * math.sqrt(factor)
    return sharpe, factor


def run_backtest(
    price_data: pd.DataFrame,
    strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    backtest_config: BacktestConfig,
) -> BacktestResult:
    """Simulate the strategy over ``price_data`` (completed OHLCV bars).

    ``price_data`` must be chronological, de-duplicated, tz-aware and
    contain open/high/low/close/volume — the shape produced by
    data_feed.normalize_bars_df.  Raises ValueError when there is not
    enough history to warm up the indicators.
    """
    strategy = TrendStrategy(strategy_config)
    risk = RiskManager(risk_config)

    config_errors = backtest_config.validate()
    if config_errors:
        raise ValueError("Invalid backtest configuration: " + "; ".join(config_errors))

    required_columns = {"open", "high", "low", "close", "volume"}
    if not isinstance(price_data, pd.DataFrame):
        raise ValueError("price_data must be a pandas DataFrame")
    missing = required_columns.difference(price_data.columns)
    if missing:
        raise ValueError(f"price_data is missing columns: {sorted(missing)}")
    if not isinstance(price_data.index, pd.DatetimeIndex) or price_data.index.tz is None:
        raise ValueError("price_data must have a timezone-aware DatetimeIndex")
    if not price_data.index.is_monotonic_increasing or price_data.index.has_duplicates:
        raise ValueError("price_data must be chronological and contain no duplicate timestamps")
    numeric = price_data[list(required_columns)].astype(float)
    if not numeric.map(math.isfinite).all(axis=None):
        raise ValueError("price_data contains NaN or infinite OHLCV values")
    price_valid = numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = (
        numeric["volume"].ge(0)
        & numeric["high"].ge(numeric[["open", "close"]].max(axis=1))
        & numeric["low"].le(numeric[["open", "close"]].min(axis=1))
        & numeric["high"].ge(numeric["low"])
    )
    if not bool((price_valid & coherent).all()):
        raise ValueError("price_data contains invalid or internally inconsistent OHLCV bars")

    warmup = strategy.required_bars()
    if price_data is None or len(price_data) < warmup + 2:
        raise ValueError(
            f"Not enough historical data: {0 if price_data is None else len(price_data)} bars "
            f"available but the configured indicators need at least {warmup + 2} "
            "(long MA / RSI / ATR / volume / ATR percentile warmup, plus one bar to "
            "execute on). Widen the date range or shorten the indicator periods."
        )

    # Indicators are causal (rolling/EWM), so computing them once over the
    # full frame is bar-for-bar identical to recomputing on each prefix —
    # generate_signal still only ever sees rows up to the current bar.
    df = compute_indicators(price_data, strategy_config)

    slippage = backtest_config.slippage_bps / 10_000.0
    commission = backtest_config.commission_per_trade

    cash = backtest_config.initial_capital
    position_qty = 0
    entry_price = 0.0
    entry_time = None
    entry_reason = ""
    stop_price: Optional[float] = None

    pending_entry: Optional[dict] = None  # signal context awaiting next-bar open
    pending_exit: Optional[str] = None  # exit reason awaiting next-bar open

    trades: list[TradeRecord] = []
    equity_values: list[float] = []

    day_start_equity: Optional[float] = None
    current_day = None
    day_halted = False

    def close_position(exit_time, exit_px: float, reason: str) -> None:
        nonlocal cash, position_qty, entry_price, entry_time, stop_price, entry_reason
        proceeds = position_qty * exit_px - commission
        cash += proceeds
        pnl = (exit_px - entry_price) * position_qty - 2 * commission
        invested = entry_price * position_qty + commission
        ret = pnl / invested * 100.0 if invested > 0 else 0.0
        trades.append(
            TradeRecord(
                symbol=strategy_config.symbol,
                qty=position_qty,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_px,
                pnl=pnl,
                return_pct=ret,
                exit_reason=reason,
                entry_reason=entry_reason,
            )
        )
        position_qty = 0
        entry_price = 0.0
        entry_time = None
        entry_reason = ""
        stop_price = None

    timestamps = df.index
    for i in range(len(df)):
        bar = df.iloc[i]
        ts = timestamps[i]
        bar_day = ts.date()
        entered_this_bar = False

        # Establish the new day's loss baseline before any fills or marks on
        # this bar.  Using the first bar's post-move equity would hide an
        # overnight/first-bar loss from the circuit breaker.
        if current_day != bar_day:
            current_day = bar_day
            day_start_equity = (
                equity_values[-1] if equity_values else backtest_config.initial_capital
            )
            day_halted = False

        # --- execute actions scheduled on the previous completed bar ------
        if pending_exit is not None and position_qty > 0:
            exit_px = float(bar["open"]) * (1.0 - slippage)
            close_position(ts, exit_px, pending_exit)
        pending_exit = None

        if pending_entry is not None and position_qty == 0 and not day_halted:
            open_px = float(bar["open"])
            fill_px = open_px * (1.0 + slippage)
            atr_at_entry = pending_entry["atr"]
            try:
                candidate_stop = risk.calculate_stop_price(
                    fill_px, atr_at_entry, strategy_config.atr_multiplier
                )
            except ValueError:
                candidate_stop = None
            if candidate_stop is not None:
                qty = int(pending_entry["qty"])
                cost = qty * fill_px + commission
                if qty > 0 and cost <= cash:
                    cash -= cost
                    position_qty = qty
                    entry_price = fill_px
                    entry_time = ts
                    entry_reason = pending_entry["reason"]
                    stop_price = candidate_stop
                    entered_this_bar = True
        pending_entry = None

        # --- protective stop check on this bar ---------------------------
        if position_qty > 0 and stop_price is not None:
            open_px = float(bar["open"])
            low_px = float(bar["low"])
            stop_fill: Optional[float] = None
            if not entered_this_bar and open_px <= stop_price:
                stop_fill = open_px  # gapped through the stop
            elif low_px <= stop_price:
                # Never better than the bar's open: with entry slippage the
                # stop can sit above the raw open on the entry bar itself.
                stop_fill = min(stop_price, open_px)
            if stop_fill is not None:
                exit_px = stop_fill * (1.0 - slippage)
                close_position(ts, exit_px, "atr_stop")

        # --- mark to market ----------------------------------------------
        equity = cash + position_qty * float(bar["close"])
        equity_values.append(equity)

        # --- daily loss circuit breaker (same rule as live) --------------
        if not day_halted and day_start_equity and risk.check_daily_loss_limit(
            day_start_equity, equity
        ):
            day_halted = True

        # --- evaluate the strategy on this completed bar ------------------
        if i >= warmup - 1 and i < len(df) - 1:
            window = df.iloc[: i + 1]
            decision = strategy.generate_signal(window)
            if decision.action == BotAction.BUY and position_qty == 0 and not day_halted:
                signal_price = float(decision.details.get("close") or bar["close"])
                try:
                    signal_stop = risk.calculate_stop_price(
                        signal_price,
                        float(decision.atr),
                        strategy_config.atr_multiplier,
                    )
                    sizing = risk.calculate_position_size(
                        equity=equity,
                        buying_power=cash,
                        entry_price=signal_price,
                        stop_price=signal_stop,
                    )
                except (TypeError, ValueError):
                    sizing = None
                if sizing is not None and sizing.approved:
                    pending_entry = {
                        "atr": decision.atr,
                        "reason": decision.reason,
                        "qty": sizing.qty,
                    }
            elif decision.action == BotAction.EXIT and position_qty > 0:
                pending_exit = "bearish_crossover"

    if position_qty > 0:
        # The simulation ended still holding shares: record the open round
        # trip (exit fields None) so the trade list matches the equity curve.
        # It is excluded from num_trades/win_rate, which count closed trades.
        trades.append(
            TradeRecord(
                symbol=strategy_config.symbol,
                qty=position_qty,
                entry_time=entry_time,
                entry_price=entry_price,
                entry_reason=entry_reason,
            )
        )

    equity_curve = pd.Series(equity_values, index=timestamps, name="strategy")

    close = df["close"].astype(float)
    benchmark_curve = (
        backtest_config.initial_capital * close / float(close.iloc[0])
    ).rename("buy_and_hold")

    closed = [t for t in trades if t.is_closed]
    wins = [t for t in closed if (t.pnl or 0) > 0]
    losses = [t for t in closed if (t.pnl or 0) <= 0]
    gross_win = float(sum(t.pnl for t in wins)) if wins else 0.0
    gross_loss = float(-sum(t.pnl for t in losses)) if losses else 0.0
    realized_pnl = float(sum(t.pnl for t in closed)) if closed else 0.0

    ending_equity = float(equity_curve.iloc[-1])
    starting_equity = backtest_config.initial_capital
    sharpe, annualization = compute_sharpe(
        equity_curve, strategy_config.timeframe, backtest_config.risk_free_rate
    )

    open_trades = [t for t in trades if not t.is_closed]
    config_snapshot = {
        "symbol": strategy_config.symbol,
        "timeframe": strategy_config.timeframe,
        "start_date": backtest_config.start_date.isoformat(),
        "end_date": backtest_config.end_date.isoformat(),
        "strategy": strategy_config.to_dict(),
        "risk": risk_config.to_dict(),
        "backtest": backtest_config.to_dict(),
        "bar_adjustment": backtest_config.adjustment,
        "execution_model": "signal on completed bar N fills at bar N+1 open",
        "stop_model": "gap-open through stop fills at open; otherwise low<=stop fills at stop",
        "account_model": "cash account, whole shares, buying power == cash",
        "bars_used": int(len(df)),
        "warmup_bars": warmup,
        "open_trade_at_end": len(open_trades),
    }

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=trades,
        total_return_pct=(ending_equity / starting_equity - 1.0) * 100.0,
        benchmark_return_pct=(float(benchmark_curve.iloc[-1]) / starting_equity - 1.0) * 100.0,
        win_rate=(len(wins) / len(closed) * 100.0) if closed else None,
        max_drawdown_pct=compute_max_drawdown(equity_curve),
        sharpe_ratio=sharpe,
        sharpe_annualization_factor=annualization,
        risk_free_rate=backtest_config.risk_free_rate,
        num_trades=len(closed),
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        realized_pnl=realized_pnl,
        avg_win=(gross_win / len(wins)) if wins else None,
        avg_loss=(-gross_loss / len(losses)) if losses else None,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else None,
        best_trade=max((t.pnl for t in closed), default=None),
        worst_trade=min((t.pnl for t in closed), default=None),
        config_snapshot=config_snapshot,
    )
