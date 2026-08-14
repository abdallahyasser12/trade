"""Deterministic long-only trend strategy with robust confirmation filters.

This is the ONLY implementation of the trading rules.  Both the
backtester and the live trade engine call ``TrendStrategy.generate_signal``
on a DataFrame of COMPLETED bars — there is no separate live/backtest
rule code, which guarantees decision parity between the two modes.

The immutable moving-average crossover remains the entry trigger.  Optional
RSI-floor, ADX, Kaufman efficiency, broad-market regime, and relative-strength
filters decide whether that fresh trigger is tradable.  A bearish crossover or
a newly broken broad-market regime may exit.  ATR-based sizing/stops live in
``RiskManager`` so the exact same decision and risk implementations are shared
by live trading, backtests, and walk-forward research.

Rules (all evaluated on the last completed bar):

* BUY  — bullish crossover (prev short_ma <= prev long_ma AND
  current short_ma > current long_ma) AND RSI < overbought AND
  volume_ratio >= minimum AND (volatility filter disabled OR
  ATR percentile <= maximum).
* EXIT — bearish crossover (prev short_ma >= prev long_ma AND
  current short_ma < current long_ma).  The engine/backtester turn EXIT
  into a no-op when no position is open.
* HOLD — anything else, including insufficient warmup data.

RSI, volume, volatility, ADX, efficiency and relative strength are entry
filters only.  No rule predicts future prices and no configuration guarantees
profit.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from config import StrategyConfig, timeframe_to_timedelta
from indicators import compute_indicators, sma
from models import BotAction, SignalDecision


def _finite(value) -> Optional[float]:
    """Convert a pandas/numpy scalar to a plain float, or None if NaN/inf."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


class TrendStrategy:
    """Deterministic trend-following signal generator (long-only)."""

    def __init__(self, config: StrategyConfig):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid strategy configuration: " + "; ".join(errors))
        self.config = config

    def required_bars(self) -> int:
        """Completed bars needed before signals can be generated safely."""
        return self.config.warmup_bars()

    def generate_signal(
        self,
        price_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
    ) -> SignalDecision:
        """Evaluate the strategy on the last completed bar of ``price_data``.

        ``price_data`` must contain only completed bars, in chronological
        order, with columns open/high/low/close/volume.  Indicator columns
        are computed here if not already present, so callers may pass raw
        OHLCV or pre-computed frames.
        """
        cfg = self.config

        if price_data is None or len(price_data) == 0:
            return self._hold(None, "insufficient_data", {"bars_available": 0})

        last_ts = price_data.index[-1]
        ts = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts

        if len(price_data) < self.required_bars():
            return self._hold(
                ts,
                "insufficient_data",
                {"bars_available": int(len(price_data)), "bars_required": self.required_bars()},
            )

        indicator_cols = {"short_ma", "long_ma", "rsi", "atr", "avg_volume", "volume_ratio"}
        if cfg.enable_adx_filter:
            indicator_cols.add("adx")
        if cfg.enable_efficiency_filter:
            indicator_cols.add("efficiency_ratio")
        if indicator_cols.issubset(price_data.columns):
            df = price_data
        else:
            df = compute_indicators(price_data, cfg)

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        short_ma = _finite(curr["short_ma"])
        long_ma = _finite(curr["long_ma"])
        prev_short = _finite(prev["short_ma"])
        prev_long = _finite(prev["long_ma"])
        rsi_val = _finite(curr["rsi"])
        atr_val = _finite(curr["atr"])
        vol = _finite(curr["volume"])
        avg_vol = _finite(curr["avg_volume"])
        vol_ratio = _finite(curr["volume_ratio"])
        atr_pct = _finite(curr["atr_percentile"]) if "atr_percentile" in df.columns else None
        adx_val = _finite(curr["adx"]) if "adx" in df.columns else None
        efficiency_val = (
            _finite(curr["efficiency_ratio"])
            if "efficiency_ratio" in df.columns
            else None
        )
        close_val = _finite(curr["close"])

        core_values = [short_ma, long_ma, prev_short, prev_long, rsi_val, atr_val, vol, avg_vol, vol_ratio]
        if cfg.enable_volatility_filter:
            core_values.append(atr_pct)
        if cfg.enable_adx_filter:
            core_values.append(adx_val)
        if cfg.enable_efficiency_filter:
            core_values.append(efficiency_val)
        if close_val is None or close_val <= 0 or any(v is None for v in core_values):
            # Some indicator is still NaN (or the data is malformed) —
            # never trade with partially initialized indicators.
            return self._hold(
                ts,
                "insufficient_data",
                {"detail": "indicators_not_initialized"},
                short_ma=short_ma,
                long_ma=long_ma,
                rsi=rsi_val,
                atr=atr_val,
                volume=vol,
                average_volume=avg_vol,
                volume_ratio=vol_ratio,
                atr_percentile=atr_pct,
                adx=adx_val,
                efficiency_ratio=efficiency_val,
            )

        invalid_indicator = (
            short_ma <= 0
            or long_ma <= 0
            or not 0.0 <= rsi_val <= 100.0
            or atr_val <= 0
            or vol < 0
            or avg_vol <= 0
            or vol_ratio < 0
            or (
                cfg.enable_volatility_filter
                and not 0.0 <= atr_pct <= 100.0
            )
            or (cfg.enable_adx_filter and not 0.0 <= adx_val <= 100.0)
            or (
                cfg.enable_efficiency_filter
                and not 0.0 <= efficiency_val <= 1.0
            )
        )
        if invalid_indicator:
            return self._hold(
                ts,
                "invalid_indicator_values",
                {"detail": "indicator_out_of_range"},
                short_ma=short_ma,
                long_ma=long_ma,
                rsi=rsi_val,
                atr=atr_val,
                volume=vol,
                average_volume=avg_vol,
                volume_ratio=vol_ratio,
                atr_percentile=atr_pct,
                adx=adx_val,
                efficiency_ratio=efficiency_val,
            )

        market_context = self._market_context(df, market_data, ts)
        if market_context.get("error"):
            return self._hold(
                ts,
                "market_context_unavailable",
                {
                    "detail": market_context["error"],
                    "regime_symbol": cfg.regime_symbol,
                    "regime_timeframe": cfg.regime_timeframe,
                },
                short_ma=short_ma,
                long_ma=long_ma,
                rsi=rsi_val,
                atr=atr_val,
                volume=vol,
                average_volume=avg_vol,
                volume_ratio=vol_ratio,
                atr_percentile=atr_pct,
                adx=adx_val,
                efficiency_ratio=efficiency_val,
            )

        # Exact crossover semantics: a NEW cross on this bar, not merely a
        # bullish/bearish state.  This prevents repeated BUYs in a trend.
        bullish_crossover = prev_short <= prev_long and short_ma > long_ma
        bearish_crossover = prev_short >= prev_long and short_ma < long_ma

        rsi_passed = rsi_val < cfg.rsi_overbought and (
            not cfg.enable_rsi_floor or rsi_val >= cfg.rsi_entry_floor
        )
        volume_passed = vol_ratio >= cfg.min_volume_ratio
        if cfg.enable_volatility_filter:
            volatility_passed = atr_pct is not None and atr_pct <= cfg.max_atr_percentile
        else:
            volatility_passed = True
        adx_passed = not cfg.enable_adx_filter or (
            adx_val is not None and adx_val >= cfg.min_adx
        )
        efficiency_passed = not cfg.enable_efficiency_filter or (
            efficiency_val is not None
            and efficiency_val >= cfg.min_efficiency_ratio
        )
        regime_passed = not cfg.enable_market_regime_filter or bool(
            market_context.get("bullish")
        )
        relative_strength = market_context.get("relative_strength")
        relative_strength_passed = not cfg.enable_relative_strength_filter or (
            relative_strength is not None
            and relative_strength >= cfg.min_relative_strength
        )

        filters = {
            "trend": bullish_crossover,
            "rsi": rsi_passed,
            "volume": volume_passed,
            "volatility": volatility_passed,
        }
        if cfg.enable_adx_filter:
            filters["adx"] = adx_passed
        if cfg.enable_efficiency_filter:
            filters["efficiency"] = efficiency_passed
        if cfg.enable_market_regime_filter:
            filters["market_regime"] = regime_passed
        if cfg.enable_relative_strength_filter:
            filters["relative_strength"] = relative_strength_passed
        common = dict(
            bullish_crossover=bullish_crossover,
            bearish_crossover=bearish_crossover,
            short_ma=short_ma,
            long_ma=long_ma,
            rsi=rsi_val,
            atr=atr_val,
            volume=vol,
            average_volume=avg_vol,
            volume_ratio=vol_ratio,
            atr_percentile=atr_pct,
            adx=adx_val,
            efficiency_ratio=efficiency_val,
            market_regime_bullish=market_context.get("bullish"),
            regime_close=market_context.get("close"),
            regime_ma=market_context.get("ma"),
            relative_strength=relative_strength,
            filters=filters,
            details={
                "close": close_val,
                "prev_short_ma": prev_short,
                "prev_long_ma": prev_long,
                "rsi_overbought": cfg.rsi_overbought,
                "min_volume_ratio": cfg.min_volume_ratio,
                "max_atr_percentile": cfg.max_atr_percentile if cfg.enable_volatility_filter else None,
                "volatility_filter_enabled": cfg.enable_volatility_filter,
                "current_volume": vol,
                "volume_passed": volume_passed,
                "rsi_floor_enabled": cfg.enable_rsi_floor,
                "rsi_entry_floor": cfg.rsi_entry_floor if cfg.enable_rsi_floor else None,
                "adx_minimum": cfg.min_adx if cfg.enable_adx_filter else None,
                "efficiency_minimum": (
                    cfg.min_efficiency_ratio if cfg.enable_efficiency_filter else None
                ),
                "regime_symbol": cfg.regime_symbol,
                "regime_timeframe": cfg.regime_timeframe,
                "regime_slope": market_context.get("slope"),
                "relative_strength_minimum": (
                    cfg.min_relative_strength
                    if cfg.enable_relative_strength_filter
                    else None
                ),
            },
        )

        if bearish_crossover:
            return SignalDecision(
                action=BotAction.EXIT, timestamp=ts, reason="bearish_crossover", **common
            )

        if (
            cfg.enable_market_regime_filter
            and cfg.exit_on_regime_break
            and market_context.get("previous_bullish") is True
            and market_context.get("bullish") is False
        ):
            return SignalDecision(
                action=BotAction.EXIT,
                timestamp=ts,
                reason="market_regime_break",
                **common,
            )

        if bullish_crossover:
            if all(ok for name, ok in filters.items() if name != "trend"):
                return SignalDecision(
                    action=BotAction.BUY,
                    timestamp=ts,
                    reason="bullish_crossover_confirmed",
                    **common,
                )
            failed = [name for name, ok in filters.items() if name != "trend" and not ok]
            reason = "entry_blocked_" + "_".join(sorted(failed)) + "_filter"
            return SignalDecision(action=BotAction.HOLD, timestamp=ts, reason=reason, **common)

        return SignalDecision(action=BotAction.HOLD, timestamp=ts, reason="no_crossover", **common)

    def _market_context(
        self,
        price_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame],
        signal_ts,
    ) -> dict:
        """Causal benchmark regime/relative-strength values as of a signal.

        Separate benchmark bars become visible only after their configured
        duration has elapsed.  The signal itself is evaluated after the symbol
        bar has elapsed, so equal-timeframe symbol and benchmark bars align on
        the same completed period.  This prevents a historical daily close
        (whose bar is timestamped near its open) from leaking into an intraday
        decision made before that close existed.
        """
        cfg = self.config
        if not (cfg.enable_market_regime_filter or cfg.enable_relative_strength_filter):
            return {}

        same_series = (
            cfg.regime_symbol == cfg.symbol
            and cfg.regime_timeframe == cfg.timeframe
        )
        source = price_data if same_series else market_data
        if source is None or not isinstance(source, pd.DataFrame) or source.empty:
            return {"error": "benchmark_bars_missing"}
        if "close" not in source.columns:
            return {"error": "benchmark_close_missing"}
        if not isinstance(source.index, pd.DatetimeIndex) or source.index.tz is None:
            return {"error": "benchmark_index_not_timezone_aware"}
        if not source.index.is_monotonic_increasing or source.index.has_duplicates:
            return {"error": "benchmark_bars_not_normalized"}

        available = source
        if not same_series:
            duration = pd.Timedelta(timeframe_to_timedelta(cfg.regime_timeframe))
            decision_time = pd.Timestamp(signal_ts) + pd.Timedelta(
                timeframe_to_timedelta(cfg.timeframe)
            )
            available = source[(source.index + duration) <= decision_time]
        required = cfg.regime_warmup_bars()
        if len(available) < required:
            return {
                "error": "benchmark_warmup_incomplete",
                "bars_available": int(len(available)),
                "bars_required": required,
            }

        market_close = available["close"].astype(float)
        result: dict = {
            "bar_timestamp": available.index[-1].isoformat(),
        }
        if cfg.enable_market_regime_filter:
            market_ma = sma(market_close, cfg.regime_ma_period)
            market_slope = market_ma - market_ma.shift(cfg.regime_slope_period)
            current_values = (
                _finite(market_close.iloc[-1]),
                _finite(market_ma.iloc[-1]),
                _finite(market_slope.iloc[-1]),
            )
            previous_values = (
                _finite(market_close.iloc[-2]),
                _finite(market_ma.iloc[-2]),
                _finite(market_slope.iloc[-2]),
            )
            if any(value is None for value in (*current_values, *previous_values)):
                return {"error": "benchmark_indicators_not_initialized"}
            close_value, ma_value, slope_value = current_values
            prev_close, prev_ma, prev_slope = previous_values
            result.update(
                {
                    "close": close_value,
                    "ma": ma_value,
                    "slope": slope_value,
                    "bullish": close_value > ma_value and slope_value > 0,
                    "previous_bullish": prev_close > prev_ma and prev_slope > 0,
                }
            )

        if cfg.enable_relative_strength_filter:
            period = cfg.relative_strength_period
            if len(price_data) < period + 1:
                return {"error": "relative_strength_warmup_incomplete"}
            symbol_close = price_data["close"].astype(float)
            comparison_index = symbol_close.index[-(period + 1) :]
            aligned_market = market_close.reindex(comparison_index)
            if aligned_market.isna().any():
                return {"error": "relative_strength_bars_not_aligned"}
            symbol_now = _finite(symbol_close.loc[comparison_index[-1]])
            symbol_then = _finite(symbol_close.loc[comparison_index[0]])
            market_now = _finite(aligned_market.iloc[-1])
            market_then = _finite(aligned_market.iloc[0])
            if (
                symbol_now is None
                or symbol_then is None
                or market_now is None
                or market_then is None
                or symbol_then <= 0
                or market_then <= 0
            ):
                return {"error": "relative_strength_not_computable"}
            result["relative_strength"] = (
                symbol_now / symbol_then - 1.0
            ) - (market_now / market_then - 1.0)
        return result

    def _hold(self, ts, reason: str, details: dict, **values) -> SignalDecision:
        filters = {
            "trend": False,
            "rsi": False,
            "volume": False,
            "volatility": False,
        }
        if self.config.enable_adx_filter:
            filters["adx"] = False
        if self.config.enable_efficiency_filter:
            filters["efficiency"] = False
        if self.config.enable_market_regime_filter:
            filters["market_regime"] = False
        if self.config.enable_relative_strength_filter:
            filters["relative_strength"] = False
        decision = SignalDecision(
            action=BotAction.HOLD,
            timestamp=ts,
            reason=reason,
            filters=filters,
            details=details,
            **values,
        )
        return decision
