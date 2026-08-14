"""Long-only moving-average trend strategy with RSI, volume and
volatility filters.

This is the ONLY implementation of the trading rules.  Both the
backtester and the live trade engine call ``TrendStrategy.generate_signal``
on a DataFrame of COMPLETED bars — there is no separate live/backtest
rule code, which guarantees decision parity between the two modes.

Rules (all evaluated on the last completed bar):

* BUY  — bullish crossover (prev short_ma <= prev long_ma AND
  current short_ma > current long_ma) AND RSI < overbought AND
  volume_ratio >= minimum AND (volatility filter disabled OR
  ATR percentile <= maximum).
* EXIT — bearish crossover (prev short_ma >= prev long_ma AND
  current short_ma < current long_ma).  The engine/backtester turn EXIT
  into a no-op when no position is open.
* HOLD — anything else, including insufficient warmup data.

RSI is an entry filter only; there is no RSI-based exit rule.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from config import StrategyConfig
from indicators import compute_indicators
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

    def generate_signal(self, price_data: pd.DataFrame) -> SignalDecision:
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
        close_val = _finite(curr["close"])

        core_values = [short_ma, long_ma, prev_short, prev_long, rsi_val, atr_val, vol, avg_vol, vol_ratio]
        if cfg.enable_volatility_filter:
            core_values.append(atr_pct)
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
            )

        # Exact crossover semantics: a NEW cross on this bar, not merely a
        # bullish/bearish state.  This prevents repeated BUYs in a trend.
        bullish_crossover = prev_short <= prev_long and short_ma > long_ma
        bearish_crossover = prev_short >= prev_long and short_ma < long_ma

        rsi_passed = rsi_val < cfg.rsi_overbought
        volume_passed = vol_ratio >= cfg.min_volume_ratio
        if cfg.enable_volatility_filter:
            volatility_passed = atr_pct is not None and atr_pct <= cfg.max_atr_percentile
        else:
            volatility_passed = True

        filters = {
            "trend": bullish_crossover,
            "rsi": rsi_passed,
            "volume": volume_passed,
            "volatility": volatility_passed,
        }
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
            },
        )

        if bearish_crossover:
            return SignalDecision(
                action=BotAction.EXIT, timestamp=ts, reason="bearish_crossover", **common
            )

        if bullish_crossover:
            if rsi_passed and volume_passed and volatility_passed:
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

    def _hold(self, ts, reason: str, details: dict, **values) -> SignalDecision:
        decision = SignalDecision(
            action=BotAction.HOLD,
            timestamp=ts,
            reason=reason,
            filters={"trend": False, "rsi": False, "volume": False, "volatility": False},
            details=details,
            **values,
        )
        return decision
