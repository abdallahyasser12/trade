"""Probabilistic multi-input market-regime engine.

Produces a ``RegimeAssessment`` — a probability for every regime label
plus a confidence — instead of a single hard classification, so consumers
can size and gate proportionally (``P(trend)=0.71``) rather than flipping
behavior on a knife-edge threshold.

Inputs are deliberately independent measurements, not one moving average:

* trend structure   — benchmark close vs long MA, MA slope, medium momentum
* realized volatility — trailing percentile of short-window volatility
* volatility ratio  — short-window vol vs long-window vol (expansion state)
* breadth           — fraction of universe members above their own trend MA
* dispersion        — cross-sectional spread of member returns
* correlation       — mean pairwise correlation of member returns
* drawdown          — benchmark distance from its trailing peak

Optional inputs (breadth/dispersion/correlation) simply lower confidence
when absent — they are never fabricated.  All mappings are transparent,
bounded, and deterministic; there is nothing to fit and nothing that can
drift silently.  Whether a STRATEGY works in a given regime is measured
downstream by the marketplace's per-regime attribution statistics — this
engine only states what the market is doing, never which strategy to like.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd

from indicators import sma
from quant.contracts import Market, RegimeAssessment, RegimeLabel


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _finite(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True)
class RegimeConfig:
    """Measurement windows for the regime engine (bars of the input frame)."""

    trend_ma_period: int = 200
    trend_slope_period: int = 20
    fast_ma_period: int = 50
    momentum_period: int = 63
    vol_short_period: int = 20
    vol_long_period: int = 100
    vol_percentile_lookback: int = 252
    drawdown_lookback: int = 252
    correlation_window: int = 40
    # Volatility ratio above which expansion probability saturates.
    vol_expansion_saturation: float = 2.0
    # Annualized-free thresholds on the vol percentile for stress scoring.
    stress_vol_percentile: float = 85.0
    stress_drawdown_pct: float = 15.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.trend_ma_period < 10:
            errors.append("Trend MA period must be at least 10 bars.")
        if self.trend_slope_period < 1:
            errors.append("Trend slope period must be at least 1 bar.")
        if self.fast_ma_period < 2:
            errors.append("Fast MA period must be at least 2 bars.")
        if self.fast_ma_period >= self.trend_ma_period:
            errors.append("Fast MA period must be below the trend MA period.")
        if self.momentum_period < 2:
            errors.append("Momentum period must be at least 2 bars.")
        if self.vol_short_period < 5:
            errors.append("Short volatility window must be at least 5 bars.")
        if self.vol_long_period <= self.vol_short_period:
            errors.append("Long volatility window must exceed the short window.")
        if self.vol_percentile_lookback < 30:
            errors.append("Volatility percentile lookback must be at least 30 bars.")
        if self.drawdown_lookback < 30:
            errors.append("Drawdown lookback must be at least 30 bars.")
        if self.correlation_window < 10:
            errors.append("Correlation window must be at least 10 bars.")
        if self.vol_expansion_saturation <= 1.0:
            errors.append("Volatility expansion saturation must exceed 1.")
        if not (50.0 <= self.stress_vol_percentile <= 100.0):
            errors.append("Stress volatility percentile must be in [50, 100].")
        if self.stress_drawdown_pct <= 0:
            errors.append("Stress drawdown threshold must be positive.")
        return errors

    def required_bars(self) -> int:
        return (
            max(
                self.trend_ma_period + self.trend_slope_period,
                self.momentum_period,
                self.vol_short_period + self.vol_percentile_lookback,
                self.vol_long_period,
                self.drawdown_lookback,
            )
            + 1
        )


class RegimeEngine:
    """Deterministic regime scorer over completed benchmark/universe bars."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid regime configuration: " + "; ".join(errors))

    def required_bars(self) -> int:
        return self.config.required_bars()

    def assess(
        self,
        benchmark_bars: pd.DataFrame,
        *,
        market: Market,
        member_closes: Optional[Mapping[str, pd.Series]] = None,
    ) -> RegimeAssessment:
        """Score the regime as of the LAST COMPLETED bar of ``benchmark_bars``.

        ``member_closes`` optionally maps universe symbols to close series
        aligned on completed bars; it powers breadth, dispersion, and
        correlation.  Missing members reduce confidence, nothing more.
        """
        cfg = self.config
        if (
            not isinstance(benchmark_bars, pd.DataFrame)
            or "close" not in benchmark_bars.columns
        ):
            raise ValueError("Regime assessment requires benchmark bars with a close column.")
        if not isinstance(benchmark_bars.index, pd.DatetimeIndex) or benchmark_bars.index.tz is None:
            raise ValueError("Benchmark bars need a timezone-aware DatetimeIndex.")
        if not benchmark_bars.index.is_monotonic_increasing or benchmark_bars.index.has_duplicates:
            raise ValueError("Benchmark bars must be chronological and de-duplicated.")

        close = benchmark_bars["close"].astype(float)
        ts_raw = benchmark_bars.index[-1]
        ts = ts_raw.to_pydatetime() if hasattr(ts_raw, "to_pydatetime") else ts_raw

        if len(close) < cfg.required_bars():
            return RegimeAssessment(
                market=market,
                timestamp=ts,
                probabilities={label.value: 0.0 for label in RegimeLabel},
                confidence=0.0,
                inputs={"bars_available": float(len(close))},
            )

        inputs: dict[str, Optional[float]] = {}

        trend_ma = sma(close, cfg.trend_ma_period)
        slope = trend_ma - trend_ma.shift(cfg.trend_slope_period)
        fast_ma = sma(close, cfg.fast_ma_period)
        price = _finite(close.iloc[-1])
        ma_value = _finite(trend_ma.iloc[-1])
        slope_value = _finite(slope.iloc[-1])
        fast_value = _finite(fast_ma.iloc[-1])
        momentum = _finite(close.iloc[-1] / close.iloc[-(cfg.momentum_period + 1)] - 1.0)

        returns = close.pct_change()
        vol_short = returns.rolling(cfg.vol_short_period, min_periods=cfg.vol_short_period).std(ddof=0)
        vol_long = returns.rolling(cfg.vol_long_period, min_periods=cfg.vol_long_period).std(ddof=0)
        vol_now = _finite(vol_short.iloc[-1])
        vol_base = _finite(vol_long.iloc[-1])
        window = vol_short.iloc[-cfg.vol_percentile_lookback :].dropna()
        vol_percentile = (
            float((window <= vol_now).mean() * 100.0)
            if vol_now is not None and len(window) >= cfg.vol_percentile_lookback // 2
            else None
        )

        peak = close.iloc[-cfg.drawdown_lookback :].max()
        drawdown_pct = (
            _finite((peak - close.iloc[-1]) / peak * 100.0) if peak and peak > 0 else None
        )

        inputs.update(
            close=price,
            trend_ma=ma_value,
            trend_slope=slope_value,
            fast_ma=fast_value,
            momentum=momentum,
            vol_short=vol_now,
            vol_long=vol_base,
            vol_percentile=vol_percentile,
            drawdown_pct=drawdown_pct,
        )

        breadth = dispersion = correlation = None
        if member_closes:
            breadth, dispersion, correlation = self._universe_inputs(member_closes)
        inputs.update(breadth=breadth, dispersion=dispersion, correlation=correlation)

        core = (price, ma_value, slope_value, fast_value, momentum, vol_now, vol_base)
        if any(value is None for value in core):
            return RegimeAssessment(
                market=market,
                timestamp=ts,
                probabilities={label.value: 0.0 for label in RegimeLabel},
                confidence=0.0,
                inputs=inputs,
            )

        # --- trend mass -------------------------------------------------
        # Independent bounded evidence scores in [-1, 1], averaged, then
        # split across five trend labels by fixed shape functions.
        above_ma = _clip((price - ma_value) / (ma_value * 0.10), -1.0, 1.0)
        slope_score = _clip(slope_value / (ma_value * 0.02), -1.0, 1.0)
        momentum_score = _clip(momentum / 0.15, -1.0, 1.0)
        fast_alignment = _clip((fast_value - ma_value) / (ma_value * 0.05), -1.0, 1.0)
        breadth_score = 0.0 if breadth is None else (breadth - 0.5) * 2.0
        evidence = [above_ma, slope_score, momentum_score, fast_alignment]
        if breadth is not None:
            evidence.append(breadth_score)
        trend_strength = sum(evidence) / len(evidence)  # in [-1, 1]

        strong_bull = _clip((trend_strength - 0.35) / 0.65) ** 1.0
        weak_bull = _clip(trend_strength / 0.35) - strong_bull
        strong_bear = _clip((-trend_strength - 0.35) / 0.65)
        weak_bear = _clip(-trend_strength / 0.35) - strong_bear
        range_mass = _clip(1.0 - abs(trend_strength) / 0.35)
        total = strong_bull + weak_bull + strong_bear + weak_bear + range_mass
        if total <= 0:
            strong_bull = weak_bull = strong_bear = weak_bear = 0.0
            range_mass = 1.0
            total = 1.0

        # --- volatility state -------------------------------------------
        vol_ratio = vol_now / vol_base if vol_base and vol_base > 0 else 1.0
        expansion = _clip((vol_ratio - 1.0) / (cfg.vol_expansion_saturation - 1.0))
        contraction = _clip((1.0 - vol_ratio) / 0.5)

        # --- liquidity stress -------------------------------------------
        stress_components: list[float] = []
        if vol_percentile is not None:
            stress_components.append(
                _clip((vol_percentile - cfg.stress_vol_percentile) / (100.0 - cfg.stress_vol_percentile))
            )
        if drawdown_pct is not None:
            stress_components.append(_clip(drawdown_pct / cfg.stress_drawdown_pct))
        if correlation is not None:
            stress_components.append(_clip((correlation - 0.5) / 0.5))
        stress = (
            sum(stress_components) / len(stress_components) if stress_components else 0.0
        )

        probabilities = {
            RegimeLabel.STRONG_BULL.value: strong_bull / total,
            RegimeLabel.WEAK_BULL.value: weak_bull / total,
            RegimeLabel.RANGE.value: range_mass / total,
            RegimeLabel.WEAK_BEAR.value: weak_bear / total,
            RegimeLabel.STRONG_BEAR.value: strong_bear / total,
            RegimeLabel.VOL_EXPANSION.value: expansion,
            RegimeLabel.VOL_CONTRACTION.value: contraction,
            RegimeLabel.LIQUIDITY_STRESS.value: stress,
        }

        # Confidence: how much of the optional evidence actually arrived,
        # scaled by history sufficiency.  Core inputs are guaranteed here.
        optional_available = sum(
            1 for value in (breadth, dispersion, correlation, vol_percentile, drawdown_pct) if value is not None
        )
        confidence = _clip(0.6 + 0.08 * optional_available)

        return RegimeAssessment(
            market=market,
            timestamp=ts,
            probabilities=probabilities,
            confidence=confidence,
            inputs=inputs,
        )

    def _universe_inputs(
        self, member_closes: Mapping[str, pd.Series]
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Breadth, dispersion, and mean pairwise correlation of members."""
        cfg = self.config
        above = 0
        counted = 0
        returns_window: dict[str, pd.Series] = {}
        member_returns_20: list[float] = []
        for symbol, series in member_closes.items():
            closes = series.astype(float).dropna()
            if len(closes) < cfg.fast_ma_period + 1:
                continue
            ma = sma(closes, cfg.fast_ma_period)
            last_close = _finite(closes.iloc[-1])
            last_ma = _finite(ma.iloc[-1])
            if last_close is None or last_ma is None:
                continue
            counted += 1
            if last_close > last_ma:
                above += 1
            if len(closes) >= 21:
                ret20 = _finite(closes.iloc[-1] / closes.iloc[-21] - 1.0)
                if ret20 is not None:
                    member_returns_20.append(ret20)
            if len(closes) >= cfg.correlation_window + 1:
                returns_window[symbol] = closes.pct_change().iloc[-cfg.correlation_window :]

        breadth = above / counted if counted >= 3 else None
        dispersion = (
            float(pd.Series(member_returns_20).std(ddof=0))
            if len(member_returns_20) >= 3
            else None
        )
        correlation = None
        if len(returns_window) >= 3:
            frame = pd.DataFrame(returns_window).dropna()
            if len(frame) >= cfg.correlation_window // 2:
                matrix = frame.corr()
                n = len(matrix)
                if n >= 2:
                    off_diagonal = (matrix.sum().sum() - n) / (n * (n - 1))
                    correlation = _finite(off_diagonal)
        return breadth, dispersion, correlation
