"""Causal intraday setup-quality intelligence shared by equity and crypto scanners.

This module deliberately sits *above* strategy signal generation and *below* the
portfolio opportunity ranker.  It does not create BUY signals and it does not add
gross alpha.  Instead it measures whether the current completed-bar setup looks
internally coherent using independent, causal feature families and supplies a
bounded [min_quality_multiplier, 1.0] ranking multiplier.

The feature set combines familiar intraday evidence with deliberately independent
path/microstructure diagnostics:

* trend alignment: fast/slow EWMA separation normalized by ATR;
* impulse-to-noise: multi-bar log return normalized by realized volatility;
* path efficiency: net displacement divided by total travelled distance;
* range location: close location inside the prior rolling range;
* participation: current volume versus the *prior* rolling median volume;
* VWAP stretch: rolling VWAP displacement normalized by ATR;
* volatility release: short/long realized-volatility ratio;
* candle pressure: close location inside the current completed candle;
* agreement quality: penalizes internally contradictory directional evidence.

All calculations slice data at ``timestamp`` before doing any rolling work.  Future
bars therefore cannot alter a historical assessment.  Missing/uninitialized data is
neutral rather than fabricated: the normal strategy/lifecycle/execution gates remain
responsible for admission.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


_EPS = 1e-12
_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _triangular(value: float, center: float, half_width: float) -> float:
    if half_width <= 0:
        return 0.0
    return _clip01(1.0 - abs(value - center) / half_width)


@dataclass(frozen=True)
class EdgeIntelligenceConfig:
    min_bars: int = 64
    impulse_lookback: int = 5
    efficiency_lookback: int = 12
    range_lookback: int = 20
    volume_lookback: int = 20
    vwap_lookback: int = 20
    volatility_short_lookback: int = 8
    volatility_long_lookback: int = 32
    trend_fast_span: int = 8
    trend_slow_span: int = 21
    atr_lookback: int = 14
    min_quality_multiplier: float = 0.55

    def validate(self) -> list[str]:
        errors: list[str] = []
        lookbacks = {
            "min_bars": self.min_bars,
            "impulse_lookback": self.impulse_lookback,
            "efficiency_lookback": self.efficiency_lookback,
            "range_lookback": self.range_lookback,
            "volume_lookback": self.volume_lookback,
            "vwap_lookback": self.vwap_lookback,
            "volatility_short_lookback": self.volatility_short_lookback,
            "volatility_long_lookback": self.volatility_long_lookback,
            "trend_fast_span": self.trend_fast_span,
            "trend_slow_span": self.trend_slow_span,
            "atr_lookback": self.atr_lookback,
        }
        for name, value in lookbacks.items():
            if not isinstance(value, int) or value < 2:
                errors.append(f"{name} must be an integer >= 2.")
        if self.trend_fast_span >= self.trend_slow_span:
            errors.append("trend_fast_span must be below trend_slow_span.")
        if self.volatility_short_lookback >= self.volatility_long_lookback:
            errors.append(
                "volatility_short_lookback must be below volatility_long_lookback."
            )
        required = max(
            self.range_lookback + 2,
            self.volume_lookback + 2,
            self.vwap_lookback + 2,
            self.volatility_long_lookback + 2,
            self.trend_slow_span * 2,
            self.atr_lookback + 2,
            self.efficiency_lookback + 2,
            self.impulse_lookback + 2,
        )
        if self.min_bars < required:
            errors.append(f"min_bars must be at least {required} for configured lookbacks.")
        if not math.isfinite(self.min_quality_multiplier) or not (
            0.0 < self.min_quality_multiplier <= 1.0
        ):
            errors.append("min_quality_multiplier must be in (0, 1].")
        return errors


@dataclass(frozen=True)
class EdgeAssessment:
    measurable: bool
    profile: str
    score: float
    quality_multiplier: float
    components: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Optional[float]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurable": self.measurable,
            "profile": self.profile,
            "score": self.score,
            "quality_multiplier": self.quality_multiplier,
            "components": dict(self.components),
            "diagnostics": dict(self.diagnostics),
            "reasons": list(self.reasons),
        }


def edge_profile(strategy_id: str, signal_reason: str) -> str:
    text = f"{strategy_id} {signal_reason}".lower()
    if "reversion" in text or "mean_revert" in text or "mean-revert" in text:
        return "mean_reversion"
    if "pullback" in text:
        return "trend_pullback"
    if "breakout" in text or "squeeze" in text or "expansion" in text:
        return "breakout"
    if "momentum" in text or "trend" in text or "continuation" in text:
        return "trend"
    return "generic"


def _atr(frame: pd.DataFrame, lookback: int) -> Optional[float]:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = true_range.rolling(lookback, min_periods=lookback).mean().iloc[-1]
    return _finite(value)


def _quote_diagnostics(quote: Any) -> dict[str, Optional[float]]:
    if quote is None:
        return {"spread_bps": None, "l1_imbalance": None}
    bid = _finite(getattr(quote, "bid_price", None))
    ask = _finite(getattr(quote, "ask_price", None))
    bid_size = _finite(getattr(quote, "bid_size", None))
    ask_size = _finite(getattr(quote, "ask_size", None))
    spread_bps = None
    if bid is not None and ask is not None and bid > 0 and ask > bid:
        midpoint = (bid + ask) / 2.0
        spread_bps = (ask - bid) / midpoint * 10_000.0
    imbalance = None
    if (
        bid_size is not None
        and ask_size is not None
        and bid_size >= 0
        and ask_size >= 0
        and (bid_size + ask_size) > 0
    ):
        imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    return {"spread_bps": spread_bps, "l1_imbalance": imbalance}


def _profile_components(profile: str, raw: dict[str, float], diagnostics: dict[str, Optional[float]]):
    components = dict(raw)
    range_location = components["range_location"]
    vwap_stretch = diagnostics.get("vwap_stretch_atr")
    volatility_ratio = diagnostics.get("volatility_release_ratio")

    if profile == "mean_reversion":
        components["path_efficiency"] = 1.0 - components["path_efficiency"]
        components["range_location"] = _triangular(range_location, 0.35, 0.55)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, -0.55, 1.45)
        if volatility_ratio is not None:
            components["volatility_release"] = _sigmoid((1.05 - volatility_ratio) * 2.0)
    elif profile == "trend_pullback":
        components["range_location"] = _triangular(range_location, 0.62, 0.55)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, 0.30, 1.50)
    elif profile == "breakout":
        components["range_location"] = _clip01((range_location - 0.45) / 0.55)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, 0.75, 1.75)
    elif profile == "trend":
        components["range_location"] = _clip01((range_location - 0.25) / 0.75)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, 0.45, 1.80)
    else:
        if vwap_stretch is not None:
            components["vwap_location"] = _clip01(1.0 - max(0.0, abs(vwap_stretch) - 1.0) / 2.5)
    return components


def _weights(profile: str) -> dict[str, float]:
    if profile == "breakout":
        return {
            "trend_alignment": 0.14,
            "impulse_to_noise": 0.18,
            "path_efficiency": 0.12,
            "range_location": 0.18,
            "participation": 0.14,
            "vwap_location": 0.06,
            "volatility_release": 0.10,
            "candle_pressure": 0.08,
        }
    if profile == "mean_reversion":
        return {
            "trend_alignment": 0.05,
            "impulse_to_noise": 0.12,
            "path_efficiency": 0.14,
            "range_location": 0.14,
            "participation": 0.10,
            "vwap_location": 0.22,
            "volatility_release": 0.08,
            "candle_pressure": 0.15,
        }
    if profile == "trend_pullback":
        return {
            "trend_alignment": 0.20,
            "impulse_to_noise": 0.12,
            "path_efficiency": 0.14,
            "range_location": 0.12,
            "participation": 0.09,
            "vwap_location": 0.16,
            "volatility_release": 0.07,
            "candle_pressure": 0.10,
        }
    if profile == "trend":
        return {
            "trend_alignment": 0.20,
            "impulse_to_noise": 0.17,
            "path_efficiency": 0.15,
            "range_location": 0.11,
            "participation": 0.10,
            "vwap_location": 0.11,
            "volatility_release": 0.07,
            "candle_pressure": 0.09,
        }
    return {
        "trend_alignment": 0.15,
        "impulse_to_noise": 0.15,
        "path_efficiency": 0.13,
        "range_location": 0.13,
        "participation": 0.11,
        "vwap_location": 0.12,
        "volatility_release": 0.09,
        "candle_pressure": 0.12,
    }


def assess_edge(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    strategy_id: str,
    signal_reason: str,
    quote: Any = None,
    config: EdgeIntelligenceConfig = EdgeIntelligenceConfig(),
) -> EdgeAssessment:
    """Measure causal setup quality at one completed timestamp.

    The result is intentionally a ranking *multiplier*, not a new signal.  A measurable
    setup can only reduce the normal opportunity score; it never raises it above the
    evidence/cost/regime score produced by the rest of Quant Desk.
    """

    errors = config.validate()
    if errors:
        raise ValueError("Invalid edge-intelligence configuration: " + "; ".join(errors))
    profile = edge_profile(strategy_id, signal_reason)
    if not isinstance(frame, pd.DataFrame) or _REQUIRED_COLUMNS.difference(frame.columns):
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("missing_ohlcv",))
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("invalid_time_index",))
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("naive_timestamp",))
    if ts not in frame.index:
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("timestamp_missing",))

    history = frame.loc[:ts, ["open", "high", "low", "close", "volume"]].copy()
    history = history.replace([np.inf, -np.inf], np.nan).dropna()
    if len(history) < config.min_bars:
        return EdgeAssessment(
            False,
            profile,
            50.0,
            1.0,
            reasons=("insufficient_edge_history",),
        )

    close = history["close"].astype(float)
    high = history["high"].astype(float)
    low = history["low"].astype(float)
    volume = history["volume"].astype(float)
    if (close <= 0).any() or (high <= 0).any() or (low <= 0).any() or (volume < 0).any():
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("invalid_ohlcv_values",))

    atr_value = _atr(history, config.atr_lookback)
    if atr_value is None or atr_value <= 0:
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("atr_uninitialized",))

    latest_close = float(close.iloc[-1])
    log_returns = np.log(close / close.shift(1)).dropna()
    long_sigma = float(
        log_returns.tail(config.volatility_long_lookback).std(ddof=1)
    )
    short_sigma = float(
        log_returns.tail(config.volatility_short_lookback).std(ddof=1)
    )
    if not math.isfinite(long_sigma) or long_sigma <= 0:
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("volatility_uninitialized",))

    impulse_raw = math.log(latest_close / float(close.iloc[-1 - config.impulse_lookback]))
    impulse_z = impulse_raw / (long_sigma * math.sqrt(config.impulse_lookback) + _EPS)
    impulse_score = _sigmoid(1.15 * impulse_z)

    path = close.tail(config.efficiency_lookback + 1)
    travelled = float(path.diff().abs().dropna().sum())
    displacement = abs(float(path.iloc[-1] - path.iloc[0]))
    efficiency = _clip01(displacement / (travelled + _EPS))

    fast_ema = float(close.ewm(span=config.trend_fast_span, adjust=False).mean().iloc[-1])
    slow_ema = float(close.ewm(span=config.trend_slow_span, adjust=False).mean().iloc[-1])
    trend_atr = (fast_ema - slow_ema) / atr_value
    trend_score = _sigmoid(1.35 * trend_atr)

    prior_range = history.iloc[-(config.range_lookback + 1) : -1]
    prior_high = float(prior_range["high"].max())
    prior_low = float(prior_range["low"].min())
    range_location = _clip01((latest_close - prior_low) / (prior_high - prior_low + _EPS))

    prior_volume = volume.iloc[-(config.volume_lookback + 1) : -1]
    volume_baseline = float(prior_volume.median()) if len(prior_volume) else 0.0
    relative_volume = float(volume.iloc[-1]) / (volume_baseline + _EPS)
    participation = _clip01(relative_volume / 2.0)

    vwap_slice = history.tail(config.vwap_lookback)
    typical = (
        vwap_slice["high"].astype(float)
        + vwap_slice["low"].astype(float)
        + vwap_slice["close"].astype(float)
    ) / 3.0
    total_volume = float(vwap_slice["volume"].astype(float).sum())
    rolling_vwap = (
        float((typical * vwap_slice["volume"].astype(float)).sum()) / total_volume
        if total_volume > 0
        else latest_close
    )
    vwap_stretch_atr = (latest_close - rolling_vwap) / atr_value
    vwap_location = _clip01(1.0 - max(0.0, abs(vwap_stretch_atr) - 1.0) / 2.5)

    volatility_release_ratio = short_sigma / (long_sigma + _EPS)
    volatility_release = _sigmoid((volatility_release_ratio - 0.90) * 2.0)

    candle_range = float(high.iloc[-1] - low.iloc[-1])
    candle_pressure = _clip01(
        (latest_close - float(low.iloc[-1])) / (candle_range + _EPS)
    )

    raw_components = {
        "trend_alignment": trend_score,
        "impulse_to_noise": impulse_score,
        "path_efficiency": efficiency,
        "range_location": range_location,
        "participation": participation,
        "vwap_location": vwap_location,
        "volatility_release": volatility_release,
        "candle_pressure": candle_pressure,
    }
    diagnostics = {
        "atr": atr_value,
        "impulse_z": impulse_z,
        "path_efficiency": efficiency,
        "trend_atr": trend_atr,
        "range_location_raw": range_location,
        "relative_volume": relative_volume,
        "rolling_vwap": rolling_vwap,
        "vwap_stretch_atr": vwap_stretch_atr,
        "volatility_release_ratio": volatility_release_ratio,
        **_quote_diagnostics(quote),
    }
    components = _profile_components(profile, raw_components, diagnostics)
    weights = _weights(profile)
    weighted = sum(weights[name] * components[name] for name in weights)

    directional = [
        components["trend_alignment"],
        components["impulse_to_noise"],
        components["range_location"],
        components["candle_pressure"],
    ]
    directional_mean = sum(directional) / len(directional)
    disagreement = 1.0 - abs(2.0 * directional_mean - 1.0)
    bullish_agreement = _clip01(directional_mean * (1.0 - 0.45 * disagreement))
    agreement_factor = 0.78 + 0.22 * bullish_agreement
    setup_quality = _clip01(weighted * agreement_factor)
    multiplier = config.min_quality_multiplier + (
        1.0 - config.min_quality_multiplier
    ) * setup_quality
    diagnostics["directional_mean"] = directional_mean
    diagnostics["directional_disagreement"] = disagreement
    diagnostics["agreement_factor"] = agreement_factor

    return EdgeAssessment(
        measurable=True,
        profile=profile,
        score=round(setup_quality * 100.0, 6),
        quality_multiplier=_clip01(multiplier),
        components={name: _clip01(value) for name, value in components.items()},
        diagnostics=diagnostics,
        reasons=(),
    )
