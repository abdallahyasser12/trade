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
* flow pressure: signed-volume balance (an order-flow proxy from bars);
* wick rejection: lower-wick versus upper-wick dominance over recent bars;
* structure persistence: higher-close and higher-low frequency (micro
  market structure rather than a smoothed average);
* horizon agreement: whether fast and medium impulse point the same way,
  interpreted per profile (a reversion entry *wants* the medium horizon
  still stretched while the fast horizon turns);
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
    impulse_medium_lookback: int = 15
    efficiency_lookback: int = 12
    range_lookback: int = 20
    volume_lookback: int = 20
    vwap_lookback: int = 20
    volatility_short_lookback: int = 8
    volatility_long_lookback: int = 32
    trend_fast_span: int = 8
    trend_slow_span: int = 21
    atr_lookback: int = 14
    flow_lookback: int = 10
    wick_lookback: int = 8
    structure_lookback: int = 10
    min_quality_multiplier: float = 0.55

    def validate(self) -> list[str]:
        errors: list[str] = []
        lookbacks = {
            "min_bars": self.min_bars,
            "impulse_lookback": self.impulse_lookback,
            "impulse_medium_lookback": self.impulse_medium_lookback,
            "efficiency_lookback": self.efficiency_lookback,
            "range_lookback": self.range_lookback,
            "volume_lookback": self.volume_lookback,
            "vwap_lookback": self.vwap_lookback,
            "volatility_short_lookback": self.volatility_short_lookback,
            "volatility_long_lookback": self.volatility_long_lookback,
            "trend_fast_span": self.trend_fast_span,
            "trend_slow_span": self.trend_slow_span,
            "atr_lookback": self.atr_lookback,
            "flow_lookback": self.flow_lookback,
            "wick_lookback": self.wick_lookback,
            "structure_lookback": self.structure_lookback,
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
        if self.impulse_lookback >= self.impulse_medium_lookback:
            errors.append("impulse_lookback must be below impulse_medium_lookback.")
        required = max(
            self.range_lookback + 2,
            self.volume_lookback + 2,
            self.vwap_lookback + 2,
            self.volatility_long_lookback + 2,
            self.trend_slow_span * 2,
            self.atr_lookback + 2,
            self.efficiency_lookback + 2,
            self.impulse_lookback + 2,
            self.impulse_medium_lookback + 2,
            self.flow_lookback + 2,
            self.wick_lookback + 2,
            self.structure_lookback + 2,
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
    if "scalp" in text:
        return "scalp"
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
    """Reinterpret raw evidence so every component means 'fit for this profile'.

    After this transformation a HIGH value always means the evidence supports
    the profile's kind of entry, which is what lets the agreement factor treat
    all profiles uniformly.
    """
    components = dict(raw)
    range_location = components["range_location"]
    vwap_stretch = diagnostics.get("vwap_stretch_atr")
    volatility_ratio = diagnostics.get("volatility_release_ratio")
    trend_atr = diagnostics.get("trend_atr")
    impulse_z = diagnostics.get("impulse_z")
    flow_balance = diagnostics.get("flow_balance")
    structure_raw = diagnostics.get("structure_raw")
    fast_sig = diagnostics.get("impulse_fast_sigmoid")
    medium_sig = diagnostics.get("impulse_medium_sigmoid")

    if profile == "mean_reversion":
        # A reversion long buys a controlled dip: chop is fine, a fresh
        # bullish impulse is late, heavy one-way selling is a falling knife.
        components["path_efficiency"] = 1.0 - components["path_efficiency"]
        components["range_location"] = _triangular(range_location, 0.35, 0.55)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, -0.55, 1.45)
        if volatility_ratio is not None:
            components["volatility_release"] = _sigmoid((1.05 - volatility_ratio) * 2.0)
        if trend_atr is not None:
            components["trend_alignment"] = _triangular(trend_atr, -0.35, 1.35)
        if impulse_z is not None:
            components["impulse_to_noise"] = _triangular(impulse_z, -0.90, 2.00)
        if flow_balance is not None:
            components["flow_pressure"] = _triangular(flow_balance, -0.20, 0.80)
        if structure_raw is not None:
            components["structure_persistence"] = _triangular(structure_raw, 0.45, 0.45)
        if fast_sig is not None and medium_sig is not None:
            # Reversion timing: fast horizon turning up while the medium
            # horizon is still stretched down.
            components["horizon_agreement"] = _clip01(
                math.sqrt(fast_sig * (1.0 - medium_sig))
            )
    elif profile == "trend_pullback":
        components["range_location"] = _triangular(range_location, 0.62, 0.55)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, 0.30, 1.50)
        if fast_sig is not None and medium_sig is not None:
            # The medium horizon carries the trend; the fast horizon only has
            # to be turning back up, not already extended.
            components["horizon_agreement"] = _clip01(
                (medium_sig**0.65) * (fast_sig**0.35)
            )
    elif profile == "breakout":
        components["range_location"] = _clip01((range_location - 0.45) / 0.55)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, 0.75, 1.75)
    elif profile == "trend":
        components["range_location"] = _clip01((range_location - 0.25) / 0.75)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, 0.45, 1.80)
    elif profile == "scalp":
        # Scalps need immediate continuation in an active tape: constructive
        # but not extended location, with the fast horizon dominant.
        components["range_location"] = _clip01((range_location - 0.40) / 0.60)
        if vwap_stretch is not None:
            components["vwap_location"] = _triangular(vwap_stretch, 0.35, 1.40)
        if fast_sig is not None and medium_sig is not None:
            components["horizon_agreement"] = _clip01(
                (fast_sig**0.65) * (medium_sig**0.35)
            )
    else:
        if vwap_stretch is not None:
            components["vwap_location"] = _clip01(1.0 - max(0.0, abs(vwap_stretch) - 1.0) / 2.5)
    return components


def _weights(profile: str) -> dict[str, float]:
    if profile == "breakout":
        return {
            "trend_alignment": 0.10,
            "impulse_to_noise": 0.14,
            "path_efficiency": 0.09,
            "range_location": 0.14,
            "participation": 0.11,
            "vwap_location": 0.05,
            "volatility_release": 0.08,
            "candle_pressure": 0.07,
            "flow_pressure": 0.08,
            "wick_rejection": 0.04,
            "structure_persistence": 0.04,
            "horizon_agreement": 0.06,
        }
    if profile == "mean_reversion":
        return {
            "trend_alignment": 0.04,
            "impulse_to_noise": 0.09,
            "path_efficiency": 0.11,
            "range_location": 0.11,
            "participation": 0.08,
            "vwap_location": 0.17,
            "volatility_release": 0.06,
            "candle_pressure": 0.12,
            "flow_pressure": 0.07,
            "wick_rejection": 0.07,
            "structure_persistence": 0.03,
            "horizon_agreement": 0.05,
        }
    if profile == "trend_pullback":
        return {
            "trend_alignment": 0.16,
            "impulse_to_noise": 0.09,
            "path_efficiency": 0.11,
            "range_location": 0.09,
            "participation": 0.07,
            "vwap_location": 0.12,
            "volatility_release": 0.05,
            "candle_pressure": 0.08,
            "flow_pressure": 0.07,
            "wick_rejection": 0.05,
            "structure_persistence": 0.05,
            "horizon_agreement": 0.06,
        }
    if profile == "trend":
        return {
            "trend_alignment": 0.16,
            "impulse_to_noise": 0.13,
            "path_efficiency": 0.12,
            "range_location": 0.08,
            "participation": 0.08,
            "vwap_location": 0.08,
            "volatility_release": 0.05,
            "candle_pressure": 0.07,
            "flow_pressure": 0.08,
            "wick_rejection": 0.04,
            "structure_persistence": 0.05,
            "horizon_agreement": 0.06,
        }
    if profile == "scalp":
        return {
            "trend_alignment": 0.05,
            "impulse_to_noise": 0.16,
            "path_efficiency": 0.08,
            "range_location": 0.10,
            "participation": 0.14,
            "vwap_location": 0.06,
            "volatility_release": 0.10,
            "candle_pressure": 0.10,
            "flow_pressure": 0.10,
            "wick_rejection": 0.04,
            "structure_persistence": 0.03,
            "horizon_agreement": 0.04,
        }
    return {
        "trend_alignment": 0.11,
        "impulse_to_noise": 0.12,
        "path_efficiency": 0.10,
        "range_location": 0.10,
        "participation": 0.09,
        "vwap_location": 0.09,
        "volatility_release": 0.07,
        "candle_pressure": 0.09,
        "flow_pressure": 0.08,
        "wick_rejection": 0.05,
        "structure_persistence": 0.05,
        "horizon_agreement": 0.05,
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
    # Defense in depth: production strategies already reject unsorted or
    # duplicated indexes, but label slicing on such an index would not be
    # causal, so an assessment is refused rather than risked.
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("unsorted_time_index",))
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
    # dropna may have removed the evaluated bar itself; measuring the prior
    # bar and presenting it as the assessment at ``ts`` would be a silent lie.
    if history.index[-1] != ts:
        return EdgeAssessment(False, profile, 50.0, 1.0, reasons=("evaluated_bar_invalid",))

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

    medium_raw = math.log(
        latest_close / float(close.iloc[-1 - config.impulse_medium_lookback])
    )
    impulse_medium_z = medium_raw / (
        long_sigma * math.sqrt(config.impulse_medium_lookback) + _EPS
    )
    impulse_fast_sig = _sigmoid(1.10 * impulse_z)
    impulse_medium_sig = _sigmoid(1.10 * impulse_medium_z)

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
    # A zero-range bar carries no close-location information; scoring it as a
    # close pinned at the low would fabricate bearish evidence.
    candle_pressure = (
        0.5
        if candle_range <= 0
        else _clip01((latest_close - float(low.iloc[-1])) / candle_range)
    )

    # Order-flow proxy: volume signed by candle direction. Bars markets do not
    # publish real trade signs; the close-vs-open sign weighted by volume is
    # the standard causal approximation.
    flow_slice = history.tail(config.flow_lookback)
    flow_signs = np.sign(
        flow_slice["close"].astype(float).to_numpy()
        - flow_slice["open"].astype(float).to_numpy()
    )
    flow_volumes = flow_slice["volume"].astype(float).to_numpy()
    flow_balance = float(
        (flow_signs * flow_volumes).sum() / (flow_volumes.sum() + _EPS)
    )
    flow_pressure = _clip01(0.5 * (1.0 + flow_balance))

    # Wick rejection: dominance of lower wicks (dip-buying) over upper wicks
    # (rally-selling) across the recent window, normalized by total range.
    wick_slice = history.tail(config.wick_lookback)
    wick_open = wick_slice["open"].astype(float)
    wick_close = wick_slice["close"].astype(float)
    wick_high = wick_slice["high"].astype(float)
    wick_low = wick_slice["low"].astype(float)
    body_top = pd.concat([wick_open, wick_close], axis=1).max(axis=1)
    body_bottom = pd.concat([wick_open, wick_close], axis=1).min(axis=1)
    upper_wicks = float((wick_high - body_top).sum())
    lower_wicks = float((body_bottom - wick_low).sum())
    total_ranges = float((wick_high - wick_low).sum())
    wick_balance = (
        (lower_wicks - upper_wicks) / total_ranges if total_ranges > 0 else 0.0
    )
    wick_rejection = _clip01(0.5 * (1.0 + wick_balance))

    # Micro market structure: how persistently the last bars print higher
    # closes and higher lows — structure, not a smoothed average.
    structure_slice = history.tail(config.structure_lookback + 1)
    higher_closes = structure_slice["close"].astype(float).diff().dropna()
    higher_lows = structure_slice["low"].astype(float).diff().dropna()
    structure_raw = 0.5 * float((higher_closes > 0).mean()) + 0.5 * float(
        (higher_lows > 0).mean()
    )
    structure_persistence = _clip01(structure_raw)

    # Multi-horizon agreement: default bullish continuation form; profiles
    # reinterpret it below (a reversion entry wants the medium horizon still
    # stretched down while the fast horizon turns).
    horizon_agreement = math.sqrt(impulse_fast_sig * impulse_medium_sig)

    raw_components = {
        "trend_alignment": trend_score,
        "impulse_to_noise": impulse_score,
        "path_efficiency": efficiency,
        "range_location": range_location,
        "participation": participation,
        "vwap_location": vwap_location,
        "volatility_release": volatility_release,
        "candle_pressure": candle_pressure,
        "flow_pressure": flow_pressure,
        "wick_rejection": wick_rejection,
        "structure_persistence": structure_persistence,
        "horizon_agreement": _clip01(horizon_agreement),
    }
    diagnostics = {
        "atr": atr_value,
        "impulse_z": impulse_z,
        "impulse_medium_z": impulse_medium_z,
        "impulse_fast_sigmoid": impulse_fast_sig,
        "impulse_medium_sigmoid": impulse_medium_sig,
        "path_efficiency": efficiency,
        "trend_atr": trend_atr,
        "range_location_raw": range_location,
        "relative_volume": relative_volume,
        "rolling_vwap": rolling_vwap,
        "vwap_stretch_atr": vwap_stretch_atr,
        "volatility_release_ratio": volatility_release_ratio,
        "flow_balance": flow_balance,
        "wick_balance": wick_balance,
        "structure_raw": structure_raw,
        **_quote_diagnostics(quote),
    }
    components = _profile_components(profile, raw_components, diagnostics)
    weights = _weights(profile)
    weighted = sum(weights[name] * components[name] for name in weights)

    # Internal coherence, computed AFTER profile transformation so that every
    # input means "fit for this profile" — a textbook dip-buy no longer reads
    # as disagreement just because its raw momentum is bearish.  Dispersion is
    # mean absolute deviation across the fit evidence (0 = unanimous).
    fit_evidence = [
        components["trend_alignment"],
        components["impulse_to_noise"],
        components["range_location"],
        components["candle_pressure"],
        components["flow_pressure"],
        components["wick_rejection"],
    ]
    fit_mean = sum(fit_evidence) / len(fit_evidence)
    fit_dispersion = _clip01(
        2.0 * sum(abs(value - fit_mean) for value in fit_evidence) / len(fit_evidence)
    )
    fit_agreement = _clip01(fit_mean * (1.0 - 0.45 * fit_dispersion))
    agreement_factor = 0.78 + 0.22 * fit_agreement
    setup_quality = _clip01(weighted * agreement_factor)
    multiplier = config.min_quality_multiplier + (
        1.0 - config.min_quality_multiplier
    ) * setup_quality
    diagnostics["fit_mean"] = fit_mean
    diagnostics["fit_dispersion"] = fit_dispersion
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
