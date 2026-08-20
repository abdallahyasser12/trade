"""Causal in-trade hold/exit intelligence: should an open long be kept?

``quant.edge_intelligence`` judges a setup *before* entry.  This module answers
the mirror question *after* entry: given only completed bars up to ``timestamp``,
does the market state still support holding the position, or has the move that
justified the trade stalled or reversed?

The output is an advisory :class:`HoldAssessment`, never an order:

* ``hold_score`` in [0, 100] — higher means the current state still supports
  holding a long;
* ``stance`` — HOLD / TIGHTEN / EXIT, derived from configured thresholds;
* per-component scores and diagnostics for the journal.

``TradeManagementPolicy`` (quant/exits.py) consumes the score only through
individually flag-gated behaviors that all default OFF, so nothing changes for
any deployed configuration until research measures the feature.  The assessment
is profile-aware for the same reason edge intelligence is: a stalled trend bar
is bad news for a momentum position but perfectly normal mid-reversion.

Causality contract: every calculation slices the frame at ``timestamp`` first;
appending future bars can never change a historical assessment.  Missing or
insufficient data yields a neutral, explicitly non-measurable HOLD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from quant.edge_intelligence import edge_profile

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


class HoldStance(str, Enum):
    HOLD = "HOLD"
    TIGHTEN = "TIGHTEN"
    EXIT = "EXIT"


@dataclass(frozen=True)
class HoldIntelligenceConfig:
    min_bars: int = 48
    impulse_lookback: int = 4
    stall_lookback: int = 6
    pressure_lookback: int = 3
    flow_lookback: int = 8
    efficiency_lookback: int = 10
    trend_fast_span: int = 8
    trend_slow_span: int = 21
    atr_lookback: int = 14
    volatility_lookback: int = 32
    # Stance thresholds on the 0-100 hold score.
    exit_threshold: float = 35.0
    tighten_threshold: float = 55.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        lookbacks = {
            "min_bars": self.min_bars,
            "impulse_lookback": self.impulse_lookback,
            "stall_lookback": self.stall_lookback,
            "pressure_lookback": self.pressure_lookback,
            "flow_lookback": self.flow_lookback,
            "efficiency_lookback": self.efficiency_lookback,
            "trend_fast_span": self.trend_fast_span,
            "trend_slow_span": self.trend_slow_span,
            "atr_lookback": self.atr_lookback,
            "volatility_lookback": self.volatility_lookback,
        }
        for name, value in lookbacks.items():
            if not isinstance(value, int) or value < 2:
                errors.append(f"{name} must be an integer >= 2.")
        if self.trend_fast_span >= self.trend_slow_span:
            errors.append("trend_fast_span must be below trend_slow_span.")
        required = max(
            self.trend_slow_span * 2,
            self.atr_lookback + 2,
            self.volatility_lookback + 2,
            self.impulse_lookback + 2,
            self.stall_lookback + 2,
            self.flow_lookback + 2,
            self.efficiency_lookback + 2,
        )
        if self.min_bars < required:
            errors.append(f"min_bars must be at least {required} for configured lookbacks.")
        if not (0.0 <= self.exit_threshold < self.tighten_threshold <= 100.0):
            errors.append("Thresholds must satisfy 0 <= exit < tighten <= 100.")
        return errors


@dataclass(frozen=True)
class HoldAssessment:
    measurable: bool
    profile: str
    hold_score: float
    stance: HoldStance
    components: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Optional[float]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurable": self.measurable,
            "profile": self.profile,
            "hold_score": self.hold_score,
            "stance": self.stance.value,
            "components": dict(self.components),
            "diagnostics": dict(self.diagnostics),
            "reasons": list(self.reasons),
        }


_NEUTRAL_SCORE = 50.0


def _neutral(profile: str, reason: str) -> HoldAssessment:
    return HoldAssessment(
        measurable=False,
        profile=profile,
        hold_score=_NEUTRAL_SCORE,
        stance=HoldStance.HOLD,
        reasons=(reason,),
    )


def _weights(profile: str) -> dict[str, float]:
    if profile == "mean_reversion":
        # A reversion long is allowed to fight the local trend; what matters is
        # whether buyers keep absorbing (flow/pressure) and the bounce advances.
        return {
            "trend_health": 0.06,
            "momentum_health": 0.16,
            "pressure_health": 0.24,
            "flow_health": 0.22,
            "advance_health": 0.20,
            "efficiency_health": 0.12,
        }
    if profile == "scalp":
        # A scalp lives or dies on immediate continuation: stall and pressure
        # dominate, slow trend structure barely matters at this horizon.
        return {
            "trend_health": 0.06,
            "momentum_health": 0.24,
            "pressure_health": 0.22,
            "flow_health": 0.16,
            "advance_health": 0.26,
            "efficiency_health": 0.06,
        }
    if profile in ("trend", "trend_pullback", "breakout"):
        return {
            "trend_health": 0.22,
            "momentum_health": 0.20,
            "pressure_health": 0.14,
            "flow_health": 0.14,
            "advance_health": 0.18,
            "efficiency_health": 0.12,
        }
    return {
        "trend_health": 0.17,
        "momentum_health": 0.19,
        "pressure_health": 0.16,
        "flow_health": 0.16,
        "advance_health": 0.18,
        "efficiency_health": 0.14,
    }


def hold_profile(strategy_id: str, signal_reason: str) -> str:
    """Profile used to interpret in-trade evidence; extends the entry mapping."""
    text = f"{strategy_id} {signal_reason}".lower()
    if "scalp" in text:
        return "scalp"
    return edge_profile(strategy_id, signal_reason)


def assess_hold(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    strategy_id: str,
    signal_reason: str,
    config: HoldIntelligenceConfig = HoldIntelligenceConfig(),
) -> HoldAssessment:
    """Measure whether completed-bar state still supports holding a long."""

    errors = config.validate()
    if errors:
        raise ValueError("Invalid hold-intelligence configuration: " + "; ".join(errors))
    profile = hold_profile(strategy_id, signal_reason)
    if not isinstance(frame, pd.DataFrame) or _REQUIRED_COLUMNS.difference(frame.columns):
        return _neutral(profile, "missing_ohlcv")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        return _neutral(profile, "invalid_time_index")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        return _neutral(profile, "unsorted_time_index")
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return _neutral(profile, "naive_timestamp")
    if ts not in frame.index:
        return _neutral(profile, "timestamp_missing")

    history = frame.loc[:ts, ["open", "high", "low", "close", "volume"]].copy()
    history = history.replace([np.inf, -np.inf], np.nan).dropna()
    if len(history) < config.min_bars:
        return _neutral(profile, "insufficient_hold_history")
    if history.index[-1] != ts:
        return _neutral(profile, "evaluated_bar_invalid")

    close = history["close"].astype(float)
    open_ = history["open"].astype(float)
    high = history["high"].astype(float)
    low = history["low"].astype(float)
    volume = history["volume"].astype(float)
    if (close <= 0).any() or (high <= 0).any() or (low <= 0).any() or (volume < 0).any():
        return _neutral(profile, "invalid_ohlcv_values")

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_value = _finite(
        true_range.rolling(config.atr_lookback, min_periods=config.atr_lookback)
        .mean()
        .iloc[-1]
    )
    if atr_value is None or atr_value <= 0:
        return _neutral(profile, "atr_uninitialized")

    log_returns = np.log(close / close.shift(1)).dropna()
    sigma = float(log_returns.tail(config.volatility_lookback).std(ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return _neutral(profile, "volatility_uninitialized")

    latest_close = float(close.iloc[-1])

    fast_ema = float(close.ewm(span=config.trend_fast_span, adjust=False).mean().iloc[-1])
    slow_ema = float(close.ewm(span=config.trend_slow_span, adjust=False).mean().iloc[-1])
    trend_atr = (fast_ema - slow_ema) / atr_value
    trend_health = _sigmoid(1.35 * trend_atr)

    impulse_raw = math.log(latest_close / float(close.iloc[-1 - config.impulse_lookback]))
    impulse_z = impulse_raw / (sigma * math.sqrt(config.impulse_lookback) + _EPS)
    momentum_health = _sigmoid(1.15 * impulse_z)

    pressure_slice = history.tail(config.pressure_lookback)
    pressure_ranges = (
        pressure_slice["high"].astype(float) - pressure_slice["low"].astype(float)
    ).to_numpy()
    pressure_raw = (
        pressure_slice["close"].astype(float) - pressure_slice["low"].astype(float)
    ).to_numpy()
    # Zero-range bars carry no close-location information: neutral, not bearish.
    pressure_values = np.where(
        pressure_ranges > 0, pressure_raw / np.where(pressure_ranges > 0, pressure_ranges, 1.0), 0.5
    )
    pressure_health = _clip01(float(pressure_values.mean()))

    flow_slice = history.tail(config.flow_lookback)
    signed = np.sign(
        flow_slice["close"].astype(float).to_numpy()
        - flow_slice["open"].astype(float).to_numpy()
    )
    flow_volume = flow_slice["volume"].astype(float).to_numpy()
    flow_balance = float((signed * flow_volume).sum() / (flow_volume.sum() + _EPS))
    flow_health = _clip01(0.5 * (1.0 + flow_balance))

    stall_closes = close.tail(config.stall_lookback)
    advances = float((stall_closes.diff().dropna() > 0).mean()) if len(stall_closes) > 1 else 0.5
    peak_position = int(np.argmax(stall_closes.to_numpy()))
    bars_since_peak = (len(stall_closes) - 1) - peak_position
    recency = 1.0 - bars_since_peak / max(1, len(stall_closes) - 1)
    advance_health = _clip01(0.5 * advances + 0.5 * recency)

    path = close.tail(config.efficiency_lookback + 1)
    travelled = float(path.diff().abs().dropna().sum())
    displacement = float(path.iloc[-1] - path.iloc[0])
    signed_efficiency = displacement / (travelled + _EPS)
    efficiency_health = _clip01(0.5 * (1.0 + signed_efficiency))

    components = {
        "trend_health": trend_health,
        "momentum_health": momentum_health,
        "pressure_health": pressure_health,
        "flow_health": flow_health,
        "advance_health": advance_health,
        "efficiency_health": efficiency_health,
    }
    weights = _weights(profile)
    hold_score = 100.0 * sum(weights[name] * components[name] for name in weights)
    hold_score = max(0.0, min(100.0, hold_score))

    if hold_score < config.exit_threshold:
        stance = HoldStance.EXIT
    elif hold_score < config.tighten_threshold:
        stance = HoldStance.TIGHTEN
    else:
        stance = HoldStance.HOLD

    diagnostics = {
        "atr": atr_value,
        "trend_atr": trend_atr,
        "impulse_z": impulse_z,
        "flow_balance": flow_balance,
        "bars_since_peak_close": float(bars_since_peak),
        "signed_efficiency": signed_efficiency,
    }
    return HoldAssessment(
        measurable=True,
        profile=profile,
        hold_score=round(hold_score, 6),
        stance=stance,
        components={name: _clip01(value) for name, value in components.items()},
        diagnostics=diagnostics,
        reasons=(),
    )
