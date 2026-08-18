"""Negative-only forward supervision for adaptive intraday alpha sleeves.

Recent winners are never levered up.  The supervisor can only leave a sleeve at its
baseline privilege, reduce its confidence, or suspend new entries after enough
current-lifecycle real trades.  This avoids the positive-feedback instability of
online PnL-chasing weights while still letting production respond when one component
of the ensemble stops working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

ADAPTIVE_SLEEVES = ("momentum", "breakout", "pullback", "reversion")


@dataclass(frozen=True)
class SleeveSupervisionConfig:
    min_trades: int = 40
    soft_profit_factor: float = 0.95
    hard_profit_factor: float = 0.80
    soft_confidence_multiplier: float = 0.70
    one_sided_z: float = 1.645
    max_records: int = 2_000

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.min_trades < 20:
            errors.append("Sleeve supervision requires at least 20 trades.")
        if not 0 < self.hard_profit_factor < self.soft_profit_factor <= 1.0:
            errors.append("Profit-factor floors must satisfy 0 < hard < soft <= 1.")
        if not 0 < self.soft_confidence_multiplier < 1.0:
            errors.append("Soft confidence multiplier must be in (0,1).")
        if not math.isfinite(self.one_sided_z) or self.one_sided_z <= 0:
            errors.append("One-sided z threshold must be finite and positive.")
        if self.max_records < self.min_trades:
            errors.append("max_records must be >= min_trades.")
        return errors


@dataclass(frozen=True)
class SleeveHealth:
    sleeve: str
    trades: int
    expectancy_r: Optional[float]
    profit_factor: Optional[float]
    upper_expectancy_bound_r: Optional[float]
    confidence_multiplier: float
    verdict: str

    def to_dict(self) -> dict:
        return {
            "sleeve": self.sleeve,
            "trades": self.trades,
            "expectancy_r": self.expectancy_r,
            "profit_factor": self.profit_factor,
            "upper_expectancy_bound_r": self.upper_expectancy_bound_r,
            "confidence_multiplier": self.confidence_multiplier,
            "verdict": self.verdict,
        }


def sleeve_from_entry_reason(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    text = str(reason)
    for sleeve in ADAPTIVE_SLEEVES:
        if text == f"adaptive_{sleeve}_entry":
            return sleeve
    return None


def _parse_time(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _profit_factor(values: list[float]) -> Optional[float]:
    wins = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    if losses > 0:
        return wins / losses
    if wins > 0:
        return math.inf
    return None


def _evaluate_values(
    sleeve: str,
    values: list[float],
    config: SleeveSupervisionConfig,
) -> SleeveHealth:
    count = len(values)
    if count < config.min_trades:
        return SleeveHealth(sleeve, count, None, None, None, 1.0, "insufficient_sample")

    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        std = math.sqrt(max(0.0, variance))
        se = std / math.sqrt(count)
    else:
        se = math.inf
    upper = mean + config.one_sided_z * se if math.isfinite(se) else None
    pf = _profit_factor(values)

    hard_negative = (
        upper is not None and upper <= 0.0
    ) or (pf is not None and math.isfinite(pf) and pf < config.hard_profit_factor)
    if hard_negative:
        return SleeveHealth(sleeve, count, mean, pf, upper, 0.0, "suspended_negative_edge")

    soft_negative = mean <= 0.0 or (
        pf is not None and math.isfinite(pf) and pf < config.soft_profit_factor
    )
    if soft_negative:
        return SleeveHealth(
            sleeve,
            count,
            mean,
            pf,
            upper,
            config.soft_confidence_multiplier,
            "reduced_negative_edge",
        )

    return SleeveHealth(sleeve, count, mean, pf, upper, 1.0, "baseline")


def compute_sleeve_health(
    store,
    strategy_id: str,
    sleeve: str,
    *,
    since: Optional[datetime] = None,
    config: SleeveSupervisionConfig = SleeveSupervisionConfig(),
) -> SleeveHealth:
    errors = config.validate()
    if errors:
        raise ValueError("Invalid sleeve supervision config: " + "; ".join(errors))
    if sleeve not in ADAPTIVE_SLEEVES:
        raise ValueError(f"Unknown adaptive sleeve {sleeve!r}.")
    if since is not None:
        if since.tzinfo is None:
            raise ValueError("Sleeve supervision 'since' must be timezone-aware.")
        since = since.astimezone(timezone.utc)

    rows = store.get_attributions(
        strategy_id=strategy_id,
        include_shadow=False,
        limit=config.max_records,
    )
    values: list[float] = []
    for row in rows:
        if sleeve_from_entry_reason(row.get("entry_reason")) != sleeve:
            continue
        if since is not None:
            recorded = _parse_time(row.get("recorded_at"))
            if recorded is None or recorded < since:
                continue
        try:
            r_value = float(row.get("r_multiple"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(r_value):
            values.append(r_value)
    return _evaluate_values(sleeve, values, config)
