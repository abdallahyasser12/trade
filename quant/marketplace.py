"""Strategy marketplace: rolling statistics from attribution rows.

Lifecycle callers may scope health to ``since`` a state transition. This is
critical for rung-specific evidence: PAPER trades must not count as CANARY trades,
and old favorable history must not hide degradation in the current deployment.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from quant.contracts import StrategyHealth, StrategyState
from quant.store import QuantStore

MIN_SAMPLE = 10
MIN_REGIME_SAMPLE = 5


def _finite(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def compute_strategy_health(
    store: QuantStore,
    strategy_id: str,
    state: StrategyState,
    *,
    include_shadow: bool = False,
    limit: int = 1000,
    since: Optional[datetime] = None,
) -> StrategyHealth:
    """Compute strategy statistics, optionally only from rows recorded since ``since``."""
    if since is not None:
        if since.tzinfo is None:
            raise ValueError("Strategy-health 'since' must be timezone-aware.")
        since = since.astimezone(timezone.utc)

    rows = store.get_attributions(
        strategy_id=strategy_id, include_shadow=include_shadow, limit=limit
    )
    if since is not None:
        filtered = []
        for row in rows:
            recorded = _parse_utc(row.get("recorded_at"))
            if recorded is not None and recorded >= since:
                filtered.append(row)
        rows = filtered

    # Store returns newest first; drawdown must be computed chronologically.
    rows = list(reversed(rows))
    r_values: list[float] = []
    net_pnl = 0.0
    regime_rows: dict[str, list[float]] = {}
    for row in rows:
        net = _finite(row.get("net_pnl"))
        if net is not None:
            net_pnl += net
        r = _finite(row.get("r_multiple"))
        if r is None:
            continue
        r_values.append(r)
        regime = row.get("regime_at_entry")
        if regime:
            regime_rows.setdefault(str(regime), []).append(r)

    trades = len(r_values)
    wins = sum(1 for r in r_values if r > 0)
    losses = trades - wins

    if trades < MIN_SAMPLE:
        return StrategyHealth(
            strategy_id=strategy_id,
            state=state,
            trades=trades,
            wins=wins,
            losses=losses,
            hit_rate=None,
            expectancy_r=None,
            avg_r=None,
            profit_factor=None,
            sharpe=None,
            sortino=None,
            max_drawdown_pct=None,
            tail_loss_r=None,
            net_pnl=net_pnl,
        )

    mean_r = sum(r_values) / trades
    variance = sum((r - mean_r) ** 2 for r in r_values) / (trades - 1)
    std = math.sqrt(variance)
    downside = [min(0.0, r) for r in r_values]
    downside_var = sum(d * d for d in downside) / (trades - 1)
    downside_std = math.sqrt(downside_var)

    positive = sum(r for r in r_values if r > 0)
    negative = abs(sum(r for r in r_values if r <= 0))
    profit_factor = positive / negative if negative > 0 else None

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for r in r_values:
        cumulative += r
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    sorted_r = sorted(r_values)
    tail_index = max(0, int(math.floor(0.05 * trades)) - 1)
    tail_loss = sorted_r[tail_index] if trades >= 20 else sorted_r[0]

    regime_performance: dict[str, dict[str, float]] = {}
    for regime, values in regime_rows.items():
        if len(values) < MIN_REGIME_SAMPLE:
            continue
        regime_performance[regime] = {
            "trades": float(len(values)),
            "avg_r": sum(values) / len(values),
            "hit_rate": sum(1 for value in values if value > 0) / len(values),
        }

    return StrategyHealth(
        strategy_id=strategy_id,
        state=state,
        trades=trades,
        wins=wins,
        losses=losses,
        hit_rate=wins / trades,
        expectancy_r=mean_r,
        avg_r=mean_r,
        profit_factor=profit_factor,
        sharpe=mean_r / std if std > 0 else None,
        sortino=mean_r / downside_std if downside_std > 0 else None,
        max_drawdown_pct=max_drawdown,
        tail_loss_r=tail_loss,
        net_pnl=net_pnl,
        regime_performance=regime_performance,
    )


def regime_compatibility(
    health: StrategyHealth, regime_label: str
) -> Optional[float]:
    bucket = health.regime_performance.get(regime_label)
    if not bucket:
        return None
    avg_r = _finite(bucket.get("avg_r"))
    if avg_r is None:
        return None
    return 1.0 / (1.0 + math.exp(-avg_r * 4.0))
