"""One friction model, shared by simulation and the live entry path.

The single number that matters for a stop-based system is not the cost in
basis points, it is the cost expressed in units of the risk the trade is
taking.  A trade that risks ``entry - stop`` and pays a round trip of the
same size must be right roughly twice as often just to break even, so a
strategy can be comfortably profitable gross and reliably negative net
without anything in the signal or sizing layer noticing.

Every consumer computes both numbers here so simulation and live trading
cannot drift apart.
"""

from __future__ import annotations

import math

BPS = 10_000.0


def round_trip_cost_bps(
    *,
    slippage_bps: float,
    spread_bps: float,
    taker_fee_bps: float,
) -> float:
    """Total friction, in bps of notional, for one full entry-and-exit cycle.

    Both legs cross half the spread, pay the taker fee, and suffer slippage.
    """

    values = (slippage_bps, spread_bps, taker_fee_bps)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Cost assumptions must be finite and non-negative.")
    return 2.0 * (slippage_bps + spread_bps / 2.0 + taker_fee_bps)


def stop_distance_bps(entry_price: float, stop_price: float) -> float:
    """The 1R risk unit, in bps of the entry price."""

    if not all(math.isfinite(value) for value in (entry_price, stop_price)):
        return math.inf
    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        return math.inf
    return (entry_price - stop_price) / entry_price * BPS


def cost_to_risk(
    *,
    entry_price: float,
    stop_price: float,
    round_trip_bps: float,
) -> float:
    """Round-trip friction as a fraction of the trade's own risk unit.

    ``0.25`` means a quarter of 1R is spent on friction.  ``1.0`` means the
    trade pays its entire risk budget in costs before it can make anything.
    Returns infinity when the stop distance is unusable, so callers that
    gate on a maximum fail closed.
    """

    distance = stop_distance_bps(entry_price, stop_price)
    if not math.isfinite(distance) or distance <= 0:
        return math.inf
    if not math.isfinite(round_trip_bps) or round_trip_bps < 0:
        return math.inf
    return round_trip_bps / distance


def break_even_win_rate(cost_to_risk_ratio: float, reward_to_risk: float = 2.0) -> float:
    """Win rate needed to break even net of costs at a given reward:risk.

    Solves ``w * (reward - c) = (1 - w) * (1 + c)`` in units of R, where
    ``c`` is the cost in R.  Returns NaN when no win rate can break even
    because friction has consumed the entire reward.
    """

    if not all(
        math.isfinite(value) and value >= 0
        for value in (cost_to_risk_ratio, reward_to_risk)
    ):
        return math.nan
    net_win = reward_to_risk - cost_to_risk_ratio
    net_loss = 1.0 + cost_to_risk_ratio
    if net_win <= 0:
        return math.nan
    return net_loss / (net_win + net_loss)
