"""Risk management shared by backtest and live trading.

Combines two independent sizing limits (risk budget from the ATR stop
distance, and maximum account exposure) plus buying power, then applies
minimum-size sanity checks.  Also owns the session loss circuit breaker.

Whole shares only: quantities are floored, never rounded up, so risk is
never silently increased to force a trade.
"""

from __future__ import annotations

import math

from config import RiskConfig
from models import PositionSizeResult


class RiskManager:
    """Deterministic position sizing and loss limits."""

    def __init__(self, config: RiskConfig):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid risk configuration: " + "; ".join(errors))
        self.config = config

    def calculate_stop_price(
        self, entry_price: float, atr_value: float, atr_multiplier: float
    ) -> float:
        """Protective stop for a long entry: fill price minus ATR distance.

        Raises ValueError instead of returning a non-positive stop — a
        stop <= 0 means the ATR distance exceeds the entry price and the
        trade must not be taken.
        """
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")
        if not math.isfinite(atr_value) or atr_value <= 0:
            raise ValueError(f"ATR must be positive, got {atr_value}")
        if not math.isfinite(atr_multiplier) or atr_multiplier <= 0:
            raise ValueError(f"ATR multiplier must be positive, got {atr_multiplier}")
        stop = entry_price - atr_value * atr_multiplier
        if stop <= 0:
            raise ValueError(
                f"Computed stop {stop:.4f} is not positive "
                f"(entry {entry_price}, ATR {atr_value}, multiplier {atr_multiplier})"
            )
        return stop

    def calculate_position_size(
        self,
        equity: float,
        buying_power: float,
        entry_price: float,
        stop_price: float,
    ) -> PositionSizeResult:
        """Size a long entry from risk budget, exposure cap and buying power.

        shares_by_risk      = floor(equity * risk_pct / (entry - stop))
        shares_by_exposure  = floor(equity * max_position_pct / entry)
        shares_by_bp        = floor(buying_power / entry)
        qty = min of the three; rejected if qty <= 0 or below the
        configured minimum share count / position value.
        """
        cfg = self.config

        def rejected(reason: str, **kw) -> PositionSizeResult:
            return PositionSizeResult(
                qty=0,
                shares_by_risk=kw.get("shares_by_risk", 0),
                shares_by_exposure=kw.get("shares_by_exposure", 0),
                shares_by_buying_power=kw.get("shares_by_buying_power", 0),
                position_value=0.0,
                risk_amount=0.0,
                rejection_reason=reason,
            )

        if not math.isfinite(equity) or equity <= 0:
            return rejected("non_positive_equity")
        if not math.isfinite(entry_price) or entry_price <= 0:
            return rejected("non_positive_entry_price")
        if not math.isfinite(stop_price) or stop_price <= 0 or stop_price >= entry_price:
            return rejected("invalid_stop_price")
        if not math.isfinite(buying_power) or buying_power <= 0:
            return rejected("no_buying_power")

        risk_per_share = entry_price - stop_price
        risk_budget = equity * (cfg.risk_per_trade_pct / 100.0)
        shares_by_risk = math.floor(risk_budget / risk_per_share)

        max_position_value = equity * (cfg.max_position_pct / 100.0)
        shares_by_exposure = math.floor(max_position_value / entry_price)

        shares_by_buying_power = math.floor(buying_power / entry_price)

        qty = min(shares_by_risk, shares_by_exposure, shares_by_buying_power)
        caps = dict(
            shares_by_risk=shares_by_risk,
            shares_by_exposure=shares_by_exposure,
            shares_by_buying_power=shares_by_buying_power,
        )
        if qty <= 0:
            return rejected("zero_quantity", **caps)

        position_value = qty * entry_price
        if qty < cfg.min_shares or position_value < cfg.min_position_value:
            return rejected("position_too_small", **caps)

        return PositionSizeResult(
            qty=int(qty),
            position_value=position_value,
            risk_amount=qty * risk_per_share,
            rejection_reason=None,
            **caps,
        )

    def session_loss_pct(self, starting_equity: float, current_equity: float) -> float:
        """Loss since session start as a positive percentage (gain -> <= 0)."""
        if starting_equity <= 0:
            return 0.0
        return (starting_equity - current_equity) / starting_equity * 100.0

    def check_daily_loss_limit(self, starting_equity: float, current_equity: float) -> bool:
        """True when the session loss limit is reached or exceeded.

        Baseline: the account equity captured when the live session (or
        backtest day) began.  Once True, the engine halts new entries for
        the remainder of the session and persists the halt.
        """
        return self.session_loss_pct(starting_equity, current_equity) >= self.config.max_daily_loss_pct
