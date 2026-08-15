"""Shared fractional sizing, portfolio concentration, and loss-latch controls."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Mapping, Optional

import pandas as pd

from config import AccountMode
from crypto_config import CryptoRiskConfig
from crypto_models import CryptoPositionSizeResult
from models import AssetInfo, BrokerPosition


def floor_to_increment(quantity: float, increment: float) -> float:
    """Floor a positive quantity without floating-point risk expansion."""

    if not math.isfinite(quantity) or quantity <= 0:
        return 0.0
    if not math.isfinite(increment) or increment <= 0:
        raise ValueError("Trade increment must be finite and positive.")
    try:
        raw = Decimal(str(quantity))
        step = Decimal(str(increment))
        units = (raw / step).to_integral_value(rounding=ROUND_FLOOR)
        return float(units * step)
    except InvalidOperation as exc:
        raise ValueError("Could not floor crypto quantity.") from exc


def portfolio_market_value(
    positions: list[BrokerPosition] | tuple[BrokerPosition, ...],
) -> float:
    total = 0.0
    for position in positions:
        value = position.market_value
        if value is None and position.current_price is not None:
            value = position.qty * position.current_price
        if value is not None and math.isfinite(value) and value > 0:
            total += value
    return total


class CryptoRiskManager:
    """Portfolio-aware risk manager shared by live and backtest."""

    def __init__(self, config: CryptoRiskConfig):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid crypto risk configuration: " + "; ".join(errors))
        self.config = config
        self._session_halted = False
        self._halt_reason: Optional[str] = None

    @property
    def session_halted(self) -> bool:
        return self._session_halted

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    def check_session_loss(self, starting_equity: float, current_equity: float) -> bool:
        """Latch permanently once loss is at or beyond the configured threshold."""

        if self._session_halted:
            return True
        if not all(math.isfinite(value) and value > 0 for value in (starting_equity, current_equity)):
            self._session_halted = True
            self._halt_reason = "Account equity is unavailable or invalid; entries halted."
            return True
        threshold = starting_equity * (1.0 - self.config.max_session_loss_pct / 100.0)
        if current_equity <= threshold:
            self._session_halted = True
            self._halt_reason = (
                f"Session equity ${current_equity:,.2f} reached the "
                f"{self.config.max_session_loss_pct:.2f}% loss limit."
            )
        return self._session_halted

    def risk_scale(self, peak_equity: float, equity: float, atr_percentile: float) -> float:
        if not all(math.isfinite(value) and value > 0 for value in (peak_equity, equity)):
            return self.config.min_risk_scale
        if not math.isfinite(atr_percentile) or not 0 <= atr_percentile <= 100:
            return self.config.min_risk_scale
        drawdown = max(0.0, (peak_equity - equity) / peak_equity * 100.0)
        start = self.config.drawdown_risk_reduction_start_pct
        halt = self.config.max_session_loss_pct
        if drawdown <= start:
            drawdown_scale = 1.0
        elif drawdown >= halt:
            drawdown_scale = self.config.min_risk_scale
        else:
            fraction = (drawdown - start) / (halt - start)
            drawdown_scale = 1.0 - fraction * (1.0 - self.config.min_risk_scale)
        volatility_scale = min(
            1.0,
            self.config.volatility_target_percentile / max(atr_percentile, 1.0),
        )
        return max(self.config.min_risk_scale, min(drawdown_scale, volatility_scale, 1.0))

    def size_position(
        self,
        *,
        equity: float,
        cash: float,
        buying_power: float,
        entry_price: float,
        stop_price: float,
        asset: AssetInfo,
        current_crypto_exposure: float,
        current_position_count: int,
        peak_equity: float,
        atr_percentile: float,
        current_open_risk: float = 0.0,
        account_mode: AccountMode | str = AccountMode.PAPER,
    ) -> CryptoPositionSizeResult:
        mode = AccountMode(account_mode)
        zero = self._rejected("Invalid sizing inputs.")
        numeric = (
            equity,
            cash,
            buying_power,
            entry_price,
            stop_price,
            current_crypto_exposure,
            peak_equity,
            atr_percentile,
            current_open_risk,
        )
        if not all(math.isfinite(value) for value in numeric):
            return zero
        if equity <= 0 or cash <= 0 or buying_power <= 0 or entry_price <= 0:
            return self._rejected("Equity, cash, buying power, and entry price must be positive.")
        if stop_price <= 0 or stop_price >= entry_price:
            return self._rejected("Protective stop must be positive and below entry price.")
        if self._session_halted:
            return self._rejected(self._halt_reason or "Session loss latch is active.")
        if current_position_count >= self.config.max_positions:
            return self._rejected("Maximum concurrent crypto positions reached.")
        if not asset.is_supported_crypto:
            return self._rejected("Asset is not supported for long-only Alpaca spot crypto.")
        increment = asset.min_trade_increment
        minimum_size = asset.min_order_size
        if not increment or increment <= 0 or not minimum_size or minimum_size <= 0:
            return self._rejected("Asset-specific crypto order increments are unavailable.")

        scale = self.risk_scale(peak_equity, equity, atr_percentile)
        risk_per_unit = entry_price - stop_price
        risk_budget = equity * self.config.risk_per_trade_pct / 100.0 * scale
        by_risk = risk_budget / risk_per_unit
        # Portfolio heat: the sum of open risk across every held position plus
        # this candidate may never exceed the configured share of equity.
        remaining_heat = (
            equity * self.config.max_portfolio_heat_pct / 100.0
            - max(0.0, current_open_risk)
        )
        if remaining_heat <= 0:
            return self._rejected(
                "Portfolio heat cap reached; open risk across held positions "
                "already consumes the configured budget."
            )
        by_heat = remaining_heat / risk_per_unit
        by_position = equity * self.config.max_position_pct / 100.0 / entry_price
        total_cap_pct = (
            self.config.max_live_total_exposure_pct
            if mode is AccountMode.LIVE
            else self.config.max_total_exposure_pct
        )
        remaining_exposure = max(
            0.0, equity * total_cap_pct / 100.0 - max(0.0, current_crypto_exposure)
        )
        by_portfolio = remaining_exposure / entry_price
        by_buying_power = min(cash, buying_power) / entry_price
        candidates = [by_risk, by_position, by_portfolio, by_buying_power, by_heat]
        if mode is AccountMode.LIVE:
            candidates.append(self.config.max_live_order_notional / entry_price)
        quantity = floor_to_increment(min(candidates), increment)
        notional = quantity * entry_price
        required_minimum = max(minimum_size, self.config.min_order_notional / entry_price)
        if quantity <= 0:
            return self._rejected("Risk and exposure caps produce zero quantity.")
        if quantity + increment / 10.0 < required_minimum:
            return self._rejected("Floored quantity is below Alpaca size or minimum notional.")
        if notional > min(cash, buying_power) + 1e-8:
            return self._rejected("Order notional exceeds available cash or buying power.")
        if quantity * risk_per_unit > risk_budget + max(1e-8, increment * risk_per_unit):
            return self._rejected("Floored quantity unexpectedly exceeds the risk budget.")
        return CryptoPositionSizeResult(
            qty=quantity,
            risk_budget=risk_budget,
            risk_per_unit=risk_per_unit,
            quantity_by_risk=by_risk,
            quantity_by_position=by_position,
            quantity_by_portfolio=by_portfolio,
            quantity_by_buying_power=by_buying_power,
            position_notional=notional,
            risk_scale=scale,
            quantity_by_heat=by_heat,
        )

    def correlation_gate(
        self,
        candidate_returns: pd.Series,
        held_returns: Mapping[str, pd.Series],
    ) -> tuple[bool, str, Optional[float]]:
        """Reject highly correlated additions; missing evidence fails closed."""

        if not held_returns:
            return True, "No existing crypto exposure.", None
        worst = -math.inf
        worst_symbol = ""
        for symbol, returns in held_returns.items():
            aligned = pd.concat(
                [candidate_returns.rename("candidate"), returns.rename("held")],
                axis=1,
                join="inner",
            ).dropna()
            aligned = aligned.iloc[-self.config.correlation_lookback :]
            if len(aligned) < min(20, self.config.correlation_lookback):
                return False, f"Insufficient aligned correlation history versus {symbol}.", None
            correlation = float(aligned["candidate"].corr(aligned["held"]))
            if not math.isfinite(correlation):
                return False, f"Correlation versus {symbol} is undefined.", None
            if correlation > worst:
                worst = correlation
                worst_symbol = symbol
        if worst > self.config.max_pairwise_correlation:
            return (
                False,
                f"Correlation {worst:.3f} versus {worst_symbol} exceeds "
                f"{self.config.max_pairwise_correlation:.3f}.",
                worst,
            )
        return True, f"Worst held-asset correlation is {worst:.3f}.", worst

    @staticmethod
    def _rejected(reason: str) -> CryptoPositionSizeResult:
        return CryptoPositionSizeResult(
            qty=0.0,
            risk_budget=0.0,
            risk_per_unit=0.0,
            quantity_by_risk=0.0,
            quantity_by_position=0.0,
            quantity_by_portfolio=0.0,
            quantity_by_buying_power=0.0,
            position_notional=0.0,
            risk_scale=0.0,
            rejection_reason=reason,
        )
