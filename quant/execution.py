"""Pre-trade execution-quality engine (spec §§31-33).

Joins three cost sources into one gate every opportunity must clear
before it may be sized:

1. the live quote's spread, when fresh;
2. the ROLLING MEASURED slippage the platform already persists in the
   executions ledger (``StateStore.execution_quality``);
3. configured conservative assumptions, as the fallback.

The measured figure is folded in as ``max(measured, assumed)`` — a
deliberately one-sided update.  Measured-worse-than-assumed must tighten
the gate immediately; measured-better-than-assumed must NOT loosen it,
because every promotion decision was made against the assumed number and
loosening live behavior ahead of a research rerun would quietly invert
the champion/challenger evidence ordering.  Research reruns are how a
better measurement becomes tradeable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

from cost_model import round_trip_cost_bps
from quant.contracts import ExecutionEstimate, Market


@dataclass(frozen=True)
class ExecutionQualityConfig:
    assumed_slippage_bps: float = 10.0
    equity_assumed_spread_bps: float = 5.0
    crypto_assumed_spread_bps: float = 10.0
    equity_taker_fee_bps: float = 0.0
    crypto_taker_fee_bps: float = 25.0
    max_spread_bps: float = 40.0
    max_participation_pct: float = 2.0
    min_measured_samples: int = 20
    uncertainty_margin_bps: float = 5.0
    quote_max_age_seconds: float = 120.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        for label, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                errors.append(f"{label} must be finite and non-negative.")
        if self.max_spread_bps <= 0 or self.max_spread_bps > 500:
            errors.append("Maximum spread must be in (0, 500] bps.")
        if self.max_participation_pct <= 0 or self.max_participation_pct > 100:
            errors.append("Maximum participation must be in (0, 100]%.")
        if self.min_measured_samples < 1:
            errors.append("Measured-sample threshold must be at least 1.")
        if self.quote_max_age_seconds <= 0:
            errors.append("Quote maximum age must be positive.")
        return errors

    def assumed_spread_for(self, market: Market) -> float:
        return (
            self.crypto_assumed_spread_bps
            if market is Market.CRYPTO
            else self.equity_assumed_spread_bps
        )

    def fee_for(self, market: Market) -> float:
        return (
            self.crypto_taker_fee_bps
            if market is Market.CRYPTO
            else self.equity_taker_fee_bps
        )


class ExecutionQualityEngine:
    """Deterministic pre-trade cost estimation and viability gating."""

    def __init__(self, config: Optional[ExecutionQualityConfig] = None, state_store=None):
        self.config = config or ExecutionQualityConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid execution-quality configuration: " + "; ".join(errors))
        self.state_store = state_store

    def slippage_estimate(self) -> tuple[float, str, int]:
        """(slippage_bps, source, samples) — measured only tightens, never loosens."""
        assumed = self.config.assumed_slippage_bps
        if self.state_store is None:
            return assumed, "assumed", 0
        try:
            quality = self.state_store.execution_quality()
        except Exception:
            return assumed, "assumed", 0
        samples = int(quality.get("samples") or 0)
        measured = quality.get("avg_slippage_bps")
        if samples < self.config.min_measured_samples or measured is None:
            return assumed, "assumed", samples
        if not math.isfinite(float(measured)):
            return assumed, "assumed", samples
        return max(float(measured), assumed), "measured", samples

    def estimate(
        self,
        symbol: str,
        market: Market,
        *,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        quote_age_seconds: Optional[float] = None,
        notional: Optional[float] = None,
        avg_bar_dollar_volume: Optional[float] = None,
    ) -> ExecutionEstimate:
        cfg = self.config
        slippage, source, samples = self.slippage_estimate()

        spread_bps: float
        quote_usable = (
            bid is not None
            and ask is not None
            and math.isfinite(bid)
            and math.isfinite(ask)
            and bid > 0
            and ask > bid
            and quote_age_seconds is not None
            and math.isfinite(quote_age_seconds)
            and quote_age_seconds <= cfg.quote_max_age_seconds
        )
        if quote_usable:
            midpoint = (bid + ask) / 2.0
            spread_bps = (ask - bid) / midpoint * 10_000.0
        else:
            spread_bps = cfg.assumed_spread_for(market)
            source = f"{source}+assumed_spread"

        participation: Optional[float] = None
        if (
            notional is not None
            and avg_bar_dollar_volume is not None
            and math.isfinite(notional)
            and math.isfinite(avg_bar_dollar_volume)
            and avg_bar_dollar_volume > 0
        ):
            participation = notional / avg_bar_dollar_volume * 100.0

        viable = True
        reason: Optional[str] = None
        if not math.isfinite(spread_bps) or spread_bps < 0:
            viable, reason = False, "spread_not_computable"
            spread_bps = cfg.assumed_spread_for(market)
        elif spread_bps > cfg.max_spread_bps:
            viable, reason = False, "spread_above_maximum"
        elif participation is not None and participation > cfg.max_participation_pct:
            viable, reason = False, "participation_above_maximum"
        elif notional is not None and (not math.isfinite(notional) or notional <= 0):
            viable, reason = False, "invalid_notional"

        fee = cfg.fee_for(market)
        return ExecutionEstimate(
            symbol=symbol,
            spread_bps=spread_bps,
            slippage_bps=slippage,
            fee_bps=fee,
            round_trip_bps=round_trip_cost_bps(
                slippage_bps=slippage, spread_bps=spread_bps, taker_fee_bps=fee
            ),
            participation_pct=participation,
            cost_source=source,
            sample_count=samples,
            viable=viable,
            reason=reason,
        )

    def net_alpha_after_costs(
        self, expected_gross_alpha_bps: float, estimate: ExecutionEstimate
    ) -> tuple[bool, float]:
        """(tradeable, net_alpha_bps) with the uncertainty margin applied.

        Fail closed: a non-finite gross alpha or a non-viable estimate can
        never be tradeable.
        """
        if not math.isfinite(expected_gross_alpha_bps):
            return False, -math.inf
        net = (
            expected_gross_alpha_bps
            - estimate.round_trip_bps
            - self.config.uncertainty_margin_bps
        )
        return (estimate.viable and net > 0), net
