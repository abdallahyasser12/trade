"""Portfolio heat book and robust allocator (spec §§28-30).

Deterministic caps and penalties only — no covariance inversion, nothing
fitted.  This layer sits ABOVE the per-trade risk managers and may only
REDUCE what a strategy requests; the hard per-trade ceilings in
``risk_manager``/``crypto_risk`` remain authoritative and untouched.

The central scenario it exists for: many individually acceptable trades
must not sum into one giant correlated bet.  Every cap failure is a named
reason so the journal can show exactly which limit turned a trade away.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from quant.contracts import Market

# Static sector map for liquid US symbols.  Unknown symbols share one
# conservative "unknown" bucket so an unmapped book still hits a sector
# cap rather than bypassing it; crypto is its own sector by definition.
SECTOR_MAP: dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "AVGO": "technology", "AMD": "technology", "CRM": "technology",
    "ORCL": "technology", "ADBE": "technology", "INTC": "technology",
    "GOOGL": "communication", "GOOG": "communication", "META": "communication",
    "NFLX": "communication", "DIS": "communication",
    "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
    "HD": "consumer_discretionary", "MCD": "consumer_discretionary",
    "NKE": "consumer_discretionary",
    "WMT": "consumer_staples", "PG": "consumer_staples", "KO": "consumer_staples",
    "PEP": "consumer_staples", "COST": "consumer_staples",
    "JPM": "financials", "BAC": "financials", "GS": "financials",
    "MS": "financials", "WFC": "financials", "V": "financials", "MA": "financials",
    "BRK.B": "financials",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "XLE": "energy",
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare",
    "PFE": "healthcare", "ABBV": "healthcare", "XLV": "healthcare",
    "BA": "industrials", "CAT": "industrials", "GE": "industrials",
    "UPS": "industrials", "XLI": "industrials",
    "SPY": "broad_index", "QQQ": "broad_index", "IWM": "broad_index",
    "DIA": "broad_index", "VTI": "broad_index",
    "XLF": "financials", "XLK": "technology", "XLU": "utilities",
    "XLP": "consumer_staples", "XLY": "consumer_discretionary",
    "GLD": "commodities", "SLV": "commodities", "USO": "commodities",
    "TLT": "rates", "IEF": "rates",
}


def sector_for(symbol: str, market: Market) -> str:
    if market is Market.CRYPTO:
        return "crypto"
    return SECTOR_MAP.get(symbol, "unknown")


@dataclass(frozen=True)
class PortfolioLimitsConfig:
    """Book-level caps, all as percent of equity unless noted."""

    max_portfolio_heat_pct: float = 2.0
    max_gross_exposure_pct: float = 60.0
    max_market_exposure_pct: float = 40.0
    max_sector_exposure_pct: float = 25.0
    max_strategy_heat_pct: float = 1.0
    max_new_risk_per_cycle_pct: float = 1.0
    max_pairwise_correlation: float = 0.85
    correlation_lookback: int = 60
    min_correlation_observations: int = 30
    reject_unknown_correlation: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        percents = {
            "portfolio heat cap": self.max_portfolio_heat_pct,
            "gross exposure cap": self.max_gross_exposure_pct,
            "market exposure cap": self.max_market_exposure_pct,
            "sector exposure cap": self.max_sector_exposure_pct,
            "strategy heat cap": self.max_strategy_heat_pct,
            "per-cycle risk budget": self.max_new_risk_per_cycle_pct,
        }
        for label, value in percents.items():
            if not math.isfinite(value) or not (0 < value <= 100):
                errors.append(f"{label} must be in (0, 100]%.")
        if not (0.0 <= self.max_pairwise_correlation <= 1.0):
            errors.append("Correlation cap must be between 0 and 1.")
        if self.correlation_lookback < 20:
            errors.append("Correlation lookback must contain at least 20 bars.")
        if self.min_correlation_observations < 10:
            errors.append("Correlation gate needs at least 10 overlapping observations.")
        if self.max_strategy_heat_pct > self.max_portfolio_heat_pct:
            errors.append("Per-strategy heat cannot exceed the portfolio heat cap.")
        return errors


@dataclass
class BookPosition:
    symbol: str
    market: Market
    sector: str
    strategy_id: str
    notional: float
    risk_at_stop: float

    @classmethod
    def from_stop(
        cls,
        symbol: str,
        market: Market,
        strategy_id: str,
        *,
        qty: float,
        price: float,
        stop_price: float,
    ) -> "BookPosition":
        return cls(
            symbol=symbol,
            market=market,
            sector=sector_for(symbol, market),
            strategy_id=strategy_id,
            notional=qty * price,
            risk_at_stop=qty * max(0.0, price - stop_price),
        )


@dataclass
class PortfolioBook:
    """Current open positions plus account equity, all marks caller-supplied."""

    equity: float
    positions: list[BookPosition] = field(default_factory=list)

    def _pct(self, value: float) -> float:
        return value / self.equity * 100.0 if self.equity > 0 else math.inf

    @property
    def heat_pct(self) -> float:
        return self._pct(sum(p.risk_at_stop for p in self.positions))

    @property
    def gross_exposure_pct(self) -> float:
        return self._pct(sum(abs(p.notional) for p in self.positions))

    def market_exposure_pct(self, market: Market) -> float:
        return self._pct(
            sum(abs(p.notional) for p in self.positions if p.market is market)
        )

    def sector_exposure_pct(self, sector: str) -> float:
        return self._pct(
            sum(abs(p.notional) for p in self.positions if p.sector == sector)
        )

    def strategy_heat_pct(self, strategy_id: str) -> float:
        return self._pct(
            sum(p.risk_at_stop for p in self.positions if p.strategy_id == strategy_id)
        )

    def breakdown(self) -> dict[str, dict[str, float]]:
        markets: dict[str, float] = {}
        sectors: dict[str, float] = {}
        strategies: dict[str, float] = {}
        for position in self.positions:
            markets[position.market.value] = markets.get(position.market.value, 0.0) + abs(
                position.notional
            )
            sectors[position.sector] = sectors.get(position.sector, 0.0) + abs(
                position.notional
            )
            strategies[position.strategy_id] = strategies.get(
                position.strategy_id, 0.0
            ) + position.risk_at_stop
        return {
            "market_pct": {k: self._pct(v) for k, v in markets.items()},
            "sector_pct": {k: self._pct(v) for k, v in sectors.items()},
            "strategy_heat_pct": {k: self._pct(v) for k, v in strategies.items()},
        }


@dataclass
class HeatCheckResult:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    headroom_risk_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pairwise_correlation(
    a: pd.Series, b: pd.Series, lookback: int
) -> tuple[Optional[float], int]:
    joined = pd.concat([a, b], axis=1, join="inner").dropna().tail(lookback)
    if len(joined) < 2:
        return None, len(joined)
    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    if corr is None or not math.isfinite(corr):
        return None, len(joined)
    return float(corr), len(joined)


class PortfolioRiskGovernor:
    """Book-level gate + deterministic greedy allocator."""

    def __init__(self, config: Optional[PortfolioLimitsConfig] = None):
        self.config = config or PortfolioLimitsConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid portfolio limits: " + "; ".join(errors))

    def check(
        self,
        book: PortfolioBook,
        *,
        candidate_symbol: str,
        market: Market,
        sector: str,
        strategy_id: str,
        add_notional: float,
        add_risk: float,
        candidate_returns: Optional[pd.Series] = None,
        held_returns: Optional[Mapping[str, pd.Series]] = None,
    ) -> HeatCheckResult:
        cfg = self.config
        reasons: list[str] = []
        if book.equity <= 0 or not math.isfinite(book.equity):
            return HeatCheckResult(approved=False, reasons=["non_positive_equity"])
        if not math.isfinite(add_notional) or not math.isfinite(add_risk) or add_risk < 0:
            return HeatCheckResult(approved=False, reasons=["invalid_candidate_size"])

        add_risk_pct = add_risk / book.equity * 100.0
        add_notional_pct = abs(add_notional) / book.equity * 100.0

        if book.heat_pct + add_risk_pct > cfg.max_portfolio_heat_pct:
            reasons.append("portfolio_heat_cap")
        if book.gross_exposure_pct + add_notional_pct > cfg.max_gross_exposure_pct:
            reasons.append("gross_exposure_cap")
        if book.market_exposure_pct(market) + add_notional_pct > cfg.max_market_exposure_pct:
            reasons.append("market_exposure_cap")
        if book.sector_exposure_pct(sector) + add_notional_pct > cfg.max_sector_exposure_pct:
            reasons.append("sector_exposure_cap")
        if book.strategy_heat_pct(strategy_id) + add_risk_pct > cfg.max_strategy_heat_pct:
            reasons.append("strategy_heat_cap")

        if candidate_returns is not None and held_returns:
            for held_symbol, held_series in held_returns.items():
                if held_symbol == candidate_symbol:
                    continue
                corr, observations = _pairwise_correlation(
                    candidate_returns, held_series, cfg.correlation_lookback
                )
                if observations < cfg.min_correlation_observations or corr is None:
                    # Too little overlap to measure: unknown, not a breach —
                    # unless the operator opted into rejecting unknowns.
                    if cfg.reject_unknown_correlation:
                        reasons.append("correlation_unknown")
                    continue
                if corr > cfg.max_pairwise_correlation:
                    reasons.append("correlated_exposure")
                    break

        headroom = max(
            0.0,
            min(
                cfg.max_portfolio_heat_pct - book.heat_pct,
                cfg.max_strategy_heat_pct - book.strategy_heat_pct(strategy_id),
            ),
        )
        return HeatCheckResult(
            approved=not reasons,
            reasons=sorted(set(reasons)),
            headroom_risk_pct=headroom,
        )

    def allocate(
        self,
        book: PortfolioBook,
        ranked: Sequence[dict[str, Any]],
    ) -> dict[str, float]:
        """Grant risk down the ranked queue under all caps.

        Each entry: {key, requested_risk_pct, notional, symbol, market,
        sector, strategy_id, returns (optional pd.Series)}.  Returns
        key -> granted risk pct (0 when rejected).  Deterministic: walks
        the given order, never reorders, never exceeds the per-cycle
        budget.
        """
        grants: dict[str, float] = {}
        cycle_budget = self.config.max_new_risk_per_cycle_pct
        working_positions = list(book.positions)
        held_returns: dict[str, pd.Series] = {}
        for entry in ranked:
            key = entry["key"]
            requested = float(entry["requested_risk_pct"])
            if cycle_budget <= 0 or requested <= 0:
                grants[key] = 0.0
                continue
            working_book = PortfolioBook(equity=book.equity, positions=working_positions)
            market = Market(entry["market"])
            check = self.check(
                working_book,
                candidate_symbol=entry["symbol"],
                market=market,
                sector=entry["sector"],
                strategy_id=entry["strategy_id"],
                add_notional=float(entry["notional"]),
                add_risk=requested / 100.0 * book.equity,
                candidate_returns=entry.get("returns"),
                held_returns=held_returns,
            )
            granted = 0.0
            if check.approved:
                granted = min(requested, check.headroom_risk_pct, cycle_budget)
            if granted > 0:
                cycle_budget -= granted
                scale = granted / requested
                working_positions = working_positions + [
                    BookPosition(
                        symbol=entry["symbol"],
                        market=market,
                        sector=entry["sector"],
                        strategy_id=entry["strategy_id"],
                        notional=float(entry["notional"]) * scale,
                        risk_at_stop=granted / 100.0 * book.equity,
                    )
                ]
                if entry.get("returns") is not None:
                    held_returns[entry["symbol"]] = entry["returns"]
            grants[key] = granted
        return grants
