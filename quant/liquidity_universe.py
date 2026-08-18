"""Liquidity-aware universe selection and deployment locking for adaptive intraday.

Selection is based only on *current executability* (quote freshness, spread and top
of book depth), never on historical/future return. The selected symbols are recorded
in research. RESEARCH -> SHADOW promotion locks that exact universe in QuantStore;
SHADOW/PAPER/LIVE must reuse it. Replacing symbols therefore requires a new research
cycle and explicit lifecycle reset rather than silent production drift.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional

ADAPTIVE_LOCKED_UNIVERSE_KEY = "crypto_intraday_adaptive.locked_universe.v2"

# Active/tradable Alpaca BASE/USD candidates verified on 2026-08-18. This is a
# discovery pool, not an approved trading universe. Broker startup still verifies
# every selected asset because listings can change.
ADAPTIVE_USD_CANDIDATE_POOL = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "DOGE/USD",
    "AVAX/USD",
    "LINK/USD",
    "LTC/USD",
    "BCH/USD",
    "ADA/USD",
    "UNI/USD",
    "AAVE/USD",
    "SKY/USD",
    "TRUMP/USD",
    "PEPE/USD",
    "CRV/USD",
    "DOT/USD",
    "FIL/USD",
    "HYPE/USD",
    "LDO/USD",
    "ONDO/USD",
    "POL/USD",
    "RENDER/USD",
    "SUSHI/USD",
    "WIF/USD",
    "YFI/USD",
    "BONK/USD",
    "SHIB/USD",
    "ARB/USD",
    "XTZ/USD",
    "BAT/USD",
    "GRT/USD",
)


@dataclass(frozen=True)
class LiquiditySelectionConfig:
    max_symbols: int = 12
    min_symbols: int = 6
    max_quote_age_seconds: float = 120.0
    max_spread_bps: float = 60.0
    min_side_depth_usd: float = 250.0
    benchmark_symbol: str = "BTC/USD"

    def validate(self) -> list[str]:
        errors = []
        if not 1 <= self.min_symbols <= self.max_symbols <= 12:
            errors.append("Liquidity universe must satisfy 1 <= min <= max <= 12.")
        for label, value in {
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "max_spread_bps": self.max_spread_bps,
            "min_side_depth_usd": self.min_side_depth_usd,
        }.items():
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{label} must be finite and positive.")
        if self.benchmark_symbol not in ADAPTIVE_USD_CANDIDATE_POOL:
            errors.append("Benchmark must be in the discovery pool.")
        return errors


@dataclass(frozen=True)
class LiquidityMetric:
    symbol: str
    eligible: bool
    score: float
    quote_age_seconds: Optional[float]
    spread_bps: Optional[float]
    side_depth_usd: Optional[float]
    reason: str


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quote_age_seconds(quote, now: datetime) -> Optional[float]:
    timestamp = getattr(quote, "timestamp", None)
    if timestamp is None:
        return None
    try:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds())
    except (AttributeError, TypeError, ValueError):
        return None


def score_quote(
    symbol: str,
    quote,
    config: LiquiditySelectionConfig,
    now: datetime,
) -> LiquidityMetric:
    if quote is None:
        return LiquidityMetric(symbol, False, 0.0, None, None, None, "missing_quote")
    age = _quote_age_seconds(quote, now)
    bid = _finite(getattr(quote, "bid_price", None))
    ask = _finite(getattr(quote, "ask_price", None))
    bid_size = _finite(getattr(quote, "bid_size", None))
    ask_size = _finite(getattr(quote, "ask_size", None))
    if age is None or bid is None or ask is None or bid <= 0 or ask <= bid:
        return LiquidityMetric(symbol, False, 0.0, age, None, None, "invalid_quote")
    midpoint = (bid + ask) / 2.0
    spread = (ask - bid) / midpoint * 10_000.0
    side_depth = None
    if bid_size is not None and ask_size is not None and bid_size >= 0 and ask_size >= 0:
        side_depth = min(bid * bid_size, ask * ask_size)

    reasons = []
    if age > config.max_quote_age_seconds:
        reasons.append("stale")
    if spread > config.max_spread_bps:
        reasons.append("wide_spread")
    if side_depth is None or side_depth < config.min_side_depth_usd:
        reasons.append("thin_depth")
    eligible = not reasons
    if not eligible:
        return LiquidityMetric(
            symbol, False, 0.0, age, spread, side_depth, "+".join(reasons)
        )

    freshness = max(0.0, 1.0 - age / config.max_quote_age_seconds)
    spread_quality = max(0.0, 1.0 - spread / config.max_spread_bps)
    depth_quality = min(
        1.0,
        math.log1p(side_depth / config.min_side_depth_usd) / math.log(21.0),
    )
    score = 0.45 * spread_quality + 0.30 * freshness + 0.25 * depth_quality
    return LiquidityMetric(symbol, True, score, age, spread, side_depth, "eligible")


def select_liquid_universe(
    quotes: Mapping[str, object],
    *,
    config: LiquiditySelectionConfig = LiquiditySelectionConfig(),
    now: Optional[datetime] = None,
) -> tuple[tuple[str, ...], list[LiquidityMetric]]:
    errors = config.validate()
    if errors:
        raise ValueError("Invalid liquidity selection config: " + "; ".join(errors))
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Liquidity selection time must be timezone-aware.")
    metrics = [
        score_quote(symbol, quotes.get(symbol), config, now)
        for symbol in ADAPTIVE_USD_CANDIDATE_POOL
    ]
    eligible = sorted(
        (metric for metric in metrics if metric.eligible),
        key=lambda metric: (-metric.score, metric.symbol),
    )
    benchmark = next(
        (metric for metric in eligible if metric.symbol == config.benchmark_symbol),
        None,
    )
    if benchmark is None:
        raise ValueError("Benchmark BTC/USD is not currently liquidity-eligible.")
    selected = [config.benchmark_symbol]
    selected.extend(
        metric.symbol
        for metric in eligible
        if metric.symbol != config.benchmark_symbol
    )
    selected = selected[: config.max_symbols]
    if len(selected) < config.min_symbols:
        raise ValueError(
            f"Only {len(selected)} liquidity-eligible symbols; "
            f"{config.min_symbols} required for adaptive research."
        )
    return tuple(selected), metrics


def universe_from_research_run(run: Optional[dict]) -> Optional[tuple[str, ...]]:
    if not isinstance(run, dict) or not bool(run.get("passed")):
        return None
    raw = run.get("universe_json")
    if raw is None:
        return None
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, list) or not values:
        return None
    symbols = tuple(str(value) for value in values)
    if (
        not 6 <= len(symbols) <= 12
        or len(set(symbols)) != len(symbols)
        or "BTC/USD" not in symbols
    ):
        return None
    return symbols


def latest_passing_universe(store, strategy_id: str) -> Optional[tuple[str, ...]]:
    for run in store.get_research_runs(strategy_id, limit=500):
        universe = universe_from_research_run(run)
        if universe is not None:
            return universe
    return None


def lock_deployment_universe(store, universe: tuple[str, ...]) -> None:
    if (
        not 6 <= len(universe) <= 12
        or len(set(universe)) != len(universe)
        or "BTC/USD" not in universe
    ):
        raise ValueError("Cannot lock an invalid adaptive deployment universe.")
    store.set_kv(ADAPTIVE_LOCKED_UNIVERSE_KEY, json.dumps(list(universe)))


def locked_deployment_universe(store) -> Optional[tuple[str, ...]]:
    raw = store.get_kv(ADAPTIVE_LOCKED_UNIVERSE_KEY)
    if not raw:
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, list):
        return None
    universe = tuple(str(value) for value in values)
    if (
        not 6 <= len(universe) <= 12
        or len(set(universe)) != len(universe)
        or "BTC/USD" not in universe
    ):
        return None
    return universe


def require_locked_universe(
    store,
    requested: Optional[tuple[str, ...]] = None,
) -> tuple[str, ...]:
    locked = locked_deployment_universe(store)
    if locked is None:
        raise ValueError(
            "Adaptive deployment universe is not locked. Promote a passing "
            "RESEARCH run to SHADOW first."
        )
    if requested is not None and tuple(requested) != locked:
        raise ValueError(
            "Requested runtime universe does not exactly match the promoted "
            f"research universe. locked={locked}, requested={tuple(requested)}"
        )
    return locked
