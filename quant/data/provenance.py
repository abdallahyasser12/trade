"""Where a bar series came from, and how much of it is actually there.

A research result is only reproducible if the data behind it is
identified. A frame of bars alone cannot answer "which feed?", "split
adjusted?", "was the range complete or did the provider truncate it?" —
so every research fetch returns bars *and* a ``BarProvenance`` describing
them, and the provenance travels into the recorded research run.

The completeness classification is deliberately conservative: a range
that cannot be proven complete is reported as INCOMPLETE rather than
assumed fine, because a silently truncated history makes a strategy look
like it simply had no signals during the missing window.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional

import pandas as pd


class DataQuality(str, Enum):
    """Quality state of a fetched series.

    Strategies declare the minimum they accept; anything worse is an
    abstention, never a silently interpolated value.
    """

    OK = "OK"
    STALE = "STALE"
    INCOMPLETE = "INCOMPLETE"
    MISSING = "MISSING"
    OUTLIER = "OUTLIER"
    PROVIDER_ERROR = "PROVIDER_ERROR"


# Ordered worst-to-best so callers can compare with an index lookup.
_QUALITY_ORDER = [
    DataQuality.PROVIDER_ERROR,
    DataQuality.MISSING,
    DataQuality.INCOMPLETE,
    DataQuality.OUTLIER,
    DataQuality.STALE,
    DataQuality.OK,
]


def quality_rank(quality: DataQuality) -> int:
    return _QUALITY_ORDER.index(quality)


def meets_quality(actual: DataQuality, required: DataQuality) -> bool:
    """True when ``actual`` is at least as good as ``required``."""
    return quality_rank(actual) >= quality_rank(required)


def worst_quality(*qualities: DataQuality) -> DataQuality:
    """The worst state among the arguments — quality never averages."""
    if not qualities:
        return DataQuality.MISSING
    return min(qualities, key=quality_rank)


@dataclass(frozen=True)
class Completeness:
    """Observed vs expected bar counts over a requested range."""

    expected_bars: int
    observed_bars: int
    missing_sessions: tuple[str, ...] = ()
    # Fraction of expected bars present.  1.0 for a range with no
    # expectation model (e.g. crypto), which reports coverage directly.
    coverage: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.coverage >= 0.995

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_bars": self.expected_bars,
            "observed_bars": self.observed_bars,
            "missing_sessions": list(self.missing_sessions),
            "coverage": round(self.coverage, 6),
        }


@dataclass(frozen=True)
class BarProvenance:
    """Identifies one fetched bar series completely enough to reproduce it."""

    provider: str  # "alpaca"
    feed: str  # "iex" | "sip" | "us" (crypto)
    market: str  # "us_equity" | "crypto"
    symbol: str
    timeframe: str
    requested_start: datetime
    requested_end: datetime
    observed_start: Optional[datetime]
    observed_end: Optional[datetime]
    adjustment: Optional[str]  # "raw" | "split" | "dividend" | "all"
    fetched_at: datetime
    quality: DataQuality
    completeness: Completeness
    regular_hours_only: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        def _iso(value: Optional[datetime]) -> Optional[str]:
            return value.isoformat() if value is not None else None

        return {
            "provider": self.provider,
            "feed": self.feed,
            "market": self.market,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "requested_start": _iso(self.requested_start),
            "requested_end": _iso(self.requested_end),
            "observed_start": _iso(self.observed_start),
            "observed_end": _iso(self.observed_end),
            "adjustment": self.adjustment,
            "fetched_at": _iso(self.fetched_at),
            "quality": self.quality.value,
            "completeness": self.completeness.to_dict(),
            "regular_hours_only": self.regular_hours_only,
            "notes": list(self.notes),
        }

    def fingerprint(self) -> str:
        """Stable hash of the *request*, not the fetch time.

        Two fetches of the same window from the same feed share a
        fingerprint, which is what makes the cache and the reproducibility
        claim meaningful.
        """
        payload = {
            "provider": self.provider,
            "feed": self.feed,
            "market": self.market,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.requested_start.astimezone(timezone.utc).isoformat(),
            "end": self.requested_end.astimezone(timezone.utc).isoformat(),
            "adjustment": self.adjustment,
            "regular_hours_only": self.regular_hours_only,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def assess_equity_completeness(
    bars: pd.DataFrame,
    timeframe: str,
    start: date,
    end: date,
) -> Completeness:
    """Compare observed equity bars against the session calendar.

    Missing sessions are reported by date so a caller can tell "the
    provider has no data for this symbol before 2021" apart from "one
    random Tuesday is absent", which are very different problems.
    """
    from quant.data.calendar import (
        expected_bar_count,
        expected_session_bars,
        session_dates,
        trading_days,
    )

    expected = expected_bar_count(start, end, timeframe)
    observed = int(len(bars))
    if expected <= 0:
        return Completeness(
            expected_bars=0,
            observed_bars=observed,
            missing_sessions=(),
            coverage=1.0 if observed == 0 else 1.0,
        )

    missing: list[str] = []
    if not bars.empty:
        local_days = set(session_dates(bars.index, timeframe))
        for day in trading_days(start, end):
            if expected_session_bars(day, timeframe) and day not in local_days:
                missing.append(day.isoformat())
    else:
        missing = [day.isoformat() for day in trading_days(start, end)]

    return Completeness(
        expected_bars=expected,
        observed_bars=observed,
        # Observed can exceed expected when a provider prints an extra
        # partial bucket; coverage is capped so it never reads as >100%.
        coverage=min(1.0, observed / expected),
        missing_sessions=tuple(missing[:64]),
    )


def assess_crypto_completeness(
    bars: pd.DataFrame,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> Completeness:
    """Crypto trades continuously, so expectation is pure elapsed time."""
    from config import timeframe_to_timedelta

    duration = timeframe_to_timedelta(timeframe)
    span = end - start
    expected = max(0, int(span / duration))
    observed = int(len(bars))
    coverage = 1.0 if expected <= 0 else min(1.0, observed / expected)
    return Completeness(
        expected_bars=expected, observed_bars=observed, coverage=coverage
    )


def classify_quality(
    completeness: Completeness,
    *,
    observed_end: Optional[datetime],
    timeframe: str,
    now: datetime,
    is_crypto: bool,
    staleness_bars: int = 3,
    check_staleness: bool = True,
) -> tuple[DataQuality, tuple[str, ...]]:
    """Derive the quality state and the reasons behind it.

    Order matters: MISSING beats INCOMPLETE beats STALE.  A caller that
    only checks "is it OK" gets the worst problem, and a caller that
    reports to an operator gets every reason.

    ``check_staleness`` must be False for a deliberately historical
    window.  Research over 2023 ends in 2023 by construction; calling
    that "stale" would mark every backtest's data as degraded and train
    the operator to ignore the field.
    """
    notes: list[str] = []
    if completeness.observed_bars == 0:
        return DataQuality.MISSING, ("provider returned no bars for the range",)

    quality = DataQuality.OK
    if not completeness.is_complete:
        quality = DataQuality.INCOMPLETE
        notes.append(
            f"coverage {completeness.coverage:.1%} "
            f"({completeness.observed_bars}/{completeness.expected_bars} bars)"
        )
        if completeness.missing_sessions:
            shown = ", ".join(completeness.missing_sessions[:5])
            notes.append(f"missing sessions include {shown}")

    if observed_end is not None and check_staleness:
        if is_crypto:
            from crypto_data_feed import is_crypto_data_stale

            stale = is_crypto_data_stale(observed_end, timeframe, now, staleness_bars)
        else:
            from data_feed import is_data_stale

            stale = is_data_stale(observed_end, timeframe, now, staleness_bars)
        if stale:
            notes.append(f"newest bar {observed_end.isoformat()} is stale as of {now.isoformat()}")
            if quality is DataQuality.OK:
                quality = DataQuality.STALE

    return quality, tuple(notes)
