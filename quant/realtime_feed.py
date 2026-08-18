"""Real-time crypto feed adapter: fresh quotes, one history fetch per completed bar.

A fast protection loop must refresh quotes on every cycle without downloading the
same OHLCV history repeatedly.  An optional quote observer receives every fresh quote
snapshot for dashboards/telemetry; observer failures are logged and never interfere
with broker protection or trading-state reconciliation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Optional

import pandas as pd

from config import timeframe_to_timedelta

logger = logging.getLogger(__name__)


class CompletedBarCachingCryptoFeed:
    """Wrap an AlpacaCryptoDataFeed-compatible object without stale quote caching."""

    def __init__(self, backend, *, quote_observer: Optional[Callable[[object], None]] = None):
        self.backend = backend
        self.quote_observer = quote_observer
        self._lock = RLock()
        self._cache: dict[tuple[str, str, int], tuple[pd.Timestamp, pd.DataFrame]] = {}
        self._quote_observer_failures = 0

    def set_interrupt_event(self, event) -> None:
        setter = getattr(self.backend, "set_interrupt_event", None)
        if callable(setter):
            setter(event)

    @property
    def rate_limit_events(self) -> int:
        return int(getattr(self.backend, "rate_limit_events", 0))

    @property
    def quote_observer_failures(self) -> int:
        return self._quote_observer_failures

    @staticmethod
    def _completed_bucket(
        now: datetime,
        timeframe: str,
        tolerance_seconds: float,
    ) -> pd.Timestamp:
        if now.tzinfo is None:
            raise ValueError("Bar-cache time must be timezone-aware.")
        duration = timeframe_to_timedelta(timeframe)
        seconds = duration.total_seconds()
        if seconds <= 0:
            raise ValueError("Timeframe duration must be positive.")
        safe_now = now.astimezone(timezone.utc) - timedelta(seconds=tolerance_seconds)
        epoch = safe_now.timestamp()
        current_start_epoch = int(epoch // seconds) * seconds
        latest_completed_start = current_start_epoch - seconds
        return pd.Timestamp(latest_completed_start, unit="s", tz="UTC")

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
        *,
        now: Optional[datetime] = None,
        tolerance_seconds: float = 5.0,
    ) -> pd.DataFrame:
        now = now or datetime.now(timezone.utc)
        bucket = self._completed_bucket(now, timeframe, tolerance_seconds)
        key = (symbol, timeframe, int(lookback_bars))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached[0] == bucket:
                return cached[1].copy()
        frame = self.backend.get_recent_bars(
            symbol,
            timeframe,
            lookback_bars,
            now=now,
            tolerance_seconds=tolerance_seconds,
        )
        with self._lock:
            self._cache[key] = (bucket, frame.copy())
        return frame

    def get_latest_quotes(self, symbols):
        # Never cache quotes: spread/freshness, depth and stop watchdogs require the
        # current market state on every engine cycle.
        quotes = self.backend.get_latest_quotes(symbols)
        if self.quote_observer is not None:
            try:
                self.quote_observer(quotes)
            except Exception:  # telemetry must never break protective execution
                self._quote_observer_failures += 1
                logger.exception("Real-time quote telemetry observer failed")
        return quotes

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
