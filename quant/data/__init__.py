"""Research-grade market data access for the quant desk.

The live engines fetch a short trailing window of bars every poll
(``data_feed.AlpacaDataFeed.get_recent_bars``).  Research needs the
opposite shape: months or years of history, for many symbols, fetched
once, cached, and carrying enough provenance that a result can be
reproduced.  This package supplies that second shape without touching
the live path.

Modules:

``calendar``    US equity session calendar (holidays, early closes,
                regular-hours windows) computed from rules — no network
                call and no new dependency.
``provenance``  What was fetched, from where, when, and how complete.
``cache``       Content-addressed on-disk bar cache keyed by the request.
``equity_history``  Chunked, calendar-aware historical equity bars.
"""

from quant.data.calendar import (
    EARLY_CLOSE_ET,
    REGULAR_CLOSE_ET,
    REGULAR_OPEN_ET,
    expected_session_bars,
    is_early_close,
    is_trading_day,
    regular_session_bounds,
    session_count,
    trading_days,
)
from quant.data.cache import BarCache
from quant.data.equity_history import (
    SUPPORTED_TIMEFRAMES,
    BarSource,
    EquityHistory,
    EquityHistoryFetcher,
    chunk_ranges,
)
from quant.data.provenance import (
    BarProvenance,
    Completeness,
    DataQuality,
    meets_quality,
    worst_quality,
)

__all__ = [
    "BarCache",
    "BarProvenance",
    "BarSource",
    "Completeness",
    "DataQuality",
    "EquityHistory",
    "EquityHistoryFetcher",
    "SUPPORTED_TIMEFRAMES",
    "chunk_ranges",
    "meets_quality",
    "worst_quality",
    "EARLY_CLOSE_ET",
    "REGULAR_CLOSE_ET",
    "REGULAR_OPEN_ET",
    "expected_session_bars",
    "is_early_close",
    "is_trading_day",
    "regular_session_bounds",
    "session_count",
    "trading_days",
]
