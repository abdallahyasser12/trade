"""Manifest-enforced deployment wrapper for the market-wide equity engine.

The base engine owns crash safety and broker execution.  This wrapper adds the
research/deployment invariant: once any strategy advances beyond RESEARCH, the runtime
must be the exact researched git/config identity and may only trade symbols from the
locked universe that still pass the current hard executability screen.
"""

from __future__ import annotations

import math
from datetime import timezone

import pandas as pd

from data_feed import DataFeedError
from equity_marketwide_engine import MarketWideEquityTradeEngine
from quant.contracts import StrategyState
from quant.equity_manifest import EQUITY_STRATEGY_IDS, require_common_locked_universe


class ManifestLockedMarketWideEquityTradeEngine(MarketWideEquityTradeEngine):
    """Production entry point for SHADOW/PAPER market-wide equities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._locked_research_universe: tuple[str, ...] = ()

    def _forward_strategy_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for strategy_id in sorted(EQUITY_STRATEGY_IDS):
            state = self.lifecycle.state(strategy_id)
            if state not in {StrategyState.RESEARCH, StrategyState.QUARANTINE}:
                ids.append(strategy_id)
        return tuple(ids)

    def _startup(self) -> None:
        forward = self._forward_strategy_ids()
        if forward:
            self._locked_research_universe = require_common_locked_universe(
                self.quant_store,
                forward,
                runtime_config=self.config,
            )
        else:
            # RESEARCH-only sessions may inspect discovery but cannot receive broker
            # allocation because lifecycle risk multipliers remain zero.
            self._locked_research_universe = ()
        super()._startup()

    def _refresh_universe_if_due(self, now) -> None:
        super()._refresh_universe_if_due(now)
        forward = self._forward_strategy_ids()
        if not forward:
            return
        if not self._locked_research_universe:
            raise DataFeedError("Forward equity strategy has no locked research universe.")
        selected_now = tuple(self._universe.selected if self._universe else ())
        allowed = tuple(
            symbol for symbol in self._locked_research_universe if symbol in set(selected_now)
        )
        if len(allowed) < 3:
            raise DataFeedError(
                "Fewer than three researched equity symbols remain currently executable; "
                "new entries are blocked while existing protective stops remain managed."
            )
        self._universe.selected = allowed
        with self._lock:
            self._status.selected_symbols = len(allowed)
            self._persist_status()

    def _execute_ranked_entries(self, decisions, quotes, bar_timestamp, account, broker_positions, now):
        """Re-check live quote freshness/spread immediately before broker submission."""
        safe = []
        now_stamp = pd.Timestamp(now).tz_convert("UTC")
        for decision in decisions:
            quote = quotes.get(decision.candidate.symbol)
            if quote is None:
                continue
            try:
                bid = float(quote.bid_price)
                ask = float(quote.ask_price)
                stamp = pd.Timestamp(quote.timestamp)
            except (AttributeError, TypeError, ValueError):
                continue
            if stamp.tzinfo is None:
                continue
            if not all(math.isfinite(value) for value in (bid, ask)) or bid <= 0 or ask <= bid:
                continue
            age = max(0.0, (now_stamp - stamp.tz_convert("UTC")).total_seconds())
            midpoint = (bid + ask) / 2.0
            spread_bps = (ask - bid) / midpoint * 10_000.0
            if age > self.config.max_quote_age_seconds:
                continue
            if spread_bps > self.config.max_spread_bps:
                continue
            if self._locked_research_universe and decision.candidate.symbol not in self._locked_research_universe:
                continue
            safe.append(decision)
        return super()._execute_ranked_entries(
            safe,
            quotes,
            bar_timestamp,
            account,
            broker_positions,
            now,
        )
