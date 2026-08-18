"""Final production composition for the market-wide US-equity PAPER desk.

A forward strategy has two distinct symbol sets:

* **entry-eligible** — researched symbols that still pass the latest hard current
  executability/liquidity screen; only these may receive a new broker allocation.
* **management-only** — already-held SHADOW symbols (real PAPER positions are added by
  the base engine independently) that must keep receiving completed bars even when a
  later universe refresh temporarily makes them ineligible for a fresh entry.

Keeping those sets separate prevents two opposite evidence errors: freezing a SHADOW
position when its spread/liquidity temporarily deteriorates, or accidentally turning a
held management symbol back into a valid new-entry candidate.
"""

from __future__ import annotations

from equity_marketwide_deployment import ManifestLockedMarketWideEquityTradeEngine
from equity_marketwide_engine import MarketWideEquityTradeEngine
from quant.contracts import StrategyState


def partition_locked_universe(
    locked_universe: tuple[str, ...],
    hard_selected: tuple[str, ...],
    shadow_held: set[str] | frozenset[str],
    *,
    min_entry_breadth: int = 3,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return (fresh-entry set, ordered scan/management set).

    Every returned symbol must belong to the immutable researched universe. A held
    SHADOW symbol can extend only the management set; it can never restore itself to
    fresh-entry eligibility after failing the current hard selector.
    """
    if min_entry_breadth < 1:
        raise ValueError("min_entry_breadth must be positive.")
    locked_set = set(locked_universe)
    hard_set = set(hard_selected)
    executable = tuple(symbol for symbol in locked_universe if symbol in hard_set)
    entry_eligible = (
        frozenset(executable)
        if len(executable) >= min_entry_breadth
        else frozenset()
    )
    held = set(shadow_held) & locked_set
    management_set = set(executable) | held
    management = tuple(
        symbol for symbol in locked_universe if symbol in management_set
    )
    return entry_eligible, management


class ProductionMarketWideEquityTradeEngine(ManifestLockedMarketWideEquityTradeEngine):
    """Manifest-locked runtime with state-aware management-only symbol handling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hard_selected_symbols: tuple[str, ...] = ()
        self._entry_eligible_symbols: frozenset[str] = frozenset()

    @property
    def entry_eligible_symbols(self) -> frozenset[str]:
        return self._entry_eligible_symbols

    def _refresh_universe_if_due(self, now) -> None:
        # Bypass ManifestLockedMarketWideEquityTradeEngine's narrower refresh override
        # so a collapse in fresh-entry eligibility cannot interrupt management of an
        # already-open SHADOW position. Startup has already enforced the same manifest.
        previous_refresh = self._last_universe_refresh
        MarketWideEquityTradeEngine._refresh_universe_if_due(self, now)
        refreshed = self._last_universe_refresh != previous_refresh
        if refreshed or not self._hard_selected_symbols:
            self._hard_selected_symbols = tuple(
                self._universe.selected if self._universe else ()
            )

        hard_selected = self._hard_selected_symbols
        forward = self._forward_strategy_ids()
        if not forward:
            self._entry_eligible_symbols = frozenset(hard_selected)
            if self._universe is not None:
                self._universe.selected = hard_selected
            return

        if not self._locked_research_universe:
            raise ValueError(
                "Forward equity strategy has no locked research universe."
            )

        shadow_held: set[str] = set()
        for strategy_id in forward:
            if self.lifecycle.state(strategy_id) is StrategyState.SHADOW:
                shadow_held.update(self.scanner.shadow.held_symbols(strategy_id))

        entry_eligible, management_symbols = partition_locked_universe(
            self._locked_research_universe,
            hard_selected,
            shadow_held,
        )
        self._entry_eligible_symbols = entry_eligible
        if self._universe is not None:
            self._universe.selected = management_symbols

        with self._lock:
            # Status reports fresh-entry breadth, not extra held symbols carried only
            # so their exits/stops continue to produce honest evidence.
            self._status.selected_symbols = len(self._entry_eligible_symbols)
            self._persist_status()

        if not self._entry_eligible_symbols and shadow_held:
            currently_executable = set(self._locked_research_universe) & set(hard_selected)
            self.store.log(
                "WARNING",
                "Fresh market-wide equity entries are blocked by current universe "
                "breadth; held SHADOW symbols remain under management.",
                context={
                    "currently_executable_researched": len(currently_executable),
                    "shadow_management_symbols": sorted(
                        set(management_symbols) - currently_executable
                    ),
                },
                session_id=self.session_id,
            )

    def _execute_ranked_entries(
        self,
        decisions,
        quotes,
        bar_timestamp,
        account,
        broker_positions,
        now,
    ):
        entry_eligible = self._entry_eligible_symbols
        if not entry_eligible:
            return None
        eligible_decisions = [
            decision
            for decision in decisions
            if decision.candidate.symbol in entry_eligible
        ]
        return super()._execute_ranked_entries(
            eligible_decisions,
            quotes,
            bar_timestamp,
            account,
            broker_positions,
            now,
        )
