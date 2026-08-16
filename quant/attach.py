"""Attaching the desk journal to a live engine.

``quant/journal.py`` has always had the adapter, and both engines have
always accepted a ``quant_journal``.  Nothing ever built one, so every
journal call site in ``trade_engine.py`` and ``crypto_engine.py`` sat
behind a permanently-``None`` guard and the desk observed nothing.  This
module is the missing constructor.

Two rules govern everything here:

**The observer never breaks the observed.**  Building a journal is
best-effort: a locked database, a read-only filesystem, a schema written
by a newer build — all of them return ``None`` and let the engine trade
exactly as it did before.  ``attach_journal`` cannot raise.

**The journal never changes a decision.**  It receives what the engine
already decided.  Nothing here is on the path between a signal and an
order, so attaching it cannot alter trading behavior — only whether that
behavior is recorded.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from quant.contracts import Market
from quant.journal import DecisionJournal, strategy_id_for
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore

logger = logging.getLogger(__name__)

_STORE_LOCK = threading.Lock()
_STORES: dict[str, QuantStore] = {}


def shared_quant_store(db_path: str = DEFAULT_QUANT_DB_PATH) -> Optional[QuantStore]:
    """One ``QuantStore`` per database path per process.

    Streamlit reruns the whole script on every interaction, and the
    engine runs on its own thread.  Opening a fresh connection each time
    would leak file handles until the process ran out; the store itself
    is already thread-safe (one connection behind an RLock, WAL mode,
    ``check_same_thread=False``), so sharing one is both cheaper and
    safer than opening many.

    Returns ``None`` rather than raising when the database cannot be
    opened — journaling is optional, trading is not.
    """
    with _STORE_LOCK:
        existing = _STORES.get(db_path)
        if existing is not None:
            return existing
        try:
            store = QuantStore(db_path)
        except Exception as error:
            # A newer schema, a locked file, a read-only mount.  Log once
            # and let every caller fall back to not journaling.
            logger.warning(
                "Quant desk journal unavailable (%s); trading continues unobserved: %s",
                db_path,
                error,
            )
            return None
        _STORES[db_path] = store
        return store


def reset_shared_stores() -> None:
    """Close and forget every shared store.  For tests and shutdown."""
    with _STORE_LOCK:
        for store in _STORES.values():
            try:
                store.close()
            except Exception as error:
                logger.warning("Closing quant store failed: %s", error)
        _STORES.clear()


def attach_journal(
    *,
    market: Market,
    timeframe: str,
    strategy: object,
    strategy_version: str = "1.0.0",
    db_path: str = DEFAULT_QUANT_DB_PATH,
    warn: Optional[Callable[[str], None]] = None,
) -> Optional[DecisionJournal]:
    """Build a journal bound to one engine's identity, or ``None``.

    ``strategy`` is the live strategy instance; its class name is mapped
    to the registry id so journal rows join against the strategy
    registry, the marketplace and the lifecycle ladder rather than
    against an ad-hoc string.

    Never raises.  A caller passes the result straight into an engine's
    ``quant_journal`` argument, and ``None`` restores the previous
    behavior exactly.
    """
    try:
        store = shared_quant_store(db_path)
        if store is None:
            return None
        return DecisionJournal(
            store,
            market=market,
            timeframe=timeframe,
            strategy_id=strategy_id_for(strategy),
            strategy_version=strategy_version,
            warn=warn or (lambda message: logger.warning("%s", message)),
        )
    except Exception as error:
        logger.warning("Could not attach the desk journal: %s", error)
        return None
