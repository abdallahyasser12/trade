"""Aligning a crypto universe to a common index, and what that costs.

The simulator requires exactly aligned indices across the universe, so the
frames are intersected to their common timestamps. That is easy to do by
accident and expensive when it goes wrong: a single symbol listed in 2023
drags an otherwise-2021 universe down to its own start date, silently
discarding years of history for every other symbol. A real run lost 65% of its
window that way and then failed inside the fold arithmetic with a message that
pointed at the folds rather than at the symbol responsible.

This lives in a module rather than inside a script because two scripts need it.
It was previously a private helper in ``scripts/run_crypto_research.py`` that
``scripts/run_sizing_study.py`` reached across for, which only worked when the
scripts directory happened to be on ``sys.path`` — a dependency on process
start-up state rather than on anything declared.
"""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd


def aligned_length(bars: Mapping[str, pd.DataFrame]) -> int:
    """How many timestamps every symbol in this universe shares."""

    common: Optional[pd.Index] = None
    for frame in bars.values():
        common = frame.index if common is None else common.intersection(frame.index)
    return 0 if common is None else len(common)


def drop_limiting_symbols(
    bars: Mapping[str, pd.DataFrame], benchmark: str, *, min_symbols: int = 3
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
    """Drop whichever symbol most limits the aligned window, repeatedly.

    Selection is by measured impact, not by start date. A late listing is the
    obvious way to truncate the common index, but it is not the only one: a
    symbol that began on day one and is missing 400 scattered days costs just
    as much, and a start-date rule would never touch it. Removing the symbol
    whose removal recovers the most bars covers both cases with one rule.

    The benchmark is never dropped — it defines the comparison — and the
    universe never falls below ``min_symbols``, because a cross-sectional
    ranking over two names is not a ranking.
    """

    kept = dict(bars)
    dropped: list[tuple[str, str]] = []
    while len(kept) > min_symbols:
        before = aligned_length(kept)
        best, recovered = None, before
        for symbol in kept:
            if symbol == benchmark:
                continue
            length = aligned_length({s: f for s, f in kept.items() if s != symbol})
            if length > recovered:
                best, recovered = symbol, length
        if best is None:
            break
        frame = kept.pop(best)
        dropped.append((
            best,
            f"starts {frame.index[0]:%Y-%m-%d}, {len(frame)} bars; "
            f"dropping it recovers {recovered - before} aligned bars",
        ))
    return kept, dropped


def align_to_common_index(
    bars: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.Index]:
    """Intersect every frame to the timestamps they all share."""

    common: Optional[pd.Index] = None
    for frame in bars.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) == 0:
        raise ValueError("The symbols share no common timestamps.")
    return {symbol: frame.loc[common] for symbol, frame in bars.items()}, common
