# Edge Intelligence / High-Velocity Intraday Ranking

## Purpose

The bot already has two distinct sources of trading frequency:

- the adaptive crypto runtime scans a broad 5-minute universe and measures a 6–40 trades/day activity objective without using that objective as a promotion target;
- the market-wide equity runtime ranks many symbols and can recycle released risk into several independent positions during the same session, while retaining PDT, session-loss, portfolio-heat, stale-data and near-close gates.

This upgrade improves *which* of those opportunities receive the next unit of risk. It does **not** force a minimum number of trades and it does not target a fixed daily profit percentage.

## Architecture

`strategy signal -> lifecycle evidence -> execution cost -> Edge Intelligence -> global ranker -> portfolio governor -> broker`

The strategy remains the alpha hypothesis. Edge Intelligence is a causal setup-quality layer. A setup-quality multiplier is bounded to `<= 1.0`, so the layer can demote a noisy candidate but can never create alpha, bypass the lifecycle, or increase an evidence-backed score.

The same scanner path is used by the adaptive crypto runtime and by the market-wide equity runtime. This keeps the feature definitions and ranking semantics identical between SHADOW/PAPER/forward states.

## Causal feature families

The implementation in `quant/edge_intelligence.py` measures the following at one completed timestamp using data sliced at or before that timestamp:

1. **Trend alignment** — fast/slow EWMA separation normalized by ATR.
2. **Impulse-to-noise** — multi-bar log return divided by realized volatility.
3. **Path efficiency** — net displacement divided by total path travelled; this identifies clean directional motion versus whipsaw.
4. **Range location** — close location inside the prior rolling range.
5. **Participation** — current volume relative to the *prior* rolling median, so the current bar cannot inflate its own baseline.
6. **VWAP stretch** — rolling VWAP displacement in ATR units to distinguish constructive location from over-extension.
7. **Volatility release** — short/long realized-volatility ratio for compression/expansion context.
8. **Candle pressure** — close location inside the completed candle.
9. **Agreement quality** — a contradiction penalty across independent directional feature groups.

Quote spread and L1 size imbalance are also journaled as diagnostics. Adverse L1 imbalance continues to be a separate negative-only execution-quality penalty.

## Strategy-aware profiles

The same raw features are interpreted differently by the current strategy families:

- `opening_range_breakout`, `volatility_squeeze`, breakout/expansion sleeves -> **breakout** profile;
- `vwap_reversion`, mean-reversion sleeves -> **mean_reversion** profile;
- adaptive trend-pullback sleeve -> **trend_pullback** profile;
- cross-sectional momentum/trend-continuation sleeves -> **trend** profile;
- unknown families -> conservative **generic** profile.

This avoids indicator soup. For example, a high range location is constructive for a breakout but not automatically constructive for a mean-reversion entry.

## Opportunity score

The shared opportunity ranker now evaluates:

`net alpha × evidence confidence × regime compatibility × strategy reliability × liquidity quality × setup quality × drawdown penalty × tail-risk penalty`

`setup_quality` is hard-clamped to `[0, 1]`. Uninitialized edge history is neutral (`1.0`) rather than fabricated, so missing edge features cannot silently block otherwise valid strategy/lifecycle evidence.

Each candidate journals the full `edge_intelligence` payload and `edge_score`, including the profile, component scores, diagnostics and the final quality multiplier.

## Multi-trade behavior

This upgrade intentionally does not add a trade quota. Frequency still comes from:

- the 5-minute crypto universe and four adaptive causal sleeves;
- the market-wide equity universe and its independent intraday strategy families;
- multiple ranked entries per completed cycle where portfolio capacity allows;
- risk recycling after exits.

Existing safeguards remain authoritative: per-trade risk, max positions, total exposure, portfolio heat, session-loss kill switches, cost-to-risk gates, correlation limits, stale quote/bar rejection, lifecycle state, broker reconciliation, protective stops, and the conservative equity PDT gate.

## Validation contract

The new tests verify:

- edge scores and multipliers are bounded;
- future bars cannot change a historical assessment;
- insufficient history is neutral rather than fabricated;
- all current strategy families route to the intended profile;
- setup quality can only reduce an opportunity score and cannot boost it above the evidence-backed baseline.

No profitability claim is made. A strategy must still pass the repository's chronological research, SHADOW/PAPER forward evidence and lifecycle gates before capital is promoted.
