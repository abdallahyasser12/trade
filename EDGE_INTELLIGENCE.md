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
8. **Candle pressure** — close location inside the completed candle; a zero-range bar is neutral, never bearish.
9. **Flow pressure** — volume signed by candle direction, the standard causal order-flow proxy available from bars.
10. **Wick rejection** — dominance of lower wicks (dip buying) over upper wicks (rally selling) across the recent window.
11. **Structure persistence** — the frequency of higher closes and higher lows: micro market structure rather than a smoothed average.
12. **Horizon agreement** — whether the fast and medium impulse horizons agree, interpreted per profile: continuation profiles want both bullish, a reversion entry wants the fast horizon turning up while the medium horizon is still stretched down, a pullback weights the medium horizon, a scalp weights the fast horizon.
13. **Coherence factor** — internal-contradiction penalty computed AFTER profile transformation, so every input means "fit for this profile". A textbook dip-buy is no longer punished merely because its raw momentum is bearish; dispersion across fit evidence is what discounts the score.

Quote spread and L1 size imbalance are also journaled as diagnostics. Adverse L1 imbalance continues to be a separate negative-only execution-quality penalty.

## Strategy-aware profiles

The same raw features are interpreted differently by the current strategy families:

- `opening_range_breakout`, `volatility_squeeze`, breakout/expansion sleeves -> **breakout** profile;
- `vwap_reversion`, mean-reversion sleeves -> **mean_reversion** profile;
- adaptive trend-pullback sleeve -> **trend_pullback** profile;
- cross-sectional momentum/trend-continuation sleeves -> **trend** profile;
- scalp sleeves (any strategy or reason containing `scalp`) -> **scalp** profile: fast-horizon impulse, participation, volatility release and candle/flow pressure dominate, slow trend structure barely matters;
- unknown families -> conservative **generic** profile.

This avoids indicator soup. For example, a high range location is constructive for a breakout but not automatically constructive for a mean-reversion entry.

## Opportunity score

The shared opportunity ranker now evaluates:

`net alpha × evidence confidence × regime compatibility × strategy reliability × liquidity quality × setup quality × drawdown penalty × tail-risk penalty`

`setup_quality` is hard-clamped to `[0, 1]`. Uninitialized edge history is neutral (`1.0`) rather than fabricated, so missing edge features cannot silently block otherwise valid strategy/lifecycle evidence.

Each candidate journals the full `edge_intelligence` payload and `edge_score`, including the profile, component scores, diagnostics and the final quality multiplier. When the assessment is not measurable, `edge_score` is journaled as `None` (never a fake mid score) and the multiplier stays exactly `1.0`.

Robustness guards, each covered by tests: an unsorted or duplicated time index refuses measurement (label slicing on such an index would not be causal), and a NaN on the evaluated bar refuses measurement instead of silently assessing the prior bar.

## Multi-trade behavior

This upgrade intentionally does not add a trade quota. Frequency still comes from:

- the 5-minute crypto universe and four adaptive causal sleeves;
- the market-wide equity universe and its independent intraday strategy families;
- multiple ranked entries per completed cycle where portfolio capacity allows;
- risk recycling after exits.

Existing safeguards remain authoritative: per-trade risk, max positions, total exposure, portfolio heat, session-loss kill switches, cost-to-risk gates, correlation limits, stale quote/bar rejection, lifecycle state, broker reconciliation, protective stops, and the conservative equity PDT gate.

## Exit intelligence: when to sell and when to keep

`quant/exit_intelligence.py` is the post-entry mirror of the edge layer. On each completed bar it measures whether the state still supports holding an open long — trend health, momentum decay, candle pressure, signed-volume flow, stall detection (bars since the last peak close plus advance frequency) and signed path efficiency — and produces a 0–100 **hold score** with a HOLD / TIGHTEN / EXIT stance. The interpretation is profile-aware: a stalled trend bar is bad news for a momentum position but normal mid-reversion, and a scalp lives or dies on immediate continuation.

`TradeManagementPolicy` (quant/exits.py) consumes the score only through individually flag-gated behaviors that all default OFF, so every deployed configuration behaves exactly as before until research measures a feature:

- **Profit target** (`enable_profit_target`) — bank a configured fraction of the original quantity at a configured R multiple; the scalp's bread and butter.
- **Smart exit** (`enable_smart_exit`) — full exit when a KNOWN hold score collapses below a threshold while the trade is at or above a configured R floor; an unmeasurable state is neutral and leaves the position to its protective stop.
- **Adaptive trail** (`enable_adaptive_trail`) — a measurably weakening hold tightens the ATR trail toward a tighter multiplier; promoted runners always keep their wide trail.

The research simulator computes the hold score per open position per completed bar whenever the policy consumes it, and journals the hold configuration in the result snapshot, so the incremental value of every exit behavior is measurable on chronological walk-forward data before anything enables it in a deployed configuration.

## Scalping profile (research hypothesis)

Scalping — many very short trades targeting small profits — is expressed as configuration, not as a new engine:

- `quant/intraday.py` adds `make_scalp_strategy_config()` (1-minute bars, rescaled horizons), `make_scalp_risk_config()` and `SCALP_INTRADAY_OBJECTIVE` (15–80 trades/day measured, never targeted). Frequency never buys risk: unit risk drops from 0.15% to 0.10%, the execution-cost gate TIGHTENS from 35% to 30% of risk, and every absolute ceiling (session loss, exposure, positions, live notional) is identical to the 5-minute profile.
- `quant/exits.py` adds `scalp_exit_policy()`: profit target at +1.2R banking 60%, break-even at +0.8R, a 12-bar stagnation stop, a 1.8x trail tightening to 1.1x on measured edge decay, and the hold-score smart exit so a stalled scalp releases its capital to the next ranked opportunity.
- `scripts/run_adaptive_intraday_research.py --exit-profile scalp` measures the scalp exit hypothesis under the full chronological walk-forward protocol; `--locked-config` refuses it because locked verification must reproduce the promoted configuration exactly.

None of this deploys by existing. A scalp sleeve reaches a broker only through the same research -> OOS -> SHADOW/PAPER -> lifecycle promotion path as everything else.

## Known evidence gap (deliberately not papered over)

Research promotion evidence for the adaptive strategy is currently measured under the simulator's `trailing+runner` exit policy, while the production crypto engine manages exits with the strategy's plain ATR ratchet plus strategy EXIT signals. The new hold-intelligence exits are therefore research-measurable but not yet production-executable; wiring `TradeManagementPolicy` into the production engines (behind the manifest/lifecycle gates) is the required follow-up before any smart-exit or scalp configuration can be promoted past PAPER honestly.

## Validation contract

The new tests verify:

- edge scores and multipliers are bounded, and every profile's weights sum to exactly 1.0 over an identical component set;
- future bars cannot change a historical assessment (edge and hold layers both);
- insufficient history is neutral rather than fabricated; the exact `min_bars` boundary is pinned;
- unsorted/duplicated indexes and NaN on the evaluated bar refuse measurement;
- all current strategy families — including scalp — route to the intended profile;
- setup quality can only reduce an opportunity score and cannot boost it above the evidence-backed baseline;
- an end-to-end scan through the REAL `EvidenceQuantScanner` path proves the assessment is measured (not silently neutral) and its multiplier reaches the ranked score, and that a starved history journals `edge_score = None` with a perfectly neutral multiplier;
- every new exit behavior is inert by default: a policy with default flags produces bit-identical management with or without a hold score.

No profitability claim is made. A strategy must still pass the repository's chronological research, SHADOW/PAPER forward evidence and lifecycle gates before capital is promoted.
