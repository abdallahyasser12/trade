# Quant Desk Layer — Architecture and Methodology

This document describes the multi-market quant-desk layer added on top of
the existing (and unchanged) stock and crypto execution engines. It covers
what was built, how it is validated, what has and has not been measured,
and exactly what remains gated.

**Nothing in this document claims profitability.** Every strategy in the
registry is a hypothesis with a recorded lifecycle state; as shipped, every
strategy is in `RESEARCH` state and deploys no capital.

## Architecture

This diagram is the *intended* data flow, not a claim that every arrow is
live. `quant/providers.py`'s Protocol layer has no production call site,
and several stages below are reachable only through `QuantScanner`, which
nothing constructs. See "What is wired into the running system" for the
component-by-component status before relying on any stage here.

```
stocks + ETFs + crypto bars
  live:     data_feed.AlpacaDataFeed / crypto_data_feed.AlpacaCryptoDataFeed
  research: quant/data/ (equities: chunked, cached, calendar-aware)
  NOTE: quant/providers.py's provider-agnostic Protocols are unused
    │
    ▼
quant/features.py        multi-timeframe causal features, session-anchored
quant/store.py           point-in-time feature store (available_at semantics)
    │
    ▼
quant/strategies/*       one UniverseStrategy protocol (the production
                         crypto protocol, formalized) + intraday library
quant/ml/*               calibrated direction model (logistic baseline,
                         PAV calibration, OOD detection)
quant/regime.py          probabilistic multi-input regime engine
    │
    ▼
quant/abstention.py      BUY / EXIT / HOLD / ABSTAIN with named reasons
quant/opportunity.py     expected NET alpha scoring + global cross-market ranker
quant/portfolio.py       portfolio heat, exposure/sector/correlation caps,
                         deterministic greedy allocator (no covariance inversion)
quant/execution.py       pre-trade cost gate: live spread + MEASURED slippage
                         (max(measured, assumed)) + participation cap
    │
    ▼
quant/scanner.py         one decision cycle; journals every verdict
quant/shadow.py          virtual fills for SHADOW-state strategies
quant/journal.py         engine adapters: decisions + trade attribution
    │
    ▼
existing engines         TradeEngine (stock paper) / CryptoTradeEngine
                         (paper + gated live) — safety logic untouched
    │
    ▼
quant/marketplace.py     rolling per-strategy statistics incl. per-regime
quant/drift.py           PSI / KS / expectancy-shift / calibration drift
quant/lifecycle.py       RESEARCH→SHADOW→PAPER→LIVE_CANARY→LIVE ladder,
                         REDUCED_RISK / QUARANTINE demotions
quant/research/*         simulator, walk-forward factory, Monte Carlo
                         stress, deflated Sharpe multiple-testing control
```

Persistence: the desk uses its own schema-versioned SQLite database
(`data/quant_desk.db`, `quant/store.py`) so the audited trading database
(`data/trading_bot.db`) and its recovery guarantees are untouched. Tables:
decisions, attributions, lifecycle, promotion_history, research_runs,
features (point-in-time), model_registry, drift_metrics, risk_events,
shadow_positions, kv.

## The strategy interface

`quant/strategies/base.py` formalizes the protocol the production crypto
strategies already implement (`required_bars`, `prepare_symbol`,
`prepare_universe`, `evaluate`, `evaluate_prepared`, `ratchet_stop`).
New strategies implement the same protocol, so they are compatible with
the existing crypto engine adapters, the existing injected research path
(`engine_research.py`), and the new simulator.

Registered families (`quant/strategies/registry.py`), each with an explicit
hypothesis string, market scope, and natural timeframes:

| strategy_id | markets | horizon | status |
|---|---|---|---|
| crypto_regime_breakout | crypto | 1H/4H | existing production strategy |
| crypto_cross_sectional | crypto | 4H/1D | existing; best measured expectancy, FAILED promotion vs benchmark |
| crypto_mean_reversion | crypto | 1H/4H | existing |
| crypto_volatility_regime | crypto | 4H/1D | existing |
| donchian_trend | crypto + equities | 4H/1D | new, RESEARCH |
| opening_range_breakout | equities | 5m/15m | new, RESEARCH |
| vwap_reversion | equities + crypto | 5m/15m | new, RESEARCH |
| volatility_squeeze | equities + crypto | 15m/1H/4H | new, RESEARCH |

## Causality guarantees (tested mechanically)

* Every feature column: value at bar N is invariant under appending future
  bars (`tests/test_quant_features.py`).
* Breakout references exclude the current bar (`shift(1)` invariant).
* Higher-timeframe context joins use **availability time** (bar timestamp +
  duration), so a daily close is invisible to intraday decisions made
  before that close existed.
* The feature store's as-of queries filter on `available_at`, so
  late-published data (fundamentals, news) is invisible before its
  publication moment.
* The simulator executes decisions strictly at the NEXT bar's open, checks
  protective stops against next-bar open/low before close-derived logic,
  and fills gaps at the adverse open.
* Direction-model labels exclude the final `horizon` rows and the
  train/validation split carries a `horizon`-sized embargo.
* The universe builder refuses stats dated after the requested as-of
  moment (survivorship-bias guard).

## Research methodology

`quant/research/factory.py` runs bounded chronological walk-forward:
rolling train → embargo → test windows; per-fold candidate selection on
training data only with an explicit multiple-testing penalty; the selected
candidate evaluated out-of-sample at base and stressed costs. On top of
the existing methodology it adds:

* **Monte Carlo trade-sequence stress** (`research/stress.py`): resampled
  drawdown distributions, losing-streak lengths, risk-of-ruin proxy, and a
  deterministic haircut variant (winners demoted/shrunk, extra cost per
  trade).
* **Deflated Sharpe** (`research/multiplicity.py`): the observed Sharpe is
  compared against the expected maximum Sharpe of N zero-skill trials,
  where N includes **every recorded prior trial** for the strategy
  (`QuantStore.total_trials`), not just the current grid.

Promotion gates (all individually recorded, none averaged): min folds,
min OOS trades, positive-fold ratio, compounded return, benchmark excess,
aggregate Sharpe, profit factor, fold drawdown ceiling, stressed return,
deflated-Sharpe probability, ruin-probability ceiling. A PASS makes a
strategy *eligible for SHADOW*, nothing more.

## Lifecycle and safety

`quant/lifecycle.py` enforces the ladder with rung-specific evidence:

* RESEARCH → SHADOW: recorded research PASS with sufficient OOS trades and
  a recorded trial count.
* SHADOW → PAPER: sufficient shadow trades with bounded shadow drawdown.
* PAPER → LIVE_CANARY: sufficient paper trades **and** measured execution
  samples **and** a research re-run with measured costs that still passes
  **and** zero open operational defects **and** an explicit operator
  acknowledgement.
* LIVE_CANARY → LIVE: sufficient canary trades and operator ack again.

Structural rules: promotions climb one rung at a time; **no transition
into a live state can ever happen without `operator_ack=True`**; demotion
to REDUCED_RISK/QUARANTINE never needs an ack and can be abrupt.
`DegradationMonitor` recommends reduce-risk on a single soft breach with
adequate sample and quarantine on hard breaches (drawdown, cost
inflation) or multiple soft breaches — and recommends nothing on
insufficient samples.

Existing safety controls are unchanged: the crypto live unlock
(`ENABLE_LIVE_TRADING` + separate live keys + explicit UI arming), stock
paper-only boundary, kill switches, session-loss halts, stop watchdogs,
reconciliation, and every deterministic risk cap in
`risk_manager`/`crypto_risk`. The quant layer can only *reduce* what those
layers would allow (lifecycle risk multipliers ≤ 1, additional rejection
gates); it cannot raise any ceiling.

## Execution intelligence

`quant/execution.py` estimates per-order cost from the live spread (when
fresh), the **measured** rolling slippage from the executions ledger, and
market-specific fees. The measured figure folds in as
`max(measured, assumed)`: a worse measurement tightens gating immediately;
a better one only takes effect after research is re-run with it (the
promotion evidence explicitly requires that re-run). Orders whose expected
net alpha after costs plus an uncertainty margin is not positive are
rejected before sizing, with journaled reasons.

## Explainability

Every decision — executed, rejected, abstained, shadow — is journaled with
its reasons in `decisions`. Every closed trade (real fills via the engine
hooks; virtual fills via the shadow executor) produces an attribution row
with entry/exit, fees, R-multiple, close-based MFE/MAE, regime at entry,
and strategy identity. The marketplace computes rolling statistics —
including per-regime performance with a minimum-sample floor — from those
rows only.

## What is wired into the running system

This section used to describe what the code *supports*. Support is not
wiring, and the gap between the two was the largest source of drift
between this document and the repository. It is now split into what
actually executes and what does not.

### Reachable at runtime

* **The desk journal is attached.** `app.py` builds one via
  `quant.attach.attach_journal` for both engines, so decisions and
  attribution rows are written to `data/quant_desk.db` during real paper
  sessions. Previously it was constructed only in tests: both engines
  accepted a `quant_journal` and had call sites behind it, but nothing
  ever passed one, so every call site was a permanently-`None` guard and
  the desk observed nothing. Journal failures can never affect trading
  (observer contract, tested).
* **Attribution covers both markets.** `record_stock_exit_fill` did not
  exist, so equity strategies would have shown zero closed trades in the
  marketplace no matter how many they took. Attribution rows are
  idempotent on the executions ledger's `event_key` (schema v2), so a
  re-reconciled fill books one row instead of inflating the trade count.
* **Equity research runs.** `scripts/run_equity_research.py` drives the
  walk-forward factory over `quant/data`, using the equity annualization
  factor and the NYSE session calendar.
* **The Quant Desk dashboard tab** reads the store.

### Present but NOT reachable at runtime

Nothing constructs `QuantScanner`, and it is the only component that wires
the decision pipeline together. Everything below is exercised by tests and
by nothing else:

* `quant/scanner.py`, `quant/shadow.py` — the decision cycle and shadow
  execution.
* `quant/regime.py`, `quant/opportunity.py`, `quant/portfolio.py`,
  `quant/abstention.py` — reached only through the scanner.
* `quant/drift.py`, `quant/universe.py`, `quant/providers.py`,
  `quant/ml/*` — no runtime importer at all.
* `QuantStore.record_features` has no caller, so the point-in-time feature
  store is never written outside tests.

The dashboard's market-state, drift and risk-event panels therefore cannot
populate: their only writer lives inside the scanner.

* `CryptoPositionState` gained three optional fields
  (`initial_stop_price`, `lowest_close`, `entry_regime`) with defaults, so
  state persisted by earlier builds loads unchanged.
* `scripts/run_crypto_research.py --engine council` was repaired (it
  previously crashed with a NameError before the walk-forward started).

## What has NOT been validated (honest status)

* **No new strategy has passed research gates on real market data.** This
  environment has no Alpaca credentials, so the research factory has been
  validated end-to-end on synthetic data only (pipeline correctness, not
  edge evidence). All new strategies are `RESEARCH` state.
* The direction model is a baseline; no claim that it improves any
  strategy until ablation runs on real data say so.
* Live trading remains exactly as gated as before; nothing here changes
  the crypto live unlock or the stock paper-only boundary.
* **Recorded research runs from before the multiple-testing fix are
  stale.** The trial count fed to the deflated Sharpe was the grid size
  when the real search is grid x folds, so N was understated by a factor
  of the fold count and that gate was easier than designed. Any prior
  recorded PASS should be re-run before it is cited as promotion evidence.

## Operating the research loop

```bash
# 1a. Crypto research on real data (requires credentials):
python scripts/run_crypto_research.py --help
python scripts/run_quant_research.py --engine donchian --timeframe 1Hour \
    --symbols BTC/USD ETH/USD SOL/USD --days 730

# 1b. Equity research on real data (requires credentials):
python scripts/run_equity_research.py --help
python scripts/run_equity_research.py --engine orb --timeframe 15Min \
    --symbols SPY QQQ IWM --start 2023-01-01 --end 2024-12-31 \
    --json research-orb.json

#     Without credentials, --synthetic runs the identical pipeline on
#     deterministic session-shaped data. That validates the machinery and
#     is NEVER evidence of edge; such runs are recorded under a strategy id
#     suffixed `_synthetic` so they cannot satisfy a promotion request.

#     Exit codes: 0 = all gates PASS (eligible for SHADOW only),
#     1 = gates FAIL, 2 = could not run.

# 2. Promote to SHADOW only through the lifecycle manager, citing the run:
#    LifecycleManager.request_promotion(strategy_id, PromotionEvidence(research_run=row))

# 3. NOT YET AVAILABLE. The operating loop below depends on QuantScanner,
#    which nothing constructs — there is no scanner CLI and no scheduled
#    entry point, so SHADOW strategies cannot accumulate virtual
#    attributions today. See "Present but NOT reachable at runtime".

# 4. PAPER and beyond follow the evidence gates in quant/lifecycle.py.
```

## Equity data

Research-grade equity history lives in `quant/data`, separate from the
live feed (`data_feed.AlpacaDataFeed.get_recent_bars`), which fetches a
short trailing window per poll and is the wrong shape for research.

* `calendar.py` derives the NYSE session calendar from the published
  rules — holidays, weekend observance, Good Friday from Easter,
  Juneteenth from 2022, early closes — so it stays correct for future
  years and adds no dependency.
* `equity_history.py` chunks on stable calendar boundaries so two
  differently-framed requests covering the same month share a cache entry.
* `provenance.py` records what was requested, what arrived, and how
  complete it is against the real calendar. A symbol with no data is
  `MISSING`, not an empty frame that reads downstream as "no signals".
* Nothing is interpolated. A gap stays a gap and is reported.

Two details that are easy to get wrong and silent when you do: daily bars
are attributed by their **UTC** date (the 00:00-UTC stamping convention
converts to the previous evening in ET, which would shift a whole daily
history back a day for the EST half of the year), and intraday session
membership is decided by **overlap**, not containment (hourly buckets do
not align to the 09:30 open, so containment drops the first and last
bucket of every session).

`4Hour` is unsupported for equities: a 6.5-hour session does not divide
into 4-hour buckets.
