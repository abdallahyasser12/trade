# Quant Desk Layer — Architecture and Methodology

This document describes the multi-market quant-desk layer added on top of
the existing (and unchanged) stock and crypto execution engines. It covers
what was built, how it is validated, what has and has not been measured,
and exactly what remains gated.

**Nothing in this document claims profitability.** Every strategy in the
registry is a hypothesis with a recorded lifecycle state; as shipped, every
strategy is in `RESEARCH` state and deploys no capital.

## Architecture

```
stocks + ETFs + crypto bars (Alpaca feeds, provider-agnostic Protocols)
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

* `CryptoTradeEngine` and `TradeEngine` accept an optional
  `quant_journal`; when attached they journal every decision (including
  block reasons) and write attribution rows on sell fills. Journal failures
  can never affect trading (observer contract, tested).
* `CryptoPositionState` gained three optional fields
  (`initial_stop_price`, `lowest_close`, `entry_regime`) with defaults, so
  state persisted by earlier builds loads unchanged.
* The dashboard gained a **Quant Desk** tab (`quant_dashboard.py`):
  market state, opportunity board with rejection reasons, strategy
  marketplace with lifecycle states, research runs with per-gate verdicts,
  measured execution quality, drift findings and risk events.
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

## Operating the research loop

```bash
# 1. Research a strategy family on real data (requires credentials):
python - <<'PY'
from quant.research.factory import run_quant_walk_forward, record_research_report
# fetch bars via data_feed / crypto_data_feed, build candidate params,
# then run and record; see tests/test_quant_factory.py for the exact shape.
PY

# 2. Promote to SHADOW only through the lifecycle manager, citing the run:
#    LifecycleManager.request_promotion(strategy_id, PromotionEvidence(research_run=row))

# 3. Run the scanner on a schedule; SHADOW strategies accumulate virtual
#    attributions; the dashboard's marketplace shows the record build up.

# 4. PAPER and beyond follow the evidence gates in quant/lifecycle.py.
```
