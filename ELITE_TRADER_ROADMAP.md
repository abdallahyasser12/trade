# Getting to elite-desk behaviour

An audit of every subsystem against how a professional systematic desk runs a
book, with the findings ranked by edge-per-unit-effort. Three items are already
implemented; the rest is a sequenced plan.

Read this alongside [`SMART_STRATEGY_PLAN.md`](SMART_STRATEGY_PLAN.md) and
[`CRYPTO_TRADING_PLAN.md`](CRYPTO_TRADING_PLAN.md), which describe what the
system is intended to do. This document is about the distance between that
intent and elite practice.

## The thesis

The engineering here is genuinely above retail grade: deterministic decisions,
strictly causal bars, idempotent client order IDs, fail-closed reconciliation, a
closed parameter allowlist, walk-forward promotion gates. That is not the
problem.

The problem is that **the system had never measured a single fact about its own
economics**, and the one number computable from its own defaults is
disqualifying:

| Quantity | Value at shipped defaults |
| --- | --- |
| Round-trip cost | `2 × (10 slippage + 2.5 half-spread + 25 taker)` = **75 bps** |
| Risk unit `1R` = `2.5 × ATR` on hourly BTC | **62–125 bps** |
| **Friction per trade** | **0.6R – 1.2R** |
| Win rate needed at 2R just to break even | **53% – 73%** |

Trend-following systems typically win 35–45% of trades. Meanwhile the exit fires
on any single close below the fast MA, which maximises turnover, and the
promotion gate — the only thing between research and money — never compared the
strategy against simply holding BTC, even though both simulators computed the
benchmark curve and discarded it.

> The gap versus an elite desk is not signal sophistication. It is that a
> well-built machine was optimising gross return against frictions roughly equal
> to its entire edge, validated by a gate that could see neither the friction nor
> the alternative of doing nothing.

Every clever signal multiplies by a number that was unknown and plausibly zero.
That is why the work below starts with measurement, not alpha.

## Shipped

### 1. The stop watchdog no longer liquidates recovered positions

`_run_stop_watchdogs` latched `breached_at` on the first quote at or below the
stop and **never cleared it**. With the 1Hour poll interval at 120s and the
watchdog grace at 15s, the grace check always passed on the next pass. A single
wick below the stop permanently armed a market exit that fired even after price
had fully recovered — converting winning trends into scratches while paying a
full round trip. Reproduced as a test before fixing.

The latch now clears when the bid recovers, escalation requires
`stop_watchdog_min_confirmations` (default 2) sustained observations, and the
engine polls every `position_open_poll_interval_seconds` (default 5s) while a
position is open so confirmation never slows a genuine escalation. The guard
order ID advances when a prior guard is terminal without a fill, so one
rejection can no longer freeze the ID and leave a position unprotected.

### 2. Promotion asks whether you beat holding the asset

The existing "excess return" check compares the *tuned* config against the
*untuned* one — whether tuning helped, not whether the strategy beat the
alternative a long-only book always has. Promotion now also requires beating
compounded buy-and-hold over the same folds, and requires that at least half the
folds individually beat their benchmark so one lucky fold cannot carry the
decision.

### 3. Cost is measured in units of risk

`cost_model.py` is now the single friction formula, consumed by the simulator and
the live engine so they cannot drift apart. `CryptoRiskConfig.max_cost_to_risk`
gates admission in both paths (live measures the real spread from the quote).

The default of `1.0` rejects only the indefensible case of paying a whole risk
unit in costs, so existing behaviour is unchanged. **The default is deliberately
weak — the reporting is the deliverable.** Each backtest now shows round-trip
cost, median cost-per-trade in R, and the break-even win rate that implies.

> **Next action for the operator:** tighten `max_cost_to_risk` toward `0.25` and
> re-run walk-forward. If trade count collapses and gates start failing, that is
> information, not failure. The honest responses are 4Hour/1Day bars or maker
> entries — not a looser ratio.

### Running it

`scripts/run_crypto_research.py` executes the protocol and prints both verdicts —
what a round trip costs in units of R, and whether the strategy beat holding the
benchmark. It reads paper credentials from the environment and touches market
data only; it never constructs a broker and cannot place an order.

```bash
python scripts/run_crypto_research.py                       # config defaults
python scripts/run_crypto_research.py --max-cost-to-risk 0.25 --timeframe 1Day
```

Exit codes: `0` promoted, `1` gates failed (a normal, informative outcome), `2`
could not run. The **Crypto walk-forward research** GitHub Actions workflow runs
the same script from repository secrets (`ALPACA_PAPER_API_KEY`,
`ALPACA_PAPER_SECRET_KEY`), publishes the verdict to the job summary, and uploads
`report.txt` and `verdict.json`. A promotion failure does not fail the pipeline —
only a run that could not complete does.

## Tier 1 — foundational (do before anything else)

Nothing below this line is trustworthy until these exist. This is overwhelmingly
correctness and measurement work, and **it will make every reported number
worse.** That is the point: you cannot improve what you are measuring wrong.

| # | Item | Effort |
| --- | --- | --- |
| 1 | **Measurement substrate.** Schema versioning (`PRAGMA user_version` plus ordered migrations); an execution-quality ledger recording decision price, arrival quote, fill price and realised shortfall in bps per order; a per-bar decision log; `planned_risk` and `initial_stop_price` on trade rows so realised R can be compared with budgeted R. Strictly advisory in v1 — do not close any loop automatically. | Medium |
| 2 | **Fee-aware live P&L.** `_record_fill_delta` books `(price − entry) × delta` with no fee subtraction, so every trade row and dashboard P&L figure is gross. Net entry and exit fees exactly once, and deduct the entry fee from available notional when sizing. | Small |
| 3 | **Stop the walk-forward leaking.** The embargo defaults to 1 bar against feature windows of 224+ bars. Derive it from `warmup_bars()`, purge the training tail, and report a Deflated Sharpe (Bailey & López de Prado) against a persistent count of trials ever run. Report it; do not gate on it yet. | Medium |
| 4 | **Make the multiple-testing penalty actually do something.** Verified no-op: `selection_penalty = sqrt(2·ln(N)/observations)` is identical for every candidate within a fold (same test window ⇒ same equity-curve length), so it cancels in `max()` and cannot change which candidate is selected. | Small |
| 5 | **Risk coherence.** `enable_adaptive_risk` defaults `False` in the dataclass but `True` in the UI, so programmatic configs silently differ from dashboard ones. The drawdown throttle uses the session-loss limit as its ramp ceiling — two different risk measures. Equity sizing consumes Alpaca's margin-inflated `buying_power`. Add a minimum stop-distance floor: sizing is currently unbounded in its most degenerate-prone input. | Small |
| 6 | **Portfolio heat.** Cap the *sum of open risk* rather than the sum of notional. Today `3 × 0.35%` is an emergent product of `max_positions × risk_per_trade_pct`, never an enforced quantity. Measure against the current ratcheted stop so heat decays as stops tighten. | Small |
| 7 | **Selection integrity.** Replace the bare grid `argmax` — the single most noise-contaminated point on the surface — with a plateau score averaged over lattice neighbours. Keep the radius a hardcoded constant with a written rationale; exposing it in the sidebar trades one overfitting surface for another. | Small |
| 8 | **Equity session filter.** Extended-hours bars silently contaminate every intraday equity series. Filtering to regular session removes roughly a 1.57× systematic overstatement from intraday research results. Bump the config fingerprint so cached results are invalidated rather than silently mixed. | Medium |
| 9 | **Close the control plane.** The dashboard publishes Start, EMERGENCY STOP, kill-switch clear and the crypto LIVE unlock to anyone who resolves the domain. Add auth at the proxy — but never gate *stopping* more tightly than *starting*. | Medium |
| 10 | **`portfolio_market_value` is not crypto-filtered.** It sums every account position and is passed as `current_crypto_exposure`, so stock holdings throttle crypto sizing on a shared paper account. Conservative, so not dangerous — but wrong. | Small |

## Tier 2 — edge (only once you can tell whether it worked)

| # | Item | Note |
| --- | --- | --- |
| 1 | **Volatility-scale the momentum signals.** Score terms are dimensionally incommensurable today: `momentum × 4.0` is a raw fractional return competing with a normalised ADX, so cross-asset comparison is not meaningful. | Watch realised concentration — vol-scaling structurally tilts selection toward the lowest-vol asset, i.e. toward BTC. |
| 2 | **Bounded signal age with decay.** Entries are one-bar events: a breakout that fires while the portfolio is full is discarded rather than remembered. | **Must ship after the cost gate.** It increases turnover, so alone it would plausibly reduce net P&L while appearing to improve signal capture. |
| 3 | **Target portfolio volatility** instead of ranking it. Replace the ATR-percentile-rank throttle with EWMA realised vol and a genuine target. | Capped at 1.0 it can only reduce exposure, so in a sustained low-vol uptrend it costs return versus today; the no-leverage constraint removes the leg that normally recovers that. |
| 4 | **Validate the entry price and cap what an entry may pay.** Every live order is currently an unpriced market order; the stock engine has no quote access at all. | Do **not** build an exit escalation ladder — a limit exit that does not fill leaves a live long with its stop already cancelled. |
| 5 | **Report OOS results as distributions.** Stationary bootstrap over portfolio bar returns; report p05/p50/p95 of terminal return and drawdown. | Resample *rows*, never columns — per-symbol resampling destroys the cross-asset correlation that produces real crypto drawdowns. |
| 6 | **Shadow replay.** Re-run the decision logic over the logged bars and assert live matches model. For a deterministic system they must agree. | Depends on the decision log from tier 1. |

## Tier 3 — scale

Gated on preconditions rather than effort: cross-sectional machinery is pointless
at N=3 symbols and 3 slots, and correlation sophistication is cosmetic on a 3×3
matrix.

1. **Expand the universe with point-in-time eligibility**, then let cross-sectional
   selection actually bind. Point-in-time is non-negotiable: applying today's
   liquid Alpaca pairs to 2023 embeds survivorship bias that can manufacture an
   entirely fictitious edge.
2. **Persist bars** so vendor revisions become observable and the live loop stops
   refetching immutable history. A cache is a new correctness surface in the live
   path — build it only once you can detect when it is wrong.
3. **Corporate-action discontinuity detection** for equities.
4. **Data-health degradation** that blocks *entries* without blinding the engine.
   Never halt the engine itself: that also stops the stop watchdog and
   reconciliation during precisely the vendor incident that triggered it.
5. **Lock-gated online backup** of the SQLite database.

## Deliberately rejected

The most valuable half of an audit is what it refuses to build. Each of these was
proposed, considered, and rejected with reason.

| Proposal | Why not |
| --- | --- |
| Isotonic calibration of score → E[net R], conviction-scaled sizing | The largest new overfitting surface on the list, for a system whose primary risk is overfitting. A few hundred trades fits noise precisely. |
| Soft regime membership (softmax + dwell half-life) | Two new parameters that add tuning surface without adding information, and pays for smoothness with lag where lag is dangerous. |
| Beta-residualised relative strength | Correct in principle, unusable at N=3: the standard error on beta over 168 bars at ρ≈0.8 lets residual momentum flip sign from estimation noise alone. |
| Effective-number-of-bets floor, correlation size haircut | The eigen-decomposition of a 3×3 correlation matrix carries almost no information. Revisit after universe expansion. |
| De-risking open positions on circuit-breaker breach | Adds a cancel-and-replace sequence to the live path that executes *only* under stress — the least-tested code running at the worst moment. |
| TWAP/VWAP child-order slicing | Building it *is* the harm. Retail-size impact in a liquid name is ~zero, so slicing converts one spread crossing into N crossings. Instrument first; revisit only if measured slippage actually rises with order size. |
| ADV/participation caps on equities | Non-binding essentially always at retail size: two knobs that do nothing 99% of the time. |
| Square-root market-impact term in the simulators | `impact_coef` is a parameter this system cannot calibrate — a guess wearing a formula's clothes. |
| Crypto funding-rate / open-interest overlay | Introduces a second uncontracted vendor into a live-money path, with its own outages and symbol-mapping drift. |
| Combinatorial purged CV (CPCV/PBO) | Duplicates the purpose of the tier-2 bootstrap and tier-1 deflation at several times the cost. |
| Websocket order-event stream | The valuable half — a shorter sleep while exposed — is already captured by the position-open cadence, in five lines with no new failure surface. |
| Atomic `replace_order_by_id` for stop re-arms | The most architecturally satisfying item and the lowest priority: a few hundred milliseconds, bought with new behaviour in the most safety-critical routine. |
| Log/equity retention pruning with VACUUM | A `DELETE` against the only record of this system's P&L. Pruning `processed_bars` is specifically dangerous — those rows *are* the duplicate-execution guard. |
| One-at-a-time sensitivity sweep over risk parameters | The report itself is the risk: an operator who sees 1.5× risk produced higher OOS return will type it into the sidebar. |
| Auto-scaling risk from realised R; drawdown-triggered auto-derisk | A feedback controller fitted to ~30 noisy samples. It shrinks risk after an ordinary losing streak and fails to restore it, which is how systematic books quietly die. |

## Sequencing

Build order differs from tier order in two places, and both matter.

1. The watchdog fix — a live defect in the only real-money engine. **Shipped.**
2. Schema versioning, because every remaining tier-1 item adds columns.
3. Measurement substrate and the benchmark gate in parallel. **Benchmark gate shipped.**
4. The cost model, which wants measured shortfall to set `max_cost_to_risk` from
   data rather than a guess. **Formula and gate shipped; calibration pending (3).**
5. Risk coherence, heat cap, selection integrity, session filter, control plane —
   mutually independent.

Three ordering constraints must not be violated:

- **Signal persistence/decay must come after the cost gate**, or turnover rises
  while net P&L falls and the two effects get attributed to each other.
- **The correlation haircut must come after the portfolio heat cap**, because it
  converts rejects into admissions and needs a backstop.
- **Shadow replay depends on the decision log** from tier 1.

Two expectations to set before starting. First, tier 1 will make every reported
number worse, and there will be pressure to revert or to shop the gates — that
pressure is the signal the work is correct. Second, the cost gate may return the
answer that this strategy cannot be traded on hourly bars at 75bps round trip.

The project's own documents decide one tie for you: `SMART_STRATEGY_PLAN.md`
commits the equity workspace to paper-only permanently, so crypto items win every
tie on real-money impact.

## What this does not change

No item here makes any strategy profitable, and none of it is investment advice.
Walk-forward PASS still means only "eligible for controlled paper testing." The
application still never copies a candidate into a running engine or unlocks live
trading. The measurement work above makes it *possible* to tell whether an edge
exists — it does not create one.
