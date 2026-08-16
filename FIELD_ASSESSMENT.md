# Field assessment — 15 August 2026

Ten completed out-of-sample experiments against real Alpaca market data, run
through promotion gates that were never loosened.

Hosted report: https://claude.ai/code/artifact/ea38f266-e60c-42eb-9af9-12b874dfa2b1

## Verdict

**Not elite — yet, but no longer edgeless.** The engineering is close to
professional grade and in two places ahead of it. Seven configurations of the
breakout/ensemble signal lost money; experiment 7 removed the cost excuse and the
sample-size excuse simultaneously and still lost 9 folds out of 9. Experiment 8
changed the hypothesis rather than the parameters and produced the first positive
expectancy at a real sample size — profit factor 1.48, Sharpe 0.52, positive under
stress — while still failing promotion because it deploys almost no capital and so
loses badly to simply holding BTC.

Engineering quality and strategy quality are separate findings; conflating them is
the mistake this document exists to prevent.

## Evidence

All figures out-of-sample, walk-forward, net of 75 bps round-trip friction.

| # | Bars | Engine | Universe | Folds | Trades | Strategy | Buy & hold | Sharpe | PF |
|---|------|--------|----------|------:|-------:|---------:|-----------:|-------:|-----:|
| 1 | 1Hour | base | 2 symbols | 31 | 88 | −6.86% | +272.14% | −2.35 | 0.26 |
| 2 | 1Day | ensemble | 2 symbols | 5 | 2 | +0.34% | −36.57% | 0.70 | 3.32 |
| 3 | 1Day | ensemble | 8 symbols | 5 | 2 | −0.16% | −36.57% | −0.93 | 0.00 |
| 4 | 1Day | base | 12 symbols | 3 | 1 | −0.30% | −44.61% | −1.11 | 0.00 |
| 5 | 1Day | base | 9 symbols | 2 | 4 | −0.07% | +19.80% | −0.14 | 0.82 |
| 6 | 4Hour | base | 8 symbols | 9 | 82 | −2.48% | +207.71% | −0.40 | 0.76 |
| 7 | 4Hour | ensemble | 8 symbols | 9 | 164 | −11.50% | +207.71% | −1.80 | 0.35 |
| 8 | 1Day | **cross** | 9 symbols | 8 | 96 | **+2.01%** | +112.48% | **+0.52** | **1.48** |
| 9 | 1Day | council | 9 symbols | 8 | 85 | +1.35% | +112.48% | +0.24 | 1.19 |
| 10 | 1Day | breadth | 9 symbols | 8 | — | +0.36% | +112.48% | +0.07 | 1.07 |

Runs 2 and 4 appear to "beat the benchmark" only because the strategy sat in cash
through a crash. With 1–2 trades they carry no statistical weight, and the
promotion gate rejected them on exactly that basis.

**Experiment 6 settles it.** 82 trades, 9 leakage-free folds, ~4.1 years, on the
timeframe where friction is cheapest (0.07R per trade). Profit factor 0.76.

**The gradient is the finding.** As friction fell from 0.46R per trade (hourly) to
0.07R (4-hour), profit factor climbed 0.26 → 0.76 and stopped short of 1.0. Costs
were a real drag and are now largely removed. What remains underneath is a signal
with negative expectancy. The entries are not predictive.

**Experiment 7 closes the cost argument for good.** 164 trades across 9 folds at
0.13R per trade — friction is no longer a plausible explanation for anything. The
break-even win rate at 2R is 37.8%, comfortably inside the 35–45% that trend
systems achieve. Both defects previously diagnosed are now fixed: trade frequency
went from 2 to 164, and cost per unit of risk fell by an order of magnitude. The
strategy still lost in **9 folds out of 9**.

The fold-by-fold shape is the real evidence, more than any aggregate:

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|
| Strategy | −1.33% | −1.09% | −2.03% | −1.28% | −1.26% | −0.26% | −1.32% | −2.11% | −1.44% |
| Benchmark | −15.95% | +57.22% | +30.75% | +83.26% | −4.66% | +38.07% | +30.21% | −38.11% | −8.38% |

Nine folds, a tight band of small losses, across markets ranging from +83% to
−38%. A strategy with a real but noisy edge scatters around zero and correlates
with something. This one bleeds a near-constant amount independent of what the
market does — the signature of a zero-expectancy signal paying friction, not of an
edge that needs better conditions. No amount of further tuning of this signal
family is indicated. That is why the next hypothesis is structurally different
rather than a reparameterisation.

## Predicted real-world behaviour

Extrapolated from experiments 6 and 7 — a central expectation with wide error bars.

**Paper trading**
- ~20 trades/year across 8 symbols; long quiet stretches are normal.
- −0.5% to −1%/year at modelled costs; ~−1.8%/year stressed.
- Shallow drawdowns; worst observed fold −1.46%.
- ~1/3 of periods green — enough to sustain false hope indefinitely.

Experiment 7 tightens this and makes it worse. At 164 trades over 9 folds the rate
is closer to ~40 trades/year, the annualised drag is roughly −2.5%/year at modelled
costs and −3.4%/year stressed, and **zero** of nine periods were green. The wider
sample removed the "~1/3 green" consolation: that figure came from small-sample
noise, not from the strategy having good stretches.

**Live money**
- Expect worse than paper. No execution-quality ledger exists, so the gap between
  assumed 10 bps slippage and reality is currently unmeasurable.
- The opportunity cost dominates: −2.5% versus +207.7% for holding BTC.
- It will not blow up. Sizing, session-loss latch, heat cap, correlation gate and
  hard live notional caps are enforced and tested.

**Failure modes**
- The seductive flat period: sitting in cash during a crash looks like skill.
- Untested market stress: no liquidity event, venue outage or unfilled stop-limit
  appears in any experiment.
- Data-window bias: Alpaca crypto history starts 2021-01-01. One cycle is not a sample.

## What is genuinely elite

- **It cannot fool itself about costs.** One friction formula feeds simulator and
  live engine, expressed in units of the risk each trade takes.
- **Its gates do not bend.** Eight runs, eight honest verdicts, zero adjustments —
  including the one that finally showed positive expectancy and was still rejected.

That machinery is the durable asset — it is what makes the negative result
trustworthy, and what any future strategy would be judged by.

## What would have to change

Every lever the codebase exposes was pulled against the original signal — three
timeframes, four universe sizes, both engines, two grid scales, a corrected
embargo — and they converged on failure. Item 1 has since been done, and changed
the picture.

1. ~~A different signal family, through the same unmodified gates.~~ **Done —
   experiment 8.** It produced the first positive expectancy. It still fails on
   deployment, not on edge, which is a different and more tractable problem.
2. Build the execution-quality ledger first; until realized slippage is measured,
   no live result can be attributed to signal versus execution.
3. Or stop: keep the measurement machine, run the ensemble in paper as a
   capital-preservation overlay, and stop paying the opportunity cost.

### Item 1 — what was built

`cross_sectional.py` is the different signal family, chosen to attack the two
diagnosed failures rather than to reparameterise around them:

- **Relative, not absolute.** Rather than asking each symbol "are ten conditions
  true at once?", it ranks the universe at each timestamp and holds the leaders.
  Something is always ranked first, so trade frequency is a property of `top_k`
  rather than an accident of filter alignment. Measured: 54 trades where the
  conjunction produced 2.
- **Dimensionally coherent.** `(close/close[-L] − 1) / (σ·√L)` is unitless, so a
  move in DOGE and a move in BTC are actually comparable. The old score summed a
  raw fractional return with a normalised ADX and therefore ranked by volatility.
- **Asymmetric turnover.** Exit requires falling `top_k + buffer` places, because
  turnover is the one cost lever repeatedly shown to matter.
- **Six parameters, not fifteen.** Every parameter is overfitting surface.

It runs through the same unmodified simulator, cost gate, benchmark comparison and
promotion criteria — only the decision layer differs, injected into the audited
code objects so the mechanics stay byte-for-byte identical.

Two things learned while wiring it up, both worth recording because both would
have produced a dishonest test:

- The engine was initially pointed at the breakout parameter grid, which varies
  keys it never reads. All candidates per fold were behaviourally identical, so
  the argmax was decided by tie-break order and the reported candidate count sized
  a multiple-testing penalty for a search that never happened. The grid now varies
  what the engine actually reads.
- With the benchmark-regime gate pinned on, a training window inside the 2022 bear
  market gives every candidate zero trades, so no candidate can be selected and
  the run aborts. Skipping such folds would have biased `positive_fold_ratio` to
  bull folds only — the strategy would score better the more often it declined to
  trade. The gate is therefore searched rather than assumed.

### Experiment 8 — the first positive expectancy

The cross-sectional engine still **FAILS** promotion, and correctly so. But it
fails for an entirely different reason than everything before it, and that
difference is the most useful thing measured so far.

| | Experiment 7 (ensemble) | Experiment 8 (cross-sectional) |
|---|---|---|
| Profit factor | 0.35 | **1.48** |
| Aggregate Sharpe | −1.80 | **+0.52** |
| Positive folds | 0% | 50% |
| Stressed return | −15.74% | **+1.11%** |
| Compounded return | −11.50% | **+2.01%** |

Eight leakage-free folds, 96 out-of-sample trades, 24 candidates searched per
fold. This is the first configuration at a meaningful sample size that makes
money rather than bleeding, and it stays positive under the stress run (35 bps
slippage, 50 bps fee, 20 bps spread). The hypothesis change did what changing the
parameters never did.

It fails on three gates, all of them fair:

- positive-fold ratio 0.50 < 0.60 — coin-flip territory
- excess return −1.35% < 0
- return versus buy-and-hold −110.47% (+2.01% strategy against +112.48% BTC)

**The parameter selections are stable, which is weak but real evidence.** Across
eight independent folds the search chose `lookback=30` seven times and a 4× ATR
stop seven times. Independent windows converging on the same values is what a
genuine optimum looks like; noise-fitting scatters. The regime gate did not
converge — off in five folds, on in three — so that filter is not reliably
earning its place and should be regarded as unresolved rather than validated.

**The binding constraint is now exposure, not signal.** Sizing is risk-based:
position notional = `risk_per_trade_pct` ÷ stop distance. At the default 0.35%
risk and the 4× ATR stop the search preferred, a daily crypto stop sits roughly
6–20% from entry, so each position is about 1.8–5.8% of equity. Three positions
means the book is **roughly 82–95% cash at all times**. A portfolio that is ~90%
in cash cannot track a benchmark that rose 112%, no matter how good its entries
are.

That reframes the problem. Every prior experiment failed because the signal had
no edge; this one has positive expectancy and simply deploys almost no capital
against it. Sizing is a far more tractable problem than edge.

Three caveats, stated because they are what would make this wrong:

1. **The search burden is real.** Sharpe 0.52 was selected from 24 candidates per
   fold. The selection penalty at 96 observations is roughly 0.26, so a deflated
   figure lands near 0.26 — positive, but weak. This is suggestive, not
   established.
2. **Scaling exposure amplifies whatever is there, including noise.** Raising
   risk per trade does not create edge. If the 1.48 profit factor is a small-sample
   artefact, larger positions turn a small drift into a large one in whichever
   direction the artefact points.
3. **Eight folds over one crypto cycle.** Alpaca history starts 2021-01-01. This
   has not been tested across an independent market epoch, and cannot be.

### Experiments 9-10 — the council does not beat its best member

A four-member council was built to test whether combining decorrelated
strategies produces better decisions than any one of them. Members: the
cross-sectional ranker, short-horizon mean reversion, a volatility-regime
state model, and universe breadth. Measured pairwise BUY-signal correlations
ran between -0.01 and +0.03, so the decorrelation was real.

Run against identical universe, period, folds and gates:

| engine | Sharpe | PF | return | positive folds | stressed |
|---|---:|---:|---:|---:|---:|
| **cross alone** | **0.52** | **1.48** | **+2.01%** | 50% | **+1.11%** |
| council of four | 0.24 | 1.19 | +1.35% | 50% | +0.21% |
| breadth alone | 0.07 | 1.07 | +0.36% | 25% | -0.76% |

The council beats breadth on every measure and loses to cross-sectional on
every measure. Since the test was whether it beats its *best* member, it
fails: combining four decorrelated strategies produced a worse book than the
single best one, at four times the compute and five times the parameters.

Averaging is the reason, and it is not a defect in the implementation. A
weighted combination pulls the best member's conviction toward the mean of a
group whose other members are weaker. Decorrelation guarantees the members
rarely reinforce each other, so what the council mostly does is dilute. That
is a general result about combining strategies of unequal quality, not a
tuning problem, and no amount of reweighting escapes it while three of four
members are worse than the fourth.

Three design findings from building it are worth keeping even though the
council is not:

1. **Unanimity is unreachable, not merely strict.** BUY rates of 15.6%, 22.9%,
   0.76% and 0.40% at near-zero correlation multiply to about one in a
   million. Unanimous mode took exactly 0 trades end to end.
2. **A single-member veto is the most aggressive setting available, not the
   most conservative.** P(at least one of four decorrelated members issues
   EXIT on a held symbol) is 94.4%. Entry needed two members to agree, exit
   needed one to dissent, and the book churned at 198 trades against its best
   member's 69 with profit factor collapsing 3.49 -> 0.51.
3. **Synthetic data was actively misleading here.** On a regime-switching
   generator, breadth scored Sharpe 2.23 and looked like the best member by a
   wide margin. On real data it scored 0.07. The generator's regime structure
   was exactly what breadth detects, which made the test circular. Only the
   structural findings above survived contact with real data.

### The sizing question remains the live one

The correct next experiment is therefore a sizing sweep at fixed signal — vary
`risk_per_trade_pct` and `max_positions` while holding the selected momentum
parameters constant — to establish whether risk-adjusted performance survives
higher deployment or was an artefact of trading tiny. That question is answerable
with the machinery already built, through the same unmodified gates.

## Limits

Walk-forward out-of-sample on Alpaca data from 2021-01-01 — one market cycle.
Experiment 7 was still computing at publication. Nothing here is investment advice.
A promotion PASS would have meant only "eligible for supervised paper testing".
The system has never traded real money, and nothing here recommends that it should.
