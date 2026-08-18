# Adaptive Intraday v2 — Production Path

`crypto_intraday_adaptive` v2.0.0 is the preferred high-velocity research path. The
legacy v1 momentum strategy remains isolated for comparison.

The objective is many **independent, positive-expectancy** intraday opportunities.
The requested 2%/day figure is an untouched-OOS diagnostic only; it is never an
optimizer or lifecycle-promotion target.

## Alpha and execution architecture

Completed 5-minute or 15-minute bars feed four causal sleeves:

1. cross-sectional momentum continuation;
2. prior-range / volatility-expansion breakout;
3. trend pullback continuation;
4. range mean reversion with reversal confirmation.

A causal cross-sectional router estimates bull/range/stress/risk-off conditions from
the benchmark, market breadth and volatility. Stress/risk-off can block entries.

Immediately before entry, Quant Desk separately applies quote freshness/spread,
measured or assumed round-trip costs, statistical expectancy confidence, adverse
depth, regime/abstention, current-book correlation, portfolio heat/exposure,
lifecycle risk scaling and the audited broker sizing/reconciliation/stop controls.
Favorable order-book depth never creates alpha; adverse depth can only reduce
execution quality.

Adaptive sleeves also have **negative-only forward supervision**. After at least 40
current-lifecycle real trades for a sleeve, deteriorating expectancy/PF can reduce its
confidence; a statistically negative sleeve can be suspended. A recent winner never
receives more than the baseline 1x confidence.

A new lifecycle rung may temporarily bridge only the immediately preceding qualified
rung so PAPER/CANARY can start from zero trades. After 40 current-rung real trades,
current-rung evidence becomes authoritative. Arbitrary older evidence is not pooled
into v2 entry confidence.

Quotes refresh every five seconds while flat and every one second while exposed.
OHLCV history is cached until the next safely completed bar. Quote/depth telemetry is
non-blocking and cannot interrupt protective execution.

## 1. Real-data research

Default run:

```bash
python scripts/run_adaptive_intraday_research.py --days 120
```

Before return research, the runner screens the active BASE/USD discovery pool by
current executability only:

- BTC/USD must be eligible;
- quote age <= 120 seconds;
- spread <= 40 bps;
- at least $250 top-of-book notional on each side;
- select up to 12 eligible pairs, with at least 6 required.

Historical return never chooses the symbols. Current selected-pair spread also enters
the research cost model: observed 75th-percentile spread becomes a floor under the
configured spread assumption. An explicit `--symbols` tuple is allowed, but its exact
order becomes part of the research identity.

The bounded normal grid contains 18 structural hypotheses. Selection is training-only
inside chronological walk-forward folds, with non-overlapping OOS windows and an
embargo. Historical trial count is accumulated into the deflated-Sharpe correction.

A PASS requires all gates, including:

- >=8 OOS folds and >=300 OOS trades;
- >=65% positive OOS folds;
- positive compounded OOS and benchmark-excess return;
- Sharpe >=0.85 and PF >=1.18;
- worst OOS-fold drawdown <=8%;
- positive stressed return under worse costs;
- deflated-Sharpe probability >=0.80;
- Monte Carlo ruin probability <=1%;
- >=6 and <=50 OOS closed trades/day;
- OOS expectancy >=+0.03R/trade;
- broad symbol participation;
- <=55% positive-PnL concentration in one symbol;
- <=45% positive-return concentration in one OOS fold.

The report separately prints geometric OOS return/day and whether 2%/day was observed.
That diagnostic cannot make a failed strategy pass.

Every recorded adaptive research run also gets an immutable manifest containing:

- full 40-character git commit;
- strategy ID and version;
- timeframe;
- exact ordered universe;
- exact selected adaptive parameters;
- research run ID.

If the git commit cannot be resolved, adaptive research refuses to record deployable
evidence.

## 2. RESEARCH -> SHADOW locks the complete strategy identity

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_adaptive

python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_adaptive
```

A passing research row **and** its matching manifest are mandatory. Promotion locks
the complete manifest, not only the symbols.

Every later SHADOW/PAPER/LIVE runtime verifies the checked-out Git SHA and reconstructs
the exact researched timeframe, symbol order and `AdaptiveIntradayConfig` from that
manifest. Runtime `--symbols` and `--timeframe` arguments are assertions only; they
cannot override the researched strategy. A code/config mismatch fails closed.

## 3. SHADOW forward evidence

```bash
python scripts/run_adaptive_intraday_runtime.py --mode shadow
```

SHADOW gets current Alpaca bars/quotes but submits no broker orders. PAPER requires
current-SHADOW evidence:

- >=100 SHADOW trades;
- expectancy >=+0.03R/trade;
- PF >=1.10;
- max R-drawdown <=5R.

Then run the same lifecycle promotion command again.

## 4. PAPER execution

```bash
python scripts/run_adaptive_intraday_runtime.py --mode paper
```

PAPER uses `data/adaptive_intraday_trading.db`. Real exits feed back into Quant Desk.
Automatic degradation can reduce risk, demote to SHADOW or quarantine the strategy.

LIVE_CANARY eligibility requires:

- >=200 current-PAPER trades;
- expectancy >=+0.03R/trade;
- PF >=1.10;
- drawdown <=6R;
- >=50 measured execution samples;
- zero open operational defects;
- a new passing measured-cost research run of the **exact same manifest strategy**.

## 5. Pre-CANARY measured-cost verification — no re-optimization

Inspect PAPER execution quality:

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_adaptive
```

Then rerun the locked strategy, for example:

```bash
python scripts/run_adaptive_intraday_research.py \
  --locked-config \
  --days 120 \
  --slippage-bps 14.2 \
  --spread-bps <CONSERVATIVE_OBSERVED_SPREAD> \
  --fee-bps 25
```

`--locked-config` loads the promoted manifest and evaluates **one exact parameter
set**. It does not search the 18-hypothesis grid. It also requires the current Git SHA
to equal the locked research SHA. Cost assumptions and the later data window may
change; code, version, timeframe, universe/order and parameters may not.

Stress costs remain at least 1.5x base assumptions and never below conservative
floors. PAPER -> LIVE_CANARY refuses the rerun if its modeled slippage is below
measured PAPER slippage or if its manifest is not the exact promoted strategy.

## 6. PAPER -> LIVE_CANARY

```bash
python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_adaptive \
  --measured-research-run-id <PASSING_LOCKED_CONFIG_RUN_ID> \
  --operational-defects 0 \
  --live-ack I_ACCEPT_LIVE_RISK
```

CANARY gets 25% of the strategy risk budget. From the 0.15% base budget, that is at
most **0.0375% equity risk/trade** before correlation, drawdown, volatility and
portfolio reductions.

## 7. Guarded LIVE canary

After separate live credentials and the independent live-deployment environment gate
are configured:

```bash
python scripts/run_adaptive_intraday_runtime.py \
  --mode live \
  --live-ack I_ACCEPT_LIVE_RISK
```

LIVE uses `data/adaptive_intraday_live_trading.db`. The base crypto LIVE config cannot
mint the Quant-controlled process-local authorization, so the legacy/base engine
cannot bypass this path. Direct intraday engine construction also enforces the strict
high-velocity lifecycle requirements and rejects weaker lifecycle managers.

## 8. LIVE_CANARY -> LIVE

Full LIVE requires another explicit acknowledgement plus current-CANARY evidence:

- >=100 real canary trades;
- expectancy >=+0.01R/trade;
- PF >=1.05;
- drawdown <=4R.

Continuous degradation monitoring remains active after promotion.

## Iteration policy

A failed run remains failed. Do not widen parameters repeatedly on the same sample
until a desired percentage appears. A materially new alpha idea gets a new
hypothesis/version, cumulative trial accounting and the same untouched-OOS and
forward lifecycle ladder.

The production loop is:

**liquidity selection -> causal hypothesis -> chronological research -> stressed OOS
-> immutable manifest -> SHADOW -> PAPER -> fixed-config measured-cost rerun ->
CANARY -> LIVE -> continuous strategy/sleeve degradation monitoring**.

This system is designed to maximize the chance that an observed edge survives real
execution. It is not designed to manufacture a backtest that says 2%/day.
