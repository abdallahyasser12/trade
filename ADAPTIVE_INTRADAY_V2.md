# Adaptive Intraday v2 — Production Path

`crypto_intraday_adaptive` v2.0.0 is the preferred high-velocity research path.
The older `crypto_intraday_momentum` v1 remains available for comparison, but its
research, lifecycle state and execution database are isolated from v2.

The objective is many **independent, positive-expectancy** intraday opportunities.
The requested 2%/day number is an untouched-OOS diagnostic only; it is never an
optimizer or lifecycle promotion target.

## Decision architecture

Completed 5-minute or 15-minute bars feed four causal sleeves:

1. cross-sectional momentum continuation;
2. prior-range / volatility-expansion breakout;
3. trend pullback continuation;
4. range mean reversion with reversal confirmation.

A cross-sectional router estimates bull/range/stress/risk-off conditions from the
benchmark, market breadth and volatility. Stress/risk-off can block entries. The
bar-driven alpha hypothesis never uses future data or live order-book state.

Immediately before an entry, Quant Desk independently applies current quote
freshness/spread, measured or assumed round-trip costs, statistical expectancy
confidence, adverse-depth penalty, regime/abstention, current-book correlation,
portfolio heat/exposure, lifecycle risk scaling and the original broker sizing,
idempotency, reconciliation and protective-stop controls. Favorable order-book depth
never creates alpha; adverse depth can only reduce execution quality.

Quotes are refreshed every five seconds while flat and every one second while
exposed. OHLCV history is cached until the next safely completed bar. Fresh quote and
depth telemetry is persisted separately; telemetry failure cannot break broker
protection.

## 1. Select executable symbols and run real-data research

Default research uses 120 calendar days of 5-minute Alpaca data:

```bash
python scripts/run_adaptive_intraday_research.py --days 120
```

Before downloading return history, the runner screens the active BASE/USD discovery
pool using **current executability only**:

- BTC/USD must be eligible;
- quote age <= 120 seconds;
- spread <= 40 bps;
- at least $250 top-of-book notional on each side;
- the 12 highest-quality eligible pairs are selected, with at least 6 required.

The selector does not use historical return to choose symbols. Current selected-pair
spread is also incorporated into the modeled research cost: the 75th percentile of
observed selected spreads becomes a floor under `--spread-bps`.

If `--symbols` is supplied, that exact 6-12-symbol tuple is the research universe and
is stored with the run. For measured-cost reruns, provide cost assumptions at least
as adverse as observed PAPER execution.

The bounded research grid contains 18 structural hypotheses. Candidate selection is
training-only inside chronological walk-forward folds; OOS folds do not overlap and
are separated by an embargo. Historical trial count is accumulated into the
deflated-Sharpe correction.

A research PASS requires all gates, including:

- at least 8 OOS folds and 300 OOS trades;
- at least 65% positive OOS folds;
- positive compounded OOS and benchmark-excess return;
- aggregate Sharpe >= 0.85 and profit factor >= 1.18;
- worst OOS-fold drawdown <= 8%;
- positive return under harsher slippage/spread/fees;
- deflated-Sharpe probability >= 0.80;
- Monte Carlo ruin probability <= 1%;
- at least 6 and no more than 50 OOS closed trades/day;
- OOS expectancy >= +0.03R/trade;
- broad symbol participation;
- no one symbol contributing >55% of positive symbol PnL;
- no one OOS fold contributing >45% of positive fold return.

The output separately reports geometric OOS return/day and whether 2%/day was
observed. A result can never pass merely because that number appeared.

## 2. Promote RESEARCH -> SHADOW and lock the universe

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_adaptive

python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_adaptive
```

A passing run is mandatory. When promotion succeeds, the exact researched symbol
tuple **including order** is locked in Quant Desk. This matters because deterministic
cross-sectional tie-breaking uses configured symbol order.

Every SHADOW/PAPER/LIVE startup then loads that lock. `--symbols` on a runtime is only
an assertion and must exactly match it. A missing or different universe fails closed.
Changing the universe requires new research and explicit lifecycle adoption; the bot
cannot silently trade assets it was not validated on.

## 3. Continuous SHADOW forward test

```bash
python scripts/run_adaptive_intraday_runtime.py --mode shadow
```

SHADOW receives current Alpaca bars and quotes but submits no broker orders. Only
forward attributions recorded since SHADOW began count. PAPER requires:

- >=100 SHADOW trades;
- expectancy >= +0.03R/trade;
- profit factor >=1.10;
- max R-drawdown <=5R.

Then run the lifecycle `promote` command again.

## 4. PAPER execution

```bash
python scripts/run_adaptive_intraday_runtime.py --mode paper
```

PAPER uses `data/adaptive_intraday_trading.db`. Every real exit is attributed back to
Quant Desk. Degradation monitoring can reduce risk, demote to SHADOW or quarantine
the strategy when expectancy, PF, drawdown or execution quality deteriorates.

LIVE_CANARY eligibility requires current-state evidence only:

- >=200 PAPER trades;
- expectancy >= +0.03R/trade;
- PF >=1.10;
- drawdown <=6R;
- >=50 measured execution samples;
- zero open operational defects;
- a new passing measured-cost research run using the **same locked universe**.

## 5. Re-run with measured execution costs

Inspect current state and execution quality:

```bash
python scripts/manage_strategy_lifecycle.py status \
  --strategy crypto_intraday_adaptive
```

If average measured slippage is 14.2 bps, for example:

```bash
python scripts/run_adaptive_intraday_research.py \
  --days 120 \
  --symbols <THE_LOCKED_SYMBOLS_IN_EXACT_ORDER> \
  --slippage-bps 14.2 \
  --spread-bps <CONSERVATIVE_OBSERVED_SPREAD> \
  --fee-bps 25
```

Stress costs are automatically at least 1.5x base assumptions and never below the
configured conservative floors. PAPER -> LIVE_CANARY additionally refuses a research
run whose modeled slippage is below measured PAPER slippage or whose universe differs
from the deployment lock.

## 6. PAPER -> LIVE_CANARY

```bash
python scripts/manage_strategy_lifecycle.py promote \
  --strategy crypto_intraday_adaptive \
  --measured-research-run-id <PASSING_RUN_ID> \
  --operational-defects 0 \
  --live-ack I_ACCEPT_LIVE_RISK
```

Canary risk is 25% of the strategy risk budget. From the default 0.15% base budget,
that is at most **0.0375% equity risk/trade** before correlation, drawdown,
volatility and portfolio reductions.

## 7. Guarded LIVE canary

After configuring separate Alpaca live credentials and the repository's independent
live-deployment gate:

```bash
python scripts/run_adaptive_intraday_runtime.py \
  --mode live \
  --live-ack I_ACCEPT_LIVE_RISK
```

LIVE uses `data/adaptive_intraday_live_trading.db`. The base crypto LIVE configuration
cannot mint the Quant-controlled process-local authorization, so the legacy UI/base
engine cannot bypass this path.

## 8. LIVE_CANARY -> LIVE

Full LIVE requires another explicit acknowledgement plus current-canary evidence:

- >=100 real canary trades;
- expectancy >= +0.01R/trade;
- PF >=1.05;
- drawdown <=4R.

Automatic health supervision remains active after promotion.

## What iteration means

A failed run remains a failed hypothesis. Do not widen parameter ranges on the same
sample until a desired percentage appears. A materially new signal idea receives a
new hypothesis/version, cumulative trial accounting and the same untouched-OOS and
forward lifecycle ladder.

The production iteration loop is:

**liquidity selection -> causal hypothesis -> chronological research -> stressed OOS
-> locked-universe SHADOW -> PAPER -> measured-cost rerun -> CANARY -> LIVE ->
continuous degradation monitoring**.

The system is designed to maximize the chance that an observed edge is real, not to
manufacture a backtest that says 2%/day.
