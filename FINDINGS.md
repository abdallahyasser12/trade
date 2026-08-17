# Confirmed defects, not yet fixed

Every entry below was reproduced by reading the named line, not inferred
from a filename or a doc. Entries that were fixed are not listed — they
are in the git history with their regression tests.

Two things this file is for: so the next session does not re-derive the
same findings, and so a promoted strategy is never scored by a component
whose known defects nobody wrote down.

Status vocabulary:

* **LIVE** — affects the running paper bot today.
* **LATENT** — real defect in code that currently has no runtime caller.
  Harmless now, a bug the moment that code is wired.
* **RESEARCH** — affects research verdicts, not execution.

---

## CRITICAL

### Execution costs are double-counted in the opportunity score
`quant/scanner.py:299` · RESEARCH (LATENT — scanner has no runtime caller)

```python
gross_alpha_bps = health.expectancy_r * distance_bps
```

`expectancy_r` is the mean `r_multiple` over attribution rows
(`quant/marketplace.py:130`), and `r_multiple = net / risk` where
`net = gross - fees` (`quant/journal.py:248`). So the figure labelled
*gross* alpha is already net of realized fees, and `quant/execution.py`
then subtracts an estimated cost on top.

Direction is conservative — it understates net alpha, so the gate rejects
rather than accepts — but the ranking is not the quantity it claims to be,
and a scanner wired in this state would reject almost everything for the
wrong reason.

Not fixed here because the correct repair is a decision, not a patch: the
realized figure nets fees but *not* spread or slippage
(`realized_slippage_bps` is always `None`, and stock fees are 0), so
"subtract only the incremental cost" needs the execution engine to expose
a fee-excluded estimate. Guessing at that accounting risks being wrong in
the permissive direction, which is worse than being wrong conservatively.

### Feature-store restatement destroys the prior vintage
`quant/store.py:604` · LATENT

The features upsert overwrites `available_at` on the existing primary key.
After a restatement, an as-of query for a moment *before* the revision
returns the revision — which is precisely the lookahead the point-in-time
store exists to prevent.

Latent only because `QuantStore.record_features` has no caller outside
tests (grep confirms: the definition and two test call sites). Must be
fixed before anything writes features.

### `available_at` defaults to the observation timestamp
`quant/store.py:652` · LATENT

```python
available = available_at or timestamp
```

A bar stamped `T` is only knowable at `T + duration`, so the default
publishes a bar's features at the bar's own open. An API footgun rather
than an active lookahead, for the same reason as above — nothing calls it.

*(An earlier audit pass reported this as CRITICAL and active at
`store.py:597`. Both the anchor and the severity were wrong; corrected
here.)*

---

## HIGH

### The registry cannot construct half its strategies
`quant/strategies/registry.py:104` · LATENT

`StrategyFactory` is typed `Callable[[CryptoStrategyConfig], ...]`, but
the intraday family takes `IntradayStrategyConfig` subclasses. Measured
directly:

```
registry.create("donchian_trend", CryptoStrategyConfig)      -> OK
registry.create("volatility_squeeze", CryptoStrategyConfig)  -> OK
registry.create("vwap_reversion", ...)      -> ValueError, needs its own
registry.create("opening_range_breakout", ...) -> ValueError, needs its own
```

Two raise; the other two are arguably worse, silently accepting a config
they were never designed for because they lack the `isinstance` guard the
other two have (`quant/strategies/donchian_trend.py:114`,
`volatility_squeeze.py:109`). The registry is unusable as a uniform
interface, which is why it has zero call sites.

### Deflated Sharpe assumes normality it was built to avoid
`quant/research/factory.py:405` · RESEARCH

`deflated_sharpe_probability` is called without `skewness`/`kurtosis`, so
it uses the Gaussian defaults. Stop-based trend returns are negatively
skewed and fat-tailed; assuming normality inflates `dsr_probability` and
makes `min_deflated_sharpe_probability` easier than designed. Fixing it
tightens the gate, which is the safe direction.

### The "conservative" stress haircut is the mildest one available
`quant/research/stress.py:95` · RESEARCH

The stressed block demotes the **smallest** winners to -1R while the
module docstring calls this the conservative choice. Demoting the smallest
preserves the entire tail that carries the expectancy, so `risk_of_ruin`
is understated and `max_ruin_probability` is easier to pass than the
documentation claims.

### The research exit stack does not exist in any deployed engine
`quant/exits.py:174` · RESEARCH

`TradeManagementPolicy` and the RUNNER / partial-profit / break-even /
time-stop layer is imported only by the simulator, the research CLIs and
tests. Both research CLIs run with `enable_runner=True`, so every promoted
number describes an exit stack no engine implements — the simulator's own
docstring calls itself "the research twin of the production engines'
execution semantics".

### The embargo is nominal
`quant/research/factory.py:301` · RESEARCH

`oos_start = test_start - (selected_warmup - 1)`. With the default
`embargo_bars=1` and a warmup of 116 bars, roughly 115 training bars are
pulled into the out-of-sample slice, where `prepare_universe` recomputes
indicators over train and test together. Not future lookahead, but the
train → embargo → test separation is not what the documentation describes.

### The stock engine never writes the executions ledger
`trade_engine.py:1065` · LIVE

`store.record_execution` is called from exactly one place in the repo
(`crypto_engine.py:798`). Stock fills reach `record_trade` and nothing
else, so execution calibration can never measure equity slippage and the
PAPER → LIVE_CANARY rung's measured-cost evidence is unobtainable for
equities.

### A failing protective stop has no escalation path
`trade_engine.py:1514` · LIVE

A persistently failing stop submit is logged at WARNING and the function
returns, leaving a live position unprotected with nothing that escalates,
halts, or arms the kill switch.

### No HTTP timeout on any market-data call
`data_feed.py:331` · LIVE

alpaca-py issues the request with no timeout, so a hung socket blocks the
engine thread indefinitely. Related: `with_retries` uses the engine stop
event for interruptible backoff, but alpaca-py independently retries
429/504 with a bare `time.sleep`, so the interruptible-backoff guarantee
is only half true (`data_feed.py:70`).

### A malformed persisted position crashes reconciliation
`trade_engine.py:459` · LIVE

```python
entry_price = float(bot_pos.get("entry_price") or 0.0)
```

Raises on a corrupt stored value, inside `_book_pending_sell_fills` on the
trading thread. The attribution helper next to it is guarded; this, on the
trade-ledger path, is not.

### NaN spread bypasses the universe spread gate
`quant/universe.py:144` · LATENT

A NaN `avg_spread_bps` is not `None`, so the `spread_unknown` branch is
skipped, and `NaN > max_spread_bps` is False, so `spread_too_wide` is
skipped too. The symbol passes a filter that never evaluated it.

### The survivorship guard is a lookahead guard
`quant/universe.py:114` · LATENT

It rejects stats whose caller-asserted `asof` postdates the requested
moment. That is a lookahead check. Nothing reconstructs historical index
membership, so survivorship bias is undetected — and the assertion is
supplied by the caller, so it is trivially bypassed.

### Regime attribution can never join
`quant/scanner.py:354` · LATENT

Lookups key on `RegimeLabel` values (STRONG_BULL, WEAK_BULL, …) while
every writer of `regime_at_entry` stores a `CryptoRegime` value. Per-regime
performance is empty by construction.

### Calibration is measured in-sample
`quant/ml/direction.py:143` · LATENT

`PoolAdjacentViolators` is fitted on `(raw_val, y_val)` and the reported
`validation_brier` / `validation_ece` are computed on the calibrated output
against the same `y_val`. A calibrator scored on its own fitting set.

### No correlation penalty in the opportunity score
`quant/opportunity.py:96` · LATENT

`score_components` multiplies net alpha by confidence, regime, reliability,
liquidity and two penalties. Correlation is absent, despite being one of
the listed score components.

### Sector caps rest on a hardcoded 60-symbol dict
`quant/portfolio.py:57` · LATENT

`SECTOR_MAP` is the only sector data in the repository. Any symbol outside
it is unclassified, so the sector cap is decorative beyond those 60 names.

### Execution quality aggregates across everything
`state_store.py:619` · LIVE

`execution_quality()` averages `slippage_bps` over the whole executions
table with no market, symbol, side or purpose filter. One market's slippage
contaminates the other's estimate.

### `run_execution_calibration.py` defaults to the wrong database
`scripts/run_execution_calibration.py:44` · LIVE

Default `--db` is `trading_bot.db`; the application uses
`config.DEFAULT_DB_PATH = data/trading_bot.db`. Run without `--db`, it
reads an empty or absent database and reports no execution history.

---

## Coverage caveat

The audit that produced these ran 10 subsystem passes and 60 planned
adversarial verifications; 35 verifications completed before the run hit a
session limit. Findings above are marked by how they were established:
everything in CRITICAL and the first four HIGH entries was reproduced by
hand. The remainder come from a single lane pass and should be treated as
strong suspicion until independently reproduced.
