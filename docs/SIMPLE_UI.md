# Simple operator UI

The primary production dashboard entrypoint is `bybit_operator_app.py`, which composes the existing `bybit_operator_ui.py` HOME/MARKETS/PERFORMANCE surfaces with the acceptance-aware SETTINGS flow. It intentionally exposes the minimum information an operator needs to supervise a Bybit Spot Demo burn-in.

The legacy research-heavy application remains available at `/research`; none of its underlying capabilities are deleted.

## Normal navigation

```text
HOME
MARKETS
PERFORMANCE
SETTINGS
```

The normal operator should not need to understand walk-forward grids, Quant Desk database internals, lifecycle administration or raw strategy research knobs.

## HOME

The top section makes environment and safety state obvious: BYBIT DEMO, Spot-only/no-leverage/no-derivatives/live-disabled state, public/private connection, backend state, uptime, HEALTHY/DEGRADED/HALTED, clock skew and unprotected quantity.

Operational summary includes strategy capital allocation, today net P&L, open bot-owned exposure and the persistent emergency kill-switch state.

Top opportunities show at most the strongest configured symbols with `BUY SETUP`, `WATCH`, `NO TRADE` or `AVOID`, plus score, price, accepted structure/ATR/R target when one exists, protective stop, risk/reward and target logic. The +0.5/+0.75/+1.0 research families are not hardcoded profit orders.

Open positions show only proven bot-owned Spot inventory, not unexplained wallet holdings. Recent activity includes decisions, no-trade decisions, order/execution events, position management and operational events.

## MARKETS

BTC/ETH/SOL cards show price, 24-hour move, decision, regime, spread, approximate multi-level ask depth, trend, momentum, volatility and accepted target when available. The **Why?** expander turns model state into plain-language setup, liquidity, target-room and cost information; raw details remain in Advanced.

## PERFORMANCE

The dashboard never hides losing trades. It displays current time-causal net P&L, trades, win rate, profit factor, expectancy, maximum drawdown, fees, average winner/loss, equity curve, and +0.25/+0.50/+0.75/+1.00 MFE reach rates. Operational and preliminary-edge verdicts are separate and the screen states that 3–5 days does not prove a durable strategy.

## SETTINGS

SETTINGS is intentionally a four-step workflow.

### 1. Connect Bybit Demo

The operator pastes the Bybit Demo API key and secret into password-masked fields. Actions are **Test connection**, **Test & Save**, **Test saved connection**, and **Clear dashboard-saved credentials**. Saved secrets are never rendered back to the browser or stored raw in SQLite/logs/reports/command arguments.

### 2. Verify tiny Demo order path

The operator selects BTC/ETH/SOL, chooses a small explicit maximum notional, acknowledges that one tiny **Bybit Demo Spot** order will be opened/closed, then runs **Demo economic smoke**.

The smoke verifies current metadata, private account access, public/private streams, Spot order submission, execution reconciliation, bot-owned inventory, clean close and zero bot-owned residual. It cannot select Live, margin, leverage, derivatives or shorts.

A PASS is bound to the current credential pair. Changing/clearing saved credentials invalidates it. Burn-in Start remains blocked until the current credentials have a PASS.

### 3. Choose run settings

Normal settings are capital allocation, BTC/ETH/SOL universe, 72h/120h duration, existing smart-exit consumers, phone push preference and optional display-only EGP reference. The smart-exit switch uses the existing `TradeManagementPolicy`; notification failure never affects trading.

### 4. Start / stop

Readiness requires Demo API configured, Demo smoke PASS, lifecycle eligible, non-empty universe and safety latch clear. Controls are **Start Bybit Demo**, **Stop gracefully**, and **EMERGENCY STOP**. Start is disabled until every gate passes and no backend is already running.

## Credential vault

Local dashboard-saved credentials live under Git-ignored `secrets/`. Production Docker mounts a dedicated `trade_secrets` volume only into the operator dashboard container. The vault is atomically written with `0600` permissions. The detached backend receives Demo credentials only in its process environment.

## Production browser authentication

Production Docker requires `TRADE_DASHBOARD_PASSWORD`. Caddy protects `/` and `/research` with HTTP Basic Auth over HTTPS using username `operator`, hashes the password at container startup, and fails closed when it is missing. `/api/*` is excluded because the mobile API has independent bearer-token authentication.

## Emergency latch and backend ownership

Emergency Stop persists the kill latch before signalling the backend. Clearing it requires explicit acknowledgement and is refused while running. Streamlit is presentation/control only; Start creates a detached server-side process, so closing the browser does not stop orders, protection, reconciliation or WebSockets.

## Advanced / Research

`/research` hosts the existing `app.py` UI and retains research, backtests, walk-forward work, Quant Desk views, lifecycle administration, Alpaca workflows and diagnostics without cluttering ordinary Bybit operation.

## Production routing

`docker-compose.production.yml` runs `bybit_operator_app.py` at root, `app.py` under `/research`, and mobile API on `/api/*`. Caddy authenticates both browser dashboards before proxying them.

## Acceptance evidence

Local and Python CI acceptance use the same command:

```bash
python scripts/run_bybit_acceptance.py
```

It writes JSON/Markdown evidence under `reports/`. Python CI runs the same command on 3.11/3.13 and uploads those receipts whenever a hosted runner is available.
