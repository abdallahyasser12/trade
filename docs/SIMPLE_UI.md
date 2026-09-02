# Simple operator UI

The primary production dashboard is `bybit_operator_ui.py`. It intentionally exposes the minimum information an operator needs to supervise a Bybit Spot Demo burn-in.

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

The top section makes environment and safety state obvious:

- BYBIT DEMO
- Spot only / no leverage / no derivatives / live disabled
- public/private connection state
- backend running/stopped
- uptime
- HEALTHY / DEGRADED / HALTED
- clock skew
- unprotected quantity

Operational summary:

- strategy capital allocation
- today net P&L
- open bot-owned exposure
- persistent emergency kill-switch state

If credentials are missing, HOME gives a one-line quick-start instruction pointing to SETTINGS.

Top opportunities show at most the strongest configured symbols with an operator-friendly decision:

- BUY SETUP
- WATCH
- NO TRADE
- AVOID

Cards show score, price, accepted structure/ATR/R target when one exists, protective stop, risk/reward and target logic. The +0.5/+0.75/+1.0 research families are not hardcoded profit orders.

Open positions show only proven bot-owned Spot inventory, not unexplained wallet holdings. The current protective stop is projected from engine-authoritative persisted position state and the accepted target plan is shown when available.

Recent activity includes decisions, no-trade decisions, order/execution events, position management and operational events.

## MARKETS

The configured BTC / ETH / SOL cards show:

- price
- 24-hour move
- decision
- regime
- spread
- approximate 10-level ask depth
- trend
- momentum
- volatility
- accepted target when available

The **Why?** expander converts model state into plain-language setup, liquidity, target-room and cost information. Raw details/microstructure/target plan remain available only in the nested Advanced expander.

## PERFORMANCE

The dashboard never hides losing trades. It displays current time-causal burn-in evidence:

- net P&L
- trade count
- win rate
- profit factor
- expectancy
- maximum drawdown
- fees
- average winner/loss
- equity curve
- +0.25%, +0.50%, +0.75% and +1.00% MFE reach rates

Operational and preliminary-edge verdicts are shown separately. The screen always states that a 3–5 day run does not prove a durable strategy.

## SETTINGS

SETTINGS is intentionally a three-step workflow.

### 1. Connect Bybit Demo

The operator can paste the Bybit Demo API key and secret directly into password-masked fields. The form clears after submit.

Actions:

- **Test connection** — read-only Demo environment/account verification; no order is submitted.
- **Test & Save** — verifies first, then stores the credentials in the isolated server-side credential vault.
- **Test saved connection** — re-validates the current saved/environment credentials.
- **Clear dashboard-saved credentials** — deletes only the dashboard vault copy while the backend is stopped.

The secret is never displayed back to the browser. It is not stored in SQLite, logs, reports or subprocess arguments.

### 2. Choose run settings

Normal settings are limited to:

- strategy lifecycle/readiness status
- capital allocation
- allowed BTC / ETH / SOL universe
- 72h / 120h duration
- researched smart-exit consumers switch
- phone push-notification preference
- optional display-only EGP reference

The smart-exit switch enables existing `TradeManagementPolicy` consumers. It does not create another exit engine.

The phone-notification preference controls the existing mobile push observer when that service/device registration is configured. Notification failure never affects trading.

### 3. Start / stop

A readiness row shows clear pass/fail markers for:

- Demo API configured
- lifecycle eligible
- permitted universe non-empty
- safety latch clear

Controls:

- **Start Bybit Demo**
- **Stop gracefully**
- **EMERGENCY STOP**

Start is disabled until all readiness checks pass and no backend is already running.

## Credential vault

Local dashboard-saved credentials live under `secrets/`, which is Git-ignored. Production Docker mounts a dedicated `trade_secrets` volume only into the operator dashboard container. The research dashboard and mobile API do not mount it.

The vault is written atomically with `0600` permissions. The detached backend receives the Demo key and secret only in its process environment; no secret-bearing command-line argument exists.

## Emergency latch

Emergency Stop persists the engine kill latch before signalling the backend. Starting remains blocked even if the process disappears.

Clearing the latch requires an explicit acknowledgement in Settings and is refused while the backend is still running.

## Backend ownership

Streamlit is presentation/control only. Start creates a detached server-side process.

Closing the browser does not stop orders, protection, reconciliation or WebSocket handling.

## Advanced / Research

The Settings expander links to `/research`, which hosts the existing `app.py` UI in production Docker. It retains research, backtests, walk-forward work, Quant Desk views, lifecycle administration, Alpaca workflows and diagnostics without cluttering ordinary Bybit operation.

## Production routing

`docker-compose.production.yml` runs:

- `bybit_operator_ui.py` on port 8501 as the root dashboard;
- `app.py` on port 8502 with Streamlit base path `/research`;
- the existing mobile API on port 8000.

Caddy routes `/api/*` to the API, `/research*` to the research dashboard, and everything else to the simple Bybit operator dashboard.
