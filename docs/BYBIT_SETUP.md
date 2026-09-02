# Bybit Spot Demo setup

This integration is **Bybit Spot Demo only** for the production acceptance mission. It does not enable margin, leverage, perpetuals, futures, inverse contracts, options, shorts, lending, borrowing, martingale, averaging down, grid-loss recovery, or loss-doubling.

The ordinary operator uses the Streamlit dashboard. The CLI runner remains an implementation/service entry point and troubleshooting tool; it is not required for normal start/stop operation.

## 1. Create a Bybit Demo account and API credentials

Use Bybit **Demo Trading** credentials. Demo trading is distinct from Testnet and from a live-money account.

Create an API key with only the read/trade permissions required for Spot Demo operation. **Do not grant withdrawal permission.** No withdrawal permission is needed by this application.

Keep Demo and future Live credentials separate. Never put Live credentials in the Demo variables.

## 2. Configure the server environment

Copy the template and edit the private `.env` file:

```bash
cp .env.example .env
```

Set only:

```text
BYBIT_DEMO_API_KEY=...
BYBIT_DEMO_API_SECRET=...
```

Do not commit `.env`. The application does not store these secrets in SQLite, reports, frontend payloads, or command-line arguments.

`BYBIT_LIVE_API_KEY` and `BYBIT_LIVE_API_SECRET` should remain blank. The current Bybit operator path cannot construct a Live broker.

## 3. Install and start the dashboard

Local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-mobile-api.txt
streamlit run bybit_operator_ui.py
```

Production Docker:

```bash
docker compose -f docker-compose.production.yml up -d --build
```

The production root route serves the simple Bybit operator dashboard. The existing full research application is preserved at `/research`.

## 4. Verify connectivity

Open **SETTINGS** in the dashboard. Verify:

- Venue: Bybit
- Environment: DEMO
- Demo credentials: Configured
- Strategy lifecycle: PAPER or another already-approved broker-backed state
- Backend: Stopped before starting a new run
- Kill switch: not armed

The dashboard never auto-promotes a strategy from RESEARCH/SHADOW just to obtain more trades.

After the backend starts, HOME shows public/private WebSocket status, engine status, health, clock skew, unprotected quantity, capital allocation, P&L, exposure, opportunities, positions and recent activity.

## 5. Choose the permitted Spot universe

Normal settings permit only:

- BTC
- ETH
- SOL

The dashboard maps these to the canonical strategy symbols `BTC/USD`, `ETH/USD`, and `SOL/USD`; the venue adapter maps them explicitly to Bybit `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`.

No meme coin or microcap is added automatically. Software does not certify Sharia compliance; the list is simply a user-controlled permitted Spot universe.

## 6. Set capital and burn-in duration

Set a **strategy capital allocation** in USDT. This is independent of the total virtual wallet balance in Bybit Demo. Sizing is capped by the configured allocation and does not automatically scale to the full Demo wallet.

Choose either:

- 72 hours / 3 days
- 120 hours / 5 days

An optional EGP reference rate in the UI is display-only and does not affect sizing or risk mathematics.

## 7. Start from the dashboard

In **SETTINGS**, click **Start Bybit Demo**.

The dashboard launches a detached server-side backend. The browser is not the trading process. Closing the tab or restarting Streamlit does not stop the backend, protective orders, reconciliation, or WebSocket supervision.

The start path refuses to run if:

- Demo credentials are missing
- the strategy lifecycle is not broker-eligible
- the emergency kill switch is armed
- another owned backend is already running
- the permitted universe is empty or outside BTC/ETH/SOL
- the capital allocation is invalid

## 8. Emergency stop

Use **EMERGENCY STOP** when the engine must halt. The dashboard persists the kill-switch latch **before** signalling the backend. A later restart remains blocked even if the process disappeared before receiving the signal.

After the incident is understood, go to SETTINGS, review the recorded reason/time, explicitly check the acknowledgement box, and clear the latch. It cannot be cleared while the backend is running.

For ordinary shutdown use **Stop gracefully**. The backend owns reconciliation and protection during shutdown.

## 9. Resume after restart

The burn-in identity includes:

- `burnin_run_id`
- Git SHA
- strategy version
- configuration hash
- frozen universe
- start time
- planned end time
- environment

A process restart resumes an active run only when the code/config/universe/environment match exactly. Changed strategy parameters create a different experiment and cannot silently resume the old identity.

## 10. Reports

HOME and PERFORMANCE read persisted evidence while a run is active. Final JSON and Markdown reports are written under `reports/` when the run ends.

A 72–120 hour run is an operational acceptance experiment and preliminary forward sample. The software never labels it "strategy proven" and never claims a guaranteed return.

## Service/diagnostic commands

Normal operators do not need these, but they remain useful for service supervision and acceptance evidence:

```bash
python scripts/bybit_demo_smoke.py --acknowledge-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
python scripts/generate_bybit_burnin_report.py --run-id <RUN_ID>
```

None of these commands can select Bybit Live trading in this mission.
