# Bybit Spot Demo setup

This integration is **Bybit Spot Demo only** for the production acceptance mission. It does not enable margin, leverage, perpetuals, futures, inverse contracts, options, shorts, lending, borrowing, martingale, averaging down, grid-loss recovery, or loss-doubling.

The ordinary operator uses the Streamlit dashboard. The CLI runner remains a service/recovery entry point; it is not required for normal operation.

## 1. Create Bybit Demo credentials

Use credentials created under Bybit **Demo Trading**. Demo Trading is distinct from Testnet and from a live-money account.

Create an API key with only the read/trade permissions required for Spot Demo operation. **Do not grant withdrawal permission.** No withdrawal permission is needed by this application.

Keep Demo and future Live credentials separate. Never paste a Live credential into the Demo dashboard.

## 2. Start the dashboard

Local:

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

## 3. Add the API key through SETTINGS

Open **SETTINGS → Connect Bybit Demo**.

1. Paste the **Bybit Demo API key**.
2. Paste the **Bybit Demo API secret**.
3. Click **Test connection** if you only want to verify it, or **Test & Save** to verify and store it for future dashboard starts.

The fields are password-masked and clear after form submission. The dashboard never renders the saved key or secret back to the browser.

A dashboard-saved credential is stored only in the server-side credential vault (`secrets/bybit_demo_credentials.json` locally, `/app/secrets/...` in production). The vault is written atomically with restrictive `0600` permissions. It is excluded from Git and production Docker mounts it through a dedicated volume that is available to the operator dashboard only. The research dashboard and mobile API do not mount that secret volume.

Credentials are **not** stored in SQLite, reports, logs, URLs, or subprocess arguments. When a run starts, the detached Bybit backend receives them only through its process environment.

You can click **Test saved connection** at any time while the backend is stopped. **Clear dashboard-saved credentials** removes the vault copy.

### Optional deployment-managed fallback

Existing deployments may still set:

```text
BYBIT_DEMO_API_KEY=...
BYBIT_DEMO_API_SECRET=...
```

in the private server `.env`. Dashboard-saved credentials take precedence when present. `BYBIT_LIVE_API_KEY` and `BYBIT_LIVE_API_SECRET` remain separate and the normal Bybit operator path cannot select Live trading.

## 4. Readiness checks

SETTINGS shows a simple readiness row before Start:

- Demo API: ready
- Strategy lifecycle: PAPER or another already-approved broker-backed state
- Coin universe: non-empty BTC/ETH/SOL subset
- Safety latch: clear
- Backend: stopped

The dashboard never auto-promotes a strategy from RESEARCH/SHADOW just to obtain more trades.

After the backend starts, HOME shows public/private WebSocket status, engine status, health, clock skew, unprotected quantity, capital allocation, P&L, exposure, opportunities, positions and recent activity.

## 5. Choose the permitted Spot universe

Normal settings permit only:

- BTC
- ETH
- SOL

The dashboard maps these to the canonical strategy symbols `BTC/USD`, `ETH/USD`, and `SOL/USD`; the venue adapter maps them explicitly to Bybit `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`.

No meme coin or microcap is added automatically. Software does not certify Sharia compliance; the list is simply a user-controlled permitted Spot universe.

## 6. Capital, duration, smart exits and notifications

Set a **strategy capital allocation** in USDT. This is independent of the total virtual wallet balance in Bybit Demo. Sizing is capped by the configured allocation and does not automatically scale to the full Demo wallet.

Choose either:

- 72 hours / 3 days
- 120 hours / 5 days

The **researched smart-exit consumers** switch uses the existing `TradeManagementPolicy` for partial profit, break-even, time/smart exit and adaptive trailing. It does not tune thresholds to chase a requested return.

The Bybit-specific entry layer additionally uses current multi-level order-book depth, an impact proxy, structural room, ATR, initial R and expected round-trip costs as negative-only gates. Accepted target plans feed into that same existing `TradeManagementPolicy`; there is still only one post-entry exit policy.

If the existing mobile app / Expo push service is configured, the **Phone push notifications** switch controls delivery of Bybit engine-state, safety-lock and new-fill notifications. The notifier watches the Bybit engine database while device tokens remain in the existing mobile API database. Notification failure can never affect trading or protection.

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

## Service / diagnostic commands

Normal operators do not need these, but they remain useful for service supervision and acceptance evidence:

```bash
python scripts/bybit_demo_smoke.py --confirm I_ACCEPT_BYBIT_DEMO_ORDER
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
python scripts/generate_bybit_burnin_report.py --run-id <RUN_ID>
```

None of these commands can select Bybit Live trading in this mission.
