# Bybit Spot Demo setup

This integration is **Bybit Spot Demo only** for the production acceptance mission. It does not enable margin, leverage, perpetuals, futures, inverse contracts, options, shorts, lending, borrowing, martingale, averaging down, grid-loss recovery, or loss-doubling.

The ordinary operator uses the Streamlit dashboard. The CLI runner remains a service/recovery entry point; it is not required for normal operation.

## 1. Create Bybit Demo credentials

Use credentials created under Bybit **Demo Trading**. Demo Trading is distinct from Testnet and from a live-money account. Create an API key with only the read/trade permissions required for Spot Demo operation. **Do not grant withdrawal permission.** Keep Demo and future Live credentials separate.

## 2. Start the dashboard

Local:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-mobile-api.txt
streamlit run bybit_operator_app.py
```

Production Docker requires a private `.env` containing at least `TRADE_DOMAIN` and strong `TRADE_DASHBOARD_PASSWORD`:

```bash
docker compose -f docker-compose.production.yml up -d --build
```

Production `/` serves the simple operator dashboard and `/research` preserves the full research application. Both browser dashboards are protected by Caddy HTTP Basic Auth over HTTPS using username `operator`; Caddy hashes `TRADE_DASHBOARD_PASSWORD` at startup and fails closed if it is missing. `/api/*` retains the mobile API's existing bearer-token authentication.

## 3. Add the API key through SETTINGS

Open **SETTINGS → Connect Bybit Demo**. Paste the Demo key and secret, then use **Test connection** or **Test & Save**. Fields are password-masked and clear after submit; saved credentials are never rendered back.

A dashboard-saved credential lives only in the server-side vault (`secrets/bybit_demo_credentials.json` locally, `/app/secrets/...` in production), atomically written with restrictive `0600` permissions. It is excluded from Git and the production secret volume is mounted only into the operator dashboard. Credentials are not stored raw in SQLite, reports, logs, URLs or subprocess arguments; detached processes receive them only through environment variables.

Existing deployments may use `BYBIT_DEMO_API_KEY` / `BYBIT_DEMO_API_SECRET` in the private server `.env` as a fallback. The normal operator path cannot select Live trading.

## 4. Run the required tiny Demo smoke

Before the 72–120 hour burn-in can start, open **SETTINGS → Verify tiny Demo order path**:

1. Choose BTC, ETH, or SOL.
2. Leave the default maximum Demo notional at 25 USDT unless current venue minimums require a slightly higher explicit cap.
3. Acknowledge that one tiny **Bybit Demo Spot** order will be opened and closed.
4. Click **Run Demo economic smoke**.

The smoke verifies private account access, current instrument metadata, public/private streams, Spot submission, execution reconciliation, bot-owned inventory accounting and clean close with no bot-owned residual. It cannot select Live, margin, leverage, derivatives or shorts.

The PASS is bound to the current credential pair. Changing or clearing saved credentials invalidates it and blocks Start until a new smoke passes.

## 5. Readiness checks

SETTINGS requires:

- Demo API ready
- Demo economic smoke PASS for current credentials
- strategy lifecycle PAPER or another already-approved broker-backed state
- non-empty BTC/ETH/SOL universe
- safety latch clear
- backend stopped

The dashboard never auto-promotes RESEARCH/SHADOW to obtain more trades.

## 6. Capital, universe, smart exits and notifications

Set a **strategy capital allocation** in USDT, independent of the full virtual Demo wallet. Choose 72h/3 days or 120h/5 days and a permitted BTC/ETH/SOL subset.

The researched smart-exit switch uses the existing `TradeManagementPolicy` for partial profit, break-even, time/smart exit and adaptive trailing. The Bybit entry layer adds negative-only depth, impact and structure/ATR/R/cost gates. There is still one post-entry policy.

The phone-push switch controls the existing mobile observer when configured. Push failure never affects trading or protection. Optional EGP reference is display-only.

## 7. Start / stop

**Start Bybit Demo** launches a detached server-side backend. Closing the browser does not stop protection, reconciliation or WebSocket supervision. Start refuses missing credentials, missing/current-smoke PASS, ineligible lifecycle, armed kill switch, another backend, invalid universe or capital.

Use **Stop gracefully** for ordinary shutdown. **EMERGENCY STOP** persists the kill latch before signalling the backend; clearing it requires explicit acknowledgement while stopped.

## 8. Restart identity

A burn-in freezes run ID, Git SHA, strategy version, configuration hash, universe, start/planned end and environment. Resume is permitted only for exact matching identity. Parameter changes create a new experiment.

## 9. Reports

HOME/PERFORMANCE read persisted evidence during the run. Final JSON/Markdown reports are written under `reports/`. A 72–120h sample is operational/preliminary evidence only and never means "strategy proven."

## 10. Unified acceptance command

Local and Python CI acceptance use exactly the same command:

```bash
python scripts/run_bybit_acceptance.py
```

It byte-compiles, runs the dedicated Bybit/dashboard safety suite, runs the complete Alpaca/Quant/mobile/Bybit regression suite, checks tracked-secret hygiene, validates Docker Compose when Docker is available, and writes `reports/bybit_acceptance_<SHA>.json` plus `.md`.

Python CI runs this same command on Python 3.11 and 3.13 and uploads the acceptance receipts. If GitHub-hosted runner allocation is unavailable, the local command is the exact source-controlled fallback rather than a different test recipe.

To append the real Demo smoke after hermetic checks pass:

```bash
python scripts/run_bybit_acceptance.py \
  --demo-smoke \
  --confirm-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
```

## Service / diagnostic commands

```bash
python scripts/bybit_demo_smoke.py --confirm I_ACCEPT_BYBIT_DEMO_ORDER
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
python scripts/generate_bybit_burnin_report.py --run-id <RUN_ID>
```

None can select Bybit Live trading in this mission.
