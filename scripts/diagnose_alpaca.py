#!/usr/bin/env python3
"""Find out exactly why market-data access is failing, in about ten seconds.

Runs the smallest possible authenticated request and classifies whatever comes
back: missing variable, wrong key type, rejected credential, entitlement,
rate limit, clock skew, network path, or a symbol that simply has no data.

Credential VALUES are never printed — only whether a variable is set, its
length, and its non-secret prefix (Alpaca key IDs are public identifiers whose
first two characters indicate the key type).

Exit codes:  0 data reachable   1 diagnosed failure   2 nothing to test
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RULE = "-" * 74
PAPER_VARS = ("ALPACA_PAPER_API_KEY", "ALPACA_PAPER_SECRET_KEY")
LEGACY_VARS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
LIVE_VARS = ("ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET_KEY")


def describe(name: str) -> str:
    """Report shape, never content."""

    raw = os.environ.get(name)
    if raw is None:
        return f"  {name:<28} not set"
    stripped = raw.strip()
    if not stripped:
        return f"  {name:<28} set but EMPTY"
    note = ""
    if raw != stripped:
        note = "  <-- has surrounding whitespace, likely a copy/paste artifact"
    if stripped[0] in "'\"" or stripped[-1] in "'\"":
        note = "  <-- contains quote characters; do not quote inside a .env file"
    kind = ""
    if name.endswith("API_KEY"):
        if stripped.startswith("PK"):
            kind = " (looks like a PAPER key id)"
        elif stripped.startswith("AK"):
            kind = " (looks like a LIVE key id)"
        else:
            kind = " (unrecognised prefix)"
    return f"  {name:<28} set, {len(stripped)} chars, starts {stripped[:2]!r}{kind}{note}"


def main() -> int:
    from config import load_credentials

    print(RULE)
    print("ALPACA MARKET-DATA DIAGNOSTIC")
    print(RULE)
    print(f"  Local UTC now   {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
    print(f"  Python          {sys.version.split()[0]}")
    print()

    print("Environment variables (values never shown):")
    for name in PAPER_VARS + LEGACY_VARS + LIVE_VARS:
        print(describe(name))
    env_path = Path(__file__).resolve().parent.parent / ".env"
    print(f"\n  .env file                    {'found' if env_path.exists() else 'not present'}")

    credentials = load_credentials()
    key, secret = credentials.api_key, credentials.secret_key
    print()
    if not key or not secret:
        print(RULE)
        print("VERDICT: no usable paper credentials were resolved.")
        print(RULE)
        print("""
Set both variables in the SAME shell that runs the script:

    export ALPACA_PAPER_API_KEY=PK...
    export ALPACA_PAPER_SECRET_KEY=...

Common causes:
  - exported in a different terminal tab than the one running the script
  - used `sudo`, which does not carry your exported variables
  - written into .env with quotes or spaces around the '='
  - only one of the two variables was set
""".strip())
        return 2

    if key.startswith("AK"):
        print("  WARNING: this key id looks like a LIVE key. Use PAPER keys — the")
        print("           data API accepts either, but never put live keys here.")
        print()

    # ---- smallest possible authenticated request -------------------------
    print(RULE)
    print("Live request: 1Hour BTC/USD bars for the last 3 days")
    print(RULE)

    try:
        from crypto_data_feed import AlpacaCryptoDataFeed
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not import the data feed: {type(exc).__name__}: {exc}")
        return 1

    feed = AlpacaCryptoDataFeed(key, secret)
    end = datetime.now(timezone.utc) - timedelta(minutes=15)
    start = end - timedelta(days=3)

    try:
        frame = feed.get_bars("BTC/USD", "1Hour", start=start, end=end)
    except Exception as exc:  # noqa: BLE001 - classification is the whole point
        return diagnose(exc)

    if frame.empty:
        print("  Request succeeded but returned ZERO bars.")
        print("\n  Alpaca authenticated you and returned nothing. Usually a date")
        print("  range problem or a symbol not carried on your data plan.")
        return 1

    print(f"  OK — {len(frame)} bars, {frame.index[0]:%Y-%m-%d %H:%M} .. "
          f"{frame.index[-1]:%Y-%m-%d %H:%M} UTC")
    print(f"  Last close: {float(frame['close'].iloc[-1]):,.2f}")

    newest_age = (datetime.now(timezone.utc) - frame.index[-1].to_pydatetime()).total_seconds()
    if newest_age > 6 * 3600:
        print(f"\n  NOTE: newest bar is {newest_age / 3600:.1f}h old. If that is far more")
        print("  than one bar, check your system clock (`timedatectl`).")

    # ---- each configured symbol, and the long history the research needs --
    print()
    print(RULE)
    print("Per-symbol availability")
    print(RULE)
    from crypto_config import DEFAULT_CRYPTO_UNIVERSE

    ok = True
    for symbol in DEFAULT_CRYPTO_UNIVERSE:
        try:
            probe = feed.get_bars(symbol, "1Hour", start=start, end=end)
            print(f"  {symbol:<10} {len(probe):>5} bars   OK")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  {symbol:<10} FAILED   {type(exc).__name__}: {str(exc)[:110]}")

    print()
    print(RULE)
    print("Deep history (what the walk-forward actually needs)")
    print(RULE)
    deep_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    try:
        deep = feed.get_bars("BTC/USD", "1Hour", start=deep_start, end=end)
        print(f"  BTC/USD from 2023-01-01: {len(deep):,} bars "
              f"({deep.index[0]:%Y-%m-%d} .. {deep.index[-1]:%Y-%m-%d})")
        if len(deep) < 3_000:
            ok = False
            print("\n  Too little history for the default 1500-train/500-test folds.")
            print("  Use a later --start, smaller --train-bars/--test-bars, or --timeframe 1Day.")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAILED   {type(exc).__name__}: {str(exc)[:160]}")
        print("\n  Short windows work but long ones do not — usually a plan limit on")
        print("  historical depth, or a timeout. Try --start 2024-01-01.")

    print()
    print(RULE)
    if ok:
        print("VERDICT: market data is reachable. Research should run.")
        print(RULE)
        return 0
    print("VERDICT: data partially reachable — see the notes above.")
    print(RULE)
    return 1


def diagnose(exc: Exception) -> int:
    """Turn a vendor exception into an instruction."""

    from data_feed import is_rate_limit_error, is_subscription_error

    name = type(exc).__name__
    text = str(exc)
    status = getattr(exc, "status_code", None)
    lowered = text.lower()

    print(f"  FAILED   {name}")
    print(f"  status   {status if status is not None else 'none reported'}")
    print(f"  message  {text[:400]}")
    print()
    print(RULE)
    print("VERDICT")
    print(RULE)

    if status in (401, 403) and not is_subscription_error(exc):
        print("""
Alpaca REJECTED the credentials (HTTP 401/403).

The keys are reaching Alpaca and being refused, so this is the credential
itself, not your network. Check, in order:

  1. Key and secret come from the SAME key pair. Regenerating a key
     invalidates the old secret; a new key with an old secret fails exactly
     like this.
  2. They are PAPER keys, from Alpaca -> Paper Trading -> API Keys, not live
     keys and not an OAuth token.
  3. Nothing was truncated on copy. The secret is long and easy to clip.
  4. The key was not revoked. Note that .env.example in this repo warns a
     paper key pair was committed in the past and must be treated as
     compromised — if you are using that pair, regenerate it.

Regenerate the pair in the Alpaca dashboard and re-export both values.
""".strip())
    elif is_subscription_error(exc):
        print("""
This is a DATA ENTITLEMENT failure, not a bad password.

The account authenticated but is not permitted this data. For crypto this is
unusual — the US crypto feed is generally available on paper accounts. Confirm
the account is fully activated in the Alpaca dashboard.
""".strip())
    elif is_rate_limit_error(exc):
        print("""
Rate limited (HTTP 429). Wait a minute and retry; if it persists, another
process is sharing these keys.
""".strip())
    elif status == 404:
        print("Endpoint or symbol not found (404). Check the symbol spelling, e.g. BTC/USD.")
    elif any(token in lowered for token in
             ("connection", "timeout", "dns", "resolve", "ssl", "certificate",
              "proxy", "unreachable", "network", "refused")):
        print("""
The request never reached Alpaca — this is a NETWORK path problem.

  - Confirm general connectivity:   curl -sS https://data.alpaca.markets/v1beta3/crypto/us/bars
  - If you are on a VPN, proxy, or Tor, try disabling it. Kali images
    sometimes ship with a proxy configured.
  - Check HTTP_PROXY / HTTPS_PROXY are not set to something stale:  env | grep -i proxy
  - If your clock is wrong, TLS fails in confusing ways:  timedatectl
""".strip())
    else:
        print(f"""
Unclassified failure: {name}.

Send the message above. If it mentions a module or import, the dependency
install may be incomplete — delete .venv and re-run the checks script.
""".strip())

    print(RULE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
