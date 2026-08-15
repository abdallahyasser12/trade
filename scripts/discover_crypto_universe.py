#!/usr/bin/env python3
"""Measure which Alpaca USD crypto pairs have enough daily history to research.

The walk-forward protocol requires exactly aligned bars across the whole
universe, so one shallow-history symbol truncates every other symbol to its
depth (adding SOL/USD once cost a third of the BTC/ETH sample). This script
probes the DATA API for every candidate pair and reports, per symbol: first
bar, last bar, and bar count — then recommends the deepest universe that fits
the engine's 12-symbol limit, with the aligned bar count that universe would
actually give the research protocol.

Asset metadata and market data only; nothing here can place an order.

Exit codes:
  0  at least one symbol qualified and a recommendation was printed
  2  the probe could not be completed (credentials, data, or configuration)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_credentials
from crypto_config import validate_crypto_symbol
from crypto_data_feed import AlpacaCryptoDataFeed

RULE = "=" * 78

# Engine validation limit: "Crypto universe is limited to 12 symbols per engine."
MAX_UNIVERSE = 12

# Used when the trading-API asset listing is unavailable. Kept deliberately
# broad; symbols Alpaca does not serve simply fail the probe and are reported.
FALLBACK_CANDIDATES = (
    "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "DOGE/USD", "SOL/USD",
    "AVAX/USD", "LINK/USD", "UNI/USD", "AAVE/USD", "DOT/USD", "SHIB/USD",
    "XTZ/USD", "YFI/USD", "GRT/USD", "BAT/USD", "CRV/USD", "MKR/USD",
    "SUSHI/USD", "XRP/USD",
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Alpaca daily-bar depth per USD crypto pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=1_800,
        help="Minimum daily bars a symbol needs to make the recommendation. "
             "Alpaca crypto history starts 2021-01-01, so full depth is "
             "roughly 2,000 bars; the default keeps only near-full histories "
             "that will not truncate the aligned sample.",
    )
    parser.add_argument(
        "--candidates",
        default="",
        help="Optional comma-separated override of the pairs to probe. "
             "Blank = ask the Alpaca asset API, falling back to a built-in list.",
    )
    parser.add_argument(
        "--json-out", default="", help="Optional path for the full result as JSON."
    )
    return parser.parse_args(argv)


def _fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return 2


def _candidate_symbols(credentials) -> tuple[list[str], str]:
    """All active tradable USD pairs from the asset API, or the fallback list.

    Uses the trading API read-only for metadata. The client is created for
    exactly one listing call and holds no order-side state in this script.
    """

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        client = TradingClient(
            credentials.api_key, credentials.secret_key, paper=True
        )
        assets = client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.CRYPTO, status=AssetStatus.ACTIVE)
        )
        symbols = sorted(
            asset.symbol
            for asset in assets
            if asset.symbol.endswith("/USD")
            and bool(asset.tradable)
            and validate_crypto_symbol(asset.symbol) is None
        )
        if symbols:
            return symbols, "alpaca asset API"
    except Exception as exc:  # noqa: BLE001 - fall back, but say why
        print(f"  Asset listing unavailable ({type(exc).__name__}); using the built-in list.")
    return list(FALLBACK_CANDIDATES), "built-in fallback list"


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.min_bars < 100:
        return _fail("--min-bars below 100 cannot support any walk-forward fold.")

    credentials = load_credentials()
    if not credentials.api_key or not credentials.secret_key:
        return _fail(
            "No Alpaca paper credentials found. Set ALPACA_PAPER_API_KEY and\n"
            "ALPACA_PAPER_SECRET_KEY in the environment, a .env file, or CI secrets."
        )

    print(RULE)
    print("CRYPTO UNIVERSE DISCOVERY — daily-bar depth per Alpaca USD pair")
    print(RULE)

    if args.candidates.strip():
        candidates = [
            symbol.strip().upper()
            for symbol in args.candidates.split(",")
            if symbol.strip()
        ]
        source = "command line"
    else:
        candidates, source = _candidate_symbols(credentials)
    print(f"  Probing {len(candidates)} candidates from the {source}.")

    bad = [s for s in candidates if validate_crypto_symbol(s) is not None]
    if bad:
        return _fail(f"Invalid symbols: {', '.join(bad)}")

    feed = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
    probe_start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    probe_end = datetime.now(timezone.utc) - timedelta(days=1)

    frames: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for symbol in candidates:
        try:
            bars = feed.get_bars(symbol, "1Day", start=probe_start, end=probe_end)
        except Exception as exc:  # noqa: BLE001 - report per symbol, keep probing
            rows.append({"symbol": symbol, "bars": 0, "first": None, "last": None,
                         "error": f"{type(exc).__name__}: {str(exc)[:90]}"})
            continue
        if bars.empty:
            rows.append({"symbol": symbol, "bars": 0, "first": None, "last": None,
                         "error": "no bars returned"})
            continue
        frames[symbol] = bars
        rows.append({
            "symbol": symbol,
            "bars": int(len(bars)),
            "first": f"{bars.index[0]:%Y-%m-%d}",
            "last": f"{bars.index[-1]:%Y-%m-%d}",
            "error": None,
        })

    rows.sort(key=lambda row: (-row["bars"], row["symbol"]))
    print(f"\n  {'Symbol':<12}{'Bars':>7}   {'First':<12}{'Last':<12}Note")
    for row in rows:
        note = row["error"] or ("OK" if row["bars"] >= args.min_bars else
                                f"below --min-bars {args.min_bars}")
        print(f"  {row['symbol']:<12}{row['bars']:>7}   "
              f"{row['first'] or '—':<12}{row['last'] or '—':<12}{note}")

    qualified = [row["symbol"] for row in rows if row["bars"] >= args.min_bars]
    if not qualified:
        return _fail(
            "No symbol met the depth requirement. Either every probe failed "
            "(see the notes above) or --min-bars is set beyond available history."
        )

    # The benchmark must be part of the universe; BTC/USD is the comparator
    # everywhere else in this repository.
    recommended = qualified[:MAX_UNIVERSE]
    if "BTC/USD" in qualified and "BTC/USD" not in recommended:
        recommended = ["BTC/USD"] + recommended[: MAX_UNIVERSE - 1]

    aligned = None
    for symbol in recommended:
        index = frames[symbol].index
        aligned = index if aligned is None else aligned.intersection(index)
    aligned_bars = int(len(aligned)) if aligned is not None else 0

    print(f"\n{RULE}\nRECOMMENDATION\n{RULE}")
    print(f"  {len(recommended)} of {len(qualified)} qualifying symbols "
          f"(engine limit is {MAX_UNIVERSE}).")
    print(f"  Aligned daily bars across the set: {aligned_bars}")
    print(f"  Fold arithmetic at 1500 train / 100 test: "
          f"{max(0, (aligned_bars - 1601) // 100 + 1)} folds")
    print("\n  --symbols \"" + ",".join(recommended) + "\"")
    print(RULE)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_bars": args.min_bars,
            "source": source,
            "probed": rows,
            "recommended": recommended,
            "aligned_bars": aligned_bars,
        }, indent=2, default=str))
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
