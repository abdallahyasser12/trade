#!/usr/bin/env bash
# Run every verification this repository supports and write one shareable report.
#
#   ./scripts/run_all_checks.sh              # tests, plus research if creds exist
#   ./scripts/run_all_checks.sh --no-research # tests only, no network, no creds
#   ./scripts/run_all_checks.sh --quick       # skip the slow strict-cost research pass
#
# Designed for Debian-family systems (Kali, Ubuntu) where PEP 668 blocks
# system-wide pip: everything installs into a local .venv.
#
# Credentials are read but NEVER printed. The report is scrubbed of any
# credential value before it is written, so it is safe to share.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO="$PWD"

RUN_RESEARCH=1
QUICK=0
for arg in "$@"; do
  case "$arg" in
    --no-research) RUN_RESEARCH=0 ;;
    --quick) QUICK=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="$REPO/check-report-$STAMP.txt"
VENV="$REPO/.venv"

PASSED=(); FAILED=(); SKIPPED=()

log()  { printf '%s\n' "$*" | tee -a "$REPORT" >/dev/null; }
say()  { printf '%s\n' "$*"; log "$*"; }
rule() { say "=============================================================================="; }

# Run a named section, streaming output to the terminal and the report.
section() {
  local name="$1"; shift
  rule; say "## $name"; rule
  local start; start=$(date +%s)
  if "$@" >>"$REPORT" 2>&1; then
    local secs=$(( $(date +%s) - start ))
    say "   -> PASS (${secs}s)"; PASSED+=("$name"); return 0
  else
    local code=$? secs=$(( $(date +%s) - start ))
    say "   -> FAIL (exit $code, ${secs}s)"; FAILED+=("$name ,exit $code"); return $code
  fi
}

skip() { say "   -> SKIPPED: $2"; SKIPPED+=("$1 ,$2"); }

: >"$REPORT"
rule
say "TRADE BOT VERIFICATION REPORT"
rule
say "  UTC time     $(date -u '+%Y-%m-%d %H:%M:%S')"
say "  Host OS      $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
say "  Kernel       $(uname -r)"
say "  Repo         $REPO"
say "  Branch       $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not a git repo')"
say "  Commit       $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
say "  Tree         $(if [ -z "$(git status --porcelain 2>/dev/null)" ]; then echo clean; else echo 'has local changes'; fi)"
say ""

# ---------------------------------------------------------------- interpreter
rule; say "## Python interpreter"; rule
PY_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,11) else 1)' 2>/dev/null; then
      PY_BIN="$candidate"; break
    fi
  fi
done
if [ -z "$PY_BIN" ]; then
  say "   -> FAIL: no Python 3.11+ found."
  say "      Install one with:  sudo apt update && sudo apt install -y python3 python3-venv"
  exit 2
fi
say "  Using $PY_BIN ($("$PY_BIN" --version 2>&1))"

# ------------------------------------------------------------------ virtualenv
rule; say "## Virtual environment"; rule
if [ ! -d "$VENV" ]; then
  say "  Creating $VENV"
  if ! "$PY_BIN" -m venv "$VENV" >>"$REPORT" 2>&1; then
    say "   -> FAIL: could not create a venv."
    say "      Kali/Debian usually needs:  sudo apt install -y python3-venv"
    exit 2
  fi
else
  say "  Reusing existing $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { say "   -> FAIL: could not activate the venv."; exit 2; }
say "  Active: $(python --version 2>&1)"

# ---------------------------------------------------------------- dependencies
say ""
section "Install dependencies" bash -c '
  python -m pip install --upgrade pip >/dev/null 2>&1
  python -m pip install -r requirements.txt
' || {
  say ""
  say "  Dependency install failed. The usual cause on a rolling distro is that"
  say "  the pinned wheels have no build for your Python version. Check the log"
  say "  above, then try a 3.11 or 3.12 interpreter."
}

# Mobile API extras are optional; their absence only skips two test files.
MOBILE_OK=0
if python -m pip install -r requirements-mobile-api.txt >>"$REPORT" 2>&1; then
  MOBILE_OK=1
  say "  Mobile API extras installed (mobile tests will run)."
else
  say "  Mobile API extras unavailable — mobile tests will be deselected."
fi

# ------------------------------------------------------------------- integrity
say ""
section "Byte-compile every module" python -m compileall -q .

# ----------------------------------------------------------------- test suite
say ""
if [ "$MOBILE_OK" -eq 1 ]; then
  section "Test suite" python -m pytest -q --durations=10
else
  section "Test suite (excluding mobile)" \
    python -m pytest -q --durations=10 \
      --ignore=tests/test_mobile_api.py --ignore=tests/test_mobile_push.py
fi

# --------------------------------------------------------------- cost arithmetic
say ""
section "Cost-versus-risk arithmetic" python - <<'PYEOF'
from cost_model import break_even_win_rate, cost_to_risk, round_trip_cost_bps

rt = round_trip_cost_bps(slippage_bps=10.0, spread_bps=5.0, taker_fee_bps=25.0)
print(f"Round-trip friction at shipped defaults: {rt:.1f} bps\n")
print(f"{'ATR/price':>10}{'1R (bps)':>12}{'cost in R':>12}{'break-even win rate @2R':>26}")
for atr_pct in (0.25, 0.35, 0.50, 0.75, 1.00):
    entry = 100.0
    stop = entry * (1 - 2.5 * atr_pct / 100.0)
    c = cost_to_risk(entry_price=entry, stop_price=stop, round_trip_bps=rt)
    w = break_even_win_rate(c, reward_to_risk=2.0)
    print(f"{atr_pct:>9.2f}%{2.5*atr_pct*100:>12.1f}{c:>12.2f}{w:>25.1%}")
print("\nTrend-following systems typically win 35-45% of trades.")
PYEOF

# -------------------------------------------------------------------- research
say ""
rule; say "## Walk-forward research (real market data)"; rule

set -a
[ -f "$REPO/.env" ] && . "$REPO/.env" 2>/dev/null
set +a
KEY="${ALPACA_PAPER_API_KEY:-${ALPACA_API_KEY:-}}"
SECRET="${ALPACA_PAPER_SECRET_KEY:-${ALPACA_SECRET_KEY:-}}"

if [ "$RUN_RESEARCH" -eq 0 ]; then
  skip "Walk-forward research" "--no-research was passed"
elif [ -z "$KEY" ] || [ -z "$SECRET" ]; then
  skip "Walk-forward research" "no Alpaca paper credentials in the environment or .env"
  say ""
  say "  To include the research pass, export your PAPER keys and re-run:"
  say "    export ALPACA_PAPER_API_KEY=...     # data access only"
  say "    export ALPACA_PAPER_SECRET_KEY=..."
  say "  Nothing here can place an order: the script never constructs a broker."
else
  say "  Credentials found (values are never printed)."
  say ""
  say "  Pass 1 of 2 — current defaults, max_cost_to_risk = 1.00"
  python scripts/run_crypto_research.py --json-out "research-default-$STAMP.json" \
    >>"$REPORT" 2>&1
  code=$?
  case $code in
    0) say "   -> PROMOTION PASS"; PASSED+=("Research (defaults): PASS") ;;
    1) say "   -> PROMOTION FAIL (a normal research outcome)"; PASSED+=("Research (defaults): gates failed") ;;
    *) say "   -> could not complete (exit $code)"; FAILED+=("Research (defaults) ,exit $code") ;;
  esac

  if [ "$QUICK" -eq 1 ]; then
    skip "Research (strict cost)" "--quick was passed"
  else
    say ""
    say "  Pass 2 of 2 — strict friction gate, max_cost_to_risk = 0.25"
    python scripts/run_crypto_research.py --max-cost-to-risk 0.25 \
      --json-out "research-strict-$STAMP.json" >>"$REPORT" 2>&1
    code=$?
    case $code in
      0) say "   -> PROMOTION PASS"; PASSED+=("Research (strict): PASS") ;;
      1) say "   -> PROMOTION FAIL (a normal research outcome)"; PASSED+=("Research (strict): gates failed") ;;
      *) say "   -> could not complete (exit $code)"; FAILED+=("Research (strict) ,exit $code") ;;
    esac
  fi
fi

# --------------------------------------------------------------------- summary
say ""
rule; say "SUMMARY"; rule
for item in "${PASSED[@]:-}";  do [ -n "$item" ] && say "  PASS     ${item/ ,/  }"; done
for item in "${SKIPPED[@]:-}"; do [ -n "$item" ] && say "  SKIP     ${item/ ,/  — }"; done
for item in "${FAILED[@]:-}";  do [ -n "$item" ] && say "  FAIL     ${item/ ,/  — }"; done
say ""
if [ "${#FAILED[@]}" -eq 0 ]; then
  say "  No failures. Note that a promotion FAIL is a research result, not a defect."
else
  say "  ${#FAILED[@]} section(s) failed — see the detail above."
fi

# ------------------------------------------------------- scrub before sharing
# Replace any credential value that reached the log with a placeholder.
for value in "$KEY" "$SECRET" "${ALPACA_LIVE_API_KEY:-}" "${ALPACA_LIVE_SECRET_KEY:-}" \
             "${MOBILE_API_TOKEN:-}" "${EXPO_ACCESS_TOKEN:-}"; do
  if [ -n "${value:-}" ] && [ "${#value}" -ge 8 ]; then
    python - "$REPORT" "$value" <<'PYEOF'
import sys, pathlib
path, secret = pathlib.Path(sys.argv[1]), sys.argv[2]
path.write_text(path.read_text(errors="replace").replace(secret, "***REDACTED***"))
PYEOF
  fi
done
# Belt and braces: mask anything that still looks like a long opaque token.
python - "$REPORT" <<'PYEOF'
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
text = path.read_text(errors="replace")
text = re.sub(r'\b(?:PK|SK|AK)[A-Z0-9]{12,}\b', '***REDACTED***', text)
path.write_text(text)
PYEOF

say ""
rule
say "Report written to:"
say "  $REPORT"
say ""
say "Credential values have been scrubbed. Review it once, then share it."
rule
