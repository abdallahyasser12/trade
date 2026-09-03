#!/usr/bin/env python3
"""Run the complete hermetic Bybit/legacy acceptance suite and write evidence.

This is the source-controlled acceptance entrypoint used locally and by Python CI. It
never selects Bybit Live. The optional Demo economic smoke remains explicitly opt-in
and requires the same exact acknowledgement as ``scripts/bybit_demo_smoke.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DEMO_ACK = "I_ACCEPT_BYBIT_DEMO_ORDER"

BYBIT_TESTS = [
    "tests/test_bybit_spot.py",
    "tests/test_bybit_state.py",
    "tests/test_bybit_market_data.py",
    "tests/test_bybit_management.py",
    "tests/test_bybit_entry_intelligence.py",
    "tests/test_bybit_burnin.py",
    "tests/test_bybit_operator_control.py",
    "tests/test_bybit_security_contract.py",
    "tests/test_bybit_chaos.py",
    "tests/test_mobile_push.py",
]

_PYTEST_TOTAL_PATTERN = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<kind>passed|failed|skipped|xfailed|xpassed|errors?|warnings?)\b",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _redact(text: str, values: Iterable[str]) -> str:
    result = str(text or "")
    for value in values:
        token = str(value or "")
        if token:
            result = result.replace(token, "<redacted>")
    return result


def _pytest_totals(output: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for match in _PYTEST_TOTAL_PATTERN.finditer(output):
        kind = match.group("kind").lower()
        if kind == "error":
            kind = "errors"
        elif kind == "warning":
            kind = "warnings"
        totals[kind] = max(totals.get(kind, 0), int(match.group("count")))
    return totals


def _run(
    name: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    redact_values: Iterable[str] = (),
) -> dict[str, Any]:
    started = _now()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        output = _redact(
            ((result.stdout or "") + "\n" + (result.stderr or "")).strip(),
            redact_values,
        )
        record: dict[str, Any] = {
            "name": name,
            "command": command,
            "started_at": started,
            "finished_at": _now(),
            "returncode": result.returncode,
            "passed": result.returncode == 0,
            "output_tail": output[-12000:],
        }
        if "pytest" in command:
            record["test_totals"] = _pytest_totals(output)
        return record
    except OSError as exc:
        return {
            "name": name,
            "command": command,
            "started_at": started,
            "finished_at": _now(),
            "returncode": None,
            "passed": False,
            "output_tail": _redact(str(exc), redact_values),
        }


def _static_secret_check() -> dict[str, Any]:
    tracked = _run(
        "tracked_private_files",
        ["git", "ls-files", "--", ".env", "secrets/**"],
    )
    exposed = [line for line in tracked["output_tail"].splitlines() if line.strip()]
    tracked["passed"] = tracked["returncode"] == 0 and not exposed
    if exposed:
        tracked["output_tail"] = "Tracked private files: " + ", ".join(exposed)
    else:
        tracked["output_tail"] = "No tracked .env or secrets/ files."
    return tracked


def _compose_check() -> dict[str, Any]:
    """Validate production Compose from a clean checkout without requiring private .env."""
    if shutil.which("docker") is None:
        return {
            "name": "docker_compose_config",
            "passed": True,
            "skipped": True,
            "reason": "docker is not installed; source-level security contract tests still run",
        }

    source = (ROOT / "docker-compose.production.yml").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="trade-bybit-acceptance-") as temp_dir:
        temp = Path(temp_dir)
        env_file = temp / "acceptance.env"
        env_file.write_text(
            "TRADE_DOMAIN=localhost\n"
            "TRADE_DASHBOARD_PASSWORD=acceptance-local-only\n"
            "MOBILE_API_TOKEN=acceptance-local-mobile-token-not-for-deployment\n",
            encoding="utf-8",
        )
        # Service-level env_file references are independent of Compose interpolation.
        # Point them at an ephemeral non-secret file rather than creating/overwriting
        # the operator's private ROOT/.env.
        compose_text = source.replace(
            "      - .env\n", f"      - {env_file.as_posix()}\n"
        )
        compose_file = temp / "docker-compose.acceptance.yml"
        compose_file.write_text(compose_text, encoding="utf-8")

        env = os.environ.copy()
        env["TRADE_DOMAIN"] = "localhost"
        env["TRADE_DASHBOARD_PASSWORD"] = "acceptance-local-only"
        env["MOBILE_API_TOKEN"] = "acceptance-local-mobile-token-not-for-deployment"
        return _run(
            "docker_compose_config",
            [
                "docker",
                "compose",
                "--project-directory",
                str(ROOT),
                "-f",
                str(compose_file),
                "config",
                "--quiet",
            ],
            env=env,
        )


def _full_regression_totals(checks: list[dict[str, Any]]) -> dict[str, int]:
    for check in checks:
        if check.get("name") == "full_alpaca_quant_mobile_bybit_regression":
            return dict(check.get("test_totals") or {})
    return {}


def _write_report(report: dict[str, Any]) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    safe_sha = str(report["git_sha"])[:12]
    json_path = REPORTS / f"bybit_acceptance_{safe_sha}.json"
    md_path = REPORTS / f"bybit_acceptance_{safe_sha}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    totals = report.get("full_regression_test_totals") or {}
    total_text = ", ".join(f"{key}={value}" for key, value in sorted(totals.items())) or "unavailable"
    lines = [
        "# Bybit production acceptance evidence",
        "",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Hermetic verdict: **{report['hermetic_verdict']}**",
        f"- Full regression totals: **{total_text}**",
        f"- Demo smoke: **{report['demo_smoke']['status']}**",
        "- Bybit Live: **DISABLED / NOT TESTED**",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        state = "SKIPPED" if check.get("skipped") else ("PASS" if check.get("passed") else "FAIL")
        lines.append(f"- **{state}** — {check['name']}")
        if check.get("test_totals"):
            values = ", ".join(
                f"{key}={value}" for key, value in sorted(check["test_totals"].items())
            )
            lines.append(f"  - pytest totals: {values}")
        if check.get("reason"):
            lines.append(f"  - {check['reason']}")
    lines += [
        "",
        "## Acceptance boundary",
        "",
        "A passing hermetic suite proves repository wiring/tests only. The production checklist still requires a real Bybit Demo economic smoke for the configured credentials and a genuine 72–120 hour forward burn-in. It does not prove profitability.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-smoke",
        action="store_true",
        help="Also run the real Bybit Demo economic smoke after hermetic tests pass.",
    )
    parser.add_argument(
        "--confirm-demo-order",
        default="",
        help=f"Required with --demo-smoke: {DEMO_ACK}",
    )
    parser.add_argument(
        "--smoke-symbol",
        choices=("BTC/USD", "ETH/USD", "SOL/USD"),
        default="BTC/USD",
    )
    parser.add_argument("--smoke-cap-usdt", type=float, default=25.0)
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    checks.append(
        _run(
            "byte_compile",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "-x",
                r"(^|/)(\.venv|venv|node_modules|mobile/ios|mobile/android)(/|$)",
                ".",
            ],
        )
    )
    checks.append(
        _run(
            "bybit_dashboard_safety_suite",
            [sys.executable, "-m", "pytest", "-q", *BYBIT_TESTS],
        )
    )
    checks.append(
        _run(
            "full_alpaca_quant_mobile_bybit_regression",
            [sys.executable, "-m", "pytest", "-q", "--durations=10"],
        )
    )
    checks.append(_static_secret_check())
    checks.append(_compose_check())

    hermetic_ok = all(check.get("passed", False) for check in checks)
    demo_smoke: dict[str, Any] = {"status": "NOT_RUN", "passed": False}
    if args.demo_smoke:
        if args.confirm_demo_order != DEMO_ACK:
            demo_smoke = {
                "status": "REFUSED",
                "passed": False,
                "reason": f"--confirm-demo-order must equal {DEMO_ACK}",
            }
        elif not hermetic_ok:
            demo_smoke = {
                "status": "BLOCKED",
                "passed": False,
                "reason": "Hermetic acceptance failed; economic smoke was not attempted.",
            }
        else:
            smoke_env = os.environ.copy()
            redact_values = (
                smoke_env.get("BYBIT_DEMO_API_KEY", ""),
                smoke_env.get("BYBIT_DEMO_API_SECRET", ""),
            )
            smoke = _run(
                "bybit_demo_economic_smoke",
                [
                    sys.executable,
                    "scripts/bybit_demo_smoke.py",
                    "--confirm",
                    DEMO_ACK,
                    "--symbol",
                    args.smoke_symbol,
                    "--max-notional-usdt",
                    str(args.smoke_cap_usdt),
                ],
                env=smoke_env,
                redact_values=redact_values,
            )
            demo_smoke = {
                "status": "PASS" if smoke["passed"] else "FAIL",
                "passed": smoke["passed"],
                "output_tail": smoke["output_tail"],
            }

    report = {
        "schema_version": 2,
        "generated_at": _now(),
        "git_sha": _git_sha(),
        "hermetic_verdict": "PASS" if hermetic_ok else "FAIL",
        "full_regression_test_totals": _full_regression_totals(checks),
        "checks": checks,
        "demo_smoke": demo_smoke,
        "live_trading_tested": False,
        "profitability_claim": False,
    }
    json_path, md_path = _write_report(report)
    print(f"Acceptance JSON: {json_path}")
    print(f"Acceptance Markdown: {md_path}")
    print(f"Hermetic verdict: {report['hermetic_verdict']}")
    totals = report["full_regression_test_totals"]
    print(
        "Full regression totals: "
        + (", ".join(f"{k}={v}" for k, v in sorted(totals.items())) or "unavailable")
    )
    print(f"Demo smoke: {demo_smoke['status']}")
    return 0 if hermetic_ok and (not args.demo_smoke or demo_smoke.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
