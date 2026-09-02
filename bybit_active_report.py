"""Time-causal reporting for an in-progress Bybit burn-in."""
from __future__ import annotations

from datetime import datetime, timezone

from bybit_report import BybitBurnInReport, _dt


class ActiveSafeBybitBurnInReport(BybitBurnInReport):
    """Never treat the planned future end as hours already observed."""

    def _window(self, run):
        start = _dt(run["start_time"])
        if start is None:
            raise ValueError("Burn-in start time is invalid")
        if str(run.get("status") or "").lower() == "running":
            return start, datetime.now(timezone.utc)
        end = _dt(run.get("stopped_at")) or _dt(run.get("planned_end_time"))
        return start, end or datetime.now(timezone.utc)
