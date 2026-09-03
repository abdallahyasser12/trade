"""Time-causal reporting for an in-progress Bybit burn-in."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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

    def write_snapshot(
        self,
        run_id: str,
        reports_dir: str = "reports",
        *,
        label: str | None = None,
    ) -> tuple[Path, Path, dict]:
        """Write an immutable in-progress JSON/Markdown snapshot.

        The snapshot uses current UTC time as the evidence cutoff while the run is
        active. It never overwrites the final report name.
        """
        report = self.build(run_id)
        out = Path(reports_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(ch for ch in suffix if ch.isalnum() or ch in {"-", "_"})
        if not safe:
            raise ValueError("Snapshot label must contain a safe filename character")
        json_path = out / f"bybit_burnin_{run_id}_{safe}.json"
        md_path = out / f"bybit_burnin_{run_id}_{safe}.md"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(self.markdown(report), encoding="utf-8")
        return json_path, md_path, report
