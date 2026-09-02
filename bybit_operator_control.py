"""Local operator process controls for the simple Bybit Demo UI.

Secrets never cross this boundary. Streamlit may launch the backend with the server's
existing environment, but API key/secret values are not read, displayed, serialized,
or passed as command-line arguments.
"""
from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_CONTROL_DB = "data/bybit_operator.db"
DEFAULT_LOG = "logs/bybit_demo_burnin.log"


class OperatorControl:
    def __init__(self, path: str = DEFAULT_CONTROL_DB) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=10.0)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS operator_state (
            id INTEGER PRIMARY KEY CHECK(id=1),
            pid INTEGER,
            run_id TEXT,
            state TEXT NOT NULL DEFAULT 'STOPPED',
            updated_at TEXT NOT NULL,
            note TEXT
            )"""
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO operator_state(id,state,updated_at) VALUES(1,'STOPPED',?)",
            (self._now(),),
        )
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def close(self) -> None:
        self._conn.close()

    def status(self) -> dict:
        row = self._conn.execute("SELECT * FROM operator_state WHERE id=1").fetchone()
        keys = ["id", "pid", "run_id", "state", "updated_at", "note"]
        result = dict(zip(keys, row)) if row else {"state": "STOPPED", "pid": None}
        pid = result.get("pid")
        result["process_alive"] = self._alive(pid)
        if pid and not result["process_alive"] and result.get("state") == "RUNNING":
            self.mark_stopped("backend process is no longer alive")
            result = self.status()
        return result

    @staticmethod
    def _alive(pid: Optional[int]) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    def mark_running(self, pid: int, run_id: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE operator_state SET pid=?,run_id=?,state='RUNNING',updated_at=?,note=NULL WHERE id=1",
            (int(pid), run_id, self._now()),
        )
        self._conn.commit()

    def set_run_id(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE operator_state SET run_id=?,updated_at=? WHERE id=1",
            (run_id, self._now()),
        )
        self._conn.commit()

    def mark_stopped(self, note: str = "") -> None:
        self._conn.execute(
            "UPDATE operator_state SET pid=NULL,state='STOPPED',updated_at=?,note=? WHERE id=1",
            (self._now(), note[:250]),
        )
        self._conn.commit()

    def start_detached(
        self,
        *,
        hours: int,
        capital_usdt: float,
        repo_root: Optional[str] = None,
        smart_exits: bool = False,
        log_path: str = DEFAULT_LOG,
    ) -> int:
        current = self.status()
        if current.get("process_alive"):
            raise RuntimeError(f"Bybit Demo backend is already running as PID {current['pid']}")
        if hours not in {72, 120}:
            raise ValueError("Operator burn-in hours must be 72 or 120")
        if capital_usdt <= 0:
            raise ValueError("Capital allocation must be positive")
        root = Path(repo_root or Path(__file__).resolve().parent)
        log = Path(log_path)
        if not log.is_absolute():
            log = root / log
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(root / "scripts" / "run_bybit_demo_burnin.py"),
            "--hours",
            str(hours),
            "--capital-usdt",
            str(float(capital_usdt)),
        ]
        if smart_exits:
            command.append("--enable-researched-smart-exits")
        stream = open(log, "ab", buffering=0)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                env=os.environ.copy(),
            )
        finally:
            stream.close()
        self.mark_running(process.pid)
        return int(process.pid)

    def request_stop(self) -> None:
        status = self.status()
        pid = status.get("pid")
        if not pid or not status.get("process_alive"):
            self.mark_stopped("stop requested while backend was not running")
            return
        os.kill(int(pid), signal.SIGTERM)
        self._conn.execute(
            "UPDATE operator_state SET state='STOPPING',updated_at=?,note='graceful stop requested' WHERE id=1",
            (self._now(),),
        )
        self._conn.commit()

    def emergency_stop(self) -> None:
        status = self.status()
        pid = status.get("pid")
        if not pid or not status.get("process_alive"):
            self.mark_stopped("emergency stop requested while backend was not running")
            return
        sig = getattr(signal, "SIGUSR1", signal.SIGTERM)
        os.kill(int(pid), sig)
        self._conn.execute(
            "UPDATE operator_state SET state='HALTING',updated_at=?,note='emergency kill switch requested' WHERE id=1",
            (self._now(),),
        )
        self._conn.commit()
