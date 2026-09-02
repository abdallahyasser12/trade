"""Server-side operator controls for the simple Bybit Demo dashboard.

Secrets never cross this boundary. Streamlit launches the backend with the server's
existing environment; API key/secret values are never read, displayed, serialized,
or passed as command-line arguments.

The controller also owns the persistent emergency latch. Arming the kill switch is a
SQLite write performed *before* signalling the engine, so a lost signal or process
crash cannot allow a later restart to silently resume economic activity.
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_CONTROL_DB = "data/bybit_operator.db"
DEFAULT_ENGINE_DB = "data/bybit_demo_trading.db"
DEFAULT_LOG = "logs/bybit_demo_burnin.log"
KILL_SWITCH_KEY = "crypto_kill_switch"


class OperatorControl:
    def __init__(
        self,
        path: str = DEFAULT_CONTROL_DB,
        *,
        engine_db: str = DEFAULT_ENGINE_DB,
    ) -> None:
        self.path = path
        self.engine_db = engine_db
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(engine_db).parent.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def _pid_alive(pid: Optional[int]) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # The process exists but is not signalable by this user.
            return True
        except OSError:
            return False

    @classmethod
    def _owned_backend_alive(cls, pid: Optional[int]) -> bool:
        if not cls._pid_alive(pid):
            return False
        # Linux production containers expose /proc. Refuse to signal a recycled PID
        # unless it is still our burn-in command. Other platforms fall back to the
        # existence check because subprocess ownership cannot be inspected portably.
        proc = Path(f"/proc/{int(pid)}/cmdline") if pid else None
        if proc is not None and proc.exists():
            try:
                command = proc.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
            except OSError:
                return False
            return "run_bybit_demo_burnin.py" in command
        return True

    def status(self) -> dict:
        row = self._conn.execute("SELECT * FROM operator_state WHERE id=1").fetchone()
        keys = ["id", "pid", "run_id", "state", "updated_at", "note"]
        result = dict(zip(keys, row)) if row else {"state": "STOPPED", "pid": None}
        pid = result.get("pid")
        result["process_alive"] = self._owned_backend_alive(pid)
        result["kill_switch"] = self.read_kill_switch()
        if pid and not result["process_alive"] and result.get("state") in {
            "RUNNING",
            "STOPPING",
            "HALTING",
        }:
            self.mark_stopped("backend process is no longer alive")
            result = self.status()
        return result

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

    def _engine_kv(self, *, create: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(self.engine_db, timeout=10.0)
        if create:
            connection.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
            connection.commit()
        return connection

    def read_kill_switch(self) -> Optional[dict]:
        if not Path(self.engine_db).exists():
            return None
        connection = self._engine_kv()
        try:
            row = connection.execute(
                "SELECT value FROM kv WHERE key=?", (KILL_SWITCH_KEY,)
            ).fetchone()
        except sqlite3.Error:
            return {"armed": True, "reason": "KILL_SWITCH_STATE_UNREADABLE"}
        finally:
            connection.close()
        if not row or not row[0]:
            return None
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return {"armed": True, "reason": "KILL_SWITCH_STATE_CORRUPT"}
        return payload if isinstance(payload, dict) else {"armed": True, "reason": "KILL_SWITCH_STATE_CORRUPT"}

    def arm_kill_switch(self, reason: str = "EMERGENCY_STOP") -> None:
        payload = json.dumps(
            {
                "armed": True,
                "reason": reason,
                "at": self._now(),
                "source": "operator_dashboard",
            },
            sort_keys=True,
        )
        connection = self._engine_kv(create=True)
        try:
            connection.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (KILL_SWITCH_KEY, payload),
            )
            connection.commit()
        finally:
            connection.close()

    def acknowledge_kill_switch(self) -> None:
        status = self.status()
        if status.get("process_alive"):
            raise RuntimeError("Cannot clear the kill switch while the trading backend is running")
        connection = self._engine_kv(create=True)
        try:
            connection.execute("DELETE FROM kv WHERE key=?", (KILL_SWITCH_KEY,))
            connection.commit()
        finally:
            connection.close()

    def start_detached(
        self,
        *,
        hours: int,
        capital_usdt: float,
        symbols: tuple[str, ...],
        repo_root: Optional[str] = None,
        smart_exits: bool = False,
        log_path: str = DEFAULT_LOG,
    ) -> int:
        current = self.status()
        if current.get("process_alive"):
            raise RuntimeError(f"Bybit Demo backend is already running as PID {current['pid']}")
        kill = current.get("kill_switch")
        if kill and kill.get("armed"):
            raise RuntimeError(
                "Emergency kill switch is armed. Review its reason and acknowledge it before starting."
            )
        if hours not in {72, 120}:
            raise ValueError("Operator burn-in hours must be 72 or 120")
        if not (20.0 <= float(capital_usdt) <= 1_000_000.0):
            raise ValueError("Capital allocation must be within 20–1,000,000 USDT")
        allowed = {"BTC/USD", "ETH/USD", "SOL/USD"}
        normalized = tuple(dict.fromkeys(symbols))
        if not normalized or any(symbol not in allowed for symbol in normalized):
            raise ValueError("Operator universe must be a non-empty subset of BTC/ETH/SOL Spot")
        root = Path(repo_root or Path(__file__).resolve().parent).resolve()
        runner = (root / "scripts" / "run_bybit_demo_burnin.py").resolve()
        if not runner.is_file():
            raise RuntimeError(f"Bybit burn-in runner is missing at {runner}")
        log = Path(log_path)
        if not log.is_absolute():
            log = root / log
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(runner),
            "--hours",
            str(hours),
            "--capital-usdt",
            str(float(capital_usdt)),
            "--symbols",
            *normalized,
            "--control-db",
            str(Path(self.path).resolve()),
            "--state-db",
            str(Path(self.engine_db).resolve()),
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
        if not self._owned_backend_alive(pid):
            raise RuntimeError("Stored backend PID is not owned by this trading service")
        os.kill(int(pid), signal.SIGTERM)
        self._conn.execute(
            "UPDATE operator_state SET state='STOPPING',updated_at=?,note='graceful stop requested' WHERE id=1",
            (self._now(),),
        )
        self._conn.commit()

    def emergency_stop(self) -> None:
        # Persist first. Even if there is no live process, a later engine startup
        # remains blocked until the operator explicitly acknowledges the latch.
        self.arm_kill_switch("EMERGENCY_STOP")
        status = self.status()
        pid = status.get("pid")
        if pid and status.get("process_alive"):
            if not self._owned_backend_alive(pid):
                raise RuntimeError("Stored backend PID is not owned by this trading service")
            sig = getattr(signal, "SIGUSR1", signal.SIGTERM)
            os.kill(int(pid), sig)
        self._conn.execute(
            "UPDATE operator_state SET state='HALTING',updated_at=?,note='emergency kill switch armed' WHERE id=1",
            (self._now(),),
        )
        self._conn.commit()
