"""Server-side operator controls for the simple Bybit Demo dashboard.

The dashboard may accept Bybit *Demo* credentials from an operator, but credentials
never enter SQLite, reports, logs, URLs, subprocess arguments, or browser-visible
status payloads.  Saved credentials live in a dedicated server-side JSON vault with
0600 permissions.  The detached trading backend receives them only in its process
environment.

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
DEFAULT_VENUE_DB = "data/bybit_demo_venue.db"
DEFAULT_LOG = "logs/bybit_demo_burnin.log"
DEFAULT_CREDENTIALS_FILE = "secrets/bybit_demo_credentials.json"
KILL_SWITCH_KEY = "crypto_kill_switch"


class OperatorControl:
    def __init__(
        self,
        path: str = DEFAULT_CONTROL_DB,
        *,
        engine_db: str = DEFAULT_ENGINE_DB,
        venue_db: str = DEFAULT_VENUE_DB,
        credentials_path: str = DEFAULT_CREDENTIALS_FILE,
    ) -> None:
        self.path = path
        self.engine_db = engine_db
        self.venue_db = venue_db
        self.credentials_path = credentials_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(engine_db).parent.mkdir(parents=True, exist_ok=True)
        Path(venue_db).parent.mkdir(parents=True, exist_ok=True)
        credential_parent = Path(credentials_path).parent
        credential_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            credential_parent.chmod(0o700)
        except OSError:
            pass
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
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS operator_preferences (
            id INTEGER PRIMARY KEY CHECK(id=1),
            notifications_enabled INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO operator_preferences(id,notifications_enabled,updated_at) VALUES(1,1,?)",
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
            return True
        except OSError:
            return False

    @classmethod
    def _owned_backend_alive(cls, pid: Optional[int]) -> bool:
        if not cls._pid_alive(pid):
            return False
        proc = Path(f"/proc/{int(pid)}/cmdline") if pid else None
        if proc is not None and proc.exists():
            try:
                command = proc.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
            except OSError:
                return False
            return "run_bybit_demo_burnin.py" in command
        return True

    def _current_backend_alive(self) -> bool:
        row = self._conn.execute("SELECT pid FROM operator_state WHERE id=1").fetchone()
        return self._owned_backend_alive(row[0] if row else None)

    # ------------------------------------------------------------------
    # Demo credential vault
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_credential(value: str, label: str) -> str:
        token = str(value or "").strip()
        if len(token) < 8 or len(token) > 256:
            raise ValueError(f"{label} does not look like a valid Bybit Demo credential")
        if any(ch.isspace() or ord(ch) < 32 for ch in token):
            raise ValueError(f"{label} must not contain whitespace/control characters")
        return token

    def _read_saved_credentials(self) -> Optional[tuple[str, str]]:
        path = Path(self.credentials_path)
        if not path.exists():
            return None
        if path.is_symlink():
            raise RuntimeError("Refusing a symlinked Bybit credential vault")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Saved Bybit Demo credential vault is unreadable/corrupt") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise RuntimeError("Saved Bybit Demo credential vault has an unsupported format")
        key = self._validate_credential(payload.get("api_key", ""), "API key")
        secret = self._validate_credential(payload.get("api_secret", ""), "API secret")
        return key, secret

    def credentials_status(self) -> dict:
        try:
            saved = self._read_saved_credentials()
        except RuntimeError as exc:
            return {"configured": False, "source": "vault_error", "message": str(exc)}
        if saved is not None:
            return {"configured": True, "source": "dashboard_vault", "message": "Saved securely on this server"}
        env_key = os.environ.get("BYBIT_DEMO_API_KEY", "").strip()
        env_secret = os.environ.get("BYBIT_DEMO_API_SECRET", "").strip()
        if env_key and env_secret:
            return {"configured": True, "source": "server_environment", "message": "Configured by server environment"}
        if env_key or env_secret:
            return {"configured": False, "source": "partial_environment", "message": "Only one Demo credential is configured"}
        return {"configured": False, "source": "none", "message": "Add Bybit Demo API credentials below"}

    def load_demo_credentials(self) -> tuple[str, str]:
        saved = self._read_saved_credentials()
        if saved is not None:
            return saved
        key = os.environ.get("BYBIT_DEMO_API_KEY", "").strip()
        secret = os.environ.get("BYBIT_DEMO_API_SECRET", "").strip()
        if not key or not secret:
            raise RuntimeError("Bybit Demo API credentials are not configured")
        return self._validate_credential(key, "API key"), self._validate_credential(secret, "API secret")

    def save_demo_credentials(self, api_key: str, api_secret: str) -> None:
        if self._current_backend_alive():
            raise RuntimeError("Stop the Bybit Demo backend before changing API credentials")
        key = self._validate_credential(api_key, "API key")
        secret = self._validate_credential(api_secret, "API secret")
        path = Path(self.credentials_path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists() and path.is_symlink():
            raise RuntimeError("Refusing to replace a symlinked Bybit credential vault")
        tmp = path.with_name(path.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(tmp), flags, 0o600)
        try:
            payload = {
                "version": 1,
                "api_key": key,
                "api_secret": secret,
                "saved_at": self._now(),
                "environment": "BYBIT_DEMO",
            }
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                fd = -1
                json.dump(payload, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def clear_saved_demo_credentials(self) -> None:
        if self._current_backend_alive():
            raise RuntimeError("Stop the Bybit Demo backend before clearing API credentials")
        path = Path(self.credentials_path)
        if path.exists():
            if path.is_symlink():
                raise RuntimeError("Refusing to follow a symlinked Bybit credential vault")
            path.unlink()

    def test_demo_credentials(self, api_key: str, api_secret: str) -> dict:
        """Read-only Demo authentication/environment check; never submits an order."""
        key = self._validate_credential(api_key, "API key")
        secret = self._validate_credential(api_secret, "API secret")
        from bybit_spot import BybitSpotBroker
        from bybit_state import BybitStateStore
        from crypto_venue import CryptoExecutionEnvironment

        state = BybitStateStore(":memory:")
        broker = BybitSpotBroker(
            key,
            secret,
            environment=CryptoExecutionEnvironment.BYBIT_DEMO,
            state=state,
            strategy_capital_budget_usdt=100.0,
            permitted_symbols=("BTC/USD",),
            allow_live=False,
        )
        try:
            result = broker.verify_environment()
            return {
                "ok": True,
                "environment": result.get("environment"),
                "rest_host": result.get("rest_host"),
                "clock_skew_ms": result.get("clock_skew_ms"),
                "spot_only": result.get("spot_only") is True,
                "isLeverage": result.get("isLeverage"),
            }
        finally:
            broker.close()
            state.close()

    def test_saved_demo_credentials(self) -> dict:
        key, secret = self.load_demo_credentials()
        return self.test_demo_credentials(key, secret)

    # ------------------------------------------------------------------
    # Operator preferences
    # ------------------------------------------------------------------
    def notifications_enabled(self) -> bool:
        row = self._conn.execute(
            "SELECT notifications_enabled FROM operator_preferences WHERE id=1"
        ).fetchone()
        return bool(row[0]) if row else True

    def set_notifications_enabled(self, enabled: bool) -> None:
        self._conn.execute(
            "UPDATE operator_preferences SET notifications_enabled=?,updated_at=? WHERE id=1",
            (1 if enabled else 0, self._now()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Durable dashboard/runtime state
    # ------------------------------------------------------------------
    def _sync_dashboard_protection(self) -> None:
        """Mirror engine-authoritative current stop into the UI evidence DB."""
        if not Path(self.engine_db).exists() or not Path(self.venue_db).exists():
            return
        engine = sqlite3.connect(self.engine_db, timeout=2.0)
        venue = sqlite3.connect(self.venue_db, timeout=2.0)
        try:
            columns = {row[1] for row in venue.execute("PRAGMA table_info(management_state)").fetchall()}
            if not columns:
                return
            if "current_stop_price" not in columns:
                venue.execute("ALTER TABLE management_state ADD COLUMN current_stop_price REAL")
            rows = engine.execute("SELECT key,value FROM kv WHERE key LIKE 'bot_position:%'").fetchall()
            for key, raw in rows:
                try:
                    payload = json.loads(raw)
                    stop = float(payload.get("stop_price"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if stop <= 0:
                    continue
                symbol = str(key).split("bot_position:", 1)[-1]
                venue.execute(
                    "UPDATE management_state SET current_stop_price=?,updated_at=? WHERE strategy_symbol=?",
                    (stop, self._now(), symbol),
                )
            venue.commit()
        except sqlite3.Error:
            return
        finally:
            engine.close()
            venue.close()

    def status(self) -> dict:
        self._sync_dashboard_protection()
        row = self._conn.execute("SELECT * FROM operator_state WHERE id=1").fetchone()
        keys = ["id", "pid", "run_id", "state", "updated_at", "note"]
        result = dict(zip(keys, row)) if row else {"state": "STOPPED", "pid": None}
        pid = result.get("pid")
        result["process_alive"] = self._owned_backend_alive(pid)
        result["kill_switch"] = self.read_kill_switch()
        if pid and not result["process_alive"] and result.get("state") in {"RUNNING", "STOPPING", "HALTING"}:
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
            row = connection.execute("SELECT value FROM kv WHERE key=?", (KILL_SWITCH_KEY,)).fetchone()
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
            {"armed": True, "reason": reason, "at": self._now(), "source": "operator_dashboard"},
            sort_keys=True,
        )
        connection = self._engine_kv(create=True)
        try:
            connection.execute(
                "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
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
            raise RuntimeError("Emergency kill switch is armed. Review its reason and acknowledge it before starting.")
        if hours not in {72, 120}:
            raise ValueError("Operator burn-in hours must be 72 or 120")
        if not (20.0 <= float(capital_usdt) <= 1_000_000.0):
            raise ValueError("Capital allocation must be within 20–1,000,000 USDT")
        allowed = {"BTC/USD", "ETH/USD", "SOL/USD"}
        normalized = tuple(dict.fromkeys(symbols))
        if not normalized or any(symbol not in allowed for symbol in normalized):
            raise ValueError("Operator universe must be a non-empty subset of BTC/ETH/SOL Spot")
        key, secret = self.load_demo_credentials()
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
            "--venue-db",
            str(Path(self.venue_db).resolve()),
        ]
        if smart_exits:
            command.append("--enable-researched-smart-exits")
        child_env = os.environ.copy()
        child_env["BYBIT_DEMO_API_KEY"] = key
        child_env["BYBIT_DEMO_API_SECRET"] = secret
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
                env=child_env,
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
