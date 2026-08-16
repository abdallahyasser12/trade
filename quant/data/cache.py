"""On-disk cache for historical bars, keyed by the request that made them.

Research re-runs the same fetch constantly: a parameter sweep over one
symbol-year hits the provider once per candidate unless something
remembers. Worse, an un-cached sweep is not reproducible — the provider's
answer can change between runs (late corrections, adjustment changes),
so two "identical" experiments quietly disagree.

The cache is keyed by ``BarProvenance.fingerprint()``, which hashes the
request (provider, feed, symbol, timeframe, window, adjustment) and NOT
the fetch time. Entries are immutable once written; a caller that needs
fresh data changes the window or passes ``refresh=True``.

Storage is Parquet when pyarrow is importable and CSV otherwise, so the
cache works on a bare ``requirements.txt`` install. The sidecar JSON
holds the provenance so a cached read reproduces the original metadata
rather than inventing new metadata at read time.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from quant.data.provenance import BarProvenance, Completeness, DataQuality

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "data/bar_cache"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return False
    return True


class BarCache:
    """Immutable, request-addressed bar cache on the local filesystem."""

    def __init__(self, root: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)
        self._use_parquet = _parquet_available()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _entry_dir(self, fingerprint: str) -> Path:
        # Two-level fan-out keeps directory listings small when a sweep
        # caches thousands of symbol-windows.
        return self.root / fingerprint[:2] / fingerprint

    def _bars_path(self, fingerprint: str) -> Path:
        suffix = "parquet" if self._use_parquet else "csv"
        return self._entry_dir(fingerprint) / f"bars.{suffix}"

    def _meta_path(self, fingerprint: str) -> Path:
        return self._entry_dir(fingerprint) / "provenance.json"

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def has(self, fingerprint: str) -> bool:
        return self._bars_path(fingerprint).exists() and self._meta_path(fingerprint).exists()

    def get(
        self, fingerprint: str
    ) -> Optional[tuple[pd.DataFrame, BarProvenance]]:
        """Return the cached series, or None on any miss or damage.

        A damaged entry is a cache miss, never an exception: the caller
        refetches. It is logged, because a cache that silently misses
        every time is a performance bug worth seeing.
        """
        if not self.has(fingerprint):
            return None
        try:
            bars = self._read_bars(self._bars_path(fingerprint))
            meta = json.loads(self._meta_path(fingerprint).read_text())
            return bars, _provenance_from_dict(meta)
        except Exception as error:  # damaged entry -> treat as a miss
            logger.warning("Bar cache entry %s unreadable (%s); refetching", fingerprint, error)
            return None

    def put(self, provenance: BarProvenance, bars: pd.DataFrame) -> str:
        """Store ``bars`` under the provenance fingerprint, atomically.

        Writes go to a temp file and are renamed into place so a crash
        mid-write cannot leave a half-written frame that later reads as
        a short history.
        """
        fingerprint = provenance.fingerprint()
        entry = self._entry_dir(fingerprint)
        entry.mkdir(parents=True, exist_ok=True)
        self._write_atomic(self._bars_path(fingerprint), bars)
        self._write_text_atomic(
            self._meta_path(fingerprint),
            json.dumps(provenance.to_dict(), indent=2, sort_keys=True),
        )
        return fingerprint

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    # ------------------------------------------------------------------
    # Storage backends
    # ------------------------------------------------------------------

    def _read_bars(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, index_col=0, parse_dates=[0])
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        frame.index.name = "timestamp"
        columns = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
        return frame[columns].astype(float)

    def _write_atomic(self, path: Path, bars: pd.DataFrame) -> None:
        handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            if path.suffix == ".parquet":
                bars.to_parquet(tmp)
            else:
                bars.to_csv(tmp)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _write_text_atomic(self, path: Path, text: str) -> None:
        handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            tmp.write_text(text)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()


def _provenance_from_dict(data: dict) -> BarProvenance:
    def _dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    completeness = data.get("completeness") or {}
    return BarProvenance(
        provider=data["provider"],
        feed=data["feed"],
        market=data["market"],
        symbol=data["symbol"],
        timeframe=data["timeframe"],
        requested_start=_dt(data["requested_start"]),
        requested_end=_dt(data["requested_end"]),
        observed_start=_dt(data.get("observed_start")),
        observed_end=_dt(data.get("observed_end")),
        adjustment=data.get("adjustment"),
        fetched_at=_dt(data.get("fetched_at")) or datetime.now(timezone.utc),
        quality=DataQuality(data.get("quality", DataQuality.OK.value)),
        completeness=Completeness(
            expected_bars=int(completeness.get("expected_bars", 0)),
            observed_bars=int(completeness.get("observed_bars", 0)),
            missing_sessions=tuple(completeness.get("missing_sessions", ())),
            coverage=float(completeness.get("coverage", 0.0)),
        ),
        regular_hours_only=bool(data.get("regular_hours_only", True)),
        notes=tuple(data.get("notes", ())),
    )
