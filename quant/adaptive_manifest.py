"""Immutable research/deployment identity for adaptive intraday v2.

A profitable report is meaningless if production later changes timeframe, universe,
parameters or code.  Every recorded adaptive research run therefore gets a manifest.
RESEARCH -> SHADOW locks one passing manifest, and every forward runtime must match
it exactly.  A measured-cost pre-CANARY rerun must evaluate the same locked model.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from quant.strategies.intraday_adaptive import AdaptiveIntradayConfig

ADAPTIVE_STRATEGY_ID = "crypto_intraday_adaptive"
ADAPTIVE_STRATEGY_VERSION = "2.0.0"
ADAPTIVE_DEPLOYMENT_MANIFEST_KEY = "crypto_intraday_adaptive.deployment_manifest.v2"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def research_manifest_key(run_id: int) -> str:
    if int(run_id) <= 0:
        raise ValueError("Research run id must be positive.")
    return f"crypto_intraday_adaptive.research_manifest.v2.{int(run_id)}"


def current_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip().lower()
    return value if _GIT_SHA.fullmatch(value) else None


@dataclass(frozen=True)
class AdaptiveManifest:
    research_run_id: int
    git_commit: str
    timeframe: str
    universe: tuple[str, ...]
    params: dict
    strategy_id: str = ADAPTIVE_STRATEGY_ID
    strategy_version: str = ADAPTIVE_STRATEGY_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.research_run_id <= 0:
            errors.append("research_run_id must be positive")
        if self.strategy_id != ADAPTIVE_STRATEGY_ID:
            errors.append("strategy_id mismatch")
        if self.strategy_version != ADAPTIVE_STRATEGY_VERSION:
            errors.append("strategy_version mismatch")
        if not _GIT_SHA.fullmatch(str(self.git_commit).lower()):
            errors.append("git_commit must be a full 40-character SHA")
        if self.timeframe not in {"5Min", "15Min"}:
            errors.append("timeframe must be 5Min or 15Min")
        if not 6 <= len(self.universe) <= 12:
            errors.append("universe must contain 6-12 symbols")
        if len(set(self.universe)) != len(self.universe):
            errors.append("universe symbols must be unique")
        if "BTC/USD" not in self.universe:
            errors.append("universe must contain BTC/USD")
        try:
            config = AdaptiveIntradayConfig(**dict(self.params))
            errors.extend(config.validate())
        except (TypeError, ValueError) as error:
            errors.append(f"adaptive params invalid: {error}")
        return errors

    def require_valid(self) -> "AdaptiveManifest":
        errors = self.validate()
        if errors:
            raise ValueError("Invalid adaptive manifest: " + "; ".join(errors))
        return self

    def to_json(self) -> str:
        self.require_valid()
        payload = asdict(self)
        payload["universe"] = list(self.universe)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "AdaptiveManifest":
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Adaptive manifest JSON is unreadable.") from error
        if not isinstance(payload, dict):
            raise ValueError("Adaptive manifest must be a JSON object.")
        payload = dict(payload)
        universe = payload.get("universe")
        if not isinstance(universe, list):
            raise ValueError("Adaptive manifest universe must be a list.")
        payload["universe"] = tuple(str(symbol) for symbol in universe)
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("Adaptive manifest params must be an object.")
        payload["params"] = dict(params)
        manifest = cls(**payload)
        return manifest.require_valid()

    def same_strategy(self, other: "AdaptiveManifest") -> bool:
        return (
            self.strategy_id == other.strategy_id
            and self.strategy_version == other.strategy_version
            and self.git_commit == other.git_commit
            and self.timeframe == other.timeframe
            and self.universe == other.universe
            and self.params == other.params
        )

    def assert_runtime_identity(
        self,
        *,
        git_commit: Optional[str],
        timeframe: Optional[str] = None,
        universe: Optional[tuple[str, ...]] = None,
    ) -> None:
        if git_commit is None or git_commit.lower() != self.git_commit.lower():
            raise ValueError(
                "Runtime git commit does not match the promoted adaptive research "
                f"commit. expected={self.git_commit}, actual={git_commit}"
            )
        if timeframe is not None and timeframe != self.timeframe:
            raise ValueError(
                f"Runtime timeframe {timeframe} does not match locked {self.timeframe}."
            )
        if universe is not None and tuple(universe) != self.universe:
            raise ValueError(
                "Runtime universe does not exactly match the promoted research "
                f"universe. expected={self.universe}, actual={tuple(universe)}"
            )


def record_research_manifest(store, manifest: AdaptiveManifest) -> None:
    manifest.require_valid()
    store.set_kv(research_manifest_key(manifest.research_run_id), manifest.to_json())


def load_research_manifest(store, run_id: int) -> Optional[AdaptiveManifest]:
    raw = store.get_kv(research_manifest_key(run_id))
    if not raw:
        return None
    return AdaptiveManifest.from_json(raw)


def lock_deployment_manifest(store, manifest: AdaptiveManifest) -> None:
    manifest.require_valid()
    store.set_kv(ADAPTIVE_DEPLOYMENT_MANIFEST_KEY, manifest.to_json())


def locked_deployment_manifest(store) -> Optional[AdaptiveManifest]:
    raw = store.get_kv(ADAPTIVE_DEPLOYMENT_MANIFEST_KEY)
    if not raw:
        return None
    return AdaptiveManifest.from_json(raw)


def require_locked_manifest(store) -> AdaptiveManifest:
    manifest = locked_deployment_manifest(store)
    if manifest is None:
        raise ValueError(
            "Adaptive deployment manifest is not locked. Promote a passing RESEARCH "
            "run to SHADOW first."
        )
    return manifest
