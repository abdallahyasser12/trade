"""Immutable research/deployment identity for market-wide US-equity strategies.

A generic strategy id is not enough evidence for deployment: changing the code,
timeframe or tradable universe after a passing walk-forward invalidates the test.  Each
market-wide equity research run therefore records an immutable manifest, and promotion
to SHADOW locks that exact identity.  Forward runtimes may *remove* temporarily
unexecutable symbols from a locked universe but may never add symbols that were not in
the promoted research run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

from quant.adaptive_manifest import current_git_commit
from quant.equity_marketwide import MarketWideEquityConfig

EQUITY_STRATEGY_IDS = frozenset(
    {"opening_range_breakout", "vwap_reversion", "volatility_squeeze"}
)
EQUITY_STRATEGY_VERSION = "1.0.0"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def research_manifest_key(strategy_id: str, run_id: int) -> str:
    if strategy_id not in EQUITY_STRATEGY_IDS:
        raise ValueError(f"Unsupported market-wide equity strategy {strategy_id!r}.")
    if int(run_id) <= 0:
        raise ValueError("Research run id must be positive.")
    return f"marketwide_equity.{strategy_id}.research_manifest.v1.{int(run_id)}"


def deployment_manifest_key(strategy_id: str) -> str:
    if strategy_id not in EQUITY_STRATEGY_IDS:
        raise ValueError(f"Unsupported market-wide equity strategy {strategy_id!r}.")
    return f"marketwide_equity.{strategy_id}.deployment_manifest.v1"


@dataclass(frozen=True)
class EquityDeploymentManifest:
    research_run_id: int
    git_commit: str
    strategy_id: str
    strategy_version: str
    timeframe: str
    universe: tuple[str, ...]
    runtime_config: dict

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.research_run_id <= 0:
            errors.append("research_run_id must be positive")
        if self.strategy_id not in EQUITY_STRATEGY_IDS:
            errors.append("strategy_id is not a market-wide equity strategy")
        if self.strategy_version != EQUITY_STRATEGY_VERSION:
            errors.append("strategy_version mismatch")
        if not _GIT_SHA.fullmatch(str(self.git_commit).lower()):
            errors.append("git_commit must be a full 40-character SHA")
        if self.timeframe not in {"5Min", "15Min"}:
            errors.append("timeframe must be 5Min or 15Min")
        if not 3 <= len(self.universe) <= 192:
            errors.append("universe must contain 3-192 symbols")
        if len(set(self.universe)) != len(self.universe):
            errors.append("universe symbols must be unique")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in self.universe):
            errors.append("universe symbols must be non-empty strings")
        if not isinstance(self.runtime_config, dict):
            errors.append("runtime_config must be an object")
        else:
            try:
                config = MarketWideEquityConfig(**dict(self.runtime_config))
                errors.extend(f"runtime_config invalid: {error}" for error in config.validate())
                if config.timeframe != self.timeframe:
                    errors.append("runtime_config timeframe differs from manifest timeframe")
                if config.max_scan_symbols < len(self.universe):
                    errors.append("runtime_config max_scan_symbols is smaller than locked universe")
            except (TypeError, ValueError) as error:
                errors.append(f"runtime_config invalid: {error}")
        return errors

    def require_valid(self) -> "EquityDeploymentManifest":
        errors = self.validate()
        if errors:
            raise ValueError("Invalid market-wide equity manifest: " + "; ".join(errors))
        return self

    def to_json(self) -> str:
        self.require_valid()
        payload = asdict(self)
        payload["universe"] = list(self.universe)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "EquityDeploymentManifest":
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Market-wide equity manifest JSON is unreadable.") from error
        if not isinstance(payload, dict):
            raise ValueError("Market-wide equity manifest must be a JSON object.")
        payload = dict(payload)
        universe = payload.get("universe")
        if not isinstance(universe, list):
            raise ValueError("Market-wide equity manifest universe must be a list.")
        payload["universe"] = tuple(str(symbol) for symbol in universe)
        runtime_config = payload.get("runtime_config")
        if not isinstance(runtime_config, dict):
            raise ValueError("Market-wide equity manifest runtime_config must be an object.")
        payload["runtime_config"] = dict(runtime_config)
        return cls(**payload).require_valid()

    def assert_code_and_config(
        self,
        *,
        git_commit: Optional[str],
        runtime_config: MarketWideEquityConfig,
    ) -> None:
        if git_commit is None or git_commit.lower() != self.git_commit.lower():
            raise ValueError(
                "Runtime git commit does not match promoted equity research. "
                f"expected={self.git_commit}, actual={git_commit}"
            )
        if runtime_config.to_dict() != self.runtime_config:
            raise ValueError(
                "Runtime market-wide equity configuration differs from the promoted "
                "research configuration; run research again before deployment."
            )

    def restrict_current_selection(self, selected: tuple[str, ...]) -> tuple[str, ...]:
        """Keep only currently executable names, in immutable researched order."""
        currently_eligible = set(selected)
        return tuple(symbol for symbol in self.universe if symbol in currently_eligible)


def record_research_manifest(store, manifest: EquityDeploymentManifest) -> None:
    manifest.require_valid()
    store.set_kv(
        research_manifest_key(manifest.strategy_id, manifest.research_run_id),
        manifest.to_json(),
    )


def load_research_manifest(
    store, strategy_id: str, run_id: int
) -> Optional[EquityDeploymentManifest]:
    raw = store.get_kv(research_manifest_key(strategy_id, run_id))
    return EquityDeploymentManifest.from_json(raw) if raw else None


def lock_deployment_manifest(store, manifest: EquityDeploymentManifest) -> None:
    manifest.require_valid()
    store.set_kv(deployment_manifest_key(manifest.strategy_id), manifest.to_json())


def locked_deployment_manifest(store, strategy_id: str) -> Optional[EquityDeploymentManifest]:
    raw = store.get_kv(deployment_manifest_key(strategy_id))
    return EquityDeploymentManifest.from_json(raw) if raw else None


def require_common_locked_universe(
    store,
    strategy_ids: tuple[str, ...] | list[str] | set[str],
    *,
    runtime_config: MarketWideEquityConfig,
    git_commit: Optional[str] = None,
) -> tuple[str, ...]:
    """Return the common researched universe for every forward-state strategy.

    All enabled market-wide strategies intentionally share one discovery universe.
    Divergent manifests are a deployment error, not something the runtime should
    silently union or intersect.
    """
    ids = tuple(sorted(set(strategy_ids)))
    if not ids:
        return ()
    commit = git_commit if git_commit is not None else current_git_commit()
    manifests: list[EquityDeploymentManifest] = []
    for strategy_id in ids:
        manifest = locked_deployment_manifest(store, strategy_id)
        if manifest is None:
            raise ValueError(
                f"{strategy_id} has forward lifecycle state but no locked equity deployment manifest."
            )
        manifest.assert_code_and_config(git_commit=commit, runtime_config=runtime_config)
        manifests.append(manifest)
    first = manifests[0].universe
    for manifest in manifests[1:]:
        if manifest.universe != first:
            raise ValueError(
                "Enabled market-wide equity strategies have different locked universes; "
                "promote research runs from one common universe."
            )
    return first
