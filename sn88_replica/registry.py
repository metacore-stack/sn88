"""Model and configuration registry - promotion, and rollback as one operation.

Guide §31. "Ship versioned artefacts" is not a lifecycle. What is actually needed:

* an immutable record per artefact, with the walk-forward evidence that justified it;
* **promotion criteria fixed in advance**, not chosen after seeing the result;
* feature-version compatibility enforced, so a model cannot load a feature set it was
  not trained against;
* rollback as a tested one-command operation, not a redeploy;
* **manual approval for anything touching risk.** Model weights may auto-promote; the
  governor's limits and the scale band may not.

The last rule is the one that matters. A system that can promote a model on evidence can
also, if you let it, quietly widen its own risk limits to make the evidence look better.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ArtifactKind",
    "Artifact",
    "PromotionCriteria",
    "Registry",
    "PromotionError",
]


class ArtifactKind(str, Enum):
    MODEL = "model"          # may auto-promote on evidence
    FEATURES = "features"
    RISK_POLICY = "risk"     # NEVER auto-promotes


class PromotionError(RuntimeError):
    """Refused to promote. The champion is unchanged."""


@dataclass(frozen=True)
class PromotionCriteria:
    """Fixed in advance. Stored with the artefact so it cannot be revised afterwards."""

    min_live_sessions: int = 20
    min_score_improvement: float = 0.05      # vs the champion, through the replica
    max_deflated_sharpe_p: float = 0.05
    must_beat_baselines: tuple[str, ...] = ("equal_weight", "inverse_volatility")
    require_manual_approval: bool = False

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Artifact:
    """One promotable thing, with the evidence that justified it."""

    artifact_id: str
    kind: ArtifactKind
    content_hash: str
    feature_version: str
    git_commit: str = ""
    created_at: str = ""
    criteria_digest: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    promoted_at: str | None = None
    rolled_back_at: str | None = None
    approver: str | None = None

    def as_row(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Artifact":
        row = dict(row)
        row["kind"] = ArtifactKind(row["kind"])
        return cls(**row)


class Registry:
    """Append-only artefact log plus a champion pointer per kind.

    Append-only matters: the rollback target must still exist, and the record of what was
    live when a result was produced must survive the thing being superseded.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "artifacts.jsonl"
        self.champions = self.root / "champions.json"

    # --- storage ---------------------------------------------------------

    def _all(self) -> list[Artifact]:
        if not self.log.exists():
            return []
        return [Artifact.from_row(json.loads(l)) for l in
                self.log.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _append(self, artifact: Artifact) -> None:
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(artifact.as_row(), separators=(",", ":")) + "\n")

    def _pointers(self) -> dict[str, str]:
        if not self.champions.exists():
            return {}
        return json.loads(self.champions.read_text(encoding="utf-8"))

    def _set_pointer(self, kind: ArtifactKind, artifact_id: str) -> None:
        p = self._pointers()
        p[kind.value] = artifact_id
        self.champions.write_text(json.dumps(p, indent=1, sort_keys=True), encoding="utf-8")

    # --- api -------------------------------------------------------------

    def register(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        content: bytes | str,
        feature_version: str,
        criteria: PromotionCriteria,
        git_commit: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> Artifact:
        """Record a candidate. Registering is not promoting."""
        if any(a.artifact_id == artifact_id for a in self._all()):
            raise PromotionError(f"artifact_id {artifact_id!r} already registered")
        blob = content.encode() if isinstance(content, str) else content
        art = Artifact(
            artifact_id=artifact_id,
            kind=kind,
            content_hash=hashlib.sha256(blob).hexdigest()[:32],
            feature_version=feature_version,
            git_commit=git_commit,
            created_at=datetime.now(timezone.utc).isoformat(),
            criteria_digest=criteria.digest(),
            evidence=dict(evidence or {}),
        )
        self._append(art)
        return art

    def champion(self, kind: ArtifactKind) -> Artifact | None:
        target = self._pointers().get(kind.value)
        if not target:
            return None
        return next((a for a in self._all() if a.artifact_id == target), None)

    def history(self, kind: ArtifactKind | None = None) -> list[Artifact]:
        return [a for a in self._all() if kind is None or a.kind is kind]

    def promote(
        self,
        artifact_id: str,
        criteria: PromotionCriteria,
        *,
        live_sessions: int,
        score_vs_champion: float,
        deflated_sharpe_p: float,
        baselines_beaten: Sequence[str],
        current_feature_version: str,
        approver: str | None = None,
    ) -> Artifact:
        """Promote if and only if the pre-registered criteria are met.

        Raises:
            PromotionError: with every unmet criterion listed. The champion is unchanged.
        """
        art = next((a for a in self._all() if a.artifact_id == artifact_id), None)
        if art is None:
            raise PromotionError(f"unknown artifact {artifact_id!r}")

        if art.criteria_digest != criteria.digest():
            raise PromotionError(
                "criteria were changed after registration - promotion criteria are fixed "
                f"in advance (registered {art.criteria_digest}, supplied {criteria.digest()})"
            )

        failures: list[str] = []
        if art.feature_version != current_feature_version:
            failures.append(
                f"feature version mismatch: trained against {art.feature_version!r}, "
                f"live is {current_feature_version!r}")
        if live_sessions < criteria.min_live_sessions:
            failures.append(f"{live_sessions} live sessions < {criteria.min_live_sessions}")
        if score_vs_champion < criteria.min_score_improvement:
            failures.append(
                f"score improvement {score_vs_champion:.4f} < "
                f"{criteria.min_score_improvement}")
        if deflated_sharpe_p > criteria.max_deflated_sharpe_p:
            failures.append(
                f"deflated Sharpe p={deflated_sharpe_p:.4f} > "
                f"{criteria.max_deflated_sharpe_p}")
        missing = set(criteria.must_beat_baselines) - set(baselines_beaten)
        if missing:
            failures.append(f"did not beat baselines {sorted(missing)}")
        if art.kind is ArtifactKind.RISK_POLICY and not approver:
            failures.append("risk-policy changes require manual approval")
        if criteria.require_manual_approval and not approver:
            failures.append("criteria require manual approval")

        if failures:
            raise PromotionError("; ".join(failures))

        art.promoted_at = datetime.now(timezone.utc).isoformat()
        art.approver = approver
        self._append(art)
        self._set_pointer(art.kind, art.artifact_id)
        return art

    def rollback(self, kind: ArtifactKind, *, to: str | None = None) -> Artifact:
        """Revert to a previous champion. One call, no redeploy.

        With no target, reverts to the most recently promoted artefact of that kind other
        than the current champion.
        """
        current = self.champion(kind)
        promoted = [a for a in self._all() if a.kind is kind and a.promoted_at]
        if to:
            target = next((a for a in promoted if a.artifact_id == to), None)
            if target is None:
                raise PromotionError(f"{to!r} was never promoted, cannot roll back to it")
        else:
            candidates = [a for a in promoted
                          if not current or a.artifact_id != current.artifact_id]
            if not candidates:
                raise PromotionError("no previous champion to roll back to")
            target = max(candidates, key=lambda a: a.promoted_at or "")

        if current:
            current.rolled_back_at = datetime.now(timezone.utc).isoformat()
            self._append(current)
        self._set_pointer(kind, target.artifact_id)
        return target
