"""Shadow operation: the full loop with the last step disarmed.

Guide §24 stage 7. Ten to twenty sessions of the complete cycle - rule check, snapshot,
policy, governor, publisher - with the publisher in ``dry_run`` so nothing reaches
``/rev``. It costs a small fraction of the ramp and retires the four operational failures
that are cheap to catch here and permanent if you catch them live: a silent ``asum > 1``
discard, an MOC leak, a publisher race, and an unvalidated ticker.

**Be honest about what shadow can prove.** You are not registered, so you have no UID and
no ``/score`` row. That splits the readiness gates in two:

* **Operational gates** - evaluable now, from the loop itself. Did every cycle complete?
  Would every file have been accepted? Were the cutoffs met? Did the governor ever get
  overruled? Did the rule detector run?
* **Projection gates** - *not* evaluable without a market-data feed, because reconciling
  a forward projection against a realised outcome needs prices for a book that does not
  exist on chain. These are reported as ``UNPROVEN``, never as passed.

A readiness report that quietly marked the second group green would be the most expensive
kind of green in this whole project, so :meth:`ShadowRun.readiness` refuses to.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ShadowRecord",
    "ShadowRun",
    "ReadinessCriteria",
    "GateResult",
    "ReadinessReport",
    "GateStatus",
]


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNPROVEN = "unproven"      # cannot be evaluated without a market-data feed


@dataclass(frozen=True)
class ShadowRecord:
    """One execution window of one session, as it would have gone."""

    session: str                    # ISO date
    window: str                     # MOO | MOC
    recorded_at: str
    published: bool                 # would the publisher have written?
    upstream_would_accept: bool     # predicted against the real parser
    governor_approved: bool
    cutoff_met: bool
    weights_hash: str = ""
    effective_cash: float | None = None
    gross: float | None = None
    turnover: float | None = None
    projected_incentive: float | None = None
    vetoes: tuple[str, ...] = ()
    faults: tuple[str, ...] = ()
    alert_action: str = "none"

    def as_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ShadowRecord":
        row = dict(row)
        row["vetoes"] = tuple(row.get("vetoes") or ())
        row["faults"] = tuple(row.get("faults") or ())
        return cls(**row)

    @staticmethod
    def hash_weights(weights: Mapping[str, float]) -> str:
        blob = json.dumps({k: round(float(v), 10) for k, v in sorted(weights.items())},
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ReadinessCriteria:
    """Fixed before the run starts, like any other promotion criterion (guide §31)."""

    min_sessions: int = 10
    max_faults: int = 0
    max_would_be_discarded: int = 0
    max_missed_cutoffs: int = 0
    max_governor_overrides: int = 0
    require_both_windows: bool = True
    require_rule_detector_exercised: bool = True


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    detail: str

    def __str__(self) -> str:
        mark = {GateStatus.PASS: "PASS", GateStatus.FAIL: "FAIL",
                GateStatus.UNPROVEN: "UNPROVEN"}[self.status]
        return f"[{mark:>8}] {self.name}: {self.detail}"


@dataclass(frozen=True)
class ReadinessReport:
    gates: tuple[GateResult, ...]
    sessions_observed: int

    @property
    def ready(self) -> bool:
        """True only if every gate PASSED. ``UNPROVEN`` is never treated as a pass."""
        return all(g.status is GateStatus.PASS for g in self.gates)

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if g.status is GateStatus.FAIL]

    @property
    def unproven(self) -> list[GateResult]:
        return [g for g in self.gates if g.status is GateStatus.UNPROVEN]

    def summary(self) -> str:
        head = ("READY to register" if self.ready
                else f"NOT READY - {len(self.failures)} failed, "
                     f"{len(self.unproven)} unproven")
        return "\n".join([head, *(str(g) for g in self.gates)])


class ShadowRun:
    """Append-only shadow log, resumable across the calendar days a real run takes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- storage ---------------------------------------------------------

    def append(self, record: ShadowRecord) -> ShadowRecord:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.as_row(), separators=(",", ":")) + "\n")
        return record

    def records(self) -> list[ShadowRecord]:
        if not self.path.exists():
            return []
        return [ShadowRecord.from_row(json.loads(l))
                for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def sessions(self) -> list[str]:
        return sorted({r.session for r in self.records()})

    # --- the gate --------------------------------------------------------

    def readiness(
        self,
        criteria: ReadinessCriteria | None = None,
        *,
        rule_detector_ran: bool = False,
        leakage_probes_green: bool | None = None,
        projection_error: float | None = None,
        market_data_available: bool = False,
    ) -> ReadinessReport:
        """Go / no-go for registration.

        Args:
            rule_detector_ran: did the rule-change detector execute each cycle?
            leakage_probes_green: did the CI leakage probes pass on the shipped code?
            projection_error: median |realised - projected| / projected. Requires a
                market-data feed; without one the gate is reported UNPROVEN.
            market_data_available: whether projection gates can be evaluated at all.
        """
        c = criteria or ReadinessCriteria()
        rs = self.records()
        gates: list[GateResult] = []
        sessions = sorted({r.session for r in rs})

        gates.append(self._gate(
            "sessions_observed", len(sessions) >= c.min_sessions,
            f"{len(sessions)} of {c.min_sessions} required"))

        faults = [f for r in rs for f in r.faults]
        gates.append(self._gate(
            "no_unhandled_faults", len(faults) <= c.max_faults,
            f"{len(faults)} fault(s)" + (f": {sorted(set(faults))[:5]}" if faults else "")))

        discarded = [r for r in rs if r.published and not r.upstream_would_accept]
        gates.append(self._gate(
            "nothing_would_be_discarded", len(discarded) <= c.max_would_be_discarded,
            f"{len(discarded)} file(s) upstream would have discarded silently"))

        missed = [r for r in rs if r.published and not r.cutoff_met]
        gates.append(self._gate(
            "cutoffs_met", len(missed) <= c.max_missed_cutoffs,
            f"{len(missed)} publish(es) after the execution cutoff"))

        overrides = [r for r in rs if r.published and not r.governor_approved]
        gates.append(self._gate(
            "governor_never_overruled", len(overrides) <= c.max_governor_overrides,
            f"{len(overrides)} publish(es) despite a governor veto"))

        if c.require_both_windows:
            windows = {r.window for r in rs}
            gates.append(self._gate(
                "both_windows_exercised", windows >= {"MOO", "MOC"},
                f"windows seen: {sorted(windows) or 'none'}"))

        if c.require_rule_detector_exercised:
            gates.append(self._gate(
                "rule_detector_ran", rule_detector_ran,
                "the rule-change detector ran each cycle" if rule_detector_ran
                else "never ran - a mid-run constant change would go unnoticed"))

        if leakage_probes_green is None:
            gates.append(GateResult("leakage_probes", GateStatus.UNPROVEN,
                                    "not reported - run the CI leakage probes"))
        else:
            gates.append(self._gate("leakage_probes", leakage_probes_green,
                                    "green" if leakage_probes_green else "FAILING"))

        # --- gates that a shadow run structurally cannot settle ------------
        if not market_data_available:
            gates.append(GateResult(
                "projection_accuracy", GateStatus.UNPROVEN,
                "unregistered shadow has no /score row and no market-data feed, so a "
                "forward projection cannot be reconciled against a realised outcome"))
            gates.append(GateResult(
                "strategy_edge", GateStatus.UNPROVEN,
                "shadow tests the PIPELINE, not the edge - a 10-20 session operational "
                "run says nothing about whether the signal works"))
        else:
            if projection_error is None:
                gates.append(GateResult("projection_accuracy", GateStatus.UNPROVEN,
                                        "market data available but no error reported"))
            else:
                gates.append(self._gate(
                    "projection_accuracy", projection_error <= 0.25,
                    f"median relative error {projection_error:.1%}"))
            gates.append(GateResult(
                "strategy_edge", GateStatus.UNPROVEN,
                "still unproven: shadow validates operations, not alpha"))

        return ReadinessReport(tuple(gates), len(sessions))

    @staticmethod
    def _gate(name: str, ok: bool, detail: str) -> GateResult:
        return GateResult(name, GateStatus.PASS if ok else GateStatus.FAIL, detail)

    # --- summary ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        rs = self.records()
        published = [r for r in rs if r.published]
        return {
            "records": len(rs),
            "sessions": len({r.session for r in rs}),
            "published": len(published),
            "skipped": len(rs) - len(published),
            "vetoed": len([r for r in rs if not r.governor_approved]),
            "faults": len([f for r in rs for f in r.faults]),
            "distinct_books": len({r.weights_hash for r in published if r.weights_hash}),
            "first": rs[0].session if rs else None,
            "last": rs[-1].session if rs else None,
        }
