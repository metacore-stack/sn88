"""Observability, alerting, and the rule-change detector.

Guide §18, §21, §23. Two jobs that are easy to confuse:

* **Metrics** answer "what is my system doing" - projected vs realised score, cash,
  gross, headroom, turnover, distance to the field, replica drift.
* **The rule-change detector** answers "is the game still the game". Validators run
  ``etc.update()`` hourly (``git pull && pip install -e .``), so the scoring constants,
  the class ratio, the similarity threshold, the cash-ETF list and the tradable universe
  can all move under you between two cycles with no announcement.

When the rules move, the correct response is **not** to trade through it. It is to
``FREEZE`` candidate selection until the replica is re-validated and the golden tests
pass again, because every projected score is computed against a function that no longer
exists.

One asymmetry worth naming: a score drop caused by the cash multiplier, or by an
upstream constant change, looks identical on the dashboard to a strategy drawdown - and
calls for the opposite response. That is what :func:`explain_score_change` is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import fingerprint, snapshot as constants_snapshot
from .universe import Universe

__all__ = [
    "Severity",
    "Alert",
    "Action",
    "RuleDiff",
    "detect_rule_change",
    "CycleMetrics",
    "check_cycle",
    "explain_score_change",
    "AlertRouter",
]


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    PAGE = "page"


class Action(str, Enum):
    NONE = "none"
    ALERT = "alert"
    FREEZE = "freeze_selection"      # stop ranking candidates; keep the last good book
    SAFE_MODE = "safe_mode"          # hold allocation, preserve logs, page a human


@dataclass(frozen=True)
class Alert:
    key: str
    severity: Severity
    message: str
    action: Action = Action.ALERT
    context: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.key}: {self.message}"


# --- rule-change detection --------------------------------------------------


@dataclass(frozen=True)
class RuleDiff:
    changed: bool
    constants: Mapping[str, tuple[Any, Any]] = field(default_factory=dict)
    ratio: tuple[Sequence[float], Sequence[float]] | None = None
    cash_tickers: tuple[frozenset[str], frozenset[str]] | None = None
    universe_added: frozenset[str] = frozenset()
    universe_removed: frozenset[str] = frozenset()
    dereg_added: frozenset[int] = frozenset()

    def alerts(self) -> list[Alert]:
        out: list[Alert] = []
        if self.constants:
            out.append(Alert(
                "constants_changed", Severity.PAGE,
                "scoring constants moved upstream: "
                + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in sorted(self.constants.items()))
                + " - every projected score is against a function that no longer exists",
                Action.FREEZE, {"constants": dict(self.constants)}))
        if self.ratio:
            before, after = self.ratio
            note = ""
            if abs(sum(after) - 1.0) > 1e-9:
                note = (f" - and the ratios now sum to {sum(after):.4f}, so "
                        f"{1 - sum(after):.1%} of miner emission is being burned at UID 0")
            out.append(Alert(
                "ratio_changed", Severity.PAGE,
                f"class emission split {list(before)} -> {list(after)}{note}",
                Action.FREEZE, {"before": list(before), "after": list(after)}))
        if self.cash_tickers:
            before, after = self.cash_tickers
            added, removed = after - before, before - after
            out.append(Alert(
                "cash_list_changed", Severity.PAGE,
                f"cash-classified list changed: +{sorted(added)} -{sorted(removed)} - "
                f"a holding may have silently become a linear score penalty",
                Action.FREEZE, {"added": sorted(added), "removed": sorted(removed)}))
        if self.universe_removed:
            out.append(Alert(
                "tickers_delisted", Severity.PAGE,
                f"{sorted(self.universe_removed)} left /assets - any holding is now "
                f"silently reclassified to cash",
                Action.FREEZE, {"removed": sorted(self.universe_removed)}))
        if self.universe_added:
            out.append(Alert(
                "tickers_added", Severity.INFO,
                f"{len(self.universe_added)} ticker(s) entered /assets: "
                f"{sorted(self.universe_added)[:8]}",
                Action.NONE, {"added": sorted(self.universe_added)}))
        if self.dereg_added:
            out.append(Alert(
                "dereg_list_changed", Severity.WARN,
                f"deregistration list gained {sorted(self.dereg_added)}",
                Action.ALERT, {"added": sorted(self.dereg_added)}))
        return out


def detect_rule_change(
    *,
    before_universe: Universe,
    after_universe: Universe,
    before_constants: Mapping[str, Any] | None = None,
    after_constants: Mapping[str, Any] | None = None,
    holdings: Iterable[str] = (),
) -> RuleDiff:
    """Diff two daily snapshots. ``holdings`` narrows the delisting alert to your book."""
    before_constants = dict(before_constants or constants_snapshot())
    after_constants = dict(after_constants or constants_snapshot())
    const_delta = {
        k: (before_constants.get(k), after_constants.get(k))
        for k in set(before_constants) | set(after_constants)
        if before_constants.get(k) != after_constants.get(k)
    }

    ratio = None
    if tuple(before_universe.asset_ratio) != tuple(after_universe.asset_ratio):
        ratio = (tuple(before_universe.asset_ratio), tuple(after_universe.asset_ratio))

    cash = None
    if before_universe.cash_tickers != after_universe.cash_tickers:
        cash = (before_universe.cash_tickers, after_universe.cash_tickers)

    before_names = set(before_universe.assets)
    after_names = set(after_universe.assets)
    removed = before_names - after_names
    held = set(holdings)
    if held:
        removed = {t for t in removed if t in held}

    return RuleDiff(
        changed=bool(const_delta or ratio or cash or removed
                     or (after_names - before_names)
                     or (set(after_universe.dereg_list) - set(before_universe.dereg_list))),
        constants=const_delta,
        ratio=ratio,
        cash_tickers=cash,
        universe_added=frozenset(after_names - before_names),
        universe_removed=frozenset(removed),
        dereg_added=frozenset(set(after_universe.dereg_list) - set(before_universe.dereg_list)),
    )


# --- per-cycle health -------------------------------------------------------


@dataclass
class CycleMetrics:
    """Everything worth alerting on, emitted once per execution window."""

    session: date
    window: str
    gross: float | None = None
    cash: float | None = None
    headroom: float | None = None
    turnover: float | None = None
    last_active_days: float | None = None
    min_field_distance: float | None = None
    replica_drift: float | None = None
    projected_score: float | None = None
    realised_score: float | None = None
    submission_confirmed: bool | None = None
    seconds_since_submission: float | None = None
    published: bool = False
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "notes"}
        d["session"] = self.session.isoformat()
        d["notes"] = list(self.notes)
        return d


def check_cycle(
    m: CycleMetrics,
    *,
    max_cash: float = 0.01,
    min_gross: float = 0.99,
    max_turnover: float = 0.15,
    max_idle_days: float = 1.0,
    min_field_distance: float = 0.02,
    replica_tolerance: float = 1e-6,
    confirm_seconds: float = 300.0,
    score_divergence: float = 0.25,
) -> list[Alert]:
    """Turn one cycle's metrics into alerts. Thresholds mirror ``config/system.yaml``."""
    out: list[Alert] = []

    if m.cash is not None and m.cash > max_cash + 1e-9:
        out.append(Alert("cash_band", Severity.WARN,
                         f"cash {m.cash:.4f} > {max_cash} - the multiplier is linear "
                         f"({1 - m.cash:.4f}x)", Action.ALERT))
    if m.gross is not None and m.gross < min_gross - 1e-9:
        out.append(Alert("under_invested", Severity.WARN,
                         f"gross {m.gross:.4f} < {min_gross}", Action.ALERT))
    if m.gross is not None and m.gross > 1.0:
        out.append(Alert("gross_over_one", Severity.PAGE,
                         f"gross {m.gross!r} > 1.0 - upstream discards the file SILENTLY",
                         Action.SAFE_MODE))
    if m.turnover is not None and m.turnover > max_turnover + 1e-9:
        out.append(Alert("turnover", Severity.WARN,
                         f"turnover {m.turnover:.4f} > {max_turnover}", Action.ALERT))
    if m.last_active_days is not None and m.last_active_days > max_idle_days:
        out.append(Alert("inactivity", Severity.PAGE,
                         f"{m.last_active_days:g} calendar days since the last rebalance - "
                         f"the DEC multiplier is "
                         f"{1 - min((m.last_active_days / 20) ** 2, 1):.4f}x",
                         Action.ALERT))
    if m.headroom is not None and m.headroom < 0:
        out.append(Alert("below_inception", Severity.PAGE,
                         f"headroom {m.headroom:.4f} - the score gate is shut and "
                         f"emission is zero until it is back above the line",
                         Action.SAFE_MODE))
    elif m.headroom is not None and m.headroom < 0.03:
        out.append(Alert("floor_warn", Severity.WARN,
                         f"headroom {m.headroom:.4f} - scale-ups blocked", Action.ALERT))
    if m.min_field_distance is not None and m.min_field_distance < min_field_distance:
        out.append(Alert("field_distance", Severity.WARN,
                         f"nearest rival at {m.min_field_distance:.4f} - the dedupe "
                         f"trigger is 0.01 and it penalises the LATER submission, "
                         f"which will be you at the next rebalance", Action.ALERT))
    if m.replica_drift is not None and m.replica_drift > replica_tolerance:
        out.append(Alert("replica_drift", Severity.PAGE,
                         f"replica differs from /score by {m.replica_drift:.2e} "
                         f"(tolerance {replica_tolerance:.0e})", Action.FREEZE))
    if m.published and m.submission_confirmed is False:
        secs = m.seconds_since_submission or 0.0
        if secs > confirm_seconds:
            out.append(Alert("unconfirmed_submission", Severity.PAGE,
                             f"unconfirmed after {secs:.0f}s - do NOT write again, the "
                             f"miner retries every ~11s on its own", Action.ALERT))
    if (m.projected_score is not None and m.realised_score is not None
            and m.projected_score > 0):
        rel = abs(m.realised_score - m.projected_score) / m.projected_score
        if rel > score_divergence:
            out.append(Alert("score_divergence", Severity.PAGE,
                             f"realised {m.realised_score:.4f} vs projected "
                             f"{m.projected_score:.4f} ({rel:.1%}) - safe mode until "
                             f"explained", Action.SAFE_MODE))
    return out


def explain_score_change(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> list[tuple[str, float, str]]:
    """Attribute a score change across the multipliers, not just to "performance".

    A drop caused by the cash multiplier or an upstream constant change looks identical
    to a drawdown on the dashboard and calls for the opposite response. Pass the
    per-component values from two :class:`~sn88_replica.pipeline.MinerScore` rows.

    Returns ``(component, multiplicative_contribution, direction)`` sorted by impact.
    """
    out: list[tuple[str, float, str]] = []
    for key in sorted(set(before) | set(after)):
        a, b = before.get(key), after.get(key)
        if a is None or b is None or a == 0:
            continue
        ratio = b / a
        if abs(ratio - 1.0) > 1e-12:
            out.append((key, ratio, "up" if ratio > 1 else "down"))
    return sorted(out, key=lambda r: abs(math_log(r[1])), reverse=True)


def math_log(x: float) -> float:
    import math
    return math.log(x) if x > 0 else float("-inf")


# --- routing ----------------------------------------------------------------


class AlertRouter:
    """Collects alerts, decides the strongest action, and appends to a JSONL log."""

    ORDER = {Action.NONE: 0, Action.ALERT: 1, Action.FREEZE: 2, Action.SAFE_MODE: 3}

    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else None
        self.alerts: list[Alert] = []

    def extend(self, alerts: Iterable[Alert]) -> "AlertRouter":
        for a in alerts:
            self.alerts.append(a)
            self._write(a)
        return self

    def action(self) -> Action:
        """The strongest action any alert demands. FREEZE beats ALERT; SAFE_MODE beats all."""
        return max((a.action for a in self.alerts), key=lambda x: self.ORDER[x],
                   default=Action.NONE)

    def should_trade(self) -> bool:
        return self.action() in (Action.NONE, Action.ALERT)

    def paging(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity is Severity.PAGE]

    def _write(self, alert: Alert) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "key": alert.key,
            "severity": alert.severity.value,
            "action": alert.action.value,
            "message": alert.message,
            "context": dict(alert.context),
            "constants_fingerprint": fingerprint(),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
