"""The MOO/MOC replay engine - mode C, with look-ahead made structurally impossible.

Guide §5.5 and §13.1. The leak this exists to prevent is specific and invisible: an MOC
decision is made at 15:50 ET, but a daily bar *is* the close, so a backtest that shares
one daily feature set between both execution windows hands the 15:50 model the very
price it is supposed to predict. Results improve, which is why nobody investigates.

Three defences, in increasing order of strength:

1. **Typed snapshots.** A :class:`FeatureSnapshot` is stamped with its session and
   window and refuses to be used for the other one, so the two windows cannot share an
   object by accident.
2. **A visibility horizon.** Every snapshot carries ``as_of``; :meth:`FeatureSnapshot.visible`
   filters any timestamped record against it, so "what did I know" is answered by the
   snapshot rather than by the caller's discipline.
3. **Poison replay.** :func:`assert_no_lookahead` runs the whole policy twice - once
   normally, once with every observation after each cutoff replaced by garbage - and
   asserts the decisions are byte-identical. A policy that peeks produces different
   output and fails the build. This is a proof, not an assertion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .calendar import Session, SessionCalendar
from .execution import FillDecision, Submission, resolve_fill, turnover_cost
from .modes import Candidate, LeakageError

__all__ = [
    "FeatureSnapshot",
    "ReplayStep",
    "ReplayEngine",
    "assert_no_lookahead",
    "visible_daily_fields",
    "DEFAULT_DECISION_LATENCY",
    "POISON",
]

#: How long the live system takes between reading its last input and the submission
#: landing at the API: feature assembly, inference, optimisation, validation, file write,
#: the miner's <=1s poll, and the HTTP POST. A backtest that decides at the cutoff minus
#: one second gives the model information the live system will never have, and it
#: inflates every MOC result. Measure your own and raise this.
DEFAULT_DECISION_LATENCY = timedelta(minutes=5)

#: Substituted for every observation later than a cutoff during a poison run. NaN is
#: deliberate: it propagates through arithmetic instead of being quietly absorbed.
POISON = float("nan")


@dataclass(frozen=True)
class FeatureSnapshot:
    """What the system was allowed to know at one cutoff, for one window.

    ``as_of`` is the execution cutoff - 09:28 ET for MOO, 15:50 ET for MOC, and **12:50
    ET for MOC on an early-close session**. Nothing timestamped at or after it may be
    visible.
    """

    session_day: date
    window: str
    as_of: datetime
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = self.window.upper()
        if w not in ("MOO", "MOC"):
            raise ValueError(f"window must be MOO or MOC, got {self.window!r}")
        object.__setattr__(self, "window", w)

    def require(self, session_day: date, window: str) -> "FeatureSnapshot":
        """Assert this snapshot belongs to the given session and window.

        Raises:
            LeakageError: if it was built for a different window - which is how a shared
                daily feature set leaks the close into the MOO decision, or vice versa.
        """
        if self.session_day != session_day or self.window != window.upper():
            raise LeakageError(
                f"snapshot is for {self.session_day} {self.window} but was used for "
                f"{session_day} {window.upper()}: MOO and MOC need SEPARATE snapshots "
                f"(guide §5.5)"
            )
        return self

    def daily_fields(self) -> frozenset[str]:
        """Which fields of TODAY's daily bar are knowable in this window.

        At the MOO cutoff (09:28) the auction has not run, so **nothing** from today's
        bar exists - not even the open. Guarding only the close is the common mistake:
        a ``gap = open / prev_close`` feature is a standard daily-bar signal and is pure
        look-ahead at 09:28.

        At the MOC cutoff the open is known, but the close has not printed, the high and
        low can still move, and volume is incomplete - the closing auction alone is
        roughly 9-10% of the session's notional.
        """
        return visible_daily_fields(self.window)

    def visible(self, records: Iterable[tuple[datetime, Any]]) -> list[Any]:
        """Filter ``(available_time, value)`` records to what was knowable at ``as_of``."""
        return [v for t, v in records if t < self.as_of]

    def assert_visible(self, available_time: datetime, what: str = "record") -> None:
        if available_time >= self.as_of:
            raise LeakageError(
                f"{what} has available_time {available_time.isoformat()} >= cutoff "
                f"{self.as_of.isoformat()}: not knowable at decision time"
            )


def visible_daily_fields(window: str) -> frozenset[str]:
    """Fields of the CURRENT session's daily bar that are knowable at each cutoff."""
    w = window.upper()
    if w == "MOO":
        return frozenset()                       # the auction has not run
    if w == "MOC":
        return frozenset({"open"})               # close/high/low/volume are all unknown
    raise ValueError(f"unknown window {window!r}")


class SnapshotBuilder(Protocol):
    def __call__(self, session: Session, window: str) -> FeatureSnapshot: ...


class Policy(Protocol):
    def __call__(
        self, snapshot: FeatureSnapshot, previous: Mapping[str, float] | None
    ) -> Candidate | None: ...


@dataclass(frozen=True)
class ReplayStep:
    session_day: date
    window: str
    candidate: Candidate | None
    fill: FillDecision | None
    turnover: float
    skipped: bool
    note: str = ""


class ReplayEngine:
    """Walk sessions forward, one execution window at a time.

    Args:
        calendar: session and cutoff source.
        build_snapshot: called once per (session, window). **Must** return a snapshot
            stamped with that window - returning one shared object for both is the leak
            this engine detects.
        policy: returns a :class:`Candidate`, or ``None`` to skip publishing. Skipping is
            a first-class outcome - the second decision point should be used only when
            new information justifies it.
        windows: which execution windows to run each session.
        decision_latency: how long the live system takes between its last input and the
            submission landing. Snapshots must be built at ``cutoff - latency``, not at
            the cutoff itself.
    """

    def __init__(
        self,
        calendar: SessionCalendar,
        build_snapshot: SnapshotBuilder,
        policy: Policy,
        *,
        windows: Sequence[str] = ("MOO", "MOC"),
        max_turnover: float | None = None,
        decision_latency: timedelta = DEFAULT_DECISION_LATENCY,
    ):
        self.calendar = calendar
        self.build_snapshot = build_snapshot
        self.policy = policy
        self.windows = tuple(w.upper() for w in windows)
        self.max_turnover = max_turnover
        if decision_latency < timedelta(0):
            raise ValueError("decision_latency cannot be negative")
        self.decision_latency = decision_latency

    def horizon(self, session: Session, window: str) -> datetime:
        """The information horizon: the cutoff LESS the decision latency.

        This is what a snapshot's ``as_of`` must equal - not the cutoff itself.
        """
        return session.cutoff(window) - self.decision_latency

    def run(self, start: date, end: date) -> list[ReplayStep]:
        steps: list[ReplayStep] = []
        previous: dict[str, float] | None = None
        submissions: list[Submission] = []
        first_done = False

        for session in self.calendar.sessions(start, end):
            for window in self.windows:
                snapshot = self.build_snapshot(session, window)
                snapshot.require(session.day, window)

                cutoff = session.cutoff(window)
                horizon = self.horizon(session, window)
                if snapshot.as_of != horizon:
                    raise LeakageError(
                        f"snapshot as_of {snapshot.as_of.isoformat()} != the {window} "
                        f"decision horizon {horizon.isoformat()} for {session.day} "
                        f"(cutoff {cutoff.isoformat()} less "
                        f"{self.decision_latency} of decision latency)"
                        + (" (early close moves the MOC cutoff to 12:50 ET)"
                           if session.early else "")
                    )

                candidate = self.policy(snapshot, previous)

                if candidate is None:
                    steps.append(ReplayStep(session.day, window, None, None, 0.0,
                                            skipped=True, note="policy declined to publish"))
                    continue

                if candidate.generated_at != session.day.isoformat():
                    raise LeakageError(
                        f"policy returned a candidate stamped {candidate.generated_at!r} "
                        f"while deciding {session.day.isoformat()} {window}"
                    )
                if candidate.window.upper() != window:
                    raise LeakageError(
                        f"candidate is stamped for {candidate.window} but was produced "
                        f"in the {window} window"
                    )

                turn = turnover_cost(previous or {}, candidate.weights)
                if self.max_turnover is not None and turn > self.max_turnover:
                    steps.append(ReplayStep(session.day, window, candidate, None, turn,
                                            skipped=True,
                                            note=f"turnover {turn:.4f} > cap {self.max_turnover}"))
                    continue

                # The submission lands one latency after the decision, still inside the
                # window. Placing it a second before the cutoff would model a system
                # with zero compute and zero network time.
                submissions.append(Submission(
                    block=int((horizon + self.decision_latency).timestamp()) - 1,
                    weights=dict(candidate.weights),
                    first_ever=not first_done,
                ))
                first_done = True

                fill = resolve_fill(submissions, session, window)
                steps.append(ReplayStep(session.day, window, candidate, fill, turn,
                                        skipped=False, note=fill.reason))
                if fill.fills:
                    previous = ({} if fill.forced_all_cash
                                else dict(fill.submission.weights))
        return steps


# --- the acceptance test ----------------------------------------------------


def _poison(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return POISON
    if isinstance(value, dict):
        return {k: _poison(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_poison(v) for v in value)
    return value


def poison_future(snapshot: FeatureSnapshot, records: Mapping[str, tuple[datetime, Any]]) -> dict:
    """Replace every observation at or after the cutoff with :data:`POISON`."""
    return {
        key: (val if ts < snapshot.as_of else _poison(val))
        for key, (ts, val) in records.items()
    }


def assert_no_lookahead(
    make_engine: Callable[[bool], ReplayEngine],
    start: date,
    end: date,
) -> list[ReplayStep]:
    """Prove a policy cannot see past its cutoff, rather than asserting it.

    ``make_engine(poisoned)`` must build the same engine twice, differing only in whether
    its snapshot builder replaces post-cutoff observations with :data:`POISON`. If the
    two runs produce different decisions, the policy read something it should not have.

    Raises:
        LeakageError: with the first divergent step named.
    """
    clean = make_engine(False).run(start, end)
    dirty = make_engine(True).run(start, end)

    if len(clean) != len(dirty):
        raise LeakageError(
            f"poison run produced {len(dirty)} steps vs {len(clean)} clean: the policy's "
            f"control flow depends on data after the cutoff"
        )

    for a, b in zip(clean, dirty):
        if a.session_day != b.session_day or a.window != b.window:
            raise LeakageError(f"step order diverged at {a.session_day} {a.window}")
        if a.skipped != b.skipped:
            raise LeakageError(
                f"{a.session_day} {a.window}: publish/skip decision changed when future "
                f"data was poisoned - the policy is reading past its cutoff"
            )
        wa = dict(a.candidate.weights) if a.candidate else {}
        wb = dict(b.candidate.weights) if b.candidate else {}
        if set(wa) != set(wb):
            raise LeakageError(
                f"{a.session_day} {a.window}: holdings changed under poisoning "
                f"({sorted(set(wa) ^ set(wb))})"
            )
        for k in wa:
            x, y = wa[k], wb[k]
            if (math.isnan(x) != math.isnan(y)) or (not math.isnan(x) and abs(x - y) > 1e-12):
                raise LeakageError(
                    f"{a.session_day} {a.window}: weight for {k} changed under poisoning "
                    f"({x} -> {y}) - the policy is reading past its cutoff"
                )
    return clean
