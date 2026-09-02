"""The three evaluation modes, kept apart by the type system.

Guide §13.1. Revision 3 of the guide said "replay the trailing 50-session window with
the candidate's weights". That is a look-ahead bug: a candidate generated from today's
features, applied backwards over sessions whose returns are already known, scores
brilliantly and means nothing. The failure is silent - the backtest simply looks
excellent - so it has to be prevented structurally rather than by convention.

    Mode A  RECONCILIATION      actual past weights, actual past P&L.
                                Validates the replica. NEVER ranks candidates.
    Mode B  FORWARD PROJECTION  history frozen; the candidate applies only from the
                                next eligible cutoff, over simulated forward paths.
                                The ONLY mode the selector may call.
    Mode C  POLICY BACKTEST     walk the clock; at each session use only data with
                                available_time <= that session's cutoff.

:class:`RealisedHistory` is immutable and carries no weights. There is no code path
by which a candidate weight can touch a realised return; attempting it raises
:class:`LeakageError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .constants import RISK_INIT_STK, WIN_SIZE_STK
from .scoring import ScoreTerms, score_terms
from .constants import DAYS_FINAL, DAYS_DELAY

__all__ = [
    "LeakageError",
    "RealisedHistory",
    "Candidate",
    "ForwardPaths",
    "reconcile",
    "project",
    "ProjectionResult",
]


class LeakageError(AssertionError):
    """Raised when an evaluation would use information unavailable at decision time.

    This is deliberately an ``AssertionError`` subclass so it cannot be swallowed by a
    broad ``except Exception`` around a scoring loop.
    """


@dataclass(frozen=True)
class RealisedHistory:
    """Immutable record of what actually happened. Carries no candidate weights.

    Attributes:
        session_dates: ISO dates, ascending. One entry per TRADING SESSION.
        swap_closes: realised close-of-session book value, same length.
        first_swap_open: the all-time inception value ($10M), for the gate.
    """

    session_dates: tuple[str, ...]
    swap_closes: tuple[float, ...]
    first_swap_open: float

    def __post_init__(self) -> None:
        if len(self.session_dates) != len(self.swap_closes):
            raise ValueError("session_dates and swap_closes must align")
        if list(self.session_dates) != sorted(self.session_dates):
            raise ValueError("session_dates must be ascending")

    @property
    def last_date(self) -> str:
        return self.session_dates[-1]

    @property
    def lifetime_days(self) -> int:
        return len(self.session_dates)

    @property
    def last_close(self) -> float:
        return self.swap_closes[-1]

    def window(self, size: int = WIN_SIZE_STK) -> tuple[float, tuple[float, ...]]:
        """(swap_open of the first session in the window, closes in the window)."""
        closes = self.swap_closes[-size:]
        # swap_open of the first windowed session == prior close, or inception if none
        idx = len(self.swap_closes) - len(closes)
        opening = self.swap_closes[idx - 1] if idx > 0 else self.first_swap_open
        return opening, closes


@dataclass(frozen=True)
class Candidate:
    """A proposed allocation, stamped with when it was generated.

    ``generated_at`` is the session date whose cutoff produced it. Any evaluation that
    would apply these weights to a return dated on or before it is a leak.
    """

    weights: dict[str, float]
    generated_at: str
    window: str = "MOC"          # "MOO" or "MOC"

    def gross(self) -> float:
        return sum(abs(w) for k, w in self.weights.items() if k not in ("_", "*", "=", "-+"))


@dataclass(frozen=True)
class ForwardPaths:
    """Simulated future session returns, in percent, one list per path.

    These must come from the §11.2 stress suite - block bootstrap, crisis replay,
    regime switching, jump diffusion - not from an i.i.d. Gaussian draw. The Gaussian
    generator is retained only as a unit test of the replica and must not set scale.
    """

    returns_pct: tuple[tuple[float, ...], ...]
    starts_after: str            # first simulated session is strictly after this date
    generator: str = "unspecified"

    def __post_init__(self) -> None:
        if self.generator == "gaussian_iid":
            # not fatal - it is a legitimate unit-test generator - but never silent
            import warnings

            warnings.warn(
                "ForwardPaths generator is gaussian_iid: acceptable for replica unit "
                "tests, NOT acceptable for sizing decisions (guide §11.2).",
                stacklevel=2,
            )


@dataclass(frozen=True)
class ProjectionResult:
    """Distribution of outcomes across forward paths - never a point estimate."""

    scores: tuple[float, ...]
    breach_probability: float
    median: float
    p10: float
    p90: float
    dispersion: float

    @classmethod
    def build(cls, scores: Sequence[float], breaches: Sequence[bool]) -> "ProjectionResult":
        s = sorted(scores)
        n = len(s)
        if n == 0:
            raise ValueError("no paths")
        mean = sum(s) / n
        var = sum((x - mean) ** 2 for x in s) / n
        return cls(
            scores=tuple(s),
            breach_probability=sum(1 for b in breaches if b) / n,
            median=s[n // 2],
            p10=s[max(0, int(0.10 * n) - 1)],
            p90=s[min(n - 1, int(0.90 * n))],
            dispersion=var**0.5,
        )


# --- Mode A -----------------------------------------------------------------


def reconcile(history: RealisedHistory, *, risk_init: float = RISK_INIT_STK) -> ScoreTerms:
    """Mode A. Reproduce the current score from realised history alone.

    No candidate is involved and none may be passed. Use this to validate the replica
    against ``/score`` and ``bin/validator`` - never to rank anything.
    """
    opening, closes = history.window()
    terms = score_terms(opening, closes, risk_init=risk_init,
                        lifetime_days=history.lifetime_days)
    value = terms.score
    if history.lifetime_days < DAYS_FINAL:
        value *= (history.lifetime_days / DAYS_FINAL) ** DAYS_DELAY
    return ScoreTerms(**{**terms.__dict__, "score": value})


# --- Mode B -----------------------------------------------------------------


def project(
    candidate: Candidate,
    history: RealisedHistory,
    paths: ForwardPaths,
    *,
    risk_init: float = RISK_INIT_STK,
    horizon: int | None = None,
    post_multipliers: Callable[[float], float] | None = None,
) -> ProjectionResult:
    """Mode B. The only mode the selector may call.

    Realised history is FROZEN. The candidate's weights drive only the simulated
    forward sessions, which begin strictly after ``paths.starts_after``.

    Raises:
        LeakageError: if the forward paths would start on or before the candidate was
            generated, or before the end of realised history - either of which would
            let the candidate see returns that were already known.
    """
    if paths.starts_after < candidate.generated_at:
        raise LeakageError(
            f"forward paths start after {paths.starts_after!r} but the candidate was "
            f"generated at {candidate.generated_at!r}: the candidate would be applied "
            f"to returns that were already known when it was produced"
        )
    if paths.starts_after < history.last_date:
        raise LeakageError(
            f"forward paths start after {paths.starts_after!r}, which is inside "
            f"realised history ending {history.last_date!r}: realised sessions must "
            f"stay frozen"
        )

    gross = candidate.gross()
    if gross > 1.0:
        raise ValueError(
            f"candidate gross exposure {gross:.6f} > 1.0 - upstream would SILENTLY "
            f"discard this file and keep your previous allocation live (guide §4.1)"
        )

    scores: list[float] = []
    breaches: list[bool] = []
    for path in paths.returns_pct:
        steps = path[:horizon] if horizon else path
        closes = list(history.swap_closes)
        equity = closes[-1]
        breached = history.last_close < history.first_swap_open
        for r in steps:
            equity *= 1 + r / 100
            closes.append(equity)
            if equity < history.first_swap_open:
                breached = True

        lifetime = len(closes)
        win = closes[-WIN_SIZE_STK:]
        idx = len(closes) - len(win)
        opening = closes[idx - 1] if idx > 0 else history.first_swap_open

        terms = score_terms(opening, win, risk_init=risk_init, lifetime_days=lifetime)
        value = terms.score
        if lifetime < DAYS_FINAL:
            value *= (lifetime / DAYS_FINAL) ** DAYS_DELAY
        # the inception gate is evaluated on the FULL simulated history
        value *= 1 if closes[-1] >= history.first_swap_open else 0
        if post_multipliers is not None:
            value = post_multipliers(value)
        scores.append(value)
        breaches.append(breached)

    return ProjectionResult.build(scores, breaches)


# --- Mode C -----------------------------------------------------------------


@dataclass
class PolicyBacktest:
    """Mode C. Walk the clock; never look forward.

    ``policy(snapshot, previous_weights) -> Candidate`` is called once per session with
    a features snapshot the caller must have built under an ``available_time`` filter
    (09:28 ET for MOO, 15:50 ET for MOC - see guide §5.5; they are DIFFERENT snapshots
    and sharing one daily bar between them leaks the close into the MOC decision).
    """

    sessions: Sequence[str]
    load_snapshot: Callable[[str, str], object]        # (session, window) -> features
    policy: Callable[[object, dict[str, float] | None], Candidate]
    realise: Callable[[Candidate, str], float]         # -> return% for that session
    window: str = "MOC"
    _weights: dict[str, float] | None = field(default=None, init=False)

    def run(self) -> list[tuple[str, Candidate, float]]:
        out: list[tuple[str, Candidate, float]] = []
        for session in self.sessions:
            snapshot = self.load_snapshot(session, self.window)
            candidate = self.policy(snapshot, self._weights)
            if candidate.generated_at != session:
                raise LeakageError(
                    f"policy returned a candidate stamped {candidate.generated_at!r} "
                    f"while evaluating session {session!r}"
                )
            realised = self.realise(candidate, session)
            out.append((session, candidate, realised))
            self._weights = candidate.weights
        return out
