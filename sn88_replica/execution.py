"""Fill semantics for US equities, ported verbatim from ``pldaily1``.

A replica reproduces upstream, including anything that looks like a quirk. Where the
code does something dimensionally surprising it is transcribed as written and flagged,
never silently "corrected" - a corrected replica disagrees with the validators, which is
the one thing it must never do.

Three rules decide whether your file trades at all::

    bk0, bk1 = first and last benchmark (SPY) bar of the session  # unix seconds
    MOO fills:  block <  bk0 - 120           # STK_MOO = -120  -> 09:28 ET
    MOC fills:  bk0 - 120 <= block < bk1 - 600   # STK_MOC = -600  -> 15:50 ET
    within a window, only the LAST submission executes

and one that catches new miners::

    if block == bk1 and no fill was found:
        fb = submissions with block >= bk1 - 600      # i.e. AFTER the MOC cutoff
        if the miner's FIRST ever strategy is in there (init == 1):
            netuid, alloc = '', 1                    # <-- forced to 100% CASH
        else:
            fb = []                                  # ignored, waits for the next MOO

On an early-close session the close is 13:00 ET, so the MOC cutoff is **12:50 ET**, not
15:50. Hard-coding the wall-clock time misses the window entirely on those days - use
:class:`sn88_replica.calendar.SessionCalendar`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from .calendar import Session
from .constants import STK_FEE

__all__ = [
    "Submission",
    "FillDecision",
    "resolve_fill",
    "first_submission_trap",
    "rebalance_shares",
    "turnover_cost",
]


@dataclass(frozen=True)
class Submission:
    """One strategy file arriving at the API. ``block`` is a unix second for equities."""

    block: int
    weights: Mapping[str, float]
    first_ever: bool = False        # upstream's `init == 1`

    def at(self) -> datetime:
        from datetime import timezone
        return datetime.fromtimestamp(self.block, tz=timezone.utc)


@dataclass(frozen=True)
class FillDecision:
    window: str | None              # "MOO", "MOC", or None if nothing fills
    submission: Submission | None
    forced_all_cash: bool = False
    reason: str = ""

    @property
    def fills(self) -> bool:
        return self.submission is not None or self.forced_all_cash


def resolve_fill(
    submissions: Sequence[Submission],
    session: Session,
    window: str,
) -> FillDecision:
    """Which submission - if any - executes in this window.

    Args:
        submissions: every submission the miner has made, any order.
        session: from :meth:`SessionCalendar.session`.
        window: ``"MOO"`` or ``"MOC"``.
    """
    w = window.upper()
    moo = session.moo_cutoff_block
    moc = session.moc_cutoff_block

    if w == "MOO":
        eligible = [s for s in submissions if s.block < moo]
    elif w == "MOC":
        eligible = [s for s in submissions if moo <= s.block < moc]
    else:
        raise ValueError(f"unknown window {window!r}")

    if eligible:
        winner = max(eligible, key=lambda s: s.block)   # only the LAST one executes
        dropped = len(eligible) - 1
        return FillDecision(
            window=w,
            submission=winner,
            reason=(f"last of {len(eligible)} in window"
                    + (f"; {dropped} earlier submission(s) discarded" if dropped else "")),
        )

    if w == "MOC":
        return first_submission_trap(submissions, session)

    return FillDecision(window=None, submission=None, reason="nothing submitted before the MOO cutoff")


def first_submission_trap(
    submissions: Sequence[Submission],
    session: Session,
) -> FillDecision:
    """The new-miner trap in ``pldaily1``'s third branch.

    A submission arriving **after** the MOC cutoff normally just waits for the next
    session's open. The exception: if it is the miner's very first strategy, upstream
    forces ``netuid=''`` and ``alloc=1`` - the book is initialised to **100% cash**.

    Consequences for a first-day plan: the inception anchor is set on an all-cash book,
    and the cash multiplier is linear, so day one is spent at a near-total score penalty.
    Publish your first strategy before 15:50 ET (12:50 on an early close).
    """
    late = [s for s in submissions if s.block >= session.moc_cutoff_block]
    if not late:
        return FillDecision(window=None, submission=None, reason="no submission in or after the window")

    newest = max(late, key=lambda s: s.block)
    if any(s.first_ever for s in late):
        return FillDecision(
            window="MOC",
            submission=newest,
            forced_all_cash=True,
            reason=("FIRST-EVER strategy arrived after the MOC cutoff: upstream forces the "
                    "book to 100% CASH for this session"),
        )
    return FillDecision(
        window=None,
        submission=None,
        reason="submitted after the MOC cutoff; ignored, waits for the next session's MOO",
    )


def rebalance_shares(
    current_shares: Mapping[str, float],
    target_weights: Mapping[str, float],
    prices: Mapping[str, float],
    portfolio_value: float,
) -> dict[str, float]:
    """Post-fee share counts, transcribed from ``pldaily1``.

    Upstream, per ticker::

        alpha  = (init * fund or swap) * alloc / price
        diffa  = alpha - current_alpha
        alpha -= abs(diffa) * STK_FEE / price * (alloc >= 0) * bool(netuid)

    The ``/ price`` resolves the units: the deduction is ``|diffa| * STK_FEE / price``
    **shares**, so its dollar value is ``|diffa| * STK_FEE``. Since ``diffa`` is a share
    count, **STK_FEE is $0.002 per share traded - a per-share commission, not 0.2% of
    notional.** On a $150 stock that is 1.3 basis points of the traded value, roughly
    150x cheaper than a percentage reading suggests.

    Combined with the fact that market impact never enters the score at all (stock
    ``swap`` is identically ``value``), **turnover is very nearly free in this
    simulation**. See :func:`fee_dollars`.

    The fee is skipped for short targets (``alloc < 0``) and for the cash key
    (``bool(netuid)`` is false for ``''``).
    """
    out: dict[str, float] = {}
    for ticker, alloc in target_weights.items():
        if ticker in ("_", "*", "=", "-+"):
            continue
        price = prices.get(ticker)
        if not price:
            continue
        target = portfolio_value * alloc / price
        diff = target - current_shares.get(ticker, 0.0)
        charge = abs(diff) * STK_FEE / price if (alloc >= 0 and ticker != "") else 0.0
        out[ticker] = target - charge
    return out


def fee_dollars(shares_traded: float) -> float:
    """Dollar cost of a rebalance: ``$0.002 per share``, and nothing else.

    There is no percentage component and no market impact - stock ``swap`` is identically
    ``value``, so the simulator never charges you for size. For a $10M book at $150/share,
    15% one-way turnover trades ~10,000 shares and costs about **$20, or 0.02bp**.
    """
    return abs(shares_traded) * STK_FEE


def turnover_cost(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    """One-way turnover, ``0.5 * sum|Δw|`` - the number the no-trade band is sized on.

    Note this is a *risk* control, not a cost control. An earlier revision priced
    turnover at 0.2% of notional and concluded a 20% cap was "fatal as an operating
    point"; at $0.002/share the fee is ~150x smaller and turnover is nearly free.
    Constrain turnover because high-turnover signals tend not to survive out of sample
    (Novy-Marx & Velikov), not because the simulator charges you for it.
    """
    names = set(current_weights) | set(target_weights)
    names -= {"_", "*", "=", "-+"}
    return 0.5 * sum(
        abs(target_weights.get(n, 0.0) - current_weights.get(n, 0.0)) for n in names
    )
