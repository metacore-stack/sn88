"""Byte-faithful port of ``simst.py`` ``pl2sc()`` + ``score()``.

Reproduces the live ``/score`` table to a max relative error of 6.8e-8 across all
226 UIDs (median 1.3e-12). See ``tests/test_golden.py``.

Four details that are easy to get wrong, and each of which was verified against
live data rather than assumed:

1. **Two different day counts.** ``pl2sc`` reads ``days = len(dd)`` *before*
   truncating to the window, and writes that LIFETIME count into the score row (it
   drives the ramp and the inactivity age guard). ``score()`` then recomputes its own
   ``days = len(dd)`` on the WINDOWED frame, which drives ``daily`` and the MAR floor.

2. **The clip loop iterates in descending ``pnl%`` order, not index order.** Upstream
   is ``for i in ii:`` where ``ii = dd[dd['pnl%']>0].sort_values('pnl%')[::-1][:N+1].index``.
   The equity rebase is order-dependent (two clipped indices i<j differ by ``d_i*clip/100``).
   Using index order instead produces up to 12% error on live rows.

3. **``pnl`` is computed before clipping and only overwritten at clipped indices.**
   Non-clipped entries keep their ORIGINAL diff, so ``lsr`` sees a mixed series.

4. **``gain`` is compounded, ``risk`` is arithmetic.** ``gain`` comes from the rebased
   ``swap_close`` while ``drawdown()`` runs on the cumsum of ``pnl%``. That wedge is why
   ``mar`` is not exactly scale-invariant (guide §2.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .constants import (
    CLIP_DEFAULT,
    CLIP_OUTLIERS,
    DAYS_DELAY,
    DAYS_FINAL,
    RISK_INIT_STK,
    WIN_SIZE_STK,
)

__all__ = ["ScoreTerms", "kelly", "drawdown", "window_and_clip", "score_terms", "raw_score"]


@dataclass(frozen=True)
class ScoreTerms:
    """The four multiplied terms plus their inputs, exactly as upstream reports them."""

    score: float
    gain: float          # return% over the window, from the REBASED equity curve
    risk: float          # max drawdown of the cumsum of clipped pnl%
    mar: float
    lsr: float
    odds: float
    daily: float
    apr: float
    swap: float          # final rebased swap_close ("clip" in upstream's printout)
    windowed_days: int
    lifetime_days: int


def kelly(p: float, b: float) -> float:
    """Upstream's Kelly fraction. Identically equal to ``mean / pavg`` (guide §2.5)."""
    return (p * (b + 1) - 1) / b


def drawdown(pct: Sequence[float]) -> float:
    """Max drawdown of the ARITHMETIC cumsum of pnl% - not of the equity curve."""
    peak = down = cum = 0.0
    for x in pct:
        cum += x
        peak = max(peak, cum)
        down = max(down, peak - cum)
    return down


def window_and_clip(
    swap_open0: float,
    swap_closes: Sequence[float],
    *,
    clip_outliers: int = CLIP_OUTLIERS,
    clip_default: float = CLIP_DEFAULT,
) -> tuple[list[float], list[float], list[float], float]:
    """Apply the outlier clip and rebase the equity curve.

    Args:
        swap_open0: ``swap_open`` of the first session IN THE WINDOW.
        swap_closes: ``swap_close`` for each session in the window, in date order.

    Returns:
        ``(rebased_swap_close, pnl, pnl_pct, clip_value)``.
    """
    n = len(swap_closes)
    if n == 0:
        raise ValueError("empty window")
    s = list(swap_closes)
    init = swap_open0

    # pnl / pnl% from the ORIGINAL series, before any clipping
    pnl = [0.0] * n
    pct = [0.0] * n
    pnl[0] = s[0] - init
    pct[0] = pnl[0] / init * 100
    for i in range(1, n):
        pnl[i] = s[i] - s[i - 1]
        pct[i] = pnl[i] / s[i - 1] * 100

    positive = [i for i in range(n) if pct[i] > 0]
    if not positive:
        return s, pnl, pct, 0.0

    # mimic pandas `sort_values('pnl%')[::-1]`: stable ascending, then reversed
    ii = sorted(positive, key=lambda i: (pct[i], i))[::-1][: clip_outliers + 1]

    if len(ii) == clip_outliers + 1:
        clip = pct[ii[-1]]                       # level the top N to the (N+1)th
    else:
        clip = min(clip_default, pct[ii[-1]])    # <= N positive sessions: cut to 1%

    for i in ii:
        pct[i] = clip

    # ORDER MATTERS - descending pnl%, exactly as upstream iterates `ii`
    for i in ii:
        opening = s[i - 1] if i else init
        delta = s[i] - opening * (1 + clip / 100)
        for j in range(i, n):
            s[j] -= delta
        pnl[i] = opening * clip / 100

    return s, pnl, pct, clip


def score_terms(
    swap_open0: float,
    swap_closes: Sequence[float],
    *,
    risk_init: float = RISK_INIT_STK,
    lifetime_days: int | None = None,
) -> ScoreTerms:
    """Port of ``simst.score()``, operating on an already-windowed series.

    ``lifetime_days`` is carried through for the ramp and the inactivity age guard;
    it is NOT used by any term here. Pass the full history length, not the window.
    """
    s, pnl, pct, _clip = window_and_clip(swap_open0, swap_closes)
    n = len(s)
    init = swap_open0

    ppos = [x for x in pct if x > 0]
    pneg = [x for x in pct if x < 0]
    prob = len(ppos) / n

    gain = max(s[-1] - init, -init) / init * 100
    risk = drawdown(pct)
    daily = ((1 + gain / 100) ** (1 / n) - 1) * 100
    apr = ((1 + daily / 100) ** 365 - 1) * 100
    mar = gain / max(risk, risk_init / n**0.5)

    denom = sum(abs(x) for x in pnl)
    lsr = sum(pnl) / (denom if denom else 1e18)

    if not ppos or not pneg:
        odds = prob * 100                        # the NaN branch: lavg or pavg undefined
    else:
        pavg = sum(ppos) / len(ppos)
        lavg = -sum(pneg) / len(pneg)
        odds = 50 + kelly(prob, pavg / lavg) / 2 * 100
    if odds <= 0 or math.isnan(odds):
        odds = 0.0 if not math.isnan(odds) else prob * 100

    value = mar * lsr * odds * daily
    if value <= 0:
        value = 0.0

    return ScoreTerms(
        score=value,
        gain=gain,
        risk=risk,
        mar=mar,
        lsr=lsr,
        odds=odds,
        daily=daily,
        apr=apr,
        swap=s[-1],
        windowed_days=n,
        lifetime_days=lifetime_days if lifetime_days is not None else n,
    )


def raw_score(
    swap_open_first_in_window: float,
    swap_closes_window: Sequence[float],
    lifetime_days: int,
    *,
    risk_init: float = RISK_INIT_STK,
) -> ScoreTerms:
    """``pl2sc`` end to end: clip, score, then the new-miner ramp.

    This is what the live ``/score`` endpoint publishes - it already includes the ramp
    but NOT the ``etc.py`` multipliers (gate, dedupe, inactivity, cash, class ratio).
    Those live in :mod:`sn88_replica.multipliers`.
    """
    terms = score_terms(
        swap_open_first_in_window,
        swap_closes_window,
        risk_init=risk_init,
        lifetime_days=lifetime_days,
    )
    value = terms.score
    if lifetime_days < DAYS_FINAL:
        value *= (lifetime_days / DAYS_FINAL) ** DAYS_DELAY
    return ScoreTerms(**{**terms.__dict__, "score": value})


def window(sessions: Sequence, size: int = WIN_SIZE_STK) -> list:
    """Trailing scoring window. Sessions must already be sorted by date ascending."""
    return list(sessions[-size:])
