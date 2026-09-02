"""Port of the ``etc.py`` multipliers - the part ``simst`` does not apply.

``Investing/bin/simst`` gives you the window, the clip and the ramp and nothing else.
Four multipliers and one hard gate live only in ``etc.py``, and between them they can
take a healthy raw score to zero. Applied in upstream's order:

    1. inception gate      x 0 or 1   (over the FULL history, not the window)
    2. dedupe / similarity x min(gap_days/30, 1) for the LATER submission
    3. inactivity (DEC)    x 1 - min((last/20)^2, 1)  once track age > 5
    4. cash and shorts     x clamp(1 - cash, 0.01, 1)
    5. class normalisation each class rescaled to its /ratio share
    6. burn                UID 0 receives sum(score)/sum(ra)*(1-sum(ra))
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from .constants import (
    CASH_DECAY,
    CASH_RESIDUE,
    DAYS_FINAL,
    DD_POWER,
    DD_TRIGGER,
    DEC1_CLIFF,
    DEC1_DECAY,
    DEC1_START,
    DEC_UID,
    DEDUPE_DIVISOR,
)

__all__ = [
    "inception_gate",
    "inactivity_multiplier",
    "cash_multiplier",
    "dedupe_multipliers",
    "normalise_classes",
    "burn_share",
]


def inception_gate(first_swap_open: float, last_swap_close: float) -> int:
    """The undocumented lifetime high-water gate.

    Upstream groups the FULL ``/pnl`` frame (up to 144 rows), not the 50-session window::

        c = int(dd['swap_close'].iat[-1] >= dd['swap_open'].iat[0])
        sc['score'] *= sc['c']

    Below your all-time starting value you score exactly zero, no matter how good the
    trailing window is. Currently zeroing 29% of the field. Guide §3.2.
    """
    return 1 if last_swap_close >= first_swap_open else 0


def inactivity_multiplier(last_active_days: float, lifetime_days: int) -> float:
    """DEC. The ``> 5`` guard is on TRACK AGE, not on idle time.

    Once your track is older than five sessions the penalty applies from the FIRST idle
    day - there is no grace period. ``last_active_days`` is in CALENDAR days (it runs
    over weekends while a stock book earns no P&L). Guide §3.4.
    """
    if lifetime_days <= DEC1_START:
        return 1.0
    dec = (last_active_days / DEC1_CLIFF) ** DEC1_DECAY
    return 1.0 - min(max(dec, 0.0), 1.0)


def cash_multiplier(cash_and_short: float) -> float:
    """Linear from the first basis point, floored at 1%.

    Nothing here is reserved for "excessive" cash: a 10% sleeve costs 10% of score, and
    the 16 cash-classified ETFs count toward it. Guide §3.5, §6.1.
    """
    return min(max((1.0 - cash_and_short) ** CASH_DECAY, CASH_RESIDUE), 1.0)


def dedupe_multipliers(
    dist_payload: Sequence,
    *,
    trigger: float | None = None,
) -> dict[int, float]:
    """Port of ``etc.dedupe()``. Consumes the raw ``/dist`` payload.

    Payload shape is ``[class0, class1, ..., [blacklist, whitelist, trigger_override]]``
    where each class block alternates distance rows and block-delta rows, both prefixed
    by ``[uid, hotkey, ...]``.

    Two properties worth stating explicitly:

    * The penalty lands on whoever holds the **later** submission - upstream keeps only
      ``du >= 0``, i.e. "I am later". The first submitter is never penalised (empty ->
      no entry -> no multiplier). A copier who clones your book and never resubmits
      flips the penalty onto you at your next rebalance. Guide §3.3.
    * A UID with no near-duplicate gets NO entry, which upstream treats as "skip",
      not as ``1.0``. Callers must distinguish absent from 1.0 only if they care about
      the distinction; multiplying by a missing key is a no-op either way.

    Returns:
        ``{uid: multiplier}``, containing only UIDs that actually carry one.
    """
    if not dist_payload:
        return {}

    tail = dist_payload[-1]
    blacklist = list(tail[0]) if len(tail) > 0 else []
    whitelist = list(tail[1]) if len(tail) > 1 else []
    override = tail[2] if len(tail) > 2 else 0.0
    dd_trigger = trigger if trigger is not None else (override or DD_TRIGGER)

    out: dict[int, float] = {}
    for cls in range(len(dist_payload) - 1):
        block = dist_payload[cls]
        if not block:
            continue
        divisor = DEDUPE_DIVISOR[cls] if cls < len(DEDUPE_DIVISOR) else DEDUPE_DIVISOR[-1]
        dist_rows = block[0::2]
        delta_rows = block[1::2]
        uids = [row[0] for row in dist_rows]
        for k, row in enumerate(dist_rows):
            distances = row[2:]
            deltas = delta_rows[k][2:]
            candidates = [
                deltas[j]
                for j in range(len(distances))
                if j != k and distances[j] < dd_trigger and deltas[j] >= 0
            ]
            if candidates:
                gap_days = min(candidates) / divisor
                out[uids[k]] = min((gap_days / DAYS_FINAL) ** DD_POWER, 1.0)

    for uid in blacklist:
        out[uid] = 0.0      # owner kill switch - score forced to zero
    for uid in whitelist:
        out[uid] = 1.0      # exempt from the similarity penalty

    return out


def normalise_classes(
    scores: Mapping[int, float],
    classes: Mapping[int, int],
    ratio: Sequence[float],
) -> dict[int, float]:
    """Rescale each asset class's summed score to its ``/ratio`` share.

    Upstream::

        scz = sc['score'].sum()
        for a in range(len(ra)):
            sca = sc[sc['a'] == a]['score'].sum()
            if sca: sc.loc[sc['a'] == a, 'score'] *= ra[a] * scz / sca
    """
    total = sum(scores.values())
    out = dict(scores)
    for a in range(len(ratio)):
        class_sum = sum(v for uid, v in scores.items() if classes.get(uid) == a)
        if class_sum:
            factor = ratio[a] * total / class_sum
            for uid in out:
                if classes.get(uid) == a:
                    out[uid] *= factor
    return out


def burn_share(normalised_scores: Iterable[float], ratio: Sequence[float]) -> float:
    """Score routed to UID 0 when the class ratios sum to less than 1.

    ``score[DEC_UID] = sum(score) / sum(ra) * (1 - sum(ra))``. With ``/ratio = [0.5, 0.5]``
    this is zero. Publishing ``[0.25, 0.25]`` would divert half of all miner emission
    with no code change and no announcement. Poll it. Guide §3.6.
    """
    rsum = sum(ratio)
    total = sum(normalised_scores)
    if not rsum:
        return 1.0
    return total / rsum * (1.0 - rsum)


BURN_UID = DEC_UID
