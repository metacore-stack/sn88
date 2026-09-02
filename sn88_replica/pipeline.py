"""Full-field scoring: raw score -> multipliers -> class normalisation -> incentive.

This is what ``Investing/bin/validator`` prints, plus the projected incentive share
that ``bin/validator`` does not compute for you.

Verified end to end against a direct chain query: this pipeline put the top miner at
11.79% where the chain reported 11.802% for the same UID.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Mapping, Sequence

from . import multipliers as mult
from .api import PNL_COLS, FieldRow
from .constants import ASSET_STOCKS, RISK_INIT_STK, WIN_SIZE_STK
from .scoring import ScoreTerms, raw_score

__all__ = ["MinerScore", "score_field", "group_pnl"]

_PI = {c: i for i, c in enumerate(PNL_COLS)}


@dataclass(frozen=True)
class MinerScore:
    uid: int
    asset: int
    terms: ScoreTerms
    gate: int
    dedupe: float | None
    inactivity: float
    cash: float
    raw: float           # post-clip, post-ramp (what /score publishes)
    adjusted: float      # after every etc.py multiplier, pre class-normalisation
    normalised: float    # after class normalisation
    incentive: float     # share of the whole miner pool, 0..1


def group_pnl(pnl_payload: Sequence[Sequence]) -> dict[int, list[Sequence]]:
    """Group the ``/pnl`` frame by UID, sorted by date ascending."""
    by: dict[int, list[Sequence]] = collections.defaultdict(list)
    for row in pnl_payload:
        by[row[_PI["uid"]]].append(row)
    for uid in by:
        by[uid].sort(key=lambda r: r[_PI["date"]])
    return dict(by)


def score_field(
    pnl_payload: Sequence[Sequence],
    days: Mapping[int, FieldRow],
    dist_payload: Sequence,
    ratio: Sequence[float],
    *,
    max_uid: int = 256,
    window: int = WIN_SIZE_STK,
    risk_init: float = RISK_INIT_STK,
) -> dict[int, MinerScore]:
    """Score every UID exactly as validators do, and derive projected incentive.

    ``days`` supplies ``cash`` and ``last`` - both are owner-computed numbers from
    ``/days``, not values a validator derives (guide §3.6).
    """
    by_uid = group_pnl(pnl_payload)
    dedupe = mult.dedupe_multipliers(dist_payload)

    raw: dict[int, ScoreTerms] = {}
    gates: dict[int, int] = {}
    classes: dict[int, int] = {}

    for uid, rows in by_uid.items():
        if uid >= max_uid:
            continue                      # benchmark rows live outside the UID space
        lifetime = len(rows)
        win = rows[-window:]
        first_open_all = rows[0][_PI["swap_open"]]
        last_close_all = rows[-1][_PI["swap_close"]]
        terms = raw_score(
            win[0][_PI["swap_open"]],
            [r[_PI["swap_close"]] for r in win],
            lifetime,
            risk_init=risk_init,
        )
        raw[uid] = terms
        gates[uid] = mult.inception_gate(first_open_all, last_close_all)
        classes[uid] = rows[0][_PI["asset"]]

    adjusted: dict[int, float] = {}
    parts: dict[int, tuple] = {}
    for uid, terms in raw.items():
        value = terms.score
        gate = gates[uid]
        value *= gate                                              # 1. inception gate

        ded = dedupe.get(uid)
        if ded is not None:
            value *= ded                                           # 2. similarity

        row = days.get(uid)
        inact = 1.0
        cash_m = 1.0
        if row is not None:
            inact = mult.inactivity_multiplier(row.last_active, terms.lifetime_days)
            value *= inact                                         # 3. inactivity
            cash_m = mult.cash_multiplier(row.cash)
            value *= cash_m                                        # 4. cash and shorts
        adjusted[uid] = value
        parts[uid] = (gate, ded, inact, cash_m)

    normalised = mult.normalise_classes(adjusted, classes, ratio)  # 5. class ratio
    burn = mult.burn_share(normalised.values(), ratio)             # 6. burn at UID 0
    denominator = sum(normalised.values()) + burn

    out: dict[int, MinerScore] = {}
    for uid, terms in raw.items():
        gate, ded, inact, cash_m = parts[uid]
        out[uid] = MinerScore(
            uid=uid,
            asset=classes[uid],
            terms=terms,
            gate=gate,
            dedupe=ded,
            inactivity=inact,
            cash=cash_m,
            raw=terms.score,
            adjusted=adjusted[uid],
            normalised=normalised[uid],
            incentive=(normalised[uid] / denominator) if denominator else 0.0,
        )
    return out


def stock_class(scores: Mapping[int, MinerScore]) -> dict[int, MinerScore]:
    """Just the US-equity miners."""
    return {u: s for u, s in scores.items() if s.asset == ASSET_STOCKS}


def leaderboard(scores: Mapping[int, MinerScore], *, top: int | None = None) -> list[MinerScore]:
    ranked = sorted(scores.values(), key=lambda s: -s.incentive)
    return ranked[:top] if top else ranked
