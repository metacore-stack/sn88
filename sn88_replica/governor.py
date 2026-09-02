"""The risk governor: final authority, and the only component that can say no.

Guide §15. It vetoes; it never proposes. The model cannot disable it, and every
rejection is logged with a machine-readable reason.

**What makes this different from an ordinary portfolio risk system.** The binding risk
on SN88 is not a rolling drawdown budget - it is an absolute floor. ``etc.py`` multiplies
every score by ``int(last_swap_close >= first_swap_open)`` computed over the *entire*
history, so below your all-time starting capital you earn exactly zero however good the
trailing fifty sessions were. On live data 65 of 226 UIDs sit under that line, and the
25th percentile of since-inception return is only -0.36%: a quarter of the field is one
bad session from losing all emission.

So the governor defends **distance to inception**, not variance.

**Recovery mode is a survival problem, not a score problem.** An earlier revision of the
guide reasoned that below the floor your score is already zero, so the rational response
is more risk. That is wrong and dangerous: it optimises today's score in isolation and
ignores that more risk can push you further under, lengthen expected recovery, burn fees
and get you pruned before you ever climb back. The gate is absorbing in *score* terms but
not in *capital* terms - you keep trading either way. :class:`RecoveryPolicy` therefore
solves for expected discounted incentive net of pruning risk and delay, and its answer
depends on headroom, edge and pruning pressure rather than on a hard-coded direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .strategy import RESERVED, Strategy
from .universe import Universe

__all__ = [
    "RiskState",
    "Verdict",
    "Veto",
    "GovernorConfig",
    "RiskGovernor",
    "RecoveryPolicy",
    "RecoveryStance",
]


class RiskState(str, Enum):
    """Where the book sits relative to the inception floor."""

    NORMAL = "normal"
    WARN = "warn"           # scale-ups blocked
    CRITICAL = "critical"   # safest admissible candidate forced
    BREACHED = "breached"   # below inception: score is already zero


@dataclass(frozen=True)
class Veto:
    rule: str
    detail: str
    fatal: bool = True


@dataclass(frozen=True)
class Verdict:
    approved: bool
    state: RiskState
    vetoes: tuple[Veto, ...] = ()
    headroom: float = 0.0
    notes: tuple[str, ...] = ()

    def reason(self) -> str:
        return "; ".join(f"{v.rule}: {v.detail}" for v in self.vetoes) or "approved"


@dataclass
class GovernorConfig:
    """Limits. Changing any of these is a manual-approval action (guide §31)."""

    inception_equity: float = 10_000_000.0

    # the floor budget, as a fraction of inception
    floor_warn: float = 0.03
    floor_crit: float = 0.01

    # hard constraints - never relaxed, in any state
    max_gross: float = 1.0
    min_gross: float = 0.99
    max_cash: float = 0.01
    allow_shorts: bool = False
    max_single_name: float = 0.10
    max_sector: float = 0.25
    min_holdings: int = 15
    max_holdings: int = 40

    # scale controls, tightened by state rather than by a fixed vol target
    max_turnover: float = 0.15
    tqqq_max_weight: float = 0.10
    tqqq_tickers: tuple[str, ...] = ("TQQQ",)

    # survival
    max_breach_probability: float = 0.05
    horizon_sessions: int = 20


class RiskGovernor:
    """Approve or veto a candidate. Never modifies it, never proposes an alternative."""

    def __init__(self, config: GovernorConfig | None = None):
        self.config = config or GovernorConfig()

    # --- state -----------------------------------------------------------

    def headroom(self, current_equity: float) -> float:
        """``current / inception - 1``. Negative means the score gate is already shut.

        This is what the gate actually tests - **current** equity against inception, not
        the minimum ever reached. A book that dipped under and recovered is scoring again.
        """
        return current_equity / self.config.inception_equity - 1.0

    def closest_approach(self, min_equity_since_inception: float) -> float:
        """Headroom at the worst point so far - risk telemetry, not a gate.

        The gate is memoryless, but the *near misses* are the signal that a strategy is
        sized too close to the floor. A book that has repeatedly come within a fraction
        of a percent has been lucky, not safe, and the optimiser should see that even
        when today's headroom looks comfortable. Live, the 25th percentile of
        since-inception return is -0.36%: a quarter of the field is one session away.
        """
        return min_equity_since_inception / self.config.inception_equity - 1.0

    def state(self, current_equity: float) -> RiskState:
        h = self.headroom(current_equity)
        if h < 0:
            return RiskState.BREACHED
        if h < self.config.floor_crit:
            return RiskState.CRITICAL
        if h < self.config.floor_warn:
            return RiskState.WARN
        return RiskState.NORMAL

    # --- the decision ----------------------------------------------------

    def review(
        self,
        strategy: Strategy,
        universe: Universe,
        *,
        current_equity: float,
        previous_weights: Mapping[str, float] | None = None,
        breach_probability: float | None = None,
        realised_vol: float | None = None,
        previous_vol: float | None = None,
        min_equity_since_inception: float | None = None,
    ) -> Verdict:
        """Every hard constraint, then the state-dependent ones.

        Hard constraints apply in **every** state including BREACHED - a breach is not a
        licence to abandon concentration or gross limits.
        """
        cfg = self.config
        vetoes: list[Veto] = []
        notes: list[str] = []
        h = self.headroom(current_equity)
        state = self.state(current_equity)

        # --- hard: things upstream would silently punish ------------------
        gross = strategy.gross()
        if gross > cfg.max_gross:
            vetoes.append(Veto("gross_cap",
                               f"{gross!r} > {cfg.max_gross}: upstream silently discards "
                               f"the file and keeps the previous allocation live"))
        if gross < cfg.min_gross - 1e-9:
            vetoes.append(Veto("min_gross",
                               f"{gross:.6f} < {cfg.min_gross}: the leftover becomes cash "
                               f"and the cash multiplier is linear"))
        if strategy.asset_class != 1:
            vetoes.append(Veto("asset_class",
                               "'_':1 is mandatory; omitting it selects the other class "
                               "permanently"))

        shorts = strategy.shorts()
        if shorts and not cfg.allow_shorts:
            vetoes.append(Veto("no_shorts",
                               f"{sorted(shorts)}: shorts count toward the cash penalty "
                               f"and consume gross budget"))

        tickers = strategy.tickers()
        unknown = [t for t in tickers if not universe.is_tradable(t)]
        if unknown:
            vetoes.append(Veto("unknown_ticker",
                               f"{sorted(unknown)} absent from today's /assets: upstream "
                               f"silently reclassifies them to cash"))

        cashish = [t for t in tickers if universe.is_cash(t)]
        cash_weight = (1.0 - gross) + sum(abs(strategy.weights[t]) for t in cashish)
        if cash_weight > cfg.max_cash + 1e-9:
            vetoes.append(Veto("cash_band",
                               f"effective cash {cash_weight:.4f} > {cfg.max_cash} "
                               f"(score multiplier {max(1 - cash_weight, 0.01):.4f}x)"
                               + (f"; cash-classified: {sorted(cashish)}" if cashish else "")))

        for t in tickers:
            w = abs(strategy.weights[t])
            if w > cfg.max_single_name + 1e-9:
                vetoes.append(Veto("single_name", f"{t} {w:.4f} > {cfg.max_single_name}"))

        by_sector: dict[str, float] = {}
        for t in tickers:
            asset = universe.assets.get(t)
            if asset:
                by_sector[asset.sector] = by_sector.get(asset.sector, 0.0) + abs(strategy.weights[t])
        for sector, w in sorted(by_sector.items()):
            if w > cfg.max_sector + 1e-9:
                vetoes.append(Veto("sector_cap", f"{sector!r} {w:.4f} > {cfg.max_sector}"))

        if not (cfg.min_holdings <= len(tickers) <= cfg.max_holdings):
            vetoes.append(Veto("holdings",
                               f"{len(tickers)} outside [{cfg.min_holdings}, {cfg.max_holdings}]"))

        if previous_weights is not None:
            names = (set(previous_weights) | set(strategy.weights)) - set(RESERVED)
            turnover = 0.5 * sum(
                abs(strategy.weights.get(n, 0.0) - previous_weights.get(n, 0.0)) for n in names
            )
            if turnover > cfg.max_turnover + 1e-9:
                vetoes.append(Veto("turnover", f"{turnover:.4f} > {cfg.max_turnover}"))

        # --- survival -----------------------------------------------------
        if breach_probability is not None and breach_probability > cfg.max_breach_probability:
            vetoes.append(Veto(
                "breach_probability",
                f"P(below inception within {cfg.horizon_sessions} sessions) = "
                f"{breach_probability:.3f} > {cfg.max_breach_probability}"))

        # --- state-dependent scale controls -------------------------------
        leveraged = [t for t in tickers if t in cfg.tqqq_tickers]
        lev_weight = sum(abs(strategy.weights[t]) for t in leveraged)
        if lev_weight > cfg.tqqq_max_weight + 1e-9:
            vetoes.append(Veto("leveraged_cap",
                               f"{sorted(leveraged)} {lev_weight:.4f} > {cfg.tqqq_max_weight}"))

        if state in (RiskState.WARN, RiskState.CRITICAL, RiskState.BREACHED):
            if lev_weight > 0:
                vetoes.append(Veto("leveraged_below_floor",
                                   f"leveraged exposure forbidden at headroom {h:.4f} "
                                   f"(state {state.value})"))
            if (realised_vol is not None and previous_vol is not None
                    and realised_vol > previous_vol + 1e-12):
                vetoes.append(Veto("scale_up_below_floor",
                                   f"volatility {realised_vol:.4f} > current "
                                   f"{previous_vol:.4f} at headroom {h:.4f}"))
            notes.append(f"headroom {h:.4f} - scale-ups blocked")

        if min_equity_since_inception is not None:
            approach = self.closest_approach(min_equity_since_inception)
            if approach < 0 and state is not RiskState.BREACHED:
                notes.append(
                    f"has been BELOW inception before (worst headroom {approach:.4f}) - "
                    f"currently scoring again, but this book has already proved it is "
                    f"sized close enough to the floor to cross it")
            elif 0 <= approach < cfg.floor_warn:
                notes.append(f"closest approach to the floor so far: {approach:.4f}")

        if state is RiskState.BREACHED:
            notes.append("BELOW INCEPTION: score is already zero. Recovery mode applies "
                         "(RecoveryPolicy) - more risk is NOT automatically correct.")

        return Verdict(
            approved=not vetoes,
            state=state,
            vetoes=tuple(vetoes),
            headroom=h,
            notes=tuple(notes),
        )


# --- recovery ---------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryStance:
    """What the recovery simulator recommends, and why."""

    target_vol: float
    expected_sessions_to_recover: float
    probability_of_recovery: float
    probability_of_pruning: float
    objective: float
    rationale: str


class RecoveryPolicy:
    """Choose a risk level below the floor by simulating, not by rule.

    Maximises::

        E[ sum_t discount^t * incentive_t ] - lambda * P(pruned before recovery)
                                            - gamma * E[sessions to recover]

    Whether that implies more risk, less risk, or a differently structured risk depends
    on how far under you are, how much edge you have, and how much pruning pressure
    exists - all of which move daily. The point of the simulator is that none of those is
    assumed.
    """

    def __init__(
        self,
        *,
        horizon: int = 60,
        discount: float = 0.99,
        prune_penalty: float = 5.0,
        delay_penalty: float = 0.02,
        paths: int = 2000,
        seed: int = 20260901,
    ):
        self.horizon = horizon
        self.discount = discount
        self.prune_penalty = prune_penalty
        self.delay_penalty = delay_penalty
        self.paths = paths
        self.seed = seed

    def evaluate(
        self,
        *,
        headroom: float,
        daily_sharpe: float,
        candidate_vols: Sequence[float],
        prune_hazard_per_session: float = 0.0,
        prune_hazard_multiplier_when_zero: float = 3.0,
    ) -> list[RecoveryStance]:
        """Score each candidate volatility. ``headroom`` is negative below the floor.

        The simulation runs the **whole horizon** rather than stopping at first recovery,
        because the gate is not one-shot: every session above the inception line earns and
        every session below earns nothing, and a book that climbs back on high volatility
        also relapses on it. Stopping at first recovery makes more risk unconditionally
        better, which is precisely the reasoning error this class exists to avoid.

        Pruning hazard is elevated while the score is zero - a miner earning nothing sinks
        toward the bottom of the incentive distribution, which is the pruning queue.
        """
        import random

        out: list[RecoveryStance] = []
        for vol in candidate_vols:
            rng = random.Random(self.seed)
            earned = 0.0
            recovered = 0
            pruned = 0
            first_recovery_total = 0.0
            for _ in range(self.paths):
                equity = 1.0 + headroom
                ever_recovered = False
                first_recovery = self.horizon
                was_pruned = False
                for t in range(self.horizon):
                    equity *= 1 + rng.gauss(daily_sharpe * vol, vol)
                    above = equity >= 1.0
                    if above:
                        earned += self.discount ** t          # only above the floor
                        if not ever_recovered:
                            ever_recovered = True
                            first_recovery = t + 1
                    hazard = prune_hazard_per_session * (
                        1.0 if above else prune_hazard_multiplier_when_zero
                    )
                    if hazard and rng.random() < hazard:
                        was_pruned = True
                        break
                if ever_recovered:
                    recovered += 1
                if was_pruned:
                    pruned += 1
                first_recovery_total += first_recovery

            n = self.paths
            p_rec = recovered / n
            p_prune = pruned / n
            mean_first = first_recovery_total / n
            objective = (earned / n
                         - self.prune_penalty * p_prune
                         - self.delay_penalty * mean_first)
            out.append(RecoveryStance(
                target_vol=vol,
                expected_sessions_to_recover=mean_first,
                probability_of_recovery=p_rec,
                probability_of_pruning=p_prune,
                objective=objective,
                rationale=(f"P(recover)={p_rec:.2f} P(prune)={p_prune:.3f} "
                           f"E[sessions to recover]={mean_first:.1f} "
                           f"earning sessions={earned / n:.1f}"),
            ))
        return sorted(out, key=lambda s: -s.objective)

    def recommend(self, **kwargs) -> RecoveryStance:
        stances = self.evaluate(**kwargs)
        if not stances:
            raise ValueError("no candidate volatilities supplied")
        return stances[0]
