"""Signal research harness: walk-forward, leak-proof by construction, scored by the replica.

Build-order stage 5. A signal here is a pure function of a :class:`SignalView`, and the
view is the enforcement mechanism: for a MOO decision on session ``t`` it exposes
returns only through ``t-1``, because the opening auction has not run and today's bar
does not exist yet. There is no argument a signal can pass to widen that.

Two rules the harness imposes rather than suggests:

* **Rank through the replica, never through Sharpe or IC.** Selecting on those
  systematically favours signals whose payoff the outlier clip then confiscates, and
  neither sees the inception gate at all.
* **Weights are rebuilt every session from information available then.** Nothing is
  fitted on the evaluation window. A signal that needs calibration must do it inside
  the view, from data the view exposes.

The bar to beat is the null strategy: equal weight over the 20 most liquid non-cash
names scores a median of **2.112** out-of-sample across 53 rolling windows on the
524-name replication panel, and would rank about **50 of 151** on the live board. It
also scores **exactly zero in 32% of windows**, because a window that closes negative
scores zero - cutting that fraction is itself a large win. See ``tools/baseline_bar.py``.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from .scoring import raw_score

__all__ = [
    "Panel",
    "SignalView",
    "Signal",
    "EvalResult",
    "evaluate",
    "equal_weight_top",
    "inverse_vol_top",
    "load_panel",
]

WIN = 50
INITIAL_EQUITY = 10_000_000.0


@dataclass(frozen=True)
class Panel:
    """The replication track as a dense panel. Raw unadjusted closes."""

    sessions: tuple[str, ...]
    names: tuple[str, ...]
    close: Mapping[str, Sequence[float]]
    returns: Mapping[str, Sequence[float]]      # length len(sessions) - 1
    sector: Mapping[str, str] = field(default_factory=dict)
    dollar_volume: Mapping[str, float] = field(default_factory=dict)
    market_cap: Mapping[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sessions)


def load_panel(path: str | Path) -> Panel:
    raw = json.loads(Path(path).read_text())
    close = raw["close"]
    returns = {
        t: [(s[i] - s[i - 1]) / s[i - 1] for i in range(1, len(s))]
        for t, s in close.items()
    }
    return Panel(
        sessions=tuple(raw["sessions"]),
        names=tuple(raw["names"]),
        close=close,
        returns=returns,
        sector=raw.get("sector", {}),
        dollar_volume=raw.get("dollar_volume", {}),
        market_cap=raw.get("market_cap", {}),
    )


class SignalView:
    """Everything a signal may legally see when deciding session ``t`` at the MOO cutoff.

    ``returns[i]`` is the move from session ``i`` to ``i+1``, so a decision for session
    ``t`` may use indices ``0 .. t-2`` - the last complete move ends at the close of
    ``t-1``. Every accessor here respects that; none can be widened.
    """

    def __init__(self, panel: Panel, t: int):
        self.panel = panel
        self.t = t
        self._end = t - 1                      # exclusive bound on the returns index

    @property
    def names(self) -> tuple[str, ...]:
        return self.panel.names

    @property
    def session(self) -> str:
        return self.panel.sessions[self.t]

    def returns(self, ticker: str, lookback: int | None = None) -> list[float]:
        r = self.panel.returns[ticker][: self._end]
        return r[-lookback:] if lookback else r

    def closes(self, ticker: str, lookback: int | None = None) -> list[float]:
        # close[i] is session i's close; the last knowable one is t-1
        c = self.panel.close[ticker][: self.t]
        return c[-lookback:] if lookback else c

    def cumulative(self, ticker: str, lookback: int) -> float:
        """Compounded return over the last ``lookback`` complete sessions."""
        r = self.returns(ticker, lookback)
        out = 1.0
        for x in r:
            out *= 1 + x
        return out - 1.0

    def volatility(self, ticker: str, lookback: int = 20) -> float:
        r = self.returns(ticker, lookback)
        return statistics.pstdev(r) if len(r) > 1 else 0.0

    def downside_volatility(self, ticker: str, lookback: int = 20) -> float:
        r = [x for x in self.returns(ticker, lookback) if x < 0]
        return statistics.pstdev(r) if len(r) > 1 else 0.0

    def beta(self, ticker: str, benchmark: str = "SPY", lookback: int = 60) -> float:
        a = self.returns(ticker, lookback)
        b = self.returns(benchmark, lookback) if benchmark in self.panel.returns else []
        n = min(len(a), len(b))
        if n < 3:
            return 1.0
        a, b = a[-n:], b[-n:]
        mb = sum(b) / n
        var = sum((x - mb) ** 2 for x in b) / n
        if var <= 0:
            return 1.0
        ma = sum(a) / n
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
        return cov / var

    def hit_rate(self, ticker: str, lookback: int = 60) -> float:
        r = self.returns(ticker, lookback)
        return sum(1 for x in r if x > 0) / len(r) if r else 0.0

    def enough_history(self, lookback: int) -> bool:
        return self._end >= lookback


#: A signal scores every name; higher is better. Ties are broken by ticker.
Signal = Callable[[SignalView], Mapping[str, float]]


# --- weight construction ----------------------------------------------------


def equal_weight_top(scores: Mapping[str, float], n: int = 20,
                     gross: float = 0.99) -> dict[str, float]:
    picks = sorted(scores, key=lambda t: (-scores[t], t))[:n]
    return {t: gross / len(picks) for t in picks} if picks else {}


def inverse_vol_top(scores: Mapping[str, float], view: SignalView, n: int = 20,
                    gross: float = 0.99, lookback: int = 20) -> dict[str, float]:
    picks = sorted(scores, key=lambda t: (-scores[t], t))[:n]
    inv = {t: (1.0 / v if (v := view.volatility(t, lookback)) > 1e-9 else 0.0) for t in picks}
    total = sum(inv.values())
    if total <= 0:
        return equal_weight_top(scores, n, gross)
    return {t: w / total * gross for t, w in inv.items()}


# --- evaluation -------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    name: str
    median: float
    p25: float
    p75: float
    zero_fraction: float
    windows: int
    mean_turnover: float
    mean_holdings: float
    scores: tuple[float, ...] = ()

    def beats(self, bar: float) -> bool:
        return self.median > bar

    def __str__(self) -> str:
        return (f"{self.name:<30} median {self.median:>9.3f}  p25 {self.p25:>8.3f}  "
                f"p75 {self.p75:>9.3f}  zero {self.zero_fraction:>4.0%}  "
                f"turnover {self.mean_turnover:>5.1%}")


def evaluate(
    signal: Signal,
    panel: Panel,
    *,
    name: str = "signal",
    start_frac: float = 0.5,
    top_n: int = 20,
    weighting: str = "equal",
    rebalance_every: int = 1,
    no_trade_band: float = 0.0,
    max_windows: int | None = 60,
    warmup: int = 60,
) -> EvalResult:
    """Walk forward, rebalancing from the view, and score through the replica.

    Args:
        start_frac: evaluation begins here; everything before is available to the signal
            as history but never scored.
        weighting: ``"equal"`` or ``"inverse_vol"`` over the top ``top_n`` names.
        rebalance_every: sessions between rebalances. 1 = every session.
        no_trade_band: skip the rebalance when one-way turnover is below this. Skipped
            opportunities count as zero realised turnover, so ``mean_turnover`` is
            turnover per opportunity rather than per executed trade.
        warmup: minimum complete sessions before the signal is called at all.
    """
    n_sessions = len(panel.sessions)
    begin = max(warmup + 1, int(n_sessions * start_frac))
    equity = INITIAL_EQUITY
    curve: list[float] = []
    weights: dict[str, float] = {}
    turnovers: list[float] = []
    holdings: list[int] = []

    for t in range(begin, n_sessions - 1):
        if not weights or (t - begin) % max(1, rebalance_every) == 0:
            view = SignalView(panel, t)
            scores = signal(view)
            if scores:
                target = (equal_weight_top(scores, top_n) if weighting == "equal"
                          else inverse_vol_top(scores, view, top_n))
                names = set(target) | set(weights)
                turn = 0.5 * sum(abs(target.get(k, 0.0) - weights.get(k, 0.0)) for k in names)
                if turn >= no_trade_band or not weights:
                    turnovers.append(turn)
                    weights = target
                else:
                    # a skipped rebalance is zero REALISED turnover, and must be counted
                    # as such - otherwise a tight band looks like high turnover because
                    # only the (large) initial trade is ever recorded.
                    turnovers.append(0.0)
        if not weights:
            continue
        holdings.append(len(weights))
        # returns[t-1] is the move from session t-1 to session t: the book held into t
        step = sum(w * panel.returns[k][t - 1] for k, w in weights.items()
                   if t - 1 < len(panel.returns[k]))
        equity *= 1 + step
        curve.append(equity)

    ends = [i for i in range(WIN, len(curve))]
    if max_windows and len(ends) > max_windows:
        ends = ends[:: max(1, len(ends) // max_windows)]
    out: list[float] = []
    for e in ends:
        out.append(raw_score(INITIAL_EQUITY, curve[e - WIN:e], e).score)
    out.sort()
    m = len(out) or 1
    return EvalResult(
        name=name,
        median=out[m // 2] if out else 0.0,
        p25=out[m // 4] if out else 0.0,
        p75=out[3 * m // 4] if out else 0.0,
        zero_fraction=(sum(1 for x in out if x <= 0) / m) if out else 1.0,
        windows=len(out),
        mean_turnover=(sum(turnovers) / len(turnovers)) if turnovers else 0.0,
        mean_holdings=(sum(holdings) / len(holdings)) if holdings else 0.0,
        scores=tuple(out),
    )


# --- the null strategy, for reference ---------------------------------------


def null_signal(view: SignalView) -> dict[str, float]:
    """Equal weight over the most liquid names - the bar every signal must clear."""
    return {t: view.panel.dollar_volume.get(t, 0.0) for t in view.names}
