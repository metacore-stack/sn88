"""Is the reversal edge the SIGNAL, or just the book shape? Size/weighting-matched nulls.

The frozen config holds 100 names inverse-vol weighted; the null bar holds 20 equal.
Those differ in diversification, weighting and turnover as well as in signal, so the
raw comparison cannot attribute anything. Each control below changes ONLY the signal
and holds the book shape fixed.
"""
from __future__ import annotations
import random, statistics, sys
sys.path.insert(0, "/home/newuser/Documents/sn88")
from sn88_replica.signals import evaluate, load_panel

FULL = load_panel("fixtures/panel-replication-2025-06-02_2026-08-31.json")
SPLIT, N, LB = 207, 100, 10
CFG = dict(start_frac=SPLIT/314, top_n=N, weighting="inverse_vol", max_windows=40)

def neutralise(raw):
    b = {}
    for t in raw:
        b.setdefault(FULL.sector.get(t, "?"), []).append(t)
    out = {}
    for _, ts in b.items():
        mu = sum(raw[t] for t in ts) / len(ts)
        for t in ts:
            out[t] = raw[t] - mu
    return out

def directional(sign):
    def f(v):
        raw = {t: sign * v.cumulative(t, LB) for t in v.names
               if len(v.returns(t, LB)) >= LB}
        return neutralise(raw) if raw else raw
    return f

def shuffled(seed):
    """Same score DISTRIBUTION, assigned to the wrong names. Kills the signal only."""
    def f(v):
        raw = {t: -v.cumulative(t, LB) for t in v.names
               if len(v.returns(t, LB)) >= LB}
        if not raw: return raw
        vals = list(neutralise(raw).values())
        rng = random.Random(seed * 100003 + v.t)
        rng.shuffle(vals)
        return dict(zip(sorted(raw), vals))
    return f

def random_pick(seed):
    def f(v):
        rng = random.Random(seed * 7919 + v.t)
        return {t: rng.random() for t in v.names}
    return f

def liquidity(v):
    return {t: FULL.dollar_volume.get(t, 0.0) for t in v.names}

def lowvol(v):
    return {t: -v.volatility(t, 20) for t in v.names}

print(f"CONFIRM block, book shape held fixed: top {N} names, inverse-vol, sector-neutral")
print(f"{'signal':<38}{'median':>9}{'p25':>8}{'p75':>9}{'zero':>7}{'turn':>7}")
print("-" * 78)
def show(label, sig):
    r = evaluate(sig, FULL, **CFG)
    print(f"{label:<38}{r.median:>9.3f}{r.p25:>8.3f}{r.p75:>9.3f}{r.zero_fraction:>6.0%}{r.mean_turnover:>7.0%}")
    return r.median

rev = show("reversal 10d  (the frozen choice)", directional(-1))
mom = show("momentum 10d  (sign-flipped)", directional(+1))
print()
sh = [show(f"shuffled reversal  seed={s}", shuffled(s)) for s in (1, 2, 3)]
print()
rp = [show(f"random 100 names   seed={s}", random_pick(s)) for s in (1, 2, 3)]
print()
show("liquidity-ranked 100", liquidity)
show("low-volatility 100", lowvol)

print(f"\n  reversal {rev:.3f}  vs shuffled median {statistics.median(sh):.3f}"
      f"  vs random median {statistics.median(rp):.3f}")
print(f"  momentum (guide 2.7 predicts this loses): {mom:.3f}")
