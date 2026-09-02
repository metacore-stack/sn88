"""Can this panel distinguish ANY effect from noise? Null spread vs candidate spread.

If a book with no information at all produces a wider spread of scores than the whole
family of candidates does, then no candidate in that family is identifiable, and every
ranking within it is a ranking of luck.
"""
from __future__ import annotations
import statistics, sys
sys.path.insert(0, "/home/newuser/Documents/sn88")
from sn88_replica.signals import (evaluate, load_panel, random_signal,
                                  shape_matched_null, exceeds_null)

FULL = load_panel("fixtures/panel-replication-2025-06-02_2026-08-31.json")
SEEDS = 12

def liquidity(v): return {t: FULL.dollar_volume.get(t, 0.0) for t in v.names}
def lowvol(v):    return {t: -v.volatility(t, 20) for t in v.names}
def rev(lb):
    def f(v):
        raw = {t: -v.cumulative(t, lb) for t in v.names if len(v.returns(t, lb)) >= lb}
        b = {}
        for t in raw: b.setdefault(FULL.sector.get(t, "?"), []).append(t)
        out = {}
        for _, ts in b.items():
            mu = sum(raw[t] for t in ts)/len(ts)
            for t in ts: out[t] = raw[t] - mu
        return out
    return f

for label, sf in [("B  sessions 207-314", 207/314), ("C  sessions 157-314", 0.5)]:
    kw = dict(start_frac=sf, top_n=100, weighting="inverse_vol", max_windows=30)
    nulls = shape_matched_null(FULL, seeds=SEEDS, **kw)
    cands = {"liquidity": liquidity, "low-vol": lowvol,
             "reversal 1d": rev(1), "reversal 10d": rev(10)}
    cs = {k: evaluate(f, FULL, **kw).median for k, f in cands.items()}
    print(f"\n{label}   book: 100 names, inverse-vol")
    print(f"  {SEEDS} NO-SIGNAL draws: min {nulls[0]:.2f}  median {statistics.median(nulls):.2f}"
          f"  max {nulls[-1]:.2f}   spread {nulls[-1]/max(nulls[0],1e-9):.1f}x")
    print(f"  candidates:")
    for k, v in sorted(cs.items(), key=lambda x: -x[1]):
        pct = 100.0 * sum(1 for n in nulls if n < v) / len(nulls)
        flag = "CLEARS" if exceeds_null(v, nulls) else "inside null"
        print(f"    {k:<14}{v:>8.3f}   {pct:>3.0f}th pct of null   {flag}")
    cspread = max(cs.values()) / max(min(cs.values()), 1e-9)
    print(f"  candidate spread {cspread:.1f}x  vs  null spread {nulls[-1]/max(nulls[0],1e-9):.1f}x"
          f"   -> {'NOT identifiable' if nulls[-1]/max(nulls[0],1e-9) >= cspread else 'identifiable'}")
