"""Book SHAPE with the signal held constant, across three disjoint regions.

The controls showed a random 100-name inverse-vol book beating a searched reversal.
That says the variable that matters is not which names you pick but how many and how
you weight them. This sweeps exactly that, using one fixed trivial signal (liquidity
rank) so nothing is being searched.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/newuser/Documents/sn88")
from sn88_replica.signals import Panel, evaluate, load_panel

FULL = load_panel("fixtures/panel-replication-2025-06-02_2026-08-31.json")

def truncate(p, upto):
    close = {t: list(s[:upto]) for t, s in p.close.items()}
    rets = {t: [(s[i]-s[i-1])/s[i-1] for i in range(1, len(s))] for t, s in close.items()}
    return Panel(sessions=p.sessions[:upto], names=p.names, close=close, returns=rets,
                 sector=p.sector, dollar_volume=p.dollar_volume, market_cap=p.market_cap)

def liquidity(v):
    return {t: FULL.dollar_volume.get(t, 0.0) for t in v.names}

REGIONS = [("A  sessions 100-207", truncate(FULL, 207), 100/207),
           ("B  sessions 207-314", FULL,                207/314),
           ("C  sessions 157-314", FULL,                0.5)]

SIZES = (10, 20, 50, 100, 200, 300, 524)
print("signal held FIXED at liquidity rank; only the book shape varies")
print(f"{'':<16}" + "".join(f"{n:>11}" for n in SIZES))
for label, panel, sf in REGIONS:
    print(f"\n{label}")
    for w in ("equal", "inverse_vol"):
        cells = []
        for n in SIZES:
            r = evaluate(liquidity, panel, start_frac=sf, top_n=n, weighting=w,
                         max_windows=40)
            cells.append(f"{r.median:>11.3f}")
        print(f"  {w:<14}" + "".join(cells))
