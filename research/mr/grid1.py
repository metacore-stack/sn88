import sys
sys.path.insert(0, "/home/newuser/Documents/sn88/research/mr")
from run import *

DV = PANEL.dollar_volume
NONCASH = [t for t in PANEL.names if t not in CASH]
LIQ = sorted(NONCASH, key=lambda t: -DV.get(t, 0.0))

def make(H, K, vol_adj=False, neutral="none", wlook=20):
    uni = LIQ[:K]
    def sig(view):
        if not view.enough_history(max(H, 25) + 1):
            return {}
        raw = {}
        for t in uni:
            c = view.cumulative(t, H)
            if vol_adj:
                v = view.volatility(t, wlook)
                c = c / v if v > 1e-6 else 0.0
            raw[t] = c
        if neutral == "mkt":
            m = sum(raw.values()) / len(raw)
            raw = {t: v - m for t, v in raw.items()}
        elif neutral == "sector":
            buckets = {}
            for t in raw:
                buckets.setdefault(PANEL.sector.get(t, "?"), []).append(t)
            out = {}
            for s, ts in buckets.items():
                m = sum(raw[t] for t in ts) / len(ts)
                for t in ts:
                    out[t] = raw[t] - m
            raw = out
        elif neutral == "beta":
            m = sum(raw.values()) / len(raw)
            out = {}
            for t in raw:
                b = view.beta(t, "SPY", 60)
                b = max(0.2, min(2.5, b))
                out[t] = raw[t] - b * m
            raw = out
        return {t: -v for t, v in raw.items()}
    return sig

if __name__ == "__main__":
    rows = []
    for H in (1, 2, 3, 5, 8, 10):
        for K in (100, 200, 350, 524):
            for va in (False, True):
                for nt in ("none", "mkt", "sector"):
                    n = f"H{H} K{K} {'vadj' if va else 'raw '} {nt}"
                    r = run(make(H, K, va, nt), n)
                    rows.append(r)
                    print(r, flush=True)
    rows.sort(key=lambda r: -r.median)
    print("\n=== TOP ===")
    for r in rows[:15]:
        print(r)
