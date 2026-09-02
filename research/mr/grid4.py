import sys, math
sys.path.insert(0, "/home/newuser/Documents/sn88/research/mr")
from lib import *
from sn88_replica.signals import evaluate, null_signal
from multiprocessing import Pool

ETFISH = {"Equity", "Commodity", "Fixed Income", "Currency"}

def make(K=524, clip=None, drop_etf=False, minprice=0.0, vlook=0, capvol=None):
    uni = [t for t in LIQ[:K] if not (drop_etf and SEC.get(t) in ETFISH)]
    def sig(view):
        if not view.enough_history(max(2, vlook + 2)):
            return {}
        out = {}
        for t in uni:
            r = view.returns(t, 1)
            if not r:
                continue
            x = r[0]
            if clip is not None and abs(x) > clip:
                continue
            if minprice:
                c = view.closes(t, 1)
                if c and c[0] < minprice:
                    continue
            if capvol is not None:
                v = fast_vol(view.returns(t, vlook or 20))
                if v > capvol:
                    continue
            out[t] = -x
        return out if len(out) >= 40 else {}
    return sig

def job(cfg):
    cfg = dict(cfg)
    ev = {k: cfg.pop(k) for k in ("top_n", "weighting") if k in cfg}
    name = " ".join(f"{k}={v}" for k, v in {**cfg, **ev}.items())
    f = make(**cfg)
    a = evaluate(f, PANEL, name=name, max_windows=40, **ev)
    b = evaluate(f, PANEL, name=name, max_windows=None, start_frac=0.25, **ev)
    c = evaluate(f, PANEL, name=name, max_windows=None, start_frac=0.5, **ev)
    return (a.median, a.p25, a.zero_fraction, a.mean_turnover,
            b.median, b.p25, b.zero_fraction, c.median, c.p25, c.zero_fraction, name)

if __name__ == "__main__":
    grid = []
    for tn in (10, 15, 20, 30, 40, 60):
        for w in ("equal", "inverse_vol"):
            grid.append(dict(top_n=tn, weighting=w))
    for cl in (0.10, 0.15, 0.25):
        grid.append(dict(clip=cl))
    grid.append(dict(drop_etf=True))
    grid.append(dict(drop_etf=True, top_n=30))
    grid.append(dict(minprice=10.0))
    for cv in (0.02, 0.03, 0.05):
        grid.append(dict(capvol=cv))
    grid.append(dict())
    with Pool(4) as p:
        res = p.map(job, grid)
    # null reference
    for sf, mw, lbl in ((0.5, 40, "NULL std"), (0.25, None, "NULL long"), (0.5, None, "NULL all")):
        r = evaluate(null_signal, PANEL, name=lbl, max_windows=mw, start_frac=sf)
        print(lbl, r)
    res.sort(reverse=True)
    print(f"{'med':>8} {'p25':>7} {'zero':>5} {'turn':>5} | {'L25med':>8} {'p25':>7} {'zero':>5} | {'ALLmed':>8} {'p25':>7} {'zero':>5} | cfg")
    for m, p25, z, to, lm, lp, lz, cm, cp, cz, n in res:
        print(f"{m:8.2f} {p25:7.2f} {z:5.0%} {to:5.0%} | {lm:8.2f} {lp:7.2f} {lz:5.0%} | {cm:8.2f} {cp:7.2f} {cz:5.0%} | {n}")
