import sys, math, json
sys.path.insert(0, "/home/newuser/Documents/sn88/research/mr")
from lib import *
from sn88_replica.signals import evaluate
from multiprocessing import Pool

def ranks(d):
    order = sorted(d, key=lambda t: (d[t], t))
    n = len(order) - 1 or 1
    return {t: i / n for i, t in enumerate(order)}

def make(H=1, K=524, gap=0, vol_adj=False, neutral="none", volscreen=1.0,
         volw=0.0, vlook=20):
    uni = LIQ[:K]
    need = max(H + gap, vlook, 60 if neutral == "beta" else 0) + 2
    def sig(view):
        if not view.enough_history(need):
            return {}
        rev, vol = {}, {}
        for t in uni:
            r = view.returns(t, H + gap)
            if len(r) < H + gap:
                continue
            seg = r[:H] if gap else r
            c = 1.0
            for x in seg:
                c *= 1 + x
            c -= 1.0
            v = fast_vol(view.returns(t, vlook))
            if v <= 1e-6:
                continue
            vol[t] = v
            rev[t] = c / v if vol_adj else c
        if len(rev) < 40:
            return {}
        if neutral == "sector":
            b = {}
            for t in rev:
                b.setdefault(SEC.get(t, "?"), []).append(t)
            out = {}
            for s, ts in b.items():
                m = sum(rev[t] for t in ts) / len(ts)
                for t in ts:
                    out[t] = rev[t] - m
            rev = out
        elif neutral == "beta":
            m = sum(rev.values()) / len(rev)
            out = {}
            for t in rev:
                bt = max(0.2, min(2.5, view.beta(t, "SPY", 60)))
                out[t] = rev[t] - bt * m
            rev = out
        if volscreen < 1.0:
            cut = sorted(vol.values())[max(0, int(len(vol) * volscreen) - 1)]
            rev = {t: v for t, v in rev.items() if vol[t] <= cut}
            if len(rev) < 25:
                return {}
        if volw:
            rr = ranks({t: -rev[t] for t in rev})
            vr = ranks({t: -vol[t] for t in rev})
            return {t: (1 - volw) * rr[t] + volw * vr[t] for t in rev}
        return {t: -v for t, v in rev.items()}
    return sig

def job(cfg):
    cfg = dict(cfg); ev = {k: cfg.pop(k) for k in ("top_n", "weighting") if k in cfg}
    name = " ".join(f"{k}={v}" for k, v in {**cfg, **ev}.items())
    f = make(**cfg)
    a = evaluate(f, PANEL, name=name, max_windows=40, **ev)                    # standard
    b = evaluate(f, PANEL, name=name, max_windows=None, start_frac=0.25, **ev) # long/extended
    return (a.median, a.p25, a.zero_fraction, a.mean_turnover,
            b.median, b.p25, b.zero_fraction, b.windows, name)

def report(res, path):
    res.sort(reverse=True)
    with open(path, "w") as f:
        f.write(f"{'med':>9} {'p25':>8} {'zero':>5} {'turn':>5} | {'Lmed':>9} {'Lp25':>8} {'Lzero':>5} | cfg\n")
        for m, p, z, to, lm, lp, lz, lw, n in res:
            f.write(f"{m:9.2f} {p:8.2f} {z:5.0%} {to:5.0%} | {lm:9.2f} {lp:8.2f} {lz:5.0%} | {n}\n")

if __name__ == "__main__":
    grid = []
    for H in (1, 2, 3, 5, 10):
        for gap in (0, 1):
            for K in (100, 200, 350, 524):
                for va in (False, True):
                    for nt in ("none", "sector"):
                        grid.append(dict(H=H, gap=gap, K=K, vol_adj=va, neutral=nt))
    with Pool(4) as p:
        res = p.map(job, grid)
    report(res, "/home/newuser/Documents/sn88/research/mr/grid3.out")
    print(len(res), "done")
