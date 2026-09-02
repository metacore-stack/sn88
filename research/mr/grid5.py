import sys, math
sys.path.insert(0, "/home/newuser/Documents/sn88/research/mr")
from lib import *
from sn88_replica.signals import evaluate
from multiprocessing import Pool

def make(clip=None, dnclip=None, K=524, top_frac=None):
    uni = list(LIQ[:K])
    def sig(view):
        if not view.enough_history(2):
            return {}
        out = {}
        for t in uni:
            r = view.returns(t, 1)
            if not r:
                continue
            x = r[0]
            if clip is not None and abs(x) > clip:
                continue
            if dnclip is not None and x < -dnclip:
                continue
            out[t] = -x
        return out if len(out) >= 40 else {}
    return sig

def job(cfg):
    cfg = dict(cfg); ev = {k: cfg.pop(k) for k in ("top_n",) if k in cfg}
    name = " ".join(f"{k}={v}" for k, v in {**cfg, **ev}.items()) or "plain"
    f = make(**cfg)
    a = evaluate(f, PANEL, name=name, max_windows=40, **ev)
    b = evaluate(f, PANEL, name=name, max_windows=None, start_frac=0.25, **ev)
    c = evaluate(f, PANEL, name=name, max_windows=None, start_frac=0.75, **ev)
    return (a.median, a.p25, a.p75, a.zero_fraction, a.mean_turnover,
            b.median, b.p25, b.zero_fraction, c.median, c.p25, c.zero_fraction, name)

if __name__ == "__main__":
    grid = [dict()]
    for cl in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.17, 0.20, 0.30, 0.50):
        grid.append(dict(clip=cl))
    for dc in (0.05, 0.08, 0.10, 0.15):
        grid.append(dict(dnclip=dc))
    for tn in (15, 25, 30):
        grid.append(dict(clip=0.10, top_n=tn))
    for K in (250, 350, 450):
        grid.append(dict(clip=0.10, K=K))
    with Pool(4) as p:
        res = p.map(job, grid)
    res.sort(reverse=True)
    print(f"{'med':>8} {'p25':>7} {'p75':>8} {'zero':>5} {'turn':>5} | {'L25med':>8} {'p25':>7} {'zero':>5} | {'H75med':>8} {'p25':>7} {'zero':>5} | cfg")
    for m,p25,p75,z,to,lm,lp,lz,cm,cp,cz,n in res:
        print(f"{m:8.2f} {p25:7.2f} {p75:8.2f} {z:5.0%} {to:5.0%} | {lm:8.2f} {lp:7.2f} {lz:5.0%} | {cm:8.2f} {cp:7.2f} {cz:5.0%} | {n}")
