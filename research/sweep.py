import sys, json, itertools, os
sys.path.insert(0,'/home/newuser/Documents/sn88')
from multiprocessing import Pool
from research.lab import build, build_combo, run, PANEL

def job(cfg):
    ev = dict(cfg.pop("_eval", {}))
    name = json.dumps(cfg, sort_keys=True) + "|" + json.dumps(ev, sort_keys=True)
    try:
        kind = cfg.pop("_kind", "build")
        if kind == "combo":
            cfg2 = dict(cfg); cfg2["horizons"] = tuple(cfg2["horizons"])
            sg = build_combo(**cfg2)
        else:
            sg = build(**cfg)
        r = run(sg, name, **ev)
        return (name, r.median, r.p25, r.p75, r.zero_fraction, r.mean_turnover, r.mean_holdings)
    except Exception as e:
        return (name, -1, -1, -1, 1.0, 0.0, 0.0)

def main(path):
    cfgs = json.load(open(path))
    with Pool(4) as p:
        res = p.map(job, cfgs)
    res.sort(key=lambda x: -x[1])
    for r in res:
        print(f"{r[1]:8.3f} p25 {r[2]:7.3f} p75 {r[3]:9.3f} zero {r[4]:5.1%} to {r[5]:5.1%} n {r[6]:4.1f}  {r[0]}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
