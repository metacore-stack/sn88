import sys, random, statistics
sys.path.insert(0, "/home/newuser/Documents/sn88")
sys.path.insert(0, "/home/newuser/Documents/sn88/research/mr")
from final import short_horizon_reversal, CASH_TICKERS
from sn88_replica.signals import load_panel, evaluate
from multiprocessing import Pool

P = load_panel("/home/newuser/Documents/sn88/fixtures/panel-replication-2025-06-02_2026-08-31.json")
UNI = [t for t in P.names if t not in CASH_TICKERS]

def rand_run(seed):
    rng = random.Random(seed)
    def sig(view):
        return {t: rng.random() for t in UNI}
    return evaluate(sig, P, name="rnd", max_windows=40).median

def mom_run(_):
    def sig(view):
        if not view.enough_history(2): return {}
        o = {}
        for t in UNI:
            r = view.returns(t, 1)
            if r and abs(r[0]) <= 0.10:
                o[t] = r[0]
        return o if len(o) >= 40 else {}
    return evaluate(sig, P, name="mom", max_windows=40).median

if __name__ == "__main__":
    with Pool(4) as p:
        meds = sorted(p.map(rand_run, range(300)))
    n = len(meds)
    print("random-20 null over %d draws:" % n)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
        print("   q%.2f %8.2f" % (q, meds[min(n-1, int(q*n))]))
    print("   max %.2f ; fraction >= 135.52 : %.4f" % (meds[-1], sum(1 for x in meds if x >= 135.52)/n))
    print("mirror image (BUY the winners, same clip):", mom_run(0))
