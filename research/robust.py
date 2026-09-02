import sys; sys.path.insert(0,'/home/newuser/Documents/sn88')
from research.lab import *

def card(sig, label):
    # full-history walk from session 70 for sub-period stability
    s,c,tn,sess = walk(sig, start_frac=0.0, warmup=70)
    n=len(s); k=n//3
    segs=[('P1',s[:k]),('P2',s[k:2*k]),('P3',s[2*k:])]
    st=stats(s)
    sub=' '.join(f'{nm} {stats(x)["sharpe_a"]:5.2f}' for nm,x in segs)
    # score medians at three evaluation start points
    meds=[]
    for sf in (0.35,0.5,0.65):
        s2,c2,_,_ = walk(sig, start_frac=sf)
        sc=sorted(scores_of(c2)); meds.append(sc[len(sc)//2] if sc else 0.0)
    r = run(sig, label)
    print(f'{label:34s} med {r.median:8.3f} p25 {r.p25:7.3f} p75 {r.p75:8.2f} zero {r.zero_fraction:5.1%} '
          f'turn {r.mean_turnover:5.1%} | ShA {st["sharpe_a"]:5.2f} skew {st["skew"]:5.2f} vol {st["vol"]*100:4.2f}% '
          f'hit {st["hit"]:.0%} | {sub} | medshift {meds[0]:7.2f}/{meds[1]:7.2f}/{meds[2]:7.2f}')
    return r
