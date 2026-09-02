"""Reversal: search on one block of sessions, confirm ONCE on a disjoint block.

The two blocks share no session and no evaluation window. The config is chosen by
median score on SEARCH only; CONFIRM is run once, after the choice is frozen.
"""
from __future__ import annotations
import sys, statistics
sys.path.insert(0, "/home/newuser/Documents/sn88")
from sn88_replica.signals import Panel, evaluate, load_panel, null_signal

FULL = load_panel("fixtures/panel-replication-2025-06-02_2026-08-31.json")
SPLIT = 207                                   # sessions [100,207) search, [207,314) confirm

def truncate(p: Panel, upto: int) -> Panel:
    close = {t: list(s[:upto]) for t, s in p.close.items()}
    rets = {t: [(s[i]-s[i-1])/s[i-1] for i in range(1, len(s))] for t, s in close.items()}
    return Panel(sessions=p.sessions[:upto], names=p.names, close=close, returns=rets,
                 sector=p.sector, dollar_volume=p.dollar_volume, market_cap=p.market_cap)

SEARCH = truncate(FULL, SPLIT)

def reversal(lookback: int, sector_neutral: bool):
    def f(view):
        raw = {t: -view.cumulative(t, lookback) for t in view.names
               if len(view.returns(t, lookback)) >= lookback}
        if not raw or not sector_neutral:
            return raw
        buckets: dict[str, list[str]] = {}
        for t in raw:
            buckets.setdefault(FULL.sector.get(t, "?"), []).append(t)
        out = {}
        for _, ts in buckets.items():
            mu = sum(raw[t] for t in ts) / len(ts)
            for t in ts:
                out[t] = raw[t] - mu
        return out
    return f

GRID = [(lb, sn, n, w)
        for lb in (1, 2, 3, 5, 10) for sn in (True, False)
        for n in (20, 50, 100, 200) for w in ("equal", "inverse_vol")]

rows = []
for lb, sn, n, w in GRID:
    r = evaluate(reversal(lb, sn), SEARCH, name=f"rev{lb}", start_frac=100/SPLIT,
                 top_n=n, weighting=w, max_windows=40)
    rows.append((r.median, lb, sn, n, w, r))
rows.sort(reverse=True, key=lambda x: x[0])

null_search = evaluate(null_signal, SEARCH, start_frac=100/SPLIT, max_windows=40)
print(f"SEARCH block: sessions 100..{SPLIT}  ({rows[0][5].windows} windows, {len(GRID)} configs)")
print(f"  null bar                                   median {null_search.median:>8.3f}")
print("  top 5 by search median:")
for md, lb, sn, n, w, r in rows[:5]:
    print(f"    rev{lb:<3} sector={str(sn):<5} n={n:<4} {w:<12} median {md:>8.3f}  zero {r.zero_fraction:>4.0%}  turn {r.mean_turnover:>5.0%}")
print(f"  median across ALL {len(GRID)} configs: {statistics.median(r[0] for r in rows):.3f}")
print(f"  how many of {len(GRID)} beat the null? {sum(1 for r in rows if r[0] > null_search.median)}")

best_md, lb, sn, n, w, _ = rows[0]
print(f"\nFROZEN CHOICE: reversal lookback={lb} sector_neutral={sn} top_n={n} weighting={w}")

print(f"\nCONFIRM block: sessions {SPLIT}..314  (disjoint - never seen during search)")
nc = evaluate(null_signal, FULL, start_frac=SPLIT/314, max_windows=40)
bc = evaluate(reversal(lb, sn), FULL, start_frac=SPLIT/314, top_n=n, weighting=w, max_windows=40)
print(f"  null bar          median {nc.median:>8.3f}  p25 {nc.p25:>7.3f}  zero {nc.zero_fraction:>4.0%}  turn {nc.mean_turnover:>5.1%}  ({nc.windows} windows)")
print(f"  frozen reversal   median {bc.median:>8.3f}  p25 {bc.p25:>7.3f}  zero {bc.zero_fraction:>4.0%}  turn {bc.mean_turnover:>5.1%}  ({bc.windows} windows)")
print(f"\n  search median {best_md:.3f} -> confirm median {bc.median:.3f}   "
      f"shrinkage {1 - bc.median/best_md:+.0%}" if best_md else "")

# what the turnover actually costs, at $0.002/share
avg_px = statistics.mean(s[-1] for s in FULL.close.values())
notional = 10_000_000 * bc.mean_turnover
print(f"\n  fee check: {bc.mean_turnover:.0%} one-way turnover on $10M at ${avg_px:.0f}/share")
print(f"    = ${notional:,.0f} traded = {notional/avg_px:,.0f} shares x $0.002 = "
      f"${notional/avg_px*0.002:,.0f}/session = {notional/avg_px*0.002/10_000_000:.4%} of book")
