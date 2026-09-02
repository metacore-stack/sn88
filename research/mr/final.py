import sys, math, statistics
sys.path.insert(0, "/home/newuser/Documents/sn88")
from sn88_replica.signals import load_panel, evaluate, SignalView, null_signal, equal_weight_top

# ---------------- THE SIGNAL ----------------
CASH_TICKERS = frozenset({
    "SGOV", "BIL", "JPST", "BSV", "VCSH", "VGIT", "GOVT", "BND",
    "IUSB", "AGG", "IEF", "BNDX", "MBB", "VCIT", "MUB", "VTEB",
})
EVENT_MOVE = 0.10          # drop names whose last observable session moved > 10%


def short_horizon_reversal(view: SignalView) -> dict[str, float]:
    """Buy yesterday's cross-sectional losers, excluding event/split-sized moves.

    Score = minus the most recent COMPLETE one-session return the view exposes
    (the move into close[t-1]).  Cash-classified tickers are excluded so the book
    never incurs the cash multiplier; names whose last move exceeded +/-10% are
    excluded because those are gaps/splits/earnings, not reversion candidates.
    Everything else in the panel is eligible - no liquidity, sector or volatility
    screen, no vol scaling, no neutralisation.  The harness equal-weights the top 20.
    """
    if not view.enough_history(2):
        return {}
    scores: dict[str, float] = {}
    for ticker in view.names:
        if ticker in CASH_TICKERS:
            continue
        last = view.returns(ticker, 1)
        if not last:
            continue
        move = last[0]
        if abs(move) > EVENT_MOVE:
            continue
        scores[ticker] = -move
    return scores if len(scores) >= 40 else {}
# --------------------------------------------

if __name__ == "__main__":
    P = load_panel("/home/newuser/Documents/sn88/fixtures/panel-replication-2025-06-02_2026-08-31.json")
    print("--- headline (standard protocol: start_frac=0.5, 53 windows, top_n=20, equal) ---")
    r = evaluate(short_horizon_reversal, P, name="reversal-1d", max_windows=40)
    n = evaluate(null_signal, P, name="NULL", max_windows=40)
    print(r, "holdings", r.mean_holdings, "windows", r.windows)
    print(n, "holdings", n.mean_holdings, "windows", n.windows)

    print("\n--- robustness ---")
    for lbl, kw in [
        ("all 106 windows",      dict(max_windows=None)),
        ("start 0.25 (186 win)", dict(max_windows=None, start_frac=0.25)),
        ("start 0.75 (28 win)",  dict(max_windows=None, start_frac=0.75)),
        ("fill_lag=2 (extra day slip)", dict(max_windows=40, fill_lag=2)),
        ("fill_lag=3",           dict(max_windows=40, fill_lag=3)),
        ("top_n=10",             dict(max_windows=40, top_n=10)),
        ("top_n=30",             dict(max_windows=40, top_n=30)),
        ("top_n=40",             dict(max_windows=40, top_n=40)),
        ("inverse_vol",          dict(max_windows=40, weighting="inverse_vol")),
        ("rebalance_every=2",    dict(max_windows=40, rebalance_every=2)),
        ("no_trade_band=0.5",    dict(max_windows=40, no_trade_band=0.5)),
    ]:
        rr = evaluate(short_horizon_reversal, P, name=lbl, **kw)
        nn = evaluate(null_signal, P, name="", **kw)
        print(f"{rr}   [null median {nn.median:.2f}]")

    print("\n--- realised return profile (standard eval span) ---")
    ns = len(P.sessions); begin = max(61, int(ns * 0.5)); rets = []
    for t in range(begin, ns - 1):
        w = equal_weight_top(short_horizon_reversal(SignalView(P, t)), 20)
        rets.append(sum(x * P.returns[k][t] for k, x in w.items()))
    m = sum(rets) / len(rets); sd = statistics.pstdev(rets)
    sk = sum(((x - m) / sd) ** 3 for x in rets) / len(rets)
    print(f"n={len(rets)} mean={m*1e4:.1f}bp sd={sd*100:.2f}%/day dailySharpe={m/sd:.3f} "
          f"ann={m/sd*math.sqrt(252):.2f} hit={sum(1 for x in rets if x>0)/len(rets):.3f} "
          f"skew={sk:+.2f} total={math.prod(1+x for x in rets):.3f}")
    h = len(rets) // 2
    for lab, seg in (("1st half", rets[:h]), ("2nd half", rets[h:])):
        mm = sum(seg)/len(seg); ss = statistics.pstdev(seg)
        print(f"  {lab}: sharpe {mm/ss:.3f} total {math.prod(1+x for x in seg):.3f}")
    # fee drag estimate at $0.002/share, $10m book
    print(f"  turnover {r.mean_turnover:.1%}/session")
