#!/usr/bin/env python3
"""Establish the bar: what rank does a null strategy hold, and what must a model beat?

Build-order stage 4. Runs the baselines over SN88's own live-era feed, scores them
through the replica on rolling 50-session windows, and maps the result onto today's live
leaderboard. The output is the number every later modelling decision is measured against.

    python3 tools/baseline_bar.py --daily1 /tmp/d1.json          # from a cached pull
    python3 tools/baseline_bar.py --fetch --since 2025-06-01     # pull fresh (~325 MB)

Weights are fitted on the first half of the history and scored only on the second, so no
baseline sees the data it is evaluated on.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.baselines import BASELINES
from sn88_replica.market import DAILY1_COLS, MarketFeed
from sn88_replica.scoring import raw_score
from sn88_replica.universe import parse_assets

ROOT = Path(__file__).resolve().parent.parent
WIN = 50
SHARPE_EXPONENT = 4.8       # measured; see guide 2.3


def load_daily(args):
    if args.daily1:
        raw = json.loads(Path(args.daily1).read_text())
        return json.loads(raw) if isinstance(raw, str) else raw
    return MarketFeed.fetch_daily(args.since, timeout=900)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily1", help="cached /daily1 json")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--since", default="2025-06-01")
    ap.add_argument("--names", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=60)
    args = ap.parse_args()
    if not args.daily1 and not args.fetch:
        ap.error("pass --daily1 <cached.json> or --fetch")

    rows = load_daily(args)
    I = {c: i for i, c in enumerate(DAILY1_COLS)}
    px = collections.defaultdict(dict)
    for r in rows:
        if r[I["ochl"]] == "day":
            px[r[I["ticker"]]][r[I["date"]]] = r[I["close"]]
    sessions = sorted({d for t in px for d in px[t]})
    print(f"feed: {len(px)} tickers, {len(sessions)} sessions, {sessions[0]} -> {sessions[-1]}")

    assets_path = sorted((ROOT / "fixtures").glob("*-assets.txt"))[-1]
    text = assets_path.read_text(errors="replace")
    u = parse_assets(text)
    cash = set(re.findall(r"cash ETFs: \[([^\]]*)\]", text)[0].replace("'", "").split(", "))
    names = [t for t in u.assets if t not in cash and len(px[t]) == len(sessions)]
    names = sorted(names, key=lambda t: -u.assets[t].dollar_volume)[: args.names]
    print(f"universe: {len(names)} liquid non-cash names with complete history\n")

    rets = {t: [(px[t][sessions[i]] - px[t][sessions[i - 1]]) / px[t][sessions[i - 1]]
                for i in range(1, len(sessions))] for t in names}
    split = len(sessions) // 2
    train = {t: rets[t][:split] for t in names}

    def score_from(weights, end):
        eq, v = [], 10_000_000.0
        for i in range(split, end):
            v *= 1 + sum(weights[t] * rets[t][i] for t in weights)
            eq.append(v)
        return raw_score(10_000_000.0, eq[-WIN:], len(eq)).score

    ends = list(range(split + WIN, len(sessions) - 1))
    if len(ends) > args.max_windows:
        step = len(ends) // args.max_windows
        ends = ends[::step]

    print(f"{'baseline':<26} {'median':>9} {'p25':>9} {'p75':>9} {'zero%':>7}   "
          f"({len(ends)} out-of-sample windows)")
    res = {}
    for name, fn in BASELINES.items():
        w = fn(train)
        s = sorted(score_from(w, e) for e in ends)
        n = len(s)
        res[name] = s[n // 2]
        zero = sum(1 for x in s if x <= 0) / n
        print(f"{name:<26} {s[n//2]:>9.3f} {s[n//4]:>9.3f} {s[3*n//4]:>9.3f} {zero:>6.0%}")

    live = json.loads(sorted((ROOT / "fixtures").glob("*-score.json"))[-1].read_text())
    live = json.loads(live) if isinstance(live, str) else live
    SI = {c: i for i, c in enumerate(
        "uid,hotkey,date,a,days,value,swap,score,apr,lsr,mar,risk,odds,daily,ret".split(","))}
    board = sorted([r[SI["score"]] for r in live
                    if r[SI["a"]] == 1 and r[SI["uid"]] < 256], reverse=True)
    earning = [s for s in board if s > 0]

    best = max(res.items(), key=lambda kv: kv[1])
    print(f"\nlive US-stock class: {len(board)} registered, {len(earning)} earning")
    print(f"\n{'baseline':<26} {'score':>9} {'would rank':>11}")
    for name, sc in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"{name:<26} {sc:>9.3f} {sum(1 for s in board if s > sc) + 1:>7} of {len(board)}")

    null = best[1]
    print(f"\n=== the bar: '{best[0]}' scores {null:.3f} with no modelling at all ===")
    print(f"{'target rank':>12} {'score needed':>13} {'x null':>8} {'x Sharpe needed':>17}")
    for r in (40, 30, 25, 20, 15, 10, 5, 1):
        if r <= len(board):
            need = board[r - 1]
            mult = need / null if null > 0 else float("inf")
            sharpe = mult ** (1.0 / SHARPE_EXPONENT) if mult > 0 else float("inf")
            print(f"{r:>12} {need:>13.3f} {mult:>7.1f}x {sharpe:>16.2f}x")
    print(f"\nscore ~ Sharpe^{SHARPE_EXPONENT}, so a {board[19]/null:.1f}x score gap is only "
          f"a {(board[19]/null) ** (1/SHARPE_EXPONENT):.2f}x Sharpe gap.")
    print("Caveats: one regime (2025-06 onward); the 'complete history' filter is itself a")
    print("mild survivorship screen; live scores move daily. Indicative, not a promise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
