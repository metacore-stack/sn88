#!/usr/bin/env python3
"""Run the daily health checks against live data and yesterday's archive.

Does what §23's runbook asks, in one command: diff the rules, reconstruct the field,
check the replica still agrees with /score, and emit alerts with the action each demands.

    python3 tools/watch.py                 # live vs the newest archived snapshot
    python3 tools/watch.py --uid 54        # add a specific miner's health
    python3 tools/watch.py --offline       # replay the two newest archives

Exit codes:  0 ok  ·  1 alert  ·  2 freeze selection  ·  3 safe mode
"""
import argparse
import glob
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api
from sn88_replica.monitor import (
    Action, AlertRouter, CycleMetrics, check_cycle, detect_rule_change,
)
from sn88_replica.pipeline import group_pnl, leaderboard, score_field
from sn88_replica.scoring import raw_score
from sn88_replica.universe import parse_assets

ROOT = Path(__file__).resolve().parent.parent
EXIT = {Action.NONE: 0, Action.ALERT: 1, Action.FREEZE: 2, Action.SAFE_MODE: 3}
_PI = {c: i for i, c in enumerate(api.PNL_COLS)}
_SI = {c: i for i, c in enumerate(api.SCORE_COLS)}


def archives(suffix):
    return sorted(glob.glob(str(ROOT / "fixtures" / f"*-{suffix}")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uid", type=int, help="also report this miner's health")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    assets = archives("assets.txt")
    if len(assets) < 2 and args.offline:
        print("need two archived /assets snapshots for an offline run")
        return 1

    before = parse_assets(Path(assets[-2 if args.offline else -1]).read_text(errors="replace"))
    if args.offline:
        after = parse_assets(Path(assets[-1]).read_text(errors="replace"))
        pnl = api.load_fixture(archives("pnl.json")[-1])
        days_raw = api.load_fixture(archives("days.json")[-1])
        dist = api.load_fixture(archives("dist.json")[-1])
        ratio = api.load_fixture(archives("ratio.json")[-1])
        score_rows = api.load_fixture(archives("score.json")[-1])
        source = f"archives {Path(assets[-2]).name[:10]} -> {Path(assets[-1]).name[:10]}"
    else:
        after = parse_assets(api.fetch("assets"))
        pnl, dist = api.fetch("pnl"), api.fetch("dist")
        ratio, days_raw = api.fetch("ratio"), api.fetch("days")
        score_rows = api.fetch("score")
        source = f"live vs {Path(assets[-1]).name[:10]}"

    router = AlertRouter(ROOT / "ops" / "alerts.jsonl")

    print(f"== rule change ({source})")
    diff = detect_rule_change(before_universe=before, after_universe=after)
    if not diff.changed:
        print("   no change: constants, ratio, cash list, universe and dereg list all stable")
    for a in diff.alerts():
        print(f"   {a}")
    router.extend(diff.alerts())

    print(f"\n== integrity")
    try:
        api.check_ratio(ratio); api.check_dist(dist)
        days = api.parse_days(days_raw)
        print(f"   /ratio {list(ratio)}  /dist tail ok  /days {len(days)} rows  "
              f"/assets {len(after)} tickers, {len(after.cash_tickers)} cash-classified")
    except api.IntegrityError as exc:
        print(f"   QUARANTINE: {exc}")
        return 3

    print(f"\n== replica vs /score")
    by_uid = group_pnl(pnl)
    worst, worst_uid, n = 0.0, None, 0
    for row in score_rows:
        uid = row[_SI["uid"]]
        rows = by_uid.get(uid)
        if not rows:
            continue
        win = rows[-50:]
        got = raw_score(win[0][_PI["swap_open"]],
                        [r[_PI["swap_close"]] for r in win], len(rows)).score
        exp = row[_SI["score"]]
        rel = abs(got - exp) / max(abs(exp), 1e-12)
        n += 1
        if rel > worst:
            worst, worst_uid = rel, uid
    print(f"   {n} rows, max relative error {worst:.3e} (uid {worst_uid})")
    router.extend(check_cycle(CycleMetrics(session=date.today(), window="MOC",
                                           replica_drift=worst)))

    print(f"\n== field")
    field = score_field(pnl, days, dist, ratio)
    earning = [s for s in field.values() if s.incentive > 0]
    board = leaderboard(field, top=5)
    print(f"   {len(field)} UIDs, {len(earning)} earning; top 5 take "
          f"{sum(s.incentive for s in board):.1%}")
    for i, s in enumerate(board, 1):
        print(f"     {i}. uid {s.uid:>3} class={s.asset} {s.incentive:7.3%}")

    if args.uid is not None:
        s = field.get(args.uid)
        if s is None:
            print(f"\n   uid {args.uid} not in the field")
        else:
            row = days.get(args.uid)
            m = CycleMetrics(session=date.today(), window="MOC",
                             cash=(row.cash if row else None),
                             last_active_days=(row.last_active if row else None))
            alerts = check_cycle(m)
            print(f"\n== uid {args.uid}")
            print(f"   incentive {s.incentive:.3%}  gate={s.gate}  "
                  f"cash weight={(row.cash if row else 0):.4f} "
                  f"(x{s.cash:.4f})  dec x{s.inactivity:.4f}  "
                  f"idle={row.last_active if row else '?'}d")
            for a in alerts:
                print(f"   {a}")
            router.extend(alerts)

    action = router.action()
    print(f"\n== verdict: {action.value}"
          + ("" if router.should_trade() else "  -- DO NOT PUBLISH"))
    return EXIT[action]


if __name__ == "__main__":
    raise SystemExit(main())
