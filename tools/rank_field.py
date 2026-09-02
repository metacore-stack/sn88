#!/usr/bin/env python3
"""Reproduce `Investing/bin/validator`, plus the projected incentive it does not show.

The dashboard shows RAW performance; validators use ADJUSTED. This shows both, and
converts adjusted score into a projected share of the miner pool.

    python3 tools/rank_field.py --class stocks --top 25
    python3 tools/rank_field.py --uid 54            # one miner, all multipliers broken out

Needs no wallet, no stake and no registration.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api
from sn88_replica.pipeline import leaderboard, score_field

POOL_ALPHA_PER_DAY = 2952.0     # structural: 1.0 alpha/block x 7200 x 41%


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", choices=("stocks", "alpha", "all"), default="all")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--uid", type=int, help="show one UID in full")
    ap.add_argument("--alpha-price-tao", type=float, default=0.0039)
    ap.add_argument("--tao-usd", type=float, default=222.0)
    args = ap.parse_args()

    days = api.parse_days(api.fetch("days"))
    ratio = api.check_ratio(api.fetch("ratio"))
    dist = api.fetch("dist")
    api.check_dist(dist)
    field = score_field(api.fetch("pnl"), days, dist, ratio)

    if args.uid is not None:
        s = field.get(args.uid)
        if s is None:
            print(f"uid {args.uid} not in the current field")
            return 1
        t = s.terms
        print(f"uid {s.uid}  class={s.asset}  lifetime={t.lifetime_days} sessions")
        print(f"  return%   {t.gain:9.4f}    risk%  {t.risk:8.4f}")
        print(f"  mar       {t.mar:9.4f}    lsr    {t.lsr:8.4f}")
        print(f"  odds      {t.odds:9.4f}    daily% {t.daily:8.5f}")
        print(f"  raw score {s.raw:9.4f}   (post-clip, post-ramp = what /score publishes)")
        print(f"  x gate        {s.gate}")
        print(f"  x dedupe      {'n/a' if s.dedupe is None else f'{s.dedupe:.4f}'}")
        print(f"  x inactivity  {s.inactivity:.4f}")
        print(f"  x cash        {s.cash:.4f}")
        print(f"  = adjusted {s.adjusted:9.4f}")
        print(f"  incentive  {s.incentive:9.4%}")
        return 0

    wanted = {"stocks": 1, "alpha": 0}.get(args.cls)
    rows = [s for s in field.values() if wanted is None or s.asset == wanted]
    earning = [s for s in rows if s.incentive > 0]
    print(f"class={args.cls}  {len(rows)} UIDs, {len(earning)} earning  "
          f"(ratio={ratio}, pool={POOL_ALPHA_PER_DAY:.0f} alpha/day)")
    print(f"{'#':>3} {'uid':>4} {'a':>2} {'sess':>5} {'ret%':>7} {'risk%':>6} "
          f"{'mar':>6} {'lsr':>6} {'cash':>5} {'dec':>5} {'gate':>4} {'incent':>8} {'$/day':>8}")
    for i, s in enumerate(leaderboard({k: v for k, v in field.items()
                                       if wanted is None or v.asset == wanted}, top=args.top), 1):
        usd = s.incentive * POOL_ALPHA_PER_DAY * args.alpha_price_tao * args.tao_usd
        print(f"{i:>3} {s.uid:>4} {s.asset:>2} {s.terms.lifetime_days:>5} "
              f"{s.terms.gain:>7.2f} {s.terms.risk:>6.2f} {s.terms.mar:>6.2f} "
              f"{s.terms.lsr:>6.3f} {s.cash:>5.3f} {s.inactivity:>5.3f} {s.gate:>4} "
              f"{s.incentive:>7.3%} {usd:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
