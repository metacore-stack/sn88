#!/usr/bin/env python3
"""Print the session calendar and its execution cutoffs.

Two clocks run against you and they disagree (guide §2.1): the scoring window and the
ramp count TRADING SESSIONS, while the inactivity penalty counts CALENDAR days. This
shows both, plus the MOO/MOC cutoffs resolved in ET - which move on early-close days.

    python3 tools/sessions.py --from 2026-11-20 --to 2026-12-31
    python3 tools/sessions.py --clocks 2026-09-01
"""
import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.calendar import NYSE_TZ, SessionCalendar, early_closes, holidays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default=date.today().isoformat())
    ap.add_argument("--to", dest="end")
    ap.add_argument("--clocks", metavar="DATE", help="show the two clocks from this date")
    ap.add_argument("--year", type=int, help="list holidays and early closes for a year")
    args = ap.parse_args()

    cal = SessionCalendar()

    if args.year:
        print(f"{args.year} NYSE holidays:")
        for d in sorted(holidays(args.year)):
            print(f"  {d.isoformat()}  {d.strftime('%a')}")
        print(f"{args.year} early closes (1:00 pm ET):")
        for d in sorted(early_closes(args.year)):
            print(f"  {d.isoformat()}  {d.strftime('%a')}")
        return 0

    if args.clocks:
        d = date.fromisoformat(args.clocks)
        ramp = cal.sessions_to_calendar_days(d, 30)
        win = cal.sessions_to_calendar_days(d, 50)
        print(f"from {d.isoformat()}:")
        print(f"  new-miner ramp   30 sessions = {ramp} calendar days  (~{ramp/7:.1f} weeks)")
        print(f"  scoring window   50 sessions = {win} calendar days  (~{win/7:.1f} weeks)")
        print(f"  inactivity clock counts CALENDAR days - refresh every day, weekends included")
        return 0

    end = date.fromisoformat(args.end) if args.end else date.fromisoformat(args.start)
    fmt = "%H:%M %Z"
    print(f"{'session':<12} {'':<4} {'MOO cutoff':<14} {'MOC cutoff':<14}")
    for s in cal.sessions(date.fromisoformat(args.start), end):
        tag = "EARLY" if s.early else ""
        moo = s.moo_cutoff.astimezone(NYSE_TZ).strftime(fmt)
        moc = s.moc_cutoff.astimezone(NYSE_TZ).strftime(fmt)
        print(f"{s.day.isoformat():<12} {tag:<4} <{moo:<13} <{moc:<13}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
