#!/usr/bin/env python3
"""Snapshot every owner endpoint. Run this daily, from day one.

These become the golden fixtures (guide §20), the rule-change detector's baseline
(§17), and the only record you will have of what the owner API used to say. You
cannot backfill them.

    python3 tools/archive_fixtures.py [--dir fixtures]

Suggested cron (daily, after the 06:00 UTC stock scoring):
    30 6 * * *  cd /path/to/sn88 && python3 tools/archive_fixtures.py >> ops/archive.log 2>&1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api
from sn88_replica.constants import fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="fixtures", help="output directory")
    args = ap.parse_args()

    written = api.archive(args.dir)
    for endpoint, path in written.items():
        print(f"  {endpoint:<7} {path.stat().st_size:>9,} bytes  {path}")
    print(f"  constants fingerprint: {fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
