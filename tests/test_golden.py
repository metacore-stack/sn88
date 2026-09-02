"""Acceptance test: the replica must reproduce the live /score table to <1e-6.

/score publishes the raw score - post-clip and post-ramp, but BEFORE the etc.py
multipliers (gate, dedupe, inactivity, cash, class ratio). So this validates
scoring.raw_score() exactly, which is the piece with all the subtle ordering.

Runs against archived fixtures if present (offline, deterministic), otherwise fetches
live. Archive daily with tools/archive_fixtures.py - you cannot backfill these.
"""

from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api                      # noqa: E402
from sn88_replica.pipeline import group_pnl       # noqa: E402
from sn88_replica.scoring import raw_score        # noqa: E402

TOLERANCE = 1e-6
FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
_PI = {c: i for i, c in enumerate(api.PNL_COLS)}
_SI = {c: i for i, c in enumerate(api.SCORE_COLS)}


def _latest(endpoint: str):
    hits = sorted(glob.glob(os.path.join(FIXTURES, f"*-{endpoint}.json")))
    return hits[-1] if hits else None


def load(endpoint: str):
    path = _latest(endpoint)
    if path:
        return api.load_fixture(path), os.path.basename(path)
    if os.environ.get("SN88_OFFLINE"):
        raise unittest.SkipTest(f"no fixture for /{endpoint} and SN88_OFFLINE is set")
    return api.fetch(endpoint), "live"


class TestGolden(unittest.TestCase):
    """The replica against upstream's own published numbers."""

    @classmethod
    def setUpClass(cls):
        cls.score_rows, cls.score_src = load("score")
        cls.pnl_rows, cls.pnl_src = load("pnl")
        cls.by_uid = group_pnl(cls.pnl_rows)

    def test_raw_score_matches_upstream(self):
        errors, worst, worst_uid, compared = [], 0.0, None, 0
        for row in self.score_rows:
            uid = row[_SI["uid"]]
            rows = self.by_uid.get(uid)
            if not rows:
                continue
            win = rows[-50:]
            got = raw_score(
                win[0][_PI["swap_open"]],
                [r[_PI["swap_close"]] for r in win],
                len(rows),
            ).score
            expected = row[_SI["score"]]
            rel = abs(got - expected) / max(abs(expected), 1e-12)
            errors.append(rel)
            compared += 1
            if rel > worst:
                worst, worst_uid = rel, uid

        self.assertGreater(compared, 100, "too few rows compared to be meaningful")
        errors.sort()
        median = errors[len(errors) // 2]
        print(
            f"\n  /score golden: {compared} rows  max={worst:.3e} (uid {worst_uid})  "
            f"median={median:.3e}  [{self.score_src}]"
        )
        self.assertLess(worst, TOLERANCE, f"uid {worst_uid} exceeds {TOLERANCE:g}")

    def test_component_terms_match(self):
        """mar, lsr, odds, daily, return% and the post-clip swap must all agree."""
        checks = {"mar": "mar", "lsr": "lsr", "odds": "odds", "daily": "daily",
                  "return": "gain", "swap": "swap"}
        worst = {k: (0.0, None) for k in checks}
        for row in self.score_rows:
            uid = row[_SI["uid"]]
            rows = self.by_uid.get(uid)
            if not rows:
                continue
            win = rows[-50:]
            t = raw_score(win[0][_PI["swap_open"]],
                          [r[_PI["swap_close"]] for r in win], len(rows))
            for col, attr in checks.items():
                expected = row[_SI[col]]
                got = getattr(t, attr)
                rel = abs(got - expected) / max(abs(expected), 1e-9)
                if rel > worst[col][0]:
                    worst[col] = (rel, uid)
        print("  component terms:", " ".join(f"{k}={v[0]:.2e}" for k, v in worst.items()))
        for col, (rel, uid) in worst.items():
            self.assertLess(rel, TOLERANCE, f"{col} mismatch on uid {uid}: {rel:.3e}")


class TestIncentiveReconstruction(unittest.TestCase):
    """The full pipeline should put the top miner near the chain's reported share."""

    def test_field_scores(self):
        from sn88_replica.pipeline import score_field, leaderboard

        pnl, _ = load("pnl")
        dist, _ = load("dist")
        ratio, _ = load("ratio")
        days_raw, _ = load("days")
        days = api.parse_days(days_raw)
        api.check_ratio(ratio)
        api.check_dist(dist)

        field = score_field(pnl, days, dist, ratio)
        board = leaderboard(field, top=5)
        total = sum(s.incentive for s in field.values())

        print(f"\n  reconstructed field: {len(field)} UIDs, "
              f"{sum(1 for s in field.values() if s.incentive > 0)} earning")
        for i, s in enumerate(board, 1):
            print(f"    {i}. uid {s.uid:>3} class={s.asset} incentive={s.incentive:7.4%} "
                  f"gate={s.gate} cash={s.cash:.3f} dec={s.inactivity:.3f}")

        self.assertLessEqual(total, 1.0 + 1e-9)
        self.assertGreater(board[0].incentive, 0.0)
        # every incentive share must be a valid probability
        for s in field.values():
            self.assertGreaterEqual(s.incentive, 0.0)
            self.assertLessEqual(s.incentive, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
