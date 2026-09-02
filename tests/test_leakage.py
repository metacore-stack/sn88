"""Leakage probes. These MUST fail the build if the guard regresses.

Guide §20. Three of revision 3's errors were of this family, and every one of them made
the numbers look better. A backtest that silently improves when you introduce a bug is
the only kind of bug you will not go looking for.
"""

from __future__ import annotations

import os
import random
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.modes import (   # noqa: E402
    Candidate,
    ForwardPaths,
    LeakageError,
    PolicyBacktest,
    RealisedHistory,
    project,
    reconcile,
)


def history(n=60, init=10_000_000.0, seed=3):
    rng = random.Random(seed)
    closes, eq, dates = [], init, []
    for i in range(n):
        eq *= 1 + rng.gauss(0.15, 1.0) / 100
        closes.append(eq)
        dates.append(f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}")
    return RealisedHistory(tuple(dates), tuple(closes), init)


def paths(starts_after, k=64, n=20, seed=7):
    rng = random.Random(seed)
    return ForwardPaths(
        returns_pct=tuple(tuple(rng.gauss(0.15, 1.0) for _ in range(n)) for _ in range(k)),
        starts_after=starts_after,
        generator="block_bootstrap",
    )


CAND = Candidate(weights={"_": 1, "AAPL": 0.5, "MSFT": 0.49}, generated_at="2026-03-01")


class TestForwardProjectionGuard(unittest.TestCase):
    """Mode B must refuse to apply a candidate to returns that predate it."""

    def test_paths_starting_before_candidate_generation_are_rejected(self):
        h = history()
        bad = paths(starts_after="2026-01-05")     # earlier than CAND.generated_at
        with self.assertRaises(LeakageError) as ctx:
            project(CAND, h, bad)
        self.assertIn("already known", str(ctx.exception))

    def test_paths_starting_inside_realised_history_are_rejected(self):
        h = history()
        cand = Candidate(weights={"_": 1, "AAPL": 0.99}, generated_at="2026-01-01")
        bad = paths(starts_after="2026-01-10")     # inside h, which ends later
        with self.assertRaises(LeakageError):
            project(cand, h, bad)

    def test_valid_projection_runs_and_returns_a_distribution(self):
        h = history()
        cand = Candidate(weights={"_": 1, "AAPL": 0.6, "MSFT": 0.39},
                         generated_at=h.last_date)
        ok = paths(starts_after=h.last_date)
        result = project(cand, h, ok)
        self.assertEqual(len(result.scores), 64)
        self.assertGreaterEqual(result.breach_probability, 0.0)
        self.assertLessEqual(result.breach_probability, 1.0)
        self.assertLessEqual(result.p10, result.median)
        self.assertLessEqual(result.median, result.p90)

    def test_realised_history_is_never_mutated(self):
        h = history()
        before = tuple(h.swap_closes)
        cand = Candidate(weights={"_": 1, "AAPL": 0.99}, generated_at=h.last_date)
        project(cand, h, paths(starts_after=h.last_date))
        self.assertEqual(before, h.swap_closes)

    def test_gross_above_one_is_refused(self):
        """Upstream SILENTLY discards it; we must not let it reach the publisher."""
        h = history()
        cand = Candidate(weights={"_": 1, "AAPL": 0.7, "MSFT": 0.5},
                         generated_at=h.last_date)
        with self.assertRaises(ValueError) as ctx:
            project(cand, h, paths(starts_after=h.last_date))
        self.assertIn("SILENTLY", str(ctx.exception))


class TestReconcileTakesNoCandidate(unittest.TestCase):
    """Mode A validates the replica. It must be structurally unable to rank anything."""

    def test_signature_accepts_no_weights(self):
        import inspect

        params = set(inspect.signature(reconcile).parameters)
        self.assertNotIn("candidate", params)
        self.assertNotIn("weights", params)

    def test_reconcile_is_deterministic(self):
        h = history()
        self.assertEqual(reconcile(h).score, reconcile(h).score)


class TestGaussianGeneratorIsFlagged(unittest.TestCase):
    """An i.i.d. Gaussian path set is a unit-test tool, never a sizing input."""

    def test_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ForwardPaths(returns_pct=((0.1, 0.2),), starts_after="2026-01-01",
                         generator="gaussian_iid")
        self.assertTrue(any("gaussian_iid" in str(w.message) for w in caught))


class TestPolicyBacktestClock(unittest.TestCase):
    """Mode C must reject a policy that returns a stale or forward-dated candidate."""

    def test_mismatched_stamp_is_rejected(self):
        sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]

        def bad_policy(_snapshot, _prev):
            return Candidate(weights={"_": 1, "AAPL": 0.99}, generated_at="2026-01-06")

        bt = PolicyBacktest(
            sessions=sessions,
            load_snapshot=lambda s, w: None,
            policy=bad_policy,
            realise=lambda c, s: 0.1,
        )
        with self.assertRaises(LeakageError):
            bt.run()

    def test_well_behaved_policy_runs(self):
        sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]

        def good_policy(_snapshot, _prev):
            return Candidate(weights={"_": 1, "AAPL": 0.99}, generated_at=good_policy.now)

        bt = PolicyBacktest(
            sessions=sessions,
            load_snapshot=lambda s, w: setattr(good_policy, "now", s),
            policy=good_policy,
            realise=lambda c, s: 0.1,
        )
        out = bt.run()
        self.assertEqual([s for s, _c, _r in out], sessions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
