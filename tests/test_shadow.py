"""Stage 7 tests: the shadow harness and the registration gate.

The most important test in this file asserts that the report REFUSES to say ready when a
gate is unproven. A readiness report that quietly counted UNPROVEN as PASS would be the
most expensive green light in the project.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.shadow import (            # noqa: E402
    GateStatus,
    ReadinessCriteria,
    ShadowRecord,
    ShadowRun,
)


def record(session, window="MOC", **kw):
    base = dict(
        session=session, window=window, recorded_at=f"{session}T20:00:00+00:00",
        published=True, upstream_would_accept=True, governor_approved=True,
        cutoff_met=True, weights_hash="abc123", effective_cash=0.005, gross=0.995,
        turnover=0.02, projected_incentive=0.01,
    )
    base.update(kw)
    return ShadowRecord(**base)


def clean_run(path, sessions=12):
    run = ShadowRun(path)
    d = date(2026, 9, 1)
    added = 0
    while added < sessions:
        if d.weekday() < 5:
            for w in ("MOO", "MOC"):
                run.append(record(d.isoformat(), w))
            added += 1
        d += timedelta(days=1)
    return run


class TestShadowLog(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "shadow.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_and_resumability(self):
        run = clean_run(self.path, sessions=3)
        self.assertEqual(len(run.records()), 6)
        # a fresh object over the same file sees the history - the run spans real days
        again = ShadowRun(self.path)
        self.assertEqual(len(again.records()), 6)
        again.append(record("2026-09-30"))
        self.assertEqual(len(ShadowRun(self.path).records()), 7)

    def test_stats(self):
        run = clean_run(self.path, sessions=5)
        run.append(record("2026-09-30", published=False))
        s = run.stats()
        self.assertEqual(s["sessions"], 6)
        self.assertEqual(s["published"], 10)
        self.assertEqual(s["skipped"], 1)

    def test_weights_hash_is_stable_and_order_independent(self):
        a = ShadowRecord.hash_weights({"AAPL": 0.5, "MSFT": 0.49})
        b = ShadowRecord.hash_weights({"MSFT": 0.49, "AAPL": 0.5})
        c = ShadowRecord.hash_weights({"AAPL": 0.5, "MSFT": 0.48})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestReadinessGate(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "shadow.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def full(self, run, **kw):
        args = dict(rule_detector_ran=True, leakage_probes_green=True,
                    market_data_available=True, projection_error=0.10)
        args.update(kw)
        return run.readiness(**args)

    def test_a_clean_run_with_everything_reported_is_still_not_ready(self):
        """`strategy_edge` is permanently UNPROVEN - shadow validates operations only."""
        rep = self.full(clean_run(self.path))
        self.assertFalse(rep.ready)
        self.assertEqual(rep.failures, [])
        self.assertEqual([g.name for g in rep.unproven], ["strategy_edge"])

    def test_unproven_is_never_counted_as_a_pass(self):
        rep = self.full(clean_run(self.path), market_data_available=False)
        self.assertFalse(rep.ready)
        names = {g.name for g in rep.unproven}
        self.assertIn("projection_accuracy", names)
        self.assertIn("strategy_edge", names)
        self.assertEqual(rep.failures, [], "unproven must not be reported as failed either")

    def test_too_few_sessions_fails(self):
        rep = self.full(clean_run(self.path, sessions=4))
        self.assertIn("sessions_observed", [g.name for g in rep.failures])

    def test_a_would_be_discarded_file_fails_the_gate(self):
        run = clean_run(self.path)
        run.append(record("2026-09-30", upstream_would_accept=False))
        rep = self.full(run)
        self.assertIn("nothing_would_be_discarded", [g.name for g in rep.failures])

    def test_a_missed_cutoff_fails_the_gate(self):
        run = clean_run(self.path)
        run.append(record("2026-09-30", cutoff_met=False))
        rep = self.full(run)
        self.assertIn("cutoffs_met", [g.name for g in rep.failures])

    def test_publishing_despite_a_veto_fails_the_gate(self):
        run = clean_run(self.path)
        run.append(record("2026-09-30", governor_approved=False, vetoes=("cash_band",)))
        rep = self.full(run)
        self.assertIn("governor_never_overruled", [g.name for g in rep.failures])

    def test_a_vetoed_but_unpublished_cycle_is_fine(self):
        """The governor working is not a failure - overruling it is."""
        run = clean_run(self.path)
        run.append(record("2026-09-30", published=False, governor_approved=False))
        rep = self.full(run)
        self.assertNotIn("governor_never_overruled", [g.name for g in rep.failures])

    def test_any_fault_fails_the_gate(self):
        run = clean_run(self.path)
        run.append(record("2026-09-30", faults=("data_quality_gate",)))
        rep = self.full(run)
        self.assertIn("no_unhandled_faults", [g.name for g in rep.failures])

    def test_only_one_window_exercised_fails(self):
        run = ShadowRun(self.path)
        for i in range(12):
            run.append(record(f"2026-09-{i + 1:02d}", "MOC"))
        rep = self.full(run)
        self.assertIn("both_windows_exercised", [g.name for g in rep.failures])

    def test_rule_detector_must_have_run(self):
        rep = self.full(clean_run(self.path), rule_detector_ran=False)
        self.assertIn("rule_detector_ran", [g.name for g in rep.failures])

    def test_unreported_leakage_probes_are_unproven_not_passed(self):
        rep = self.full(clean_run(self.path), leakage_probes_green=None)
        self.assertIn("leakage_probes", [g.name for g in rep.unproven])
        self.assertFalse(rep.ready)

    def test_failing_leakage_probes_fail_the_gate(self):
        rep = self.full(clean_run(self.path), leakage_probes_green=False)
        self.assertIn("leakage_probes", [g.name for g in rep.failures])

    def test_poor_projection_accuracy_fails_when_measurable(self):
        rep = self.full(clean_run(self.path), projection_error=0.80)
        self.assertIn("projection_accuracy", [g.name for g in rep.failures])

    def test_summary_names_what_is_missing(self):
        rep = self.full(clean_run(self.path, sessions=2), market_data_available=False)
        text = rep.summary()
        self.assertIn("NOT READY", text)
        self.assertIn("sessions_observed", text)
        self.assertIn("UNPROVEN", text)

    def test_criteria_are_configurable_but_default_to_zero_tolerance(self):
        c = ReadinessCriteria()
        self.assertEqual(c.max_faults, 0)
        self.assertEqual(c.max_would_be_discarded, 0)
        self.assertEqual(c.max_missed_cutoffs, 0)
        self.assertEqual(c.max_governor_overrides, 0)

    def test_an_empty_run_is_not_ready(self):
        rep = self.full(ShadowRun(self.path))
        self.assertFalse(rep.ready)
        self.assertIn("sessions_observed", [g.name for g in rep.failures])


if __name__ == "__main__":
    unittest.main(verbosity=2)
