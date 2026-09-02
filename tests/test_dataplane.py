"""Stage 4 tests: point-in-time storage and the baseline portfolios.

The PIT tests are leakage tests. Guide §8: reading by ``event_time`` instead of
``available_time`` is the most common way an equity backtest lies, and it lies in the
direction that makes results look better.
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.baselines import (          # noqa: E402
    BASELINES,
    covariance_matrix,
    equal_risk_contribution,
    equal_weight,
    ewma_covariance,
    hierarchical_risk_parity,
    inverse_volatility,
    minimum_variance,
    shrink_covariance,
    stdev,
)
from sn88_replica.modes import LeakageError   # noqa: E402
from sn88_replica.pit import (                # noqa: E402
    Observation,
    PointInTimeStore,
)

UTC = timezone.utc


def dt(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


class TestPointInTime(unittest.TestCase):
    def setUp(self):
        self.s = PointInTimeStore()

    def test_available_time_before_event_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.s.add(Observation(available_time=dt(2026, 1, 1),
                                   event_time=dt(2026, 2, 1),
                                   key="AAPL:eps", value=1.0))

    def test_asof_is_strict(self):
        self.s.add(Observation(dt(2026, 1, 10), dt(2026, 1, 9), "AAPL:close", 100.0))
        self.assertIsNone(self.s.asof("AAPL:close", dt(2026, 1, 10)))     # exactly at
        self.assertEqual(self.s.asof("AAPL:close", dt(2026, 1, 10, 0, 1)), 100.0)

    def test_a_fact_is_invisible_before_it_is_published(self):
        """The publication-lag discipline: Q3 earnings are not knowable in September."""
        self.s.add(Observation(available_time=dt(2026, 10, 28),
                               event_time=dt(2026, 9, 30),
                               key="AAPL:eps", value=1.64))
        self.assertIsNone(self.s.asof("AAPL:eps", dt(2026, 10, 1)),
                          "reading by event_time would have leaked this")
        self.assertEqual(self.s.asof("AAPL:eps", dt(2026, 11, 1)), 1.64)

    def test_revisions_return_what_was_known_then_not_the_restatement(self):
        self.s.add(Observation(dt(2026, 10, 28), dt(2026, 9, 30), "AAPL:eps", 1.64))
        self.s.add(Observation(dt(2027, 2, 1), dt(2026, 9, 30), "AAPL:eps", 1.58))  # restated
        self.assertEqual(self.s.asof("AAPL:eps", dt(2026, 12, 1)), 1.64)
        self.assertEqual(self.s.asof("AAPL:eps", dt(2027, 3, 1)), 1.58)
        self.assertEqual(len(self.s.revisions("AAPL:eps", dt(2026, 9, 30))), 2)

    def test_history_dedupes_to_the_revision_current_at_the_horizon(self):
        self.s.add(Observation(dt(2026, 1, 2), dt(2026, 1, 1), "X:v", 1.0))
        self.s.add(Observation(dt(2026, 1, 3), dt(2026, 1, 1), "X:v", 9.0))  # correction
        self.s.add(Observation(dt(2026, 1, 4), dt(2026, 1, 3), "X:v", 2.0))
        early = self.s.history("X:v", dt(2026, 1, 3))
        later = self.s.history("X:v", dt(2026, 1, 5))
        self.assertEqual([v for _t, v in early], [1.0])
        self.assertEqual([v for _t, v in later], [9.0, 2.0])

    def test_snapshot_bakes_the_horizon_in(self):
        self.s.add(Observation(dt(2026, 1, 10), dt(2026, 1, 9), "AAPL:close", 100.0))
        self.s.add(Observation(dt(2026, 2, 10), dt(2026, 2, 9), "AAPL:close", 120.0))
        snap = self.s.snapshot(dt(2026, 1, 20))
        self.assertEqual(snap.get("AAPL:close"), 100.0)
        self.assertEqual(snap.series("AAPL:close"), [100.0])
        with self.assertRaises(LeakageError):
            snap.assert_horizon(dt(2026, 2, 20))

    def test_require_refuses_to_backfill(self):
        snap = self.s.snapshot(dt(2026, 1, 1))
        with self.assertRaises(LeakageError) as ctx:
            snap.require("MISSING:key")
        self.assertIn("not knowable", str(ctx.exception))

    def test_there_is_no_horizonless_read(self):
        """A method that returns 'the latest value' would defeat the whole module."""
        import inspect

        for name in ("asof", "history"):
            params = inspect.signature(getattr(PointInTimeStore, name)).parameters
            self.assertIn("as_of", params, f"{name} must require a horizon")

    def test_round_trip_through_disk(self):
        self.s.add(Observation(dt(2026, 1, 10), dt(2026, 1, 9), "AAPL:close", 100.0))
        self.s.add(Observation(dt(2026, 1, 11), dt(2026, 1, 10), "AAPL:close", 101.5))
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pit.jsonl")
            self.assertEqual(self.s.save(path), 2)
            back = PointInTimeStore.load(path)
        self.assertEqual(back.asof("AAPL:close", dt(2026, 1, 12)), 101.5)
        self.assertEqual(len(back), 2)

    def test_access_log_records_what_was_consulted(self):
        self.s.access.enabled = True
        self.s.add(Observation(dt(2026, 1, 10), dt(2026, 1, 9), "AAPL:close", 100.0))
        snap = self.s.snapshot(dt(2026, 1, 20))
        snap.get("AAPL:close")
        snap.get("MSFT:close")
        self.assertEqual(self.s.access.keys(), {"AAPL:close", "MSFT:close"})


def synthetic_returns(n_assets=8, n_obs=120, seed=5):
    rng = random.Random(seed)
    market = [rng.gauss(0.0004, 0.010) for _ in range(n_obs)]
    out = {}
    for i in range(n_assets):
        beta = 0.5 + i * 0.15
        idio = 0.004 + i * 0.002
        out[f"A{i:02d}"] = [beta * market[t] + rng.gauss(0, idio) for t in range(n_obs)]
    return out


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.r = synthetic_returns()

    def test_every_baseline_is_long_only_and_hits_the_gross_band(self):
        for name, fn in BASELINES.items():
            w = fn(self.r)
            gross = sum(abs(v) for v in w.values())
            self.assertAlmostEqual(gross, 0.99, places=9, msg=name)
            self.assertTrue(all(v >= -1e-12 for v in w.values()), name)
            self.assertEqual(set(w), set(self.r), name)

    def test_equal_weight(self):
        w = equal_weight(sorted(self.r))
        self.assertEqual(len(set(round(v, 12) for v in w.values())), 1)

    def test_inverse_volatility_favours_the_calmer_asset(self):
        w = inverse_volatility(self.r)
        vols = {k: stdev(v) for k, v in self.r.items()}
        calm = min(vols, key=vols.get)
        wild = max(vols, key=vols.get)
        self.assertGreater(w[calm], w[wild])

    def test_erc_equalises_risk_contributions(self):
        w = equal_risk_contribution(self.r)
        names, cov = covariance_matrix(self.r)
        vec = [w[n] for n in names]
        marginal = [sum(cov[i][j] * vec[j] for j in range(len(names)))
                    for i in range(len(names))]
        contrib = [vec[i] * marginal[i] for i in range(len(names))]
        spread = (max(contrib) - min(contrib)) / max(contrib)
        self.assertLess(spread, 0.05, "risk contributions should be nearly equal")

    def test_minimum_variance_beats_equal_weight_in_sample_variance(self):
        names, cov = covariance_matrix(self.r)
        def port_var(w):
            v = [w[n] for n in names]
            return sum(v[i] * cov[i][j] * v[j] for i in range(len(names)) for j in range(len(names)))
        self.assertLess(port_var(minimum_variance(self.r)), port_var(equal_weight(names)))

    def test_hrp_runs_and_is_diversified(self):
        w = hierarchical_risk_parity(self.r)
        self.assertEqual(len(w), len(self.r))
        self.assertLess(max(w.values()), 0.6, "HRP should not concentrate in one name")
        self.assertGreater(min(w.values()), 0.0)

    def test_shrinkage_conditions_a_singular_matrix(self):
        dup = {"A": [0.01, -0.01, 0.02, -0.02] * 10}
        dup["B"] = list(dup["A"])          # perfectly collinear -> singular
        _names, cov = covariance_matrix(dup)
        w = minimum_variance(dup)          # must not raise
        self.assertAlmostEqual(sum(w.values()), 0.99, places=9)
        shrunk = shrink_covariance(cov, 0.3)
        self.assertGreater(shrunk[0][0], 0.0)
        self.assertLess(abs(shrunk[0][1]), abs(cov[0][1]) + 1e-18)

    def test_ewma_weights_recent_observations_more(self):
        r = {"A": [0.001] * 60 + [0.05] * 10}
        _n, sample = covariance_matrix(r)
        _n, ewma = ewma_covariance(r, halflife=5)
        self.assertGreater(ewma[0][0], sample[0][0],
                           "a recent volatility burst must dominate the EWMA")

    def test_single_asset_edge_case(self):
        one = {"A": [0.01, -0.005, 0.002]}
        for name, fn in BASELINES.items():
            w = fn(one)
            self.assertAlmostEqual(sum(w.values()), 0.99, places=9, msg=name)

    def test_baselines_are_deterministic(self):
        for name, fn in BASELINES.items():
            self.assertEqual(fn(self.r), fn(self.r), name)


class TestBaselinesThroughTheReplica(unittest.TestCase):
    """Guide §19: the bar is the replica's score, not Sharpe."""

    def test_each_baseline_produces_a_scoreable_book(self):
        from sn88_replica.scoring import score_terms

        r = synthetic_returns(n_assets=6, n_obs=60)
        length = min(len(v) for v in r.values())
        results = {}
        for name, fn in BASELINES.items():
            w = fn(r)
            equity, init = [], 10_000_000.0
            value = init
            for t in range(length):
                port = sum(w[k] * r[k][t] for k in w)
                value *= 1 + port
                equity.append(value)
            results[name] = score_terms(init, equity[-50:]).score
        print("\n  baseline scores through the replica:")
        for name, sc in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<26} {sc:12.4f}")
        self.assertTrue(all(s >= 0 for s in results.values()))
        self.assertEqual(len(results), len(BASELINES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
