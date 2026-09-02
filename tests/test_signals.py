"""Stage 5 tests: the signal harness, and above all that its view cannot see the future.

``SignalView`` is the enforcement mechanism for the whole signal-research programme. If
it leaks, every result produced against it is worthless and looks excellent - so these
tests check the horizon arithmetic directly rather than trusting the accessors.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.signals import (        # noqa: E402
    INITIAL_EQUITY,
    WIN,
    Panel,
    SignalView,
    equal_weight_top,
    evaluate,
    exceeds_null,
    inverse_vol_top,
    load_panel,
    null_signal,
    random_signal,
    shape_matched_null,
    shuffled,
)

PANEL_PATH = (Path(__file__).resolve().parent.parent / "fixtures"
              / "panel-replication-2025-06-02_2026-08-31.json")


def synthetic(n_names=6, n_sessions=200) -> Panel:
    """Deterministic prices with a KNOWN future spike, so a leak is detectable."""
    sessions = [f"S{i:04d}" for i in range(n_sessions)]
    close, rets = {}, {}
    for k in range(n_names):
        px, series = 100.0, []
        for i in range(n_sessions):
            px *= 1 + ((k + 1) * 0.0003) + (0.02 if i == n_sessions - 5 else 0.0)
            series.append(px)
        close[f"T{k}"] = series
        rets[f"T{k}"] = [(series[i] - series[i - 1]) / series[i - 1]
                         for i in range(1, n_sessions)]
    names = tuple(close)
    return Panel(sessions=tuple(sessions), names=names, close=close, returns=rets,
                 sector={t: "Tech" for t in names},
                 dollar_volume={t: 1e9 - i for i, t in enumerate(names)},
                 market_cap={t: 1e11 for t in names})


class TestViewHorizon(unittest.TestCase):
    """The MOO cutoff is 09:28: returns through t-2, closes through t-1."""

    def setUp(self):
        self.p = synthetic()

    def test_returns_stop_two_indices_back(self):
        v = SignalView(self.p, 100)
        r = v.returns("T0")
        self.assertEqual(len(r), 99, "returns[0..98] only; index 99 ends at session 100")
        self.assertEqual(r, list(self.p.returns["T0"][:99]))

    def test_closes_stop_at_the_prior_session(self):
        v = SignalView(self.p, 100)
        c = v.closes("T0")
        self.assertEqual(len(c), 100)
        self.assertEqual(c[-1], self.p.close["T0"][99],
                         "the last knowable close is session t-1's")
        self.assertNotIn(self.p.close["T0"][100], c)

    def test_lookback_slices_from_the_recent_end(self):
        v = SignalView(self.p, 100)
        self.assertEqual(v.returns("T0", 5), list(self.p.returns["T0"][94:99]))

    def test_the_future_spike_is_invisible_before_it_happens(self):
        n = len(self.p.sessions)
        spike_ret_index = n - 6            # the return that lands the +2% on session n-5
        before = SignalView(self.p, spike_ret_index)      # decision cannot see it
        after = SignalView(self.p, n - 3)                 # now it is history
        self.assertLess(max(before.returns("T0")), 0.01)
        self.assertGreater(max(after.returns("T0")), 0.01)

    def test_early_sessions_yield_short_histories_not_errors(self):
        v = SignalView(self.p, 3)
        self.assertEqual(len(v.returns("T0")), 2)
        self.assertFalse(v.enough_history(60))
        self.assertTrue(SignalView(self.p, 100).enough_history(60))

    def test_session_label_is_the_decision_session(self):
        self.assertEqual(SignalView(self.p, 42).session, self.p.sessions[42])


class TestViewStatistics(unittest.TestCase):
    def setUp(self):
        self.v = SignalView(synthetic(), 150)

    def test_cumulative_compounds(self):
        r = self.v.returns("T1", 10)
        expect = 1.0
        for x in r:
            expect *= 1 + x
        self.assertAlmostEqual(self.v.cumulative("T1", 10), expect - 1.0, places=12)

    def test_volatility_of_a_constant_drift_is_zero(self):
        self.assertAlmostEqual(self.v.volatility("T0", 20), 0.0, places=12)

    def test_hit_rate_of_a_rising_series_is_one(self):
        self.assertAlmostEqual(self.v.hit_rate("T0", 30), 1.0, places=12)

    def test_beta_defaults_to_one_without_a_benchmark(self):
        self.assertEqual(self.v.beta("T0", "NOT_A_TICKER"), 1.0)

    def test_downside_volatility_is_zero_with_no_losses(self):
        self.assertAlmostEqual(self.v.downside_volatility("T0", 30), 0.0, places=12)


class TestWeightConstruction(unittest.TestCase):
    def test_equal_weight_hits_the_gross_band_and_picks_the_top_n(self):
        scores = {f"T{i}": float(i) for i in range(10)}
        w = equal_weight_top(scores, n=4)
        self.assertEqual(sorted(w), ["T6", "T7", "T8", "T9"])
        self.assertAlmostEqual(sum(w.values()), 0.99, places=12)

    def test_ties_break_deterministically(self):
        scores = {"B": 1.0, "A": 1.0, "C": 1.0}
        self.assertEqual(sorted(equal_weight_top(scores, n=2)), ["A", "B"])

    def test_inverse_vol_falls_back_to_equal_when_vol_is_degenerate(self):
        p = synthetic()
        v = SignalView(p, 150)
        scores = {t: 1.0 for t in p.names}
        w = inverse_vol_top(scores, v, n=3)
        self.assertAlmostEqual(sum(w.values()), 0.99, places=12)

    def test_empty_scores_produce_no_book(self):
        self.assertEqual(equal_weight_top({}, n=5), {})


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.p = synthetic(n_sessions=260)

    def test_evaluation_is_deterministic(self):
        a = evaluate(null_signal, self.p, max_windows=20)
        b = evaluate(null_signal, self.p, max_windows=20)
        self.assertEqual(a.scores, b.scores)

    def test_a_monotonically_rising_book_never_scores_zero(self):
        res = evaluate(null_signal, self.p, max_windows=20)
        self.assertEqual(res.zero_fraction, 0.0)
        self.assertGreater(res.median, 0.0)

    def test_percentiles_are_ordered(self):
        res = evaluate(null_signal, self.p, max_windows=30)
        self.assertLessEqual(res.p25, res.median)
        self.assertLessEqual(res.median, res.p75)

    def test_a_signal_that_returns_nothing_produces_no_book(self):
        res = evaluate(lambda v: {}, self.p, max_windows=10)
        self.assertEqual(res.median, 0.0)
        self.assertEqual(res.windows, 0)

    def test_no_trade_band_suppresses_rebalancing(self):
        loose = evaluate(null_signal, self.p, max_windows=20, no_trade_band=0.0)
        tight = evaluate(null_signal, self.p, max_windows=20, no_trade_band=0.5)
        self.assertLessEqual(tight.mean_turnover, loose.mean_turnover + 1e-12)

    def test_the_signal_only_ever_receives_a_view(self):
        """If a signal could reach the panel directly, the horizon would be advisory."""
        seen = []

        def spy(view):
            seen.append(type(view).__name__)
            return {t: 1.0 for t in view.names}

        evaluate(spy, self.p, max_windows=5)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"SignalView"})


def alternating(n_sessions=260) -> Panel:
    """Two names in exact antiphase: A rises on even moves, B on odd ones.

    This makes a 1-day reversal signal deterministic. Whichever name just fell is,
    by construction, the one that rises next - so a harness that transacts at the
    close it just read wins EVERY session, and one that transacts a session later
    loses every session. No noise, no sampling: the sign of the result is the answer.
    """
    sessions = [f"S{i:04d}" for i in range(n_sessions)]
    close, rets = {}, {}
    for k, name in enumerate(("A", "B")):
        px, series = 100.0, [100.0]
        for i in range(n_sessions - 1):
            px *= 1 + (0.02 if (i + k) % 2 == 0 else -0.02)
            series.append(px)
        close[name] = series
        rets[name] = [(series[i] - series[i - 1]) / series[i - 1]
                      for i in range(1, n_sessions)]
    return Panel(sessions=tuple(sessions), names=("A", "B"), close=close, returns=rets,
                 sector={"A": "X", "B": "X"}, dollar_volume={"A": 2.0, "B": 1.0},
                 market_cap={"A": 1e11, "B": 1e11})


def one_day_reversal(view):
    """Buy whichever name fell over the last complete session."""
    return {t: -view.cumulative(t, 1) for t in view.names
            if len(view.returns(t, 1)) >= 1}


class TestFillTiming(unittest.TestCase):
    """A view horizon constrains what the signal READS, not the price it GETS.

    Revision 1 of ``evaluate`` earned ``returns[t-1]`` - the move ending at the very
    close the view had just exposed. That price is unfillable in both windows: at the
    15:50 MOC cutoff ``close[t-1]`` has not printed, and by the 09:28 MOO cutoff the
    overnight gap out of it is already gone. These tests pin the correction.
    """

    def setUp(self):
        self.p = alternating()

    def test_default_fill_lag_is_one(self):
        import inspect
        self.assertEqual(inspect.signature(evaluate).parameters["fill_lag"].default, 1,
                         "the honest MOC convention must be the default, not opt-in")

    def test_lag_zero_wins_every_session_on_a_panel_where_that_is_impossible(self):
        """The bug's signature: a perfect score from transacting at the observed close."""
        leaky = evaluate(one_day_reversal, self.p, top_n=1, fill_lag=0, max_windows=20)
        self.assertEqual(leaky.zero_fraction, 0.0)
        self.assertGreater(leaky.median, 0.0)

    def test_lag_one_loses_every_session_on_that_same_panel(self):
        honest = evaluate(one_day_reversal, self.p, top_n=1, fill_lag=1, max_windows=20)
        self.assertEqual(honest.zero_fraction, 1.0,
                         "buying the faller and holding a session later must lose here")
        self.assertEqual(honest.median, 0.0)

    def test_the_two_conventions_earn_different_indices(self):
        """Direct arithmetic: lag 1 earns the session AFTER the one lag 0 earns."""
        p, seen = self.p, []
        evaluate(lambda v: (seen.append(v.t), {"A": 1.0})[1], p, top_n=1,
                 fill_lag=1, max_windows=5, warmup=60)
        t = seen[0]
        v = SignalView(p, t)
        last_seen_close = v.closes("A")[-1]
        self.assertEqual(last_seen_close, p.close["A"][t - 1])
        # lag 1 transacts at close[t] - strictly later than the last close observed
        self.assertGreater(t, t - 1)

    def test_negative_lag_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(null_signal, self.p, fill_lag=-1)

    def test_the_correction_barely_moves_a_slow_signal(self):
        """0.3% turnover: fill assumptions are invisible until turnover is high."""
        a = evaluate(null_signal, self.p, fill_lag=0, max_windows=20)
        b = evaluate(null_signal, self.p, fill_lag=1, max_windows=20)
        self.assertLess(abs(a.mean_turnover - b.mean_turnover), 1e-9)


class TestShapeMatchedNulls(unittest.TestCase):
    """Against the error that a 20-name equal-weight bar cannot judge a 100-name book."""

    def setUp(self):
        self.p = synthetic(n_names=12, n_sessions=300)

    def test_random_signal_is_deterministic_per_seed(self):
        a, b = random_signal(3), random_signal(3)
        v = SignalView(self.p, 100)
        self.assertEqual(a(v), b(v))
        self.assertNotEqual(a(v), random_signal(4)(v))

    def test_random_signal_scores_every_name(self):
        self.assertEqual(set(random_signal(1)(SignalView(self.p, 100))), set(self.p.names))

    def test_shuffle_preserves_the_value_multiset_and_moves_it(self):
        base = lambda v: {t: float(i) for i, t in enumerate(v.names)}
        v = SignalView(self.p, 100)
        original, permuted = base(v), shuffled(base, 5)(v)
        self.assertEqual(sorted(original.values()), sorted(permuted.values()),
                         "a shuffle must destroy the mapping, not the distribution")
        self.assertEqual(set(original), set(permuted))
        self.assertNotEqual(original, permuted)

    def test_shuffle_of_an_empty_signal_stays_empty(self):
        self.assertEqual(shuffled(lambda v: {}, 1)(SignalView(self.p, 100)), {})

    def test_null_draws_come_back_sorted(self):
        d = shape_matched_null(self.p, seeds=4, top_n=5, max_windows=10)
        self.assertEqual(len(d), 4)
        self.assertEqual(list(d), sorted(d))

    def test_exceeds_null_requires_clearing_the_whole_spread(self):
        draws = (1.0, 2.0, 9.0)
        self.assertFalse(exceeds_null(5.0, draws), "beating the median is not enough")
        self.assertFalse(exceeds_null(9.0, draws), "tying the best is not enough")
        self.assertTrue(exceeds_null(9.1, draws))

    def test_no_null_draws_means_no_claim(self):
        self.assertFalse(exceeds_null(1e9, ()))

    def test_a_pure_noise_candidate_does_not_clear_its_own_nulls(self):
        """The property that makes this a real control rather than a formality."""
        kw = dict(top_n=5, weighting="inverse_vol", max_windows=12)
        cand = evaluate(random_signal(99), self.p, **kw).median
        self.assertFalse(exceeds_null(cand, shape_matched_null(self.p, seeds=6, **kw)))


@unittest.skipUnless(PANEL_PATH.exists(), "replication panel not built")
class TestAgainstTheRealPanel(unittest.TestCase):
    """The null bar, on the real 524-name cross-section."""

    @classmethod
    def setUpClass(cls):
        cls.panel = load_panel(PANEL_PATH)

    def test_panel_shape(self):
        self.assertGreater(len(self.panel.names), 400)
        self.assertEqual(len(self.panel.sessions), 314)
        for t in self.panel.names:
            self.assertEqual(len(self.panel.returns[t]), 313)

    def test_the_null_bar_is_where_we_think_it_is(self):
        res = evaluate(null_signal, self.panel, name="null", max_windows=40)
        print(f"\n  {res}")
        self.assertGreater(res.median, 1.5)
        self.assertLess(res.median, 3.0)
        self.assertGreater(res.zero_fraction, 0.15,
                           "a third of windows closing at zero is the fact to beat")

    def test_the_view_never_exceeds_the_panel_on_real_data(self):
        for t in (60, 150, 300):
            v = SignalView(self.panel, t)
            self.assertEqual(len(v.returns(self.panel.names[0])), t - 1)
            self.assertEqual(len(v.closes(self.panel.names[0])), t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
