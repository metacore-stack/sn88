"""Stage 3 tests: the session calendar, the fill model, and the look-ahead proof.

The leakage tests here are build-breakers. Guide §20: three of an earlier revision's
errors were of this family and every one made the numbers look better.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.calendar import (          # noqa: E402
    NYSE_TZ,
    SessionCalendar,
    early_closes,
    good_friday,
    holidays,
)
from sn88_replica.execution import (         # noqa: E402
    Submission,
    resolve_fill,
    turnover_cost,
)
from sn88_replica.modes import Candidate, LeakageError   # noqa: E402
from sn88_replica.replay import (            # noqa: E402
    FeatureSnapshot,
    ReplayEngine,
    assert_no_lookahead,
    visible_daily_fields,
)

CAL = SessionCalendar()


class TestCalendarRules(unittest.TestCase):
    def test_ten_holidays_a_year(self):
        for y in (2025, 2026, 2027):
            self.assertEqual(len(holidays(y)), 10, f"{y}: {sorted(holidays(y))}")

    def test_known_holidays(self):
        self.assertIn(date(2026, 1, 1), holidays(2026))      # New Year
        self.assertIn(date(2026, 1, 19), holidays(2026))     # MLK, 3rd Monday
        self.assertIn(date(2026, 2, 16), holidays(2026))     # Washington, 3rd Monday
        self.assertIn(date(2026, 5, 25), holidays(2026))     # Memorial, last Monday
        self.assertIn(date(2026, 9, 7), holidays(2026))      # Labor, 1st Monday
        self.assertIn(date(2026, 11, 26), holidays(2026))    # Thanksgiving, 4th Thursday
        self.assertIn(date(2026, 12, 25), holidays(2026))

    def test_good_friday(self):
        self.assertEqual(good_friday(2026), date(2026, 4, 3))
        self.assertEqual(good_friday(2025), date(2025, 4, 18))
        self.assertIn(good_friday(2026), holidays(2026))

    def test_juneteenth_only_from_2022(self):
        self.assertIn(date(2026, 6, 19), holidays(2026))
        self.assertNotIn(date(2021, 6, 18), holidays(2021))
        self.assertNotIn(date(2021, 6, 21), holidays(2021))

    def test_weekend_observance(self):
        # 2026-07-04 is a Saturday -> observed Friday the 3rd
        self.assertEqual(date(2026, 7, 4).weekday(), 5)
        self.assertIn(date(2026, 7, 3), holidays(2026))
        # 2027-07-04 is a Sunday -> observed Monday the 5th
        self.assertEqual(date(2027, 7, 4).weekday(), 6)
        self.assertIn(date(2027, 7, 5), holidays(2027))

    def test_early_close_rules(self):
        self.assertIn(date(2026, 11, 27), early_closes(2026))   # day after Thanksgiving
        self.assertIn(date(2026, 12, 24), early_closes(2026))   # Christmas Eve
        # July 3 2026 is the observed holiday, so it is NOT an early close
        self.assertNotIn(date(2026, 7, 3), early_closes(2026))
        self.assertIn(date(2025, 7, 3), early_closes(2025))

    def test_an_early_close_is_never_also_a_holiday(self):
        for y in (2024, 2025, 2026, 2027):
            self.assertEqual(early_closes(y) & holidays(y), frozenset())

    def test_unscheduled_closures_are_honoured(self):
        self.assertFalse(CAL.is_session(date(2012, 10, 30)))    # Hurricane Sandy
        self.assertFalse(CAL.is_session(date(2025, 1, 9)))      # day of mourning

    def test_session_count_is_plausible(self):
        n = CAL.count_sessions(date(2026, 1, 1), date(2026, 12, 31))
        self.assertGreaterEqual(n, 250)
        self.assertLessEqual(n, 253)


class TestCutoffs(unittest.TestCase):
    """The cutoffs must be computed in ET, not as a fixed UTC offset."""

    def _et(self, s, which):
        return s.cutoff(which).astimezone(NYSE_TZ).strftime("%H:%M")

    def test_regular_session_cutoffs(self):
        s = CAL.session(date(2026, 9, 1))
        self.assertEqual(self._et(s, "MOO"), "09:28")
        self.assertEqual(self._et(s, "MOC"), "15:50")

    def test_cutoffs_survive_dst_transitions(self):
        for day in (date(2026, 3, 6), date(2026, 3, 9),
                    date(2026, 10, 30), date(2026, 11, 2)):
            s = CAL.session(day)
            self.assertEqual(self._et(s, "MOO"), "09:28", day)
            self.assertEqual(self._et(s, "MOC"), "15:50", day)

    def test_dst_actually_changes_the_utc_offset(self):
        a = CAL.session(date(2026, 3, 6)).moo_cutoff
        b = CAL.session(date(2026, 3, 9)).moo_cutoff
        self.assertNotEqual(a.astimezone(timezone.utc).hour,
                            b.astimezone(timezone.utc).hour,
                            "a fixed UTC offset would pass this test wrongly")

    def test_early_close_moves_the_MOC_cutoff_to_1250(self):
        s = CAL.session(date(2026, 12, 24))
        self.assertTrue(s.early)
        self.assertEqual(self._et(s, "MOO"), "09:28")
        self.assertEqual(self._et(s, "MOC"), "12:50",
                         "hard-coding 15:50 misses the window entirely on early closes")

    def test_window_for_classifies_submissions(self):
        day = date(2026, 9, 1)
        s = CAL.session(day)
        before = s.moo_cutoff - timedelta(minutes=1)
        between = s.moo_cutoff + timedelta(hours=1)
        after = s.moc_cutoff + timedelta(minutes=1)
        self.assertEqual(CAL.window_for(before, day), "MOO")
        self.assertEqual(CAL.window_for(between, day), "MOC")
        self.assertIsNone(CAL.window_for(after, day))


class TestTwoClocks(unittest.TestCase):
    """Sessions vs calendar days - the asymmetry that taxes a Friday-only cadence."""

    def test_ramp_and_window_in_calendar_days(self):
        ramp = CAL.sessions_to_calendar_days(date(2026, 9, 1), 30)
        window = CAL.sessions_to_calendar_days(date(2026, 9, 1), 50)
        self.assertGreater(ramp, 40)        # ~6 calendar weeks, not 30 days
        self.assertGreater(window, 65)      # ~10 calendar weeks
        print(f"\n  30 sessions = {ramp} calendar days;  50 sessions = {window} calendar days")

    def test_a_long_weekend_costs_idle_days_with_no_pnl(self):
        # Thanksgiving 2026: Thu 26th closed, Fri 27th early close
        thu = date(2026, 11, 26)
        self.assertFalse(CAL.is_session(thu))
        prev = CAL.previous_session(thu)
        nxt = CAL.next_session(thu)
        self.assertEqual(CAL.count_calendar_days(prev, nxt), 3)
        self.assertEqual(CAL.count_sessions(prev, nxt), 2)


class TestFillResolution(unittest.TestCase):
    def setUp(self):
        self.s = CAL.session(date(2026, 9, 1))
        self.moo = self.s.moo_cutoff_block
        self.moc = self.s.moc_cutoff_block

    def w(self, x):
        return {"_": 1, "AAPL": x}

    def test_moo_takes_submissions_before_the_open_cutoff(self):
        subs = [Submission(self.moo - 60, self.w(0.9))]
        d = resolve_fill(subs, self.s, "MOO")
        self.assertEqual(d.window, "MOO")
        self.assertEqual(d.submission.weights["AAPL"], 0.9)

    def test_only_the_last_submission_in_a_window_executes(self):
        subs = [Submission(self.moo - 300, self.w(0.1)),
                Submission(self.moo - 200, self.w(0.2)),
                Submission(self.moo - 100, self.w(0.3))]
        d = resolve_fill(subs, self.s, "MOO")
        self.assertEqual(d.submission.weights["AAPL"], 0.3)
        self.assertIn("2 earlier submission(s) discarded", d.reason)

    def test_moc_window_is_between_the_two_cutoffs(self):
        subs = [Submission(self.moo + 10, self.w(0.5))]
        self.assertEqual(resolve_fill(subs, self.s, "MOC").submission.weights["AAPL"], 0.5)
        self.assertIsNone(resolve_fill(subs, self.s, "MOO").submission)

    def test_late_submission_is_ignored_and_waits(self):
        subs = [Submission(self.moc + 60, self.w(0.9))]
        d = resolve_fill(subs, self.s, "MOC")
        self.assertIsNone(d.submission)
        self.assertIn("waits for the next session", d.reason)

    def test_FIRST_EVER_late_submission_is_forced_to_all_cash(self):
        """The new-miner trap: a first strategy after 15:50 initialises to 100% cash."""
        subs = [Submission(self.moc + 60, self.w(0.99), first_ever=True)]
        d = resolve_fill(subs, self.s, "MOC")
        self.assertTrue(d.forced_all_cash)
        self.assertIn("100% CASH", d.reason)

    def test_a_later_non_first_submission_does_not_trigger_the_trap(self):
        subs = [Submission(self.moc + 60, self.w(0.99), first_ever=False)]
        self.assertFalse(resolve_fill(subs, self.s, "MOC").forced_all_cash)


class TestTurnover(unittest.TestCase):
    def test_one_way_turnover(self):
        self.assertAlmostEqual(
            turnover_cost({"AAPL": 0.5, "MSFT": 0.5}, {"AAPL": 0.6, "MSFT": 0.4}), 0.1)
        self.assertAlmostEqual(turnover_cost({}, {"AAPL": 1.0}), 0.5)
        self.assertAlmostEqual(turnover_cost({"_": 1, "AAPL": 0.5}, {"_": 1, "AAPL": 0.5}), 0.0)


class TestSnapshotSeparation(unittest.TestCase):
    def test_a_snapshot_refuses_the_other_window(self):
        s = CAL.session(date(2026, 9, 1))
        snap = FeatureSnapshot(s.day, "MOO", s.moo_cutoff)
        snap.require(s.day, "MOO")
        with self.assertRaises(LeakageError) as ctx:
            snap.require(s.day, "MOC")
        self.assertIn("SEPARATE snapshots", str(ctx.exception))

    def test_visible_filters_on_available_time(self):
        s = CAL.session(date(2026, 9, 1))
        snap = FeatureSnapshot(s.day, "MOC", s.moc_cutoff)
        recs = [(s.moc_cutoff - timedelta(minutes=1), "knowable"),
                (s.moc_cutoff, "at the cutoff"),
                (s.close_utc, "the close itself")]
        self.assertEqual(snap.visible(recs), ["knowable"])

    def test_assert_visible_rejects_the_close(self):
        s = CAL.session(date(2026, 9, 1))
        snap = FeatureSnapshot(s.day, "MOC", s.moc_cutoff)
        with self.assertRaises(LeakageError):
            snap.assert_visible(s.close_utc, "session close")


def _prices(day: date, cheat: bool) -> float:
    """A deterministic 'future' value the honest policy must not consult."""
    return 100.0 + (day.toordinal() % 7) + (5.0 if cheat else 0.0)


class TestReplayEngineAndLookaheadProof(unittest.TestCase):
    START, END = date(2026, 9, 1), date(2026, 9, 18)

    LATENCY = timedelta(minutes=5)

    def _engine(self, policy, *, windows=("MOO", "MOC"), builder=None):
        def default_builder(session, window):
            return FeatureSnapshot(session.day, window,
                                   session.cutoff(window) - self.LATENCY,
                                   {"px": _prices(session.day, False)})
        return ReplayEngine(CAL, builder or default_builder, policy, windows=windows,
                            decision_latency=self.LATENCY)

    def test_engine_walks_sessions_and_windows(self):
        def policy(snap, prev):
            return Candidate({"_": 1, "AAPL": 0.99}, snap.session_day.isoformat(), snap.window)
        steps = self._engine(policy).run(self.START, self.END)
        days = {s.session_day for s in steps}
        self.assertEqual(len(days), CAL.count_sessions(self.START, self.END))
        self.assertEqual(len(steps), len(days) * 2)

    def test_policy_may_decline_to_publish(self):
        def policy(snap, prev):
            return None if snap.window == "MOC" else Candidate(
                {"_": 1, "AAPL": 0.99}, snap.session_day.isoformat(), snap.window)
        steps = self._engine(policy).run(self.START, self.END)
        self.assertTrue(all(s.skipped for s in steps if s.window == "MOC"))
        self.assertFalse(any(s.skipped for s in steps if s.window == "MOO"))

    def test_shared_snapshot_between_windows_is_rejected(self):
        """The exact leak: one daily feature set used for both execution windows."""
        def shared_builder(session, window):
            return FeatureSnapshot(session.day, "MOO",
                                   session.moo_cutoff - self.LATENCY)        # always MOO

        def policy(snap, prev):
            return Candidate({"_": 1, "AAPL": 0.99}, snap.session_day.isoformat(), snap.window)

        with self.assertRaises(LeakageError):
            self._engine(policy, builder=shared_builder).run(self.START, self.END)

    def test_wrong_cutoff_is_rejected(self):
        def bad_builder(session, window):
            return FeatureSnapshot(session.day, window, session.close_utc)   # the close!

        def policy(snap, prev):
            return Candidate({"_": 1, "AAPL": 0.99}, snap.session_day.isoformat(), snap.window)

        with self.assertRaises(LeakageError) as ctx:
            self._engine(policy, builder=bad_builder).run(self.START, self.END)
        self.assertIn("cutoff", str(ctx.exception))

    def test_stale_candidate_stamp_is_rejected(self):
        def policy(snap, prev):
            return Candidate({"_": 1, "AAPL": 0.99}, "2020-01-01", snap.window)
        with self.assertRaises(LeakageError):
            self._engine(policy).run(self.START, self.END)

    # --- the proof --------------------------------------------------------

    def _make(self, cheating: bool):
        def factory(poisoned: bool):
            def builder(session, window):
                data = {"px": _prices(session.day, False)}
                # the "future" value: only a cheating policy reads it, and poisoning
                # replaces it exactly as a real post-cutoff observation would be
                data["future_px"] = POISON_SENTINEL if poisoned else _prices(session.day, True)
                return FeatureSnapshot(session.day, window,
                                       session.cutoff(window) - self.LATENCY, data)

            def policy(snap, prev):
                src = "future_px" if cheating else "px"
                w = round((snap.data[src] % 5) / 100 + 0.9, 4) if snap.data[src] == snap.data[src] else 0.9
                return Candidate({"_": 1, "AAPL": w}, snap.session_day.isoformat(), snap.window)

            return ReplayEngine(CAL, builder, policy, decision_latency=self.LATENCY)
        return factory

    def test_honest_policy_passes_the_poison_replay(self):
        steps = assert_no_lookahead(self._make(cheating=False), self.START, self.END)
        self.assertGreater(len(steps), 10)

    def test_cheating_policy_is_caught_by_the_poison_replay(self):
        with self.assertRaises(LeakageError) as ctx:
            assert_no_lookahead(self._make(cheating=True), self.START, self.END)
        self.assertIn("reading past its cutoff", str(ctx.exception))


POISON_SENTINEL = float("nan")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestCalendarAgainstPublishedFixtures(unittest.TestCase):
    """Fixtures derived independently from NYSE Rule 7.2 and the NYSE Group press releases.

    The rule engine is the implementation; these are the answers. If a rule is ever
    "simplified", one of these breaks.
    """

    #: Published session counts. 2025 is 250 only if the unscheduled Carter closure is encoded.
    SESSION_COUNTS = {2018: 251, 2024: 252, 2025: 250, 2026: 251, 2027: 251, 2028: 251}

    #: (day, moo_unix, moc_unix, is_early) - exact cutoffs across DST and early closes.
    CUTOFFS = [
        ("2025-03-07", 1741357680, 1741380600, False),   # EST, regular
        ("2025-03-10", 1741613280, 1741636200, False),   # EDT, regular
        ("2025-07-03", 1751549280, 1751561400, True),    # EDT, early close
        ("2025-11-28", 1764340080, 1764352200, True),    # EST, early close
        ("2025-12-24", 1766586480, 1766598600, True),    # EST, early close
    ]

    def test_session_counts_match_published_figures(self):
        for year, expected in self.SESSION_COUNTS.items():
            got = CAL.count_sessions(date(year, 1, 1), date(year, 12, 31))
            self.assertEqual(got, expected, f"{year}")

    def test_cutoffs_match_to_the_second(self):
        for day, moo, moc, early in self.CUTOFFS:
            s = CAL.session(date.fromisoformat(day))
            self.assertEqual(s.early, early, day)
            self.assertEqual(s.moo_cutoff_block, moo, f"{day} MOO")
            self.assertEqual(s.moc_cutoff_block, moc, f"{day} MOC")

    def test_new_years_on_a_saturday_is_not_observed_at_all(self):
        """Rule 7.2's 'unusual business conditions' exception: the Friday is year-end."""
        self.assertNotIn(date(2022, 1, 1), holidays(2022))
        self.assertNotIn(date(2021, 12, 31), holidays(2021))
        self.assertNotIn(date(2028, 1, 1), holidays(2028))
        self.assertEqual(len(holidays(2028)), 9, "2028 has nine, not ten")

    def test_saturday_holidays_shift_back_to_friday(self):
        self.assertIn(date(2027, 6, 18), holidays(2027))    # Juneteenth 6/19 is a Sat
        self.assertIn(date(2027, 12, 24), holidays(2027))   # Christmas 12/25 is a Sat

    def test_christmas_eve_early_close_suppressed_when_it_is_the_holiday(self):
        self.assertNotIn(date(2027, 12, 24), early_closes(2027))
        self.assertEqual(sorted(early_closes(2027)), [date(2027, 11, 26)])

    def test_exact_early_close_sets(self):
        self.assertEqual(sorted(early_closes(2024)),
                         [date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)])
        self.assertEqual(sorted(early_closes(2025)),
                         [date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24)])
        self.assertEqual(sorted(early_closes(2026)),
                         [date(2026, 11, 27), date(2026, 12, 24)])
        self.assertEqual(sorted(early_closes(2028)),
                         [date(2028, 7, 3), date(2028, 11, 24)])

    def test_every_early_close_in_the_live_data_era(self):
        """From SN88's FIRST_DATE. These are the only days the MOC cutoff is not 15:50."""
        expected = [date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
                    date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26)]
        got = [d for y in (2025, 2026, 2027) for d in sorted(early_closes(y))
               if d >= date(2025, 3, 20)]
        self.assertEqual(got, expected)

    def test_offsets_come_from_tzdata_not_a_hard_coded_table(self):
        """The Sunshine Protection Act passed the House 2026-07-14 and is pending.

        If DST is abolished mid-flight, a hard-coded transition table silently shifts
        every cutoff by an hour. zoneinfo picks the change up from tzdata; a constant
        table would not. This test documents the dependency.
        """
        import zoneinfo
        self.assertIn("America/New_York", zoneinfo.available_timezones())
        march = CAL.session(date(2026, 3, 6)).moo_cutoff.astimezone(timezone.utc)
        april = CAL.session(date(2026, 4, 6)).moo_cutoff.astimezone(timezone.utc)
        self.assertNotEqual(march.hour, april.hour)


class TestLeakageProbes(unittest.TestCase):
    """One probe per look-ahead vector. Each of these makes a backtest look BETTER.

    Derived from an adversarial enumeration of equity MOO/MOC leaks. The ones already
    covered elsewhere (shared snapshot, wrong cutoff, poison replay) are not repeated.
    """

    LATENCY = timedelta(minutes=5)

    def test_zero_latency_submission_is_rejected(self):
        """Deciding one second before the cutoff models zero compute and zero network.

        Live, feature assembly + inference + optimiser + validation + file write + the
        miner's poll + the HTTP POST take seconds to minutes. A backtest that ignores it
        gives the model information the live system will never have.
        """
        def builder(session, window):
            return FeatureSnapshot(session.day, window, session.cutoff(window))

        def policy(snap, prev):
            return Candidate({"_": 1, "AAPL": 0.99}, snap.session_day.isoformat(), snap.window)

        engine = ReplayEngine(CAL, builder, policy, decision_latency=self.LATENCY)
        with self.assertRaises(LeakageError) as ctx:
            engine.run(date(2026, 9, 1), date(2026, 9, 4))
        self.assertIn("decision latency", str(ctx.exception))

    def test_horizon_is_the_cutoff_less_latency(self):
        engine = ReplayEngine(CAL, lambda s, w: None, lambda s, p: None,
                              decision_latency=self.LATENCY)
        s = CAL.session(date(2026, 9, 1))
        self.assertEqual(engine.horizon(s, "MOC"), s.moc_cutoff - self.LATENCY)
        self.assertEqual(engine.horizon(s, "MOO"), s.moo_cutoff - self.LATENCY)

    def test_negative_latency_is_refused(self):
        with self.assertRaises(ValueError):
            ReplayEngine(CAL, lambda s, w: None, lambda s, p: None,
                         decision_latency=timedelta(seconds=-1))

    def test_todays_OPEN_is_a_leak_at_the_MOO_cutoff(self):
        """Everyone guards the close; almost nobody guards the open.

        A `gap = open / prev_close` feature is a standard daily-bar signal and is pure
        look-ahead at 09:28 - the opening auction has not run.
        """
        self.assertEqual(visible_daily_fields("MOO"), frozenset())
        s = CAL.session(date(2026, 9, 1))
        snap = FeatureSnapshot(s.day, "MOO", s.moo_cutoff - self.LATENCY)
        self.assertNotIn("open", snap.daily_fields())

    def test_close_high_low_and_volume_are_all_leaks_at_the_MOC_cutoff(self):
        """The close has not printed; high/low can still move; volume is incomplete.

        The closing auction alone is roughly 9-10% of the session's notional, so
        full-day volume, full-day VWAP and relative-volume features are unknowable.
        """
        fields = visible_daily_fields("MOC")
        self.assertEqual(fields, frozenset({"open"}))
        for leaked in ("close", "high", "low", "volume"):
            self.assertNotIn(leaked, fields, leaked)

    def test_early_close_hardcoded_at_1550_would_leak_the_actual_close(self):
        """On a 13:00 session, a 15:50 cutoff hands the model 2h50m that does not exist.

        Those are low-volume half-days where a leaked close is maximally predictive, and
        there are ~3 of them a year.
        """
        s = CAL.session(date(2026, 12, 24))
        self.assertTrue(s.early)
        naive = s.open_utc.replace(hour=20, minute=50)      # a hardcoded 15:50 ET in EST
        self.assertGreater(naive, s.close_utc,
                           "a hardcoded 15:50 cutoff falls AFTER the early close")
        self.assertLess(s.moc_cutoff, s.close_utc)

    def test_a_snapshot_dated_after_the_session_close_is_rejected(self):
        def builder(session, window):
            return FeatureSnapshot(session.day, window, session.close_utc)

        def policy(snap, prev):
            return Candidate({"_": 1, "AAPL": 0.99}, snap.session_day.isoformat(), snap.window)

        with self.assertRaises(LeakageError):
            ReplayEngine(CAL, builder, policy,
                         decision_latency=self.LATENCY).run(date(2026, 9, 1), date(2026, 9, 2))

    def test_unscheduled_closure_produces_no_session(self):
        """A rule-only calendar would trade on a day the market was shut."""
        steps = ReplayEngine(
            CAL,
            lambda s, w: FeatureSnapshot(s.day, w, s.cutoff(w) - self.LATENCY),
            lambda snap, prev: Candidate({"_": 1, "AAPL": 0.99},
                                         snap.session_day.isoformat(), snap.window),
            decision_latency=self.LATENCY,
        ).run(date(2025, 1, 8), date(2025, 1, 10))
        self.assertNotIn(date(2025, 1, 9), {s.session_day for s in steps})

    def test_the_clip_absorbs_small_leaks_so_probes_must_assert_on_decisions(self):
        """A leak inflating 1-3 sessions is largely eaten by the outlier clip.

        Only a leak that lifts all 50 sessions - survivorship, restated fundamentals,
        back-adjusted prices - shows up in the score. So a probe that watches SCORE will
        miss most leaks; ``assert_no_lookahead`` compares DECISIONS for that reason.
        """
        from sn88_replica.scoring import score_terms

        base = [0.10] * 50
        leaked = list(base)
        leaked[10] += 5.0                       # one hugely inflated session
        def equity(pct):
            out, v = [], 10_000_000.0
            for r in pct:
                v *= 1 + r / 100
                out.append(v)
            return 10_000_000.0, out
        clean = score_terms(*equity(base)).score
        dirty = score_terms(*equity(leaked)).score
        self.assertLess(abs(dirty - clean) / clean, 0.5,
                        "the clip should absorb most of a single-session leak")


class TestCalendarAgainstTheValidatorsOwnSessionData(unittest.TestCase):
    """The strongest calendar test available: the bars the validator actually uses.

    ``pldaily1`` does not hard-code 09:30/16:00. It takes ``bk0, bk1`` from the FIRST and
    LAST intraday SPY bar of the session and derives both cutoffs from them, so the
    validator's effective calendar is "does SPY have bars that date, and when". A
    rule-based calendar is only correct if it reproduces that.

    Fixture: every SPY session boundary in the live-era ``/daily1`` feed,
    2025-06-02 .. 2026-08-31 (314 sessions), as ``{date: [bk0, bk1]}``.
    """

    FIXTURE = (Path(__file__).resolve().parent.parent / "fixtures"
               / "spy-sessions-2025-06-02_2026-08-31.json")

    @classmethod
    def setUpClass(cls):
        if not cls.FIXTURE.exists():
            raise unittest.SkipTest("SPY session fixture not present")
        cls.sessions = json.loads(cls.FIXTURE.read_text())

    def test_session_membership_matches_exactly(self):
        feed = {date.fromisoformat(d) for d in self.sessions}
        lo, hi = min(feed), max(feed)
        rule = set(CAL.session_days(lo, hi))
        self.assertEqual(feed - rule, set(), "traded but my calendar says closed")
        self.assertEqual(rule - feed, set(), "my calendar says open but nothing traded")
        self.assertEqual(len(feed), 314)

    def test_open_and_close_match_bk0_and_bk1_to_the_second(self):
        bad_open = bad_close = 0
        for d, (bk0, bk1) in self.sessions.items():
            s = CAL.session(date.fromisoformat(d))
            bad_open += int(int(s.open_utc.timestamp()) != bk0)
            bad_close += int(int(s.close_utc.timestamp()) != bk1)
        self.assertEqual(bad_open, 0, "session open disagrees with the first SPY bar")
        self.assertEqual(bad_close, 0, "session close disagrees with the last SPY bar")

    def test_cutoffs_derived_from_real_bars_match_the_rule_calendar(self):
        """bk0 - 120 and bk1 - 600, computed from the feed, must equal my cutoffs."""
        for d, (bk0, bk1) in self.sessions.items():
            s = CAL.session(date.fromisoformat(d))
            self.assertEqual(s.moo_cutoff_block, bk0 - 120, d)
            self.assertEqual(s.moc_cutoff_block, bk1 - 600, d)

    def test_every_short_session_is_flagged_early(self):
        short = {d for d, (a, b) in self.sessions.items() if (b - a) < 6 * 3600}
        self.assertEqual(sorted(short),
                         ["2025-07-03", "2025-11-28", "2025-12-24"])
        for d in short:
            s = CAL.session(date.fromisoformat(d))
            self.assertTrue(s.early, d)
            self.assertEqual(
                s.close_utc.astimezone(NYSE_TZ).strftime("%H:%M"), "13:00", d)

    def test_no_full_session_is_wrongly_flagged_early(self):
        for d, (a, b) in self.sessions.items():
            if (b - a) >= 6 * 3600:
                self.assertFalse(CAL.session(date.fromisoformat(d)).early, d)
