"""Competition-replication track: parsing SN88's own feed and loading it point-in-time.

The load tests use a small synthetic feed so they run offline and fast. The
availability semantics they pin were verified against the real 3.9M-row ``/daily1``
pull, on AAPL around 2026-08-31::

    raw feed   08-28: open=316.845 close=319.70
               08-31: open=319.600 close=316.85

    09:28 MOO cutoff  -> open 316.845  close 319.70   (yesterday's; the auction has not run)
    09:31             -> open 319.600  close 319.70   (today's open is now known)
    15:50 MOC cutoff  -> open 319.600  close 319.70   (TODAY'S CLOSE IS INVISIBLE)
    16:01             -> open 319.600  close 316.85   (and only now does it appear)
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.calendar import SessionCalendar          # noqa: E402
from sn88_replica.market import (                          # noqa: E402
    DAILY1_COLS,
    MarketFeed,
    Track,
)
from sn88_replica.pit import PointInTimeStore              # noqa: E402

CAL = SessionCalendar()
D1, D2 = date(2026, 8, 28), date(2026, 8, 31)      # a Friday and the following Monday


def daily(ticker, day, o, c, v=1e6):
    s = CAL.session(day)
    return [day.isoformat(), int(s.open_utc.timestamp()), ticker, o, max(o, c), min(o, c), c, v, "day"]


def intraday(ticker, day, minutes_in, price):
    s = CAL.session(day)
    block = int((s.open_utc + timedelta(minutes=minutes_in)).timestamp())
    return [day.isoformat(), block, ticker, price, price, price, price, 1e5, "ochl"]


FEED = [
    daily("AAPL", D1, 316.845, 319.70),
    daily("AAPL", D2, 319.600, 316.85),
    daily("MSFT", D1, 500.0, 505.0),
    daily("MSFT", D2, 505.0, 498.0),
    intraday("AAPL", D2, 0, 319.6),
    intraday("AAPL", D2, 330, 318.0),      # 15:00 bar
    intraday("AAPL", D2, 360, 317.0),      # 15:30 bar - does not close until 16:00
]
SPLITS = [["2026-08-31", "AAPL", 1.0, 4.0]]
DIVIDENDS = [["2026-08-31", "2026-09-15", "MSFT", 0.83, "USD", "CD"]]


class TestParsing(unittest.TestCase):
    def setUp(self):
        self.feed = MarketFeed(CAL)

    def test_daily_and_intraday_are_separated_by_the_ochl_flag(self):
        d = list(self.feed.parse_daily(FEED))
        i = list(self.feed.parse_intraday(FEED))
        self.assertEqual(len(d), 4)
        self.assertEqual(len(i), 3)
        self.assertEqual({b.ticker for b in i}, {"AAPL"})

    def test_daily_fields(self):
        bar = next(b for b in self.feed.parse_daily(FEED)
                   if b.ticker == "AAPL" and b.session == D2)
        self.assertAlmostEqual(bar.open, 319.6)
        self.assertAlmostEqual(bar.close, 316.85)

    def test_split_factor(self):
        s = next(MarketFeed.parse_splits(SPLITS))
        self.assertEqual(s.kind, "split")
        self.assertAlmostEqual(s.split_factor, 4.0, msg="from=1 to=4 is a 4-for-1")

    def test_dividend_carries_ex_and_pay_dates(self):
        d = next(MarketFeed.parse_dividends(DIVIDENDS))
        self.assertEqual(d.effective, D2)               # applied on the EX date
        self.assertEqual(d.pay_date, date(2026, 9, 15))
        self.assertAlmostEqual(d.amount, 0.83)


class TestPointInTimeLoading(unittest.TestCase):
    """The whole reason this module exists: available_time is not the session date."""

    def setUp(self):
        self.store = PointInTimeStore()
        MarketFeed(CAL).load_daily(self.store, FEED, fields=("open", "close"))
        self.s = CAL.session(D2)

    def see(self, when, field, ticker="AAPL"):
        return self.store.snapshot(when).get(f"replication:{ticker}:{field}")

    def test_todays_close_is_invisible_at_the_MOC_cutoff(self):
        """The leak everyone guards. Verified against the real feed."""
        self.assertAlmostEqual(self.see(self.s.moc_cutoff, "close"), 319.70)
        self.assertNotAlmostEqual(self.see(self.s.moc_cutoff, "close"), 316.85)

    def test_todays_close_appears_only_after_the_session_ends(self):
        after = self.s.close_utc + timedelta(minutes=1)
        self.assertAlmostEqual(self.see(after, "close"), 316.85)

    def test_todays_open_is_invisible_at_the_MOO_cutoff(self):
        """The leak almost nobody guards: at 09:28 the auction has not run."""
        self.assertAlmostEqual(self.see(self.s.moo_cutoff, "open"), 316.845)
        self.assertNotAlmostEqual(self.see(self.s.moo_cutoff, "open"), 319.600)

    def test_todays_open_appears_after_the_open(self):
        after = self.s.open_utc + timedelta(minutes=1)
        self.assertAlmostEqual(self.see(after, "open"), 319.600)
        self.assertAlmostEqual(self.see(after, "close"), 319.70,
                               msg="the close must still be yesterday's")

    def test_open_and_close_have_different_horizons(self):
        """Storing a whole bar under one timestamp is what makes `gap` leak."""
        mid = self.s.open_utc + timedelta(hours=1)
        self.assertAlmostEqual(self.see(mid, "open"), 319.600)    # today's
        self.assertAlmostEqual(self.see(mid, "close"), 319.70)    # yesterday's


class TestIntradayHorizons(unittest.TestCase):
    def setUp(self):
        self.store = PointInTimeStore()
        MarketFeed(CAL).load_intraday(self.store, FEED)
        self.s = CAL.session(D2)

    def test_a_bar_is_knowable_only_after_it_closes(self):
        """The 15:30 bar does not close until 16:00, so 15:50 must not see it."""
        snap = self.store.snapshot(self.s.moc_cutoff)
        series = snap.series("replication:AAPL:intraday_close")
        self.assertIn(318.0, series, "the 15:00 bar has closed and is usable")
        self.assertNotIn(317.0, series, "the 15:30 bar has NOT closed at 15:50")


class TestCorporateActions(unittest.TestCase):
    def test_actions_are_knowable_on_their_effective_date_only(self):
        store = PointInTimeStore()
        n = MarketFeed(CAL).load_actions(store, SPLITS, DIVIDENDS)
        self.assertEqual(n, 2)
        s = CAL.session(D2)
        before = store.snapshot(CAL.session(D1).close_utc)
        after = store.snapshot(s.close_utc)
        self.assertIsNone(before.get("replication:AAPL:split"))
        self.assertAlmostEqual(after.get("replication:AAPL:split"), 4.0)
        self.assertAlmostEqual(after.get("replication:MSFT:dividend"), 0.83)


class TestTrackLabelling(unittest.TestCase):
    """Survivorship status lives in the data, not in the documentation."""

    def test_tracks_are_namespaced_apart(self):
        store = PointInTimeStore()
        feed = MarketFeed(CAL)
        feed.load_daily(store, FEED, track=Track.REPLICATION, fields=("close",))
        feed.load_daily(store, FEED, track=Track.LONG_REGIME, fields=("close",))
        keys = store.keys()
        self.assertIn("replication:AAPL:close", keys)
        self.assertIn("long_regime:AAPL:close", keys)

    def test_a_model_cannot_read_the_wrong_track_by_accident(self):
        store = PointInTimeStore()
        MarketFeed(CAL).load_daily(store, FEED, track=Track.LONG_REGIME, fields=("close",))
        snap = store.snapshot(CAL.session(D2).close_utc)
        self.assertIsNone(snap.get("replication:AAPL:close"),
                          "asking for the authoritative track must not silently return "
                          "survivorship-biased data")


class TestUniverseReconstruction(unittest.TestCase):
    FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "fixtures")

    def test_eligibility_uses_the_nearest_EARLIER_snapshot(self):
        got = MarketFeed.eligible_on(self.FIXTURES, date(2026, 9, 2))
        if not got:
            self.skipTest("no archived /assets snapshots")
        self.assertGreater(len(got), 400)
        self.assertIn("AAPL", got)

    def test_a_date_before_any_snapshot_returns_empty_rather_than_todays_universe(self):
        """Returning today's list would be a membership look-ahead."""
        self.assertEqual(MarketFeed.eligible_on(self.FIXTURES, date(2000, 1, 1)),
                         frozenset())

    def test_departed_is_not_a_delisting_list(self):
        """AA, AAL, ACM, AEE and AGNC all still trade - they left the ELIGIBLE set."""
        gone = MarketFeed.departed(["AAPL", "AA", "AAL", "AGNC"], ["AAPL"])
        self.assertEqual(gone, ["AA", "AAL", "AGNC"])
        doc = MarketFeed.departed.__doc__
        self.assertIn("Not a delisting list", doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
