"""Stage 2 tests: the strategy parser prediction, and the publisher's safety properties.

The point of these is that upstream fails SILENTLY. Every test below asserts we catch a
failure that would otherwise show up as an unexplained zero score two sessions later.
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api                                    # noqa: E402
from sn88_replica.publisher import Publisher, PublishError      # noqa: E402
from sn88_replica.strategy import (                             # noqa: E402
    Strategy,
    parse_like_upstream,
    predict_acceptance,
    sanitize,
    serialize,
    validate,
)
from sn88_replica.universe import parse_assets                  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
HOTKEY = "5DFqEA000000000000000000000000000000000000000000"


def load_universe():
    hits = sorted(glob.glob(os.path.join(FIXTURES, "*-assets.txt")))
    if hits:
        return parse_assets(Path(hits[-1]).read_text(encoding="utf-8", errors="replace"))
    if os.environ.get("SN88_OFFLINE"):
        raise unittest.SkipTest("no /assets fixture and SN88_OFFLINE is set")
    return parse_assets(api.fetch("assets"))


class TestUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = load_universe()

    def test_shape(self):
        self.assertGreater(len(self.u), 400)
        self.assertEqual(len(self.u.cash_tickers), 16)
        print(f"\n  universe: {len(self.u)} tickers, {len(self.u.cash_tickers)} cash-classified, "
              f"dereg={list(self.u.dereg_list)}, ratio={list(self.u.asset_ratio)}")

    def test_cash_list_is_curated_not_a_bond_rule(self):
        """The asymmetry worth real score: TLT/GLD are free, SGOV/AGG are taxed."""
        for free in ("TLT", "LQD", "GLD", "IAU", "SLV", "JEPI", "JEPQ", "VNQ"):
            if self.u.is_tradable(free):
                self.assertFalse(self.u.is_cash(free), f"{free} should NOT be cash-classified")
        for taxed in ("SGOV", "BIL", "AGG", "BND", "IEF"):
            self.assertTrue(self.u.is_cash(taxed), f"{taxed} should be cash-classified")

    def test_tickers_are_case_sensitive(self):
        self.assertTrue(self.u.is_tradable("AAPL"))
        self.assertFalse(self.u.is_tradable("aapl"))

    def test_malformed_payload_raises_rather_than_emptying_the_universe(self):
        with self.assertRaises(ValueError):
            parse_assets("nothing useful here")


class TestUpstreamParserPrediction(unittest.TestCase):
    """We must predict acceptance exactly, because upstream never tells you."""

    def test_comment_is_silently_discarded(self):
        text = "{'_':1,'AAPL':0.99}   # top pick"
        self.assertEqual(sanitize(text), "{'_':1,'AAPL':0.99}toppick")
        self.assertIsNone(parse_like_upstream(text), "a comment must be detected as fatal")

    def test_expression_silently_changes_meaning(self):
        parsed = parse_like_upstream("{'_':1,'AAPL':(0.1+0.2)}")
        self.assertIsNotNone(parsed)
        self.assertNotEqual(parsed["AAPL"], 0.3)          # 0.30000000000000004
        self.assertAlmostEqual(parsed["AAPL"], 0.3, places=15)

    def test_gross_above_one_is_discarded(self):
        self.assertIsNone(parse_like_upstream("{'_':1,'AAPL':0.7,'MSFT':0.5}"))

    def test_reserved_keys_do_not_consume_gross(self):
        self.assertIsNotNone(parse_like_upstream("{'_':1,'*':5,'AAPL':0.99}"))

    def test_asset_class_two_is_discarded(self):
        self.assertIsNone(parse_like_upstream("{'_':2,'AAPL':0.5}"))

    def test_whitespace_and_newlines_are_harmless(self):
        text = "{'_': 1,\n 'AAPL': 0.5,\n 'MSFT': 0.49,\n}"
        parsed = parse_like_upstream(text)
        self.assertEqual(parsed["AAPL"], 0.5)

    def test_dotted_and_hyphenated_tickers_survive(self):
        parsed = parse_like_upstream("{'_':1,'BRK.B':0.5,'BF-B':0.49}")
        self.assertIn("BRK.B", parsed)
        self.assertIn("BF-B", parsed)

    def test_predict_acceptance_flags_reclassification(self):
        u = load_universe()
        text = serialize(Strategy({"AAPL": 0.5, "SGOV": 0.49}))
        acc = predict_acceptance(text, u)
        self.assertTrue(acc.accepted)
        self.assertIn("SGOV", acc.reclassified)
        self.assertAlmostEqual(acc.effective_cash, 0.5, places=6)


class TestSerializationRoundTrip(unittest.TestCase):
    def test_serialized_form_is_sanitizer_stable(self):
        s = Strategy({"AAPL": 0.15, "MSFT": 0.2, "BRK.B": 0.09, "NVDA": 0.25,
                      "GOOGL": 0.1, "SPY": 0.2})
        text = serialize(s)
        self.assertEqual(sanitize(text), text, "serialised form must survive untouched")
        parsed = parse_like_upstream(text)
        for k, v in s.weights.items():
            self.assertAlmostEqual(parsed[k], v, places=9)

    def test_tiny_weights_round_trip(self):
        s = Strategy({"AAPL": 0.9899999999, "MSFT": 0.0000001})
        parsed = parse_like_upstream(serialize(s))
        self.assertIsNotNone(parsed)


class TestValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = load_universe()

    def good(self):
        names = [a.ticker for a in self.u.most_liquid(40)
                 if not self.u.is_cash(a.ticker)][:20]
        return Strategy({t: 0.0495 for t in names})     # 20 x 0.0495 = 0.99

    def test_a_good_strategy_validates(self):
        r = validate(self.good(), self.u)
        self.assertTrue(r.ok, r.errors)
        self.assertLessEqual(r.effective_cash, 0.01 + 1e-9)

    def test_gross_over_one_is_an_error(self):
        s = Strategy({"AAPL": 0.7, "MSFT": 0.5})
        r = validate(s, self.u)
        self.assertFalse(r.ok)
        self.assertTrue(any("SILENTLY DISCARD" in e for e in r.errors))

    def test_underinvested_is_an_error(self):
        s = Strategy({"AAPL": 0.10, "MSFT": 0.10})
        r = validate(s, self.u, max_single_name=1.0)
        self.assertFalse(r.ok)
        self.assertTrue(any("becomes cash" in e for e in r.errors))

    def test_unknown_ticker_is_an_error(self):
        s = Strategy({"AAPL": 0.5, "NOTATICKER": 0.49})
        r = validate(s, self.u, max_single_name=1.0)
        self.assertFalse(r.ok)
        self.assertTrue(any("NOTATICKER" in e for e in r.errors))

    def test_lowercase_ticker_is_rejected(self):
        s = Strategy({"aapl": 0.99})
        r = validate(s, self.u, max_single_name=1.0)
        self.assertFalse(r.ok)

    def test_shorts_rejected_by_default(self):
        s = Strategy({"AAPL": 0.6, "MSFT": -0.39})
        r = validate(s, self.u, max_single_name=1.0)
        self.assertFalse(r.ok)
        self.assertTrue(any("shorts" in e for e in r.errors))

    def test_cash_classified_holding_is_caught(self):
        s = Strategy({"AAPL": 0.5, "SGOV": 0.49})
        r = validate(s, self.u, max_single_name=1.0)
        self.assertFalse(r.ok)
        self.assertTrue(any("SGOV" in e for e in r.errors))

    def test_missing_asset_class_marker_is_an_error(self):
        s = Strategy({"AAPL": 0.99}, asset_class=0)
        r = validate(s, self.u, max_single_name=1.0)
        self.assertFalse(r.ok)
        self.assertTrue(any("mandatory" in e for e in r.errors))

    def test_explicit_cash_key_warns(self):
        s = Strategy({"AAPL": 0.79, "": 0.20})
        r = validate(s, self.u, max_single_name=1.0)
        self.assertTrue(any("IGNORED" in w for w in r.warnings))


class TestPublisher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = load_universe()

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.strat = Path(self.tmp.name)
        (self.strat / ".last-update").touch()
        self.pub = Publisher(self.strat, HOTKEY)
        names = [a.ticker for a in self.u.most_liquid(40)
                 if not self.u.is_cash(a.ticker)][:20]
        self.s = Strategy({t: 0.0495 for t in names})

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_writes_nothing(self):
        r = self.pub.publish(self.s, self.u, dry_run=True)
        self.assertFalse(r.written)
        self.assertFalse(self.pub.path.exists())
        self.assertTrue(r.text.startswith("{'_':1,"))

    def test_invalid_strategy_never_reaches_disk(self):
        bad = Strategy({"AAPL": 0.7, "MSFT": 0.5})
        with self.assertRaises(PublishError):
            self.pub.publish(bad, self.u)
        self.assertFalse(self.pub.path.exists(), "a rejected file must not be written")

    def test_write_is_atomic_and_leaves_no_temp_files(self):
        self.pub.publish(self.s, self.u, confirm_timeout=0.01, poll_interval=0.01)
        self.assertTrue(self.pub.path.exists())
        leftovers = [p for p in self.strat.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [], "temp files must be cleaned up")

    def test_written_file_is_what_upstream_will_accept(self):
        self.pub.publish(self.s, self.u, confirm_timeout=0.01, poll_interval=0.01)
        acc = predict_acceptance(self.pub.path.read_text(), self.u)
        self.assertTrue(acc.accepted)
        self.assertLessEqual(acc.effective_cash, 0.01 + 1e-9)

    def test_unconfirmed_when_last_update_does_not_advance(self):
        r = self.pub.publish(self.s, self.u, confirm_timeout=0.05, poll_interval=0.01)
        self.assertTrue(r.written)
        self.assertFalse(r.confirmed)
        self.assertIn("NOT CONFIRMED", r.note)

    def test_confirmed_when_the_miner_touches_last_update(self):
        def miner():
            time.sleep(0.05)
            os.utime(self.strat / ".last-update", None)

        threading.Thread(target=miner, daemon=True).start()
        r = self.pub.publish(self.s, self.u, confirm_timeout=5.0, poll_interval=0.01)
        self.assertTrue(r.confirmed)
        self.assertTrue(r.submitted)

    def test_concurrent_publishers_cannot_race(self):
        import fcntl

        self.pub.lock_path.touch()
        holder = open(self.pub.lock_path, "r+")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        try:
            with self.assertRaises(PublishError) as ctx:
                self.pub.publish(self.s, self.u, confirm_timeout=0.01)
            self.assertIn("another publisher", str(ctx.exception))
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

    def test_mtime_strictly_advances(self):
        self.pub.publish(self.s, self.u, confirm_timeout=0.01, poll_interval=0.01)
        first = self.pub.path.stat().st_mtime
        self.pub.publish(self.s, self.u, confirm_timeout=0.01, poll_interval=0.01)
        self.assertGreater(self.pub.path.stat().st_mtime, first)

    def test_touch_refreshes_without_changing_weights(self):
        self.pub.publish(self.s, self.u, confirm_timeout=0.01, poll_interval=0.01)
        before = self.pub.path.read_text()
        time.sleep(0.01)
        self.pub.touch()
        self.assertEqual(self.pub.path.read_text(), before)

    def test_touch_refuses_when_there_is_nothing_to_refresh(self):
        with self.assertRaises(PublishError):
            self.pub.touch()

    def test_hotkey_path_traversal_is_refused(self):
        with self.assertRaises(PublishError):
            Publisher(self.strat, "../../etc/passwd")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestUniverseColumnParsing(unittest.TestCase):
    """The /assets table is FIXED-WIDTH. Two parsers that look correct are not.

    A plain .split() merges `name` into `sector` (552 unique "sectors" for 555 tickers,
    so a max_sector cap groups nothing). Splitting on runs of spaces fixes that but
    silently drops the rows whose long name butts against a neighbouring column.
    """

    @classmethod
    def setUpClass(cls):
        cls.u = load_universe()

    def test_no_rows_are_dropped(self):
        self.assertEqual(len(self.u), 555)

    def test_name_and_sector_are_separated(self):
        self.assertEqual(self.u.assets["NVDA"].name, "NVIDIA Corporation")
        self.assertEqual(self.u.assets["NVDA"].sector, "Semiconductors")

    def test_rows_whose_name_touches_the_next_column_survive(self):
        for tk, sector in (("DIA", "Equity"),
                           ("HLN", "Drug Manufacturers - Specialty & Generic"),
                           ("TAK", "Drug Manufacturers - Specialty & Generic"),
                           ("TEVA", "Drug Manufacturers - Specialty & Generic"),
                           ("ZTS", "Drug Manufacturers - Specialty & Generic")):
            self.assertIn(tk, self.u.assets, tk)
            self.assertEqual(self.u.assets[tk].sector, sector, tk)

    def test_sectors_actually_group(self):
        """Otherwise a max_sector constraint is silently inert."""
        groups = self.u.by_sector()
        self.assertLess(len(groups), 150, "sectors are not grouping - name leaked in")
        self.assertGreaterEqual(max(len(v) for v in groups.values()), 20)

    def test_a_header_change_raises_rather_than_emptying_the_universe(self):
        with self.assertRaises(ValueError):
            parse_assets("asset ratio: [0.5, 0.5]\ncash ETFs: ['SGOV']\nno header here\n")
