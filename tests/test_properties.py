"""Property tests for the scoring algebra.

These encode the facts from guide §2 that the rest of the system reasons from. If one
of them ever fails, a scoring constant changed upstream and candidate selection must
freeze until the replica is re-validated (guide §21).
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.constants import CLIP_DEFAULT, RISK_INIT_STK   # noqa: E402
from sn88_replica.scoring import (                                # noqa: E402
    drawdown,
    kelly,
    score_terms,
    window_and_clip,
)


def equity_from_returns(pct, init=10_000_000.0):
    out, eq = [], init
    for r in pct:
        eq *= 1 + r / 100
        out.append(eq)
    return init, out


class TestScaleIdentity(unittest.TestCase):
    """Guide §2.2, corrected.

    Through a REAL compounded equity curve, only ``odds`` is exactly scale-invariant.
    ``lsr`` uses DOLLAR pnl, so it falls with scale; ``mar`` rises (the compounding
    wedge); ``daily`` is sublinear. Net: score has DECREASING returns to scale.

    An earlier revision computed ``lsr`` from ``pnl%`` instead of dollar pnl and
    concluded it was invariant. It is not - it moves 21% across a 32x range.
    """

    KS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    def setUp(self):
        rng = random.Random(11)
        self.path = [rng.gauss(0.20, 1.10) for _ in range(50)]

    def _terms(self, k):
        init, closes = equity_from_returns([x * k for x in self.path])
        return score_terms(init, closes)

    def test_odds_is_exactly_invariant(self):
        """odds derives from pnl% ratios only, so scale cancels exactly."""
        base = self._terms(1.0).odds
        for k in self.KS:
            self.assertAlmostEqual(self._terms(k).odds, base, places=9,
                                   msg=f"odds moved at k={k}")

    def test_risk_is_exactly_linear_in_scale(self):
        """drawdown() runs on the arithmetic cumsum, so it scales exactly."""
        base = self._terms(1.0).risk
        for k in self.KS:
            self.assertAlmostEqual(self._terms(k).risk / k, base, places=6)

    def test_mar_rises_with_scale_compounding_wedge(self):
        """gain is compounded, risk is arithmetic -> the ratio drifts up."""
        prev = None
        for k in self.KS:
            mar = self._terms(k).mar
            if prev is not None:
                self.assertGreater(mar, prev, f"mar should rise at k={k}")
            prev = mar

    def test_lsr_falls_with_scale_because_it_uses_dollar_pnl(self):
        prev = None
        for k in self.KS:
            lsr = self._terms(k).lsr
            if prev is not None:
                self.assertLess(lsr, prev, f"lsr should fall at k={k}")
            prev = lsr
        spread = self._terms(0.25).lsr - self._terms(8.0).lsr
        self.assertGreater(spread, 0.01, "lsr's scale dependence is material, not noise")

    def test_score_has_DECREASING_returns_to_scale(self):
        """The headline correction: score/k falls as k rises. Scale is not free."""
        base = self._terms(1.0).score
        ratios = {k: self._terms(k).score / base / k for k in self.KS}
        self.assertGreater(ratios[0.25], 1.0)
        self.assertLess(ratios[8.0], 0.8)
        ordered = [ratios[k] for k in self.KS]
        self.assertEqual(ordered, sorted(ordered, reverse=True),
                         "score/k must decrease monotonically in k")


class TestOddsIdentity(unittest.TestCase):
    """Guide §2.5: kelly(p, pavg/lavg) == mean/pavg, so odds = 50*(1 + mean/pavg)."""

    def test_identity_holds(self):
        for seed in range(25):
            rng = random.Random(seed)
            pct = [rng.gauss(0.2, 1.2) for _ in range(50)]
            pos = [x for x in pct if x > 0]
            neg = [x for x in pct if x < 0]
            if not pos or not neg:
                continue
            prob = len(pos) / len(pct)
            pavg = sum(pos) / len(pos)
            lavg = -sum(neg) / len(neg)
            via_kelly = 50 + kelly(prob, pavg / lavg) / 2 * 100
            via_mean = 50 * (1 + (sum(pct) / len(pct)) / pavg)
            self.assertAlmostEqual(via_kelly, via_mean, places=9, msg=f"seed {seed}")

    def test_win_rate_alone_does_not_move_odds(self):
        """Same mean and same average up-day, very different hit rates -> same odds."""
        a = [1.0] * 30 + [-1.0] * 20          # 60% win rate
        b = [1.0] * 40 + [-3.0] * 10          # 80% win rate, same mean (+0.10), same pavg
        self.assertAlmostEqual(sum(a) / len(a), sum(b) / len(b), places=9)
        for series in (a, b):
            pos = [x for x in series if x > 0]
            neg = [x for x in series if x < 0]
            prob = len(pos) / len(series)
            pavg, lavg = sum(pos) / len(pos), -sum(neg) / len(neg)
            series.append(50 + kelly(prob, pavg / lavg) / 2 * 100)
        self.assertAlmostEqual(a[-1], b[-1], places=9)


class TestClipBranches(unittest.TestCase):
    """Guide §2.7. All three branches, plus the equity rebase."""

    def test_no_positive_sessions(self):
        init, closes = equity_from_returns([-0.5] * 10)
        s, pnl, pct, clip = window_and_clip(init, closes)
        self.assertEqual(clip, 0.0)
        self.assertEqual(list(s), list(closes))

    def test_single_positive_session_cut_to_one_percent(self):
        pct_in = [-0.1] * 49 + [5.98]
        init, closes = equity_from_returns(pct_in)
        _s, _pnl, pct, clip = window_and_clip(init, closes)
        self.assertAlmostEqual(clip, CLIP_DEFAULT, places=9)
        self.assertAlmostEqual(max(pct), CLIP_DEFAULT, places=6)

    def test_top_three_levelled_to_third_best(self):
        pct_in = [0.1] * 47 + [3.0, 2.0, 1.5]
        init, closes = equity_from_returns(pct_in)
        _s, _pnl, pct, clip = window_and_clip(init, closes)
        self.assertGreater(clip, 0.0)
        self.assertLessEqual(max(pct), clip + 1e-9)
        levelled = sum(1 for x in pct if abs(x - clip) < 1e-9)
        self.assertGreaterEqual(levelled, 3)

    def test_rebase_reduces_total_gain(self):
        """Clipping cuts total return, not just the daily statistics."""
        rng = random.Random(4)
        pct_in = [rng.gauss(0.2, 1.5) for _ in range(50)]
        init, closes = equity_from_returns(pct_in)
        s, _pnl, _pct, _clip = window_and_clip(init, closes)
        self.assertLess(s[-1], closes[-1], "rebased curve must end below the raw curve")

    def test_clip_order_is_descending_pnl_not_index(self):
        """Index order gives up to 12% error on live rows - lock the order in."""
        pct_in = [0.05] * 47 + [4.0, 0.1, 2.5]
        init, closes = equity_from_returns(pct_in)
        s_desc, _p, _q, clip = window_and_clip(init, closes)

        # replicate the WRONG (index-ascending) order for comparison
        n = len(closes)
        s = list(closes)
        pos = [i for i in range(n) if _pct_of(closes, init, i) > 0]
        ii = sorted(pos, key=lambda i: (_pct_of(closes, init, i), i))[::-1][:3]
        for i in sorted(ii):
            opening = s[i - 1] if i else init
            delta = s[i] - opening * (1 + clip / 100)
            for j in range(i, n):
                s[j] -= delta
        self.assertNotAlmostEqual(s_desc[-1], s[-1], places=3,
                                  msg="order-dependence must be preserved")


def _pct_of(closes, init, i):
    prev = closes[i - 1] if i else init
    return (closes[i] - prev) / prev * 100


class TestMarFloor(unittest.TestCase):
    """Guide §2.6. floor = RISK_INIT/sqrt(days); harsher for new miners."""

    def test_floor_value(self):
        self.assertAlmostEqual(RISK_INIT_STK / math.sqrt(50), 0.7071, places=4)
        self.assertAlmostEqual(RISK_INIT_STK / math.sqrt(5), 2.2361, places=4)

    def test_floor_caps_mar_for_a_flawless_book(self):
        """No drawdown at all -> mar = gain / floor = 1.414 * gain at a full window."""
        init, closes = equity_from_returns([0.10] * 50)
        t = score_terms(init, closes)
        self.assertEqual(t.risk, 0.0)
        self.assertAlmostEqual(t.mar, t.gain / (RISK_INIT_STK / math.sqrt(50)), places=6)


class TestSaturation(unittest.TestCase):
    """Guide §2.5: lsr and odds hit their maxima when there are no losing sessions."""

    def test_no_losing_sessions_saturates_both(self):
        init, closes = equity_from_returns([0.10] * 50)
        t = score_terms(init, closes)
        self.assertAlmostEqual(t.lsr, 1.0, places=9)
        self.assertAlmostEqual(t.odds, 100.0, places=9)

    def test_single_jackpot_session_collapses_odds(self):
        init, closes = equity_from_returns([0.0] * 49 + [12.5])
        t = score_terms(init, closes)
        self.assertLess(t.odds, 5.0, "one winning session in 50 -> odds ~ 2")

    def test_negative_window_scores_zero(self):
        init, closes = equity_from_returns([-0.2] * 50)
        self.assertEqual(score_terms(init, closes).score, 0.0)


class TestDrawdown(unittest.TestCase):
    def test_arithmetic_cumsum_not_equity(self):
        self.assertAlmostEqual(drawdown([1.0, -2.0, 1.0]), 2.0, places=9)
        self.assertAlmostEqual(drawdown([1.0, 1.0, 1.0]), 0.0, places=9)
        # peak starts at 0, so an immediate loss counts in full
        self.assertAlmostEqual(drawdown([-3.0, 1.0]), 3.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
