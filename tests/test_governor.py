"""Stage 5 tests: the risk governor and the recovery policy.

The governor is final authority, so the tests are mostly about what it REFUSES - and in
particular that hard limits survive every state, including a breach.
"""

from __future__ import annotations

import glob
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api                                        # noqa: E402
from sn88_replica.governor import (                                 # noqa: E402
    GovernorConfig,
    RecoveryPolicy,
    RiskGovernor,
    RiskState,
)
from sn88_replica.strategy import Strategy                          # noqa: E402
from sn88_replica.universe import parse_assets                      # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
INCEPTION = 10_000_000.0


def load_universe():
    hits = sorted(glob.glob(os.path.join(FIXTURES, "*-assets.txt")))
    if hits:
        return parse_assets(Path(hits[-1]).read_text(encoding="utf-8", errors="replace"))
    if os.environ.get("SN88_OFFLINE"):
        raise unittest.SkipTest("no /assets fixture and SN88_OFFLINE is set")
    return parse_assets(api.fetch("assets"))


class GovernorCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = load_universe()
        # 20 liquid, non-cash names spread across sectors, 0.0495 each = 0.99 gross
        seen: dict[str, int] = {}
        picks: list[str] = []
        for a in sorted(cls.u.assets.values(), key=lambda a: -a.dollar_volume):
            if cls.u.is_cash(a.ticker) or a.ticker in ("TQQQ",):
                continue
            if seen.get(a.sector, 0) >= 3:
                continue
            seen[a.sector] = seen.get(a.sector, 0) + 1
            picks.append(a.ticker)
            if len(picks) == 20:
                break
        cls.names = picks

    def book(self, **overrides) -> Strategy:
        w = {t: 0.0495 for t in self.names}
        w.update(overrides)
        return Strategy(w)

    def gov(self, **cfg) -> RiskGovernor:
        return RiskGovernor(GovernorConfig(inception_equity=INCEPTION, **cfg))


class TestStates(GovernorCase):
    def test_headroom_and_state_thresholds(self):
        g = self.gov()
        self.assertEqual(g.state(INCEPTION * 1.10), RiskState.NORMAL)
        self.assertEqual(g.state(INCEPTION * 1.02), RiskState.WARN)
        self.assertEqual(g.state(INCEPTION * 1.005), RiskState.CRITICAL)
        self.assertEqual(g.state(INCEPTION * 0.999), RiskState.BREACHED)
        self.assertAlmostEqual(g.headroom(INCEPTION * 1.05), 0.05, places=9)


class TestHardConstraints(GovernorCase):
    def test_a_good_book_is_approved(self):
        v = self.gov().review(self.book(), self.u, current_equity=INCEPTION * 1.10)
        self.assertTrue(v.approved, v.reason())
        self.assertEqual(v.state, RiskState.NORMAL)

    def test_gross_above_one_is_vetoed(self):
        s = Strategy({t: 0.06 for t in self.names})      # 20 x 0.06 = 1.20
        v = self.gov().review(s, self.u, current_equity=INCEPTION * 1.1)
        self.assertFalse(v.approved)
        self.assertIn("gross_cap", v.reason())

    def test_underinvested_is_vetoed(self):
        s = Strategy({t: 0.04 for t in self.names})      # 0.80 gross
        v = self.gov().review(s, self.u, current_equity=INCEPTION * 1.1)
        self.assertIn("min_gross", v.reason())

    def test_shorts_vetoed(self):
        w = {t: 0.0495 for t in self.names}
        w[self.names[0]] = -0.0495
        v = self.gov().review(Strategy(w), self.u, current_equity=INCEPTION * 1.1)
        self.assertIn("no_shorts", v.reason())

    def test_unknown_ticker_vetoed(self):
        v = self.gov().review(self.book(NOTREAL=0.0), self.u, current_equity=INCEPTION * 1.1)
        self.assertIn("unknown_ticker", v.reason())

    def test_cash_classified_holding_vetoed(self):
        w = {t: 0.0495 for t in self.names[:19]}
        w["SGOV"] = 0.0495
        v = self.gov().review(Strategy(w), self.u, current_equity=INCEPTION * 1.1)
        self.assertIn("cash_band", v.reason())

    def test_single_name_cap(self):
        w = {t: 0.0295 for t in self.names}
        w[self.names[0]] = 0.40
        v = self.gov().review(Strategy(w), self.u, current_equity=INCEPTION * 1.1)
        self.assertIn("single_name", v.reason())

    def test_sector_cap_actually_binds(self):
        """Only possible because /assets is parsed by column offset - see stage 4."""
        by_sector = self.u.by_sector()
        big = max(by_sector, key=lambda s: len(by_sector[s]))
        names = [t for t in by_sector[big] if not self.u.is_cash(t)][:20]
        if len(names) < 15:
            self.skipTest("not enough same-sector names in the fixture")
        s = Strategy({t: 0.99 / len(names) for t in names})
        v = self.gov().review(s, self.u, current_equity=INCEPTION * 1.1)
        self.assertIn("sector_cap", v.reason())

    def test_holdings_bounds(self):
        s = Strategy({self.names[0]: 0.99})
        v = self.gov().review(s, self.u, current_equity=INCEPTION * 1.1,
                              )
        self.assertIn("holdings", v.reason())

    def test_turnover_cap(self):
        prev = {t: 0.0495 for t in self.names}
        fresh = {t: 0.0495 for t in self.names[10:] + self.names[:10]}
        other = [a.ticker for a in self.u.most_liquid(80)
                 if a.ticker not in self.names and not self.u.is_cash(a.ticker)][:20]
        if len(other) < 20:
            self.skipTest("fixture too small")
        v = self.gov().review(Strategy({t: 0.0495 for t in other}), self.u,
                              current_equity=INCEPTION * 1.1, previous_weights=prev)
        self.assertIn("turnover", v.reason())

    def test_breach_probability_veto(self):
        v = self.gov().review(self.book(), self.u, current_equity=INCEPTION * 1.1,
                              breach_probability=0.20)
        self.assertIn("breach_probability", v.reason())


class TestFloorBehaviour(GovernorCase):
    def test_leveraged_sleeve_forbidden_below_the_warn_line(self):
        w = {t: 0.0495 for t in self.names[:19]}
        w["TQQQ"] = 0.0495
        s = Strategy(w)
        ok = self.gov().review(s, self.u, current_equity=INCEPTION * 1.20)
        self.assertTrue(ok.approved, ok.reason())
        blocked = self.gov().review(s, self.u, current_equity=INCEPTION * 1.01)
        self.assertFalse(blocked.approved)
        self.assertIn("leveraged_below_floor", blocked.reason())

    def test_scale_up_blocked_below_the_warn_line(self):
        v = self.gov().review(self.book(), self.u, current_equity=INCEPTION * 1.01,
                              realised_vol=0.25, previous_vol=0.18)
        self.assertIn("scale_up_below_floor", v.reason())

    def test_hard_limits_still_apply_when_breached(self):
        """A breach is not a licence to abandon gross or concentration limits."""
        w = {t: 0.06 for t in self.names}                # gross 1.20
        v = self.gov().review(Strategy(w), self.u, current_equity=INCEPTION * 0.95)
        self.assertEqual(v.state, RiskState.BREACHED)
        self.assertFalse(v.approved)
        self.assertIn("gross_cap", v.reason())

    def test_breach_note_refuses_the_more_risk_shortcut(self):
        v = self.gov().review(self.book(), self.u, current_equity=INCEPTION * 0.95)
        self.assertEqual(v.state, RiskState.BREACHED)
        self.assertTrue(any("NOT automatically correct" in n for n in v.notes))

    def test_governor_never_mutates_the_candidate(self):
        s = self.book()
        before = dict(s.weights)
        self.gov().review(s, self.u, current_equity=INCEPTION * 0.5)
        self.assertEqual(dict(s.weights), before)

    def test_it_vetoes_and_never_proposes(self):
        import inspect

        src = inspect.getsource(RiskGovernor)
        self.assertNotIn("return Strategy", src, "the governor must not build portfolios")


class TestRecoveryPolicy(unittest.TestCase):
    """Below the floor, the right risk level is simulated - never assumed."""

    def setUp(self):
        self.p = RecoveryPolicy(horizon=60, paths=400)
        self.vols = [0.004, 0.008, 0.012, 0.020, 0.030]

    def test_it_returns_a_ranked_stance_per_candidate(self):
        out = self.p.evaluate(headroom=-0.03, daily_sharpe=0.06, candidate_vols=self.vols)
        self.assertEqual(len(out), len(self.vols))
        self.assertEqual([s.objective for s in out],
                         sorted([s.objective for s in out], reverse=True))
        for s in out:
            self.assertGreaterEqual(s.probability_of_recovery, 0.0)
            self.assertLessEqual(s.probability_of_recovery, 1.0)

    def test_deeper_underwater_needs_more_volatility_to_recover_at_all(self):
        shallow = {s.target_vol: s.probability_of_recovery for s in
                   self.p.evaluate(headroom=-0.005, daily_sharpe=0.05, candidate_vols=self.vols)}
        deep = {s.target_vol: s.probability_of_recovery for s in
                self.p.evaluate(headroom=-0.15, daily_sharpe=0.05, candidate_vols=self.vols)}
        self.assertGreater(shallow[0.004], deep[0.004])
        self.assertGreater(deep[0.030], deep[0.004],
                           "far under water, the lowest vol cannot climb back")

    def test_pruning_hazard_pushes_the_answer_toward_less_delay(self):
        no_prune = self.p.recommend(headroom=-0.05, daily_sharpe=0.05,
                                    candidate_vols=self.vols, prune_hazard_per_session=0.0)
        heavy = self.p.recommend(headroom=-0.05, daily_sharpe=0.05,
                                 candidate_vols=self.vols, prune_hazard_per_session=0.02)
        self.assertIsNotNone(no_prune.target_vol)
        self.assertGreater(heavy.probability_of_pruning, no_prune.probability_of_pruning)

    def test_the_answer_is_not_always_maximum_risk(self):
        """The whole point: 'below the floor, take more risk' is not a rule."""
        answers = set()
        for hr in (-0.002, -0.01, -0.05, -0.20):
            for sharpe in (0.02, 0.08, 0.15):
                answers.add(self.p.recommend(headroom=hr, daily_sharpe=sharpe,
                                             candidate_vols=self.vols).target_vol)
        self.assertGreater(len(answers), 1,
                           "if every scenario returns the same vol, this is a rule, "
                           "not a policy")
        print(f"\n  recovery policy picked {sorted(answers)} across scenarios "
              f"(candidates {self.vols})")

    def test_it_is_deterministic(self):
        a = self.p.recommend(headroom=-0.05, daily_sharpe=0.05, candidate_vols=self.vols)
        b = self.p.recommend(headroom=-0.05, daily_sharpe=0.05, candidate_vols=self.vols)
        self.assertEqual(a.target_vol, b.target_vol)
        self.assertAlmostEqual(a.objective, b.objective, places=12)

    def test_no_candidates_raises(self):
        with self.assertRaises(ValueError):
            self.p.recommend(headroom=-0.05, daily_sharpe=0.05, candidate_vols=[])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestClosestApproach(GovernorCase):
    """The gate is memoryless; the near misses are still the risk signal."""

    def test_closest_approach_is_reported_but_is_not_a_gate(self):
        g = self.gov()
        self.assertAlmostEqual(g.closest_approach(INCEPTION * 0.98), -0.02, places=9)
        v = g.review(self.book(), self.u, current_equity=INCEPTION * 1.20,
                     min_equity_since_inception=INCEPTION * 0.98)
        self.assertTrue(v.approved, "a past dip must not veto a currently-healthy book")
        self.assertTrue(any("has been BELOW inception before" in n for n in v.notes))

    def test_a_near_miss_is_surfaced(self):
        v = self.gov().review(self.book(), self.u, current_equity=INCEPTION * 1.20,
                              min_equity_since_inception=INCEPTION * 1.005)
        self.assertTrue(any("closest approach" in n for n in v.notes))

    def test_a_comfortable_history_is_quiet(self):
        v = self.gov().review(self.book(), self.u, current_equity=INCEPTION * 1.20,
                              min_equity_since_inception=INCEPTION * 1.15)
        self.assertEqual(v.notes, ())
