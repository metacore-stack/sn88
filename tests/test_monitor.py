"""Stage 6 tests: monitoring, alerting, rollback, and fault injection.

The fault-injection class walks every row of the guide's §21 failure matrix. The point is
not that each alert fires, but that each one demands the RIGHT ACTION - several of these
failures look identical on a dashboard and call for opposite responses.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica.monitor import (            # noqa: E402
    Action,
    AlertRouter,
    CycleMetrics,
    Severity,
    check_cycle,
    detect_rule_change,
    explain_score_change,
)
from sn88_replica.registry import (           # noqa: E402
    Artifact,
    ArtifactKind,
    PromotionCriteria,
    PromotionError,
    Registry,
)
from sn88_replica.universe import Universe, parse_assets   # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def universes():
    """The two real daily snapshots, if both exist."""
    hits = sorted(glob.glob(str(FIXTURES / "*-assets.txt")))
    if len(hits) < 2:
        raise unittest.SkipTest("need two archived /assets snapshots")
    return (parse_assets(Path(hits[-2]).read_text(errors="replace")),
            parse_assets(Path(hits[-1]).read_text(errors="replace")))


class TestRuleChangeDetection(unittest.TestCase):
    def test_two_real_consecutive_snapshots(self):
        before, after = universes()
        diff = detect_rule_change(before_universe=before, after_universe=after)
        print(f"\n  real snapshot diff: changed={diff.changed} "
              f"+{len(diff.universe_added)} -{len(diff.universe_removed)} tickers, "
              f"ratio={diff.ratio}, cash={'changed' if diff.cash_tickers else 'same'}")
        for a in diff.alerts():
            print(f"    {a}")
        self.assertIsNotNone(diff)

    def test_a_constant_change_freezes_selection(self):
        before, after = universes()
        diff = detect_rule_change(
            before_universe=before, after_universe=after,
            before_constants={"CLIP_OUTLIERS": 2, "WIN_SIZE_STK": 50},
            after_constants={"CLIP_OUTLIERS": 3, "WIN_SIZE_STK": 50})
        self.assertTrue(diff.changed)
        alerts = diff.alerts()
        self.assertTrue(any(a.key == "constants_changed" for a in alerts))
        self.assertEqual(AlertRouter().extend(alerts).action(), Action.FREEZE)

    def test_a_ratio_change_that_introduces_burn_is_called_out(self):
        before, after = universes()
        after.asset_ratio = (0.25, 0.25)
        diff = detect_rule_change(before_universe=before, after_universe=after)
        msg = next(a.message for a in diff.alerts() if a.key == "ratio_changed")
        self.assertIn("burned at UID 0", msg)
        self.assertIn("50.0%", msg)

    def test_a_delisted_holding_freezes(self):
        before, after = universes()
        victim = sorted(after.assets)[0]
        after.assets.pop(victim)
        diff = detect_rule_change(before_universe=before, after_universe=after,
                                  holdings=[victim])
        self.assertIn(victim, diff.universe_removed)
        self.assertEqual(AlertRouter().extend(diff.alerts()).action(), Action.FREEZE)

    def test_a_delisting_outside_your_book_does_not_freeze(self):
        before, after = universes()
        victim = sorted(after.assets)[0]
        after.assets.pop(victim)
        diff = detect_rule_change(before_universe=before, after_universe=after,
                                  holdings=["AAPL"])
        self.assertEqual(diff.universe_removed, frozenset())

    def test_cash_list_change_freezes(self):
        before, after = universes()
        after.cash_tickers = after.cash_tickers | {"TLT"}
        diff = detect_rule_change(before_universe=before, after_universe=after)
        alerts = diff.alerts()
        self.assertTrue(any(a.key == "cash_list_changed" for a in alerts))
        self.assertEqual(AlertRouter().extend(alerts).action(), Action.FREEZE)

    def test_no_change_is_quiet(self):
        before, _ = universes()
        diff = detect_rule_change(before_universe=before, after_universe=before)
        self.assertFalse(diff.changed)
        self.assertEqual(diff.alerts(), [])


class TestFailureMatrix(unittest.TestCase):
    """Guide §21, row by row. Each must demand the right ACTION, not merely fire."""

    def m(self, **kw):
        return CycleMetrics(session=date(2026, 9, 2), window="MOC", **kw)

    def action_for(self, **kw) -> Action:
        return AlertRouter().extend(check_cycle(self.m(**kw))).action()

    def test_healthy_cycle_is_silent(self):
        alerts = check_cycle(self.m(gross=0.995, cash=0.005, headroom=0.12,
                                    turnover=0.03, last_active_days=0,
                                    min_field_distance=0.28, replica_drift=1e-12))
        self.assertEqual(alerts, [])

    def test_gross_over_one_goes_to_safe_mode(self):
        """Upstream discards it silently - trading on is worse than stopping."""
        self.assertEqual(self.action_for(gross=1.0000001), Action.SAFE_MODE)

    def test_below_inception_goes_to_safe_mode(self):
        self.assertEqual(self.action_for(headroom=-0.004), Action.SAFE_MODE)

    def test_replica_drift_freezes_selection(self):
        """Every projected score is against a function we no longer reproduce."""
        self.assertEqual(self.action_for(replica_drift=1e-4), Action.FREEZE)

    def test_score_divergence_goes_to_safe_mode(self):
        self.assertEqual(self.action_for(projected_score=10.0, realised_score=4.0),
                         Action.SAFE_MODE)

    def test_inactivity_pages_but_keeps_trading(self):
        """Refreshing is the fix; stopping would make it worse."""
        alerts = check_cycle(self.m(last_active_days=4))
        self.assertEqual(alerts[0].severity, Severity.PAGE)
        self.assertEqual(AlertRouter().extend(alerts).action(), Action.ALERT)

    def test_cash_band_warns(self):
        self.assertEqual(self.action_for(cash=0.10), Action.ALERT)

    def test_under_invested_warns(self):
        self.assertEqual(self.action_for(gross=0.90), Action.ALERT)

    def test_turnover_warns(self):
        self.assertEqual(self.action_for(turnover=0.40), Action.ALERT)

    def test_floor_warn_below_three_percent(self):
        alerts = check_cycle(self.m(headroom=0.01))
        self.assertTrue(any(a.key == "floor_warn" for a in alerts))

    def test_field_distance_warns_about_the_grief_vector(self):
        alerts = check_cycle(self.m(min_field_distance=0.005))
        self.assertIn("LATER submission", alerts[0].message)

    def test_unconfirmed_submission_pages_but_does_not_rewrite(self):
        alerts = check_cycle(self.m(published=True, submission_confirmed=False,
                                    seconds_since_submission=600))
        self.assertIn("do NOT write again", alerts[0].message)

    def test_unconfirmed_within_the_grace_window_is_quiet(self):
        alerts = check_cycle(self.m(published=True, submission_confirmed=False,
                                    seconds_since_submission=30))
        self.assertEqual(alerts, [])

    def test_freeze_beats_alert_and_safe_mode_beats_freeze(self):
        r = AlertRouter()
        r.extend(check_cycle(self.m(cash=0.10)))                  # ALERT
        self.assertEqual(r.action(), Action.ALERT)
        r.extend(check_cycle(self.m(replica_drift=1e-3)))         # FREEZE
        self.assertEqual(r.action(), Action.FREEZE)
        r.extend(check_cycle(self.m(headroom=-0.01)))             # SAFE_MODE
        self.assertEqual(r.action(), Action.SAFE_MODE)
        self.assertFalse(r.should_trade())

    def test_alerts_are_logged_as_jsonl_with_the_constants_fingerprint(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.jsonl"
            AlertRouter(path).extend(check_cycle(self.m(cash=0.2)))
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertIn("constants_fingerprint", rows[0])
            self.assertEqual(rows[0]["key"], "cash_band")


class TestAttribution(unittest.TestCase):
    """A cash-multiplier drop and a drawdown look identical on the dashboard."""

    def test_the_dominant_component_is_identified(self):
        before = {"raw": 100.0, "cash": 1.00, "inactivity": 1.00, "dedupe": 1.0}
        after = {"raw": 98.0, "cash": 0.50, "inactivity": 1.00, "dedupe": 1.0}
        out = explain_score_change(before, after)
        self.assertEqual(out[0][0], "cash")
        self.assertAlmostEqual(out[0][1], 0.5, places=9)
        self.assertEqual(out[0][2], "down")

    def test_unchanged_components_are_omitted(self):
        out = explain_score_change({"raw": 10.0, "cash": 1.0}, {"raw": 12.0, "cash": 1.0})
        self.assertEqual([k for k, _r, _d in out], ["raw"])


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.r = Registry(self.tmp.name)
        self.c = PromotionCriteria(min_live_sessions=20, min_score_improvement=0.05)

    def tearDown(self):
        self.tmp.cleanup()

    def reg(self, aid, kind=ArtifactKind.MODEL, fv="fv1", criteria=None):
        return self.r.register(artifact_id=aid, kind=kind, content=aid,
                               feature_version=fv, criteria=criteria or self.c)

    def passing(self, **over):
        args = dict(live_sessions=25, score_vs_champion=0.20, deflated_sharpe_p=0.01,
                    baselines_beaten=["equal_weight", "inverse_volatility"],
                    current_feature_version="fv1")
        args.update(over)
        return args

    def test_register_then_promote(self):
        self.reg("m1")
        self.r.promote("m1", self.c, **self.passing())
        self.assertEqual(self.r.champion(ArtifactKind.MODEL).artifact_id, "m1")

    def test_registering_is_not_promoting(self):
        self.reg("m1")
        self.assertIsNone(self.r.champion(ArtifactKind.MODEL))

    def test_duplicate_id_refused(self):
        self.reg("m1")
        with self.assertRaises(PromotionError):
            self.reg("m1")

    def test_criteria_cannot_be_changed_after_registration(self):
        """Otherwise the bar moves to wherever the result landed."""
        self.reg("m1")
        relaxed = PromotionCriteria(min_live_sessions=1, min_score_improvement=0.0)
        with self.assertRaises(PromotionError) as ctx:
            self.r.promote("m1", relaxed, **self.passing())
        self.assertIn("fixed in advance", str(ctx.exception))

    def test_feature_version_mismatch_blocks_promotion(self):
        self.reg("m1", fv="fv1")
        with self.assertRaises(PromotionError) as ctx:
            self.r.promote("m1", self.c, **self.passing(current_feature_version="fv2"))
        self.assertIn("feature version mismatch", str(ctx.exception))

    def test_every_unmet_criterion_is_reported(self):
        self.reg("m1")
        with self.assertRaises(PromotionError) as ctx:
            self.r.promote("m1", self.c, **self.passing(
                live_sessions=3, score_vs_champion=0.0, deflated_sharpe_p=0.5,
                baselines_beaten=[]))
        msg = str(ctx.exception)
        for expect in ("live sessions", "score improvement", "deflated Sharpe",
                       "did not beat baselines"):
            self.assertIn(expect, msg)

    def test_risk_policy_never_auto_promotes(self):
        """Model weights may auto-promote. Risk limits may not."""
        self.reg("risk1", kind=ArtifactKind.RISK_POLICY)
        with self.assertRaises(PromotionError) as ctx:
            self.r.promote("risk1", self.c, **self.passing())
        self.assertIn("manual approval", str(ctx.exception))
        self.r.promote("risk1", self.c, **self.passing(), approver="operator")
        self.assertEqual(self.r.champion(ArtifactKind.RISK_POLICY).artifact_id, "risk1")

    def test_a_failed_promotion_leaves_the_champion_untouched(self):
        self.reg("m1"); self.r.promote("m1", self.c, **self.passing())
        self.reg("m2")
        with self.assertRaises(PromotionError):
            self.r.promote("m2", self.c, **self.passing(live_sessions=1))
        self.assertEqual(self.r.champion(ArtifactKind.MODEL).artifact_id, "m1")

    def test_rollback_is_one_call(self):
        self.reg("m1"); self.r.promote("m1", self.c, **self.passing())
        self.reg("m2"); self.r.promote("m2", self.c, **self.passing())
        self.assertEqual(self.r.champion(ArtifactKind.MODEL).artifact_id, "m2")
        back = self.r.rollback(ArtifactKind.MODEL)
        self.assertEqual(back.artifact_id, "m1")
        self.assertEqual(self.r.champion(ArtifactKind.MODEL).artifact_id, "m1")

    def test_rollback_to_a_named_target(self):
        for a in ("m1", "m2", "m3"):
            self.reg(a); self.r.promote(a, self.c, **self.passing())
        self.assertEqual(self.r.rollback(ArtifactKind.MODEL, to="m1").artifact_id, "m1")

    def test_cannot_roll_back_to_something_never_promoted(self):
        self.reg("m1"); self.r.promote("m1", self.c, **self.passing())
        self.reg("m2")
        with self.assertRaises(PromotionError):
            self.r.rollback(ArtifactKind.MODEL, to="m2")

    def test_cannot_roll_back_with_no_history(self):
        self.reg("m1"); self.r.promote("m1", self.c, **self.passing())
        with self.assertRaises(PromotionError):
            self.r.rollback(ArtifactKind.MODEL)

    def test_the_log_is_append_only(self):
        self.reg("m1"); self.r.promote("m1", self.c, **self.passing())
        self.reg("m2"); self.r.promote("m2", self.c, **self.passing())
        self.r.rollback(ArtifactKind.MODEL)
        ids = [a.artifact_id for a in self.r.history(ArtifactKind.MODEL)]
        self.assertGreaterEqual(len(ids), 5, "every state transition must be recorded")
        self.assertIn("m1", ids)

    def test_content_hash_detects_a_changed_artifact(self):
        a = self.reg("m1")
        b = self.r.register(artifact_id="m1b", kind=ArtifactKind.MODEL,
                            content="different", feature_version="fv1", criteria=self.c)
        self.assertNotEqual(a.content_hash, b.content_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
