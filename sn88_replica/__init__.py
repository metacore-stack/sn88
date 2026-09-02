"""An exact replica of the SN88 scoring pipeline, in pure stdlib Python.

Stage 1 of the build order: everything else in the system is downstream of this.

    from sn88_replica import api, pipeline
    days  = api.parse_days(api.fetch("days"))
    field = pipeline.score_field(api.fetch("pnl"), days, api.fetch("dist"), api.fetch("ratio"))

Acceptance: reproduces the live /score table to <1e-6 relative on every row
(measured max 6.8e-8, median 1.3e-12). See tests/test_golden.py.
"""

from .constants import fingerprint, snapshot
from .scoring import ScoreTerms, raw_score, score_terms, window_and_clip, drawdown, kelly
from .strategy import Strategy, ValidationError, predict_acceptance, sanitize, serialize, validate
from .universe import Universe, parse_assets
from .publisher import Publisher, PublishError, PublishResult
from .calendar import SessionCalendar, Session, holidays, early_closes
from .execution import Submission, FillDecision, resolve_fill, turnover_cost
from .replay import FeatureSnapshot, ReplayEngine, ReplayStep, assert_no_lookahead
from .pit import Observation, PointInTimeStore, PITSnapshot
from .baselines import BASELINES, covariance_matrix, ewma_covariance, shrink_covariance
from .universe import Universe, parse_assets as _pa
from .governor import (
    GovernorConfig, RecoveryPolicy, RecoveryStance, RiskGovernor, RiskState, Verdict, Veto,
)
from .monitor import (
    Action, Alert, AlertRouter, CycleMetrics, Severity, check_cycle, detect_rule_change,
    explain_score_change,
)
from .registry import Artifact, ArtifactKind, PromotionCriteria, PromotionError, Registry
from .signals import (
    EvalResult, Panel, Signal, SignalView, equal_weight_top, evaluate,
    inverse_vol_top, load_panel, null_signal,
)
from .market import (
    CorporateAction, DailyBar, IntradayBar, MarketFeed, Track,
)
from .shadow import (
    GateStatus, ReadinessCriteria, ReadinessReport, ShadowRecord, ShadowRun,
)
from .modes import (
    Candidate,
    ForwardPaths,
    LeakageError,
    ProjectionResult,
    RealisedHistory,
    project,
    reconcile,
)

__all__ = [
    "fingerprint", "snapshot",
    "ScoreTerms", "raw_score", "score_terms", "window_and_clip", "drawdown", "kelly",
    "Candidate", "ForwardPaths", "LeakageError", "ProjectionResult", "RealisedHistory",
    "project", "reconcile",
    "Strategy", "ValidationError", "predict_acceptance", "sanitize", "serialize", "validate",
    "Universe", "parse_assets",
    "Publisher", "PublishError", "PublishResult",
    "SessionCalendar", "Session", "holidays", "early_closes",
    "Submission", "FillDecision", "resolve_fill", "turnover_cost",
    "FeatureSnapshot", "ReplayEngine", "ReplayStep", "assert_no_lookahead",
    "Observation", "PointInTimeStore", "PITSnapshot",
    "BASELINES", "covariance_matrix", "ewma_covariance", "shrink_covariance",
    "GovernorConfig", "RecoveryPolicy", "RecoveryStance", "RiskGovernor", "RiskState",
    "Verdict", "Veto",
    "Action", "Alert", "AlertRouter", "CycleMetrics", "Severity", "check_cycle",
    "detect_rule_change", "explain_score_change",
    "Artifact", "ArtifactKind", "PromotionCriteria", "PromotionError", "Registry",
    "GateStatus", "ReadinessCriteria", "ReadinessReport", "ShadowRecord", "ShadowRun",
    "CorporateAction", "DailyBar", "IntradayBar", "MarketFeed", "Track",
    "EvalResult", "Panel", "Signal", "SignalView", "equal_weight_top", "evaluate",
    "inverse_vol_top", "load_panel", "null_signal",
]
__version__ = "0.9.0"
