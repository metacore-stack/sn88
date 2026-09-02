#!/usr/bin/env python3
"""Run one shadow cycle, or report readiness to register.

Stage 7 of the build order. The full loop with the publisher disarmed: rule check,
snapshot, policy, governor, publish (dry run), record. Resumable - a real run spans the
10-20 calendar sessions the gate requires.

    python3 tools/shadow.py --cycle --window MOC     # one cycle, cron-able
    python3 tools/shadow.py --report                 # go / no-go

Exit codes:  0 ready  ·  1 not ready  ·  2 cycle recorded a fault
"""
import argparse
import glob
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api
from sn88_replica.calendar import SessionCalendar
from sn88_replica.governor import GovernorConfig, RiskGovernor
from sn88_replica.monitor import AlertRouter, detect_rule_change
from sn88_replica.replay import DEFAULT_DECISION_LATENCY
from sn88_replica.shadow import ReadinessCriteria, ShadowRecord, ShadowRun
from sn88_replica.strategy import Strategy, predict_acceptance, serialize, validate
from sn88_replica.universe import parse_assets

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "ops" / "shadow.jsonl"


def newest(suffix):
    hits = sorted(glob.glob(str(ROOT / "fixtures" / f"*-{suffix}")))
    return hits[-1] if hits else None


def baseline_book(u, n=20):
    """A conservative equal-weight book over liquid, non-cash, sector-spread names.

    This is what stage 8 registers with: something defensible you actually hold, not a
    model output. The point of the shadow run is the pipeline, not the signal.
    """
    seen, picks = {}, []
    for a in sorted(u.assets.values(), key=lambda a: -a.dollar_volume):
        if u.is_cash(a.ticker) or a.ticker == "TQQQ":
            continue
        if seen.get(a.sector, 0) >= 3:
            continue
        seen[a.sector] = seen.get(a.sector, 0) + 1
        picks.append(a.ticker)
        if len(picks) == n:
            break
    return Strategy({t: round(0.99 / len(picks), 8) for t in picks})


def cycle(args) -> int:
    cal = SessionCalendar()
    today = date.fromisoformat(args.date) if args.date else date.today()
    faults, vetoes = [], []

    if not cal.is_session(today):
        print(f"{today.isoformat()} is not a trading session - nothing to do "
              f"(but the inactivity clock still runs: touch the timestamp)")
        return 0

    session = cal.session(today)
    cutoff = session.cutoff(args.window)
    horizon = cutoff - DEFAULT_DECISION_LATENCY
    now = datetime.now(timezone.utc)
    cutoff_met = now < cutoff

    try:
        after = parse_assets(api.fetch("assets"))
    except Exception as exc:                                  # noqa: BLE001
        faults.append(f"assets_fetch:{type(exc).__name__}")
        print(f"FAULT fetching /assets: {exc}")
        return 2

    prev_path = newest("assets.txt")
    rule_ran = False
    if prev_path:
        before = parse_assets(Path(prev_path).read_text(errors="replace"))
        diff = detect_rule_change(before_universe=before, after_universe=after)
        rule_ran = True
        router = AlertRouter(ROOT / "ops" / "alerts.jsonl").extend(diff.alerts())
        for a in diff.alerts():
            print(f"  {a}")
        if not router.should_trade():
            faults.append(f"rules_changed:{router.action().value}")

    book = baseline_book(after)
    report = validate(book, after)
    gov = RiskGovernor(GovernorConfig())
    verdict = gov.review(book, after, current_equity=10_000_000.0)
    vetoes = [v.rule for v in verdict.vetoes]
    acc = predict_acceptance(serialize(book), after)

    published = report.ok and verdict.approved and cutoff_met and not faults

    rec = ShadowRecord(
        session=today.isoformat(),
        window=args.window,
        recorded_at=now.isoformat(),
        published=published,
        upstream_would_accept=acc.accepted and not acc.reclassified,
        governor_approved=verdict.approved,
        cutoff_met=cutoff_met,
        weights_hash=ShadowRecord.hash_weights(book.weights),
        effective_cash=acc.effective_cash,
        gross=book.gross(),
        turnover=None,
        projected_incentive=None,
        vetoes=tuple(vetoes),
        faults=tuple(faults),
    )
    ShadowRun(LOG).append(rec)

    print(f"{today.isoformat()} {args.window}: "
          f"{'PUBLISH (dry run)' if published else 'skip'}  "
          f"holdings={len(book.tickers())} gross={book.gross():.4f} "
          f"cash={acc.effective_cash:.4f} cutoff_met={cutoff_met}")
    print(f"  horizon {horizon.isoformat()}  cutoff {cutoff.isoformat()}"
          + ("  (EARLY CLOSE)" if session.early else ""))
    if report.errors:
        print(f"  validation: {report.errors}")
    if vetoes:
        print(f"  governor vetoes: {vetoes}")
    return 2 if faults else 0


def report(args) -> int:
    run = ShadowRun(LOG)
    s = run.stats()
    print(f"shadow log: {s['records']} records over {s['sessions']} session(s) "
          f"{s['first']} .. {s['last']}")
    print(f"  published {s['published']}  skipped {s['skipped']}  "
          f"vetoed {s['vetoed']}  faults {s['faults']}  "
          f"distinct books {s['distinct_books']}\n")
    rep = run.readiness(
        ReadinessCriteria(min_sessions=args.min_sessions),
        rule_detector_ran=args.rule_detector_ran,
        leakage_probes_green=(None if args.leakage is None else bool(args.leakage)),
        market_data_available=args.market_data,
    )
    print(rep.summary())
    if rep.unproven:
        print("\nUNPROVEN gates are not failures - they are things a shadow run cannot")
        print("settle. Do not treat them as passed.")
    return 0 if rep.ready else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cycle", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--window", choices=("MOO", "MOC"), default="MOC")
    ap.add_argument("--date")
    ap.add_argument("--min-sessions", type=int, default=10)
    ap.add_argument("--rule-detector-ran", action="store_true")
    ap.add_argument("--leakage", type=int, choices=(0, 1), default=None)
    ap.add_argument("--market-data", action="store_true")
    args = ap.parse_args()
    return cycle(args) if args.cycle else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
