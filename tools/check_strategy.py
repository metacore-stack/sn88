#!/usr/bin/env python3
"""Predict exactly what upstream will do with a strategy file, before you publish it.

Upstream fails silently: an unparseable file is discarded with `except: continue`, an
unknown ticker becomes cash, an expression changes meaning. This tells you which.

    python3 tools/check_strategy.py --file Investing/strat/<hotkey>
    python3 tools/check_strategy.py --text "{'_':1,'AAPL':0.99}"
    python3 tools/check_strategy.py --text "{'_':1,'AAPL':0.99}  # oops"
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn88_replica import api
from sn88_replica.strategy import parse_like_upstream, predict_acceptance, sanitize
from sn88_replica.universe import parse_assets

ROOT = Path(__file__).resolve().parent.parent


def load_universe(offline: bool):
    hits = sorted(glob.glob(str(ROOT / "fixtures" / "*-assets.txt")))
    if hits and offline:
        return parse_assets(Path(hits[-1]).read_text(encoding="utf-8", errors="replace"))
    try:
        return parse_assets(api.fetch("assets"))
    except Exception:
        if not hits:
            raise
        return parse_assets(Path(hits[-1]).read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--text")
    ap.add_argument("--offline", action="store_true", help="use the archived /assets")
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    u = load_universe(args.offline)

    cleaned = sanitize(text)
    print(f"  as written : {text!r}")
    print(f"  sanitised  : {cleaned!r}")
    if cleaned != text.strip():
        print("  NOTE       : the sanitiser altered the text (whitespace is harmless;")
        print("               stray letters from a comment are NOT).")

    acc = predict_acceptance(text, u)
    print()
    if not acc.accepted:
        print("  VERDICT    : *** SILENTLY DISCARDED ***")
        print("               No error is raised anywhere. Your previous allocation stays")
        print("               live and the inactivity clock keeps running.")
        print(f"               reason: {acc.reason}")
        return 2

    parsed = acc.parsed
    gross = sum(abs(v) for k, v in parsed.items() if k not in ("_", "*", "=", "-+") and k != "")
    print("  VERDICT    : accepted")
    print(f"  asset class: {parsed.get('_', 0)}  {'(US stocks)' if parsed.get('_') == 1 else '(ALPHA - wrong class!)'}")
    print(f"  holdings   : {len([k for k in parsed if k not in ('_','*','=','-+') and k != ''])}")
    print(f"  gross      : {gross:.6f}")
    print(f"  eff. cash  : {acc.effective_cash:.6f}  -> score multiplier "
          f"{max(1 - acc.effective_cash, 0.01):.4f}x")
    if acc.reclassified:
        print(f"  RECLASSIFIED TO CASH: {list(acc.reclassified)}")
        print("               (unknown/delisted tickers, or the 16 cash-classified ETFs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
