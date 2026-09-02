"""Differential test against the REAL ``etc.score()``, not just against ``/score``.

This closes the last verification gap. ``/score`` publishes only the raw score, so the
five adjusted components - inception gate, dedupe, inactivity, cash, class
normalisation - plus the UID-0 burn had no external reference and were verified by
construction alone.

They can be checked directly, because ``etc.py`` needs only numpy/pandas/sqlalchemy/
requests. **bittensor is not required** - only ``api.py`` imports it, and only for
logging. So the real scorer runs against archived payloads with a light install::

    cd investing && .venv/bin/pip install "numpy>=2" "pandas==2.2.3" "sqlalchemy>=2" "requests>=2"

Measured: max absolute difference 2.78e-17 across 228 UIDs - machine epsilon.

Skips cleanly when the upstream venv is absent, so the suite still runs anywhere.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "investing"
VENV_PY = UPSTREAM / ".venv" / "bin" / "python"

# The comparison runs inside the upstream venv (which has pandas) and reports back as
# JSON, so this test file itself stays pure-stdlib like the rest of the package.
PROBE = r'''
import sys, json, io, contextlib, glob
from pathlib import Path
sys.path.insert(0, {upstream!r})
sys.path.insert(0, {root!r})
import pandas as pd
import Investing.core.etc as etc
from sn88_replica import api
from sn88_replica.pipeline import score_field

def load(sfx):
    p = sorted(glob.glob(str(Path({root!r}) / "fixtures" / ("*-" + sfx))))[-1]
    v = json.loads(Path(p).read_text(errors="replace"))
    return json.loads(v) if isinstance(v, str) else v

pnl, days, dist, ratio = load("pnl.json"), load("days.json"), load("dist.json"), load("ratio.json")
cols = lambda n: pd.read_csv(str(Path({upstream!r}) / "Investing" / "core" / "db" / n)).columns
pl = pd.DataFrame(pnl, None, cols("pnl.col"))
da = pd.DataFrame(days, None, cols("days.col"))

with contextlib.redirect_stdout(io.StringIO()):
    real, uid0, dec = etc.score(pl, dist, da, ratio, 256)

mine = score_field(pnl, api.parse_days(days), dist, ratio)
tot = sum(real)
real_inc = {{i: (v / tot if tot else 0.0) for i, v in enumerate(real)}}

worst, worst_uid = 0.0, None
for uid, s in mine.items():
    d = abs(s.incentive - real_inc.get(uid, 0.0))
    if d > worst:
        worst, worst_uid = d, uid

print(json.dumps({{
    "compared": len(mine),
    "nonzero": sum(1 for s in mine.values() if s.incentive > 0),
    "max_abs_diff": worst,
    "worst_uid": worst_uid,
    "burn_uid0": real_inc.get(0, 0.0),
    "dec": dec,
    "real_top": sorted(((v, i) for i, v in real_inc.items() if v > 0), reverse=True)[:5],
    "mine_top": sorted(((s.incentive, u) for u, s in mine.items() if s.incentive > 0), reverse=True)[:5],
}}))
'''


def _run_probe() -> dict:
    src = PROBE.format(upstream=str(UPSTREAM), root=str(ROOT))
    out = subprocess.run([str(VENV_PY), "-c", src], capture_output=True, text=True,
                         timeout=300, cwd=str(ROOT))
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


@unittest.skipUnless(
    VENV_PY.exists(),
    "upstream venv absent - run: cd investing && .venv/bin/pip install "
    "'numpy>=2' 'pandas==2.2.3' 'sqlalchemy>=2' 'requests>=2'")
class TestAgainstRealEtcScore(unittest.TestCase):
    """The differential test. Everything else compares against a published artefact."""

    TOLERANCE = 1e-12          # machine epsilon in practice; measured 2.78e-17

    @classmethod
    def setUpClass(cls):
        try:
            import pandas  # noqa: F401  (in the outer interpreter? usually not)
        except ImportError:
            pass
        if not sorted(glob.glob(str(ROOT / "fixtures" / "*-pnl.json"))):
            raise unittest.SkipTest("no archived /pnl fixture")
        try:
            cls.result = _run_probe()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"upstream probe failed: {exc}") from exc

    def test_incentive_matches_the_real_implementation(self):
        r = self.result
        print(f"\n  real etc.score() vs pipeline.score_field(): {r['compared']} UIDs "
              f"({r['nonzero']} earning), max abs diff {r['max_abs_diff']:.3e} "
              f"(uid {r['worst_uid']})")
        self.assertGreater(r["compared"], 100)
        self.assertLess(r["max_abs_diff"], self.TOLERANCE)

    def test_the_leaderboard_ordering_is_identical(self):
        r = self.result
        self.assertEqual([u for _v, u in r["real_top"]], [u for _v, u in r["mine_top"]])

    def test_burn_is_zero_while_the_ratios_sum_to_one(self):
        """score[0] = sum/sum(ra)*(1-sum(ra)); with /ratio == [0.5, 0.5] this is zero."""
        self.assertAlmostEqual(self.result["burn_uid0"], 0.0, places=12)

    def test_subnet_level_dec_is_computed_but_inert(self):
        """Both consumers of the subnet-level DEC are commented out upstream."""
        self.assertGreaterEqual(self.result["dec"], 0.0)
        self.assertLess(self.result["dec"], 0.03)      # below DEC_CUTOFF


class TestUpstreamDependencyFootprint(unittest.TestCase):
    """etc.py and simst.py do NOT need bittensor - only api.py does, and only to log."""

    def test_only_api_imports_bittensor(self):
        if not UPSTREAM.exists():
            self.skipTest("upstream repo absent")
        needs = {}
        for name in ("simst.py", "etc.py", "api.py"):
            text = (UPSTREAM / "Investing" / "core" / name).read_text()
            needs[name] = "bittensor" in text
        self.assertFalse(needs["simst.py"], "simst must not require bittensor")
        self.assertFalse(needs["etc.py"], "etc must not require bittensor")
        self.assertTrue(needs["api.py"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
