"""Mirror of ``Investing/core/const.py`` for the US-equity class.

Every value here decides income. Validators run ``etc.update()`` hourly, which does
``git pull && pip install -e .``, so upstream can change these without notice.

``fingerprint()`` exists so the rule-change detector can alert and freeze candidate
selection the moment the mirror stops matching upstream (guide §12, §21, §33).
"""

from __future__ import annotations

import hashlib
import json

# --- scoring window ---------------------------------------------------------
WIN_SIZE_STK = 50          # TRADING SESSIONS, not calendar days
RISK_INIT_STK = 5.0        # MAR denominator floor numerator: floor = RISK_INIT/sqrt(days)

# --- outlier clip -----------------------------------------------------------
CLIP_OUTLIERS = 2          # top (N+1) positive sessions are levelled to the (N+1)th
CLIP_DEFAULT = 1.0         # percent; used when the window holds <= N positive sessions

# --- new-miner ramp ---------------------------------------------------------
DAYS_FINAL = 30            # ramp length AND dedupe recovery length
DAYS_DELAY = 1             # ramp exponent - TUNABLE upstream, watch it

# --- inactivity (DEC) -------------------------------------------------------
DEC1_START = 5             # guard is on TRACK AGE, not idle time
DEC1_CLIFF = 20            # score reaches zero here
DEC1_DECAY = 2

# --- cash and shorts --------------------------------------------------------
CASH_DECAY = 1             # linear from the first basis point
CASH_RESIDUE = 0.01        # floor

# --- similarity (dedupe) ----------------------------------------------------
DD_TRIGGER = 0.01          # normalised L2 distance; API can override via /dist tail
DD_POWER = 1

# --- burn -------------------------------------------------------------------
DEC_UID = 0                # receives sum(score)/sum(ra)*(1-sum(ra))

# --- US-equity execution ----------------------------------------------------
STK_MOO = -120             # seconds before the open  -> 09:28 ET
STK_MOC = -600             # seconds before the close -> 15:50 ET
STK_FEE = 0.002            # $0.002 PER SHARE traded - NOT 0.2% of notional
STK_BENCH = "SPY"
STK_TZ = "America/New_York"

# --- class ------------------------------------------------------------------
ASSET_STOCKS = 1           # '_': 1 - MANDATORY in every strategy file you write
ASSET_ALPHA = 0            # the other class; omitting '_' silently selects it

# blocks-per-day divisor used by dedupe(), indexed by asset class.
# asset 0 counts chain blocks (7200/day); asset 1 counts unix seconds (86400/day).
DEDUPE_DIVISOR = (7200, 86400)

API_ROOT = "http://api.investing88.ai"   # plain http, unsigned - see guide §33

_MIRRORED = (
    "WIN_SIZE_STK RISK_INIT_STK CLIP_OUTLIERS CLIP_DEFAULT DAYS_FINAL DAYS_DELAY "
    "DEC1_START DEC1_CLIFF DEC1_DECAY CASH_DECAY CASH_RESIDUE DD_TRIGGER DD_POWER "
    "DEC_UID STK_MOO STK_MOC STK_FEE"
).split()


def snapshot() -> dict:
    """The mirrored constants, for archiving and diffing."""
    g = globals()
    return {k: g[k] for k in _MIRRORED}


def fingerprint() -> str:
    """Stable hash of the mirrored constants.

    Store this alongside each archived fixture set. When it changes, the rule-change
    detector must freeze candidate selection until the replica is re-validated.
    """
    blob = json.dumps(snapshot(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
