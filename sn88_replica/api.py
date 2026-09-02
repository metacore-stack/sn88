"""Client for the owner API, with the integrity checks guide §33 requires.

Every field that decides your score - ``cash``, ``last``, the distance matrix, the
class ratio, the blacklist and the similarity threshold - arrives over unsigned plain
HTTP from a single operator. You cannot make that source trustworthy; you can make the
system refuse implausible input and keep an archive of what it used to say.

Nothing here is allowed to move a position on its own.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .constants import API_ROOT, fingerprint

__all__ = ["Endpoint", "fetch", "load_fixture", "archive", "IntegrityError", "PNL_COLS", "SCORE_COLS", "DAYS_COLS"]

PNL_COLS = (
    "uid hotkey date asset block_open block_high block_low block_close "
    "value_open value_high value_low value_close "
    "swap_open swap_high swap_low swap_close"
).split()

SCORE_COLS = "uid hotkey date a days value swap score apr lsr mar risk odds daily return".split()

DAYS_COLS = "uid hotkey date rank a days last cash".split()

Endpoint = str
ENDPOINTS: tuple[Endpoint, ...] = ("score", "pnl", "days", "dist", "ratio", "assets")


class IntegrityError(RuntimeError):
    """Raised when a payload fails a plausibility check. Quarantine, do not trade."""


def _get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sn88-replica/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed host
        return resp.read()


def fetch(endpoint: Endpoint, *, root: str = API_ROOT, timeout: float = 120.0) -> Any:
    """Fetch one endpoint. ``assets`` is HTML-ish text; the rest are JSON strings."""
    raw = _get(f"{root}/{endpoint}", timeout)
    if endpoint == "assets":
        return raw.decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    # every JSON endpoint double-encodes: the body is a JSON *string* of JSON
    return json.loads(parsed) if isinstance(parsed, str) else parsed


def load_fixture(path: str | Path) -> Any:
    """Load an archived payload. Same double-decode as :func:`fetch`."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("<") or "cash ETFs" in text[:400]:
        return text
    parsed = json.loads(text)
    return json.loads(parsed) if isinstance(parsed, str) else parsed


def archive(out_dir: str | Path, *, root: str = API_ROOT, on: date | None = None) -> dict[str, Path]:
    """Snapshot every endpoint to ``out_dir/<YYYY-MM-DD>-<endpoint>.json``.

    Run this daily from day one, before you have anywhere to put it. These become the
    golden fixtures, the rule-change detector's baseline, and the only record you will
    have of what the owner API used to say. You cannot backfill it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (on or date.today()).isoformat()
    written: dict[str, Path] = {}
    for ep in ENDPOINTS:
        raw = _get(f"{root}/{ep}", 120.0)
        suffix = "txt" if ep == "assets" else "json"
        path = out_dir / f"{stamp}-{ep}.{suffix}"
        path.write_bytes(raw)
        written[ep] = path
    (out_dir / f"{stamp}-constants.json").write_text(
        json.dumps({"fingerprint": fingerprint()}, indent=1), encoding="utf-8"
    )
    return written


# --- integrity checks (guide §33) -------------------------------------------


@dataclass(frozen=True)
class FieldRow:
    uid: int
    asset: int
    lifetime_days: int
    last_active: float
    cash: float
    rank: int


def parse_days(payload) -> dict[int, FieldRow]:
    """Parse ``/days`` with range checks. Raises :class:`IntegrityError` on bad input."""
    out: dict[int, FieldRow] = {}
    for row in payload:
        if len(row) != len(DAYS_COLS):
            raise IntegrityError(f"/days row has {len(row)} fields, expected {len(DAYS_COLS)}")
        uid, _hk, _dt, rank, a, days, last, cash = row
        if not (0 <= cash <= 1):
            raise IntegrityError(f"/days uid {uid}: cash={cash} outside [0,1]")
        if last < 0:
            raise IntegrityError(f"/days uid {uid}: last={last} negative")
        if days < 0:
            raise IntegrityError(f"/days uid {uid}: days={days} negative")
        if a not in (0, 1):
            raise IntegrityError(f"/days uid {uid}: asset class {a} not in (0,1)")
        out[uid] = FieldRow(uid=uid, asset=a, lifetime_days=days, last_active=last,
                            cash=cash, rank=rank)
    return out


def check_ratio(ratio) -> list[float]:
    """Validate ``/ratio``. A sum below 1.0 means emission is being burned at UID 0."""
    if not isinstance(ratio, (list, tuple)) or not ratio:
        raise IntegrityError(f"/ratio malformed: {ratio!r}")
    for r in ratio:
        if not (0 <= r <= 1):
            raise IntegrityError(f"/ratio entry {r} outside [0,1]")
    if sum(ratio) > 1.0 + 1e-9:
        raise IntegrityError(f"/ratio sums to {sum(ratio)} > 1")
    return list(ratio)


def check_dist(payload) -> None:
    """Validate the ``/dist`` tail: ``[blacklist, whitelist, trigger_override]``."""
    if not payload:
        raise IntegrityError("/dist empty")
    tail = payload[-1]
    if len(tail) < 3:
        raise IntegrityError(f"/dist tail malformed: {tail!r}")
    if not isinstance(tail[0], list) or not isinstance(tail[1], list):
        raise IntegrityError("/dist blacklist/whitelist not lists")
    override = tail[2]
    if override and not (0 < override < 2):
        raise IntegrityError(f"/dist trigger override {override} implausible")
