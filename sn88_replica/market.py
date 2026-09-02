"""The competition-replication track: SN88's own market feed, loaded point-in-time.

Guide §8 and §30, and the two-track design. This is the **authoritative** backtest
source, and it is authoritative for four reasons that a vendor feed cannot match:

1. **They are the prices the validator uses.** No adjustment disagreement, no
   reconciliation step, no vendor-vs-vendor arbitration.
2. **They are raw and unadjusted.** Required, not merely preferable: upstream scales the
   share count on a split (``alpha0 *= spl2/spli``) and reinvests dividends as shares
   (``alpha0 *= 1 + divd/price``). Feeding back-adjusted prices double-counts every
   corporate action.
3. **They are membership-aware.** The feed carries 1,279 tickers against ~555 currently
   eligible, so historical eligibility can be reconstructed rather than assumed from
   today's ``/assets``.
4. **They carry the session grid the cutoffs are derived from** - 30-minute intraday
   bars whose first and last timestamps are the ``bk0``/``bk1`` that define the MOO and
   MOC windows.

**What it is not.** 314 sessions from 2025-06-02 is one market regime. It cannot test a
recession, a rate shock or a crash, which is why the long-regime track exists - and why
:class:`Track` labels every dataset so a model cannot silently treat the two as
interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .api import fetch, load_fixture
from .calendar import SessionCalendar
from .pit import Observation, PointInTimeStore

__all__ = [
    "Track",
    "DailyBar",
    "IntradayBar",
    "CorporateAction",
    "MarketFeed",
    "DAILY1_COLS",
    "SPLIT_COLS",
    "DIVIDEND_COLS",
]

#: ``/daily1/{date}`` rows. ``ochl`` is ``'day'`` for the daily bar, otherwise intraday.
DAILY1_COLS = ("date", "block", "ticker", "open", "high", "low", "close", "volume", "ochl")
SPLIT_COLS = ("date", "ticker", "from", "to")
DIVIDEND_COLS = ("ex_date", "pay_date", "ticker", "amount", "currency", "type")


class Track(str, Enum):
    """Which backtest a dataset belongs to. Carried in the data, not the documentation.

    A long-regime series is survivorship-biased by construction and must never be used
    to produce an authoritative performance estimate. Labelling it in the store makes
    that structural rather than a matter of remembering.
    """

    REPLICATION = "replication"        # SN88's own feed - authoritative, one regime
    LONG_REGIME = "long_regime"        # vendor history - SURVIVORSHIP BIASED, stress only


@dataclass(frozen=True)
class DailyBar:
    ticker: str
    session: date
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def available_time_close(self) -> datetime:
        """When the close became knowable: the session close, not the session date."""
        return SessionCalendar().session(self.session).close_utc


@dataclass(frozen=True)
class IntradayBar:
    ticker: str
    session: date
    block: int                 # unix seconds - the bar timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def at(self) -> datetime:
        return datetime.fromtimestamp(self.block, tz=timezone.utc)


@dataclass(frozen=True)
class CorporateAction:
    kind: str                  # "split" | "dividend"
    ticker: str
    effective: date            # split date, or dividend ex-date
    ratio_from: float = 1.0
    ratio_to: float = 1.0
    amount: float = 0.0
    currency: str = ""
    pay_date: date | None = None

    @property
    def split_factor(self) -> float:
        return self.ratio_to / self.ratio_from if self.ratio_from else 1.0


class MarketFeed:
    """Load ``/daily1``, ``/split`` and ``/dividend`` into a point-in-time store.

    The whole point is the ``available_time`` assignment, and it is not the session date::

        daily close   -> knowable at the session CLOSE
        daily open    -> knowable at the session OPEN
        intraday bar  -> knowable at the bar's own timestamp
        split         -> effective on its date; announcement date is NOT in the feed
        dividend      -> applied on ex_date; announcement date is NOT in the feed

    The last two are a real limitation and are recorded as such: ``/split`` and
    ``/dividend`` carry no declaration date, so nothing in SN88's own data says when an
    action became public. Any "dividend in N days" feature built from this table is
    unfilterable by availability - treat corporate actions as knowable only on their
    effective date, which is conservative but honest.
    """

    def __init__(self, calendar: SessionCalendar | None = None):
        self.calendar = calendar or SessionCalendar()

    # --- fetching --------------------------------------------------------

    @staticmethod
    def fetch_daily(since: str | date, *, timeout: float = 600.0) -> list[list]:
        """``/daily1/{since}``. The full live-era pull is ~325 MB - do it once."""
        day = since.isoformat() if isinstance(since, date) else since
        return fetch(f"daily1/{day}", timeout=timeout)

    @staticmethod
    def fetch_actions(*, timeout: float = 120.0) -> tuple[list[list], list[list]]:
        return fetch("split", timeout=timeout), fetch("dividend", timeout=timeout)

    # --- parsing ---------------------------------------------------------

    def parse_daily(self, rows: Sequence[Sequence]) -> Iterator[DailyBar]:
        idx = {c: i for i, c in enumerate(DAILY1_COLS)}
        for r in rows:
            if r[idx["ochl"]] != "day":
                continue
            yield DailyBar(
                ticker=r[idx["ticker"]],
                session=date.fromisoformat(r[idx["date"]]),
                open=float(r[idx["open"]]), high=float(r[idx["high"]]),
                low=float(r[idx["low"]]), close=float(r[idx["close"]]),
                volume=float(r[idx["volume"]]),
            )

    def parse_intraday(self, rows: Sequence[Sequence]) -> Iterator[IntradayBar]:
        idx = {c: i for i, c in enumerate(DAILY1_COLS)}
        for r in rows:
            if r[idx["ochl"]] == "day":
                continue
            yield IntradayBar(
                ticker=r[idx["ticker"]],
                session=date.fromisoformat(r[idx["date"]]),
                block=int(r[idx["block"]]),
                open=float(r[idx["open"]]), high=float(r[idx["high"]]),
                low=float(r[idx["low"]]), close=float(r[idx["close"]]),
                volume=float(r[idx["volume"]]),
            )

    @staticmethod
    def parse_splits(rows: Sequence[Sequence]) -> Iterator[CorporateAction]:
        idx = {c: i for i, c in enumerate(SPLIT_COLS)}
        for r in rows:
            yield CorporateAction(
                kind="split", ticker=r[idx["ticker"]],
                effective=date.fromisoformat(r[idx["date"]]),
                ratio_from=float(r[idx["from"]]), ratio_to=float(r[idx["to"]]),
            )

    @staticmethod
    def parse_dividends(rows: Sequence[Sequence]) -> Iterator[CorporateAction]:
        idx = {c: i for i, c in enumerate(DIVIDEND_COLS)}
        for r in rows:
            pay = r[idx["pay_date"]]
            yield CorporateAction(
                kind="dividend", ticker=r[idx["ticker"]],
                effective=date.fromisoformat(r[idx["ex_date"]]),
                amount=float(r[idx["amount"]]), currency=r[idx["currency"]],
                pay_date=date.fromisoformat(pay) if pay else None,
            )

    # --- loading ---------------------------------------------------------

    def load_daily(
        self,
        store: PointInTimeStore,
        rows: Sequence[Sequence],
        *,
        track: Track = Track.REPLICATION,
        fields: Iterable[str] = ("open", "close", "volume"),
    ) -> int:
        """Load daily bars with per-field availability.

        The open and the close are knowable at different instants, so they are stored as
        separate observations. Storing a whole bar under one timestamp is what makes a
        ``gap = open/prev_close`` feature leak at the 09:28 cutoff.
        """
        n = 0
        for bar in self.parse_daily(rows):
            try:
                session = self.calendar.session(bar.session)
            except ValueError:
                continue                       # feed disagrees with the calendar; skip
            for field in fields:
                available = session.open_utc if field == "open" else session.close_utc
                store.add(Observation(
                    available_time=available,
                    event_time=session.open_utc,
                    key=f"{track.value}:{bar.ticker}:{field}",
                    value=getattr(bar, field),
                    source="sn88/daily1",
                ))
                n += 1
        return n

    def load_intraday(
        self,
        store: PointInTimeStore,
        rows: Sequence[Sequence],
        *,
        track: Track = Track.REPLICATION,
        tickers: Iterable[str] | None = None,
    ) -> int:
        """Load 30-minute bars, each knowable at its own timestamp.

        A bar stamped 15:30 has not closed until 16:00, so an MOC decision at 15:50 may
        use the 15:00 bar and not the 15:30 one. ``available_time`` is set to the bar
        timestamp plus the 30-minute interval for exactly that reason.
        """
        keep = set(tickers) if tickers else None
        n = 0
        for bar in self.parse_intraday(rows):
            if keep and bar.ticker not in keep:
                continue
            store.add(Observation(
                available_time=bar.at + timedelta(minutes=30),   # the bar must CLOSE first
                event_time=bar.at,
                key=f"{track.value}:{bar.ticker}:intraday_close",
                value=bar.close,
                source="sn88/daily1",
            ))
            n += 1
        return n

    def load_actions(
        self,
        store: PointInTimeStore,
        splits: Sequence[Sequence],
        dividends: Sequence[Sequence],
        *,
        track: Track = Track.REPLICATION,
    ) -> int:
        """Load corporate actions, knowable only on their effective date.

        Conservative by necessity: neither ``/split`` nor ``/dividend`` carries a
        declaration date, so the feed cannot tell you when an action became public.
        """
        n = 0
        for action in list(self.parse_splits(splits)) + list(self.parse_dividends(dividends)):
            try:
                when = self.calendar.session(action.effective).open_utc
            except ValueError:
                when = datetime.combine(action.effective, datetime.min.time(),
                                        tzinfo=timezone.utc)
            store.add(Observation(
                available_time=when,
                event_time=when,
                key=f"{track.value}:{action.ticker}:{action.kind}",
                value=(action.split_factor if action.kind == "split" else action.amount),
                source=f"sn88/{action.kind}",
            ))
            n += 1
        return n

    # --- universe reconstruction ----------------------------------------

    @staticmethod
    def eligible_on(fixtures_dir: str | Path, day: date) -> frozenset:
        """Who was eligible on ``day``, from the archived ``/assets`` snapshots.

        Falls back to the nearest EARLIER snapshot - never a later one, which would be a
        membership look-ahead. Returns an empty set when nothing earlier exists, so a
        backtest fails loudly rather than silently using today's universe.
        """
        import re

        best: tuple[str, Path] | None = None
        for path in sorted(Path(fixtures_dir).glob("*-assets.txt")):
            stamp = path.name[:10]
            if stamp <= day.isoformat() and (best is None or stamp > best[0]):
                best = (stamp, path)
        if best is None:
            return frozenset()
        text = best[1].read_text(encoding="utf-8", errors="replace")
        return frozenset(re.findall(r"quote/([A-Z.\-]+)/", text))

    @staticmethod
    def departed(history_tickers: Iterable[str], current: Iterable[str]) -> list[str]:
        """Tickers seen historically but absent from the current universe.

        **Not a delisting list.** A symbol leaves ``/assets`` for market-cap, liquidity,
        eligibility-rule, rename, acquisition or delisting reasons, and the feed does not
        say which - AA, AAL, ACM, AEE and AGNC are all in this set and all still trade.
        Reconcile against an external ``active=false`` / ``delisted_utc`` source before
        calling any of them delisted.
        """
        return sorted(set(history_tickers) - set(current))
