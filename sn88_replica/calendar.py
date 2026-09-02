"""The NYSE session calendar - load-bearing in two different places.

SN88 runs two clocks against you and they disagree (guide §2.1):

* the **scoring window** (50) and the **new-miner ramp** (30) count TRADING SESSIONS;
* the **inactivity penalty** counts CALENDAR days, including weekends and holidays.

So a Friday-only refresh cadence is a permanent tax for nothing, and a long weekend is
worse. Both numbers come out of this module.

It also owns the execution cutoffs. ``STK_MOO = -120`` and ``STK_MOC = -600`` are
**seconds relative to the session open and close**, and ``pldaily1`` compares them
against unix-second timestamps, so the cutoffs must be computed in America/New_York and
converted - not assumed to be a fixed UTC offset. The offset moves twice a year.

Holidays are computed from rules, but an unscheduled closure (weather, a national day
of mourning) obeys no rule at all. ``CLOSURE_OVERRIDES`` exists for those, and a
rule-only calendar will silently disagree with reality on exactly those days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

from .constants import STK_MOC, STK_MOO, STK_TZ

__all__ = ["SessionCalendar", "Session", "NYSE_TZ", "easter", "good_friday"]

NYSE_TZ = ZoneInfo(STK_TZ)

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

#: Closures that follow no rule. Extend this from the NYSE announcements; a rule-based
#: calendar is silently wrong on these days and a backtest will disagree with reality.
CLOSURE_OVERRIDES: frozenset[date] = frozenset({
    date(2012, 10, 29), date(2012, 10, 30),   # Hurricane Sandy
    date(2018, 12, 5),                        # George H. W. Bush, national day of mourning
    date(2025, 1, 9),                         # Jimmy Carter, national day of mourning
})

#: Early closes that follow no rule, or that the rules below would get wrong.
EARLY_CLOSE_OVERRIDES: frozenset[date] = frozenset()

#: Days the rules would call a holiday but which actually traded.
OPEN_OVERRIDES: frozenset[date] = frozenset()


def easter(year: int) -> date:
    """Gregorian Easter (Meeus/Jones/Butcher). Needed only for Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def good_friday(year: int) -> date:
    return easter(year) - timedelta(days=2)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th ``weekday`` (Mon=0) of a month; ``n = -1`` means the last one."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """Weekend observance: Saturday -> the preceding Friday, Sunday -> the following Monday.

    Returns ``None`` when the shift would land outside the holiday's own year, which the
    NYSE handles by simply not observing it (New Year's Day falling on a Saturday).
    """
    if d.weekday() == 5:                       # Saturday
        shifted = d - timedelta(days=1)
        return shifted if shifted.year == d.year else None
    if d.weekday() == 6:                       # Sunday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def holidays(year: int) -> frozenset[date]:
    """NYSE full-day closures for a calendar year, computed from the rules."""
    out: set[date] = set()

    def add(d: date | None) -> None:
        if d is not None:
            out.add(d)

    add(_observed(date(year, 1, 1)))                       # New Year's Day
    add(_nth_weekday(year, 1, 0, 3))                       # MLK Jr Day - 3rd Monday
    add(_nth_weekday(year, 2, 0, 3))                       # Washington's Birthday - 3rd Monday
    add(good_friday(year))
    add(_nth_weekday(year, 5, 0, -1))                      # Memorial Day - last Monday
    if year >= 2022:
        add(_observed(date(year, 6, 19)))                  # Juneteenth - NYSE from 2022
    add(_observed(date(year, 7, 4)))                       # Independence Day
    add(_nth_weekday(year, 9, 0, 1))                       # Labor Day - 1st Monday
    add(_nth_weekday(year, 11, 3, 4))                      # Thanksgiving - 4th Thursday
    add(_observed(date(year, 12, 25)))                     # Christmas Day

    out -= OPEN_OVERRIDES
    return frozenset(out)


@lru_cache(maxsize=64)
def early_closes(year: int) -> frozenset[date]:
    """1:00 pm ET closes.

    Three rules, each conditional on the weekday - this is where hand-rolled calendars
    usually go wrong, so every one of these is pinned by a test fixture.
    """
    out: set[date] = set()
    hol = holidays(year)

    # Day after Thanksgiving
    friday = _nth_weekday(year, 11, 3, 4) + timedelta(days=1)
    if friday not in hol:
        out.add(friday)

    # July 3, when Independence Day is observed on a weekday and July 3 itself trades
    july4 = date(year, 7, 4)
    july3 = date(year, 7, 3)
    if july3.weekday() < 5 and july3 not in hol and july4.weekday() < 5:
        out.add(july3)

    # Christmas Eve, when it is a weekday and Christmas itself is a weekday
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5 and dec24 not in hol and date(year, 12, 25).weekday() < 5:
        out.add(dec24)

    out |= {d for d in EARLY_CLOSE_OVERRIDES if d.year == year}
    return frozenset(d for d in out if d not in hol)


@dataclass(frozen=True)
class Session:
    """One trading session, with its execution cutoffs resolved to unix seconds.

    ``moo_cutoff`` / ``moc_cutoff`` are the instants a strategy file must be submitted
    **before**, matching ``pldaily1``'s ``block < bk0 - 120`` and ``block < bk1 - 600``.
    """

    day: date
    open_utc: datetime
    close_utc: datetime
    early: bool

    @property
    def moo_cutoff(self) -> datetime:
        return self.open_utc + timedelta(seconds=STK_MOO)      # STK_MOO is negative

    @property
    def moc_cutoff(self) -> datetime:
        return self.close_utc + timedelta(seconds=STK_MOC)     # STK_MOC is negative

    @property
    def moo_cutoff_block(self) -> int:
        return int(self.moo_cutoff.timestamp())

    @property
    def moc_cutoff_block(self) -> int:
        return int(self.moc_cutoff.timestamp())

    def cutoff(self, window: str) -> datetime:
        w = window.upper()
        if w == "MOO":
            return self.moo_cutoff
        if w == "MOC":
            return self.moc_cutoff
        raise ValueError(f"unknown execution window {window!r}; expected MOO or MOC")

    def et(self, moment: datetime) -> str:
        return moment.astimezone(NYSE_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

    def __str__(self) -> str:
        tag = " (early close)" if self.early else ""
        return f"{self.day.isoformat()}{tag} MOO<{self.et(self.moo_cutoff)} MOC<{self.et(self.moc_cutoff)}"


class SessionCalendar:
    """Trading sessions and execution cutoffs, DST-correct.

    Deliberately has no notion of "today": every method takes explicit dates, so a
    backtest cannot accidentally consult the wall clock.
    """

    def __init__(self, *, tz: ZoneInfo = NYSE_TZ):
        self.tz = tz

    # --- session membership ---------------------------------------------

    def is_session(self, day: date) -> bool:
        if day.weekday() >= 5:
            return False
        if day in CLOSURE_OVERRIDES:
            return False
        return day not in holidays(day.year)

    def is_early_close(self, day: date) -> bool:
        return self.is_session(day) and day in early_closes(day.year)

    def session(self, day: date) -> Session:
        """Resolve one session. Raises if ``day`` is not a trading day."""
        if not self.is_session(day):
            raise ValueError(f"{day.isoformat()} is not an NYSE trading session")
        early = self.is_early_close(day)
        close_local = EARLY_CLOSE if early else REGULAR_CLOSE
        open_dt = datetime.combine(day, REGULAR_OPEN, tzinfo=self.tz)
        close_dt = datetime.combine(day, close_local, tzinfo=self.tz)
        return Session(
            day=day,
            open_utc=open_dt.astimezone(ZoneInfo("UTC")),
            close_utc=close_dt.astimezone(ZoneInfo("UTC")),
            early=early,
        )

    # --- ranges ----------------------------------------------------------

    def sessions(self, start: date, end: date) -> list[Session]:
        out, d = [], start
        while d <= end:
            if self.is_session(d):
                out.append(self.session(d))
            d += timedelta(days=1)
        return out

    def session_days(self, start: date, end: date) -> list[date]:
        return [s.day for s in self.sessions(start, end)]

    def count_sessions(self, start: date, end: date) -> int:
        """Trading sessions in ``[start, end]`` - the scoring-window clock."""
        return len(self.session_days(start, end))

    def count_calendar_days(self, start: date, end: date) -> int:
        """Calendar days in ``[start, end]`` - the INACTIVITY clock. Different number."""
        return (end - start).days + 1

    def previous_session(self, day: date) -> date | None:
        d = day - timedelta(days=1)
        for _ in range(15):
            if self.is_session(d):
                return d
            d -= timedelta(days=1)
        return None

    def next_session(self, day: date) -> date | None:
        d = day + timedelta(days=1)
        for _ in range(15):
            if self.is_session(d):
                return d
            d += timedelta(days=1)
        return None

    def sessions_to_calendar_days(self, start: date, n_sessions: int) -> int:
        """How many calendar days ``n_sessions`` actually spans from ``start``.

        The ramp is 30 sessions and the window is 50 - about six and ten calendar weeks.
        Use this whenever you are tempted to say "30 days".
        """
        d, seen = start, 0
        while seen < n_sessions:
            if self.is_session(d):
                seen += 1
            if seen < n_sessions:
                d += timedelta(days=1)
        return (d - start).days + 1

    # --- the trap this module exists to prevent --------------------------

    def window_for(self, submitted_at: datetime, day: date) -> str | None:
        """Which execution window a submission lands in - or ``None``.

        Mirrors ``pldaily1``: before the MOO cutoff fills at the open; between the two
        cutoffs fills at the close; **after the MOC cutoff it does not fill at all** and
        waits for the next session - with one exception, handled by
        :func:`sn88_replica.execution.first_submission_trap`.
        """
        s = self.session(day)
        if submitted_at < s.moo_cutoff:
            return "MOO"
        if submitted_at < s.moc_cutoff:
            return "MOC"
        return None
