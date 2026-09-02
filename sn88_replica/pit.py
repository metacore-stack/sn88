"""Point-in-time storage: the data-layer defence against look-ahead.

Guide §8. Every observation carries two timestamps and they are not the same thing:

* ``event_time`` - when the thing happened (the session the price belongs to, the
  quarter the earnings cover, the ex-date of the dividend).
* ``available_time`` - when *you could have known it*. A Q3 earnings figure has an
  ``event_time`` in September and an ``available_time`` in late October.

Reading by ``event_time`` is the single most common way an equity backtest lies, and it
lies in the direction that makes results look better. This store makes it awkward:
:meth:`PointInTimeStore.asof` requires an ``as_of`` and filters on ``available_time``,
and there is no method that returns "the latest value" without one.

**Revisions are first-class.** A restated fundamental is a *new* observation with the
same ``event_time`` and a later ``available_time``. ``asof`` returns what you would have
seen at the time, not what the vendor believes today - which is why a restated database
silently inflates every backtest built on it.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .modes import LeakageError

__all__ = [
    "Observation",
    "PointInTimeStore",
    "PITSnapshot",
    "AccessLog",
]


@dataclass(frozen=True, order=True)
class Observation:
    """One fact, with both timestamps. Immutable; a correction is a new observation."""

    available_time: datetime
    event_time: datetime
    key: str = field(compare=False)
    value: Any = field(compare=False, default=None)
    source: str = field(compare=False, default="")

    def as_row(self) -> dict:
        return {
            "key": self.key,
            "event_time": self.event_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "value": self.value,
            "source": self.source,
        }

    @classmethod
    def from_row(cls, row: dict) -> "Observation":
        return cls(
            available_time=datetime.fromisoformat(row["available_time"]),
            event_time=datetime.fromisoformat(row["event_time"]),
            key=row["key"],
            value=row["value"],
            source=row.get("source", ""),
        )


@dataclass
class AccessLog:
    """Records every read, so a replay can prove what the policy actually consulted."""

    entries: list[tuple[str, datetime]] = field(default_factory=list)
    enabled: bool = False

    def record(self, key: str, as_of: datetime) -> None:
        if self.enabled:
            self.entries.append((key, as_of))

    def keys(self) -> set[str]:
        return {k for k, _ in self.entries}

    def clear(self) -> None:
        self.entries.clear()


class PointInTimeStore:
    """Append-only observations, queryable only with an explicit horizon.

    Observations are kept sorted by ``available_time`` per key, so ``asof`` is a binary
    search rather than a scan.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, list[Observation]] = {}
        self.access = AccessLog()

    # --- writing ---------------------------------------------------------

    def add(self, obs: Observation) -> None:
        if obs.available_time < obs.event_time:
            raise ValueError(
                f"{obs.key}: available_time {obs.available_time.isoformat()} precedes "
                f"event_time {obs.event_time.isoformat()} - you cannot know something "
                f"before it happens"
            )
        bucket = self._by_key.setdefault(obs.key, [])
        bisect.insort(bucket, obs)

    def extend(self, observations: Iterable[Observation]) -> None:
        for obs in observations:
            self.add(obs)

    # --- reading (horizon mandatory) -------------------------------------

    def asof(self, key: str, as_of: datetime, *, default: Any = None) -> Any:
        """The value as it was known at ``as_of``. Strict: ``available_time < as_of``.

        When a fact has been revised, this returns the revision that was current then -
        never today's restated value.
        """
        self.access.record(key, as_of)
        bucket = self._by_key.get(key)
        if not bucket:
            return default
        idx = bisect.bisect_left(bucket, (as_of,), key=lambda o: (o.available_time,))
        if idx == 0:
            return default
        return bucket[idx - 1].value

    def history(
        self, key: str, as_of: datetime, *, limit: int | None = None
    ) -> list[tuple[datetime, Any]]:
        """``(event_time, value)`` for everything knowable at ``as_of``, oldest first.

        Deduplicated by ``event_time``, keeping the latest revision available at
        ``as_of`` - which is what a feature calculation wants.
        """
        self.access.record(key, as_of)
        bucket = self._by_key.get(key, [])
        latest: dict[datetime, Any] = {}
        for obs in bucket:
            if obs.available_time < as_of:
                latest[obs.event_time] = obs.value
            else:
                break
        rows = sorted(latest.items())
        return rows[-limit:] if limit else rows

    def revisions(self, key: str, event_time: datetime) -> list[Observation]:
        """Every version of one fact, oldest first. Use this to audit a vendor."""
        return [o for o in self._by_key.get(key, []) if o.event_time == event_time]

    def keys(self) -> list[str]:
        return sorted(self._by_key)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    # --- snapshots -------------------------------------------------------

    def snapshot(self, as_of: datetime) -> "PITSnapshot":
        """A frozen view with the horizon baked in. Pass this to feature code."""
        return PITSnapshot(self, as_of)

    # --- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> int:
        """JSONL, one observation per line. Raw vendor pulls stay immutable elsewhere."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with p.open("w", encoding="utf-8") as fh:
            for key in sorted(self._by_key):
                for obs in self._by_key[key]:
                    fh.write(json.dumps(obs.as_row(), separators=(",", ":")) + "\n")
                    n += 1
        return n

    @classmethod
    def load(cls, path: str | Path) -> "PointInTimeStore":
        store = cls()
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    store.add(Observation.from_row(json.loads(line)))
        return store


@dataclass(frozen=True)
class PITSnapshot:
    """A store plus a fixed horizon. Feature code should take one of these, not a store.

    Because the horizon is baked in, a feature function physically cannot widen it, and
    an ``as_of`` cannot be forgotten at a call site.
    """

    store: PointInTimeStore
    as_of: datetime

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.asof(key, self.as_of, default=default)

    def history(self, key: str, limit: int | None = None) -> list[tuple[datetime, Any]]:
        return self.store.history(key, self.as_of, limit=limit)

    def series(self, key: str, limit: int | None = None) -> list[Any]:
        return [v for _t, v in self.history(key, limit=limit)]

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise LeakageError(
                f"{key!r} is not knowable at {self.as_of.isoformat()} - a feature "
                f"depending on it must be skipped, not back-filled from a later value"
            )
        return value

    def assert_horizon(self, as_of: datetime) -> "PITSnapshot":
        if self.as_of != as_of:
            raise LeakageError(
                f"snapshot horizon {self.as_of.isoformat()} != required "
                f"{as_of.isoformat()}"
            )
        return self
