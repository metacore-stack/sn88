"""Write the strategy file safely, and confirm the miner actually submitted it.

Guide §16 and §29. The division of labour is easy to get wrong:

* **You** write ``Investing/strat/<hotkey_ss58>``.
* **The miner daemon** notices the mtime once a second and POSTs it to ``/rev``.
* ``.last-update`` is touched **only** when the API returns ``<= 201``.

So the publisher never speaks to the API, never needs the network, and never touches a
wallet. Verification is watching ``.last-update`` advance. And you do **not** write a
retry loop - the miner retries every ~11 seconds forever, because ``.last-update`` never
advanced (guide §5.3).

Four failure modes this module exists to prevent:

1. **A partial or interleaved write.** Two publishers racing produce a file that may
   parse to something neither intended, or fail to parse at all - and upstream discards
   an unparseable file *silently*. Hence the ``flock``.
2. **A torn write surviving a crash.** Hence write-temp, ``fsync``, atomic ``rename``.
3. **Publishing something upstream will discard.** Hence mandatory validation, which
   simulates upstream's own parser (:mod:`sn88_replica.strategy`).
4. **Believing you rebalanced when you did not.** Hence ``.last-update`` verification -
   and note the inactivity clock keeps running while you are wrong about this.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .strategy import Strategy, ValidationReport, serialize, validate
from .universe import Universe

__all__ = ["Publisher", "PublishResult", "PublishError"]


class PublishError(RuntimeError):
    """Refused to publish, or published and could not confirm."""


@dataclass(frozen=True)
class PublishResult:
    written: bool
    dry_run: bool
    path: Path
    text: str
    mtime_before: float | None
    mtime_after: float | None
    last_update_before: float | None
    last_update_after: float | None
    confirmed: bool
    report: ValidationReport
    note: str = ""

    @property
    def submitted(self) -> bool:
        """True only when the miner acknowledged the API accepted it."""
        return self.confirmed


class Publisher:
    """Atomic, locked, validated writer for one hotkey's strategy file.

    Args:
        strat_dir: ``Investing/strat`` in the cloned repo.
        hotkey_ss58: the strategy filename. **Never** a coldkey - the coldkey holds the
            192 staked alpha and must not be on this host at all (guide §29).
        lock_dir: where the advisory lock lives; defaults to ``strat_dir``.
    """

    def __init__(self, strat_dir: str | Path, hotkey_ss58: str, *, lock_dir: str | Path | None = None):
        self.strat_dir = Path(strat_dir)
        self.hotkey = hotkey_ss58
        if not self.strat_dir.is_dir():
            raise PublishError(f"strategy directory does not exist: {self.strat_dir}")
        if "/" in hotkey_ss58 or hotkey_ss58 in (".", ".."):
            raise PublishError(f"implausible hotkey filename: {hotkey_ss58!r}")
        self.path = self.strat_dir / hotkey_ss58
        self.last_update = self.strat_dir / ".last-update"
        self.lock_path = Path(lock_dir or self.strat_dir) / f".{hotkey_ss58}.publock"

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _mtime(p: Path) -> float | None:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return None

    def _atomic_write(self, text: str) -> None:
        """Write via temp + fsync + rename, so no reader ever sees a partial file."""
        fd, tmp = tempfile.mkstemp(dir=str(self.strat_dir), prefix=f".{self.hotkey}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            dir_fd = os.open(str(self.strat_dir), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # --- the operation ---------------------------------------------------

    def publish(
        self,
        strategy: Strategy,
        universe: Universe,
        *,
        dry_run: bool = False,
        confirm_timeout: float = 300.0,
        poll_interval: float = 2.0,
        **limits,
    ) -> PublishResult:
        """Validate, write atomically under a lock, then confirm submission.

        Args:
            dry_run: validate and render, write nothing. This is what stage 7's shadow
                operation uses - the full loop with the last step disarmed.
            confirm_timeout: seconds to wait for ``.last-update`` to advance. Guide §23
                alerts if unconfirmed after 300s.
            **limits: forwarded to :func:`sn88_replica.strategy.validate`.

        Raises:
            PublishError: if validation fails. The file is never written.
        """
        report = validate(strategy, universe, **limits)
        text = serialize(strategy)

        if not report.ok:
            raise PublishError(
                "refusing to publish:\n  - " + "\n  - ".join(report.errors)
            )

        if dry_run:
            return PublishResult(
                written=False, dry_run=True, path=self.path, text=text,
                mtime_before=self._mtime(self.path), mtime_after=None,
                last_update_before=self._mtime(self.last_update), last_update_after=None,
                confirmed=False, report=report, note="dry run - nothing written",
            )

        self.lock_path.touch(exist_ok=True)
        with open(self.lock_path, "r+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PublishError(
                    "another publisher holds the lock - refusing to race "
                    f"({self.lock_path})"
                ) from exc
            try:
                mtime_before = self._mtime(self.path)
                lu_before = self._mtime(self.last_update)

                self._atomic_write(text)

                mtime_after = self._mtime(self.path)
                if mtime_after is None:
                    raise PublishError("strategy file missing immediately after write")
                if mtime_before is not None and mtime_after <= mtime_before:
                    # the miner keys off mtime strictly increasing; force it forward
                    bump = max(mtime_before + 1.0, time.time())
                    os.utime(self.path, (bump, bump))
                    mtime_after = self._mtime(self.path)
                if self.path.stat().st_size == 0:
                    raise PublishError("wrote a zero-byte strategy file - never submitted")

                confirmed, lu_after = self._await_confirmation(
                    lu_before, confirm_timeout, poll_interval
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        note = "" if confirmed else (
            f"NOT CONFIRMED after {confirm_timeout:g}s - .last-update did not advance. "
            f"The miner retries every ~11s on its own; do not write again. Alert."
        )
        return PublishResult(
            written=True, dry_run=False, path=self.path, text=text,
            mtime_before=mtime_before, mtime_after=mtime_after,
            last_update_before=lu_before, last_update_after=lu_after,
            confirmed=confirmed, report=report, note=note,
        )

    def _await_confirmation(
        self, before: float | None, timeout: float, interval: float
    ) -> tuple[bool, float | None]:
        """Poll ``.last-update``. It advances only on an API status <= 201."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            now = self._mtime(self.last_update)
            if now is not None and (before is None or now > before):
                return True, now
            time.sleep(interval)
        return False, self._mtime(self.last_update)

    # --- read-back -------------------------------------------------------

    def current(self) -> str | None:
        """What is on disk right now, if anything."""
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def touch(self) -> float:
        """Refresh the timestamp without changing weights - a free rebalance.

        Guide §3.4: inactivity is charged on a CALENDAR clock from the first idle day
        once your track is older than five sessions, so this runs every day including
        weekends. It still rebalances to target and pays fees on the drift, roughly
        0.5bp per refresh on a 25-name book.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            raise PublishError("nothing to refresh - no non-empty strategy file on disk")
        now = time.time()
        os.utime(self.path, (now, now))
        return now
