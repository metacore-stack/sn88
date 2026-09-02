"""Build, serialise and validate a strategy file - predicting exactly what upstream does.

Upstream's parser fails silently in four different ways (guide §5.2). This module
reproduces its semantics so the publisher can refuse a file that would be discarded or
would mean something other than what you intended, instead of finding out from a zero
score two sessions later.

The sanitiser is the sharp edge::

    st['strat'] = st['strat'].str.replace(r'''[^{'\\w":.,}\\[\\]\\*=+-]''', '', regex=True)

Whitespace, parentheses and ``#`` are all outside that class. Consequences, each
verified against the real regex:

* ``{'_':1,'AAPL':0.15}   # top pick`` -> ``{'_':1,'AAPL':0.15}toppick`` -> ``SyntaxError``
  -> upstream's ``except: continue`` -> **the whole file is silently discarded** and your
  previous allocation stays live while the inactivity clock keeps running.
* ``{'_': 1, 'AAPL': (0.1+0.2)}`` -> ``{'_':1,'AAPL':0.1+0.2}`` -> evaluates fine but to
  ``0.30000000000000004``. An expression does not fail; it silently **changes meaning**.
* Newlines and spaces vanish harmlessly, so pretty-printing is safe.
* ``BRK.B`` and ``BF-B`` survive - ``.`` and ``-`` are inside the class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .constants import ASSET_STOCKS
from .universe import Universe

__all__ = [
    "Strategy",
    "ValidationError",
    "ValidationReport",
    "sanitize",
    "serialize",
    "parse_like_upstream",
    "predict_acceptance",
]

# byte-for-byte the upstream sanitiser
SANITIZER = re.compile(r"""[^{'\w":.,}\[\]\*=+-]""")

RESERVED = ("_", "*", "=", "-+")


class ValidationError(Exception):
    """A strategy that must never reach the strategy directory."""


@dataclass(frozen=True)
class Strategy:
    """A US-equity allocation. ``'_': 1`` is added on serialisation, never by you."""

    weights: Mapping[str, float]
    asset_class: int = ASSET_STOCKS

    def gross(self) -> float:
        """``sum(|w|)`` over non-reserved keys - what upstream's ``asum`` computes."""
        return sum(abs(v) for k, v in self.weights.items() if k not in RESERVED)

    def implied_cash(self) -> float:
        """The leftover upstream will assign to ``''``. Your explicit value is ignored."""
        return 1.0 - self.gross()

    def shorts(self) -> dict[str, float]:
        return {k: v for k, v in self.weights.items() if v < 0 and k not in RESERVED}

    def tickers(self) -> list[str]:
        return [k for k in self.weights if k not in RESERVED and k != ""]


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    effective_cash: float = 0.0
    reclassified: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_bad(self) -> "ValidationReport":
        if self.errors:
            raise ValidationError("; ".join(self.errors))
        return self


def sanitize(text: str) -> str:
    """Apply upstream's sanitiser. Use this to prove a round-trip before publishing."""
    return SANITIZER.sub("", text)


def serialize(strategy: Strategy) -> str:
    """Render a strategy file that survives the sanitiser unchanged in meaning.

    Deliberately conservative: no comments, no expressions, no exponent notation, keys
    sorted with ``'_'`` first so diffs are readable.
    """
    parts = [f"'_':{int(strategy.asset_class)}"]
    for ticker in sorted(strategy.tickers()):
        w = strategy.weights[ticker]
        parts.append(f"'{ticker}':{w:.10f}".rstrip("0").rstrip("."))
    return "{" + ",".join(parts) + "}"


def parse_like_upstream(text: str) -> dict | None:
    """Reproduce ``initfund``'s parse. Returns ``None`` where upstream would ``continue``.

    ``None`` means **silently discarded**: no error surfaces anywhere, and your previous
    allocation stays live.
    """
    cleaned = sanitize(text)
    try:
        value = eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307 - mirroring upstream
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    asset = value.get("_", 0)
    if asset not in (0, 1):                    # `an = 2`; '_':2 drops the strategy
        return None
    try:
        asum = sum(abs(value[k]) for k in value if k not in RESERVED)
    except Exception:
        return None
    if asum > 1:                               # NO LEVERAGE - whole file dropped
        return None
    return value


@dataclass(frozen=True)
class Acceptance:
    accepted: bool
    reason: str
    parsed: dict | None
    effective_cash: float
    reclassified: tuple[str, ...]


def predict_acceptance(text: str, universe: Universe) -> Acceptance:
    """What will upstream actually do with this file?

    Answers the only question that matters at publish time: accepted as written,
    accepted but meaning something else, or silently thrown away.
    """
    parsed = parse_like_upstream(text)
    if parsed is None:
        return Acceptance(False, "silently discarded (unparseable, wrong class, or gross > 1)",
                          None, 1.0, ())

    reclassified = tuple(
        t for t in parsed
        if t not in RESERVED and t != "" and universe.would_become_cash(t)
    )
    live = sum(
        abs(v) for k, v in parsed.items()
        if k not in RESERVED and k != "" and not universe.would_become_cash(k)
    )
    return Acceptance(
        accepted=True,
        reason="accepted",
        parsed=parsed,
        effective_cash=max(0.0, 1.0 - live),
        reclassified=reclassified,
    )


def validate(
    strategy: Strategy,
    universe: Universe,
    *,
    max_gross: float = 1.0,
    min_gross: float = 0.99,
    max_cash: float = 0.01,
    allow_shorts: bool = False,
    max_single_name: float = 0.10,
    min_holdings: int = 1,
    max_holdings: int = 40,
) -> ValidationReport:
    """Every check that must pass before a file is written. Guide §15, §16.

    Defaults mirror ``config/system.yaml``: full invested exposure, no shorts in v1,
    and a cash band that keeps the linear cash tax near zero.
    """
    r = ValidationReport()

    if strategy.asset_class != ASSET_STOCKS:
        r.errors.append(
            f"asset_class={strategy.asset_class}; '_':1 is mandatory for US equities and "
            f"omitting it silently selects the OTHER class"
        )

    gross = strategy.gross()
    eps = 1e-9
    # NO tolerance on the upper bound: upstream is `if asum > 1: continue` with no
    # rounding slack at all, so a vector summing to 1.0000000000000002 in float is
    # discarded outright. Being more permissive here would publish files that vanish.
    if gross > max_gross:
        r.errors.append(
            f"gross exposure {gross!r} > {max_gross}: upstream would SILENTLY DISCARD "
            f"this file (no float tolerance) and keep your previous allocation live"
        )
    elif gross < min_gross - eps:
        r.errors.append(
            f"gross exposure {gross:.6f} < {min_gross}: leftover becomes cash and the "
            f"cash multiplier is linear ({(1 - (1 - gross)):.4f}x on score)"
        )

    if not allow_shorts and strategy.shorts():
        r.errors.append(
            f"shorts present ({sorted(strategy.shorts())}): they count toward the cash "
            f"penalty AND consume gross budget"
        )

    tickers = strategy.tickers()
    if len(tickers) < min_holdings:
        r.errors.append(f"{len(tickers)} holdings < min {min_holdings}")
    if len(tickers) > max_holdings:
        r.errors.append(f"{len(tickers)} holdings > max {max_holdings}")

    unknown = [t for t in tickers if not universe.is_tradable(t)]
    if unknown:
        r.errors.append(
            f"not in today's /assets: {sorted(unknown)} - upstream silently reclassifies "
            f"these to cash"
        )

    cashish = [t for t in tickers if universe.is_cash(t)]
    if cashish:
        weight = sum(abs(strategy.weights[t]) for t in cashish)
        r.reclassified = sorted(cashish)
        msg = (f"cash-classified holdings {sorted(cashish)} carry {weight:.4f} weight and "
               f"count toward the cash penalty")
        (r.errors if weight > max_cash + eps else r.warnings).append(msg)

    for t in tickers:
        w = abs(strategy.weights[t])
        if w > max_single_name + eps:
            r.errors.append(f"{t} weight {w:.4f} > max_single_name {max_single_name}")

    if "" in strategy.weights:
        r.warnings.append(
            "explicit '' cash key is IGNORED by upstream - cash is recomputed as the "
            "leftover, so this value only consumes gross budget"
        )

    text = serialize(strategy)
    if sanitize(text) != text:
        r.errors.append("serialised form does not survive the upstream sanitiser unchanged")
    round_trip = parse_like_upstream(text)
    if round_trip is None:
        r.errors.append("serialised form would be SILENTLY DISCARDED by upstream")
    else:
        for t in tickers:
            if abs(round_trip.get(t, 0.0) - strategy.weights[t]) > 1e-9:
                r.errors.append(f"round-trip changed {t}: "
                                f"{strategy.weights[t]} -> {round_trip.get(t)}")

    live = sum(abs(w) for t, w in strategy.weights.items()
               if t not in RESERVED and t != "" and not universe.would_become_cash(t))
    r.effective_cash = max(0.0, 1.0 - live)
    if r.effective_cash > max_cash + eps and not any("cash-classified" in e for e in r.errors):
        r.errors.append(
            f"effective cash {r.effective_cash:.4f} > {max_cash} "
            f"(score multiplier {1 - r.effective_cash:.4f}x)"
        )

    for t in tickers:
        a = universe.assets.get(t)
        if a and a.dollar_volume <= 0:
            r.warnings.append(f"{t} reports zero dollar volume in today's snapshot")

    return r
