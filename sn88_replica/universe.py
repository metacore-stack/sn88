"""Parse ``/assets``: the tradable universe, the cash list, and the delisting queue.

Three things live in that response and only one of them is obvious:

* **555 tradable tickers**, case-sensitive, ranked by market cap.
* **16 cash-classified tickers.** They fill normally but count toward ``cash``, which
  multiplies your score linearly (guide §3.5). It is a curated list, not a rule about
  bonds - TLT, LQD, GLD, IAU, SLV, JEPI, JEPQ and VNQ are all tradable and are NOT
  cash, so a duration or gold sleeve is a defensive posture at zero cash penalty.
* **A delisting list.** Holdings in a delisted market are force-converted to cash,
  which then feeds the same penalty.

An unsupported, delisted or misspelled ticker is **silently reclassified to cash** by
upstream. That is the single most likely silent failure in a live equity system,
because tickers change - so every symbol must be checked against today's snapshot
before a strategy file is written (guide §5.2, §21).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Universe", "parse_assets"]

_TICKER = re.compile(r"^[A-Z][A-Z.\-]{0,6}$")
_TAG = re.compile(r"<a [^>]*>|</a>")


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    sector: str
    price: float
    volume: float
    dollar_volume: float
    market_cap: float


@dataclass
class Universe:
    """Today's tradable set. Rebuild it every session before publishing."""

    assets: dict[str, Asset] = field(default_factory=dict)
    cash_tickers: frozenset[str] = frozenset()
    dereg_list: tuple[int, ...] = ()
    asset_ratio: tuple[float, ...] = ()
    as_of: str | None = None

    def __len__(self) -> int:
        return len(self.assets)

    def is_tradable(self, ticker: str) -> bool:
        """Case-sensitive, exactly as upstream matches."""
        return ticker in self.assets

    def is_cash(self, ticker: str) -> bool:
        """True if the ticker counts toward the cash penalty despite being tradable."""
        return ticker in self.cash_tickers

    def would_become_cash(self, ticker: str) -> bool:
        """Upstream drops unknown tickers, and the leftover is recomputed as cash."""
        return (not self.is_tradable(ticker)) or self.is_cash(ticker)

    def most_liquid(self, n: int = 20) -> list[Asset]:
        return sorted(self.assets.values(), key=lambda a: -a.dollar_volume)[:n]

    def by_sector(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for a in self.assets.values():
            out.setdefault(a.sector, []).append(a.ticker)
        return out


def parse_assets(text: str, *, as_of: str | None = None) -> Universe:
    """Parse the ``/assets`` payload.

    The response is a header block followed by an HTML-ish table. Both are parsed
    defensively - a shape change upstream should raise, not silently yield an empty
    universe that then reclassifies your whole book to cash.
    """
    clean = _TAG.sub("", text)

    ratio: tuple[float, ...] = ()
    m = re.search(r"asset ratio:\s*\[([^\]]*)\]", clean)
    if m:
        ratio = tuple(float(x) for x in m.group(1).split(",") if x.strip())

    dereg: tuple[int, ...] = ()
    m = re.search(r"dereg list:\s*\[([^\]]*)\]", clean)
    if m:
        dereg = tuple(int(x) for x in m.group(1).split(",") if x.strip())

    cash: frozenset[str] = frozenset()
    m = re.search(r"cash ETFs:\s*\[([^\]]*)\]", clean)
    if m:
        cash = frozenset(x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip())

    # The table is FIXED-WIDTH with right-aligned columns, not delimiter-separated.
    # Neither .split() nor splitting on runs of spaces works: `name` and `sector` both
    # contain single spaces (so .split() merges them into one useless field, making
    # every row's "sector" unique and a max_sector constraint silently inert), and long
    # names butt straight up against the neighbouring column with no gap at all - DIA,
    # HLN, TAK, TEVA and ZTS each lose a boundary that way. Slice on the header's own
    # column offsets instead.
    header = next((ln for ln in clean.splitlines() if ln.lstrip().startswith("netuid")), None)
    if header is None:
        raise ValueError("/assets has no column header - shape changed, do not publish")
    bounds: list[int] = []
    for label in ("netuid", "name", "sector", "price", "volume", "pv"):
        idx = header.find(label, bounds[-1] if bounds else 0)
        if idx < 0:
            raise ValueError(f"/assets header is missing the {label!r} column")
        bounds.append(idx + len(label))

    assets: dict[str, Asset] = {}
    for line in clean.splitlines():
        if len(line) < bounds[-1]:
            continue
        cells = [line[:bounds[0]]]
        cells += [line[bounds[i - 1]:bounds[i]] for i in range(1, len(bounds))]
        cells.append(line[bounds[-1]:])
        cells = [c.strip() for c in cells]
        ticker = cells[0]
        if not _TICKER.match(ticker):
            continue
        try:
            price, vol, pv, mc = (float(cells[i]) for i in (3, 4, 5, 6))
        except (ValueError, IndexError):
            continue
        assets[ticker] = Asset(ticker, cells[1], cells[2], price, vol, pv, mc)

    if not assets:
        raise ValueError("parsed zero tradable assets - /assets shape changed, do not publish")
    if not cash:
        raise ValueError("parsed zero cash tickers - /assets shape changed, do not publish")

    return Universe(
        assets=assets,
        cash_tickers=cash,
        dereg_list=dereg,
        asset_ratio=ratio,
        as_of=as_of,
    )
