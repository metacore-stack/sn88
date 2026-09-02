"""Cross-sectional short-horizon mean-reversion lab (family: buy the recent losers)."""
from __future__ import annotations
import statistics, math, itertools, sys, time
from sn88_replica.signals import load_panel, evaluate, null_signal, SignalView

PANEL = load_panel('fixtures/panel-replication-2025-06-02_2026-08-31.json')

CASH = {"SGOV","BIL","JPST","BSV","VCSH","VGIT","GOVT","BND","IUSB","AGG",
        "IEF","BNDX","MBB","VCIT","MUB","VTEB"}

# static liquidity ranking (metadata, not future data)
_DV = PANEL.dollar_volume
LIQ_RANK = sorted((t for t in PANEL.names if t not in CASH),
                  key=lambda t: (-_DV.get(t, 0.0), t))

# --- per-view caches keyed by view.t ---------------------------------------
_cache_t = {"t": None, "ret": {}, "cum": {}, "vol": {}}

def _ret(view, t, n):
    if _cache_t["t"] != view.t:
        _cache_t.update(t=view.t, ret={}, cum={}, vol={})
    key = (t, n)
    r = _cache_t["ret"].get(key)
    if r is None:
        r = view.returns(t, n)
        _cache_t["ret"][key] = r
    return r

def cum(view, t, n):
    r = _ret(view, t, n)
    o = 1.0
    for x in r:
        o *= 1 + x
    return o - 1.0

def _pstdev(r):
    n = len(r)
    if n < 2:
        return 0.0
    m = sum(r) / n
    return (sum((x - m) ** 2 for x in r) / n) ** 0.5

def vol(view, t, n):
    key = ("v", t, n)
    if _cache_t["t"] != view.t:
        _cache_t.update(t=view.t, ret={}, cum={}, vol={})
    v = _cache_t["vol"].get(key)
    if v is None:
        v = _pstdev(_ret(view, t, n))
        _cache_t["vol"][key] = v
    return v

def make_mr(horizon=5, univ=100, norm="vol", volwin=20, sector_neut=False,
            skip=0, exclude_cash=True, mktname="SPY", beta_neut=False,
            vol_cap=None, min_hist=70):
    """norm: 'raw' | 'vol' | 'rank'"""
    universe = [t for t in LIQ_RANK[:univ]]
    def sig(view: SignalView) -> dict[str, float]:
        if not view.enough_history(min_hist):
            return {t: _DV.get(t, 0.0) for t in universe}
        raw = {}
        for t in universe:
            c = cum(view, t, horizon + skip)
            if skip:
                c2 = cum(view, t, skip)
                c = (1 + c) / (1 + c2) - 1.0
            v = vol(view, t, volwin)
            if v <= 1e-9:
                continue
            if vol_cap is not None and v > vol_cap:
                continue
            if norm == "vol":
                raw[t] = -c / v
            elif norm == "raw":
                raw[t] = -c
            else:
                raw[t] = -c
        if not raw:
            return {t: _DV.get(t, 0.0) for t in universe}
        if norm == "rank":
            order = sorted(raw, key=lambda t: raw[t])
            raw = {t: i for i, t in enumerate(order)}
        if beta_neut:
            m = cum(view, mktname, horizon + skip) if mktname in PANEL.returns else 0.0
            raw2 = {}
            for t in raw:
                b = view.beta(t, mktname, 60)
                c = cum(view, t, horizon + skip)
                v = vol(view, t, volwin)
                raw2[t] = -(c - b * m) / (v if norm == "vol" and v > 1e-9 else 1.0)
            raw = raw2
        if sector_neut:
            groups = {}
            for t in raw:
                groups.setdefault(PANEL.sector.get(t, "?"), []).append(t)
            out = {}
            for g, ts in groups.items():
                mu = sum(raw[t] for t in ts) / len(ts)
                for t in ts:
                    out[t] = raw[t] - mu
            raw = out
        return raw
    return sig

def run(sig, name, **kw):
    return evaluate(sig, PANEL, name=name, max_windows=40, **kw)
