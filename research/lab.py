"""Cross-sectional short-horizon mean-reversion lab. Leak-proof: SignalView only."""
from __future__ import annotations
import statistics
from sn88_replica.signals import load_panel, evaluate, null_signal, SignalView

PANEL = load_panel('/home/newuser/Documents/sn88/fixtures/panel-replication-2025-06-02_2026-08-31.json')

CASH = {"SGOV","BIL","JPST","BSV","VCSH","VGIT","GOVT","BND","IUSB","AGG",
        "IEF","BNDX","MBB","VCIT","MUB","VTEB"}
ETFISH = {"SPY","QQQ","IWM","DIA","VTI","VOO","TQQQ","TLT","LQD","GLD","IAU","SLV",
          "JEPI","JEPQ","VNQ","XLV","SCHD","XLF","XLE","XLK","XLY","XLP","XLI","XLU",
          "XLB","XLC","XLRE","EFA","EEM","IEFA","IEMG","VEA","VWO","SMH","ARKK","IBIT"}
_DV = PANEL.dollar_volume
LIQ = sorted((t for t in PANEL.names if t not in CASH), key=lambda t: (-_DV.get(t,0.0), t))
NONSTOCK_SECTORS = {"Equity", "Fixed Income", "Commodity", "Currency"}
def _is_stock(t):
    sec = PANEL.sector.get(t)
    return bool(sec) and sec not in NONSTOCK_SECTORS
LIQ_NOETF = [t for t in LIQ if _is_stock(t)]

_C = {"t": None}
def _fresh(view):
    if _C["t"] != view.t:
        _C.clear(); _C["t"] = view.t; _C["r"] = {}; _C["v"] = {}; _C["c"] = {}; _C["b"] = {}

def rets(view, t, n):
    _fresh(view); k=(t,n); r=_C["r"].get(k)
    if r is None: r = view.returns(t, n); _C["r"][k]=r
    return r

def cum(view, t, n):
    _fresh(view); k=(t,n); c=_C["c"].get(k)
    if c is None:
        o=1.0
        for x in rets(view,t,n): o*=1+x
        c=o-1.0; _C["c"][k]=c
    return c

def vol(view, t, n):
    _fresh(view); k=(t,n); v=_C["v"].get(k)
    if v is None:
        r=rets(view,t,n); m=len(r)
        v = (sum((x-sum(r)/m)**2 for x in r)/m)**0.5 if m>1 else 0.0
        _C["v"][k]=v
    return v

def dvol(view, t, n):
    r=[x for x in rets(view,t,n) if x<0]; m=len(r)
    return (sum((x-sum(r)/m)**2 for x in r)/m)**0.5 if m>1 else 0.0

def build(univ=100, horizon=5, skip=0, norm="vol", volwin=20,
          sector_neut=False, beta_neut=False, exclude_etf=False,
          vol_band=None, hit_min=None, min_hist=70, mkt="SPY",
          reverse=False, blend_lv=0.0):
    base = LIQ_NOETF if exclude_etf else LIQ
    universe = base[:univ]
    def sig(view: SignalView) -> dict[str, float]:
        if not view.enough_history(min_hist):
            return {t: _DV.get(t,0.0) for t in universe}
        pool = list(universe)
        if vol_band is not None:
            lo, hi = vol_band
            vs = sorted((vol(view,t,volwin), t) for t in pool)
            n = len(vs)
            pool = [t for i,(v,t) in enumerate(vs) if lo <= i/n < hi]
        if hit_min is not None:
            pool = [t for t in pool if view.hit_rate(t, 60) >= hit_min]
        if not pool:
            return {t: _DV.get(t,0.0) for t in universe}
        raw = {}
        m = cum(view, mkt, horizon+skip) if (beta_neut and mkt in PANEL.returns) else 0.0
        for t in pool:
            c = cum(view, t, horizon+skip)
            if skip:
                cs = cum(view, t, skip)
                c = (1+c)/(1+cs) - 1.0
            if beta_neut:
                c = c - view.beta(t, mkt, 60) * m
            v = vol(view, t, volwin)
            if norm == "vol":
                if v <= 1e-9: continue
                s = -c / v
            else:
                s = -c
            if reverse: s = -s
            if blend_lv:
                s = s - blend_lv * (v / 0.02)
            raw[t] = s
        if not raw:
            return {t: _DV.get(t,0.0) for t in universe}
        if sector_neut:
            g = {}
            for t in raw: g.setdefault(PANEL.sector.get(t,"?"), []).append(t)
            out = {}
            for _, ts in g.items():
                mu = sum(raw[t] for t in ts)/len(ts)
                for t in ts: out[t] = raw[t]-mu
            raw = out
        return raw
    return sig

def run(sig, name, **kw):
    return evaluate(sig, PANEL, name=name, max_windows=40, **kw)


# --- diagnostics: same walk-forward as evaluate(), but returns the daily series ---
from sn88_replica.signals import equal_weight_top, inverse_vol_top, WIN, INITIAL_EQUITY
from sn88_replica.scoring import raw_score

def walk(signal, panel=PANEL, start_frac=0.5, top_n=20, weighting="equal", warmup=60):
    n = len(panel.sessions)
    begin = max(warmup+1, int(n*start_frac))
    eq = INITIAL_EQUITY; curve=[]; w={}; steps=[]; turns=[]; sess=[]
    for t in range(begin, n-1):
        view = SignalView(panel, t)
        sc = signal(view)
        if sc:
            tgt = equal_weight_top(sc, top_n) if weighting=="equal" else inverse_vol_top(sc, view, top_n)
            nm = set(tgt)|set(w)
            turns.append(0.5*sum(abs(tgt.get(k,0.)-w.get(k,0.)) for k in nm))
            w = tgt
        if not w: continue
        step = sum(x*panel.returns[k][t-1] for k,x in w.items() if t-1 < len(panel.returns[k]))
        eq *= 1+step; curve.append(eq); steps.append(step); sess.append(panel.sessions[t])
    return steps, curve, turns, sess

def stats(steps):
    n=len(steps); m=sum(steps)/n
    sd=(sum((x-m)**2 for x in steps)/n)**0.5
    sk=(sum((x-m)**3 for x in steps)/n)/sd**3 if sd>0 else 0.0
    return dict(n=n, mean=m, vol=sd, sharpe_d=m/sd if sd else 0, sharpe_a=(m/sd*252**0.5) if sd else 0,
                skew=sk, hit=sum(1 for x in steps if x>0)/n, worst=min(steps), best=max(steps))

def scores_of(curve, max_windows=40):
    ends=[i for i in range(WIN, len(curve))]
    if max_windows and len(ends)>max_windows: ends=ends[::max(1,len(ends)//max_windows)]
    return [raw_score(INITIAL_EQUITY, curve[e-WIN:e], e).score for e in ends]


def build_combo(univ=50, horizons=(2,3,4,5), skip=0, volwin=20, rank=True,
                sector_neut=False, exclude_etf=False, min_hist=70, weights=None,
                mkt="SPY", beta_neut=False, dvol_norm=False):
    """Average the cross-sectional z (or rank) of -cum/vol over several horizons."""
    base = LIQ_NOETF if exclude_etf else LIQ
    universe = base[:univ]
    ws = weights or [1.0]*len(horizons)
    def sig(view: SignalView):
        if not view.enough_history(min_hist):
            return {t: _DV.get(t,0.0) for t in universe}
        m = {h: (cum(view, mkt, h+skip) if mkt in PANEL.returns else 0.0) for h in horizons} if beta_neut else None
        agg = {t: 0.0 for t in universe}
        ok = True
        for wgt, h in zip(ws, horizons):
            raw = {}
            for t in universe:
                c = cum(view, t, h+skip)
                if skip:
                    c = (1+c)/(1+cum(view,t,skip)) - 1.0
                if beta_neut:
                    c = c - view.beta(t, mkt, 60)*m[h]
                v = dvol(view,t,volwin) if dvol_norm else vol(view, t, volwin)
                if v <= 1e-9: continue
                raw[t] = -c/v
            if not raw: ok=False; break
            if sector_neut:
                g={}
                for t in raw: g.setdefault(PANEL.sector.get(t,"?"),[]).append(t)
                o={}
                for _,ts in g.items():
                    mu=sum(raw[t] for t in ts)/len(ts)
                    for t in ts: o[t]=raw[t]-mu
                raw=o
            if rank:
                order=sorted(raw,key=lambda t:(raw[t],t)); n=len(order)
                raw={t:(i/(n-1) if n>1 else 0.5) for i,t in enumerate(order)}
            else:
                vals=list(raw.values()); mu=sum(vals)/len(vals)
                sd=(sum((x-mu)**2 for x in vals)/len(vals))**0.5 or 1.0
                raw={t:(x-mu)/sd for t,x in raw.items()}
            for t,x in raw.items(): agg[t]=agg.get(t,0.0)+wgt*x
            for t in list(agg):
                if t not in raw: agg.pop(t,None)
        if not ok or not agg:
            return {t: _DV.get(t,0.0) for t in universe}
        return agg
    return sig
