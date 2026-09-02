import sys, math, statistics
sys.path.insert(0, "/home/newuser/Documents/sn88")
from sn88_replica.signals import load_panel, evaluate, SignalView

PANEL = load_panel("/home/newuser/Documents/sn88/fixtures/panel-replication-2025-06-02_2026-08-31.json")
CASH = {"SGOV","BIL","JPST","BSV","VCSH","VGIT","GOVT","BND","IUSB","AGG","IEF","BNDX","MBB","VCIT","MUB","VTEB"}
DV = PANEL.dollar_volume
NONCASH = [t for t in PANEL.names if t not in CASH]
LIQ = sorted(NONCASH, key=lambda t: -DV.get(t, 0.0))
SEC = PANEL.sector

def fast_vol(r):
    n = len(r)
    if n < 2: return 0.0
    m = sum(r) / n
    return math.sqrt(sum((x - m) ** 2 for x in r) / n)

def run(fn, name, **kw):
    return evaluate(fn, PANEL, name=name, max_windows=kw.pop("max_windows", 40), **kw)
