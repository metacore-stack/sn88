import sys, math, statistics, itertools, json
sys.path.insert(0, "/home/newuser/Documents/sn88")
from sn88_replica.signals import load_panel, evaluate, SignalView

PANEL = load_panel("/home/newuser/Documents/sn88/fixtures/panel-replication-2025-06-02_2026-08-31.json")

CASH = {"SGOV","BIL","JPST","BSV","VCSH","VGIT","GOVT","BND","IUSB","AGG","IEF","BNDX","MBB","VCIT","MUB","VTEB"}

def run(fn, name, **kw):
    r = evaluate(fn, PANEL, name=name, max_windows=40, **kw)
    return r
