#!/usr/bin/env python3
"""compare.py truth.csv gpu.csv  -> pair agreement + area error stats (square degrees)"""
import sys, numpy as np, pandas as pd
t = pd.read_csv(sys.argv[1], dtype={"sid": str}); g = pd.read_csv(sys.argv[2], dtype={"sid": str})
if __import__("os").environ.get("SELFJOIN"):
    g = g[g.landrecid.astype(str) < g.sid.astype(str)]   # uuid order == lowercase hex text order
key = lambda d: d.landrecid.astype(str) + "|" + d.sid.astype(str)
t["k"] = key(t); g["k"] = key(g)
t = t.set_index("k"); g = g.set_index("k")
common = t.index.intersection(g.index)
fn = len(t) - len(common); fp = len(g) - len(common)
print(f"truth pairs={len(t)} gpu pairs={len(g)} common={len(common)} missed(FN)={fn} extra(FP)={fp}")
ta = t.loc[common, "area"].to_numpy(); ga = g.loc[common, "area"].to_numpy()
err = ga - ta
print(f"area sum truth={ta.sum():.9g} gpu={ga.sum():.9g} rel={(ga.sum()-ta.sum())/ta.sum():.3e}")
big = ta > 1e-12
rel = np.abs(err[big]) / ta[big]
print(f"pairs with area>1e-12: {big.sum()}  abs err max={np.abs(err).max():.3e}  rel err median={np.median(rel):.3e} p99={np.percentile(rel,99):.3e} max={rel.max():.3e}")
lat = float(__import__("os").environ.get("LAT", "41.66"))  # sample latitude for deg^2 -> m^2 (approx.)
m2 = 111319.49**2 * np.cos(np.radians(lat))
ae = np.abs(err) * m2
print(f"  abs err m^2: median={np.median(ae):.3e} p99={np.percentile(ae,99):.3e} max={ae.max():.3e}")
print(f"  >1e-3 rel: {(rel>1e-3).sum()}   >1e-6 rel: {(rel>1e-6).sum()}")
# extra/missing pairs that matter (truth area > 0)
if fn:
    miss = t.loc[t.index.difference(g.index)]
    print(f"missed pairs with truth area>0: {(miss.area>0).sum()}  (touching-only: {(miss.area<=0).sum()})")
if fp:
    ex = g.loc[g.index.difference(t.index)]
    print(f"extra pairs: area>1e-12: {(ex.area>1e-12).sum()}")
