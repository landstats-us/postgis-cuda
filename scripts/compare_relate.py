#!/usr/bin/env python3
"""compare_relate.py truth.csv gpu.csv -> pair agreement and DE-9IM agreement (truth from ST_Relate)"""
import os, sys, pandas as pd
t = pd.read_csv(sys.argv[1], dtype={"sid": str}); g = pd.read_csv(sys.argv[2], dtype={"sid": str})
t = t.rename(columns={t.columns[3]: "relate"})
if os.environ.get("SELFJOIN"):
    g = g[g.landrecid.astype(str) < g.sid.astype(str)]
m = t.merge(g, on=["landrecid", "sid"], how="outer", suffixes=("_t", "_g"), indicator=True)
both = m[m._merge == "both"]
bad = both[both.relate_t != both.relate_g]
print(f"truth={len(t)} gpu={len(g)} missed={int((m._merge == 'left_only').sum())} extra={int((m._merge == 'right_only').sum())} "
      f"relate_mismatch={len(bad)} ({len(bad) / max(len(both), 1):.2e})")
if len(bad):
    print(bad.groupby(["relate_t", "relate_g"]).size().sort_values(ascending=False).head(8).to_string())
    bad.head(20)[["landrecid", "sid", "relate_t", "relate_g"]].to_csv(sys.argv[2] + ".mismatch.csv", index=False)
