#!/usr/bin/env python3
"""End-to-end benchmarks with the one-time layer load included.

Loads every benchmark dataset into a running gpujoind straight from Postgres (its --conninfo), then runs each
workload on those layers in stats mode (no row delivery) and reports load time, query time and their sum.

  GSP_TOKEN=... bench_amortized.py SOCKET [--keep]

Layers are named b_* and dropped at the end unless --keep.
"""
import json, os, re, subprocess, sys

sock = sys.argv[1]
keep = "--keep" in sys.argv
cat = os.path.join(os.path.dirname(__file__), "..", "build", "gsp_cat.next")
LA_ENV = "st_makeenvelope(-118.944703, 32.7952055, -117.646386, 34.8233041, 4326)"
JO_ENV = "st_makeenvelope(-91.83406181721983, 41.42324126132862, -91.36588056341255, 41.86194550524961, 4326)"

def call(*args):
    out = subprocess.run([cat, sock, *map(str, args)], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr.strip())
    first, js = out.stdout.strip().split("\n", 1)
    wall = float(re.search(r"time=([0-9.]+)", first).group(1))
    return wall, json.loads(js)

LAYERS = {   # name: (op, args...) -- LOAD = 1 (table), LOAD_QUERY = 2
    "b_parcels_19103": (1, "combine.parcel_enriched_19103", "landrecid", "geom", "", 256, 1),
    "b_parcels_06037": (1, "combine.parcel_enriched_06037", "landrecid", "geom", "", 256, 1),
    "b_counties": (1, "tiger_data.county_4326", "gid", "geom", "", 64, 1),
    "b_soils_19103": (2, f"select ctid::text, geom from insights.nrcs_ssurgo_mupolygon where geom && {JO_ENV}", 64, 1),
    "b_soils_06037": (2, f"select ctid::text, geom from insights.nrcs_ssurgo_mupolygon where geom && {LA_ENV}", 64, 1),
    "b_addr_06037": (2, f"select id, geom from combine.overturemaps_address where geom && {LA_ENV}", 0, 0),
}
# (label, layers used, op, args, PostGIS reference)
WORK = [
    ("Johnson parcels x soils (area)", ["b_parcels_19103", "b_soils_19103"], 6, ("b_parcels_19103", "b_soils_19103", 1, "all", "stats"), "1.45 s (48 conns)"),
    ("Johnson parcels x parcels (area)", ["b_parcels_19103"], 6, ("b_parcels_19103", "b_parcels_19103", 1, "distinct", "stats"), "2.16 s (48 conns)"),
    ("Johnson parcels x counties (area)", ["b_parcels_19103", "b_counties"], 6, ("b_parcels_19103", "b_counties", 1, "all", "stats"), "0.27 s (48 conns)"),
    ("Johnson ST_ClusterIntersecting", ["b_parcels_19103"], 7, ("b_parcels_19103",), "1.83 s (1 core)"),
    ("Johnson parcels x parcels dwithin 5 m", ["b_parcels_19103"], 9, ("b_parcels_19103", "b_parcels_19103", "dwithin", 5, 0, "distinct", "stats"), "76.2 s (48 conns)"),
    ("Johnson parcels x counties dwithin 1 km", ["b_parcels_19103", "b_counties"], 9, ("b_parcels_19103", "b_counties", "dwithin", 1000, 0, "all", "stats"), "0.41 s (48 conns)"),
    ("Johnson buffer(-1) x buffer(-1)", ["b_parcels_19103"], 9, ("b_parcels_19103", "b_parcels_19103", "buffer", -1, -1, "distinct", "stats"), "6.6 s (1 conn)"),
    ("LA parcels x soils (area)", ["b_parcels_06037", "b_soils_06037"], 6, ("b_parcels_06037", "b_soils_06037", 1, "all", "stats"), "18.0 s (48 conns)"),
    ("LA parcels x counties (area)", ["b_parcels_06037", "b_counties"], 6, ("b_parcels_06037", "b_counties", 1, "all", "stats"), "17.7 s (48 conns)"),
    ("LA parcels x parcels (area)", ["b_parcels_06037"], 6, ("b_parcels_06037", "b_parcels_06037", 1, "distinct", "stats"), "61.5 s (48 conns)"),
    ("LA addresses x parcels", ["b_parcels_06037", "b_addr_06037"], 6, ("b_addr_06037", "b_parcels_06037", 0, "all", "stats"), "13.3 s (48 conns)"),
    ("LA ST_ClusterIntersecting", ["b_parcels_06037"], 7, ("b_parcels_06037",), "32.5 s (1 core)"),
    ("LA parcels x parcels dwithin 5 m", ["b_parcels_06037"], 9, ("b_parcels_06037", "b_parcels_06037", "dwithin", 5, 0, "distinct", "stats"), "79.3 s (48 conns)"),
    ("LA addresses x parcels dwithin 5 m", ["b_parcels_06037", "b_addr_06037"], 9, ("b_addr_06037", "b_parcels_06037", "dwithin", 5, 0, "all", "stats"), "21.0 s (48 conns)"),
    ("LA buffer(-1) x buffer(-1)", ["b_parcels_06037"], 9, ("b_parcels_06037", "b_parcels_06037", "buffer", -1, -1, "distinct", "stats"), "14.8 s (48 conns)"),
    ("LA addresses in buffer(parcel, -1)", ["b_parcels_06037", "b_addr_06037"], 9, ("b_parcels_06037", "b_addr_06037", "buffer", -1, 0, "all", "stats"), "12.1 s (48 conns)"),
]

load = {}
for name, spec in LAYERS.items():
    wall, js = call(spec[0], name, *spec[1:])
    load[name] = wall
    print(f"load {name:18s} {js['rows']:>10,} rows  {wall:7.2f} s  (copy {js['copy_s']:.2f} s, prepare {js['prepare_s']:.2f} s, "
          f"upload {js['upload_s']:.2f} s, {js['gpu_mb']:.0f} MB)", flush=True)

print()
print(f"| {'Workload':42s} | {'Rows':>12s} | {'Load':>7s} | {'Query':>8s} | {'Load + query':>12s} | PostGIS |")
print(f"|{'-' * 44}|{'-' * 14}|{'-' * 9}|{'-' * 10}|{'-' * 14}|---------|")
for label, used, op, args, pg in WORK:
    wall, js = call(op, *args)
    lt = sum(load[u] for u in used)
    rows = js.get("rows", 0)
    print(f"| {label:42s} | {rows:>12,} | {lt:6.2f}s | {wall * 1000:6.0f}ms | {lt + wall:11.2f}s | {pg} |", flush=True)

if not keep:
    for name in LAYERS:
        call(3, name)
