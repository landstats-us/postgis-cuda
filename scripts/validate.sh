#!/usr/bin/env bash
# Run the GPU engine against every PostGIS ground-truth set and summarize agreement.
# usage: validate.sh [quick]   (quick = Johnson County only)
cd "$(dirname "$0")/.."
H="./build/harness"; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
out=$(mktemp -p data gpu_val.XXXX.csv)
check() {  # name truth lat selfjoin -- harness args
  local name=$1 truth=$2 lat=$3 sj=$4; shift 5
  local t; t=$($H "$@" --reps 2 --out "$out" | grep -oP 'traverse=\S+ refine=\S+')
  local r; r=$(SELFJOIN=$sj LAT=$lat python3 scripts/compare.py "$truth" "$out" 2>/dev/null)
  local pairs; pairs=$(grep -oP 'missed\(FN\)=\d+ extra\(FP\)=\d+' <<<"$r")
  local arel; arel=$(grep -oP 'area sum .* rel=\K\S+' <<<"$r")
  local amax; amax=$(grep -oP 'abs err m\^2: .*max=\K\S+' <<<"$r")
  printf "%-44s %-28s area_rel=%-11s max_err_m2=%-10s %s\n" "$name" "$pairs" "$arel" "$amax" "$t"
}
cluster() {  # name truth gset: GPU ST_ClusterIntersecting vs ST_ClusterIntersectingWin partition
  local name=$1 truth=$2 g=$3
  local t; t=$($H "$g" "$g" --cluster --dedup --subdiv 256 --reps 2 --out "$out" | grep -oP 'traverse=\S+ refine=\S+')
  local r; r=$(python3 - "$truth" "$out" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1], header=None, names=["id", "ct"]); g = pd.read_csv(sys.argv[2])
m = t.merge(g, on="id"); a = m.groupby("ct").cluster.nunique(); b = m.groupby("cluster").ct.nunique()
print(f"clusters truth={t.ct.nunique()} gpu={g.cluster.nunique()} identical_partition={bool((a == 1).all() and (b == 1).all())}")
PY
)
  printf "%-44s %-62s %s\n" "$name" "$r" "$t"
}
pairs() {  # name truth selfjoin -- harness args: pair sets only (dwithin / buffered intersects)
  local name=$1 truth=$2 sj=$3; shift 4
  local t; t=$($H "$@" --reps 2 --out "$out" | grep -oP 'traverse=\S+ refine=\S+')
  local r; r=$(SELFJOIN=$sj python3 scripts/compare.py "$truth" "$out" 2>/dev/null | grep -oP 'truth pairs=\d+ .*')
  printf "%-44s %-62s %s\n" "$name" "$r" "$t"
}
relate() {  # name truth selfjoin -- harness args: GPU DE-9IM vs ST_Relate
  local name=$1 truth=$2 sj=$3; shift 4
  local t; t=$($H "$@" --reps 1 --out "$out" | grep -oP 'traverse=\S+ refine=\S+')
  local r; r=$(SELFJOIN=$sj python3 scripts/compare_relate.py "$truth" "$out" | head -1)
  printf "%-44s %-62s %s\n" "$name" "$r" "$t"
}
check "19103 parcels x soils"            data/truth_soils.csv 41.66 "" -- data/parcels_19103.gset data/soils_19103.gset --area
check "19103 parcels x soils subdiv64"   data/truth_soils.csv 41.66 "" -- data/parcels_19103.gset data/soils_19103.gset --area --subdiv 64
check "19103 parcels x parcels dedup"    data/truth_pp_19103.csv 41.66 1 -- data/parcels_19103.gset data/parcels_19103.gset --area --dedup
check "19103 parcels x parcels dd sd256"  data/truth_pp_19103.csv 41.66 1 -- data/parcels_19103.gset data/parcels_19103.gset --area --dedup --subdiv 256
cluster "19103 ST_ClusterIntersecting sd256" data/truth_cl_19103.csv data/parcels_19103.gset
relate "19103 ST_Relate parcels x parcels sd256" data/truth_rel_pp_19103.csv 1 -- data/parcels_19103.gset data/parcels_19103.gset --relate --dedup --subdiv 256
relate "19103 ST_Relate parcels x county sd64" data/truth_rel_pc_19103.csv "" -- data/parcels_19103.gset data/counties.gset --relate --dedup --subdiv 64
check "19103 parcels x county dd sd64"   data/truth_pc_19103.csv 41.66 "" -- data/parcels_19103.gset data/counties.gset --area --dedup --subdiv 64
pairs "19103 ST_DWithin pp 5 m sd256"      data/truth_dw5_pp_19103.csv 1 -- data/parcels_19103.gset data/parcels_19103.gset --dwithin 5 --dedup --subdiv 256
pairs "19103 ST_DWithin pc 1 km sd64"      data/truth_dw1000_pc_19103.csv "" -- data/parcels_19103.gset data/counties.gset --dwithin 1000 --dedup --subdiv 64
pairs "19103 buffer(-1) x buffer(-1) pp"   data/truth_er1_pp_19103.csv 1 -- data/parcels_19103.gset data/parcels_19103.gset --erode 1 --dedup --subdiv 256
[ "$1" = quick ] && { rm -f "$out"; exit 0; }
check "06037 parcels x soils subdiv64"   data/truth_soils_06037.csv 34.2 "" -- data/parcels_06037.gset data/soils_06037.gset --area --subdiv 64
check "06037 parcels x county dd sd64"   data/truth_pc_06037.csv 34.2 "" -- data/parcels_06037.gset data/counties.gset --area --dedup --subdiv 64
check "06037 parcels x parcels dedup"    data/truth_pp_06037.csv 34.2 1 -- data/parcels_06037.gset data/parcels_06037.gset --area --dedup
check "06037 parcels x parcels dd sd256"  data/truth_pp_06037.csv 34.2 1 -- data/parcels_06037.gset data/parcels_06037.gset --area --dedup --subdiv 256
check "06037 addresses x parcels"         data/truth_pt_06037.csv 34.2 "" -- data/addr_06037.pset data/parcels_06037.gset --dedup --subdiv-b 256
relate "06037 ST_Relate parcels x parcels sd256" data/truth_rel_pp_06037.csv 1 -- data/parcels_06037.gset data/parcels_06037.gset --relate --dedup --subdiv 256
relate "06037 ST_Relate parcels x county sd64" data/truth_rel_pc_06037.csv "" -- data/parcels_06037.gset data/counties.gset --relate --dedup --subdiv 64
relate "06037 ST_Relate addresses x parcels" data/truth_rel_pt_06037.csv "" -- data/addr_06037.pset data/parcels_06037.gset --dedup --subdiv-b 256
cluster "06037 ST_ClusterIntersecting sd256" data/truth_cl_06037.csv data/parcels_06037.gset
# the 21 pairs PostGIS reports for pp dwithin but the GPU does not are 7-12 m apart: ST_DWithin(geography) gives
# them in this join, but not when called on the pair alone (_ST_DWithinUncached, UTM distance agree)
pairs "06037 ST_DWithin pp 5 m sd256"      data/truth_dw5_pp_06037.csv 1 -- data/parcels_06037.gset data/parcels_06037.gset --dwithin 5 --dedup --subdiv 256
pairs "06037 ST_DWithin addresses 5 m"     data/truth_dw5_pt_06037.csv "" -- data/addr_06037.pset data/parcels_06037.gset --dwithin 5 --dedup --subdiv-b 256
pairs "06037 buffer(-1) x buffer(-1) pp"   data/truth_er1_pp_06037.csv 1 -- data/parcels_06037.gset data/parcels_06037.gset --erode 1 --dedup --subdiv 256
pairs "06037 addresses in buffer(-1)"      data/truth_er1_pt_06037.csv "" -- data/addr_06037.pset data/parcels_06037.gset --erode 1 --dedup --subdiv-b 256
rm -f "$out"
