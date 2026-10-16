#!/usr/bin/env bash
# PostGIS CPU scaling for the county sample: parcels(geoid=$GEOID) x layer, st_intersects + st_area(st_intersection).
# Mode A: in-SQL parallel query (Gather) with N workers. Mode B: N concurrent connections over hash shards.
# usage: bench_postgis.sh soils|wetlands [workers...]
set -uo pipefail
layer=$1; shift
case $layer in soils) tbl=insights.nrcs_ssurgo_mupolygon;; wetlands) tbl=insights.usfws_nwi_wetlands;; esac
GEOID=${GEOID:-19103}
W=${*:-"0 1 2 4 8 16 32 48"}
PSQL="psql -h graybase -U claude -d landstats -w -X -q -At"
Q="from combine.parcel_enriched p join $tbl s on st_intersects(p.geom, s.geom) where p.geoid='$GEOID' and p.geom is not null"
echo "mode,workers,launched,ms,pairs,area_sum"
for w in $W; do
  out=$($PSQL <<SQL
set max_parallel_workers_per_gather=$w; set parallel_setup_cost=0; set parallel_tuple_cost=0;
set min_parallel_table_scan_size=0; set min_parallel_index_scan_size=0; set statement_timeout='30min';
set enable_seqscan=off;  -- otherwise the planner drives from a parallel seq scan of the whole national layer
explain (analyze, timing off, costs off, verbose) select count(*), sum(st_area(st_intersection(p.geom, s.geom))) $Q;
SQL
)
  ms=$(grep -oP 'Execution Time: \K[0-9.]+' <<<"$out")
  la=$(grep -oP 'Workers Launched: \K[0-9]+' <<<"$out" | head -1); la=${la:-0}
  echo "parallel_query,$w,$la,$ms,,"
done
for n in $W; do
  [ "$n" = 0 ] && continue
  t0=$(date +%s.%N)
  for ((i=0;i<n;i++)); do
    $PSQL -c "set max_parallel_workers_per_gather=0; select count(*), sum(st_area(st_intersection(p.geom, s.geom))) $Q and abs(hashtext(p.landrecid::text)) % $n = $i" > /tmp/claude-1000/-mnt-big2-home-tjwebb-workspace-postgis-hopper/af3a34fb-64a2-4deb-bcff-ae1ab80effce/scratchpad/shard_$i.out &
  done
  wait
  t1=$(date +%s.%N)
  agg=$(cat /tmp/claude-1000/-mnt-big2-home-tjwebb-workspace-postgis-hopper/af3a34fb-64a2-4deb-bcff-ae1ab80effce/scratchpad/shard_*.out | grep '|' | awk -F'|' '{c+=$1; a+=$2} END {printf "%d,%.9g", c, a}')
  rm -f /tmp/claude-1000/-mnt-big2-home-tjwebb-workspace-postgis-hopper/af3a34fb-64a2-4deb-bcff-ae1ab80effce/scratchpad/shard_*.out
  echo "sharded_conns,$n,$n,$(echo "($t1-$t0)*1000" | bc),$agg"
done
