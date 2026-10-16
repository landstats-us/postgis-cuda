#!/usr/bin/env bash
# PostGIS ground truth over N concurrent connections (hash shards of the parcels); also the N-core timing.
# usage: truth_sharded.sh <soils|wetlands> <geoid> <out.csv> [N=48]
set -euo pipefail
layer=$1; geoid=$2; out=$3; N=${4:-48}
case $layer in soils) tbl=insights.nrcs_ssurgo_mupolygon;; wetlands) tbl=insights.usfws_nwi_wetlands;; esac
tmp=$(mktemp -d -p "$(dirname "$out")" shards.XXXX)
t0=$(date +%s.%N)
for ((i=0;i<N;i++)); do
  psql -h graybase -U claude -d landstats -w -X -q -c "set max_parallel_workers_per_gather=0; set statement_timeout='2h';" \
    -c "\copy (select p.landrecid, s.ctid::text as sid, true as isect, st_area(st_intersection(p.geom,s.geom)) as area from combine.parcel_enriched p join $tbl s on st_intersects(p.geom,s.geom) where p.geoid='$geoid' and p.geom is not null and abs(hashtext(p.landrecid::text)) % $N = $i) to '$tmp/$i.csv' csv" &
done
wait
t1=$(date +%s.%N)
{ echo "landrecid,sid,isect,area"; cat "$tmp"/*.csv; } > "$out"
rm -rf "$tmp"
echo "$layer geoid=$geoid N=$N wall_ms=$(echo "($t1-$t0)*1000" | bc) rows=$(($(wc -l < "$out")-1))"
