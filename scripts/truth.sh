#!/usr/bin/env bash
# PostGIS CPU ground truth + baseline timing for the county sample. Read-only.
# usage: truth.sh <layer: soils|wetlands> <out.csv>
set -euo pipefail
layer=$1; out=$2
case $layer in
  soils) tbl=insights.nrcs_ssurgo_mupolygon;;
  wetlands) tbl=insights.usfws_nwi_wetlands;;
esac
psql -h graybase -U claude -d landstats -w -v ON_ERROR_STOP=1 <<SQL
\timing on
set statement_timeout='30min';
set max_parallel_workers_per_gather=0;   -- single core baseline
\copy (select p.landrecid, s.ctid::text as sid, st_intersects(p.geom,s.geom) as isect, st_area(st_intersection(p.geom,s.geom)) as area from combine.parcel_enriched p join $tbl s on p.geom && s.geom and st_intersects(p.geom,s.geom) where p.geoid='19103' and p.geom is not null) to '$out' csv header
SQL
