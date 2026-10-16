#!/usr/bin/env bash
# PostGIS ground truth over N concurrent connections; also the N-core wall time.
# usage: [CONN="psql connection args"] truth_sql.sh <out.csv> <N> "<select ... producing (a_id, b_id, isect, area)>"
#   the query must contain {i} and {n}; shard i of n should filter e.g. abs(hashtext(p.landrecid::text)) % {n} = {i}
set -euo pipefail
out=$1; N=$2; tmpl=$3
CONN=${CONN:-"-h graybase -U claude -d landstats"}
tmp=$(mktemp -d -p "$(dirname "$out")" shards.XXXX)
t0=$(date +%s.%N)
for ((i=0;i<N;i++)); do
  q=${tmpl//\{i\}/$i}; q=${q//\{n\}/$N}
  psql $CONN -w -X -q -v ON_ERROR_STOP=1 -c "set max_parallel_workers_per_gather=0; set statement_timeout='3h';" \
    -c "\copy ($q) to '$tmp/$i.csv' csv" 2> "$tmp/$i.err" &
done
wait
t1=$(date +%s.%N)
errs=$(cat "$tmp"/*.err | grep -v '^$' | sort | uniq -c | head -5 || true)
{ echo "landrecid,sid,isect,area"; cat "$tmp"/*.csv; } > "$out"
rm -rf "$tmp"
echo "$(basename "$out") N=$N wall_ms=$(echo "($t1-$t0)*1000" | bc) rows=$(($(wc -l < "$out")-1)) ${errs:+ERRORS: $errs}"
