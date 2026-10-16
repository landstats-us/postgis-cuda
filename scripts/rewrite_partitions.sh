#!/usr/bin/env bash
# Rewrite (VACUUM FULL + ANALYZE) every leaf partition of a partitioned table to remove heap bloat.
# Largest partitions first, P at a time. Each partition is ACCESS EXCLUSIVE locked only while it is rewritten;
# lock_timeout makes busy partitions fail fast instead of queueing behind (and blocking) other sessions —
# those are retried in later passes.
#   usage: rewrite_partitions.sh [table=combine.parcel_enriched] [P=6] [passes=3]
set -uo pipefail
TABLE=${1:-combine.parcel_enriched}; P=${2:-6}; PASSES=${3:-3}
PSQL="psql -h graybase -U claude -d landstats -w -X -q -At"
LOG=${LOG:-$(dirname "$0")/../data/rewrite_$(date +%Y%m%d_%H%M%S).log}
todo=$(mktemp); failed=$(mktemp)

$PSQL -c "select c.oid::regclass from pg_inherits i join pg_class c on c.oid = i.inhrelid
          where i.inhparent = '$TABLE'::regclass and c.relkind = 'r' order by pg_relation_size(c.oid) desc" > "$todo"
before=$($PSQL -c "select sum(pg_relation_size(c.oid)), sum(pg_total_relation_size(c.oid)) from pg_inherits i join pg_class c on c.oid = i.inhrelid where i.inhparent = '$TABLE'::regclass")
echo "$(date -Is) start $TABLE: $(wc -l < "$todo") partitions, heap,total bytes before = $before, P=$P" | tee -a "$LOG"

rewrite_one() {
  local rel=$1 t0 out rc b a
  b=$($PSQL -c "select pg_relation_size('$rel'), pg_total_relation_size('$rel')")
  t0=$(date +%s.%N)
  out=$($PSQL -c "set lock_timeout = '10s'" -c "set statement_timeout = '4h'" -c "vacuum (full, analyze) $rel" 2>&1); rc=$?
  a=$($PSQL -c "select pg_relation_size('$rel'), pg_total_relation_size('$rel')")
  if [ $rc -eq 0 ]; then
    echo "$(date -Is) ok   $rel $(printf %.1f "$(echo "$(date +%s.%N) - $t0" | bc)")s heap,total $b -> $a"
  else
    echo "$(date -Is) FAIL $rel: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
    echo "$rel" >> "$FAILED"
  fi
}
export -f rewrite_one; export PSQL

for pass in $(seq "$PASSES"); do
  [ -s "$todo" ] || break
  : > "$failed"
  echo "$(date -Is) pass $pass: $(wc -l < "$todo") partitions" | tee -a "$LOG"
  FAILED=$failed xargs -P "$P" -I{} bash -c 'rewrite_one "$@"' _ {} < "$todo" >> "$LOG" 2>&1
  cp "$failed" "$todo"
  [ -s "$todo" ] && sleep 60
done

after=$($PSQL -c "select sum(pg_relation_size(c.oid)), sum(pg_total_relation_size(c.oid)) from pg_inherits i join pg_class c on c.oid = i.inhrelid where i.inhparent = '$TABLE'::regclass")
echo "$(date -Is) done $TABLE: heap,total bytes $before -> $after; still failing: $(wc -l < "$todo")" | tee -a "$LOG"
cat "$todo" >> "$LOG"
rm -f "$todo" "$failed"
