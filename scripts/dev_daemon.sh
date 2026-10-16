#!/usr/bin/env bash
# (Re)start gpujoind on GPU 3 against the dev instance (port 5434) and load the test layers.
cd "$(dirname "$0")/.."
TOKEN_ARGS=""; [ -f run/gpujoind.token ] && TOKEN_ARGS="--token-file run/gpujoind.token" && export PGOPTIONS="-c gpu_spatial.token=$(cat run/gpujoind.token)"
[ -f run/gpujoind.pid ] && kill "$(cat run/gpujoind.pid)" 2>/dev/null && sleep 1
rm -f run/gpujoind.sock
CUDA_VISIBLE_DEVICES=${GPU:-3} nohup ${BIN:-./build/gpujoind} --conninfo "host=127.0.0.1 port=5434 dbname=postgres user=postgres" \
  --socket run/gpujoind.sock --threads 16 ${TOKEN_ARGS} >> data/gpujoind.log 2>&1 &
echo $! > run/gpujoind.pid
for i in $(seq 50); do [ -S run/gpujoind.sock ] && psql -h 127.0.0.1 -p 5434 -U postgres -w -X -qAt -c "select gpu_status()" >/dev/null 2>&1 && break; sleep 0.2; done
psql -h 127.0.0.1 -p 5434 -U postgres -w -X -qAt <<'SQL'
select gpu_load('parcels', 'combine.parcel_enriched', id_column => 'landrecid', subdivide => 256, dedup => true)->>'rows';
select gpu_load('counties', 'tiger_data.county_4326', id_column => 'gid', subdivide => 64)->>'rows';
select gpu_load('addresses', 'combine.overturemaps_address', id_column => 'id')->>'rows';
SQL
