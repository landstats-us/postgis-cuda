#!/usr/bin/env bash
# build/gpujoind: the daemon (needs libpq headers + library; see scripts/build_env.sh for NVCC, CCBIN, ARCH)
#   PG_CONFIG  pg_config used to find libpq (default: pg_config on PATH, else /usr/include/postgresql)
set -euo pipefail
source "$(dirname "$0")/build_env.sh"
cd "$ROOT"
PQ_INC=${PQ_INC:-$( (${PG_CONFIG:-pg_config} --includedir) 2>/dev/null || echo /usr/include/postgresql)}
PQ_LIB=${PQ_LIB:-$( (${PG_CONFIG:-pg_config} --libdir) 2>/dev/null || echo /usr/lib/x86_64-linux-gnu)}
"$NVCC" "${NVFLAGS[@]}" -I"$PQ_INC" daemon/gpujoind.cu -o "${OUT:-build/gpujoind}" -L"$PQ_LIB" -lpq -lgomp "$@"
