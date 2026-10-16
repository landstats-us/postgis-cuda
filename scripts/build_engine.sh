#!/usr/bin/env bash
# build/harness: file-based test / benchmark driver (see scripts/build_env.sh for NVCC, CCBIN, ARCH)
set -euo pipefail
source "$(dirname "$0")/build_env.sh"
cd "$ROOT"
"$NVCC" "${NVFLAGS[@]}" -Iengine engine/harness.cu -o "${OUT:-build/harness}" -lgomp "$@"
