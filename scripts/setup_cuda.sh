#!/usr/bin/env bash
# Private CUDA toolchain copy with a glibc >= 2.41 header fix: glibc declares cospi/sinpi/rsqrt noexcept and CUDA <= 12.9's
# math_functions.h does not, so nvcc fails on every file. Only needed on such hosts (e.g. Ubuntu 25.04+, Fedora 42+);
# needs no root. The build scripts use toolchain/cuda/bin/nvcc automatically once it exists.
#   usage: [CUDA_SRC=/usr/local/cuda-12.9] setup_cuda.sh [dest=toolchain/cuda]
set -euo pipefail
C=${CUDA_SRC:-/usr/local/cuda-12.9}
D=${1:-$(dirname "$0")/../toolchain/cuda}
mkdir -p "$D/targets/x86_64-linux"
cp -r $C/bin $C/nvvm "$D"/
cp -r $C/targets/x86_64-linux/include "$D/targets/x86_64-linux/"
ln -sfn $C/targets/x86_64-linux/lib "$D/targets/x86_64-linux/lib"
ln -sfn targets/x86_64-linux/include "$D/include"
ln -sfn $C/lib64 "$D/lib64"; cp $C/version.json "$D"/
chmod -R u+w "$D"
F="$D/targets/x86_64-linux/include/crt/math_functions.h"
for fn in rsqrt rsqrtf sinpi sinpif cospi cospif; do
  sed -i -E "s/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +$fn\([a-z]+ x\));/\1 noexcept(true);/" "$F"
done
echo "nvcc: $D/bin/nvcc"
