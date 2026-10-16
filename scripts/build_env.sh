# Shared build settings (sourced by build_engine.sh / build_daemon.sh). Override any of them in the environment:
#   NVCC   nvcc to use (default: toolchain/cuda/bin/nvcc if scripts/setup_cuda.sh made one, else nvcc on PATH)
#   CCBIN  host compiler for nvcc (default: g++-13 if installed, else nvcc's default)
#   ARCH   GPU architecture (default: sm_90a, Hopper; the code needs sm_80+ for warp reductions)
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${NVCC:-}" ]; then
  if [ -x "$ROOT/toolchain/cuda/bin/nvcc" ]; then NVCC="$ROOT/toolchain/cuda/bin/nvcc"
  elif command -v nvcc >/dev/null; then NVCC=$(command -v nvcc)
  elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc
  else echo "nvcc not found: set NVCC, or run scripts/setup_cuda.sh" >&2; exit 1; fi
fi
if [ -z "${CCBIN+x}" ]; then CCBIN=$(command -v g++-13 || true); fi
ARCH=${ARCH:-sm_90a}
NVFLAGS=(-std=c++17 -O3 -lineinfo "-arch=$ARCH" --expt-relaxed-constexpr -Xcompiler -fopenmp)
[ -n "$CCBIN" ] && NVFLAGS+=(-ccbin "$CCBIN")
mkdir -p "$ROOT/build"
