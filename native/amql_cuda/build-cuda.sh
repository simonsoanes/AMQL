#!/usr/bin/env bash
# build-cuda.sh — build the optional amql_cuda native backend (Linux).
# Requires: CUDA Toolkit (nvcc), GCC toolchain.
# Produces: native/amql_cuda/bin/libamql_cuda.so
# Exit code 0 on success, 1 on any failure.
#
# Environment overrides:
#   CUDA_ROOT  — path to the CUDA Toolkit (default: /usr/local/cuda)
#   CUDA_ARCH  — target SM architecture (default: sm_120; e.g. sm_121)
#   CLI_OUT    — optional deployment directory

set -euo pipefail
cd "$(dirname "$0")"

# ── CUDA Toolkit detection ───────────────────────────────────────────
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
# Also check common DGX Spark / NVIDIA SDK Manager paths.
for candidate in "$CUDA_ROOT" /usr/local/cuda-13 /opt/nvidia/hpc_sdk/Linux_aarch64/*/cuda; do
    if [ -x "$candidate/bin/nvcc" ]; then
        CUDA_ROOT="$candidate"
        break
    fi
done
if [ ! -x "$CUDA_ROOT/bin/nvcc" ]; then
    echo "[amql-cuda] nvcc not found — set CUDA_ROOT to the CUDA Toolkit install." >&2
    exit 1
fi

# ── Architecture ─────────────────────────────────────────────────────
CUDA_ARCH="${CUDA_ARCH:-sm_120}"

# ── Build ────────────────────────────────────────────────────────────
mkdir -p bin

echo "[amql-cuda] nvcc: $("$CUDA_ROOT/bin/nvcc" --version | grep -i release || true)"
echo "[amql-cuda] compiling amql_cuda.cu -arch=$CUDA_ARCH ..."

"$CUDA_ROOT/bin/nvcc" \
    -arch="$CUDA_ARCH" \
    -O3 -std=c++17 \
    -Xcompiler -fPIC \
    --shared \
    -o bin/libamql_cuda.so \
    amql_cuda.cu \
    -lcublasLt -lcublas -lcudart

echo "[amql-cuda] built bin/libamql_cuda.so"

# ── Stage next to the CLI / server ───────────────────────────────────
if [ -n "${CLI_OUT:-}" ] && [ -d "$CLI_OUT" ]; then
    cp bin/libamql_cuda.so "$CLI_OUT/"
    echo "[amql-cuda] staged libamql_cuda.so into $CLI_OUT"
fi