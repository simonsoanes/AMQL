#!/usr/bin/env bash
# build-cuda.sh — build the optional amql_cuda native backend (Linux).
# Requires: CUDA Toolkit (nvcc), GCC toolchain.
# Produces: native/amql_cuda/bin/libamql_cuda.so (the engine, and Inkling 2
#           looped's kernels beside it)
# Exit code 0 on success, 1 on any failure.
#
# Environment overrides:
#   CUDA_ROOT  — path to the CUDA Toolkit (default: /usr/local/cuda)
#   CUDA_ARCH  — target SM architecture (default: the installed GPU's, from
#                nvidia-smi — sm_121 on a DGX Spark's GB10 — else sm_120)
#   CLI_OUT    — optional deployment directory (one or more, space-separated):
#                the libraries are copied there, e.g. the folder
#                scripts/publish-linux.sh writes

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
if [ -z "${CUDA_ARCH:-}" ]; then
    # compute_cap is "12.1" on GB10 (DGX Spark), "12.0" on RTX 50-series.
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .' || true)"
    if [[ "$cap" =~ ^[0-9]+$ ]]; then
        CUDA_ARCH="sm_${cap}"
    else
        CUDA_ARCH="sm_120"
    fi
fi
echo "[amql-cuda] target: $CUDA_ARCH ($(uname -m))"

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
    amql_cuda.cu amql_engine.cu amql_moe.cu amql_vision.cu amql_graph.cu amql_inkling.cu \
    -lcublas -lcudart

echo "[amql-cuda] built bin/libamql_cuda.so"

# ── Stage next to the CLI / server ───────────────────────────────────
for out in ${CLI_OUT:-}; do
    if [ -d "$out" ]; then
        cp bin/libamql_*.so "$out/"
        echo "[amql-cuda] staged $(cd bin && echo libamql_*.so) into $out"
    fi
done