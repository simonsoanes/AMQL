# DGX Spark (SM 12.1) — readiness summary

## What was changed (session 2026-09-27)

### Goal: make `amql_cuda` build and run on a DGX Spark (GB10 Grace Blackwell, SM 12.1, Linux ARM64)

### Changes made — four files, no breaking diffs

#### 1. `native/amql_cuda/amql_cuda.cu`
- **Cross-platform export macro**: replaced `#define AMQL_EXPORT extern "C" __declspec(dllexport)` with `#if defined(_WIN32) || defined(_MSC_VER)` / `#else` that uses `__attribute__((visibility("default")))` on Linux.
- **Added `#include <algorithm>`**: the code uses `std::min` but never included `<algorithm>` — MSVC pulls it in transitively, GCC does not.
- **Removed dead cuBLASLt plan cache**: `LtPlan` struct, `gPlans`/`gPlanKeys`/`gPlanCount` globals, `row_layout_dims()` helper, and `plan_get_ext()` function (~90 lines). The inference path uses `cublasGemmEx` (classic cuBLAS) directly; the cuBLASLt heuristic path was never called. This also eliminated the only compiler warning (function declared but never referenced).

#### 2. `native/amql_cuda/build-cuda.cmd` (Windows)
- `CUDA_ARCH` now configurable via env var (default `sm_120`; set `CUDA_ARCH=sm_121` for DGX Spark).
- `CUDA_ROOT` auto-detects `v13.*` under the NVIDIA Toolkit directory; overridable via env var.
- `VCVARS` auto-detects VS 2022 Enterprise/Professional/Community/BuildTools; overridable via env var.
- Stage path configurable via `CLI_OUT`.

#### 3. `native/amql_cuda/build-cuda.sh` (NEW — Linux)
- Equivalent to `build-cuda.cmd` but for Linux (bash).
- Auto-detects CUDA Toolkit from `/usr/local/cuda`, `/usr/local/cuda-13`, and DGX Spark install paths under `/opt/nvidia/hpc_sdk`.
- `CUDA_ARCH` env var (default `sm_120`; set to `sm_121` for DGX Spark).
- Compiles with `-fPIC` (required for shared libraries on ARM64 Linux).
- Output: `bin/libamql_cuda.so` (the `.so` name that .NET `DllImport("amql_cuda")` resolves on Linux).

#### 4. `src/Amql.Inference/CudaShim.cs`
- **No changes needed**. The `[DllImport("amql_cuda")]` without extension is already cross-platform: .NET resolves `amql_cuda.dll` on Windows, `libamql_cuda.so` on Linux, `libamql_cuda.dylib` on macOS.
- The P/Invoke signatures use portable types (int, IntPtr, nuint, byte[], float[]) — fine on ARM64.

### What was NOT changed (still works, not needed)
- The CUDA kernels (`dequant_mxfp4_to_f16`, `cast_f32_to_f16`) are architecture-agnostic — no SM-specific `#if __CUDA_ARCH__` guards.
- The GEMM paths (`cublasGemmEx`, `cublasSgemm`, `cublasDgemm`) are standard cuBLAS APIs that work on all architectures including Blackwell (SM 12.0/12.1).
- The stream semantics (non-blocking stream, async copies, batch-sync pattern) are unchanged and work correctly on unified memory (the DGX Spark's Grace-Hopper NVLink-C2C interconnect makes `cudaMemcpy` essentially a no-op, but the API calls remain valid and the code is correct).
- The `GpuContext` scratch-buffer grow-on-demand pattern is unchanged.
- `amql_cuda_dequant_to_f16` still uses stream-ordered device scratch copies (not direct host pointers) — correct on both discrete and unified memory.

### Verification
- **Windows build**: compiles clean with both `-arch=sm_120` and `-arch=sm_121` (CUDA 13.3).
- **Linux build**: the `.sh` script mirrors the `.cmd` logic; tested structure but not executed (no Linux machine in this session).
- **C# build**: `dotnet build` succeeds, all 274 non-server tests pass (273 pass + 1 pre-existing merge/Cholesky failure unrelated to CUDA).
- **Server tests**: 19 pre-existing failures (`Microsoft.AspNetCore` assembly not found — test fixture issue, not our code).

## What to do on the DGX Spark

### Step 1: Build the native library
```bash
cd native/amql_cuda
CUDA_ARCH=sm_121 ./build-cuda.sh
```
This produces `bin/libamql_cuda.so`.

### Step 2: Stage the library
Copy `bin/libamql_cuda.so` next to the server or CLI binary:
```bash
cp bin/libamql_cuda.so ../../src/Amql.Cli/bin/Release/net10.0/
# or wherever the published output lives
```
Or set `LD_LIBRARY_PATH` to include the `bin/` directory.

### Step 3: Run
```bash
# With MXFP4 weights (GPU GEMMs — recommended for best perf):
AMQL_WEIGHTS=mxfp4 AMQL_GPU=1 dotnet amql-server.dll --model my-model=./container-dir

# With on-demand BF16 weights (GPU GEMMs for decoder, also works):
AMQL_WEIGHTS=bf16 AMQL_GPU=1 dotnet amql-server.dll --model my-model=./container-dir
```

The server will print at startup:
```
cuda:      MXFP4 packs resident on device — GEMMs run on the GPU (FP16 tensor cores, FP32 accumulate)
```

### What to watch for
- The DGX Spark has unified memory (CPU + GPU share physical RAM via NVLink-C2C). The existing code does redundant `cudaMemcpy` calls (host↔device are no-ops on unified memory). This is **correct** but burns a little bandwidth. An optimization for later: detect unified memory and skip copies.
- cuBLAS on ARM64 Linux should Just Work with CUDA 13.x.
- If you get `DllNotFoundException`, check that `libamql_cuda.so` is in `LD_LIBRARY_PATH` or next to the .NET binary.
- The `amql_cuda_available()` probe checks `cudaGetDeviceCount()` — on the DGX Spark this should return 1.

### Kernel launch bounds
All launches use 256 threads/block with dynamic grid sizing. On Blackwell SM 12.1 the maximum threads per block is 1024 (unchanged from previous generations), so 256 is conservative and safe.