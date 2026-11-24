// amql_cuda.cu — the AMQL optional CUDA execution backend.
//
// Scope (first milestone): offload the GEMM core — every weight projection
// in the runtime (attention q/k/v/o, linear-attn in/out, FFN gate/up/down,
// MoE router/experts and the output head) — while norms, activations, the
// GatedDelta recurrence and the token loop stay on the managed CPU side.
// Weights are uploaded to the device as MXFP4 packs and dequantised to
// FP16 once per session; FP4×E8M0 values (small integers times powers of
// two) are exactly representable in FP16, so the dequant is lossless and
// the GEMM accumulates in FP32 — quality matches the CPU FP4 path within
// accumulation order, and Blackwell's tensor cores do the work.
//
// The ABI is flat C (extern "C"), loaded via P/Invoke. Every entry returns
// 0 on success and a negative CUDA status otherwise, so the managed side
// falls back to the CPU path on any failure.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

// Cross-platform ABI: Windows needs __declspec(dllexport); Linux/macOS use
// visibility("default") with -fvisibility=hidden (or default visibility).
// Both paths use extern "C" for symbol-name stability.
#if defined(_WIN32) || defined(_MSC_VER)
  #define AMQL_EXPORT extern "C" __declspec(dllexport)
#else
  #define AMQL_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ── FP4 E2M1 codec (mirror of the managed BitPattern grid) ────────────────

__constant__ float kFp4Grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float fp4_decoded(unsigned char nibble)
{
    float v = kFp4Grid[nibble & 0x07];          // __constant__ only in .cu — this is .cu
    return (nibble & 0x08) != 0 ? -v : v;
}

__device__ __forceinline__ float e8m0_decoded(unsigned char b)
{
    return b == 0xFF ? 0.f : exp2f((float)b - 127.f);   // NaN pins to 0 for safety
}

// ── Dequant kernel: MXFP4 pack [rows × cols] → FP16 row-major ─────────────

__global__ void dequant_mxfp4_to_f16(
    const unsigned char* __restrict__ packed,   // 2 elements/byte, row-major
    const unsigned char* __restrict__ scales,   // [rows × ceil(cols/32)], row-major
    __half* __restrict__ out,
    int rows, int cols)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * cols;
    if (index >= total)
    {
        return;
    }
    int col = index % cols;
    int row = index / cols;
    int blocksPerRow = (cols + 31) / 32;
    int block = col / 32;
    float scale = e8m0_decoded(scales[(long)row * blocksPerRow + block]);
    unsigned char byte = packed[index / 2];
    unsigned char nibble = (index & 1) == 0 ? (byte & 0x0F) : (byte >> 4);
    float value = fp4_decoded(nibble) * scale;
    out[index] = __float2half_rn(value);
}

// ── B = (m,k) @ (k,n) row-major, weight W = [n,k] row-major → B= x·Wᵀ ─────
//
// The runtime's dominant call is MatMulTransposedB(x, w): x = [m,k],
// w = [n,k] (rows are the output space), result C = [m,n] = x @ wᵀ.
// Classical cublas GemmEx, FP16 × FP16 with FP32 accumulate: the resident
// weights are FP16 and the activations are cast to FP16 on the device
// (cuBLAS gives NOT_SUPPORTED for an FP32-A × FP16-B mix on this
// platform/driver — neither cuBLASLt's heuristic nor GemmEx accepts it).

__global__ void cast_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, long count)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        dst[i] = __float2half_rn(src[i]);
    }
}

struct GpuContext
{
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    cudaStream_t stream;
    void* scratchA;      // reusable host→device staging for A (grows)
    size_t scratchABytes;
    void* scratchC;      // reusable device→host staging for C (grows)
    size_t scratchCBytes;
    void* scratchPack;   // the MXFP4 dequant's device copy of the pack (grows)
    size_t scratchPackBytes;
    void* scratchScale;  // ...and the block scales (grows)
    size_t scratchScaleBytes;
    void* scratchAF16;   // inference activations cast to FP16 for the GEMM (grows)
    size_t scratchAF16Bytes;
};

static GpuContext gCtx{};
static bool gCtxValid = false;
static bool gUnifiedMemory = false;

static int ctx_ensure()
{
    if (gCtxValid)
    {
        return 0;
    }
    if (cublasCreate(&gCtx.cublas) != CUBLAS_STATUS_SUCCESS)
    {
        return -15;
    }
    if (cudaStreamCreateWithFlags(&gCtx.stream, cudaStreamNonBlocking) != cudaSuccess)
    {
        return -2;
    }
    if (cublasSetStream(gCtx.cublas, gCtx.stream) != CUBLAS_STATUS_SUCCESS)
    {
        return -16;
    }
    if (cusolverDnCreate(&gCtx.cusolver) != CUSOLVER_STATUS_SUCCESS)
    {
        return -22;
    }
    if (cusolverDnSetStream(gCtx.cusolver, gCtx.stream) != CUSOLVER_STATUS_SUCCESS)
    {
        return -23;
    }
    // Probe unified memory (Grace-Hopper, integrated GPUs) — on these
    // platforms host↔device cudaMemcpy is a page-table no-op and the GPU
    // can access system memory directly through the NVLink-C2C coherent
    // fabric.  The hot paths skip the scratch-buffer staging when set.
    int integrated = 0;
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0);
    gUnifiedMemory = (integrated != 0);

    if (cudaMalloc(&gCtx.scratchPack, 1u << 20) != cudaSuccess)
    {
        return -19;
    }
    gCtx.scratchPackBytes = 1u << 20;
    if (cudaMalloc(&gCtx.scratchScale, 1u << 20) != cudaSuccess)
    {
        return -20;
    }
    gCtx.scratchScaleBytes = 1u << 20;
    if (cudaMalloc(&gCtx.scratchAF16, 1u << 20) != cudaSuccess)
    {
        return -21;
    }
    gCtx.scratchAF16Bytes = 1u << 20;
    gCtxValid = true;
    return 0;
}

static int scratch_reserve(void** slot, size_t* slotBytes, size_t want)
{
    if (*slot != nullptr && *slotBytes >= want)
    {
        return 0;
    }
    if (*slot != nullptr)
    {
        cudaFree(*slot);
    }
    if (cudaMalloc(slot, want) != cudaSuccess)
    {
        return -1;
    }
    *slotBytes = want;
    return 0;
}

// ── C ABI ─────────────────────────────────────────────────────────────────

AMQL_EXPORT int amql_cuda_available(void)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
    {
        return 0;
    }
    return 1;
}

AMQL_EXPORT int amql_cuda_init(void)
{
    return ctx_ensure();
}

AMQL_EXPORT int amql_cuda_unified_memory(void)
{
    if (ctx_ensure() != 0)
    {
        return 0;
    }
    return gUnifiedMemory ? 1 : 0;
}

AMQL_EXPORT void amql_cuda_shutdown(void)
{
    if (!gCtxValid)
    {
        return;
    }
    if (gCtx.scratchA) cudaFree(gCtx.scratchA);
    if (gCtx.scratchC) cudaFree(gCtx.scratchC);
    if (gCtx.scratchPack) cudaFree(gCtx.scratchPack);
    if (gCtx.scratchScale) cudaFree(gCtx.scratchScale);
    if (gCtx.scratchAF16) cudaFree(gCtx.scratchAF16);
    cudaStreamDestroy(gCtx.stream);
    cublasDestroy(gCtx.cublas);
    cusolverDnDestroy(gCtx.cusolver);
    gCtxValid = false;
}

AMQL_EXPORT int amql_cuda_malloc(void** ptr, size_t bytes)
{
    return cudaMalloc(ptr, bytes) == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_free(void* ptr)
{
    return cudaFree(ptr) == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_host_to_device(void* dst, const void* src, size_t bytes)
{
    if (gUnifiedMemory)
    {
        return 0;   // GPU accesses host memory directly via NVLink-C2C
    }
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_host_to_device_f32(void* dst, const float* src, size_t bytes)
{
    if (gUnifiedMemory)
    {
        return 0;
    }
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_device_to_host(void* dst, const void* src, size_t bytes)
{
    if (gUnifiedMemory)
    {
        return 0;
    }
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_dequant_to_f16(
    const unsigned char* packed, const unsigned char* scales,
    __half* out, int rows, int cols, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    long total = (long)rows * cols;
    if (total == 0)
    {
        return 0;
    }
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    // The kernel runs on the device, so it needs DEVICE copies of the
    // pack and scale bytes — the caller hands in host pointers (the
    // P/Invoke side pins byte[] as host memory). Up to the dequant-rework
    // of this function the raw host pointers were passed straight to the
    // kernel, which on a discrete GPU is an illegal memory access from
    // every thread; the fault was asynchronous, silently corrupted the
    // context (error 700), and every later upload fell back to CPU — the
    // GPU inference path never actually engaged. The copies go through
    // the stream-ordered scratch, so consecutive uploads reuse the same
    // device buffers without a per-call synchronise (the next memcpy on
    // the stream is ordered after the previous kernel consumed them).
    //
    // On unified-memory platforms (Grace-Hopper, integrated GPUs) the GPU
    // can access host memory directly through the NVLink-C2C coherent
    // fabric — the kernel reads from the host pointers and the scratch
    // copies are skipped.
    size_t packedBytes = (size_t)((total + 1) / 2);
    size_t scaleBytes = (size_t)rows * ((cols + 31) / 32);
    const unsigned char* packSrc;
    const unsigned char* scaleSrc;
    if (gUnifiedMemory)
    {
        packSrc = packed;
        scaleSrc = scales;
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchPack, &gCtx.scratchPackBytes, packedBytes) != 0)
        {
            return -2;
        }
        if (scratch_reserve(&gCtx.scratchScale, &gCtx.scratchScaleBytes, scaleBytes) != 0)
        {
            return -3;
        }
        if (cudaMemcpyAsync(gCtx.scratchPack, packed, packedBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(gCtx.scratchScale, scales, scaleBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return -4;
        }
        packSrc = (const unsigned char*)gCtx.scratchPack;
        scaleSrc = (const unsigned char*)gCtx.scratchScale;
    }
    dequant_mxfp4_to_f16<<<blocks, threads, 0, stream>>>(
        packSrc, scaleSrc, out, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ── GEMM core: C[m,n] = A[m,k] @ W[n,k]ᵀ, all row-major, FP32 accumulate ──

AMQL_EXPORT int amql_cuda_gemm_transposed_b(
    const float* a,       // [m,k] row-major host
    const __half* w,      // [n,k] row-major device (pre-dequantised, resident)
    float* c,             // [m,n] row-major host out
    int m, int k, int n, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0)
    {
        return 0;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    const float* aSrc;
    if (gUnifiedMemory)
    {
        aSrc = a;   // GPU accesses host memory directly via NVLink-C2C
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)m * k * 4) != 0)
        {
            return -2;
        }
        if (cudaMemcpyAsync(gCtx.scratchA, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return -4;
        }
        aSrc = (const float*)gCtx.scratchA;
    }
    if (scratch_reserve(&gCtx.scratchAF16, &gCtx.scratchAF16Bytes, (size_t)m * k * 2) != 0)
    {
        return -5;
    }
    void* aF16 = gCtx.scratchAF16;
    long actCount = (long)m * k;
    int actThreads = 256;
    int actBlocks = (int)((actCount + actThreads - 1) / actThreads);
    cast_f32_to_f16<<<actBlocks, actThreads, 0, stream>>>(
        aSrc, (__half*)aF16, actCount);
    if (cudaGetLastError() != cudaSuccess)
    {
        return -6;
    }

    // C[m,n] = A[m,k] @ W[n,k]ᵀ with both operands FP16 and FP32
    // accumulate (the only mixed-format configuration this platform
    // supports). The column-major formulation mirrors
    // amql_cuda_gemm_transposed_b_f32: D[n,m] = W[n,k]·Aᵀ[k,m] with both
    // operands viewed column-major at ld=k; D stored with ldc=n lands at
    // j + i·n, exactly the row-major [m,n] host output.
    float alpha = 1.0f, beta = 0.0f;
    float* cOut;
    if (gUnifiedMemory)
    {
        cOut = c;   // cuBLAS writes directly to host output buffer
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
        {
            return -3;
        }
        cOut = (float*)gCtx.scratchC;
    }
    cublasStatus_t status = cublasGemmEx(
        gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const void*)w, CUDA_R_16F, k,
        (const void*)aF16, CUDA_R_16F, k,
        &beta,
        (void*)cOut, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (!gUnifiedMemory)
    {
        cudaMemcpyAsync(c, (float*)gCtx.scratchC, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    return status == CUBLAS_STATUS_SUCCESS ? 0 : -7;
}

// ── Async GEMM: same as amql_cuda_gemm_transposed_b but does NOT sync ──────
// the stream — the caller is responsible for eventually calling
// amql_cuda_sync before reading the host output buffer.  Batched callers
// launch several async GEMMs and sync once.

AMQL_EXPORT int amql_cuda_gemm_transposed_b_async(
    const float* a,       // [m,k] row-major host
    const __half* w,      // [n,k] row-major device (pre-dequantised, resident)
    float* c,             // [m,n] row-major host out (filled after sync)
    int m, int k, int n, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0)
    {
        return 0;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    const float* aSrc;
    if (gUnifiedMemory)
    {
        aSrc = a;
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)m * k * 4) != 0)
        {
            return -2;
        }
        if (cudaMemcpyAsync(gCtx.scratchA, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return -4;
        }
        aSrc = (const float*)gCtx.scratchA;
    }
    if (scratch_reserve(&gCtx.scratchAF16, &gCtx.scratchAF16Bytes, (size_t)m * k * 2) != 0)
    {
        return -5;
    }
    void* aF16 = gCtx.scratchAF16;
    long actCount = (long)m * k;
    int actThreads = 256;
    int actBlocks = (int)((actCount + actThreads - 1) / actThreads);
    cast_f32_to_f16<<<actBlocks, actThreads, 0, stream>>>(
        aSrc, (__half*)aF16, actCount);
    if (cudaGetLastError() != cudaSuccess)
    {
        return -6;
    }

    float alpha = 1.0f, beta = 0.0f;
    float* cOut;
    if (gUnifiedMemory)
    {
        cOut = c;
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
        {
            return -3;
        }
        cOut = (float*)gCtx.scratchC;
    }
    cublasStatus_t status = cublasGemmEx(
        gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const void*)w, CUDA_R_16F, k,
        (const void*)aF16, CUDA_R_16F, k,
        &beta,
        (void*)cOut, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (!gUnifiedMemory)
    {
        cudaMemcpyAsync(c, (float*)gCtx.scratchC, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream);
    }
    // NO sync — caller batches and syncs once.

    return status == CUBLAS_STATUS_SUCCESS ? 0 : -7;
}

// ── Upload host FP32 activation to device as FP16 ─────────────────────────
// Allocates a device buffer, copies and casts on the stream.  The caller
// frees the returned pointer with amql_cuda_free_activation when done.
// This is the batched-GEMM fast path: upload once, launch N GEMMs against
// the same activation, sync once, free.

AMQL_EXPORT int amql_cuda_upload_activation_f16(
    const float* a,       // [m,k] row-major host
    __half** out,         // allocated device FP16 pointer
    int m, int k, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0)
    {
        return 0;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;
    size_t bytes = (size_t)m * k * 2;
    if (cudaMalloc((void**)out, bytes) != cudaSuccess)
    {
        return -2;
    }

    const float* aSrc;
    if (gUnifiedMemory)
    {
        aSrc = a;
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)m * k * 4) != 0)
        {
            cudaFree(*out);
            *out = nullptr;
            return -3;
        }
        if (cudaMemcpyAsync(gCtx.scratchA, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            cudaFree(*out);
            *out = nullptr;
            return -4;
        }
        aSrc = (const float*)gCtx.scratchA;
    }
    long count = (long)m * k;
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    cast_f32_to_f16<<<blocks, threads, 0, stream>>>(
        aSrc, *out, count);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(*out);
        *out = nullptr;
        return -5;
    }
    return 0;
}

// ── GEMM with device-resident FP16 activation (no upload, no cast) ────────
// C[m,n] = A_dev[m,k] @ W[n,k]ᵀ.  No host→device copy, no FP32→FP16 cast
// — the activation is already on the device in FP16 from a prior upload.
// Async (no sync); the caller batches and syncs once.

AMQL_EXPORT int amql_cuda_gemm_device_a(
    const __half* a_dev,  // [m,k] row-major device FP16
    const __half* w,      // [n,k] row-major device FP16
    float* c,             // [m,n] row-major host out (filled after sync)
    int m, int k, int n, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0)
    {
        return 0;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    float* cOut;
    if (gUnifiedMemory)
    {
        cOut = c;
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
        {
            return -2;
        }
        cOut = (float*)gCtx.scratchC;
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const void*)w, CUDA_R_16F, k,
        (const void*)a_dev, CUDA_R_16F, k,
        &beta,
        (void*)cOut, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (!gUnifiedMemory)
    {
        cudaMemcpyAsync(c, (float*)gCtx.scratchC, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream);
    }
    // NO sync.

    return status == CUBLAS_STATUS_SUCCESS ? 0 : -7;
}

// ── Stream synchronisation — the caller's batch fence ─────────────────────

AMQL_EXPORT int amql_cuda_sync(int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        return -2;
    }
    // Scratch buffers are now free for the next batch.
    return 0;
}

// ── FP32 GEMM: C[m,n] = A[m,k] @ W[n,k]ᵀ, all row-major, FP32 × FP32 ─────
// Used by the merge path (embedding alignment, map-apply, moe-ify usage
// matmuls) which works entirely in fp32 — no FP16 dequant involved.
// Streams A in blocks of blockRows rows so a multi-GiB token table never
// has to be fully resident (the merge's other rows are ~5 GiB on the
// 27B). Classic cublas (no Lt heuristic): row-major A is viewed as the
// column-major transpose, opA='T' recovers [m,k]·[k,n], and the result is
// transposed-copied back into the row-major host layout.

AMQL_EXPORT int amql_cuda_gemm_transposed_b_f32(
    const float* a,       // [m,k] row-major host
    const float* w,       // [n,k] row-major device (fp32, resident)
    float* c,             // [m,n] row-major host out
    int m, int k, int n, int blockRows, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0)
    {
        return 0;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;
    if (blockRows <= 0)
    {
        blockRows = 4096;
    }
    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)blockRows * k * 4) != 0)
    {
        return -2;
    }
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)blockRows * n * 4) != 0)
    {
        return -3;
    }
    void* aD = gCtx.scratchA;
    void* cD = gCtx.scratchC;
    float alpha = 1.0f, beta = 0.0f;

    for (int base = 0; base < m; base += blockRows)
    {
        int rows = std::min(blockRows, m - base);
        if (cudaMemcpyAsync(aD, a + (size_t)base * k, (size_t)rows * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return -4;
        }
        // Column-major cublas view: D[n,rows] = W[n,k]·Aᵀ[k,rows] gives
        // D[j,i] = Σ_p W[j,p]·A[i,p] = C_row[i,j], and D stored with
        // ldc=n lands at offset j + i*n — exactly the row-major [rows,n]
        // host layout, so no transpose copy is needed on download.
        // A_stored is the row-major block viewed column-major [k,rows]
        // (ld=k), transa=T recovers A [rows,k].
        cublasStatus_t status = cublasSgemm(
            gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n, rows, k,
            &alpha, (const float*)w, k, (const float*)aD, k, &beta, (float*)cD, n);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            return -5;
        }
        if (cudaMemcpyAsync(c + (size_t)base * n, cD, (size_t)rows * n * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            return -6;
        }
    }
    cudaStreamSynchronize(stream);
    return 0;
}

// ── Map-apply with in-function map upload ─────────────────────────────────
// Same blocked Sgemm as above, but the weight (map) is uploaded here, on
// the cublas stream, so the caller never straddles a sync NULL-stream
// copy with cublas work (the discipline gram_cross proved necessary on
// this driver: default-stream ops next to cublas kernels are unreliable).

AMQL_EXPORT int amql_cuda_gemm_transposed_b_f32_u(
    const float* a,    // [m × k] host rows
    const float* w,    // [n × k] host weight (the alignment map)
    float* c,          // [m × n] host output
    int m, int k, int n, int blockRows, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0)
    {
        return 0;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;
    if (blockRows <= 0)
    {
        blockRows = 4096;
    }
    void* wD = nullptr;
    if (cudaMalloc(&wD, (size_t)n * k * 4) != cudaSuccess)
    {
        return -2;
    }
    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)blockRows * k * 4) != 0)
    {
        cudaFree(wD);
        return -3;
    }
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)blockRows * n * 4) != 0)
    {
        cudaFree(wD);
        return -4;
    }
    void* aD = gCtx.scratchA;
    void* cD = gCtx.scratchC;
    float alpha = 1.0f, beta = 0.0f;

    if (cudaMemcpyAsync(wD, w, (size_t)n * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cudaFree(wD);
        return -5;
    }
    for (int base = 0; base < m; base += blockRows)
    {
        int rows = std::min(blockRows, m - base);
        if (cudaMemcpyAsync(aD, a + (size_t)base * k, (size_t)rows * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            cudaFree(wD);
            return -6;
        }
        cublasStatus_t status = cublasSgemm(
            gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n, rows, k,
            &alpha, (const float*)wD, k, (const float*)aD, k, &beta, (float*)cD, n);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            cudaFree(wD);
            return -7;
        }
        if (cudaMemcpyAsync(c + (size_t)base * n, cD, (size_t)rows * n * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            cudaFree(wD);
            return -8;
        }
    }
    cudaStreamSynchronize(stream);
    cudaFree(wD);
    return 0;
}

// ── Blocked Gram/Cross: G = Σᵢ bᵢbᵢᵀ = BᵀB, C = Σᵢ aᵢbᵢᵀ = BᵀA ─────────
// The merge alignment's normal-equation matrices over the shared-token
// anchors. Runs in row blocks (blockRows at a time) so a huge anchor set
// never has to be resident. Uses classic cublas (no Lt heuristic): the
// row-major [blockRows × d] slab viewed column-major is Bᵀ
// ([d × blockRows], ld=d), so gram = BᵀB is opB='T'·opA='N' on the same
// buffer, and cross = BᵀA likewise. Block partials are fp64-accumulated
// on the host. This is the operation that pegged ~50 CPU cores for 40+
// minutes on the 27B merge; on the 5090 each block is a pair of small
// GEMMs (seconds total).

AMQL_EXPORT int amql_cuda_gram_cross(
    const float* a,   // [n × d] host — the target (scaffold) rows
    const float* b,   // [n × d] host — the source (other-model) rows
    double* gramOut,  // [d × d] host — BᵀB
    double* crossOut, // [d × d] host — BᵀA
    int n, int d, int blockRows, int streamOrdinal)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (n <= 0 || d <= 0)
    {
        return 0;
    }
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    // fp64 is the alignment authority (the managed normal equations run in
    // double). The Gram/Cross products stay fp64: each block's slabs are
    // converted to double on the host, uploaded, and fed to cublasDgemm —
    // Blackwell consumer fp64 is slow (1:64 of fp32) but the Gram work is
    // only O(n·d²) ≈ tens of seconds at 27B scale versus ~50 CPU minutes.
    auto* bT = new double[blockRows * d];
    auto* aT = new double[blockRows * d];
    size_t ddF64 = (size_t)d * d * 8;
    void* bD = nullptr; void* aD = nullptr;
    void* gramD = nullptr; void* crossD = nullptr;
    if (cudaMalloc(&bD, (size_t)blockRows * d * 8) != cudaSuccess) { delete[] bT; delete[] aT; return -2; }
    if (cudaMalloc(&aD, (size_t)blockRows * d * 8) != cudaSuccess) { cudaFree(bD); delete[] bT; delete[] aT; return -3; }
    if (cudaMalloc(&gramD, ddF64) != cudaSuccess) { cudaFree(bD); cudaFree(aD); delete[] bT; delete[] aT; return -4; }
    if (cudaMalloc(&crossD, ddF64) != cudaSuccess)
    {
        cudaFree(bD); cudaFree(aD); cudaFree(gramD);
        delete[] bT; delete[] aT;
        return -5;
    }
    // Host staging for each block's d×d partials; accumulated into the
    // caller's zero-initialised totals (fp64 authority, block-ordered).
    auto* gramStaged = new double[d * d];
    auto* crossStaged = new double[d * d];

    double alpha = 1.0, beta = 0.0;
    for (int base = 0; base < n; base += blockRows)
    {
        int rows = std::min(blockRows, n - base);
        const float* aBlock = a + (size_t)base * d;
        const float* bBlock = b + (size_t)base * d;
        // Widen the block to double and clear the tail beyond `rows` so
        // the fixed-size layout never reads stale rows.
        for (int i = 0; i < rows * d; i++)
        {
            bT[i] = bBlock[i];
            aT[i] = aBlock[i];
        }
        for (size_t i = (size_t)rows * d; i < (size_t)blockRows * d; i++)
        {
            bT[i] = 0.0;
            aT[i] = 0.0;
        }
        if (cudaMemcpyAsync(bD, bT, (size_t)blockRows * d * 8, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(aD, aT, (size_t)blockRows * d * 8, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            cudaFree(bD); cudaFree(aD); cudaFree(gramD); cudaFree(crossD);
            delete[] bT; delete[] aT;
            return -9;
        }
        // gram = BᵀB: the row-major block viewed column-major with ld=d is
        // Bᵀ ([d × blockRows]); transa='N' keeps Bᵀ, transb='T' recovers B,
        // so C[d,d] = Bᵀ·B. cross = BᵀA with A = aD likewise.
        if (cublasDgemm(gCtx.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                d, d, blockRows,
                &alpha, (const double*)bD, d, (const double*)aD, d, &beta, (double*)crossD, d) != CUBLAS_STATUS_SUCCESS)
        {
            cudaFree(bD); cudaFree(aD); cudaFree(gramD); cudaFree(crossD);
            delete[] bT; delete[] aT;
            return -11;
        }
        if (cublasDgemm(gCtx.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                d, d, blockRows,
                &alpha, (const double*)bD, d, (const double*)bD, d, &beta, (double*)gramD, d) != CUBLAS_STATUS_SUCCESS)
        {
            cudaFree(bD); cudaFree(aD); cudaFree(gramD); cudaFree(crossD);
            delete[] bT; delete[] aT;
            return -10;
        }
        // Download this block's partials and fp64-accumulate into the
        // caller's running totals (each block is a fresh beta=0 Dgemm).
        // All copies stay on the cublas stream — no default-stream ops in
        // the loop (same discipline as amql_cuda_gemm_transposed_b_f32).
        if (cudaMemcpyAsync(gramStaged, gramD, ddF64, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(crossStaged, crossD, ddF64, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            cudaFree(bD); cudaFree(aD); cudaFree(gramD); cudaFree(crossD);
            delete[] bT; delete[] aT; delete[] gramStaged; delete[] crossStaged;
            return -12;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
            cudaFree(bD); cudaFree(aD); cudaFree(gramD); cudaFree(crossD);
            delete[] bT; delete[] aT; delete[] gramStaged; delete[] crossStaged;
            return -13;
        }
        // cublas stores C column-major, so the downloaded partials are
        // Cᵀ relative to our row-major layout. gram = BᵀB is symmetric,
        // so it is layout-invariant; cross = BᵀA is not — accumulate the
        // transpose (crossOut[r·d+c] += staged[r + c·d]) so the solver's
        // cross matches the managed normal equations exactly.
        for (int i = 0; i < d * d; i++)
        {
            gramOut[i] += gramStaged[i];
        }
        for (int r = 0; r < d; r++)
        {
            for (int c = 0; c < d; c++)
            {
                crossOut[r * d + c] += crossStaged[r + c * d];
            }
        }
    }

    cudaFree(bD); cudaFree(aD); cudaFree(gramD); cudaFree(crossD);
    delete[] bT; delete[] aT; delete[] gramStaged; delete[] crossStaged;
    return 0;
}

// ── Elementwise kernels (P2) ──────────────────────────────────────────────

// RMSNorm: out[r,c] = in[r,c] * (weight[c] + wOff) / sqrt(mean_sq + eps)
// One thread per row (cols ≤ 16384 → one block per row with 256 threads
// doing the reduction), in-place on device memory.

__global__ void rms_norm_f32_kernel(
    float* __restrict__ x,       // [rows, cols] row-major, in-place
    const float* __restrict__ w, // [cols] weight
    float wOff, float eps,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float* rowPtr = x + (long)row * cols;

    // Sum of squares across the row (parallel reduction within the block).
    __shared__ float sSq[256];
    float local = 0.f;
    for (int c = tid; c < cols; c += stride)
    {
        float v = rowPtr[c];
        local += v * v;
    }
    sSq[tid] = local;
    __syncthreads();
    // Reduce within the block (cols ≤ 16384, so 256 threads is enough).
    for (int s = 128; s > 0; s >>= 1)
    {
        if (tid < s) sSq[tid] += sSq[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sSq[0] / (float)cols + eps);

    // Apply weight.
    for (int c = tid; c < cols; c += stride)
    {
        rowPtr[c] = rowPtr[c] * inv * (w[c] + wOff);
    }
}

AMQL_EXPORT int amql_cuda_rms_norm_f32(
    float* x, const float* w, float wOff, float eps,
    int rows, int cols, int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (rows <= 0 || cols <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    int threads = 256;
    rms_norm_f32_kernel<<<rows, threads, 0, stream>>>(x, w, wOff, eps, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Host-facing wrapper: copies x to device, runs the kernel there, copies
// back. Scratch buffers for the staging copies.
AMQL_EXPORT int amql_cuda_rms_norm_f32_host(
    float* x, const float* w, float wOff, float eps,
    int rows, int cols, int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (rows <= 0 || cols <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;
    long count = (long)rows * cols;
    size_t xBytes = (size_t)count * 4;
    size_t wBytes = (size_t)cols * 4;

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, xBytes) != 0) return -2;
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, wBytes) != 0) return -3;

    if (cudaMemcpyAsync(gCtx.scratchA, x, xBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -4;
    if (cudaMemcpyAsync(gCtx.scratchC, w, wBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -5;

    int threads = 256;
    rms_norm_f32_kernel<<<rows, threads, 0, stream>>>(
        (float*)gCtx.scratchA, (const float*)gCtx.scratchC, wOff, eps, rows, cols);
    if (cudaGetLastError() != cudaSuccess) return -6;

    if (cudaMemcpyAsync(x, gCtx.scratchA, xBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return -7;
    cudaStreamSynchronize(stream);
    return 0;
}

// SiLU: out[i] = x[i] / (1 + exp(-x[i])), element-wise in-place on device.

__global__ void silu_f32_kernel(float* __restrict__ x, long count)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = x[i];
    x[i] = v / (1.f + expf(-v));
}

AMQL_EXPORT int amql_cuda_silu_f32(float* x, int rows, int cols, int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (rows <= 0 || cols <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;
    long count = (long)rows * cols;
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    silu_f32_kernel<<<blocks, threads, 0, stream>>>(x, count);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_silu_f32_host(
    float* x, int rows, int cols, int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (rows <= 0 || cols <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;
    long count = (long)rows * cols;
    size_t bytes = (size_t)count * 4;

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, bytes) != 0) return -2;
    if (cudaMemcpyAsync(gCtx.scratchA, x, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -3;

    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    silu_f32_kernel<<<blocks, threads, 0, stream>>>((float*)gCtx.scratchA, count);
    if (cudaGetLastError() != cudaSuccess) return -4;

    if (cudaMemcpyAsync(x, gCtx.scratchA, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return -5;
    cudaStreamSynchronize(stream);
    return 0;
}

// RoPE: apply rotary position embedding to a [rows, heads * headDim] tensor,
// in-place on device.  Rotary pairs are (i, i + rotaryWidth/2) for i <
// pairCount within each head, where rotaryWidth ≤ headDim.

__global__ void rope_f32_kernel(
    float* __restrict__ x,   // [rows, heads * headDim] row-major, in-place
    const float* __restrict__ invFreq, // [pairCount] precomputed 1/theta^(2i/rotaryWidth)
    const int* __restrict__ positions, // [rows] absolute positions
    int rows, int heads, int headDim, int pairCount)
{
    int elem = (int)((long)blockIdx.x * blockDim.x + threadIdx.x);
    long total = (long)rows * heads * headDim;
    if (elem >= total) return;

    int col = elem % (heads * headDim);
    int row = elem / (heads * headDim);
    int h = col / headDim;
    int d = col % headDim;

    // Only the first rotaryWidth dims are rotated; the rest are pass-through.
    if (d >= pairCount * 2) return;

    int pair = d < pairCount ? d : d - pairCount;
    int otherOff = d < pairCount ? pairCount : -pairCount;
    int otherIdx = (int)((long)row * heads * headDim + (long)h * headDim + d + otherOff);

    float angle = (float)positions[row] * invFreq[pair];
    float c = cosf(angle);
    float s = sinf(angle);
    float v1 = x[elem];
    float v2 = x[otherIdx];
    if (d < pairCount)
    {
        x[elem] = v1 * c - v2 * s;
    }
    else
    {
        x[elem] = v1 * s + v2 * c;
    }
}

AMQL_EXPORT int amql_cuda_rope_f32(
    float* x, const float* invFreq, const int* positions,
    int rows, int heads, int headDim, int pairCount, int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (rows <= 0 || heads <= 0 || headDim <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    long total = (long)rows * heads * headDim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    rope_f32_kernel<<<blocks, threads, 0, stream>>>(
        x, invFreq, positions, rows, heads, headDim, pairCount);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_rope_f32_host(
    float* x, const float* invFreq, const int* positions,
    int rows, int heads, int headDim, int pairCount, int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (rows <= 0 || heads <= 0 || headDim <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    long total = (long)rows * heads * headDim;
    size_t xBytes = (size_t)total * 4;
    size_t freqBytes = (size_t)pairCount * 4;
    size_t posBytes = (size_t)rows * 4;

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, xBytes) != 0) return -2;
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, freqBytes + posBytes) != 0) return -3;

    float* dFreq = (float*)gCtx.scratchC;
    int* dPos = (int*)(dFreq + pairCount);

    if (cudaMemcpyAsync(gCtx.scratchA, x, xBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -4;
    if (cudaMemcpyAsync((void*)dFreq, invFreq, freqBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -5;
    if (cudaMemcpyAsync((void*)dPos, positions, posBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -6;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    rope_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)gCtx.scratchA, dFreq, dPos, rows, heads, headDim, pairCount);
    if (cudaGetLastError() != cudaSuccess) return -7;

    if (cudaMemcpyAsync(x, gCtx.scratchA, xBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return -8;
    cudaStreamSynchronize(stream);
    return 0;
}

// Embedding gather: out[r,c] = table[ids[r], c].  The full embedding table
// is copied to the device, ids are uploaded, rows are gathered, and the
// result is copied back.  Small-op only — the table upload dominates for
// large vocabularies.  P7 keeps the table resident.

__global__ void gather_rows_f32_kernel(
    const float* __restrict__ table, // [vocab, hidden] row-major
    const int* __restrict__ ids,     // [rows]
    float* __restrict__ out,         // [rows, hidden] row-major
    int rows, int hidden)
{
    int col = (int)((long)blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= hidden) return;
    for (int r = 0; r < rows; r++)
    {
        int id = ids[r];
        out[(long)r * hidden + col] = table[(long)id * hidden + col];
    }
}

AMQL_EXPORT int amql_cuda_gather_rows_f32_host(
    const float* table, int vocab, const int* ids, float* out,
    int rows, int hidden, int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (rows <= 0 || hidden <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    size_t tableBytes = (size_t)vocab * hidden * 4;
    size_t idBytes = (size_t)rows * 4;
    size_t outBytes = (size_t)rows * hidden * 4;

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, tableBytes) != 0) return -2;
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, outBytes) != 0) return -3;

    // Copy table to scratchA, ids to scratchPack, output will go to scratchC.
    if (scratch_reserve(&gCtx.scratchPack, &gCtx.scratchPackBytes, idBytes) != 0) return -4;

    if (cudaMemcpyAsync(gCtx.scratchA, table, tableBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -5;
    if (cudaMemcpyAsync(gCtx.scratchPack, ids, idBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -6;

    int threads = 256;
    int blocks = (hidden + threads - 1) / threads;
    gather_rows_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)gCtx.scratchA, (const int*)gCtx.scratchPack,
        (float*)gCtx.scratchC, rows, hidden);
    if (cudaGetLastError() != cudaSuccess) return -7;

    if (cudaMemcpyAsync(out, gCtx.scratchC, outBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return -8;
    cudaStreamSynchronize(stream);
    return 0;
}

// ── P3: Softmax attention + KV on device ──────────────────────────────────

// Fused GQA attention: QK^T + causal mask + window + sinks + softcap +
// softmax + PV.  One thread block per (query position, query head) pair.
// Uses shared memory for the score row and separate block-level reductions.
//
// Q: [seqQ, numQHeads * headDim] row-major
// K: [seqKV, numKvHeads * headDim] row-major
// V: [seqKV, numKvHeads * headDim] row-major
// out: [seqQ, numQHeads * headDim] row-major
// sinks: [numQHeads] per-head additive bias (nullptr → no sinks)
//
// GQA: q_head → kv_head = q_head * numKvHeads / numQHeads.

__global__ void attention_gqa_f32_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int seqQ, int seqKV,
    int numQHeads, int numKvHeads, int headDim,
    float scoreScale, float softCap, int window,
    const float* __restrict__ sinks,
    const int* __restrict__ qPos,
    const int* __restrict__ kvPos)
{
    // sData layout: [0..seqKV-1] = scores, [seqKV..seqKV+1] = max/exp
    extern __shared__ float sData[];

    int qi = blockIdx.x;
    int qh = blockIdx.y;
    if (qi >= seqQ || qh >= numQHeads) return;

    float* scores = sData;
    __shared__ float sRed[256]; // blockDim.x-sized reduction workspace

    int kvh = qh * numKvHeads / numQHeads;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 1. ── QK^T ──
    const float* qPtr = Q + (long)qi * numQHeads * headDim + (long)qh * headDim;

    float myMax = -INFINITY;
    int qp = qPos[qi];

    for (int j = tid; j < seqKV; j += stride)
    {
        int kp = kvPos[j];
        if (kp > qp) { scores[j] = -INFINITY; continue; }
        if (window > 0 && qp - kp >= window) { scores[j] = -INFINITY; continue; }

        const float* kPtr = K + (long)j * numKvHeads * headDim + (long)kvh * headDim;
        float dot = 0.f;
        for (int d = 0; d < headDim; d++)
            dot += qPtr[d] * kPtr[d];
        float s = dot * scoreScale;
        if (softCap > 0.f && !isinf(s))
            s = tanhf(s / softCap) * softCap;
        if (sinks && kp == 0)
            s += sinks[qh];
        scores[j] = s;
        if (s > myMax) myMax = s;
    }

    // 2. ── softmax ──
    // Block-wide max reduction.
    sRed[tid] = myMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && sRed[tid + s] > sRed[tid])
            sRed[tid] = sRed[tid + s];
        __syncthreads();
    }
    float maxScore = sRed[0];
    // Every thread must have read the max before sRed is reused for the sum
    // below — without this barrier thread 0 can overwrite sRed[0] with its
    // partial sum while a slower warp is still reading the max.
    __syncthreads();

    float mySum = 0.f;
    if (!isinf(maxScore))
    {
        for (int j = tid; j < seqKV; j += stride)
        {
            if (isinf(scores[j])) { scores[j] = 0.f; continue; }
            scores[j] = expf(scores[j] - maxScore);
            mySum += scores[j];
        }
    }
    else
    {
        for (int j = tid; j < seqKV; j += stride)
            scores[j] = 0.f;
    }

    // Block-wide sum reduction.
    sRed[tid] = mySum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sRed[tid] += sRed[tid + s];
        __syncthreads();
    }
    float sumExp = sRed[0];
    float invSum = (sumExp > 0.f) ? (1.f / sumExp) : 0.f;

    // Normalise scores in-place.
    for (int j = tid; j < seqKV; j += stride)
        scores[j] *= invSum;
    __syncthreads();

    // 3. ── PV ──
    float* outPtr = out + (long)qi * numQHeads * headDim + (long)qh * headDim;
    for (int d = tid; d < headDim; d += stride)
    {
        float acc = 0.f;
        for (int j = 0; j < seqKV; j++)
        {
            float w = scores[j];
            if (w == 0.f) continue;
            acc += w * V[(long)j * numKvHeads * headDim + (long)kvh * headDim + d];
        }
        outPtr[d] = acc;
    }
}

// Host staging wrapper: copies Q, K, V, positions to device, runs the
// fused kernel, copies output back.
AMQL_EXPORT int amql_cuda_attention_gqa_f32_host(
    const float* q, int seqQ,
    const float* kCache, int seqKV,
    const float* vCache,
    float* output,
    int numQHeads, int numKvHeads, int headDim,
    float scoreScale, float softCap, int window,
    const float* sinks,
    const int* qPositions,
    const int* kvPositions,
    int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (seqQ <= 0 || seqKV <= 0 || numQHeads <= 0 || headDim <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    size_t qBytes    = (size_t)seqQ * numQHeads * headDim * 4;
    size_t kvBytes   = (size_t)seqKV * numKvHeads * headDim * 4;
    size_t outBytes  = (size_t)seqQ * numQHeads * headDim * 4;
    size_t posBytes  = (size_t)(seqQ + seqKV) * 4;
    size_t sinkBytes = sinks ? (size_t)numQHeads * 4 : 0;

    // ScratchA: Q + output
    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, qBytes) != 0) return -2;
    // ScratchC: K + V
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, kvBytes * 2) != 0) return -3;
    // ScratchPack: positions + sinks
    if (scratch_reserve(&gCtx.scratchPack, &gCtx.scratchPackBytes, posBytes + sinkBytes) != 0) return -4;

    float* dQ  = (float*)gCtx.scratchA;
    float* dK  = (float*)gCtx.scratchC;
    float* dV  = dK + (long)seqKV * numKvHeads * headDim;
    int*   dPos = (int*)  gCtx.scratchPack;
    int*   dKvPos = dPos + seqQ;
    float* dSinks = sinks ? (float*)(dKvPos + seqKV) : nullptr;

    if (cudaMemcpyAsync(dQ, q, qBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -5;
    if (cudaMemcpyAsync(dK, kCache, kvBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -6;
    if (cudaMemcpyAsync(dV, vCache, kvBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -7;
    if (cudaMemcpyAsync(dPos, qPositions, (size_t)seqQ * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -8;
    if (cudaMemcpyAsync(dKvPos, kvPositions, (size_t)seqKV * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -9;
    if (sinks && cudaMemcpyAsync(dSinks, sinks, sinkBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return -10;

    int blockThreads = 256;
    dim3 grid(seqQ, numQHeads);
    size_t shmBytes = (size_t)seqKV * sizeof(float) + sizeof(float); // scores + max/exp

    attention_gqa_f32_kernel<<<grid, blockThreads, shmBytes, stream>>>(
        dQ, dK, dV, dQ, seqQ, seqKV, numQHeads, numKvHeads, headDim,
        scoreScale, softCap, window, dSinks, dPos, dKvPos);
    if (cudaGetLastError() != cudaSuccess) return -11;

    if (cudaMemcpyAsync(output, dQ, outBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return -12;
    cudaStreamSynchronize(stream);
    return 0;
}

// ── Device KV cache management ────────────────────────────────────────────

// Append new rows to a device ring buffer.
// newRows: [seqQ, numKvHeads * headDim] row-major
// cache:   [maxSeq, numKvHeads * headDim] pre-allocated
// Returns 0 on success, negative on error.  The kernel is a plain copy;
// no synchronisation — caller owns the ring position.
__global__ void copy_contig_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    long count)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i];
}

AMQL_EXPORT int amql_cuda_kv_append_f32(
    const float* newRows, int seqQ,
    float* cache, int maxSeq,
    int numKvHeads, int headDim, int writePos,
    int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (seqQ <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    long kvStride = (long)numKvHeads * headDim;
    long total = (long)seqQ * kvStride;
    if (writePos + seqQ > maxSeq) return -2; // ring overflow

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    copy_contig_f32_kernel<<<blocks, threads, 0, stream>>>(
        newRows, cache + (long)writePos * kvStride, total);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

// Allocate a device KV cache buffer.  amql_cuda_free_kv_cache to free.
AMQL_EXPORT int amql_cuda_alloc_kv_cache(float** ptr, int maxSeq, int numKvHeads, int headDim)
{
    if (ctx_ensure() != 0) return -1;
    size_t bytes = (size_t)maxSeq * numKvHeads * headDim * 4;
    cudaError_t e = cudaMalloc((void**)ptr, bytes);
    return e == cudaSuccess ? 0 : -2;
}

// Free a device KV cache buffer allocated by amql_cuda_alloc_kv_cache.
AMQL_EXPORT int amql_cuda_free_kv_cache(float* ptr)
{
    if (!ptr) return 0;
    cudaError_t e = cudaFree(ptr);
    return e == cudaSuccess ? 0 : -1;
}

// Read a contiguous slice of a device KV cache back to host.
// Reads rows [startRow, startRow+numRows) from cache into host buffer.
AMQL_EXPORT int amql_cuda_kv_read_f32(
    const float* cache, int startRow, int numRows,
    float* host, int numKvHeads, int headDim,
    int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (numRows <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    long kvStride = (long)numKvHeads * headDim;
    size_t bytes = (size_t)numRows * kvStride * 4;
    if (cudaMemcpyAsync(host, cache + (long)startRow * kvStride, bytes,
            cudaMemcpyDeviceToHost, stream) != cudaSuccess) return -2;
    cudaStreamSynchronize(stream);
    return 0;
}

// ── P6: cuSOLVER Cholesky + triangular solve ─────────────────────────────

// Solves AX = B where A is symmetric positive-definite [n × n], B is
// [n × nrhs] (both column-major).  Uses cusolverDnDpotrf + cusolverDnDpotrs.
// On success, B is overwritten with the solution X.  fp64 matches the
// managed Cholesky path exactly.

AMQL_EXPORT int amql_cuda_cholesky_solve_f64(
    double* A, int n,    // A[n*n] column-major, overwritten with Cholesky factor
    double* B, int nrhs, // B[n*nrhs] column-major, overwritten with solution
    int streamOrdinal)
{
    if (ctx_ensure() != 0) return -1;
    if (n <= 0 || nrhs <= 0) return 0;
    cudaStream_t stream = streamOrdinal == 0 ? gCtx.stream : 0;

    size_t aBytes = (size_t)n * n * sizeof(double);
    size_t bBytes = (size_t)n * nrhs * sizeof(double);

    double *dA = nullptr, *dB = nullptr;
    if (cudaMalloc(&dA, aBytes) != cudaSuccess) return -2;
    if (cudaMalloc(&dB, bBytes) != cudaSuccess) { cudaFree(dA); return -3; }

    if (cudaMemcpyAsync(dA, A, aBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        { cudaFree(dA); cudaFree(dB); return -4; }
    if (cudaMemcpyAsync(dB, B, bBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        { cudaFree(dA); cudaFree(dB); return -5; }

    int workspaceSize = 0;
    if (cusolverDnDpotrf_bufferSize(gCtx.cusolver, CUBLAS_FILL_MODE_LOWER, n, dA, n,
            &workspaceSize) != CUSOLVER_STATUS_SUCCESS)
        { cudaFree(dA); cudaFree(dB); return -6; }

    double* dWorkspace = nullptr;
    if (cudaMalloc((void**)&dWorkspace, (size_t)workspaceSize) != cudaSuccess)
        { cudaFree(dA); cudaFree(dB); return -7; }

    int* dInfo = nullptr;
    if (cudaMalloc(&dInfo, sizeof(int)) != cudaSuccess)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); return -8; }

    if (cusolverDnDpotrf(gCtx.cusolver, CUBLAS_FILL_MODE_LOWER, n, dA, n,
            dWorkspace, workspaceSize, dInfo) != CUSOLVER_STATUS_SUCCESS)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); cudaFree(dInfo); return -9; }

    int info = 0;
    if (cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); cudaFree(dInfo); return -10; }
    cudaStreamSynchronize(stream);
    if (info != 0)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); cudaFree(dInfo); return info; }

    if (cusolverDnDpotrs(gCtx.cusolver, CUBLAS_FILL_MODE_LOWER, n, nrhs, dA, n,
            dB, n, dInfo) != CUSOLVER_STATUS_SUCCESS)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); cudaFree(dInfo); return -11; }

    if (cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); cudaFree(dInfo); return -12; }
    cudaStreamSynchronize(stream);
    if (info != 0)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); cudaFree(dInfo); return info; }

    if (cudaMemcpyAsync(B, dB, bBytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        { cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace); cudaFree(dInfo); return -13; }
    cudaStreamSynchronize(stream);

    cudaFree(dA); cudaFree(dB); cudaFree((void*)dWorkspace);
    cudaFree(dInfo);
    return 0;
}
// ── NVFP4 weights resident as packs ───────────────────────────────────────
//
// A weight stored as NVFP4 (FP4 E2M1 elements two to a byte, low nibble
// first; one E4M3 scale per 16 elements of a row; an F32 tensor scale) stays
// on the device in that form: the packed elements, then (256-byte aligned)
// the block scales in cuBLASLt's tiled layout, so the tensor cores can read
// them as they are. 0.5625 bytes an element (plus tile padding) against
// FP16's 2. Products never widen the whole matrix:
//   - up to 32 activation rows (decode, and the small batches a routed
//     expert gets), 8 per pass, a warp per output row decodes its
//     packs in registers and accumulates in FP32 against FP32 activations;
//   - more rows (prefill), the activations are quantised to NVFP4 on the
//     device and the FP4 tensor cores multiply (cuBLASLt, block-scaled
//     E2M1 x E2M1, FP32 accumulate): W4A4, the arithmetic NVFP4 checkpoints
//     are trained for. Where cuBLASLt declines a shape, or with
//     AMQL_FP4_PREFILL=0, a tile of the weight is decoded to FP16 (exact)
//     for an FP16 cuBLAS GEMM instead.
//
// Scale tiles (cuBLAS "1D Block Scaling Factors Layout"): 128 rows x 4
// blocks of 16 elements per 512-byte tile, tiles row-major, rows padded to
// 128 and blocks to 4 with zero scales.

static void* gNvfp4Tile = nullptr;
static size_t gNvfp4TileBytes = 0;

static size_t nvfp4_scale_offset(int rows, int cols)
{
    size_t packed = (size_t)rows * (size_t)cols / 2;
    return (packed + 255) & ~(size_t)255;
}

__host__ __device__ __forceinline__ int fp4_scale_tiles_inner(int cols)
{
    return (cols / 16 + 3) / 4;
}

__host__ __device__ __forceinline__ size_t fp4_scale_bytes(int rows, int cols)
{
    return (size_t)((rows + 127) / 128) * 128 * (size_t)fp4_scale_tiles_inner(cols) * 4;
}

// Offset of the scale of (row, 16-element block) in the tiled layout.
__host__ __device__ __forceinline__ size_t fp4_scale_index(int row, int block, int tilesInner)
{
    return ((size_t)(row / 128) * tilesInner + block / 4) * 512
         + (size_t)(row % 32) * 16 + (size_t)((row % 128) / 32) * 4 + (size_t)(block % 4);
}

__device__ __forceinline__ float e4m3_decoded(unsigned char b)
{
    int exponent = (b >> 3) & 0x0F;
    int mantissa = b & 0x07;
    float v;
    if (exponent == 0)
    {
        v = ldexpf((float)mantissa, -9);              // subnormal: m/8 x 2^-6
    }
    else if (exponent == 15 && mantissa == 7)
    {
        v = 0.f;                                      // NaN: pinned to 0
    }
    else
    {
        v = ldexpf(1.f + (float)mantissa * 0.125f, exponent - 7);
    }
    return (b & 0x80) ? -v : v;
}

// Hardware conversions (sm_89+ for E4M3, Blackwell for E2M1): two elements
// to an FP16 pair per instruction, the low byte or nibble first. A lookup
// table indexed by the element would serialise across the warp.
__device__ __forceinline__ float2 e4m3x2_decoded(unsigned short pair)
{
    __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)pair, __NV_E4M3);
    return __half22float2(*reinterpret_cast<__half2*>(&raw));
}

__device__ __forceinline__ float2 e2m1x2_decoded(unsigned char pair)
{
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)pair, __NV_E2M1);
    return __half22float2(*reinterpret_cast<__half2*>(&raw));
}

__global__ void tile_fp4_scales(const unsigned char* __restrict__ linear, unsigned char* __restrict__ tiled,
    int rows, int cols)
{
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int blocks = cols / 16;
    if (index >= (long)rows * blocks)
    {
        return;
    }
    int row = (int)(index / blocks);
    int block = (int)(index % blocks);
    tiled[fp4_scale_index(row, block, fp4_scale_tiles_inner(cols))] = linear[index];
}

template <int M>
__global__ void nvfp4_gemv(
    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scales,
    const float* __restrict__ a, float* __restrict__ c, int n, int k, float global)
{
    int row = (int)((blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (row >= n)
    {
        return;
    }
    int blocks = k / 16;
    int tilesInner = fp4_scale_tiles_inner(k);
    const unsigned char* rowPack = packed + (size_t)row * (k / 2);
    float acc[M];
#pragma unroll
    for (int i = 0; i < M; i++)
    {
        acc[i] = 0.f;
    }
    for (int b = lane; b < blocks; b += 32)
    {
        float scale = e4m3_decoded(scales[fp4_scale_index(row, b, tilesInner)]);
        uint2 bytes = *reinterpret_cast<const uint2*>(rowPack + (size_t)b * 8);
        float w[16];
#pragma unroll
        for (int j = 0; j < 16; j += 2)
        {
            unsigned int word = (j >> 3) == 0 ? bytes.x : bytes.y;
            float2 pair = e2m1x2_decoded((unsigned char)((word >> (((j >> 1) & 3) * 8)) & 0xFF));
            w[j] = pair.x * scale;
            w[j + 1] = pair.y * scale;
        }
#pragma unroll
        for (int i = 0; i < M; i++)
        {
            const float4* x = reinterpret_cast<const float4*>(a + (size_t)i * k + (size_t)b * 16);
            float4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
            acc[i] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w
                    + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w
                    + w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w
                    + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
        }
    }
#pragma unroll
    for (int i = 0; i < M; i++)
    {
        float v = acc[i];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane == 0)
        {
            c[(size_t)i * n + row] = v * global;
        }
    }
}

static const int kGemvRows = 32;
// Up to kGemvRows activation rows through the in-register GEMV, 8 at a time
// (each group re-reads the weight, from L2 after the first): for the small
// batches a routed expert gets, cheaper than quantising for the tensor cores.
static void nvfp4_gemv_rows(const unsigned char* packed, const unsigned char* scales, float global, const float* aDev, float* cDev, int m, int n, int k, cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (int)(((long)n * 32 + threads - 1) / threads);
    for (int done = 0; done < m; done += 8)
    {
        const float* a = aDev + (size_t)done * k;
        float* c = cDev + (size_t)done * n;
        switch (std::min(8, m - done))
        {
            case 1: nvfp4_gemv<1><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
            case 2: nvfp4_gemv<2><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
            case 3: nvfp4_gemv<3><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
            case 4: nvfp4_gemv<4><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
            case 5: nvfp4_gemv<5><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
            case 6: nvfp4_gemv<6><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
            case 7: nvfp4_gemv<7><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
            default: nvfp4_gemv<8><<<blocks, threads, 0, stream>>>(packed, scales, a, c, n, k, global); break;
        }
    }
}

__global__ void dequant_nvfp4_to_f16(
    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scales,
    __half* __restrict__ out, int firstRow, int rows, int k)
{
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long)rows * k)
    {
        return;
    }
    int row = firstRow + (int)(index / k);
    int col = (int)(index % k);
    unsigned char byte = packed[(size_t)row * (k / 2) + col / 2];
    float2 pair = e2m1x2_decoded(byte);
    float scale = e4m3_decoded(scales[fp4_scale_index(row, col / 16, fp4_scale_tiles_inner(k))]);
    out[index] = __float2half_rn(((col & 1) ? pair.y : pair.x) * scale);
}

// ── Activations to NVFP4 (the prefill's B operand) ───────────────────────

__global__ void abs_max(const float* __restrict__ x, long count, unsigned int* __restrict__ result)
{
    float local = 0.f;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (long)gridDim.x * blockDim.x)
    {
        local = fmaxf(local, fabsf(x[i]));
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local = fmaxf(local, __shfl_down_sync(0xffffffff, local, offset));
    }
    if ((threadIdx.x & 31) == 0)
    {
        atomicMax(result, __float_as_uint(local));    // non-negative floats order as their bits
    }
}

// One thread per 16-element block of an activation row: ModelOpt's NVFP4
// recipe with the tensor scale from this call's absolute maximum. Thread 0
// also writes alpha = (weight tensor scale) x (activation tensor scale).
__global__ void quantise_nvfp4_rows(
    const float* __restrict__ x, int m, int k, const unsigned int* __restrict__ amaxBits,
    unsigned char* __restrict__ packed, unsigned char* __restrict__ scales, float weightGlobal,
    float* __restrict__ alphaBeta)
{
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int blocks = k / 16;
    float amax = __uint_as_float(*amaxBits);
    float global = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
    if (index == 0)
    {
        alphaBeta[0] = weightGlobal * global;
        alphaBeta[1] = 0.f;
    }
    if (index >= (long)m * blocks)
    {
        return;
    }
    int row = (int)(index / blocks);
    int block = (int)(index % blocks);
    const float* src = x + (size_t)row * k + (size_t)block * 16;
    float v[16];
    float peak = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        v[i] = src[i];
        peak = fmaxf(peak, fabsf(v[i]));
    }
    __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(peak / 6.f / global, __NV_SATFINITE, __NV_E4M3);
    float step = e4m3_decoded((unsigned char)code) * global;
    float inv = step > 0.f ? 1.f / step : 0.f;
    unsigned char* out = packed + (size_t)row * (k / 2) + (size_t)block * 8;
#pragma unroll
    for (int i = 0; i < 16; i += 2)
    {
        out[i / 2] = (unsigned char)__nv_cvt_float2_to_fp4x2(make_float2(v[i] * inv, v[i + 1] * inv),
            __NV_E2M1, cudaRoundNearest);
    }
    scales[fp4_scale_index(row, block, fp4_scale_tiles_inner(k))] = (unsigned char)code;
}

// ── cuBLASLt plans, one per (m, n, k) ─────────────────────────────────────

#include <map>
#include <tuple>

struct Fp4Plan
{
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool ok = false;
};

static cublasLtHandle_t gLt = nullptr;
static std::map<std::tuple<int, int, int>, Fp4Plan> gFp4Plans;
static void* gLtWorkspace = nullptr;
static const size_t kLtWorkspaceBytes = 32u << 20;
static void* gActPacked = nullptr;   static size_t gActPackedBytes = 0;
static void* gActScales = nullptr;   static size_t gActScalesBytes = 0;
static void* gAmax = nullptr;        // unsigned int, then float alpha, beta

static int gFp4PrefillState = -1;

static bool fp4_prefill_enabled()
{
    if (gFp4PrefillState < 0)
    {
        const char* env = getenv("AMQL_FP4_PREFILL");
        gFp4PrefillState = (env != nullptr && (env[0] == '0' || env[0] == 'f' || env[0] == 'o')) ? 0 : 1;
    }
    return gFp4PrefillState == 1;
}

/// Turns the FP4 tensor-core prefill on (1) or off (0, FP16 tiles instead).
AMQL_EXPORT void amql_cuda_set_fp4_prefill(int enabled)
{
    gFp4PrefillState = enabled != 0 ? 1 : 0;
}

// D[n,m] (column-major, = C[m,n] row-major) = W[n,k] x X[m,k]^T: the weight
// is A (k x n column-major, transposed), the activations B (k x m, as is).
static Fp4Plan* fp4_plan(int m, int n, int k, void* aScales, void* bScales)
{
    auto key = std::make_tuple(m, n, k);
    auto found = gFp4Plans.find(key);
    if (found != gFp4Plans.end())
    {
        return found->second.ok ? &found->second : nullptr;
    }
    Fp4Plan& plan = gFp4Plans[key];
    if (gLt == nullptr && cublasLtCreate(&gLt) != CUBLAS_STATUS_SUCCESS)
    {
        return nullptr;
    }
    cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_DEVICE;
    if (cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScales, sizeof(aScales)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScales, sizeof(bScales)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_4F_E2M1, k, n, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_4F_E2M1, k, m, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan.d, CUDA_R_32F, n, m, n) != CUBLAS_STATUS_SUCCESS)
    {
        return nullptr;
    }
    cublasLtMatmulPreference_t preference = nullptr;
    size_t workspace = kLtWorkspaceBytes;
    cublasLtMatmulHeuristicResult_t result{};
    int found_count = 0;
    if (cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulAlgoGetHeuristic(gLt, plan.desc, plan.a, plan.b, plan.d, plan.d, preference, 1, &result, &found_count) == CUBLAS_STATUS_SUCCESS &&
        found_count > 0)
    {
        plan.algo = result.algo;
        plan.ok = true;
    }
    if (preference != nullptr)
    {
        cublasLtMatmulPreferenceDestroy(preference);
    }
    return plan.ok ? &plan : nullptr;
}

// The tensor-core prefill: quantise, multiply. Returns 0, or nonzero to
// fall back (nothing has been written to C that matters).
static int nvfp4_tensor_core_gemm(const unsigned char* packed, const unsigned char* scales,
    const float* aDev, float* cDev, int m, int k, int n, float global, cudaStream_t stream)
{
    if (!fp4_prefill_enabled())
    {
        return -1;
    }
    size_t actPacked = (size_t)m * k / 2;
    size_t actScales = fp4_scale_bytes(m, k);
    if (scratch_reserve(&gActPacked, &gActPackedBytes, actPacked) != 0 ||
        scratch_reserve(&gActScales, &gActScalesBytes, actScales) != 0 ||
        (gAmax == nullptr && cudaMalloc(&gAmax, 16) != cudaSuccess) ||
        (gLtWorkspace == nullptr && cudaMalloc(&gLtWorkspace, kLtWorkspaceBytes) != cudaSuccess))
    {
        return -2;
    }
    // Scale pointers are plan attributes; the scratch only moves when it
    // grows, so a plan is rebuilt for a new activation scale buffer.
    Fp4Plan* plan = fp4_plan(m, n, k, (void*)scales, gActScales);
    if (plan == nullptr)
    {
        return -3;
    }
    void* aScalesAttr = (void*)scales;
    void* bScalesAttr = gActScales;
    cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScalesAttr, sizeof(aScalesAttr));
    cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScalesAttr, sizeof(bScalesAttr));

    unsigned int* amax = (unsigned int*)gAmax;
    float* alphaBeta = (float*)((unsigned char*)gAmax + 8);
    long count = (long)m * k;
    if (cudaMemsetAsync(amax, 0, 4, stream) != cudaSuccess ||
        cudaMemsetAsync(gActScales, 0, actScales, stream) != cudaSuccess)
    {
        return -4;
    }
    abs_max<<<(int)std::min<long>(1024, (count + 255) / 256), 256, 0, stream>>>(aDev, count, amax);
    long blocks = (long)m * (k / 16);
    quantise_nvfp4_rows<<<(int)((blocks + 255) / 256), 256, 0, stream>>>(
        aDev, m, k, amax, (unsigned char*)gActPacked, (unsigned char*)gActScales, global, alphaBeta);
    if (cudaGetLastError() != cudaSuccess)
    {
        return -5;
    }
    cublasStatus_t status = cublasLtMatmul(gLt, plan->desc,
        alphaBeta, packed, plan->a, gActPacked, plan->b,
        alphaBeta + 1, cDev, plan->d, cDev, plan->d,
        &plan->algo, gLtWorkspace, kLtWorkspaceBytes, stream);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -6;
}

AMQL_EXPORT int amql_cuda_nvfp4_upload(
    const unsigned char* packed, const unsigned char* scales, int rows, int cols, void** out)
{
    *out = nullptr;
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (rows <= 0 || cols <= 0 || cols % 16 != 0)
    {
        return -2;
    }
    size_t offset = nvfp4_scale_offset(rows, cols);
    size_t linearScales = (size_t)rows * (cols / 16);
    size_t tiledScales = fp4_scale_bytes(rows, cols);
    void* dev = nullptr;
    if (cudaMalloc(&dev, offset + tiledScales) != cudaSuccess)
    {
        return -3;
    }
    // The linear scales pass through the A staging scratch on their way to
    // their tiles.
    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, linearScales) != 0 ||
        cudaMemcpy(dev, packed, (size_t)rows * cols / 2, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(gCtx.scratchA, scales, linearScales, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset((unsigned char*)dev + offset, 0, tiledScales) != cudaSuccess)
    {
        cudaFree(dev);
        return -4;
    }
    tile_fp4_scales<<<(int)((linearScales + 255) / 256), 256>>>(
        (const unsigned char*)gCtx.scratchA, (unsigned char*)dev + offset, rows, cols);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(dev);
        return -5;
    }
    *out = dev;
    return 0;
}

/// Device bytes an NVFP4 upload of a [rows, cols] weight takes.
AMQL_EXPORT long long amql_cuda_nvfp4_bytes(int rows, int cols)
{
    return (long long)(nvfp4_scale_offset(rows, cols) + fp4_scale_bytes(rows, cols));
}

AMQL_EXPORT int amql_cuda_nvfp4_gemm(
    const float* a,        // [m,k] row-major host
    const void* w,         // NVFP4 upload of an [n,k] weight
    float* c,              // [m,n] row-major host out
    int m, int k, int n, float global)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0)
    {
        return 0;
    }
    if (k % 16 != 0)
    {
        return -2;
    }
    cudaStream_t stream = gCtx.stream;
    const unsigned char* packed = (const unsigned char*)w;
    const unsigned char* scales = packed + nvfp4_scale_offset(n, k);

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)m * k * 4) != 0 ||
        scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
    {
        return -3;
    }
    if (cudaMemcpyAsync(gCtx.scratchA, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return -4;
    }
    float* cDev = (float*)gCtx.scratchC;
    const float* aDev = (const float*)gCtx.scratchA;

    if (m <= kGemvRows)
    {
        nvfp4_gemv_rows(packed, scales, global, aDev, cDev, m, n, k, stream);
        if (cudaGetLastError() != cudaSuccess)
        {
            return -5;
        }
    }
    else if (nvfp4_tensor_core_gemm(packed, scales, aDev, cDev, m, k, n, global, stream) != 0)
    {
        cudaGetLastError(); // clear anything the declined path left
        if (scratch_reserve(&gCtx.scratchAF16, &gCtx.scratchAF16Bytes, (size_t)m * k * 2) != 0)
        {
            return -6;
        }
        long actCount = (long)m * k;
        cast_f32_to_f16<<<(int)((actCount + 255) / 256), 256, 0, stream>>>(
            aDev, (__half*)gCtx.scratchAF16, actCount);
        // Decode the weight a tile of rows at a time (at most 64 MiB of FP16).
        int tileRows = (int)std::max<long>(1, std::min<long>(n, (32L << 20) / k));
        if (scratch_reserve(&gNvfp4Tile, &gNvfp4TileBytes, (size_t)tileRows * k * 2) != 0)
        {
            return -7;
        }
        float beta = 0.f;
        for (int first = 0; first < n; first += tileRows)
        {
            int rows = std::min(tileRows, n - first);
            long count = (long)rows * k;
            dequant_nvfp4_to_f16<<<(int)((count + 255) / 256), 256, 0, stream>>>(
                packed, scales, (__half*)gNvfp4Tile, first, rows, k);
            if (cudaGetLastError() != cudaSuccess)
            {
                return -8;
            }
            // D[n,m] column-major (= C[m,n] row-major); this tile is rows
            // [first, first + rows) of D, so it starts at cDev + first, ldc n.
            if (cublasGemmEx(gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    rows, m, k,
                    &global,
                    gNvfp4Tile, CUDA_R_16F, k,
                    gCtx.scratchAF16, CUDA_R_16F, k,
                    &beta,
                    cDev + first, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            {
                return -9;
            }
        }
    }
    if (cudaMemcpyAsync(c, cDev, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return -10;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -11;
}

/// Whether NVFP4 products of this shape run on the tensor cores: 1 yes, 0 FP16 tiles.
AMQL_EXPORT int amql_cuda_nvfp4_tensor_cores(int m, int n, int k)
{
    auto found = gFp4Plans.find(std::make_tuple(m, n, k));
    return fp4_prefill_enabled() && found != gFp4Plans.end() && found->second.ok ? 1 : 0;
}

// ── MXFP8 weights resident as stored ──────────────────────────────────────
//
// A weight stored as MXFP8 (E4M3 elements, one E8M0 power-of-two scale per
// 32 elements of a row) stays on the device in that form: 1.03 bytes an
// element against FP16's 2. One allocation, the elements then (256-byte
// aligned) the scale exponents. As for NVFP4:
//   - up to 32 activation rows (8 per pass), a warp per output row decodes its elements in
//     registers (the hardware E4M3 conversion) and accumulates in FP32;
//   - more rows, a tile is decoded to FP16 for cuBLAS. E4M3 times a power of
//     two is exact in FP16 while the product stays inside FP16's range,
//     which a trained weight's does.

static size_t mxfp8_scale_offset(int rows, int cols)
{
    size_t elements = (size_t)rows * (size_t)cols;
    return (elements + 255) & ~(size_t)255;
}

template <int M>
__global__ void mxfp8_gemv(
    const unsigned char* __restrict__ elements, const unsigned char* __restrict__ scales,
    const float* __restrict__ a, float* __restrict__ c, int n, int k)
{
    int row = (int)((blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (row >= n)
    {
        return;
    }
    // Lanes take 16-element halves of the 32-element blocks, so a warp reads
    // 512 contiguous bytes of the row at a time.
    int halves = k / 16;
    const unsigned char* rowElements = elements + (size_t)row * k;
    const unsigned char* rowScale = scales + (size_t)row * (k / 32);
    float acc[M];
#pragma unroll
    for (int i = 0; i < M; i++)
    {
        acc[i] = 0.f;
    }
    for (int h = lane; h < halves; h += 32)
    {
        float scale = e8m0_decoded(rowScale[h >> 1]);
        uint4 bytes = *reinterpret_cast<const uint4*>(rowElements + (size_t)h * 16);
        unsigned int words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        float w[16];
#pragma unroll
        for (int j = 0; j < 16; j += 2)
        {
            float2 pair = e4m3x2_decoded((unsigned short)((words[j >> 2] >> ((j & 2) * 8)) & 0xFFFF));
            w[j] = pair.x * scale;
            w[j + 1] = pair.y * scale;
        }
#pragma unroll
        for (int i = 0; i < M; i++)
        {
            const float4* x = reinterpret_cast<const float4*>(a + (size_t)i * k + (size_t)h * 16);
            float4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
            acc[i] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w
                    + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w
                    + w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w
                    + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
        }
    }
#pragma unroll
    for (int i = 0; i < M; i++)
    {
        float v = acc[i];
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane == 0)
        {
            c[(size_t)i * n + row] = v;
        }
    }
}

// Up to kGemvRows activation rows through the in-register GEMV, 8 at a time
// (each group re-reads the weight, from L2 after the first): for the small
// batches a routed expert gets, cheaper than quantising for the tensor cores.
static void mxfp8_gemv_rows(const unsigned char* elements, const unsigned char* scales, const float* aDev, float* cDev, int m, int n, int k, cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (int)(((long)n * 32 + threads - 1) / threads);
    for (int done = 0; done < m; done += 8)
    {
        const float* a = aDev + (size_t)done * k;
        float* c = cDev + (size_t)done * n;
        switch (std::min(8, m - done))
        {
            case 1: mxfp8_gemv<1><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
            case 2: mxfp8_gemv<2><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
            case 3: mxfp8_gemv<3><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
            case 4: mxfp8_gemv<4><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
            case 5: mxfp8_gemv<5><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
            case 6: mxfp8_gemv<6><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
            case 7: mxfp8_gemv<7><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
            default: mxfp8_gemv<8><<<blocks, threads, 0, stream>>>(elements, scales, a, c, n, k); break;
        }
    }
}

__global__ void dequant_mxfp8_to_f16(
    const unsigned char* __restrict__ elements, const unsigned char* __restrict__ scales,
    __half* __restrict__ out, int rows, int k)
{
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long)rows * k)
    {
        return;
    }
    long row = index / k;
    int col = (int)(index % k);
    float scale = e8m0_decoded(scales[row * (k / 32) + col / 32]);
    out[index] = __float2half_rn(e4m3x2_decoded(elements[index]).x * scale);
}

AMQL_EXPORT int amql_cuda_mxfp8_upload(
    const unsigned char* elements, const unsigned char* scales, int rows, int cols, void** out)
{
    *out = nullptr;
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (rows <= 0 || cols <= 0 || cols % 32 != 0)
    {
        return -2;
    }
    size_t offset = mxfp8_scale_offset(rows, cols);
    size_t scaleBytes = (size_t)rows * (cols / 32);
    void* dev = nullptr;
    if (cudaMalloc(&dev, offset + scaleBytes) != cudaSuccess)
    {
        return -3;
    }
    if (cudaMemcpy(dev, elements, (size_t)rows * cols, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy((unsigned char*)dev + offset, scales, scaleBytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(dev);
        return -4;
    }
    *out = dev;
    return 0;
}

AMQL_EXPORT int amql_cuda_mxfp8_gemm(
    const float* a,        // [m,k] row-major host
    const void* w,         // MXFP8 upload of an [n,k] weight
    float* c,              // [m,n] row-major host out
    int m, int k, int n)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0)
    {
        return 0;
    }
    if (k % 32 != 0)
    {
        return -2;
    }
    cudaStream_t stream = gCtx.stream;
    const unsigned char* elements = (const unsigned char*)w;
    const unsigned char* scales = elements + mxfp8_scale_offset(n, k);

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)m * k * 4) != 0 ||
        scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
    {
        return -3;
    }
    if (cudaMemcpyAsync(gCtx.scratchA, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return -4;
    }
    float* cDev = (float*)gCtx.scratchC;
    const float* aDev = (const float*)gCtx.scratchA;

    if (m <= kGemvRows)
    {
        mxfp8_gemv_rows(elements, scales, aDev, cDev, m, n, k, stream);
        if (cudaGetLastError() != cudaSuccess)
        {
            return -5;
        }
    }
    else
    {
        if (scratch_reserve(&gCtx.scratchAF16, &gCtx.scratchAF16Bytes, (size_t)m * k * 2) != 0)
        {
            return -6;
        }
        long actCount = (long)m * k;
        cast_f32_to_f16<<<(int)((actCount + 255) / 256), 256, 0, stream>>>(
            (const float*)gCtx.scratchA, (__half*)gCtx.scratchAF16, actCount);
        int tileRows = (int)std::max<long>(1, std::min<long>(n, (32L << 20) / k));
        if (scratch_reserve(&gNvfp4Tile, &gNvfp4TileBytes, (size_t)tileRows * k * 2) != 0)
        {
            return -7;
        }
        float alpha = 1.f, beta = 0.f;
        for (int first = 0; first < n; first += tileRows)
        {
            int rows = std::min(tileRows, n - first);
            long count = (long)rows * k;
            dequant_mxfp8_to_f16<<<(int)((count + 255) / 256), 256, 0, stream>>>(
                elements + (size_t)first * k, scales + (size_t)first * (k / 32),
                (__half*)gNvfp4Tile, rows, k);
            if (cudaGetLastError() != cudaSuccess)
            {
                return -8;
            }
            if (cublasGemmEx(gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    rows, m, k,
                    &alpha,
                    gNvfp4Tile, CUDA_R_16F, k,
                    gCtx.scratchAF16, CUDA_R_16F, k,
                    &beta,
                    cDev + first, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            {
                return -9;
            }
        }
    }
    if (cudaMemcpyAsync(c, cDev, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return -10;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -11;
}

// ── Device-resident FFN: activations stay on the GPU between products ─────
//
// The managed path multiplies one weight at a time, copying the activations
// up and the product down each time; for a routed MoE layer in prefill that
// is most of the time spent (each picked expert re-uploads its rows and
// downloads both of its intermediates). amql_cuda_ffn runs a whole gated
// FFN, or every device-resident expert of a routed layer, as one call:
// the activations go up once, each job gathers its rows, multiplies gate
// and up, applies SiLU(gate) * up, multiplies down and adds its weighted
// rows into the output on the device, and the output comes down once.
//
// Weight formats per product: 1 NVFP4 upload, 2 MXFP8 upload, 3 FP16.

static void* gFfnX = nullptr;     static size_t gFfnXBytes = 0;
static void* gFfnRows = nullptr;  static size_t gFfnRowsBytes = 0;
static void* gFfnG = nullptr;     static size_t gFfnGBytes = 0;
static void* gFfnU = nullptr;     static size_t gFfnUBytes = 0;
static void* gFfnY = nullptr;     static size_t gFfnYBytes = 0;
static void* gFfnOut = nullptr;   static size_t gFfnOutBytes = 0;
static void* gFfnIdx = nullptr;   static size_t gFfnIdxBytes = 0;
static void* gFfnW = nullptr;     static size_t gFfnWBytes = 0;

__global__ void gather_rows(const float* __restrict__ x, const int* __restrict__ rows, int count, int k,
    float* __restrict__ out)
{
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long)count * k)
    {
        return;
    }
    int r = (int)(index / k);
    out[index] = x[(size_t)rows[r] * k + index % k];
}

__global__ void silu_times(float* __restrict__ gate, const float* __restrict__ up, long count)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        float g = gate[i];
        gate[i] = g / (1.f + expf(-g)) * up[i];
    }
}

__global__ void scatter_add_rows(const float* __restrict__ y, const int* __restrict__ rows,
    const float* __restrict__ weights, int count, int width, float* __restrict__ out)
{
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long)count * width)
    {
        return;
    }
    int r = (int)(index / width);
    out[(size_t)rows[r] * width + index % width] += weights[r] * y[index];
}

// One product C[m,n] = A[m,k] W[n,k]^T with A and C on the device.
static int device_product(int format, const void* w, float global,
    const float* aDev, float* cDev, int m, int k, int n, cudaStream_t stream)
{
    if (format == 1)
    {
        const unsigned char* packed = (const unsigned char*)w;
        const unsigned char* scales = packed + nvfp4_scale_offset(n, k);
        if (k % 16 != 0)
        {
            return -1;
        }
        if (m <= kGemvRows)
        {
            nvfp4_gemv_rows(packed, scales, global, aDev, cDev, m, n, k, stream);
            return cudaGetLastError() == cudaSuccess ? 0 : -2;
        }
        if (nvfp4_tensor_core_gemm(packed, scales, aDev, cDev, m, k, n, global, stream) == 0)
        {
            return 0;
        }
        cudaGetLastError();
    }
    else if (format == 2)
    {
        if (k % 32 != 0)
        {
            return -1;
        }
        const unsigned char* elements = (const unsigned char*)w;
        const unsigned char* scales = elements + mxfp8_scale_offset(n, k);
        if (m <= kGemvRows)
        {
            mxfp8_gemv_rows(elements, scales, aDev, cDev, m, n, k, stream);
            return cudaGetLastError() == cudaSuccess ? 0 : -2;
        }
    }
    else if (format != 3)
    {
        return -3;
    }

    // FP16 activations against FP16 weight tiles (decoded for formats 1 and
    // 2, resident for 3), FP32 accumulate.
    if (scratch_reserve(&gCtx.scratchAF16, &gCtx.scratchAF16Bytes, (size_t)m * k * 2) != 0)
    {
        return -4;
    }
    long actCount = (long)m * k;
    cast_f32_to_f16<<<(int)((actCount + 255) / 256), 256, 0, stream>>>(aDev, (__half*)gCtx.scratchAF16, actCount);
    float beta = 0.f;
    if (format == 3)
    {
        return cublasGemmEx(gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &global,
            w, CUDA_R_16F, k, gCtx.scratchAF16, CUDA_R_16F, k, &beta,
            cDev, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -5;
    }
    int tileRows = (int)std::max<long>(1, std::min<long>(n, (32L << 20) / k));
    if (scratch_reserve(&gNvfp4Tile, &gNvfp4TileBytes, (size_t)tileRows * k * 2) != 0)
    {
        return -6;
    }
    for (int first = 0; first < n; first += tileRows)
    {
        int rows = std::min(tileRows, n - first);
        long count = (long)rows * k;
        if (format == 1)
        {
            const unsigned char* packed = (const unsigned char*)w;
            dequant_nvfp4_to_f16<<<(int)((count + 255) / 256), 256, 0, stream>>>(
                packed, packed + nvfp4_scale_offset(n, k), (__half*)gNvfp4Tile, first, rows, k);
        }
        else
        {
            const unsigned char* elements = (const unsigned char*)w;
            dequant_mxfp8_to_f16<<<(int)((count + 255) / 256), 256, 0, stream>>>(
                elements + (size_t)first * k, elements + mxfp8_scale_offset(n, k) + (size_t)first * (k / 32),
                (__half*)gNvfp4Tile, rows, k);
        }
        if (cudaGetLastError() != cudaSuccess ||
            cublasGemmEx(gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, m, k, &global,
                gNvfp4Tile, CUDA_R_16F, k, gCtx.scratchAF16, CUDA_R_16F, k, &beta,
                cDev + first, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        {
            return -7;
        }
    }
    return 0;
}

// ── Grouped expert GEMV: every routed expert's product in one launch ─────
//
// A routed layer in decode gives each picked expert a row or two. Run one
// expert at a time and a token costs six launches an expert a layer
// (gather, gate, up, SiLU, down, scatter): ~2,900 for 10 experts over 48
// layers, most of them too small to fill the GPU. As in grouped GEMM for
// dropless MoE (one kernel over every expert's actual rows, no padding to a
// capacity), a task here is one NVFP4 product over up to 8 rows, and one
// launch runs them all: blockIdx.y picks the task, the warps of blockIdx.x
// its output rows. Each warp's arithmetic is nvfp4_gemv's, term for term,
// so the result is bit for bit the one-expert-at-a-time path's.

struct Fp4Task
{
    const unsigned char* packed;
    const unsigned char* scales;
    const float* a;      // rows of the input, k apart
    float* c;            // rows of the output, n apart
    float global;
    int rows;            // 1..8
    int n;
    int k;
};

__global__ void nvfp4_grouped_gemv(const Fp4Task* __restrict__ tasks)
{
    const Fp4Task t = tasks[blockIdx.y];
    int row = (int)((blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (row >= t.n)
    {
        return;
    }
    int k = t.k;
    int blocks = k / 16;
    int tilesInner = fp4_scale_tiles_inner(k);
    const unsigned char* rowPack = t.packed + (size_t)row * (k / 2);
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        acc[i] = 0.f;
    }
    for (int b = lane; b < blocks; b += 32)
    {
        float scale = e4m3_decoded(t.scales[fp4_scale_index(row, b, tilesInner)]);
        uint2 bytes = *reinterpret_cast<const uint2*>(rowPack + (size_t)b * 8);
        float w[16];
#pragma unroll
        for (int j = 0; j < 16; j += 2)
        {
            unsigned int word = (j >> 3) == 0 ? bytes.x : bytes.y;
            float2 pair = e2m1x2_decoded((unsigned char)((word >> (((j >> 1) & 3) * 8)) & 0xFF));
            w[j] = pair.x * scale;
            w[j + 1] = pair.y * scale;
        }
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            if (i < t.rows)
            {
                const float4* x = reinterpret_cast<const float4*>(t.a + (size_t)i * k + (size_t)b * 16);
                float4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
                acc[i] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w
                        + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w
                        + w[8] * x2.x + w[9] * x2.y + w[10] * x2.z + w[11] * x2.w
                        + w[12] * x3.x + w[13] * x3.y + w[14] * x3.z + w[15] * x3.w;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        if (i < t.rows)
        {
            float v = acc[i];
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                v += __shfl_down_sync(0xffffffff, v, offset);
            }
            if (lane == 0)
            {
                t.c[(size_t)i * t.n + row] = v * t.global;
            }
        }
    }
}

// out[r] = sum of rowWeight[s] * y[s] over row r's slots, in job order:
// the same additions, in the same order, as scatter_add_rows job by job.
__global__ void combine_slots(const float* __restrict__ y, const int* __restrict__ rowSlotStart,
    const int* __restrict__ rowSlots, const float* __restrict__ slotWeights, int m, int width,
    float* __restrict__ out)
{
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long)m * width)
    {
        return;
    }
    int r = (int)(index / width);
    int col = (int)(index % width);
    float acc = 0.f;
    for (int i = rowSlotStart[r]; i < rowSlotStart[r + 1]; i++)
    {
        int slot = rowSlots[i];
        acc += slotWeights[slot] * y[(size_t)slot * width + col];
    }
    out[index] = acc;
}

static void* gGroupTasks[2] = {nullptr, nullptr};  static size_t gGroupTasksBytes[2] = {0, 0};
static void* gGroupSlots = nullptr;  static size_t gGroupSlotsBytes = 0;
static std::vector<Fp4Task> gGroupHostTasks[2];
static std::vector<int> gGroupHostSlots;

// AMQL_GROUPED_FFN=0 (or amql_cuda_set_grouped_ffn(0)) runs routed experts
// one at a time, as before.
static int gGroupedFfn = -1;
static long long gGroupedFfnCalls = 0;

static bool grouped_ffn_enabled()
{
    if (gGroupedFfn < 0)
    {
        const char* v = getenv("AMQL_GROUPED_FFN");
        gGroupedFfn = !(v != nullptr && (strcmp(v, "0") == 0 || strcmp(v, "off") == 0 || strcmp(v, "false") == 0));
    }
    return gGroupedFfn == 1;
}

AMQL_EXPORT void amql_cuda_set_grouped_ffn(int on)
{
    gGroupedFfn = on ? 1 : 0;
}

AMQL_EXPORT int amql_cuda_get_grouped_ffn()
{
    return grouped_ffn_enabled() ? 1 : 0;
}

/// Routed layers run as grouped launches since the library loaded.
AMQL_EXPORT long long amql_cuda_grouped_ffn_calls()
{
    return gGroupedFfnCalls;
}

// One launch of every task in tasks, n rows at most. The descriptors go up
// from pageable memory, which the copy has taken before it returns; each of
// a layer's two launches has its own buffer so neither overwrites the other's.
static int run_fp4_tasks(int which, const std::vector<Fp4Task>& tasks, int maxN, cudaStream_t stream)
{
    if (tasks.empty())
    {
        return 0;
    }
    size_t bytes = tasks.size() * sizeof(Fp4Task);
    if (scratch_reserve(&gGroupTasks[which], &gGroupTasksBytes[which], bytes) != 0 ||
        cudaMemcpyAsync(gGroupTasks[which], tasks.data(), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return -1;
    }
    const int threads = 256;
    dim3 grid((unsigned)(((long)maxN * 32 + threads - 1) / threads), (unsigned)tasks.size());
    nvfp4_grouped_gemv<<<grid, threads, 0, stream>>>((const Fp4Task*)gGroupTasks[which]);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

// The routed layer's jobs as grouped launches, when every product is NVFP4
// with at most kGemvRows rows (decode and small batches); 1 when not this
// path's, 0 when done (out on the device in gFfnOut), negative on error.
static int grouped_nvfp4_ffn(int m, int k, int width, int jobs, const int* rowStart,
    const int* inter, const void* const* weights, const int* formats, const float* globals,
    int totalRows, int maxInter, cudaStream_t stream)
{
    if (!grouped_ffn_enabled() || jobs < 2 || k % 16 != 0)
    {
        return 1;
    }
    for (int j = 0; j < jobs; j++)
    {
        int count = rowStart[j + 1] - rowStart[j];
        if (formats[3 * j] != 1 || formats[3 * j + 1] != 1 || formats[3 * j + 2] != 1 ||
            count > kGemvRows || inter[j] % 16 != 0)
        {
            return 1;
        }
    }
    if (scratch_reserve(&gFfnG, &gFfnGBytes, (size_t)totalRows * maxInter * 4) != 0 ||
        scratch_reserve(&gFfnU, &gFfnUBytes, (size_t)totalRows * maxInter * 4) != 0 ||
        scratch_reserve(&gFfnY, &gFfnYBytes, (size_t)totalRows * width * 4) != 0 ||
        scratch_reserve(&gFfnRows, &gFfnRowsBytes, (size_t)totalRows * k * 4) != 0)
    {
        return -3;
    }
    // Every job's rows, gathered at once: slot s is rows[s].
    long gathered = (long)totalRows * k;
    gather_rows<<<(int)((gathered + 255) / 256), 256, 0, stream>>>(
        (const float*)gFfnX, (const int*)gFfnIdx, totalRows, k, (float*)gFfnRows);

    // Gate and up, every job, one launch: slot s's hidden units at s * maxInter.
    float* g = (float*)gFfnG;
    float* u = (float*)gFfnU;
    const float* xr = (const float*)gFfnRows;
    auto& tasks = gGroupHostTasks[0];
    tasks.clear();
    for (int j = 0; j < jobs; j++)
    {
        int first = rowStart[j], count = rowStart[j + 1] - first, n = inter[j];
        for (int done = 0; done < count; done += 8)
        {
            int rows = std::min(8, count - done);
            for (int p = 0; p < 2; p++)
            {
                const unsigned char* packed = (const unsigned char*)weights[3 * j + p];
                Fp4Task t;
                t.packed = packed;
                t.scales = packed + nvfp4_scale_offset(n, k);
                t.a = xr + (size_t)(first + done) * k;
                // A job's hidden rows are n apart, as the one-job path lays them out.
                t.c = (p == 0 ? g : u) + (size_t)first * maxInter + (size_t)done * n;
                t.global = globals[3 * j + p];
                t.rows = rows;
                t.n = n;
                t.k = k;
                tasks.push_back(t);
            }
        }
    }
    int code;
    if ((code = run_fp4_tasks(0, tasks, maxInter, stream)) != 0)
    {
        return -10 + code;
    }
    // SiLU(gate) * up over every job's hidden units at once.
    long hidden = (long)totalRows * maxInter;
    silu_times<<<(int)((hidden + 255) / 256), 256, 0, stream>>>(g, u, hidden);

    // Down, every job, one launch: slot s's output at s * width.
    float* y = (float*)gFfnY;
    auto& downs = gGroupHostTasks[1];
    downs.clear();
    for (int j = 0; j < jobs; j++)
    {
        int first = rowStart[j], count = rowStart[j + 1] - first, n = inter[j];
        const unsigned char* packed = (const unsigned char*)weights[3 * j + 2];
        for (int done = 0; done < count; done += 8)
        {
            Fp4Task t;
            t.packed = packed;
            t.scales = packed + nvfp4_scale_offset(width, n);
            t.a = g + (size_t)first * maxInter + (size_t)done * n;
            t.c = y + (size_t)(first + done) * width;
            t.global = globals[3 * j + 2];
            t.rows = std::min(8, count - done);
            t.n = width;
            t.k = n;
            downs.push_back(t);
        }
    }
    if ((code = run_fp4_tasks(1, downs, width, stream)) != 0)
    {
        return -20 + code;
    }
    gGroupedFfnCalls++;
    return 0;
}

/// Runs gated SiLU FFN jobs over rows of x, all on the device, and writes
/// the weighted sum of their outputs to out ([m, width], overwritten).
/// Job j covers rows[rowStart[j] .. rowStart[j + 1]) with rowWeights, and
/// has inter[j] hidden units; its gate, up and down weights are
/// weights[3j .. 3j + 2] with formats[...] and globals[...].
AMQL_EXPORT int amql_cuda_ffn(
    const float* x, int m, int k, int width,
    int jobs, const int* rowStart, const int* rows, const float* rowWeights,
    const int* inter, const void* const* weights, const int* formats, const float* globals,
    float* out)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || jobs <= 0)
    {
        return -2;
    }
    cudaStream_t stream = gCtx.stream;
    int totalRows = rowStart[jobs];
    int maxRows = 0, maxInter = 0;
    for (int j = 0; j < jobs; j++)
    {
        maxRows = std::max(maxRows, rowStart[j + 1] - rowStart[j]);
        maxInter = std::max(maxInter, inter[j]);
    }
    if (scratch_reserve(&gFfnX, &gFfnXBytes, (size_t)m * k * 4) != 0 ||
        scratch_reserve(&gFfnRows, &gFfnRowsBytes, (size_t)maxRows * k * 4) != 0 ||
        scratch_reserve(&gFfnG, &gFfnGBytes, (size_t)maxRows * maxInter * 4) != 0 ||
        scratch_reserve(&gFfnU, &gFfnUBytes, (size_t)maxRows * maxInter * 4) != 0 ||
        scratch_reserve(&gFfnY, &gFfnYBytes, (size_t)maxRows * width * 4) != 0 ||
        scratch_reserve(&gFfnOut, &gFfnOutBytes, (size_t)m * width * 4) != 0 ||
        scratch_reserve(&gFfnIdx, &gFfnIdxBytes, (size_t)std::max(1, totalRows) * 4) != 0 ||
        scratch_reserve(&gFfnW, &gFfnWBytes, (size_t)std::max(1, totalRows) * 4) != 0)
    {
        return -3;
    }
    if (cudaMemcpyAsync(gFfnX, x, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(gFfnIdx, rows, (size_t)totalRows * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(gFfnW, rowWeights, (size_t)totalRows * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemsetAsync(gFfnOut, 0, (size_t)m * width * 4, stream) != cudaSuccess)
    {
        return -4;
    }
    const int* idx = (const int*)gFfnIdx;
    const float* wts = (const float*)gFfnW;
    int grouped = grouped_nvfp4_ffn(m, k, width, jobs, rowStart, inter, weights, formats, globals,
        totalRows, maxInter, stream);
    if (grouped < 0)
    {
        return -300 + grouped;
    }
    if (grouped == 0)
    {
        // Each row's slots in job order, so the sum adds as the per-job scatter does.
        auto& slots = gGroupHostSlots;
        slots.assign((size_t)m + 1 + totalRows, 0);
        int* start = slots.data();
        int* list = start + m + 1;
        for (int s = 0; s < totalRows; s++)
        {
            start[rows[s] + 1]++;
        }
        for (int r = 0; r < m; r++)
        {
            start[r + 1] += start[r];
        }
        std::vector<int> fill(start, start + m);
        for (int s = 0; s < totalRows; s++)
        {
            list[fill[rows[s]]++] = s;   // slots run in job order
        }
        size_t bytes = slots.size() * 4;
        if (scratch_reserve(&gGroupSlots, &gGroupSlotsBytes, bytes) != 0 ||
            cudaMemcpyAsync(gGroupSlots, slots.data(), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            return -8;
        }
        long outCount = (long)m * width;
        combine_slots<<<(int)((outCount + 255) / 256), 256, 0, stream>>>(
            (const float*)gFfnY, (const int*)gGroupSlots, (const int*)gGroupSlots + m + 1, wts, m, width,
            (float*)gFfnOut);
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpyAsync(out, gFfnOut, (size_t)m * width * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            return -9;
        }
        return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -7;
    }
    for (int j = 0; j < jobs; j++)
    {
        int first = rowStart[j];
        int count = rowStart[j + 1] - first;
        if (count == 0)
        {
            continue;
        }
        int hiddenUnits = inter[j];
        long gathered = (long)count * k;
        gather_rows<<<(int)((gathered + 255) / 256), 256, 0, stream>>>(
            (const float*)gFfnX, idx + first, count, k, (float*)gFfnRows);
        int code;
        if ((code = device_product(formats[3 * j], weights[3 * j], globals[3 * j],
                 (const float*)gFfnRows, (float*)gFfnG, count, k, hiddenUnits, stream)) != 0 ||
            (code = device_product(formats[3 * j + 1], weights[3 * j + 1], globals[3 * j + 1],
                 (const float*)gFfnRows, (float*)gFfnU, count, k, hiddenUnits, stream)) != 0)
        {
            return -100 + code;
        }
        long hiddenCount = (long)count * hiddenUnits;
        silu_times<<<(int)((hiddenCount + 255) / 256), 256, 0, stream>>>(
            (float*)gFfnG, (const float*)gFfnU, hiddenCount);
        if ((code = device_product(formats[3 * j + 2], weights[3 * j + 2], globals[3 * j + 2],
                 (const float*)gFfnG, (float*)gFfnY, count, hiddenUnits, width, stream)) != 0)
        {
            return -200 + code;
        }
        long outCount = (long)count * width;
        scatter_add_rows<<<(int)((outCount + 255) / 256), 256, 0, stream>>>(
            (const float*)gFfnY, idx + first, wts + first, count, width, (float*)gFfnOut);
        if (cudaGetLastError() != cudaSuccess)
        {
            return -5;
        }
    }
    if (cudaMemcpyAsync(out, gFfnOut, (size_t)m * width * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return -6;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -7;
}

/// Several products sharing one input, C_i[m, n_i] = A[m, k] W_i[n_i, k]^T:
/// attention's q/k/v, the linear attention's input projections. A goes up
/// once and every product comes down in one copy, into out as C_0 then C_1
/// and so on, each row-major.
static void* gMultiOut = nullptr; static size_t gMultiOutBytes = 0;

AMQL_EXPORT int amql_cuda_multi(
    const float* a, int m, int k, int count,
    const void* const* weights, const int* formats, const float* globals, const int* ns,
    float* out)
{
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    if (m <= 0 || count <= 0)
    {
        return -2;
    }
    cudaStream_t stream = gCtx.stream;
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += (size_t)m * ns[i];
    }
    if (scratch_reserve(&gFfnX, &gFfnXBytes, (size_t)m * k * 4) != 0 ||
        scratch_reserve(&gMultiOut, &gMultiOutBytes, total * 4) != 0)
    {
        return -3;
    }
    if (cudaMemcpyAsync(gFfnX, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return -4;
    }
    // All NVFP4 with few rows (decode): every product in one grouped launch.
    bool grouped = grouped_ffn_enabled() && count > 1 && m <= kGemvRows && k % 16 == 0;
    for (int i = 0; grouped && i < count; i++)
    {
        grouped = formats[i] == 1;
    }
    if (grouped)
    {
        auto& tasks = gGroupHostTasks[0];
        tasks.clear();
        size_t at = 0;
        int maxN = 0;
        for (int i = 0; i < count; i++)
        {
            const unsigned char* packed = (const unsigned char*)weights[i];
            for (int done = 0; done < m; done += 8)
            {
                Fp4Task t;
                t.packed = packed;
                t.scales = packed + nvfp4_scale_offset(ns[i], k);
                t.a = (const float*)gFfnX + (size_t)done * k;
                t.c = (float*)gMultiOut + at + (size_t)done * ns[i];
                t.global = globals[i];
                t.rows = std::min(8, m - done);
                t.n = ns[i];
                t.k = k;
                tasks.push_back(t);
            }
            at += (size_t)m * ns[i];
            maxN = std::max(maxN, ns[i]);
        }
        if (run_fp4_tasks(0, tasks, maxN, stream) != 0)
        {
            return -7;
        }
        gGroupedFfnCalls++;
        if (cudaMemcpyAsync(out, gMultiOut, total * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            return -5;
        }
        return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -6;
    }
    size_t offset = 0;
    for (int i = 0; i < count; i++)
    {
        int code = device_product(formats[i], weights[i], globals[i],
            (const float*)gFfnX, (float*)gMultiOut + offset, m, k, ns[i], stream);
        if (code != 0)
        {
            return -100 + code;
        }
        offset += (size_t)m * ns[i];
    }
    if (cudaMemcpyAsync(out, gMultiOut, total * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        return -5;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : -6;
}

// ── Device-only timing of the NVFP4 prefill paths (benchmarks) ────────────

__global__ void fill_pattern(float* x, long count)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        unsigned int h = (unsigned int)i * 2654435761u;
        x[i] = ((float)(h >> 8) / 16777216.f) * 2.f - 1.f;
    }
}

/// Times one NVFP4 product of m rows against an uploaded [n,k] weight with
/// the activations already on the device: tensor cores (W4A4, including the
/// activation quantisation) and FP16 tiles (including the tile decode).
/// Milliseconds per product into msCores / msTiles; -1 where a path is unavailable.
AMQL_EXPORT int amql_cuda_nvfp4_bench(const void* w, int m, int k, int n, float global, int reps,
    float* msCores, float* msTiles)
{
    *msCores = *msTiles = -1.f;
    if (ctx_ensure() != 0)
    {
        return -1;
    }
    cudaStream_t stream = gCtx.stream;
    const unsigned char* packed = (const unsigned char*)w;
    const unsigned char* scales = packed + nvfp4_scale_offset(n, k);
    float *a = nullptr, *c = nullptr;
    if (cudaMalloc(&a, (size_t)m * k * 4) != cudaSuccess || cudaMalloc(&c, (size_t)m * n * 4) != cudaSuccess)
    {
        return -2;
    }
    long count = (long)m * k;
    fill_pattern<<<(int)((count + 255) / 256), 256, 0, stream>>>(a, count);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (nvfp4_tensor_core_gemm(packed, scales, a, c, m, k, n, global, stream) == 0)
    {
        cudaEventRecord(start, stream);
        for (int r = 0; r < reps; r++)
        {
            nvfp4_tensor_core_gemm(packed, scales, a, c, m, k, n, global, stream);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(msCores, start, stop);
        *msCores /= reps;
    }
    cudaGetLastError();
    // FP16 tiles: cast the activations, decode tiles, GEMM (as the fallback does).
    if (scratch_reserve(&gCtx.scratchAF16, &gCtx.scratchAF16Bytes, (size_t)m * k * 2) == 0)
    {
        int tileRows = (int)std::max<long>(1, std::min<long>(n, (32L << 20) / k));
        if (scratch_reserve(&gNvfp4Tile, &gNvfp4TileBytes, (size_t)tileRows * k * 2) == 0)
        {
            float beta = 0.f;
            cudaEventRecord(start, stream);
            for (int r = 0; r < reps; r++)
            {
                cast_f32_to_f16<<<(int)((count + 255) / 256), 256, 0, stream>>>(a, (__half*)gCtx.scratchAF16, count);
                for (int first = 0; first < n; first += tileRows)
                {
                    int rows = std::min(tileRows, n - first);
                    long tile = (long)rows * k;
                    dequant_nvfp4_to_f16<<<(int)((tile + 255) / 256), 256, 0, stream>>>(
                        packed, scales, (__half*)gNvfp4Tile, first, rows, k);
                    cublasGemmEx(gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, m, k, &global,
                        gNvfp4Tile, CUDA_R_16F, k, gCtx.scratchAF16, CUDA_R_16F, k, &beta,
                        c + first, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                }
            }
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(msTiles, start, stop);
            *msTiles /= reps;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a);
    cudaFree(c);
    return 0;
}
