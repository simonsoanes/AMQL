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
#include <cusolverDn.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

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