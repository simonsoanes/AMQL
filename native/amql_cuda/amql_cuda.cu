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
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstring>

#define AMQL_EXPORT extern "C" __declspec(dllexport)

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
    cublasLtHandle_t lt;
    cublasHandle_t cublas;   // classic cublas — the merge GEMMs use this
    cudaStream_t stream;
    void* workspace;
    size_t workspaceBytes;
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

// cuBLASLt plan cache: decode hits the same (m,k,n) every token, and the
// descriptor/layout/heuristic setup dwarfs a small GEMM — cache it once.
struct LtPlan
{
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algo;
    bool valid;
};

static LtPlan gPlans[16];
static int64_t gPlanKeys[16];
static int gPlanCount = 0;

static int ctx_ensure()
{
    if (gCtxValid)
    {
        return 0;
    }
    if (cublasLtCreate(&gCtx.lt) != CUBLAS_STATUS_SUCCESS)
    {
        return -1;
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
    gCtx.workspaceBytes = 64u << 20;   // cuBLASLt scratch
    if (cudaMalloc(&gCtx.workspace, gCtx.workspaceBytes) != cudaSuccess)
    {
        return -3;
    }
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

/// <summary>Row-major layout dims (stored rows/cols/ld) for the A/B/C
/// tensors of an opA/opB combo. C is always [m, n] with opC=N.</summary>
static void row_layout_dims(
    int m, int k, int n, cublasOperation_t opA, cublasOperation_t opB,
    int* aRows, int* aCols, int* bRows, int* bCols)
{
    if (opA == CUBLAS_OP_T)
    {
        *aRows = k; *aCols = m;      // stored [k, m], reused as [m, k]ᵀ
    }
    else
    {
        *aRows = m; *aCols = k;
    }
    if (opB == CUBLAS_OP_T)
    {
        *bRows = n; *bCols = k;      // stored [n, k], reused as [k, n]ᵀ
    }
    else
    {
        *bRows = k; *bCols = n;
    }
}

/// <summary>Finds or builds a cached plan for the (m,k,n) GEMM with
/// explicit opA/opB and B dtype. Returns 0 and the plan, or a negative
/// error.</summary>
static int plan_get_ext(int m, int k, int n, cublasOperation_t opA, cublasOperation_t opB,
    cudaDataType_t bDtype, LtPlan** outPlan)
{
    int64_t key = ((int64_t)m << 40) ^ ((int64_t)k << 20) ^ (int64_t)n
        ^ ((int64_t)opA << 4) ^ ((int64_t)opB << 2) ^ (int64_t)bDtype;
    for (int i = 0; i < gPlanCount; i++)
    {
        if (gPlanKeys[i] == key)
        {
            *outPlan = &gPlans[i];
            return 0;
        }
    }
    if (gPlanCount >= 16)
    {
        return -7; // plan cache full — fall back without caching
    }

    cublasStatus_t status = cublasLtMatmulDescCreate(&gPlans[gPlanCount].desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return -6;
    }
    cublasLtMatmulDescSetAttribute(gPlans[gPlanCount].desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(gPlans[gPlanCount].desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    int aRows, aCols, bRows, bCols;
    row_layout_dims(m, k, n, opA, opB, &aRows, &aCols, &bRows, &bCols);
    cublasLtMatrixLayoutCreate(&gPlans[gPlanCount].a, CUDA_R_32F, aRows, aCols, aCols);
    cublasLtMatrixLayoutCreate(&gPlans[gPlanCount].b, bDtype, bRows, bCols, bCols);
    cublasLtMatrixLayoutCreate(&gPlans[gPlanCount].c, CUDA_R_32F, m, n, n);
    int32_t rowMajor = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(gPlans[gPlanCount].a, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor));
    cublasLtMatrixLayoutSetAttribute(gPlans[gPlanCount].b, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor));
    cublasLtMatrixLayoutSetAttribute(gPlans[gPlanCount].c, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor));

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &gCtx.workspaceBytes, sizeof(gCtx.workspaceBytes));

    gPlans[gPlanCount].algo = cublasLtMatmulAlgo_t{};
    gPlans[gPlanCount].valid = false;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        gCtx.lt, gPlans[gPlanCount].desc,
        gPlans[gPlanCount].a, gPlans[gPlanCount].b,
        gPlans[gPlanCount].c, gPlans[gPlanCount].c,
        pref, 1, &heuristic, &returned);
    if (status == CUBLAS_STATUS_SUCCESS && returned > 0)
    {
        gPlans[gPlanCount].algo = heuristic.algo;
        gPlans[gPlanCount].valid = true;
    }
    if (pref)
    {
        cublasLtMatmulPreferenceDestroy(pref);
    }

    gPlanKeys[gPlanCount] = key;
    *outPlan = &gPlans[gPlanCount];
    gPlanCount++;
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

AMQL_EXPORT void amql_cuda_shutdown(void)
{
    if (!gCtxValid)
    {
        return;
    }
    if (gCtx.workspace) cudaFree(gCtx.workspace);
    if (gCtx.scratchA) cudaFree(gCtx.scratchA);
    if (gCtx.scratchC) cudaFree(gCtx.scratchC);
    if (gCtx.scratchPack) cudaFree(gCtx.scratchPack);
    if (gCtx.scratchScale) cudaFree(gCtx.scratchScale);
    if (gCtx.scratchAF16) cudaFree(gCtx.scratchAF16);
    cudaStreamDestroy(gCtx.stream);
    cublasLtDestroy(gCtx.lt);
    cublasDestroy(gCtx.cublas);
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
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_host_to_device_f32(void* dst, const float* src, size_t bytes)
{
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

AMQL_EXPORT int amql_cuda_device_to_host(void* dst, const void* src, size_t bytes)
{
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
    size_t packedBytes = (size_t)((total + 1) / 2);
    size_t scaleBytes = (size_t)rows * ((cols + 31) / 32);
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
    dequant_mxfp4_to_f16<<<blocks, threads, 0, stream>>>(
        (const unsigned char*)gCtx.scratchPack, (const unsigned char*)gCtx.scratchScale, out, rows, cols);
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

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)m * k * 4) != 0)
    {
        return -2;
    }
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
    {
        return -3;
    }
    void* aD = gCtx.scratchA;
    void* cD = gCtx.scratchC;
    if (cudaMemcpyAsync(aD, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return -4;
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
        (const float*)aD, (__half*)aF16, actCount);
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
    cublasStatus_t status = cublasGemmEx(
        gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const void*)w, CUDA_R_16F, k,
        (const void*)aF16, CUDA_R_16F, k,
        &beta,
        (void*)cD, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpyAsync(c, cD, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream);
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

    if (scratch_reserve(&gCtx.scratchA, &gCtx.scratchABytes, (size_t)m * k * 4) != 0)
    {
        return -2;
    }
    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
    {
        return -3;
    }
    void* aD = gCtx.scratchA;
    void* cD = gCtx.scratchC;
    if (cudaMemcpyAsync(aD, a, (size_t)m * k * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        return -4;
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
        (const float*)aD, (__half*)aF16, actCount);
    if (cudaGetLastError() != cudaSuccess)
    {
        return -6;
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const void*)w, CUDA_R_16F, k,
        (const void*)aF16, CUDA_R_16F, k,
        &beta,
        (void*)cD, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpyAsync(c, cD, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream);
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

    // Staging: copy host→device FP32, then cast to FP16 on device.
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
    long count = (long)m * k;
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    cast_f32_to_f16<<<blocks, threads, 0, stream>>>(
        (const float*)gCtx.scratchA, *out, count);
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

    if (scratch_reserve(&gCtx.scratchC, &gCtx.scratchCBytes, (size_t)m * n * 4) != 0)
    {
        return -2;
    }
    void* cD = gCtx.scratchC;

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        gCtx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const void*)w, CUDA_R_16F, k,
        (const void*)a_dev, CUDA_R_16F, k,
        &beta,
        (void*)cD, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpyAsync(c, cD, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream);
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