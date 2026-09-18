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
// cuBLASLt row-major layout + opB=transpose implements this directly.

struct GpuContext
{
    cublasLtHandle_t lt;
    cudaStream_t stream;
    void* workspace;
    size_t workspaceBytes;
    void* scratchA;      // reusable host→device staging for A (grows)
    size_t scratchABytes;
    void* scratchC;      // reusable device→host staging for C (grows)
    size_t scratchCBytes;
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
    if (cudaStreamCreateWithFlags(&gCtx.stream, cudaStreamNonBlocking) != cudaSuccess)
    {
        return -2;
    }
    gCtx.workspaceBytes = 64u << 20;   // cuBLASLt scratch
    if (cudaMalloc(&gCtx.workspace, gCtx.workspaceBytes) != cudaSuccess)
    {
        return -3;
    }
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

/// <summary>Finds or builds a cached plan for the (m,k,n) transposed-B
/// GEMM. Returns 0 and the plan, or a negative error.</summary>
static int plan_get(int m, int k, int n, LtPlan** outPlan)
{
    int64_t key = ((int64_t)m << 40) ^ ((int64_t)k << 20) ^ (int64_t)n;
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

    const cublasOperation_t opA = CUBLAS_OP_N;
    const cublasOperation_t opB = CUBLAS_OP_T;
    cublasStatus_t status = cublasLtMatmulDescCreate(&gPlans[gPlanCount].desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return -6;
    }
    cublasLtMatmulDescSetAttribute(gPlans[gPlanCount].desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(gPlans[gPlanCount].desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    cublasLtMatrixLayoutCreate(&gPlans[gPlanCount].a, CUDA_R_32F, m, k, k);
    cublasLtMatrixLayoutCreate(&gPlans[gPlanCount].b, CUDA_R_16F, n, k, k);
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
    cudaStreamDestroy(gCtx.stream);
    cublasLtDestroy(gCtx.lt);
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
    dequant_mxfp4_to_f16<<<blocks, threads, 0, stream>>>(packed, scales, out, rows, cols);
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

    // Row-major: C[m,n] = A[m,k] @ W[n,k]ᵀ. FP32 accumulate over FP16
    // weights; the cuBLASLt plan (desc + layouts + heuristic algo) is
    // cached per (m,k,n) so repeated decode GEMMs skip the setup.
    float alpha = 1.0f, beta = 0.0f;
    LtPlan* plan = nullptr;
    int planStatus = plan_get(m, k, n, &plan);
    if (planStatus != 0 || plan == nullptr)
    {
        return planStatus != 0 ? planStatus : -8;
    }

    const cublasLtMatmulAlgo_t* algoPtr = nullptr;
    if (plan->valid)
    {
        algoPtr = &plan->algo;
    }

    cublasStatus_t status = cublasLtMatmul(
        gCtx.lt, plan->desc, &alpha, aD, plan->a, w, plan->b, &beta,
        cD, plan->c, cD, plan->c, algoPtr, gCtx.workspace, gCtx.workspaceBytes, stream);

    cudaMemcpyAsync(c, cD, (size_t)m * n * 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return status == CUBLAS_STATUS_SUCCESS ? 0 : -5;
}