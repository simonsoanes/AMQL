// amql_engine.cu — the device-resident decode engine.
//
// Where amql_cuda.cu offloads single operators from a host-driven loop (a
// round trip per GEMM), this file holds the pieces of a whole forward pass
// that stays on the device: weights uploaded once as BF16 (or f32 when the
// container or a patch makes them so), activations, KV cache and recurrent
// state all resident, and one decode step captured into a CUDA graph so a
// token costs one launch and two small copies.
//
// The managed side (Amql.Inference.GpuEngine) walks the op plan and issues
// these entry points in order; every kernel takes raw device pointers so
// the plan, not this file, decides the model's shape. Positions are read
// from a device counter (*pos), never baked into a launch, which is what
// lets the same captured graph serve every decode step.
//
// Numerics: f32 activations and accumulation throughout. Weights are the
// container's own values (BF16 widened exactly in registers). Results agree
// with the managed path to rounding (reduction order differs), not bit for
// bit.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <utility>

#if defined(_WIN32)
  #define AMQE_EXPORT extern "C" __declspec(dllexport)
#else
  #define AMQE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ── engine context ─────────────────────────────────────────────────────────

struct Engine
{
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    float* widen = nullptr;     // f32 scratch a BF16 weight widens into for a GEMM
    size_t widenElems = 0;
    __nv_bfloat16* xb = nullptr; // BF16 copy of a GEMM's activations
    size_t xbElems = 0;
    int prefillF32 = 0;          // 1: batched rows use FP32 SGEMM over widened weights
    float* scores = nullptr;     // prompt attention's [group x rows x keys] scores
    size_t scoresElems = 0;
    float* chunk = nullptr;      // chunked Gated DeltaNet's W, U, G and P
    size_t chunkElems = 0;
};

static int ok(cudaError_t e) { return e == cudaSuccess ? 0 : -(int)e; }

AMQE_EXPORT int amqe_create(Engine** out)
{
    auto* e = new Engine();
    if (cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking) != cudaSuccess) { delete e; return -1; }
    if (cublasCreate(&e->blas) != CUBLAS_STATUS_SUCCESS) { cudaStreamDestroy(e->stream); delete e; return -2; }
    cublasSetStream(e->blas, e->stream);
    // Plain FP32 SGEMM: no TF32 down-conversion.
    cublasSetMathMode(e->blas, CUBLAS_DEFAULT_MATH);
    *out = e;
    return 0;
}

AMQE_EXPORT void amqe_destroy(Engine* e)
{
    if (!e) return;
    cudaStreamSynchronize(e->stream);
    if (e->exec) cudaGraphExecDestroy(e->exec);
    if (e->graph) cudaGraphDestroy(e->graph);
    if (e->widen) cudaFree(e->widen);
    if (e->xb) cudaFree(e->xb);
    if (e->scores) cudaFree(e->scores);
    if (e->chunk) cudaFree(e->chunk);
    if (e->blas) cublasDestroy(e->blas);
    if (e->stream) cudaStreamDestroy(e->stream);
    delete e;
}

AMQE_EXPORT int amqe_alloc(void** ptr, size_t bytes)
{
    *ptr = nullptr;
    return ok(cudaMalloc(ptr, bytes == 0 ? 16 : bytes));
}

AMQE_EXPORT int amqe_free(void* ptr) { return ptr ? ok(cudaFree(ptr)) : 0; }

AMQE_EXPORT int amqe_host_alloc(void** ptr, size_t bytes) { return ok(cudaMallocHost(ptr, bytes)); }
AMQE_EXPORT int amqe_host_free(void* ptr) { return ptr ? ok(cudaFreeHost(ptr)) : 0; }

AMQE_EXPORT int amqe_sm_count(void)
{
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms;
}

AMQE_EXPORT int amqe_mem_info(size_t* freeBytes, size_t* totalBytes) { return ok(cudaMemGetInfo(freeBytes, totalBytes)); }

AMQE_EXPORT int amqe_upload(Engine* e, void* dst, const void* src, size_t bytes)
{
    int r = ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, e->stream));
    return r ? r : ok(cudaStreamSynchronize(e->stream));
}

AMQE_EXPORT int amqe_download(Engine* e, void* dst, const void* src, size_t bytes)
{
    int r = ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, e->stream));
    return r ? r : ok(cudaStreamSynchronize(e->stream));
}

/// Stream-ordered copies (captured into the graph when issued inside one).
AMQE_EXPORT int amqe_copy_async(Engine* e, void* dst, const void* src, size_t bytes, int kind)
{
    return ok(cudaMemcpyAsync(dst, src, bytes, (cudaMemcpyKind)kind, e->stream));
}

AMQE_EXPORT int amqe_memset(Engine* e, void* dst, int value, size_t bytes)
{
    return ok(cudaMemsetAsync(dst, value, bytes, e->stream));
}

AMQE_EXPORT int amqe_sync(Engine* e) { return ok(cudaStreamSynchronize(e->stream)); }

AMQE_EXPORT int amqe_last_error(void) { return (int)cudaGetLastError(); }

// ── programmatic dependent launch ─────────────────────────────────────────
//
// Every kernel here is launched with programmatic stream serialisation: the
// next kernel in the stream may start as soon as every block of this one
// has started, and waits (griddepcontrol.wait) for this one to finish before
// it touches anything this one writes. What a kernel may do before its wait
// is read data no kernel writes — the weights — so a projection streams its
// weights into L2 while the norm, attention or recurrence before it runs,
// and DRAM stays busy across kernel boundaries. A kernel's wait returns
// only once its predecessor has completed, and that one waited on its own,
// so the ordering is the stream's, transitively.

static bool pdl_enabled()
{
    static int on = -1;
    if (on < 0)
    {
        const char* v = getenv("AMQE_PDL");
        on = (v && v[0] == '0') ? 0 : 1;
    }
    return on == 1;
}

template <typename... KArgs, typename... Args>
static void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args&&... args)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl_enabled() ? 1 : 0;
    cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

// Let the next kernel launch, then wait for the previous one's results.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_enter() { pdl_trigger(); pdl_wait(); }

// ── graph capture ──────────────────────────────────────────────────────────

AMQE_EXPORT int amqe_capture_begin(Engine* e)
{
    if (e->exec) { cudaGraphExecDestroy(e->exec); e->exec = nullptr; }
    if (e->graph) { cudaGraphDestroy(e->graph); e->graph = nullptr; }
    return ok(cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal));
}

AMQE_EXPORT int amqe_capture_end(Engine* e)
{
    int r = ok(cudaStreamEndCapture(e->stream, &e->graph));
    if (r) return r;
    return ok(cudaGraphInstantiate(&e->exec, e->graph, 0));
}

AMQE_EXPORT int amqe_capture_drop(Engine* e)
{
    if (e->exec) { cudaGraphExecDestroy(e->exec); e->exec = nullptr; }
    if (e->graph) { cudaGraphDestroy(e->graph); e->graph = nullptr; }
    return 0;
}

AMQE_EXPORT int amqe_graph_launch(Engine* e)
{
    if (!e->exec) return -1;
    int r = ok(cudaGraphLaunch(e->exec, e->stream));
    return r ? r : ok(cudaStreamSynchronize(e->stream));
}

/// Launch only — the caller synchronises (amqe_sync) when it needs the result.
AMQE_EXPORT int amqe_graph_launch_async(Engine* e)
{
    if (!e->exec) return -1;
    return ok(cudaGraphLaunch(e->exec, e->stream));
}

// ── helpers ────────────────────────────────────────────────────────────────

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum; every thread gets the total. `shared` holds ≥ 32 floats.
__device__ __forceinline__ float block_sum(float v, float* shared)
{
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float t = 0.f;
    if (threadIdx.x < 32)
    {
        t = threadIdx.x < warps ? shared[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) shared[0] = t;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float gelu_tanh(float x)
{
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Activation codes shared with the managed side.
enum { ACT_NONE = 0, ACT_SILU = 1, ACT_GELU = 2, ACT_RELU = 3 };

__device__ __forceinline__ float activate(float x, int act)
{
    switch (act)
    {
        case ACT_SILU: return silu(x);
        case ACT_GELU: return gelu_tanh(x);
        case ACT_RELU: return x > 0.f ? x : 0.f;
        default: return x;
    }
}

// Eight weights of row `row` starting at column c (c % 8 == 0, row 16-byte aligned).
__device__ __forceinline__ void load8(const __nv_bfloat16* w, long long idx, float* o)
{
    uint4 v = __ldg(reinterpret_cast<const uint4*>(w + idx));
    o[0] = bf16_lo(v.x); o[1] = bf16_hi(v.x);
    o[2] = bf16_lo(v.y); o[3] = bf16_hi(v.y);
    o[4] = bf16_lo(v.z); o[5] = bf16_hi(v.z);
    o[6] = bf16_lo(v.w); o[7] = bf16_hi(v.w);
}
__device__ __forceinline__ void load8(const float* w, long long idx, float* o)
{
    float4 a = __ldg(reinterpret_cast<const float4*>(w + idx));
    float4 b = __ldg(reinterpret_cast<const float4*>(w + idx + 4));
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
    o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ float load1(const __nv_bfloat16* w, long long idx) { return __bfloat162float(w[idx]); }
__device__ __forceinline__ float load1(const float* w, long long idx) { return w[idx]; }

// ── GEMV: y[r, j] = epilogue(Σ_c W[j, c] · x[r, c]) for R ≤ 8 rows ─────────
//
// One warp per output row j (grid-stride, so a wide head reuses its blocks);
// each lane streams 16-byte runs of the weight row and dots them against
// every activation row. Optionally the rows are normalised first (the norm
// that precedes this projection, fused: each block normalises x into shared
// memory, which costs a re-read of x per block instead of a launch). The
// epilogue adds a bias, applies an activation, and adds into a residual (in
// place when y == residual — each element is read once, then written).

struct GemvArgs
{
    const void* w; int n; int k;
    const float* x; int ldx;
    float* y; int ldy;
    const float* bias;
    const float* residual; int ldr; float resScale;
    int act;
    int glu;          // 1: rows j and n + j are gate and up; y = act(g)·u (n = intermediate)
    const float* normW; float normOff; const float* normBias; float normEps; int normKind;
    int prefetch;     // pull the first rows into L2 before waiting on the previous kernel
};

// SPLIT warps share each output row (each takes every SPLIT-th 256-column
// run), so a narrow matrix — 1024 outputs over 170 SMs is six warps each —
// still keeps enough loads in flight to saturate memory.
template <typename WT, int R, int SPLIT, int RW>
__global__ void __launch_bounds__(256) gemv_kernel(GemvArgs a)
{
    constexpr int WARPS = 8;
    constexpr int GROUP = WARPS / SPLIT * RW;   // output rows per block iteration
    extern __shared__ float xs[];          // [R × k] when the norm is fused
    __shared__ float red[32];
    __shared__ float part[SPLIT > 1 ? WARPS : 1][RW][2][R];
    const WT* W = static_cast<const WT*>(a.w);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int sub = warp % SPLIT, slot = warp / SPLIT;
    int k = a.k;
    pdl_trigger();
    if (a.prefetch)
    {
        // The weights are never written: pull this block's first rows into L2
        // while the kernel before is still finishing.
        int rowsW0 = a.glu ? 2 : 1;
        long long rowBytes = (long long)k * sizeof(WT);
        int jb0 = blockIdx.x * GROUP + slot * RW;
        for (int q = 0; q < RW; q++)
        {
            int j = jb0 + q;
            if (j >= a.n) break;
            for (int s0 = 0; s0 < rowsW0; s0++)
            {
                const char* row = reinterpret_cast<const char*>(W + (long long)(s0 == 0 ? j : a.n + j) * k);
                for (long long off = (long long)(sub * 32 + lane) * 128; off < rowBytes; off += 128LL * 32 * SPLIT)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(row + off));
            }
        }
    }
    pdl_wait();
    const float* xb = a.x;
    int ldx = a.ldx;
    if (a.normW)
    {
        for (int r = 0; r < R; r++)
        {
            const float* xr = a.x + (long long)r * a.ldx;
            float mean = 0.f;
            if (a.normKind == 1)
            {
                float t = 0.f;
                for (int c = threadIdx.x; c < k; c += blockDim.x) t += xr[c];
                mean = block_sum(t, red) / k;
            }
            float ss = 0.f;
            for (int c = threadIdx.x; c < k; c += blockDim.x) { float d = xr[c] - mean; ss += d * d; }
            float inv = rsqrtf(block_sum(ss, red) / k + a.normEps);
            for (int c = threadIdx.x; c < k; c += blockDim.x)
            {
                float v = (xr[c] - mean) * inv * (a.normW[c] + a.normOff);
                if (a.normBias) v += a.normBias[c];
                xs[r * k + c] = v;
            }
        }
        __syncthreads();
        xb = xs;
        ldx = k;
    }

    const int rowsW = a.glu ? 2 : 1;
    bool vec = (k & 7) == 0 && (ldx & 3) == 0;
    int kv = vec ? k : 0;
    int groups = (a.n + GROUP - 1) / GROUP;
    for (int grp = blockIdx.x; grp < groups; grp += gridDim.x)
    {
        int jb = grp * GROUP + slot * RW;
        float acc[RW][2][R];
        #pragma unroll
        for (int q = 0; q < RW; q++)
            #pragma unroll
            for (int s = 0; s < 2; s++)
                #pragma unroll
                for (int r = 0; r < R; r++) acc[q][s][r] = 0.f;

        for (int s = 0; s < rowsW; s++)
        {
            long long rowBase[RW];
            bool live[RW];
            #pragma unroll
            for (int q = 0; q < RW; q++)
            {
                live[q] = jb + q < a.n;
                rowBase[q] = (long long)(s == 0 ? jb + q : a.n + jb + q) * k;
            }
            #pragma unroll 4
            for (int c = (sub * 32 + lane) * 8; c < kv; c += 256 * SPLIT)
            {
                float wv[RW][8];
                #pragma unroll
                for (int q = 0; q < RW; q++)
                {
                    if (live[q]) load8(W, rowBase[q] + c, wv[q]);
                    else
                        #pragma unroll
                        for (int i = 0; i < 8; i++) wv[q][i] = 0.f;
                }
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    const float4* xp = reinterpret_cast<const float4*>(xb + (long long)r * ldx + c);
                    float4 x0 = xp[0], x1 = xp[1];
                    #pragma unroll
                    for (int q = 0; q < RW; q++)
                    {
                        float t = acc[q][s][r];
                        t = fmaf(wv[q][0], x0.x, t); t = fmaf(wv[q][1], x0.y, t);
                        t = fmaf(wv[q][2], x0.z, t); t = fmaf(wv[q][3], x0.w, t);
                        t = fmaf(wv[q][4], x1.x, t); t = fmaf(wv[q][5], x1.y, t);
                        t = fmaf(wv[q][6], x1.z, t); t = fmaf(wv[q][7], x1.w, t);
                        acc[q][s][r] = t;
                    }
                }
            }
            for (int c = kv + sub * 32 + lane; c < k; c += 32 * SPLIT)
            {
                #pragma unroll
                for (int q = 0; q < RW; q++)
                {
                    if (!live[q]) continue;
                    float wv = load1(W, rowBase[q] + c);
                    #pragma unroll
                    for (int r = 0; r < R; r++) acc[q][s][r] = fmaf(wv, xb[(long long)r * ldx + c], acc[q][s][r]);
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < RW; q++)
            #pragma unroll
            for (int s = 0; s < 2; s++)
                #pragma unroll
                for (int r = 0; r < R; r++) acc[q][s][r] = (s == 0 || a.glu) ? warp_sum(acc[q][s][r]) : 0.f;

        if (SPLIT > 1)
        {
            if (lane == 0)
                #pragma unroll
                for (int q = 0; q < RW; q++)
                    #pragma unroll
                    for (int s = 0; s < 2; s++)
                        #pragma unroll
                        for (int r = 0; r < R; r++) part[warp][q][s][r] = acc[q][s][r];
            __syncthreads();
            if (sub == 0)
            {
                #pragma unroll
                for (int t = 1; t < SPLIT; t++)
                    #pragma unroll
                    for (int q = 0; q < RW; q++)
                        #pragma unroll
                        for (int s = 0; s < 2; s++)
                            #pragma unroll
                            for (int r = 0; r < R; r++) acc[q][s][r] += part[warp + t][q][s][r];
            }
            __syncthreads();
        }

        if (sub == 0 && lane == 0)
        {
            #pragma unroll
            for (int q = 0; q < RW; q++)
            {
                int j = jb + q;
                if (j >= a.n) continue;
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    float g = acc[q][0][r], u = acc[q][1][r];
                    float v;
                    if (a.glu)
                    {
                        if (a.bias) { g += a.bias[j]; u += a.bias[a.n + j]; }
                        v = activate(g, a.act) * u;
                    }
                    else
                    {
                        v = g;
                        if (a.bias) v += a.bias[j];
                        v = activate(v, a.act);
                    }
                    if (a.residual) v = a.residual[(long long)r * a.ldr + j] + a.resScale * v;
                    a.y[(long long)r * a.ldy + j] = v;
                }
            }
        }
    }
}

static int gSms = 0;

template <typename WT, int SPLIT>
static void launch_gemv_split(Engine* e, const GemvArgs& a, int rows, int blocks, size_t smem)
{
    switch (rows)
    {
        case 1: launch(gemv_kernel<WT, 1, SPLIT, 2>, blocks, 256, smem, e->stream, a); break;
        case 2: launch(gemv_kernel<WT, 2, SPLIT, 2>, blocks, 256, smem, e->stream, a); break;
        case 3: launch(gemv_kernel<WT, 3, SPLIT, 1>, blocks, 256, smem, e->stream, a); break;
        case 4: launch(gemv_kernel<WT, 4, SPLIT, 1>, blocks, 256, smem, e->stream, a); break;
        case 5: launch(gemv_kernel<WT, 5, SPLIT, 1>, blocks, 256, smem, e->stream, a); break;
        case 6: launch(gemv_kernel<WT, 6, SPLIT, 1>, blocks, 256, smem, e->stream, a); break;
        case 7: launch(gemv_kernel<WT, 7, SPLIT, 1>, blocks, 256, smem, e->stream, a); break;
        default: launch(gemv_kernel<WT, 8, SPLIT, 1>, blocks, 256, smem, e->stream, a); break;
    }
}

template <typename WT>
static int launch_gemv(Engine* e, const GemvArgs& a, int rows)
{
    if (rows < 1 || rows > 8) return -100;
    if (gSms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&gSms, cudaDevAttrMultiProcessorCount, dev);
        if (gSms <= 0) gSms = 64;
    }
    static int perSm = -1;
    if (perSm < 0)
    {
        const char* env = getenv("AMQE_NORM_BLOCKS_PER_SM");
        perSm = env ? atoi(env) : 4;
        if (perSm <= 0) perSm = 4;
    }
    // Enough warps per row that the machine holds ~32 warps per SM.
    int split = a.n >= gSms * 32 ? 1 : (a.n >= gSms * 12 ? 2 : 4);
    if (a.k < 256 * split) split = 1;
    int perGroup = (8 / split) * (rows <= 2 ? 2 : 1);
    long long groups = ((long long)a.n + perGroup - 1) / perGroup;
    // With the norm fused every block re-reads x, so a wide matrix runs on a
    // bounded grid and its blocks loop over rows instead.
    long long cap = a.normW ? (long long)gSms * perSm : groups;
    int blocks = (int)(groups < cap ? groups : cap);
    size_t smem = a.normW ? (size_t)rows * a.k * sizeof(float) : 0;
    if (smem > 48 * 1024) return -101;
    switch (split)
    {
        case 1: launch_gemv_split<WT, 1>(e, a, rows, blocks, smem); break;
        case 2: launch_gemv_split<WT, 2>(e, a, rows, blocks, smem); break;
        default: launch_gemv_split<WT, 4>(e, a, rows, blocks, smem); break;
    }
    return ok(cudaGetLastError());
}

// ── batched rows: widen + SGEMM, then the same epilogue ───────────────────

__global__ void widen_bf16_kernel(const __nv_bfloat16* src, float* dst, long long count)
{
    pdl_enter();
    long long i = (blockIdx.x * (long long)blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= count)
    {
        float v[8];
        load8(src, i, v);
        float4* d = reinterpret_cast<float4*>(dst + i);
        d[0] = make_float4(v[0], v[1], v[2], v[3]);
        d[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
    else
    {
        for (; i < count; i++) dst[i] = __bfloat162float(src[i]);
    }
}

// dst[r, c] = bf16(x[r, c]) — round to nearest even, packed [rows × k].
__global__ void to_bf16_kernel(const float* x, int ldx, int rows, int k, __nv_bfloat16* dst)
{
    pdl_enter();
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int k2 = k / 2;
    if (idx >= (long long)rows * k2) return;
    int r = (int)(idx / k2), c = (int)(idx % k2) * 2;
    float2 v = *reinterpret_cast<const float2*>(x + (long long)r * ldx + c);
    *reinterpret_cast<__nv_bfloat162*>(dst + (long long)r * k + c) = __floats2bfloat162_rn(v.x, v.y);
}

// out[r, j] = epilogue(raw[r, j]) — raw is the GEMM's [rows, n or 2n] output.
__global__ void epilogue_kernel(const float* raw, int ldraw, int rows, int n,
    float* y, int ldy, const float* bias, const float* residual, int ldr, float resScale, int act, int glu)
{
    pdl_enter();
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)rows * n) return;
    int r = (int)(idx / n), j = (int)(idx % n);
    float v;
    if (glu)
    {
        float g = raw[(long long)r * ldraw + j], u = raw[(long long)r * ldraw + n + j];
        if (bias) { g += bias[j]; u += bias[n + j]; }
        v = activate(g, act) * u;
    }
    else
    {
        v = raw[(long long)r * ldraw + j];
        if (bias) v += bias[j];
        v = activate(v, act);
    }
    if (residual) v = residual[(long long)r * ldr + j] + resScale * v;
    y[(long long)r * ldy + j] = v;
}

/// y = epilogue(x · Wᵀ). wType 0 = BF16, 1 = f32. `raw` is scratch for the
/// batched path, [rows × (glu ? 2n : n)] floats.
AMQE_EXPORT int amqe_linear(Engine* e, const void* w, int wType, int n, int k,
    const float* x, int ldx, int rows, float* y, int ldy,
    const float* bias, const float* residual, int ldr, float resScale, int act, int glu, float* raw,
    const float* normW, float normOff, const float* normBias, float normEps, int normKind)
{
    if (rows <= 0) return 0;
    static int prefetch = -1;
    if (prefetch < 0) { const char* v = getenv("AMQE_GEMV_PREFETCH"); prefetch = v ? atoi(v) : 1; }
    GemvArgs a{ w, n, k, x, ldx, y, ldy, bias, residual, ldr, resScale, act, glu,
                normW, normOff, normBias, normEps, normKind, prefetch };
    if (normW && rows > 8) return -101;
    if (rows <= 8)
    {
        return wType == 0 ? launch_gemv<__nv_bfloat16>(e, a, rows) : launch_gemv<float>(e, a, rows);
    }

    int outRows = glu ? 2 * n : n;
    bool direct = !glu && act == ACT_NONE && bias == nullptr &&
                  ((residual == nullptr) || (residual == y && ldr == ldy && resScale == 1.f));
    const float alpha = 1.f;
    if (wType == 0 && !e->prefillF32 && (k % 8) == 0 && (ldx % 2) == 0)
    {
        // BF16 tensor cores: activations rounded to BF16, the weights as
        // stored, products accumulated in f32.
        size_t need = (size_t)rows * k;
        if (need > e->xbElems)
        {
            if (e->xb) cudaFree(e->xb);
            e->xb = nullptr;
            e->xbElems = 0;
            if (cudaMalloc(&e->xb, need * sizeof(__nv_bfloat16)) != cudaSuccess) return -2;
            e->xbElems = need;
        }
        long long pairs = (long long)rows * (k / 2);
        launch(to_bf16_kernel, (unsigned)((pairs + 255) / 256), 256, 0, e->stream, x, ldx, rows, k, e->xb);
        const float beta = (direct && residual) ? 1.f : 0.f;
        float* c = direct ? y : raw;
        int ldc = direct ? ldy : outRows;
        int m = direct ? n : outRows;
        auto st = cublasGemmEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, rows, k,
            &alpha, w, CUDA_R_16BF, k, e->xb, CUDA_R_16BF, k, &beta, c, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
        if (direct) return ok(cudaGetLastError());
        long long total = (long long)rows * n;
        launch(epilogue_kernel, (unsigned)((total + 255) / 256), 256, 0, e->stream,
            raw, outRows, rows, n, y, ldy, bias, residual, ldr, resScale, act, glu);
        return ok(cudaGetLastError());
    }
    const float* wf = static_cast<const float*>(w);
    if (wType == 0)
    {
        size_t need = (size_t)outRows * k;
        if (need > e->widenElems)
        {
            if (e->widen) cudaFree(e->widen);
            e->widen = nullptr;
            e->widenElems = 0;
            if (cudaMalloc(&e->widen, need * sizeof(float)) != cudaSuccess) return -2;
            e->widenElems = need;
        }
        long long count = (long long)need;
        int threads = 256;
        long long blocks = (count / 8 + threads) / threads + 1;
        launch(widen_bf16_kernel, (unsigned)blocks, threads, 0, e->stream, static_cast<const __nv_bfloat16*>(w), e->widen, count);
        wf = e->widen;
    }

    // Fast path: plain residual add straight out of the GEMM (beta = 1).
    if (direct)
    {
        const float beta = residual ? 1.f : 0.f;
        auto st = cublasSgemm(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k,
            &alpha, wf, k, x, ldx, &beta, y, ldy);
        return st == CUBLAS_STATUS_SUCCESS ? ok(cudaGetLastError()) : -(1000 + (int)st);
    }
    const float beta0 = 0.f;
    auto st = cublasSgemm(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, outRows, rows, k,
        &alpha, wf, k, x, ldx, &beta0, raw, outRows);
    if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
    long long total = (long long)rows * n;
    launch(epilogue_kernel, (unsigned)((total + 255) / 256), 256, 0, e->stream, 
        raw, outRows, rows, n, y, ldy, bias, residual, ldr, resScale, act, glu);
    return ok(cudaGetLastError());
}

/// Batched rows (prompts): 0 = BF16 tensor cores (default), 1 = FP32 SGEMM
/// over widened weights, which keeps activations at f32.
AMQE_EXPORT int amqe_set_prefill_f32(Engine* e, int f32)
{
    e->prefillF32 = f32 ? 1 : 0;
    return 0;
}

// ── embedding ──────────────────────────────────────────────────────────────

template <typename WT>
__global__ void embed_kernel(const WT* table, const int* tokens, int hidden, float scale, int hasScale, float* out)
{
    pdl_enter();
    int r = blockIdx.x;
    long long base = (long long)tokens[r] * hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
    {
        float v = load1(table, base + c);
        out[(long long)r * hidden + c] = hasScale ? v * scale : v;
    }
}

AMQE_EXPORT int amqe_embed(Engine* e, const void* table, int wType, const int* tokens, int rows, int hidden,
    float scale, int hasScale, float* out)
{
    if (wType == 0)
        launch(embed_kernel<__nv_bfloat16>, rows, 256, 0, e->stream, static_cast<const __nv_bfloat16*>(table), tokens, hidden, scale, hasScale, out);
    else
        launch(embed_kernel<float>, rows, 256, 0, e->stream, static_cast<const float*>(table), tokens, hidden, scale, hasScale, out);
    return ok(cudaGetLastError());
}

// ── norms ──────────────────────────────────────────────────────────────────
// kind 0 = RMSNorm, 1 = LayerNorm. out = norm(x) · (w + wOff) (+ bias).
// out may equal x. Rows come from x with stride ldx (so a norm can read
// only the last row of a batch).

__global__ void norm_kernel(const float* x, long long ldx, float* out, long long ldo, const float* w, float wOff,
    const float* bias, float eps, int cols, int kind)
{
    pdl_enter();
    __shared__ float red[32];
    const float* xr = x + blockIdx.x * ldx;
    float* orow = out + blockIdx.x * ldo;
    float mean = 0.f;
    if (kind == 1)
    {
        float s = 0.f;
        for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c];
        mean = block_sum(s, red) / cols;
    }
    float ss = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) { float d = xr[c] - mean; ss += d * d; }
    float total = block_sum(ss, red);
    float inv = rsqrtf(total / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
    {
        float v = (xr[c] - mean) * inv * (w[c] + wOff);
        if (bias) v += bias[c];
        orow[c] = v;
    }
}

AMQE_EXPORT int amqe_norm(Engine* e, const float* x, long long ldx, float* out, long long ldo, int rows, int cols,
    const float* w, float wOff, const float* bias, float eps, int kind)
{
    int threads = cols >= 1024 ? 1024 : (cols >= 512 ? 512 : 256);
    launch(norm_kernel, rows, threads, 0, e->stream, x, ldx, out, ldo, w, wOff, bias, eps, cols, kind);
    return ok(cudaGetLastError());
}

// y = y + scale · b (the generic residual join).
__global__ void add_scaled_kernel(float* y, const float* b, float scale, long long count)
{
    pdl_enter();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < count) y[i] += scale * b[i];
}

AMQE_EXPORT int amqe_add_scaled(Engine* e, float* y, const float* b, float scale, long long count)
{
    launch(add_scaled_kernel, (unsigned)((count + 255) / 256), 256, 0, e->stream, y, b, scale, count);
    return ok(cudaGetLastError());
}

// ── softmax attention: q/k prep (norm, rope, cache append) ────────────────

struct AttnPrepArgs
{
    const float* qkv; int ldqkv;     // [rows, qProjWidth + 2·kvDim]
    int rows, hq, hkv, d;
    int gated;                       // q_proj interleaves per head [q_h | gate_h]
    float* q;                        // [rows, hq·d]
    float* gate;                     // [rows, hq·d] when gated
    float* kCache; float* vCache;    // [cap, hkv·d]
    const int* pos;                  // position of row 0
    const float* qNormW; float qNormOff; float qNormEps;
    const float* kNormW; float kNormOff; float kNormEps;
    const float* invFreq; int pairs; double ropeScale; int interleaved; int rope;
};

__global__ void attn_prep_kernel(AttnPrepArgs a)
{
    pdl_enter();
    extern __shared__ float buf[];   // d floats
    __shared__ float red[32];
    int r = blockIdx.x;
    int head = blockIdx.y;           // [0, hq) q heads, [hq, hq + hkv) kv heads
    int d = a.d;
    bool isQ = head < a.hq;
    int h = isQ ? head : head - a.hq;
    const float* row = a.qkv + (long long)r * a.ldqkv;
    int qw = a.hq * d * (a.gated ? 2 : 1);
    int kvDim = a.hkv * d;
    const float* src = isQ ? row + (long long)h * d * (a.gated ? 2 : 1) : row + qw + (long long)h * d;
    int position = *a.pos + r;

    for (int c = threadIdx.x; c < d; c += blockDim.x) buf[c] = src[c];
    if (isQ && a.gated)
        for (int c = threadIdx.x; c < d; c += blockDim.x) a.gate[(long long)r * a.hq * d + h * d + c] = src[d + c];
    __syncthreads();

    const float* nw = isQ ? a.qNormW : a.kNormW;
    if (nw)
    {
        float off = isQ ? a.qNormOff : a.kNormOff, eps = isQ ? a.qNormEps : a.kNormEps;
        float ss = 0.f;
        for (int c = threadIdx.x; c < d; c += blockDim.x) ss += buf[c] * buf[c];
        float inv = rsqrtf(block_sum(ss, red) / d + eps);
        for (int c = threadIdx.x; c < d; c += blockDim.x) buf[c] = buf[c] * inv * (nw[c] + off);
        __syncthreads();
    }

    if (a.rope)
    {
        for (int i = threadIdx.x; i < a.pairs; i += blockDim.x)
        {
            int ia = a.interleaved ? 2 * i : i;
            int ib = a.interleaved ? ia + 1 : i + a.pairs;
            double angle = (double)position * a.invFreq[i];
            double s, c;
            sincos(angle, &s, &c);
            c *= a.ropeScale; s *= a.ropeScale;
            double x1 = buf[ia], x2 = buf[ib];
            buf[ia] = (float)(x1 * c - x2 * s);
            buf[ib] = (float)(x1 * s + x2 * c);
        }
        __syncthreads();
    }

    if (isQ)
    {
        for (int c = threadIdx.x; c < d; c += blockDim.x) a.q[(long long)r * a.hq * d + h * d + c] = buf[c];
    }
    else
    {
        long long dst = (long long)position * kvDim + h * d;
        const float* vsrc = row + qw + kvDim + (long long)h * d;
        for (int c = threadIdx.x; c < d; c += blockDim.x)
        {
            a.kCache[dst + c] = buf[c];
            a.vCache[dst + c] = vsrc[c];
        }
    }
}

AMQE_EXPORT int amqe_attn_prep(Engine* e, const float* qkv, int ldqkv, int rows, int hq, int hkv, int d, int gated,
    float* q, float* gate, float* kCache, float* vCache, const int* pos,
    const float* qNormW, float qNormOff, float qNormEps,
    const float* kNormW, float kNormOff, float kNormEps,
    const float* invFreq, int pairs, double ropeScale, int interleaved, int rope)
{
    AttnPrepArgs a{ qkv, ldqkv, rows, hq, hkv, d, gated, q, gate, kCache, vCache, pos,
        qNormW, qNormOff, qNormEps, kNormW, kNormOff, kNormEps, invFreq, pairs, ropeScale, interleaved, rope };
    dim3 grid(rows, hq + hkv);
    int threads = d >= 128 ? 128 : 64;
    launch(attn_prep_kernel, grid, threads, d * sizeof(float), e->stream, a);
    return ok(cudaGetLastError());
}

// ── softmax attention ──────────────────────────────────────────────────────
//
// Decode (a few query rows) splits the key span into fixed 64-key slices,
// one block each, and a second pass merges the slices' partial softmaxes —
// the grid is sized by the cache capacity so a captured graph stays valid
// as the sequence grows (slices past the query's position exit at once).
// Prefill tiles queries and keys through shared memory with an online
// softmax.

__device__ __forceinline__ float warp_max(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float block_max(float v, float* shared)
{
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float t = threadIdx.x < warps ? shared[threadIdx.x] : -INFINITY;
        t = warp_max(t);
        if (threadIdx.x == 0) shared[0] = t;
    }
    __syncthreads();
    return shared[0];
}

constexpr int SPLIT_KEYS = 32;
constexpr int SPLIT_THREADS = 256;

__device__ __forceinline__ float softcapped(float s, float softcap)
{
    return softcap > 0.f ? softcap * tanhf(s / softcap) : s;
}

template <int DPL>
__global__ void __launch_bounds__(SPLIT_THREADS) attention_split_kernel(const float* q, const float* kCache,
    const float* vCache, float* part, const int* pos, int hq, int hkv, int nsplit,
    float scale, float softcap, int window, int d)
{
    pdl_enter();
    const int D = d;   // DPL = ceil(d / 32) dims per lane
    constexpr int WARPS = SPLIT_THREADS / 32;
    __shared__ float sp[SPLIT_KEYS];
    __shared__ float red[32];
    int h = blockIdx.x, r = blockIdx.y, split = blockIdx.z;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int qPos = *pos + r;
    int first = window > 0 ? max(0, qPos - window + 1) : 0;
    int k0 = max(split * SPLIT_KEYS, first);
    int k1 = min(split * SPLIT_KEYS + SPLIT_KEYS, qPos + 1);
    float* out = part + (((long long)r * hq + h) * nsplit + split) * (D + 2);
    if (k0 >= k1)
    {
        if (threadIdx.x == 0) { out[0] = -INFINITY; out[1] = 0.f; }
        return;
    }
    int n = k1 - k0;
    int kvh = h / (hq / hkv);
    int kvDim = hkv * D;
    const float* qr = q + ((long long)r * hq + h) * D;
    float qv[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; i++) qv[i] = lane + 32 * i < D ? qr[lane + 32 * i] : 0.f;

    // Scores: each warp takes every WARPS-th key, four keys in flight.
    for (int j0 = warp * 4; j0 < n; j0 += WARPS * 4)
    {
        float s[4] = { 0.f, 0.f, 0.f, 0.f };
        #pragma unroll
        for (int u = 0; u < 4; u++)
        {
            int j = j0 + u;
            if (j < n)
            {
                const float* kr = kCache + (long long)(k0 + j) * kvDim + kvh * D;
                #pragma unroll
                for (int i = 0; i < DPL; i++) if (lane + 32 * i < D) s[u] = fmaf(qv[i], kr[lane + 32 * i], s[u]);
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; u++)
        {
            float t = warp_sum(s[u]) * scale;
            if (lane == 0 && j0 + u < n) sp[j0 + u] = softcapped(t, softcap);
        }
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, sp[j]);
    m = block_max(m, red);
    float l = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        float p = __expf(sp[j] - m);
        sp[j] = p;
        l += p;
    }
    l = block_sum(l, red);   // its barriers also publish sp
    for (int c = threadIdx.x; c < D; c += blockDim.x)
    {
        const float* vr = vCache + (long long)k0 * kvDim + kvh * D + c;
        float acc = 0.f;
        #pragma unroll 16
        for (int j = 0; j < n; j++) acc = fmaf(sp[j], vr[(long long)j * kvDim], acc);
        out[2 + c] = acc;
    }
    if (threadIdx.x == 0) { out[0] = m; out[1] = l; }
}

__global__ void attention_combine_kernel(const float* part, const float* gate, float* out, int hq, int d, int nsplit)
{
    pdl_enter();
    extern __shared__ float wts[];   // nsplit
    __shared__ float red[32];
    int h = blockIdx.x, r = blockIdx.y;
    const float* base = part + ((long long)r * hq + h) * nsplit * (d + 2);
    float m = -INFINITY;
    for (int s = threadIdx.x; s < nsplit; s += blockDim.x) m = fmaxf(m, base[(long long)s * (d + 2)]);
    m = block_max(m, red);
    float l = 0.f;
    for (int s = threadIdx.x; s < nsplit; s += blockDim.x)
    {
        float ms = base[(long long)s * (d + 2)];
        float w = ms == -INFINITY ? 0.f : __expf(ms - m);
        wts[s] = w;
        l += w * base[(long long)s * (d + 2) + 1];
    }
    l = block_sum(l, red);
    float inv = l > 0.f ? 1.f / l : 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x)
    {
        float v = 0.f;
        for (int s = 0; s < nsplit; s++)
            if (wts[s] != 0.f) v = fmaf(wts[s], base[(long long)s * (d + 2) + 2 + c], v);
        v *= inv;
        long long o = ((long long)r * hq + h) * d + c;
        if (gate) v *= sigmoidf_(gate[o]);
        out[o] = v;
    }
}

constexpr int PF_QT = 16;   // query rows per block (4 per warp)
constexpr int PF_KT = 32;   // keys per tile (one per lane)

template <int DPL>
__global__ void __launch_bounds__(128) attention_prefill_kernel(const float* q, const float* kCache, const float* vCache,
    const float* gate, float* out, const int* pos, int rows, int hq, int hkv, float scale, float softcap, int window, int d)
{
    pdl_enter();
    const int D = d;   // DPL = ceil(d / 32) dims per lane
    extern __shared__ float smem[];
    float* sK = smem;                         // [KT][D + 1]
    float* sV = sK + PF_KT * (D + 1);         // [KT][D]
    float* sQ = sV + PF_KT * D;               // [QT][D]
    int h = blockIdx.x;
    int t0 = blockIdx.y * PF_QT;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int kvh = h / (hq / hkv);
    int kvDim = hkv * D;
    int pos0 = *pos;
    int lastRow = min(t0 + PF_QT, rows) - 1;
    int maxQ = pos0 + lastRow;
    int minQ = pos0 + t0;
    int kStart = window > 0 ? max(0, minQ - window + 1) : 0;
    kStart -= kStart % PF_KT;

    for (int i = threadIdx.x; i < PF_QT * D; i += blockDim.x)
    {
        int rr = t0 + i / D;
        sQ[i] = rr < rows ? q[((long long)rr * hq + h) * D + i % D] : 0.f;
    }

    float m[4], l[4], acc[4][DPL];
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        m[i] = -INFINITY; l[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < DPL; c++) acc[i][c] = 0.f;
    }

    for (int kt = kStart; kt <= maxQ; kt += PF_KT)
    {
        __syncthreads();
        for (int i = threadIdx.x; i < PF_KT * D; i += blockDim.x)
        {
            int j = i / D, c = i % D;
            int key = kt + j;
            bool valid = key <= maxQ;
            long long src = (long long)key * kvDim + kvh * D + c;
            sK[j * (D + 1) + c] = valid ? kCache[src] : 0.f;
            sV[j * D + c] = valid ? vCache[src] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int rr = warp * 4 + i;
            int qPos = pos0 + t0 + rr;
            if (t0 + rr >= rows || kt > qPos) continue;   // warp-uniform
            const float* qrow = sQ + rr * D;
            const float* krow = sK + lane * (D + 1);
            float s = 0.f;
            #pragma unroll 16
            for (int c = 0; c < D; c++) s = fmaf(qrow[c], krow[c], s);
            s = softcapped(s * scale, softcap);
            int key = kt + lane;
            bool masked = key > qPos || (window > 0 && qPos - key >= window);
            if (masked) s = -INFINITY;
            float tmax = warp_max(s);
            float mNew = fmaxf(m[i], tmax);
            if (mNew == -INFINITY) continue;
            float corr = __expf(m[i] - mNew);
            float p = masked ? 0.f : __expf(s - mNew);
            l[i] = l[i] * corr + warp_sum(p);
            #pragma unroll
            for (int c = 0; c < DPL; c++) acc[i][c] *= corr;
            #pragma unroll 8
            for (int j = 0; j < PF_KT; j++)
            {
                float pj = __shfl_sync(0xffffffffu, p, j);
                #pragma unroll
                for (int c = 0; c < DPL; c++) if (lane + 32 * c < D) acc[i][c] = fmaf(pj, sV[j * D + lane + 32 * c], acc[i][c]);
            }
            m[i] = mNew;
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int rr = t0 + warp * 4 + i;
        if (rr >= rows) continue;
        float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        #pragma unroll
        for (int c = 0; c < DPL; c++)
        {
            if (lane + 32 * c >= D) continue;
            long long o = ((long long)rr * hq + h) * D + lane + 32 * c;
            float v = acc[i][c] * inv;
            if (gate) v *= sigmoidf_(gate[o]);
            out[o] = v;
        }
    }
}

template <int DPL>
static int launch_attention(Engine* e, const float* q, const float* kCache, const float* vCache,
    const float* gate, float* out, const int* pos, int rows, int hq, int hkv,
    float scale, float softcap, int window, float* part, int nsplit, int d)
{
    const int D = d;
    if (rows <= 8)
    {
        dim3 grid(hq, rows, nsplit);
        launch(attention_split_kernel<DPL>, grid, SPLIT_THREADS, 0, e->stream, q, kCache, vCache, part, pos, hq, hkv, nsplit, scale, softcap, window, d);
        launch(attention_combine_kernel, dim3(hq, rows), D < 128 ? 128 : D, nsplit * sizeof(float), e->stream, part, gate, out, hq, D, nsplit);
        return ok(cudaGetLastError());
    }
    size_t smem = (size_t)(PF_KT * (D + 1) + PF_KT * D + PF_QT * D) * sizeof(float);
    static bool configured[9] = {};
    if (!configured[DPL])
    {
        constexpr int DM = 32 * DPL;   // the widest head this instantiation serves
        size_t most = (size_t)(PF_KT * (DM + 1) + PF_KT * DM + PF_QT * DM) * sizeof(float);
        cudaFuncSetAttribute(attention_prefill_kernel<DPL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)most);
        configured[DPL] = true;
    }
    dim3 grid(hq, (rows + PF_QT - 1) / PF_QT);
    launch(attention_prefill_kernel<DPL>, grid, 128, smem, e->stream, q, kCache, vCache, gate, out, pos, rows, hq, hkv, scale, softcap, window, d);
    return ok(cudaGetLastError());
}

// ── prompt attention as GEMMs ──────────────────────────────────────────────
//
// For a chunk of query rows at positions pos0.. over keys kStart..pos0+rows-1:
// per KV head, S = Q·Kᵀ for its group of query heads (one strided-batched
// GEMM), a masked softmax over each score row, then O = P·V. Tensor cores
// (TF32) unless the engine keeps prompts exact, where it is plain FP32.

// One block per (row, head in group): scale, softcap, causal and window
// mask, softmax, written back in place as probabilities.
__global__ void attn_softmax_rows_kernel(float* scores, int rows, int keys, int kStart, int pos0,
    float scale, float softcap, int window)
{
    pdl_enter();
    __shared__ float red[32];
    int r = blockIdx.x;
    float* row = scores + ((long long)blockIdx.y * rows + r) * keys;
    int qPos = pos0 + r;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < keys; j += blockDim.x)
    {
        int key = kStart + j;
        bool masked = key > qPos || (window > 0 && qPos - key >= window);
        float v = masked ? -INFINITY : softcapped(row[j] * scale, softcap);
        row[j] = v;
        m = fmaxf(m, v);
    }
    m = block_max(m, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < keys; j += blockDim.x)
    {
        float v = row[j];
        float ev = v == -INFINITY ? 0.f : __expf(v - m);
        row[j] = ev;
        sum += ev;
    }
    float inv = 1.f / block_sum(sum, red);
    for (int j = threadIdx.x; j < keys; j += blockDim.x) row[j] *= inv;
}

__global__ void attn_gate_kernel(float* out, const float* gate, long long count)
{
    pdl_enter();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] *= sigmoidf_(gate[i]);
}

/// Prompt attention through cuBLAS; pos0 is the chunk's first position (the
/// host's copy of the device counter). Returns -200 when the score scratch
/// would pass its budget, and the caller falls back to amqe_attention.
AMQE_EXPORT int amqe_attention_rows(Engine* e, const float* q, const float* kCache, const float* vCache,
    const float* gate, float* out, int pos0, int rows, int hq, int hkv, int d,
    float scale, float softcap, int window)
{
    if (rows <= 0 || hkv <= 0 || hq % hkv != 0) return -100;
    int g = hq / hkv;
    int kvDim = hkv * d;
    int last = pos0 + rows - 1;
    int kStart = window > 0 ? max(0, pos0 - window + 1) : 0;
    int keys = last + 1 - kStart;
    size_t need = (size_t)g * rows * keys;
    if (need * sizeof(float) > ((size_t)1 << 30)) return -200;
    if (need > e->scoresElems)
    {
        if (e->scores) cudaFree(e->scores);
        e->scores = nullptr;
        e->scoresElems = 0;
        if (cudaMalloc(&e->scores, need * sizeof(float)) != cudaSuccess) return -2;
        e->scoresElems = need;
    }
    cublasComputeType_t ct = e->prefillF32 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
    const float one = 1.f, zero = 0.f;
    long long strideS = (long long)rows * keys;
    for (int kh = 0; kh < hkv; kh++)
    {
        const float* K = kCache + (long long)kStart * kvDim + kh * d;
        const float* V = vCache + (long long)kStart * kvDim + kh * d;
        const float* Q = q + (long long)kh * g * d;
        float* O = out + (long long)kh * g * d;
        // Column-major: S[keys x rows] = K^T . Q, per head of the group.
        auto st = cublasGemmStridedBatchedEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, keys, rows, d,
            &one, K, CUDA_R_32F, kvDim, 0, Q, CUDA_R_32F, hq * d, d,
            &zero, e->scores, CUDA_R_32F, keys, strideS, g, ct, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
        launch(attn_softmax_rows_kernel, dim3(rows, g), 256, 0, e->stream, e->scores, rows, keys, kStart, pos0,
            scale, softcap, window);
        // O[d x rows] = V . P.
        st = cublasGemmStridedBatchedEx(e->blas, CUBLAS_OP_N, CUBLAS_OP_N, d, rows, keys,
            &one, V, CUDA_R_32F, kvDim, 0, e->scores, CUDA_R_32F, keys, strideS,
            &zero, O, CUDA_R_32F, hq * d, d, g, ct, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
    }
    if (gate)
    {
        long long count = (long long)rows * hq * d;
        launch(attn_gate_kernel, (unsigned)((count + 255) / 256), 256, 0, e->stream, out, gate, count);
    }
    return ok(cudaGetLastError());
}

/// part: scratch of [8 × hq × nsplit × (d + 2)] floats for the decode split;
/// nsplit = ceil(cache capacity / SPLIT_KEYS) — see amqe_attention_split_keys.
AMQE_EXPORT int amqe_attention_split_keys(void) { return SPLIT_KEYS; }

AMQE_EXPORT int amqe_attention(Engine* e, const float* q, const float* kCache, const float* vCache,
    const float* gate, float* out, const int* pos, int rows, int hq, int hkv, int d,
    float scale, float softcap, int window, float* part, int nsplit)
{
    if (d < 1 || d > 256) return -100;
    switch ((d + 31) / 32)
    {
#define ATTN_CASE(n) case n: return launch_attention<n>(e, q, kCache, vCache, gate, out, pos, rows, hq, hkv, scale, softcap, window, part, nsplit, d);
        ATTN_CASE(1) ATTN_CASE(2) ATTN_CASE(3) ATTN_CASE(4) ATTN_CASE(5) ATTN_CASE(6) ATTN_CASE(7) ATTN_CASE(8)
#undef ATTN_CASE
        default: return -100;
    }
}

// ── Gated DeltaNet ─────────────────────────────────────────────────────────

// Depthwise causal conv + SiLU over the qkv stream. history[c·(K−1) + i] is
// the input i + 1 steps before row 0. Many rows: every (row, channel) at
// once, then the history advances in a second pass (a few rows go through
// gdn_conv_prep_kernel instead).
__global__ void gdn_conv_rows_kernel(const float* proj, int ldp, int rows, int channels, const float* w, int kernel,
    const float* history, float* out)
{
    pdl_enter();
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)rows * channels) return;
    int r = (int)(idx / channels), c = (int)(idx % channels);
    int hl = kernel - 1;
    float acc = 0.f;
    for (int i = 0; i < kernel; i++)
    {
        int p = r - hl + i;
        float x = p >= 0 ? proj[(long long)p * ldp + c] : history[(long long)c * hl + (-p - 1)];
        acc += w[(long long)c * kernel + i] * x;
    }
    out[idx] = silu(acc);
}

__global__ void gdn_conv_advance_kernel(const float* proj, int ldp, int rows, int channels, int kernel, float* history)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int hl = kernel - 1;
    // New entry i is the input i + 1 steps before the next row: row rows-1-i,
    // or an older history entry when the batch is shorter than the window.
    for (int i = hl - 1; i >= 0; i--)
    {
        int p = rows - 1 - i;
        history[(long long)c * hl + i] = p >= 0 ? proj[(long long)p * ldp + c] : history[(long long)c * hl + (i - rows)];
    }
}

// A few rows (decode): the conv, the q/k L2 norm and decay/β in one launch.
// Blocks are KD channels wide, so each q or k block is exactly one key
// head and normalises its own outputs; the last block computes decay and β.
__global__ void gdn_conv_prep_kernel(const float* proj, int ldp, int rows, int channels, int keyDim, int nk,
    const float* w, int kernel, float* history, float* conv, float* qk,
    int bOff, int aOff, int nv, const float* aLog, const float* dtBias, float* gb)
{
    pdl_enter();
    __shared__ float red[32];
    if (blockIdx.x == gridDim.x - 1)
    {
        for (int i = threadIdx.x; i < rows * nv; i += blockDim.x)
        {
            int r = i / nv, v = i % nv;
            const float* pr = proj + (long long)r * ldp;
            float x = pr[aOff + v] + dtBias[v];
            float sp = x > 20.f ? x : log1pf(expf(x));
            gb[(long long)r * 2 * nv + v] = expf(-expf(aLog[v]) * sp);
            gb[(long long)r * 2 * nv + nv + v] = sigmoidf_(pr[bOff + v]);
        }
        return;
    }
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    bool live = c < channels;
    bool isQk = blockIdx.x < 2 * nk;   // block-uniform
    int hl = kernel - 1;
    float hist[8];
    float wc[9];
    if (live)
    {
        for (int i = 0; i < hl; i++) hist[i] = history[(long long)c * hl + i];
        for (int i = 0; i < kernel; i++) wc[i] = w[(long long)c * kernel + i];
    }
    for (int r = 0; r < rows; r++)
    {
        float y = 0.f, x = 0.f;
        if (live)
        {
            x = proj[(long long)r * ldp + c];
            float acc = 0.f;
            for (int i = 0; i < hl; i++) acc += wc[i] * hist[hl - 1 - i];
            acc += wc[hl] * x;
            y = silu(acc);
            conv[(long long)r * channels + c] = y;
            for (int i = hl - 1; i >= 1; i--) hist[i] = hist[i - 1];
            if (hl > 0) hist[0] = x;
        }
        if (isQk)
        {
            float ss = block_sum(y * y, red);
            qk[(long long)r * 2 * keyDim + c] = y * rsqrtf(ss + 1e-6f);
        }
    }
    if (live)
        for (int i = 0; i < hl; i++) history[(long long)c * hl + i] = hist[i];
}

// Per (row, key head): q and k L2-normalised (eps 1e-6) into qk [rows,
// 2·keyDim]; per (row, value head): decay exp(g), g = −exp(A_log)·
// softplus(a + dt_bias), and β = sigmoid(b) into gb [rows, 2·nv]. With
// logDecay, g itself instead of exp(g) (the chunked recurrence's input).
__global__ void gdn_prep_kernel(const float* conv, int keyDim, int convDim, int kd,
    const float* proj, int ldp, int bOff, int aOff, int nv,
    const float* aLog, const float* dtBias, float* qk, float* gb, int logDecay)
{
    pdl_enter();
    __shared__ float red[32];
    int r = blockIdx.x, kh = blockIdx.y;
    const float* cr = conv + (long long)r * convDim;
    float qss = 0.f, kss = 0.f;
    for (int i = threadIdx.x; i < kd; i += blockDim.x)
    {
        float qv = cr[kh * kd + i], kv = cr[keyDim + kh * kd + i];
        qss += qv * qv; kss += kv * kv;
    }
    qss = block_sum(qss, red);
    kss = block_sum(kss, red);
    float qInv = rsqrtf(qss + 1e-6f);
    float kInv = rsqrtf(kss + 1e-6f);
    float* out = qk + (long long)r * 2 * keyDim;
    for (int i = threadIdx.x; i < kd; i += blockDim.x)
    {
        out[kh * kd + i] = cr[kh * kd + i] * qInv;
        out[keyDim + kh * kd + i] = cr[keyDim + kh * kd + i] * kInv;
    }
    if (kh == 0)
    {
        const float* pr = proj + (long long)r * ldp;
        for (int v = threadIdx.x; v < nv; v += blockDim.x)
        {
            float x = pr[aOff + v] + dtBias[v];
            float sp = x > 20.f ? x : log1pf(expf(x));
            float g = -expf(aLog[v]) * sp;
            gb[(long long)r * 2 * nv + v] = logDecay ? g : expf(g);
            gb[(long long)r * 2 * nv + nv + v] = sigmoidf_(pr[bOff + v]);
        }
    }
}

// The recurrence over every row, sequential in position: block (head, 32
// value columns); four lanes share a column, each holding a quarter of its
// key rows of the state in registers, so no barrier sits inside the loop.
//   S ← decay·S;  δ = β·(v − Sᵀk);  S ← S + k ⊗ δ;  o = Sᵀq / √K
// FUSED (decode): one block covers the whole head and applies the z-gated
// RMSNorm itself — a barrier per row, which a few rows can afford.
template <int KD, bool FUSED>
__global__ void __launch_bounds__(FUSED ? 512 : 128) gdn_scan_kernel(const float* conv, const float* qk, const float* gb,
    int rows, int keyDim, int convDim, int nk, int nv, int vd, float* state, float* out, int valueDim,
    const float* proj, int ldp, int zOff, const float* normW, float normEps)
{
    pdl_enter();
    constexpr int KP = KD / 4;
    __shared__ float red[32];
    int h = blockIdx.x;
    int col = blockIdx.y * 32 + (threadIdx.x >> 2);
    int part = threadIdx.x & 3;
    int kh = h / (nv / nk);
    float S[KP];
    float* st = state + (long long)h * KD * vd;
    #pragma unroll
    for (int i = 0; i < KP; i++) S[i] = st[(long long)(part * KP + i) * vd + col];
    const float qScale = rsqrtf((float)KD);

    for (int r = 0; r < rows; r++)
    {
        const float* qr = qk + (long long)r * 2 * keyDim + kh * KD + part * KP;
        const float* kr = qr + keyDim;
        float decay = gb[(long long)r * 2 * nv + h];
        float beta = gb[(long long)r * 2 * nv + nv + h];
        float v = conv[(long long)r * convDim + 2 * keyDim + h * vd + col];
        float kv[KP], qv[KP];
        #pragma unroll
        for (int i = 0; i < KP; i += 4)
        {
            float4 a = *reinterpret_cast<const float4*>(kr + i);
            float4 b = *reinterpret_cast<const float4*>(qr + i);
            kv[i] = a.x; kv[i + 1] = a.y; kv[i + 2] = a.z; kv[i + 3] = a.w;
            qv[i] = b.x; qv[i + 1] = b.y; qv[i + 2] = b.z; qv[i + 3] = b.w;
        }
        float m0 = 0.f, m1 = 0.f;
        #pragma unroll
        for (int i = 0; i < KP; i += 2)
        {
            S[i] *= decay; S[i + 1] *= decay;
            m0 = fmaf(S[i], kv[i], m0);
            m1 = fmaf(S[i + 1], kv[i + 1], m1);
        }
        float kvMem = m0 + m1;
        kvMem += __shfl_xor_sync(0xffffffffu, kvMem, 1);
        kvMem += __shfl_xor_sync(0xffffffffu, kvMem, 2);
        float delta = (v - kvMem) * beta;
        float o0 = 0.f, o1 = 0.f;
        #pragma unroll
        for (int i = 0; i < KP; i += 2)
        {
            S[i] = fmaf(kv[i], delta, S[i]);
            S[i + 1] = fmaf(kv[i + 1], delta, S[i + 1]);
            o0 = fmaf(S[i], qv[i], o0);
            o1 = fmaf(S[i + 1], qv[i + 1], o1);
        }
        float o = o0 + o1;
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        o *= qScale;
        if (FUSED)
        {
            float ss = block_sum(part == 0 ? o * o : 0.f, red);
            if (part == 0)
            {
                float z = proj[(long long)r * ldp + zOff + h * vd + col];
                out[(long long)r * valueDim + h * vd + col] = o * rsqrtf(ss / vd + normEps) * normW[col] * silu(z);
            }
        }
        else if (part == 0)
        {
            out[(long long)r * valueDim + h * vd + col] = o;
        }
    }
    #pragma unroll
    for (int i = 0; i < KP; i++) st[(long long)(part * KP + i) * vd + col] = S[i];
}

// The recurrence over a prompt: the same arithmetic as gdn_scan_kernel, laid
// out for latency. LANES = KD/16 lanes share a value column (16 key rows of
// state each, so the dependent FMA chains are half as long), a block is 8
// columns (so 4x the blocks reach the SMs), and the next row's q, k, v, decay
// and β are loaded before this row's arithmetic, so the loads' latency hides
// behind it instead of opening every step.
template <int KD>
__global__ void __launch_bounds__(8 * (KD / 16)) gdn_scan_rows_kernel(const float* __restrict__ qk,
    const float* __restrict__ conv, const float* __restrict__ gb,
    int rows, int keyDim, int convDim, int nk, int nv, int vd, float* __restrict__ state, float* __restrict__ out, int valueDim)
{
    pdl_enter();
    constexpr int KP = 16, LANES = KD / KP;
    int h = blockIdx.x;
    int col = blockIdx.y * 8 + threadIdx.x / LANES;
    int part = threadIdx.x % LANES;
    int kh = h / (nv / nk);
    float S[KP];
    float* st = state + (long long)h * KD * vd;
    #pragma unroll
    for (int i = 0; i < KP; i++) S[i] = st[(long long)(part * KP + i) * vd + col];
    const float qScale = rsqrtf((float)KD);
    const long long qkStride = 2LL * keyDim;
    const float* qBase = qk + kh * KD + part * KP;
    const float* vBase = conv + 2 * keyDim + h * vd + col;

    float kv[KP], qv[KP], v, decay, beta;
    auto load = [&](int r, float* kd_, float* qd_, float& v_, float& dc_, float& bt_)
    {
        const float* qr = qBase + r * qkStride;
        const float* kr = qr + keyDim;
        #pragma unroll
        for (int i = 0; i < KP; i += 4)
        {
            float4 a = __ldg(reinterpret_cast<const float4*>(kr + i));
            float4 b = __ldg(reinterpret_cast<const float4*>(qr + i));
            kd_[i] = a.x; kd_[i + 1] = a.y; kd_[i + 2] = a.z; kd_[i + 3] = a.w;
            qd_[i] = b.x; qd_[i + 1] = b.y; qd_[i + 2] = b.z; qd_[i + 3] = b.w;
        }
        v_ = __ldg(vBase + (long long)r * convDim);
        dc_ = __ldg(gb + (long long)r * 2 * nv + h);
        bt_ = __ldg(gb + (long long)r * 2 * nv + nv + h);
    };
    load(0, kv, qv, v, decay, beta);
    for (int r = 0; r < rows; r++)
    {
        float kn[KP], qn[KP], vn = 0.f, dn = 0.f, bn = 0.f;
        if (r + 1 < rows) load(r + 1, kn, qn, vn, dn, bn);
        float m0 = 0.f, m1 = 0.f;
        #pragma unroll
        for (int i = 0; i < KP; i += 2)
        {
            S[i] *= decay; S[i + 1] *= decay;
            m0 = fmaf(S[i], kv[i], m0);
            m1 = fmaf(S[i + 1], kv[i + 1], m1);
        }
        float kvMem = m0 + m1;
        #pragma unroll
        for (int o = 1; o < LANES; o <<= 1) kvMem += __shfl_xor_sync(0xffffffffu, kvMem, o);
        float delta = (v - kvMem) * beta;
        float o0 = 0.f, o1 = 0.f;
        #pragma unroll
        for (int i = 0; i < KP; i += 2)
        {
            S[i] = fmaf(kv[i], delta, S[i]);
            S[i + 1] = fmaf(kv[i + 1], delta, S[i + 1]);
            o0 = fmaf(S[i], qv[i], o0);
            o1 = fmaf(S[i + 1], qv[i + 1], o1);
        }
        float o = o0 + o1;
        #pragma unroll
        for (int x = 1; x < LANES; x <<= 1) o += __shfl_xor_sync(0xffffffffu, o, x);
        if (part == 0) out[(long long)r * valueDim + h * vd + col] = o * qScale;
        #pragma unroll
        for (int i = 0; i < KP; i++) { kv[i] = kn[i]; qv[i] = qn[i]; }
        v = vn; decay = dn; beta = bn;
    }
    #pragma unroll
    for (int i = 0; i < KP; i++) st[(long long)(part * KP + i) * vd + col] = S[i];
}

// ── the recurrence over a prompt, in chunks (WY form) ──────────────────────
//
// Over a chunk of C rows with state S₀ at its start and G_t the running sum
// of log-decays inside it, the delta rule unrolls to
//   (I + A)·Δ = diag(β)·V − diag(β·e^G)·K·S₀,   A_ts = β_t·e^(G_t−G_s)·k_t·k_s  (s < t)
// so with W = (I+A)⁻¹·diag(β·e^G)·K and U = (I+A)⁻¹·diag(β)·V, Δ = U − W·S₀,
//   S_C = e^(G_C)·S₀ + Σ_s e^(G_C−G_s)·k_s ⊗ Δ_s
//   o_t = (e^(G_t)·q_tᵀ·S₀ + Σ_{s≤t} e^(G_t−G_s)·(q_t·k_s)·Δ_s) / √K.
// Three kernels: W, U, the decayed keys and the masked, decayed Q·Kᵀ depend
// only on the chunk, so they are built for every chunk at once; the state is
// then carried chunk to chunk (two products per chunk, not a step per row);
// and the outputs, which need only each chunk's starting state and Δ, are
// formed for every chunk at once again. Decays stay in log space, so no
// ratio of two underflowed products ever forms.
//
// Scratch, with rows padded to whole chunks (Rp = chunks·C, padded rows 0):
//   WT [nv × chunks × KD × C]   W, transposed per chunk
//   QT [nv × chunks × KD × C]   e^(G_t)·q_t, transposed per chunk
//   U  [nv × Rp × vd]            U, then Δ in place
//   KT [nv × Rp × KD]            e^(G_C−G_t)·k_t
//   G  [nv × Rp]                 running log-decay
//   PT [nv × chunks × C × C]     P_ts = e^(G_t−G_s)·q_t·k_s (s ≤ t) at [s][t]
//   S0 [nv × chunks × KD × vd]   the state at each chunk's start

constexpr int GDN_CHUNK = 64;
constexpr int GDN_LD = GDN_CHUNK + 4;   // padded row of a transposed chunk in shared memory

// The prep kernel's first region: q and k transposed, later the solved rows.
template <int KD> __host__ __device__ constexpr int gdn_prep_region()
{
    return 2 * KD * GDN_LD > (GDN_CHUNK - 16) * 256 ? 2 * KD * GDN_LD : (GDN_CHUNK - 16) * 256;
}

// Block (value head, chunk), 256 threads. gb holds log-decays here.
template <int KD>
__global__ void __launch_bounds__(256) gdn_chunk_prep_kernel(const float* __restrict__ qk,
    const float* __restrict__ conv, const float* __restrict__ gb, int rows, int keyDim, int convDim,
    int nk, int nv, int vd, float* __restrict__ WT, float* __restrict__ QT, float* __restrict__ U, float* __restrict__ KT,
    float* __restrict__ G, float* __restrict__ PT)
{
    pdl_enter();
    constexpr int C = GDN_CHUNK, LD = GDN_LD;
    extern __shared__ float sm[];
    float* kT = sm;                  // [KD][LD]: k_t[i] at i·LD + t
    float* qT = kT + KD * LD;        // [KD][LD]
    float* A = sm + gdn_prep_region<KD>(); // [C][LD]: A_ts at s·LD + t
    float* g = A + C * LD;           // [C] running log-decay
    float* bt = g + C;               // [C] β
    int h = blockIdx.x, c = blockIdx.y, nC = gridDim.y;
    int r0 = c * C, L = min(C, rows - r0), Rp = nC * C;
    int kh = h / (nv / nk);
    int tid = threadIdx.x;

    // q and k as float4s, every load issued before the first store (few
    // warps per SM, so nothing else would hide their latency).
    {
        constexpr int N = C * KD / 4 / 256;
        float4 kv[N], qv[N];
        #pragma unroll
        for (int n = 0; n < N; n++)
        {
            int idx = tid + n * 256, t = idx / (KD / 4), i = (idx % (KD / 4)) * 4;
            const float* row = qk + (long long)(r0 + t) * 2 * keyDim + kh * KD + i;
            float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
            kv[n] = t < L ? __ldg(reinterpret_cast<const float4*>(row + keyDim)) : z;
            qv[n] = t < L ? __ldg(reinterpret_cast<const float4*>(row)) : z;
        }
        #pragma unroll
        for (int n = 0; n < N; n++)
        {
            int idx = tid + n * 256, t = idx / (KD / 4), i = (idx % (KD / 4)) * 4;
            kT[i * LD + t] = kv[n].x; kT[(i + 1) * LD + t] = kv[n].y; kT[(i + 2) * LD + t] = kv[n].z; kT[(i + 3) * LD + t] = kv[n].w;
            qT[i * LD + t] = qv[n].x; qT[(i + 1) * LD + t] = qv[n].y; qT[(i + 2) * LD + t] = qv[n].z; qT[(i + 3) * LD + t] = qv[n].w;
        }
    }
    if (tid < C)
    {
        g[tid] = tid < L ? gb[(long long)(r0 + tid) * 2 * nv + h] : 0.f;
        bt[tid] = tid < L ? gb[(long long)(r0 + tid) * 2 * nv + nv + h] : 0.f;
    }
    __syncthreads();
    if (tid < 32)
    {
        // Running sum over 64 entries: two per lane, then a warp scan.
        float a = g[2 * tid], b = a + g[2 * tid + 1];
        float run = b;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            float n = __shfl_up_sync(0xffffffffu, run, o);
            if (tid >= o) run += n;
        }
        float before = run - b;
        g[2 * tid] = before + a;
        g[2 * tid + 1] = run;
    }
    __syncthreads();
    if (tid < C) G[(long long)h * Rp + r0 + tid] = g[tid];
    float gL = g[L - 1];
    for (int idx = tid; idx < C * KD; idx += blockDim.x)
    {
        int t = idx / KD, i = idx % KD;
        KT[((long long)h * Rp + r0 + t) * KD + i] = kT[i * LD + t] * expf(gL - g[t]);
    }
    float* qt = QT + ((long long)h * nC + c) * KD * C;
    for (int idx = tid; idx < C * KD; idx += blockDim.x)
    {
        int i = idx / C, t = idx % C;
        qt[idx] = qT[i * LD + t] * expf(g[t]);
    }

    // K·Kᵀ and Q·Kᵀ, a 4×4 tile of (t, s) per thread.
    {
        int ty = tid / 16, tx = tid % 16;
        float kk[4][4] = {}, qq[4][4] = {};
        #pragma unroll 4
        for (int i = 0; i < KD; i++)
        {
            float4 kt = *reinterpret_cast<const float4*>(kT + i * LD + ty * 4);
            float4 qt = *reinterpret_cast<const float4*>(qT + i * LD + ty * 4);
            float4 ks = *reinterpret_cast<const float4*>(kT + i * LD + tx * 4);
            float a[4] = { kt.x, kt.y, kt.z, kt.w }, q[4] = { qt.x, qt.y, qt.z, qt.w }, b[4] = { ks.x, ks.y, ks.z, ks.w };
            #pragma unroll
            for (int x = 0; x < 4; x++)
            #pragma unroll
            for (int y = 0; y < 4; y++)
            {
                kk[x][y] = fmaf(a[x], b[y], kk[x][y]);
                qq[x][y] = fmaf(q[x], b[y], qq[x][y]);
            }
        }
        float* pt = PT + ((long long)h * nC + c) * C * C;
        #pragma unroll
        for (int x = 0; x < 4; x++)
        #pragma unroll
        for (int y = 0; y < 4; y++)
        {
            int t = ty * 4 + x, s = tx * 4 + y;
            float d = s <= t ? expf(g[t] - g[s]) : 0.f;
            A[s * LD + t] = s < t ? bt[t] * d * kk[x][y] : 0.f;
            pt[s * C + t] = d * qq[x][y];
        }
    }
    __syncthreads();

    // Forward substitution through (I + A), one column of [β·e^G·K | β·V] per
    // thread (KD + vd ≤ 256). Padded rows have β = 0, so their A rows and
    // right-hand sides are 0 and they solve to 0. Rows go in blocks of 16:
    // a block takes its earlier blocks' share as 16 independent FMAs per
    // earlier row, then solves its own 16×16 triangle; solved rows wait in
    // shared memory (the keys' space, free once every right-hand side is
    // read), each thread reading back only its own column.
    int j = tid;
    int nj = KD + vd;
    bool active = j < nj;
    bool isW = j < KD;
    float rhs[C];
    {
        const float* vcol = conv + (long long)r0 * convDim + 2 * keyDim + h * vd + (j - KD);
        #pragma unroll
        for (int t = 0; t < C; t++)
            rhs[t] = !active ? 0.f : isW ? bt[t] * expf(g[t]) * kT[j * LD + t]
                : (t < L ? bt[t] * __ldg(vcol + (long long)t * convDim) : 0.f);
    }
    __syncthreads();
    float* X = kT;                   // [C][nj], rows of earlier blocks
    #pragma unroll
    for (int b = 0; b < C / 16; b++)
    {
        float x[16];
        #pragma unroll
        for (int r = 0; r < 16; r++) x[r] = rhs[16 * b + r];
        #pragma unroll 4
        for (int s = 0; s < 16 * b; s++)
        {
            float xs = active ? X[s * nj + j] : 0.f;
            const float4* a = reinterpret_cast<const float4*>(A + s * LD + 16 * b);
            #pragma unroll
            for (int q = 0; q < 4; q++)
            {
                float4 v = a[q];
                x[4 * q] = fmaf(-v.x, xs, x[4 * q]);
                x[4 * q + 1] = fmaf(-v.y, xs, x[4 * q + 1]);
                x[4 * q + 2] = fmaf(-v.z, xs, x[4 * q + 2]);
                x[4 * q + 3] = fmaf(-v.w, xs, x[4 * q + 3]);
            }
        }
        #pragma unroll
        for (int s = 0; s < 15; s++)
        {
            const float* a = A + (16 * b + s) * LD + 16 * b;
            #pragma unroll
            for (int t = s + 1; t < 16; t++) x[t] = fmaf(-a[t], x[s], x[t]);
        }
        if (active)
        {
            if (b + 1 < C / 16)
            {
                #pragma unroll
                for (int r = 0; r < 16; r++) X[(16 * b + r) * nj + j] = x[r];
            }
            if (isW)
            {
                float4* dst = reinterpret_cast<float4*>(WT + (((long long)h * nC + c) * KD + j) * C + 16 * b);
                #pragma unroll
                for (int q = 0; q < 4; q++) dst[q] = make_float4(x[4 * q], x[4 * q + 1], x[4 * q + 2], x[4 * q + 3]);
            }
            else
            {
                float* dst = U + ((long long)h * Rp + r0 + 16 * b) * vd + (j - KD);
                #pragma unroll
                for (int r = 0; r < 16; r++) dst[(long long)r * vd] = x[r];
            }
        }
    }
}

// Block (value head, VT value columns), 256 threads, sequential over chunks;
// the block's slice of the state stays in shared memory throughout. Per
// chunk it records the starting state, turns U into Δ = U − W·S and moves S
// on. Each chunk's keys load (cp.async) while its W·S runs, and the next
// chunk's W while its state update runs.
template <int KD, int VT>
__global__ void __launch_bounds__(256) gdn_chunk_state_kernel(const float* __restrict__ WT, float* __restrict__ U,
    const float* __restrict__ KT, const float* __restrict__ G, int rows, int nv, int vd,
    float* __restrict__ state, float* __restrict__ S0)
{
    pdl_enter();
    constexpr int C = GDN_CHUNK;
    constexpr int TY = 256 / VT, RPT = C / TY, SPT = KD / TY;
    extern __shared__ float sm[];
    float* w = sm;                   // [KD][C]: W transposed
    float* k = w + KD * C;           // [C][KD]: decayed keys
    float* S = k + C * KD;           // [KD][VT]
    float* dv = S + KD * VT;         // [C][VT]
    int h = blockIdx.x;
    int tid = threadIdx.x, tx = tid % VT, ty = tid / VT;
    int col = blockIdx.y * VT + tx;
    int nC = (rows + C - 1) / C, Rp = nC * C;
    float* st = state + (long long)h * KD * vd;
    #pragma unroll
    for (int j = 0; j < SPT; j++) S[(ty * SPT + j) * VT + tx] = st[(long long)(ty * SPT + j) * vd + col];

    // Both blocks are KD·C contiguous floats.
    auto stage = [&](float* dst, const float* src)
    {
        for (int idx = tid; idx < KD * C / 4; idx += blockDim.x)
        {
            unsigned a = (unsigned)__cvta_generic_to_shared(dst + idx * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(a), "l"(src + idx * 4));
        }
        asm volatile("cp.async.commit_group;");
    };
    stage(w, WT + (long long)h * nC * KD * C);

    for (int c = 0; c < nC; c++)
    {
        int r0 = c * C, L = min(C, rows - r0);
        stage(k, KT + ((long long)h * Rp + r0) * KD);
        float eL = expf(__ldg(G + (long long)h * Rp + r0 + L - 1));
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        float* s0 = S0 + ((long long)h * nC + c) * KD * vd;
        #pragma unroll
        for (int j = 0; j < SPT; j++) s0[(long long)(ty * SPT + j) * vd + col] = S[(ty * SPT + j) * VT + tx];

        float* u = U + ((long long)h * Rp + r0) * vd + col;
        float uv[RPT];
        #pragma unroll
        for (int a = 0; a < RPT; a++) uv[a] = u[(long long)(ty * RPT + a) * vd];
        float aw[RPT] = {};
        #pragma unroll 4
        for (int i = 0; i < KD; i++)
        {
            float s = S[i * VT + tx];
            #pragma unroll
            for (int a = 0; a < RPT; a += 4)
            {
                float4 w4 = *reinterpret_cast<const float4*>(w + i * C + ty * RPT + a);
                aw[a] = fmaf(w4.x, s, aw[a]); aw[a + 1] = fmaf(w4.y, s, aw[a + 1]);
                aw[a + 2] = fmaf(w4.z, s, aw[a + 2]); aw[a + 3] = fmaf(w4.w, s, aw[a + 3]);
            }
        }
        #pragma unroll
        for (int a = 0; a < RPT; a++)
        {
            int t = ty * RPT + a;
            float d = uv[a] - aw[a];
            dv[t * VT + tx] = d;
            u[(long long)t * vd] = d;
        }
        __syncthreads();
        if (c + 1 < nC)
        {
            stage(w, WT + ((long long)h * nC + c + 1) * KD * C);
            asm volatile("cp.async.wait_group 1;");
        }
        else
        {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        // S ← e^(G_C)·S + K̃ᵀ·Δ over this thread's own SPT rows.
        float acc[SPT];
        #pragma unroll
        for (int j = 0; j < SPT; j++) acc[j] = S[(ty * SPT + j) * VT + tx] * eL;
        #pragma unroll 4
        for (int t = 0; t < C; t++)
        {
            float d = dv[t * VT + tx];
            #pragma unroll
            for (int j = 0; j < SPT; j += 4)
            {
                float4 k4 = *reinterpret_cast<const float4*>(k + t * KD + ty * SPT + j);
                acc[j] = fmaf(k4.x, d, acc[j]); acc[j + 1] = fmaf(k4.y, d, acc[j + 1]);
                acc[j + 2] = fmaf(k4.z, d, acc[j + 2]); acc[j + 3] = fmaf(k4.w, d, acc[j + 3]);
            }
        }
        #pragma unroll
        for (int j = 0; j < SPT; j++) S[(ty * SPT + j) * VT + tx] = acc[j];
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < SPT; j++) st[(long long)(ty * SPT + j) * vd + col] = S[(ty * SPT + j) * VT + tx];
}

// Block (value head, chunk, VT value columns), 256 threads: o for the chunk's
// rows from its starting state and Δ, written before the gated norm like
// gdn_scan_rows_kernel.
template <int KD, int VT>
__global__ void __launch_bounds__(256) gdn_chunk_out_kernel(const float* __restrict__ QT,
    const float* __restrict__ U, const float* __restrict__ PT, const float* __restrict__ S0,
    int rows, int nv, int vd, float* __restrict__ out, int valueDim)
{
    pdl_enter();
    constexpr int C = GDN_CHUNK;
    constexpr int TY = 256 / VT, RPT = C / TY;
    extern __shared__ float sm[];
    float* qT = sm;                  // [KD][C]: e^(G_t)·q_t
    float* pT = qT + KD * C;         // [C][C]
    float* S = pT + C * C;           // [KD][VT]
    float* dv = S + KD * VT;         // [C][VT]
    int h = blockIdx.x, c = blockIdx.y, nC = gridDim.y;
    int tid = threadIdx.x, tx = tid % VT, ty = tid / VT;
    int col = blockIdx.z * VT + tx;
    int r0 = c * C, L = min(C, rows - r0), Rp = nC * C;

    // Everything through cp.async, so the loads are all in flight at once.
    auto copy = [&](float* dst, const float* src)
    {
        unsigned a = (unsigned)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(a), "l"(src));
    };
    const float* qg = QT + ((long long)h * nC + c) * KD * C;
    for (int idx = tid; idx < KD * C / 4; idx += blockDim.x) copy(qT + idx * 4, qg + idx * 4);
    const float* pg = PT + ((long long)h * nC + c) * C * C;
    for (int idx = tid; idx < C * C / 4; idx += blockDim.x) copy(pT + idx * 4, pg + idx * 4);
    const float* s0 = S0 + ((long long)h * nC + c) * KD * vd + blockIdx.z * VT;
    for (int idx = tid; idx < KD * VT / 4; idx += blockDim.x)
        copy(S + idx * 4, s0 + (long long)(idx / (VT / 4)) * vd + (idx % (VT / 4)) * 4);
    const float* u = U + ((long long)h * Rp + r0) * vd + blockIdx.z * VT;
    for (int idx = tid; idx < C * VT / 4; idx += blockDim.x)
        copy(dv + idx * 4, u + (long long)(idx / (VT / 4)) * vd + (idx % (VT / 4)) * 4);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    float o[RPT] = {};
    #pragma unroll 4
    for (int i = 0; i < KD; i++)
    {
        float s = S[i * VT + tx];
        #pragma unroll
        for (int a = 0; a < RPT; a += 4)
        {
            float4 q4 = *reinterpret_cast<const float4*>(qT + i * C + ty * RPT + a);
            o[a] = fmaf(q4.x, s, o[a]); o[a + 1] = fmaf(q4.y, s, o[a + 1]);
            o[a + 2] = fmaf(q4.z, s, o[a + 2]); o[a + 3] = fmaf(q4.w, s, o[a + 3]);
        }
    }
    #pragma unroll 4
    for (int s = 0; s < C; s++)
    {
        float d = dv[s * VT + tx];
        #pragma unroll
        for (int a = 0; a < RPT; a += 4)
        {
            float4 p4 = *reinterpret_cast<const float4*>(pT + s * C + ty * RPT + a);
            o[a] = fmaf(p4.x, d, o[a]); o[a + 1] = fmaf(p4.y, d, o[a + 1]);
            o[a + 2] = fmaf(p4.z, d, o[a + 2]); o[a + 3] = fmaf(p4.w, d, o[a + 3]);
        }
    }
    const float qScale = rsqrtf((float)KD);
    #pragma unroll
    for (int a = 0; a < RPT; a++)
    {
        int t = ty * RPT + a;
        if (t < L) out[(long long)(r0 + t) * valueDim + h * vd + col] = o[a] * qScale;
    }
}

template <int KD> static size_t gdn_chunk_prep_smem()
{
    return sizeof(float) * (gdn_prep_region<KD>() + GDN_CHUNK * GDN_LD + 2 * GDN_CHUNK);
}

template <int KD, int VT> static size_t gdn_chunk_state_smem()
{
    return sizeof(float) * (2 * KD * GDN_CHUNK + KD * VT + GDN_CHUNK * VT);
}

template <int KD, int VT> static size_t gdn_chunk_out_smem()
{
    return sizeof(float) * (KD * GDN_CHUNK + GDN_CHUNK * GDN_CHUNK + KD * VT + GDN_CHUNK * VT);
}

template <int KD>
static int gdn_chunked(Engine* e, const float* qk, const float* conv, const float* gb, int rows, int keyDim, int convDim,
    int nk, int nv, int vd, float* state, float* out, int valueDim, int vt)
{
    constexpr int C = GDN_CHUNK;
    int nC = (rows + C - 1) / C, Rp = nC * C;
    size_t nWT = (size_t)nv * Rp * KD, nU = (size_t)nv * Rp * vd, nG = (size_t)nv * Rp;
    size_t nP = (size_t)nv * nC * C * C, nS = (size_t)nv * nC * KD * vd;
    size_t need = 3 * nWT + nU + nG + nP + nS;
    if (need > e->chunkElems)
    {
        if (e->chunk) cudaFree(e->chunk);
        e->chunk = nullptr;
        e->chunkElems = 0;
        if (cudaMalloc(&e->chunk, need * sizeof(float)) != cudaSuccess) return -2;
        e->chunkElems = need;
    }
    float* WT = e->chunk;
    float* KT = WT + nWT;
    float* QT = KT + nWT;
    float* U = QT + nWT;
    float* PT = U + nU;
    float* S0 = PT + nP;
    float* G = S0 + nS;
    static bool ready = false;
    if (!ready)
    {
        cudaFuncSetAttribute(gdn_chunk_prep_kernel<KD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gdn_chunk_prep_smem<KD>());
        cudaFuncSetAttribute(gdn_chunk_state_kernel<KD, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gdn_chunk_state_smem<KD, 16>());
        cudaFuncSetAttribute(gdn_chunk_state_kernel<KD, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gdn_chunk_state_smem<KD, 32>());
        cudaFuncSetAttribute(gdn_chunk_out_kernel<KD, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gdn_chunk_out_smem<KD, 32>());
        ready = true;
    }
    launch(gdn_chunk_prep_kernel<KD>, dim3(nv, nC), 256, gdn_chunk_prep_smem<KD>(), e->stream,
        qk, conv, gb, rows, keyDim, convDim, nk, nv, vd, WT, QT, U, KT, G, PT);
    if (vt == 16)
        launch(gdn_chunk_state_kernel<KD, 16>, dim3(nv, vd / 16), 256, gdn_chunk_state_smem<KD, 16>(), e->stream,
            WT, U, KT, G, rows, nv, vd, state, S0);
    else
        launch(gdn_chunk_state_kernel<KD, 32>, dim3(nv, vd / 32), 256, gdn_chunk_state_smem<KD, 32>(), e->stream,
            WT, U, KT, G, rows, nv, vd, state, S0);
    launch(gdn_chunk_out_kernel<KD, 32>, dim3(nv, nC, vd / 32), 256, gdn_chunk_out_smem<KD, 32>(), e->stream,
        QT, U, PT, S0, rows, nv, vd, out, valueDim);
    return 0;
}

// z-gated RMSNorm over each value head, in place: y = ô · w · silu(z).
__global__ void gdn_gate_norm_kernel(float* o, int valueDim, int vd, const float* proj, int ldp, int zOff,
    const float* normW, float eps)
{
    pdl_enter();
    __shared__ float red[32];
    int r = blockIdx.x, h = blockIdx.y;
    float* row = o + (long long)r * valueDim + h * vd;
    float ss = 0.f;
    for (int d = threadIdx.x; d < vd; d += blockDim.x) ss += row[d] * row[d];
    float inv = rsqrtf(block_sum(ss, red) / vd + eps);
    const float* z = proj + (long long)r * ldp + zOff + h * vd;
    for (int d = threadIdx.x; d < vd; d += blockDim.x) row[d] = row[d] * inv * normW[d] * silu(z[d]);
}

/// The whole Gated DeltaNet mixer after its in-projection: proj holds
/// [qkv | z | b | a] per row; out receives the gated, normalised [rows ×
/// valueDim]. conv, qk and gb are scratch ([rows × convDim], [rows ×
/// 2·keyDim], [rows × 2·nv]); state and history advance in place.
AMQE_EXPORT int amqe_gdn(Engine* e, const float* proj, int ldp, int rows, int keyDim, int valueDim,
    int nk, int nv, int kd, int vd, const float* convW, int kernel, float* history,
    const float* aLog, const float* dtBias, const float* normW, float normEps,
    float* state, float* conv, float* qk, float* gb, float* out)
{
    if (vd % 32 != 0 || (kd != 64 && kd != 128) || kernel < 1 || kernel > 9) return -100;
    int convDim = 2 * keyDim + valueDim;
    int zOff = convDim, bOff = convDim + valueDim, aOff = convDim + valueDim + nv;
    bool few = rows <= 4;
    static int chunkMin = -1, chunkVt = -1;
    if (chunkMin < 0)
    {
        const char* v = getenv("AMQE_GDN_CHUNK");     // 0 disables; else the fewest rows it runs for
        chunkMin = v ? atoi(v) : 256;
        if (chunkMin == 0) chunkMin = 1 << 30;
        const char* t = getenv("AMQE_GDN_VT");
        chunkVt = t ? atoi(t) : 0;
    }
    bool chunked = !few && rows >= chunkMin && vd % 32 == 0 && kd + vd <= 256;
    if (few)
    {
        int blocks = (convDim + kd - 1) / kd + 1;
        launch(gdn_conv_prep_kernel, blocks, kd, 0, e->stream, proj, ldp, rows, convDim, keyDim, nk, convW, kernel,
            history, conv, qk, bOff, aOff, nv, aLog, dtBias, gb);
    }
    else
    {
        long long total = (long long)rows * convDim;
        launch(gdn_conv_rows_kernel, (unsigned)((total + 255) / 256), 256, 0, e->stream, proj, ldp, rows, convDim, convW, kernel, history, conv);
        launch(gdn_conv_advance_kernel, (convDim + 127) / 128, 128, 0, e->stream, proj, ldp, rows, convDim, kernel, history);
        launch(gdn_prep_kernel, dim3(rows, nk), kd, 0, e->stream, conv, keyDim, convDim, kd, proj, ldp, bOff, aOff, nv, aLog, dtBias, qk, gb, chunked ? 1 : 0);
    }
    bool fused = few && vd * 4 <= 512;
    static int rowsScan = -1;
    if (rowsScan < 0) { const char* v = getenv("AMQE_GDN_ROWS_SCAN"); rowsScan = v ? atoi(v) : 1; }
    if (chunked)
    {
        // Narrower column slices when the heads alone would leave SMs idle.
        static int sms = amqe_sm_count();
        int vt = chunkVt == 16 || chunkVt == 32 ? chunkVt : (nv * (vd / 32) < sms / 2 ? 16 : 32);
        int rc = kd == 128
            ? gdn_chunked<128>(e, qk, conv, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, vt)
            : gdn_chunked<64>(e, qk, conv, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, vt);
        if (rc) return rc;
        launch(gdn_gate_norm_kernel, dim3(rows, nv), vd < 64 ? 64 : (vd > 256 ? 256 : vd), 0, e->stream, out, valueDim, vd, proj, ldp, zOff, normW, normEps);
        return ok(cudaGetLastError());
    }
    if (!few && rowsScan && vd % 8 == 0)
    {
        dim3 g(nv, vd / 8);
        if (kd == 128) launch(gdn_scan_rows_kernel<128>, g, 64, 0, e->stream, qk, conv, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim);
        else launch(gdn_scan_rows_kernel<64>, g, 32, 0, e->stream, qk, conv, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim);
        launch(gdn_gate_norm_kernel, dim3(rows, nv), vd < 64 ? 64 : (vd > 256 ? 256 : vd), 0, e->stream, out, valueDim, vd, proj, ldp, zOff, normW, normEps);
        return ok(cudaGetLastError());
    }
    dim3 grid(nv, fused ? 1 : vd / 32);
    int threads = fused ? vd * 4 : 128;
    if (kd == 128)
    {
        if (fused) launch(gdn_scan_kernel<128, true>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps);
        else launch(gdn_scan_kernel<128, false>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps);
    }
    else
    {
        if (fused) launch(gdn_scan_kernel<64, true>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps);
        else launch(gdn_scan_kernel<64, false>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps);
    }
    if (!fused)
        launch(gdn_gate_norm_kernel, dim3(rows, nv), vd < 64 ? 64 : (vd > 256 ? 256 : vd), 0, e->stream, out, valueDim, vd, proj, ldp, zOff, normW, normEps);
    return ok(cudaGetLastError());
}

// ── head epilogue and position counter ─────────────────────────────────────

__global__ void logit_post_kernel(float* logits, long long count, float mult, int hasMult, float cap)
{
    pdl_enter();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = logits[i];
    if (hasMult) v = (float)((double)v * mult);
    if (cap > 0.f) v = (float)(cap * tanh((double)v / cap));
    logits[i] = v;
}

AMQE_EXPORT int amqe_logit_post(Engine* e, float* logits, long long count, float mult, int hasMult, float cap)
{
    launch(logit_post_kernel, (unsigned)((count + 255) / 256), 256, 0, e->stream, logits, count, mult, hasMult, cap);
    return ok(cudaGetLastError());
}

__global__ void advance_kernel(int* pos, int by) { pdl_enter(); *pos += by; }

AMQE_EXPORT int amqe_advance(Engine* e, int* pos, int by)
{
    launch(advance_kernel, 1, 1, 0, e->stream, pos, by);
    return ok(cudaGetLastError());
}
