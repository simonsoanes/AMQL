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
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <utility>
#include <type_traits>

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
    int rowInvariant = 0;        // 1: a decode GEMV computes each row the same for 1 to 8 rows
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

// ── 4-bit weights ──────────────────────────────────────────────────────────
//
// FP4 E2M1 elements two to a byte (low nibble first), row-major [rows × k],
// then a block scale per 16 elements (NVFP4: an E4M3 byte) or per 32 (MXFP4:
// an E8M0 exponent byte), row-major [rows × k/block]. A tensor scale (NVFP4's
// weight_scale_2) is not folded in: it is one f32 per output row, applied to
// the dot product (rows of stacked weights keep their own tensor's). Element
// times block scale is exact in f32 (and in BF16): at most 6 significant bits.
#ifndef FP4_UNROLL
#define FP4_UNROLL 1   // past one activation row the 4-bit step loop is long already: unrolling it thrashed the instruction cache
#endif
struct Fp4Nv { unsigned char b; };   // tags: the pointer type picks the reader
struct Fp4Mx { unsigned char b; };

template <typename WT> struct WTraits
{
    static constexpr bool fp4 = false;
    static constexpr int block = 1;
    __host__ __device__ static long long rowBytes(int k) { return (long long)k * sizeof(WT); }
};
template <> struct WTraits<Fp4Nv>
{
    static constexpr bool fp4 = true;
    static constexpr int block = 16;
    __host__ __device__ static long long rowBytes(int k) { return k / 2; }
};
template <> struct WTraits<Fp4Mx>
{
    static constexpr bool fp4 = true;
    static constexpr int block = 32;
    __host__ __device__ static long long rowBytes(int k) { return k / 2; }
};

__device__ __forceinline__ float e4m3_value(unsigned char b)
{
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&h));
}

__device__ __forceinline__ float e8m0_value(unsigned char b)
{
    // 2^(b − 127); b = 0 is 2^−127, below f32's normals.
    return b == 0 ? __int_as_float(0x00400000) : __int_as_float((int)b << 23);
}

__device__ __forceinline__ float fp4_block_scale(const Fp4Nv*, const unsigned char* scales, long long idx)
{
    return e4m3_value(__ldg(scales + (idx >> 4)));
}
__device__ __forceinline__ float fp4_block_scale(const Fp4Mx*, const unsigned char* scales, long long idx)
{
    return e8m0_value(__ldg(scales + (idx >> 5)));
}

__device__ __forceinline__ float2 e2m1x2_value(unsigned char pair)
{
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)pair, __NV_E2M1);
    return __half22float2(*reinterpret_cast<__half2*>(&raw));
}

// Eight E2M1 elements packed in one word, as f32, unscaled. Integer work
// only (conversion instructions run at a fraction of the integer rate): the
// magnitudes 0, ½, 1, 1½, 2, 3, 4, 6 are f32s whose low two bytes are zero,
// so a byte permute looks up their top two bytes for four elements at once,
// and the element's sign bit moves to bit 31.
__device__ __forceinline__ void decode8_e2m1(unsigned int word, float* o)
{
    const unsigned int topLo = 0x3F3F3F00u, topHi = 0x40404040u;   // byte 3, magnitudes 0..3 and 4..7
    const unsigned int midLo = 0xC0800000u, midHi = 0xC0804000u;   // byte 2
    #pragma unroll
    for (int half = 0; half < 2; half++)
    {
        unsigned int nib = word >> (16 * half);
        unsigned int sel = nib & 0x7777u;
        unsigned int top = __byte_perm(topLo, topHi, sel);
        unsigned int mid = __byte_perm(midLo, midHi, sel);
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            unsigned int bits = __byte_perm(mid, top, j | ((4 + j) << 4)) << 16;
            bits |= (nib << (28 - 4 * j)) & 0x80000000u;
            o[4 * half + j] = __uint_as_float(bits);
        }
    }
}

// Eight elements packed in one word, times their block scale.
__device__ __forceinline__ void decode8_fp4(unsigned int word, float s, float* o)
{
    decode8_e2m1(word, o);
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] *= s;
}

// Eight elements from idx (a multiple of 8, so one block scale covers them).
template <typename F>
__device__ __forceinline__ void load8_fp4(const F* w, const unsigned char* scales, long long idx, float* o)
{
    unsigned int word = __ldg(reinterpret_cast<const unsigned int*>(reinterpret_cast<const unsigned char*>(w) + (idx >> 1)));
    float s = fp4_block_scale(w, scales, idx);
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 p = e2m1x2_value((unsigned char)(word >> (8 * i)));
        o[2 * i] = p.x * s;
        o[2 * i + 1] = p.y * s;
    }
}

template <typename F>
__device__ __forceinline__ float load1_fp4(const F* w, const unsigned char* scales, long long idx)
{
    unsigned char pair = __ldg(reinterpret_cast<const unsigned char*>(w) + (idx >> 1));
    float2 p = e2m1x2_value(pair);
    return ((idx & 1) ? p.y : p.x) * fp4_block_scale(w, scales, idx);
}

__device__ __forceinline__ void loadw(const __nv_bfloat16* w, const unsigned char*, long long idx, float* o) { load8(w, idx, o); }
__device__ __forceinline__ void loadw(const float* w, const unsigned char*, long long idx, float* o) { load8(w, idx, o); }
__device__ __forceinline__ void loadw(const Fp4Nv* w, const unsigned char* sc, long long idx, float* o) { load8_fp4(w, sc, idx, o); }
__device__ __forceinline__ void loadw(const Fp4Mx* w, const unsigned char* sc, long long idx, float* o) { load8_fp4(w, sc, idx, o); }
__device__ __forceinline__ float loadw1(const __nv_bfloat16* w, const unsigned char*, long long idx) { return load1(w, idx); }
__device__ __forceinline__ float loadw1(const float* w, const unsigned char*, long long idx) { return load1(w, idx); }
__device__ __forceinline__ float loadw1(const Fp4Nv* w, const unsigned char* sc, long long idx) { return load1_fp4(w, sc, idx); }
__device__ __forceinline__ float loadw1(const Fp4Mx* w, const unsigned char* sc, long long idx) { return load1_fp4(w, sc, idx); }

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
    const unsigned char* wScales;   // 4-bit weights: the block scales
    const float* rowScale;          // 4-bit weights: each output row's tensor scale (null: 1)
};

// SPLIT warps share each output row (each takes every SPLIT-th 256-column
// run), so a narrow matrix — 1024 outputs over 170 SMs is six warps each —
// still keeps enough loads in flight to saturate memory.
template <typename WT, int R, int SPLIT, int RW, bool RMS>
__global__ void __launch_bounds__(256) gemv_kernel(GemvArgs a)
{
    constexpr int WARPS = 8;
    constexpr int GROUP = WARPS / SPLIT * RW;   // output rows per block iteration
    extern __shared__ float xs[];          // [R × k] when a norm other than RMS is fused
    __shared__ float red[32];
    __shared__ float part[SPLIT > 1 ? WARPS : 1][RW][2][R];
    __shared__ float partSs[SPLIT > 1 ? WARPS : 1][R];
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
        long long rowBytes = WTraits<WT>::rowBytes(k);
        int jb0 = blockIdx.x * GROUP + slot * RW;
        for (int q = 0; q < RW; q++)
        {
            int j = jb0 + q;
            if (j >= a.n) break;
            for (int s0 = 0; s0 < rowsW0; s0++)
            {
                const char* row = reinterpret_cast<const char*>(a.w) + (long long)(s0 == 0 ? j : a.n + j) * rowBytes;
                for (long long off = (long long)(sub * 32 + lane) * 128; off < rowBytes; off += 128LL * 32 * SPLIT)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(row + off));
            }
        }
    }
    pdl_wait();
    const float* xb = a.x;
    int ldx = a.ldx;
    // RMS norm without a bias (the common case): each warp reads x and the
    // norm's weights straight from cache inside the dot loop and sums x² over
    // its own columns; 1/rms is one number per row, applied in the epilogue.
    // Nothing waits on a block-wide pass before the weights start streaming.
    constexpr bool rms = RMS;
    if (a.normW && !rms)
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
    const float* gw = a.normW;
    const float off = a.normOff;

    const int rowsW = a.glu ? 2 : 1;
    bool vec = (k & (WTraits<WT>::fp4 ? 31 : 7)) == 0 && (ldx & 3) == 0;
    int kv = vec ? k : 0;
    int groups = (a.n + GROUP - 1) / GROUP;
    for (int grp = blockIdx.x; grp < groups; grp += gridDim.x)
    {
        int jb = grp * GROUP + slot * RW;
        float acc[RW][2][R];
        float ssq[R];
        #pragma unroll
        for (int r = 0; r < R; r++) ssq[r] = 0.f;
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
            // A lane takes E columns a step: 8 BF16 or f32 (16 or 32 bytes), or
            // 32 FP4 elements (16 bytes, one load for four groups of 8).
            constexpr bool fp4 = WTraits<WT>::fp4;
            constexpr int E = fp4 ? 32 : 8;
            #pragma unroll (fp4 ? (R == 1 ? 2 : FP4_UNROLL) : 4)
            for (int c0 = (sub * 32 + lane) * E; c0 < kv; c0 += 32 * E * SPLIT)
            {
            uint4 packs[RW];
            if constexpr (fp4)
            {
                #pragma unroll
                for (int q = 0; q < RW; q++)
                    packs[q] = live[q]
                        ? __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(a.w) + ((rowBase[q] + c0) >> 1)))
                        : make_uint4(0, 0, 0, 0);
            }
            #pragma unroll
            for (int g = 0; g < E / 8; g++)
            {
                const int c = c0 + 8 * g;
                float wv[RW][8];
                float gsc[RW];   // 4-bit: the group's block scale, applied to its partial sum
                #pragma unroll
                for (int q = 0; q < RW; q++)
                {
                    if constexpr (fp4)
                    {
                        unsigned int word = g == 0 ? packs[q].x : g == 1 ? packs[q].y : g == 2 ? packs[q].z : packs[q].w;
                        decode8_e2m1(word, wv[q]);
                        gsc[q] = live[q] ? fp4_block_scale(W, a.wScales, rowBase[q] + c) : 0.f;
                    }
                    else if (live[q]) loadw(W, a.wScales, rowBase[q] + c, wv[q]);
                    else
                        #pragma unroll
                        for (int i = 0; i < 8; i++) wv[q][i] = 0.f;
                }
                float4 g0 = make_float4(1.f, 1.f, 1.f, 1.f), g1 = g0;
                if (rms)
                {
                    g0 = *reinterpret_cast<const float4*>(gw + c);
                    g1 = *reinterpret_cast<const float4*>(gw + c + 4);
                    g0.x += off; g0.y += off; g0.z += off; g0.w += off;
                    g1.x += off; g1.y += off; g1.z += off; g1.w += off;
                }
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    const float4* xp = reinterpret_cast<const float4*>(xb + (long long)r * ldx + c);
                    float4 x0 = xp[0], x1 = xp[1];
                    if (rms)
                    {
                        if (s == 0)
                        {
                            float t = ssq[r];
                            t = fmaf(x0.x, x0.x, t); t = fmaf(x0.y, x0.y, t); t = fmaf(x0.z, x0.z, t); t = fmaf(x0.w, x0.w, t);
                            t = fmaf(x1.x, x1.x, t); t = fmaf(x1.y, x1.y, t); t = fmaf(x1.z, x1.z, t); t = fmaf(x1.w, x1.w, t);
                            ssq[r] = t;
                        }
                        x0.x *= g0.x; x0.y *= g0.y; x0.z *= g0.z; x0.w *= g0.w;
                        x1.x *= g1.x; x1.y *= g1.y; x1.z *= g1.z; x1.w *= g1.w;
                    }
                    #pragma unroll
                    for (int q = 0; q < RW; q++)
                    {
                        if constexpr (fp4)
                        {
                            float t = wv[q][0] * x0.x;
                            t = fmaf(wv[q][1], x0.y, t);
                            t = fmaf(wv[q][2], x0.z, t); t = fmaf(wv[q][3], x0.w, t);
                            t = fmaf(wv[q][4], x1.x, t); t = fmaf(wv[q][5], x1.y, t);
                            t = fmaf(wv[q][6], x1.z, t); t = fmaf(wv[q][7], x1.w, t);
                            acc[q][s][r] = fmaf(gsc[q], t, acc[q][s][r]);
                        }
                        else
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
            }
            }
            for (int c = kv + sub * 32 + lane; c < k; c += 32 * SPLIT)
            {
                float gc = rms ? gw[c] + off : 1.f;
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    float xv = xb[(long long)r * ldx + c];
                    if (rms && s == 0) ssq[r] = fmaf(xv, xv, ssq[r]);
                    xv *= gc;
                    #pragma unroll
                    for (int q = 0; q < RW; q++)
                    {
                        if (!live[q]) continue;
                        acc[q][s][r] = fmaf(loadw1(W, a.wScales, rowBase[q] + c), xv, acc[q][s][r]);
                    }
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < RW; q++)
            #pragma unroll
            for (int s = 0; s < 2; s++)
                #pragma unroll
                for (int r = 0; r < R; r++) acc[q][s][r] = (s == 0 || a.glu) ? warp_sum(acc[q][s][r]) : 0.f;
        if (rms)
            #pragma unroll
            for (int r = 0; r < R; r++) ssq[r] = warp_sum(ssq[r]);

        if (SPLIT > 1)
        {
            if (lane == 0)
            {
                #pragma unroll
                for (int q = 0; q < RW; q++)
                    #pragma unroll
                    for (int s = 0; s < 2; s++)
                        #pragma unroll
                        for (int r = 0; r < R; r++) part[warp][q][s][r] = acc[q][s][r];
                #pragma unroll
                for (int r = 0; r < R; r++) partSs[warp][r] = ssq[r];
            }
            __syncthreads();
            if (sub == 0)
            {
                #pragma unroll
                for (int t = 1; t < SPLIT; t++)
                {
                    #pragma unroll
                    for (int q = 0; q < RW; q++)
                        #pragma unroll
                        for (int s = 0; s < 2; s++)
                            #pragma unroll
                            for (int r = 0; r < R; r++) acc[q][s][r] += part[warp + t][q][s][r];
                    #pragma unroll
                    for (int r = 0; r < R; r++) ssq[r] += partSs[warp + t][r];
                }
            }
            __syncthreads();
        }

        if (sub == 0 && lane == 0)
        {
            float inv[R];
            #pragma unroll
            for (int r = 0; r < R; r++) inv[r] = rms ? rsqrtf(ssq[r] / k + a.normEps) : 1.f;
            #pragma unroll
            for (int q = 0; q < RW; q++)
            {
                int j = jb + q;
                if (j >= a.n) continue;
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    float g = acc[q][0][r] * inv[r], u = acc[q][1][r] * inv[r];
                    if (a.rowScale) { g *= a.rowScale[j]; if (a.glu) u *= a.rowScale[a.n + j]; }
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

// RW output rows per warp: each activation load then serves RW weight rows.
template <typename WT, int SPLIT, int RW, bool RMS>
static void launch_gemv_rows(Engine* e, const GemvArgs& a, int rows, int blocks, size_t smem)
{
    switch (rows)
    {
        case 3: launch(gemv_kernel<WT, 3, SPLIT, RW, RMS>, blocks, 256, smem, e->stream, a); break;
        case 4: launch(gemv_kernel<WT, 4, SPLIT, RW, RMS>, blocks, 256, smem, e->stream, a); break;
        case 5: launch(gemv_kernel<WT, 5, SPLIT, RW, RMS>, blocks, 256, smem, e->stream, a); break;
        case 6: launch(gemv_kernel<WT, 6, SPLIT, RW, RMS>, blocks, 256, smem, e->stream, a); break;
        case 7: launch(gemv_kernel<WT, 7, SPLIT, RW, RMS>, blocks, 256, smem, e->stream, a); break;
        default: launch(gemv_kernel<WT, 8, SPLIT, RW, RMS>, blocks, 256, smem, e->stream, a); break;
    }
}

template <typename WT, int SPLIT, bool RMS>
static void launch_gemv_split(Engine* e, const GemvArgs& a, int rows, int rw, int blocks, size_t smem)
{
    if (rows == 1)
    {
        if (rw >= 8) launch(gemv_kernel<WT, 1, SPLIT, 8, RMS>, blocks, 256, smem, e->stream, a);
        else if (rw >= 4) launch(gemv_kernel<WT, 1, SPLIT, 4, RMS>, blocks, 256, smem, e->stream, a);
        else launch(gemv_kernel<WT, 1, SPLIT, 2, RMS>, blocks, 256, smem, e->stream, a);
        return;
    }
    if (rows == 2)
    {
        if (rw >= 4) launch(gemv_kernel<WT, 2, SPLIT, 4, RMS>, blocks, 256, smem, e->stream, a);
        else launch(gemv_kernel<WT, 2, SPLIT, 2, RMS>, blocks, 256, smem, e->stream, a);
        return;
    }
#ifndef AMQE_GEMV_BENCH_FEW_ROWS   // a benchmark build that only runs one or two rows
    if (rw >= 4) launch_gemv_rows<WT, SPLIT, 4, RMS>(e, a, rows, blocks, smem);
    else if (rw == 2) launch_gemv_rows<WT, SPLIT, 2, RMS>(e, a, rows, blocks, smem);
    else launch_gemv_rows<WT, SPLIT, 1, RMS>(e, a, rows, blocks, smem);
#endif
}

template <typename WT, bool RMS>
static void launch_gemv_kernel(Engine* e, const GemvArgs& a, int rows, int split, int rw, int blocks, size_t smem)
{
    switch (split)
    {
        case 1: launch_gemv_split<WT, 1, RMS>(e, a, rows, rw, blocks, smem); break;
        case 2: launch_gemv_split<WT, 2, RMS>(e, a, rows, rw, blocks, smem); break;
        case 8: launch_gemv_split<WT, 8, RMS>(e, a, rows, rw, blocks, smem); break;
        default: launch_gemv_split<WT, 4, RMS>(e, a, rows, rw, blocks, smem); break;
    }
}

// Output rows per warp for 3+ activation rows: 2, or 4 from 6 rows (measured on
// Qwen3.5 0.8B, RTX 5090); AMQE_GEMV_RW=1, 2 or 4 fixes it. Each output keeps
// its own accumulation order, so the values do not depend on it.
static int gemv_rw(int rows)
{
    static int rw = -1;
    if (rw < 0) { const char* v = getenv("AMQE_GEMV_RW"); rw = v ? atoi(v) : 0; if (rw != 1 && rw != 2 && rw != 4) rw = 0; }
    return rw ? rw : (rows >= 6 ? 4 : 2);
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
    // Warps per output row. One or two activation rows (decode) are pure
    // weight streaming: four warps a row keeps the most loads in flight, and
    // eight when the matrix is so narrow (n = 1024) that four leave the SMs
    // under ~24 warps each — within 2% of the best of 1/2/4/8 on every
    // Qwen3.5 0.8B and 4B projection (RTX 5090). From three rows the FMAs per
    // weight grow and fewer warps a row win: about 32 warps per SM.
    // AMQE_GEMV_SPLIT=1, 2, 4 or 8 fixes it.
    // Row-invariant (speculative decoding verifies drafts as extra rows and
    // needs each row's logits exactly as one-row decode gives them): the
    // one-row rule for every row count, since the split sets the order each
    // output is summed in. RW and the grid do not.
    // 4-bit weights: a row of k = 2560 is only 1,280 bytes, so the fixed cost
    // of a row (the warps' reduction, the epilogue) weighs four times as much,
    // and rows under 4096 take one warp each (four from 4096). Measured on
    // Qwen 3.5 4B, RTX 5090: a GDN in-projection or the head gains most, and a
    // verify of several rows (speculative decoding, which must keep the
    // one-row split) runs a fifth faster than at four warps a row.
    constexpr bool fp4w = WTraits<WT>::fp4;
    int split;
    if (fp4w && (rows <= 2 || e->rowInvariant)) split = a.k >= 4096 ? 4 : 1;
    else if (rows <= 2 || e->rowInvariant) split = (long long)a.n * 4 / 2 < (long long)gSms * 24 ? 8 : 4;
    else split = a.n >= gSms * 32 ? 1 : (a.n >= gSms * 12 ? 2 : 4);
    static int splitEnv = -1;
    if (splitEnv < 0) { const char* v = getenv("AMQE_GEMV_SPLIT"); splitEnv = v ? atoi(v) : 0; }
    if (splitEnv == 1 || splitEnv == 2 || splitEnv == 4 || splitEnv == 8) split = splitEnv;
    constexpr int stepCols = 32 * (WTraits<WT>::fp4 ? 32 : 8);   // columns one warp covers a step
    while (split > 1 && a.k < stepCols * split) split /= 2;
    static int rwDecode = -1;   // AMQE_GEMV_RW_DECODE=2|4: output rows per warp at one or two rows
    if (rwDecode < 0) { const char* v = getenv("AMQE_GEMV_RW_DECODE"); rwDecode = v ? atoi(v) : 0; }
    // Output rows per warp do not change any output's sum: 4 for 4-bit rows under 4096 wide.
    int rwDefault = fp4w && a.k < 4096 ? 4 : 2;
    int rw = rows <= 2 ? (rwDecode == 2 || rwDecode == 4 || rwDecode == 8 ? rwDecode : rwDefault)
                       : (fp4w && !getenv("AMQE_GEMV_RW") ? 4 : gemv_rw(rows));
    int perGroup = (8 / split) * rw;
    long long groups = ((long long)a.n + perGroup - 1) / perGroup;
    // An RMS norm (no bias) is applied inside the dot loop; any other norm is
    // staged in shared memory by every block first, which re-reads x per
    // block, so a wide matrix then runs on a bounded grid and its blocks loop
    // over rows instead. The RMS path is bounded too, more loosely, so the
    // head's ~60k row groups do not each re-read x and the norm weights.
    bool rmsNorm = a.normW && a.normKind == 0 && !a.normBias;
    bool staged = a.normW && !rmsNorm;
    long long cap = staged ? (long long)gSms * perSm : (rmsNorm ? (long long)gSms * 8 : groups);
    int blocks = (int)(groups < cap ? groups : cap);
    size_t smem = staged ? (size_t)rows * a.k * sizeof(float) : 0;
    if (smem > 48 * 1024) return -101;
    if (rmsNorm) launch_gemv_kernel<WT, true>(e, a, rows, split, rw, blocks, smem);
    else launch_gemv_kernel<WT, false>(e, a, rows, split, rw, blocks, smem);
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
    float* y, int ldy, const float* bias, const float* residual, int ldr, float resScale, int act, int glu,
    const float* rowScale)
{
    pdl_enter();
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)rows * n) return;
    int r = (int)(idx / n), j = (int)(idx % n);
    float v;
    if (glu)
    {
        float g = raw[(long long)r * ldraw + j], u = raw[(long long)r * ldraw + n + j];
        if (rowScale) { g *= rowScale[j]; u *= rowScale[n + j]; }
        if (bias) { g += bias[j]; u += bias[n + j]; }
        v = activate(g, act) * u;
    }
    else
    {
        v = raw[(long long)r * ldraw + j];
        if (rowScale) v *= rowScale[j];
        if (bias) v += bias[j];
        v = activate(v, act);
    }
    if (residual) v = residual[(long long)r * ldr + j] + resScale * v;
    y[(long long)r * ldy + j] = v;
}

// A 4-bit weight decoded, eight elements a thread: to BF16 (exact) without
// the row scale, for the tensor-core GEMM whose epilogue applies it, or to
// f32 with it, as the managed path decodes (element × block scale × tensor scale).
template <typename F, typename OUT>
__global__ void dequant_fp4_kernel(const F* w, const unsigned char* scales, const float* rowScale,
    int k, long long count, OUT* out)
{
    pdl_enter();
    long long i = (blockIdx.x * (long long)blockDim.x + threadIdx.x) * 8;
    if (i >= count) return;
    float v[8];
    load8_fp4(w, scales, i, v);
    float rs = rowScale ? rowScale[i / k] : 1.f;
    #pragma unroll
    for (int t = 0; t < 8; t++)
    {
        if constexpr (sizeof(OUT) == 2) out[i + t] = __float2bfloat16_rn(v[t]);
        else out[i + t] = v[t] * rs;
    }
}

static int linear_fp4_gemm(Engine* e, const void* w, int wType, const unsigned char* wScales, const float* rowScale,
    int n, int k, const float* x, int ldx, int rows, float* y, int ldy,
    const float* bias, const float* residual, int ldr, float resScale, int act, int glu, float* raw);

/// y = epilogue(x · Wᵀ). wType 0 = BF16, 1 = f32, 2 = NVFP4, 3 = MXFP4 (the
/// 4-bit layout above: wScales the block scales, rowScale each output row's
/// tensor scale or null). `raw` is scratch for the batched path,
/// [rows × (glu ? 2n : n)] floats.
AMQE_EXPORT int amqe_linear(Engine* e, const void* w, int wType, int n, int k,
    const float* x, int ldx, int rows, float* y, int ldy,
    const float* bias, const float* residual, int ldr, float resScale, int act, int glu, float* raw,
    const float* normW, float normOff, const float* normBias, float normEps, int normKind,
    const unsigned char* wScales, const float* rowScale)
{
    if (rows <= 0) return 0;
    if (wType >= 2 && (k % 32 != 0 || wScales == nullptr)) return -100;
    static int prefetch = -1;
    if (prefetch < 0) { const char* v = getenv("AMQE_GEMV_PREFETCH"); prefetch = v ? atoi(v) : 1; }
    GemvArgs a{ w, n, k, x, ldx, y, ldy, bias, residual, ldr, resScale, act, glu,
                normW, normOff, normBias, normEps, normKind, prefetch, wScales, rowScale };
    // AMQE_GEMM_FROM=N: N rows or more go to cuBLAS instead of the GEMV (default 9).
    static int gemmFrom = -1;
    if (gemmFrom < 0) { const char* v = getenv("AMQE_GEMM_FROM"); gemmFrom = v ? atoi(v) : 9; if (gemmFrom < 2) gemmFrom = 9; }
    if (normW && rows >= gemmFrom) return -101;
    if (rows < gemmFrom)
    {
        switch (wType)
        {
            case 0: return launch_gemv<__nv_bfloat16>(e, a, rows);
            case 1: return launch_gemv<float>(e, a, rows);
            case 2: return launch_gemv<Fp4Nv>(e, a, rows);
            case 3: return launch_gemv<Fp4Mx>(e, a, rows);
            default: return -100;
        }
    }
    if (wType >= 2)
        return linear_fp4_gemm(e, w, wType, wScales, rowScale, n, k, x, ldx, rows, y, ldy, bias, residual, ldr, resScale, act, glu, raw);
    if (rowScale) return -100;

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
            raw, outRows, rows, n, y, ldy, bias, residual, ldr, resScale, act, glu, (const float*)nullptr);
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
        raw, outRows, rows, n, y, ldy, bias, residual, ldr, resScale, act, glu, (const float*)nullptr);
    return ok(cudaGetLastError());
}

// Batched rows over a 4-bit weight: decoded whole into scratch, then the BF16
// path (exact BF16 weights, the row scale in the epilogue) or, with prefill at
// f32, the SGEMM path over the f32 decode. The decode is a read of the 4-bit
// weight and a write of its wide copy, small next to the GEMM over 512 rows.
static int linear_fp4_gemm(Engine* e, const void* w, int wType, const unsigned char* wScales, const float* rowScale,
    int n, int k, const float* x, int ldx, int rows, float* y, int ldy,
    const float* bias, const float* residual, int ldr, float resScale, int act, int glu, float* raw)
{
    int outRows = glu ? 2 * n : n;
    long long count = (long long)outRows * k;
    unsigned blocks = (unsigned)((count / 8 + 255) / 256);
    const float alpha = 1.f, beta0 = 0.f;
    if (!e->prefillF32)
    {
        size_t need = (size_t)count;
        if (need * 2 > e->widenElems * 4)
        {
            if (e->widen) cudaFree(e->widen);
            e->widen = nullptr;
            e->widenElems = 0;
            if (cudaMalloc(&e->widen, need * sizeof(float)) != cudaSuccess) return -2;
            e->widenElems = need;
        }
        __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(e->widen);
        if (wType == 2) launch(dequant_fp4_kernel<Fp4Nv, __nv_bfloat16>, blocks, 256, 0, e->stream, (const Fp4Nv*)w, wScales, (const float*)nullptr, k, count, wb);
        else launch(dequant_fp4_kernel<Fp4Mx, __nv_bfloat16>, blocks, 256, 0, e->stream, (const Fp4Mx*)w, wScales, (const float*)nullptr, k, count, wb);
        size_t needX = (size_t)rows * k;
        if (needX > e->xbElems)
        {
            if (e->xb) cudaFree(e->xb);
            e->xb = nullptr;
            e->xbElems = 0;
            if (cudaMalloc(&e->xb, needX * sizeof(__nv_bfloat16)) != cudaSuccess) return -2;
            e->xbElems = needX;
        }
        long long pairs = (long long)rows * (k / 2);
        launch(to_bf16_kernel, (unsigned)((pairs + 255) / 256), 256, 0, e->stream, x, ldx, rows, k, e->xb);
        auto st = cublasGemmEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, outRows, rows, k,
            &alpha, wb, CUDA_R_16BF, k, e->xb, CUDA_R_16BF, k, &beta0, raw, CUDA_R_32F, outRows,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
        long long total = (long long)rows * n;
        launch(epilogue_kernel, (unsigned)((total + 255) / 256), 256, 0, e->stream,
            raw, outRows, rows, n, y, ldy, bias, residual, ldr, resScale, act, glu, rowScale);
        return ok(cudaGetLastError());
    }
    size_t need = (size_t)count;
    if (need > e->widenElems)
    {
        if (e->widen) cudaFree(e->widen);
        e->widen = nullptr;
        e->widenElems = 0;
        if (cudaMalloc(&e->widen, need * sizeof(float)) != cudaSuccess) return -2;
        e->widenElems = need;
    }
    if (wType == 2) launch(dequant_fp4_kernel<Fp4Nv, float>, blocks, 256, 0, e->stream, (const Fp4Nv*)w, wScales, rowScale, k, count, e->widen);
    else launch(dequant_fp4_kernel<Fp4Mx, float>, blocks, 256, 0, e->stream, (const Fp4Mx*)w, wScales, rowScale, k, count, e->widen);
    auto st = cublasSgemm(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, outRows, rows, k,
        &alpha, e->widen, k, x, ldx, &beta0, raw, outRows);
    if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
    long long total = (long long)rows * n;
    launch(epilogue_kernel, (unsigned)((total + 255) / 256), 256, 0, e->stream,
        raw, outRows, rows, n, y, ldy, bias, residual, ldr, resScale, act, glu, (const float*)nullptr);
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
    // A batch (null otherwise): row r is its own sequence, with its own caches and position.
    float* const* kCaches; float* const* vCaches; const int* const* poss;
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
    int position = a.poss ? *a.poss[r] : *a.pos + r;
    float* kCache = a.kCaches ? a.kCaches[r] : a.kCache;
    float* vCache = a.vCaches ? a.vCaches[r] : a.vCache;

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
            kCache[dst + c] = buf[c];
            vCache[dst + c] = vsrc[c];
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
        qNormW, qNormOff, qNormEps, kNormW, kNormOff, kNormEps, invFreq, pairs, ropeScale, interleaved, rope,
        nullptr, nullptr, nullptr };
    dim3 grid(rows, hq + hkv);
    int threads = d >= 128 ? 128 : 64;
    launch(attn_prep_kernel, grid, threads, d * sizeof(float), e->stream, a);
    return ok(cudaGetLastError());
}

// One decode row for each of `batch` sequences: row r appends to kCaches[r] and
// vCaches[r] at position *poss[r].
AMQE_EXPORT int amqe_attn_prep_batch(Engine* e, const float* qkv, int ldqkv, int batch, int hq, int hkv, int d, int gated,
    float* q, float* gate, float* const* kCaches, float* const* vCaches, const int* const* poss,
    const float* qNormW, float qNormOff, float qNormEps,
    const float* kNormW, float kNormOff, float kNormEps,
    const float* invFreq, int pairs, double ropeScale, int interleaved, int rope)
{
    AttnPrepArgs a{ qkv, ldqkv, batch, hq, hkv, d, gated, q, gate, nullptr, nullptr, nullptr,
        qNormW, qNormOff, qNormEps, kNormW, kNormOff, kNormEps, invFreq, pairs, ropeScale, interleaved, rope,
        kCaches, vCaches, poss };
    dim3 grid(batch, hq + hkv);
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
    float scale, float softcap, int window, int d,
    const float* const* kCaches, const float* const* vCaches, const int* const* poss)
{
    pdl_enter();
    const int D = d;   // DPL = ceil(d / 32) dims per lane
    constexpr int WARPS = SPLIT_THREADS / 32;
    __shared__ float sp[SPLIT_KEYS];
    __shared__ float red[32];
    int h = blockIdx.x, r = blockIdx.y, split = blockIdx.z;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // A batch: row r is its own sequence, with its own caches and position.
    if (kCaches) { kCache = kCaches[r]; vCache = vCaches[r]; }
    int qPos = poss ? *poss[r] : *pos + r;
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
        launch(attention_split_kernel<DPL>, grid, SPLIT_THREADS, 0, e->stream, q, kCache, vCache, part, pos, hq, hkv, nsplit, scale, softcap, window, d,
            (const float* const*)nullptr, (const float* const*)nullptr, (const int* const*)nullptr);
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

template <int DPL>
static int launch_attention_batch(Engine* e, const float* q, const float* const* kCaches, const float* const* vCaches,
    const float* gate, float* out, const int* const* poss, int batch, int hq, int hkv,
    float scale, float softcap, int window, float* part, int nsplit, int d)
{
    dim3 grid(hq, batch, nsplit);
    launch(attention_split_kernel<DPL>, grid, SPLIT_THREADS, 0, e->stream, q, (const float*)nullptr, (const float*)nullptr, part,
        (const int*)nullptr, hq, hkv, nsplit, scale, softcap, window, d, kCaches, vCaches, poss);
    launch(attention_combine_kernel, dim3(hq, batch), d < 128 ? 128 : d, nsplit * sizeof(float), e->stream, part, gate, out, hq, d, nsplit);
    return ok(cudaGetLastError());
}

// One decode row for each of up to 8 sequences: row r attends over its own
// caches up to *poss[r]. nsplit covers the longest cache; a slice past a row's
// position exits at once.
AMQE_EXPORT int amqe_attention_batch(Engine* e, const float* q, const float* const* kCaches, const float* const* vCaches,
    const float* gate, float* out, const int* const* poss, int batch, int hq, int hkv, int d,
    float scale, float softcap, int window, float* part, int nsplit)
{
    if (d < 1 || d > 256 || batch < 1 || batch > 8) return -100;
    switch ((d + 31) / 32)
    {
#define ATTN_CASE(n) case n: return launch_attention_batch<n>(e, q, kCaches, vCaches, gate, out, poss, batch, hq, hkv, scale, softcap, window, part, nsplit, d);
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
    int bOff, int aOff, int nv, const float* aLog, const float* dtBias, float* gb, float* histSnap)
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
            // The history after row r, for a verify that keeps only rows 0..r.
            if (histSnap && r < rows - 1)
                for (int i = 0; i < hl; i++) histSnap[((long long)r * channels + c) * hl + i] = hist[i];
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
    const float* proj, int ldp, int zOff, const float* normW, float normEps, float* stateSnap)
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
        // The state after row r, for a verify that keeps only rows 0..r.
        if (stateSnap && r < rows - 1)
        {
            float* sn = stateSnap + (long long)r * nv * KD * vd + (long long)h * KD * vd;
            #pragma unroll
            for (int i = 0; i < KP; i++) sn[(long long)(part * KP + i) * vd + col] = S[i];
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
    float* state, float* conv, float* qk, float* gb, float* out, float* stateSnap, float* histSnap)
{
    if (vd % 32 != 0 || (kd != 64 && kd != 128) || kernel < 1 || kernel > 9) return -100;
    int convDim = 2 * keyDim + valueDim;
    int zOff = convDim, bOff = convDim + valueDim, aOff = convDim + valueDim + nv;
    // Up to 8 rows (decode, and a speculative verify) take the one-launch
    // conv and the fused scan, whose arithmetic per row does not depend on
    // how many rows there are.
    bool few = rows <= 8;
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
            history, conv, qk, bOff, aOff, nv, aLog, dtBias, gb, histSnap);
    }
    else if (stateSnap || histSnap) return -100;
    else
    {
        long long total = (long long)rows * convDim;
        launch(gdn_conv_rows_kernel, (unsigned)((total + 255) / 256), 256, 0, e->stream, proj, ldp, rows, convDim, convW, kernel, history, conv);
        launch(gdn_conv_advance_kernel, (convDim + 127) / 128, 128, 0, e->stream, proj, ldp, rows, convDim, kernel, history);
        launch(gdn_prep_kernel, dim3(rows, nk), kd, 0, e->stream, conv, keyDim, convDim, kd, proj, ldp, bOff, aOff, nv, aLog, dtBias, qk, gb, chunked ? 1 : 0);
    }
    bool fused = few && vd * 4 <= 512;
    if ((stateSnap || histSnap) && !fused) return -100;
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
        if (fused) launch(gdn_scan_kernel<128, true>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps, stateSnap);
        else launch(gdn_scan_kernel<128, false>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps, stateSnap);
    }
    else
    {
        if (fused) launch(gdn_scan_kernel<64, true>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps, stateSnap);
        else launch(gdn_scan_kernel<64, false>, grid, threads, 0, e->stream, conv, qk, gb, rows, keyDim, convDim, nk, nv, vd, state, out, valueDim, proj, ldp, zOff, normW, normEps, stateSnap);
    }
    if (!fused)
        launch(gdn_gate_norm_kernel, dim3(rows, nv), vd < 64 ? 64 : (vd > 256 ? 256 : vd), 0, e->stream, out, valueDim, vd, proj, ldp, zOff, normW, normEps);
    return ok(cudaGetLastError());
}

// ── Gated DeltaNet, a batch of sequences ───────────────────────────────────
//
// One decode token for each of B independent sequences (row r of proj, conv,
// qk, gb and out is sequence r), each with its own conv history and recurrent
// state: histories[r] and states[r]. The arithmetic is gdn_conv_prep_kernel's
// and gdn_scan_kernel's for one row; the sequence is the grid's last
// dimension, so the whole batch is two launches instead of two per sequence.

__global__ void gdn_conv_prep_batch_kernel(const float* proj, int ldp, int channels, int keyDim, int nk,
    const float* w, int kernel, float* const* histories, float* conv, float* qk,
    int bOff, int aOff, int nv, const float* aLog, const float* dtBias, float* gb)
{
    pdl_enter();
    __shared__ float red[32];
    int r = blockIdx.y;
    const float* pr = proj + (long long)r * ldp;
    if (blockIdx.x == gridDim.x - 1)
    {
        for (int v = threadIdx.x; v < nv; v += blockDim.x)
        {
            float x = pr[aOff + v] + dtBias[v];
            float sp = x > 20.f ? x : log1pf(expf(x));
            gb[(long long)r * 2 * nv + v] = expf(-expf(aLog[v]) * sp);
            gb[(long long)r * 2 * nv + nv + v] = sigmoidf_(pr[bOff + v]);
        }
        return;
    }
    float* history = histories[r];
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    bool live = c < channels;
    bool isQk = blockIdx.x < 2 * nk;   // block-uniform
    int hl = kernel - 1;
    float y = 0.f;
    if (live)
    {
        float x = pr[c];
        float acc = 0.f;
        float* hc = history + (long long)c * hl;
        for (int i = 0; i < hl; i++) acc += w[(long long)c * kernel + i] * hc[hl - 1 - i];
        acc += w[(long long)c * kernel + hl] * x;
        y = silu(acc);
        conv[(long long)r * channels + c] = y;
        for (int i = hl - 1; i >= 1; i--) hc[i] = hc[i - 1];
        if (hl > 0) hc[0] = x;
    }
    if (isQk)
    {
        float ss = block_sum(y * y, red);
        qk[(long long)r * 2 * keyDim + c] = y * rsqrtf(ss + 1e-6f);
    }
}

template <int KD, bool FUSED>
__global__ void __launch_bounds__(FUSED ? 512 : 128) gdn_scan_batch_kernel(const float* conv, const float* qk, const float* gb,
    int keyDim, int convDim, int nk, int nv, int vd, float* const* states, float* out, int valueDim,
    const float* proj, int ldp, int zOff, const float* normW, float normEps)
{
    pdl_enter();
    constexpr int KP = KD / 4;
    __shared__ float red[32];
    int r = blockIdx.z;
    int h = blockIdx.x;
    int col = blockIdx.y * 32 + (threadIdx.x >> 2);
    int part = threadIdx.x & 3;
    int kh = h / (nv / nk);
    float S[KP];
    float* st = states[r] + (long long)h * KD * vd;
    #pragma unroll
    for (int i = 0; i < KP; i++) S[i] = st[(long long)(part * KP + i) * vd + col];
    const float qScale = rsqrtf((float)KD);

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
    #pragma unroll
    for (int i = 0; i < KP; i++) st[(long long)(part * KP + i) * vd + col] = S[i];
}

AMQE_EXPORT int amqe_gdn_batch(Engine* e, const float* proj, int ldp, int batch, int keyDim, int valueDim,
    int nk, int nv, int kd, int vd, const float* convW, int kernel, float* const* histories,
    const float* aLog, const float* dtBias, const float* normW, float normEps,
    float* const* states, float* conv, float* qk, float* gb, float* out)
{
    if (vd % 32 != 0 || (kd != 64 && kd != 128) || kernel < 1 || kernel > 9 || batch < 1) return -100;
    int convDim = 2 * keyDim + valueDim;
    int zOff = convDim, bOff = convDim + valueDim, aOff = convDim + valueDim + nv;
    int blocks = (convDim + kd - 1) / kd + 1;
    launch(gdn_conv_prep_batch_kernel, dim3(blocks, batch), kd, 0, e->stream, proj, ldp, convDim, keyDim, nk, convW, kernel,
        histories, conv, qk, bOff, aOff, nv, aLog, dtBias, gb);
    bool fused = vd * 4 <= 512;
    dim3 grid(nv, fused ? 1 : vd / 32, batch);
    int threads = fused ? vd * 4 : 128;
    if (kd == 128)
    {
        if (fused) launch(gdn_scan_batch_kernel<128, true>, grid, threads, 0, e->stream, conv, qk, gb, keyDim, convDim, nk, nv, vd, states, out, valueDim, proj, ldp, zOff, normW, normEps);
        else launch(gdn_scan_batch_kernel<128, false>, grid, threads, 0, e->stream, conv, qk, gb, keyDim, convDim, nk, nv, vd, states, out, valueDim, proj, ldp, zOff, normW, normEps);
    }
    else
    {
        if (fused) launch(gdn_scan_batch_kernel<64, true>, grid, threads, 0, e->stream, conv, qk, gb, keyDim, convDim, nk, nv, vd, states, out, valueDim, proj, ldp, zOff, normW, normEps);
        else launch(gdn_scan_batch_kernel<64, false>, grid, threads, 0, e->stream, conv, qk, gb, keyDim, convDim, nk, nv, vd, states, out, valueDim, proj, ldp, zOff, normW, normEps);
    }
    if (!fused)
        launch(gdn_gate_norm_kernel, dim3(batch, nv), vd < 64 ? 64 : (vd > 256 ? 256 : vd), 0, e->stream, out, valueDim, vd, proj, ldp, zOff, normW, normEps);
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

__global__ void set_int_kernel(int* p, int v) { pdl_enter(); *p = v; }

/// *p = v, in stream order (a position a captured graph or a later kernel reads).
AMQE_EXPORT int amqe_set_int(Engine* e, int* p, int v)
{
    launch(set_int_kernel, 1, 1, 0, e->stream, p, v);
    return ok(cudaGetLastError());
}

// The index of the largest of n values (the lowest index on a tie), written
// to out and, when given, to out2: a draft token the next kernel embeds.
__global__ void __launch_bounds__(1024) argmax_kernel(const float* x, int n, int* out, int* out2)
{
    pdl_enter();
    __shared__ float bv[32];
    __shared__ int bi[32];
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        if (v > best) { best = v; at = i; }
    }
    for (int o = 16; o > 0; o >>= 1)
    {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, at, o);
        if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { bv[warp] = best; bi[warp] = at; }
    __syncthreads();
    if (warp == 0)
    {
        int warps = blockDim.x >> 5;
        best = lane < warps ? bv[lane] : -INFINITY;
        at = lane < warps ? bi[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1)
        {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, at, o);
            if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
        }
        if (lane == 0)
        {
            if (at == 0x7fffffff) at = 0;
            *out = at;
            if (out2) *out2 = at;
        }
    }
}

AMQE_EXPORT int amqe_argmax(Engine* e, const float* x, int n, int* out, int* out2)
{
    launch(argmax_kernel, 1, 1024, 0, e->stream, x, n, out, out2);
    return ok(cudaGetLastError());
}

/// 1: every decode GEMV sums each row in the order one-row decode does, so a
/// row's values do not depend on how many rows run with it.
AMQE_EXPORT int amqe_set_row_invariant(Engine* e, int on)
{
    e->rowInvariant = on ? 1 : 0;
    return 0;
}

// ── 4-bit quantisation with calibration (GPTQ) ─────────────────────────────
//
// Rounding a weight to the nearest 4-bit value ignores what the rounding
// costs the layer's output. GPTQ (Frantar et al., 2022) rounds a row's columns
// one at a time and moves each column's error onto the columns still to come,
// weighted by how the layer's inputs correlate (H = XᵀX over calibration
// tokens), so the output error, not the weight error, is what stays small.
// Rows are independent: each thread takes one row of a 32-column block, with
// the block's piece of U (the upper Cholesky factor of H⁻¹) in shared memory;
// a cuBLAS rank-32 update then carries the block's errors to the rest of the
// row. The block scale of each group (16 for NVFP4, 32 for MXFP4) is chosen
// when the group is reached, from its error-corrected values, by trying the
// scales near the peak's and keeping the one with the least weighted error.

/// H += XᵀX over the rows of x (row-major [rows × k], stride ldx); H is k × k.
AMQE_EXPORT int amqe_gram(Engine* e, const float* x, int ldx, int rows, int k, float* h)
{
    if (rows <= 0) return 0;
    const float one = 1.f;
    auto st = cublasSgemm(e->blas, CUBLAS_OP_N, CUBLAS_OP_T, k, k, rows, &one, x, ldx, x, ldx, &one, h, k);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : -1000 - (int)st;
}

/// From a damped H (k × k, symmetric, device; overwritten), U: the upper
/// Cholesky factor of H⁻¹, into u (column-major; only the upper triangle is
/// meaningful). Returns 0, or a positive cuSOLVER info when a factorisation
/// fails (the caller damps more and tries again).
AMQE_EXPORT int amqe_gptq_prepare(Engine* e, float* h, int k, float* u)
{
    cusolverDnHandle_t s = nullptr;
    if (cusolverDnCreate(&s) != CUSOLVER_STATUS_SUCCESS) return -1;
    cusolverDnSetStream(s, e->stream);
    int lwork = 0, lwork2 = 0, info = 0, rc = 0;
    int* dInfo = nullptr;
    float* work = nullptr;
    const float one = 1.f;
    cusolverDnSpotrf_bufferSize(s, CUBLAS_FILL_MODE_LOWER, k, h, k, &lwork);
    cusolverDnSpotrf_bufferSize(s, CUBLAS_FILL_MODE_UPPER, k, u, k, &lwork2);
    if (lwork2 > lwork) lwork = lwork2;
    if (cudaMalloc(&dInfo, sizeof(int)) != cudaSuccess || cudaMalloc(&work, (size_t)lwork * 4 + 16) != cudaSuccess)
    {
        rc = -2;
    }
    // H = L Lᵀ; H⁻¹ by solving against the identity; H⁻¹ = Uᵀ U.
    if (rc == 0 && cusolverDnSpotrf(s, CUBLAS_FILL_MODE_LOWER, k, h, k, work, lwork, dInfo) != CUSOLVER_STATUS_SUCCESS) rc = -3;
    if (rc == 0)
    {
        cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, e->stream);
        cudaStreamSynchronize(e->stream);
        if (info != 0) rc = info;
    }
    if (rc == 0)
    {
        cudaMemsetAsync(u, 0, (size_t)k * k * 4, e->stream);
        cudaMemcpyAsync(work, &one, 4, cudaMemcpyHostToDevice, e->stream);
        if (cublasScopy(e->blas, k, work, 0, u, k + 1) != CUBLAS_STATUS_SUCCESS) rc = -4;
        cudaStreamSynchronize(e->stream);
    }
    if (rc == 0 && cusolverDnSpotrs(s, CUBLAS_FILL_MODE_LOWER, k, k, h, k, u, k, dInfo) != CUSOLVER_STATUS_SUCCESS) rc = -5;
    if (rc == 0 && cusolverDnSpotrf(s, CUBLAS_FILL_MODE_UPPER, k, u, k, work, lwork, dInfo) != CUSOLVER_STATUS_SUCCESS) rc = -6;
    if (rc == 0)
    {
        cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, e->stream);
        cudaStreamSynchronize(e->stream);
        if (info != 0) rc = info;
    }
    if (dInfo) cudaFree(dInfo);
    if (work) cudaFree(work);
    cusolverDnDestroy(s);
    if (rc == 0) rc = ok(cudaGetLastError());
    return rc;
}

/// The E2M1 code nearest to x (ties to the smaller magnitude, as the managed
/// encoder rounds).
__device__ __forceinline__ unsigned int e2m1_code(float x)
{
    float a = fabsf(x);
    unsigned int m = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u
                   : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.f ? 6u : 7u;
    return m | (x < 0.f ? 8u : 0u);
}

__device__ __forceinline__ float e2m1_of(unsigned int code)
{
    unsigned int m = code & 7u;
    float v = m < 4 ? 0.5f * m : m == 4 ? 2.f : m == 5 ? 3.f : m == 6 ? 4.f : 6.f;
    return (code & 8u) ? -v : v;
}

constexpr int GPTQ_B = 32;   // columns a block takes; both group sizes divide it

template <int G>
__global__ void gptq_block_kernel(float* __restrict__ w, int n, int k, const float* __restrict__ u, int c0,
    float global, int mx, unsigned char* __restrict__ packed, unsigned char* __restrict__ scales, float* __restrict__ err)
{
    __shared__ float su[GPTQ_B][GPTQ_B + 1];
    for (int t = threadIdx.x; t < GPTQ_B * GPTQ_B; t += blockDim.x)
    {
        int i = t / GPTQ_B, j = t % GPTQ_B;
        su[i][j] = j >= i ? u[(size_t)(c0 + i) + (size_t)(c0 + j) * k] : 0.f;
    }
    __syncthreads();
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float* row = w + (size_t)r * k + c0;
    float v[GPTQ_B];
    #pragma unroll
    for (int j = 0; j < GPTQ_B; j++) v[j] = row[j];
    unsigned int codes[GPTQ_B];
    #pragma unroll
    for (int g = 0; g < GPTQ_B / G; g++)
    {
        // The group's scale, from its values as corrected so far.
        float amax = 0.f;
        #pragma unroll
        for (int j = g * G; j < g * G + G; j++) amax = fmaxf(amax, fabsf(v[j]));
        float best = 0.f, bestErr = INFINITY;
        unsigned char bestByte = mx ? 127 : 0;   // a zero group: any scale serves
        if (amax > 0.f)
        {
            int first, last;   // the candidate scale codes
            if (mx)
            {
                int e0 = (int)ceilf(log2f(amax / 6.f));
                last = min(max(e0 + 127, 2), 254);
                first = last - 1;
            }
            else
            {
                int c = (int)__nv_cvt_float_to_fp8(amax / 6.f / global, __NV_SATFINITE, __NV_E4M3);
                first = max(c - 4, 1); last = min(c + 1, 0x7E);
            }
            for (int c = first; c <= last; c++)
            {
                float s = mx ? e8m0_value((unsigned char)c) : e4m3_value((unsigned char)c) * global;
                if (!(s > 0.f)) continue;
                float inv = 1.f / s, se = 0.f;
                #pragma unroll
                for (int j = g * G; j < g * G + G; j++)
                {
                    float d = v[j] - e2m1_of(e2m1_code(v[j] * inv)) * s;
                    se = fmaf(d * d, 1.f / (su[j][j] * su[j][j]), se);
                }
                if (se < bestErr) { bestErr = se; best = s; bestByte = (unsigned char)c; }
            }
        }
        scales[(size_t)r * (k / G) + (c0 + g * G) / G] = bestByte;
        float inv = best > 0.f ? 1.f / best : 0.f;
        #pragma unroll
        for (int j = g * G; j < g * G + G; j++)
        {
            unsigned int code = best > 0.f ? e2m1_code(v[j] * inv) : 0u;
            codes[j] = code;
            float e = (v[j] - e2m1_of(code) * best) / su[j][j];
            err[(size_t)r * GPTQ_B + j] = e;
            #pragma unroll
            for (int l = j + 1; l < GPTQ_B; l++) v[l] = fmaf(-e, su[j][l], v[l]);
        }
    }
    unsigned char* out = packed + ((size_t)r * k + c0) / 2;
    #pragma unroll
    for (int j = 0; j < GPTQ_B; j += 2) out[j / 2] = (unsigned char)(codes[j] | (codes[j + 1] << 4));
}

/// GPTQ over W (row-major [n × k] f32, device, overwritten) with U from
/// amqe_gptq_prepare: FP4 codes into packed ([n × k/2]) and block scales into
/// scales (E4M3 per 16 for NVFP4, with the tensor scale global; E8M0 per 32
/// for MXFP4). err is scratch of n × 32 floats. With U the identity this is
/// rounding to nearest, each group at its best scale.
AMQE_EXPORT int amqe_gptq(Engine* e, float* w, int n, int k, const float* u, int mx, float global,
    unsigned char* packed, unsigned char* scales, float* err)
{
    if (k % GPTQ_B != 0) return -1;
    const float one = 1.f, minus = -1.f;
    for (int c0 = 0; c0 < k; c0 += GPTQ_B)
    {
        int threads = 128, blocks = (n + threads - 1) / threads;
        if (mx) gptq_block_kernel<32><<<blocks, threads, 0, e->stream>>>(w, n, k, u, c0, global, 1, packed, scales, err);
        else gptq_block_kernel<16><<<blocks, threads, 0, e->stream>>>(w, n, k, u, c0, global, 0, packed, scales, err);
        int rest = k - c0 - GPTQ_B;
        if (rest > 0)
        {
            // Wᵀ[c0+B.., :] -= U[c0..c0+B, c0+B..]ᵀ · Errᵀ, column-major views of the row-major W and Err.
            auto st = cublasSgemm(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, rest, n, GPTQ_B, &minus,
                u + c0 + (size_t)(c0 + GPTQ_B) * k, k, err, GPTQ_B, &one, w + c0 + GPTQ_B, k);
            if (st != CUBLAS_STATUS_SUCCESS) return -1000 - (int)st;
        }
    }
    cudaStreamSynchronize(e->stream);
    return ok(cudaGetLastError());
}
