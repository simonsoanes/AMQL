// amql_engine.h — what the engine's sources share: the context, the launch
// helper and the weight readers.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

struct MoePrompt;

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
    int kvFp4 = 0;               // 1: KV caches hold NVFP4 rows (see kv4_row_bytes), else f32
    float* kvWiden = nullptr;    // prompt attention's f32 copy of an NVFP4 key/value span
    size_t kvWidenElems = 0;
    // Where one-row attention lands: when set, each decode-path attention adds the
    // share every SPLIT_KEYS-key window takes, summed over heads (amqe_set_attention_mass).
    float* massOut = nullptr;       // [massSplits]
    float* massScratch = nullptr;   // [heads x (2 x massSplits + 2)]
    int massSplits = 0;
    MoePrompt* moe = nullptr;       // routed prompts' scratch (amql_moe.cu)
    const int* reach = nullptr;     // how far ahead a prompt's rows see (amqe_set_reach)
};

void moe_prompt_free(Engine* e);

static int ok(cudaError_t e) { return e == cudaSuccess ? 0 : -(int)e; }

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

// BF16 and f32 weights gain from it. 4-bit weights lose (4 to 9% of a decode step
// on Qwen 3.5 0.8B and 4B, RTX 5090): the engine says which it holds
// (amqe_set_pdl), and AMQE_PDL=0 or 1 settles it.
extern int gPdlWanted;   // amql_engine.cu

static bool pdl_enabled()
{
    static int forced = -2;
    if (forced == -2)
    {
        const char* v = getenv("AMQE_PDL");
        forced = v ? (v[0] == '0' ? 0 : 1) : -1;
    }
    return (forced >= 0 ? forced : gPdlWanted) == 1;
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

static int gSms = 0;

static int sm_count()
{
    if (gSms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&gSms, cudaDevAttrMultiProcessorCount, dev);
        if (gSms <= 0) gSms = 64;
    }
    return gSms;
}
