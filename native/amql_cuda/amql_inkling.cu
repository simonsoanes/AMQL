// amql_inkling.cu — Inkling 2 looped (inkling2_loop) on the GPU.
//
// Built into amql_cuda.dll beside the engine (amql_engine.cu) and run on an
// engine's stream: a prompt chunk's projections and experts are the engine's
// (amqe_linear, amqe_moe_experts_paired: BF16 tensor cores), and what is
// Inkling's own (its attention, the short convolutions, the loop and its
// messages, a step's GEMVs) is here. It mirrors src/Amql.Inference/Inkling2Model.cs,
// the managed runtime transcribed from transformers, kernel for step:
//
//   h = embed_norm(embed[token]);  blocks = []
//   layers [0, start), then the core [start, end) three times (passes 1 and 2
//   restart from the prelude's state and read messages from the pass before),
//   then [end, n); logits = softcap(unembed . norm(mix(output_query, blocks, h)))
//
// Every layer visit has its own cache slot (keys, values and the short
// convolutions' histories): 24 physical layers and 2 x 12 for the core's extra
// passes. A sliding slot is a ring of `window` rows; a full slot grows with the
// sequence and holds f32, f16 or NVFP4 rows. The experts stay as stored
// (NVFP4 with its E4M3 block scales and per-expert factor, MXFP8 with E8M0
// exponents, or plain f32/BF16) and only the chosen ones are read.
//
// A step reads its token and position from device memory, so the whole step is
// captured once as a CUDA graph and replayed for every token. A prompt runs in
// chunks of up to kChunk tokens with a token dimension: the projections and the
// experts (grouped, each chosen expert decoded once per chunk) as the engine's
// tensor-core GEMMs, or, with prompts kept at f32 (ink_set_prefill_f32), the
// projections as f32 SGEMMs and the experts as the f32 kernels here; the
// convolutions and the cache rows advance token by token inside their kernels.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <initializer_list>

#if defined(_WIN32)
  #define INK_EXPORT extern "C" __declspec(dllexport)
#else
  #define INK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

static int ok(cudaError_t e) { return e == cudaSuccess ? 0 : -(int)e; }
#define TRY(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) return -(int)_e; } while (0)

// -- the engine (amql_engine.cu, amql_moe.cu), in the same library --

struct Engine;
extern "C" int amqe_create(Engine** out);
extern "C" void amqe_destroy(Engine* e);
extern "C" cudaStream_t amqe_stream(Engine* e);
extern "C" int amqe_set_prefill_f32(Engine* e, int f32);
extern "C" int amqe_linear(Engine* e, const void* w, int wType, int n, int k,
    const float* x, int ldx, int rows, float* y, int ldy,
    const float* bias, const float* residual, int ldr, float resScale, int act, int glu, float* raw,
    const float* normW, float normOff, const float* normBias, float normEps, int normKind,
    const unsigned char* wScales, const float* rowScale);
extern "C" int amqe_moe_experts_paired_step(Engine* e,
    const void* gateUp, int guType, const unsigned char* guScales, const float* guRowScale,
    const void* down, int dType, const unsigned char* dScales, const float* dRowScale,
    int experts, int hidden, int inter, int topK, float gateCap, float upCap,
    const float* x, int ldx, int rows, const int* idx, float* glu, float* slots);
extern "C" int amqe_moe_experts_paired(Engine* e,
    const void* gateUp, int guType, const unsigned char* guScales, const float* guRowScale,
    const void* down, int dType, const unsigned char* dScales, const float* dRowScale,
    int experts, int hidden, int inter, int topK, float gateCap, float upCap,
    const float* x, int ldx, int rows, const int* idx, int* order, int* groups, float* slots);

// ── types shared with the managed side ───────────────────────────────────────

enum { W_F32 = 0, W_BF16 = 1, W_NVFP4 = 2, W_MXFP8 = 3 };
enum { KV_F32 = 0, KV_F16 = 1, KV_FP4 = 2 };

// Layer matrices (ink_set_mat).
enum { M_WQ = 0, M_WK, M_WV, M_WR, M_WO, M_GATE, M_SINK, M_KPROJ, M_VPROJ, M_ROUTER,
       M_CTRL = 10 /* + 3 g + {down, forget, gain}, g in q, k, v, mlp */, M_COUNT = 22 };
// Layer vectors (ink_set_vec), always f32.
enum { V_QSCALE = 0, V_RELSCALE, V_RELPROJ, V_KCONV, V_VCONV, V_KSTATIC, V_VSTATIC, V_ATTNNORM, V_QNORM,
       V_KNORM, V_VNORM, V_MLPNORM, V_ATTNCONV, V_MLPCONV, V_ROUTERBIAS, V_ATTNQUERY, V_MLPQUERY, V_COUNT };
// Model-wide tensors (ink_set_global): two matrices, then f32 vectors.
enum { G_EMBED = 0, G_UNEMBED, G_EMBEDNORM, G_FINALNORM, G_OUTPUTQUERY, G_MESSAGES, G_COUNT };

struct InkConfig
{
    int d, layers, experts, topK, width, convK, block, loopStart, loopEnd, passes, rank;
    int paddedVocab, vocab, relDim, relExtent, window, logFloor;
    float alpha, qkClip, relClip, eps, gateCap, upCap, finalCap, routeScale;
};

struct Mat { const void* w = nullptr; int type = 0; int rows = 0; int cols = 0; };
struct Experts { const void* w = nullptr; const unsigned char* scales = nullptr; const float* factor = nullptr;
                 int type = 0; int rows = 0; int cols = 0;
                 const float* rowScale = nullptr; };   // NVFP4: the factor for every row, [experts x rows], for the engine

struct Layer
{
    int full = 0, heads = 0, kvHeads = 0, headDim = 0, extent = 0, loop = 0, queries = 0;
    Mat m[M_COUNT];
    const float* v[V_COUNT] = {};
    Experts w13, w2;
};

// ── device helpers ───────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum, every thread gets it; `sh` holds 32 floats.
__device__ float block_sum(float v, float* sh)
{
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = threadIdx.x < warps ? sh[threadIdx.x] : 0.f;
    if (warp == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float bf16_bits(unsigned short b) { return __uint_as_float((unsigned int)b << 16); }

__device__ __forceinline__ float ld(const void* w, int type, long long i)
{
    return type == W_BF16 ? bf16_bits(((const unsigned short*)w)[i]) : ((const float*)w)[i];
}

__device__ __forceinline__ float e4m3(unsigned char b)
{
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&h));
}

__device__ __forceinline__ float e8m0(unsigned char b) { return b == 255 ? __int_as_float(0x7fc00000) : exp2f((float)b - 127.f); }

// Two E2M1 codes of a byte (low nibble first), by the conversion instruction:
// a table in constant memory serialises when the lanes' codes differ.
__device__ __forceinline__ float2 e2m1x2(unsigned int pair)
{
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(pair & 0xffu), __NV_E2M1);
    return __half22float2(*reinterpret_cast<__half2*>(&raw));
}


// One row of a matrix dotted with x, by a whole warp.
__device__ float row_dot(const void* w, int type, long long row, int cols, const float* x)
{
    int lane = threadIdx.x & 31;
    float s = 0.f;
    if (type == W_BF16)
    {
        const unsigned short* r = (const unsigned short*)w + row * cols;
        if ((cols & 7) == 0)
        {
            const uint4* r4 = (const uint4*)r;
            for (int c = lane; c < cols / 8; c += 32)
            {
                uint4 q = __ldg(r4 + c);
                const float* xx = x + c * 8;
                s += __uint_as_float(q.x << 16) * xx[0] + __uint_as_float(q.x & 0xffff0000u) * xx[1]
                   + __uint_as_float(q.y << 16) * xx[2] + __uint_as_float(q.y & 0xffff0000u) * xx[3]
                   + __uint_as_float(q.z << 16) * xx[4] + __uint_as_float(q.z & 0xffff0000u) * xx[5]
                   + __uint_as_float(q.w << 16) * xx[6] + __uint_as_float(q.w & 0xffff0000u) * xx[7];
            }
        }
        else
        {
            for (int c = lane; c < cols; c += 32) s += bf16_bits(r[c]) * x[c];
        }
    }
    else
    {
        const float* r = (const float*)w + row * cols;
        for (int c = lane; c < cols; c += 32) s += r[c] * x[c];
    }
    return warp_sum(s);
}

// One row of an expert stack dotted with x, by a whole warp. NVFP4: the
// per-expert factor multiplies the dot product (element x block scale is
// exact in f32, so only the order of the sum differs from decoding first).
__device__ float expert_dot(const Experts& e, int expert, int row, const float* x)
{
    int lane = threadIdx.x & 31, cols = e.cols;
    long long r = (long long)expert * e.rows + row;
    float s = 0.f;
    if (e.type == W_NVFP4)
    {
        const unsigned char* el = (const unsigned char*)e.w + r * (cols / 2);
        const unsigned char* sc = e.scales + r * (cols / 16);
        if ((cols & 31) == 0)
        {
            // 32 elements (16 bytes, two scale blocks) a lane a turn.
            for (int b = lane; b < cols / 32; b += 32)
            {
                uint4 q = __ldg((const uint4*)(el + b * 16));
                unsigned short sp = __ldg((const unsigned short*)(sc + b * 2));
                const float4* xx = (const float4*)(x + b * 32);
                unsigned int words[4] = { q.x, q.y, q.z, q.w };
                float t0 = 0.f, t1 = 0.f;
                #pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    float4 xa = xx[2 * k], xb = xx[2 * k + 1];
                    float2 v0 = e2m1x2(words[k]), v1 = e2m1x2(words[k] >> 8), v2 = e2m1x2(words[k] >> 16), v3 = e2m1x2(words[k] >> 24);
                    float t = v0.x * xa.x + v0.y * xa.y + v1.x * xa.z + v1.y * xa.w + v2.x * xb.x + v2.y * xb.y + v3.x * xb.z + v3.y * xb.w;
                    if (k < 2) t0 += t; else t1 += t;
                }
                s += e4m3(sp & 255) * t0 + e4m3(sp >> 8) * t1;
            }
            return warp_sum(s) * e.factor[expert];
        }
        for (int b = lane; b < cols / 16; b += 32)
        {
            uint2 q = __ldg((const uint2*)(el + b * 8));
            float scale = e4m3(__ldg(sc + b));
            const float* xx = x + b * 16;
            float t = 0.f;
            #pragma unroll
            for (int k = 0; k < 8; k++)
            {
                float2 v = e2m1x2((k < 4 ? q.x : q.y) >> (8 * (k & 3)));
                t += v.x * xx[2 * k] + v.y * xx[2 * k + 1];
            }
            s += scale * t;
        }
        return warp_sum(s) * e.factor[expert];
    }
    if (e.type == W_MXFP8)
    {
        const unsigned char* el = (const unsigned char*)e.w + r * cols;
        const unsigned char* sc = e.scales + r * (cols / 32);
        // 16 elements (16 bytes) a lane a turn, so a warp reads 512 contiguous bytes.
        for (int b = lane; b < cols / 16; b += 32)
        {
            uint4 q = __ldg((const uint4*)(el + b * 16));
            unsigned int words[4] = { q.x, q.y, q.z, q.w };
            const float4* xx = (const float4*)(x + b * 16);
            float t = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; k++)
            {
                unsigned int wd = words[k];
                __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wd & 0xffffu), __NV_E4M3);
                __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wd >> 16), __NV_E4M3);
                float2 a = __half22float2(*reinterpret_cast<__half2*>(&lo));
                float2 c = __half22float2(*reinterpret_cast<__half2*>(&hi));
                float4 xv = xx[k];
                t += a.x * xv.x + a.y * xv.y + c.x * xv.z + c.y * xv.w;
            }
            s += e8m0(__ldg(sc + b / 2)) * t;
        }
        return warp_sum(s);
    }
    return row_dot(e.w, e.type, r, cols, x);
}

// Two consecutive rows of an expert stack (gate 2j and up 2j + 1) dotted with
// x by one warp: they are contiguous, so the warp streams both as one span and
// keeps twice the loads in flight.
__device__ void expert_dot_pair(const Experts& e, int expert, int row, const float* x, float& g, float& u)
{
    int lane = threadIdx.x & 31, cols = e.cols;
    long long r = (long long)expert * e.rows + row;
    float sg = 0.f, su = 0.f;
    if (e.type == W_NVFP4 && (cols & 31) == 0)
    {
        int n = cols / 32;
        const unsigned char* el = (const unsigned char*)e.w + r * (cols / 2);
        const unsigned char* sc = e.scales + r * (cols / 16);
        for (int b = lane; b < 2 * n; b += 32)
        {
            int bb = b < n ? b : b - n;
            uint4 q = __ldg((const uint4*)(el + b * 16));
            unsigned short sp = __ldg((const unsigned short*)(sc + b * 2));
            const float4* xx = (const float4*)(x + bb * 32);
            unsigned int words[4] = { q.x, q.y, q.z, q.w };
            float t0 = 0.f, t1 = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; k++)
            {
                float4 xa = xx[2 * k], xb = xx[2 * k + 1];
                float2 v0 = e2m1x2(words[k]), v1 = e2m1x2(words[k] >> 8), v2 = e2m1x2(words[k] >> 16), v3 = e2m1x2(words[k] >> 24);
                float t = v0.x * xa.x + v0.y * xa.y + v1.x * xa.z + v1.y * xa.w + v2.x * xb.x + v2.y * xb.y + v3.x * xb.z + v3.y * xb.w;
                if (k < 2) t0 += t; else t1 += t;
            }
            float v = e4m3(sp & 255) * t0 + e4m3(sp >> 8) * t1;
            if (b < n) sg += v; else su += v;
        }
        g = warp_sum(sg) * e.factor[expert];
        u = warp_sum(su) * e.factor[expert];
        return;
    }
    if (e.type == W_MXFP8)
    {
        int n = cols / 16;
        const unsigned char* el = (const unsigned char*)e.w + r * cols;
        const unsigned char* sc = e.scales + r * (cols / 32);
        for (int b = lane; b < 2 * n; b += 32)
        {
            int bb = b < n ? b : b - n;
            uint4 q = __ldg((const uint4*)(el + b * 16));
            unsigned int words[4] = { q.x, q.y, q.z, q.w };
            const float4* xx = (const float4*)(x + bb * 16);
            float t = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; k++)
            {
                __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] & 0xffffu), __NV_E4M3);
                __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] >> 16), __NV_E4M3);
                float2 a = __half22float2(*reinterpret_cast<__half2*>(&lo));
                float2 c = __half22float2(*reinterpret_cast<__half2*>(&hi));
                float4 xv = xx[k];
                t += a.x * xv.x + a.y * xv.y + c.x * xv.z + c.y * xv.w;
            }
            float v = e8m0(__ldg(sc + b / 2)) * t;
            if (b < n) sg += v; else su += v;
        }
        g = warp_sum(sg);
        u = warp_sum(su);
        return;
    }
    g = expert_dot(e, expert, row, x);
    u = expert_dot(e, expert, row + 1, x);
}

// One small matrix (a router, a message gate's projection): a block a row, so
// a few dozen rows still spread over the device.
__global__ void k_gemv_rows(Mat m, const float* __restrict__ x, float* __restrict__ y)
{
    __shared__ float sh[32];
    int row = blockIdx.x, cols = m.cols;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += ld(m.w, m.type, (long long)row * cols + c) * x[c];
    s = block_sum(s, sh);
    if (threadIdx.x == 0) y[row] = s;
}

// ── kernels ──────────────────────────────────────────────────────────────────

// y[r] = W[r] . x, a warp a row (rows of W from `first`).
__global__ void k_gemv(Mat m, const float* __restrict__ x, float* __restrict__ y, int rows)
{
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    float s = row_dot(m.w, m.type, row, m.cols, x);
    if ((threadIdx.x & 31) == 0) y[row] = s;
}

struct Gemvs { Mat m[4]; const float* x[4]; float* y[4]; int first[5]; int n; };

__global__ void k_gemvs(Gemvs g)
{
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= g.first[g.n]) return;
    int i = 0;
    while (row >= g.first[i + 1]) i++;
    int r = row - g.first[i];
    float s = row_dot(g.m[i].w, g.m[i].type, r, g.m[i].cols, g.x[i]);
    if ((threadIdx.x & 31) == 0) g.y[i][r] = s;
}

// Several matrices times T rows each (row t of x[i] is x[i] + t * cols): a
// warp a weight row, 16 tokens at a time, so a weight is read once per 16.
struct Gemms { Mat m[4]; const float* x[4]; float* y[4]; int first[5]; int n; int T; };

__device__ __forceinline__ void w8(const Mat& m, long long idx, float* o)
{
    if (m.type == W_BF16)
    {
        uint4 v = __ldg((const uint4*)((const unsigned short*)m.w + idx));
        o[0] = __uint_as_float(v.x << 16); o[1] = __uint_as_float(v.x & 0xffff0000u);
        o[2] = __uint_as_float(v.y << 16); o[3] = __uint_as_float(v.y & 0xffff0000u);
        o[4] = __uint_as_float(v.z << 16); o[5] = __uint_as_float(v.z & 0xffff0000u);
        o[6] = __uint_as_float(v.w << 16); o[7] = __uint_as_float(v.w & 0xffff0000u);
    }
    else
    {
        float4 a = __ldg((const float4*)((const float*)m.w + idx)), b = __ldg((const float4*)((const float*)m.w + idx + 4));
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
    }
}

__global__ void k_gemms(Gemms g)
{
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= g.first[g.n]) return;
    int i = 0;
    while (row >= g.first[i + 1]) i++;
    int r = row - g.first[i];
    const Mat m = g.m[i];
    const float* X = g.x[i];
    float* Y = g.y[i];
    int cols = m.cols;
    for (int t0 = 0; t0 < g.T; t0 += 16)
    {
        int nt = min(16, g.T - t0);
        float acc[16];
        #pragma unroll
        for (int tt = 0; tt < 16; tt++) acc[tt] = 0.f;
        if ((cols & 7) == 0)
        {
            for (int c = lane; c < cols / 8; c += 32)
            {
                float w[8];
                w8(m, (long long)r * cols + c * 8, w);
                #pragma unroll
                for (int tt = 0; tt < 16; tt++)
                {
                    if (tt < nt)
                    {
                        const float4* xx = (const float4*)(X + (long long)(t0 + tt) * cols + c * 8);
                        float4 a = xx[0], b = xx[1];
                        acc[tt] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
                    }
                }
            }
        }
        else
        {
            for (int c = lane; c < cols; c += 32)
            {
                float w = ld(m.w, m.type, (long long)r * cols + c);
                #pragma unroll
                for (int tt = 0; tt < 16; tt++) if (tt < nt) acc[tt] += w * X[(long long)(t0 + tt) * cols + c];
            }
        }
        #pragma unroll
        for (int tt = 0; tt < 16; tt++)
        {
            float sum = warp_sum(acc[tt]);
            if (lane == 0 && tt < nt) Y[(long long)(t0 + tt) * m.rows + r] = sum;
        }
    }
}

// The (token, choice) pairs of a chunk grouped by expert: offsets[e] ..
// offsets[e + 1] in `order` are expert e's pairs (pair = token * topK + k).
__global__ void k_group(const int* __restrict__ idx, int pairs, int experts, int* __restrict__ offsets, int* __restrict__ order)
{
    __shared__ int cnt[512], cur[512];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) atomicAdd(&cnt[idx[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int run = 0;
        for (int e = 0; e < experts; e++) { offsets[e] = run; cur[e] = run; run += cnt[e]; }
        offsets[experts] = run;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) order[atomicAdd(&cur[idx[p]], 1)] = p;
}

// 32 elements of one expert row (columns 32b .. 32b + 31) as f32, without the
// NVFP4 per-expert factor (it multiplies the dot product).
__device__ __forceinline__ void expert_chunk(const Experts& e, int expert, int row, int b, float* w)
{
    long long r = (long long)expert * e.rows + row;
    int cols = e.cols;
    if (e.type == W_NVFP4)
    {
        uint4 q = __ldg((const uint4*)((const unsigned char*)e.w + r * (cols / 2) + b * 16));
        unsigned short sp = __ldg((const unsigned short*)(e.scales + r * (cols / 16) + b * 2));
        float s0 = e4m3(sp & 255), s1 = e4m3(sp >> 8);
        unsigned int words[4] = { q.x, q.y, q.z, q.w };
        #pragma unroll
        for (int k = 0; k < 4; k++)
        {
            float s = k < 2 ? s0 : s1;
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                float2 v = e2m1x2(words[k] >> (8 * j));
                w[8 * k + 2 * j] = v.x * s;
                w[8 * k + 2 * j + 1] = v.y * s;
            }
        }
        return;
    }
    if (e.type == W_MXFP8)
    {
        const uint4* q4 = (const uint4*)((const unsigned char*)e.w + r * cols + b * 32);
        uint4 a = __ldg(q4), c = __ldg(q4 + 1);
        float s = e8m0(__ldg(e.scales + r * (cols / 32) + b));
        unsigned int words[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] & 0xffffu), __NV_E4M3);
            __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] >> 16), __NV_E4M3);
            float2 x0 = __half22float2(*reinterpret_cast<__half2*>(&lo)), x1 = __half22float2(*reinterpret_cast<__half2*>(&hi));
            w[4 * k] = x0.x * s; w[4 * k + 1] = x0.y * s; w[4 * k + 2] = x1.x * s; w[4 * k + 3] = x1.y * s;
        }
        return;
    }
    #pragma unroll
    for (int k = 0; k < 32; k++) w[k] = ld(e.w, e.type, r * cols + b * 32 + k);
}

__device__ __forceinline__ float dot32(const float* w, const float* x)
{
    const float4* x4 = (const float4*)x;
    float t = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; k++)
    {
        float4 v = x4[k];
        t += w[4 * k] * v.x + w[4 * k + 1] * v.y + w[4 * k + 2] * v.z + w[4 * k + 3] * v.w;
    }
    return t;
}

// Eight columns (8g .. 8g + 7) of one expert row as f32, without the NVFP4
// per-expert factor (it multiplies the dot product).
__device__ __forceinline__ void expert8(const Experts& e, int expert, int row, int g, float* w)
{
    long long r = (long long)expert * e.rows + row;
    int cols = e.cols;
    if (e.type == W_NVFP4)
    {
        unsigned int q = __ldg((const unsigned int*)((const unsigned char*)e.w + r * (cols / 2) + g * 4));
        float s = e4m3(__ldg(e.scales + r * (cols / 16) + g / 2));
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float2 v = e2m1x2(q >> (8 * j));
            w[2 * j] = v.x * s;
            w[2 * j + 1] = v.y * s;
        }
        return;
    }
    if (e.type == W_MXFP8)
    {
        uint2 q = __ldg((const uint2*)((const unsigned char*)e.w + r * cols + g * 8));
        float s = e8m0(__ldg(e.scales + r * (cols / 32) + g / 4));
        unsigned int words[2] = { q.x, q.y };
        #pragma unroll
        for (int k = 0; k < 2; k++)
        {
            __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] & 0xffffu), __NV_E4M3);
            __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] >> 16), __NV_E4M3);
            float2 x0 = __half22float2(*reinterpret_cast<__half2*>(&lo)), x1 = __half22float2(*reinterpret_cast<__half2*>(&hi));
            w[4 * k] = x0.x * s; w[4 * k + 1] = x0.y * s; w[4 * k + 2] = x1.x * s; w[4 * k + 3] = x1.y * s;
        }
        return;
    }
    #pragma unroll
    for (int k = 0; k < 8; k++) w[k] = ld(e.w, e.type, r * cols + g * 8 + k);
}

// Input rows a warp holds at once in the prompt expert kernels.
static constexpr int kP = 4;

// Eight weight rows (rows0 .. rows0 + 7 of `expert`) against up to kP
// input rows (x + in[q] * cols), a warp over 8-column granules: each input
// granule loaded serves all eight rows, each weight granule all kP inputs.
// out[r][q] holds the warp's sums (every lane), times the NVFP4 factor.
__device__ __forceinline__ void expert_block(const Experts& e, int expert, int row0, const float* __restrict__ x,
                                             const long long* in, int np, float out[8][kP])
{
    int lane = threadIdx.x & 31, cols = e.cols;
    float acc[8][kP];
    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int q = 0; q < kP; q++) acc[r][q] = 0.f;
    #pragma unroll 1
    for (int g = lane; g < cols / 8; g += 32)
    {
        float4 xa[kP], xb[kP];
        #pragma unroll
        for (int q = 0; q < kP; q++)
        {
            const float4* xx = (const float4*)(x + in[q] * cols + g * 8);   // in[q] is a valid row for q >= np too
            xa[q] = __ldg(xx);
            xb[q] = __ldg(xx + 1);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++)
        {
            float w[8];
            expert8(e, expert, row0 + r, g, w);
            #pragma unroll
            for (int q = 0; q < kP; q++)
                acc[r][q] += w[0] * xa[q].x + w[1] * xa[q].y + w[2] * xa[q].z + w[3] * xa[q].w
                           + w[4] * xb[q].x + w[5] * xb[q].y + w[6] * xb[q].z + w[7] * xb[q].w;
        }
    }
    float factor = e.type == W_NVFP4 ? e.factor[expert] : 1.f;
    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int q = 0; q < kP; q++) out[r][q] = warp_sum(acc[r][q]) * factor;
}

// A chunk's experts, each expert's rows read once for all its pairs. Grid
// (experts, width / 32): a warp four gate/up outputs (rows 2j .. 2j + 7).
__global__ void __launch_bounds__(256) k_expert_up_g(Experts w13, const float* __restrict__ x, const int* __restrict__ offsets,
                                                     const int* __restrict__ order, int topK, float* __restrict__ act, int width,
                                                     float gateCap, float upCap)
{
    int e = blockIdx.x, lane = threadIdx.x & 31;
    int j0 = (blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 4;
    int first = offsets[e], count = offsets[e + 1] - first;
    if (count == 0 || j0 >= width) return;
    for (int p0 = 0; p0 < count; p0 += kP)
    {
        int np = min(kP, count - p0);
        long long in[kP];
        #pragma unroll
        for (int q = 0; q < kP; q++) in[q] = q < np ? order[first + p0 + q] / topK : 0;
        float out[8][kP];
        expert_block(w13, e, 2 * j0, x, in, np, out);
        if (lane == 0)
        {
            #pragma unroll
            for (int jj = 0; jj < 4; jj++)
            {
                if (j0 + jj >= width) break;
                #pragma unroll
                for (int q = 0; q < kP; q++)
                {
                    if (q >= np) break;
                    float g = out[2 * jj][q], u = out[2 * jj + 1][q];
                    act[(long long)order[first + p0 + q] * width + j0 + jj] =
                        tanhf(g / gateCap) * gateCap / (1.f + expf(-g)) * (tanhf(u / upCap) * upCap);
                }
            }
        }
    }
}

// Grid (experts, d / 64): a warp eight output rows, every pair of the expert.
__global__ void __launch_bounds__(256) k_expert_down_g(Experts w2, const float* __restrict__ act, const int* __restrict__ offsets,
                                                       const int* __restrict__ order, const float* __restrict__ wt,
                                                       float* __restrict__ part, int d, int width)
{
    int e = blockIdx.x, lane = threadIdx.x & 31;
    int i0 = (blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 8;
    int first = offsets[e], count = offsets[e + 1] - first;
    if (count == 0 || i0 >= d) return;
    for (int p0 = 0; p0 < count; p0 += kP)
    {
        int np = min(kP, count - p0);
        long long in[kP];
        #pragma unroll
        for (int q = 0; q < kP; q++) in[q] = q < np ? order[first + p0 + q] : 0;
        float out[8][kP];
        expert_block(w2, e, i0, act, in, np, out);
        if (lane == 0)
        {
            #pragma unroll
            for (int r = 0; r < 8; r++)
            {
                if (i0 + r >= d) break;
                #pragma unroll
                for (int q = 0; q < kP; q++)
                {
                    if (q >= np) break;
                    part[in[q] * d + i0 + r] = wt[in[q]] * out[r][q];
                }
            }
        }
    }
}

__global__ void k_softcap(float* x, int n, float cap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}

// out = x / rms(x) (times w when given).
// Up to three norms at once: grid (T, n).
struct Norms { const float* x[3]; const float* w[3]; float* out[3]; };

__global__ void k_rmsnorm(Norms a, int d, float eps)
{
    __shared__ float sh[32];
    const float* x = a.x[blockIdx.y] + (long long)blockIdx.x * d;
    const float* w = a.w[blockIdx.y];
    float* out = a.out[blockIdx.y] + (long long)blockIdx.x * d;
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += x[i] * x[i];
    float inv = rsqrtf(block_sum(s, sh) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) out[i] = x[i] * inv * (w ? w[i] : 1.f);
}

// h = embed_norm(embed[token]).
__global__ void k_embed(Mat embed, const int* __restrict__ token, const float* __restrict__ norm, float* __restrict__ out, float eps)
{
    __shared__ float sh[32];
    long long row = token[blockIdx.x];
    int d = embed.cols;
    out += (long long)blockIdx.x * d;
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
        float v = ld(embed.w, embed.type, row * d + i);
        out[i] = v;
        s += v * v;
    }
    float inv = rsqrtf(block_sum(s, sh) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) out[i] = out[i] * inv * norm[i];
}

// Attention over depth (Inkling2Loop{Block,Message}AttentionResidual): each
// state RMS-normed without a gain, weighted by softmax(query . state / sqrt d).
struct States { const float* s[10]; int n; };
// The norm that follows a mix, done in the same block: query i's output is
// also written RMS-normed with w[i] (null: no gain) into out[i] + row * d.
struct MixNorm { const float* w[4]; float* out[4]; float eps; };

__global__ void k_mix(States st, const float* __restrict__ queries, float* __restrict__ out, int d, MixNorm mn)
{
    __shared__ float sh[32];
    __shared__ float inv[10], w[10];
    const float* q = queries + (long long)blockIdx.x * d;
    long long ro = (long long)blockIdx.y * d;
    // Every state's sum of squares and dot with the query in one pass.
    __shared__ float part[10][2][32];
    float ss[10], qs[10];
    #pragma unroll
    for (int i = 0; i < 10; i++) { ss[i] = 0.f; qs[i] = 0.f; }
    for (int j = threadIdx.x; j < d; j += blockDim.x)
    {
        float qj = q[j];
        #pragma unroll
        for (int i = 0; i < 10; i++) if (i < st.n) { float v = st.s[i][ro + j]; ss[i] += v * v; qs[i] += qj * v; }
    }
    int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < 10; i++)
    {
        float a = warp_sum(ss[i]), b = warp_sum(qs[i]);
        if (lane == 0 && i < st.n) { part[i][0][wp] = a; part[i][1][wp] = b; }
    }
    __syncthreads();
    if (threadIdx.x < st.n)
    {
        int i = threadIdx.x;
        float a = 0.f, b = 0.f;
        for (int k = 0; k < (int)(blockDim.x >> 5); k++) { a += part[i][0][k]; b += part[i][1][k]; }
        inv[i] = rsqrtf(a / d + 1e-6f);
        w[i] = b * inv[i] * rsqrtf((float)d);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float m = -INFINITY, sum = 0.f;
        for (int i = 0; i < st.n; i++) m = fmaxf(m, w[i]);
        for (int i = 0; i < st.n; i++) { w[i] = expf(w[i] - m); sum += w[i]; }
        for (int i = 0; i < st.n; i++) w[i] = w[i] / sum * inv[i];
    }
    __syncthreads();
    float* o = out + ((long long)blockIdx.x * gridDim.y + blockIdx.y) * d;
    float sq = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
    {
        float acc = 0.f;
        for (int i = 0; i < st.n; i++) acc += w[i] * st.s[i][ro + j];
        o[j] = acc;
        sq += acc * acc;
    }
    float* no = mn.out[blockIdx.x];
    if (no == nullptr) return;
    // Each thread reads back only what it wrote, so no barrier is needed for o.
    float rinv = rsqrtf(block_sum(sq, sh) / d + mn.eps);
    const float* nw = mn.w[blockIdx.x];
    no += ro;
    for (int j = threadIdx.x; j < d; j += blockDim.x) no[j] = o[j] * rinv * (nw ? nw[j] : 1.f);
}

// The same with each thread holding one float4 of every state in registers
// (d % 4 == 0, d / 4 <= 1024 threads): one round of loads, the weighting from
// registers, the softmax in one warp.
__global__ void __launch_bounds__(1024) k_mix4(States st, const float* __restrict__ queries, float* __restrict__ out, int d, MixNorm mn)
{
    __shared__ float sh[32];
    __shared__ float part[10][2][32];
    __shared__ float w[10];
    const int n = st.n, d4 = d >> 2, j = threadIdx.x;
    const bool live = j < d4;
    long long ro = (long long)blockIdx.y * d;
    float4 q = live ? reinterpret_cast<const float4*>(queries + (long long)blockIdx.x * d)[j] : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v[10];
    #pragma unroll
    for (int i = 0; i < 10; i++)
        v[i] = live && i < n ? reinterpret_cast<const float4*>(st.s[i] + ro)[j] : make_float4(0.f, 0.f, 0.f, 0.f);
    const int lane = j & 31, wp = j >> 5, warps = blockDim.x >> 5;
    #pragma unroll
    for (int i = 0; i < 10; i++)
    {
        if (i >= n) break;
        float a = warp_sum(v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w);
        float b = warp_sum(q.x * v[i].x + q.y * v[i].y + q.z * v[i].z + q.w * v[i].w);
        if (lane == 0) { part[i][0][wp] = a; part[i][1][wp] = b; }
    }
    __syncthreads();
    if (wp == 0)
    {
        float a = 0.f, b = 0.f;
        if (lane < n)
            for (int k = 0; k < warps; k++) { a += part[lane][0][k]; b += part[lane][1][k]; }
        float inv = rsqrtf(a / d + 1e-6f);
        float logit = lane < n ? b * inv * rsqrtf((float)d) : -INFINITY;
        float m = warp_max(logit);
        float e = lane < n ? expf(logit - m) : 0.f;
        float sum = warp_sum(e);
        if (lane < n) w[lane] = e / sum * inv;
    }
    __syncthreads();
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int i = 0; i < 10; i++)
    {
        if (i >= n) break;
        const float wi = w[i];
        acc.x += wi * v[i].x; acc.y += wi * v[i].y; acc.z += wi * v[i].z; acc.w += wi * v[i].w;
    }
    float* o = out + ((long long)blockIdx.x * gridDim.y + blockIdx.y) * d;
    if (live) reinterpret_cast<float4*>(o)[j] = acc;
    float* no = mn.out[blockIdx.x];
    if (no == nullptr) return;
    float rinv = rsqrtf(block_sum(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w, sh) / d + mn.eps);
    if (!live) return;
    const float* nw = mn.w[blockIdx.x];
    float4 g = nw ? reinterpret_cast<const float4*>(nw)[j] : make_float4(1.f, 1.f, 1.f, 1.f);
    reinterpret_cast<float4*>(no + ro)[j] = make_float4(acc.x * rinv * g.x, acc.y * rinv * g.y, acc.z * rinv * g.z, acc.w * rinv * g.w);
}

// The message gate's input: [h / rms(h); m / rms(m)].
// Up to three gates at once (q, k and v take theirs side by side): grid (T, n).
struct GateCats { const float* h[3]; const float* m[3]; float* cat[3]; };

__global__ void k_gate_cat(GateCats args, int d, float eps)
{
    __shared__ float sh[32];
    const int g = blockIdx.y;
    const float* h = args.h[g] + (long long)blockIdx.x * d;
    const float* m = args.m[g] + (long long)blockIdx.x * d;
    float* cat = args.cat[g] + (long long)blockIdx.x * 2 * d;
    float a = 0.f, b = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) { a += h[i] * h[i]; b += m[i] * m[i]; }
    float ia = rsqrtf(block_sum(a, sh) / d + eps);
    float ib = rsqrtf(block_sum(b, sh) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) { cat[i] = h[i] * ia; cat[d + i] = m[i] * ib; }
}

// (1 - tanh(F g)) h + tanh(G g) m, up to three gates at once: grid (d / 8, T, n).
struct GateApplies { Mat forget[3], gain[3]; const float* g[3]; const float* h[3]; const float* m[3]; float* out[3]; };

__global__ void k_gate_apply(GateApplies args, int d, int rank)
{
    int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= d) return;
    const int z = blockIdx.z;
    const Mat forget = args.forget[z], gain = args.gain[z];
    long long ro = (long long)blockIdx.y * d;
    const float* g = args.g[z] + (long long)blockIdx.y * rank;
    const float* h = args.h[z] + ro;
    const float* m = args.m[z] + ro;
    float* out = args.out[z] + ro;
    float f = row_dot(forget.w, forget.type, i, rank, g);
    float a = row_dot(gain.w, gain.type, i, rank, g);
    if ((threadIdx.x & 31) == 0) out[i] = (1.f - tanhf(f)) * h[i] + tanhf(a) * m[i];
}

// Queries and relative-bias coefficients, a head a block (z = 0 queries, 1
// coefficients): a learned scale per channel, clipped to `clip`, then the full
// layers' log scaling. Grid (heads, T, 2).
struct HeadPreps { float* x[2]; const float* scale[2]; int dim[2]; float clip[2]; };

__global__ void k_head_prep(HeadPreps a, int full, const int* __restrict__ pos, int floor, float alpha)
{
    __shared__ float sh[32];
    const int z = blockIdx.z, dim = a.dim[z];
    const float clip = a.clip[z];
    float* hx = a.x[z] + ((long long)blockIdx.y * gridDim.x + blockIdx.x) * dim;
    const float* sc = a.scale[z] + blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = hx[i] * sc[i]; hx[i] = v; s += v * v; }
    float norm = sqrtf(block_sum(s, sh));
    float f = fminf(1.f, clip / (norm + 1e-6f));
    if (full)
    {
        float t = (float)(*pos + blockIdx.y + 1);
        f *= 1.f + alpha * logf(fmaxf(1.f, t / (float)floor));
    }
    if (f != 1.f)
        for (int i = threadIdx.x; i < dim; i += blockDim.x) hx[i] *= f;
}

// ── the key/value rows ───────────────────────────────────────────────────────

struct KvSlot
{
    void* k = nullptr; void* v = nullptr;                  // rows x C elements (FP4: rows x C/2 bytes)
    unsigned char* ks = nullptr; unsigned char* vs = nullptr; // FP4: rows x C/16 E4M3 block scales
    float* kg = nullptr; float* vg = nullptr;               // FP4: rows x kvHeads tensor scales
    int cap = 0;                                           // rows held
    int ring = 0;                                          // 1: a sliding window, row = position % cap
    int format = KV_F32;
};

struct KvView
{
    const void* k; const void* v; const unsigned char* ks; const unsigned char* vs; const float* kg; const float* vg;
    int cap, ring, format;
};

__device__ __forceinline__ float kv_load(const void* p, const unsigned char* sc, const float* g, int format,
                                         long long row, int C, int hd, int i)
{
    long long idx = row * C + i;
    if (format == KV_F32) return ((const float*)p)[idx];
    if (format == KV_F16) return __half2float(((const __half*)p)[idx]);
    unsigned char b = ((const unsigned char*)p)[idx >> 1];
    float2 pair = e2m1x2(b);
    float e = (idx & 1) ? pair.y : pair.x;
    return e * e4m3(sc[idx >> 4]) * g[row * (C / hd) + i / hd];
}

// The conv taps, clips and norm of one step's keys and values, then the row
// stored. Grid (kvHeads, 2): y = 0 keys, 1 values. Sliding layers: a static
// depthwise convolution; full layers: static taps plus one per head projected
// from the layer's input. The histories hold the projections before the conv.
struct KvPrep
{
    const float* raw[2];       // projections
    const float* dyn[2];       // full: [kvHeads x K] dynamic taps
    const float* conv[2];      // sliding [C x K]; full: static [K x C]
    float* hist[2];            // [(K-1) x C]
    KvView slot;
    void* dst[2]; unsigned char* dsc[2]; float* dg[2];
    int C, hd, K, full, count, kvHeads;
    float clip, eps;
};

__global__ void k_kv_prep(KvPrep a, const int* __restrict__ pos)
{
    __shared__ float sh[32];
    __shared__ float vals[256];
    int which = blockIdx.y, head = blockIdx.x, i = threadIdx.x;
    int K = a.K, C = a.C, hd = a.hd;
    int ch = head * hd + i;
    for (int tok = 0; tok < a.count; tok++)
    {
    int t = *pos + tok;
    float out = 0.f;
    if (i < hd)
    {
        const float* raw = a.raw[which] + (long long)tok * C;
        const float* w = a.conv[which];
        float* hist = a.hist[which];
        float x = raw[ch];
        float sum;
        if (!a.full)
        {
            sum = w[ch * K + K - 1] * x;
            for (int lag = 1; lag < K; lag++)
                if (t - lag >= 0) sum += w[ch * K + K - 1 - lag] * hist[((t - lag) % (K - 1)) * C + ch];
            out = (sum + x) * rsqrtf((float)(K + 1));
        }
        else
        {
            const float* dyn = a.dyn[which] + (long long)tok * a.kvHeads * K + head * K;
            sum = (w[ch] + dyn[0]) * x;
            for (int lag = 1; lag < K; lag++)
                if (t - lag >= 0) sum += (w[lag * C + ch] + dyn[lag]) * hist[((t - lag) % (K - 1)) * C + ch];
            out = (sum + x) * rsqrtf((float)(2 * K + 1));
        }
        hist[(t % (K - 1)) * C + ch] = x;
    }
    float ss = block_sum(i < hd ? out * out : 0.f, sh);
    if (which == 0)
    {
        float f = fminf(1.f, a.clip / (sqrtf(ss) + 1e-6f));
        out *= f;
    }
    else
    {
        out *= rsqrtf(ss / hd + a.eps);
    }
    long long row = a.slot.ring ? t % a.slot.cap : t;
    if (a.slot.format == KV_F32) { if (i < hd) ((float*)a.dst[which])[row * C + ch] = out; continue; }
    if (a.slot.format == KV_F16) { if (i < hd) ((__half*)a.dst[which])[row * C + ch] = __float2half(out); continue; }
    // NVFP4: a tensor scale per (row, head) so the largest block scale is
    // E4M3's 448, an E4M3 scale per 16, elements to the nearest E2M1.
    float amax = 0.f;
    if (i < hd) { vals[i] = out; }
    float am = i < hd ? fabsf(out) : 0.f;
    am = warp_max(am);
    if ((i & 31) == 0) sh[i >> 5] = am;
    __syncthreads();
    if (i == 0) { for (int w = 0; w < (blockDim.x + 31) / 32; w++) amax = fmaxf(amax, sh[w]); sh[0] = amax; }
    __syncthreads();
    amax = sh[0];
    float g = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
    if (i == 0) a.dg[which][row * (C / hd) + head] = g;
    if (i < hd && (i & 15) == 0)
    {
        float bm = 0.f;
        for (int k = 0; k < 16; k++) bm = fmaxf(bm, fabsf(vals[i + k]));
        __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(bm / 6.f / g, __NV_SATFINITE, __NV_E4M3);
        float scale = e4m3((unsigned char)s8);
        a.dsc[which][(row * C + ch) >> 4] = (unsigned char)s8;
        float step = scale * g;
        unsigned char* e = (unsigned char*)a.dst[which] + ((row * C + ch) >> 1);
        for (int k = 0; k < 16; k += 2)
        {
            auto code = [&](float v) -> unsigned int {
                if (!(step > 0.f)) return 0u;
                float x = v / step, ax = fabsf(x);
                unsigned int m = ax <= 0.25f ? 0u : ax < 0.75f ? 1u : ax <= 1.25f ? 2u : ax < 1.75f ? 3u
                               : ax <= 2.5f ? 4u : ax < 3.5f ? 5u : ax <= 5.f ? 6u : 7u;
                return m | (x < 0.f ? 8u : 0u);
            };
            e[k / 2] = (unsigned char)(code(vals[i + k]) | (code(vals[i + k + 1]) << 4));
        }
    }
    __syncthreads();
    }
}

// Split-key attention for one step: grid (heads, splits), 4 warps a block,
// each warp an online softmax over its keys, merged per block into
// (max, sum, acc[hd]). The relative bias r . P[:, dist] applies within
// `extent` of the newest key.
struct AttnArgs
{
    const float* q; const float* r; const float* relProj;
    KvView kv;
    float* part;               // [heads x splits x (hd + 2)]
    int heads, kvHeads, hd, dRel, extent, window, splits;
    float scale;
};

__global__ void k_attn(AttnArgs a, const int* __restrict__ pos)
{
    __shared__ float sm[4], sl[4], sacc[4][128];
    int h = blockIdx.x, s = blockIdx.y, z = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int t = *pos + z, hd = a.hd, C = a.kvHeads * hd, kvh = h / (a.heads / a.kvHeads);
    int n = a.window ? min(t + 1, a.window) : t + 1;
    int first = t + 1 - n;
    int chunk = (n + a.splits - 1) / a.splits;
    int lo = first + s * chunk, hi = min(first + n, lo + chunk);
    float qv[4], acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { int i = lane + 32 * j; qv[j] = i < hd ? a.q[((long long)z * a.heads + h) * hd + i] : 0.f; acc[j] = 0.f; }
    float rr = lane < a.dRel ? a.r[((long long)z * a.heads + h) * a.dRel + lane] : 0.f;
    float m = -INFINITY, l = 0.f;
    const KvView& kv = a.kv;
    for (int p = lo + warp; p < hi; p += 4)
    {
        long long row = kv.ring ? p % kv.cap : p;
        float dot = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int i = lane + 32 * j;
            if (i < hd) dot += qv[j] * kv_load(kv.k, kv.ks, kv.kg, kv.format, row, C, hd, kvh * hd + i);
        }
        int dist = t - p;
        float b = dist < a.extent && lane < a.dRel ? rr * a.relProj[lane * a.extent + dist] : 0.f;
        float sc = warp_sum(dot * a.scale + b);
        float mn = fmaxf(m, sc);
        float corr = expf(m - mn), pe = expf(sc - mn);
        l = l * corr + pe;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int i = lane + 32 * j;
            float vv = i < hd ? kv_load(kv.v, kv.vs, kv.vg, kv.format, row, C, hd, kvh * hd + i) : 0.f;
            acc[j] = acc[j] * corr + pe * vv;
        }
        m = mn;
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    #pragma unroll
    for (int j = 0; j < 4; j++) { int i = lane + 32 * j; if (i < 128) sacc[warp][i] = acc[j]; }
    __syncthreads();
    float* out = a.part + (((long long)z * a.heads + h) * a.splits + s) * (hd + 2);
    float M = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
    {
        float v = 0.f;
        for (int w = 0; w < 4; w++) if (sm[w] > -INFINITY) v += sacc[w][i] * expf(sm[w] - M);
        out[2 + i] = v;
    }
    if (threadIdx.x == 0)
    {
        float L = 0.f;
        for (int w = 0; w < 4; w++) if (sm[w] > -INFINITY) L += sl[w] * expf(sm[w] - M);
        out[0] = M; out[1] = L;
    }
}

// The splits merged with the sink logit, then the output gate.
__global__ void k_attn_merge(const float* __restrict__ part, const float* __restrict__ sinks, const float* __restrict__ gate,
                             float* __restrict__ out, int splits, int hd)
{
    int h = blockIdx.x, z = blockIdx.y, heads = gridDim.x;
    const float* p = part + ((long long)z * heads + h) * splits * (hd + 2);
    sinks += (long long)z * heads;
    gate += (long long)z * heads * hd;
    out += (long long)z * heads * hd;
    float M = sinks[h];
    for (int s = 0; s < splits; s++) M = fmaxf(M, p[s * (hd + 2)]);
    float L = expf(sinks[h] - M);
    for (int s = 0; s < splits; s++) { float ms = p[s * (hd + 2)]; if (ms > -INFINITY) L += p[s * (hd + 2) + 1] * expf(ms - M); }
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
    {
        float v = 0.f;
        for (int s = 0; s < splits; s++) { float ms = p[s * (hd + 2)]; if (ms > -INFINITY) v += p[s * (hd + 2) + 2 + i] * expf(ms - M); }
        float g = gate[h * hd + i];
        out[h * hd + i] = v / L * (1.f / (1.f + expf(-g)));
    }
}

// ── a prompt chunk's attention on the tensor cores ───────────────────────────
//
// FlashAttention-2 over FP16 tiles: a block takes one head, 64 queries of the
// chunk (16 a warp) and one key split; every key tile (64 keys) is decoded from
// the slot's format into shared memory once and serves all 64 queries. Q·Kᵀ and
// P·V are mma.m16n8k16 with f32 sums; the online softmax is f32. The scale,
// the relative bias (r · P[:, dist], from the table k_relbias makes) and the
// causal and window masks are applied to the scores; the output is the split
// partials k_attn_merge takes, which adds the sink and the output gate.

static const int kTQ = 64, kTK = 64;

// rb[z][h][dist] = r[z][h] · P[:, dist] for dist < extent.
__global__ void k_relbias(const float* __restrict__ r, const float* __restrict__ relProj, float* __restrict__ rb,
                          int heads, int dRel, int extent)
{
    int z = blockIdx.x, h = blockIdx.y;
    const float* rr = r + ((long long)z * heads + h) * dRel;
    float* o = rb + ((long long)z * heads + h) * extent;
    for (int dist = threadIdx.x; dist < extent; dist += blockDim.x)
    {
        float b = 0.f;
        for (int j = 0; j < dRel; j++) b += rr[j] * relProj[j * extent + dist];
        o[dist] = b;
    }
}

// Eight channels (ch0 .. ch0 + 7, one head's) of a key or value row as f32.
__device__ __forceinline__ void kv_load8(const void* p, const unsigned char* sc, const float* g, int format,
                                         long long row, int C, int hd, int ch0, float* o)
{
    long long idx = row * C + ch0;
    if (format == KV_F32)
    {
        float4 a = *(const float4*)((const float*)p + idx), b = *(const float4*)((const float*)p + idx + 4);
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
        return;
    }
    if (format == KV_F16)
    {
        uint4 u = *(const uint4*)((const __half*)p + idx);
        unsigned int w[4] = { u.x, u.y, u.z, u.w };
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[j]));
            o[2 * j] = f.x; o[2 * j + 1] = f.y;
        }
        return;
    }
    unsigned int w = *(const unsigned int*)((const unsigned char*)p + (idx >> 1));
    float scale = e4m3(sc[idx >> 4]) * g[row * (C / hd) + ch0 / hd];
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        float2 v = e2m1x2(w >> (8 * j));
        o[2 * j] = v.x * scale; o[2 * j + 1] = v.y * scale;
    }
}

__device__ __forceinline__ unsigned int half2_bits(float lo, float hi)
{
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned int*>(&h);
}

__device__ __forceinline__ void mma16816(float* c, const unsigned int* a, unsigned int b0, unsigned int b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct TileArgs
{
    const float* q; const float* rb;
    KvView kv;
    float* part;               // [T x heads x splits x (hd + 2)], as k_attn writes
    int heads, kvHeads, extent, window, splits, T;
    float scale;
};

// Grid (heads, splits, query tiles), 128 threads.
template <int HD>
__global__ void __launch_bounds__(128) k_attn_tiles(TileArgs a, const int* __restrict__ pos)
{
    constexpr int KS = HD + 8, VS = kTK + 8;    // padded strides (halves): conflict-free fragment reads
    __shared__ __align__(16) __half sk[kTK * KS];   // keys [key][channel]
    __shared__ __align__(16) __half sv[HD * VS];    // values transposed [channel][key]
    const int h = blockIdx.x, split = blockIdx.y, tile = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int p0 = *pos, C = a.kvHeads * HD, kvh = h / (a.heads / a.kvHeads);
    const int zq0 = tile * kTQ, zLast = min(a.T, zq0 + kTQ) - 1;
    const int tFirst = p0 + zq0, tLast = p0 + zLast;
    const int lo = a.window ? max(0, tFirst + 1 - a.window) : 0;
    const int chunk = (tLast + 1 - lo + a.splits - 1) / a.splits;
    const int klo = lo + split * chunk, khi = min(tLast + 1, klo + chunk);

    // This thread's two query rows: g and g + 8 of the warp's sixteen.
    const int zr[2] = { zq0 + warp * 16 + g, zq0 + warp * 16 + g + 8 };
    const bool live[2] = { zr[0] < a.T, zr[1] < a.T };
    const int tq[2] = { p0 + zr[0], p0 + zr[1] };
    const int wlo[2] = { a.window ? tq[0] + 1 - a.window : 0, a.window ? tq[1] + 1 - a.window : 0 };
    const float* rb[2] = { a.rb + ((long long)(live[0] ? zr[0] : 0) * a.heads + h) * a.extent,
                           a.rb + ((long long)(live[1] ? zr[1] : 0) * a.heads + h) * a.extent };

    unsigned int qf[HD / 16][4];
    {
        const float* q0 = a.q + ((long long)(live[0] ? zr[0] : 0) * a.heads + h) * HD;
        const float* q1 = a.q + ((long long)(live[1] ? zr[1] : 0) * a.heads + h) * HD;
        #pragma unroll
        for (int kk = 0; kk < HD / 16; kk++)
        {
            int c = kk * 16 + tig * 2;
            qf[kk][0] = live[0] ? half2_bits(q0[c], q0[c + 1]) : 0u;
            qf[kk][1] = live[1] ? half2_bits(q1[c], q1[c + 1]) : 0u;
            qf[kk][2] = live[0] ? half2_bits(q0[c + 8], q0[c + 9]) : 0u;
            qf[kk][3] = live[1] ? half2_bits(q1[c + 8], q1[c + 9]) : 0u;
        }
    }
    float o[HD / 8][4];
    #pragma unroll
    for (int nb = 0; nb < HD / 8; nb++) o[nb][0] = o[nb][1] = o[nb][2] = o[nb][3] = 0.f;
    float m[2] = { -INFINITY, -INFINITY }, l[2] = { 0.f, 0.f };
    const KvView& kv = a.kv;

    for (int k0 = klo; k0 < khi; k0 += kTK)
    {
        __syncthreads();
        for (int it = threadIdx.x; it < kTK * HD / 8; it += blockDim.x)
        {
            int kr = it / (HD / 8), d0 = (it % (HD / 8)) * 8, p = k0 + kr;
            float kx[8], vx[8];
            if (p < khi)
            {
                long long row = kv.ring ? p % kv.cap : p;
                kv_load8(kv.k, kv.ks, kv.kg, kv.format, row, C, HD, kvh * HD + d0, kx);
                kv_load8(kv.v, kv.vs, kv.vg, kv.format, row, C, HD, kvh * HD + d0, vx);
            }
            else
            {
                #pragma unroll
                for (int j = 0; j < 8; j++) kx[j] = vx[j] = 0.f;
            }
            *reinterpret_cast<uint4*>(&sk[kr * KS + d0]) =
                make_uint4(half2_bits(kx[0], kx[1]), half2_bits(kx[2], kx[3]), half2_bits(kx[4], kx[5]), half2_bits(kx[6], kx[7]));
            #pragma unroll
            for (int j = 0; j < 8; j++) sv[(d0 + j) * VS + kr] = __float2half_rn(vx[j]);
        }
        __syncthreads();

        float sc[kTK / 8][4];
        #pragma unroll
        for (int nb = 0; nb < kTK / 8; nb++)
        {
            sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = 0.f;
            #pragma unroll
            for (int kk = 0; kk < HD / 16; kk++)
            {
                const __half* kp = &sk[(nb * 8 + g) * KS + kk * 16 + tig * 2];
                mma16816(sc[nb], qf[kk], *reinterpret_cast<const unsigned int*>(kp), *reinterpret_cast<const unsigned int*>(kp + 8));
            }
        }
        float mx[2] = { -INFINITY, -INFINITY };
        #pragma unroll
        for (int nb = 0; nb < kTK / 8; nb++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                int rr = e >> 1, p = k0 + nb * 8 + tig * 2 + (e & 1), dist = tq[rr] - p;
                bool ok = live[rr] && p < khi && dist >= 0 && p >= wlo[rr];
                float x = sc[nb][e] * a.scale;
                if (ok && dist < a.extent) x += rb[rr][dist];
                sc[nb][e] = ok ? x : -INFINITY;
                mx[rr] = fmaxf(mx[rr], sc[nb][e]);
            }
        }
        float corr[2];
        #pragma unroll
        for (int rr = 0; rr < 2; rr++)
        {
            mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffffu, mx[rr], 1));
            mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffffu, mx[rr], 2));
            float mn = fmaxf(m[rr], mx[rr]);
            corr[rr] = mn == -INFINITY ? 1.f : expf(m[rr] - mn);
            m[rr] = mn;
            l[rr] *= corr[rr];
        }
        #pragma unroll
        for (int nb = 0; nb < kTK / 8; nb++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                int rr = e >> 1;
                float pe = sc[nb][e] == -INFINITY ? 0.f : expf(sc[nb][e] - m[rr]);
                sc[nb][e] = pe;
                l[rr] += pe;
            }
        }
        #pragma unroll
        for (int nb = 0; nb < HD / 8; nb++)
        {
            o[nb][0] *= corr[0]; o[nb][1] *= corr[0];
            o[nb][2] *= corr[1]; o[nb][3] *= corr[1];
        }
        #pragma unroll
        for (int j = 0; j < kTK / 16; j++)
        {
            unsigned int pa[4] = { half2_bits(sc[2 * j][0], sc[2 * j][1]), half2_bits(sc[2 * j][2], sc[2 * j][3]),
                                   half2_bits(sc[2 * j + 1][0], sc[2 * j + 1][1]), half2_bits(sc[2 * j + 1][2], sc[2 * j + 1][3]) };
            #pragma unroll
            for (int nb = 0; nb < HD / 8; nb++)
            {
                const __half* vp = &sv[(nb * 8 + g) * VS + j * 16 + tig * 2];
                mma16816(o[nb], pa, *reinterpret_cast<const unsigned int*>(vp), *reinterpret_cast<const unsigned int*>(vp + 8));
            }
        }
    }
    #pragma unroll
    for (int rr = 0; rr < 2; rr++)
    {
        l[rr] += __shfl_xor_sync(0xffffffffu, l[rr], 1);
        l[rr] += __shfl_xor_sync(0xffffffffu, l[rr], 2);
        if (!live[rr]) continue;
        float* out = a.part + (((long long)zr[rr] * a.heads + h) * a.splits + split) * (HD + 2);
        if (tig == 0) { out[0] = m[rr]; out[1] = l[rr]; }
        #pragma unroll
        for (int nb = 0; nb < HD / 8; nb++)
        {
            out[2 + nb * 8 + tig * 2] = o[nb][2 * rr];
            out[2 + nb * 8 + tig * 2 + 1] = o[nb][2 * rr + 1];
        }
    }
}

// A depthwise causal short convolution (Inkling2LoopShortConvolution) over
// this slot's history, (conv + x) / sqrt(K + 1), plus `add` when given.
__global__ void k_shortconv(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ hist,
                            const float* __restrict__ add, float* __restrict__ out, int C, int K, const int* __restrict__ pos, int count)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C) return;
    for (int tok = 0; tok < count; tok++)
    {
        int t = *pos + tok;
        long long o = (long long)tok * C + ch;
        float xv = x[o];
        float sum = w[ch * K + K - 1] * xv;
        for (int lag = 1; lag < K; lag++)
            if (t - lag >= 0) sum += w[ch * K + K - 1 - lag] * hist[((t - lag) % (K - 1)) * C + ch];
        hist[(t % (K - 1)) * C + ch] = xv;
        out[o] = (sum + xv) * rsqrtf((float)(K + 1)) + (add ? add[o] : 0.f);
    }
}

// Router (Inkling2LoopTopKRouter): sigmoid scores, the top k by score + bias
// (ties to the lower index), weights renormalised over the chosen, times route_scale.
__global__ void k_router(const float* __restrict__ logits, const float* __restrict__ bias, int experts, int topK,
                         float routeScale, int* __restrict__ idx, float* __restrict__ wt)
{
    __shared__ float score[512];
    logits += (long long)blockIdx.x * experts;
    idx += blockIdx.x * topK;
    wt += blockIdx.x * topK;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) score[e] = 1.f / (1.f + expf(-logits[e]));
    __syncwarp();
    // Each lane holds the keys of experts lane, lane + 32, ...; a round takes
    // the warp's best (ties to the lower index) and strikes it out.
    int lane = threadIdx.x & 31;
    float key[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) { int e = lane + 32 * j; key[j] = e < experts ? score[e] + bias[e] : -INFINITY; }
    float total = 0.f;
    for (int k = 0; k < topK; k++)
    {
        float bv = -INFINITY;
        int be = 0x7fffffff;
        #pragma unroll
        for (int j = 0; j < 16; j++)
        {
            int e = lane + 32 * j;
            if (e < experts && (key[j] > bv || (key[j] == bv && e < be))) { bv = key[j]; be = e; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; }
        }
        if ((be & 31) == lane)
        {
            #pragma unroll
            for (int j = 0; j < 16; j++) if (lane + 32 * j == be) key[j] = -INFINITY;
        }
        if (lane == 0) idx[k] = be;
        total += score[be];
    }
    total += 1e-20f;
    __syncwarp();
    if (lane < topK) wt[lane] = score[idx[lane]] / total * routeScale;
}

// The chosen experts' gate and up rows (interleaved: gate j = row 2j), through
// the softcapped SwiGLU. Grid (topK, rows / 8 of the width), a warp an output.
__global__ void k_expert_up(Experts w13, const float* __restrict__ x, const int* __restrict__ idx, float* __restrict__ act,
                            int width, float gateCap, float upCap)
{
    int j = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= width) return;
    int e = idx[blockIdx.x];
    float g, u;
    expert_dot_pair(w13, e, 2 * j, x, g, u);
    if ((threadIdx.x & 31) == 0)
        act[blockIdx.x * width + j] = tanhf(g / gateCap) * gateCap / (1.f + expf(-g)) * (tanhf(u / upCap) * upCap);
}

__global__ void k_expert_down(Experts w2, const float* __restrict__ act, const int* __restrict__ idx, const float* __restrict__ wt,
                              float* __restrict__ part, int d, int width)
{
    int i = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= d) return;
    int k = blockIdx.x;
    float s = expert_dot(w2, idx[k], i, act + k * width);
    if ((threadIdx.x & 31) == 0) part[k * d + i] = wt[k] * s;
}

// The experts summed in their chosen order, then the MLP's short convolution
// and the residual: out = residual + conv(ffn).
__global__ void k_expert_sum(const float* __restrict__ part, int topK, int d, float* __restrict__ ffn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    part += (long long)blockIdx.y * topK * d;
    float s = 0.f;
    for (int k = 0; k < topK; k++) s += part[k * d + i];
    ffn[(long long)blockIdx.y * d + i] = s;
}

// The same for the engine's experts, whose outputs are not yet weighted.
__global__ void k_expert_wsum(const float* __restrict__ part, const float* __restrict__ wt, int topK, int d, float* __restrict__ ffn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    part += (long long)blockIdx.y * topK * d;
    wt += blockIdx.y * topK;
    float s = 0.f;
    for (int k = 0; k < topK; k++) s += wt[k] * part[k * d + i];
    ffn[(long long)blockIdx.y * d + i] = s;
}

// rowScale[e * rows + r] = factor[e]: the per-expert factor as the engine's row scale.
__global__ void k_expand_factor(const float* __restrict__ factor, int rows, long long count, float* __restrict__ out)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = factor[i / rows];
}

// ── the model and a sequence ─────────────────────────────────────────────────

// Tokens a prompt chunk holds: every per-token buffer has this many rows.
static const int kChunk = 256;
// Key splits a prompt row's attention takes at most (a decode step takes more).
static const int kRowSplits = 8;

struct Ctx
{
    InkConfig c;
    std::vector<Layer> layers;
    Mat g[2];
    const float* gv[G_COUNT] = {};
    std::vector<void*> owned;
    long long bytes = 0;
    cudaStream_t stream = nullptr;   // the engine's
    Engine* eng = nullptr;
    int err = 0;                     // the first engine call of a chunk that failed
    int exactPrefill = 0;            // 1: prompts at f32 (the engine's SGEMMs, the f32 experts and attention here)
};

struct Slot
{
    KvSlot kv;
    float* khist = nullptr; float* vhist = nullptr; float* ahist = nullptr; float* mhist = nullptr;
    int full = 0, C = 0;
};

struct Seq
{
    int capacity = 0, format = KV_F32, pos = 0;
    std::vector<Slot> slots;
    int* dToks = nullptr; int* dPos = nullptr;   // device: kChunk tokens, then the first position
    int* hToks = nullptr;                        // pinned, the same layout
    float* hLogits = nullptr;
    std::vector<float*> hbuf;                    // per visit, kChunk rows of d
    float *rlogits, *q, *k, *v, *r, *sinks, *gate, *kdyn, *vdyn, *attn, *part, *mixed, *msg, *cat, *gvec, *xq, *xk, *xv,
          *nq, *nk, *nv, *res, *mlpIn, *mlpGated, *normed, *act, *epart, *ffn, *logits, *wt, *tmp;
    float* every = nullptr;                      // kChunk x vocab, made when a prompt's every position is asked for
    float* rb = nullptr;                         // a chunk's relative bias [kChunk x heads x extent], for k_attn_tiles
    int *idx, *offsets, *order;
    int *gorder = nullptr, *ggroups = nullptr;   // the engine's grouping: [3 x (pairs + experts)] each
    std::vector<void*> owned;
    long long bytes = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};

// Zeroed on the legacy stream, which the context's non-blocking stream does
// not wait for: whoever allocates synchronises the device before the next step.
static int dev_alloc(std::vector<void*>& owned, long long& bytes, void** p, size_t n)
{
    if (n == 0) n = 4;
    TRY(cudaMalloc(p, n));
    TRY(cudaMemset(*p, 0, n));
    owned.push_back(*p);
    bytes += (long long)n;
    return 0;
}

template <typename T> static int alloc_f(Seq* s, T** p, size_t count)
{
    return dev_alloc(s->owned, s->bytes, (void**)p, count * sizeof(T));
}

static int splits_for(const Slot& s)
{
    if (s.kv.ring) return std::max(1, std::min(16, (s.kv.cap + 63) / 64));
    return std::max(1, std::min(64, s.kv.cap / 128));
}

static long long kv_bytes(int rows, int C, int kvHeads, int format)
{
    if (format == KV_F32) return 2LL * rows * C * 4;
    if (format == KV_F16) return 2LL * rows * C * 2;
    return 2LL * ((long long)rows * C / 2 + (long long)rows * C / 16 + (long long)rows * kvHeads * 4);
}

static int alloc_kv(Seq* s, KvSlot& kv, int rows, int C, int kvHeads)
{
    kv.cap = rows;
    if (kv.format == KV_F32 || kv.format == KV_F16)
    {
        size_t el = kv.format == KV_F32 ? 4 : 2;
        if (int rc = dev_alloc(s->owned, s->bytes, &kv.k, (size_t)rows * C * el)) return rc;
        return dev_alloc(s->owned, s->bytes, &kv.v, (size_t)rows * C * el);
    }
    int rc = 0;
    if ((rc = dev_alloc(s->owned, s->bytes, &kv.k, (size_t)rows * C / 2))) return rc;
    if ((rc = dev_alloc(s->owned, s->bytes, &kv.v, (size_t)rows * C / 2))) return rc;
    if ((rc = dev_alloc(s->owned, s->bytes, (void**)&kv.ks, (size_t)rows * C / 16))) return rc;
    if ((rc = dev_alloc(s->owned, s->bytes, (void**)&kv.vs, (size_t)rows * C / 16))) return rc;
    if ((rc = dev_alloc(s->owned, s->bytes, (void**)&kv.kg, (size_t)rows * kvHeads * 4))) return rc;
    return dev_alloc(s->owned, s->bytes, (void**)&kv.vg, (size_t)rows * kvHeads * 4);
}

static int phys_of(const Ctx* x, int slot)
{
    int n = x->c.layers, core = x->c.loopEnd - x->c.loopStart;
    return slot < n ? slot : x->c.loopStart + (slot - n) % core;
}

// The rows a slot holds for a sequence of `capacity` positions: a full slot
// every position; a sliding slot its window and a chunk more, so the rows a
// chunk writes never overwrite keys its earlier rows still read.
static int slot_rows(const Ctx* x, int full, int capacity)
{
    return full ? capacity : std::min(capacity, x->c.window + kChunk);
}

// Projections that share nothing but a launch: one row (a step) as GEMVs, T
// rows (a chunk) as GEMMs.
static void proj(Ctx* x, int T, std::initializer_list<std::tuple<Mat, const float*, float*>> list)
{
    Gemms g = {};
    g.first[0] = 0;
    for (auto& [m, in, out] : list)
    {
        g.m[g.n] = m; g.x[g.n] = in; g.y[g.n] = out;
        g.first[g.n + 1] = g.first[g.n] + m.rows;
        g.n++;
    }
    g.T = T;
    int rows = g.first[g.n];
    if (T == 1 && g.n == 1 && rows <= 256)
    {
        k_gemv_rows<<<rows, 256, 0, x->stream>>>(g.m[0], g.x[0], g.y[0]);
    }
    else if (T == 1)
    {
        Gemvs v = {};
        for (int i = 0; i < g.n; i++) { v.m[i] = g.m[i]; v.x[i] = g.x[i]; v.y[i] = g.y[i]; }
        for (int i = 0; i <= g.n; i++) v.first[i] = g.first[i];
        v.n = g.n;
        k_gemvs<<<(rows + 7) / 8, 256, 0, x->stream>>>(v);
    }
    else
    {
        // The engine's GEMMs (BF16 tensor cores, or f32 SGEMM with prompts kept at f32).
        for (int i = 0; i < g.n; i++)
        {
            const Mat& m = g.m[i];
            int wType = m.type == W_BF16 ? 0 : m.type == W_F32 ? 1 : -1;
            int rc = wType < 0 ? -100
                : amqe_linear(x->eng, m.w, wType, m.rows, m.cols, g.x[i], m.cols, T, g.y[i], m.rows,
                              nullptr, nullptr, 0, 1.f, 0, 0, nullptr, nullptr, 0.f, nullptr, 0.f, 0, nullptr, nullptr);
            if (rc == -100)
            {
                Gemms one = {};
                one.m[0] = m; one.x[0] = g.x[i]; one.y[0] = g.y[i];
                one.first[1] = m.rows; one.n = 1; one.T = T;
                k_gemms<<<(m.rows + 7) / 8, 256, 0, x->stream>>>(one);
            }
            else if (rc && !x->err) x->err = rc;
        }
    }
}

// The engine's code for an expert stack's format: 0 BF16, 2 NVFP4, 4 MXFP8; -1 none.
static int engine_type(int type) { return type == W_BF16 ? 0 : type == W_NVFP4 ? 2 : type == W_MXFP8 ? 4 : -1; }
// The same for a step's experts, whose kernels also read f32 (1).
static int engine_step_type(int type) { return type == W_F32 ? 1 : engine_type(type); }

static void rmsnorm(Ctx* x, const float* in, const float* w, float* out, int T)
{
    Norms n = {};
    n.x[0] = in; n.w[0] = w; n.out[0] = out;
    k_rmsnorm<<<dim3(T, 1), 256, 0, x->stream>>>(n, x->c.d, x->c.eps);
}

// Three norms in one launch.
static void rmsnorm3(Ctx* x, const float* const in[3], const float* const w[3], float* const out[3], int T)
{
    Norms n = {};
    for (int i = 0; i < 3; i++) { n.x[i] = in[i]; n.w[i] = w[i]; n.out[i] = out[i]; }
    k_rmsnorm<<<dim3(T, 3), 256, 0, x->stream>>>(n, x->c.d, x->c.eps);
}

static void mix(Ctx* x, const std::vector<float*>& blocks, const float* partial, const float* queries, int count, float* out, int T,
                const float* const normW[] = nullptr, float* const normOut[] = nullptr)
{
    States st = {};
    st.n = (int)blocks.size() + 1;
    for (size_t i = 0; i < blocks.size(); i++) st.s[i] = blocks[i];
    st.s[blocks.size()] = partial;
    MixNorm mn = {};
    mn.eps = x->c.eps;
    if (normOut)
        for (int i = 0; i < count; i++) { mn.w[i] = normW[i]; mn.out[i] = normOut[i]; }
    const int d = x->c.d;
    // The float4 kernel when every vector it reads or writes is 16-byte aligned.
    auto al = [](const void* p) { return ((uintptr_t)p & 15) == 0; };
    bool fit = d % 4 == 0 && d / 4 <= 1024 && st.n <= 10 && al(queries) && al(out);
    for (int i = 0; i < st.n; i++) fit = fit && al(st.s[i]);
    for (int i = 0; i < count; i++) fit = fit && al(mn.w[i]) && al(mn.out[i]);
    if (fit)
        k_mix4<<<dim3(count, T), ((d / 4 + 31) / 32) * 32, 0, x->stream>>>(st, queries, out, d, mn);
    else
        k_mix<<<dim3(count, T), 1024, 0, x->stream>>>(st, queries, out, d, mn);
}

// Message gates g0 .. g0 + n - 1 (n is 1 or 3): gate g reads h[i] and the
// message m[i], and writes out[i]. The q, k and v gates are independent, so
// their three steps (cat, the down projection, apply) each take one launch.
static void message_gates(Ctx* x, Seq* s, const Layer& L, int g0, int n, const float* const h[3], const float* const m[3],
                          float* const out[3], int T)
{
    int d = x->c.d, rank = x->c.rank;
    GateCats gc = {};
    GateApplies ga = {};
    float* cat[3];
    float* gv[3];
    for (int i = 0; i < n; i++)
    {
        cat[i] = s->cat + (long long)i * kChunk * 2 * d;
        gv[i] = s->gvec + (long long)i * kChunk * rank;
        gc.h[i] = h[i]; gc.m[i] = m[i]; gc.cat[i] = cat[i];
        ga.forget[i] = L.m[M_CTRL + 3 * (g0 + i) + 1];
        ga.gain[i] = L.m[M_CTRL + 3 * (g0 + i) + 2];
        ga.g[i] = gv[i]; ga.h[i] = h[i]; ga.m[i] = m[i]; ga.out[i] = out[i];
    }
    k_gate_cat<<<dim3(T, n), 256, 0, x->stream>>>(gc, d, x->c.eps);
    if (n == 3)
        proj(x, T, { { L.m[M_CTRL + 3 * g0], cat[0], gv[0] }, { L.m[M_CTRL + 3 * (g0 + 1)], cat[1], gv[1] },
                     { L.m[M_CTRL + 3 * (g0 + 2)], cat[2], gv[2] } });
    else
        proj(x, T, { { L.m[M_CTRL + 3 * g0], cat[0], gv[0] } });
    k_gate_apply<<<dim3((d + 7) / 8, T, n), 256, 0, x->stream>>>(ga, d, rank);
}

static KvView view_of(const KvSlot& kv) { return KvView{ kv.k, kv.v, kv.ks, kv.vs, kv.kg, kv.vg, kv.cap, kv.ring, kv.format }; }

// One layer visit (Inkling2Loop{Sliding,Full}[Loop]DecoderLayer) for T rows
// (consecutive positions from *dPos): every buffer holds T rows.
static float* run_layer(Ctx* x, Seq* s, int l, Slot& slot, float* h, std::vector<float*>& blocks, const float* messages,
                        float* outBuf, int T)
{
    const Layer& L = x->layers[l];
    const InkConfig& c = x->c;
    int d = c.d, K = c.convK;
    long long Td = (long long)T * d;
    const float *qIn, *kIn, *vIn;
    if (L.full)
    {
        const float *xq = h, *xk = h, *xv = h;
        const float* qkvNorm[3] = { L.v[V_QNORM], L.v[V_KNORM], L.v[V_VNORM] };
        float* const qkvOut[3] = { s->nq, s->nk, s->nv };
        bool normed = false;
        if (L.v[V_ATTNQUERY])
        {
            mix(x, blocks, h, L.v[V_ATTNQUERY], 3, s->mixed, T, messages ? nullptr : qkvNorm, messages ? nullptr : qkvOut);
            normed = !messages;
            xq = s->mixed; xk = s->mixed + Td; xv = s->mixed + 2 * Td;
        }
        if (messages)
        {
            const float* hs[3] = { xq, xk, xv };
            const float* ms[3] = { messages, messages + Td, messages + 2 * Td };
            float* const outs[3] = { s->xq, s->xk, s->xv };
            message_gates(x, s, L, 0, 3, hs, ms, outs, T);
            xq = s->xq; xk = s->xk; xv = s->xv;
        }
        if (!normed)
        {
            const float* ins[3] = { xq, xk, xv };
            rmsnorm3(x, ins, qkvNorm, qkvOut, T);
        }
        qIn = s->nq; kIn = s->nk; vIn = s->nv;
    }
    else
    {
        const float* xx = h;
        const float* attnNorm[1] = { L.v[V_ATTNNORM] };
        float* const attnOut[1] = { s->nq };
        bool normed = false;
        if (L.v[V_ATTNQUERY])
        {
            mix(x, blocks, h, L.v[V_ATTNQUERY], 1, s->mixed, T, messages ? nullptr : attnNorm, messages ? nullptr : attnOut);
            normed = !messages;
            xx = s->mixed;
        }
        if (messages)
        {
            const float* hs[3] = { xx, xx, xx };
            const float* ms[3] = { messages, messages + Td, messages + 2 * Td };
            float* const outs[3] = { s->xq, s->xk, s->xv };
            message_gates(x, s, L, 0, 3, hs, ms, outs, T);
            const float* ins[3] = { s->xq, s->xk, s->xv };
            const float* ws[3] = { L.v[V_ATTNNORM], L.v[V_ATTNNORM], L.v[V_ATTNNORM] };
            float* const nouts[3] = { s->nq, s->nk, s->nv };
            rmsnorm3(x, ins, ws, nouts, T);
            qIn = s->nq; kIn = s->nk; vIn = s->nv;
        }
        else
        {
            if (!normed) rmsnorm(x, xx, L.v[V_ATTNNORM], s->nq, T);
            qIn = kIn = vIn = s->nq;
        }
    }

    // Attention (Inkling2Loop{Sliding,Full}Attention).
    int heads = L.heads, kvh = L.kvHeads, hd = L.headDim, C = kvh * hd;
    proj(x, T, { { L.m[M_WQ], qIn, s->q }, { L.m[M_WR], qIn, s->r }, { L.m[M_SINK], qIn, s->sinks }, { L.m[M_GATE], qIn, s->gate } });
    if (L.full)
        proj(x, T, { { L.m[M_WK], kIn, s->k }, { L.m[M_WV], vIn, s->v }, { L.m[M_KPROJ], kIn, s->kdyn }, { L.m[M_VPROJ], vIn, s->vdyn } });
    else
        proj(x, T, { { L.m[M_WK], kIn, s->k }, { L.m[M_WV], vIn, s->v } });
    {
        HeadPreps hp = {};
        hp.x[0] = s->q; hp.scale[0] = L.v[V_QSCALE]; hp.dim[0] = hd; hp.clip[0] = c.qkClip;
        hp.x[1] = s->r; hp.scale[1] = L.v[V_RELSCALE]; hp.dim[1] = c.relDim; hp.clip[1] = c.relClip;
        k_head_prep<<<dim3(heads, T, 2), 128, 0, x->stream>>>(hp, L.full, s->dPos, c.logFloor, c.alpha);
    }
    KvPrep kp = {};
    kp.raw[0] = s->k; kp.raw[1] = s->v;
    if (L.full)
    {
        kp.dyn[0] = s->kdyn; kp.dyn[1] = s->vdyn;
        kp.conv[0] = L.v[V_KSTATIC]; kp.conv[1] = L.v[V_VSTATIC];
    }
    else
    {
        kp.conv[0] = L.v[V_KCONV]; kp.conv[1] = L.v[V_VCONV];
    }
    kp.hist[0] = slot.khist; kp.hist[1] = slot.vhist;
    kp.slot = view_of(slot.kv);
    kp.dst[0] = slot.kv.k; kp.dst[1] = slot.kv.v;
    kp.dsc[0] = slot.kv.ks; kp.dsc[1] = slot.kv.vs;
    kp.dg[0] = slot.kv.kg; kp.dg[1] = slot.kv.vg;
    kp.C = C; kp.hd = hd; kp.K = K; kp.full = L.full; kp.clip = c.qkClip; kp.eps = c.eps;
    kp.count = T; kp.kvHeads = kvh;
    k_kv_prep<<<dim3(kvh, 2), 128, 0, x->stream>>>(kp, s->dPos);

    AttnArgs aa = {};
    aa.q = s->q; aa.r = s->r; aa.relProj = L.v[V_RELPROJ];
    aa.kv = view_of(slot.kv);
    aa.part = s->part;
    aa.heads = heads; aa.kvHeads = kvh; aa.hd = hd; aa.dRel = c.relDim; aa.extent = L.extent;
    aa.window = L.full ? 0 : c.window;
    aa.splits = T == 1 ? splits_for(slot) : std::min(splits_for(slot), kRowSplits);
    aa.scale = 1.f / sqrtf((float)hd);
    // AMQL_TILED_ATTENTION=0 keeps a chunk's attention on the per-query kernel (to compare, or as a way out).
    static int tiledOn = -1;
    if (tiledOn < 0) { const char* v = getenv("AMQL_TILED_ATTENTION"); tiledOn = v && v[0] == '0' ? 0 : 1; }
    bool tiled = tiledOn && T > 1 && !x->exactPrefill && (hd == 16 || hd == 32 || hd == 64 || hd == 128) && heads <= 64;
    if (tiled)
    {
        // A chunk on the tensor cores: splits enough to fill the device, a few hundred keys each at least.
        int keys = L.full ? s->pos + T : std::min(s->pos + T, c.window + kTQ);
        int sp = std::max(1, std::min(aa.splits, keys / 512));
        if (L.extent > 0)
            k_relbias<<<dim3(T, heads), 256, 0, x->stream>>>(s->r, L.v[V_RELPROJ], s->rb, heads, c.relDim, L.extent);
        TileArgs ta = {};
        ta.q = s->q; ta.rb = s->rb; ta.kv = view_of(slot.kv); ta.part = s->part;
        ta.heads = heads; ta.kvHeads = kvh; ta.extent = L.extent; ta.window = aa.window; ta.splits = sp; ta.T = T;
        ta.scale = aa.scale;
        dim3 grid(heads, sp, (T + kTQ - 1) / kTQ);
        switch (hd)
        {
            case 16: k_attn_tiles<16><<<grid, 128, 0, x->stream>>>(ta, s->dPos); break;
            case 32: k_attn_tiles<32><<<grid, 128, 0, x->stream>>>(ta, s->dPos); break;
            case 64: k_attn_tiles<64><<<grid, 128, 0, x->stream>>>(ta, s->dPos); break;
            default: k_attn_tiles<128><<<grid, 128, 0, x->stream>>>(ta, s->dPos); break;
        }
        aa.splits = sp;
    }
    else
    {
        k_attn<<<dim3(heads, aa.splits, T), 128, 0, x->stream>>>(aa, s->dPos);
    }
    k_attn_merge<<<dim3(heads, T), 128, 0, x->stream>>>(s->part, s->sinks, s->gate, s->tmp, aa.splits, hd);
    proj(x, T, { { L.m[M_WO], s->tmp, s->attn } });

    // The attention's short convolution, and the residual: a block's first
    // layer keeps its input as a block and restarts the residual.
    bool blockStart = l % c.block == 0;
    if (blockStart) blocks.push_back(h);
    k_shortconv<<<(d + 255) / 256, 256, 0, x->stream>>>(s->attn, L.v[V_ATTNCONV], slot.ahist, blockStart ? nullptr : h, s->res, d, K, s->dPos, T);

    const float* mlpNorm[1] = { L.v[V_MLPNORM] };
    float* const mlpOut[1] = { s->normed };
    mix(x, blocks, s->res, L.v[V_MLPQUERY], 1, s->mlpIn, T, messages ? nullptr : mlpNorm, messages ? nullptr : mlpOut);
    const float* mlpIn = s->mlpIn;
    if (messages)
    {
        const float* hs[3] = { s->mlpIn };
        const float* ms[3] = { messages + 3 * Td };
        float* const outs[3] = { s->mlpGated };
        message_gates(x, s, L, 3, 1, hs, ms, outs, T);
        mlpIn = s->mlpGated;
    }
    if (messages) rmsnorm(x, mlpIn, L.v[V_MLPNORM], s->normed, T);

    // The experts (Inkling2LoopTopKRouter, Inkling2LoopExperts): a step reads
    // its chosen experts' rows; a chunk groups its pairs by expert first.
    proj(x, T, { { L.m[M_ROUTER], s->normed, s->rlogits } });
    k_router<<<T, 32, 0, x->stream>>>(s->rlogits, L.v[V_ROUTERBIAS], c.experts, c.topK, c.routeScale, s->idx, s->wt);
    bool grouped = T > 1 && d % 32 == 0 && c.width % 32 == 0 && d % 8 == 0;
    bool summed = false;
    if (!grouped && T <= 8 && engine_step_type(L.w13.type) >= 0 && engine_step_type(L.w2.type) >= 0)
    {
        // A step's experts on the engine's decode kernels: the up rows with the
        // capped SwiGLU, then each pair's down projection; weighed and summed here.
        int rc = amqe_moe_experts_paired_step(x->eng, L.w13.w, engine_step_type(L.w13.type), L.w13.scales, L.w13.rowScale,
                                              L.w2.w, engine_step_type(L.w2.type), L.w2.scales, L.w2.rowScale,
                                              c.experts, d, c.width, c.topK, c.gateCap, c.upCap, s->normed, d, T, s->idx,
                                              s->act, s->epart);
        if (rc == 0)
        {
            k_expert_wsum<<<dim3((d + 255) / 256, T), 256, 0, x->stream>>>(s->epart, s->wt, c.topK, d, s->ffn);
            summed = true;
        }
        else if (rc != -100 && !x->err) x->err = rc;
    }
    if (summed)
    {
    }
    else if (!grouped)
    {
        for (int t = 0; t < T; t++)
        {
            k_expert_up<<<dim3(c.topK, (c.width + 7) / 8), 256, 0, x->stream>>>(L.w13, s->normed + (long long)t * d, s->idx + t * c.topK,
                                                                               s->act + (long long)t * c.topK * c.width, c.width, c.gateCap, c.upCap);
            k_expert_down<<<dim3(c.topK, (d + 7) / 8), 256, 0, x->stream>>>(L.w2, s->act + (long long)t * c.topK * c.width, s->idx + t * c.topK,
                                                                           s->wt + t * c.topK, s->epart + (long long)t * c.topK * d, d, c.width);
        }
    }
    else
    {
        int gu = engine_type(L.w13.type), dn = engine_type(L.w2.type);
        int rc = gu < 0 || dn < 0 ? -100
            : amqe_moe_experts_paired(x->eng, L.w13.w, gu, L.w13.scales, L.w13.rowScale, L.w2.w, dn, L.w2.scales, L.w2.rowScale,
                                      c.experts, d, c.width, c.topK, c.gateCap, c.upCap, s->normed, d, T, s->idx,
                                      s->gorder, s->ggroups, s->epart);
        if (rc == 0)
        {
            k_expert_wsum<<<dim3((d + 255) / 256, T), 256, 0, x->stream>>>(s->epart, s->wt, c.topK, d, s->ffn);
        }
        else
        {
            if (rc != -100 && !x->err) x->err = rc;
            k_group<<<1, 1024, 0, x->stream>>>(s->idx, T * c.topK, c.experts, s->offsets, s->order);
            k_expert_up_g<<<dim3(c.experts, (c.width + 31) / 32), 256, 0, x->stream>>>(L.w13, s->normed, s->offsets, s->order, c.topK,
                                                                                    s->act, c.width, c.gateCap, c.upCap);
            k_expert_down_g<<<dim3(c.experts, (d + 63) / 64), 256, 0, x->stream>>>(L.w2, s->act, s->offsets, s->order, s->wt, s->epart, d, c.width);
            k_expert_sum<<<dim3((d + 255) / 256, T), 256, 0, x->stream>>>(s->epart, c.topK, d, s->ffn);
        }
    }
    if (!grouped && !summed) k_expert_sum<<<dim3((d + 255) / 256, T), 256, 0, x->stream>>>(s->epart, c.topK, d, s->ffn);
    k_shortconv<<<(d + 255) / 256, 256, 0, x->stream>>>(s->ffn, L.v[V_MLPCONV], slot.mhist, s->res, outBuf, d, K, s->dPos, T);
    return outBuf;
}

// T rows through the model, as launches on the context's stream (a step's are
// captured into a graph). The logits: the last row's into s->logits, or
// every row's into s->every.
static void record(Ctx* x, Seq* s, int T, bool every)
{
    const InkConfig& c = x->c;
    int d = c.d, core = c.loopEnd - c.loopStart;
    long long Td = (long long)T * d;
    int buf = 0;
    float* h = s->hbuf[buf++];
    k_embed<<<T, 256, 0, x->stream>>>(x->g[G_EMBED], s->dToks, x->gv[G_EMBEDNORM], h, c.eps);
    std::vector<float*> blocks;
    for (int l = 0; l < c.loopStart; l++) h = run_layer(x, s, l, s->slots[l], h, blocks, nullptr, s->hbuf[buf++], T);
    float* prelude = h;
    std::vector<float*> preludeBlocks = blocks;
    for (int pass = 0; pass < c.passes; pass++)
    {
        std::vector<float*> bank;
        float* bankOut = nullptr;
        if (pass > 0)
        {
            bank.assign(blocks.begin() + preludeBlocks.size() + 1, blocks.end());
            bankOut = h;
            h = prelude;
            blocks = preludeBlocks;
        }
        for (int ip = c.loopStart; ip < c.loopEnd; ip++)
        {
            int slot = pass == 0 ? ip : c.layers + (pass - 1) * core + (ip - c.loopStart);
            float* messages = nullptr;
            if (pass > 0)
            {
                // q, k, v and mlp messages from the previous pass: [4][T][d].
                mix(x, bank, bankOut, x->gv[G_MESSAGES] + (long long)(ip - c.loopStart) * 4 * d, 4, s->msg, T);
                messages = s->msg;
            }
            h = run_layer(x, s, ip, s->slots[slot], h, blocks, messages, s->hbuf[buf++], T);
        }
    }
    for (int l = c.loopEnd; l < c.layers; l++) h = run_layer(x, s, l, s->slots[l], h, blocks, nullptr, s->hbuf[buf++], T);
    {
        const float* finalNorm[1] = { x->gv[G_FINALNORM] };
        float* const finalOut[1] = { s->normed };
        mix(x, blocks, h, x->gv[G_OUTPUTQUERY], 1, s->mixed, T, finalNorm, finalOut);
    }
    Mat head = x->g[G_UNEMBED];
    head.rows = c.vocab;   // the padded rows are never read
    if (every)
    {
        proj(x, T, { { head, s->normed, s->every } });
        long long n = (long long)T * c.vocab;
        k_softcap<<<(unsigned)((n + 255) / 256), 256, 0, x->stream>>>(s->every, (int)n, c.finalCap);
    }
    else
    {
        proj(x, 1, { { head, s->normed + Td - d, s->logits } });
        k_softcap<<<(c.vocab + 255) / 256, 256, 0, x->stream>>>(s->logits, c.vocab, c.finalCap);
    }
}

// ── exports ──────────────────────────────────────────────────────────────────

INK_EXPORT int ink_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0 ? 1 : 0;
}

INK_EXPORT int ink_mem_info(long long* freeBytes, long long* totalBytes)
{
    size_t f = 0, t = 0;
    TRY(cudaMemGetInfo(&f, &t));
    *freeBytes = (long long)f;
    *totalBytes = (long long)t;
    return 0;
}

INK_EXPORT int ink_create(const InkConfig* cfg, void** out)
{
    if (cfg->topK > 32 || cfg->experts > 512 || cfg->relDim > 32 || cfg->convK < 2 || cfg->topK * kChunk > 1 << 16)
        return -1000;
    Ctx* x = new Ctx();
    x->c = *cfg;
    x->layers.resize(cfg->layers);
    if (int rc = amqe_create(&x->eng)) { delete x; return -2100 + (rc < 0 ? rc : 0); }
    x->stream = amqe_stream(x->eng);
    *out = x;
    return 0;
}

INK_EXPORT int ink_upload(void* ctx, const void* host, long long bytes, void** dev)
{
    Ctx* x = (Ctx*)ctx;
    TRY(cudaMalloc(dev, (size_t)std::max(bytes, 4LL)));
    TRY(cudaMemcpy(*dev, host, (size_t)bytes, cudaMemcpyHostToDevice));
    x->owned.push_back(*dev);
    x->bytes += bytes;
    return 0;
}

INK_EXPORT int ink_set_geometry(void* ctx, int l, int full, int heads, int kvHeads, int headDim, int extent, int loop)
{
    Ctx* x = (Ctx*)ctx;
    if (l < 0 || l >= x->c.layers || headDim > 128 || headDim % 16 != 0 || heads % kvHeads != 0) return -1001;
    Layer& L = x->layers[l];
    L.full = full; L.heads = heads; L.kvHeads = kvHeads; L.headDim = headDim; L.extent = extent; L.loop = loop;
    return 0;
}

INK_EXPORT int ink_set_mat(void* ctx, int l, int which, const void* dev, int type, int rows, int cols)
{
    Ctx* x = (Ctx*)ctx;
    if (l < 0 || l >= x->c.layers || which < 0 || which >= M_COUNT || (type != W_F32 && type != W_BF16)) return -1002;
    x->layers[l].m[which] = Mat{ dev, type, rows, cols };
    return 0;
}

INK_EXPORT int ink_set_vec(void* ctx, int l, int which, const float* dev)
{
    Ctx* x = (Ctx*)ctx;
    if (l < 0 || l >= x->c.layers || which < 0 || which >= V_COUNT) return -1003;
    x->layers[l].v[which] = dev;
    return 0;
}

INK_EXPORT int ink_set_experts(void* ctx, int l, int which, const void* w, const unsigned char* scales, const float* factor,
                               int type, int rows, int cols)
{
    Ctx* x = (Ctx*)ctx;
    if (l < 0 || l >= x->c.layers || type < W_F32 || type > W_MXFP8) return -1004;
    if ((type == W_NVFP4 && cols % 16) || (type == W_MXFP8 && cols % 32)) return -1005;
    Experts e{ w, scales, factor, type, rows, cols };
    if (type == W_NVFP4 && factor)
    {
        long long count = (long long)x->c.experts * rows;
        float* rs = nullptr;
        TRY(cudaMalloc((void**)&rs, (size_t)count * sizeof(float)));
        x->owned.push_back(rs);
        x->bytes += count * (long long)sizeof(float);
        k_expand_factor<<<(unsigned)((count + 255) / 256), 256, 0, x->stream>>>(factor, rows, count, rs);
        TRY(cudaGetLastError());
        TRY(cudaStreamSynchronize(x->stream));
        e.rowScale = rs;
    }
    (which == 0 ? x->layers[l].w13 : x->layers[l].w2) = e;
    return 0;
}

INK_EXPORT int ink_set_global(void* ctx, int which, const void* dev, int type, int rows, int cols)
{
    Ctx* x = (Ctx*)ctx;
    if (which < 0 || which >= G_COUNT) return -1006;
    if (which <= G_UNEMBED) x->g[which] = Mat{ dev, type, rows, cols };
    else x->gv[which] = (const float*)dev;
    return 0;
}

INK_EXPORT long long ink_device_bytes(void* ctx) { return ((Ctx*)ctx)->bytes; }

static void seq_release(Seq* s)
{
    if (s->exec) cudaGraphExecDestroy(s->exec);
    if (s->graph) cudaGraphDestroy(s->graph);
    for (void* p : s->owned) cudaFree(p);
    if (s->hToks) cudaFreeHost(s->hToks);
    if (s->hLogits) cudaFreeHost(s->hLogits);
}

static int max_parts(const Ctx* x, const Seq* s)
{
    int most = 1;
    for (size_t i = 0; i < s->slots.size(); i++)
    {
        int heads = x->layers[phys_of(x, (int)i)].heads;
        int sp = splits_for(s->slots[i]);
        most = std::max(most, std::max(sp * heads, kChunk * std::min(sp, kRowSplits) * heads));
    }
    return most;
}

// A sequence of up to `capacity` positions: every slot's rows and histories,
// and the scratch of a chunk of kChunk tokens.
INK_EXPORT int ink_seq_create(void* ctx, int capacity, int format, void** out)
{
    Ctx* x = (Ctx*)ctx;
    const InkConfig& c = x->c;
    Seq* s = new Seq();
    s->capacity = capacity;
    s->format = format;
    int core = c.loopEnd - c.loopStart, nslots = c.layers + (c.passes - 1) * core, d = c.d;
    const size_t R = kChunk;
    int rc = 0;
    auto fail = [&](int code) { seq_release(s); delete s; return code; };
    s->slots.resize(nslots);
    int maxHeads = 1, maxC = 1;
    for (int i = 0; i < nslots; i++)
    {
        const Layer& L = x->layers[phys_of(x, i)];
        Slot& sl = s->slots[i];
        sl.full = L.full;
        sl.C = L.kvHeads * L.headDim;
        sl.kv.ring = L.full ? 0 : 1;
        // Sliding slots keep f32 rows of their window (a window is small); full slots the format asked for.
        sl.kv.format = L.full ? format : KV_F32;
        if ((rc = alloc_kv(s, sl.kv, slot_rows(x, L.full, capacity), sl.C, L.kvHeads))) return fail(rc);
        if ((rc = alloc_f(s, &sl.khist, (size_t)(c.convK - 1) * sl.C))) return fail(rc);
        if ((rc = alloc_f(s, &sl.vhist, (size_t)(c.convK - 1) * sl.C))) return fail(rc);
        if ((rc = alloc_f(s, &sl.ahist, (size_t)(c.convK - 1) * d))) return fail(rc);
        if ((rc = alloc_f(s, &sl.mhist, (size_t)(c.convK - 1) * d))) return fail(rc);
        maxHeads = std::max(maxHeads, L.heads * L.headDim);
        maxC = std::max(maxC, sl.C);
    }
    int visits = c.loopStart + c.passes * core + (c.layers - c.loopEnd) + 2;
    s->hbuf.resize(visits);
    for (auto& p : s->hbuf) if ((rc = alloc_f(s, &p, R * d))) return fail(rc);
    float** heads[] = { &s->q, &s->gate, &s->tmp };
    for (auto p : heads) if ((rc = alloc_f(s, p, R * maxHeads))) return fail(rc);
    float** kv[] = { &s->k, &s->v };
    for (auto p : kv) if ((rc = alloc_f(s, p, R * maxC))) return fail(rc);
    if ((rc = alloc_f(s, &s->r, R * 64 * c.relDim))) return fail(rc);
    if ((rc = alloc_f(s, &s->sinks, R * 256))) return fail(rc);
    if ((rc = alloc_f(s, &s->rlogits, R * c.experts))) return fail(rc);
    if ((rc = alloc_f(s, &s->kdyn, R * 64 * c.convK))) return fail(rc);
    if ((rc = alloc_f(s, &s->vdyn, R * 64 * c.convK))) return fail(rc);
    if ((rc = alloc_f(s, &s->part, (size_t)max_parts(x, s) * 130))) return fail(rc);
    if ((rc = alloc_f(s, &s->mixed, R * 3 * d))) return fail(rc);
    if ((rc = alloc_f(s, &s->msg, R * 4 * d))) return fail(rc);
    if ((rc = alloc_f(s, &s->cat, 3 * R * 2 * d))) return fail(rc);      // three gates at once (message_gates)
    if ((rc = alloc_f(s, &s->gvec, 3 * R * c.rank))) return fail(rc);
    float** rows[] = { &s->attn, &s->xq, &s->xk, &s->xv, &s->nq, &s->nk, &s->nv, &s->res, &s->mlpIn, &s->mlpGated, &s->normed, &s->ffn };
    for (auto p : rows) if ((rc = alloc_f(s, p, R * d))) return fail(rc);
    if ((rc = alloc_f(s, &s->act, R * c.topK * c.width))) return fail(rc);
    if ((rc = alloc_f(s, &s->epart, R * c.topK * d))) return fail(rc);
    if ((rc = alloc_f(s, &s->logits, c.vocab))) return fail(rc);
    if ((rc = alloc_f(s, &s->wt, R * c.topK))) return fail(rc);
    if ((rc = alloc_f(s, &s->idx, R * c.topK))) return fail(rc);
    if ((rc = alloc_f(s, &s->order, R * c.topK))) return fail(rc);
    if ((rc = alloc_f(s, &s->offsets, c.experts + 1))) return fail(rc);
    if ((rc = alloc_f(s, &s->gorder, 3 * (R * c.topK + c.experts)))) return fail(rc);
    {
        size_t most = 1;
        for (const Layer& L : x->layers) most = std::max(most, (size_t)L.heads * (size_t)std::max(L.extent, 1));
        if ((rc = alloc_f(s, &s->rb, R * most))) return fail(rc);
    }
    if ((rc = alloc_f(s, &s->ggroups, 3 * (R * c.topK + c.experts)))) return fail(rc);
    if ((rc = alloc_f(s, &s->dToks, R + 1))) return fail(rc);
    s->dPos = s->dToks + R;
    if (cudaHostAlloc((void**)&s->hToks, (R + 1) * sizeof(int), cudaHostAllocDefault) != cudaSuccess) return fail(-2000);
    if (cudaHostAlloc((void**)&s->hLogits, (size_t)c.vocab * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return fail(-2001);
    if ((rc = ok(cudaDeviceSynchronize()))) return fail(rc);
    *out = s;
    return 0;
}

INK_EXPORT long long ink_seq_bytes(void* seq) { return ((Seq*)seq)->bytes; }
INK_EXPORT int ink_seq_position(void* seq) { return ((Seq*)seq)->pos; }
INK_EXPORT int ink_seq_capacity(void* seq) { return ((Seq*)seq)->capacity; }

INK_EXPORT int ink_seq_reset(void* ctx, void* seq)
{
    ((Seq*)seq)->pos = 0;   // every row and history is written before it is read
    return 0;
}

INK_EXPORT int ink_seq_free(void* ctx, void* seq)
{
    Seq* s = (Seq*)seq;
    cudaDeviceSynchronize();
    seq_release(s);
    delete s;
    return 0;
}

// Room for `capacity` positions: each slot whose rows depend on it moves to a
// larger allocation (the rows so far are kept), and the step graph is captured
// again.
INK_EXPORT int ink_seq_grow(void* ctx, void* seq, int capacity)
{
    Ctx* x = (Ctx*)ctx;
    Seq* s = (Seq*)seq;
    if (capacity <= s->capacity) return 0;
    TRY(cudaStreamSynchronize(x->stream));
    auto drop = [&](void* p) {
        if (!p) return;
        auto it = std::find(s->owned.begin(), s->owned.end(), p);
        if (it != s->owned.end()) s->owned.erase(it);
        cudaFree(p);
    };
    for (size_t i = 0; i < s->slots.size(); i++)
    {
        Slot& sl = s->slots[i];
        const Layer& L = x->layers[phys_of(x, (int)i)];
        int rows = slot_rows(x, sl.full, capacity);
        if (rows <= sl.kv.cap) continue;
        // A ring that grows has not wrapped yet: it held fewer rows than the
        // positions so far would need only once the capacity exceeded it.
        KvSlot old = sl.kv;
        long long before = kv_bytes(old.cap, sl.C, L.kvHeads, old.format);
        if (int rc = alloc_kv(s, sl.kv, rows, sl.C, L.kvHeads)) { sl.kv = old; return rc; }
        size_t n = (size_t)std::min(s->pos, old.cap);
        size_t el = old.format == KV_F32 ? 4 : 2;
        if (old.format == KV_FP4)
        {
            TRY(cudaMemcpy(sl.kv.k, old.k, n * sl.C / 2, cudaMemcpyDeviceToDevice));
            TRY(cudaMemcpy(sl.kv.v, old.v, n * sl.C / 2, cudaMemcpyDeviceToDevice));
            TRY(cudaMemcpy(sl.kv.ks, old.ks, n * sl.C / 16, cudaMemcpyDeviceToDevice));
            TRY(cudaMemcpy(sl.kv.vs, old.vs, n * sl.C / 16, cudaMemcpyDeviceToDevice));
            TRY(cudaMemcpy(sl.kv.kg, old.kg, n * L.kvHeads * 4, cudaMemcpyDeviceToDevice));
            TRY(cudaMemcpy(sl.kv.vg, old.vg, n * L.kvHeads * 4, cudaMemcpyDeviceToDevice));
        }
        else
        {
            TRY(cudaMemcpy(sl.kv.k, old.k, n * sl.C * el, cudaMemcpyDeviceToDevice));
            TRY(cudaMemcpy(sl.kv.v, old.v, n * sl.C * el, cudaMemcpyDeviceToDevice));
        }
        drop(old.k); drop(old.v); drop(old.ks); drop(old.vs); drop(old.kg); drop(old.vg);
        s->bytes -= before;
    }
    // A larger cache may take more splits than the scratch was sized for.
    drop(s->part);
    if (int rc = alloc_f(s, &s->part, (size_t)max_parts(x, s) * 130)) return rc;
    TRY(cudaDeviceSynchronize());
    s->capacity = capacity;
    if (s->exec) { cudaGraphExecDestroy(s->exec); s->exec = nullptr; }
    if (s->graph) { cudaGraphDestroy(s->graph); s->graph = nullptr; }
    return 0;
}

// The KV bytes a sequence of `capacity` positions would hold.
INK_EXPORT long long ink_seq_estimate(void* ctx, int capacity, int format)
{
    Ctx* x = (Ctx*)ctx;
    const InkConfig& c = x->c;
    int core = c.loopEnd - c.loopStart, nslots = c.layers + (c.passes - 1) * core;
    long long total = 0;
    for (int i = 0; i < nslots; i++)
    {
        const Layer& L = x->layers[phys_of(x, i)];
        int C = L.kvHeads * L.headDim;
        total += kv_bytes(slot_rows(x, L.full, capacity), C, L.kvHeads, L.full ? format : KV_F32);
        total += 4LL * (c.convK - 1) * (2LL * C + 2LL * c.d);
    }
    return total;
}

// Feeds one token; copies the logits out when `logits` is given. `graphs`
// 0 runs the launches directly (for debugging), otherwise the step's graph.
INK_EXPORT int ink_step(void* ctx, void* seq, int token, float* logits, int graphs)
{
    Ctx* x = (Ctx*)ctx;
    Seq* s = (Seq*)seq;
    if (s->pos >= s->capacity) return -3000;
    if (token < 0 || token >= x->c.paddedVocab) return -3001;
    s->hToks[0] = token;
    s->hToks[kChunk] = s->pos;
    TRY(cudaMemcpyAsync(s->dToks, s->hToks, sizeof(int), cudaMemcpyHostToDevice, x->stream));
    TRY(cudaMemcpyAsync(s->dPos, s->hToks + kChunk, sizeof(int), cudaMemcpyHostToDevice, x->stream));
    x->err = 0;
    if (!graphs)
    {
        record(x, s, 1, false);
        if (x->err) return x->err;
        TRY(cudaGetLastError());
    }
    else
    {
        if (!s->exec)
        {
            TRY(cudaStreamBeginCapture(x->stream, cudaStreamCaptureModeThreadLocal));
            record(x, s, 1, false);
            cudaError_t e = cudaStreamEndCapture(x->stream, &s->graph);
            if (e != cudaSuccess) return -(int)e;
            if (x->err) { cudaGraphDestroy(s->graph); s->graph = nullptr; return x->err; }
            TRY(cudaGraphInstantiate(&s->exec, s->graph, 0));
        }
        TRY(cudaGraphLaunch(s->exec, x->stream));
    }
    if (logits)
    {
        TRY(cudaMemcpyAsync(s->hLogits, s->logits, (size_t)x->c.vocab * sizeof(float), cudaMemcpyDeviceToHost, x->stream));
    }
    TRY(cudaStreamSynchronize(x->stream));
    if (logits) memcpy(logits, s->hLogits, (size_t)x->c.vocab * sizeof(float));
    s->pos++;
    return 0;
}

// Feeds `count` tokens in chunks of up to kChunk; the logits of every position
// into `logits` (count x vocab) when `every`, else only the last position's
// (vocab), or none when `logits` is null. A chunk of one is a step.
INK_EXPORT int ink_prefill(void* ctx, void* seq, const int* tokens, int count, float* logits, int every, int graphs)
{
    Ctx* x = (Ctx*)ctx;
    Seq* s = (Seq*)seq;
    int vocab = x->c.vocab;
    if (s->pos + count > s->capacity) return -3000;
    for (int i = 0; i < count; i++) if (tokens[i] < 0 || tokens[i] >= x->c.paddedVocab) return -3001;
    if (every && logits && !s->every)
    {
        if (int rc = alloc_f(s, &s->every, (size_t)kChunk * vocab)) return rc;
        TRY(cudaDeviceSynchronize());
    }
    for (int done = 0; done < count;)
    {
        int T = std::min(kChunk, count - done);
        bool last = done + T == count;
        if (T == 1)
        {
            float* row = logits == nullptr ? nullptr : every ? logits + (size_t)done * vocab : last ? logits : nullptr;
            if (int rc = ink_step(ctx, seq, tokens[done], row, graphs)) return rc;
            done++;
            continue;
        }
        memcpy(s->hToks, tokens + done, (size_t)T * sizeof(int));
        s->hToks[kChunk] = s->pos;
        TRY(cudaMemcpyAsync(s->dToks, s->hToks, (size_t)T * sizeof(int), cudaMemcpyHostToDevice, x->stream));
        TRY(cudaMemcpyAsync(s->dPos, s->hToks + kChunk, sizeof(int), cudaMemcpyHostToDevice, x->stream));
        bool all = every && logits;
        x->err = 0;
        record(x, s, T, all);
        if (x->err) return x->err;
        TRY(cudaGetLastError());
        if (all)
        {
            TRY(cudaMemcpyAsync(logits + (size_t)done * vocab, s->every, (size_t)T * vocab * sizeof(float), cudaMemcpyDeviceToHost, x->stream));
        }
        else if (logits && last)
        {
            TRY(cudaMemcpyAsync(s->hLogits, s->logits, (size_t)vocab * sizeof(float), cudaMemcpyDeviceToHost, x->stream));
        }
        TRY(cudaStreamSynchronize(x->stream));
        if (!all && logits && last) memcpy(logits, s->hLogits, (size_t)vocab * sizeof(float));
        s->pos += T;
        done += T;
    }
    return 0;
}

// The output of layer visit `visit` (0 .. 47 in run order) for the first
// `rows` rows of the last chunk a prefill ran: every visit of a prompt that fit
// in one chunk, the layer outputs a reference records. Diagnostics only.
INK_EXPORT int ink_seq_visit(void* ctx, void* seq, int visit, float* out, int rows)
{
    Ctx* x = (Ctx*)ctx;
    Seq* s = (Seq*)seq;
    if (visit < 0 || visit + 1 >= (int)s->hbuf.size() || rows < 1 || rows > kChunk) return -3002;
    TRY(cudaStreamSynchronize(x->stream));
    TRY(cudaMemcpy(out, s->hbuf[visit + 1], (size_t)rows * x->c.d * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

INK_EXPORT int ink_free(void* ctx)
{
    Ctx* x = (Ctx*)ctx;
    cudaDeviceSynchronize();
    for (void* p : x->owned) cudaFree(p);
    amqe_destroy(x->eng);   // and its stream
    delete x;
    return 0;
}

INK_EXPORT const char* ink_error_string(int code)
{
    return code < 0 && code > -1000 ? cudaGetErrorString((cudaError_t)(-code)) : "amql_inkling error";
}

/// Prompts at f32 (1): projections as f32 SGEMMs, and the experts and the
/// attention as the f32 kernels here, which keeps a chunk within 1e-3 of the
/// managed runtime. 0 (the default): tensor cores, the engine's BF16 GEMMs and
/// the FP16 tiled attention (k_attn_tiles).
INK_EXPORT int ink_set_prefill_f32(void* ctx, int f32)
{
    ((Ctx*)ctx)->exactPrefill = f32 ? 1 : 0;
    return amqe_set_prefill_f32(((Ctx*)ctx)->eng, f32);
}
