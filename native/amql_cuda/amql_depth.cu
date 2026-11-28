// amql_depth.cu — the engine's kernels for looped depth and its residuals, and
// for attention conditioned without a rotary embedding.
//
// Primitives the managed engine (src/Amql.Inference/GpuEngine.Visits.cs) calls
// for the plan ops GenericRuntime's row-exact path runs on the CPU
// (docs/depth-ops.md), each on the engine's stream so a decode step's calls are
// captured into one CUDA graph:
//
//   depth mixes over a bank of block inputs, with the norm that follows fused
//   (DepthMixOp, BlockResidualOp); message gates, up to three batched
//   (MessageGateOp); residual short convolutions, static or with dynamic taps on
//   keys and values (ResidualConvOp), with the key clip and the gainless value
//   RMS; the query and relative-coefficient preparation (QueryChannelScale,
//   QkClip, LogScaling); attention with a relative score bias, a sink logit and
//   a separate output gate, split-key for a step and FlashAttention-2 tiles on
//   the tensor cores for a prompt chunk; sigmoid routing normalised over the
//   chosen experts with a selection bias and a route scale; packed experts in
//   their stored formats (NVFP4, MXFP8, plain) through a softcapped GLU; an
//   embedding with a weighted norm, the head's softcap.
//
// Projections that share a launch run as GEMVs for one row and as the engine's
// GEMMs (amqe_linear) for a chunk; a chunk's experts are the engine's grouped
// tensor-core experts (amqe_moe_experts_paired) and a step's its decode kernels
// (amqe_moe_experts_paired_step). With prompts kept at f32 the engine's SGEMMs
// and the f32 expert kernels here run instead. Convolution histories and cache
// rows advance token by token inside their kernels, reading the first position
// from device memory.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#if defined(_WIN32)
  #define DEPTH_EXPORT extern "C" __declspec(dllexport)
#else
  #define DEPTH_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define TRY(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) return -(int)_e; } while (0)

// -- the engine (amql_engine.cu, amql_moe.cu), in the same library --

struct Engine;
extern "C" cudaStream_t amqe_stream(Engine* e);
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

// Weight forms, the engine's codes: BF16, f32, NVFP4 (E2M1 + E4M3 per 16 + a
// factor per expert), MXFP8 (E4M3 + E8M0 per 32).
enum { W_BF16 = 0, W_F32 = 1, W_NVFP4 = 2, W_MXFP8 = 4 };
enum { KV_F32 = 0, KV_F16 = 1, KV_FP4 = 2 };

// A matrix [rows x cols], BF16 or f32 (GpuEngine.DepthMat).
struct Mat { const void* w; int type; int rows; int cols; int pad; };
// An expert stack [experts x rows x cols] as stored (GpuEngine.DepthExperts).
// rowScale: NVFP4's factor for every row, [experts x rows], for the engine's kernels.
struct Experts { const void* w; const unsigned char* scales; const float* factor; const float* rowScale; int type; int rows; int cols; int pad; };

// A cache slot's key and value rows (GpuEngine.DepthKv).
struct KvView
{
    const void* k; const void* v;                  // rows x C elements (FP4: rows x C/2 bytes)
    const unsigned char* ks; const unsigned char* vs; // FP4: rows x C/16 E4M3 block scales
    const float* kg; const float* vg;               // FP4: rows x kvHeads tensor scales
    int cap;                                       // rows held
    int ring;                                      // 1: a sliding window, row = position % cap
    int format;
    int pad;
};

namespace {

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

// h = norm(embed[token]): RMS-normed with `norm` as the gain (null: none) when
// `normed`, else the row as it is.
__global__ void k_embed(Mat embed, const int* __restrict__ token, const float* __restrict__ norm, float* __restrict__ out, float eps,
                        int normed)
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
    if (!normed) return;
    float inv = rsqrtf(block_sum(s, sh) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) out[i] = out[i] * inv * (norm ? norm[i] : 1.f);
}

// A depth mix over a bank of states (a block residual, a message): each state
// RMS-normed without a gain (eps), weighted by softmax(query . state / sqrt d).
struct States { const float* s[10]; int n; float eps; };
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
        inv[i] = rsqrtf(a / d + st.eps);
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
        float inv = rsqrtf(a / d + st.eps);
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
// coefficients): a learned scale per channel (null: none), L2-clipped to `clip`
// (inf: none), then the position log scaling where `logScale` is set.
// Grid (heads, T, 1 or 2).
struct HeadPreps { float* x[2]; const float* scale[2]; int dim[2]; float clip[2]; };

__global__ void k_head_prep(HeadPreps a, int logScale, const int* __restrict__ pos, int floor, float alpha)
{
    __shared__ float sh[32];
    const int z = blockIdx.z, dim = a.dim[z];
    const float clip = a.clip[z];
    float* hx = a.x[z] + ((long long)blockIdx.y * gridDim.x + blockIdx.x) * dim;
    const float* sc = a.scale[z] ? a.scale[z] + blockIdx.x * dim : nullptr;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = hx[i] * (sc ? sc[i] : 1.f); hx[i] = v; s += v * v; }
    float norm = sqrtf(block_sum(s, sh));
    float f = fminf(1.f, clip / (norm + 1e-6f));
    if (logScale)
    {
        float t = (float)(*pos + blockIdx.y + 1);
        f *= 1.f + alpha * logf(fmaxf(1.f, t / (float)floor));
    }
    if (f != 1.f)
        for (int i = threadIdx.x; i < dim; i += blockDim.x) hx[i] *= f;
}

// ── the key/value rows ───────────────────────────────────────────────────────

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

// The residual convolutions, clip and norm of a chunk's keys and values, then
// the rows stored, token by token. Grid (kvHeads, 2): y = 0 keys, 1 values. A
// conv is none, static (depthwise [C x K], the last tap on the current token)
// or dynamic (static taps [K x C] plus one per head and lag projected from the
// projection's input). The histories hold the projections before the conv.
// Keys are L2-clipped per head (clip = inf: none); values RMS-normed per head
// without a gain when vnorm is set.
enum { CONV_NONE = 0, CONV_STATIC = 1, CONV_DYNAMIC = 2 };

struct KvPrep
{
    const float* raw[2];       // projections
    const float* dyn[2];       // dynamic: [kvHeads x K] taps per token
    const float* conv[2];      // static [C x K]; dynamic: static taps [K x C]
    float* hist[2];            // [(K-1) x C]
    KvView slot;
    void* dst[2]; unsigned char* dsc[2]; float* dg[2];
    int C, hd, K, count, kvHeads, vnorm;
    int mode[2];
    float clip, eps;
};

__global__ void k_kv_prep(KvPrep a, const int* __restrict__ pos)
{
    __shared__ float sh[32];
    __shared__ float vals[256];
    int which = blockIdx.y, head = blockIdx.x, i = threadIdx.x;
    int K = a.K, C = a.C, hd = a.hd;
    int ch = head * hd + i;
    const int mode = a.mode[which];
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
        if (mode == CONV_NONE)
        {
            out = x;
        }
        else if (mode == CONV_STATIC)
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
        if (mode != CONV_NONE) hist[(t % (K - 1)) * C + ch] = x;
    }
    float ss = block_sum(i < hd ? out * out : 0.f, sh);
    if (which == 0)
    {
        float f = fminf(1.f, a.clip / (sqrtf(ss) + 1e-6f));
        out *= f;
    }
    else if (a.vnorm)
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

// The splits merged with the sink logit (sinks null: none; sinkStride 0: one
// logit per head for every row), then the sigmoid output gate (null: none).
__global__ void k_attn_merge(const float* __restrict__ part, const float* __restrict__ sinks, int sinkStride,
                             const float* __restrict__ gate, float* __restrict__ out, int splits, int hd)
{
    int h = blockIdx.x, z = blockIdx.y, heads = gridDim.x;
    const float* p = part + ((long long)z * heads + h) * splits * (hd + 2);
    if (sinks) sinks += (long long)z * sinkStride;
    if (gate) gate += (long long)z * heads * hd;
    out += (long long)z * heads * hd;
    float M = sinks ? sinks[h] : -INFINITY;
    for (int s = 0; s < splits; s++) M = fmaxf(M, p[s * (hd + 2)]);
    float L = sinks ? expf(sinks[h] - M) : 0.f;
    for (int s = 0; s < splits; s++) { float ms = p[s * (hd + 2)]; if (ms > -INFINITY) L += p[s * (hd + 2) + 1] * expf(ms - M); }
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
    {
        float v = 0.f;
        for (int s = 0; s < splits; s++) { float ms = p[s * (hd + 2)]; if (ms > -INFINITY) v += p[s * (hd + 2) + 2 + i] * expf(ms - M); }
        out[h * hd + i] = gate ? v / L * (1.f / (1.f + expf(-gate[h * hd + i]))) : v / L;
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

// A residual short convolution over this slot's history, depthwise and causal,
// (conv + x) / sqrt(K + 1), plus `add` when given; no weights: x + add.
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
        if (w == nullptr) { out[o] = xv + (add ? add[o] : 0.f); continue; }
        float sum = w[ch * K + K - 1] * xv;
        for (int lag = 1; lag < K; lag++)
            if (t - lag >= 0) sum += w[ch * K + K - 1 - lag] * hist[((t - lag) % (K - 1)) * C + ch];
        hist[(t % (K - 1)) * C + ch] = xv;
        out[o] = (sum + xv) * rsqrtf((float)(K + 1)) + (add ? add[o] : 0.f);
    }
}

// Sigmoid routing normalised over the chosen: sigmoid scores, the top k by
// score + selection bias (ties to the lower index), the chosen scores
// renormalised (sum + 1e-20), times the route scale. One warp a row.
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
    for (int j = 0; j < 16; j++) { int e = lane + 32 * j; key[j] = e < experts ? score[e] + (bias ? bias[e] : 0.f) : -INFINITY; }
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
// the softcapped GLU. Grid (topK, rows / 8 of the width), a warp an output.
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

// The chosen experts' weighted outputs summed in their chosen order.
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

// ── host helpers ─────────────────────────────────────────────────────────────

// Projections that share nothing but a launch: one row (a step) as GEMVs, T
// rows (a chunk) as the engine's GEMMs (BF16 tensor cores, or f32 SGEMM with
// prompts kept at f32).
int proj(Engine* e, const Mat* m, const float* const* x, float* const* y, int n, int T)
{
    cudaStream_t st = amqe_stream(e);
    if (n < 1 || n > 4) return -4000;
    int first[5] = { 0 };
    for (int i = 0; i < n; i++) first[i + 1] = first[i] + m[i].rows;
    int rows = first[n];
    if (T == 1 && n == 1 && rows <= 256)
    {
        k_gemv_rows<<<rows, 256, 0, st>>>(m[0], x[0], y[0]);
        return 0;
    }
    if (T == 1)
    {
        Gemvs v = {};
        for (int i = 0; i < n; i++) { v.m[i] = m[i]; v.x[i] = x[i]; v.y[i] = y[i]; }
        for (int i = 0; i <= n; i++) v.first[i] = first[i];
        v.n = n;
        k_gemvs<<<(rows + 7) / 8, 256, 0, st>>>(v);
        return 0;
    }
    for (int i = 0; i < n; i++)
    {
        const Mat& w = m[i];
        int rc = w.type == W_BF16 || w.type == W_F32
            ? amqe_linear(e, w.w, w.type, w.rows, w.cols, x[i], w.cols, T, y[i], w.rows,
                          nullptr, nullptr, 0, 1.f, 0, 0, nullptr, nullptr, 0.f, nullptr, 0.f, 0, nullptr, nullptr)
            : -100;
        if (rc == -100)
        {
            Gemms one = {};
            one.m[0] = w; one.x[0] = x[i]; one.y[0] = y[i];
            one.first[1] = w.rows; one.n = 1; one.T = T;
            k_gemms<<<(w.rows + 7) / 8, 256, 0, st>>>(one);
        }
        else if (rc) return rc;
    }
    return 0;
}

int mix(cudaStream_t st, const float* const* states, int n, const float* queries, int count, float* out, int T, int d, float eps,
        const float* const* normW, float* const* normOut, float normEps)
{
    if (n < 1 || n > 10 || count < 1 || count > 4) return -4001;
    States s = {};
    s.n = n;
    s.eps = eps;
    for (int i = 0; i < n; i++) s.s[i] = states[i];
    MixNorm mn = {};
    mn.eps = normEps;
    if (normOut)
        for (int i = 0; i < count; i++) { mn.w[i] = normW ? normW[i] : nullptr; mn.out[i] = normOut[i]; }
    // The float4 kernel when every vector it reads or writes is 16-byte aligned.
    auto al = [](const void* p) { return ((uintptr_t)p & 15) == 0; };
    bool fit = d % 4 == 0 && d / 4 <= 1024 && al(queries) && al(out);
    for (int i = 0; i < n; i++) fit = fit && al(s.s[i]);
    for (int i = 0; i < count; i++) fit = fit && al(mn.w[i]) && al(mn.out[i]);
    if (fit)
        k_mix4<<<dim3(count, T), ((d / 4 + 31) / 32) * 32, 0, st>>>(s, queries, out, d, mn);
    else
        k_mix<<<dim3(count, T), 1024, 0, st>>>(s, queries, out, d, mn);
    return 0;
}

} // namespace

// ── exports ──────────────────────────────────────────────────────────────────

/// Up to four projections in one launch: y[i] = x[i] · m[i]ᵀ for T rows each.
DEPTH_EXPORT int amqe_projections(Engine* e, const Mat* m, const float* const* x, float* const* y, int n, int T)
{
    return proj(e, m, x, y, n, T);
}

/// Up to three RMS norms of T rows of d in one launch: out[i] = x[i] / rms · w[i] (null: no gain).
DEPTH_EXPORT int amqe_rms_rows(Engine* e, const float* const* x, const float* const* w, float* const* out, int n, int T, int d, float eps)
{
    if (n < 1 || n > 3) return -4002;
    Norms a = {};
    for (int i = 0; i < n; i++) { a.x[i] = x[i]; a.w[i] = w ? w[i] : nullptr; a.out[i] = out[i]; }
    k_rmsnorm<<<dim3(T, n), 256, 0, amqe_stream(e)>>>(a, d, eps);
    return 0;
}

/// The depth mix of `n` states (the bank, then the current state; each T x d)
/// for `count` queries [count x d]: out[q][t] (query-major, T x d each). With
/// normOut, query q's mix is also written RMS-normed with normW[q] (null: no
/// gain) to normOut[q], in the same kernel.
DEPTH_EXPORT int amqe_depth_mix(Engine* e, const float* const* states, int n, const float* queries, int count, float* out,
                                int T, int d, float eps, const float* const* normW, float* const* normOut, float normEps)
{
    return mix(amqe_stream(e), states, n, queries, count, out, T, d, eps, normW, normOut, normEps);
}

/// Message gates on `n` input sites (1 to 3, independent, so each of their three
/// steps is one launch): out[i] = (1 - tanh(F g)) h[i] + tanh(G g) m[i], g = D [rms h; rms m].
/// cat: n x stride x 2d and g: n x stride x rank scratch (stride >= T rows).
DEPTH_EXPORT int amqe_message_gates(Engine* e, int n, const float* const* h, const float* const* m, float* const* out,
                                    const Mat* down, const Mat* forget, const Mat* gain, float* cat, float* g, int stride,
                                    int T, int d, int rank, float eps)
{
    if (n < 1 || n > 3 || T > stride) return -4003;
    cudaStream_t st = amqe_stream(e);
    GateCats gc = {};
    GateApplies ga = {};
    const float* cats[3];
    float* gs[3];
    for (int i = 0; i < n; i++)
    {
        float* ci = cat + (long long)i * stride * 2 * d;
        gs[i] = g + (long long)i * stride * rank;
        cats[i] = ci;
        gc.h[i] = h[i]; gc.m[i] = m[i]; gc.cat[i] = ci;
        ga.forget[i] = forget[i]; ga.gain[i] = gain[i];
        ga.g[i] = gs[i]; ga.h[i] = h[i]; ga.m[i] = m[i]; ga.out[i] = out[i];
    }
    k_gate_cat<<<dim3(T, n), 256, 0, st>>>(gc, d, eps);
    if (int rc = proj(e, down, cats, gs, n, T)) return rc;
    k_gate_apply<<<dim3((d + 7) / 8, T, n), 256, 0, st>>>(ga, d, rank);
    return 0;
}

/// Queries [T x heads x qDim] (and relative coefficients [T x heads x rDim] when r
/// is given), in place: a channel scale (null: none), an L2 clip per head (inf:
/// none), then 1 + alpha ln(max(1, (t+1)/floor)) where logScale is set.
DEPTH_EXPORT int amqe_head_prep(Engine* e, float* q, const float* qScale, int qDim, float qClip,
                                float* r, const float* rScale, int rDim, float rClip,
                                int heads, int T, int logScale, const int* pos, int floor, float alpha)
{
    HeadPreps hp = {};
    hp.x[0] = q; hp.scale[0] = qScale; hp.dim[0] = qDim; hp.clip[0] = qClip;
    hp.x[1] = r; hp.scale[1] = rScale; hp.dim[1] = rDim; hp.clip[1] = rClip;
    k_head_prep<<<dim3(heads, T, r ? 2 : 1), 128, 0, amqe_stream(e)>>>(hp, logScale, pos, floor, alpha);
    return 0;
}

/// A chunk's keys and values (projections kRaw, vRaw [T x C]) through their
/// residual convolutions (mode 0 none, 1 static, 2 dynamic with taps kDyn, vDyn
/// [T x kvHeads x K]), the key clip and the value norm, into the slot's rows at
/// positions *pos.. (ring: modulo its rows).
DEPTH_EXPORT int amqe_kv_prep(Engine* e, const KvView* slot, const float* kRaw, const float* vRaw,
                              const float* kDyn, const float* vDyn, const float* kConv, const float* vConv,
                              float* kHist, float* vHist, int kMode, int vMode,
                              int C, int hd, int K, int kvHeads, int T, float clip, int vnorm, float eps, const int* pos)
{
    if (hd > 128 || ((kMode || vMode) && K < 2)) return -4004;
    KvPrep kp = {};
    kp.raw[0] = kRaw; kp.raw[1] = vRaw;
    kp.dyn[0] = kDyn; kp.dyn[1] = vDyn;
    kp.conv[0] = kConv; kp.conv[1] = vConv;
    kp.hist[0] = kHist; kp.hist[1] = vHist;
    kp.slot = *slot;
    kp.dst[0] = (void*)slot->k; kp.dst[1] = (void*)slot->v;
    kp.dsc[0] = (unsigned char*)slot->ks; kp.dsc[1] = (unsigned char*)slot->vs;
    kp.dg[0] = (float*)slot->kg; kp.dg[1] = (float*)slot->vg;
    kp.C = C; kp.hd = hd; kp.K = K; kp.count = T; kp.kvHeads = kvHeads; kp.vnorm = vnorm;
    kp.mode[0] = kMode; kp.mode[1] = vMode;
    kp.clip = clip; kp.eps = eps;
    k_kv_prep<<<dim3(kvHeads, 2), 128, 0, amqe_stream(e)>>>(kp, pos);
    return 0;
}

/// Key splits one row's attention takes over a slot: a ring a split per 64
/// rows up to 16; full rows one per 128 up to 64.
DEPTH_EXPORT int amqe_depth_attention_splits(int cap, int ring)
{
    if (ring) return std::max(1, std::min(16, (cap + 63) / 64));
    return std::max(1, std::min(64, cap / 128));
}

/// Split-key attention of T rows at positions *pos.. over a slot, into the split
/// partials `part` [T x heads x splits x (hd + 2)]: per query an online softmax
/// over its keys (the window's when window > 0), the relative bias r · P[:, dist]
/// within `extent` of the query (extent 0: none). tiled: FlashAttention-2 on the
/// tensor cores (FP16 tiles, f32 sums) for a chunk, the bias first made as a
/// table rb [T x heads x extent].
DEPTH_EXPORT int amqe_depth_attention(Engine* e, const float* q, const float* r, const float* relProj, float* rb,
                                      const KvView* kv, float* part, int heads, int kvHeads, int hd, int dRel, int extent,
                                      int window, int splits, float scale, int T, int tiled, const int* pos)
{
    cudaStream_t st = amqe_stream(e);
    if (hd > 128 || heads % kvHeads != 0 || dRel > 32 || (extent > 0 && (r == nullptr || relProj == nullptr))) return -4005;
    if (tiled)
    {
        if (!(hd == 16 || hd == 32 || hd == 64 || hd == 128) || heads > 64) return -4006;
        if (extent > 0)
            k_relbias<<<dim3(T, heads), 256, 0, st>>>(r, relProj, rb, heads, dRel, extent);
        TileArgs ta = {};
        ta.q = q; ta.rb = rb; ta.kv = *kv; ta.part = part;
        ta.heads = heads; ta.kvHeads = kvHeads; ta.extent = extent; ta.window = window; ta.splits = splits; ta.T = T;
        ta.scale = scale;
        dim3 grid(heads, splits, (T + kTQ - 1) / kTQ);
        switch (hd)
        {
            case 16: k_attn_tiles<16><<<grid, 128, 0, st>>>(ta, pos); break;
            case 32: k_attn_tiles<32><<<grid, 128, 0, st>>>(ta, pos); break;
            case 64: k_attn_tiles<64><<<grid, 128, 0, st>>>(ta, pos); break;
            default: k_attn_tiles<128><<<grid, 128, 0, st>>>(ta, pos); break;
        }
        return 0;
    }
    AttnArgs aa = {};
    aa.q = q; aa.r = r; aa.relProj = relProj;
    aa.kv = *kv;
    aa.part = part;
    aa.heads = heads; aa.kvHeads = kvHeads; aa.hd = hd; aa.dRel = dRel; aa.extent = extent;
    aa.window = window; aa.splits = splits; aa.scale = scale;
    k_attn<<<dim3(heads, splits, T), 128, 0, st>>>(aa, pos);
    return 0;
}

/// The split partials merged with each head's sink logit (sinks null: none;
/// sinkStride the logits' row stride, 0 for one per head), then the sigmoid
/// output gate (null: none): out [T x heads x hd].
DEPTH_EXPORT int amqe_depth_attention_merge(Engine* e, const float* part, const float* sinks, int sinkStride, const float* gate,
                                            float* out, int heads, int splits, int hd, int T)
{
    k_attn_merge<<<dim3(heads, T), 128, 0, amqe_stream(e)>>>(part, sinks, sinkStride, gate, out, splits, hd);
    return 0;
}

/// out = conv(x) + add, a residual short convolution over this slot's history
/// [(K-1) x C] for T rows at positions *pos.. (w null: out = x + add); add may be null.
DEPTH_EXPORT int amqe_residual_conv(Engine* e, const float* x, const float* w, float* hist, const float* add, float* out,
                                    int C, int K, int T, const int* pos)
{
    if (w && K < 2) return -4007;
    k_shortconv<<<(C + 255) / 256, 256, 0, amqe_stream(e)>>>(x, w, hist, add, out, C, K, pos, T);
    return 0;
}

/// Sigmoid routing normalised over the chosen experts, T rows of router logits
/// [T x experts]: idx, wt [T x topK]; bias (null: none) chooses only.
DEPTH_EXPORT int amqe_route_sigmoid_selected(Engine* e, const float* logits, const float* bias, int experts, int topK,
                                             float routeScale, int* idx, float* wt, int T)
{
    if (experts > 512 || topK > 32) return -4008;
    k_router<<<T, 32, 0, amqe_stream(e)>>>(logits, bias, experts, topK, routeScale, idx, wt);
    return 0;
}

/// Packed experts through a softcapped GLU (gate and up rows interleaved, gate
/// j = row 2j: cap(g)·sigmoid(g)·cap(u)) for T rows of x [T x d] routed to idx with
/// weights wt [T x topK]: ffn [T x d]. A step's experts on the engine's decode
/// kernels, a chunk's grouped by expert on its tensor cores, and with prompts at
/// f32 (or a form or shape the engine's kernels do not take) the f32 kernels here.
/// Scratch: act [T x topK x width], part [T x topK x d], order [T x topK],
/// offsets [experts + 1], gorder and ggroups [3 x (T x topK + experts)].
DEPTH_EXPORT int amqe_packed_experts(Engine* e, const Experts* gateUp, const Experts* down, int experts, int d, int width, int topK,
                                     float gateCap, float upCap, const float* x, int T, const int* idx, const float* wt,
                                     float* act, float* part, float* ffn, int* order, int* offsets, int* gorder, int* ggroups)
{
    cudaStream_t st = amqe_stream(e);
    if (experts > 512 || topK > 32) return -4009;
    const Experts& w13 = *gateUp;
    const Experts& w2 = *down;
    bool grouped = T > 1 && d % 32 == 0 && width % 32 == 0 && d % 8 == 0;
    if (!grouped && T <= 8)
    {
        int rc = amqe_moe_experts_paired_step(e, w13.w, w13.type, w13.scales, w13.rowScale, w2.w, w2.type, w2.scales, w2.rowScale,
                                              experts, d, width, topK, gateCap, upCap, x, d, T, idx, act, part);
        if (rc == 0)
        {
            k_expert_wsum<<<dim3((d + 255) / 256, T), 256, 0, st>>>(part, wt, topK, d, ffn);
            return 0;
        }
        if (rc != -100) return rc;
    }
    if (!grouped)
    {
        for (int t = 0; t < T; t++)
        {
            k_expert_up<<<dim3(topK, (width + 7) / 8), 256, 0, st>>>(w13, x + (long long)t * d, idx + t * topK,
                                                                     act + (long long)t * topK * width, width, gateCap, upCap);
            k_expert_down<<<dim3(topK, (d + 7) / 8), 256, 0, st>>>(w2, act + (long long)t * topK * width, idx + t * topK,
                                                                  wt + t * topK, part + (long long)t * topK * d, d, width);
        }
        k_expert_sum<<<dim3((d + 255) / 256, T), 256, 0, st>>>(part, topK, d, ffn);
        return 0;
    }
    int gu = w13.type == W_F32 ? -1 : w13.type, dn = w2.type == W_F32 ? -1 : w2.type;
    int rc = gu < 0 || dn < 0 ? -100
        : amqe_moe_experts_paired(e, w13.w, gu, w13.scales, w13.rowScale, w2.w, dn, w2.scales, w2.rowScale,
                                  experts, d, width, topK, gateCap, upCap, x, d, T, idx, gorder, ggroups, part);
    if (rc == 0)
    {
        k_expert_wsum<<<dim3((d + 255) / 256, T), 256, 0, st>>>(part, wt, topK, d, ffn);
        return 0;
    }
    if (rc != -100) return rc;
    k_group<<<1, 1024, 0, st>>>(idx, T * topK, experts, offsets, order);
    k_expert_up_g<<<dim3(experts, (width + 31) / 32), 256, 0, st>>>(w13, x, offsets, order, topK, act, width, gateCap, upCap);
    k_expert_down_g<<<dim3(experts, (d + 63) / 64), 256, 0, st>>>(w2, act, offsets, order, wt, part, d, width);
    k_expert_sum<<<dim3((d + 255) / 256, T), 256, 0, st>>>(part, topK, d, ffn);
    return 0;
}

/// NVFP4's per-expert factor as the engine's row scale: out[e x rows + r] = factor[e].
DEPTH_EXPORT int amqe_expert_row_scale(Engine* e, const float* factor, int rows, long long count, float* out)
{
    k_expand_factor<<<(unsigned)((count + 255) / 256), 256, 0, amqe_stream(e)>>>(factor, rows, count, out);
    return 0;
}

/// T tokens' embedding rows, RMS-normed with normW (null: no gain) when normed.
DEPTH_EXPORT int amqe_embed_rows(Engine* e, const Mat* table, const int* tokens, const float* normW, int normed, float eps,
                                 float* out, int T)
{
    k_embed<<<T, 256, 0, amqe_stream(e)>>>(*table, tokens, normW, out, eps, normed);
    return 0;
}

/// x = cap · tanh(x / cap), n values.
DEPTH_EXPORT int amqe_softcap(Engine* e, float* x, long long n, float cap)
{
    k_softcap<<<(unsigned)((n + 255) / 256), 256, 0, amqe_stream(e)>>>(x, (int)n, cap);
    return 0;
}
