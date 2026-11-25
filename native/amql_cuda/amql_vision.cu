// amql_vision.cu — what a vision tower needs beside the projections, norms and
// prompt attention the text engine already has: positions on patches, heads
// rotated by a patch's two coordinates, and the pooling into soft tokens.

#include "amql_engine.h"

// x[r] += xs[px] + ys[py]: each patch's position, from two tables of BF16 rows.
__global__ void vision_positions_kernel(float* x, const __nv_bfloat16* xs, const __nv_bfloat16* ys, const int* pos, int hidden)
{
    pdl_enter();
    const int r = blockIdx.x;
    const long long px = pos[2 * r], py = pos[2 * r + 1];
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        x[(long long)r * hidden + c] += load1(xs, px * hidden + c) + load1(ys, py * hidden + c);
}

AMQE_EXPORT int amqe_vision_positions(Engine* e, float* x, const void* xs, const void* ys, const int* pos, int rows, int hidden)
{
    launch(vision_positions_kernel, rows, 256, 0, e->stream, x, static_cast<const __nv_bfloat16*>(xs),
        static_cast<const __nv_bfloat16*>(ys), pos, hidden);
    return ok(cudaGetLastError());
}

struct VisionPrepArgs
{
    const float* qkv; int ldqkv; int heads, d;      // [rows, 3·heads·d]
    float* q; float* k; float* v;                   // [rows, heads·d] each
    const int* pos;                                 // [rows, 2]
    const float* qNormW; const float* kNormW; float eps;
    const float* invFreq;                           // [d / 4]
};

// One block per row, head and q, k or v: RMS norm (weighted for q and k), and
// for q and k the first half of the head rotated by x, the second by y.
__global__ void vision_prep_kernel(VisionPrepArgs a)
{
    pdl_enter();
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int r = blockIdx.x, kind = blockIdx.y / a.heads, h = blockIdx.y % a.heads, d = a.d;
    const float* src = a.qkv + (long long)r * a.ldqkv + ((long long)kind * a.heads + h) * d;
    float ss = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) { buf[c] = src[c]; ss += src[c] * src[c]; }
    const float inv = rsqrtf(block_sum(ss, red) / d + a.eps);
    const float* w = kind == 0 ? a.qNormW : kind == 1 ? a.kNormW : nullptr;
    for (int c = threadIdx.x; c < d; c += blockDim.x) buf[c] = buf[c] * inv * (w ? w[c] : 1.f);
    __syncthreads();
    if (kind < 2)
    {
        const int half = d / 2, n = d / 4;
        for (int i = threadIdx.x; i < 2 * n; i += blockDim.x)
        {
            const int part = i / n, p = i % n;
            const int ia = part * half + p, ib = ia + n;
            float s, c;
            sincosf((float)a.pos[2 * r + part] * a.invFreq[p], &s, &c);
            const float x1 = buf[ia], x2 = buf[ib];
            buf[ia] = x1 * c - x2 * s;
            buf[ib] = x1 * s + x2 * c;
        }
        __syncthreads();
    }
    float* out = (kind == 0 ? a.q : kind == 1 ? a.k : a.v) + ((long long)r * a.heads + h) * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) out[c] = buf[c];
}

AMQE_EXPORT int amqe_vision_prep(Engine* e, const float* qkv, int ldqkv, int rows, int heads, int d,
    float* q, float* k, float* v, const int* pos, const float* qNormW, const float* kNormW, float eps, const float* invFreq)
{
    if (d % 4 != 0) return -100;
    VisionPrepArgs a{ qkv, ldqkv, heads, d, q, k, v, pos, qNormW, kNormW, eps, invFreq };
    launch(vision_prep_kernel, dim3(rows, 3 * heads), 64, d * sizeof(float), e->stream, a);
    return ok(cudaGetLastError());
}

// Soft token t: the mean of its side x side patches (a grid `width` patches wide),
// times scale, shifted and scaled per channel, and RMS-normalised without a weight.
__global__ void vision_pool_kernel(const float* x, float* out, int hidden, int width, int side, float scale,
    const float* bias, const float* stdScale, float eps)
{
    pdl_enter();
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int t = blockIdx.x, across = width / side;
    const int ty = t / across, tx = t % across;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
    {
        float sum = 0.f;
        for (int j = 0; j < side; j++)
            for (int i = 0; i < side; i++)
                sum += x[((long long)(ty * side + j) * width + tx * side + i) * hidden + c];
        float v = sum / (side * side) * scale;
        if (bias) v = (v - bias[c]) * stdScale[c];
        buf[c] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / hidden + eps);
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) out[(long long)t * hidden + c] = buf[c] * inv;
}

AMQE_EXPORT int amqe_vision_pool(Engine* e, const float* x, float* out, int tokens, int hidden, int width, int side, float scale,
    const float* bias, const float* stdScale, float eps)
{
    size_t smem = (size_t)hidden * sizeof(float);
    if (smem > 44 * 1024) return -101;
    launch(vision_pool_kernel, tokens, 256, smem, e->stream, x, out, hidden, width, side, scale, bias, stdScale, eps);
    return ok(cudaGetLastError());
}
