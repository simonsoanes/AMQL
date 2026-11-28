// amql_moe.cu — routed experts for the device-resident engine (amql_engine.cu).

#include "amql_engine.h"

// ── MXFP8 weights ──────────────────────────────────────────────────────────
//
// E4M3 elements, an E8M0 exponent per 32 (packed experts stored MXFP8).
// The tag picks the reader, as Fp4Nv and Fp4Mx do; a row is k bytes.

struct Fp8Mx { unsigned char b; };

__device__ __forceinline__ void loadw(const Fp8Mx* w, const unsigned char* sc, long long idx, float* o)
{
    const uint2 q = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const unsigned char*>(w) + idx));
    const float s = e8m0_value(__ldg(sc + (idx >> 5)));
    const unsigned int words[2] = { q.x, q.y };
    #pragma unroll
    for (int k = 0; k < 2; k++)
    {
        __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] & 0xffffu), __NV_E4M3);
        __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[k] >> 16), __NV_E4M3);
        const float2 a = __half22float2(*reinterpret_cast<__half2*>(&lo)), b = __half22float2(*reinterpret_cast<__half2*>(&hi));
        o[4 * k] = a.x * s; o[4 * k + 1] = a.y * s; o[4 * k + 2] = b.x * s; o[4 * k + 3] = b.y * s;
    }
}

__device__ __forceinline__ float loadw1(const Fp8Mx* w, const unsigned char* sc, long long idx)
{
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)__ldg(reinterpret_cast<const unsigned char*>(w) + idx), __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&h)) * e8m0_value(__ldg(sc + (idx >> 5)));
}

// The softcapped GLU (ExpertGateSoftcappedGlu): cap(g) . sigmoid(g) . cap(u), cap(v) = c . tanh(v / c).
__device__ __forceinline__ float softcapped_glu(float g, float u, float gateCap, float upCap)
{
    return tanhf(g / gateCap) * gateCap / (1.f + expf(-g)) * (tanhf(u / upCap) * upCap);
}

// ── timing marks ──────────────────────────────────────────────────────────
//
// An event after each device call of a captured step: the time between two is
// what the call took on the GPU. For profiling alone.

struct Marks { cudaEvent_t at[4096]; int n = 0, made = 0; };
static Marks* gMarks = nullptr;

AMQE_EXPORT int amqe_mark(Engine* e)
{
    if (!gMarks || gMarks->n >= 4096) return 0;
    if (gMarks->n == gMarks->made)
    {
        if (cudaEventCreate(&gMarks->at[gMarks->made]) != cudaSuccess) return -1;
        gMarks->made++;
    }
    return ok(cudaEventRecord(gMarks->at[gMarks->n++], e->stream));
}

AMQE_EXPORT int amqe_marks_begin(Engine* e)
{
    if (!gMarks) gMarks = new Marks();
    gMarks->n = 0;
    return 0;
}

/// milliseconds[k] = from mark k to the one before it (mark 0: 0). Returns how many.
AMQE_EXPORT int amqe_marks_end(Engine* e, float* milliseconds, int most)
{
    if (!gMarks) return 0;
    int n = gMarks->n < most ? gMarks->n : most;
    for (int k = 0; k < n; k++)
    {
        milliseconds[k] = 0.f;
        if (k > 0 && cudaEventElapsedTime(&milliseconds[k], gMarks->at[k - 1], gMarks->at[k]) != cudaSuccess) milliseconds[k] = 0.f;
    }
    return n;
}

// ── routed experts ─────────────────────────────────────────────────────────
//
// A layer's experts are stacked into one matrix a projection: expert e's rows
// are e·rowsPerExpert on. Each row routes to topK experts, a (row, slot) pair
// each. Pairs of one expert run together in groups of up to R, so a prompt
// reads an expert's weights once for R pairs rather than once a pair; a
// decode step's pairs mostly have experts of their own. A pair's output is the
// same sum in the same order whatever group it falls in, so a token's result
// does not depend on what ran beside it.

struct MoeRouteArgs
{
    const void* w; int experts; int hidden; int topK;
    const float* x; int ldx;
    float normEps;            // > 0: what the router reads is x RMS-normalised (no weight)
    const float* inScale;     // [hidden] times 1/sqrt(hidden), or null
    const float* expertScale; // [experts], or null
    int normaliseSelected;
    int* idx; float* wt;      // [rows x topK]
    // With an input norm for the experts, x normalised by it is written here too
    // (the sum of squares is taken once for both).
    const float* inNormW; float inNormOff; float inNormEps; float* normed;
};

template <typename WT>
__global__ void __launch_bounds__(1024) moe_route_kernel(MoeRouteArgs a)
{
    pdl_trigger();
    {
        // The router never changes: into L2 while the kernel before is finishing.
        const char* w = static_cast<const char*>(a.w);
        const long long bytes = (long long)a.experts * a.hidden * sizeof(WT);
        for (long long off = (long long)threadIdx.x * 128; blockIdx.x == 0 && off < bytes; off += 128LL * blockDim.x)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(w + off));
    }
    pdl_wait();
    extern __shared__ float buf[];        // hidden + experts floats
    __shared__ float red[32];
    __shared__ int pick[64];
    const int r = blockIdx.x, H = a.hidden, E = a.experts;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int WARPS = blockDim.x >> 5;
    constexpr int RW = 4;
    float* xs = buf;
    float* p = buf + H;
    const float* xr = a.x + (long long)r * a.ldx;
    float inv = 1.f;
    if (a.normEps > 0.f || a.inNormW)
    {
        float ss = 0.f;
        for (int c = threadIdx.x; c < H; c += blockDim.x) ss += xr[c] * xr[c];
        const float mean = block_sum(ss, red) / H;
        if (a.normEps > 0.f) inv = rsqrtf(mean + a.normEps);
        if (a.inNormW)
        {
            const float ninv = rsqrtf(mean + a.inNormEps);
            float* out = a.normed + (long long)r * H;
            for (int c = threadIdx.x; c < H; c += blockDim.x) out[c] = xr[c] * ninv * (a.inNormW[c] + a.inNormOff);
        }
    }
    const float root = a.inScale ? rsqrtf((float)H) : 1.f;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        xs[c] = xr[c] * inv * (a.inScale ? a.inScale[c] * root : 1.f);
    __syncthreads();
    const WT* W = static_cast<const WT*>(a.w);
    const int kv = H & ~7;
    for (int e0 = warp * RW; e0 < E; e0 += WARPS * RW)
    {
        float t[RW];
        #pragma unroll
        for (int q = 0; q < RW; q++) t[q] = 0.f;
        #pragma unroll 4
        for (int c = lane * 8; c < kv; c += 256)
        {
            float wv[RW][8];
            #pragma unroll
            for (int q = 0; q < RW; q++) load8(W, (long long)min(e0 + q, E - 1) * H + c, wv[q]);
            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
                const float v = xs[c + i];
                #pragma unroll
                for (int q = 0; q < RW; q++) t[q] = fmaf(wv[q][i], v, t[q]);
            }
        }
        for (int c = kv + lane; c < H; c += 32)
            #pragma unroll
            for (int q = 0; q < RW; q++) t[q] = fmaf(load1(W, (long long)min(e0 + q, E - 1) * H + c), xs[c], t[q]);
        #pragma unroll
        for (int q = 0; q < RW; q++)
        {
            const float v = warp_sum(t[q]);
            if (lane == 0 && e0 + q < E) p[e0 + q] = v;
        }
    }
    __syncthreads();
    float m = -INFINITY;
    for (int e = threadIdx.x; e < E; e += blockDim.x) m = fmaxf(m, p[e]);
    m = block_max(m, red);
    float sum = 0.f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) { p[e] = __expf(p[e] - m); sum += p[e]; }
    sum = block_sum(sum, red);   // its barriers also publish p
    // Highest first, an earlier expert first among equals: an expert's place is
    // the number that come before it.
    for (int e = threadIdx.x; e < E; e += blockDim.x)
    {
        int rank = 0;
        float v = p[e];
        for (int o = 0; o < E; o++) rank += (p[o] > v || (p[o] == v && o < e)) ? 1 : 0;
        if (rank < a.topK) pick[rank] = e;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float total = 0.f;
        for (int k = 0; k < a.topK; k++) total += p[pick[k]];
        float scale = a.normaliseSelected ? (total > 0.f ? 1.f / total : 0.f) : 1.f / sum;
        for (int k = 0; k < a.topK; k++)
        {
            int e = pick[k];
            a.idx[(long long)r * a.topK + k] = e;
            a.wt[(long long)r * a.topK + k] = p[e] * scale * (a.expertScale ? a.expertScale[e] : 1.f);
        }
    }
}

/// Each row's topK experts and their weights. With inNormW, normed[r] = x[r]
/// normalised by it (what the experts read, when the router reads x itself).
AMQE_EXPORT int amqe_moe_route(Engine* e, const void* w, int wType, int experts, int hidden, int topK,
    const float* x, int ldx, int rows, float normEps, const float* inScale, const float* expertScale, int normaliseSelected,
    int* idx, float* wt, const float* inNormW, float inNormOff, float inNormEps, float* normed)
{
    if (topK < 1 || topK > 64 || topK > experts) return -100;
    MoeRouteArgs a{ w, experts, hidden, topK, x, ldx, normEps, inScale, expertScale, normaliseSelected, idx, wt,
        inNormW, inNormOff, inNormEps, normed };
    size_t smem = (size_t)(hidden + experts) * sizeof(float);
    if (smem > 44 * 1024) return -101;
    const int threads = experts > 64 ? 1024 : experts > 32 ? 512 : 256;
    if (wType == 0) launch(moe_route_kernel<__nv_bfloat16>, rows, threads, smem, e->stream, a);
    else if (wType == 1) launch(moe_route_kernel<float>, rows, threads, smem, e->stream, a);
    else return -100;
    return ok(cudaGetLastError());
}

// The pairs ordered by expert, and within one by row, so a prompt's blocks (and the
// GEMMs' rounding) come out the same every run. A row's experts are distinct, so a
// pair's place among its expert's is the rows before it with that expert: a bit a
// row and a popcount. groups[3g] = expert, first entry of order, pairs (0 past the
// last group); with R = 0, groups[e] = the expert's pairs instead.
__global__ void __launch_bounds__(256) moe_group_kernel(const int* idx, int rows, int topK, int experts, int R, int maxGroups,
    int* order, int* groups)
{
    pdl_enter();
    extern __shared__ int moeSm[];
    const int words = (rows + 31) / 32;
    unsigned int* bits = reinterpret_cast<unsigned int*>(moeSm);      // [experts x words]
    int* before = moeSm + experts * words;                            // [experts x words] pairs in the words before
    int* first = before + experts * words;                            // [experts] its first entry of order
    for (int i = threadIdx.x; i < experts * words; i += blockDim.x) bits[i] = 0;
    __syncthreads();
    const int pairs = rows * topK;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x)
    {
        const int r = i / topK;
        atomicOr(&bits[idx[i] * words + (r >> 5)], 1u << (r & 31));
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
    {
        int c = 0;
        for (int w = 0; w < words; w++) { before[e * words + w] = c; c += __popc(bits[e * words + w]); }
        first[e] = c;   // its count, until the pass below
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int at = 0, g = 0;
        for (int e = 0; e < experts; e++)
        {
            const int c = first[e];
            first[e] = at;
            if (R == 0) groups[e] = c;
            else for (int j = 0; j < c; j += R)
            {
                groups[3 * g] = e;
                groups[3 * g + 1] = at + j;
                groups[3 * g + 2] = min(R, c - j);
                g++;
            }
            at += c;
        }
        for (; g < maxGroups; g++) groups[3 * g + 2] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x)
    {
        const int r = i / topK, e = idx[i], w = r >> 5;
        order[first[e] + before[e * words + w] + __popc(bits[e * words + w] & ((1u << (r & 31)) - 1u))] = i;
    }
}

// Shared memory for that.
static size_t moe_group_smem(int rows, int experts)
{
    return (2 * ((size_t)(rows + 31) / 32) + 1) * experts * sizeof(int);
}

static int launch_moe_group(Engine* e, const int* idx, int rows, int topK, int experts, int R, int maxGroups, int* order, int* groups)
{
    size_t smem = moe_group_smem(rows, experts);
    if (smem > 47 * 1024)
    {
        // Past what a kernel may have unasked: what the device allows a block.
        static int most = 0;
        if (most == 0)
        {
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || most <= 0) most = 1;
            cudaFuncSetAttribute(moe_group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, most);
            cudaGetLastError();
        }
        if (smem > (size_t)most) return -101;
    }
    launch(moe_group_kernel, 1, 256, smem, e->stream, idx, rows, topK, experts, R, maxGroups, order, groups);
    return ok(cudaGetLastError());
}

struct MoeGemvArgs
{
    const void* w; const unsigned char* wScales; const float* rowScale;
    int n; int k; int rowsPerExpert;
    const float* x; long long ldx; int xDiv;   // pair p reads row p / xDiv of x
    const int* order; const int* groups;       // null order: pair g is group g, expert idx[g]
    const int* idx;
    float* y;                                  // [pairs x n]
    int act; int glu;
    float gateCap, upCap;                      // PAIR: the softcapped GLU's caps
};

// One warp per RW output rows of a group's expert, over the group's R inputs.
// A GLU takes an output's gate and up rows together, so their loads are in
// flight at once: rows j and n + j, or with PAIR rows 2j and 2j + 1 (gate and
// up interleaved) through the softcapped GLU.
template <typename WT, int R, int RW, bool GLU, bool PAIR = false>
__global__ void __launch_bounds__(256) moe_gemv_kernel(MoeGemvArgs a)
{
    pdl_enter();
    constexpr int WARPS = 8;
    constexpr int S = GLU ? 2 : 1;
    const int g = blockIdx.y;
    int expert, start = 0, live = 1;
    if (a.order)
    {
        expert = a.groups[3 * g];
        start = a.groups[3 * g + 1];
        live = a.groups[3 * g + 2];
        if (live == 0) return;
    }
    else expert = a.idx[g];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int k = a.k;
    const WT* W = static_cast<const WT*>(a.w);
    const float* xr[R];
    float* yr[R];
    #pragma unroll
    for (int r = 0; r < R; r++)
    {
        int pair = a.order ? a.order[start + (r < live ? r : 0)] : g;
        xr[r] = a.x + (long long)(pair / a.xDiv) * a.ldx;
        yr[r] = a.y + (long long)pair * a.n;
    }
    const int jb = (blockIdx.x * WARPS + warp) * RW;
    if (jb >= a.n) return;
    const long long rowBase0 = (long long)expert * a.rowsPerExpert;
    constexpr bool fp4 = WTraits<WT>::fp4;
    constexpr int E = fp4 ? 32 : 8;
    const bool vec = (k & (E - 1)) == 0 && (a.ldx & 3) == 0;
    const int kv = vec ? k : 0;
    float acc[S][RW][R];
    long long base[S][RW];
    #pragma unroll
    for (int s = 0; s < S; s++)
        #pragma unroll
        for (int q = 0; q < RW; q++)
        {
            const int jj = jb + q < a.n ? jb + q : jb;
            base[s][q] = (rowBase0 + (PAIR ? 2 * jj + s : jj + (s ? a.n : 0))) * k;
            #pragma unroll
            for (int r = 0; r < R; r++) acc[s][q][r] = 0.f;
        }
    #pragma unroll 2
    for (int c0 = lane * E; c0 < kv; c0 += 32 * E)
    {
        uint4 packs[S][RW];
        if constexpr (fp4)
        {
            #pragma unroll
            for (int s = 0; s < S; s++)
                #pragma unroll
                for (int q = 0; q < RW; q++)
                    packs[s][q] = __ldg(reinterpret_cast<const uint4*>(static_cast<const unsigned char*>(a.w) + ((base[s][q] + c0) >> 1)));
        }
        #pragma unroll
        for (int t = 0; t < E / 8; t++)
        {
            const int c = c0 + 8 * t;
            float4 x0[R], x1[R];
            #pragma unroll
            for (int r = 0; r < R; r++)
            {
                x0[r] = reinterpret_cast<const float4*>(xr[r] + c)[0];
                x1[r] = reinterpret_cast<const float4*>(xr[r] + c)[1];
            }
            #pragma unroll
            for (int s = 0; s < S; s++)
                #pragma unroll
                for (int q = 0; q < RW; q++)
                {
                    float wv[8];
                    float gsc = 1.f;
                    if constexpr (fp4)
                    {
                        const uint4 p = packs[s][q];
                        decode8_e2m1(t == 0 ? p.x : t == 1 ? p.y : t == 2 ? p.z : p.w, wv);
                        gsc = fp4_block_scale(W, a.wScales, base[s][q] + c);
                    }
                    else loadw(W, a.wScales, base[s][q] + c, wv);
                    #pragma unroll
                    for (int r = 0; r < R; r++)
                    {
                        float v = wv[0] * x0[r].x;
                        v = fmaf(wv[1], x0[r].y, v);
                        v = fmaf(wv[2], x0[r].z, v); v = fmaf(wv[3], x0[r].w, v);
                        v = fmaf(wv[4], x1[r].x, v); v = fmaf(wv[5], x1[r].y, v);
                        v = fmaf(wv[6], x1[r].z, v); v = fmaf(wv[7], x1[r].w, v);
                        acc[s][q][r] = fp4 ? fmaf(gsc, v, acc[s][q][r]) : acc[s][q][r] + v;
                    }
                }
        }
    }
    for (int c = kv + lane; c < k; c += 32)
        #pragma unroll
        for (int s = 0; s < S; s++)
            #pragma unroll
            for (int q = 0; q < RW; q++)
            {
                const float wv = loadw1(W, a.wScales, base[s][q] + c);
                #pragma unroll
                for (int r = 0; r < R; r++) acc[s][q][r] = fmaf(wv, xr[r][c], acc[s][q][r]);
            }
    #pragma unroll
    for (int q = 0; q < RW; q++)
        #pragma unroll
        for (int r = 0; r < R; r++)
        {
            float v[S];
            #pragma unroll
            for (int s = 0; s < S; s++) v[s] = warp_sum(acc[s][q][r]);
            const int j = jb + q;
            if (lane != 0 || j >= a.n || r >= live) continue;
            if (a.rowScale)
                #pragma unroll
                for (int s = 0; s < S; s++) v[s] *= a.rowScale[rowBase0 + (PAIR ? 2 * j + s : (s ? a.n : 0) + j)];
            if constexpr (PAIR) yr[r][j] = softcapped_glu(v[0], v[S - 1], a.gateCap, a.upCap);
            else yr[r][j] = GLU ? activate(v[0], a.act) * v[S - 1] : activate(v[0], a.act);
        }
}

// How a row count runs: pairs of one expert to a group (R), and output rows to
// a warp (RW: more where the rows are short, so a warp's fixed cost is spread).
static void moe_shape(int rows, int k, int glu, int* R, int* RW)
{
    static int alone = -1;   // AMQE_MOE_ALONE: up to this many rows, each pair is a group of its own
    if (alone < 0) { const char* v = getenv("AMQE_MOE_ALONE"); alone = v ? atoi(v) : 8; }
    *R = rows <= alone ? 1 : rows <= 64 ? 2 : 8;
    *RW = k * (glu ? 2 : 1) >= 4096 ? 1 : *R >= 8 ? 2 : 4;
}

template <typename WT>
static int launch_moe_gemv(Engine* e, const MoeGemvArgs& a, int R, int RW, int groups)
{
    const dim3 grid((a.n + 8 * RW - 1) / (8 * RW), groups);
#define MOE_CASE(r, w) case r * 16 + w: \
        if (a.glu) launch(moe_gemv_kernel<WT, r, w, true>, grid, 256, 0, e->stream, a); \
        else launch(moe_gemv_kernel<WT, r, w, false>, grid, 256, 0, e->stream, a); break;
    switch (R * 16 + RW)
    {
        MOE_CASE(1, 1) MOE_CASE(1, 4) MOE_CASE(2, 1) MOE_CASE(2, 4) MOE_CASE(8, 1) MOE_CASE(8, 2)
        default: return -100;
    }
#undef MOE_CASE
    return ok(cudaGetLastError());
}

static int moe_gemv(Engine* e, int wType, const MoeGemvArgs& a, int R, int RW, int groups)
{
    switch (wType)
    {
        case 0: return launch_moe_gemv<__nv_bfloat16>(e, a, R, RW, groups);
        case 1: return launch_moe_gemv<float>(e, a, R, RW, groups);
        case 2: return launch_moe_gemv<Fp4Nv>(e, a, R, RW, groups);
        case 3: return launch_moe_gemv<Fp4Mx>(e, a, R, RW, groups);
        case 4: return launch_moe_gemv<Fp8Mx>(e, a, R, RW, groups);   // MXFP8: amqe_moe_experts_paired_step only
        default: return -100;
    }
}

struct MoeMixArgs
{
    const void* w; const unsigned char* wScales; const float* rowScale;
    int n; int k; int topK;
    const float* glu;            // [rows x topK x k]
    const int* idx; const float* wt;
    float* out;                  // [rows x n]
};

// The down projection of a decode step: an output takes its row's topK experts
// together, weighted, in the order they were picked. One warp per RW outputs.
template <typename WT, int RW>
__global__ void __launch_bounds__(256) moe_mix_kernel(MoeMixArgs a)
{
    pdl_enter();
    constexpr int WARPS = 8, AHEAD = 4;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.y, k = a.k;
    const int jb = (blockIdx.x * WARPS + warp) * RW;
    if (jb >= a.n) return;
    const WT* W = static_cast<const WT*>(a.w);
    constexpr bool fp4 = WTraits<WT>::fp4;
    constexpr int E = fp4 ? 32 : 8;
    const int kv = (k & (E - 1)) == 0 ? k : 0;
    float out[RW];
    #pragma unroll
    for (int q = 0; q < RW; q++) out[q] = 0.f;
    // AHEAD slots' first loads are in flight together.
    for (int s0 = 0; s0 < a.topK; s0 += AHEAD)
    {
        float acc[AHEAD][RW];
        long long rowAt[AHEAD];
        const float* x[AHEAD];
        #pragma unroll
        for (int t = 0; t < AHEAD; t++)
        {
            const int slot = min(s0 + t, a.topK - 1);
            rowAt[t] = (long long)a.idx[(long long)r * a.topK + slot] * a.n;
            x[t] = a.glu + ((long long)r * a.topK + slot) * k;
            #pragma unroll
            for (int q = 0; q < RW; q++) acc[t][q] = 0.f;
        }
        for (int c0 = lane * E; c0 < kv; c0 += 32 * E)
        {
            uint4 packs[AHEAD][RW];
            if constexpr (fp4)
            {
                #pragma unroll
                for (int t = 0; t < AHEAD; t++)
                    #pragma unroll
                    for (int q = 0; q < RW; q++)
                        packs[t][q] = __ldg(reinterpret_cast<const uint4*>(static_cast<const unsigned char*>(a.w) +
                            (((rowAt[t] + min(jb + q, a.n - 1)) * k + c0) >> 1)));
            }
            #pragma unroll
            for (int t = 0; t < AHEAD; t++)
                #pragma unroll
                for (int u = 0; u < E / 8; u++)
                {
                    const int c = c0 + 8 * u;
                    const float4 x0 = reinterpret_cast<const float4*>(x[t] + c)[0];
                    const float4 x1 = reinterpret_cast<const float4*>(x[t] + c)[1];
                    #pragma unroll
                    for (int q = 0; q < RW; q++)
                    {
                        const long long base = (rowAt[t] + min(jb + q, a.n - 1)) * k + c;
                        float wv[8];
                        float gsc = 1.f;
                        if constexpr (fp4)
                        {
                            const uint4 p = packs[t][q];
                            decode8_e2m1(u == 0 ? p.x : u == 1 ? p.y : u == 2 ? p.z : p.w, wv);
                            gsc = fp4_block_scale(W, a.wScales, base);
                        }
                        else loadw(W, a.wScales, base, wv);
                        float v = wv[0] * x0.x;
                        v = fmaf(wv[1], x0.y, v);
                        v = fmaf(wv[2], x0.z, v); v = fmaf(wv[3], x0.w, v);
                        v = fmaf(wv[4], x1.x, v); v = fmaf(wv[5], x1.y, v);
                        v = fmaf(wv[6], x1.z, v); v = fmaf(wv[7], x1.w, v);
                        acc[t][q] = fp4 ? fmaf(gsc, v, acc[t][q]) : acc[t][q] + v;
                    }
                }
        }
        for (int c = kv + lane; c < k; c += 32)
            #pragma unroll
            for (int t = 0; t < AHEAD; t++)
                #pragma unroll
                for (int q = 0; q < RW; q++)
                    acc[t][q] = fmaf(loadw1(W, a.wScales, (rowAt[t] + min(jb + q, a.n - 1)) * k + c), x[t][c], acc[t][q]);
        #pragma unroll
        for (int t = 0; t < AHEAD; t++)
        {
            if (s0 + t >= a.topK) break;
            const float wt = a.wt[(long long)r * a.topK + s0 + t];
            #pragma unroll
            for (int q = 0; q < RW; q++)
                out[q] = fmaf(wt * (a.rowScale ? a.rowScale[rowAt[t] + min(jb + q, a.n - 1)] : 1.f), acc[t][q], out[q]);
        }
    }
    #pragma unroll
    for (int q = 0; q < RW; q++)
    {
        const float v = warp_sum(out[q]);
        if (lane == 0 && jb + q < a.n) a.out[(long long)r * a.n + jb + q] = v;
    }
}

template <typename WT>
static void launch_moe_mix(Engine* e, const MoeMixArgs& a, int rows)
{
    launch(moe_mix_kernel<WT, 2>, dim3((a.n + 15) / 16, rows), 256, 0, e->stream, a);
}

struct MoeFinishArgs
{
    const float* slots; const float* wt; int topK; int hidden;
    const float* dense;                         // the dense FFN's output, or null
    const float* denseNormW; float denseNormOff, denseNormEps;   // its norm, or null
    const float* routedNormW; float routedNormOff, routedNormEps; // the mixture's norm, or null
    const float* outNormW; float outNormOff, outNormEps;          // the sum's norm, or null
    float* x; float resScale; float mul;        // x = (x + resScale · sum) · mul
    int replace;                                // 1: x = sum
};

// The end of a routed layer in one pass: the mixture (slots in the order they were
// picked), the norms around it, the dense FFN beside it, and the residual.
__global__ void __launch_bounds__(1024) moe_finish_kernel(MoeFinishArgs a)
{
    pdl_enter();
    extern __shared__ float fin[];   // hidden
    __shared__ float red[32];
    const int r = blockIdx.x, H = a.hidden;
    float ssR = 0.f, ssD = 0.f;
    const float* dense = a.dense ? a.dense + (long long)r * H : nullptr;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
    {
        float t = 0.f;
        if (a.topK == 0) t = a.slots[(long long)r * H + c];   // mixed already
        for (int s = 0; s < a.topK; s++)
        {
            const long long pair = (long long)r * a.topK + s;
            t = fmaf(a.wt[pair], a.slots[pair * H + c], t);
        }
        fin[c] = t;
        ssR += t * t;
        if (dense) ssD += dense[c] * dense[c];
    }
    const float invR = a.routedNormW ? rsqrtf(block_sum(ssR, red) / H + a.routedNormEps) : 1.f;
    const float invD = a.denseNormW ? rsqrtf(block_sum(ssD, red) / H + a.denseNormEps) : 1.f;
    __syncthreads();
    float ss = 0.f;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
    {
        float v = fin[c];
        if (a.routedNormW) v = v * invR * (a.routedNormW[c] + a.routedNormOff);
        if (dense) v += a.denseNormW ? dense[c] * invD * (a.denseNormW[c] + a.denseNormOff) : dense[c];
        fin[c] = v;
        ss += v * v;
    }
    const float inv = a.outNormW ? rsqrtf(block_sum(ss, red) / H + a.outNormEps) : 1.f;
    float* xr = a.x + (long long)r * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
    {
        float v = fin[c];
        if (a.outNormW) v = v * inv * (a.outNormW[c] + a.outNormOff);
        xr[c] = a.replace ? v : (xr[c] + a.resScale * v) * a.mul;
    }
}

/// The experts of amqe_moe_experts joined to the residual x: the weighted slots
/// (with topK 0, the mixture it made already),
/// then each norm that is given (RMS, no bias), the dense FFN's output and
/// x = (x + resScale · sum) · mul, or with replace x = sum.
AMQE_EXPORT int amqe_moe_finish(Engine* e, const float* slots, const float* wt, int topK, int hidden, int rows,
    const float* dense, const float* denseNormW, float denseNormOff, float denseNormEps,
    const float* routedNormW, float routedNormOff, float routedNormEps,
    const float* outNormW, float outNormOff, float outNormEps,
    float* x, float resScale, float mul, int replace)
{
    size_t smem = (size_t)hidden * sizeof(float);
    if (smem > 44 * 1024) return -101;
    MoeFinishArgs a{ slots, wt, topK, hidden, dense, denseNormW, denseNormOff, denseNormEps,
        routedNormW, routedNormOff, routedNormEps, outNormW, outNormOff, outNormEps, x, resScale, mul, replace };
    launch(moe_finish_kernel, rows, hidden >= 1024 ? 1024 : 256, smem, e->stream, a);
    return ok(cudaGetLastError());
}

// ── prompts: each expert's rows gathered, tensor-core GEMMs ───────────────
//
// The pairs of one expert are gathered into blocks of MOE_BLOCK rows (the last
// one padded with zeros), in BF16 like every other prompt activation. The
// experts in use are decoded to BF16 a few at a time into scratch that stays in
// L2, and one batched GEMM runs their blocks.

constexpr int MOE_BLOCK = 32;
// Expert weights as MXFP8 (E4M3 elements, E8M0 per 32): prompts only (amqe_moe_experts_paired).
constexpr int WT_MXFP8 = 4;

// x[block][row] = bf16(x of the pair there), zero past the block's last pair.
__global__ void moe_gather_kernel(const float* x, long long ldx, int xDiv, const int* order, const int* blocks, int k, __nv_bfloat16* out)
{
    pdl_enter();
    const int b = blockIdx.x, r = blockIdx.y;
    const int start = blocks[3 * b + 1], live = blocks[3 * b + 2];
    __nv_bfloat16* o = out + ((long long)b * MOE_BLOCK + r) * k;
    const float* xr = r < live ? x + (long long)(order[start + r] / xDiv) * ldx : nullptr;
    for (int c = threadIdx.x; c < k; c += blockDim.x) o[c] = __float2bfloat16_rn(xr ? xr[c] : 0.f);
}

// The GLU of a block's raw products, as the down projection's BF16 input.
__global__ void moe_glu_kernel(const float* raw, int n, const int* blocks, const float* rowScale, int rowsPerExpert, int act, __nv_bfloat16* out)
{
    pdl_enter();
    const int b = blockIdx.x, r = blockIdx.y;
    const float* in = raw + ((long long)b * MOE_BLOCK + r) * 2 * n;
    __nv_bfloat16* o = out + ((long long)b * MOE_BLOCK + r) * n;
    const float* rs = rowScale ? rowScale + (long long)blocks[3 * b] * rowsPerExpert : nullptr;
    for (int c = threadIdx.x; c < n; c += blockDim.x)
    {
        float g = in[c], u = in[n + c];
        if (rs) { g *= rs[c]; u *= rs[n + c]; }
        o[c] = __float2bfloat16_rn(activate(g, act) * u);
    }
}

// The GLU of interleaved rows (gate 2c, up 2c + 1), each softcapped:
// cap(g) · sigmoid(g) · cap(u), with cap(v) = c · tanh(v / c).
__global__ void moe_glu_pairs_kernel(const float* raw, int n, const int* blocks, const float* rowScale, int rowsPerExpert,
                                     float gateCap, float upCap, __nv_bfloat16* out)
{
    pdl_enter();
    const int b = blockIdx.x, r = blockIdx.y;
    const float* in = raw + ((long long)b * MOE_BLOCK + r) * 2 * n;
    __nv_bfloat16* o = out + ((long long)b * MOE_BLOCK + r) * n;
    const float* rs = rowScale ? rowScale + (long long)blocks[3 * b] * rowsPerExpert : nullptr;
    for (int c = threadIdx.x; c < n; c += blockDim.x)
    {
        float g = in[2 * c], u = in[2 * c + 1];
        if (rs) { g *= rs[2 * c]; u *= rs[2 * c + 1]; }
        o[c] = __float2bfloat16_rn(tanhf(g / gateCap) * gateCap / (1.f + expf(-g)) * (tanhf(u / upCap) * upCap));
    }
}

// Each pair's output back in pair order.
__global__ void moe_scatter_kernel(const float* raw, int n, const int* order, const int* blocks, const float* rowScale, float* slots)
{
    pdl_enter();
    const int b = blockIdx.x, r = blockIdx.y;
    const int start = blocks[3 * b + 1], live = blocks[3 * b + 2];
    if (r >= live) return;
    const float* in = raw + ((long long)b * MOE_BLOCK + r) * n;
    float* o = slots + (long long)order[start + r] * n;
    const float* rs = rowScale ? rowScale + (long long)blocks[3 * b] * n : nullptr;
    for (int c = threadIdx.x; c < n; c += blockDim.x) o[c] = rs ? in[c] * rs[c] : in[c];
}

// The weights of the experts listed, 32 elements a thread. Element times block
// scale is exact in BF16, so the top half of the f32 is the value.
template <typename WT>
__global__ void moe_decode_kernel(const WT* w, const unsigned char* scales, const int* experts, long long elements, __nv_bfloat16* out)
{
    pdl_enter();
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 32;
    if (i >= elements) return;
    const long long at = (long long)experts[blockIdx.y] * elements + i;
    const uint4 packed = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(w) + (at >> 1)));
    const unsigned int words[4] = { packed.x, packed.y, packed.z, packed.w };
    const float s0 = fp4_block_scale(w, scales, at), s1 = fp4_block_scale(w, scales, at + 16);
    uint4* o = reinterpret_cast<uint4*>(out + (long long)blockIdx.y * elements + i);
    #pragma unroll
    for (int t = 0; t < 4; t++)
    {
        float v[8];
        decode8_e2m1(words[t], v);
        const float sc = t < 2 ? s0 : s1;
        unsigned int p[4];
        #pragma unroll
        for (int q = 0; q < 4; q++)
            p[q] = (__float_as_uint(v[2 * q + 1] * sc) & 0xffff0000u) | (__float_as_uint(v[2 * q] * sc) >> 16);
        o[t] = make_uint4(p[0], p[1], p[2], p[3]);
    }
}

// MXFP8 (E4M3 elements, an E8M0 exponent per 32), 32 elements a thread. Element
// times a power of two is exact in BF16.
__global__ void moe_decode_mxfp8_kernel(const unsigned char* w, const unsigned char* scales, const int* experts, long long elements, __nv_bfloat16* out)
{
    pdl_enter();
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 32;
    if (i >= elements) return;
    const long long at = (long long)experts[blockIdx.y] * elements + i;
    const uint4* q = reinterpret_cast<const uint4*>(w + at);
    const uint4 a = __ldg(q), c = __ldg(q + 1);
    const unsigned int words[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
    const float s = e8m0_value(__ldg(scales + (at >> 5)));
    uint4* o = reinterpret_cast<uint4*>(out + (long long)blockIdx.y * elements + i);
    #pragma unroll
    for (int t = 0; t < 4; t++)
    {
        unsigned int p[4];
        #pragma unroll
        for (int h = 0; h < 2; h++)
        {
            const unsigned int wd = words[2 * t + h];
            __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wd & 0xffffu), __NV_E4M3);
            __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wd >> 16), __NV_E4M3);
            const float2 x0 = __half22float2(*reinterpret_cast<__half2*>(&lo)), x1 = __half22float2(*reinterpret_cast<__half2*>(&hi));
            p[2 * h] = (__float_as_uint(x0.y * s) & 0xffff0000u) | (__float_as_uint(x0.x * s) >> 16);
            p[2 * h + 1] = (__float_as_uint(x1.y * s) & 0xffff0000u) | (__float_as_uint(x1.x * s) >> 16);
        }
        o[t] = make_uint4(p[0], p[1], p[2], p[3]);
    }
}

static int moe_decode(Engine* e, const void* w, int type, const unsigned char* scales, const int* experts, int count, long long elements, __nv_bfloat16* out)
{
    dim3 grid((unsigned)((elements / 32 + 255) / 256), count);
    switch (type)
    {
        case 2: launch(moe_decode_kernel<Fp4Nv>, grid, 256, 0, e->stream, (const Fp4Nv*)w, scales, experts, elements, out); break;
        case 3: launch(moe_decode_kernel<Fp4Mx>, grid, 256, 0, e->stream, (const Fp4Mx*)w, scales, experts, elements, out); break;
        case WT_MXFP8: launch(moe_decode_mxfp8_kernel, grid, 256, 0, e->stream, (const unsigned char*)w, scales, experts, elements, out); break;
        default: return -100;
    }
    return ok(cudaGetLastError());
}

struct MoePrompt   // scratch of the prompt path, grown as needed, and the side stream
{
    void* buf[5] = {};
    size_t bytes[5] = {};
    int* counts = nullptr;         // host, pinned
    char* table = nullptr;         // host, pinned: blocks, expert ids and GEMM pointers
    size_t tableBytes = 0;
    cudaStream_t side = nullptr, main = nullptr;   // amqe_side_begin
    cudaEvent_t fork = nullptr, join = nullptr;
    int inSide = 0;
    void* get(int i, size_t need)
    {
        if (need > bytes[i])
        {
            if (buf[i]) cudaFree(buf[i]);
            buf[i] = nullptr;
            bytes[i] = 0;
            if (cudaMalloc(&buf[i], need) != cudaSuccess) return nullptr;
            bytes[i] = need;
        }
        return buf[i];
    }
};

void moe_prompt_free(Engine* e)
{
    if (!e->moe) return;
    for (int i = 0; i < 5; i++) if (e->moe->buf[i]) cudaFree(e->moe->buf[i]);
    if (e->moe->counts) cudaFreeHost(e->moe->counts);
    if (e->moe->table) cudaFreeHost(e->moe->table);
    if (e->moe->side) { cudaStreamSynchronize(e->moe->side); cudaStreamDestroy(e->moe->side); }
    if (e->moe->fork) cudaEventDestroy(e->moe->fork);
    if (e->moe->join) cudaEventDestroy(e->moe->join);
    delete e->moe;
    e->moe = nullptr;
}

#ifndef MOE_DECODE_EXPERTS
#define MOE_DECODE_EXPERTS 8   // decoded at once
#endif

static int moe_experts_prompt(Engine* e,
    const void* gateUp, int guType, const unsigned char* guScales, const float* guRowScale,
    const void* down, int dType, const unsigned char* dScales, const float* dRowScale,
    int experts, int hidden, int inter, int act,
    const float* x, int ldx, int topK, int pairs, int* order, int* groups, float* slots,
    int paired = 0, float gateCap = 0.f, float upCap = 0.f)
{
    if (!e->moe) e->moe = new MoePrompt();
    MoePrompt& m = *e->moe;
    if (!m.counts && cudaMallocHost(&m.counts, 4096 * sizeof(int)) != cudaSuccess) return -2;
    // How many pairs each expert has is the host's to know: it shapes the GEMMs.
    if (cudaMemcpyAsync(m.counts, groups, experts * sizeof(int), cudaMemcpyDeviceToHost, e->stream) != cudaSuccess) return -3;
    if (cudaStreamSynchronize(e->stream) != cudaSuccess) return -3;
    int blocks = 0, used = 0;
    for (int x0 = 0; x0 < experts; x0++)
    {
        blocks += (m.counts[x0] + MOE_BLOCK - 1) / MOE_BLOCK;
        used += m.counts[x0] > 0;
    }
    // Host table: blocks [3 ints], used experts' ids, then the GEMM pointers.
    size_t ints = (size_t)(3 * blocks + used) * sizeof(int);
    ints = (ints + 15) & ~(size_t)15;
    const size_t bytes = ints + (size_t)6 * blocks * sizeof(void*);
    if (bytes > m.tableBytes)
    {
        if (m.table) cudaFreeHost(m.table);
        m.table = nullptr;
        m.tableBytes = 0;
        if (cudaMallocHost(&m.table, bytes * 2) != cudaSuccess) return -2;
        m.tableBytes = bytes * 2;
    }
    const long long guElems = 2LL * inter * hidden, dElems = (long long)hidden * inter;
    const long long most = guElems > dElems ? guElems : dElems;
    const size_t rowsAll = (size_t)blocks * MOE_BLOCK;
    const size_t inCols = hidden > inter ? hidden : inter, outCols = hidden > 2 * inter ? hidden : 2 * inter;
    auto* xb = (__nv_bfloat16*)m.get(0, rowsAll * inCols * sizeof(__nv_bfloat16));
    auto* raw = (float*)m.get(1, rowsAll * outCols * sizeof(float));
    auto* wb = (__nv_bfloat16*)m.get(2, (size_t)MOE_DECODE_EXPERTS * most * sizeof(__nv_bfloat16));
    auto* table = (char*)m.get(3, bytes);
    if (!xb || !raw || !wb || !table) return -2;
    int* hb = (int*)m.table;
    int* ids = hb + 3 * blocks;
    void** ptrs = (void**)(m.table + ints);
    void** pa = ptrs, **pb = ptrs + blocks, **pc = ptrs + 2 * blocks;             // gate and up
    void** qa = ptrs + 3 * blocks, **qb = ptrs + 4 * blocks, **qc = ptrs + 5 * blocks;   // down
    int b = 0, at = 0, u = 0;
    int firstBlock[4097];
    for (int x0 = 0; x0 < experts; x0++)
    {
        const int c = m.counts[x0];
        if (c == 0) continue;
        firstBlock[u] = b;
        ids[u] = x0;
        for (int j = 0; j < c; j += MOE_BLOCK, b++)
        {
            hb[3 * b] = x0;
            hb[3 * b + 1] = at + j;
            hb[3 * b + 2] = c - j < MOE_BLOCK ? c - j : MOE_BLOCK;
            const int slot = u % MOE_DECODE_EXPERTS;
            pa[b] = guType >= 2 ? (void*)(wb + slot * guElems) : (void*)((const __nv_bfloat16*)gateUp + (long long)x0 * guElems);
            pb[b] = xb + (size_t)b * MOE_BLOCK * hidden;
            pc[b] = raw + (size_t)b * MOE_BLOCK * 2 * inter;
            qa[b] = dType >= 2 ? (void*)(wb + slot * dElems) : (void*)((const __nv_bfloat16*)down + (long long)x0 * dElems);
            qb[b] = xb + (size_t)b * MOE_BLOCK * inter;
            qc[b] = raw + (size_t)b * MOE_BLOCK * hidden;
        }
        at += c;
        u++;
    }
    firstBlock[u] = b;
    if (cudaMemcpyAsync(table, m.table, bytes, cudaMemcpyHostToDevice, e->stream) != cudaSuccess) return -3;
    const int* blk = (const int*)table;
    const int* idd = blk + 3 * blocks;
    void** dev = (void**)(table + ints);
    const float one = 1.f, zero = 0.f;
    const dim3 grid(blocks, MOE_BLOCK);
    launch(moe_gather_kernel, grid, 256, 0, e->stream, x, (long long)ldx, topK, order, blk, hidden, xb);
    for (int u0 = 0; u0 < used; u0 += MOE_DECODE_EXPERTS)
    {
        const int n = used - u0 < MOE_DECODE_EXPERTS ? used - u0 : MOE_DECODE_EXPERTS;
        if (guType >= 2) { int rc = moe_decode(e, gateUp, guType, guScales, idd + u0, n, guElems, wb); if (rc) return rc; }
        const int f = firstBlock[u0], count = firstBlock[u0 + n] - f;
        auto st = cublasGemmBatchedEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * inter, MOE_BLOCK, hidden, &one,
            (const void* const*)(dev + f), CUDA_R_16BF, hidden, (const void* const*)(dev + blocks + f), CUDA_R_16BF, hidden,
            &zero, (void* const*)(dev + 2 * blocks + f), CUDA_R_32F, 2 * inter, count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
    }
    if (paired) launch(moe_glu_pairs_kernel, grid, 256, 0, e->stream, raw, inter, blk, guRowScale, 2 * inter, gateCap, upCap, xb);
    else launch(moe_glu_kernel, grid, 256, 0, e->stream, raw, inter, blk, guRowScale, 2 * inter, act, xb);
    for (int u0 = 0; u0 < used; u0 += MOE_DECODE_EXPERTS)
    {
        const int n = used - u0 < MOE_DECODE_EXPERTS ? used - u0 : MOE_DECODE_EXPERTS;
        if (dType >= 2) { int rc = moe_decode(e, down, dType, dScales, idd + u0, n, dElems, wb); if (rc) return rc; }
        const int f = firstBlock[u0], count = firstBlock[u0 + n] - f;
        auto st = cublasGemmBatchedEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, hidden, MOE_BLOCK, inter, &one,
            (const void* const*)(dev + 3 * blocks + f), CUDA_R_16BF, inter, (const void* const*)(dev + 4 * blocks + f), CUDA_R_16BF, inter,
            &zero, (void* const*)(dev + 5 * blocks + f), CUDA_R_32F, hidden, count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
    }
    launch(moe_scatter_kernel, grid, 256, 0, e->stream, raw, hidden, order, blk, dRowScale, slots);
    return ok(cudaGetLastError());
}

// ── a branch beside the main stream ───────────────────────────────────────
//
// In a decode step a layer's dense FFN and its experts read the same residual and
// neither needs the other: the dense one runs on a side stream, forked from the
// main one and joined again before the two are summed. A captured graph holds
// them as parallel branches.

AMQE_EXPORT int amqe_side_begin(Engine* e)
{
    if (!e->moe) e->moe = new MoePrompt();
    MoePrompt& m = *e->moe;
    if (m.inSide) return -100;
    if (!m.side)
    {
        if (cudaStreamCreateWithFlags(&m.side, cudaStreamNonBlocking) != cudaSuccess) return -1;
        if (cudaEventCreateWithFlags(&m.fork, cudaEventDisableTiming) != cudaSuccess) return -1;
        if (cudaEventCreateWithFlags(&m.join, cudaEventDisableTiming) != cudaSuccess) return -1;
    }
    int r = ok(cudaEventRecord(m.fork, e->stream));
    if (!r) r = ok(cudaStreamWaitEvent(m.side, m.fork, 0));
    if (r) return r;
    m.main = e->stream;
    e->stream = m.side;
    m.inSide = 1;
    return 0;
}

/// Back to the main stream, which runs on beside what the side one was given.
AMQE_EXPORT int amqe_side_end(Engine* e)
{
    if (!e->moe || !e->moe->inSide) return -100;
    MoePrompt& m = *e->moe;
    int r = ok(cudaEventRecord(m.join, m.side));
    e->stream = m.main;
    m.inSide = 0;
    return r;
}

/// The main stream waits for the side one.
AMQE_EXPORT int amqe_side_join(Engine* e) { return ok(cudaStreamWaitEvent(e->stream, e->moe->join, 0)); }

/// slots[pair] = each of a row's topK gated experts over x: amqe_moe_finish weighs
/// and joins them. A few rows (decode) are mixed here: slots[row] is then the
/// weighted sum and *mixed is 1. gateUp: [experts x 2·inter] rows of hidden (an expert's gate
/// rows, then its up rows); down: [experts x hidden] rows of inter. Scratch: order
/// and groups [3 x (pairs + experts)] ints, glu [pairs x inter] floats. Prompts (BF16,
/// enough rows) run on tensor cores; decode, and prompts kept at f32, on the f32 kernels.
AMQE_EXPORT int amqe_moe_experts(Engine* e,
    const void* gateUp, int guType, const unsigned char* guScales, const float* guRowScale,
    const void* down, int dType, const unsigned char* dScales, const float* dRowScale,
    int experts, int hidden, int inter, int topK, int act,
    const float* x, int ldx, int rows, const int* idx, const float* wt,
    int* order, int* groups, float* glu, float* slots, int* mixed)
{
    *mixed = 0;
    if ((guType >= 2 && (hidden % 32 != 0 || !guScales)) || (dType >= 2 && (inter % 32 != 0 || !dScales))) return -100;
    const int pairs = rows * topK;
    static int from = -1;   // AMQE_MOE_GEMM_FROM: rows from which a prompt takes the tensor cores
    if (from < 0) { const char* v = getenv("AMQE_MOE_GEMM_FROM"); from = v ? atoi(v) : 64; if (from < 2) from = 64; }
    if (rows >= from && !e->prefillF32 && experts <= 4096 && guType != 1 && dType != 1 && hidden % 8 == 0 && inter % 8 == 0)
    {
        int rc = launch_moe_group(e, idx, rows, topK, experts, 0, 0, order, groups);
        if (rc) return rc;
        return moe_experts_prompt(e, gateUp, guType, guScales, guRowScale, down, dType, dScales, dRowScale,
            experts, hidden, inter, act, x, ldx, topK, pairs, order, groups, slots);
    }
    int R, rwUp, rwDown;
    moe_shape(rows, hidden, 1, &R, &rwUp);
    moe_shape(rows, inter, 0, &R, &rwDown);
    int count = pairs;
    if (R > 1)
    {
        int most = (pairs + R - 1) / R + experts;
        if (count > most) count = most;
        int rc = launch_moe_group(e, idx, rows, topK, experts, R, count, order, groups);
        if (rc) return rc;
    }
    const int* ord = R > 1 ? order : nullptr;
    MoeGemvArgs a{ gateUp, guScales, guRowScale, inter, hidden, 2 * inter, x, ldx, topK, ord, groups, idx, glu, act, 1 };
    int rc = moe_gemv(e, guType, a, R, rwUp, count);
    if (rc) return rc;
    if (rows <= 8)
    {
        MoeMixArgs mixArgs{ down, dScales, dRowScale, hidden, inter, topK, glu, idx, wt, slots };
        switch (dType)
        {
            case 0: launch_moe_mix<__nv_bfloat16>(e, mixArgs, rows); break;
            case 1: launch_moe_mix<float>(e, mixArgs, rows); break;
            case 2: launch_moe_mix<Fp4Nv>(e, mixArgs, rows); break;
            case 3: launch_moe_mix<Fp4Mx>(e, mixArgs, rows); break;
            default: return -100;
        }
        *mixed = 1;
        return ok(cudaGetLastError());
    }
    MoeGemvArgs b{ down, dScales, dRowScale, hidden, inter, hidden, glu, inter, 1, ord, groups, idx, slots, ACT_NONE, 0 };
    return moe_gemv(e, dType, b, R, rwDown, count);
}

/// A prompt's experts on the tensor cores, for gate and up rows interleaved
/// (gate 2c, up 2c + 1) and each softcapped (a softcapped GLU): slots[pair] = the
/// expert's output for that pair, unweighted. guType and dType: 0 BF16, 2 NVFP4,
/// 3 MXFP4, 4 MXFP8; a row scale is [experts x rows], or null. Scratch as for
/// amqe_moe_experts. -100: not a shape or mode this path takes (the caller runs
/// its own kernels), as with prompts kept at f32.
AMQE_EXPORT int amqe_moe_experts_paired(Engine* e,
    const void* gateUp, int guType, const unsigned char* guScales, const float* guRowScale,
    const void* down, int dType, const unsigned char* dScales, const float* dRowScale,
    int experts, int hidden, int inter, int topK, float gateCap, float upCap,
    const float* x, int ldx, int rows, const int* idx, int* order, int* groups, float* slots)
{
    auto takes = [](int t) { return t == 0 || t == 2 || t == 3 || t == WT_MXFP8; };
    if (e->prefillF32 || experts > 4096 || !takes(guType) || !takes(dType) || hidden % 32 != 0 || inter % 32 != 0) return -100;
    if ((guType != 0 && !guScales) || (dType != 0 && !dScales)) return -100;
    int rc = launch_moe_group(e, idx, rows, topK, experts, 0, 0, order, groups);
    if (rc) return rc;
    return moe_experts_prompt(e, gateUp, guType, guScales, guRowScale, down, dType, dScales, dRowScale,
        experts, hidden, inter, ACT_NONE, x, ldx, topK, rows * topK, order, groups, slots, 1, gateCap, upCap);
}

/// A decode step's experts (rows <= 8) for gate and up rows interleaved and
/// softcapped: glu[pair] = the softcapped GLU of the pair's expert,
/// then slots[pair] = its down projection, unweighted (the caller weighs and
/// sums them). A warp takes several short down rows of one pair, which suits
/// narrow experts (1,152 wide, say) better than amqe_moe_experts' mix. guType
/// and dType: 0 BF16, 1 f32, 2 NVFP4, 3 MXFP4, 4 MXFP8; a row scale is
/// [experts x rows] or null. glu: [rows x topK x inter], slots [rows x topK x hidden].
AMQE_EXPORT int amqe_moe_experts_paired_step(Engine* e,
    const void* gateUp, int guType, const unsigned char* guScales, const float* guRowScale,
    const void* down, int dType, const unsigned char* dScales, const float* dRowScale,
    int experts, int hidden, int inter, int topK, float gateCap, float upCap,
    const float* x, int ldx, int rows, const int* idx, float* glu, float* slots)
{
    if (rows < 1 || rows > 8 || guType < 0 || guType > 4 || dType < 0 || dType > 4) return -100;
    if ((guType >= 2 && (hidden % 32 != 0 || !guScales)) || (dType >= 2 && (inter % 32 != 0 || !dScales))) return -100;
    int R, rw;
    moe_shape(rows, hidden, 1, &R, &rw);
    if (R != 1) return -100;
    MoeGemvArgs a{ gateUp, guScales, guRowScale, inter, hidden, 2 * inter, x, ldx, topK, nullptr, nullptr, idx, glu, ACT_NONE, 1, gateCap, upCap };
    const dim3 grid((inter + 8 * rw - 1) / (8 * rw), rows * topK);
    auto up = [&](auto tag) {
        using WT = decltype(tag);
        if (rw == 1) launch(moe_gemv_kernel<WT, 1, 1, true, true>, grid, 256, 0, e->stream, a);
        else launch(moe_gemv_kernel<WT, 1, 4, true, true>, grid, 256, 0, e->stream, a);
    };
    switch (guType)
    {
        case 0: up(__nv_bfloat16{}); break;
        case 1: up(float{}); break;
        case 2: up(Fp4Nv{}); break;
        case 3: up(Fp4Mx{}); break;
        default: up(Fp8Mx{}); break;
    }
    int rd;
    moe_shape(rows, inter, 0, &R, &rd);
    MoeGemvArgs b{ down, dScales, dRowScale, hidden, inter, hidden, glu, inter, 1, nullptr, nullptr, idx, slots, ACT_NONE, 0, 0.f, 0.f };
    return moe_gemv(e, dType, b, 1, rd, rows * topK);
}

// y = (y + scale · b) · mul: the residual join of a layer that scales what it passes on.
__global__ void add_scaled_mul_kernel(float* y, const float* b, float scale, float mul, long long count)
{
    pdl_enter();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < count) y[i] = (y[i] + scale * b[i]) * mul;
}

AMQE_EXPORT int amqe_add_scaled_mul(Engine* e, float* y, const float* b, float scale, float mul, long long count)
{
    launch(add_scaled_mul_kernel, (unsigned)((count + 255) / 256), 256, 0, e->stream, y, b, scale, mul, count);
    return ok(cudaGetLastError());
}

// ── calibration: each expert's Gram ───────────────────────────────────────

__global__ void moe_gram_gather_kernel(const float* x, long long ldx, int xDiv, const int* order, const float* wt, int k, float* out)
{
    pdl_enter();
    const int pair = order[blockIdx.x];
    const float w = wt[pair];
    const float* xr = x + (long long)(pair / xDiv) * ldx;
    for (int c = threadIdx.x; c < k; c += blockDim.x) out[(long long)blockIdx.x * k + c] = xr[c] * w;
}

/// h[expert] += Xᵀ X over the rows routed to it, each scaled by its routing weight
/// (what its error is scaled by too). x: [pairs / xDiv x ldx]; h: [experts x k x k];
/// tmp: [pairs x k] floats; rows: host, the rows each expert had are added to it.
AMQE_EXPORT int amqe_moe_gram(Engine* e, const int* idx, const float* wt, int rows, int topK, int experts,
    const float* x, int ldx, int xDiv, int k, int* order, int* groups, float* tmp, float* h, long long* rowsOf)
{
    int rc = launch_moe_group(e, idx, rows, topK, experts, 0, 0, order, groups);
    if (rc) return rc;
    int counts[4096];
    if (experts > 4096) return -100;
    if (cudaMemcpyAsync(counts, groups, experts * sizeof(int), cudaMemcpyDeviceToHost, e->stream) != cudaSuccess) return -3;
    if (cudaStreamSynchronize(e->stream) != cudaSuccess) return -3;
    launch(moe_gram_gather_kernel, rows * topK, 256, 0, e->stream, x, (long long)ldx, xDiv, order, wt, k, tmp);
    const float one = 1.f;
    int at = 0;
    for (int g = 0; g < experts; g++)
    {
        const int c = counts[g];
        if (c == 0) continue;
        auto st = cublasSgemm(e->blas, CUBLAS_OP_N, CUBLAS_OP_T, k, k, c, &one, tmp + (size_t)at * k, k, tmp + (size_t)at * k, k,
            &one, h + (size_t)g * k * k, k);
        if (st != CUBLAS_STATUS_SUCCESS) return -(1000 + (int)st);
        rowsOf[g] += c;
        at += c;
    }
    return ok(cudaGetLastError());
}

// x = (x + resScale · rmsnorm(b)) · mul: a post norm and the residual join in one.
__global__ void __launch_bounds__(1024) norm_join_kernel(float* x, const float* b, int hidden, const float* w, float off, float eps,
    float resScale, float mul)
{
    pdl_enter();
    __shared__ float red[32];
    const int r = blockIdx.x;
    const float* br = b + (long long)r * hidden;
    float* xr = x + (long long)r * hidden;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) ss += br[c] * br[c];
    const float inv = rsqrtf(block_sum(ss, red) / hidden + eps);
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        xr[c] = (xr[c] + resScale * (br[c] * inv * (w[c] + off))) * mul;
}

AMQE_EXPORT int amqe_norm_join(Engine* e, float* x, const float* b, int rows, int hidden, const float* w, float off, float eps,
    float resScale, float mul)
{
    launch(norm_join_kernel, rows, hidden >= 1024 ? 1024 : 256, 0, e->stream, x, b, hidden, w, off, eps, resScale, mul);
    return ok(cudaGetLastError());
}
