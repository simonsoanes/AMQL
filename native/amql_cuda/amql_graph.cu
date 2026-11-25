// amql_graph.cu — captured passes kept beside the decode step's own graph (a verify
// for each number of rows, a run of drafts), and the choice of a draft.

#include "amql_engine.h"

struct Pass
{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};

AMQE_EXPORT int amqe_pass_begin(Engine* e)
{
    return ok(cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal));
}

/// What was launched since amqe_pass_begin, as a pass that can be run again; *out is
/// null when it could not be captured.
AMQE_EXPORT int amqe_pass_end(Engine* e, Pass** out)
{
    *out = nullptr;
    Pass* p = new Pass();
    int r = ok(cudaStreamEndCapture(e->stream, &p->graph));
    if (!r) r = ok(cudaGraphInstantiate(&p->exec, p->graph, 0));
    if (r)
    {
        if (p->graph) cudaGraphDestroy(p->graph);
        delete p;
        return r;
    }
    *out = p;
    return 0;
}

AMQE_EXPORT int amqe_pass_run(Engine* e, Pass* p) { return ok(cudaGraphLaunch(p->exec, e->stream)); }

AMQE_EXPORT int amqe_pass_free(Pass* p)
{
    if (!p) return 0;
    if (p->exec) cudaGraphExecDestroy(p->exec);
    if (p->graph) cudaGraphDestroy(p->graph);
    delete p;
    return 0;
}

// The index of the largest of n logits (the lowest on a tie), to out and, when
// given, out2 (a draft token the next pass embeds), and its share of their softmax.
__global__ void __launch_bounds__(1024) draft_pick_kernel(const float* x, int n, int* out, int* out2, float* share)
{
    pdl_enter();
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float red[32];
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > best) { best = x[i]; at = i; }
    for (int o = 16; o > 0; o >>= 1)
    {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, at, o);
        if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { bv[warp] = best; bi[warp] = at; }
    __syncthreads();
    if (warp == 0)
    {
        best = bv[lane];
        at = bi[lane];
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, at, o);
            if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
        }
        if (lane == 0) { bv[0] = best; bi[0] = at; }
    }
    __syncthreads();
    const float top = bv[0];
    const int pick = bi[0] == 0x7fffffff ? 0 : bi[0];
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += __expf(x[i] - top);
    sum = block_sum(sum, red);
    if (threadIdx.x == 0)
    {
        *out = pick;
        if (out2) *out2 = pick;
        *share = sum > 0.f ? 1.f / sum : 0.f;
    }
}

AMQE_EXPORT int amqe_draft_pick(Engine* e, const float* x, int n, int* out, int* out2, float* share)
{
    launch(draft_pick_kernel, 1, 1024, 0, e->stream, x, n, out, out2, share);
    return ok(cudaGetLastError());
}
