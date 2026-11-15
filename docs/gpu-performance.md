# GPU inference performance

How fast the generic runtime serves one request on an NVIDIA GPU once a model is
loaded, how that compares with llama.cpp's CUDA backend, and how the device engine
(`GpuEngine`, `native/amql_cuda/amql_engine.cu`) gets there.

## The device engine

`DecodeSession` runs a sequence on the device engine when the CUDA backend is
present and the plan is within the engine's reach. Otherwise it runs on the managed
path described in [cpu-performance.md](cpu-performance.md).

What the engine does:

- **Weights are uploaded once.** They go up as BF16 when stored as BF16 and nothing
  changes them, otherwise as f32 (a patch, an added delta or an f32 container).
  Weights that share an input are stacked into one device matrix: q/k/v, an FFN's
  gate and up, and the four Gated DeltaNet in-projections.
- **Everything stays on the device.** Activations, the KV cache and the recurrent
  state never leave it.
- **Each decode step is one CUDA graph.** The token id goes up, every layer runs,
  and the logits come back. The graph is captured once and replayed. Kernels read
  the position from a device counter, so the same graph serves every step. It is
  recaptured only when the KV cache grows.
- **Prompts run in chunks of up to 512 rows.**
  - Projections are BF16 tensor-core GEMMs (`cublasGemmEx`): the activations are
    rounded to BF16, the weights are used as stored, and products accumulate in f32.
  - Attention is two strided-batched GEMMs per KV-head group (S = Q·Kᵀ, O = P·V, in
    TF32) around a masked softmax kernel.
  - The conv and q/k normalisation run in parallel over rows. Only the recurrence
    runs position by position, one 64-thread block per (value head, 8 columns).

The engine covers:

- dense FFNs (gated or not, SiLU, GELU, ReLU, biases)
- softmax attention: GQA, the Qwen 3.5 output gate, QK norms per head or over the
  row, rotary (half-split, interleaved, partial, and scaling that does not depend on
  sequence length), sliding windows and softcaps
- Gated DeltaNet linear attention
- RMSNorm and LayerNorm in every placement (pre, post, sandwich, parallel)
- residual scaling, embedding scale, logit multipliers and softcaps, tied or untied
  heads

These stay on the managed path:

- routed experts (MoE)
- the LFM short conv
- masked diffusion
- length-dependent rope (dynamic, longrope)
- an embedding norm
- any runtime with a trace, capture or residual patch set

The engine is also not used with the `mxfp4` or `disk` working sets, under `--cpu`,
or with `AMQL_GPU_ENGINE=0`. The session's `Backend` and `GpuRefusal` properties say
which path ran and why (`bench` prints them).

**Numerics:** decode steps use f32 activations and accumulation and the container's
weights exactly. The reduction order differs from the managed kernels, so the two
agree to rounding, not bit for bit.

Prompts, by default, round their activations to BF16 for the projections and use
TF32 for attention. On Qwen 3.5 0.8B this moves the largest logit by ~0.04, about
0.3% of the largest logit. Greedy decoding still agrees token for token after
prompts of 32, 512 and 2,000 tokens.

`AMQL_GPU_PREFILL=f32` (or `DecodeSession.ExactGpuPrefill = true`) keeps prompts at
f32: FP32 SGEMM over widened weights and FP32 attention GEMMs. The largest logit
difference is then about 2e-5 (logits up to ~20). `GpuEngineTests` checks every dense
arch-zoo family against the managed path in both modes.

## Measuring

```bash
amql-cli bench <container> --prompt-tokens 32 --steps 256 --repeat 3
amql-cli bench <container> --prompt-tokens 512 --steps 128 --repeat 3 --compare
```

`--compare` runs the same greedy request on a managed-path session. It reports how
many tokens agree and the largest logit difference.

llama.cpp was measured with `llama-server -ngl 99 -fa on` (the LM Studio CUDA 12
build) on the same model exported with `amql-cli to-gguf` (F16 weights). It used the
`timings` from requests with `cache_prompt: false`, best of three.

## Results

Qwen 3.5 0.8B (24 layers: 18 Gated DeltaNet + 6 gated softmax attention, a 248k
vocabulary, tied head). RTX 5090 (170 SMs, ~1.79 TB/s), Windows 11, driver 617, CUDA
13.3.

| | prefill pp32 | prefill pp512 | prefill pp2000 | decode tg128 |
|---|---|---|---|---|
| AMQL before (per-operator CUDA path) | — | — | — | ~41 tok/s |
| **AMQL device engine** (BF16 prompts, the default) | **7,472 tok/s** | **55,354 tok/s** | **53,755 tok/s** | **655 tok/s** |
| AMQL device engine, `AMQL_GPU_PREFILL=f32` | 5,296 tok/s | 27,906 tok/s | 25,693 tok/s | 650 tok/s |
| llama.cpp, CUDA, F16, flash attention | 1,930 tok/s | 20,728 tok/s | 26,491 tok/s | 438–497 tok/s |

How to read this:

- **Decode stays flat with context.** It is ~610 tok/s after a 512- or 2000-token
  prompt. llama.cpp drops to ~460 after 512.
- **Exact prompts trade speed for f32.** At 2,000 tokens llama.cpp is ~3% ahead of
  the f32 mode. The default is ~2× ahead of llama.cpp there.
- **Measured on an idle machine.** Best of three runs each. The engine's decode
  varies by about ±1% between runs and llama.cpp's by more (438 and 497 tok/s on
  two sessions).
- **Long prompts slow prefill a little.** It is ~15,000 tok/s at 2,000 tokens,
  because attention grows with the prompt.

**Where a decode step goes:** about 1.45 ms on the GPU, plus ~70 µs for the graph
launch, the logits copy and the argmax.

- **Matrix-vector products: ~1.15 ms.** The model streams ~1.5 GB of BF16 per token.
  The 508 MB head reaches ~1.65 TB/s (92% of peak); the smaller projections reach
  1.2–1.4 TB/s.
- **Everything else: ~0.3 ms.** That is the GDN conv and recurrence, attention, and
  the q/k prep.

## What makes it fast

- **One graph per step.** A step is ~150 kernels, so launching them one by one would
  cost more than running them.
- **GEMV that saturates memory.**
  - One warp streams 16-byte runs of a weight row against up to eight activation
    rows.
  - Each warp carries two rows when decoding, so the loads of both are in flight.
  - Split-K across warps for narrow matrices: 1,024 outputs on 170 SMs would
    otherwise be six warps per SM.
  - A grid-stride loop for the 248k-row head.
  - Epilogues fused in: bias, activation, the gate·up product, and the residual add,
    done in place.
- **Norms fused into the projection that reads them.** Each block normalises its
  activation rows into shared memory, which replaces a launch per norm.
- **Programmatic dependent launch.** Every kernel lets the next one start early. A
  projection prefetches its first weight rows into L2 before waiting on the kernel
  before it (`griddepcontrol.wait`), so DRAM keeps streaming across kernel
  boundaries.
  - Decode gains ~12% (`AMQE_PDL=0` and `AMQE_GEMV_PREFETCH=0` turn it off for
    comparison).
  - A separate prefetch stream was tried as well. It helped only at one block count
    and hurt once other kernels changed, so it was dropped.
- **Tensor cores for prompts.** The projections read the BF16 weights straight into
  `cublasGemmEx`, with no widened f32 copy written and read back per chunk. Attention
  is two cuBLAS GEMMs per KV-head group rather than a scalar kernel, which took it
  from ~50 ms to a few ms of a 2,000-token prompt.
- **A prompt scan built for latency.** The recurrence is sequential in position, so
  each step's latency is what counts.
  - Eight lanes share a value column (16 key rows each), which halves the dependent
    FMA chains.
  - Eight columns per block puts four times the blocks on the SMs.
  - The next row's q, k, v, decay and β are loaded before this row's arithmetic.

  Together these make the scan ~2× faster.
- **Decode attention split by key.** Each 32-key slice is a block, and a second pass
  merges their softmaxes. The grid is sized by the cache's capacity, so the graph
  stays valid as the sequence grows.
- **Gated DeltaNet without barriers in the loop.**
  - Four lanes share a value column, each holding a quarter of its key rows in
    registers.
  - When decoding, the conv, the q/k L2 norm and decay/β are one kernel, and the
    recurrence and the gated RMSNorm are another.

## What would go further

- **Smaller weights.** Decode is bound by the bytes streamed per token. 8-bit or FP4
  weights read in the GEMV would roughly halve or quarter them, at the cost of
  exactness.
- **Device-side sampling.** It would skip the 1 MB logits copy per token. That would
  mean returning a token rather than logits from `Step`.
- **Chunked Gated DeltaNet.** The prompt recurrence is still one position at a time
  and is now the largest prompt cost at 512 tokens. The chunked (WY) form would turn
  most of it into GEMMs.
- **Fused prompt attention.** The GEMM attention writes its scores to a scratch
  buffer, falling back to the tiled kernel past a 1 GiB budget. A tensor-core flash
  kernel would avoid both.
