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
  - The conv and q/k normalisation run in parallel over rows.
  - From 256 rows the Gated DeltaNet recurrence runs in its chunked (WY) form, 64
    rows at a time (below). Shorter prompts scan it position by position, one
    64-thread block per (value head, 8 columns).

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
0.3% of the largest logit. Greedy decoding still agrees token for token over 64
steps after prompts of 32, 512 and 1,300 tokens. After the 2,000-token prompt it
parts at step 5, where the managed path's top two logits are 0.0012 apart.
The f32 mode agrees there.

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
| **AMQL device engine** (BF16 prompts, the default) | **7,956 tok/s** | **64,310 tok/s** | **63,354 tok/s** | **679 tok/s** |
| AMQL device engine, `AMQL_GPU_PREFILL=f32` | 6,045 tok/s | 31,784 tok/s | 30,795 tok/s | 682 tok/s |
| llama.cpp, CUDA, F16, flash attention | 1,930 tok/s | 20,728 tok/s | 26,491 tok/s | 438–497 tok/s |

How to read this:

- **Decode stays flat with context.** It is ~610 tok/s after a 512- or 2000-token
  prompt. llama.cpp drops to ~460 after 512.
- **Exact prompts trade speed for f32.** The f32 mode runs at about half the
  default's prompt speed. It is still ~15% ahead of llama.cpp at 2,000 tokens, and
  the default ~2.4×.
- **The two AMQL rows come from a later session than llama.cpp's.** In that
  session the engine without the chunked recurrence ran at 8,205, 58,115 and 58,324
  tok/s (pp32, pp512, pp2000) and decoded at ~672. That is ~5% above the earlier
  session's 7,472, 55,354 and 53,755 tok/s and 655 tok/s on the same code, so
  compare rows from different sessions with that margin in mind.
- **Measured on an idle machine.** Best of three runs each. The engine's decode
  varies by about ±1% between runs and llama.cpp's by more (438 and 497 tok/s on
  two sessions).
- **Long prompts hold their speed.** Prefill at 2,000 tokens is within ~2% of 512,
  since attention is a small share of a prompt's time at these lengths.

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
  - Split-K across warps. For one or two activation rows (decode), four warps share
    each output row, and eight when the matrix is so narrow (n = 1,024) that four
    leave fewer than ~24 warps per SM. A sweep of 1/2/4/8 warps per row, 1/2/4
    output rows per warp, and prefetch on or off, over every Qwen 3.5 0.8B and 4B
    projection on an RTX 5090, found this within 2% of the best for each shape.
    From three rows the FMAs per weight grow, and fewer warps per row win (about
    32 warps per SM). `AMQE_GEMV_SPLIT=1|2|4|8` fixes it.
  - A grid-stride loop for the 248k-row head.
  - Epilogues fused in: bias, activation, the gate·up product, and the residual add,
    done in place.
- **Norms fused into the projection that reads them.**
  - An RMS norm without a bias is applied inside the dot loop. Each warp reads x and
    the norm's weights from cache and sums x² over its own columns, alongside the
    products. 1/rms is one number per row, applied in the epilogue. Nothing waits on
    a block-wide pass before the weights start streaming.
  - Staging the normalised rows in shared memory first, as every block used to,
    cost 13% of these projections' bandwidth.
  - Any other norm (LayerNorm, or one with a bias) is still staged that way.
- **What the GEMV reaches** on Qwen 3.5 4B at one row, with the RTX 5090 otherwise
  idle. It is measured over weight copies too big for L2, as one graph of
  back-to-back projections.

  | Projection | Before | Now |
  |---|---|---|
  | GDN in (norm fused) | 1,425 GB/s | 1,664 GB/s |
  | Attention qkv (norm fused) | 1,481 GB/s | 1,658 GB/s |
  | FFN gate·up (norm fused) | 1,397 GB/s | 1,661 GB/s |
  | FFN down | 1,477 GB/s | 1,641 GB/s |
  | Out projections | 1,650 GB/s | 1,670 GB/s |
  | Head | 1,683 GB/s | 1,706 GB/s |

  Across a token's projections that is 5.7 ms → 5.05 ms. With a game sharing the
  card, the served model decoded at ~85 tok/s against ~68 tok/s on the old kernels,
  measured old and new alternately under the same load.
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
- **The Gated DeltaNet recurrence in chunks.** Over 64 rows the delta rule unrolls
  to a triangular system, (I + A)·Δ = β·V − β·e^G·K·S₀. Its W and U factors, the
  decayed keys and the masked, decayed Q·Kᵀ depend only on the chunk's own rows.
  - The first kernel builds them for every chunk at once. It solves the triangle in
    16-row blocks, one column per thread.
  - The second kernel carries the state between chunks: Δ = U − W·S, then
    S ← e^(G_C)·S + K̃ᵀ·Δ. That is two small products per chunk instead of 64
    dependent steps. The next chunk's W and keys load (`cp.async`) while the current
    one computes.
  - The third kernel forms every chunk's outputs at once from the chunk's starting
    state and Δ.

  Decays stay in log space, so no ratio of two underflowed products forms. All of
  it is f32 on CUDA cores. At 512 rows the three kernels take ~90 µs per layer
  against ~140 µs for the row scan, which makes 512-token prompts ~11% faster. Below
  256 rows a single block's latency outweighs that, and the row scan is kept
  (`AMQE_GDN_CHUNK` moves the threshold; `0` turns the chunked form off).
- **A prompt scan built for latency.** Short prompts scan the recurrence position by
  position, so each step's latency is what counts.
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

## Speculative decoding

A model that ships an MTP drafter (Qwen 3.5's `mtp.*` tensors: one decoder layer, a
projector and three norms) drafts tokens ahead, and the model checks them all in one
pass. Decode reads every weight once per pass whether it has one row or eight, so
each draft the model would have chosen anyway is a token for the price of a GEMV row.

- **The drafter** runs on the device with a KV cache of its own. It reads the
  final-normed hidden state at position t beside the embedding of token t+1,
  `fc([norm(embed(t+1)) ; norm(h_t)])`, and through one layer, its norm and the
  shared head predicts t+2. Each draft is fed back for the next. It catches up on
  every prompt chunk and every kept verify row, so its cache holds every true pair.
- **A draft reads 32k of the head's 248k rows** (`AMQL_MTP_VOCAB`; a BPE vocabulary
  runs roughly from common to rare). As many drafts hold as with the whole head, at
  ~0.37 ms a draft instead of ~1.05 ms.
- **The verify** is the chosen token and the drafts as rows of one pass, each row's
  logits kept. With `RowInvariant` (on with a drafter) the GEMV's split no longer
  depends on the row count, a fused RMS norm runs in the dot loop at any row count,
  and the Gated DeltaNet takes its one-launch path up to 8 rows, so each row's logits
  are bit for bit what a one-token step gives. Attention already was.
- **A miss** keeps the rows before it: each Gated DeltaNet layer saved its state and
  conv history after every verify row, `Accept` restores the one at the last row
  kept, and the position moves back. The KV cache needs nothing; later rows are
  overwritten.
- **Suffix drafts** (server): when the reply's last 8 tokens appeared earlier in the
  context, what followed them there is drafted instead. These cost nothing to make
  and suit an agent copying paths and code.

Qwen 3.5 4B, RTX 5090, greedy code generation (300 tokens, `GpuEngine` alone):

| Drafts per step | tok/s | Tokens per step | Draft | Verify |
|---|---|---|---|---|
| none | 174 | 1 | | 5.7 ms |
| 2 | 303 | 2.56 | 0.78 ms | 6.85 ms |
| 3 | 325 | 3.02 | 1.11 ms | 7.42 ms |
| 4 | 330 | 3.38 | 1.43 ms | 7.95 ms |

Served (3 drafts, thinking on, idle GPU), the replies are identical to those without
speculation, greedy and seeded sampling alike. A 400-token essay runs 1.38–1.41×
faster and a 600-token sampled reply to a real 23k-token Hydra request 1.65×. Prose
under sampling keeps about half the drafts.

`AMQL_MTP_DRAFTS` (default 3, 0 off), `AMQL_SUFFIX_DRAFT` (default 8, 0 off),
`AMQL_MTP=0` (no drafter, and no row invariance). Speculation runs while one reply
runs alone. A batch step passes the drafter by, and that sequence drafts again only
after a reset or a rewind.

## 4-bit weights (NVFP4, MXFP4)

Decode reads every weight once a token, so it runs as fast as the weights are
small. The engine runs projections stored at 4 bits as they are stored.

- **Compile once.** `amql-cli quantise <container> --out <dir> --format nvfp4|mxfp4`
  writes every projection of the decoder stack and the MTP drafter as FP4 E2M1
  elements with their block scales. That is q/k/v/o, the Gated DeltaNet's
  in- and out-projections, the FFN and the drafter's fc. NVFP4 has an E4M3 scale
  per 16 and an F32 tensor scale; MXFP4 has an E8M0 scale per 32. The embedding,
  the head, norms and convolutions stay as they were. On Qwen 3.5 4B the stack
  and drafter go from 7.38 GB to 2.08 GB (NVFP4) or 1.96 GB (MXFP4), in about 80 s.
- **Every runtime then multiplies by the same weights.** The operand store
  decodes them for the managed path, and the GPU engine reads them as stored. The
  two agree to f32 rounding, as at full precision (tested on both formats). An
  Entail adapter runs its own program over the token stream at full precision,
  whatever the model's weights are. A drafter's acceptance measured on the 4-bit
  model holds wherever it runs.
- **Or quantise at load.** `AMQL_GPU_FORMAT=nvfp4|mxfp4` (`--gpu-format` in
  `amql-cli`) quantises an ordinary container's projections as the engine uploads
  them. It uses the same quantiser on the same set, so the engine's logits are bit
  for bit those of the compiled container. The managed path still reads the
  originals. `AMQL_GPU_FORMAT_HEAD=1` quantises the head as well (the engine only).
- **Decode** decodes FP4 in registers. A lane loads 16 bytes (32 elements) a step,
  and a byte permute turns four elements at a time into f32s: E2M1's magnitudes
  are f32s whose low two bytes are zero. There are no conversion instructions,
  which run at a fraction of the integer rate. The block scale multiplies each
  8-element partial sum, and the tensor scale multiplies each output row's sum.
  Stacked weights (q/k/v) keep each part's own tensor scale per row.
- **Short rows are costly.** A 4-bit row of 2,560 columns is only 1,280 bytes, so
  a row's fixed cost (the warps' reduction, the epilogue) weighs four times as
  much. A warp takes four output rows, and rows under 4,096 wide take one warp
  each (four from 4,096).
- **Code size matters at several rows.** Past one activation row, the 4-bit step
  loop is not unrolled. Unrolling it thrashed the instruction cache: two rows cost
  twice one.
- **Prompts** decode the weight to BF16 (exact: element × block scale has at most
  6 significant bits) for the BF16 tensor-core GEMM, and apply the tensor scale in
  the epilogue. With `ExactPrefill` they decode to f32 for the SGEMM instead.
- **Verify rows stay bit-identical to steps** over 4-bit weights: the split rule
  is fixed per shape, never tuned by timing, so a reply is the same from run to run.

Qwen 3.5 4B decode GEMVs on an RTX 5090, per token (all of them, head included):

| Weights | 1 row | 4 rows |
|---|---|---|
| BF16 | 5.06 ms (1,661 GB/s) | 5.40 ms |
| NVFP4 | 2.36 ms (1,002 GB/s) | 4.76 ms |
| MXFP4 | 2.46 ms (906 GB/s) | 4.67 ms |

Qwen 3.5 4B end to end: compiled containers (the head stays BF16), greedy code
generation (`GpuEngine` alone, idle GPU). Quality is teacher-forced on 4,095
tokens of this repo's docs and C#:

| Weights | Decode | With 2 MTP drafts | Perplexity | Top-1 as BF16 |
|---|---|---|---|---|
| BF16 | 169 tok/s | 316 tok/s | 7.54 | — |
| NVFP4 | 272 tok/s | 348 tok/s | 7.86 (+4.3%) | 90.4% |
| MXFP4 | 267 tok/s | 345 tok/s | 8.24 (+9.4%) | 88.0% |

Speculation gains less at 4 bits, because each verify row costs about 1 ms against
0.2 ms at BF16. Two drafts do best.

### Rounding: GPTQ

Rounding each weight to its nearest 4-bit value keeps the *weights* close.
What matters is the *layer's output*. With `--calibration <file-or-folder>`,
`amql-cli quantise` rounds by GPTQ (Frantar et al., 2022):

- The source model reads the calibration text on the GPU one layer at a time.
  For each input a projection reads, it collects H = XᵀX. Those inputs are the
  normed residual for q/k/v, the GDN in-projections, gate and up; the attention
  context or GDN output for o/out; and the gated activation for down.
- Each row is rounded a column at a time. Every column's rounding error is
  carried onto the columns still to come, weighted by H⁻¹, so the output over
  those inputs moves least. Rows are independent: a thread takes one row of a
  32-column block, and a cuBLAS rank-32 update carries the block's error on.
- Each block's scale is chosen when the block is reached, from its corrected
  values. NVFP4 tries six E4M3 scales around the peak's and MXFP4 tries two
  powers of two; the one with the least weighted error wins.
- `--rounding search` does the scale search alone, without calibration.
- On the 4B the whole thing takes about 70 s: 1.2 s of Grams per layer, H⁻¹ and
  rounding on the GPU. The stored layout is unchanged, so every runtime reads it
  as before. `index.json` records the rounding used.

`amql-cli quantise compare <reference> <candidate>... --text <file> [--tokens 4096]`
measures a quantisation where it matters. For each candidate it reports the KL
from the reference's next-token distribution at every position, perplexity, and
top-1 agreement. Perplexity alone hides a lot: a model can keep its average loss
while changing its choices.

GPTQ is the default. `amql-cli quantise` uses it without being asked,
calibrating on a built-in text (this repository's docs and C#, which
carried over to Python it never saw) unless `--calibration` names your own.
Asking to *run* at 4 bits (`--gpu-format nvfp4`, `AMQL_GPU_FORMAT`, or
`"format": "nvfp4"` for a model in amql-server's `--config`) loads the
container's GPTQ copy, `{container}-nvfp4-gptq`, built beside it the
first time (about a minute for the 4B) and reused while it is newer than
the container. Rounding each weight to nearest as it loads is what is left
for a container GPTQ cannot calibrate.

Qwen 3.5 4B, calibrated on 32k tokens of this repo's other docs and C#.
Evaluated on 4,096 tokens of held-out docs and C#, and on 4,096 tokens of
Python (vLLM model code, a domain the calibration never saw):

| Weights | Docs/C#: perplexity | KL | Top-1 as BF16 | Python: perplexity | KL | Top-1 |
|---|---|---|---|---|---|---|
| BF16 | 7.533 | — | — | 3.352 | — | — |
| NVFP4, nearest | 7.856 (+4.3%) | 0.046 | 90.4% | 3.517 (+4.9%) | 0.054 | 92.4% |
| NVFP4, searched scales | 7.790 (+3.4%) | 0.038 | 91.7% | 3.485 (+4.0%) | 0.042 | 93.4% |
| NVFP4, GPTQ | **7.618 (+1.1%)** | **0.013** | **95.3%** | **3.395 (+1.3%)** | **0.019** | **95.4%** |
| MXFP4, nearest | 8.240 (+9.4%) | 0.081 | 88.0% | 4.062 (+21%) | 0.167 | 86.5% |
| MXFP4, GPTQ (16k tokens) | 7.691 (+2.1%) | 0.027 | 92.5% | 3.492 (+4.2%) | 0.042 | 93.8% |

GPTQ cuts the KL to a third on NVFP4 and MXFP4 alike, on the held-out domain as
well as the calibrated one. The worst position's KL falls from 5.3 to 0.5. Decode
speed is that of the format: the weights are the same size and are read the same
way. Calibration on 16k tokens instead of 32k gave KL 0.0149 / 0.0211.

The 4-bit kernels move fewer bytes per second than BF16: the decode is work BF16
does not do. A 4-bit row also finishes in a quarter of the time, so its fixed
costs show.

## Routed experts: grouped launches

Routed (MoE) layers do not run on the device engine yet; the managed path
runs them, with every expert held on the GPU done in one call
(`amql_cuda_ffn`). That call used to run expert by expert: six launches each
(gather, gate, up, SiLU, down, scatter), so a decode token on Flash-Next's
geometry (10 experts routed of 512, 48 layers) took ~2,900 launches, each too
small to fill the GPU.

Following NVIDIA's grouped GEMM for dropless MoE (one kernel over every
expert's actual rows, no padding to a capacity, no per-expert host
round trip), when every expert is NVFP4 with at most 32 rows (decode, small
batches) a layer now runs as five launches whatever the expert count:

- one gather of every expert's rows;
- one launch of every expert's gate and up products. A task is one product over
  up to 8 rows; `blockIdx.y` picks the task and the warps of `blockIdx.x` its
  output rows;
- one SiLU over every hidden unit;
- one launch of every down product;
- one combine that adds each row's experts in the order the per-expert scatter
  did.

Each warp does `nvfp4_gemv`'s arithmetic term for term, so the output is
bit for bit the per-expert path's (tested). Products that share an input
(q/k/v, the GDN's in-projections) group the same way when all are NVFP4.
`AMQL_GROUPED_FFN=0` turns both off.

A routed layer at hidden 2560 with experts 640 wide, top 10 of 64, NVFP4, timed
end to end through `FfnKernel.Routed` (routing on the host included, Debug build):

| Rows | Per expert | Grouped | Speed-up |
|---|---|---|---|
| 1 (decode) | 566–595 µs | 181–194 µs | 3.1x |
| 4 | 2.28–2.50 ms | 306–331 µs | 7.5x |
| 8 | 3.40–3.70 ms | 507–545 µs | 6.8x |

The weights the 1-row case reads (30 matrices of 0.9 MB) take ~15 µs at the
card's bandwidth, so the remaining ~180 µs is fixed cost. That is the next
target: routing on the host, three uploads and a download, and a stream
sync each layer.

## What would go further

- **Routed experts on the device engine.** Bringing MoE layers into `GpuEngine`
  would remove that fixed cost, as the post's own gains did:
  - a fused router on the device (softmax, top-k, and the per-expert slot lists
    built there);
  - the grouped launches above reading device-side counts, so no host round trip
    and no copy of expert sizes back to the host;
  - the whole step captured in a CUDA graph, which a host-sized expert loop rules
    out.

  The same grouping extends to MXFP8 and FP16 experts and, for prefill, to a
  grouped tensor-core GEMM (cuBLASLt grouped, or one W4A4 launch over a
  ragged batch) in place of a GEMM per expert.

- **Faster 4-bit GEMVs.** They reach 0.9–1.0 TB/s against 1.66 for BF16, and
  each extra activation row costs ~0.6 ms a token, where BF16 hides it under
  memory. The kernel is bound by issue and latency, not bandwidth. Candidates are
  fewer registers per warp (more warps to hide latency), the RMS norm applied once
  per row outside the dot loop, and packed half2 products.

- **A cheaper verify.** A row costs ~0.55 ms: four rows run at 7.4 ms against 5.7 ms
  for one. Part of that is the decode split rule at 3+ rows, and part is a verify
  launched kernel by kernel rather than from a captured graph.

- **Closer 4-bit weights.** GPTQ calibrates each layer on the full-precision
  model's inputs. Calibrating on the inputs the already-quantised earlier layers
  produce would let each layer correct what came before it. Past that,
  distillation from the BF16 model (training the 4-bit model's scales or a
  low-rank correction against BF16 logits) is how NVIDIA's QAD checkpoints get
  within noise of full precision.

- **Smaller weights.** Decode is bound by the bytes streamed per token. 8-bit or FP4
  weights read in the GEMV would roughly halve or quarter them, at the cost of
  exactness.
- **Device-side sampling.** It would skip the 1 MB logits copy per token. That would
  mean returning a token rather than logits from `Step`.
- **Tensor cores in the chunked recurrence.** Its products run as f32 on CUDA cores,
  and each kernel has only one block per (head, chunk) or (head, column slice).
  - The chunk products, Q·Kᵀ and K·Kᵀ, take ~90 µs per layer at 512 rows, about a
    fifth of the prompt's time.
  - In the default mode, TF32 `mma` tiles could cut most of that. The f32 mode would
    keep the present kernels.
- **Fused prompt attention.** The GEMM attention writes its scores to a scratch
  buffer, falling back to the tiled kernel past a 1 GiB budget. A tensor-core flash
  kernel would avoid both.
