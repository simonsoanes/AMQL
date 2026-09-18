# CUDA Backend Plan — AMQL

Status: **partially implemented** · Target: net10.0 + native CUDA interop

## 0. Implemented so far (this session)

- **On-demand working sets, CPU (§4, P0.5 done).** `WeightLoader` has
  three modes, chosen by `AMQL_WEIGHTS` (`f32` default | `bf16` |
  `mxfp4`), inherited by every runtime consumer automatically:
  - `OnDemandBf16`: stack projections resident as stored bytes, widened
    into a bounded LRU on access — bit-exact with the f32 path (tested),
    ~2× the memory saving.
  - `Mxfp4`: stack projections resident as MXFP4 packs (~13% of f32),
    dequantised into the bounded LRU on access — deterministic, tolerance-
    gated, never bit-exact. The per-element E8M0 `pow` in the dequant hot
    loop was replaced with a per-block decode (the original was the 5×
    slowdown on small models).
  - Embeddings, head, norms, `A_log` stay f32-resident in every mode.
- **CUDA GEMM core, opt-in (§2, P0/P1 subset done).** `native/amql_cuda`
  builds `amql_cuda.dll` (nvcc + cuBLASLt, `scripts/../build-cuda.cmd`):
  device-resident FP16 weights dequantised from the MXFP4 packs (lossless:
  FP4×E8M0 values are exact in FP16), row-major transposed-B GEMM with
  FP32 accumulate, per-shape cuBLASLt plan cache, graceful CPU fallback.
  Wired behind `AMQL_GPU` + `AMQL_WEIGHTS=mxfp4`; GEMMs ≥ 8M MACs with a
  device copy route to the GPU, everything else stays managed. Parity
  test: GPU-MXFP4 vs managed-MXFP4 to 1e-4.
- **Not yet done:** elementwise/norm/attention/gated-delta/RoPE kernels on
  device, KV on device, streams/overlap, per-layer FP32 mirror LRU, the
  full op inventory of §5.2 (merge/MTP/moe-ify numerics on device), and
  the read-model acceptance table of P7. Current per-token decode is
  memory-bound on host↔device round trips for m=1 GEMMs, so end-to-end
  speedup on small models is not yet a win — the win so far is the flat
  working set (the 27B-pruned runs at ~30-85 GiB RSS instead of ~95 GiB
  thrash, and the machine stays responsive).

## 1. Goal and scope

Speed up every numerically heavy operation in AMQL by (a) cutting the
runtime working set to **MXFP4 weights + FP32 accumulate** — the change
that makes the 27B runnable at all — and (b) offloading the hot kernels to
the GPU, while keeping the CPU path as the always-available reference. The
runtime today is a managed `float[]` / `Tensor2D` executor with hand-written
SIMD kernels; the 27B pipeline (encode → prune → merge → moe-ify →
generate-mtp → export) taught us the machine reality that motivates this:

- A full 27B BF16 weight set (~47 GiB) doubled to f32 (~94 GiB) exceeds the
  127 GiB host RAM comfortably and **far exceeds the 32 GiB VRAM**.
- MXFP4 projection weights shrink the stack to ~12.5 GiB (full) / ~6.2 GiB
  (pruned) — the full 27B becomes host-runnable and even VRAM-resident.
- Decode is position-major with a stateful linear-attention recurrence
  (48 of 64 layers on the full model); the softmax layers prefill batched.

"All operations we can optimise" = the operations inventory in §5. Ops
that are I/O-bound or dominated by graph file work (encode, safetensors
sharding, export's byte copy) are out of scope; only their compute stages
(quantisation, widening) are candidates.

Environment (verified on this machine):
- NVIDIA GeForce RTX 5090, compute capability 12.0, 32607 MiB VRAM
  (≈27.8 GiB free at idle).
- CUDA Toolkit v13.3 (also v13.0) installed, `nvcc` available.
- .NET 10.0.303, Windows. No external numerics dependency in the managed
  stack today, by design ("deliberately no external numerics dependency").

## 2. Design principle

**Native CUDA library + P/Invoke, not a managed-CUDA dependency.** The
managed core keeps zero third-party nuget packages. A thin native library
(`Amql.Cuda`) exposes a flat C ABI of kernels/helpers; `Amql.Inference`
gains a small interop layer that:

- probes for the device once and toggles a global `GpuEnabled` flag
  (env `AMQL_GPU=off|0|1`, default: available);
- is **always optional** — every call site falls back to the existing
  managed op on detect-failure, allocation-failure, OOM, or small-op
  cutoff (see §7);
- respects the existing `ComputeBudget` (cores now also partitions the
  "use GPU vs leave it free" decision; a new VRAM budget parallel to
  `AMQL_MEMORY_GB`, e.g. `AMQL_VRAM_GB`, defaults to 80% of device memory).

Surgery is incremental: existing ops stay intact; GPU variants are added
beside them (a `GpuOps` static class mirroring `TensorOps`/`FfnKernel`/
`AttentionKernel`/`GatedDeltaKernel`/`Norms`), and the runtime dispatch
chooses per-call based on size. No architectural rewrite of
`GenericRuntime` / `Planner` / `OperandStore` is required for the compute
layer.

## 3. Host↔device memory model

The container is the authority; device memory mirrors it.

- **Device weight cache**: a `CudaWeights` cache keyed by
  `(objectId, tensorName)` exactly like `WeightLoader`, populated lazily
  from the segment bytes **without CPU-side widening**:
  - keep **BF16 storage on device**; the numeric kernels use cuBLAS
    BF16×BF16→FP32 (or a convert-on-first-use FP32 mirror **per layer**,
    evicted on VRAM pressure — see §6). This is the difference between
    "pruned model fits" and "doesn't fit".
  - upload once per session; reuse the segment-relative tensor name.
- **Activation staging**: small fixed device arena for h, q/k/v, logits,
  KV rows; host↔device copies only at op boundaries, pinned host staging
  buffers for the copies that must happen.
- **KV / recurrent state**: softmax KV rows remain a device growable ring
  per layer; the GatedDelta `LinearAttentionState` (S per key head,
  conv history) becomes a device struct per layer, advanced in place by
  the recurrence kernel.
- **32-bit ceiling audit**: tensors with element counts near 2^31 (the
  embedding 248320×5120 = 1.27e9 f32, the head logits row 248320) fit in
  `int` index space but every cuBLAS/custom kernel dimension must be
  checked; use 64-bit shape plumbing where the reference uses `long[]`
  shapes today (weights already carry `long[]`; the runtime should pass
  `int` dims only after an explicit overflow guard, mirroring the existing
  `checked` element-count paths).

## 4. Numerics and determinism policy

**Working-set precision: MXFP4 weights, FP32 accumulate — never FP4
activations.** The internal execution format is:
- **MXFP4 (FP4 E2M1, per-32 E8M0 block scales)** for stack projection
  weights only — q/k/v/o, linear-attn in/out projections, FFN
  gate/up/down, MoE router/experts. This is exactly the tensor split
  `export --quant mxfp4` already implements (embeddings, norms, biases,
  `A_log`, output head stay full precision), so the format the runtime
  computes in matches the format the exporter produces.
- **BF16 for embeddings, output head, norms, `A_log`; FP32 for
  activations, residuals, the KV cache, and all GatedDelta state `S`.**
  FP4×FP4 accumulation is not a supported path — it would destroy quality
  and the recurrent state would drift.
- Workset arithmetic: 0.5B/element + 1B per 32 → ~0.53 B/elem ≈ 27% of
  BF16 ≈ 13% of f32. Full 27B ≈ 12.5 GiB; pruned 24-layer ≈ 6.2 GiB.
- **On Blackwell (CC 12.0, the 5090)**: native FP4 tensor cores — FP4
  GEMM with FP32 accumulate via cuBLASLt. **On CPU**: block-dequantise to
  f32 scratch per layer on read (a memory win only; the win is exactly
  what is "killing things" today — the f32 working set).
- **Precision-critical one-shot numerics stay BF16/f32/fp64**: merge
  alignment (fp64 Gram + Cholesky), MTP ridge fits, prune corpus ranking,
  moe-ify PPL gates. These are not the per-token hot loop; accuracy
  matters more there than throughput.
- The canonical container stays **BF16, byte-exact** — encode/verify/hash/
  re-encode are untouched. MXFP4 is applied at the `WeightLoader` boundary
  as an internal execution format, never written back to the container.
- **Deterministic-but-not-bit-identical**: the repo has byte-exactness
  guarantees (fit determinism hash, `Profile.Exact`, byte-identical
  re-encode). GPU accumulation order and MXFP4 dequant paths differ from
  the managed f32 kernels, so:
  - all existing determinism/hash/byte-exact tests continue to gate the
    **CPU BF16→f32 path only** (unchanged);
  - a new parity suite asserts *tolerance* (per-op max abs/rel error,
    e.g. 1e-3 for FP4-weight GEMMs with FP32 accumulate), not bit
    equality; plus GPU-repeatability (same GPU twice → same result);
  - `verify`/re-encode remain CPU-side and byte-exact — GPU and the MXFP4
    working set never write container segments.
- Stateful recurrence (GatedDelta) has a strict position order on CPU; the
  GPU kernel must keep the same position-serialisation (only heads/data
  parallelise), otherwise the kvMem update order changes numerics.

## 5. Operations inventory (what gets offloaded)

Each row: managed surface today → GPU plan → win estimate.

### 5.0 The shared working-set change (applies to every row)

`WeightLoader` / `CudaWeights` serve **one** resident weight format per
session: MXFP4 for stack projections, BF16 for embeddings/norms/`A_log`/
head (matching `export --quant mxfp4`'s tensor split), with per-layer
f32-dequant scratch on CPU and FP4 tensor-core GEMMs with FP32 accumulate
on Blackwell. Nothing below repeats this; the rows list where the kernels
land.

### 5.1 Inference core (priority 1)

| Op | Managed site | GPU plan | Win |
|---|---|---|---|
| `MatMul` / `MatMulTransposedB` | `TensorOps` in `Tensor2D.cs` (SIMD, row-parallel, transpose-free) | cuBLAS `cublasSgemm` (fp32 or bf16→fp32); transposed-B encoded as cuBLAS op-tranpose, no host transpose | dominant: ~all projections |
| Embedding gather | `TensorOps.GatherRows` / `GenericRuntime.Embed` | gather kernel (or cublas with one-hot when batch is large; direct copy kernel for 1-8 rows) | small but lazy |
| Norms (RMSNorm family) | `Norms.ApplyInPlace/ApplyRow` (`Kernels.cs`) | row-normalise kernel reading the same weight vector on device | every layer |
| Activations (SiLU / GELU-tanh / ReLU) | `Activations` (`Kernels.cs`), used inside `FfnKernel` | fused elementwise kernel | fused with FFN |
| Dense FFN (gate/up/down) | `FfnKernel.Dense` | 3 GEMMs + fused SiLU-gate kernel (or single fused kernel with in-register weights) | big (intermediate 17408) |
| Routed MoE FFN | `FfnKernel.Routed` | router-score GEMM, softmax + top-k select kernel, then **one** expert-matrix GEMM per selected set (ungrouped select first; grouped/bucketed experts later) | moe-ify outputs |
| Softmax attention (GQA, causal, window, sinks, softcap) | `AttentionKernel.Execute` + `SoftmaxInPlace` + `Rope.Apply` (`Attention.cs`) | QKᵀ GEMM, fused causal-softmax (max/exp/sum online), PV GEMM; RoPE as a fused elementwise on k/v on write + q on read | softmax layers |
| Logits head + final norm | `GenericRuntime.FinalNormAndHead`, head GEMM [1×5120]@[248320×5120]ᵀ | cuBLAS norm + bf16 GEMM emitting [1×248320], fused final-RMSNorm | head is a big GEMM per token |
| Sampling | `Sampler.ArgMax/Sample` (`Sampler.cs`) | argmax reduce kernel; multinomial via device RNG + prefix scan (or return logits row to host — 1 MB, fine) | minor; keep host for now |
| GatedDelta conv1d | `GatedDeltaKernel.ConvForward/ConvStep` | depthwise causal conv kernel over [channels×T], channels-parallel | every linear layer |
| GatedDelta recurrence | `GatedDeltaKernel.Recurrent` (position-sequential, head/data inner) | custom kernel: **position-serial**, heads×data parallel (48 v-heads×128 on 27B); state S updated in device memory | the 48-layer majority on 27B |
| Gated RMSNorm | `GatedDeltaKernel.GatedRMSNorm` | fused with recurrence tail | linear layers |
| q/k L2-norm rows | `GatedDeltaKernel.L2NormRow` | fused into recurrence entry | linear layers |
| Per-head norms / chunk/col helpers | `GenericRuntime.ApplyPerHead/ChunkBlocks/ColRange` | elementwise kernels or fold into neighbours | small |

### 5.2 Tooling / merge / MTP (priority 2)

| Op | Managed site | GPU plan | Win |
|---|---|---|---|
| Ridge fit Gram (BᵀB, BᵀA) | `LeastSquares.Fit` (`LeastSquares.cs`) — the 5120-wide merge fit we saw take ~45 min | `cublasSgemm` for both Gram products (fp32; the managed accumulator is f64 — decide: fp32 accumulate + Cholesky in f64 host copy, or fp64 gemm `cublasDgemm` for parity) | merge alignment ~10-30× |
| Cholesky + triangular solves | `LeastSquares.Cholesky/SolveLower/SolveUpperTranspose` (also `FitProjection`) | cuSOLVER `potrf`/`trsm` (fp64 to match), keep small | fits |
| Embedding alignment residual | `LeastSquares.ResidualL2` | reduction kernel | merge |
| Merge blend / consensus / energy restore | `ModelMerger` (ApplyMap, Cosine, RestoreEnergy, per-token loop) | per-token kernels (map apply via GEMM + row cosines) | merge |
| MTP pair collection | `PairCollector` (forward pass over corpus) | inherits §5.1 inference kernels | generate-mtp --fit |
| MTP projector fits | `MtpFitter`, `MtpKProjectors` (ridge, K-projector mixture) | Gram+solve as above; K-projector routing = small | fit |
| Draft acceptance gate | `DraftContext.Score` (trunk forward per block) | inherits §5.1 | fit sweep |
| moe-ify clustering | `MoeIfy.ClusterUnits` (k-means over 17408×T usage, balanced re-assign) + `DistanceSquared` | distance/centroid kernels (data-parallel over units) | moe-ify |
| moe-ify slicing | `GatherRows/GatherCols`, `RouterRows` | device gathers/means | moe-ify |
| Prune corpus scoring | `LayerPruner` corpus approach (one forward) | inherits §5.1 | prune --approach corpus |
| Export widening / mxfp4 quant | `ModelExporter`/`BitPattern` widen + `Mxfp4` encode | optional: device convert/quant kernels; keep host initially (I/O-bound) | low |

### 5.3 Explicitly out of scope
- Encode/mapping, safetensors hashing, segment writes (I/O-bound, byte-exact).
- Tokenizer, graph planning, `route`/`path` template probing (small compute,
  host RNG heavy), patch machinery (writes container-related artifacts).

## 6. VRAM budget and the MXFP4-on-device decision

- Full 27B f32 on device = ~94 GiB → **impossible**. Even the pruned
  24-layer container in f32 ≈ 46.8 GiB → **impossible**.
- Therefore projection weights live on device **as MXFP4** (FP4 E2M1 +
  per-32 E8M0 block scales): the pruned stack ≈ 6.2 GiB, the full 27B
  stack ≈ 12.5 GiB. Embeddings, norms, `A_log` and the output head stay
  BF16 on device (small), and activations/state stay FP32. 32 GiB VRAM
  then comfortably holds the **full 27B** + activations + KV + recurrent
  state — the same decision that makes the host working set tractable
  (§4).
- On Blackwell the GEMMs run as FP4 tensor-core ops with FP32 accumulate
  (cuBLASLt); on CPU the same MXFP4 working set is block-dequantised per
  layer to f32 scratch (memory win; compute stays f32).
- When a model exceeds `AMQL_VRAM_GB` (default 80% of device), the loader
  falls back to a **per-layer FP32 mirror with LRU eviction** (convert one
  layer's weights on demand, reuse the existing compute-budget-style
  refusal message when even one layer cannot stage). If that fails → CPU
  path, exactly like today.
- Host-side `WeightLoader` f32 cache (the ~2× memory blowup) is *not*
  needed: the MXFP4 working set is built once from segment bytes (via the
  existing `Mxfp4.Quantize` path used by export) and uploaded, widened
  only in per-layer scratch or on the GPU.

## 7. Dispatch and small-op cutoff

GPU launch overhead (~10-50 µs) dwarfs tiny ops. `GpuOps.Try*` gate every
call on `GpuEnabled && workBelowCutoff == false`:

- cutoff by minimum FLOPs (e.g. ≥ 1M MACs) or matrix size (≥ 64×64);
- norms/activation/gather on single-row decode inputs stay fused into the
  surrounding GEMM kernels where possible, not separate launches;
- prefill (batch ≥ 8 positions) is the GPU sweet spot; decode
  (1 row/token) still benefits from the big head/FFN GEMMs but overlaps
  with the position-serial state advance on a second stream.

## 8. Implementation phases

Each phase ends with a runnable gate (build + existing CPU tests green +
new GPU test green + a 30 s smoke on the demo container).

- **P0 — Interop skeleton.** `Amql.Cuda` native lib buildable via
  `scripts/build-cuda.cmd` (nvcc + CUDA 13.3, x64 release); C ABI:
  `cuda_available`, `cuda_init(vramBudgetBytes)`, `cuda_alloc/copy/free`,
  device info dump; P/Invoke shim in `Amql.Inference/Cuda` with lazy
  load, graceful no-device fallback; `AMQL_GPU` env honor.
  *Gate: `cuda probe` prints device; demo generate byte-identical to CPU.*
- **P0.5 — Host MXFP4 working set (CPU, no GPU).** `WeightLoader` gains an
  MXFP4 resident path: build the FP4+scales working set once per session
  from segment bytes (`Mxfp4.Quantize`), block-dequantise each layer into
  f32 scratch on read, LRU the scratch. The memory blowup (f32 widening)
  dies on the CPU path alone, before any CUDA lands. The canonical
  container and `verify` stay BF16/byte-exact.
  *Gate: pruned-27B `generate` completes without the 90+ GiB working set;
  demo generate tolerance-parity with the BF16→f32 path; a
  `--weights mxfp4|bf16` runtime switch (default bf16 for now).*
- **P1 — GEMM core.** `cuda_sgemm_nt/tt` etc. wired into
  `TensorOps.MatMul/MatMulTransposedB` through the Try-gate; **MXFP4
  weight paths in `CudaWeights`** (FP4 tensor-core GEMM with FP32
  accumulate via cuBLASLt; bf16/f32 fallback per layer); shape-overflow
  guards.
  *Gate: parity tests on random 5120-shape GEMMs (tol 1e-3 for the FP4
  path); demo + pruned-24 container `generate` runs with ≥2× speedup vs
  CPU.*
- **P2 — Elementwise bundle.** Norms, activations (fused with the P1 FFN
  GEMMs), RoPE, embedding gather, per-head helpers.
  *Gate: hybrid 0.8B container parity (linear+softmax layers) on a corpus
  forward; error ≤ 1e-4.*
- **P3 — Softmax attention + KV on device.** QKᵀ, fused causal softmax
  (sinks, window, softcap), PV, device KV ring; prefill path first,
  decode after.
  *Gate: full-attention demo parity; pruned container prefill parity.*
- **P4 — GatedDelta stack.** Device `LinearAttentionState`; conv kernel;
  position-serial/head-parallel recurrence; gated RMSNorm; Q/K L2-norm
  fused. This is the bespoke, highest-risk kernel.
  *Gate: linear-attention parity on the real 0.8B hybrid; the 27B pruned
  single-step smoke (must go from "never finished on CPU" to < 60 s).*
- **P5 — Routed MoE FFN.** Router GEMM + softmax/top-k + selected-experts
  GEMM; then bucketed/grouped expert GEMM (phase 2 of P5).
  *Gate: moe-ified demo parity; moe-ified pruned-27B smoke.*
- **P6 — Tooling numerics.** `LeastSquares` Gram via GEMM, cuSOLVER
  potrf/trsm; merge blend kernels; moe-ify clustering kernels; MTP fits.
  *Gate: 5120-wide merge alignment finishes in minutes, residual within
  1e-4 of CPU; generate-mtp --fit sweeps in minutes.*
- **P7 — Streaming/overlap polish.** Second stream for KV/state advance,
  per-layer FP32 mirror LRU, VRAM pressure handling, multi-request
  batching hooks, `verify --device gpu` style diagnostics.
  *Gate: pruned-27B generate token-rate table (CPU-BF16 vs CPU-MXFP4 vs
  GPU-MXFP4 vs FP32-mirror-GPU).*

## 9. Verification strategy

- All 166 existing tests stay green on the CPU path (unchanged semantics;
  the `WorkerCount` budget test remains valid).
- New `tests/Amql.Tests/CudaTests.cs`:
  - probe/allocate/copy round-trip;
  - per-op parity vs the managed op on random tensors at 27B shapes
    (5120×17408 FFN, 12288×5120 q_proj, 248320×5120 head, 1×248320
    logits) with tolerance 1e-4;
  - hybrid-stack forward parity on a real encoded 0.8B container;
  - correctness of GatedDelta recurrence vs the managed kernel over many
    positions (same seed corpus, tolerance-gated);
  - OOM drill: force `AMQL_VRAM_GB` small → fallback to CPU mid-session.
- End-to-end: after P4/P5/P6, one pipeline run
  (encode·cache · prune · merge · generate-mtp · moe-ify) on the pruned
  27B using the GPU path; the PR and export use the CPU byte-exact paths
  unchanged.

## 10. Risks and open questions

1. **Numerics drift** — GPU accumulation order changes results at ~1e-5;
   the fp64 Gram/Cholesky merge path must decide fp32 vs fp64 parity
   (recommend: keep the managed fp64 solve as the authority for the merge
   alignment, GPU accelerates only the Gram products, then copies back).
2. **BF16 GEMM support** — cuBLAS BF16×BF16→FP32 (`CUBLAS_COMPUTE_32F`
   with `CUBLASLT_ORDER_COL` etc.) is available on CC 12.0; if the pair
   of input dtypes is restricted in practice, fall back to one-time
   per-layer FP32 conversion (kept in the plan as the LRU mirror).
3. **Position-serial recurrence** — the GatedDelta loop is sequential;
   expect a ~heads×vectorization speedup, not a 1000× one. The win on the
   27B is still large because 48/64 layers are linear-attention and the
   FFN/attention GEMMs around them are the real cost.
4. **P/Invoke + spans** — large `float[]`/`byte[]` marshalling must use
   pinned buffers (or unsafe fixed) to avoid copying; the 2 GiB array
   ceiling still applies host-side (chunked upload, matching the segment
   layer already built for ResolveWidened).
5. **Driver/toolchain** — CC 12.0 needs CUDA ≥ 12.8; 13.3 is present.
   `dotnet build` must not require nvcc (the native lib builds via
   `scripts/build-cuda.cmd`, checked-in or CI-built; managed builds stay
   green with the lib absent).
6. **Determinism tests** — byte-exact tests are CPU-only by construction;
   a GPU determinism test asserts repeatability (same GPU twice) not
   cross-backend equality.

## 11. Deliverable shape

- `native/amql_cuda/` — CMake or raw nvcc build, `amql_cuda.dll`, C ABI
  headers + a `cuda_api.natvis`-style IDL comment block.
- `src/Amql.Inference/Cuda/` — `CudaShim.cs` (load/probe), `CudaWeights.cs`
  (device weight cache + VRAM budget), `GpuOps.cs` (Try-gated op mirrors),
  `CudaOpsDispatch.cs` (cutoff + fallback).
- `scripts/build-cuda.cmd`, `README` section, phase gates as runnable
  smoke commands.

The CPU remains the byte-exact reference and the fallback for any
capacity/numerics condition; the GPU is an accelerator behind a
Try-gated, budget-aware seam — never a second source of truth for
artifacts.