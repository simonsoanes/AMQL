# CPU inference performance

How fast the generic runtime (`GenericRuntime` / `DecodeSession`) serves one request on
the CPU once a model is loaded, how that compares with llama.cpp, and what the runtime
does to get there. Every change described here leaves the numerics **bit-identical**:
`amql-cli bench` prints a checksum of the logits' bit patterns, and it is the same
before and after (`f8ddb1c22f5fa2e0` for the run below).

## Measuring

```bash
amql-cli --cpu bench <container> --prompt-tokens 32 --steps 128 --repeat 3
amql-cli --cpu bench <container> --prompt-tokens 512 --steps 128 --repeat 3 --profile
```

`bench` loads the model, runs one untimed warm-up request, then times `--repeat`
requests of a `--prompt-tokens` prefill followed by `--steps` greedy decode steps.
`--profile` splits one prefill and one decode run by layer operator into time inside
the parallel kernels and serial time.

llama.cpp was measured with `llama-server` (the CPU backend, `-ngl 0`) on the same
model exported with `amql-cli to-gguf` (F16 weights — the same 16 bits per weight AMQL
streams from its BF16 container), using the `timings` it returns for a request with
`cache_prompt: false`, best of three, at 16 threads (its fastest; 32 and 64 were slower).

## Results

Qwen 3.5 0.8B (24 layers: 18 Gated DeltaNet + 6 gated softmax attention, 248k
vocabulary, tied head). AMD Threadripper 9970X (32 cores), 2 × 64 GiB DDR5-6400
(two channels: ~85 GB/s measured), Windows 11, .NET 11. Other software was running on
the machine; figures are best of three runs.

| | prefill pp32 | prefill pp512 | decode tg128 |
|---|---|---|---|
| AMQL before | 36 tok/s | 263 tok/s | 10.5 tok/s |
| **AMQL now** | **362 tok/s** | **975 tok/s** | **43.9 tok/s** (41.1 after a 512-token prompt) |
| llama.cpp, CPU, F16 | 390 tok/s | 1858 tok/s | 39.2 tok/s (39.0 after 512) |

Decode, which dominates a single request's time unless the prompt is long, is now
~12% ahead of llama.cpp. End to end, a request of 32 prompt tokens and 128 generated
takes 3.0 s against 3.35 s. At 512 prompt tokens and 128 generated AMQL is ~2% behind
(3.64 s against 3.56 s), because llama.cpp's prefill is still faster.

Without `--cpu` on a machine with the CUDA backend, `DecodeSession` runs on the
device engine instead (see [gpu-performance.md](gpu-performance.md)); the figures here
are the managed path's, as under `--cpu` or `AMQL_GPU_ENGINE=0`.

Decode is memory-bound: each token streams every weight once (~1.5 GB in BF16 for
this model, a third of it the 248k-row head). At ~85 GB/s that is a floor of ~18 ms a
token; AMQL runs at ~23 ms, ~75% of the machine's bandwidth.

On the RTX 5090, the device engine decodes this model at ~655 tok/s against
llama.cpp's ~500; see [gpu-performance.md](gpu-performance.md).

## What changed

**Weights stay 16-bit.** In the default `f32` working set a matrix stored as BF16 (and
not changed by a patch) is kept as its BF16 bits (`Tensor2D.FromBf16`). The GEMV kernel
widens each run of bits to f32 in registers — a zero-extend and a 16-bit shift, which
is exact — so it computes the same products as before while streaming half the bytes.
`Tensor2D.Data` still returns the f32 values: the tensor widens itself on first touch,
and from then on the f32 array is the tensor (so in-place edits such as
`WeightLoader.AddDelta` are seen by every later operation). `AMQL_RESIDENT_BF16=0`
restores widening at load.

**A GEMV kernel that hides latency.** `TensorOps.DotRows` / `DotRowsBf16` compute four
weight rows at once with four independent accumulators. Each row's arithmetic is exactly
`TensorOps.Dot`'s: one `Vector<float>` accumulator, multiply then add (no FMA
contraction), `Vector.Dot` fold, scalar tail — hence bit-identical.

**A persistent thread pool.** `KernelPool` keeps compute threads spinning briefly
between loops, so the few hundred parallel loops of one decode step start in
microseconds instead of paying `Parallel.For` dispatch each time. With it the floor
below which a projection runs single-threaded fell from 4M to 256K multiply-adds; the
old floor sat just above every FFN and linear-attention projection of a 1B-class model,
which therefore ran on one core. The claim counter and the completion counter live on
separate cache lines, and each thread reports its completions once.

**Fused projections.** `TensorOps.MatMulTransposedBMany` runs the weights that share an
input — q/k/v, an FFN's gate and up, the four Gated DeltaNet in-projections — as one
pool loop.

**A register-blocked GEMM for prefill.** Batches of rows use a 4 × 4 micro-kernel (four
activation rows × four weight rows, sixteen accumulators, each weight run widened once
per four rows), and the work is tiled by activation rows as well as weight rows. Each
accumulator still sums one (i, j) dot in increasing k with the same fold. The hot
kernels are marked `AggressiveOptimization`, so the first request does not run them as
unoptimised tier-0 code.

**Gated DeltaNet.** The recurrence is vectorised along the value dimension with the
reference's exact operation order (decay in double rounded to float, the S·k and S·q
reductions as float products accumulated in double) and runs in parallel over heads; a
long batch also splits each head by value-dim columns, each on a private copy of its
columns of the state. A single token's conv reads the projected row directly instead of
going through column-major 2-D arrays, and a batch's conv runs row-major across
channels in parallel.

**Attention and the KV cache.** `RowKvCache` keeps each layer's rows in one contiguous,
geometrically grown buffer; the runtime used to rebuild every layer's K and V matrices
with LINQ on every step, an O(context) copy per layer per token. The attention kernel
reads the buffers in place, runs (query row, head) pairs in parallel and accumulates
P·V along whole value rows.

**Less serial work.** RoPE computes each row's cos/sin once for all heads; RMSNorm,
per-head q/k norms, the attention output gate, the FFN activation and the gated
RMSNorm split rows across the pool when there are many; outputs that are overwritten in
full are allocated without zero-filling. `DecodeSession.Prefill` puts only the last
position through the final norm and head, as it only returns that row — before, a
512-token prompt computed 512 rows of 248k logits and kept one.

## What would go further

- **Prefill.** llama.cpp is still ~1.8× faster at 512 tokens. The remaining gap is the
  GEMM's compute rate (AMQL keeps multiply-then-add on 256-bit vectors to stay
  bit-identical; FMA and 512-bit vectors would change the rounding) and the Gated
  DeltaNet recurrence, which is sequential over positions — a chunked formulation would
  parallelise it but changes the summation order.
- **Decode.** At ~75% of memory bandwidth, the remaining headroom on BF16 weights is
  small. Fewer bytes per token — a quantised CPU kernel reading MXFP4 or 8-bit weights
  in registers — is the next large step, at the cost of exactness.
