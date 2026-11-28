# Inkling 2 looped (Inkling-2-Mini)

`thinkingmachines/Inkling-2-Mini`, `model_type` `inkling2_loop`: 24 layers of
2304, 128 experts (top 8, width 1152), layers 6-17 run three times, no rotary.
A model note: how the checkpoint is imported, what was found about its expert
scales, and what 1M context needs. Nothing in the runtimes is this model's: its
config adapter (`src/Amql.Hf/Inkling2.cs`) describes the stack with the generic
surfaces of [depth-ops.md](depth-ops.md), and both runtimes run that plan.

## Import: experts kept as stored

```
amql-cli encode D:\Models\Inkling-2-Mini --out <container> --no-vision
```

- Text model only. The vision tower (`model.visual.*`, 0.64 GB) is refused
  unless `--no-vision` leaves it out. The drafter (`mtp.safetensors`, its own
  file with unprefixed names such as `fuse_weight`) is not listed in
  `model.safetensors.index.json`, so it is never treated as the model's and is
  never encoded.
- Tensors keep their checkpoint names under `model.llm.` in four objects
  (`target.embedding`, `target.decoder_stack`, `target.final_norm`,
  `target.output_head`), so export writes the original names back.
- The experts are stored as they ship (nothing is dequantised to 16-bit):

  | layers | form  | tensors per stack (`w13_weight`, `w2_weight`) |
  |--------|-------|-----------------------------------------------|
  | 0-19   | NVFP4 | `U8 [128, rows, cols/2]` (two E2M1 a byte, even element low), `_scale F8_E4M3 [128, rows, cols/16]`, `_amax F32 [128]`, `_outer_scale_denominator F32 [1]` |
  | 20-23  | MXFP8 | `F8_E4M3 [128, rows, cols]`, `_scale U8 [128, rows, cols/32]` (E8M0 exponents), no amax |

  The NVFP4 element tensor is recorded as `FP4` at its logical shape
  (`[128, 2304, 2304]` for `w13`), which `ArchMapper.PackedFp4` now does for a
  3-D stack as well as a 2-D weight; the companions stay as their own tensors,
  in their own dtypes. `w13` interleaves gate and up rows (gate j = row 2j).
  `hf_quant_config.json` names MXFP8 `weight_block_shape [32, 32]`, but the
  stored scales are one per 32 along a row (`[1, 32]`); llama.cpp's converter
  makes the same observation and uses the stored shape, as does AMQL.
- The encoded container is 17.6 GiB (the checkpoint less the vision tower).
- `hf_quant_config.json` travels with the container and comes back out on
  export, with the source `config.json` verbatim, since the expert bytes come
  back unchanged (export writes FP4 back as `U8` at half the last dimension,
  for any rank).

## Described generically

The config adapter describes the stack by structure: a depth loop (layers 6-17
three times, each pass with its own cache slots), a block residual (a bank of
block inputs every 6 layers, depth mixes at each sub-block and at the final
norm), cross-pass message gates, residual short convolutions, and the
attention's conditioning (separate q/k/v inputs on the full layers, query scale
and clip, log-scaling on the full layers, a relative bias, projected sinks, a
separate output gate), with packed experts routed by sigmoid scores and a
softcapped GLU, a weighted embedding norm and a padded vocabulary. See
[depth-ops.md](depth-ops.md) for each surface, its plan op and how the CPU's
row-exact path and the device engine's visit path run it. `amql-server`,
`generate` and `DecodeSession.Open` open it like any planned model: the device
engine when there is one (one request at a time), else the CPU. A container
encoded before these surfaces needs `amql-cli graph upgrade <container>`
(graph only; tensors untouched).

## Per-pass expert scales: there are none

The question was whether the loop's passes use different NVFP4 expert scales
from transformers' single per-expert factor (llama.cpp PR #32704 was cited).

- The checkpoint: every expert stack has one `_amax [128]` and one
  `_outer_scale_denominator [1]`, and no tensor anywhere has a pass index
  (the only names with "pass" are `middle_cycle_cross_pass_queries` /
  `_kv_queries`, the message queries). `hf_quant_config.json` has one NVFP4
  group (layers 0-19, `weight_scale_format: amax_denominator`) and one MXFP8
  group (20-23); nothing per pass. The loop runs the same physical layers.
- llama.cpp PR #32704 (4 files, +7 lines) does not add per-pass scales. It adds
  a hook, `post_arch_tensors`, that runs after `load_tensors`' generic pass
  which creates the optional `.scale` tensors (`ffn_*_exps_s`, the per-expert
  scale that an NVFP4 GGUF carries, and the attention/FFN `.scale`s), and moves
  `inkling2-loop.cpp`'s aliasing of the extra cache slots,
  `layers[il] = layers[layer_phys(il)]` for slots 24-47, into it. Before the
  fix the alias copied each layer struct before its scale pointers were set, and
  the generic pass then looked for `blk.24..47.*.scale`, which do not exist:
  passes 1 and 2 ran their experts without the per-expert scale. After it,
  every pass shares pass 0's weights and scales.
- So the checkpoint needs exactly what transformers does: one factor per
  expert, the same in every pass. AMQL's runtime has a single physical layer
  object per layer that every pass reads (only the KV and convolution slots
  are per pass), so it cannot have llama.cpp's bug; nothing pass-specific was
  added. A GGUF made by a llama.cpp without #32704 will be wrong in passes 1-2
  and should not be used as a reference.

## Checking against transformers

- `LoopedCheckpointTests.PackedExperts_Are_Kept_As_Stored_And_Match_Their_Decoded_Values`:
  the tiny arch-zoo model with its experts quantised in all three forms
  (plain, NVFP4 with amax/denominator, MXFP8, mixed inside the loop) gives
  bit-identical logits to the same model with those values decoded to F32 by
  transformers' formula.
- `LoopedCheckpointTests.PackedExperts_Export_As_Stored`: export gives back every
  tensor with its dtype, shape and bytes, and re-imports to the same logits.
- `LoopedCheckpointTests.LoopedDepth_Real_Checkpoint_Matches_Transformers_When_Available` runs
  when two variables are set: `AMQL_TEST_CONTAINER` (a container encoded as
  above) and `AMQL_TEST_REFERENCE` (the output of
  `scripts/reference/stream_reference.py`, which runs transformers a layer at a
  time in a few GB). It reads three files:
  - `summary.json`: per prompt, the ids, the top-k logits, log-sum-exp and
    entropy at every position, and a greedy continuation.
  - `visits.json`: per layer visit and position, the output's RMS and first
    coordinates.
  - `{name}.last.f32`: the last position's logits.

  It compares the planned CPU path and, where it runs, the device engine. The
  reference may run in bf16, so the test asserts decisions: the top token at
  90% of positions and at the last, the last top five, and the first four
  greedy tokens. It prints the differences.
- `LoopedCheckpointTests.LoopedDepth_Real_Checkpoint_Completes_A_Plain_Prompt_When_Available`
  (`AMQL_TEST_CONTAINER` alone) is a smoke test without a reference.

The first reference, `golden.py`, loaded the whole model in transformers (bf16
on CPU). It reached 97 GB private memory and took the machine to its commit
limit, so it was stopped before writing anything. The streaming reference
replaces it.

Measured on 2026-11-27 against the bf16 reference, which has two prompts: the
plain "The capital of France is" (5 tokens) and the chat-templated "What is 17
times 23? Answer briefly." (26 tokens). The CPU runtime and the GPU path (then
the dedicated ones this build replaced, which the planned paths match) gave the
same numbers to the printed precision:

| | plain0 | chat0 |
|---|---|---|
| top token agrees | 5/5 | 25/26 (position 5: a near tie) |
| top-5 overlap | 5/5 everywhere | 5/5 at 22 positions, 4/5 at 3 |
| max \|diff\| on the reference's top 20 | 0.04-0.13 | 0.05-1.1 |
| last position, all 200,058 logits | max \|diff\| 0.34, cosine 0.99992 | max \|diff\| 0.35, cosine 0.99999 |
| greedy continuation | 8/8 tokens | 8/8 tokens |
| layer outputs (48 visits) | RMS ratio 0.987-1.009, fingerprint cosine ≥ 0.992 | RMS ratio 0.960-1.035, fingerprint cosine ≥ 0.951 |

The layer outputs are compared on the reference's 16 fixed directions, made
once with torch's generator (`torch.Generator().manual_seed(20261127 + 2304)`,
`randn(16, 2304)`, normalised). They are read from `AMQL_TEST_DIRECTIONS`.
The differences are the size bf16 arithmetic gives, and none grows across the
loop's passes.

## Context: 1M needs no positional work

There is no rotary, so there is nothing to extend: no NTK, no YaRN, and
AMQL's context-extension machinery does not apply.

- The relative bias acts only on the last `rel_extent` = 1024 keys (and the
  sliding layers attend over a 1024 window anyway); it has no notion of
  absolute position.
- The full layers' log scaling, `tau = 1 + 0.1 ln(max(1, (t+1)/8192))`,
  multiplies q and the relative-bias projection; it is defined at every
  position and reaches 1.485 at 1M. `model_max_length` is 1,048,576, so 1M is
  inside the trained range.
- Sinks add one logit per head to the softmax denominator; no position.

What 1M does cost is memory. Each layer visit has its own KV slot: 24 physical
layers plus 2 × 12 for the core's extra passes, 48 slots. Of them 8 are full
attention (layers 5 and 23 once, 11 and 17 three times each) and 40 sliding.
A slot holds 8 KV heads × 128 × (K + V) = 2048 values per token.

| | per token | at 1M tokens |
|---|---|---|
| 8 full slots, FP16 | 32 KiB | 32 GiB |
| 8 full slots, FP8 | 16 KiB | 16 GiB |
| 8 full slots, NVFP4 (+ scales) | ~9 KiB | ~9 GiB |
| 40 sliding slots, window 1024, FP16 | fixed | 160 MiB |

Plus the convolution histories (K-1 = 3 states per convolution per slot,
negligible). The device engine holds the sliding slots as rings of their window
plus a prompt chunk (1,280 rows, f32) and the full slots in the KV format
asked for (sizes below). Remaining work for 1M:

- The CPU's row-exact path keeps every key of every slot in f32 (including the
  sliding ones), about 384 KiB per token. It is a reference, not a server.
- With 17.6 GiB of weights on a 32 GB card, 1M fits only with NVFP4 rows
  (about 10 GiB). FP16 rows reach about 256K.
- The CPU path's prompt attention is per query over its keys. The device
  engine's runs on tensor cores in tiles, so prompt speed stays nearly flat to
  64K tokens there.

## On the GPU

The device engine's visit path ([depth-ops.md](depth-ops.md#runtime-the-device-engines-visit-path))
runs it with the experts as stored. Measured on the RTX 5090 (32 GB) on
2026-11-28 with nothing else running:

- **Against the planned CPU path** (f32 rows and f32 prompts, four prompts of
  5, 7, 20 and 582 tokens, the last over two chunks): the top five at all 614
  positions agree in set and order, the worst logit difference is 2.6e-4, and
  the greedy continuations are token for token the same. With BF16 prompts (the
  default) the top token agrees at 614 of 614; the largest logit differences
  (up to 10) are in the improbable tail, where the softcap at 30 compresses
  values and BF16 then shifts them.
- **Speed**, a prompt fed in chunks and then 128 greedy steps after it
  (`LoopedDepth_Real_Checkpoint_Gpu_Speed`); in brackets the same before the
  step's fusions, the lane-group experts and the prompt's chunks taken
  token-parallel (2026-11-28, measured the same way). Prompts past 512 tokens
  borrow a chunk of up to 1,024 rows while memory allows (see
  [depth-ops.md](depth-ops.md)); every prompt below did:

  | KV rows | prompt tokens | prompt tok/s | decode tok/s after it |
  |---|---|---|---|
  | fp16 | 5 | — | 172 (143) |
  | fp16 | 512 | 3,894 (2,007) | 172 (136) |
  | fp16 | 4,096 | 4,400 (2,019) | 150 (128) |
  | fp16 | 16,384 | 4,549 (1,999) | 141 (121) |
  | fp16 | 65,536 | 3,993 (1,855) | 118 (101) |
  | nvfp4 | 4,096 | 4,496 (1,956) | 148 (122) |
  | nvfp4 | 65,536 | 3,940 (1,791) | 113 (63) |

  The 512-token prompt is the nvfp4 run's (the first prompt of a run is slower,
  under 1,000 tok/s, while cuBLAS picks its algorithms). In the engine's own
  chunks of 512 (`AMQL_DEPTH_CHUNK=512`) the 64K prompt is about 3,300 tok/s.
  `amql-cli generate <container> --prompt "The capital of France is" --steps 128`
  prints 160–176 tok/s: over 10 runs a median of 172, standard deviation 4.0
  (2.3%), over 5 runs of 512 steps 170–175 (GPU clocks steady at 2.84–2.87 GHz;
  the short run is 0.73 s, so a stall shows).
- **Device memory on Windows.** Once the device's use passes about 23.4 GB,
  the driver's reported use rises by about 6 GB at once (a 64K fp16 sequence, a
  borrowed chunk, or an idle sequence reserved as padding all set it off); the
  CUDA allocations do not (Nsight Systems' memory trace), the process's later
  allocations fit in it, and it goes when the sequence is released. Other
  processes, a game say, see the higher figure.
- **Decode** reads about 5.4 GB a token (2.6 GB of attention projections over
  48 visits, 1.85 GB of chosen experts, 0.9 GB of head), so the bandwidth bound
  is near 330 tok/s. A step is about 5.8 ms; on the GPU (Nsight Systems, graph
  nodes, per step): the attention projections 1.08 ms (one launch a visit, 40 MB
  in 22.7 µs, 1.76 TB/s, their input norms computed from the same loads), the
  experts 1.46 ms (NVFP4 gate/up 19.3 µs and down 10.0 µs a visit on lane-group
  kernels, the routing inside the first), the head 0.54 ms, the output
  projection with its conv 0.51 ms, and about 14 launches a visit for the rest
  (attention 6.5 µs with its merge, mixes 2.9 µs, the queries' and keys' prep
  4.8 µs, the FFN conv with the experts' sum 6.5 µs, gates and router 2-4 µs).
  While those small kernels run, the next matrices come into L2 on a side branch
  of the step's graph: 171 against 168 tok/s without it (`AMQL_GPU_PREFETCH=0`).
  Programmatic dependent launch gained nothing inside the graph. At long
  contexts the attention reads the cache: from 1,024 keys a block takes a key
  head for its three query heads, so each row is read once (at 64K tokens a
  full slot's attention took 418 µs a visit with a block a query head).
- **Memory.** The weights take 17.6 GiB, the chunk scratch and a first
  sequence about 1.5 GiB more. A sequence's KV rows and convolution histories
  (40 sliding rings of 1,536 f32 rows, plus the 8 full slots):

  | positions | f32 | fp16 | nvfp4 |
  |---|---|---|---|
  | 4,096 | 0.72 GiB | 0.60 GiB | 0.51 GiB |
  | 32,768 | 2.47 GiB | 1.47 GiB | 0.77 GiB |
  | 131,072 | 8.47 GiB | 4.47 GiB | 1.66 GiB |
  | 262,144 | 16.47 GiB | 8.47 GiB | 2.85 GiB |
  | 1,048,576 | 64.47 GiB | 32.47 GiB | 9.97 GiB |

  So 1M fits on the 32 GB card only with nvfp4 rows.

Not done yet: batching several requests in one step, marks and paging on the
visit path, W4A4 prefill (`hf_quant_config.json`'s
`required_activation_quantization`), the vision tower and the drafter.
