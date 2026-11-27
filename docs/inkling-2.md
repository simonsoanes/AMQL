# Inkling 2 looped (Inkling-2-Mini)

`thinkingmachines/Inkling-2-Mini`, `model_type` `inkling2_loop`: 24 layers of
2304, 128 experts (top 8, width 1152), layers 6-17 run three times, no rotary.
This note covers what AMQL does with the real checkpoint, what was found about
its expert scales, what 1M context needs, and its GPU path.

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

## Runtime (host memory)

`Inkling2Model` keeps every expert stack packed and decodes only the rows of
the 8 experts a token routes to, row by row, exactly as transformers'
`Inkling2LoopDequantizeExperts` computes them in f32: `(E2M1 × E4M3 block
scale) × (amax / denominator)` for NVFP4, `E4M3 × 2^(s-127)` for MXFP8. Large
BF16 matrices (embedding, head, attention projections) stay BF16 and are
widened as read (exact); small tensors are widened to f32 once. Host memory is
about 19 GB, 15.7 GB of it the experts. A tensor stored F32 (the tiny arch-zoo
fixture) runs exactly as before. Rows of large matrices are spread over the
`ComputeBudget` cores.

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

- `Inkling2Tests.Quantised_Experts_Are_Kept_As_Stored_And_Match_Their_Decoded_Values`:
  the tiny arch-zoo model with its experts quantised in all three forms
  (plain, NVFP4 with amax/denominator, MXFP8, mixed inside the loop) gives
  bit-identical logits to the same model with those values decoded to F32 by
  transformers' formula.
- `Inkling2Tests.Quantised_Experts_Export_As_Stored`: export gives back every
  tensor with its dtype, shape and bytes, and re-imports to the same logits.
- `Inkling2Tests.Real_Checkpoint_Matches_Transformers_When_Available` runs
  when two variables are set: `AMQL_INKLING_CONTAINER` (a container encoded as
  above) and `AMQL_INKLING_REFERENCE` (the output of
  `scripts/reference/stream_reference.py`, which runs transformers a layer at a
  time in a few GB). It reads three files:
  - `summary.json`: per prompt, the ids, the top-k logits, log-sum-exp and
    entropy at every position, and a greedy continuation.
  - `visits.json`: per layer visit and position, the output's RMS and first
    coordinates.
  - `{name}.last.f32`: the last position's logits.

  It compares the managed runtime and, where it runs, the GPU path. The
  reference may run in bf16, so the test asserts decisions: the top token at
  90% of positions and at the last, the last top five, and the first four
  greedy tokens. It prints the differences.
- `Inkling2Tests.Real_Checkpoint_Completes_A_Plain_Prompt_When_Available`
  (`AMQL_INKLING_CONTAINER` alone) is a smoke test without a reference.

The first reference, `golden.py`, loaded the whole model in transformers (bf16
on CPU). It reached 97 GB private memory and took the machine to its commit
limit, so it was stopped before writing anything. The streaming reference
replaces it.

Measured on 2026-11-27 against the bf16 reference, which has two prompts: the
plain "The capital of France is" (5 tokens) and the chat-templated "What is 17
times 23? Answer briefly." (26 tokens). The managed runtime and the GPU path
give the same numbers to the printed precision:

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
`randn(16, 2304)`, normalised). They are read from `AMQL_INKLING_DIRECTIONS`.
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
negligible). The GPU path holds the sliding slots as rings of their window
plus a prompt chunk (1,280 rows, f32) and the full slots in the KV format
asked for; their sizes as measured are in the GPU section. Remaining work for
1M:

- The managed runtime keeps every key of every slot in f32 (including the
  sliding ones), about 384 KiB per token. It is a reference, not a server.
- With 17.6 GiB of weights on a 32 GB card, 1M fits only with NVFP4 rows
  (about 10 GiB). FP16 rows reach about 256K.
- A prompt's attention is per query over its keys (split-key, CUDA cores), so
  very long prompts are attention-bound. A tiled tensor-core attention is the
  next step there.

## GPU path

`native/amql_cuda/amql_inkling.cu`, built by `build-inkling.cmd` (with
`VCVARS_VER=14.44`) into its own `amql_inkling.dll`, and
`src/Amql.Inference/Inkling2Gpu.cs`. It is a library of its own so nothing in
`amql_cuda.dll`, which runs the other models, changes. The kernels mirror
`Inkling2Model` step for step.

- **Weights as stored.** The experts go up as the checkpoint holds them: NVFP4
  elements, E4M3 block scales and the per-expert factor `amax / denominator`
  (layers 0-19), MXFP8 elements and E8M0 exponents (20-23), or plain F32/BF16
  (the tiny fixture). The large matrices stay BF16 and the small tensors are
  widened to f32. The weights take 17.59 GiB of device memory, the
  container's size.
- **A step** reads its token and position from device memory, so the whole
  step (48 layer visits, the loop's three passes, the messages between them)
  is captured once as a CUDA graph and replayed for every token. Projections
  that share an input run as one launch. Only the 8 chosen experts' rows are
  read, each decoded in registers (E2M1 pairs and E4M3 by the conversion
  instructions). The router's top 8 is one warp.
- **A prompt** runs in chunks of up to 256 tokens through the same kernels
  with a token dimension. The projections are GEMMs that read each weight row
  once per 16 tokens. The chunk's (token, choice) pairs are grouped by expert,
  so each chosen expert is read once per chunk. The short convolutions and the
  cache rows advance token by token inside their kernels.
- **Per-pass slots.** Every layer visit has its own slot: 24 physical layers
  plus 2 × 12 for the core's extra passes, 48 slots. A sliding slot is a ring
  of its window plus one chunk (1,024 + 256 rows, f32), so the rows a chunk
  writes never overwrite keys its own earlier rows still read. A full slot
  holds every position in the KV format asked for: `f32`, `fp16` (the
  default) or `nvfp4` (E2M1 codes, an E4M3 scale per 16 values and an f32
  scale per head and row). The format is `GpuEngine.RequestedKvFormat`
  (`AMQL_KV_FORMAT`, `--kv-format`). Full slots grow by doubling, keeping
  their rows, and the step graph is captured again.
- **Attention** is split-key flash decoding with the model's extras: the sink
  logit in the softmax's denominator, the relative bias `r · P[:, dist]`
  within 1,024 of the newest key, the full layers' log scaling on q and r, the
  q and k L2 clips, the value RMS, and the sigmoid output gate.

**Served.** `ModelHost` sends an Inkling 2 container to
`DecodeSession.ForInkling2`: the GPU path when a device and the library are
there and `AMQL_GPU_ENGINE` is not 0, otherwise the managed runtime. It runs
one request at a time, with no continuous batching. `amql-cli generate` does
the same and prints both speeds:

```
amql-cli generate <container> --prompt "The capital of France is" --steps 64
```

Measured on the RTX 5090 (32 GB) on 2026-11-27 with nothing else running:

- **Matches the managed runtime.** The tiny model agrees to 1e-3 of the
  largest logit with f32 rows. Each prompt is 3, 21 or 200 tokens; the sliding
  rings wrap and the full slots grow. FP16 rows agree to 2e-2, and NVFP4 rows
  give the same top token at at least 80% of positions. A 600-token prompt in
  three chunks matches the same tokens fed one at a time, and a replayed step
  graph matches its direct launches bit for bit. The tiny model with NVFP4,
  MXFP8 and plain experts mixed also agrees to 1e-3.
- **On the real checkpoint** (f32 rows) there are four prompts of 5, 7, 20 and
  582 tokens; the last spans three chunks. The top five at all 614 positions
  agree in set and order, the worst logit difference is 2.4e-4, and the greedy
  continuations are token for token the same.
- **Speed**, with a prompt fed in chunks and then 128 greedy steps after it:

  | KV rows | prompt tokens | prompt tok/s | decode tok/s after it |
  |---|---|---|---|
  | fp16 | 512 | 629 | 80-97 |
  | fp16 | 4,096 | 580 | 97 |
  | fp16 | 16,384 | 536 | 93 |
  | fp16 | 65,536 | 397 | 81 |
  | nvfp4 | 4,096 | 522 | 91 |
  | nvfp4 | 65,536 | 208 | 55 |
  | f32 | 4,096 | 575 | 96 |

  `amql-cli generate` on a 5-token prompt decodes at 110 tok/s. Decode reads
  about 5.4 GB a token: 2.6 GB of attention projections over 48 visits,
  1.85 GB of chosen experts and 0.9 GB of head. That puts the bandwidth bound
  near 330 tok/s, so decode runs at about a third of it. The rest is the many
  small kernels a visit runs, about 30 each.
- **Memory.** The weights take 17.59 GiB. A sequence's KV rows and
  convolution histories (40 sliding rings of 1,280 f32 rows, plus the 8 full
  slots) take:

  | positions | f32 | fp16 | nvfp4 |
  |---|---|---|---|
  | 4,096 | 0.64 GiB | 0.52 GiB | 0.43 GiB |
  | 32,768 | 2.39 GiB | 1.39 GiB | 0.69 GiB |
  | 131,072 | 8.39 GiB | 4.39 GiB | 1.58 GiB |
  | 262,144 | 16.39 GiB | 8.39 GiB | 2.77 GiB |
  | 1,048,576 | 64.39 GiB | 32.39 GiB | 9.89 GiB |

  The step's and a chunk's scratch add about 0.3 GiB. At a 65,536-token
  prompt the device held 22.2 GiB in all with fp16 rows, and 20.8 GiB with
  nvfp4 rows. So 1M fits on the 32 GB card only with nvfp4 rows: 17.6 + 9.9
  GiB plus scratch.

Prompts are the slow part. The grouped experts and the projections run on CUDA
cores in f32, about 65% and 25% of a chunk, so the prompt speed is a fraction
of what tensor cores would give.

Not done yet:

- Batching several requests in one step.
- Prompts on the tensor cores: the projections and experts as BF16/FP4
  matrix products, and a tiled attention. They are CUDA-core f32 kernels now,
  which keeps them within 1e-3 of the managed path.
- Decode: fewer, fused kernels per visit (norms into the mixes, the gates'
  three steps into one).
- W4A4 prefill, which `hf_quant_config.json` describes (`required_activation_quantization`).
- KV paging and marks.
- The vision tower and the drafter.
