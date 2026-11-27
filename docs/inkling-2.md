# Inkling 2 looped (Inkling-2-Mini)

`thinkingmachines/Inkling-2-Mini`, `model_type` `inkling2_loop`: 24 layers of
2304, 128 experts (top 8, width 1152), layers 6-17 run three times, no rotary.
This note covers what AMQL does with the real checkpoint, what was found about
its expert scales, what 1M context needs, and a design for the GPU path.

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
- `Inkling2Tests.Real_Checkpoint_Matches_Transformers_When_Available`: runs
  when `AMQL_INKLING_CONTAINER` (a container encoded as above) and
  `AMQL_INKLING_GOLDEN` (the output of the setup-only `golden.py`: per prompt
  `{name}.ids.npy`, `{name}.logits.npy`, `{name}.hidden.npy`, and
  `summary.json` with the greedy continuation) are set; `AMQL_INKLING_GREEDY`
  sets the greedy steps compared (default 8). transformers runs in bf16 and
  this runtime in f32, so it judges decisions: last-position argmax, top-5
  overlap, argmax at 90% of positions, and the first 4 greedy tokens. It prints
  the worst logit difference and the per-visit hidden-state cosine.

- `Inkling2Tests.Real_Checkpoint_Completes_A_Plain_Prompt_When_Available`
  (`AMQL_INKLING_CONTAINER` alone) is a smoke test without a reference.

Measured on the real checkpoint (2026-11-27, managed path, CPU at low
priority): the container loads in 21 s (experts NVFP4 on layers 0-19, MXFP8 on
20-23, as stored), a 5-token prefill takes 4.1 s, decode about 0.8 s a token,
and "The capital of France is" continues greedily as " Paris." The
transformers reference was not produced: `golden.py` in bf16 on CPU reached
97 GB private memory after loading (its dequantising loader keeps far more
than the 53 GB of bf16 weights) and took the machine to its commit limit, so
it was stopped before writing any output. A leaner reference (layers loaded
and run a few at a time, or experts left quantised) is needed before the
logit comparison can run.

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
negligible). So a 24 GB card holding the ~17.6 GB of weights cannot also hold
1M of FP16 KV; with NVFP4 KV and the existing disk paging of idle states, the
full slots at 1M are about 9 GiB, which needs either weights partly on the host
or a larger card. Work needed:

- The managed runtime keeps every key of every slot in f32 (including the
  sliding ones), about 384 KiB per token; it is a reference, not a server.
  Trimming the 40 sliding slots to a 1024 ring would make it O(window) there.
- The server's `--max-context` must allow 1,048,576 for this model (its
  `model_max_length`); no RoPE check applies.
- Prefill attention at 1M is O(T²) on the 8 full slots (plus the 1024-wide
  bias band), so it needs the split-key attention path, not a dense one.

## GPU path: design

Fits from `GpuEngine` / `native/amql_cuda` today:

- NVFP4 and MXFP8 weights as stored: `nvfp4_gemv`, `nvfp4_grouped_gemv`,
  `mxfp8_gemv`, and the FP4 tensor-core prefill GEMM. The per-expert factor
  `amax / denominator` is a per-expert scalar, the role `rowScale` /
  `expertScale` already plays in `amql_moe.cu` (`moe_glu_kernel`,
  `moe_scatter_kernel`). The 3-D stacks would be uploaded as 128 row-blocks of
  `[rows, cols]` each, as the split-expert path holds them, with no copy.
- MoE grouping, gather and scatter (`moe_group_kernel`, `moe_gemv_kernel`,
  `moe_mix_kernel`, `moe_finish_kernel`). Routing is new in part:
  `moe_route_kernel` is softmax top-k; Inkling needs sigmoid scores, a
  selection-only bias (`mlp.gate.bias` picks the top 8 but is not in the
  weights), renormalisation over the chosen 8 and `route_scale` 8.
- Sliding-window attention, split-key attention and NVFP4/FP16 KV, KV paging,
  marks/rewind.
- The depthwise causal short convolution with history (`gdn_conv_*` kernels)
  for the static convolutions (`attn_sconv`, `mlp_sconv`, sliding `k/v_sconv`),
  with the `(conv + x)/sqrt(K+1)` residual form added.
- Final logit softcap (Gemma 2 path).

New:

- Cache slots decoupled from physical layers: 48 slots over 24 layers, as
  llama.cpp's `n_layer_all`; KV row sizes, windows and marks per slot.
- Attention: sinks (an extra logit per head, no value), the query-side log
  scaling (a per-position scalar on q and r; cheap and flash-compatible), q and
  k L2 clips with the per-channel q scale, value RMS without gain, the gated
  output (`sigmoid(gate_attn · x)`), and the relative bias: per query and
  head, `r_h · P[:, dist]` for `dist < 1024`, a 1024-wide additive band on the
  most recent keys. The band is computed per query (16 × 1024 per head) and
  added inside the split-key kernel for the keys in range.
- Dynamic short convolution on the full layers' k and v: taps are
  `static[lag, c] + dyn[head, lag]`, `dyn` projected from the layer input;
  scale `1/sqrt(2K+1)`.
- The attention residual: per layer, softmax over depth of the blocks so far
  and the current state (each RMS-normed without gain), against 1 or 3 learned
  queries; a small batched kernel over at most 5 states. Blocks restart every
  6 layers.
- The loop: passes 1-2 restore the prelude state and blocks, read a "bank"
  (the previous pass's blocks and output) through the message queries, and
  gate each of q/k/v/mlp inputs with a rank-64 message gate
  (`(1 - tanh(F g)) h + tanh(G g) m`). Graph capture per pass, three copies of
  the core's 12 layers with different slots.
- Softcapped SwiGLU (`tanh(g/4)·4 · sigmoid(g) · tanh(u/25)·25`) with gate/up
  interleaved by row in `w13`; the FP4 grouped GEMM can split even/odd rows
  into the two outputs or the GLU kernel can read them interleaved.
- Optional: W4A4 prefill, which `hf_quant_config.json` asks for
  (`required_activation_quantization`: per-token NVFP4 activations for layers
  0-19, MXFP8 for 20-23). transformers ignores it (bf16 activations), so the
  first GPU target should match transformers and treat W4A4 as a later choice.

Memory on a 24 GB card: weights 17.6 GiB as stored, plus activations, the 40
sliding slots and the full slots' KV. 32K of FP16 KV on the 8 full slots is
1 GiB; 256K is 8 GiB (NVFP4: ~2.3 GiB). So 256K fits with NVFP4 KV; 1M needs
host-resident experts or a bigger card.
