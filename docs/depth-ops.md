# Depth and conditioning ops

Generic operations for stacks that loop their layers, mix their residual over
depth, and condition their attention without a rotary embedding. Each is a
structural fact on the execution surface (`system_graph.json`), planned by
`Planner` into `ComponentOpPlan` / `LayerPlan` ops, and run by
`GenericRuntime`. Nothing here names a model family: a checkpoint's spellings
live in its layer role map (`execution.roles`), written by its config adapter
on import, and both runtimes, the CPU's and the device engine's, run the plan. Inkling 2 looped is the first stack described this way
([inkling-2.md](inkling-2.md)).

## Surfaces and plan ops

| Surface (execution) | Plan op | What it computes |
|---|---|---|
| `depth_loop {start, end, passes}` | `ComponentOpPlan.DepthLoop`, `.Visits` | Layers `[start, end)` run `passes` times. The plan's visit schedule lists each run of a physical layer with its own cache slot (pass 0 in the layer's slot, later passes past the last layer); weights are shared. A later pass restarts from the state (and bank) saved before `start`. |
| `block_residual {block_size, eps, query_object, mixer_queries[], ffn_queries[], output_query}` | `BlockResidualOp`, `LayerPlan.MixerInputMix`, `.FfnInputMix`, `.OpensBlock` | A bank of block inputs: a layer whose index is a multiple of `block_size` adds its input to the bank and restarts the residual from the mixer's output. A sub-block's input is a depth mix `Σ softmax(q·rms(s)/√d) rms(s)` over bank + current state; three mixer queries give the attention separate q, k, v inputs; the final norm reads the `output_query` mix. |
| `message_gates {rank, eps, skip_blocks, sites[]}` | `LayerPlan.MessageGates` (`MessageGateOp`) | On a looped layer in a later pass, each input site (q, k, v, ffn) is gated on a message from the previous pass: `(1 − tanh(F·g))·h + tanh(G·g)·m`, `g = D·[rms h; rms m]`, `m` the site's depth mix over the previous pass's bank (past its first `skip_blocks`) and output. |
| `residual_conv {kernel}` | `ResidualConvOp` on `LayerPlan.MixerOutputConv`, `.FfnOutputConv`, `AttentionOp.KeyConv`, `.ValueConv` | Short causal depthwise convolutions with a residual, `(Σ taps·x + x)/√(n + 1)`: static taps `[C, 1, K]` (n = K), or on keys/values dynamic taps `static[lag, c] + (P·input)[head, lag]` (n = 2K). Histories are per cache slot. |
| `attention.qk_clip`, role `q_channel_scale` | `AttentionOp.QkClip`, `.QueryChannelScale` | Per-channel query scale, then each query and key head L2-clipped. |
| `attention.log_scaling {floor, alpha, layers}` | `AttentionOp.LogScaling` | Queries and relative states × `1 + α·ln(max(1, (t+1)/floor))` on the listed layers. |
| `attention.relative_bias {rank, clip}`, roles `rel_proj`, `rel_channel_scale`, `rel_table` | `RelativeBiasOp` | A learned score bias over backward distance δ < extent (the table's width): `r·T[:, δ]`, `r` a clipped, scaled projection of the query input. Stands where rotary would (position `none`). |
| `attention.sink_logits` (`static` / `projected`), roles `sink` / `sink_proj` | `AttentionSinkOp` | A per-head logit in every softmax that takes mass without a value. (The legacy `attention.sinks` JSON stays refused.) |
| `attention.separate_output_gate`, role `out_gate_proj` | `AttentionOp.OutputGateProjection` | Context × `sigmoid(G·q_input)` before the output projection (the existing `output_gate` is the gate folded into `q_proj`). |
| roles `q_in_norm`, `k_in_norm`, `v_in_norm` | `AttentionOp.SeparateInputs`, `LayerPlan.Q/K/VInputNorm` | q, k and v are projected from separately normalised inputs. |
| `attention.parameter_free_qk_norm.v` | (existing) | Values RMS-normalised per head without a gain — after the value conv. |
| `ffn.moe.routing_policy: sigmoid_normalised_over_selected`, `selection_bias`, `route_scale` | `RoutedFfnOp.RoutingPolicy`, `.SelectionBias`, `.RouteScale` | Sigmoid scores; the router bias only chooses the experts; chosen scores normalised (sum + 1e-20) × route scale. |
| `ffn.gate_policy {kind: softcapped_glu, gate_cap, up_cap}` | `RoutedFfnOp.GatePolicy` | `cap(g)·sigmoid(g)·cap(u)`, `cap(x, c) = c·tanh(x/c)`. |
| `ffn.moe.packed_experts`, `gate_up_layout: interleaved_rows`, roles `packed_experts_gate_up` / `_down` (parts `weight`, `scale`, `amax`, `denominator`) | `PackedExpertsOp` | One tensor per projection for all experts, each layer in its stored format (plain, NVFP4, MXFP8); only the chosen experts' rows are decoded. |
| `head.embedding_norm.weight`, `head.embedding_weight`, `head.unpadded_vocab_size`, `norm.final_norm_weight` | `EmbeddingOp.NormWeight`, `OutputOp.TableRows` | A weighted embedding RMS norm; tensor names other than `weight`; a head with more rows than the vocabulary (the padding is never scored). |

All fields are optional: graphs without them read and plan as before, and the
schema stays 7.

## Runtime: the row-exact path

A plan with a depth loop, a block residual or one of these layer ops
(`ComponentOpPlan.RunsByVisit`) runs on `GenericRuntime`'s row-exact path
(`GenericRuntime.Visits.cs`, with `DepthMix`, `RowAttention`, `ResidualConv`
and `PackedExperts` beside it): one token at a time through the visit
schedule, every product accumulated in double row by row, large BF16 matrices
kept as their bits and expert stacks as stored (`RowOperands`). Prefill is
position by position; `RunLayerInternal` refuses such a plan, and the final
norm reads one row (the depth bank is the last token's). `RowPathRefusal`
names anything else in the plan the path does not implement (rotary, post
norms, dense FFNs, …).

On the tiny looped fixture the planned path's logits equal those of the
dedicated runtime it replaced bit for bit at every step of a 12-token prompt
and 8 greedy tokens, and so do all 400 visit outputs; on Inkling-2-Mini every
logit of a 58-token prompt and 8 greedy tokens did too. Its logits are held to
transformers' goldens in `ArchZooTests`, and the session's prefill and steps
to the runtime's own in `DepthOpsTests`.

## Runtime: the device engine's visit path

`GpuEngine` runs the same plan (`GpuEngine.Visits.cs`, over the primitives of
`native/amql_cuda/amql_depth.cu`); `GpuEngine.Unsupported` takes such a plan
when `RowPathRefusal` does and the geometry fits the kernels (head dim up to
128, a relative bias of rank up to 32, up to 512 experts and top 32, at most
nine blocks in the bank). The C# side walks `plan.Schedule` the way
`GenericRuntime.StepVisits` does and calls one primitive per op:

| Plan op | Primitive (amqe_*) | What it fuses |
|---|---|---|
| `EmbeddingOp` (+ `NormWeight`) | `embed_rows` | the norm in the gather |
| `DepthMixOp`, `BlockResidualOp.OutputQuery` | `depth_mix` | each thread holds a float4 of every state in registers; the softmax in one warp; the norm that follows (q/k/v input norms, the FFN norm, the final norm) in the same kernel |
| `MessageGateOp` | `message_gates` | the q, k and v gates' three steps (cat, down projection, apply) one launch each; every gate's message from one mix |
| `LayerPlan.Q/K/VInputNorm`, `PreFfnNorm` | `rms_rows` | up to three norms a launch |
| projections sharing an input | `projections` | up to four in one launch: GEMVs for a step, the engine's GEMMs (`amqe_linear`) for a chunk |
| `QueryChannelScale`, `QkClip`, `LogScaling`, `RelativeBiasOp` coefficients | `head_prep` | queries and coefficients in one launch |
| `AttentionOp.KeyConv/ValueConv` (static or dynamic), key clip, `ParameterFreeQkNorm.V` | `kv_prep` | the conv, clip, norm and the row stored (f32, FP16 or NVFP4) in one kernel |
| attention with `RelativeBiasOp` | `depth_attention` | a step: split-key flash decoding; a chunk: FlashAttention-2 on the tensor cores (FP16 tiles, f32 sums), the bias as a table first |
| `AttentionSinkOp`, `OutputGateProjection` | `depth_attention_merge` | the sink and the sigmoid gate where the key splits merge |
| `MixerOutputConv`, `FfnOutputConv`, the residual join | `residual_conv` | the conv and the residual add |
| `SigmoidNormalisedOverSelected` + `SelectionBias` + `RouteScale` | `route_sigmoid_selected` | the top k in one warp |
| `PackedExpertsOp` + `ExpertGateSoftcappedGlu` | `packed_experts` | a step: the engine's decode kernels (`amqe_moe_experts_paired_step`), only the chosen rows read; a chunk: grouped by expert, each decoded to BF16 once, on its tensor cores (`amqe_moe_experts_paired`); f32 prompts: f32 kernels |
| `OutputOp.TableRows`, softcap | `projections`, `softcap` | only the vocabulary's rows of a padded head are read |

- **Weights once.** Each physical layer's weights go up once and every visit
  of it reads them: large matrices as stored (BF16) or f32, small tensors f32,
  packed experts as stored (NVFP4 with the per-expert factor as a row scale,
  MXFP8, plain).
- **Slots.** A sequence holds every cache slot of the schedule: a windowed slot
  is a ring of its window and a chunk more (f32 rows), so a chunk's rows never
  overwrite keys its own earlier rows read; a full slot holds every position in
  the requested KV format (`f32`, `fp16`, `nvfp4`; `GpuEngine.RequestedKvFormat`,
  `AMQL_KV_FORMAT`). Full slots grow by doubling, keeping their rows. Each slot
  has its convolutions' histories.
- **A step** reads its token and position from device memory, so its launches
  are captured once per sequence as a CUDA graph (`amqe_pass_*`) and replayed.
  **A prompt** runs in chunks of 256 tokens; convolutions and cache rows advance
  token by token inside their kernels. `ExactPrefill` (`AMQL_GPU_PREFILL=f32`)
  keeps a chunk at f32 (SGEMMs, the f32 expert kernels, the per-query
  attention); `AMQL_TILED_ATTENTION=0` keeps the per-query attention only.
- **One sequence at a time.** `GpuEngine.BatchRefusal` says so: no batched
  steps, marks, rewinds or drafts on this path, and the server runs such a
  model one request at a time.

Held to the planned CPU path (`DepthGpuTests`): on the tiny fixture, plain and
with NVFP4, MXFP8 and plain experts mixed in the loop, every position of 3-,
21- and 200-token prompts and greedy steps, to 1e-3 of the largest logit with
f32 rows and f32 prompts and 2e-2 with FP16 rows (NVFP4 rows by the decision);
a replayed step graph equals its direct launches bit for bit; a 600-token
prompt in three chunks matches the same tokens stepped. On the tiny fixture
the visit path also gave exactly the logits of the dedicated GPU path it
replaced, at every position of 3-, 40- and 600-token prompts and a step after
them, in all three KV formats, with BF16 and f32 prompts.

On Inkling-2-Mini (RTX 5090, 2026-11-28; `LoopedCheckpointTests`):

- f32 rows and f32 prompts against the planned CPU path: the top five at all
  614 positions of four prompts (5, 7, 20 and 582 tokens) in set and order,
  worst logit difference 4.0e-4, the greedy continuations the same; BF16
  prompts (the default): the same top token at 612 of 614.
- Decode about 142 tok/s from a 5-token prompt, prompts 1,800-2,000 tok/s from
  4K to 64K tokens: the same as the dedicated path it replaced, measured the
  same way (table in [inkling-2.md](inkling-2.md#on-the-gpu)).

## Containers encoded before these surfaces

A graph written by an older build records facts this build does not read
(Inkling 2's config facts under `execution.inkling2`, from before these
surfaces). The planner refuses any surface key it does not read, naming it,
with the command that fixes it:

```
amql-cli graph upgrade <container> [--out <dir>] [--dry-run]
```

It rebuilds the execution surface from those facts (`Amql.Hf.GraphUpgrade`,
through the config adapter that wrote them) — exactly what a fresh encode
writes — and leaves the tensors, the index and every other file alone, keeping
the old graph as `system_graph.json.pre-upgrade` (or writing to `--out`). It
reads only `index.json` and `system_graph.json`, so it works on a copy without
segments.
