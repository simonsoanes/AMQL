# Depth and conditioning ops

Generic operations for stacks that loop their layers, mix their residual over
depth, and condition their attention without a rotary embedding. Each is a
structural fact on the execution surface (`system_graph.json`), planned by
`Planner` into `ComponentOpPlan` / `LayerPlan` ops, and run by
`GenericRuntime`. Nothing here names a model family: a checkpoint's spellings
live in its layer role map (`execution.roles`), written by its config adapter
on import. Inkling 2 looped is the first stack described this way
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

The device engine does not run these ops yet: `GpuEngine.Unsupported` refuses
each by name ("looped depth", "block attention residual", "cross-pass message
gates", "residual short convolutions", "a learned relative score bias",
"attention sink logits", "attention conditioning", "packed experts…", "a padded
vocabulary"), so a session falls back to the CPU.

On the tiny looped fixture the planned path's logits equal Inkling2Model's
bit for bit at every step of a 12-token prompt and 8 greedy tokens, and so do
all 400 visit outputs (`DepthOpsTests`). On Inkling-2-Mini (CPU, a graph-upgraded copy) every logit of a 58-token prompt and 8 greedy tokens is the same as well (66 steps in about a minute for either path, load included).

## Containers encoded before these surfaces

A graph that records only a dedicated session's facts (Inkling 2 before this
change) is refused by the planner with the command that fixes it:

```
amql-cli graph upgrade <container> [--out <dir>] [--dry-run]
```

It rebuilds the execution surface from the facts the graph already holds —
exactly what a fresh encode writes — and leaves the tensors, the index and
every other file alone, keeping the old graph as
`system_graph.json.pre-upgrade` (or writing to `--out`). It reads only
`index.json` and `system_graph.json`, so it works on a copy without segments.

`AMQL_RUNTIME=planned` opens such a container on the planned path in
`DecodeSession.Open`; otherwise the dedicated session (Inkling2Gpu or
Inkling2Model) is still the one used.
