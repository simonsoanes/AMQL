# Star-Elastic Pruning

Status: **Phase 1 built** (FFN channels, `prune --approach star-elastic`). Phases 2–3 and the
no-grad recovery refit are designed below but **not built yet**.

## Purpose

Add a `prune --approach star-elastic` mode that shrinks a container by *structured
width pruning* — slicing the model's **FFN intermediate dimension** (and later its
softmax attention heads and MoE experts) to the most important channels, ranked by
**activation-based importance scores** from a single forward pass over a corpus.

This is the zero-shot core of [Star Elastic](https://arxiv.org/abs/2605.07182)
(its stages 1 + 2: importance estimation + nested slicing). AMQL has no autograd, so
stage 3 (the learnable router + knowledge-distillation *elastic training* that makes
the paper's sliced submodels recover full accuracy) is out of scope here — see
*Risks & limitations*.

## What already exists

- `prune` currently drops **whole decoder layers** only (`Amql.Merge.LayerPruner`),
  three ranking approaches (`provenance`/`corpus`/`random`), hitting `--target-bytes`
  with an exact byte estimate and a fail-closed floor.
- `moe-ify` (`Amql.Merge.MoeIfy`) is the precedent for everything this feature does:
  it samples FFN inputs through `GenericRuntime.FfnInputCapture`, computes
  `hf @ weightᵀ`, slices tensors at the **byte level** (`GatherRows`/`GatherCols`
  over `OperandStore.Resolve` → `Dtype`/`Payload`), and rebuilds the stack segment
  via `SegmentWriter.Write` + a rewritten `system_graph.json`/`index.json`.
- The `ExecutionSurface` facts (`FfnSurface.IntermediateSize`,
  `AttentionSurface.NumQHeads`/`NumKvHeads`, `MoeSurface.Experts`) are the single,
  homogeneous source of truth the planner reads.

## Data model

Stack tensors are named `{layer}.<op>` and resolved through the operand store:

| Axis | Tensors sliced | Surface fact rewritten |
|------|----------------|------------------------|
| FFN channels | `{l}.mlp.gate_proj.weight`, `{l}.mlp.up_proj.weight`, `{l}.mlp.down_proj.weight` (aliases `feed_forward.w1/w3/w2`) | `FfnSurface.IntermediateSize` |
| Attention heads | `{l}.self_attn.{q,k,v}_proj.weight`, `{l}.self_attn.{o,out}_proj.weight` | `AttentionSurface.NumQHeads`/`NumKvHeads` |
| MoE experts | `{l}.mlp.router.weight`, `{l}.mlp.experts.{e}.{gate,up,down}_proj.weight` | `MoeSurface.Experts` |

Weight shape convention (HF `nn.Linear`): projections are `[out, in]`, so

- gate/up `[intermediate, hidden]` — slicing a channel = dropping a **row**;
- down `[hidden, intermediate]` — slicing a channel = dropping a **column**.

`FfnSurface.IntermediateSize`, `AttentionSurface.NumQHeads/NumKvHeads`, and
`MoeSurface.Experts` are each a **single** value, so AMQL's slice is **homogeneous**:
the same kept channels/heads/experts on every layer. (The paper supports
per-layer-heterogeneous FFN; AMQL's surface does not.)

## The importance formulas (as implemented)

The paper aggregates `Σ` over batch `B` and sequence `L`. AMQL runs a single
token-serial forward over the corpus (its existing `--sample` cap), so `B = 1` and
the sum is over the sampled positions:

- **FFN channels** — `F_j = Σ_layers Σ_t |(hf · W_upᵀ)[t,j]|` for channel `j`,
  where `hf` is the normed pre-FFN input. (The paper scores the `up` branch
  `X·W_upᵀ`; `gate` shares the channel index, so gate/up/down are all sliced by the
  same kept set.)
- **Attention heads** (*Phase 2*) — `F_h = Σ_t ‖Attn(XW_q^h, XW_k^h, XW_v^h)‖₂`,
  the per-head context-vector norm before `o_proj`.
- **MoE experts** (*Phase 3*) — REAP: `S_e = mean_{x active-e} g_e(x)·‖f_e(x)‖₂`.

Scores are aggregated across layers into one global ranking (homogeneous slice).
Kept units are the top-K indices **restored to original index order** — a plain
smaller model, not a permuted one. Nesting as a *contiguous prefix* of the ranking
is only meaningful when one artifact emits several budgets, which AMQL's
single-output `prune` does not.

## Budget model

Same shape as `LayerPruner`: `--target-bytes` is the whole-output budget. Let
`inter` be the original FFN intermediate, `K` the kept width. Removing one channel
from one dense layer frees `3·hidden·elemSize` bytes (one gate row + one up row +
one down column). The pruner scans `K = inter … 1`, estimates each candidate's exact
output size (rebuilt stack header + payload, index, graph — with the token-map and
every other file byte-copied), and commits the largest `K` that fits. Below the
floor it fails closed, like the layer pruner.

## Phasing

| Phase | Axis | Status |
|-------|------|--------|
| 1 | FFN channels (homogeneous intermediate width) | **built** |
| 2 | Softmax attention heads (homogeneous head count, GQA-aware) | designed, not built |
| 3 | MoE experts (opt-in `--slice-moe`, REAP ranking) | designed, not built |
| — | No-grad recovery refit (`down_proj` least-squares on calibration activations) | future work |

Each phase lands as its own reviewable change, wired through `prune --approach
star-elastic`, with tests proving the output is a complete, planable, verifiable
container and that the slice is byte-exact for the chosen top-K.

## Risks & limitations

- **No elastic training.** Zero-shot slicing a model that was never trained
  "elastified" degrades quality more than the paper's reported numbers — the same
  trade-off AMQL's `prune`/`moe-ify` already accept (quality gated by perplexity,
  not fine-tuned). The least-squares `down_proj` refit (above) is the cheap no-grad
  recovery and the likely next milestone.
- **Homogeneous only.** Per-layer-heterogeneous widths (the paper's option) cannot
  be expressed in a single `FfnSurface`/`AttentionSurface`; extending the surface to
  per-layer dimensions is a much larger planner/exporter change and is not planned.
- **Flat encodings only.** Slicing is byte-granular (`Dtype.ElementSize()`). A stack
  stored in a packed/blockwise encoding (FP4, or a `strip`-quantized Q4_0/MXFP4
  stack) is refused with a clear message rather than silently corrupted.
- **Softmax attention only** for the head axis. Linear-attention (GDN) layers have
  no per-head `Attn(·)` to score; they are left untouched and their FFN is still
  pruned (the mixer and the FFN are independent operators).
- **Families the planner refuses** (`Mamba2`/`Kda`/`Mla`/`ConvQkv`, which have no
  managed forward) cannot be scored for the head axis; their dense FFN channels are
  pruned only if the planner serves them.