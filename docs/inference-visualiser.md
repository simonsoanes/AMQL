# Inference Visualiser — watching a request flow through the model

**Status:** Phases 1, 2, 3 and 6 (compare) implemented; 4 (live run) and the
causal/logit-lens overlays not started
**Author:** AMQL / Ariadne
**Date:** 2026-09-25 (design), updated 2026-09-25 (implementation status)

## 1. Purpose

Everything AMQL does to a model today is *structural*: encode it, merge it, prune it,
quantize it, export it. What is missing is the ability to watch one concrete inference
run happen and answer a question the structural views cannot:

> For **this** prompt, which tensors actually mattered?

The intended workflow is a loop, and the visualiser only earns its place if it closes it:

1. Run a prompt through a container.
2. See the forward pass on a large zoomable map — which operators fired, how much signal
   each carried, which experts were routed to, which logits were in play at each step.
3. Rank the weight tensors by how much they contributed.
4. Edit one of them (scale it, zero it, patch it) from the map.
5. Re-run and see the difference, on the same map, against the previous run.

Step 4 and 5 are the point. Steps 1-3 exist to make them *aimed* rather than random.

## 2. What already exists

This is not a greenfield feature. The runtime already has the seams:

| Seam | Location | What it gives |
|---|---|---|
| `LayerNormTrace` | `GenericRuntime.cs` | per-layer attention-output and FFN-output L2 norms |
| `AttentionTrace` | `GenericRuntime.cs` | `LayerHeadAttention(Layer, Head, Weights)` — post-softmax attention rows |
| `FfnInputCapture` | `GenericRuntime.cs` | the pre-FFN normed input per layer (the MoE-ification seam) |
| `TensorLoadTrace` | `GenericRuntime.cs` | weight load events: name, shape, cache hit |
| `Trace` / `BeginTrace` / `EndTrace` | `GenericRuntime.cs` | collected `(Layer, R, D)` per forward |
| `SetPatch` / `ResidualPatch` | `GenericRuntime.cs` | residual-stream override after a given layer — activation injection |
| `RunLayerInternal` | `GenericRuntime.cs` | the per-layer choke point every forward passes through |
| `CausalTracer` | `CausalTracer.cs` | **ROME-style per-layer attribution**: clean vs corrupted vs restored `P(target)`, `LayerDelta[]`, `LayerShare[]` |
| `StepOutcome` | `InferenceCommands.cs` | token, position, top-k candidates with softmax probabilities, per-layer trace |
| `TensorPatchTools.ApplyEdit` | `Amql.Merge` | `Set` / `Add` / `Scale` edits producing a `WeightPatch` |

`CausalTracer` is the important one: it already computes *causal* importance per layer,
which is a much stronger answer to "which tensors matter" than activation magnitude. The
visualiser should present it, not reinvent it.

Two things do **not** exist and are the bulk of the work:

- **Per-operator granularity.** Everything above is per-*layer* (two scalars per layer).
  A map of 64 boxes is not what was asked for; the ask is per-tensor.
- **Any GUI rendering.** `Amql.Gui` references only `Vindex3` and `Safetensors` and shells
  out to the CLI for everything else. There is no canvas, no zoom/pan, no graph layout
  anywhere in the codebase.

## 3. The map

### 3.1 Node kinds

Two kinds, deliberately distinct:

- **Op nodes** — a compute step inside a layer (`attn_q` matmul, `rope`, `softmax`,
  `attn_output`, `residual_add`, `ffn_gate`, `silu`, `ffn_down`, `rms_norm`, the linear
  -attention `conv`/`gate`/`decay` steps, the MoE router and expert combine). These carry
  *activations* and answer "how much signal flowed here".
- **Weight nodes** — the parameter tensors feeding those ops (`blk.12.attn_qkv.weight`,
  `blk.7.ffn_gate_exps.weight[3]`, …). These are leaves, and they are what the user edits.

Keeping them separate matters: the thing you can *measure* is an op, the thing you can
*change* is a weight, and conflating them makes the edit affordance ambiguous.

### 3.2 Edges

- op → op: activation flow. Thickness = L2 norm of the tensor on that edge (log-scaled,
  normalised per run). Colour = a second selectable metric.
- weight → op: a short stub, drawn in the weight's own colour once selected/ranked.

### 3.3 Layout

Hierarchical, left-to-right by layer, operators stacked vertically within a layer, with a
collapsible group header per layer. Layout is computed **once per model** and cached —
never per frame. A 64-layer model at ~20 ops is ~1,300 op nodes plus weight leaves, which
is comfortably renderable if it is not a WPF visual per node (see §5).

Level of detail: below a zoom threshold, layer groups collapse to a single bar showing the
layer's aggregate; labels disappear; only the trunk edges draw. This is what makes "giant"
tractable.

## 4. Metrics

Two tiers, because the honest answer is expensive and the cheap one is still useful.

### 4.1 Cheap tier — captured on every run

| Metric | Definition | Why |
|---|---|---|
| `l2` | L2 norm of the op's output | raw signal magnitude |
| `residualShare` | ‖op output‖ / ‖residual stream‖ at that point | comparable *across* layers; the standard contribution measure |
| `maxAbs`, `meanAbs` | peak / average magnitude | spots dead or exploding tensors |
| `ms` | wall time | spots the hot path |
| `topK` | per step: token id, text, probability | "which logits were involved" |
| `entropy`, `top1Margin` | per step | how decided the model was |
| `routedExperts` | per MoE layer: selected expert ids + router weights | genuinely sparse, and the most interpretable signal in a MoE |
| `attentionRow` | optional, per (layer, head) | already captured by `AttentionTrace` |

### 4.2 Expensive tier — on demand

- **Causal attribution** via `CausalTracer`: one clean forward, one corrupted forward, then
  one forward per traced layer with the clean residual restored. `LayerShare[]` recolours
  the map by *causal* importance rather than magnitude. A high-`l2` tensor that changes
  nothing when restored is noise; a modest one with a large share is the target. This is
  the difference between a pretty picture and an answer.
- **Logit lens**: project each layer's residual through the output head, so you can see at
  which depth the eventual prediction forms. Cheap (one matmul per layer) and very legible.

Both are buttons, not defaults — they cost extra forwards.

## 5. Rendering

WPF will not survive 1,300+ live `Shape` elements animated per token. Approach:

- One custom `FrameworkElement` with `OnRender(DrawingContext)`. All geometry drawn
  imperatively from a cached layout model.
- Pan/zoom as a `Matrix` applied to coordinates during render (crisper text than a
  `ScaleTransform`), mouse-wheel zoom about the cursor, drag to pan, fit-to-view, and
  explicit zoom-to-selection.
- Manual hit testing against cached node rects — no per-node visual tree, so picking is a
  loop over rects rather than a tree walk.
- Redraw only on step change, interaction, or metric change (`InvalidateVisual`), not on a
  timer, except in live mode where it is throttled to ~20 fps through the `Dispatcher`.

## 6. Live vs post-hoc

Both, over the same recorder:

- **Post-hoc (primary).** Run to completion, capture the whole trace, then scrub. Analysis
  needs comparison across steps, which a live animation is bad at. A timeline slider,
  play/pause, and per-metric aggregation over the run (max / mean / last) are the core of
  the tool.
- **Live.** The GUI subscribes to recorder events and animates as tokens are produced.
  Inference runs on a background thread; UI updates are marshalled and coalesced.

## 7. Trace format

Serialisable to JSON so runs can be saved, diffed and reproduced:

```
RunTrace   { model, componentId, promptTokens, sampling, weightWorkingSet, graph: NodeDef[], steps: StepTrace[] }
NodeDef    { nodeId, layer, op, weightRefs: [tensorRef] }        // topology, once per run
StepTrace  { index, position, token, tokenId, topK, entropy, top1Margin, nodes: NodeTrace[], moe?, attention? }
NodeTrace  { nodeId, l2, residualShare, maxAbs, meanAbs, ms }
```

Topology is factored out into `NodeDef[]` and referenced by id from each step, which is the
difference between a 10 MB trace and a 300 MB one. `--trace-steps` caps the step count;
an aggregate-only mode keeps just the per-node reductions for very long runs.

## 8. Where the code lives

- `src/Amql.Inference/Tracing/` — `TraceRecorder`, the trace model, the per-operator hook.
  Shared by CLI and GUI so there is one definition of a trace.
- `src/Amql.Cli` — `generate --trace-json <path>` (and `--trace-steps`), so a trace can be
  captured headlessly and opened later.
- `src/Amql.Gui/Visualiser/` — the window, the canvas element, layout, the panels.
- `Amql.Gui.csproj` gains a reference to `Amql.Inference`. This is a real change in
  character for the GUI, which currently shells out for anything computational; running
  in-process is what makes live animation possible at all.

## 9. Phasing

| # | Phase | Status |
|---|---|---|
| 1 | Recorder + JSON + `generate --trace-json` | **Done.** `OpTrace` / `ExpertRoutingTrace` hooks in `GenericRuntime`, `TraceRecorder`, `--trace-json`. Verified headlessly on the 0.8B; instrumentation produces byte-identical text with and without the flag. |
| 2 | Static map: layout, zoom/pan, LOD, colour by aggregate | **Done.** `ModelMapCanvas` + `TraceMetrics`. |
| 3 | Step scrubber, per-step metrics, top-k panel | **Done.** |
| 4 | Live in-process run from the GUI | **Done, but not in-process.** A `▶ Run…` button drives `generate` in a CLI child process and tails its `--trace-stream` file, applying each step as it lands (no animation, per request). Chosen over hosting `DecodeSession` in the window: the model stays out of the GUI process, the existing kill-the-process-tree path is the cancel mechanism (there is no `CancellationToken` in `Amql.Inference`), and a runtime crash cannot take the window with it. |
| 5 | Ranking + edit → re-run loop | **Done, with one deliberate gap.** Ranking and the right-click menu exist, but the menu *copies* the `edit-tensor` / `generate --patch` commands rather than running them — the main window already has a command runner with output capture, and a second place that knows how to invoke the CLI is a second place to keep in step. `edit-tensor` (whole-tensor scale/zero/offset) was added for this, because `change-tensor` edits a single cell and a single cell of a 4M-element matrix is unmeasurable downstream. |
| 6 | Compare mode; causal and logit-lens overlays | **All done.** Compare recolours on a diverging ramp with nodes matched on layer+op+weight. `generate --attribute` runs `CausalTracer` and stores per-layer shares; `--logit-lens` projects each layer's residual through the head; `--trace-attention` records per-head weights for the last query row. |

## 10. Risks — as resolved

- **Does the CUDA path bypass CPU-side hooks?** No. `StepForward` already reads
  `hidden.Row(...)` unconditionally to feed `LayerNormTrace`, so activations are always
  materialised in managed memory even when GEMMs run on device. The hooks see real values
  on both paths.
- **Native deps in the GUI process.** `Amql.Gui` now references `Amql.Inference`, which
  brings the CUDA shim with it. Startup was checked on a machine with the runtime present
  and is clean; startup on a machine *without* it has not been checked.
- **Hook overhead.** Opt-in only: with no delegate attached, each observation site costs one
  null check and the clock is never read. Not yet benchmarked under tracing.
- **Trace size.** 211 KB for 16 tokens over 150 operator nodes on the 0.8B, with topology
  interned once. A 27B run has roughly four times the operators; the per-step cost is what
  scales, and `--trace-steps` is not implemented yet.
- **Linear-attention layers have no attention matrix.** Still true, and now
  handled in the UI: the attention panel says how many softmax layers it is
  showing out of the total, so three absent columns in four read as expected
  rather than as missing data. The linear-attention mixer itself is broken out
  into its eight projections plus the recurrent core, so those layers are not
  second-class on the map.

## 11. Known limitations of what shipped

- **The rendering has not been judged by a human for the lens, attention or live
  panels.** The window has been confirmed to open and load a real trace, and the
  stream tail has been proven against a file being actively appended to, but
  nobody has clicked `▶ Run…` or looked at the two new panels.
- **Prefill is not traced.** The recorder attaches after prefill, so the map shows
  decode steps only. For a long prompt that is the cheaper half of the run. The
  lens and attention hooks are inside `RunLayerInternal` and `StepForward`, so
  extending coverage means attaching before prefill and giving it its own step
  record.
- **One operator per conv mixer.** `RunConv` is a single node; its `in_proj` and
  `out_proj` are not separately instrumented. The linear-attention mixer and the
  softmax mixer are fully broken out.
- **Attention is the last query row only**, which is what `AttentionTrace`
  captures. It is not the full attention matrix, and the panel says so.
- **Causal attribution is per layer**, not per operator, and its cost is one
  forward per traced layer plus two. On a 64-layer model that is 66 forwards.
- **The logit lens costs a full head GEMM per layer per step**, so it is
  impractical on a large model. It is opt-in and the help text says so.
- **No GUI tests.** `Amql.Tests` targets `net10.0` and cannot reference the
  `net10.0-windows` GUI project, which is why every reduction, layout and
  comparison rule lives in `TraceMetrics` and `TraceRecorder` where it can be
  tested. The WPF layer is deliberately thin.
