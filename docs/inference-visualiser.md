# Inference Visualiser — watching a request flow through the model

**Status:** Design proposal
**Author:** AMQL / Ariadne
**Date:** 2026-09-25

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

Each phase is independently useful and testable.

1. **Recorder + JSON + CLI flag.** Per-operator hook in `GenericRuntime`, trace model,
   `--trace-json`. Verifiable headlessly on the 0.8B with no GUI involved.
2. **Static map.** Layout + zoom/pan + LOD, colour by an aggregate over a loaded trace.
3. **Step scrubber.** Timeline, play/pause, per-step metrics, top-k logit panel.
4. **Live in-process run** from the GUI.
5. **Ranking + edit loop.** Sortable tensor table; right-click → scale/zero/patch → re-run.
6. **Compare mode.** Two traces, edges coloured by Δ; plus the causal-attribution and
   logit-lens overlays.

## 10. Risks to resolve in Phase 1

- **The CUDA path may bypass CPU-side hooks.** `CudaShim` runs GEMMs on device; if operator
  outputs never materialise as managed `float[]`, the recorder sees nothing. Needs an
  explicit answer early — either force CPU when tracing, or instrument the device path too.
  Tracing is a debugging tool, so forcing CPU is an acceptable answer if it is *stated*.
- **Native deps in the GUI process.** Referencing `Amql.Inference` pulls CUDA natives into
  `amql-gui`. Must confirm `CudaShim.Enabled` is lazy and that the GUI still starts on a
  machine without the runtime.
- **Hook overhead.** Per-operator callbacks on a hot loop cost something. Opt-in only, and
  measured before it ships.
- **Trace size.** Bounded by §7, but needs a real number from a 27B run before the format
  is frozen.
- **Linear-attention layers have no attention matrix.** 3 of every 4 layers in Qwen3.5 are
  recurrent; their "attention" overlay is the recurrent state, not a softmax grid. The map
  must not imply otherwise.
