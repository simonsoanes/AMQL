# Embedding Models in AMQL — nomic-embed-text-v1.5 Ingest, VIndex3 Export, and `embed`

**Status:** Design proposal
**Author:** AMQL / Ariadne
**Date:** 2026-09-22

## 1. Purpose

AMQL today is a **generative** pipeline: `encode` a decoder checkpoint, transform it (`merge`,
`prune`, `moe-ify`, `generate-mtp`), `generate` tokens from it, and `export` it back out. Every
one of those steps assumes an autoregressive model with an output head over a vocabulary.

This document designs the **embedding** side of the same system, with three deliverables:

1. **Ingest** — `encode` an embedding checkpoint into a VIndex3 container, with
   **nomic-embed-text-v1.5** as the first supported family.
2. **Export** — materialise an embedding model *from a VIndex3 container*, potentially any
   container, into a checkpoint that sentence-transformers / HF / ONNX consumers load.
3. **Serve** — a CLI command that turns a chunk of content into a vector:
   `amql-cli embed <container> --text "…"`.

The design follows the repo's existing discipline: **judged facts only, refuse by name rather
than approximate, calculation over training, and byte-exact round-trip where the container is
the authority.** Where nomic's contract and AMQL's current model of the world genuinely
disagree, §3 says so plainly instead of bending either side.

---

## 2. What nomic-embed-text-v1.5 actually is (researched)

Everything in this section was read from the published repository
(`nomic-ai/nomic-embed-text-v1.5`) on 2026-09-22: `config.json`, the `model.safetensors`
header, `1_Pooling/config.json`, `modules.json`, `sentence_bert_config.json`,
`config_sentence_transformers.json` and `README.md`.

### 2.1 It is not a BERT, and it is not stock Transformers

| Property | Value |
|----------|-------|
| `model_type` | `nomic_bert` |
| `architectures` | `["NomicBertModel"]` |
| Loading | `auto_map` → `nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel` (**trust_remote_code / custom modelling**) |
| Parameters | ≈137M (23.4M embedding + 113.2M encoder + norms) |
| `torch_dtype` | `float32` — every tensor in the checkpoint is **F32** |
| Shape | hidden 768, 12 layers, 12 heads, head_dim 64, intermediate 3072, vocab 30528 |
| Served context | **8192** (`sentence_bert_config.max_seq_length`), though `max_position_embeddings` and `max_trained_positions` both say 2048 |

It descends from GPT-2's config lineage (hence the vestigial `summary_type`, `n_embd`,
`n_layer`, `attn_pdrop` spellings alongside the standard HF ones), but three facts make it
structurally unlike both BERT and a decoder LM:

- **Bidirectional.** `causal: false`. Attention is full, not masked-causal.
- **RoPE, not learned positions.** `rotary_emb_base`/`rope_parameters.rope_theta` = 1000.0,
  `rotary_emb_fraction` 1.0 (full head width), `rotary_emb_interleaved` **false** (half-pairing,
  NeoX-style), no NTK scaling (`rotary_emb_scale_base: null`, `rotary_scaling_factor: null`).
  Consequently **there is no `position_embeddings` tensor in the checkpoint at all** — which is
  also why the 2048 `max_position_embeddings` is vestigial and 8192 is the real served limit.
- **Post-LayerNorm, with biases.** `prenorm: false`, `use_rms_norm: false`,
  `layer_norm_eps`/`layer_norm_epsilon` = 1e-12. Every LayerNorm ships **weight and bias**.
  By contrast the linear layers are bias-free: `qkv_proj_bias: false`, `mlp_fc1_bias: false`,
  `mlp_fc2_bias: false`.
- **SwiGLU MLP, not GELU.** `activation_function: "swiglu"`, `hidden_act: "silu"`,
  `fused_bias_fc: true`.

### 2.2 The tensor contract (read from the safetensors header, all F32)

| Tensor | Shape | Notes |
|---|---|---|
| `embeddings.word_embeddings.weight` | 30528×768 | WordPiece vocab padded to a multiple of 64 (`pad_vocab_size_multiple`) |
| `embeddings.token_type_embeddings.weight` | 2×768 | `type_vocab_size` 2 |
| *(absent)* | — | **no `embeddings.position_embeddings.weight`** — RoPE |
| `emb_ln.weight` / `emb_ln.bias` | 768 each | post-embedding LayerNorm; nomic-bert spelling, **not** `embeddings.LayerNorm.*` |
| `encoder.layers.N.attn.Wqkv.weight` | **2304×768** | fused Q,K,V (3×768), no bias |
| `encoder.layers.N.attn.out_proj.weight` | 768×768 | |
| `encoder.layers.N.mlp.fc11.weight` | 3072×768 | SwiGLU **gate** |
| `encoder.layers.N.mlp.fc12.weight` | 3072×768 | SwiGLU **up** |
| `encoder.layers.N.mlp.fc2.weight` | 768×3072 | down |
| `encoder.layers.N.norm1.weight` / `.bias` | 768 each | post-attention-residual norm |
| `encoder.layers.N.norm2.weight` / `.bias` | 768 each | post-FFN-residual norm |

for `N ∈ 0..11`. **There is no `lm_head`, no pooled-output projection, and no classifier.**
The embedding is produced by *pooling the final token states*, not by a tensor.

The repo also ships nine `onnx/*` variants (`model.onnx`, `fp16`, `int8`, `q4`, `q4f16`,
`quantized`, `uint8`, `bnb4`) — out of scope for ingest (§13 Q6), but evidence that consumers
expect this model to be quantised for serving.

### 2.3 The embedding recipe (this is the part most implementations get wrong)

From `modules.json` + `1_Pooling/config.json` + the model card:

```
tokenize (WordPiece, adds [CLS] … [SEP])
  → NomicBertModel forward (bidirectional, RoPE θ=1000, post-LN, SwiGLU)
  → mean pooling:            pooling_mode_mean_tokens = true   (all other modes false)
  → F.layer_norm(pooled, normalized_shape=(dim,))              ← v1.5 Matryoshka step
  → slice to matryoshka_dim  (e.g. [:, :512])
  → F.normalize(p, p=2, dim=1)                                  ← L2, done by the CALLER
```

Three details that change the output vector and must be reproduced exactly:

1. **There is no `Normalize` module in `modules.json`** — only `Transformer` then `Pooling`.
   L2 normalisation is the *caller's* responsibility, and the model card does it explicitly.
2. **v1.5 applies an unweighted `layer_norm` before truncation.** This is the Matryoshka
   recipe: normalise → slice → L2. Slicing before the layer_norm produces a different vector.
3. **Mean pooling averages every non-pad token, including `[CLS]` and `[SEP]`.**
   sentence-transformers' mean pooling masks padding only; it does not drop special tokens.

`max_seq_length` is 8192 and `do_lower_case` is **false** in `sentence_bert_config.json` —
because the HF tokenizer's own normalizer already lowercases (see §13 Q4).

### 2.4 Task prefixes

The model card requires a task prefix on the input text:

| Prefix | Use |
|---|---|
| `search_query: ` | queries (asymmetric retrieval) |
| `search_document: ` | documents / passages |
| `classification: ` | classification inputs |
| `clustering: ` | clustering inputs |

These are **plain text prepended before tokenisation**, not special tokens. Getting them wrong
degrades retrieval measurably, so the CLI must apply them from a recorded table and refuse an
unknown task name rather than silently embedding bare text.

---

## 3. What AMQL can express today (assessed)

### 3.1 What is already reusable — more than expected

| Need | Already present |
|---|---|
| Bidirectional-safe batched forward | `DecodeSession.Prefill` already runs a fully-softmax component as **one batched pass** over all positions (`GenericRuntime.Embed` → `RunLayerInternal` per layer). That is exactly the encoder shape |
| RoPE at an arbitrary θ, full width | `Rope.Apply` + `PositionRope { Theta }`; `rotary_emb_fraction` 1.0 → plain `PositionRope`, and the **half-pairing** convention (`i` with `i + rotaryWidth/2`) matches `rotary_emb_interleaved: false` |
| Attention scaling | `AttentionSurface.ScoreScale = 1/√head_dim` matches `scale_attn_weights: true` |
| SwiGLU MLP with a gate/up pair | `DenseFfnOp { Gate, Up, Down }`, `IsGated`, `Activation.Silu`, and `ExpertGateGated`'s `activation(gate) * up` arithmetic — **`fc11`/`fc12`/`fc2` map onto this with no new kernel** |
| No output head | `ComponentOpPlan.Output` is already **optional**; `GenericRuntime.FinalNormAndHead` already handles the headless case and returns the normed hidden states |
| LayerNorm as a norm kind | `NormType.LayerNorm` exists in the enum |
| Tokenizer travel | `ModelToContainer.Encode` already copies `tokenizer.json` into the container root |
| Judgement/refusal discipline | `ModelConfigException`, `UnsupportedOperatorException`, `Planner`'s operand-closure rule |

The headless case is the lucky break: **an embedding model is a decoder plan with no `Output`
op, bidirectional masking, and a pooling step.** Most of the runtime is reusable as-is.

### 3.2 The real gaps

These are structural, not cosmetic. Each one is a place where AMQL currently *cannot* record the
fact, so the ingest must refuse rather than approximate.

| # | Gap | Where | Why it blocks nomic |
|---|---|---|---|
| G1 | **No bidirectional attention.** `AttentionKernel.Execute` is documented "Causal by construction" and breaks out of the key loop at the first future position | `Attention.cs` | nomic is `causal: false`. A causal encoder produces different vectors — silently, not loudly |
| G2 | **Norms have no bias operand.** `NormOp`/`NormSpec` carry `Weight` + `Eps` + `WeightOffset` only | `Plan.cs`, `Surface.cs` | every nomic LayerNorm ships a bias (26 bias tensors) |
| G3 | **No post-LN placement.** `NormPlacement` is `PreOnly` \| `PrePost` | `Surface.cs` | nomic is `prenorm: false`: `h = LN(h + attn(h))`. Post-LN is a *different function*, not a relabelling |
| G4 | **No encoder object kind.** `ObjectKind` has `DecoderStack` but no `EncoderStack`; `Planner` *requires* a `decoder_stack` object | `GraphModel.cs`, `Planner.cs` | naming nomic's stack a "decoder" would be a lie in the graph |
| G5 | **No pooling representation** anywhere | — | mean-over-non-pad-tokens has no home in `ExecutionSurface` |
| G6 | **The embedding norm is weightless.** `HeadSurface.EmbeddingNorm` is deliberately "no learned weights ship" | `Surface.cs`, `Plan.cs` | `emb_ln` has learned weight **and** bias |
| G7 | **No token-type embedding term.** `EmbeddingOp` gathers one table | `Plan.cs`, `GenericRuntime.cs` | `embeddings.token_type_embeddings.weight` (2×768) must be added |
| G8 | **Ingest refuses the config outright.** `ModelConfig.ReadTextFacts` *throws* when `layer_types` is absent, and `ArchMapper.DetectTextPrefix` only accepts `model` / `model.language_model` / `language_model` | `ModelConfig.cs`, `ArchMapper.cs` | nomic has no `layer_types` (all layers are identical) and its tensors live under `embeddings.` / `encoder.layers.` |
| G9 | **Fused `Wqkv`.** `AttentionOp` wants separate `QProj`/`KProj`/`VProj` | `Plan.cs`, `Planner.cs` | nomic ships one 2304×768 matrix |
| G10 | **No masking/padding model.** The runtime takes `int[] tokens` with no attention mask | `DecodeSession.cs` | mean pooling must exclude pad tokens in a batch |

G1, G2, G3 and G5 are the load-bearing ones. G8 and G9 are mechanical. G4, G6, G7, G10 are
small but must be explicit, because the alternative in each case is a silent approximation.

---

## 4. Representing an embedding model in VIndex3

The container stays the sole authority; these are additive schema facts. **Schema bumps 6 → 7**
(§4.6 explains why a bump rather than a nullable field).

### 4.1 `ObjectKind.EncoderStack` (new)

A stack whose layers run bidirectionally and which produces representations rather than logits.
`Planner` accepts *either* `DecoderStack` *or* `EncoderStack` for the component's stack object,
and the kind is what makes the directionality requirement (§4.2) non-optional — an encoder stack
with no recorded directionality refuses to plan.

`ObjectKind.OutputHead` stays **absent** for embedding models. That is already legal: `Head` is
optional on the surface and `Output` is optional on the plan.

### 4.2 `AttentionSurface.Directionality` (new)

```csharp
public enum AttentionDirectionality { Causal, Bidirectional }
```

Required on the attention surface for a softmax operator. `ArchMapper` writes `Causal`
explicitly for every decoder ingest — it is a *judged fact of the family*, not an executor
default — and `Bidirectional` for nomic. `AttentionKernel.Execute` gains a directionality
parameter: `Bidirectional` skips the future-position break and the causal mask entirely (the
window/sink/softcap paths stay as they are).

The planner refuses an absent value by name. Nothing infers bidirectionality from the absence of
a head.

### 4.3 Norm biases and post-LN

```csharp
public sealed class NormSpec {
    public required NormType Kind { get; init; }
    public required double Eps { get; init; }
    public float WeightOffset { get; init; }
    public bool HasBias { get; init; }          // NEW — operand-closure then requires "<site>.bias"
}

public enum NormPlacement { PreOnly, PrePost, PostOnly }   // NEW value
```

`NormOp` gains `OperandRef? Bias`. With `Placement = PostOnly` the layer loop becomes
`h = LN₁(h + attn(h))`, `h = LN₂(h + ffn(h))`, binding `norm1` to the post-attention site and
`norm2` to the post-FFN site. `Norms.ApplyInPlace` gains the bias term.

`HasBias` defaults false, so every existing container is unaffected and the operand-closure rule
(`Require(store, …)`) does the enforcement: a surface that says "bias" with no bias tensor in the
segment refuses the whole component.

### 4.4 `EmbeddingSurface` (new) — the input side, complete

The current `EmbeddingOp` is too narrow for nomic. Replace the borrowed `HeadSurface.EmbeddingNorm`
with a first-class surface on the component:

```csharp
public sealed class EmbeddingSurface {
    public required int VocabSize { get; init; }
    public required int HiddenSize { get; init; }
    public bool HasTokenTypeEmbedding { get; init; }   // operand: token_type.weight
    public int TokenTypeCount { get; init; }           // 2 for nomic; inputs use row 0
    public NormSpec? PostEmbeddingNorm { get; init; }  // emb_ln: LayerNorm + weight + bias
    public double? Scale { get; init; }                // √hidden where the family scales
}
```

`EmbeddingOp` gains `OperandRef? TokenTypeTable` and a weighted `NormOp? PostNorm`. The runtime
adds **row 0** of the type table (the single-sequence case) and refuses, by name, a request that
would need a non-zero segment — rather than quietly averaging segment types.

> **Why not fold row 0 into the word embedding?** It would be numerically exact for
> single-segment input and cheaper, but it destroys the round-trip: `export` could not reproduce
> `embeddings.token_type_embeddings.weight` byte-for-byte. The container is the authority, so the
> tensor is carried and added.

### 4.5 `PoolingSurface` (new) — the output side

```csharp
public enum PoolingKind { Mean, MeanSqrtLen, Max, Cls, Last, WeightedMean }

public sealed class PoolingSurface {
    public required PoolingKind Kind { get; init; }
    public bool MaskPadding { get; init; } = true;
    public bool IncludeSpecialTokens { get; init; } = true;  // nomic: [CLS]/[SEP] ARE averaged
    public NormSpec? PreTruncationNorm { get; init; }        // v1.5's unweighted layer_norm
    public IReadOnlyList<int>? MatryoshkaDims { get; init; }  // advertised truncation widths
    public bool L2Normalise { get; init; }                   // recorded, applied by the CLI
}
```

All six `PoolingKind` values exist because `1_Pooling/config.json` enumerates exactly them and the
ingest should record what it reads rather than hard-coding one. Only `Mean` has a managed
implementation in this build; the rest are **carried and refused at plan time**, exactly like an
unjudged position policy or layer operator. That keeps the fact and keeps the honesty.

`PreTruncationNorm` is *weightless* on purpose: v1.5's step is `F.layer_norm(x, (dim,))` with no
affine parameters, which is a different operation from the checkpoint's `emb_ln`.

### 4.6 Why a schema bump, not nullable fields

The repo's rule is that an executor never defaults and a graph of another schema is refused, not
upgraded. Adding `Directionality` as nullable would force exactly the choice the rule forbids:
either infer causal-on-absent (a silent default) or refuse every existing container (a
regression). Bumping to 7 and re-encoding is cheaper than either — the containers are derived
artifacts, `encode` is deterministic, and the re-encoded result is byte-exact by construction.
Existing schema-6 containers then fail with a clear "graph schema 6, this build requires 7"
rather than behaving subtly differently.

---

## 5. Ingest: `encode` a nomic checkpoint

### 5.1 A separate facts reader — do not loosen the decoder path

`ModelConfig.ReadTextFacts` refusing a config with no `layer_types` is **correct behaviour for a
decoder** and must stay. Embedding checkpoints get their own reader:

```csharp
public sealed record EncoderArchitectureFacts(
    string ModelType, string Architecture,
    int HiddenSize, int NumLayers, int NumHeads, int HeadDim,
    int IntermediateSize, string Activation, string ActivationFunction,
    int VocabSize, int TypeVocabSize,
    bool Causal, bool Prenorm, bool UseRmsNorm, double LayerNormEps,
    bool QkvProjBias, bool MlpFc1Bias, bool MlpFc2Bias,
    double RotaryTheta, double RotaryFraction, bool RotaryInterleaved,
    long MaxPositionEmbeddings, long MaxTrainedPositions, long ServedMaxLength,
    bool UseFlashAttn, bool ParallelBlock, int PadVocabSizeMultiple,
    string TorchDtype, int? PadTokenId);
```

`ModelToContainer.Encode` dispatches on `model_type` + `architectures`: `nomic_bert` /
`NomicBertModel` → the encoder path; anything else → the existing decoder path unchanged.

### 5.2 Facts judged, facts refused

**Judged and served** (nomic-embed-text-v1.5 passes all of these):

| Config fact | Judgement |
|---|---|
| `causal: false` | `Directionality.Bidirectional` |
| `prenorm: false`, `use_rms_norm: false` | `NormPlacement.PostOnly`, `NormType.LayerNorm`, `Eps = 1e-12`, `HasBias = true` |
| `rotary_emb_fraction: 1.0`, `rope_type: "default"`, `rotary_emb_interleaved: false` | `PositionRope { Theta = 1000 }` — full-width half-paired rotary, the existing served case |
| `activation_function: "swiglu"` + `hidden_act: "silu"` | `FfnType.Gated`, `Activation.Silu`, `ExpertGateGated` arithmetic |
| `scale_attn_weights: true` | `ScoreScale = 1/√64` |
| `qkv_proj_bias`/`mlp_fc1_bias`/`mlp_fc2_bias` all false | no bias operands on the linear layers — matches what AMQL already has |
| `type_vocab_size: 2` | `HasTokenTypeEmbedding`, `TokenTypeCount = 2` |
| `n_positions: 8192` + `sentence_bert_config.max_seq_length: 8192` | `ContextLength = 8192`; the 2048 pair is recorded as `MaxTrainedPositions` metadata |

**Refused by name** (a nomic-variant or other encoder hitting these must fail loudly):

| Config fact | Refusal |
|---|---|
| `prenorm: true` | post/pre mismatch is a different function — but note this *is* servable once `PreOnly` + `LayerNorm` + bias is wired, so the refusal message should say "not yet judged", not "impossible" |
| `use_rms_norm: true` | `LayerNorm`/`RmsNorm` are distinct kinds; an RMS encoder needs its own judgement |
| `rotary_emb_fraction < 1.0` | `PositionPartialRope` exists, but the *fraction-of-head-dim* semantics for nomic's rope have not been verified against its modelling code |
| `rotary_emb_interleaved: true` | `Rope.Apply` implements half-pairing only |
| `rotary_emb_scale_base` / `rotary_scaling_factor` non-null | NTK-by-parts scaling — carried unresolved, refused at plan time |
| `parallel_block: true` | a genuinely different block structure |
| `qkv_proj_bias`/`mlp_fc1_bias`/`mlp_fc2_bias` true | `AttentionOp`/`DenseFfnOp` have no bias operands |
| `pad_vocab_size_multiple` inconsistent with the tensor shape | refuse rather than pad silently |
| a pooling mode other than mean | carried in `PoolingSurface`, refused at plan time |

### 5.3 Tensor mapping and the two encode-time transforms

Prefix detection gains an encoder branch: `embeddings.` and `encoder.layers.` (never assumed —
probed, like `DetectTextPrefix`).

| Source tensor | Object | Segment-relative name |
|---|---|---|
| `embeddings.word_embeddings.weight` | `target.embedding` | `weight` |
| `embeddings.token_type_embeddings.weight` | `target.embedding` | `token_type.weight` |
| `emb_ln.weight` / `emb_ln.bias` | `target.embedding` | `post_norm.weight` / `post_norm.bias` |
| `encoder.layers.N.*` | `target.encoder_stack` | `N.*` (prefix stripped, as `BindLayers` does) |
| *(none)* | `target.output_head` | **object not created** |

Per layer, two transforms are needed:

**T1 — split `Wqkv`.** `encoder.layers.N.attn.Wqkv.weight` (2304×768) → three 768×768 operands
`N.self_attn.{q,k,v}_proj.weight`, matching what `AttentionOp` and the planner's operand-closure
check already expect.

> **The one ingest fact I have not verified (§13 Q1): the row order inside `Wqkv`.** Two
> conventions are in the wild for GPT-2-lineage fused QKV: **block-major** (`[all Q; all K;
> all V]`) and **head-interleaved** (`[q₀k₀v₀; q₁k₁v₁; …]`). nomic-bert's modelling file
> reshapes it, and I did not retrieve that file. The split is therefore written behind a single
> `QkvLayout` judgement with two implementations, **defaulting to neither**: the ingest must
> resolve it against the reference implementation's outputs (§10.2) before the first container is
> written. A wrong guess does not error — it produces plausible, wrong vectors. This is the
> highest-risk item in the design.

**T2 — rename the SwiGLU pair.** `mlp.fc11` → `gate_proj`, `mlp.fc12` → `up_proj`,
`mlp.fc2` → `down_proj`, `norm1` → the post-attention site, `norm2` → the post-FFN site.
No arithmetic — `DenseFfnOp` already computes `silu(gate) * up` then `down`.

Both transforms are **deterministic and invertible**, so `export` (§7.1) can reproduce the
original bytes exactly. The inverse of T1 is a concatenation in the same recorded layout; the
inverse of T2 is a rename. Neither widens nor narrows dtype.

### 5.4 What the container looks like afterwards

```
nomic-embed-text-v1.5/
  index.json            family "nomic_bert", encoding "F32", schema 7
  system_graph.json     component "target" (PrimaryText, EncoderStack, 12 layers, hidden 768)
                        objects: target.embedding, target.encoder_stack, target.final_norm
                        surface: Attention{12 q heads, 12 kv heads, head 64, Bidirectional},
                                 Ffn{3072, Silu, Gated}, Norm{LayerNorm, 1e-12, HasBias, PostOnly},
                                 Embedding{30528, token-type ×2, post-norm emb_ln},
                                 Pooling{Mean, mask padding, include specials,
                                         pre-truncation layer_norm, L2}
                        no OutputHead, no Head surface
  .segments/*.bin
  tokenizer.json        copied verbatim (WordPiece)
  pooling.json          NEW: the sentence-transformers metadata, carried for export
```

`final_norm` deserves a note: nomic has **no** top-level norm after the last encoder layer
(post-LN puts it inside the block), so `target.final_norm` binds to layer 11's `norm2` rather
than a separate tensor. `Planner` requires a `final_norm` object, so it must exist — but the
mapping has to be recorded as a *binding*, not fabricated as a tensor, or export would invent one.
This is §13 Q2.

---

## 6. Runtime: producing a vector

### 6.1 `EmbedSession`, not `DecodeSession`

`DecodeSession` is a *logits* contract: `Prefill` ends in `FinalNormAndHead` and slices the last
row. An embedding run wants every row, pooled, with no head. A parallel session type keeps both
contracts honest:

```csharp
public sealed class EmbedSession {
    public EmbedSession(ComponentOpPlan plan, OperandStore store, WeightPatch? patch = null);
    public Tensor2D Encode(int[] tokens, bool[]? mask = null);   // [n, hidden] token states
    public float[] Pool(Tensor2D states, bool[]? mask, PoolingSurface spec);
}
```

Internally it reuses `GenericRuntime.Embed` and `RunLayerInternal` in the **batched** path (an
encoder has no stateful layers, so `IsStateful` is false everywhere and one pass suffices), then
applies the final norm. No KV cache is retained, no position counter advances, and there is no
`Step`. The `WeightPatch` seam is kept, so the existing LoRA-style patching and `inspect`
tooling work on embedding containers unchanged.

`DecodeSession.Prefill`'s refusal to run twice is not inherited: `Encode` is a pure function of
its input and may be called repeatedly. That is the whole point of an embedding model.

### 6.2 The pooling kernel

Mean over rows where `mask` is true (default: all rows), including special tokens per
§4.5. Then, in order, exactly as the model card does:

1. optional unweighted `layer_norm` (`PreTruncationNorm`) — **before** truncation;
2. truncate to `--dims` (validated against `MatryoshkaDims` when recorded);
3. L2-normalise (`--no-normalize` to skip, recorded in the output metadata).

Truncation and normalisation are *not* fused, and the order is not configurable — it is part of
the v1.5 contract.

### 6.3 Batching and masking

G10. `Encode` takes an optional mask; the CLI's batch mode right-pads to the longest sequence in
the batch and passes the mask to both attention (pad keys masked to −∞) and pooling (pad rows
excluded). Single-text mode never pads, so the mask is all-true and the result is identical to
the unpadded path — which is a testable invariant (§10.3).

---

## 7. Export: an embedding model from a VIndex3 container

The user asked for export "from a VIndex3 — potentially any VIndex3". The honest answer has two
tiers, because **the target format is a function of what the container records, not a flag that
can be forced.**

### 7.1 Tier 1 — faithful: an encoder container → `nomic_bert` checkpoint

When the graph records an `EncoderStack` with `Bidirectional` attention, `PostOnly` LayerNorm
with biases, and a `PoolingSurface`, the export is the **exact inverse of §5.3**: re-fuse
`q/k/v_proj` → `Wqkv` in the recorded layout, rename the SwiGLU pair back to `fc11`/`fc12`/`fc2`,
re-emit `emb_ln.{weight,bias}` and `embeddings.token_type_embeddings.weight`, and write
`config.json` regenerated **from judged graph facts** (never copied from the source checkpoint —
the existing export rule).

Emitted alongside the weights, so the output is a drop-in sentence-transformers model:

```
<out>/
  config.json                     # model_type nomic_bert, architectures [NomicBertModel],
                                  # causal false, prenorm false, rope theta/fraction/interleaved,
                                  # swiglu/silu, biases false, type_vocab_size, pad_vocab_multiple
  model.safetensors               # F32, original nomic tensor names
  tokenizer.json                  # verbatim from the container
  1_Pooling/config.json           # from PoolingSurface
  modules.json                    # Transformer → Pooling (no Normalize module, per §2.3)
  sentence_bert_config.json       # max_seq_length, do_lower_case
  config_sentence_transformers.json
```

`auto_map` is **not** emitted: it points at `nomic-ai/nomic-bert-2048`, and claiming someone
else's remote code for our weights would be both wrong and fragile. The checkpoint declares
`model_type: nomic_bert` and relies on the consumer having the modelling code (or on the ONNX
tier, §7.3). This is a real portability limitation and is stated in the README section rather
than hidden.

**Round-trip gate:** `encode` the published checkpoint → `export` → byte-compare
`model.safetensors` against the original. Because §5.3's transforms are invertible and the
canonical encoding is F32 (no widening, no quantisation), this must be **byte-exact**, not
tolerance-bounded. That is the strongest correctness statement available for the ingest, and it
tests T1's layout guess at the same time.

### 7.2 Tier 2 — derived: a causal container → an embedding-capable artifact

This is where the design has to say no to something attractive. **A decoder container cannot be
faithfully exported as `nomic_bert`.** The blockers, in order of severity:

| Blocker | Convertible by renaming? |
|---|---|
| Pre-LN vs post-LN (`prenorm` differs) | **No** — a different function, not a different spelling |
| RMSNorm (no bias, `1+w` affine on Qwen3.x) vs LayerNorm (bias) | **No** |
| Causal vs bidirectional attention | **No** — changes every output vector |
| Partial rotary (Qwen3.x factor 0.25) vs full-width RoPE | **No** |
| Separate `q/k/v_proj` vs fused `Wqkv` | Yes (concatenate) |
| `gate/up/down_proj` vs `fc11/fc12/fc2` | Yes (rename) |
| `model.layers.N.self_attn.*` vs `encoder.layers.N.attn.*` | Yes (rename) |

Four of seven are arithmetic, not naming. Emitting a `nomic_bert` config over a Qwen3.x container
would therefore produce a checkpoint that **loads successfully and computes the wrong thing** —
precisely the failure mode the repo's refuse-by-name discipline exists to prevent.

So Tier 2 exports in a format that *is* faithful to what the container records:

- **`--arch amql-encoder` (default for a causal container):** the existing HF export path,
  unchanged, **plus** the embedding metadata — `1_Pooling/config.json`, `modules.json`,
  `sentence_bert_config.json` — and an `amql_embedding.json` recording the pooling spec, the
  task prefixes, the Matryoshka dims and `directionality: "causal"`. The artifact is honest:
  here is the model, here is how to pool it, and here is the fact that it was not trained as an
  embedding model.
- **The pooling itself is the derived part, and it is labelled.** Mean-pooling a causal LM's
  hidden states is a real technique, but its quality on this container is **unmeasured**. The
  exporter writes `"derived": true, "trained_as_embedding_model": false` into
  `amql_embedding.json` and prints a warning. It never writes `nomic_bert` as the `model_type`.

**Which pooling for a causal container?** `last` (final token) is the conventional choice for
decoder embeddings and is the default for Tier 2; `mean` is offered and warned about, since
mean-pooling causal hidden states is known to be weaker without training. Both are recorded in
the artifact, so a consumer can see exactly what was chosen. `Cls`, `Max`, `WeightedMean` and
`MeanSqrtLen` remain carried-but-refused unless a container records them.

**Refusal cases.** `export --arch nomic-bert` on a container that does not record
`Bidirectional` + `PostOnly` + LayerNorm-with-bias **refuses by name**, listing the specific
mismatching facts (the `layer_types` refusal pattern). There is no `--force`.

### 7.3 GGUF and ONNX tiers

Deferred, deliberately. GGUF has no encoder/pooling story in `HfCheckpointToGguf.cs` today, and
the ONNX variants nomic ships suggest that consumers wanting a quantised embedding model expect
ONNX rather than GGUF. Both are recorded as future work (§13 Q6) rather than half-designed here;
the T1 checkpoint is the input to either.

### 7.4 Quantisation

`--quant mxfp4` composes with the embedding export on the same eligibility split the decoder path
already uses: quantise `Wqkv` (or `q/k/v_proj`), `out_proj`, `fc11`/`fc12`/`fc2`; keep
**embeddings, token-type embeddings, all norms and all biases** at full precision. The n-gram/PLE
lessons from the Flash-Next work apply here too — a lookup table is not a GEMM weight and does
not belong in an MXFP4 split.

Note the round-trip consequence: an MXFP4 export is **not** byte-exact against the F32 original,
so the §7.1 gate runs on the unquantised path and the quantised path is tolerance-gated instead
(1e-3 on the pooled vector's cosine similarity to the F32 path, per `docs/CUDA_PLAN.md` §4).

---

## 8. CLI

### 8.1 `embed` — the command the user asked for

```
amql-cli embed <container> [--component target]
               (--text "…" | --file <path> | --stdin | --jsonl <path>)
               [--task search_query|search_document|classification|clustering|none]
               [--dims 768] [--no-normalize]
               [--format json|base64|raw|csv] [--batch-size 32]
               [--patch <p.safetensors>]
```

Behaviour:

- **Task prefix** is prepended verbatim from the container's recorded prefix table. An unknown
  `--task` refuses by name and lists the valid ones. `--task none` embeds bare text and is
  recorded as such in the output — it is opt-in because it is usually a mistake.
- **`--dims`** validates against `PoolingSurface.MatryoshkaDims` when recorded, and always
  against `1 ≤ dims ≤ hidden`. Truncation happens *after* the pre-truncation layer_norm (§6.2).
- **`--format json`** (default) emits `{"text":…, "task":…, "dims":…, "normalised":true,
  "embedding":[…]}` — self-describing, so a stored vector can never be silently misread.
  `base64` packs the float32 array for compact storage; `raw` writes little-endian float32 to
  stdout for piping; `csv` is one vector per line for bulk jobs.
- **`--jsonl`** reads `{"text": …}` per line and writes one result object per line — the bulk
  path for building an index. `--batch-size` controls padding groups; results are emitted in
  input order.
- Tokenises with the container's own `tokenizer.json` (no `--tokenizer` needed), adds `[CLS]`/
  `[SEP]` per the tokenizer's post-processor, and refuses input longer than the recorded
  `ContextLength` rather than truncating silently.
- The `--patch` seam is threaded through, so embedding containers work with the existing
  live-adapter / patching tooling.

Exit codes follow the existing CLI convention; every refusal names the fact that was missing.

### 8.2 `encode` and `export`

No new commands — `encode` gains the encoder dispatch (§5.1) and `export` gains `--arch
nomic-bert` alongside the existing default and the Flash-Next `--arch` (§7). `export
--list-architectures` grows the entry.

---

## 9. New and changed files

| File | Change |
|---|---|
| `src/Amql.Vindex3/GraphModel.cs` | `ObjectKind.EncoderStack`; `CurrentSchema` 6 → 7 |
| `src/Amql.Vindex3/Surface.cs` | `AttentionDirectionality` + `AttentionSurface.Directionality`; `NormSpec.HasBias`; `NormPlacement.PostOnly`; `EmbeddingSurface`; `PoolingSurface` + `PoolingKind` |
| `src/Amql.Vindex3/PositionPolicy.cs` | unchanged (nomic uses the already-served `PositionRope`) |
| `src/Amql.Hf/EncoderConfig.cs` (new) | `EncoderArchitectureFacts` + `ReadEncoderFacts`; the nomic-specific judgements and refusals (§5.2) |
| `src/Amql.Hf/ArchMapper.cs` | encoder dispatch, `embeddings.`/`encoder.` prefix probe, `QkvLayout` split (T1), SwiGLU rename (T2), the `final_norm` binding, no `output_head` object |
| `src/Amql.Hf/ModelToContainer.cs` | copy `1_Pooling/config.json`, `modules.json`, `sentence_bert_config.json` into the container as `pooling.json` |
| `src/Amql.Inference/Plan.cs` | `NormOp.Bias`; `EmbeddingOp.TokenTypeTable` + weighted `PostNorm`; `PoolingOp` on `ComponentOpPlan` |
| `src/Amql.Inference/Planner.cs` | accept `EncoderStack`; require `Directionality`; bind bias operands under `HasBias`; bind the pooling spec; keep operand-closure refusals |
| `src/Amql.Inference/Attention.cs` | `Directionality` parameter on `AttentionKernel.Execute`; an explicit padding mask |
| `src/Amql.Inference/Kernels.cs` | bias term in `Norms.ApplyInPlace`; a weightless layer_norm for the Matryoshka step |
| `src/Amql.Inference/EmbedSession.cs` (new) | `Encode` → token states; `Pool` → vector (§6) |
| `src/Amql.Cli/EmbedCommands.cs` (new) | `embed` command: task prefixes, `--dims`, formats, JSONL batching |
| `src/Amql.Cli/Program.cs` | `"embed" => Embed(args[1..])` |
| `src/Amql.Cli/ModelExporter.cs` | `--arch nomic-bert` (Tier 1 inverse map + ST metadata); `--arch amql-encoder` (Tier 2 + `amql_embedding.json`); the §7.2 refusal |
| `src/Amql.Cli/ExportConfig.cs` | regenerate the `nomic_bert` config from judged encoder facts |
| `README.md`, `docs/` | an embedding-models section with measured numbers |

---

## 10. Validation gates

### 10.1 Round-trip (the primary gate)

`encode` the published nomic-embed-text-v1.5 → `export --arch nomic-bert` → **byte-compare
`model.safetensors` with the original**, and diff the regenerated `config.json` against the
source field by field. Any difference is a defect in T1/T2 or in the config writer. This gate
runs offline, needs no reference implementation, and is the one that catches a wrong `QkvLayout`.

### 10.2 Numerical parity against the reference

Run the same input through sentence-transformers and through `amql-cli embed`, and compare:

- **token states** at layer 11 output: max abs diff < 1e-5 (F32, same arithmetic order);
- **pooled vector** at 768 dims: cosine similarity > 0.999999;
- **Matryoshka 512**: identical after slice + L2;
- the model card's own example pair (`search_query: What is TSNE?` vs
  `search_query: Who is Laurens van der Maaten?`) reproduces the published similarity ordering.

Marked to skip in offline builds, like the Flash-Next engine-load gate. **This gate is what
resolves §13 Q1** — a wrong `Wqkv` layout fails here immediately and unambiguously.

### 10.3 Invariants

- **padding invariance:** a single text embedded alone and embedded in a padded batch produce
  bit-identical vectors;
- **determinism:** two runs → byte-identical output (no RNG anywhere on this path);
- **prefix sensitivity:** `search_query: X` and `search_document: X` differ, confirming the
  prefix is applied and not dropped;
- **no-head plan:** the container plans and runs with no `OutputHead` object and no `Head`
  surface, and `generate` on it refuses by name rather than producing garbage;
- **refusals:** each §5.2 refusal fires on a synthesised config (`SyntheticCheckpoint` already
  exists for exactly this), naming the fact.

### 10.4 Suite

All existing CPU tests stay green after the schema bump, including the re-encoded decoder
containers. New tests cover the split/concat inverse, post-LN ordering, bias application, mean
pooling with and without a mask, the Matryoshka order, and both export tiers' refusals.

---

## 11. Phases (each gated)

**Phase A — schema + ingest, no runtime.** Add the §4 schema facts, `EncoderConfig`, the encoder
branch of `ArchMapper` (with `QkvLayout` resolved by §10.1/§10.2), and `pooling.json` carriage.
*Gate:* `encode` nomic-embed-text-v1.5 → `verify` passes; `export --arch nomic-bert` round-trips
byte-exact; every existing decoder container re-encodes at schema 7 with all tests green.

**Phase B — runtime.** `Directionality` in `AttentionKernel`, norm biases, `PostOnly` placement,
the token-type and post-embedding-norm terms, `EmbedSession`, and the pooling kernel.
*Gate:* §10.2 numerical parity and all §10.3 invariants.

**Phase C — `embed` CLI.** Task prefixes, `--dims`, the four output formats, JSONL batching, the
`--patch` seam.
*Gate:* the model card's examples reproduce end-to-end from the CLI; a 10k-line JSONL run
completes and is deterministic; refusal messages name the missing fact.

**Phase D — export from any container.** Tier 1 inverse map + sentence-transformers metadata;
Tier 2 `--arch amql-encoder` with `amql_embedding.json` and the `derived: true` labelling; the
§7.2 refusal on a causal container asked for `nomic-bert`.
*Gate:* Tier 1 byte-exact; Tier 2 loads in sentence-transformers as a custom-module model and
its `amql_embedding.json` states the derivation honestly; `--arch nomic-bert` on
`Qwen3.5-0.8B` refuses listing the four mismatching facts.

**Phase E — quantisation, measurement, docs.** MXFP4 split for the encoder path; measured numbers
(throughput, on-disk size at F32/F16/MXFP4, MRL quality at 64/128/256/512/768 on a held-out
similarity set); README section.
*Gate:* the MXFP4 cosine-similarity tolerance holds; the docs carry measured values, not
estimates.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Wrong `Wqkv` row order** — plausible vectors, silently wrong | `QkvLayout` has no default; resolved by the §10.2 parity gate before any container is written; §10.1 byte-exact round-trip catches it independently |
| **Causal masking left on** — every vector subtly wrong | `Directionality` is a required, judged fact; the planner refuses an absent value; §10.2 compares against the reference at layer granularity, where a causal mask shows up immediately |
| **Post-LN implemented as pre-LN** | `NormPlacement.PostOnly` is a distinct value with its own code path and a dedicated ordering test; §10.2 would fail outright |
| **Pooling omits or includes `[CLS]`/`[SEP]` wrongly** | `IncludeSpecialTokens` is recorded, not assumed; the model card's example pair reproduces only if this is right |
| **Matryoshka order wrong** (slice before layer_norm) | The order is fixed in code (§6.2), not configurable, and tested at 512 dims |
| **Tokenizer normalizer mismatch** (`do_lower_case: false` vs a lowercasing tokenizer) | Ingest reads the normalizer from `tokenizer.json` and records it; §13 Q4; a parity test on mixed-case input |
| **Tier 2 read as "we trained an embedding model"** | `derived: true` + `trained_as_embedding_model: false` in the artifact, a printed warning, and `model_type` never set to `nomic_bert` |
| **Schema bump breaks existing containers** | Containers are derived artifacts; re-encode is deterministic and byte-exact; the refusal message names the schema mismatch |
| **`auto_map` absence breaks naive HF loading** | documented in the README section; the ONNX tier (§7.3) is the portable answer |
| **Embedding quality of a derived causal export assumed good** | Phase E measures MRL/retrieval quality on a held-out set and reports it; an unmeasured artifact is labelled unmeasured |

---

## 13. Open questions

**Q1 — the `Wqkv` row order (block-major vs head-interleaved).** *Unverified.* I read the tensor
shape from the safetensors header but did not retrieve
`modeling_hf_nomic_bert.py`, which is where the reshape lives. This must be settled in Phase A by
either reading that file or by the §10.2 parity gate. **Highest-risk open item.**

**Q2 — what `target.final_norm` binds to.** nomic has no post-stack norm; layer 11's `norm2` *is*
the final norm under `PostOnly`. `Planner` requires a `final_norm` object, so the binding must be
recorded rather than a tensor invented. Provisional: bind it to `11.norm2` and mark the binding
shared, so export does not emit a duplicate tensor. Needs a decision, because a duplicated
`norm2` in the export would break the byte-exact gate.

**Q3 — the exact `eps` of the Matryoshka `F.layer_norm`.** The model card snippet shows
`F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))` with no `eps` argument, i.e.
PyTorch's default **1e-5** — which differs from the checkpoint's 1e-12. Provisional: 1e-5,
recorded in `PoolingSurface.PreTruncationNorm` so it is a fact and not a constant in code.
Confirmed by §10.2 either way.

**Q4 — the tokenizer's lowercasing.** `sentence_bert_config.json` says `do_lower_case: false`
while the underlying tokenizer is BERT-uncased. Reading: ST does not lowercase *additionally*
because the HF tokenizer's normalizer already does. Verify by inspecting `tokenizer.json`'s
normalizer list at ingest and record the fact.

**Q5 — should `[CLS]`-pooled or last-token variants be supported for nomic?**
`1_Pooling/config.json` says mean and only mean, so the ingest records mean. But the same
container could legitimately be *served* with a different pooling by a consumer. Provisional:
`embed --pooling mean|cls|last` overrides at serve time, is **recorded in the output metadata**,
and defaults to what the container says. Serving a different pooling than the model was trained
with is a quality question, not a correctness one, so it is allowed but labelled.

**Q6 — ONNX / GGUF tiers.** nomic ships nine ONNX variants, which is strong evidence about what
consumers actually deploy. Deferred (§7.3); revisit after Phase E with measured demand.

**Q7 — other embedding families.** The design is nomic-first but the schema (§4) is
family-neutral: `EncoderConfig` is where per-family judgements live, and BERT/RoBERTa (learned
positions, pre-LN, GELU, `[CLS]` pooling), GTE/E5 (XLM-RoBERTa lineage) and modern decoder-based
embedders (last-token pooling, causal) each need their own facts reader. Nothing in §4 blocks
them; `PositionUnresolved` and the carried-but-refused pooling kinds are exactly the mechanism
for admitting a family's facts before its kernels exist.

**Q8 — do we want an embedding *index* artifact?** The user's framing ("generate an embedding
for a chunk of content") suggests retrieval use. A VIndex3-native vector index (store, search,
incremental add) is a natural follow-on and would reuse `embed --jsonl` as its producer.
Explicitly out of scope here; noted so the CLI's output formats stay index-friendly.

---

## 14. Deliverable shape

- `docs/embedding-models-nomic-embed-text.md` (this design).
- `src/Amql.Hf/EncoderConfig.cs` — the encoder facts reader and the nomic judgements/refusals.
- `src/Amql.Inference/EmbedSession.cs` — encode → pool → vector.
- `src/Amql.Cli/EmbedCommands.cs` — the `embed` command.
- Schema additions in `Amql.Vindex3` (`EncoderStack`, `Directionality`, `HasBias`, `PostOnly`,
  `EmbeddingSurface`, `PoolingSurface`) at schema 7.
- Encoder ingest in `ArchMapper` (including the `QkvLayout` split) and the two export tiers in
  `ModelExporter` / `ExportConfig`.
- Tests for §10.1–§10.4 and a README section with measured numbers.

The CPU stays the byte-exact reference. Nothing on this path trains, nothing defaults, and an
unjudged fact refuses by name — the same contract the rest of AMQL already keeps.
