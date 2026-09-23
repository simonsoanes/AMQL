# Classifier Models in AMQL — sequence-classification ("Jev") support

**Status:** Design proposal
**Author:** AMQL / Ariadne
**Date:** 2026-09-23

## 1. Purpose

AMQL's pipeline is now generative (`encode` a decoder, transform, `generate` tokens, `export`)
and — fresh from the embedding work — bidirectional-encoder (nomic-embed-text-v1.5: `embed`,
mean pooling, no head). This document adds the third family: **classifiers**, in the shape
popularised by the "Jev" line of models and by standard HF *sequence classification*.

A Jev-style model is the pragmatic union of the two things AMQL already handles, in the **causal**
orientation with a small **linear score head**:

- It is a **decoder** (causal, Qwen3.5 or similar), so the existing `DecoderStack`,
  `DecodeSession`/prefill, RoPE partial-rotary, GatedDelta linear-attention and MoE surfaces all
  apply **unchanged**.
- But it has **no `lm_head` over the vocabulary**. Instead, after the last token's hidden state is
  pooled, a `score` linear maps `hidden_size → num_labels`. Output is a **label distribution**, not
  a next-token distribution.

The concrete reference is `AlexWortega/openjev` (`qwen3.5-4b-nli`), whose `config.json` and
`modeling_openjev.py` were read on 2026-09-23:

| Property | Value (from the real checkpoint) |
|---|---|
| `architectures` | `["Qwen3_5ForSequenceClassification"]` |
| `model_type` | `qwen3_5` (text stack `qwen3_5_text`) |
| `problem_type` | `single_label_classification` |
| labels | `id2label`: `0→contradiction, 1→entailment, 2→neutral`; `label2id` inverse |
| template | `nli_template`: `"Premise: {premise}\nHypothesis: {hypothesis}"` |
| tokens | `pad_token_id` / `eos_token_id` = 248044, `image_token_id` 248056, vision tokens |
| pooling | **last non-pad token** (`attention_mask.sum(1) - 1`), not mean, not `[CLS]` |
| head | **`score`** linear: `model.score.weight` is `[num_labels, hidden_size]` |
| loss | cross-entropy over the three classes (plain CE for the NLI head; soft-BCE for the latent MLP heads) |
| backbone | standard Qwen3.5 text config: 32 layers, hidden 2560, 3-of-4 `linear_attention`, tied embeddings |

**What AMQL must, must not, and may do:**

- **Must** be able to *create a container* from such a checkpoint (`encode`), *inspect* it
  (`inspect` / new `classify`), and *export* it back out (`export`).
- **Must not** claim to represent the raw `score` computation unless it really can. The backbone is
  fully expressible today; the missing piece is one **`ClassifierSurface`** carrying the score-head
  geometry, the label table, the pooling rule and the template.
- **May** (phases) offer the `classify` serve command that applies the template, pools the last
  non-pad token, runs the score head and emits a probability distribution.

This design mirrors the embedding-models document's structure and reuses its schema machinery
(`EncoderStack`, `PoolingSurface`) where the two families genuinely overlap, while making the
**causal / last-token / linear-head** differences explicit.

---

## 2. What a Jev-style classifier actually is (researched)

### 2.1 The backbone is a plain causal decoder

Everything in `qwen3.5-4b-nli/`'s `text_config` is a normal Qwen3.5 decoder — the exact family AMQL
already encodes, serves and transforms:

```
layer_types: 32 rows of (linear_attention ×3, full_attention ×1) repeating  — the 3-of-4 GDN pattern
head_dim 256, num_attention_heads 16, num_key_value_heads 4, hidden 2560, intermediate 9216
partial_rotary_factor 0.25, rope_theta 10_000_000, mrope_interleaved true
linear_conv_kernel_dim 4, linear_key_head_dim 128, linear_key_heads 16,
linear_value_head_dim 128, linear_value_heads 32
tie_word_embeddings true, rms_norm_eps 1e-6
attention_bias false, attn_output_gate true
```

There is **no architectural novelty in the backbone**. The only structural difference from the
models AMQL already handles is **at the head** — the piece that reads `text_config` ends and a
`score` projector begins. This is why the design is cheap: the costly parts (decoder ingest,
linear-attention execution, RoPE, MoE, prefill) are all present and tested.

### 2.2 The classifier head is a single linear + a pooling rule

From `modeling_openjev.py`:

```python
def _pooled(self, enc):
    h = self.backbone(**enc).last_hidden_state            # [B, T, hidden]
    last = enc["attention_mask"].sum(1) - 1               # index of last non-pad token
    return h[torch.arange(h.shape[0]), last]              # [B, hidden]

logits = self.model.score(self._pooled(...))               # score: hidden → 3
probs  = torch.softmax(logits, -1)                         # [contradiction, entailment, neutral]
```

Three facts are load-bearing:

1. **Pooling = last *non-pad* token** — `attention_mask.sum(1) - 1`, i.e. the position of the final
   real token in a right-padded batch. That is *not* the same as "the last row of the sequence"
   when batching pads; the mask is required.
2. **No `lm_head`, no tied output projection reused as logits.** The head is its own `score`
   tensor (`model.score.weight`, `[num_labels, hidden]`). AMQL's `OutputOp` / `HeadSurface` is for
   a vocabulary decoder; a classifier head is a *different* object kind, not a degenerate output
   head.
3. **The template is applied before tokenisation.** `nli_template.format(premise=…,
   hypothesis=…)` turns a pair into one string. The template is a **recorded config fact**, not a
   hard-coded constant, because it is task-specific and the container must carry it.

### 2.3 The configuration surface

`config.json` carries the classification-relevant facts at the **top level** (`id2label`,
`label2id`, `nli_template`, `problem_type`, `pad_token_id`) alongside the ordinary `text_config`.
So a classifier's graph must record, per component, an additional surface that captures these
"above the stack" facts without disturbing the decoder surface.

### 2.4 Contrast with the embedding family

| Aspect | nomic-embed (done) | Jev classifier (this design) |
|---|---|---|
| Attention | bidirectional | **causal** (decoder) |
| Pooling | mean over non-pad, incl. specials | **last non-pad token** |
| Output | headless (L2 vector) | **linear `score` → label logits** |
| Template | task prefix (`search_query:`) | **`nli_template` pair format** |
| Object kind | `EncoderStack` | **`DecoderStack` + `ClassifierHead`** |
| Execution surface | `EmbeddingSurface`, `PoolingSurface` | **`ClassifierSurface`** (additive) |

The two share the *idea* of "pool the hidden states and project; there is no auto-regressive
head", but every operator differs. The design therefore reuses the embedding work's *schema
mechanics* (a new `*Surface`, carried metadata, refuse-by-name) rather than its *assumptions*.

---

## 3. AMQL today: what a classifier needs vs what exists

### 3.1 Reusable as-is (the whole decoder half)

| Need | Present |
|---|---|
| Causal Qwen3.5 decoder ingest (`layer_types`, GDN, partial RoPE, tied embeddings) | full `ModelConfig.ReadTextFacts` + `ArchMapper` decoder path |
| GatedDeltaNet run | `GatedDeltaKernel`, `linear_attention` operator, `DecodeSession` position-major prefill |
| Full-attention layers | `AttentionKernel.Execute` (causal) |
| MoE / routed FFN | `RoutedFfnOp`, `MoeSurface` |
| Prefill over an input sequence (batch or single) | `DecodeSession.Prefill` |
| Export of the backbone | `ModelExporter` + `ExportConfig` (Qwen3.x names) |
| Refuse-by-name discipline | `ModelConfigException`, `UnsupportedOperatorException`, planner operand-closure |

A `Qwen3_5ForSequenceClassification` container is **the same container AMQL already produces** for
`Qwen3_5ForCausalLM`, plus one extra logical object and one extra surface. If the ingest ignores
the classification facts today and encodes only the decoder, it would already *run* as a generator
(through the tied `lm_head`), but it would be **wrong as a classifier** and the classification
metadata would be silently dropped. The design's job is to stop that silent drop.

### 3.2 The gaps

| # | Gap | Where | Why it blocks a classifier |
|---|---|---|---|
| C1 | **No classification head object kind.** `ObjectKind` has no `ClassifierHead`/`ScoreHead`; the graph can only express a `DecoderStack` + `OutputHead` | `Amql.Vindex3/GraphModel.cs` | a `score` linear is neither an `OutputHead` (vocabulary) nor nothing |
| C2 | **No representation of the score tensor.** `model.score.weight` `[num_labels, hidden]` has no binding | `ArchMapper`, `ModelToContainer` | without it, ingest would have to *throw away* the head weights — data loss |
| C3 | **No last-non-pad pooling rule.** `PoolingSurface` (embedding work) has `Mean`/`Cls`/`Last` etc. but not the *mask-aware* last-token rule (`attention_mask.sum−1`) | `Amql.Vindex3/Surface.cs` | a right-padded batch needs the mask to find the true last token |
| C4 | **No label table / template / problem_type in the graph.** The top-level `id2label`, `label2id`, `nli_template`, `problem_type` facts have no home | — | export could not regenerate them; `classify` would not know the output labels |
| C5 | **Ingest refuses or drops the top-level facts.** `ReadTextFacts` reads only `text_config`-level facts; the top-level classifier keys are ignored | `ModelConfig.ReadTextFacts` | silent loss of the very facts that make it a classifier |
| C6 | **No `classify` serve command.** `generate`, `embed` exist; a "run the score head" path does not | `Amql.Cli` | nothing applies the template + pooling + linear |
| C7 | **Export doesn't know the head.** `ExportConfig.BuildJson` rebuilds a decoder `config.json`; it has no classifier branch | `ExportConfig` | a classifier exporting as a causal LM would load but classify wrong — the exact failure the repo forbids |

C1–C4 are the same "carry the fact, refuse rather than approximate" pattern used for encoder and
embedding surfaces. C5–C7 are the user-facing surface (create / inspect / export / serve).

---

## 4. Schema: representing a classifier in VIndex3

Additive, schema **6 → 7** — the same bump the embedding/encoder work already needs, so the two
designs land on one graph version. Nothing a non-classifier container carries changes.

### 4.1 `ObjectKind.ClassifierHead` (new)

```csharp
public enum ObjectKind {
    Embedding,
    DecoderStack,
    FinalNorm,
    OutputHead,
    PerceptionTower,
    PerceptionAdapter,
    FeatureProjector,
    ExpertBank,
    EncoderStack,        // (embedding/encoder work, schema 7)
    ClassifierHead,      // NEW — score linear over pooled hidden state
}
```

A `ClassifierHead` object owns exactly one tensor (`score.weight`, `[num_labels, hidden]`) with a
source binding to the checkpoint's `model.score.weight`. It is materialised (has a
representation) whenever the source checkpoint carries it, so export is lossless.

### 4.2 `PoolingRule` — extend for mask-aware last-token

The embedding work's `PoolingSurface.PoolingKind` carries `Mean, MeanSqrtLen, Max, Cls, Last,
WeightedMean`. Jev needs `Last` with the mask semantics spelled out. Rather than silently
overload `Last`, add a field:

```csharp
public sealed class PoolingSurface {
    public required PoolingKind Kind { get; init; }
    // NEW: for `Last`, the index is attention_mask.sum(seq)-1 — the last NON-PAD token,
    // not the final row. Required when Kind == Last on a causal classifier; ignored for Mean.
    public bool LastNonPad { get; init; } = false;
    public bool MaskPadding { get; init; } = true;
    public bool IncludeSpecialTokens { get; init; } = true;
    public NormSpec? PreTruncationNorm { get; init; }
    public IReadOnlyList<int>? MatryoshkaDims { get; init; }
    public bool L2Normalise { get; init; }
}
```

For a Jev classifier: `Kind = Last`, `LastNonPad = true`, `MaskPadding = true`,
`IncludeSpecialTokens = false` (the last real token is the decision token). The embedding
`Mean` path is untouched; `Last` with `LastNonPad=false` remains the row-`T-1` pooling for
decoder cases where that is actually wanted.

> **Why a field and not a new enum value?** `Last` poolable is one operation; the *selector* —
> "which row is 'last'" — is a separate, orthogonal fact (row `T-1` vs mask-aware `Σmask−1`).
> Overloading the enum with `LastMasked` would bake a pairing that the export/round-trip then has
> to reconstruct from two facts anyway. Both variants must be representable because some decoder
> embedders genuinely use plain last-row pooling even in batches (no padding), while NLI
> cross-encoders pad and need mask-awareness.

### 4.3 `ClassifierSurface` (new) — the head's full contract

```csharp
public sealed class ClassifierSurface {
    public required int NumLabels { get; init; }
    public required string ProblemType { get; init; }        // "single_label_classification" | allowed-kind
    public required PoolingSurface Pooling { get; init; }    // Kind=Last, LastNonPad=true
    // Label table + template as carried JSON so round-trip never invents a mapping.
    public required JsonElement Id2Label { get; init; }      // {"0":"contradiction", ...}
    public required JsonElement Label2Id { get; init; }
    public string? Template { get; init; }                   // "Premise: {premise}\\nHypothesis: {hypothesis}"
    public bool ScoreHeadReusesEmbeddingLayout { get; init; } // false for a dedicated score tensor
}
```

Components:

- `NumLabels` is the width of the `score` tensor's row 0 (judged from the tensor, never from
  `id2label` alone — the tensor is the authority and the map must agree with it).
- `ProblemType` is inverted from `problem_type`; this build **serves** `single_label_classification`
  (softmax over logits) and `single_label_regression` (scalar) and **carries** anything else
  (`multi_label_classification`, `regression`) verbatim until a kernel for it is judged.
- `Template` is carried so `classify` can apply it; a missing template is a carrier/annotation-only
  classifier, and `classify` then refuses with "the container records no input template".

`ExecutionSurface` gains an optional `Classifier` member, exactly like `Head` is optional:

```csharp
public sealed class ExecutionSurface {
    ...
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ClassifierSurface? Classifier { get; init; }
}
```

A `ClassifierHead` object **ownership contract**: a component that owns a `ClassifierHead` must
have a `Classifier` surface; a component with none must have none. `Planner` enforces this and
refuses a mismatch by name (the operand-closure pattern). A component may own a `DecoderStack`
**and** a `ClassifierHead` — that is the Jev shape — or an `EncoderStack` and a `ClassifierHead`
(BERT-style classifier, future), but never both a `ClassifierHead` and a vocabulary `OutputHead`
(labels are not tokens; the planner refuses that combto as a defect).

---

## 5. Create: `encode` a classifier checkpoint

### 5.1 Dispatch and fact-carriage

The decoder-half fact reader stays exactly as it is (Qwen3.5 `text_config` already satisfies it).
Above it, `ReadTextFacts` (or a thin wrapper) additionally lifts the **top-level classifier keys**
that the decoder path ignores today:

```csharp
public sealed record ClassificationFacts(
    string? Architecture,              // "Qwen3_5ForSequenceClassification"
    string ProblemType,
    int NumLabels,
    JsonElement Id2Label,
    JsonElement Label2Id,
    string? Template,
    bool HasScoreTensor);              // inventory has model.score.weight
```

Dispatch rule in `ModelToContainer.Encode`:

- `architectures` contains `…ForSequenceClassification` (or `README`/config declares a classifier
  head) **and** `model.score.weight` is in the inventory → **classifier ingest**: build the full
  decoder container **and** a `ClassifierHead` object + `ClassifierSurface`. Nothing about the
  decoder changes; this is purely additive.
- A `…ForSequenceClassification` config whose `score.weight` is **absent** → a defect. The
  checkpoint claims a head it does not ship; refuse by name (same discipline as a missing
  `lm_head` when `tie_word_embeddings:false`).
- No classifier marker → ordinary decoder ingest, unchanged.

### 5.2 The objects and the score binding

```csharp
var objects = new List<LogicalObject> {
    TextObject("target.embedding", ObjectKind.Embedding, prefix, "embed_tokens", encoding),
    TextObject("target.decoder_stack", ObjectKind.DecoderStack, prefix, "layers", encoding),
    TextObject("target.final_norm", ObjectKind.FinalNorm, prefix, "norm", encoding),
    // head: tied lm_head is *not* created as a separate object (reuses embedding), as today.
};
// classifier head rides its own object + segment:
if (classification is { HasScoreTensor: true }) {
    objects.Add(new LogicalObject {
        Id = "target.classifier_head",
        Component = "target",
        Kind = ObjectKind.ClassifierHead,
        SourceBindings = { new() { Artifact = prefix, TensorPrefix = "score", Tensors = 1, Bytes = 0 } },
        Representations = { new() { Encoding = encoding, Fidelity = Fidelity.Canonical } },
    });
}
```

Representation binding mirrors `BindOne`:

```csharp
Rep("target.classifier_head", encoding, BindOne(inventory, prefix, "score.weight"));
```

Note: the classifier head is **un-tied by construction** — it is its own tensor. There is no
`tie` concept for a score head. The existing `BindOutputHead` is for `lm_head`; this is a
separate `BindScoreHead` that requires exactly `model.score.weight`.

### 5.3 The container afterwards

```
<container>/
  index.json             family "qwen3_5", encoding "BF16" (or stored dtype), schema 7
  system_graph.json      component "target" (PrimaryText, DecoderStack, 32 layers, hidden 2560)
                         objects: target.embedding, target.decoder_stack, target.final_norm,
                                  target.classifier_head
                         execution.Classifier: { NumLabels 3, problem_type single_label_classification,
                               Pooling {Last, LastNonPad=true}, id2label/label2id, template }
                         decoder surface: unchanged from a Qwen3.5 causal LM
  .segments/*.bin        classifier_head segment holds `weight` ([3, 2560])
  tokenizer.json         copied verbatim
  classifier.json        NEW: carried top-level facts (id2label, label2id, nli_template,
                         problem_type, pad_token_id) for lossless export — the graph already
                         has them, this is a convenience mirror
```

`target.final_norm` binds the same `norm.weight` as a causal Qwen3.5 — the backbone is a normal
decoder, and the `score` head reads the **post-final-norm, pooled** last-token hidden state.

### 5.4 Refusals on ingest

| Condition | Action |
|---|---|
| `model.score.weight` present but shape `[·,·]` ≠ `[num_labels, hidden]` | refuse, naming both shapes |
| `…ForSequenceClassification` but no `score.weight` | refuse: claim without a head |
| `id2label` keys don't agree with `label2id` / `NumLabels` | refuse: the map is not self-consistent |
| unknown `problem_type` | carry verbatim; a serve/export that needs a softmax refuses until a kernel is judged |
| `score.weight` dtype not re-encodable, under a patch | refuse at export like any un-encoding dtype |

---

## 6. Inspect: understanding a classifier container

`inspect-token` today inspects embedded rows and (optionally) logits over the vocabulary. A
classifier needs two tools:

### 6.1 `inspect` gains a classifier report

`amql-cli inspect <container> --classifier` prints:

- the `ClassifierSurface`: `NumLabels`, `ProblemType`, `Pooling {Kind, LastNonPad}`;
- the label table (`id2label`), so the user reads `"entailment"` not `1`;
- the recorded `Template`;
- the score tensor profile: shape `[num_labels, hidden]`, stored dtype, min/max/mean/norm per row
  (reusing the embedding-profile machinery for the per-row numeric profile);
- a note if the pooling rule is unsupported (not `LastNonPad`), so it is inspected, not silently
  served.

This reuses `TokenInspector`'s shape/profile/report pattern but over the `ClassifierHead` object
instead of the embedding — no new kernel, only a new reader.

### 6.2 `--label` and `--score` on existing inspect surfaces

Where today's `inspect-token --logits` reports vocabulary ranks, a classifier's equivalent is
**score logits over labels**. `inspect-token <container> <idx> --classifier-score --tokens <ids>`
runs the pooled last-token state through the `score` head and prints, per label:
`contradiction  -1.234 / softmax 0.18`, etc. (Softmax is judged for `single_label_classification`;
anything else refuses.)

---

## 7. Export: a classifier container back to a checkpoint

The export must regenerate a **classifier** `config.json`, not a causal-LM one. `ExportConfig`
gains a classifier branch:

### 7.1 `config.json` regeneration

| `config.json` key | Source |
|---|---|
| `architectures` | `["Qwen3_5ForSequenceClassification"]` |
| `model_type`, `text_config` (all decoder facts) | unchanged decoder writer — the backbone is identical |
| `id2label`, `label2id`, `problem_type`, `nli_template`, `pad_token_id` | **from `ClassifierSurface`** (carried, never invented) |
| `score_head` metadata | nothing beyond the tensor itself + `problem_type` |

### 7.2 Tensors

- The **decoder** tensors export under the exact Qwen3.x names today's exporter already rebuilds
  (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`, tied `lm_head` skipped as
  it reuses the embedding).
- The **classifier head** exports as `model.score.weight` from the `ClassifierHead` segment —

  this is the only tensor the existing exporter would otherwise drop, which is the whole reason a
  classifier needs its own object/segment rather than riding `OutputHead`.

So `export` of a classifier container is: **today's decoder export, plus one tensor and a
classifier config branch**. MXFP4/patching compose unchanged; `score.weight` follows the same
rule as any 2-D per-layer weight.

### 7.3 Round-trip gate

`encode` the openjev checkpoint → `export` → re-`encode` the result → the second container is
**byte-exact** (or documented tolerance-bounded for quantised export) with the first, including the
`classifier_head` object and its `score.weight`. This is the same strongest-possible statement the
embedding and Flash-Next designs gate on.

---

## 8. Serve: the `classify` command

This is the user-visible payoff and the direct analogue of `amql-cli embed`:

```
amql-cli classify <container> [--component target]
                  (--text "premise|hypothesis" | --premise "…" --hypothesis "…"
                   | --file <path> | --stdin | --jsonl <path>)
                  [--format labels|json|jsonl|csv]
                  [--batch-size 32] [--patch <p.safetensors>]
```

Behaviour, per the recorded facts (never inferred):

1. **Apply the template.** Read `ClassifierSurface.Template` and format `{premise}` / `{hypothesis}`
   (single-string form `--text "p|h"` is split on the first recorded separator; explicit
   `--premise`/`--hypothesis` avoids ambiguity). Refuse if no template is recorded.
2. **Tokenise** with the container's own `tokenizer.json`, right-padded across the batch.
3. **Encode + pool.** Run the decoder prefill (batched, position-major because linear layers are
   stateful), compute the last non-pad token index from the mask, take that hidden row, apply the
   final norm.
4. **Score.** `logits = score.pooled` ([batch, num_labels]); softmax for
   `single_label_classification`.
5. **Emit.** Default `--format labels` prints the argmax label name from `id2label`;
   `--format json` emits `{premise, hypothesis, labels:[…], probabilities:[…], argmax:"entailment"}` —
   self-describing like the embedding output; `--jsonl` for batches, `--csv` for tables.

`--format json` output includes the pooling rule actually used and the applied template inputs, so
a stored classification can never be silently misread as something computed a different way. The
`--patch` seam and `--batch-size` (mask-aware padding) thread through like `embed`.

Refusals: unknown labels in `--text`, an unsupported `problem_type` for the serve (only
`single_label_classification` runs a softmax), a component that owns no `ClassifierHead` — each
names the missing/mismatched fact.

---

## 9. New and changed files

| File | Change |
|---|---|
| `src/Amql.Vindex3/GraphModel.cs` | `ObjectKind.ClassifierHead`; `CurrentSchema` 6 → 7 (shared with encoder work) |
| `src/Amql.Vindex3/Surface.cs` | `ClassifierSurface`; `PoolingSurface.LastNonPad`; `ExecutionSurface.Classifier` |
| `src/Amql.Hf/ModelConfig.cs` | lift top-level `ClassificationFacts` (id2label/label2id/problem_type/template) |
| `src/Amql.Hf/ArchMapper.cs` | `BindScoreHead`; build `ClassifierHead` object + `ClassifierSurface`; the §5.4 refusals |
| `src/Amql.Hf/ModelToContainer.cs` | classifier dispatch; write `classifier.json` mirror |
| `src/Amql.Inference/Plan.cs` | `ClassifierOp { Score, Pooling, ProblemType }` on `ComponentOpPlan` |
| `src/Amql.Inference/Planner.cs` | enforce the head-ownership contract; bind `score.weight` and the pooling rule |
| `src/Amql.Inference/GenericRuntime.cs` | pooled last-token extraction + `score` projection; keep the mask |
| `src/Amql.Cli/ClassifyCommands.cs` (new) | the `classify` command (§8) |
| `src/Amql.Cli/ModelExporter.cs` + `ExportConfig.cs` | classifier `config.json` branch; emit `model.score.weight`; round-trip gate |
| `src/Amql.Cli/Program.cs` | `"classify" => Classify(args[1..])`; `inspect --classifier` and `--classifier-score` |
| `README.md`, `docs/` | classifier section with measured numbers |

---

## 10. Validation gates

1. **Round-trip.** `encode` openjev (`qwen3.5-4b-nli`) → `export` → re-`encode` → container
   byte-exact, including `classifier_head` + `score.weight`.
2. **Numeral parity.** Same premise/hypothesis through `classify --format json` and the reference
   `OpenJevCrossEncoder.predict`: labels identical; softmax probs within 1e-4 (F32, same arithmetic
   order). Requires the local runtime; marked skip in offline builds.
3. **Mask correctness.** A single text prefill == the same text inside a padded batch (bit-identical
   pooled row) — proves `LastNonPad` selects the same token.
4. **Arginvs.** `inspect --classifier` reports `entailment` (name) not `1`; a component with no
   head refuses `classify`; a `multi_label_classification` refuses a softmax serve.
5. **Full suite green.** All existing CPU tests stay green after the schema bump; re-encoded
   decoder containers `verify` byte-exact.

---

## 11. Phases

**Phase A — schema + ingest + export (container lifecycle).** `ClassifierHead`,
`ClassifierSurface`, `PoolingSurface.LastNonPad`, ingest binding, export branch, round-trip gate.
*Gate:* §10.1 round-trip green; `inspect --classifier` prints the label table; export regenerates
a `Qwen3_5ForSequenceClassification` config.

**Phase B — serve + inspect logits.** `classify` (template → pool → score → softmax), the pooled
last-token path, `--classifier-score` on inspect, refusals for unsupported problem types.
*Gate:* §10.2 parity, §10.3 mask invariance.

**Phase C — breadth + hardening.** Broad checks (BERT-style `EncoderStackClassifier` on the schema,
multi-label/regression carried-and-refused, template variants), MXFP4 on `score.weight`, README
numbers, quantified latency/throughput on the demo NLI checkpoint.
*Gate:* full suite green, measured serve latency in the README.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pooling picks the wrong row (off-by-one, or last-row-not-last-non-pad) | `LastNonPad` is a recorded, required fact; §10.3 invariants force bit-identical alone-vs-padded; §10.2 parity with the reference |
| Score-head dropped on export (data loss) | `ClassifierHead` is its own object/segment; export emits `model.score.weight`; §10.1 round-trip byte-exact catches a drop immediately |
| Classifier exported as a causal LM → loads but classifies wrong | `ExportConfig` classifier branch forces `architectures=[…ForSequenceClassification]` and refuses a headless-vs-headed mismatch |
| `id2label`/`label2id` disagree with the `score` tensor | `NumLabels` judged from the tensor; map cross-checked; §5.4 refusal |
| Softmax served for an unjudged problem type | only `single_label_classification` runs a softmax; others carried and refused |
| Template loss on export | `Template` carried in `ClassifierSurface`; `classifier.json` mirror |
| Schema bump regression | containers are derived; re-encode deterministic and byte-exact; refusal names the mismatch |

---

## 13. Open questions

1. **Score-head tensor names beyond openjev.** `model.score.weight` / `model.score.bias` is the
   spine of HF `ForSequenceClassification`, but some classifiers name the head differently
   (`classifier.weight`, `head.weight`, or a BiLSTM). The design binds **`score`** as the initial
   judgement; other spellings are a `BindScoreHead`-table extension, not a schema change. Confirm
   by checking one or two more checkpoints in Phase A.
2. **`score.bias` presence.** openjev's head is `Linear(hidden, num_labels)` which by default has a
   bias; the safetensors index was not retrieved (404'd for a single-shard file), so whether a
   `score.bias` tensor ships is **unverified**. The schema's `ClassifierSurface` should carry a
   `HasBiаs`-style flag (via the existing selection pattern) and the binding should probe for
   `score.bias` / `score.weight` independently. The tensor is the authority; a missing probe
   simply means "no bias", a present one is bound.
3. **Multi-label / regression serve.** Out of scope for the initial `classify` (sigmoid path is
   a separate kernel); the facts are carried in the graph so a later phase is purely additive.
4. **Template separator for `--text "p|h"`.** The model card uses `{premise}` / `{hypothesis}`
   placeholders; a single-string CLI is a convenience split on a recorded separator. Decide the
   default (e.g. first `\n` or `|`) in Phase B and document it; explicit flags remain the
   unambiguous path.
5. **Do we serve a `LatentMLPHead`?** The 35B variant trains per-task MLP heads
   (`d → 512 → 1`, GELU, dropout, soft-BCE) on a frozen cross-encoder latent. This is a
   *trained* head, not a checkpoint tensor — explicitly out of scope (no training on AMQL's
   text path). Noted so the pooled-latent escape hatch (`latents`, `latents_hypotheses`) is
   recorded as an export/serve facility but never trained.

---

## 14. Deliverable shape

- `docs/classifier-models-jev.md` (this design).
- `Amql.Vindex3`: `ObjectKind.ClassifierHead`, `ClassifierSurface`, `PoolingSurface.LastNonPad`,
  `ExecutionSurface.Classifier` — schema 7.
- `Amql.Hf`: top-level `ClassificationFacts` reader, classifier dispatch + `BindScoreHead`, the
  `classifier.json` mirror.
- `Amql.Inference`: `ClassifierOp` and the pooled-last-token + score runtime.
- `Amql.Cli`: `classify` command, `inspect --classifier` / `--classifier-score`, exporter +
  export-config classifier branches.
- Tests for §10 and a README section with measured numbers.

The backbone needs nothing new — a Jev classifier is a Qwen3.5 decoder plus one linear head and
one pooling rule. The design's entire job is to **carry the classifier facts through create →
inspect → export (→ serve) without ever letting the score head, the label table, or the
pooling rule be silently reshaped into a generator.** The CPU stays the byte-exact reference; an
unjudged problem type or an absent head refuses by name.