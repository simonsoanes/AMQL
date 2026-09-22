# Exporting AMQL Containers to the Qwen3.8-Flash-Next Inference Format

**Status:** Design proposal
**Author:** AMQL / Ariadne
**Date:** 2026-09-22

## 1. Purpose

This document proposes a new export target for AMQL: take a VIndex3 container (the output of
`encode`, `merge`, `prune`, `moe-ify`, `generate-mtp`, or any composition thereof) and
materialise it as a **checkpoint that loads in the same inference stack as
Qwen3.8-Flash-Next** — the HF Transformers / vLLM / SGLang / TokenSpeed family, with GGUF
for llama.cpp as a secondary tier. The goal is a **cost- and time-efficient instance** of
the new architecture, derived from the AMQL model objects we already produce, with **no
retraining**.

The driving requirement: *"use the same inference format as Qwen3.8-Flash-Next."* Stated
plainly, that means:

- The **on-disk checkpoint contract** — tensor names, shapes, dtypes, `model_type`,
  `config.json` facts, and GGUF metadata — must be one the Flash-Next runtime stack already
  understands, so existing serving engines load it unmodified.
- AMQL does **not** need to reproduce Qwen3.8-Flash-Next's *weights*. It needs to make the
  **architecture it already runs** expressible in that format.

This is a design document, not an implementation. It assesses the current export surface,
identifies the target format, and lays out an incremental, measurement-gated plan with
detailed implementation guidelines.

---

## 2. What we are exporting from (current state, assessed)

### 2.1 Containers we hold today

From the repository working tree (`containers/`, `exports/`):

| Container | Shape | Notes |
|-----------|-------|-------|
| `Qwen3.5-0.8B` | 0.8B, 28 layers, hidden 2048 | hybrid: 4× linear-attention + 24× softmax |
| `demo-model` / `demo` | tiny, executable | synth + encode routine |
| `Qwen3.8-27B` | 27B dense, BF16 | base encode; 48/64 layers GatedDeltaNet |
| `Qwen3.8-27B-merged` | derived (`import`) | consensus-gated merge |
| `Qwen3.8-27B-pruned` | derived (`prune`) | 24-layer, ~6.2 GiB MXFP4 working set |
| `Qwen3.8-27B-moe` | derived (`moe-ify`) | routed MoE FFN, per-expert separable segments |
| `Qwen3.8-27B-mtp` / `-mtp-pairs` | derived (`generate-mtp`) | drafter materialised / fitted |
| `exports/Qwen3.8-27B-moe-hf` (+ `.gguf`) | exported HF / GGUF | existing `export` + `to-gguf` output (24.1 GB GGUF) |

These are the **sources**. They already encode the *hybrid linear-attention + softmax*
stack, MoE routing, and an MTP drafter — the same structural building blocks
Qwen3.8-Flash-Next uses (GatedDeltaNet + Qwen Sparse Attention + MoE). **None of them is
structured like Flash-Next today** — that is the gap this document closes.

### 2.2 The current export surface (what runs today)

Two CLI commands in `src/Amql.Cli/Program.cs`:

1. **`amql-cli export <container> --out <dir> [--patch <p.safetensors>] [--quant mxfp4]`**
   (`ModelExporter.Export`, `src/Amql.Cli/ModelExporter.cs`): the inverse of `encode`.

   | File | Contents |
   |------|----------|
   | `model.safetensors` | primary text component tensors, plus a materialised vision tower under `model.visual.*` when present |
   | `config.json` | regenerated **only from judged graph facts**; an operator without a judged `layer_types` spelling refuses the export by name |
   | `tokenizer.json` | copied from the container (encoded in at `encode` time) |
   | `mtp.safetensors` + `mtp.config.json` | automatic companion when an MTP drafter is materialised |

   Properties: HF tensor names are **rebuilt from the graph's source bindings** (a
   round-trip, not a remap); tensors a patch never touches are copied **byte-identically**,
   so an unpatched export is byte-exact; the tied output head is skipped with a note;
   `ExportFormat = "amql-export-v1"` is written as **provenance metadata** into the
   safetensors header.

2. **`amql-cli to-gguf <checkpoint-dir> --out <file.gguf>`**
   (`GgufConverter.Convert`, `src/Amql.Gguf/HfCheckpointToGguf.cs`): converts an *exported*
   HF checkpoint to GGUF v3. It already emits the `qwen35` / `qwen35moe` architectures —
   the **Qwen3-Next hybrid spelling llama.cpp accepts** — applying `A_log = -exp(A_log)`,
   norms `+1`, conv1d squeeze, tiled V-head reorder, stacked 3-D `ffn_*_exps`, transposed
   `ffn_gate_inp` router, weights as F16 with the ssm scalars kept F32, and the required
   `rope.dimension_sections`, `attention.recurrent_layers`, `full_attention_interval`
   metadata. The vision tower is skipped; the MTP drafter stays a separate companion shard.

### 2.3 Quantisation today: `--quant mxfp4`

`export --quant mxfp4` emits an **OCP microscaling** checkpoint using the MXFP4 codec in
`Amql.Safetensors`:

- Stack projections (`q/k/v/o_proj`, `gate/up/down_proj`, linear-attention projections) →
  FP4 E2M1 elements packed two-per-byte, one FP8-E8M0 scale per 32-element block.
- Each such weight becomes **two tensors**: `...weight` (dtype `FP4`) and `...weight_scale`
  (dtype `F8_E8M0`, per block, laid out per row). Dequant: `x ≈ DecodeFp4(q) × DecodeE8M0(scale)`.
- Embeddings, norms, biases, `A_log`, and the output head **keep full precision**.
- The element grid `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and
  `quantization_config.quant_method: "mxfp4"` are serialised, so the artifact is
  self-describing.

This is the same tensor split the **runtime** computes in (`AMQL_WEIGHTS=mxfp4`) and the
same working set the CUDA backend targets (FP4 tensor cores, FP32 accumulate) — export
format and engine are already aligned.

### 2.4 What the container already carries

- `index.json` — model identity, `Representations` (object@encoding → segment), `Profiles`,
  `PrecisionMap`, `Authority`.
- `system_graph.json` — components (`PrimaryText`, `Perception`, `Drafter`), logical objects
  (`Embedding`, `DecoderStack`, `FinalNorm`, `OutputHead`, `PerceptionTower`,
  `PerceptionAdapter`, `FeatureProjector`, `ExpertBank`), source bindings back to HF tensor
  names, hidden-state edges (with optional `BlockSize`).
- `.segments/*.bin` — payload plus the `ExecutionSurface`: attention geometry, norm config,
  FFN type incl. MoE routing, head, and optional operators (linear attention /
  GatedDeltaNet, KDA, MLA, Mamba2, conv-qkv, residual-in-fp32).
- Optional extra JSON: `tokenizer.json`, `token-map.json`, `mtp.*`, `prune.dropped_layers`.

**Key takeaway:** the container is already *architecture-aware and self-describing* enough
to drive a faithful export to a different on-disk convention. The exporter's job is
**translation, not invention** — except where the target architecture has structures AMQL
does not yet materialise (HyperConnection streams, the QSA indexer, the PLE n-gram
table), which must be *synthesised* or *explicitly declined* under measured rules.

### 2.5 Gaps vs. the Flash-Next target

1. **Naming / layout** — Flash-Next uses a distinct per-layer factory (GatedDeltaNet blocks,
   Qwen-Sparse-Attention blocks, routed+shared MoE), mandatory HyperConnection streams, and a
   separate PLE n-gram embedding module. Our export currently rebuilds the *original*
   Qwen3.x names.
2. **MoE topology** — Flash-Next is 512-total / 10-activated / 1-shared (≈60× sparsity,
   ~6B active). Our moe-ify defaults are small (8 experts, top-2).
3. **Attention mix** — Flash-Next is `12 × ((3 × GDN → MoE) → (1 × QSA → MoE))`. Our 27B is
   48/64 GDN with softmax interleaved. The export must carry a layer-type table the target
   `config.json` declares, and record sparse geometry.
4. **N-gram embedding memory (PLE)** — a hashed, config-sized row-shard table, not a sidecar.
   AMQL has no equivalent object; §5.4 defines what the exporter must write, and §12.1 asks
   whether we write it at all.
5. **Backend acceptance** — the export must pass our own planner's judgement gate **and** be
   accepted by HF/vLLM/SGLang via a `config.json` those backends recognise.

---

## 3. What the Qwen3.8-Flash-Next format is (target, researched)

### 3.1 Reference architecture facts

Qwen3.8-Flash-Next (released 2026-08-26) is the "Qwen 4 architecture preview". Published
facts (Qwen blog, HF model card, SGLang day-0 post, ModelScope):

| Property | Value |
|----------|-------|
| Main model params | **125B** |
| N-gram embedding params | **+51B** (near-weightless activated lookup) |
| Activated per token | **~6B** |
| Layers / hidden | **48 layers**, hidden **2048** |
| Hidden layout | `12 × (3 × (GatedDeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))` |
| Attention | hybrid **Gated DeltaNet** (linear, 3 of every 4 blocks, fixed-size state, O(1)/token) + **Qwen Sparse Attention (QSA)** for retrieval |
| MoE | **512 total / 10 activated / 1 shared** per layer (inherited from Qwen3-Next-80B-A3B) |
| Context | 1M (native 262k, extensible) |
| Distribution | **HF Transformers format**; loads in HF / vLLM / SGLang / TokenSpeed; GGUF for llama.cpp |

### 3.2 The three faces of "the inference format"

1. **HF `config.json` facts** — `model_type` / `architectures`, `hidden_size`,
   `num_hidden_layers`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`,
   `head_dim`, the per-layer `layer_types` table (hybrid linear / sparse / MoE), sparse
   attention parameters (window, sinks, block size), GatedDeltaNet conv/state geometry, MoE
   `experts` / `top_k` / `shared_experts` / `expert_intermediate_size`, rope parameters, the
   n-gram embedding block geometry, and `quantization_config` when quantised.
2. **HF safetensors tensor contract** — tensor *names* and *shapes* such that a
   Transformers/vLLM module loads them. **This is the compatibility-critical contract.**
3. **GGUF metadata** — architecture enum plus hyper-parameters. The repo already emits
   `qwen35`/`qwen35moe`; Flash-Next is the `qwen4`-family branch.

### 3.3 Why this is the right compatibility target

- AMQL already **plans and executes** the hybrid stack on the very containers we would
  export. The `ExecutionSurface` already distinguishes GatedDeltaNet from softmax layers and
  already encodes MoE routing — we are serialising an architecture we run, not guessing one.
- The existing `export` + `to-gguf` path **already proves** the hybrid → HF/GGUF translation
  works (Qwen3-Next transforms, tiled V-head reorder, F16 emission). Flash-Next is a strict
  superset of that surface.
- The **n-gram embedding** and **QSA geometry** are the only genuinely new elements, and both
  are *additive, low-cost* structures on the existing hybrid + MoE skeleton.

**Design stance (crucial):** we do **not** copy Qwen's proprietary weights. We emit a
checkpoint that *describes* the same architecture and carries **our own**
(merged/pruned/moe-ified/fitted) weights underneath Flash-Next-recognisable names. The result
is an open, locally-run instance of the same **format** — which is exactly what "compatible
with existing inference services" requires.

---

## 4. Proposed export formats

### 4.1 Two-tier output

Target **vLLM/SGLang/HF first**, **GGUF/llama.cpp second**. Do **not** build a new container
format — the VIndex3 container stays the authority.

| Tier | Output | Consumer | Build cost | Reuses |
|------|--------|----------|-----------|--------|
| **T1 (primary)** | HF checkpoint dir: `config.json`, `model.safetensors`, `tokenizer.json`, `n_gram_emb.safetensors` + `n_gram_config.json`, `mtp.*` (if drafter) | vLLM, SGLang, TokenSpeed, HF Transformers | new `--arch qwen4-next` exporter path | `ModelExporter` graph→name machinery, `ExportConfig` |
| **T2 (secondary)** | GGUF v3 (`qwen4`-family arch) | llama.cpp, LM Studio | new mapping on top of T1 | `GgufConverter` / `HfCheckpointToGguf.cs` |

T1 is where the cost and time wins live (one model served by the industry-standard engines,
MXFP4 being the memory lever). T2 is downstream convenience, batch-produced once T1 is
correct and its tensor names are grounded against a real engine load.

### 4.2 Three strictly-additive formats

All are opt-in; the existing `export` (default) and `export --quant mxfp4` remain untouched
and byte-exact.

- **Format A — `--arch qwen4-next`**: remaps tensor names and `config.json` so the output is
  recognised as the Qwen4 / Flash-Next architecture family, at a shape that matches (or is a
  cost-efficient approximation of) the target.
- **Format B — `--arch qwen4-next --quant mxfp4`**: the same remap with the MXFP4 codec
  applied along the Flash-Next tensor split. ~4× smaller than BF16 on disk and ~13% of an f32
  working set. **This is the primary cost-efficient artifact.**
- **Format C — structured export**: additionally emits the MTP/speculative-drafter companion
  (`mtp.*`) and, when enabled, the PLE n-gram table. The PLE table is **not** a separate
  addressable module — vLLM reads it as ordinary rows inside the normal safetensors shards
  (`…ple_embedding.ngram_embedding.shard_{i}.weight`), with its geometry in `config.json`.

Written artifacts for T1:

```
<out>/
  config.json                     # Flash-Next (qwen4-family) discriminant + judged facts
  model.safetensors               # text stack + (materialised) vision tower
  tokenizer.json                  # copied verbatim from the container
  # PLE n-gram rows (when --ple-layers is non-empty) live INSIDE the model shards as
  #   model.layers.{L}.ple.ple_embedding.ngram_embedding.shard_{i}.weight   — no sidecar file
  mtp.safetensors / mtp.config.json   # only when a drafter is materialised
```

### 4.3 Where the cost and time efficiency comes from

The point of targeting Flash-Next's *format* is to inherit its *compute economics* through
**structural sparsity**, not by shrinking a dense model:

- **MoE sparsity** — only the routed experts for the token are computed. A 27B-class container
  whose FFN is moe-ified into a wide expert bank spends a small constant fraction per token.
- **3-of-4 GDN layers** → O(1) per-token recurrent state instead of quadratic attention; only
  1-in-4 layers pay attention. (QSA would cut that further, but AMQL cannot express it — §5.3.)
- **MXFP4 projections** → ~13% of an f32 working set (proven in `docs/CUDA_PLAN.md`: full 27B
  ≈12.5 GiB, pruned ≈6.2 GiB), which is what lets a large-total-parameter model run in 32 GiB
  VRAM or on host RAM with unified memory.
- **N-gram embedding memory** runs at embedding-lookup cost, not transformer cost.

**Export time:** the export step is I/O-bound (quantisation, byte copy — explicitly out of the
GPU plan's scope, `docs/CUDA_PLAN.md` §1), so it is not the bottleneck. The compute-heavy
*transformation* steps (merge alignment, moe-ify clustering, MTP/n-gram fits) are exactly the
ones §5.2 of the CUDA plan targets for 10–30× speedups. The export is a pure function of an
already-prepared container, so each pipeline step is **one** pass.

---

## 5. Architecture mapping and the new pieces

### 5.1 Already expressible (no new tensors)

AMQL's `ExecutionSurface` already models every operator Flash-Next's text stack needs in
hybrid+MoE form:

- Gated DeltaNet (linear attention) — first-class operator (`GatedDeltaKernel`; `ssm_*`
  transforms already applied in the GGUF path).
- Gated / softmax attention — already emitted (`AttentionKernel.Execute`: causal, window,
  sinks, softcap).
- Dense + routed MoE FFN — already emitted (`RoutedFfnOp`, stacked per-expert tensors,
  materialised linear router).
- RoPE, RMSNorm (with the `+1` value transform), residual-in-fp32, conv-qkv.
- MTP drafter — already exported as the `mtp.safetensors` companion.

Execution on the AMQL side is unchanged; only the *serialisation target* differs.

### 5.2 The object → Flash-Next name shadow table

The system graph already names logical objects. The new export needs a per-object →
Flash-Next-tensor-name table — the inverse of `encode`'s source-binding rebuild — in a new
`Qwen4NextLayout` type. Indicative mapping:

| AMQL logical object / surface | Flash-Next checkpoint tensor name |
|---|---|
| component `target` embedding | `model.embed_tokens.weight` |
| GDN layer `N` (in_proj_qkv, in_proj_z, in_proj_a/b, A_log, dt, conv1d, out_proj, norms) | `model.layers.N.linear_attn.*` — vLLM packs `in_proj_qkv`+`in_proj_z` → `in_proj_qkvz`, so the **checkpoint keeps them separate** |
| QSA / full-attention layer `N` q/k/v (+ q/k norms) | `model.layers.N.self_attn.{q,k,v}_proj.weight`, `.q_norm`/`.k_norm` — vLLM packs q/k/v → `qkv_proj` |
| QSA indexer (**new, no AMQL equivalent**) | `model.layers.N.self_attn.indexer.index_qk_proj.*` |
| attention output | `model.layers.N.self_attn.o_proj.weight` |
| MoE router | `model.layers.N.mlp.gate.weight` |
| MoE routed experts | `model.layers.N.mlp.experts.{e}.{gate,up}_proj.weight` (vLLM packs → `gate_up_proj`) + `.down_proj.weight` |
| MoE shared expert | `model.layers.N.mlp.shared_expert.{gate,up,down}_proj.weight` (ckpt prefix confirmed as `mlp.shared_expert`) |
| Final norm | `model.norm.weight` |
| Output head (untied) | `lm_head.weight`; tied → skipped (reuses embed) |
| **Per-layer HyperConnection modules (new)** | `model.layers.N.{attn_hc,mlp_hc}.{hc_norm.weight,input_mix_weight_down.weight,input_mix_weight_up.weight,block_inject_weight.weight}` |
| **Final HyperConnection mixer (new)** | `model.hyper_connection_mixer.{hc_norm.weight,input_mix_weight_down.weight,input_mix_weight_up.weight}` |
| **PLE / n-gram block (new)** | `model.layers.{L}.ple.*` — see §5.4 |
| MTP drafter | `mtp.embed_tokens.weight`, `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight`, `mtp.layers.{i}.*`, `mtp.hyper_connection_mixer.*` (+ shared `lm_head`) |
| Multimodal wrapper | checkpoint may prefix text tensors `model.language_model.` — vLLM rewrites that to `model.` |

> **Verification rule — this table is derived from vLLM source, not from a checkpoint fixture.**
> Every name above was read out of `vllm/models/qwen4_exp/` on `main` (2026-09-22):
> `config.py`, `nvidia/model.py` (`packed_modules_map`, `_EXTRA_WEIGHTS_MAPPER`,
> `hf_to_vllm_mapper`, `ckpt_prefix="mlp.shared_expert"`), `nvidia/qsa.py`,
> `nvidia/indexer_qsa.py`, `nvidia/ple_layer.py`, `nvidia/ngram_embedding.py`,
> `nvidia/mtp.py`. **Phase A must still pin a reference `config.json` +
> `model.safetensors.index.json` from the published `Qwen/Qwen3.8-Flash-Next` Hub repo as the
> judgement fixture**, because (a) `packed_modules_map` proves the *runtime* name, and the
> checkpoint spelling is inferred from it, and (b) Transformers and SGLang may differ from
> vLLM. A name with no fixture evidence refuses the export by name rather than being
> approximated.

**Naming verdict on the earlier drafts:** `model.norm.weight` and `mlp.gate.weight` are
confirmed correct (`model.final_norm` / `mlp.router` were wrong and have been removed). The
GDN block is `linear_attn`, matching the existing Qwen3-Next spelling AMQL already emits.

The same table is the basis for the **reverse mapper** the encoder needs if Flash-Next
checkpoints are to `encode` in (round-trip), which the acceptance gate verifies.

### 5.3 Qwen Sparse Attention (QSA) — **not** window/sink attention

**Correction to the earlier drafts.** QSA is *not* a sliding-window + sink-token scheme. In
vLLM it is an **indexer-driven block-sparse attention**: a separate MQA indexer scores
compressed blocks and selects a top-k block budget per query. The config fields are
(`vllm/models/qwen4_exp/config.py`, `_QSA_CONFIG_FIELDS`):

| Field | Constraint enforced by vLLM |
|---|---|
| `indexer_n_heads` | positive |
| `indexer_kv_heads` | **must equal 1** (MQA operators require it) |
| `indexer_head_dim` | positive, and **≥ `head_dim × partial_rotary_factor`** (must cover the rotary dim) |
| `indexer_budget` | positive, divisible by `indexer_compress_ratio` |
| `indexer_compress_ratio` | positive; `indexer_budget / indexer_compress_ratio` **must be 512 or 2048** |

Layer-type spelling: `QSA_LAYER_TYPE = "qwen_sparse_attention"`, with
`ATTENTION_LAYER_TYPES = ("full_attention", "qwen_sparse_attention")`. Older checkpoints
label these layers plain `full_attention` and mark QSA by setting the indexer fields for the
whole model. `_validate_qsa_config` **raises** if any `qwen_sparse_attention` layer type is
present without all five indexer fields — so a half-specified QSA config is a hard load failure,
not a silent fallback.

**Consequence for AMQL — this is the honest position.** AMQL's `AttentionKernel` implements
causal + window + sinks + softcap. It has **no indexer**, no block-compression scoring, and no
top-k block selection. Therefore AMQL **cannot faithfully produce or execute QSA**, and the
exporter must not claim to. The design choice is:

1. **Export those layers as `full_attention`** and omit all five `indexer_*` fields. This is
   explicitly supported — it is the older-checkpoint spelling vLLM still accepts, and
   `_validate_qsa_config` returns cleanly when the fields are all absent *and* no layer is typed
   `qwen_sparse_attention`. The result loads and runs correctly; it simply does not get QSA's
   long-context cost saving.
2. **Never emit `qwen_sparse_attention` without a real indexer.** Doing so either fails config
   validation or, worse, loads and silently computes different attention.

So the earlier rule stands, with a sharper edge: **emit what the graph records; never invent
sparse geometry.** For a Flash-Next-shaped export from today's containers that means
`layer_types` entries are `linear_attention` and `full_attention` only. Implementing a real QSA
indexer in AMQL is a *runtime* feature (new operator, new `ExecutionSurface` fields, new
kernel) and is **out of scope for the export work** — recorded here as a known fidelity gap
rather than papered over.

### 5.4 N-gram embedding — it is a **PLE hashed table**, not a fitted row list

**Correction to the earlier drafts.** The earlier design proposed a bounded table keyed by the
distinct n-grams that actually occur in a fit corpus, materialised as one `n_gram_emb.weight`
shard plus a self-describing `n_gram_config.json`. vLLM's implementation is materially
different, and the difference is not cosmetic — it changes what the exporter must write.
From `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` (`Qwen4ExpNGramEmbedding`) and
`common/ple.py`:

- **It is a hashed vocabulary, not an observed-n-gram list.** Row ids are computed at
  runtime by a splitmix64-based prime-multiplier hash (`_splitmix64`,
  `_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15`, `_make_layer_multipliers`, `_make_vocab_layout`).
  There is **no n-gram → id dictionary in the checkpoint**: ids are recomputed by the engine
  from the token stream, and `ple_ngram_ids` (`.ops.ple`) does the lookup. A bounded table of
  only observed n-grams would therefore not address correctly.
- **Table size is config-derived and huge.** `ngram_vocab_size_base = 20_000_000`, padded up
  to `make_ngram_vocab_size_divisible_by = 128`. Per-row width is a *head* dim, not the model
  hidden size: `head_dim = ple_embed_dim / ngram_heads` where
  `ngram_heads = (ngram_size - 1) * heads_per_ngram` (defaults `ngram_size=3`,
  `heads_per_ngram=8` → 16 heads). The config enforces
  `ple_embed_dim % ((ngram_size-1) * heads_per_ngram) == 0`.
- **The table is stored as row shards.** Loader names are
  `…ple_embedding.ngram_embedding.shard_{i}.weight` (`shard_prefix =
  "ngram_embedding.shard_"`, `shard_text.isdigit()` → `shard_index`), reassembled by
  `PLEVocabParallelEmbedding.weight_loader(param, loaded_weight, checkpoint_start=…)` /
  `copy_ple_embedding_shard_`, which computes `PLEShardOverlap` against the TP vocab range.
  `split_ngram_parts` (default 512) controls the split.
- **Quantisation is FP8-or-unquantised, per-tensor global scale.**
  `Qwen4ExpPLEEmbeddingMethod.from_quant_config` returns the unquantised method when the
  prefix (or any `…shard_` name) is in `ignored_layers`, otherwise it *requires*
  `quant_config.is_checkpoint_fp8_serialized` and raises `NotImplementedError("Qwen4Exp PLE
  embedding only supports serialized FP8 checkpoints")`. The FP8 path registers one global
  `weight_scale` (float32, `PerTensorScaleParameter`) and rejects a checkpoint whose scale is
  still the `float32.min` sentinel. **MXFP4 is not an option for this tensor.**
- **Some buffers are optional and some weights are skipped.** `layer_multipliers`,
  `ngram_heads_offsets`, `ngram_heads_vocab_sizes` are persistent buffers copied from the
  checkpoint *if present* (otherwise recomputed). `model.py`'s `load_weights` skips names
  containing `hashstats_` or `token_lookup`, so an exporter may omit those and stay valid.
- **Placement is offloadable.** Two backends: `Qwen4ExpPLEDeviceEmbedding` (on-device) and
  `Qwen4ExpPLEPinnedHostEmbedding` (pinned host + UVA, prefetch on a side stream), chosen by
  `engram_config.cpu_offload`. The host path raises `RuntimeError("Engram CPU offload
  requires UVA support")` without UVA — note vLLM's own naming here is *Engram*, matching
  `deepseek_v41/{common,nvidia}/engram.py`.
- **It is wired through a conv, not added straight to the embedding.** PLE layers implement
  `MambaBase` with `mamba_type = SHORT_CONV`: `conv1d` is depthwise over
  `hc_hidden_size = hidden_size * hc_count`, `kernel_size = ple_conv_kernel_size` (4),
  `dilation = ngram_size`, `padding = (ple_conv_kernel_size-1) * ngram_size`, **zero-initialised
  and flagged `_no_reinit = True`**, with state shape `(hidden_size*hc_count,
  (ple_conv_kernel_size-1)*ngram_size)`. So the earlier "additive to the boundary-token
  embedding" description is wrong: the n-gram contribution enters via the PLE block's
  conv/gate path on `ple_layer_ids` layers.

**Revised AMQL position.** Keep the *calculation-first* stance — the ridge solve
(`Amql.Merge.LeastSquares.Fit`) is still how we produce values without training — but the
**addressing must be vLLM's hash**, not our own occurrence list. Concretely:

1. Port `_splitmix64` / `_make_layer_multipliers` / `_make_vocab_layout` exactly, so row ids
   agree bit-for-bit with the engine. Determinism (an existing acceptance gate) then covers
   this by construction.
2. Size the table from config (`ngram_vocab_size_base`, padded by 128), with row width
   `ple_embed_dim / ((ngram_size-1)*heads_per_ngram)`. At the defaults this is ~20M rows ×
   `head_dim` — **the dominant on-disk cost of the whole export**, which is why the FP8 path
   matters and why `ple_layer_ids = []` must remain a legal, cheap configuration.
3. Emit it as row shards under `ngram_embedding.shard_{i}.weight`, unquantised (BF16) or
   FP8 with one global `weight_scale`. **No separate `n_gram_emb.safetensors` /
   `n_gram_config.json` files** — the geometry lives in `config.json`'s text config, and the
   weights live in the normal shards.
4. **Disabled = `ple_layer_ids: []`.** That is the real off switch (`ngram_context_len`
   becomes 0, `short_conv_layer_ids` empty, no PLE modules constructed). It replaces the
   invented `enabled: false` block. The value gate (§7.7.3) still decides whether to ship it
   on.
5. Never claim this is Qwen's trained table; it is a calculated one at the correct
   addressing, and the ablation gate must prove it is ≥ neutral before it ships enabled.

### 5.5 MoE: carry the count, don't build 512 experts

The moe-ified container ships 8 experts / top-2 today; the *format* accepts any
`experts` / `top_k` / `shared_experts`. **Do not retrain or re-cluster to 512 experts.**
Cheapest first:

a. **Export the existing `experts=8, top_k=2` container as-is** with the Flash-Next config
   shape. Fully format-compatible; the engine loads it; it is simply a smaller MoE.
b. **Re-slice wider by re-clustering** (`moe-ify --experts N --top-k K`) into e.g.
   `experts=32, top_k=4` — still one forward pass, no training, reusing the balanced k-means +
   router materialisation already built. Cost scales with FFN rows sliced, **not** with expert
   count.
c. **Always emit `shared_experts=1`** (the Flash-Next convention) by materialising one
   always-on expert as the shared path — a small addition to the existing router kernel's
   export.

Net: the export **carries the 512/10/1 shape without requiring 512 experts to exist**. Build
the widest MoE that fits the time/disk budget, and always emit the shared expert so the config
is Flash-Next-shaped. Each expert remains its own segment (already true), so experts stay
separable for offload and the exporter writes `experts.{e}.*` without regrouping.

**Cost target.** "Cost- and time-efficient instance" is quantified as: total on-disk
≤ ~14–16 GiB (MXFP4), active-per-token ≈ a 6B–10B-class model (via MoE sparsity + 3-of-4 GDN),
runnable on a single 32 GiB GPU or a unified-memory host. That is the practical, runnable
instance of the new architecture produced from assets we already have.

### 5.6 MTP drafter — vLLM uses two projectors, not one

Already exported as a companion (`mtp.safetensors` + `mtp.config.json`); keep that behaviour,
but **the tensor names in the earlier drafts are wrong**. `vllm/models/qwen4_exp/nvidia/mtp.py`
constructs:

- `mtp.embed_tokens.weight` — `VocabParallelEmbedding(vocab_size, hidden_size)`
- `mtp.fc_embedding.weight` — `ColumnParallelLinear`, prefix `f"{prefix}.fc_embedding"`
- `mtp.fc_hidden.weight` — `ColumnParallelLinear`, prefix `f"{prefix}.fc_hidden"`
- `mtp.layers.{mtp_start_layer_idx + i}.*` — the drafter decoder layers
- `mtp.hyper_connection_mixer.*` and a shared `lm_head` (prefix `maybe_prefix(prefix,
  "lm_head")`); the module prefix itself is `maybe_prefix(prefix, "mtp")`

So the drafter projects the **embedding stream and the hidden stream separately** and
concatenates — it is *not* AMQL's single `fc.weight`, and not the K-projector mixture
(`fc.router.weight` + `fc.experts.{e}.weight`). The registry name is `Qwen4ExpMTP`.

**Reconciling this.** AMQL's fitted K-projector mixture is a genuinely different (and, by
measurement, better-accepted: 0.3% boot → 16.8% fitted at K=8 held-out) formulation. Two
honest options, to be chosen in Phase 3:

a. **Collapse to the vLLM contract**: fold the K-projector mixture into a single
   `fc_embedding`/`fc_hidden` pair (e.g. `fc_hidden` ← the fitted projector, `fc_embedding` ←
   identity-scaled or a second fitted block). Loads unmodified; loses the mixture.
b. **Keep the mixture and target the AMQL-native drafter shape**, documented as a deviation
   that only AMQL's own runtime consumes — and say so plainly rather than shipping a
   checkpoint that a stock engine will reject on a missing `fc_embedding`.

Option (a) is the default for anything claiming Flash-Next compatibility; (b) is an
explicitly-labelled AMQL extension.

---

## 6. Dtype / quantisation strategy

Three complementary levers:

1. **Keep the canonical container BF16 / byte-exact** (unchanged, per `docs/CUDA_PLAN.md` §4).
   `encode`, `verify`, hash guarantees and re-encode stay untouched. MXFP4 is an *export-side*
   and *runtime working-set* format, never written back to the container.
2. **Export-side MXFP4** (`--quant mxfp4`): stack projections → FP4-E2M1 + per-32 E8M0 scales
   (~27% of BF16, ~13% of f32); embeddings, norms, biases, `A_log`, and the head stay full
   precision. This is the working-set reducer that makes a large Flash-Next-shaped instance fit
   the available hardware.
3. **PLE/n-gram table is FP8-or-unquantised, never MXFP4.** vLLM's
   `Qwen4ExpPLEEmbeddingMethod.from_quant_config` accepts exactly two storage modes: unquantised
   (when the prefix or any `…shard_` name appears in `ignored_layers`), or an FP8-serialised
   checkpoint carrying **one global float32 `weight_scale`** — anything else raises
   `NotImplementedError`. So the PLE table is excluded from the MXFP4 split by construction, and
   the shared expert and the attention blocks follow the same MXFP4-or-BF16 rule as the rest of
   the stack. Routers, norms, `A_log` and the recurrent scalars (`dt`, `in_proj_a/b`, conv) stay
   F32 — exactly the split `HfCheckpointToGguf.cs` already decides.

**MXFP4 split for Format B:** extend the existing `Mxfp4.Quantize` eligibility split to the
Flash-Next tensor set — quantise all routed *and* shared expert matrices, the GDN projections
and the HyperConnection mixers; keep embeddings, the PLE table, norms, biases, `A_log` and the
output head out of it. Keep the `weight` + `weight_scale` companion convention and the
serialised grid so the artifact stays self-describing to an MXFP4-aware backend, and add the
PLE prefix to `quantization_config.ignored_layers` so the loader picks its unquantised path.

---

## 7. Implementation guidelines

Follow the repo conventions: model-family code lives in the encoder/planner layer, the runtime
stays generic, exports are byte-exact or tolerance-gated, and anything "learned" is
**calculated** (ridge least-squares for MTP, deterministic hashing for the PLE table), never
stochastically trained, and measurement-gated.
stochastically trained, and measurement-gated.

### 7.1 CLI design — `--arch`, not `--format`

```
amql-cli export <container> --out <dir> --arch qwen4-next [--quant mxfp4]
                [--patch <p.safetensors>]
                [--ple-layers <ids>] [--ple-corpus <corpus.txt>] [--no-ple]
amql-cli export --list-architectures     # qwen3.x (default), qwen4-next
```

`--arch` defaults to the existing Qwen3.x behaviour, so nothing changes for current users.

**Why `--ple-layers` and not `--ngram-n`:** `ngram_size` is not a runtime knob. It fixes
`ngram_heads = (ngram_size-1) * heads_per_ngram`, which must divide `ple_embed_dim`, and it sets
the conv dilation and the short-conv state length `(ple_conv_kernel_size-1) * ngram_size`.
Changing it invalidates any table already built, so it is **one value chosen at build time and
recorded in `config.json`** (default 3), validated before anything is written — the same
discipline as `layer_types`. The MTP `K ∈ {1,4,8}` sweep convention deliberately does not
transfer. The knob worth exposing is *which layers get PLE at all*, because `ple_layer_ids: []`
is the cheap, legal, off configuration.

**Why `--arch` and not `--format`:** `ModelExporter.ExportFormat = "amql-export-v1"` is
*provenance metadata* written into the safetensors header — it identifies the exporter, not
the target architecture. Overloading it with `--format flash-next` would conflate two
independent axes. Keep them separate: `--arch` selects the name/config layout; provenance
stays `amql-export-v1` and the chosen architecture is recorded both in `config.json` and as an
`arch` key in the safetensors metadata. `--arch` also composes cleanly with the existing
`--quant` and `--patch` flags.

### 7.2 Naming / layout module

- Add **`Qwen4NextLayout`** in `Amql.Cli` (promote to a new `Amql.Qwen4Next` project only if
  the surface grows): the single source of truth for the object→name table (§5.2) and the
  Flash-Next `config.json` writer.
- **Never guess.** An object or surface with no Flash-Next judgement refuses the export by
  name, exactly like the existing `layer_types` refusal. Unknown config fields round-trip
  verbatim (the serde-flatten rule the index already follows).
- Reuse the **pattern-probe discipline** (`route`/`path` validation style): an
  `--arch qwen4-next` export must first confirm every tensor maps and every layer's attention
  type is representable as GDN or QSA, **before writing anything**.

### 7.3 `config.json` authority

- Always **regenerate from judged graph facts**; never copy a source `config.json` verbatim
  (the current export already does this — keep the rule).
- Emit `model_type: "qwen4_exp"` with `architectures: ["Qwen4ExpForCausalLM"]` (or
  `Qwen4ExpForConditionalGeneration` for a multimodal wrapper), and put every text fact in the
  `text_config` sub-config (`model_type: "qwen4_exp_text"`): `hidden_size`, `num_hidden_layers`,
  `num_attention_heads` + `head_dim`, `num_key_value_heads`, `rope_parameters` / `rope_theta` /
  `partial_rotary_factor`, `max_position_embeddings`, the per-layer `layer_types` table
  (`linear_attention` / `full_attention` — see §5.3 for why `qwen_sparse_attention` is off
  limits), MoE `num_experts` / `num_experts_per_tok` / `shared_expert_intermediate_size` /
  `moe_intermediate_size`, the HyperConnection fields (`hc_count` **must be > 1**, `hc_lowrank`,
  `output_gate_type`), and the PLE block (`ple_layer_ids` 1-based, `ple_embed_dim`,
  `ple_conv_kernel_size`, `ngram_size`, `heads_per_ngram`, `ngram_vocab_size_base`,
  `make_ngram_vocab_size_divisible_by`, `split_ngram_parts`) — plus `quantization_config` for
  Format B. `vision_config` is required by the config class even for a text-only export.
- **Tolerance-correctness gate:** an exported checkpoint must re-encode cleanly (round-trip)
  and `verify` on the rebuilt container.

### 7.4 New / changed files

| File | Change |
|---|---|
| `src/Amql.Cli/ModelExporter.cs` | extend `Export(…)` with an `arch` parameter; reuse the graph→name reconstruction, the parallel worker and the shape-overflow guard; invoke the n-gram builder when a corpus is supplied, else write `enabled:false`; keep the MTP companion path |
| `src/Amql.Cli/ExportConfig.cs` | add the Flash-Next fact builder (judge each fact from `ExecutionSurface` + `PrecisionMap`; refuse by name on anything unjudged) |
| `src/Amql.Cli/Qwen4NextLayout.cs` (new) | the §5.2 shadow table + fixture-derived name validation |
| `src/Amql.Ngram/` (new project, or fold into `Amql.Merge`) | `PleTableBuilder`: splitmix64 / prime-multiplier hashing matching `Qwen4ExpNGramEmbedding`, row-shard writer emitting `…ngram_embedding.shard_{i}.weight`, and the optional `layer_multipliers` / `ngram_heads_offsets` / `ngram_heads_vocab_sizes` buffers. Reuses `PairCollector` for corpus statistics; **no ridge solve** |
| `src/Amql.Cli/MoeIfy.cs` | `--experts N --top-k K --shared 1 --layout qwen4-next` (enforces the GDN/QSA block pattern) |
| `src/Amql.Gguf/HfCheckpointToGguf.cs` | `qwen4`-family architecture branch (T2) |
| `README.md`, `docs/` | new section with measured numbers; `--list-architectures` help |

### 7.5 GGUF tier (T2)

Extend `HfCheckpointToGguf.cs` with a `qwen4`-family branch:

- Map `sparse_attention` layers into the new arch's hyper-parameters
  (`rope.dimension_sections`, `attention.recurrent_layers`, `full_attention_interval`, sparse
  window/sink, n-gram vocab size).
- **Reuse** the existing Qwen3-Next transforms unchanged: tiled V-head reorder,
  `A_log = -exp(A_log)`, norms `+1`, conv squeeze, F16 emission with F32 ssm scalars.
- The n-gram table follows as its own tensor set.
- **Ship only after T1 produces a checkpoint that loads in a target engine**, so the GGUF map
  is grounded against real tensor names rather than guessed ones.

### 7.6 Pipeline composition & determinism

Prefer composing existing, tested, byte-exact steps over new ones:

```
prune → moe-ify (--layout qwen4-next) → generate-mtp --fit → export --arch qwen4-next --quant mxfp4
```

Every transformation step is deterministic (no random seeds in the learning numeric paths —
the `fit-mtp` and merge rules already hold): two identical pipelines produce byte-identical
artifacts. Each phase ends with a **runnable gate** in the CUDA plan's style — build green,
existing CPU tests green, a new round-trip/verify test green, and a smoke on the demo
container.

### 7.7 Validation gates

1. **Round-trip** — `export --arch qwen4-next` then `encode` the result → byte-exact container
   (or a documented tolerance-bounded diff where the n-gram table is present).
2. **Engine load** — load the exported checkpoint in a real Flash-Next-compatible runtime
   (vLLM / SGLang / unpatched HF Transformers) and run a smoke generation; compare held-out
   PPL / token output against the AMQL `generate` path on the same container. Tolerance-gated
   for quantised exports — **never bit-identical for MXFP4**, per `docs/CUDA_PLAN.md` §4. Mark
   to skip in offline builds.
3. **N-gram value** — run `enabled:false` vs `enabled:true` on held-out tokens and report the
   delta, like MTP's acceptance gate. **If the table does not measurably help, ship it disabled
   by default.**
4. **Determinism** — two identical runs → byte-identical PLE shards and identical hash buffers.
5. **Full suite green** — all 166 existing CPU tests stay green; `verify` passes on the
   exported → re-encoded container.

---

## 8. Phases (each gated)

### Phase 0/A — Mapping + config surface (no topology change)
Implement `Qwen4NextLayout` (fixture-derived from a pinned Flash-Next reference) and the
`config.json` writer; wire `--arch qwen4-next` to a **pure remap** of an existing container,
emitting the Flash-Next discriminant over the *existing* tensor set (no new tensors).
**Gate:** a demo / 0.8B / pruned-27B container exports with `--arch qwen4-next`, re-encodes
and `verify`s byte-exact; and the result loads in the target engine and generates identically
to a `qwen35moe` export.

### Phase 1/B — Format B (MXFP4) + sparse geometry + MoE width
Bridge the `--quant mxfp4` tensor split to the Flash-Next layout (emit `weight` +
`weight_scale` companions under Flash-Next names); record sparse geometry; enable
`--experts` / `--shared` re-slicing via existing `moe-ify`.
**Gate:** the MXFP4 Flash-Next checkpoint loads; parity suite per-op tolerance (e.g. 1e-3 for
FP4-weight GEMMs with FP32 accumulate) against the BF16 path; 27B-pruned engine-load + PPL
parity vs the AMQL path.

### Phase 2 — moe-ify at target scale
`--experts N --top-k K --shared 1 --layout qwen4-next`, per-layer expert bank, hybrid GDN/QSA
block pattern enforced by layout.
**Gate:** held-out `dense → moe` PPL at the target topology; the exported checkpoint runs under
the reference backend with the expected cost profile. Fall back to a smaller `num_experts` if
the PPL gate fails — the layout is format-faithful, the size is our choice.

### Phase 3/C — MTP drafter + PLE table
Materialise `mtp.*` in the vLLM two-projector shape (`fc_embedding` + `fc_hidden`, §5.6); add
`PleTableBuilder` over the repo `corpus.txt`; activation-cost and value gate (§7.7.3). Gate this
phase on the §12.1 HyperConnection decision — if HC streams are not implemented, PLE has nothing
to attach to.
**Gate:** held-out drafter acceptance measured; the PLE shards are byte-reproducible, the
on/off ablation is ≥ 0, and the checkpoint loads with `ple_layer_ids: []` as the default.

### Phase 4/D — GGUF (T2), CUDA tooling acceleration, hardening
Add the `qwen4`-family GGUF branch; wire the §5.2 tooling numerics (merge Gram via GEMM,
cuSOLVER solve, moe-ify clustering kernels, MTP/n-gram fits) behind the Try-gated GPU path so
each pipeline step is fast — export itself stays host-side, I/O-bound, byte-exact. Write the
README section with measured numbers.
**Gate:** the full pipeline (prune → moe-ify → fit-mtp → export Flash-Next MXFP4 → GGUF)
completes in a small measured time within the hardware budget; full test suite green.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Flash-Next `config.json` / tensor names drift across backend versions | **Phase A pins a reference `config.json` + index from the Hub as the judgement fixture**; refresh deliberately. Every name is tied to a real engine load before n-gram/GGUF work |
| "Same format" read as "must load in unpatched vLLM at full architectural fidelity" | Separate *loadable-and-runs* (format) from *exact arch parity*; this design targets the former, honestly labelled |
| **QSA mislabelling changes quality** | AMQL has no indexer, so it **cannot** emit `qwen_sparse_attention` at all (§5.3). Export those layers as `full_attention` and omit all five `indexer_*` fields — the older-checkpoint spelling vLLM still accepts. Half-specified QSA is a hard load failure, not a fallback |
| **HyperConnections are mandatory and AMQL does not model them** | `Qwen4ExpTextConfig` raises `ValueError("Qwen4Exp requires hc_count > 1")`, so there is no single-stream configuration. Hidden states between layers are `[..., hc_count × hidden_size]` (HC outer, HS inner). Either implement the HC modules (mix/combine + `GroupedGemmaRMSNorm`'s Gemma-style `1+w`, and `input_mix_weight_down/up` + `block_inject_weight`) or **do not target `qwen4_exp`** — this is the largest scope risk in the design |
| **PLE/n-gram table dwarfs the rest of the checkpoint** | ~20M rows × `head_dim` at the config defaults. `ple_layer_ids: []` is a legal zero-cost configuration and should be the default until the value gate proves the table pays for itself |
| PLE table adds cost without value, or corrupts outputs | `ple_layer_ids: []` by default; value gate before enabling; hash-and-fill is deterministic; never claim it is Qwen's trained table |
| MXFP4 export is not bit-identical | Documented tolerance policy already in `docs/CUDA_PLAN.md` §4; the engine-load gate uses tolerance, not byte-equality |
| 512-expert config without 512 experts | `experts` in config is a **count**; build the widest MoE that fits budget, always emit `shared_experts`; no retraining implied |
| MoE-ifying a dense 27B degrades quality beyond the PPL gate | Keep the gate; fall back to a smaller `num_experts` |
| Expert-bank export size at high expert counts | Each expert is its own segment (already true); MXFP4 + prune-first keeps total on-disk in budget |
| PLE table inflates disk (~20M rows × `head_dim`) | FP8 storage with one global scale halves it vs BF16; `ngram_vocab_size_base` is config-tunable; offload path (`engram_config.cpu_offload`) keeps it out of VRAM |
| Round-trip loss on Flash-Next names | The reverse mapper (encoder) is part of Phase A's gate; a name with no inverse refuses |

---

## 10. Cost / time summary

Producing a **Flash-Next-format instance** of an existing container costs, in run time,
roughly today's `export --quant mxfp4` plus, when PLE is enabled, **one corpus pass to hash and
fill the n-gram table** (I/O- and hash-bound, no solve) and, optionally, **one re-cluster pass of
`moe-ify`** to widen experts. There is **no training**, **no new container
format**, and **no change to the BF16 canonical authority**. The result is a checkpoint the
Flash-Next inference stack serves unaltered, at an MXFP4 working set that fits 32 GiB VRAM.

---

## 11. What this does *not* do

- It does **not** reproduce Qwen's proprietary 176B weights or training.
- It does **not** change the canonical container or the default `export` path — byte-exactness,
  `verify` and hash guarantees are all unchanged.
- It does **not** require training/SGD anywhere: moe-ify and fit-mtp stay calculation-first,
  and the PLE table is hash-and-fill, not fitted.
- It does **not** implement QSA (§5.3) or HyperConnection streams (§12.1) — both are runtime
  features, and the design says so rather than emitting a checkpoint that would load and
  compute something different.
- It does **not** introduce a new container format; VIndex3 remains the sole authority.

---

## 12. Open questions — **resolved** (vLLM `main`, 2026-09-22)

All three were answered from source: `vllm/model_executor/models/registry.py` and the
`vllm/models/qwen4_exp/` package (`config.py`, `__init__.py`, `nvidia/{model,mtp,qsa,
indexer_qsa,ple_layer,ngram_embedding}.py`, `common/{ple,hyperconnection}.py`). Local copies
of the retrieved files are kept under `.research/` as the Phase A judgement fixture.

1. **`model_type` and per-layer tensor-name spelling — resolved.**
   `Qwen4ExpConfig.model_type = "qwen4_exp"` with sub-configs `vision_config`
   (`Qwen4ExpVisionConfig`, also `model_type = "qwen4_exp"`) and `text_config`
   (`Qwen4ExpTextConfig`, `model_type = "qwen4_exp_text"`, deriving from `Qwen3NextConfig`).
   Architectures registered: `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`
   (multimodal; text weights prefixed `model.language_model.` and rewritten to `model.`), and
   `Qwen4ExpMTP`. Note vLLM resolves CUDA/ROCm implementations lazily by platform and raises
   `NotImplementedError` on XPU/TPU. Layer types are `"linear_attention"`,
   `"full_attention"` and `"qwen_sparse_attention"`; per-layer names are `linear_attn.*`,
   `self_attn.{qkv_proj,o_proj,indexer.index_qk_proj}`, `ple.*`, and
   `{attn,mlp}_hyper_connection.hyper_connection.*`. Full table in §5.2.
2. **N-gram representation — resolved: neither.** It is not a `quantization_config` entry and
   not a dedicated sidecar file. The *geometry* is ordinary text-config fields
   (`ple_layer_ids`, `ple_embed_dim`, `ple_conv_kernel_size`, `ngram_size`, `heads_per_ngram`,
   `ngram_vocab_size_base`, `make_ngram_vocab_size_divisible_by`, `split_ngram_parts`,
   `output_gate_type`); the *weights* are row shards named
   `…ple_embedding.ngram_embedding.shard_{i}.weight` inside the normal safetensors shards.
   Quantisation is orthogonal and restricted: FP8-serialised with one global `weight_scale`,
   or unquantised via `ignored_layers`. See §5.4.
3. **Expose n-gram `n` as a CLI sweep — resolved: no, not as a sweep.** `ngram_size` is not a
   free runtime knob: it fixes `ngram_heads = (ngram_size-1) * heads_per_ngram`, which must
   divide `ple_embed_dim`, and it sets the conv dilation and the short-conv state length
   `(ple_conv_kernel_size-1) * ngram_size`. Changing it invalidates any table already built.
   So make it **one value chosen at build time and recorded in config**, defaulting to 3, with
   the divisibility constraint validated before anything is written — the same discipline as
   `layer_types`. The MTP `K ∈ {1,4,8}` sweep convention deliberately does *not* transfer.
   The knob worth exposing instead is **`--ple-layers`** (which layers get PLE at all), since
   `ple_layer_ids: []` is the cheap, legal, off configuration.

### 12.1 New questions this research opened

1. **Do we target `qwen4_exp` at all, given mandatory HyperConnections?** `hc_count > 1` is
   enforced, so a faithful Flash-Next export requires AMQL to model HC streams end-to-end
   (`[..., hc_count × hidden_size]` hidden layout, `GroupedGemmaRMSNorm`, mix/combine, and a
   final `hyper_connection_mixer`). This is a runtime feature, not a serialisation change, and
   it is the single biggest scope decision in the project. Alternative: keep
   `--arch qwen4-next` as a *name/config* target for the subset we can express and document
   the fidelity gap, as §5.3 already does for QSA.
2. **SGLang / HF Transformers parity is still unverified.** Everything above is vLLM. The
   Hub fixture (`Qwen/Qwen3.8-Flash-Next` `config.json` +
   `model.safetensors.index.json`) still has to be pinned in Phase A to confirm the checkpoint
   spelling inferred from `packed_modules_map`, and to check the other engines.
3. **Is the PLE table worth its disk cost at our scale?** ~20M rows × `head_dim` is likely
   larger than the rest of an MXFP4 27B-pruned export combined. The §7.7.3 value gate must
   run before `--ple-layers` is ever non-empty by default.

---

## 13. Deliverable shape

- `docs/export-qwen3.8-flash-next.md` (this design).
- `src/Amql.Cli/Qwen4NextLayout.cs` — the shadow table + `config.json` writer.
- `src/Amql.Ngram/PleTableBuilder.cs` — the hashed PLE n-gram table and its row shards.
- Extensions to `ModelExporter` / `ExportConfig` (`--arch`), `MoeIfy` (Flash-Next scale +
  layout), the MXFP4 split bridge, and `HfCheckpointToGguf` (`qwen4` branch).
- A CLI `--list-architectures`, a README section with measured numbers, and the phase gates as
  runnable smoke commands.

The CPU stays the byte-exact reference; any quantisation/GPU path sits behind a budget-aware,
Try-gated seam and is never a second source of truth for artifacts.
