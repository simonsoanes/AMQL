# AMQL — a .NET port of LARQL (high-level building blocks)

Status: **initial scaffold**. This document is the design companion for the
first .NET slice. Scope is deliberately bounded: the port implements the
*high-level building blocks* of the [LARQL](https://github.com/HuggingFace/larql)
stack — a safetensors loader, the VINDEX3 model-system container format, and a
generic (architecture-agnostic) inference runtime. Total parity is not a goal;
every deliberate divergence is called out below.

Reference: `D:\Dev\Open Source\larql` (Rust workspace), studied 2026-09-03.

---

## 1. Why .NET / what this is

LARQL answers:

> What are these model objects, what operations can consume them, which
> representations are equivalent, which parts should be resident, and what
> future computation will need them?

VINDEX3 is a *model-system container format*: `index.json` is the root
authority, `system_graph.json` is the semantic IR, and `segments/*.bin` are
physical payloads addressed by logical object id. The runtime executes a model
**from the container alone** — no HF checkpoint, no transformeRs config, no
`if family == X` branches.

AMQL reproduces that spine in C#/.NET 10 so the format and the generic
execution contract are usable from .NET tooling (hosting, tooling utilities,
embedded inference).

## 2. Solution layout

```
D:\Dev\AMQL\
├── AMQL.slnx                      # .NET 10 solution (slnx format)
├── docs/DESIGN.md                 # this document
├── containers/                    # encoder output (e.g. Qwen3.5-0.8B/)
├── src/
│   ├── Amql.Safetensors/          # ↦ larql-models::loading::safetensors
│   ├── Amql.Vindex3/              # ↦ larql-vindex::format::vindex3 (+ graph)
│   ├── Amql.Inference/            # ↦ larql-inference::vindex3 (generic runtime)
│   ├── Amql.Hf/                   # ↦ larql-models detection: G0 inventory, G1 facts, G2 graph
│   └── Amql.Cli/                  # the loader front-end: encode + verify
└── tests/
    └── Amql.Tests/                # xunit: format, runtime, and loader pipeline tests
```

Dependency spine mirrors the Rust crate graph:

```
Amql.Safetensors   (no deps — only System.Text.Json / System.IO.MemoryMappedFiles)
   ▲
Amql.Vindex3       (↦ larql-vindex)
   ▲
Amql.Inference     (↦ larql-inference)
   ▲
Amql.Hf            (↦ larql-models detection: G0/G1/G2)
   ▲
Amql.Cli           (shell: encode <model-dir> --out <container-dir> | verify <container-dir>)
```

## 3. Building-block map (Rust → C#)

| LARQL (Rust) | AMQL (.NET) | Notes |
|---|---|---|
| `larql-models/src/loading/safetensors` | `Amql.Safetensors` | header/payload parse, dtype decode, shard walking, MLX `weights/` fallback, HF-cache path resolution |
| `larql-vindex/src/format/vindex3` (index, graph, encode/segment) | `Amql.Vindex3` | `index.json`, `system_graph.json`, `.bin` segments, container open/inspect/encode |
| `format/vindex3/exec/operands.rs` (`OperandStore`) | `Amql.Vindex3.OperandStore` | `object id → representation → segment → tensor` resolution |
| `larql-inference/src/vindex3/runtime.rs` (GenericRuntime) | `Amql.Inference` | op plan, generic layer loop, KV cache, decode session |
| `opplan/*` (ComponentOpPlan, LayerPlan, ops) | `Amql.Inference.Plan` | same op vocabulary: embedding, norms, attention, ffn, head |
| `larql-compute` attention/ffn/rope kernels | `Amql.Inference.Ops` | plain managed implementations, `Vector<float>`-accelerated dot products |
| `inspect-hf` → invent → represent → encode | `Amql.Hf` + `Amql.Cli` | **shipped in the loader**: G0 inventory, G1 config facts, G2 graph/surface, G3 canonical encode; the reference's auto-detection depth is explicitly bounded (see §6) |
| quantisation / `represent`/k-quants / MXFP4 | — | **out of scope**: canonical raw encodings only |
| remote MoE / Metal / GPU backends, LQL, router, server | — | **out of scope** |

## 4. On-disk formats (ported faithfully)

### 4.1 safetensors (`model.safetensors`)

```
[8 bytes u64 LE: header JSON length]
[header JSON: space-padded so 8 + header_len ≡ 0 (mod 8)]
[payload bytes, concatenated in header order]
```

Header entries per tensor: `dtype` (uppercase safetensors label), `shape`,
`data_offsets` `[begin, end)` relative to payload start. Dtype table:

| label | element size | decode |
|---|---|---|
| `F64` | 8 | (accepted as raw; not widened) |
| `F32` | 4 | LE bits → f32 |
| `F16` | 2 | half decode |
| `BF16` | 2 | bfloat16 decode |
| `I8` | 1 | sign-extend |
| `I16/I32/I64`, `U8`, `BOOL` | — | accepted as raw bytes, refused for f32 widening (mirrors `UnsupportedDtype`) |
| `F8_E4M3`, `F8_E5M2`, `F8_E8M0` | 1 | bit-pattern decoders (Open Compute) |

The loader memory-maps shards (`MemoryMappedFile`), parses headers only for
fast inventory, and decodes tensors on demand — matching LARQL's walk-only /
filtered loading pattern.

### 4.2 VINDEX3 container

```
<container>/
├── index.json          # version, model, representations {}, segments {}, …
├── system_graph.json   # schema 6: components, objects, edges
└── segments/*.bin      # one per representation: {object}@{encoding}
```

The `.bin` segment mirrors safetensors framing:

```
[8 bytes u64 LE: header JSON length]
[header JSON: space-padded so 8 + header_len ≡ 0 (mod 16)]
[payload bytes, in tensor-table order (table sorted by name)]
```

```jsonc
// segment header
{ "schema": 1,
  "representation": "target.decoder_stack@BF16",
  "tensors": [
    { "name": "3.self_attn.q_proj.weight",
      "dtype": "BF16", "shape": [128, 256], "offset": 0, "len": 65536 }
  ] }
```

`index.json` keeps the sole root authority:

```jsonc
{ "version": 4,
  "model": "…", "family": "…", "hidden_size": 6656, "num_layers": 52,
  "system_graph": "system_graph.json",
  "representations": {
    "target.decoder_stack@BF16": {
      "object": "target.decoder_stack", "encoding": "BF16",
      "segment": "segments/target.decoder_stack.bin",
      "tensor_count": 12, "payload_bytes": 12345,
      "payload_sha256": "…", "segment_sha256": "…" } },
  "profiles": [{ "name": "exact", "selects": {} }],
  "segments": { "segments/target.decoder_stack": 1 } }
```

`system_graph.json` (schema 6) carries components (role
`primary_text|perception|drafter`), per-layer `AttentionLayerPolicy`
(operator, span, window, position), logical objects (kind, source bindings,
representations), `HiddenStateEdge`s, and the `ExecutionSurface` (attention /
ffn / norm / head — presence means the program runs it).

**Verification contract.** `Container.VerifyIntegrity()` recomputes
payload + full-segment SHA-256s and compares against `index.json`, and
re-derives the segment tensor table from the persisted graph (four-authority
style: Declared ≡ Graph ≡ Encoded). Byte equivalence first; semantic checks
are the runtime's job.

### 4.3 Naming / casing convention

All JSON round-trips use **snake_case field names** identical to the Rust
serde spellings (mirrors larql's `[JsonPropertyName]`-free conventions). C#
types use PascalCase; only the JSON layer converts. This keeps containers
byte-compatible with the Rust implementation.

## 5. The generic runtime (Amql.Inference)

Architecture follows `Vindex3Runtime`:

```
container ──open──▶ SystemInspection ──plan──▶ ComponentOpPlan
                                                    │ (ops + OperandRef, never tensor names)
                                                    ▼
                             GenericRuntime ──▶ layer loop: pre-norm → attention
                             (embed → layers → final-norm → head)
                                                    │
                                                    ▼
                             RowKvCache + DecodeSession (prefill / step)
```

1. **Planner** builds `ComponentOpPlan` from the graph + surfaces only:
   per-layer `AttentionLayerPolicy` rows are read from the persisted table
   (no layer-pattern arithmetic), spans/windows/positions are table reads,
   norms come from `NormSurface` (placement PreOnly vs PrePost derived
   from operand topology evidence when ambiguous, else from the surface).
   An unsupported layer operator (gated_delta, mamba2, kda, mla, conv_qkv)
   **refuses to plan** with the primitive named — fail-closed, never guessed.
2. **Operands** resolve `OperandRef {object, tensor} → representation →
   segment table entry → payload bytes`, widened to f32 at load time.
3. **Ops** are the generic primitives: embedding (+ optional norm/scale),
   RMSNorm/LayerNorm, RoPE (`inv_freq = 1/θ^(2i/d)`, position offset,
   divisor, freq scaling data carried but identity implemented),
   GQA softmax attention parameterised by query/score scale, logit
   softcapping (`tanh(s/c)·c`), sliding-window masking, per-head sinks,
   dense FFN (gate/up/down, SiLU | tanh-GELU) and routed MoE (top-k
   softmax router, per-expert gate/up/down), output head (projection,
   multiplier, softcapping).
4. **KV cache** is row-based and caller-owned: per layer, position-ordered
   key/value rows (post-norm, post-RoPE); sliding windows mask at
   attention time, never evict. `DecodeSession` exposes the
   `prefill(tokens) → logits`, `step(token) → logits`, `position`
   contract; sampling is greedy + temperature/top-k/top-p.

The deletion invariant holds by construction: the runtime reads only the
container; no original HF tensor name appears on the execution path.

## 5.5 The loader pipeline (Amql.Hf + Amql.Cli)

`amql-cli encode <model-dir> --out <container-dir>` runs the reference's
G0→G3 chain, .NET-shaped:

1. **G0 inventory** (`HfInventory`): discovers `*.safetensors` shards
   (root or MLX `weights/`), exposes tensor names/facts/payloads.
2. **G1 facts** (`ModelConfig`): `config.json` → `TextArchitectureFacts`.
   The multimodal wrapper (`Qwen3_5ForConditionalGeneration`) is unwrapped
   to its `text_config`; `layer_types` (a judged per-layer operator table)
   is mandatory — an absent table refuses the checkpoint.
3. **G2 graph** (`ArchMapper`): facts → `SystemGraph` + `ExecutionSurface`.
   The text prefix is detected from the inventory (`model.language_model`,
   `model`, …), never assumed. Vision and MTP side-components are recorded
   as carried objects (source bindings only, no segments).
4. **G3 encode** (`ContainerEncoder`): object-relative segment names
   (`layers.3.self_attn.q_proj.weight → 3.self_attn.q_proj.weight`,
   `embed_tokens.weight → weight`), payload bytes copied verbatim, the
   canonical encoding judged from the stored dtype. Tensors deliberately
   kept in another dtype (Qwen3.5 keeps `A_log` and the recurrent norm in
   F32 inside the BF16 stack) are recorded in `index.precision_map` as
   exceptions — never promoted, never refused.

`amql-cli verify <container-dir>` re-derives byte equivalence from disk
alone (integrity hashes), resolves + widens real payloads through the
operand store, prints the operator census, and reports the runtime
boundary by attempting to plan the component (fail-closed refusal names
the primitives this build does not serve).

`amql-cli generate <container> --tokens 0,1 [--steps N]` runs
autoregressive generation against the planned component: greedy by
default, or temperature/top-k/top-p with a session-stable RNG (one
`Random` per run — repeated steps are draws, not replays). `--logits K`
prints each step's top-K window with probabilities. `amql-cli
inspect-token <container> <id>` inspects a token in vocabulary space from
the container alone: embedding profile (row, min/max/mean, L2) and
nearest neighbours by cosine; with `--tokens ctx,...` and an executable
component it also reports the model's logit/rank for the token at the
end of that context. `amql-cli synth-model <dir>` writes the executable
2-layer demo checkpoint so the whole chain (encode → generate →
inspect-token) is exercisable without a servable HF model.

Rope judgment in the loader: plain default rotary (no MRoPE sections,
full factor, no frequency scaling) is served as standard `PositionRope`;
every other rope fact is carried-unresolved and refused by name.

**Executable boundary for Qwen3.5-0.8B:** the loaded container is fully
faithful (24 layers, 320 tensors, ~1.4 GiB), but this build's executor
serves none of its layers yet — 18 are `linear_attention` (conv +
recurrent-key hybrid) and even the 6 `softmax` layers carry a hard output
gate (second half of `q_proj`) plus partial-MRoPE (rotary factor 0.25).
Each of those is refused by name at plan time rather than approximated.
Weighted QK norm is likewise refused (a stack with `q_norm`/`k_norm`
tensors would otherwise silently skip normalising Q/K).

## 6. Explicit non-goals (this slice)

- **Reference-depth auto-detection** (G0–G2): the loader ships G0/G1/G2 for
  the Qwen3.5 text family's judged surface, carried verbatim rather than
  approximated. Generalised `inspect-hf` heuristics, representability
  arithmetic (choosing encodings), and quantisation policy remain absent.
- **Linear-attention / partial-MRoPE / output-gate / weighted-QK-norm
  kernels**: declared in the graph, refused at plan time by name. Porting
  them is future work, not approximation.
- **Quantization** (k-quants, MXFP4, FP8 serving), GGUF loading, remote
  expert serving, Metal/GPU backends.
- **LQL**, the router/server surface, KV dispatch tiers, drafter/inference
  pipelines on top of the runtime.
- **Tokenizer support**: the loader and runtime operate on token ids;
  `tokenizer.json`/`vocab.json` are not parsed in this slice.
- **Byte-parity with Rust for every kernel**: numerics follow the same
  formulas; tests validate against independent hand-computed references,
  not against Rust goldens (no cross-compiler A/B harness in this slice).

## 7. Verification story

`tests/Amql.Tests`:

- **Safetensors**: golden dtype decodes (F16/BF16/FP8 bit patterns),
  writer→reader round-trip, alignment invariants, sharded directory walk.
- **VINDEX3**: container encode → open round-trip; segment byte-layout
  golden (`8 + header_len ≡ 0 mod 16`, offsets relative to payload start,
  table sorted by name); SHA-256 integrity verification passes and fails on
  a mutated byte.
- **Inference**: end-to-end synthetic model — build a tiny Llama-shaped
  checkpoint in safetensors (deterministic weights), encode to a container,
  open it, run prefill + decode, and check hidden states / logits against
  hand-computed references (embedding lookup, RMSNorm, RoPE at position,
  causal sliding attention, FFN, head, routed MoE). Fail-closed guards:
  gated attention, sinks, weighted QK norm, linear-attention and unknown
  position/operator rows all refuse by name.
- **Loader + inference CLI**: a synthetic multimodal checkpoint (text_config
  wrapper, mixed linear/softmax `layer_types`, F32 precision exceptions,
  vision + MTP side-components) encodes → opens → verifies → resolves
  payloads end-to-end; the real Qwen3.5-0.8B config facts are asserted
  when the checkpoint is present on the machine. CLI commands are tested
  against the encoded demo checkpoint: greedy determinism + vocab bounds,
  sampled replay for a fixed seed + different seed divergence, embedding
  profile / neighbour ranking vs a brute-force oracle, logits top-1 ==
  library argmax, and the named `linear_attention` refusal.