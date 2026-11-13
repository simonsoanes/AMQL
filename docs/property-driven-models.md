# Property-driven model support

AMQL imports, runs and exports a model from what its checkpoint states: the
config, the tensor names and the tensor shapes. A new family that is built
from operators AMQL already runs needs no code. This page covers how that
works, how it is measured, and where it stops.

## How a model is judged

Import (`ArchMapper`) reads three kinds of evidence, in this order:

1. **Tensors.** Every layer tensor is given a *role* by `TensorRoles`
   (`src/Amql.Vindex3/TensorRoles.cs`): `q`, `k`, `v`, `o`, `q_norm`,
   `k_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, and four norm sites
   (`attn_in_norm`, `attn_out_norm`, `ffn_in_norm`, `ffn_out_norm`), each
   with an optional `.bias`. A role has a list of known spellings
   (`self_attn.q_proj`, `attention.wq`, `mlp.c_fc`, …). The norm sites are
   told apart by structure, because one name can mean two things.
   `post_attention_layernorm` is the pre-FFN norm in Llama and the
   attention-output norm in Gemma 2 and OLMo 2; which one it is follows from
   the other norms the layer carries.
   - The block shape follows from which sites exist: pre-only (Llama),
     sandwich (Gemma 2/3, GLM-4, LFM3), post-only (OLMo 2, EXAONE 4), or
     parallel (one input norm and nothing between attention and FFN:
     Cohere, GPT-J).
   - Contiguous fusions (`qkv_proj`, `gate_up_proj`) are split on import as
     zero-copy byte ranges and recorded, so export can join them back.
     Interleaved fusions (GPT-NeoX `query_key_value`) are refused by name.
   - Biases, gated or plain FFN, QK-norm width (one head or the whole
     projection), and LayerNorm vs RMSNorm (a norm with a bias) are all read
     from the tensors.
2. **Config.** `ModelConfig` reads every standard key under its known
   aliases. That includes `layer_types` (or a `sliding_window_pattern`
   period), `sliding_window` with `use_sliding_window`, per-layer-type
   `rope_parameters` blocks, `partial_rotary_factor`, `no_rope_layers`,
   `query_pre_attn_scalar`, the attention and final logit softcaps,
   Granite's four multipliers, Cohere's `logit_scale`, and which epsilon key
   is used (`layer_norm_eps` means LayerNorm). The whole config is also kept
   in the container as `source_config.json`.
3. **Conventions.** A few modelling choices are stated nowhere in a
   checkpoint: whether RMSNorm multiplies by `w` or `1 + w`, whether
   embeddings are scaled by √hidden, whether rope pairs are adjacent or
   half-split. These live in `src/Amql.Hf/family-conventions.json`, keyed
   by `model_type`. Adding a family that follows a known convention is one
   line there, or in a file named by `AMQL_CONVENTIONS`, with no rebuild.
   Every convention applied is recorded in the surface's `provenance`, so a
   fact that came from this table is never presented as read from the
   checkpoint.

The role map is persisted on the execution surface (`roles`). The planner
binds operands by role, never by spelling. For containers written before
roles existed, it judges the map from the segment with the same rules.

## Export formats

`amql-cli export <container> --out <path> [--format hf|gguf|onnx]`

| Format | What it writes | How it is generic |
|---|---|---|
| `hf` (default) | `model.safetensors` + `config.json` | The checkpoint's own config with every judged fact written over it (layer count, sizes, `layer_types`, `no_rope_layers`, MoE), so unknown keys survive. Split fusions are joined back under their original names. A per-layer config array the build cannot regenerate refuses the export if the layer count changed. |
| `gguf` | one llama.cpp GGUF | Tensors take llama.cpp's role names (`blk.N.attn_q`, `post_ffw_norm`, …). The llama.cpp architecture is chosen from the model's properties: `llama` (Q/K permuted to adjacent pairs, as `convert_hf_to_gguf.py` does), `qwen2`, `qwen3`, `granite`, `olmo2`. A model with a property none of these computes is refused, naming the property. |
| `onnx` | `model.onnx` + `.data` | The plan, op for op, in standard ONNX ops. Every norm kind and site, biases, QK norms, half-split, interleaved, partial and zero-tail rope, NoPE layers, sliding windows, softcaps, scale knobs and parallel blocks. |

## How it is measured

`tests/Amql.Tests/fixtures/make_arch_zoo.py` builds a tiny random model for
each of 20 transformers architectures that no AMQL code names: Llama (with
and without biases), Mistral, Ministral, Qwen2, Qwen3, Phi-3, Gemma, Gemma 2,
Gemma 3, OLMo 2, OLMo 3, Granite, SmolLM3, StableLM, Starcoder2, Cohere,
Helium, GLM-4 and EXAONE 4. Each one stores transformers' own logits,
computed with eager attention: SDPA silently drops Gemma 2's attention
softcap.

| Check | Result |
|---|---|
| Import + run (`ArchZooTests.Encodes_And_Matches_Transformers_Logits`) | 20 / 20 match transformers (< 2e-3) |
| HF export round trip (`ArchZooTests.Hf_Export_Round_Trips`): same tensor names, every original config key kept, re-imports to the same logits | 20 / 20 |
| HF export loaded by transformers under its own class (`scripts/export/check_hf.py`) | 20 / 20 reproduce the original logits (to the goldens' 6-decimal rounding) |
| ONNX export, run in onnxruntime (`scripts/export/check_onnx.py`) | 20 / 20, worst difference 1.7e-5 |
| GGUF export, read back with the `gguf` package under llama.cpp's rules (`scripts/export/check_gguf.py`) | 8 / 8 exported match (llama, mistral, helium, phi3, qwen2, qwen3, granite, olmo2); the other 12 are refused by property |
| Real checkpoint: Phi-3-mini-4k-instruct | Encodes unchanged. 8 greedy tokens match transformers, including a step where the top two differ by 0.013. HF export re-fuses `qkv_proj` / `gate_up_proj` and reloads in transformers with identical logprobs. |

`check_gguf.py` is an independent reading of the file, not llama.cpp.
Loading the GGUFs in llama.cpp itself is still the final check.

## Fixes made on the way

- **Granite's residual multiplier** was applied to the residual stream
  (`h·rm + branch`). transformers applies it to the branch
  (`h + branch·rm`). The old test's hand-written reference encoded the same
  mistake. Both are corrected; the zoo's Granite now matches.
- **LayerNorm biases** on the final norm were never bound.
- **`gelu_pytorch_tanh`** was rewritten to `gelu` on import. It is now kept
  exactly as the checkpoint states it.
- **HF-cache symlinks**: safetensors shards that are symlinks (the Hugging
  Face cache layout) read as empty. They are now resolved.
- **ONNX**: the exporter now builds the graph from the plan. It previously
  multiplied by untransposed `[out, in]` weights, fixed theta at 1e6 and
  used a single-head attention stand-in. Small initializers now stay inline,
  where shape inference can read them, and nodes are named, which
  onnxruntime's optimiser requires.

## Limits

- **Operators.** A family needing an operator the runtime does not have is
  refused by name, not approximated: MLA, Mamba, a new recurrent mixer,
  learned attention sinks, interleaved fused QKV, ALiBi or relative position
  biases. Each needs runtime code once. Every family using it then works
  through the same judging.
- **Rope scaling.** `linear`, `llama3`, `yarn` and `longrope` are carried
  unresolved and refused at plan time. Llama 3.1+, Phi-3-128k and Gemma 3's
  global layers need this.
- **GELU.** The runtime computes exact `gelu` with the tanh approximation,
  as it did before. ONNX mirrors the runtime.
- **GGUF.** Only the five architectures above. llama.cpp's Gemma layouts
  hard-code layer patterns and rope bases that need checking against
  llama.cpp before they can be chosen from properties. MoE and the hybrid
  families keep their own converters (`amql-cli to-gguf`). The tokenizer is
  written as BPE with the generic `default` pre-tokenizer, and
  SentencePiece-only tokenizers are not converted.
- **Separate import paths.** ModernBERT and nomic-bert encoders still have
  their own import paths.
- **Kept family layouts.** The LFM3, Qwen3.8-Flash-Next and ModernBERT HF
  export layouts and the Qwen3.5/LFM3/A2D GGUF converters are unchanged.
  The generic paths sit beside them.
