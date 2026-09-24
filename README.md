# AMQL — VIndex3 Model Container & Graph Database

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![.NET 10.0](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/download/dotnet)
[![Research — Pre 1.0](https://img.shields.io/badge/status-research%20%7C%20pre--1.0-orange.svg)

AMQL turns LLMs into a queryable graph database — encode a checkpoint into a VINDEX3
container, then probe token relationships, trace causal pathways, edit weights, generate
LoRA adapters, merge models, prune layers, and export back to a standard checkpoint.
Built in C# against .NET 10.0 with an optional CUDA backend and a WPF desktop GUI.

Credit for the VIndex3 design: [Chris Hay](https://github.com/chrishay).

## What is this for?

AMQL is the model-introspection engine behind a continuous-cognition platform. It exposes
the internal structure of a language model as a **graph** so you can query how tokens
relate, find the edges that carry one concept into another, and then **edit those edges**
— by patching weights, generating a LoRA adapter, or rewriting the base model.

At the token level you can remove false associations or add new ones in a familiar form
(e.g. "PlaceA is the capital of PlaceB"). At the next layer of abstraction you can adjust
relationships that haven't been described linguistically but are inferred — hallucinations,
incorrect tool selection, biases introduced during post-training.

Blending models via consensus-gated token alignment adds solution-space vectors from one
model into another. Combined with automated self-learning, this extends a model's
understanding beyond what fine-tuning alone can achieve — because fine-tuning is inherently
gated at human-level textual representations. Generating genuinely novel solution vectors
is a separate challenge, but AMQL lets you apply them once they exist.

## Quick Start

```bash
# Create a tiny demo model, encode it, and generate a few tokens
amql-cli synth-model demo-model
amql-cli encode demo-model --out demo-container
amql-cli generate demo-container --prompt "The capital of France is" --tokenizer demo-model

# Probe a token relationship
amql-cli route demo-container France Paris --tokenizer demo-model --top 5
```

**Prerequisites:** [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) and Git.

## Architecture

AMQL is nine projects with a CLI front-end and a WPF desktop GUI:

| Project | Purpose |
|---------|---------|
| `Amql.Cli` | CLI (`amql-cli`) — encode, infer, patch, merge, inspect |
| `Amql.Gui` | WPF GUI — container browser, command launcher, parameter editor, tensor explorer |
| `Amql.Vindex3` | Container graph, schema, token-index management |
| `Amql.Safetensors` | Safetensors I/O, MXFP4, ternary, and NVFP4 codecs |
| `Amql.Inference` | Inference engine, tracing, LoRA adapter execution |
| `Amql.Hf` | HuggingFace checkpoint loading, config parsing, architecture mapping |
| `Amql.Merge` | Model merging, MoE-ification, fine-tuning, pruning, model conversion |
| `Amql.Onnx` | ONNX graph builder — zero-dependency protobuf writer |
| `Amql.Gguf` | GGUF v3 converter for llama.cpp deployment |

```
AMQL/
├── src/
│   ├── Amql.Cli/        Amql.Gui/       Amql.Safetensors/
│   ├── Amql.Vindex3/    Amql.Inference/ Amql.Hf/
│   ├── Amql.Merge/      Amql.Onnx/      Amql.Gguf/
├── native/amql_cuda/    # CUDA backend (GEMM, dequant, elementwise)
├── tests/Amql.Tests/    # ~200 unit & integration tests
└── docs/                # Design docs and architecture reference
```

## CLI Reference

```
amql-cli encode <model-dir> --out <container-dir>
amql-cli verify <container-dir>
amql-cli generate <container> --prompt "text" --tokenizer <dir> [--steps N] [--patch <file>]
                     [--trace] [--trace-tensors] [--weights f32|bf16|mxfp4]
amql-cli route <container> <A> <B> --tokenizer <dir> [--top N] [--patch <file>]
amql-cli path <container> <A> <B> [--topk N] [--max-nodes N]
amql-cli inspect-token <container> <token> [--neighbors N] [--tokenizer <dir>]
amql-cli change-tensor <container> <object> <tensor> <cell> (--set V|--add V) --out <patch>
amql-cli save-lora <patch> --out <dir> [--rank 8] [--alpha 16]
amql-cli export <container> --out <dir> [--patch <file>] [--quant mxfp4|ptq1|pq2] [--arch qwen4-next]
amql-cli import <container> <model> --out <dir> [--container]
amql-cli merge <containerA> <containerB> --out <dir>
amql-cli moe-ify <container> --out <dir> --text <corpus> [--experts 8] [--top-k 2]
amql-cli prune <container> --out <dir> --target-bytes <size> [--approach provenance|corpus|random]
amql-cli fine-tune <container> --data <pairs.tsv> --out <patch> [--lr 1e-4]
amql-cli create-model --hidden 768 --layers 12 --vocab 32000 --out <dir>
amql-cli convert-to-classifier <container> --num-labels N --out <dir>
amql-cli convert-to-embedding <container> --out <dir>
amql-cli export-onnx <container> --out <model.onnx>
amql-cli classify <container> --premise "A" --hypothesis "B"
amql-cli layers <container>
amql-cli to-gguf <checkpoint-dir> --out <file.gguf>
amql-cli export --list-architectures
```

Two directory types appear throughout: **containers** (`<container-dir>`, encode output,
weights only) and **checkpoints** (`--tokenizer`, the original HF model directory whose
`tokenizer.json` converts text to ids).

## Key Capabilities

### Model Families

Import and export these checkpoint formats:

| Family | `model_type` | Notes |
|--------|-------------|-------|
| Qwen 3.5–3.8 | `qwen3_5_text` | Hybrid: GDN (linear attention) + softmax, MoE, vision |
| Qwen Flash-Next | `qwen4_exp_text` | `--arch qwen4-next` export with HC placeholders |
| Gemma 4 | `gemma4_text` | GeGLU activation, QK norms, nested `text_config` |
| LFM 2.5 | `lfm2` | Short-convolution layers + GQA, SwiGLU MLP |
| Nomic Embed | `nomic_bert` | Bidirectional encoder, mean pooling |
| Jev Classifier | `*ForSequenceClassification` | Score head on decoder backbone |
| Binary/Ternary | any decoder | Via `--quant mxfp4`, `--quant ptq1`, or `--quant pq2` |

### Inference & Weight Management

Three weight working-set modes, selected by `AMQL_WEIGHTS` (or `--weights` on `generate`):

| Mode | Resident size | Description |
|------|-------------|-------------|
| `f32` | All weights widened to f32 | Byte-exact reference |
| `bf16` | Stored BF16, widened on access into LRU cache | ~2× smaller |
| `mxfp4` | MXFP4 packs dequantised into LRU cache | ~4× smaller |

When `AMQL_GPU=1` and the CUDA backend is built (`native/amql_cuda/build-cuda.cmd`),
MXFP4 packs are dequantised to FP16 on-device and GEMMs run on tensor cores with FP32
accumulate. Embeddings, norms, and the output head stay f32 in every mode.

Tune memory with `AMQL_MEMORY_GB`, `AMQL_F32_CACHE_GB`, and `AMQL_CORES`.

### Token Relationship Probing

`route` names the relationship between two tokens using template probing and identifies the
exact (layer, head, position) attention coordinates carrying one token's meaning into the
other. Causal tracing then reports per-layer residual weights — the targets to patch or LoRA.

`path` shows the model's own continuation route between two tokens via bidirectional
best-first search over the next-token graph.

### Patching, LoRA & Fine-Tuning

`change-tensor` edits a single weight cell and records the delta as a patch. Every
command (`generate`, `route`, `path`, `inspect-token`, `export`) accepts `--patch` to
apply deltas at load time. `save-lora` factors a patch's 2-D deltas into a LoRA adapter
(`lora_A` / `lora_B`) via truncated SVD.

`fine-tune` performs supervised output-head adaptation: for each (prompt, completion)
pair, the model forward-passes the prompt, teacher-forces each completion token, and
accumulates per-token head deltas — no autograd, no gradient infrastructure, fully
deterministic.

### Model Merging & Transformation

- **`merge`** — Two models into one via consensus-gated token alignment. Shared tokens
  that agree are blended and the result restored to full energy; disagreements keep the
  scaffold's row. The larger model scaffolds the stack shape.
- **`import`** — Merge a second model into an existing container.
- **`moe-ify`** — Turn a dense model into a routed Mixture of Experts by clustering FFN
  co-activations. No training, quality gated by perplexity.
- **`prune`** — Drop whole decoder layers to hit a byte budget. Three ranking approaches:
  provenance (merge history), corpus (residual delta), or random (ablation baseline).
- **`create-model`** — New container with Xavier-uniform random weights and a
  user-defined architecture. Ready for `fine-tune`.
- **`convert-to-classifier`** — Add a random score head to a generative model.
- **`convert-to-embedding`** — Add pooling facts; no new tensors.

### Quantized Export

| Flag | Format | Bits/Weight | Scheme |
|------|--------|------------|--------|
| `--quant mxfp4` | MXFP4 (OCP) | ~4.0 | FP4 E2M1, E8M0 block scales, 32-element blocks |
| `--quant ptq1` | PTQ1_0 (type 143) | ~1.75 | 5 trits/byte (base-3 dense), FP16 scale, 128-block, Hadamard |
| `--quant pq2` | PQ2_0 (type 142) | ~2.13 | 2 bits/trit, FP16 scale, 128-block, Hadamard |

All three quantise stack projections only; embeddings, norms, and the output head stay
full precision. `ptq1`/`pq2` produce PrismML Bonsai-compatible checkpoints for use with
the PrismML llama.cpp fork.

### ONNX Export

`export-onnx` produces a standard `.onnx` model using only built-in operators (no custom
ops). Includes full RoPE (dynamic cos/sin tables), RMSNorm from primitives, SiLU, and
scaled dot-product attention with mask. Compatible with ONNX Runtime 1.21+.

### GUI & Explorer

`amql-gui` provides a WPF desktop interface: command catalog with parameter editing, run
history, and a **Container Explorer** tab. The explorer drills into the VIndex3 structure
(components → layers → objects → edges) with right-click context menus for viewing tensor
values in a spreadsheet grid, browsing the tokenizer vocabulary, and inspecting
hidden-state edge connections.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMQL_WEIGHTS` | `f32` | Weight working-set: `f32`, `bf16`, or `mxfp4` |
| `AMQL_GPU` | `0` | Enable CUDA GPU acceleration |
| `AMQL_MEMORY_GB` | auto | Total memory budget for LRU cache |
| `AMQL_F32_CACHE_GB` | ¼ of budget | F32 LRU cache cap |
| `AMQL_CORES` | auto | CPU core budget for parallel work |
| `AMQL_MERGE_GPU` | `0` | Enable GPU-accelerated merge GEMM |

## Documentation

- [VIndex3 Overview](docs/vindex3-overview.md) — container format, schema, segment layout
- [System Architecture](docs/architecture.md) — layered design, component responsibilities
- [Flash-Next Export](docs/export-qwen3.8-flash-next.md) — qwen4-next architecture mapping
- [Classifier Models](docs/classifier-models-jev.md) — Jev/NLI classification design
- [Embedding Models](docs/embedding-models-nomic-embed-text.md) — nomic-bert encoder
- [CUDA Backend Plan](docs/CUDA_PLAN.md) — GPU kernel roadmap
- [MTP Drafter](docs/MTP_DRAFTER.md) — speculative decoding drafter design

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. The project is research-status
(pre-1.0) — issues and PRs welcome, expect the surface to evolve.