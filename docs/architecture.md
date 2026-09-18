# Architecture Overview

## Design Philosophy

AMQL is a **model-agnostic** framework. Its inference engine is built around a **generic operation program** — a sequence of abstract tensor operations that are independent of any specific model family. The same engine can run Qwen3.5, SmolMol, Gemma, Mistral, and any architecture that can be expressed as a composition of supported operations.

The key insight: model-specific code is pushed to the encoder/planner layer. The inference runtime sees only operations (`EmbeddingOp`, `AttentionOp`, `DenseFfnOp`, `RoutedFfnOp`, etc.) with operand references and resolved parameters.

## Layered Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  CLI (amql-cli)                                                        │
│  ┌─────────────┐ ┌──────────┐ ┌───────────┐ ┌───────────────────┐   │
│  │ encode      │ │ verify   │ │ infer     │ │ inspect / export  │   │
│  └─────────────┘ └──────────┘ └───────────┘ └───────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  VINDEX3 Container                                               │ │
│  │  ┌───────────┐  ┌────────────────┐  ┌────────────────────────┐   │ │
│  │  │ index.json│  │ system_graph   │  │ .segments/*.bin        │   │ │
│  │  │ + profiles│  │ (semantic IR)  │  │ (payload + hashed)     │   │ │
│  │  └───────────┘  └────────────────┘  └────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Encoder / Planner Layer                                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐              │ │
│  │  │ Hf       │  │ Merge    │  │ Inference/Plan   │              │ │
│  │  │ (convert)│  │ (patch)  │  │ (graph → ops)    │              │ │
│  │  └──────────┘  └──────────┘  └──────────────────┘              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Core Libraries                                                   │ │
│  │  ┌──────────────┐ ┌─────────────┐ ┌────────────────────────┐    │ │
│  │  │ Vindex3      │ │ Inference   │ │ Safetensors            │    │ │
│  │  │ (container)  │ │ (IR + plan) │ │ (tensor readers)       │    │ │
│  │  └──────────────┘ └─────────────┘ └────────────────────────┘    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Amql.Safetensors

Low-level safetensors file readers. Provides `NamedTensorData` records and dtype definitions (`FP32`, `BF16`, `Q4_0`, `FP4`, etc.). Used by both the Vindex3 encoder and the Hf converter.

### Amql.Vindex3

The container format layer. Three core subsystems:

1. **Segment writer** (`SegmentWriter`) — single-pass streaming write with dual SHA-256 computation. Handles tensors beyond the 2 GiB single-buffer ceiling by chunking.
2. **Segment reader** (`SegmentFile`) — memory-mapped read with lazy tensor access. Parses JSON header once, reads payload on demand via `ReadBytes()` and `ReadBytes(name, offset, count)`.
3. **Execution surface** (`ExecutionSurface`) — runtime parameters that the planner resolves at build time. Encodes norm types, attention geometry, FFN configuration, MoE routing, and optional operators (linear attention, KDA, MLA, Mamba2).

### Amql.Hf

Hugging Face model converter. Reads from the Hugging Face Hub, converts model weights into VIndex3 container format. Handles the full pipeline from HF tensor naming conventions to the abstract logical object model used by system graphs.

### Amql.Merge

Model patching and merge operations. Supports:

- **Patching** — apply delta tensors to an existing model (e.g. LoRA adapters)
- **Merging** — combine multiple models using different interpolation strategies
- Uses `Amql.Inference`'s graph resolution to ensure semantic consistency across merged components

### Amql.Inference

The inference engine. Two main phases:

1. **Planning** (`Plan.cs`) — converts a VIndex3 container's system graph into a sequence of generic operations (`ComponentOpPlan`). The planner resolves hidden-state edges between components, assigns segment-relative tensor references, and constructs `LayerPlan` structures that encode the full forward pass.

2. **Execution** — executes the operation program using the resolved operand references and surface parameters. The runtime is model-family agnostic.

## Data Flow

### Encoding (Model → Container)

```
HF Model → Hf converter → Logical objects → Vindex3 encoder → Segments
     ↓                                    ↓
  Raw tensors                    System graph (semantic IR)
     ↓                                    ↓
  safetensors                   index.json (authority)
```

### Inference (Container → Output)

```
Container read → Graph parse → Planner → ComponentOpPlan → Execution
     ↓             ↓           ↓           ↓              ↓
  index.json  Components   Attention   Softmax,       Embedding,
  graph.json  Objects      Edge map   FFN ops, norms  Norm ops
  segments/   Edges        Norm       Linear attn     Output head
                          surface     MLA, KDA,       (or tied)
                          spec        Mamba2, etc.
```

## Execution Surface

The `ExecutionSurface` type is the contract between planner and executor. It captures everything needed to run a component without any model-family knowledge:

- **Attention geometry** — query/KV heads, head dimension, scope (per-head vs. full-projection)
- **Norm configuration** — RMS/Layer norm, placement (pre/post), final norm, weight offset convention
- **FFN type** — dense (Gated/DotGated/Glu), MoE (router, experts, top-k, routing policy)
- **Head** — vocab size, embedding tie, output scale, logit soft-capping
- **Optional operators** — linear attention (GatedDeltaNet), KDA, MLA, Mamba2, conv-qkv, residual-in-fp32

An unknown operator appears as a `JsonElement` carry-forward; the planner refuses execution rather than guessing.

## System Graph

The system graph is the semantic IR — a directed graph of components connected at the hidden-state level. It encodes:

- **Component roles** — `PrimaryText` (decoder stack), `Perception` (encoder), `Drafter` (SmolMol-style)
- **Object kinds** — embedding tables, decoder stacks, final norm, output heads, perception towers, expert banks
- **Source bindings** — traceability back to original HF tensor names and byte counts
- **Hidden state edges** — which layers in which component feed into which consumer object, with optional block size

The graph is **optional** for simple single-model containers but required for merged/patched models where cross-component data flow must be explicit.

## Integrity Model

VIndex3's integrity guarantees are two-tier:

1. **Structure validation** — on container open, every cross-reference is verified: representations resolve to known objects, components/objects/edges resolve to each other, segment files exist on disk.
2. **Hash verification** — on-demand via `VerifyIntegrity()`, recomputes payload SHA-256 and whole-segment SHA-256 from disk and compares against the index. Detects both source drift (checkpoint changed) and corruption (encoding differs).

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Memory-mapped reads | Zero-copy tensor access for large payloads; only header parsed upfront |
| Dual SHA-256 | Payload hash enables content-addressing; file hash detects corruption |
| Streaming write | Handles tensors larger than 2 GiB single-buffer limit |
| Schema-gated validation | Fails closed on unreadable schema; no guessing |
| Extra field passthrough | Unknown index fields round-trip verbatim (serde-flatten pattern) |
| Canonical/derived authority | Distinguishes primary containers from those built by patching/merging |
