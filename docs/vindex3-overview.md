# VIndex3 Container Format

## Overview

VIndex3 (VINDEX3) is AMQL's custom container format for storing and managing model data independently of any single model family. It provides a schema-gated, integrity-verified structure that separates the semantic description of a model from its physical tensor storage.

The format is inspired by [Chris Hay's Larq](https://www.kavita-gupta.com/larq.html) research and is implemented here as a C# port. The container is the sole root authority for all model representations within it.

## Directory Structure

A VIndex3 container is a directory containing:

| File | Purpose |
|------|---------|
| `index.json` | Root authority — format version, model identity, representation directory, segment census |
| `system_graph.json` | Semantic IR — components, logical objects, hidden-state edges (optional, present when the container has a graph) |
| `.segments/` | Directory containing individual `.bin` segment files, one per representation |

## index.json Schema

The index is a JSON document with the following top-level fields:

### Version and Model Identity

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `Version` | int | Yes | Schema version (current: 4). Must be in range `[MinReadableSchema, CurrentSchema]`. |
| `Model` | string | Yes | Model identifier (e.g. `"Qwen/Qwen3.5-14B-A2.7B"`). |
| `Family` | string | Yes | Model family (e.g. `"Qwen3.5"`). |
| `HiddenSize` | int | Yes | Hidden dimension size. |
| `NumLayers` | int | Yes | Number of layers. |
| `DerivedFromModel` | string? | No | Parent model for derived containers (built from another container's representations). |

### Representation Directory

`Representations` maps `{object}@{encoding}` string keys to `RepresentationEntry` objects:

| Field | Type | Description |
|-------|------|-------------|
| `Object` | string | Logical object ID from the system graph. |
| `Encoding` | string | Storage encoding (e.g. `"BF16"`, `"Q4_0"`). |
| `Segment` | string | Relative path to the segment file. |
| `TensorCount` | int | Number of tensors in this representation. |
| `PayloadBytes` | long | Payload size in bytes. |
| `PayloadSha256` | string | SHA-256 of the payload region only. |
| `SegmentSha256` | string | SHA-256 of the entire segment file (header + payload). |
| `CompiledFrom` | string? | Optional source model reference. |

### Selection Profiles

`Profiles` is a list of selection profiles, each with a `Name` and `Selects` dictionary mapping object IDs to representation IDs. The `"exact"` profile is always present by default.

### Segment Census

`Segments` maps path stems to physical file counts for integrity auditing.

### Precision Map

`PrecisionMap` documents the stored-precision policy — which encoding is canonical and which segment-relative tensor names deviate (e.g. Qwen3.5 keeps `A_log` and the recurrent norm in F32 inside a BF16 stack).

### Authority

`Authority` indicates whether the container is `Canonical` (primary authority) or `Derived` (built from another container).

## System Graph Schema

The optional `system_graph.json` contains the semantic intermediate representation of the model system. Schema: 6 (current).

### Components

Each component has:
- `Id` — unique identifier
- `Role` — `PrimaryText`, `Perception`, or `Drafter`
- `SourceArtifact` — the original model artifact it came from
- `NumLayers` — layer count
- `HiddenSize` — hidden dimension
- `Attention` — per-layer attention policy table (optional, absent for perception towers)
- `Execution` — the resolved execution surface (schema ≥ 6 only)
- `Perception` — opaque JSON for modality/transform facts

### Logical Objects

Each logical object has:
- `Id` — conceptual identity (`{component}.{kind}`)
- `Component` — owning component ID
- `Kind` — `Embedding`, `DecoderStack`, `FinalNorm`, `OutputHead`, `PerceptionTower`, `PerceptionAdapter`, `FeatureProjector`, or `ExpertBank`
- `SourceBindings` — physical trace (artifact, tensor prefix, tensor count, bytes)
- `Representations` — materialisations with encoding and fidelity

### Hidden State Edges

Edges connect components at the hidden-state level:
- `ProducerComponent` — source component
- `ProducerLayers` — producing layer indices
- `ConsumerComponent` — target component
- `ConsumerObject` — consuming logical object
- `BlockSize` — optional block size

### Component Roles

| Role | Description |
|------|-------------|
| `PrimaryText` | Main text-generation component (decoder stack, head, etc.) |
| `Perception` | Modality perception tower (e.g. vision encoder) |
| `Drafter` | SmolMol-style drafter component |

### Object Kinds

| Kind | Description |
|------|-------------|
| `Embedding` | Token embedding table |
| `DecoderStack` | Multi-layer transformer stack |
| `FinalNorm` | Final normalisation layer |
| `OutputHead` | Projection to vocabulary space |
| `PerceptionTower` | Modality perception encoder |
| `PerceptionAdapter` | Adapter between modalities |
| `FeatureProjector` | Feature projection (e.g. LoRA adapter) |
| `ExpertBank` | Mixture-of-experts expert bank |

## Segment File Format

Segments are `.bin` files with the following layout:

```
[8 bytes: u64 LE — padded header length]
[padded header JSON]
[tensor payload data, in sorted name order]
```

### Header Format

The JSON header contains:
- `schema` — segment format version (current: 1)
- `representation` — representation ID string
- `tensors` — array of tensor entries, each with:
  - `name` — tensor name relative to the object
  - `dtype` — storage dtype label (safetensors uppercase spelling)
  - `shape` — array of dimension lengths
  - `offset` — byte offset from payload start
  - `len` — payload length in bytes

### Physical Layout Rules

- Header length is stored as a 64-bit little-endian integer at bytes 0–7.
- Header JSON is space-padded so that `8 + headerLen ≡ 0 (mod 16)`.
- Payload region starts at byte `8 + storedHeaderLength` (always 16-byte aligned).
- Tensors are written in sorted name order; offsets remain relative to payload start.
- Two SHA-256 hashes are computed incrementally during a single streaming pass: payload-only and whole-file.

### Why Streaming?

A segment can exceed the 2 GiB single-array ceiling — Qwen3.8-27B's decoder stack is 48 GB. The hashes are computed over streams, never over a whole-file buffer. Large tensors are written in chunks when their payload exceeds the single-buffer limit.

## Integrity Verification

VIndex3 supports byte-equivalence integrity checking (G4-style):

1. **Structure validation** (on open): Every recorded reference resolves — index representations name known objects (when a graph exists), graph components/objects/edges resolve to each other, and every representation entry's segment file exists on disk.

2. **Hash verification** (on-demand via `VerifyIntegrity()`): Payload SHA-256 and whole-segment SHA-256 are recomputed from disk and compared against the index. A drifted checkpoint (source ≠ recorded) and a corrupted container (encoded ≠ recorded) both fail here.

## Schema Evolution

| Type | Current | Min Readable |
|------|---------|--------------|
| Index schema | 4 | 3 |
| Segment format | 1 | 1 |
| System graph | 6 | 6 |

Unknown fields in the index round-trip verbatim (mirrors the reference's `#[serde(flatten)] extra`). Schema-gated validation fails closed — an unreadable schema is a typed error, never a guess.

## Encoding Support

VIndex3 supports multiple tensor dtypes including:
- `FP32`, `FP16`, `BF16` — full and half-precision floating point
- `Q4_0`, `Q8_0` — quantised formats
- `FP4` — 4-bit float with packed representation (two elements per byte)

A precision map may be recorded to document when some tensors legitimately deviate from the canonical stack encoding.
