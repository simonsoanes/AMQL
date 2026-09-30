# VIndex3 Slim Containers — Design

**Status:** design · **Date:** 2026-09-30

## 1. Motivation

A full VIndex3 container stores every imported encoding for every logical object
— the canonical BF16 (or FP32) plus any quantised variants the user requested
(Q4_0, Q8_0, FP4). This makes the container the single authority for every
precision level, but it means a container that was imported at BF16 *and* Q4_0
holds ~1.5× the payload of a BF16-only container even when the user only ever
runs inference at Q4_0.

A **slim container** stores exactly one encoding per object — the one the user
asked to import or the one a post-hoc strip retained. It is not "compressed" in
the data-compression sense; it is **sparse**: it simply omits the encodings it
does not need. The goal is that a Q4_0-only slim container of a 7B model should
be roughly the same size on disk as the equivalent Q4_0 GGUF (~3.6 GiB for
Qwen3.5-7B at Q4_0), plus VIndex3's intrinsic overhead (segment headers, index,
system graph — typically < 1 MiB).

## 2. Non-goals

- **Compression.** Re-compressing the tensor payload with zstd/lz4 — the tensors
  themselves are already quantised; shaving another 10-20% off the segment files
  would break memory-mapped zero-copy access (the current design's key strength).
- **Mixed encodings within a slim container.** A slim container is single-encoding
  by construction. If a user wants both Q4_0 and FP4, they keep the full
  container or create two slim ones.
- **On-the-fly re-quantisation.** The slim container is produced at encode time
  (or via a `strip` command that drops segments), never by re-reading and
  re-encoding the payload.

## 3. Design principle

A slim container is a **valid VIndex3 container at schema level 4 (or 7 for the
graph)** with three differences from a full container:

1. **Single representation per object.** Every `LogicalObject` has exactly one
   `Representation` (not two or three).
2. **Single segment per object.** The `.segments/` directory has exactly one
   `.bin` file per object (plus maybe one for the classifier head or expert bank).
3. **Explicit slim marker.** The `index.json` carries an `"Authority": "Slim"`
   field so tooling knows this container is intentionally incomplete. A slim
   container is not "derived" from another — it is its own authority for the
   encoding it carries.

### 3.1 Why not "Derived"?

The current `Authority.Derived` means "built from another container's
representations" (merge output, patched model). A slim container is built
directly from a checkpoint, at a specific encoding — it is not derived from
another container. Reusing `Derived` would confuse the merge/patch tooling.

### 3.2 Container naming convention

The family name or a suffix identifies the encoding:

```
<model>-<quant>.amql/          e.g.  Qwen3.5-7B-Q4_0.amql/
<model>/                        e.g.  Qwen3.5-7B/          (full, all encodings)
```

This mirrors GGUF's `q4_0.gguf` suffix convention. The `index.json`'s
`PrecisionMap` is still the authority for which tensor names deviate from the
stack encoding, so a file-system-level convention never contradicts the payload.

## 4. Implementation paths

There are two ways to produce a slim container:

### Path A: Encode directly at the requested encoding (preferred)

`amql-cli encode <checkpoint> --dtype Q4_0` imports only the Q4_0 encoding.
Internally:

1. `ModelToContainer.Encode` still reads the safetensors at their native dtype
   (BF16 or FP32).
2. For each logical object's weight tensors, instead of storing the canonical
   BF16 representation and optionally quantising from it, the encoder quantises
   on the fly and stores ONLY the requested encoding.
3. The `Representation` list for each `LogicalObject` has one entry: `Q4_0`
   with `Fidelity.Approximate`.
4. The `PrecisionMap` records the stack encoding as Q4_0, with F32 exceptions
   for embeddings, norms, `A_log`, and the output head (these stay F32 because
   they are small and precision-sensitive — exactly the same exceptions the
   MXFP4 working set already makes).

**Pros:** Single pass. No temporary full container on disk.  
**Cons:** Requires the quantisation code path to work during encode (Q4_0 is
already used by GGUF export, so the quantiser exists; FP4/MXFP4 also exists;
Q8_0 exists).

### Path B: Strip an existing full container (post-hoc)

`amql-cli strip <container> --keep Q4_0` produces a slim copy:

1. Read the full container's `index.json` and system graph.
2. For each logical object, keep only the representation whose encoding matches
   `--keep`.
3. Copy the corresponding segment file into a new `.segments/` directory.
4. Write a new `index.json` with `Authority: Slim`, the reduced representation
   map, and updated `Segments` census.
5. Copy `tokenizer.json`, `classifier.json`, and other ancillary files verbatim.

**Pros:** Works on any existing container. No re-encoding. Deterministic
output from a given container (byte-identical segments).  
**Cons:** Requires the full container to exist first (2× disk space transiently).

### Recommendation

Support both. `encode --dtype Q4_0` is the direct path for new imports;
`strip --keep Q4_0` is the migration path for existing containers. They share
the same output structure.

## 5. Schema changes

### 5.1 `Authority` enum (index.json)

Add `Slim`:

```csharp
public enum ContainerAuthority
{
    Canonical,   // primary authority, all encodings
    Derived,     // built from another container (merge, patch)
    Slim,        // single-encoding subset of what a full import would hold
}
```

Existing containers are unaffected. The `Canonical` vs `Slim` distinction tells
tooling: "verify" on a slim container checks the stored encoding only, not "the
canonical BF16 is missing."

### 5.2 Graph schema

No change. The system graph records the same components, objects, and edges.
The `Representation` list simply has one entry instead of two or three. Schema
7 already allows this — the `Representations` field is a `List<Representation>`
with no minimum-length constraint.

### 5.3 `index.json` — `PrecisionMap`

The `PrecisionMap` carries the stack encoding (e.g., `Q4_0`) and the
per-tensor-name exceptions (e.g., `A_log: F32`, `model.norm.weight: F32`).
This already exists and works for a single-encoding container.

## 6. Segment layout

No change. A slim container uses the exact same segment binary format (8-byte
header length + padded JSON header + tensor payload). The only difference is
that there is one segment per object instead of N.

## 7. File-level structure

```
Qwen3.5-7B-Q4_0.amql/
  index.json           Authority: Slim, PrecisionMap: Q4_0 + exceptions
  system_graph.json    unchanged (schema 7)
  .segments/
    target.embedding.bin     Q4_0 (or F32 if the embedding is an exception)
    target.decoder_stack.bin Q4_0 (with F32 exceptions for A_log, norms)
    target.final_norm.bin    F32
  tokenizer.json        copied verbatim
  classifier.json       if classifier
```

## 8. Size estimates

### 8.1 Qwen3.5-7B at Q4_0

| Component | Elements | Q4_0 bytes | Notes |
|-----------|----------|-----------|-------|
| Embedding | 152064 × 4096 | ~311 MB | F32 kept (~623 MB) as exception |
| Decoder stack (28 layers) | ~28 × 12 × 4096² | ~5.6 GiB | projections only |
| Final norm | 4096 | ~16 KB | F32 |
| Output head | tied to embedding | 0 | head reuses embedding |
| **Total (Q4_0 for projections)** | | **~5.9 GiB** | |
| **Total (with F32 embedding)** | | **~6.2 GiB** | |

Equivalent Q4_0 GGUF for the same model: ~3.6 GiB for the weight data. The
difference is the F32 embedding (which GGUF also keeps at F32, and the
embedding is ~623 MB), plus per-tensor block scales in Q4_0 which add
~1/16 overhead. VIndex3 Q4_0 uses the same block size (32) as GGUF Q4_0, so
the per-block scale overhead is identical. The remaining gap is:

- Segment headers (one per object, ~1 KB each, negligible)
- `index.json` and `system_graph.json` (< 1 MiB together)
- VIndex3 stores BF16/F32 norms and `A_log` at full precision (~20 MB for 7B)

Expected: a Q4_0 slim container of Qwen3.5-7B should land at **~3.8–4.0 GiB**,
which is within ~10% of the equivalent GGUF. This is the "similar relative
size" target.

### 8.2 Qwen3.5-0.8B at Q4_0

| Component | Elements | Q4_0 bytes |
|-----------|----------|-----------|
| Embedding | 151936 × 896 | ~68 MB raw, ~34 MB Q4_0 (if quantised) |
| Decoder stack (16 layers) | ~16 × 12 × 896² | ~155 MB |
| Final norm | 896 | ~3.5 KB |
| Output head | tied | 0 |
| **Total** | | **~255 MB** |

Equivalent GGUF: ~250 MB. Slim container: ~255–270 MB.

### 8.3 Qwen3.8-27B-pruned at MXFP4

The pruned 24-layer 27B is ~6.2 GiB of MXFP4 stack projections. A full
container (BF16 + MXFP4) is ~48 GiB + 6.2 GiB ≈ 54 GiB. A slim MXFP4-only
container is ~6.2 GiB + ~1 GiB F32 exceptions (embedding, head, norms,
`A_log`) ≈ **~7.2 GiB**. This makes the 27B distributable on consumer media.

## 9. Tooling

### 9.1 Encode

```
amql-cli encode <checkpoint> [--dtype Q4_0|Q8_0|FP4|BF16|FP32]
```

`--dtype` selects the single encoding to import. Without it, the current
behaviour (canonical BF16) is unchanged.

### 9.2 Strip

```
amql-cli strip <container> --keep Q4_0 [--output <slim-dir>]
```

Produces a slim container at `--output` (default: `<container>-Q4_0`).
Refuses if `--keep` names an encoding the container does not hold.

### 9.3 Inspect

```
amql-cli inspect <slim-container>
```

Reports `Authority: Slim` and the single encoding. The existing `--verify`
checks the segments present; it does not fail because BF16 is absent (the
authority is Slim, not Canonical).

### 9.4 Export

```
amql-cli export <slim-container> --format safetensors
```

Exports the tensors at their stored dtype. A downstream tool receiving Q4_0
safetensors must handle the quantised format (just as it must for a GGUF).

## 10. Verification

- `encode --dtype Q4_0` of a known checkpoint → container opens, `inspect`
  reports `Authority: Slim`, `--verify` passes.
- `strip --keep Q4_0` of an existing full container → byte-identical segments
  to the encode path for the same checkpoint and encoding.
- Slim container `generate` produces the same output as the full container at
  the same encoding (tolerance-gated for quantised encodings, bit-exact for
  BF16/FP32).
- Slim container size is within 15% of the equivalent GGUF for Q4_0 and Q8_0.
- A `Canonical` container whose BF16 segments are deleted → `open` fails
  structure validation (segments named in index don't exist). A `Slim`
  container whose single encoding segment is deleted → same failure. The
  integrity guarantee is unchanged: every recorded reference resolves.

## 11. Risks

1. **Q4_0 quantisation during encode.** The encoder currently stores BF16 and
   optionally quantises to MXFP4 for the working set. Q4_0 quantisation for
   GGUF export exists in `Amql.Gguf.GgmlQuant`. The encode path would need to
   call this quantiser and write the result directly into the segment. Risk:
   `GgmlQuant` operates on `float[]` widened from safetensors; this is already
   how GGUF export works, so the code path exists. The only new work is wiring
   it into `ContainerEncoder`.

2. **PrecisionMap exceptions.** Q4_0 quantisation of 1-D tensors (norms) is
   undesirable — quantising a 4096-element RMSNorm weight to 4 bits is a
   rounding error on top of a small vector. The `PrecisionMap` must mark these
   as F32 exceptions. The encode path already builds the precision map for
   MXFP4; the same logic applies to Q4_0.

3. **Untied embedding.** Models with `tie_word_embeddings: false` have a
   separate output head tensor of size `[vocab, hidden]` (hundreds of MB).
   Quantising this to Q4_0 degrades output quality measurably; it should stay
   F32 (or BF16) even in a slim container. The `PrecisionMap` handles this.