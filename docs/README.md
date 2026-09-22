# AMQL Documentation

Deeper technical reference for the VIndex3 model container format and AMQL's architecture.

## Contents

- [VIndex3 Container Format](vindex3-overview.md) — The VINDEX3 specification: directory structure, `index.json` schema, `system_graph.json` semantic IR, segment file binary format, integrity model, and encoding support.
- [System Architecture](architecture.md) — Layered design, component responsibilities, data flow diagrams, execution surface, and key design decisions.
- [Embedding Models — nomic-embed-text-v1.5](embedding-models-nomic-embed-text.md) — Ingesting encoder/embedding checkpoints into VIndex3, exporting an embedding model from any container, and the `embed` CLI command (task prefixes, Matryoshka truncation, mean pooling).

## Quick Reference

| Document | For |
|----------|-----|
| [vindex3-overview.md](vindex3-overview.md) | Understanding the container format, segment layout, schema versions, and encoding (FP32/BF16/Q4_0/FP4) |
| [architecture.md](architecture.md) | How AMQL's components fit together, the planner, and the model-agnostic inference model |
| [embedding-models-nomic-embed-text.md](embedding-models-nomic-embed-text.md) | Embedding-model support: the nomic-embed-text-v1.5 contract, schema additions (encoder stack, bidirectional attention, pooling), the `embed` command, and the two export tiers |
