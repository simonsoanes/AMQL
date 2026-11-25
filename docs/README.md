# AMQL Documentation

Deeper technical reference for the VIndex3 model container format and AMQL's architecture.

## Contents

- [VIndex3 Container Format](vindex3-overview.md) — The VINDEX3 specification: directory structure, `index.json` schema, `system_graph.json` semantic IR, segment file binary format, integrity model, and encoding support.
- [GPU Inference Performance](gpu-performance.md) — The device-resident engine: single-request prefill and decode against llama.cpp's CUDA backend, what it covers, and the graph, GEMV, launch and recurrence techniques behind it.
- [The KV cache](kv-cache.md) — How big prompt state is and how far it compresses, the NVFP4 KV format and what it costs, and which rows a conversation's later questions actually read.
- [Running a model past its trained context](context-extension.md) — `--max-context`: what it does to a model, the 8K to 1M measurements on Qwen 3.5 4B, what it costs in memory and time, and how to choose a method for another model.
- [CPU Inference Performance](cpu-performance.md) — Single-request prefill and decode speed against llama.cpp, how to measure it with `amql-cli bench`, and the kernel, threading and memory changes behind it, all bit-identical to the code they replaced.
- [Property-driven Model Support](property-driven-models.md) — How AMQL imports, runs and exports a model family it has no code for: tensor roles, config facts, the conventions table, the `hf` / `gguf` / `onnx` export formats, the 33-model zoo that measures it (MoE and length-dependent rope included), and the limits.
- [System Architecture](architecture.md) — Layered design, component responsibilities, data flow diagrams, execution surface, and key design decisions.
- [Embedding Models — nomic-embed-text-v1.5](embedding-models-nomic-embed-text.md) — Ingesting encoder/embedding checkpoints into VIndex3, exporting an embedding model from any container, and the `embed` CLI command (task prefixes, Matryoshka truncation, mean pooling).
- [Server — amql-server](server.md) — Serving containers over HTTP: TypeSafe `/v1/decisions`, OpenAI `/v1/embeddings`, `/v1/chat/completions` and `/v1/responses`, capability detection, routing, streaming, refusals, and how each was verified.
- [AMQL Studio](amql-studio.md) — The desktop app: choosing and serving models with memory limits, the live server monitor, the chat playground and API tester, every CLI command as a form, and the container explorer.
- [N-gram packs](ngram-packs.md) — A domain's next-token statistics blended into the logits (kNN-LM style): building, serving, choosing per request, and why a .NET pack did not raise build success for a 0.8B model (n-grams do not know types).
- [Decision Models — Von](decision-models-von.md) — Importing, exporting and running the Von option-marker decision model (ModernBERT backbone + `option_marker.pt` scoring head): the PyTorch checkpoint reader/writer, the in-process encoder, `amql-cli decide` for TypeSafe `/v1/decisions` requests, and how each was verified against the real weights.
- [Classifier Models — Jev sequence-classification](classifier-models-jev.md) — Ingesting classifier/sequence-classification checkpoints (Qwen3_5ForSequenceClassification, openjev-style) into VIndex3, the `score` head + label table + last-non-pad pooling contract, and the `classify` / inspect / export paths.

## Quick Reference

| Document | For |
|----------|-----|
| [vindex3-overview.md](vindex3-overview.md) | Understanding the container format, segment layout, schema versions, and encoding (FP32/BF16/Q4_0/FP4) |
| [architecture.md](architecture.md) | How AMQL's components fit together, the planner, and the model-agnostic inference model |
| [embedding-models-nomic-embed-text.md](embedding-models-nomic-embed-text.md) | Embedding-model support: the nomic-embed-text-v1.5 contract, schema additions (encoder stack, bidirectional attention, pooling), the `embed` command, and the two export tiers |
| [decision-models-von.md](decision-models-von.md) | Von / ModernBERT: the checkpoint layout, `encoder` and `option_marker` surface facts, `TorchCheckpoint`, and what is (not) served |
| [classifier-models-jev.md](classifier-models-jev.md) | Classifier-model support: the Jev/openjev `score`-head contract, `ClassifierSurface` and last-non-pad pooling schema additions, the `classify` command, and classifier create/inspect/export |
