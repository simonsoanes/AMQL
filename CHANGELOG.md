# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **feat(gui):** "▶ Run and display inference…" in the container explorer (header button, right-click menu, and *Explorer* menu) — opens the run dialog with the loaded container pre-filled and starts a live run in a new inference visualiser
- **feat:** `amql-cli --progress` (or `AMQL_PROGRESS=1`) — a machine-readable progress/result protocol on stdout: newline-delimited `##amql-progress` and `##amql-result` JSON lines, so front-ends bind progress exactly instead of scraping percentages out of prose
- **feat:** `amql-cli --verbose` — full stack traces for unexpected errors are now opt-in
- **feat:** `to-gguf --force` overwrites an existing output file (idempotent re-runs)
- **docs:** the exit-code table is documented in `amql-cli help`: 0 success, 1 a legitimate negative result (`path` found nothing within budget, `verify` integrity failed), 2 usage/runtime error

### Changed

- **cli:** `change-tensor` collects every argument-validation problem and reports them in one run instead of one error per invocation
- **cli:** `verify` skips shape-specific operand probes that the container does not have (e.g. the 2-layer `synth-model` demo, a stack without linear attention) instead of failing after integrity passed
- **cli:** `route` and `path` progress fragments are newline-terminated and flushed under redirected stdout, so a piped consumer sees updates as they happen
- **gui:** AMQL Studio requests the progress protocol, binds the bar to it (heuristic parsing kept as fallback), and records exit 1 as `Negative` rather than `Failed`
- **build:** the solution builds warning-clean — the eight `HfCheckpointToGguf` nullability warnings are gone (`PlanEntry.RequiredSource` throws a typed error on a malformed plan)

### Removed

- **cli:** unexpected errors no longer print a full stack trace by default — one `error: …` line plus a hint to re-run with `--verbose`

---

## v0.x — Changelog (from git history)

### 2026-09 — Calculated MTP drafter & MoE

- **feat:** finish the calculated MTP fit (phase 3) — `generate-mtp --fit` one-shot CLI (boot + collect + fit, acceptance gate ships the K)
- **feat:** add the K-projector mixture to the calculated MTP fit (phase 2) — balanced clusters, per-cluster ridge experts, argmax router; held-out acceptance 14.8% → 16.4% → 16.8% as K grows 1 → 4 → 8 on the real 0.8B
- **feat:** fit the calculated MTP drafter's free block by ridge least squares (phase 1) — closed-form projector, held-out acceptance 0.0% → 14.8% on the real 0.8B
- **feat:** collect the calculated MTP drafter's continuation pairs (phase 0) — deterministic, held-out split
- **feat:** `moe-ify` a dense container into a routed mixture of experts (clustered experts + materialised router + PPL gate)
- **docs:** record the measured Qwen3.5-0.8B zero-shot MTP draft acceptance in the generate-mtp section
- **docs:** list amql-cli prune and generate-mtp in the CLI examples summary so the command index matches the usage/help text

### 2026-09 — Pruning & large-model support

- **feat:** add `amql-cli prune` — drop decoder layers to a byte budget (`--target-bytes` with provenance/corpus/random ranking, exact pre-commit size verification) so the merged-container workflow closes: merge two large models, prune, then import a smaller model into the pruned container
- **feat:** import Qwen3.8-27B (untied head, chunked payloads, paged export past the 2 GiB ceiling)
- **perf:** parallelize the export payload pass by core count, two cores spare over ten
- **docs:** Revise README for model version and feature updates

### 2026-09 — Quantisation

- **feat:** export the model as MXFP4 (OCP microscaling standard) instead of NVFP4
- **feat:** export the model as an NVFP4 checkpoint (`--quant nvfp4`)

### 2026-09 — Model merging

- **feat:** consensus-gated token merge in the scaffold's space with energy reinforcement
- **feat:** import a second model into a container via anchored-alignment merge; stream large file writes
- **docs:** Updated documentation

### 2026-09 — Model export & inspection

- **feat:** export the patched model back to a checkpoint; add `layers` command
- **feat:** keep the MoE facts in the exported config so re-encoding retains the routed judgment
- **docs:** merge: integrate the README revision with the moe-ify documentation

### 2026-09 — Patching & LoRA

- **feat:** manual weight patches, patch-aware pathways, and LoRA export

### 2026-09 — Path finding

- **feat:** bidirectional-best-first path search between tokens; container-carried tokenizer

### Earlier

- **feat:** `route` — relationship probing between tokens (patch-targeting weights)
- **cli:** clarify container vs checkpoint dirs; rename `--model-dir` to `--tokenizer`
- **build:** publish amql-cli as a single-file self-contained EXE
- **docs:** Initialize README with project details
- **docs:** Revise README title and improve project description

---

## Versioning Note

AMQL is pre-1.0 research software. Version numbers before 1.0 indicate major structural changes. API surface (CLI commands, container format) may change between minor versions until 1.0.
