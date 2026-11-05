# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **feat:** `amql-cli entail find|extract|background|optimise|review|accept` — from a trained model to a subject's Entail and back. `find` ranks the neurons or experts that carry a subject and tests the claim by ablation; `extract` probes a subject's declared relations and writes a table of what the model did, with each answer's softmax margin; `optimise` applies the documented rewrite rules in `docs/entail-optimisation.md`, including deriving a whole-number law by trimmed least squares; `review` puts the documentation and the program in front of any model, and every change, from any author, must pass a gate on held-out agreement, false fires on ordinary text, and cost. On Qwen 3.5 0.8B this derived `a + b`, `a - b` and `a*b` from the model's answers; all 47 rows the program overruled were the model's mistakes, and reloading took unseen two-digit arithmetic from 84% to 100% with 767/767 top-1 agreement on prose
- **feat:** `amql-cli entail decompile|fidelity` — the raw extract: every neuron seen across a corpus as Entail rules and observed `vote`s, with a saved study (`--from`), and its agreement with the model on held-out text. `scripts/entail/python_coverage.py` measures it against Python's own documentation
- **feat:** Entail `vote`, a soft output the interpreter runs and `compile` and `graft` refuse, and the string escapes `\q` and `\{`
- **feat:** Synthetic Python: `scripts/entail/subjects/python.synthetic.entail.txt`, its generator and scoring, and the steps in [docs/entail-synthesis.md](docs/entail-synthesis.md)
- **feat:** `scripts/entail/physics.entail.txt` and `chemistry.entail.txt`, with `make_chemistry.py`, `science_cases.py` and [docs/entail-sciences.md](docs/entail-sciences.md)
- **feat:** `amql-cli entail run --tokenizer <dir> --cases <file>` tests a program on a model's real tokens, without weights
- **fix:** `entail graft` no longer fails on a token that is only ever emitted
- **feat:** `amql-cli entail decompile --only-holdout` records a test set from a corpus without studying it
- **feat:** `amql-cli entail eval --batch N` runs N cases at once on sessions sharing one copy of the weights, with `DecodeSession.Sibling`
- **feat:** `DecodeSession.PrefillLogits` — logits of every position from one batched forward pass
- **feat:** Entail strings take `\n`, and `floor()` is available in constant expressions
- **feat:** `amql-cli entail check|run|compile|graft|eval` and the Entail language (src/Amql.Entail) — statements of what must follow from what, compiled into weights with nothing learned or fitted: `token` → embedding rows, `state` → residual lanes, `attend back` → RoPE-exact attention heads, `entail` → SwiGLU threshold and gate neurons, `emit` → output rows, with `for`/`if`/`def` as generators; a dataflow scheduler places statements in layers and reuses dead lanes. A power-of-four ballast makes every RMSNorm's gain exactly 1, so whole numbers pass bit-exact through the stack. `compile` writes a new, empty model; `graft` widens a Qwen3.5 checkpoint and puts the same layers in front of the host, which stays unchanged wherever the program is silent. `scripts/entail/mathematics.entail.txt` defines + - * / on decimal digits: Qwen 3.5 0.8B goes from 43.8% to 100% on direct arithmetic and 51.7% to 86.7% in chat, with 767/767 top-1 agreement on prose. See `docs/entail.md`
- **feat:** `amql-cli bake-predicate <container> --predicate "<rule>" --out <patch>` — bakes a predicate into the weights by closed-form context distillation. The model reads the rule in context (system message under the chat template, preamble with `--raw`) and generates its own answers to probe prompts (`--prompts`, or `--self-probes N` written by the model for that rule plus generic turns); then, shallow to deep, each chosen layer's FFN down-projection gets the minimum-norm ridge update ΔWᵀ = Kᵀ(KKᵀ + λI)⁻¹T that maps the bare model's FFN activations onto its share of the residual gap to the predicated model at every aligned answer position. Reports KL(predicated ‖ bare) and top-1 agreement before/after with sample answers; the result is a standard weight patch. `LeastSquares.FitMinimumNormUpdate` solves in sample space (n×n rather than dIn×dIn). Also in the GUI catalog under Training

- **feat:** masked diffusion language models — dLLM's A2D-Qwen3 ([`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)). `encode` judges `a2d-qwen3` as Qwen3 with a bidirectional attention surface and a new `diffusion` surface (objective `masked_diffusion`, the mask token id resolved from the tokenizer files, the HF class and `auto_map`); any other `a2d-*` refuses. The planner binds non-causal attention (CPU kernel flag; the GPU kernel is given query positions that mask nothing), step decoding a diffusion plan refuses, and `generate` on such a container runs `MaskedDiffusionSampler` — block-wise low-confidence (or random) remasking as in the model card, `--gen/--steps/--block/--remasking/--temperature/--show-steps`, the vocabulary head projected for the current block only, and an exact early stop once a finished block holds the stop token. Logits match a float32 PyTorch reference (argmax at every position of the real 0.6B, sampled logits within 0.05) and generations match it token for token. `export` restores the class, `auto_map` and top-level `rope_theta` and carries the `modeling_*.py` remote code verbatim (never executed); `to-gguf` writes llama.cpp's `rnd1` arch with the dense FFN as a one-expert MoE, `attention.causal=false`, `diffusion.shift_logits=false` and the mask token — `llama-diffusion-cli` at `--top-k 1` reproduces the reference exactly at F32, BF16 and Q8_0
- **feat:** `synth-model --family a2d-qwen3`
- **feat:** LFM3 / LFM3-VL ([`LiquidAI/LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B)) import, inference, export and GGUF. `encode` judges the three layer types (gated short conv width 7, full attention, sliding-window attention), the per-head sigmoid output gate (`gate_proj`), the sandwich norms (`post_operator_norm` / `post_ffn_norm`) and the zero-tail partial rope (new `zero_tail_rope` position policy: the full head dim's frequency progression with the trailing pairs at frequency zero); the VL wrapper config rides the perception component so `export` rebuilds the checkpoint byte-for-byte (tensors) with an identical config. Prefill and step-by-step decode match a float32 PyTorch forward written from Liquid's `modeling_lfm3.py` (synthetic golden in the tests; the real 2B agrees to the 4th significant figure on CPU and GPU). `to-gguf` writes llama.cpp's `lfm3` arch (`--quant none|f16|q8_0|mxfp4`); with `none` every tensor and tokenizer key is byte-identical to `convert_hf_to_gguf.py`'s output
- **feat:** `synth-model --family lfm3` writes a 4-layer LFM3-shaped checkpoint (conv, full, conv, sliding)
- **feat:** `amql-server` (src/Amql.Server) — serves containers over HTTP, each on the endpoints its facts allow: TypeSafe `/v1/decisions` (+ `/v1/systemone`, and `/api/v1/decisions` in the jevai envelope) for Von, OpenAI `/v1/embeddings` for embedding containers, `/v1/chat/completions` and `/v1/responses` (SSE streaming) for generative decoders; `/v1/models`, `/health`, optional bearer API key. Chat rendering matches HF `apply_chat_template` on Qwen3.5's template; greedy generation matches transformers token for token
- **feat:** `ChatTemplate.ApplyMessages` renders whole conversations (Qwen3 ChatML with think handling, ChatML, Llama 3); `HfTokenizer.DecodeText` decodes through bytes so split characters come out whole; `DecodeSession.PrefillHiddenStates` for pooled embeddings
- **feat:** `amql-cli classify` serves sequence-classification containers (docs/classifier-models-jev.md C1–C6): template → tokenise → last-non-pad pooling → score head (with bias) → softmax / sigmoid / identity by `problem_type`, labelled from the carried `id2label`; `--format labels|json|jsonl|csv`, `--jsonl` batches, `--patch`. Matches transformers on `openjev` qwen3.5-0.8b-nli-v5 to 8e-6 in the logits
- **feat:** classifier ingest carries `score.bias`, the label table, `pad_token_id` and a `classifier.json` mirror, checks every fact against the score tensor, and export regenerates the real labels instead of `LABEL_i`
- **feat:** `amql-cli decide <container> --request <json | @file | ->` — answers TypeSafe `/v1/decisions` (OpenJEV `/v1/systemone`) requests with a Von container; `--envelope jevai` wraps the response as `{ code, message, data }`, invalid requests exit 2 with the API's 422 message. Answers are identical to the Von 1.2 SDK's on the real weights
- **feat:** `ModernBertEncoder` / `OptionMarkerScorer` — a ModernBERT encoder written directly in C# (global + sliding bidirectional attention on position ids, RoPE per layer type, GeGLU with exact GELU, Von's order-invariant option mask); `verify` reports such containers as served
- **feat:** the tokenizer honours `lstrip`/`rstrip` added tokens and `TemplateProcessing` post-processors (`EncodeWithSpecialTokens`)
- **feat:** ModernBERT and the Von decision model ([`wfzyx/von`](https://huggingface.co/wfzyx/von)) import and export — `encode` reads the backbone plus the option-marker scoring head from `option_marker.pt`, `export` writes both back; verified bit-identical (weights and Von's own logits) on the real checkpoint. See `docs/decision-models-von.md`
- **feat:** `TorchCheckpoint` / `TorchCheckpointWriter` — read and write PyTorch zip checkpoints without Python; the reader only accepts the globals a state dict needs
- **feat(gui):** "▶ Run and display inference…" in the container explorer (header button, right-click menu, and *Explorer* menu) — opens the run dialog with the loaded container pre-filled and starts a live run in a new inference visualiser
- **feat:** `amql-cli --progress` (or `AMQL_PROGRESS=1`) — a machine-readable progress/result protocol on stdout: newline-delimited `##amql-progress` and `##amql-result` JSON lines, so front-ends bind progress exactly instead of scraping percentages out of prose
- **feat:** `amql-cli --verbose` — full stack traces for unexpected errors are now opt-in
- **feat:** `to-gguf --force` overwrites an existing output file (idempotent re-runs)
- **docs:** the exit-code table is documented in `amql-cli help`: 0 success, 1 a legitimate negative result (`path` found nothing within budget, `verify` integrity failed), 2 usage/runtime error
- **feat:** `prune --approach star-elastic` — structured FFN-channel width pruning (Star Elastic's zero-shot importance estimation + nested slice): ranks the intermediate channels by activation magnitude `Σ|x·W_upᵀ|` on one forward pass over `--text`, slices the gate/up rows and down columns to the top channels on every dense layer, and writes a smaller complete container under the same exact byte budget + fail-closed floor as layer prune. No elastic training (autograd/distillation is out of scope), so quality is gated, not recovered. See docs/star-elastic-pruning.md

### Changed

- **fix(hf):** configs in the pre-`rope_parameters` layout (top-level `rope_theta`, `rope_scaling`) had their rope facts ignored, so every such checkpoint (Qwen3, Qwen2, Llama exports from transformers ≤ 4.x) imported at theta 10 000; they are now folded into the grouped form, and a legacy `rope_scaling` is carried (and refused by the planner) instead of silently dropped
- **fix(gguf):** the file-type picker counted 2-D weights only and threw on an all-F32 file; expert banks now count by their full element count and an all-F32 file is `ALL_F32`
- **fix(merge):** the embedding alignment applied the transpose of the fitted map — `LeastSquares.Fit` solves the normal equations for Mᵀ but returned it as M, while `ApplyMap`, `ResidualL2` and the GPU map GEMM all compute M·b. Every merge mapped the imported rows through the wrong matrix (LFM3 ← Qwen3.5-0.8B: shared-token agreement read 0.000; with the fix 0.794, matching an independent PyTorch ridge fit)
- **fix(merge):** the CPU Gram/Cross reduction added each worker's partials to the shared totals without a lock, losing updates under load (the "1e-3 map spread on identical input" the GPU parity test worked around)
- **fix(merge):** a cross-family merge no longer grafts the other stack's tensors into the scaffold (they belong to operators the scaffold does not run, and grafting crashed when a tensor would have had to shrink); the other stack stays preserved under `segments/source/`. The reported mean agreement no longer counts non-shared tokens as 0
- **fix(cuda):** the fused GPU attention passed an empty array for "no sinks", which P/Invoke marshals as a valid pointer — the kernel copied `numQHeads` floats of garbage and added them to every head's position-0 score on any prefill of 8+ tokens. Null now means no sinks. Also adds the missing barrier between the softmax max-reduction and the sum-reduction that reuse the same shared buffer
- **fix(inference):** the LFM short conv is the real gated form — `in_proj` → (B, C, x), depthwise causal conv over `B⊙x` with no activation, gated by `C`, kernel width read from the weight — and convolves a whole sequence against its carried history, so conv plans prefill layer-major too. The previous form (conv then SiLU over a hidden-wide projection) matched no LFM checkpoint
- **fix(hf):** the final norm of an LFM checkpoint binds as `embedding_norm`, so export rebuilds the name the checkpoint actually uses
- **inference:** the post-mixer norm (four-norm / sandwich placement) applies after any mixer, not only softmax attention
- **fix(inference):** GatedDeltaNet decays its state before predicting `v` from it, as transformers does — the old order was exact at the first token and drifted from the second, so every multi-token Qwen3.5 forward (generate, classify) was off by up to ~5% in the final hidden state; pinned by a test against `torch_recurrent_gated_delta_rule`
- **fix(tokenizer):** pair-form merges are enumerated once instead of indexed — loading Qwen3.5's tokenizer drops from ~3 minutes to 0.5 s
- **perf(inference):** plans whose only stateful layers are GatedDeltaNet prefill layer-major (bit-identical, ~2.6× faster)
- **perf:** `TensorOps.MatMulTransposedB` walks weight rows in cache-sized blocks (same arithmetic, so bit-identical results); `BitPattern.WidenToF32` copies an F32 payload once instead of twice
- **inference:** planning an encoder container refuses with `UnsupportedOperatorException` naming the encoder stack, so `verify` reports it as not served instead of exiting with an error
- **hf:** the config reader understands ModernBERT's `hidden_activation`, `norm_eps` and period-style `global_attn_every_n_layers` (previously a GELU encoder would have defaulted to SiLU)
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
