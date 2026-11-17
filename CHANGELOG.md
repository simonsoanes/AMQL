# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **feat:** `lora-train`, LoRA training in AMQL without an autograd engine, on FFN down-projections. The backward pass through the output head and final norm is exact, so the last layer's gradient is exact; earlier layers follow the direct residual path. Every run checks its gradient against finite differences at the first step and logs the result (`train.log.jsonl`), and also caches the frozen prefix, guards against loss spikes, stops early and supports self-anchors: unlabelled prompts the base model answers itself, kept as targets. Writes an `amql-lora-v1` adapter and a patch.
- **feat:** `entail from-lora` writes a LoRA as Entail: one state per rank component, with the token pairs it fires on, its column read through the final norm and head as votes, and, by occlusion, which parts of the prompt the decision depends on.
- **feat:** `tool-trace` traces a model's own tool call: attention over each tool's name and description, the request and the format instructions at the call, tool and argument decisions, and request-to-description similarity by layer.
- **feat(entail):** `carry DEST from SOURCE`, a lane that starts each token with another lane's value from the token before, so a program can fold over a text of any length. Runs in the interpreter, the sparse adapter and the new `entail run --file`, which streams a file in fixed memory; compiled weights refuse it. `scan.entail.txt` answers from anywhere in an 8 MB text with 2,640 bytes of state.
- **docs:** [entail-strategies.md](docs/entail-strategies.md), the tool-calling and long-context experiment on Qwen 3.5 0.8B.
- **feat(entail):** `prefer "TEXT" at P when LANE` and `decide "TEXT" at K times prior when LANE`, outputs decided by the host's own probability. `prefer` says TEXT when the host gives it at least P. `times prior` makes the threshold relative to the host's prior: the same conversation with the latest message blanked to "N/A", measured with one more prefill (`ChatPrior`, `SideBySideSession.PriorPrompt`). `decide` also rules TEXT out below the threshold. They run in both adapter forms; `compile` and `graft` refuse them.
- **feat:** `scripts/entail/toolcall/call-when-likely.entail.txt`, a fix for the 0.8B model not calling tools, with no word lists. A 10-neuron fold finds the reply's first token and applies `decide "<tool_call>" at 2 times prior`. With `qwen3.5-calls.jinja`, Qwen's template minus two lines that invite prose, a held-out set of implied requests goes from 23 to 36 calls of 36 with no new false call. On 180 decision points from real policy conversations, calls go from 64 to 94 of 120 and right tools from 46 to 70, with 4 more calls among 60 text turns.
- **feat:** `entail eval --watch TOKEN` records a token's probability on the reply's first row (and the prior's, when one is measured), plus the five likeliest tokens there. `--prefix TEXT` forces the reply's opening, and `--logprobs` records each reply token's probability.
- **feat:** `lora-train` can run with CUDA on. The LoRA is now its own f32 low-rank path beside the frozen down-projection (`GenericRuntime.FfnAdapter`), not a delta added into the weight. So the managed path's device GEMMs and f16 weight copies never go stale and never round the update away. A 4-step run on Qwen 3.5 0.8B gives the same losses on CPU and GPU (0.2362 → 0.0044), with gradient checks within 0.14%. It is not yet faster: attention, the Gated DeltaNet and the head backward stay on the CPU. The next step is the device engine exposing an FFN-input capture and an FFN add, the seam this change defines.
- **feat:** `make_real_cases.py` (decision points from recorded multi-turn tool conversations), `make_need_cases.py` (a held-out omission set), `variants.py` (prompt variants), `score_real.py`, `watch_auc.py` and `draft_features.py`.

### Fixed

- **fix(gpu):** the device engine replayed a stale decode graph after a prefill grew the KV cache. A session that had decoded a short prompt, and was then reset and given a longer one (past 1,024, 2,048, … rows), failed its next step with CUDA status 700 or read freed memory. `GpuEngineTests.Engine_Recaptures_After_A_Prefill_Grows_The_Cache` covers it.
- **fix(entail):** `entail run --tokenizer` dropped special tokens such as `<|im_start|>` when splitting a text, so no program could read a chat template's markers. Adapters were not affected.

- **perf:** the device engine runs a Gated DeltaNet over a prompt in its chunked (WY) form. On Qwen 3.5 0.8B (RTX 5090) prefill goes from ~58,000 to ~64,000 tok/s at 512 and 2,000 tokens (+11%), and from ~30,000 to ~31,500 with `AMQL_GPU_PREFILL=f32`.
  - Chunks are 64 rows. One kernel builds every chunk's W, U, decayed keys and masked Q·Kᵀ at once, solving the chunk's triangular system in 16-row blocks.
  - A second kernel carries the state from chunk to chunk: two small products per chunk instead of a step per row. The next chunk's operands load (`cp.async`) while the current one computes.
  - A third kernel forms every chunk's outputs at once from its starting state.

  Decays stay in log space, and everything is f32: with `AMQL_GPU_PREFILL=f32` logits agree with the managed path to ~2e-5 over 2,000 tokens. It runs for prompt chunks of 256 rows or more; shorter ones keep the row scan, which is faster there. `AMQE_GDN_CHUNK` sets that threshold (`0` turns the chunked form off). The arch zoo gains `qwen3_5-gdn`, a hybrid with 64-wide linear-attention heads, so `GpuEngineTests` covers the engine's Gated DeltaNet. It now also runs an 800-token prompt, which takes the chunked form in two calls.
- **perf:** device-engine prompts are 2–4× faster than llama.cpp's CUDA backend. On Qwen 3.5 0.8B (RTX 5090) prefill goes from ~5,800 to ~7,500 tok/s at 32 tokens (llama.cpp: 1,930), from ~20,000 to ~55,000 at 512 (20,700) and from ~15,000 to ~54,000 at 2,000 (26,500).
  - Prompt projections run as BF16 tensor-core GEMMs on the weights as stored (`cublasGemmEx`, f32 accumulation), instead of widening the weights to f32 for SGEMM.
  - Prompt attention runs as strided-batched GEMMs per KV-head group (TF32) around a masked softmax kernel, instead of a scalar tiled kernel.
  - The Gated DeltaNet recurrence over a prompt has its own scan: 8 lanes per value column, 8 columns per block, and the next row's inputs loaded ahead.

  Rounding prompt activations to BF16 moves logits by ~0.3% of the largest (greedy tokens agree at 32, 512 and 2,000 tokens). `AMQL_GPU_PREFILL=f32` or `DecodeSession.ExactGpuPrefill` keeps prompts at f32 (max logit diff ~2e-5), at ~28,000 tok/s for 512 tokens. `GpuEngineTests` checks both modes. The native library is now copied beside the binaries on build.
- **perf:** a device-resident GPU engine (`GpuEngine`, `native/amql_cuda/amql_engine.cu`) that `DecodeSession` uses when CUDA is available and the plan is within reach. Single-request decode is faster than llama.cpp's CUDA backend: on Qwen 3.5 0.8B (RTX 5090) it goes from ~41 to ~655 tok/s (llama.cpp, F16, flash attention: ~500), and prefill reaches ~5,800 tok/s at 32 tokens (2,227). At 512 tokens prefill is ~20,000 tok/s, ~10% behind llama.cpp's 22,072. How it works:
  - Weights are uploaded once as BF16 and stacked where projections share an input. Activations, the KV cache and the Gated DeltaNet state stay on the device.
  - Each decode step is one CUDA graph, with kernels reading the position from a device counter.
  - GEMVs fuse the preceding norm and their epilogues, and split K for narrow matrices.
  - Programmatic dependent launch lets each projection prefetch its weights into L2 while the previous kernel finishes.
  - Decode attention is split by key; prefill attention is tiled.
  - The GDN conv, q/k norm, recurrence and gated norm take two kernels when decoding.

  It covers dense FFNs, softmax attention (GQA, output gate, QK norms, rotary, windows, softcaps), Gated DeltaNet, every norm placement and residual scaling; routed experts, short convs, length-dependent rope, traces and patches stay on the managed path. Results agree with the managed path to rounding (max logit diff ~2e-5 on Qwen 3.5 0.8B, greedy tokens identical); `GpuEngineTests` checks every dense arch-zoo family. `AMQL_GPU_ENGINE=0`, `DecodeSession.AllowGpuEngine`, `Backend` and `GpuRefusal`. See [docs/gpu-performance.md](docs/gpu-performance.md)
- **feat:** `amql-cli bench --compare` runs the same greedy request on the managed path and reports token agreement and the largest logit difference; `bench` prints which backend ran. `Sampler.ArgMax` is vectorised, with the same first-maximum and NaN behaviour
- **perf:** single-request CPU inference is faster than llama.cpp at decode, with bit-identical results. On Qwen 3.5 0.8B (Threadripper 9970X, two DDR5 channels) decode goes from 10.5 to 43.9 tok/s (llama.cpp CPU, F16: 39.2) and prefill from 36 to 362 tok/s at 32 tokens and from 263 to 975 at 512. In the `f32` working set a BF16-stored matrix is held as its bits and widened in registers by the GEMV (`Tensor2D.FromBf16`, `TensorOps.DotRowsBf16`; `Data` widens on first touch; `AMQL_RESIDENT_BF16=0` restores widening at load); a persistent `KernelPool` replaces per-projection `Parallel.For`, which lets every FFN and linear-attention projection run on all cores; `MatMulTransposedBMany` fuses projections that share an input; prefill gets a 4×4 register-blocked GEMM tiled by activation rows; the Gated DeltaNet recurrence and conv are vectorised and parallel; `RowKvCache` keeps contiguous per-layer buffers instead of being rebuilt with LINQ on every step; `DecodeSession.Prefill` runs the head on the last position only. Every result is bit-identical to the code it replaced (`KernelExactnessTests`). See [docs/cpu-performance.md](docs/cpu-performance.md)
- **feat:** `amql-cli bench <container>` — single-request prefill and decode tokens/s after a warm-up, a checksum of the logits' bit patterns, and `--profile` for a per-operator split into parallel and serial time
- **feat:** Entail grafts into any host the importer can judge. `EntailHost.Judge` reads the host with `ArchMapper`'s rules and the block is written in the host's own shape and spelling: its heads and rotary pairs (half-split, interleaved, partial, zero-tail; linear, llama3, yarn, longrope), score scale and softcap, q/k norm per head, whole or absent, `w` or `1 + w` norms, sandwich and post-only norms (a ballast head and neuron fix their divisors), fused projections, zero biases, MoE layers (shared expert, or every expert alike), power-of-two residual and embedding multipliers, √hidden embeddings (a power-of-four width), and tied or untied heads. Host tensors widen by role, and every per-layer config fact shifts with the inserted layers. The program is compiled for the host's heads and FFN width. On Qwen 3.5 0.8B the mathematics graft is tensor-for-tensor identical to before; 16 zoo hosts keep their logits and answer, and 14 of them reload in transformers under their own class; LayerNorm hosts, uneven expert splits, non-power-of-two multipliers and softcaps that flatten a lookup (Gemma 2) are refused by name. `entail graft --output-norm`; `EntailCompiler.Compile(..., headsPerLayer)`
- **feat:** `longrope` and `dynamic` rope scaling, whose frequencies depend on the sequence length, taken as transformers does (largest position + 1). Served by the runtime and ONNX; refused for GGUF. Early Phi-3's `su` is read as longrope
- **feat:** MRoPE: `mrope_section` and `mrope_interleaved` (and the legacy `rope_scaling.type: mrope`) become the position policy's `mrope`. `GenericRuntime.RunLayerInternal(..., positionStreams)` rotates each pair by its (t, h, w) stream; tiny Qwen2-VL and Qwen3-VL decoders match transformers on text and on an image grid's positions
- **feat:** mixture-of-experts checkpoints judged from their tensors and standard config keys: Mixtral, Qwen2-MoE (shared expert and its sigmoid gate), Qwen3-MoE (dense layers by their own tensors), OLMoE. `norm_topk_prob` picks the routing rule; experts keep their checkpoint names through HF export. Sigmoid or grouped routers, router biases and packed experts are refused by name
- **test:** the zoo grows to 33 models (longrope in both regimes, dynamic NTK, four MoE families), plus `fixtures/mrope-zoo`; `EntailGraftTests`, `MropeTests`, `scripts/entail/check_graft.py`
- **feat:** rope scaling — `linear`, `llama3` and `yarn` rope blocks are served instead of refused, per layer type and on partial rotary, through one `RopeScaling.Frequencies` that mirrors transformers and is shared by the runtime, ONNX export and GGUF export (linear as `rope.scaling.*`, llama3 as `rope_freqs`; yarn refused for GGUF). Older Gemma 3 configs (`rope_local_base_freq`) become per-layer-type blocks. Six scaled-rope zoo models match transformers; real Gemma 3 12B matches greedy token for token
- **feat:** the text stack's prefix is found from its `layers.N` tensors (older `language_model.model` wrappers), and when a config does not state `tie_word_embeddings` the absence of a head tensor means tied
- **feat:** property-driven model support — a model family AMQL has no code for imports, runs and exports from its config and tensors alone. Layer tensors are bound by role (`TensorRoles`), with norm sites told apart by structure (pre-only, sandwich, post-only, parallel); contiguous fusions (`qkv_proj`, `gate_up_proj`) split zero-copy and join back on export; biases, LayerNorm, interleaved and per-layer-type rope, NoPE layers, sliding windows for every family, softcaps, `query_pre_attn_scalar` and `logit_scale` are read from the checkpoint; the few facts no checkpoint states (`1 + w` norms, √hidden embeddings, interleaved rope) come from `family-conventions.json` and are recorded as provenance. 20 transformers architectures with no AMQL code (Llama, Mistral, Qwen2/3, Phi-3, Gemma 1/2/3, OLMo 2/3, Granite, SmolLM3, StableLM, Starcoder2, Cohere, Helium, GLM-4, EXAONE 4, Ministral) match transformers' logits; real Phi-3-mini matches greedy token for token. See [docs/property-driven-models.md](docs/property-driven-models.md)
- **feat:** `amql-cli export --format hf|gguf|onnx`. `hf` writes the checkpoint's own config with the judged facts written over it; `gguf` picks the llama.cpp architecture (llama, qwen2, qwen3, granite, olmo2) from the model's properties and refuses what none computes; `onnx` translates the plan op for op and matches transformers in onnxruntime on all 20 zoo architectures
- **fix:** Granite's `residual_multiplier` scales the branch (`h + branch·rm`), as transformers does, not the residual stream
- **fix:** a LayerNorm final norm's bias is now bound; `gelu_pytorch_tanh` is kept as written instead of becoming `gelu`; safetensors shards that are symlinks (the Hugging Face cache) are read through to their target
- **fix:** the ONNX exporter computed the wrong function (untransposed `[out, in]` weights, a fixed rope theta, single-head attention); it is now built from the plan, keeps small initializers inline and names its nodes
- **feat:** disk-based grafting, for a host whose graft would not fit in RAM or on the GPU. `entail graft` streams: it reads the host through its memory map (sharded hosts via `model.safetensors.index.json` too) and writes each tensor as it is made with `SafetensorsStreamWriter`; `encode` reads each tensor lazily as its segment is written. On the Qwen 3.5 0.8B mathematics graft, peak private memory fell from 8.5 to 1.4 GiB for `graft` and 6.4 to 0.9 GiB for `encode`, with byte-identical output. The new `disk` working set (`--weights disk` or `AMQL_WEIGHTS=disk`) runs exact f32 on the CPU, keeping what fits in `AMQL_F32_CACHE_GB` (half the budget by default) and widening the rest from the container on every pass; `entail eval` and `entail side` choose it by themselves when a model's f32 weights exceed the memory budget. With a 1 GiB cache that graft's eval gave the same report as f32 at 4.7 instead of 10.6 GiB, 3.7× slower. Widening to f32 now runs chunked on every core straight from the map, which made resident f32 loading faster too
- **feat:** `amql-cli entail adapter|side` and `entail eval --adapter` — run Entail programs beside an unchanged model, reading its tokens and ruling on the next one at the sampler, in doubles whatever the model's precision. `SideBySideSession`, `EntailSparseAdapter` and `EntailVerdict`; see [docs/entail-adapter.md](docs/entail-adapter.md)
- **feat:** `amql-cli entail find|extract|background|optimise|review|accept` — from a trained model to a subject's Entail and back. `find` ranks the neurons or experts that carry a subject and tests the claim by ablation; `extract` probes a subject's declared relations and writes a table of what the model did, with each answer's softmax margin; `optimise` applies the documented rewrite rules in `docs/entail-optimisation.md`, including deriving a whole-number law by trimmed least squares; `review` puts the documentation and the program in front of any model, and every change, from any author, must pass a gate on held-out agreement, false fires on ordinary text, and cost. On Qwen 3.5 0.8B this derived `a + b`, `a - b` and `a*b` from the model's answers; all 47 rows the program overruled were the model's mistakes, and reloading took unseen two-digit arithmetic from 84% to 100% with 767/767 top-1 agreement on prose
- **feat:** `amql-cli entail decompile|fidelity` — the raw extract: every neuron seen across a corpus as Entail rules and observed `vote`s, with a saved study (`--from`), and its agreement with the model on held-out text. `scripts/entail/python_coverage.py` measures it against Python's own documentation
- **feat:** Entail `vote`, a soft output the interpreter runs and `compile` and `graft` refuse, and the string escapes `\q` and `\{`
- **feat:** Synthetic Python: `scripts/entail/subjects/python.synthetic.entail.txt`, its generator and scoring, and the steps in [docs/entail-synthesis.md](docs/entail-synthesis.md)
- **feat:** `scripts/entail/physics.entail.txt` and `chemistry.entail.txt`, with `make_chemistry.py`, `science_cases.py` and [docs/entail-sciences.md](docs/entail-sciences.md)
- **feat:** `scripts/entail/dates.entail.txt`, `bases.entail.txt`, `logic.entail.txt` and `json.entail.txt`, with `make_bases.py`, `make_logic.py`, `make_json.py`, `formal_cases.py` and [docs/entail-formal.md](docs/entail-formal.md)
- **feat:** Entail `only` and `forbid`, masks that rule tokens out without choosing one, for a program that keeps a host's text valid as it is written; and the string escapes `\t`, `\r` and `\e`
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
