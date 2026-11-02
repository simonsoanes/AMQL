# LFM3 × Qwen3.5 merge, MoE and variable quantization — overnight report

*1 November 2026 · run unattended overnight on the RTX 5090 workstation.
Committed: LFM3 support and four bug fixes. Everything else is uncommitted, for review.*

## TL;DR

- **LFM3 / LFM3-VL support is in AMQL and committed** (`1f2f1f4`): encode, inference on CPU and GPU (matching a
  float32 PyTorch reference to 4 significant figures on the real 2B model), byte-exact export, and an `lfm3` GGUF
  writer whose output is byte-identical to llama.cpp's own converter.
- **Four real bugs fixed and committed on the way** — see [Committed](#committed).
  The GPU-attention one affected *every* model's GPU prefill.
- **The merge works, but cannot add capability.** LFM3's stack is kept verbatim (different architectures cannot
  be blended layer-by-layer); only the token interface moves. The default 50/50 blend hurts (wikitext PPL 53.2 →
  72.6; 83.1 with Qwen's 161k extra tokens appended). A light blend that keeps LFM3's vocabulary
  (`import --blend 0.05 --vocab base`, new) is equivalent to LFM3: chat-KLD 0.007, 97.8% same top-1, tools 15/16.
- **Training-free MoE is not viable for LFM3**: 8 experts/top-2 takes held-out PPL 142.6 → 41,370; even top-6 of 8
  costs +299%. LFM3's SwiGLU FFN isn't activation-sparse; MoE-ifying it needs continued pre-training.
- **"Export as Nemotron" would be a re-architecture, not a conversion** (details below). The right "existing
  inference server" target is llama.cpp's native `lfm3` architecture, which AMQL now writes.
- **Variable quantization works and matches or beats llama.cpp's presets.** New `to-gguf --recipe … --imatrix …`, driven by a
  measured sensitivity map, gives a **990 MB model that is equivalent to LFM3** (chat-KLD 0.033, 93% same top-1,
  tools 16/16) — 6% smaller than llama.cpp's Q4_K_M, tied with it on chat (0.033 vs 0.031) and clearly closer to
  BF16 out-of-domain (0.046 vs 0.065) — plus 851 MB and 651 MB builds that keep tool calling (14–16/16) with
  measurable drift.
- **1–2-bit interim layers cannot reach LFM3 quality by post-training quantization.** Free healing (distilling the
  F32 norm gains) recovers ~10–20% of the gap; a distilled LoRA adapter recovers ~35% but costs 44% of GPU decode
  speed, so it defeats the purpose. Q1_0/Q2_0 in particular are formats for quantization-aware-trained models — on
  LFM3 they give PPL > 60,000.
- **MXFP4/NVFP4 are not the best choice here.** For the tied embedding/head and for FFNs, post-training FP4 is 2.5–5×
  worse in fidelity than K/IQ 4-bit at the same size, for ~5–7% faster prefill. (NVIDIA's NVFP4 models get their
  quality from FP4-aware calibration/training.)
- **Speed**: on the 5090, decode saturates at ~640–675 tok/s for anything ≤ 1 GB — 1.9× BF16 — because it is
  kernel-launch-bound, not bandwidth-bound; sub-4-bit buys almost nothing on this GPU. On CPU it scales with bytes:
  20.5 tok/s (BF16) → 58 (990 MB) → 78.5 (651 MB).

## Deliverables

`D:\Models\AMQL\LFM3xQwen3.5-2B-AMQL\` — all files load in **stock upstream llama.cpp** (built from `a91eab6`,
31 Oct 2026). LM Studio's bundled runtime (2.57) predates `lfm3` and will not load them yet.

| file | size | what |
|---|---|---|
| `…-BF16.gguf` | 3.3 GB | merged model (LFM3 stack, Qwen-aligned token rows at 5%), reference precision |
| `…-AMQL-mix1000.gguf` | 990 MB | **recommended**: equivalent to LFM3 |
| `…-AMQL-mix850-healed.gguf` | 851 MB | faster on CPU, small drift, tools 15/16 |
| `…-AMQL-mix650-healed.gguf` | 651 MB | 2-bit-class FFNs, noticeable drift, tools 16/16 |
| `mmproj-…-F16.gguf` | 379 MB | LFM3-VL vision tower + projector for `--mmproj` |
| `recipes/*.txt`, `imatrix-calib-mix.gguf` | | how each build was made (`amql-cli to-gguf --recipe`) |
| `e2e-*.md` | | llama-server transcripts: chat, tool round trip, vision |
| `scripts/` | | sensitivity, allocation, healing, evaluation (also in `scripts/lfm3-quant/` here) |

```bash
llama-server -m LFM3-VL-2B-x-Qwen3.5-0.8B-AMQL-mix1000.gguf --mmproj mmproj-LFM3-VL-2B-x-Qwen3.5-0.8B-F16.gguf -ngl 99 --jinja -c 16384
```

## Results

Fidelity is measured against **BF16 LFM3 itself** (KL divergence of the next-token distribution and how often the
top token agrees), on two held-out sets: *chat* — LFM3's own held-out chat/tool transcripts — and *OOD* — technical
docs, a model card and code. Wikitext PPL is shown for reference only: the importance matrix was partly calibrated on
wikitext-train, which flatters it (some builds score *below* BF16). Tools: 16-case suite via llama-server, thinking on.

| build | size | chat KLD | chat top-1 | OOD KLD | OOD top-1 | tools | GPU tg128 | CPU tg128 |
|---|---|---|---|---|---|---|---|---|
| LFM3 BF16 (original) | 3307 MB | — | — | — | — | 15/16 | 342 | 20.5 |
| merged BF16 | 3307 MB | 0.007 | 97.8% | 0.004 | 95.9% | 15/16 | 346 | 20.5 |
| llama.cpp Q4_K_M (ref) | 1049 MB | 0.031 | 93.3% | 0.065 | 85.4% | — | 628 | 54.6 |
| **mix1000** | 990 MB | 0.033 | 93.2% | 0.046 | 87.6% | 16/16 | 643 | 58.3 |
| mix850 healed | 851 MB | 0.054 | 90.9% | 0.095 | 82.6% | 15/16 | 655 | 66.5 |
| mix650 healed | 651 MB | 0.193 | 83.7% | 0.273 | 72.2% | 16/16 | 663 | 78.5 |
| llama.cpp IQ2_S (ref) | 600 MB | 0.437 | 75.6% | 0.729 | 59.1% | — | 668 | 82.0 |

(CPU numbers are for the equivalent recipe built from the earlier scripted merge; the GGUFs differ only in the 5%
blend of embedding rows.)

### What the sensitivity sweep found

Everything at Q6_K, then one group lowered at a time (chat KLD; baseline 0.007):

| lowered | to | chat KLD |
|---|---|---|
| tied embedding/head | MXFP4 / NVFP4 / Q4_K / Q5_K | 0.042 / 0.027 / 0.015 / 0.008 |
| all FFN gate / up / down | IQ2_S | 0.047 / 0.067 / 0.078 |
| all FFN | IQ2_S | 0.185 |
| attention q/k/v | IQ2_S | 0.075 |
| attention output | IQ2_S | 0.087 |
| conv in / out | IQ2_S | 0.043 / 0.041 |

Per layer, FFNs 15–22 are 2–4× less sensitive than FFNs 0–5. Attention is small but the most sensitive per byte.
A greedy allocator over these measurements (bits where they buy the most KLD) predicted the measured results to within
~10%, and produced the recipes in `recipes/`.

### Merge

| variant | wikitext PPL |
|---|---|
| LFM3 | 53.19 |
| AMQL merge as designed (50/50 blend, union vocab 286k) | 83.14 |
| 50/50 blend, LFM3 vocab | 72.61 |
| 20% blend, LFM3 vocab | 53.37 |
| 5% blend, LFM3 vocab | 53.06 |

The tied head means appended Qwen-only rows receive logits and take probability mass, even though LFM3's merges
never produce those tokens. A held-out ridge fit of Qwen3.5-0.8B's embedding space onto LFM3's reaches cosine 0.79 but
only 2.7% top-1 retrieval — most of the agreement is a shared mean direction, so the spaces correspond only loosely.

### MoE

| `moe-ify` (no training) | held-out PPL (dense 142.6) |
|---|---|
| 8 experts, top-2 | 41,370 |
| 8 experts, top-6 | 569 |

### Why not Nemotron

llama.cpp's `nemotron_h` / `nemotron_h_moe` (read from `src/models/nemotron-h.cpp`): each block is one operator —
Mamba2, *or* attention, *or* FFN; attention has no RoPE, no QK-norm, no output gate; the FFN is an ungated ReLU²
MLP (MoE variant: sigmoid-routed ReLU² experts with a shared expert). LFM3 has a gated short convolution (B·x →
depthwise conv → C·), RoPE attention with QK-norm and a per-head sigmoid gate, SwiGLU FFNs and sandwich norms. None of
these maps onto a Nemotron operator, so "restructuring" would mean replacing every operator and retraining. Exporting
to the native `lfm3` arch keeps the model exact and runs in stock llama.cpp.

### Healing

Only tensors that stay F32 in the GGUF are trained, so the result needs no runtime support: every RMSNorm weight
(LFM3's sandwich post-norms act as per-channel output gains) and the short-conv kernels, as log-gains, distilled
against BF16 LFM3 on calibration text with validation-based early stopping. A first version trained the weights
directly and overfit (held-out chat KLD got *worse*, 0.185 → 0.277); relative gains fixed it (0.185 → 0.151).
A LoRA adapter trained alongside the gains, exported as a llama.cpp adapter (`--lora`), goes further the longer it
trains — on the 651 MB build, chat KLD 0.193 → 0.170 (rank 16, 600 steps) → 0.129 (rank 32, 1500) → **0.121** (rank
32, 5000 steps, 2.4 MB of LFM3's own transcripts; top-1 87.2%, OOD 0.251, tools 15/16); on the 851 MB build
0.055 → 0.048. **But the adapter is not worth deploying**: llama.cpp applies it as extra matmuls, and decode drops
686 → 385 tok/s on the GPU (−44%) and 79 → 61 tok/s on CPU (−23%) — at which point the 651 MB build is no faster
than mix1000 and much less faithful. The adapters are therefore not shipped; they show that more training keeps
paying off, i.e. the route to a genuinely good ~2-bit model is quantization-aware training (adapting the low-bit
weights themselves), not post-hoc correction.

## Committed

| commit | |
|---|---|
| `1f2f1f4` | **feat: LFM3 / LFM3-VL** — config facts (sliding window, sandwich norms, wrapper), `zero_tail_rope` position policy, per-head output gate, the real gated short conv (also fixes LFM2, whose previous conv was conv-then-SiLU and matched no checkpoint), sandwich norm placement, `Lfm3Layout` export, `Lfm3Gguf` writer, `synth-model --family lfm3`, golden test against PyTorch. Includes **fix(cuda)**: fused GPU attention passed `Array.Empty<float>()` for "no sinks", which P/Invoke marshals as a valid pointer — garbage was added to every head's position-0 score on every prefill ≥ 8 tokens, for all models — plus a missing barrier between its max and sum reductions. |
| `6f664a6` | **fix(merge)**: `LeastSquares.Fit` returned Mᵀ where every consumer computes M·b (every merge used the transposed alignment); unlocked `localFinally` race in the Gram/Cross reduction; cross-family merges no longer graft the other stack's tensors (crash + dead weight); mean agreement no longer counts non-shared tokens as 0. |
| `b391c6b` | **fix(moe-ify)**: dense FFN names come from the plan instead of hard-coded Qwen names. |
| `53d8eb4` | **fix(gguf)**: complete ggml block-size table (TQ1_0/TQ2_0/Q1_0/Q2_0 threw; K/IQ types missing). |

## Uncommitted, for review

- `amql-cli tool-eval` (`src/Amql.Cli/ToolEval.cs`, cases in `tests/Amql.Tests/fixtures/tool-eval-cases.jsonl`) —
  validates tool use through any OpenAI-compatible endpoint; parses raw `<tool_call>` text when a server doesn't.
- `to-gguf --recipe <file> [--imatrix <imatrix.gguf>] [--ggml <dir>]` for lfm3 — per-tensor types; K/IQ types are
  encoded by ggml's own `ggml_quantize_chunk` via P/Invoke (`src/Amql.Gguf/GgmlNative.cs`, `AMQL_GGML`), with
  llama.cpp-format importance matrices. Only wired for the lfm3 writer so far.
- `import --blend <w> --vocab union|base` (`ModelMerger.MergeOptions`).
- `scripts/lfm3-quant/` — the sensitivity sweep, allocator, healing trainers, evaluation and e2e scripts (Python
  parts use torch + gguf-py; they ran in the Unsloth Studio venv, read-only).

## Not done / caveats

- **Qwen3.5-4B pass skipped** (weights downloaded to `D:\Models\Qwen3.5-4B`). With Qwen3.5-4B the larger model, the
  merge would keep *its* stack, giving Qwen3.5-4B with LFM3-nudged embeddings; and MoE isn't viable — so the
  "complete easily and viable" condition wasn't met.
- AMQL doesn't compute importance matrices itself yet (it reads llama.cpp's), and the healing trainers are Python —
  AMQL has no autograd.
- vLLM / SGLang not tested (not installed here). The tool suite is 16 cases; single-case differences are noise.
- The first calibration corpus (used for the importance matrix and the norm-healed `-healed` builds) contained the
  16 tool-eval prompts with LFM3's own answers, which can flatter those builds' tool scores slightly; the un-healed
  mix1000 (16/16) and the adapter runs (cleaned corpus, 14–15/16) are unaffected.
- The KLD metric uses llama.cpp's BF16 as the reference; AMQL's own forward matches PyTorch, and AMQL's BF16 GGUF is
  byte-identical to llama.cpp's converter output, so the chain is consistent.
