# Masked diffusion LM support — dLLM Qwen3-0.6B (MDLM)

**Model:** [`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)
(downloaded to `D:\Models\Qwen3-0.6B-diffusion-mdlm`). It is Qwen3-0.6B converted
from autoregressive to diffusion ("A2D") and trained with MDLM by the dLLM framework.

I picked it because it is the most-downloaded small text diffusion LM. Its
architecture is a family AMQL already runs. The published modeling code changes only
one thing from Qwen3: the causal mask becomes a padding-only mask. Logits at position
*i* predict the token *at* *i*, with no shift. Generation starts from a run of
`<|mask|>` tokens and unmasks them block by block over repeated full-sequence passes.

## What AMQL now does

| Step | Behaviour |
|---|---|
| `encode` | Reads `a2d-qwen3` as Qwen3 plus `attention.bidirectional = true` plus a `diffusion` surface: `masked_diffusion`, the mask token id from the tokenizer files, the HF class and `auto_map`. Any other `a2d-*` is refused. Before this change the checkpoint imported **silently as a causal Qwen3**. |
| plan / runtime | Non-causal attention, using a CPU kernel flag. The GPU kernel needs no change, because it uses positions only for masking. Each query gets the last position, so it masks nothing. Step-decoding a diffusion plan is refused. |
| `generate` | Detects a diffusion container and runs `MaskedDiffusionSampler`. This is the model card's sampler: per block, each pass commits the *k* most confident masked positions on an even schedule. Flags: `--gen 128 --steps <gen> --block 32 --remasking low_confidence\|random --temperature 0 --show-steps --chat [--think]`. The vocabulary head is computed only for the current block's rows. Generation stops once a finished block holds `<|im_end|>`; this is exact, because committed tokens never change. |
| `export` | Restores `A2DQwen3LMHeadModel`, `auto_map` and top-level `rope_theta`, and carries `modeling_qwen3.py` byte for byte without executing it. All 310 tensors are bit-identical to the original. The original's `lm_head` duplicates the embedding, and the export ties it. |
| `to-gguf` | Writes llama.cpp's **`rnd1`** architecture, which is Qwen3-MoE with non-causal attention. Each dense FFN becomes a **one-expert MoE**: a softmax over one router logit is exactly 1, so the result is exact. Also writes `attention.causal=false`, `diffusion.shift_logits=false` and the mask token id. `--quant none\|f16\|q8_0\|mxfp4` all work. The per-tensor `--recipe` path is part of the separate, uncommitted variable-quantization work. |

## Validation

- **Reference:** `scripts/diffusion/mdlm_ref.py` is a float32 PyTorch forward built from the safetensors. Running the checkpoint's remote code was blocked. The reference was checked in causal mode against transformers' `Qwen3ForCausalLM`, with a max difference of 2e-5.
- **AMQL vs reference, real 0.6B:**
  - Argmax agrees at all 56 positions (a chat prompt plus 32 masks).
  - Sampled logits are within 0.05 (`MaskedDiffusionTests.Real_Checkpoint_Matches_Reference_When_Available`, gated on `AMQL_MDLM_CONTAINER`).
  - Full generations match **token for token** for "capital of France" (32/32/32) and "prime function" (128/128/32 and 140/140/20), on both the f32 CPU path and the FP16 GPU path.
- **AMQL vs reference, synthetic 2-layer checkpoint:**
  - Every logit matches within 1e-3.
  - The full commit trace of a two-block generation matches exactly.
- **llama.cpp vs reference:** `llama-diffusion-cli --top-k 1` produces exactly the reference's text over 140 positions, with the F32 GGUF on CPU and the BF16 and Q8_0 GGUFs on CUDA. `--top-k 1` makes decoding argmax and left to right (`--order ltr` in the reference script).
- **Tests:** 10 new tests. The full suite passes: 335/335.

### Comparing with llama.cpp: two things to know

1. **Sequence length matters.** In a bidirectional model every trailing mask is attended to. `llama-diffusion-cli` always masks out to `-ub` total positions, so compare like with like. Use a block length that divides both the prompt and the generation length, then set `--gen` = `-ub` − prompt.
2. **`--temp 0` is not greedy there.** llama-diffusion-cli applies top-k 40 and top-p 0.95, then still *samples* through `dist`. Its "confidence" is the probability renormalised over that truncated set. With default flags it therefore gives a plausible but different answer: for the prime prompt, `i = 5 … i += 6` instead of `i = 3 … i += 2`. This is a property of the sampler, not of the conversion.

## Speed (RTX 5090, 0.6B, ~150-token sequences)

| Runtime | Weights | Time per denoising pass |
|---|---|---|
| AMQL `generate` | `--weights bf16` (FP16 GEMMs on GPU) | ~0.3 s (3.1 passes/s) |
| AMQL `generate` | f32 on CPU | ~2.8 s |
| llama-diffusion-cli | BF16 GGUF, CUDA | 14–21 ms |
| llama-diffusion-cli | Q8_0 GGUF (749 MiB), CUDA | 10–14 ms |
| llama-diffusion-cli | F32 GGUF, CPU | 170–360 ms |

AMQL's pass costs about the same as one of its own prefills. The managed runtime
copies activations to and from the device around every GEMM and runs norms, RoPE and
the residual on the CPU. Diffusion does a full prefill every step, so that overhead
dominates. Making the runtime device-resident would be a general AMQL improvement,
not specific to diffusion. For serving, the GGUF and llama.cpp path is about 20× faster.

## Bugs found and fixed along the way

- **Legacy rope config ignored (affects many models).** In configs without a `rope_parameters` block (top-level `rope_theta` / `rope_scaling`), the rope facts were dropped and theta defaulted to 10 000. This affects any Qwen3/Qwen2/Llama checkpoint saved by transformers 4.x. These are now folded into the grouped form, and a legacy `rope_scaling` is carried and refused by name instead of being silently ignored.
- **GGUF file-type picker** threw on an all-F32 file and ignored 3-D expert banks.

## Not done / follow-ups

- The `bd3lm` sibling ([`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)) is **not** supported. It uses a block-causal mask and a different sampler, so `a2d-*` types other than `a2d-qwen3` refuse.
- AMQL has no prefix KV cache for diffusion (Fast-dLLM style), because exact caching is impossible in a bidirectional stack. Approximate caching would be a separate, opt-in feature.
- Classifier-free guidance (`cfg_scale`) from the card's sampler is not implemented. The card's default is 0.
