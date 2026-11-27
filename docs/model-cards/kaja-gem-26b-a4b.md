---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - amql
  - gemma4
  - nvfp4
  - mixture-of-experts
  - long-context
  - entail
---

# Kaja-gem 26B A4B

[Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) in one 4-bit container for
[AMQL](https://github.com/simonsoanes/AMQL), with three changes that need no training:

1. **Long contexts.** The query norms of its five global attention layers are multiplied by 1.5.
   Asked for one fact among 126 of the same form at 256K tokens, it answers 38 of 48 instead of 14.
2. **NVFP4 weights rounded by GPTQ**, the embedding and head too, with Google's drafter attached for
   speculative decoding and the vision tower for images.
3. **Five Entail programs** that run beside the model at the sampler and rule on the next token:
   exact dates, number bases, propositional logic and JSON validity, and `stop-circling`, which ends
   thinking that has started to repeat itself.

| | Gemma 4 26B A4B, NVFP4 | Kaja-gem 26B A4B |
|---|---|---|
| One fact among 126 at 256K tokens | 14/48 | **38/48** |
| ... at 128K | 20/48 | **37/48** |
| Dates, number bases, logic, JSON validity (1,156 cases) | 267 | **1,151** |
| Hard reasoning suite (80 generated problems), thinking, 3,072 tokens | 2/80 | **77/80** |
| Hard reasoning suite, no thinking | 80/80 | 80/80 |
| Decode, one reply with the drafter: prose, explanation, code (tok/s) | 274, 337, 433 | 254, 316, 411 |
| Decode, 80 replies in turn, no drafter (tok/s) | 242 | 236 |

Measured with AMQL on one RTX 5090. The formal cases and the reasoning suite are generated, with
answers from Python's own libraries; details are in AMQL's `docs/kaja-gem.md`.

## Running it

This is an AMQL container, not a transformers checkpoint: its weights are stored in AMQL's segment
format, 4-bit where they were rounded.

```bash
amql-server --model kaja=<this folder> --kv-format nvfp4
```

It serves OpenAI-compatible chat completions and responses, with streaming, tools in Gemma's own call
form, thinking (`reasoning_content`), images, and the whole 256K context in 18.4 GiB of GPU memory.

## What was changed

From [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), under the
[Apache 2.0 licence](https://ai.google.dev/gemma/docs/gemma_4_license):

- the weights of the attention, experts, shared MLP, embedding and head rounded to NVFP4 (4-bit values,
  FP8 scales per 16) by GPTQ against a calibration text;
- the `q_norm` weights of layers 5, 11, 17, 23 and 29 multiplied by 1.5;
- the drafter [google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)
  attached at NVFP4;
- Entail adapters added in `adapters/`. They hold no weights: each is a program, compiled when the model
  loads and run in doubles beside it.

The chat template, tokenizer and vision tower are Google's, unchanged.

## Limits

- The long-context edit was found and measured on one kind of document (source code and its docs) with
  one form of fact. It does not lose the hard reasoning suite or tool calls, but other long-context work
  (summaries, multi-turn retrieval) was not measured.
- An Entail program speaks only where it reads the whole question. What it says replaces the model's own
  token, formatting included.
- Rounded to 4 bits, the model is not bit for bit Gemma 4.
