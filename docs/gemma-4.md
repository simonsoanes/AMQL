# Gemma 4 26B A4B

Google's mixture-of-experts Gemma 4: 25 billion parameters of text model, of which a token
uses about 3 billion (128 experts a layer, 8 routed). This page is what it took to run it from
one 14 GB container with its whole 256K context, its drafter, images and tool calls, inside 24 GB of
GPU memory, and how it compares with
Qwen 3.5 4B. The checkpoint is
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (Apache-2.0).

```bash
amql-cli encode google/gemma-4-26B-A4B-it --out containers/gemma-4-26b-a4b --no-vision
```

```bash
amql-cli quantise containers/gemma-4-26b-a4b --out containers/gemma-4-26b-a4b-nvfp4-gptq --embedding --calibration <text> --calibration-tokens 131072
```

```bash
amql-server --model containers/gemma-4-26b-a4b-nvfp4-gptq --kv-format nvfp4
```

Its drafter and its vision tower are added to the container as it stands:

```bash
amql-cli attach containers/gemma-4-26b-a4b-nvfp4-gptq --drafter google/gemma-4-26B-A4B-it-assistant --vision google/gemma-4-26B-A4B-it
```

The encode takes 3 minutes and the quantise 22, on an RTX 5090. The 49 GB stack does not fit the
GPU, so `quantise` first makes a nearest-rounded copy of it to read the calibration text with,
and deletes the copy at the end. The full-precision container is only needed to quantise again.

## Beside Qwen 3.5 4B

One RTX 5090, both NVFP4 (GPTQ), the same build, measured on 2026-11-25.

| | Gemma 4 26B A4B | Qwen 3.5 4B |
|---|---|---|
| Container | 14.2 GB (from 51.6 GB) | 3.8 GB |
| Decode, after a 512, 2K and 8K prompt | 266, 255, 231 tok/s | 275, 270, 247 tok/s |
| Prompt, 2K and 8K | 8,003, 7,463 tok/s | 11,413, 10,543 tok/s |
| 8 replies together | 633 tok/s | 730 tok/s |
| Reading 256K tokens | 71.9 s | 72.1 s |
| Decode at 256K, FP16 KV | 39 tok/s | 47 tok/s |
| GPU memory at 256K, NVFP4 KV | 18.1 GB | |
| GPU memory at 256K, FP16 KV | 22.6 GB | 15.0 GB |
| Hard reasoning suite, held-out seed | 80 of 80 without thinking (739 tokens a reply) | 75 of 80 with thinking closed at once, 80 at a 1,536-token budget, 60 with no limit |
| Facts recalled among look-alikes at 28K, 127K, 258K | 12 of 14, 14 of 24, 5 of 24 | 14 of 14, 22 of 24, 17 of 24 |

- The decode figures are without drafters, which both models have (see [below](#the-drafter)).
- **Long contexts are Gemma's weak side.** It reads 256K and quotes from it, but tells one of 126
  facts of one form from another far worse than the 4B does (see
  [context-extension.md](context-extension.md) for the measure). The arithmetic was checked: 7,032
  tokens agree with transformers where the model writes its own turns, f32 prompts and FP16 or
  NVFP4 caches recall the same, and 4-bit rounding is part of it: nearest rounding recalled 8 of 24
  at 127K and 2 at 258K, GPTQ 12 to 14 and 5 to 6. The original weights do not fit the GPU, so what
  they would recall was not measured.
- The memory is over the desktop's own. The KV cache is 5.0 GB of it at FP16 and 1.5 GB at NVFP4:
  only the 5 full-attention layers grow with the context (see below).
- Conversations reuse what they share: over a 203,949-token document through the server, the first
  question took 74 s and each one after it 1.0 s.

## What the 4 bits cost

Against the original BF16 weights, run by transformers a layer at a time (f32 arithmetic), over
3,575 positions of text the calibration never read, each in the model's own turn:

| Container | KL | Next token agrees |
|---|---|---|
| NVFP4, nearest | 0.112 | 90.4% |
| NVFP4, GPTQ | 0.057 | 93.1% |
| NVFP4, GPTQ, embedding and head too (`--embedding`) | 0.064 | 93.4% |

`--embedding` takes 1.06 GB off the container and the GPU and decode from 218 to 254 tok/s, because
the tied head is 1.5 GB read for every token.

Scoring raw text, outside the model's own turn, says nothing here: the original weights give it 4.8
to 7.1 nats a token.

Google also publishes weights trained to be rounded to 4 bits
(`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`). They were not tried.

## 24 GB

With everything attached and NVFP4 KV, reading 252,058 tokens, then an image, then writing: 18.4 GiB
at the peak. It was measured with ballast on the card, so that 21.3 GiB were left, and nothing spilled into
host memory. The weights are 14.2 GB of it and the drafter 0.24 GB. Between images the GPU holds nothing
of the vision tower. FP16 KV took 22.6 GB before the drafter was there, which a 24 GB card with a desktop on
it does not have: use `--kv-format nvfp4` there.

## The drafter

Google's assistant (`gemma4_assistant`) is four layers, 1,024 wide, with queries only: it reads the keys
and values of the model's last sliding and last full layer, from one position, and so keeps no cache, needs
no prompt and catches up with nothing. Its drafts match transformers' on a random pair of models
(`fixtures/gemma4-assistant`). Greedy, on an RTX 5090, NVFP4 weights, the replies token for token those
of plain decode:

| | Plain | With drafts | Tokens a step |
|---|---|---|---|
| Gemma 4 26B, prose | 250 tok/s | 331 tok/s | 2.4 |
| Gemma 4 26B, code | 258 tok/s | 391 tok/s | 3.6 |
| Qwen 3.5 4B (its MTP layer), prose | 263 tok/s | 365 tok/s | 2.3 |
| Qwen 3.5 4B, code | 268 tok/s | 380 tok/s | 2.5 |

- The number of drafts follows the reply: one more when all held twice running, one fewer when most did
  not, four at most. A draft whose chance of holding, by the drafter's own reckoning along with those before
  it, is under 45% is not checked. This replaces the fixed three drafts for the 4B too.
- Verifies and draft runs are captured as CUDA graphs, like the plain step. Launched kernel by kernel a one-row
  verify took 6.4 ms instead of 3.9.
- An expert pair is a group of its own up to 8 rows: grouping a few rows by expert cost more than it saved.
- At 4 bits the drafter keeps as many drafts as at BF16 and takes half the time.
- It runs while one reply runs alone, as every drafter here does.

## Images

A picture is resized to its shape's best fit of at most 280 tokens, as the model's own processor does it, and
its tokens stand in the prompt where the picture does. The tower (27 layers) runs on the GPU from weights
in host memory, copied over a layer at a time: 0.15 s a picture, after 0.9 s for the first. PNG and JPEG
files are read by code of AMQL's own, bit for bit as Pillow reads them. On the real tower the tokens are
0.7% from transformers' in BF16, the precision everything here runs at, and the same in f32.

In the layers with a window each token of an image sees the whole image, ahead of it too (the
full-attention layers stay causal). An image is therefore never cut between two chunks of a prompt.

In a prompt an image's ids are below zero and made from its pixels, so kept state is only reused for the same
picture. A conversation about a picture reads it once.

- In the server: chat's `image_url` and the Responses API's `input_image`, as `data:` URLs, in user messages
  and tool results. `detail` `low` is 70 tokens and `high` 560.
- On the command line: `amql-cli generate <container> --chat --prompt "…" --image photo.jpg`.

## Tool calls

The server runs the model's own template for declarations, calls and results (checked against
`fixtures/gemma4_tool_golden.json`, made by transformers), and learns from it how calls are written, as it does
for [any model](server.md#chat-templates). Gemma's are `call:NAME{key:value}`. Each is held to
the tools of the request, in the model's notation: bare keys, strings between `<|"|>` marks. Three things differ from
the template on purpose. What was said along with a call is written before it (the template
puts it after the results and closes the turn, so the reply would start outside any turn). A result that names
no call answers the one in its place. And a reply that goes on from an image without thinking starts with
the empty channel: straight from the image the model writes nonsense.

In a loop of calls the results stay in one model turn, and kept state is saved before the last of them, so
each step reads only what is new (0.1 to 0.8 s a step).

## What had to be built

None of it names Gemma at run time. The importer judges each of these from the checkpoint, and
three facts that no checkpoint states are entries in `family-conventions.json`.

| | |
|---|---|
| Bare parameters (`layer_scalar`), experts packed outside the MLP | import |
| Heads that differ by layer, read from each layer's own projections | import, planner |
| One projection for keys and values; a weightless norm on values | planner, engine |
| A dense FFN beside routed experts, with norms of their own; a scaled router; a multiplier on a layer's output | planner, both runtimes |
| Zero-tail rotary (`proportional`), 512-wide heads | engine |
| **Routed experts on the GPU engine** | engine (`amql_moe.cu`) |
| **Window-bounded caches** | engine |
| SentencePiece-style tokenizer (marked spaces, byte fallback) | tokenizer |
| The model's own chat template, and its turns, thinking and calls learned from it | `Jinja/`, `ChatDialect`, `CallShape` |
| GPTQ for experts, and for a model larger than the GPU | `quantise` |

Mixtral, OLMoE and Qwen3-MoE run on the GPU engine now too. A shared expert (Qwen2-MoE) is still
left to the managed path.

### Routed experts

A layer's experts are one matrix a projection. A row's router scores, top 8 and weights are one
kernel. Then:

- **A decode step** takes each routed expert's gate and up rows with one warp an output, and the
  down projection takes all 8 experts of an output together. The dense FFN runs beside them as a
  branch of the captured graph (194 to 218 tok/s).
- **A prompt** gathers the rows sent to each expert into blocks of 32, in row order so that a prompt
  gives the same bits every run, decodes 8 experts at a time to BF16 and runs their blocks as one
  batched tensor-core GEMM. Decoding the experts costs the same whatever the rows, so routed models
  read prompts 1,024 tokens at a time. This took prompts from 1,047 to 7,986 tok/s.

### Window-bounded caches

25 of the 30 layers attend to the last 1,024 positions. Kept whole, their cache would be 50 GB at
256K and FP16. A windowed layer's cache stops at about two windows and slides: when it is full its
last rows move to the front. The layer counts rows from there, and its rotary position adds it back.
All 25 take 0.6 GB at any context. Rewinding and paging keep the rows before the mark; a restored
sequence's steps agree to f32 rounding rather than to the bit. Every model with sliding windows
(Gemma 2 and 3, Mistral, OLMo 3) gets this.

### For every model

- **Loading** reads 4-bit weights as stored. It used to decode every block scale to a float, keep
  it, and encode it again: 70 s and 6 GB of host memory for this model, now 13 s and 20 MB.
- **Dependent launch** is chosen by format: it helps BF16 weights and costs 4-bit ones 4 to 9% of a
  decode step. Qwen 3.5 4B: 248 to 270 tok/s.
- **Prompt attention's scratch** is 256 MB rather than 1 GB: 1.5 GB less memory, and Qwen 3.5 4B reads
  128K tokens in 20 s rather than 35.
- `bench --profile` times a decode step call by call on the GPU.

## Limits

- **Images** are PNG and JPEG; WebP, GIF, video and sound are not read. At most 560 tokens an image: the
  model's 1,120-token setting would not fit a prompt chunk. They need the GPU engine, and no Entail adapters.
- **Gemma 4's smaller models** (E2B, E4B) are refused by name: per-layer input embeddings and layers
  that share another's cache are operators AMQL does not have.
- `nvcc -split-compile` builds the engine in 4 minutes rather than 12 and makes Qwen 3.5 4B decode
  10% slower, so the build scripts do not use it.
