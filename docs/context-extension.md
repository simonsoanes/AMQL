# Running a model past its trained context

`--max-context` runs any model AMQL serves at a context length other than the one it was
trained to: 256K Qwen 3.5 4B at 1M tokens on one 32 GB GPU, for example. This page says how,
what it costs, and how well it works, measured rather than assumed.

```bash
amql-server --model containers/qwen3.5-4b-nvfp4-gptq --max-context 1M --kv-format nvfp4 --kv-gpu-budget 14
```

```bash
amql-cli --max-context 1M --kv-format nvfp4 generate <container> --prompt ...
```

- `--max-context N` takes tokens, or K and M (`1048576`, `512K`, `1M`). A length shorter than
  the model's own only limits it. Per model in a server `--config`: `"max_context": "1M"`.
  In AMQL Studio: *Context length* on the model's panel (512K, 1M or any length), with
  *Prompt state (KV) format* under the server's defaults. With nothing set, every model keeps its
  trained length.
- `--rope-extend auto|none|yarn|ntk|linear` says how positions past the trained length rotate.
  `auto` is the default. Per model: `"rope_extend"`.
- The container's files never change. The extension is made to the system graph as the
  container opens, so the planner, the engine, the server's limit and the exports all see one
  model. The load notes say what changed.

## What `auto` does

Only attention layers with rotary positions are affected. Recurrent layers (Gated DeltaNet)
hold a fixed-size state and have no position limit, and sliding-window layers whose window is
inside the trained length never see a new distance.

- A conversation that fits the trained length runs at the trained frequencies. Nothing a model
  does today changes: its logits are the same to the bit.
- A conversation that will go past it runs NTK-aware frequencies (the rotary base grown so the
  slowest frequency covers the new length and the fastest is unchanged) from its first token.
  Keys are stored rotated, so which it is has to be settled when the sequence starts: from the
  prompt and the reply's token limit. A conversation that outgrows the trained length is read
  again once, extended.
- Both kinds run side by side in one batch. Saved state (`--kv-disk`) remembers which it is.
- A model whose rope is already scaled (llama3, yarn, linear) has that scaling taken further
  for every sequence. `dynamic` and `longrope` already follow the length and are left alone.
- The managed (CPU) path switches on each call's positions, as it does for longrope.
- `auto` has no checkpoint spelling, so exports (HF, GGUF, ONNX) refuse it. Name a method to
  export one: it is written as the model's own rope settings.

## Qwen 3.5 4B from 8K to 1M

`amql-cli context bench` reads one long document (this repository's docs and source) with a
fact planted every 2,048 tokens ("the maintenance code for the *copper lantern* is 73914"),
and stops at each length to ask about facts spread evenly over everything read, and to have
passages quoted from a 64-token excerpt. Reading once and rewinding is what makes 1M
affordable, and gives every length in one run. NVFP4 weights (GPTQ) and NVFP4 KV.

The facts all have one form, and there are more of them the longer the document (126 at 258K,
510 at 1.04M), so this is a test of telling them apart, not only of reach.

**The comparison to rely on**: one frozen copy of the text, 64 questions at each length.

| Context | `none` | `yarn` | **`ntk`** |
|---|---|---|---|
| 258,048 | 44/64 | 38/64 | **52/64** |
| 520,192 | 28/64 | 30/64 | **41/64** |

**The wider sweep**, 24 questions at each length. Its text was this repository while it was
being worked on, so the runs read slightly different documents: read it for the shape, not for
differences of a few facts.

| Context | `none` | `yarn` | `linear` | **`ntk`** |
|---|---|---|---|---|
| 28,672 | 14/14 | 13/14 | 14/14 | 14/14 |
| 126,976 | **23** | 18 | 21 | 19 |
| 258,048 | 16 | 19 | **20** | **20** |
| 389,120 | 17 | 11 | **18** | 15 |
| 520,192 | 8 | 8 | 16 | **20** |
| 782,336 | 6 | | | **11** |
| 1,044,480 | 3 | | | **9** |

**Quoting**, nats a token (lower is better; a passage it cannot find costs 3 or more):

| Context | `none` | `yarn` | `linear` | `ntk` |
|---|---|---|---|---|
| 126,976 | 0.109 | 0.120 | 0.108 | 0.123 |
| 258,048 | 0.066 | 0.084 | 0.068 | 0.070 |
| 520,192 | 0.058 | 0.059 | 0.066 | 0.056 |
| 782,336 | 0.199 | | | 0.148 |
| 1,044,480 | 0.269 | | | **0.127** |

- **Past the trained length NTK and linear hold up; YaRN does not.** On the frozen text
  NTK recalls 93 of 128, against 72 with the trained frequencies and 68 with YaRN. Linear was as
  good as NTK in the sweep.
- **Inside it, the trained frequencies did best** in the sweep (37 of 38 at 28K and 127K, against
  33, 35 and 31), which is why `auto` keeps them there.
- **Quoting barely degrades** with NTK to 1M: 0.127 nats against 0.109 to 0.123 at 127K.
- **Recall among look-alikes degrades well before the context ends**, inside the trained
  length too. Over every run, 399 of 400 misses gave some five-digit code; 240 are identifiable
  as another fact's (only a third of the codes can be matched), and 171 of those 240 share the
  colour or the object asked for. With a fact every 16,384 tokens instead of 2,048, the trained frequencies
  recall 15 of 16 at 258K.
- The KV format is not the cause: FP16 KV recalls 17 of 24 at 258K where NVFP4 recalls 16.
- The numbers are noisy: greedy, one document, and the same model and method recalled 20, 20
  and 17 of 24 at 520K on three readings. The 1M figures (9 of 24 against 3) are from the sweep.

## A fully rotary model: MiniCPM5 2B

Qwen 3.5 rotates a quarter of each head, at a base of 10 million. MiniCPM5 2B rotates all of
it, at 5 million, and was trained to 128K: the ordinary case. It has no chat template here, so
the document is read as bare text (`--plain`), and the measure is repeating a passage from
earlier in it (nats a token, the mean of 8 depths):

| Context | `none` | `yarn` | **`ntk`** | `linear` |
|---|---|---|---|---|
| 12,288 | 0.040 | 0.071 | 0.035 | 0.946 |
| 61,440 | 0.068 | 0.077 | 0.076 | 1.019 |
| 126,976 | 0.290 | 0.255 | 0.236 | 1.493 |
| 192,512 | 0.655 | 0.168 | **0.117** | |
| 258,048 | 0.917 | 0.246 | **0.231** | |

Past its trained length the trained frequencies lose the passages; NTK keeps them best, with
YaRN close. Linear interpolation, which did well on Qwen, breaks this model even at 12K
(perplexity 10.5 against 4.8), so it is not a safe default.

`auto` is NTK because it was the best or joint best past the trained length on both.

### Memory and speed

| | |
|---|---|
| KV at 1M, NVFP4 | 9.25 KiB a token, 9.5 GB, and 1.2 GB for the MTP drafter |
| KV at 1M, FP16 | 36 GB: does not fit a 32 GB GPU |
| Reading 1M tokens | about 32 minutes on an RTX 5090 shared with a served model |
| Reading 258K, 520K | 2.0 and 6.6 minutes with the GPU to itself |
| Decode at 258K, 520K, 1M | about 58, 35 and 16 tokens/s |

Reading costs the square of the length, because every token attends to all before it. Reuse
makes it bearable: through the server, a 303,821-token conversation took 316 s to read and each
further question 1.2 s. For that, put the document in a turn before the question (the state is
saved at the last turn), not in the same message. Of three facts asked there, two came back
right.

## What had to change in the engine

- **Prompt attention in segments.** Prompt attention held every score of a chunk at once
  (`heads x 512 x keys`), and past 128K keys fell back to a kernel several times slower. Long
  spans now run 8,192 keys at a time, each segment softmaxed against its own maximum and
  merged into a running result, so the scratch is the same size at any length.
- **Decode attention in segments.** The merge kernel's shared memory ran out near 400K keys
  and its scratch was 4.3 GB at 1M. Caches longer than 64K keys are taken in segments too.
- **Decode reads each cached row once per head group** rather than once per query head. The
  results are the same to the bit; it is 5% faster at 61K and 15% at 520K.
- **The cache grows by a quarter past 128K rows** instead of doubling, and an allocation that
  fails leaves the engine as it was.
- **The server reads long prompts in slices** of 4,096 tokens and gives running replies a
  turn between them. With one 35K prompt being read beside a streaming reply, the longest gap
  in the reply was 2.0 s. Without room for a prompt, kept state leaves the GPU (oldest first)
  before the request fails. A cancelled request stops reading.
- **`--kv-exact-prefix auto` keeps at most 16,384 positions.** It kept the whole first user
  message at FP16, which for a 1M-token document is 32 GB.

## Reading the question's own attention

The model loses its way among similar passages far back long before it stops attending to
them. `GpuEngine.ReadAttention` reads a question and returns where the attention went, by
window of 32 positions, from the partial sums the decode kernels already compute.
`AttentionFocus` picks the passages that drew most, less what the same model attends to when
asked nothing in particular. `context bench --focus N` measures it.

Taking off idle attention matters: at 258K the fact's passage was among the 8 chosen for 23 of
24 questions, against 20 without. Reading those passages again beside the question, in place,
does not help: 20 of 24 against 19 without it at 258K, and 16 against 17 at 520K (where the
passage was found for 17). Whatever the passage's distance, the answer is still given from
inside the long context, among the same look-alikes. Answering from the passages in a fresh
conversation did not do better either: 16 of 24 at 258K, where the passage was found for 17
and plain recall was 13. It is kept as a measurement, not a feature.

## Choosing a method for another model

Measure it, inside its trained length as well as past it:

```bash
amql-cli --max-context 512K --rope-extend ntk --kv-format nvfp4 context bench <container> --text docs;src --lengths 32K,128K,256K,512K
```

Copy the text somewhere it will not change first: a file edited between two runs is a different
document. `--plain` reads the document as bare text instead and reports perplexity, for models without a
chat template. Do not score text inside a chat model's user turn: its predictions there were
never trained, and a passage it can quote perfectly costs several nats a token.
