# Kaja-gem 26B A4B

Gemma 4 26B A4B as AMQL serves it best, in one container: the long-context edit, NVFP4 weights
rounded by GPTQ, its drafter and vision tower, and five Entail programs that run beside it. The
model card that goes with it is [model-cards/kaja-gem-26b-a4b.md](model-cards/kaja-gem-26b-a4b.md).

| | Gemma 4 26B A4B (NVFP4, GPTQ) | Kaja-gem 26B A4B |
|---|---|---|
| Recall of one fact among 126 at 256K | 14/48 | **38/48** |
| Dates, number bases, logic, JSON validity (1,156 cases) | 267 | **1,151** |
| Hard reasoning suite, thinking, 3,072 tokens | 2/80 | **77/80** |
| Hard reasoning suite, no thinking | 80/80 | 80/80 |
| Need-a-tool cases: right calls, and silent when none is due | 36/36, 24/24 | 36/36, 24/24 |
| Decode, one reply with the drafter: prose, explanation, code (tok/s) | 274, 337, 433 | 254, 316, 411 |
| Decode, 80 replies in turn, no drafter (tok/s) | 242 | 236 |
| Size | 15.6 GB | 15.6 GB, the five adapters 0.74 MB of it |

## What is in it

- **The weights** of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
  (Apache 2.0) at NVFP4, rounded by GPTQ, with the embedding ([gemma-4.md](gemma-4.md)).
- **The long-context edit**: the query norms of its five global layers ×1.5, found by the search in
  [long-context-editing.md](long-context-editing.md). Nothing was trained.
- **Its drafter** (Google's assistant model) and **its vision tower**, attached.
- **Five Entail adapters**, in `adapters/`, which a server runs beside the model unless told otherwise
  ([entail-adapter.md](entail-adapter.md#carried-in-a-container)):

| Adapter | Does | Beside Gemma | Gemma alone |
|---|---|---|---|
| `stop-circling` | closes the thinking at a line break once it repeats itself, or after 1,536 tokens | 77/80 (78/80 at 4,096 tokens) | 2/80 |
| `dates` | weekdays, date sums and differences, leap years, month lengths | 250/250 | 29/250 |
| `bases` | whole numbers between bases 2 to 16 | 312/315 | 21/315 |
| `logic` | propositional constants, truth tables, tautologies | 225/226 | 14/226 |
| `json` | whether a JSON text is valid, and no ending inside one | 364/365 | 203/365 |

The four formal programs answer exactly, in doubles, where the question is theirs, and are silent
otherwise, so what none of them reads the model answers as before. Of the five misses, three are
bases' right number with the model running on after it (`66237A`, `48879y`); logic's is brackets two
deep, which it does not read, as on Qwen; JSON's is a spoilt word it took for valid.

**Thinking.** Asked to think with 3,072 tokens to do it in, Gemma alone closes its thinking on 2 of
the 80 hard problems, both right; on the other 78 it is still thinking at the limit, most of them going
round the same steps again. Beside `stop-circling` all 80 close, and 77 are right. With 4,096 tokens
it is 78: both misses closed their thinking early and then ran their reply to the limit.

**No thinking.** Gemma answers all 80 without thinking, and the five programs together leave every one
of those answers as it was: no problem won or lost, and no token steered.

### Left out: `call-gate`

`call-gate` opens a call on the reply's first token when the request touches a concept a tool covers.
Gemma does not need it: on the 60 need cases it calls every one of the 36 that need a tool, with the
right values, and none of the 24 that need none. Beside it the gate calls 8 of those 24, on questions
that only mention a tool's subject ("Who invented the egg timer?" gets `set_timer`; "What does CC mean
in an email?" gets `send_email`). It is built and measured, and can be attached.

## Making it

From a copy of the base container (`g4-qs1.5`: the GPTQ container with the edit, see
[long-context-editing.md](long-context-editing.md); it holds the vision tower already):

```bash
amql-cli attach containers/kaja-gem-26b-a4b --drafter google/gemma-4-26B-A4B-it-assistant
```

Each adapter is built against the container's own tokenizer. The generated programs are written for
its vocabulary first:

```bash
python scripts/entail/make_bases.py containers/kaja-gem-26b-a4b/tokenizer.json bases.entail.txt
python scripts/entail/make_logic.py containers/kaja-gem-26b-a4b/tokenizer.json logic.entail.txt
python scripts/entail/make_json.py containers/kaja-gem-26b-a4b/tokenizer.json json.entail.txt
```

```bash
amql-cli entail adapter scripts/entail/reasoning/stop-circling.entail.txt --tokenizer containers/kaja-gem-26b-a4b --out adapters/stop-circling
```

and likewise `dates` (from `scripts/entail/dates.entail.txt`), `bases`, `logic` and `json`. Then:

```bash
amql-cli attach containers/kaja-gem-26b-a4b --adapter adapters/stop-circling,adapters/dates,adapters/bases,adapters/logic,adapters/json
```

The model card goes in as `README.md`, and the folder is published with
`amql-cli upload containers/kaja-gem-26b-a4b <owner>/Kaja-gem-26b-a4b` (private unless `--public`).

## Serving it

```bash
amql-server --model kaja=containers/kaja-gem-26b-a4b --kv-format nvfp4
```

The five adapters run beside it, and images, tools and the drafter work with them. `"adapters": []` in a
`--config` entry serves the model alone, and `"adapters": ["stop-circling"]` only that one.

## What was found on the way

Porting the programs from Qwen to Gemma turned up three faults that touched any model whose
tokenizer is not Qwen's, all fixed:

- **Spelling.** A text was looked up in its byte-level spelling first. Gemma's vocabulary holds `Ġ` and
  `Ċ` as letters, so `" "` and `"\n"` bound to them, never to Gemma's space and newline: dates read no
  gap, and stop-circling would never have seen a line break.
- **Declarations.** A token declared a second time lost what the first declaration meant. Gemma's
  `<channel|>` both closes reasoning and leads the reply, so `call-gate` could never fire on it.
- **Cut operators.** Gemma writes ` <->` as ` <` and `->`, and `->` is an operator of its own: logic
  read a biconditional as an implication and answered 179 of 226. Read as one operator, 225.

`entail eval`, `side` and `bench` ran any model too large for f32 in host memory from disk, as a
graft must run. A quantised container is no graft, so they run it as served, on the GPU.

With the drafter on, the five adapters first cost one reply 35–45% of its speed (163, 194 and 218 tok/s
against 253, 310 and 395). Speculation fed them each kept token after the GPU had verified it, one adapter
after another, so their 1.3 ms per token (bases is most of it) was added to every step. An adapter can now
take back the latest tokens it read, so a step feeds them the token and all its drafts while the GPU verifies,
and takes back those not kept: 254, 316 and 411 tok/s against 274, 337 and 433 without them, and the same
replies, hash for hash.

## Not done

- The faster and smaller quantisation, and the engineered experts (BAR, [moe-bar.md](moe-bar.md)),
  meant for this release. Both are next.
- Images and long contexts were not measured with the adapters on.
- The hard suite is one seed, greedy.
