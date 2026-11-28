# amql-server — HTTP endpoints for AMQL containers

`amql-server` loads one or more VINDEX3 containers and serves each on the endpoints its recorded facts say it
can answer. Nothing is guessed from a model's name: a container that cannot serve an endpoint is not offered
on it, and the startup log says why.

```bash
dotnet build -c Release src/Amql.Server
./bin/win-x64/Release/amql-server --model von=<von-container> --model chat=<qwen-container> --model emb=<embedding-container> \
                          --urls http://127.0.0.1:8000 [--api-key KEY] [--max-tokens N] [--no-warm] \
                          [--adapter [id=]<dir>[,<dir>…]] [--think] [--temperature T] [--top-p P] [--top-k K] \
                          [--config <models.json>] [--max-ram GB] [--gpu-memory GB] [--cpu] [--no-batch] [--batch-bf16]
```

[AMQL Studio](amql-studio.md) starts and stops the server with the models you choose and shows what it is
doing; everything below applies whichever way it is started.

| Container | Endpoints |
|---|---|
| Von decision model (ModernBERT + option-marker head) | `POST /v1/decisions`, `POST /v1/systemone` (OpenJEV's path), `POST /api/v1/decisions` (jevai.org's `{ code, message, data }` envelope) |
| Embedding container (`convert-to-embedding`, an `embedding` surface) | `POST /v1/embeddings` |
| Generative decoder with a chat template | `POST /v1/chat/completions`, `POST /v1/responses`, streaming (SSE) or not, with the reasoning and tools its [template](#chat-templates) writes |
| Every model | `GET /v1/models`, `GET /v1/models/{id}` (with `capabilities`, and for chat models `context_length`, `tools`, `thinking`, `adapters` and `defaults`), `GET /health` |
| The server itself | `GET /amql/status`: what it is doing (see [Status](#status)) |

A sequence classifier or a bare encoder loads but is offered no endpoint (the log notes `amql-cli classify`).

## Several models

One server serves any number of models, each under its own id, and a request picks one with `model`. Give them
as `--model id=<dir>` (repeatable), or in a `--config` JSON file, where each model also gets its own adapters
and defaults. Paths are relative to the file, comments and trailing commas are allowed, and an unknown setting
is refused at startup:

```jsonc
{
  "models": [
    { "id": "qwen-think", "path": "qwen3.5-0.8b", "adapters": ["adapters/stop-circling"],
      "think": true, "temperature": 0 },
    { "id": "qwen", "path": "qwen3.5-0.8b", "think": false, "max_tokens": 1024 }
  ]
}
```

`"kv_dir"` gives a model its own folder for paged-out prompt state (see [Concurrency](#concurrency)).
A model's own settings (`think`, `max_tokens`, `temperature`, `top_p`, `top_k`) come first, then the server's flags,
then the built-in defaults; a request's own fields override all of them. A model that thinks by default gets
4096 tokens unless something says otherwise. The same container may be served under several ids, here with and
without adapters; each loads its own copy. `GET /v1/models` lists every model with what it serves and the
defaults it applies, and `GET /v1/models/{id}` returns one (404 `model_not_found` otherwise).

## Routing, concurrency, auth

- A request's `model` names a loaded model by id (`--model id=<dir>`, default: the directory name). An unknown
  id is a 404 `model_not_found`; a model that does not serve the endpoint is a 400 `model_not_supported`.
  Omitting `model` is allowed when exactly one loaded model serves the endpoint. For decisions, `model` names
  TypeSafe's hosted Jev (`openjev`, `typesafe-ai/jev`), so any unmatched name falls back to the sole Von model;
  the response reports the version actually served.
- A chat model on the GPU engine batches its requests (see [Concurrency](#concurrency)). Anything else serves one
  request at a time per model; on the CPU, different models run in parallel, and with CUDA on the managed GPU
  path takes turns across all models, because its kernels stage through process-wide device buffers and two
  sessions on the GPU at once would overwrite each other's activations. Generation runs off the request thread,
  streams through a channel, and stops at the next token when the client disconnects.
- `--api-key` (or `AMQL_API_KEY`) requires `Authorization: Bearer <key>` on `/v1/*`, `/api/*` and `/amql/*`;
  `/health` is open. Without a key the server logs that it is unauthenticated — bind it to loopback.
- Weights are warmed at startup (one short forward pass per model) so the first request does not pay for
  widening them; `--no-warm` skips that.

## Memory and device

- `--max-ram GB` is the RAM the process aims to stay under. It selects the lean working set, as
  `amql-cli --max-ram` does, unless `AMQL_WEIGHTS` chose a mode: weights stream through a cache sized to fit.
  That is slower, but it serves models larger than memory.
- `--gpu-memory GB` caps the GPU memory the weights may take. The rest stay on the host, and 0 keeps every
  weight there. Without it the loader takes the device's free memory, less a margin.
- `--cpu` serves on the CPU even when a GPU is available.
- The GPU engine's KV cache is FP16 by default (`--kv-format fp16`): half f32's memory (32 KiB a token on
  Qwen 3.5 4B), a little faster decode at long contexts, and on the 4B no measurable difference from f32
  (perplexity 10.325 against 10.326, the same top token at all 8,192 positions). Paged-out files are half the size
  too. `--kv-format f32` (or `AMQL_KV_FORMAT=f32`) keeps the values as computed.
- `--max-context N` (`1M`, `512K`, a token count; `"max_context"` per model in `--config`) serves a model at a
  context length other than its own. A conversation that fits the trained length runs exactly as before; one
  that will go past it runs extended rotary frequencies (`--rope-extend`). Past 256K, use `--kv-format nvfp4` and
  a `--kv-gpu-budget` that holds the state (about 11 GB at 1M on Qwen 3.5 4B), or it is released after each
  reply. Reading 1M tokens takes about half an hour. See [Running a model past its trained
  context](context-extension.md).
- `--kv-format nvfp4` stores the GPU engine's KV cache as NVFP4: ~7x less memory a token on Qwen 3.5 4B
  (592 bytes a position instead of 4,096), faster decode at long contexts, and paged-out files 7x smaller.
  Replies differ slightly from f32 KV's (perplexity +0.4%, KL 0.008 on the 4B). See [The KV cache](kv-cache.md).
  With it, each conversation's system prompt, tools and first user message stay out of the 4-bit rounding
  (`--kv-exact-prefix auto`, the default; `N` tokens, or `0` for none), so the instructions every later token
  reads keep their precision (at most 16,384 tokens of them, so a document as the first message does not fill
  the GPU). They are kept at FP16 by default: half f32's memory, and on the 4B no measurable
  difference from f32 (KL under 0.00005, the same top token at all 8,192 positions). `--kv-exact-format f32`
  keeps them at f32, bit for bit an f32 cache's.

## Status

`GET /amql/status` reports what the server is doing, for a monitor to poll. AMQL Studio's Monitor reads it once
a second. It returns:

- `process`: pid, working set, private bytes, managed heap, CPU seconds, and whether the GPU is in use.
- `totals`: requests and failures since start, and requests and completion tokens over the last minute.
- `models`: for each model, its capabilities, the requests in flight, requests, errors, prompt and completion
  tokens, mean decode tokens/s, last use, context length, adapters and load notes.
- `active` and `recent` (the newest 50 of 200 kept): each request's path, model, status, prompt and
  completion tokens (streamed so far for an active one), seconds, `ttft_seconds` (time to first token, from
  when generation starts), `decode_tps` (tokens after the first over the time after it) and finish reason.

Health checks, `/v1/models` and status polls are left out of the counts and lists. The endpoint needs the API
key when one is set.

## Concurrency

A chat model the GPU engine runs is served by **continuous batching**. One thread owns the engine; each request
gets a sequence of its own on it (its own KV cache, recurrent state and position, over the one uploaded copy of
the weights), is prefilled when it arrives, and then joins the running batch: every step advances up to 8
requests by one token in a single pass, and a request leaves the batch the moment it finishes, so a waiting one
takes its place at once. Up to 64 requests per model are accepted; past 8 they wait their turn.

- **The same answers.** A request's tokens do not depend on what else is running: the projections share each
  weight read across the batch's rows, but every row keeps its own accumulation order, and attention and the
  Gated DeltaNet recurrence run per sequence. Tests hold batched logits to each sequence alone (within 1e-4,
  sequences joining and leaving), and many requests at once to the same answers as one at a time, greedy and
  seeded sampling alike.
- **Per step:** the GEMVs take up to 8 rows; attention and the recurrence take the whole batch in one launch each
  (per-sequence cache, state and position pointers in a table the captured graph reads); the head writes every
  row's logits to pinned host memory, and each request samples from its row in place, on its own core.
- **Entail adapters** run per request (each request gets its own copy), so `stop-circling` and `no-copy` batch
  too. The dense adapter form, which cannot be copied on the GPU, falls back to one request at a time.
- `--no-batch` turns it off. `--batch-bf16` runs batches of 4 or more on tensor cores (cuBLAS, activations
  rounded to BF16, as prompts already are): about 1.6x the throughput at 8, but a reply then depends slightly
  on what shares its batch.
- **Speculative decoding.** A reply running alone on a model with a drafter (Qwen 3.5's MTP layer, or one
  [attached](gemma-4.md#the-drafter) to its container) takes several tokens a step. The drafter proposes as many
  as have been holding, or the context does when the reply's last 8 tokens appeared in it before. The model runs them as rows of one pass, and the reply takes each row exactly as a plain step would:
  adapters, constraints and the sampler, in order, for as long as it chooses the next draft. The replies are
  the same token for token, greedy or sampled with a seed; on Qwen 3.5 4B they come 1.4–1.65x faster.
  `AMQL_MTP_DRAFTS=0` turns it off, `AMQL_SUFFIX_DRAFT=0` the context's drafts, and `AMQL_SERVER_TIMING=1` logs
  drafts kept and where a step's time goes. See [GPU inference performance](gpu-performance.md#speculative-decoding).
- **4-bit weights.** Point a model's `path` at a container compiled with `amql-cli quantise --format nvfp4` (or
  `mxfp4`) and the GPU engine runs its projections at 4 bits. On Qwen 3.5 4B decode is ~1.6x faster, and perplexity
  rises ~4% (NVFP4). Entail adapters are unaffected: they run their own programs over the tokens.
  `AMQL_GPU_FORMAT=nvfp4|mxfp4` quantises an ordinary container at load instead, the same weights the compiled
  container holds. See [GPU inference performance](gpu-performance.md#4-bit-weights-nvfp4-mxfp4).
- `/v1/models` gives each chat model's `concurrency` (8 when it batches, else 1) and, when it does not batch,
  why (`batching`). `AMQL_SERVER_TIMING=1` logs where a batch step's time goes; `AMQL_GPU_ENGINE_TIMING=1` the
  engine's launch, wait and copy.

Qwen 3.5 0.8B on an RTX 5090, agent-sized prompts with 12 tools, 128 to 256 tokens a reply:

| Clients | One at a time (`--no-batch`) | Batched, exact | Batched, `--batch-bf16` |
|---|---|---|---|
| 1 | 547 tok/s | 579 tok/s | 557 tok/s |
| 2 | 496 | 875 | 890 |
| 4 | 524 | 1,251 | 1,228 |
| 8 | 528 | 1,520 | 2,016 |
| 16 | 527 | 1,594 | 2,218 |

Mean latency at 8 clients drops from 1.7 s to 0.9 s (0.7 s with `--batch-bf16`), at 16 from 3.1 s to 1.4 s.

With Qwen's recommended sampling (temperature 0.6, top-p 0.95, top-k 20), exact: 315 tok/s for one client and 866
at 8, against 290 at any count one at a time, after a rewrite of the top-p filter (it ranked and hashed the whole 248k vocabulary each token; it now ranks
only the top-k survivors, with the same draws). The softmax over the whole vocabulary that remained has gone
too: with top-k up to 256, the sampler finds the k largest logits and the softmax's denominator in vectorised
passes (skipping blocks more than 30 below the largest, under float rounding of the sum) and gives a probability
only to the survivors; top-p and the draw then run over those, in the same order and with the same full-vocabulary
normalisation, so the draws match the whole-vocabulary path (`Sampler.SampleWholeVocabulary`, held to it seed for
seed in `SamplerTopKTests`). On a 248k vocabulary a draw takes 64 µs instead of 1,099 µs. An Entail adapter's
choice no longer walks the vocabulary with two set lookups per token, nor sums its softmax in scalar doubles:
on Qwen 3.5 4B, choosing a sampled token went from ~1,150 µs to 120-330 µs of a ~7 ms step, and in an agent's
tool calls from ~2,350 µs to 340-880 µs. The engine alone (`amql-cli bench <container> --batch 8`) does 668 tok/s at 1 sequence and 2,256 at 8,
3,577 with tensor-core batches.

**Prompt prefixes are reused.** An agent's calls share all but their last few hundred tokens: Hydra sends an
18k-token system prompt and tool list, and each call adds a reply and a tool result. Qwen 3.5 4B prefills such a
prompt in 1.5-2.9 s on an RTX 5090, which was half of every call. So the scheduler keeps finished sequences,
each with its prompt and the recurrent state saved at that prompt's mark (`GpuEngine.Mark`). A new prompt that
agrees with a kept one up to its mark rewinds that sequence there (`GpuEngine.Rewind`: the Gated DeltaNet
state and conv history are copied back, and the KV entries before the mark were never overwritten) and
prefills only the rest.

- **The mark** is at the prompt's second-last `<|im_start|>`. The last opens the reply's turn. The message
  before it is the newest, and an agent may move it: Hydra sends its notes as a user message after the
  newest message, so in the next call that position holds the reply instead. Everything before the mark is
  the same from one call to the next.
- **What it saves:** replaying eight calls of a Hydra turn on the 4B, prefill went from 1.5-2.0 s to 125-212 ms
  for every call after the first (17-20k tokens reused of 18-21k), and a 64-token reply from 2.2-2.8 s to 0.8 s.
- **Kept by time and memory, not by count.** Every finished sequence is kept, so any number of conversations
  pick up where they left off, until it has sat unused for `--kv-idle` seconds (default 300) or the kept
  sequences pass `--kv-gpu-budget` GiB of GPU memory (default 4), least recently used first. Then its memory is
  released. Each holds its KV cache at its capacity: about 2.3 GiB at 32k positions on the 4B (KV is f32, 64 KiB
  a token, plus the drafter's 8 KiB), 150 MiB for a 1.7k-token prompt on the 0.8B. `--kv-idle 0` keeps them
  until the budget needs the room.
- **Numerics:** a rewound sequence gives exactly the logits it gave at the mark, steps after it included
  (tested on plain and Gated DeltaNet models). A prompt prefilled in two pieces rather than one agrees with
  the one-piece prefill to f32 rounding in exact mode. In the default BF16 mode the split moves rounding,
  as any change of chunk boundaries does, so greedy replies can part on a near-tie (three of eight in the
  replay above, with similar wording).
- `--no-prefix-reuse` prefills every prompt whole. `AMQL_SERVER_TIMING=1` logs each admission's prompt, reused
  tokens and prefill time, and how far the prompt agrees with each kept sequence.

**Paging prompt state to disk** (off by default). With `--kv-disk`, a kept sequence that leaves GPU memory
(idle, over the budget, or at shutdown) is written to disk first, and a later prompt that starts the same way
loads it back instead of prefilling. It works for any model the GPU engine batches: what is written is exactly
what `GpuEngine.Rewind` needs (`GpuEngine.SaveMark`): the prompt's tokens up to its mark, every attention layer's
KV rows before the mark, the recurrent state and conv history saved there, and the MTP drafter's cache and
pending state. Rows after the mark are left out. `GpuEngine.RestoreMark` builds a new sequence from it that
carries on bit for bit as the one saved, drafter included (`KvPagingTests`).

- **Each model has its own folder:** `kv-cache` in its container by default (the NVFP4/MXFP4 copy has its own),
  `--kv-dir DIR` for `DIR/<model id>`, or `"kv_dir"` per model in `--config` (which turns the disk tier on for
  that model). Every file records the model's fingerprint, a hash of its engine geometry, prefill precision and
  container files, and one that does not match is deleted when found, so a rebuilt container never loads stale
  state. Files left from an earlier run are loaded at start when they are still in date.
- **Limits:** `--kv-disk-budget GB` per model (default 1), oldest deleted first; a state larger than the budget is
  not written (a 1.7k-token prompt on the 0.8B is 65 MiB, on the 4B 64 KiB a token, so 1 GiB holds ~15k tokens).
  `--kv-expire S` deletes a state that long after its last use (default 1800; 0 never). Prompts saved before
  fewer than 512 tokens are not written: they prefill faster than they load.
- **Off the batch:** the write runs in the background, and the read runs on the request's thread before it joins
  the batch, so other replies keep stepping. A state is taken (its file deleted) when it is loaded; it is written
  again with its newer mark when that sequence leaves GPU memory. A state still being written can be taken from
  memory.
- **Measured** on the 0.8B, a 1,707-token conversation idle for 5 s: paged out to a 65 MiB file, loaded back
  with 43 tokens left to prefill (99 ms for the whole 40-token reply against 39 ms from GPU memory), and the
  reply the same as the one picked up from GPU memory. Larger models and longer prompts gain the most: the 4B
  prefills an 18k-token prompt in 1.5-2.9 s, against ~1.2 GB to read back.
- **Privacy:** the files hold the conversation in a recoverable form. Keep the folder on a disk only the server's
  account can read; deleting a file on an SSD does not erase its blocks.

The log (stderr) names each page-out, restore, expiry and release.

**Device memory.** Keep the GPU's memory clear of other models. With three other models loaded beside the 4B
(and LM Studio and a game on the same card, 28.5 of 32.6 GB in use), the same 18k-token prefill took 32.6 s
instead of 2.9 s: its scratch buffers no longer fit, and the driver paged device memory to the host.

## Decisions

The body and response are TypeSafe's, answered by the same engine as `amql-cli decide` — see
[decision-models-von.md](decision-models-von.md). Invalid requests are 422: an OpenAI-shaped `error` on `/v1`,
`{ "code": 422, "message": …, "data": null }` on `/api/v1/decisions`.

## Embeddings

`input` may be a string, an array of strings, an array of token ids, or an array of token-id arrays. Text is
tokenised as HF's `tokenizer(text)` does (post-processor template included) and truncated to the context.
The container's recorded pooling (`mean`, `last` or `cls`) runs over the post-final-norm hidden states, then
L2-normalises if recorded. `encoding_format: "base64"` returns little-endian float32 bytes. `dimensions` must
equal the hidden size — the container records no Matryoshka truncation, so none is invented. Usage counts
prompt tokens.

## Chat completions and responses

Messages are rendered by the container's own chat template ([below](#chat-templates)) and tokenised without extra
special tokens, as `apply_chat_template` does.

- **Content**: strings or arrays of text parts (`text`, `input_text`, `output_text`); `developer` is a system
  message; the Responses API's `instructions` is a system message placed first.
- **Sampling**: `temperature` (0 = greedy; default 1), `top_p`, `top_k` (an extension; 0 = off), `seed`, `stop`
  (up to 4 strings, never streamed past), `max_tokens` / `max_completion_tokens` / `max_output_tokens` (default
  `--max-tokens`, bounded by the context). `--temperature`, `--top-p` and `--top-k` set the defaults for requests
  that leave them out. Generation stops at the tokens named by `generation_config.json`'s `eos_token_id`, the
  tokenizer's `eos_token`, and the tokens the template ends a turn with, after words and after calls. Other sampling fields (penalties, `min_p`) are
  ignored.
- **Text** is decoded from the accumulated bytes at every step, so a character split across tokens is sent once,
  whole.
- **Thinking** is on when `chat_template_kwargs.enable_thinking` is true; else when `reasoning_effort` (chat) or
  `reasoning.effort` (Responses) is anything but `none`/`minimal`; else when the server runs with `--think`. The
  thinking block comes back apart from the answer: `reasoning_content` on the chat message (and in stream
  deltas), a `reasoning` output item with `reasoning_text` in Responses. Sent back on a later turn
  (`reasoning_content` on an assistant message, or a `reasoning` input item), it is rendered as the template
  renders it. The other `chat_template_kwargs`, and `reasoning_effort`, are passed to the template.
- **Finish**: `stop`, `length` or `tool_calls` for chat; Responses reports `completed`, or `incomplete` with
  `incomplete_details.reason: "max_output_tokens"`.
- **Streaming**: chat sends `chat.completion.chunk` deltas (role first, `finish_reason` last, a `usage` chunk
  with `stream_options.include_usage`) and `data: [DONE]`. Responses sends `response.created`,
  `response.in_progress`, then for each output item `response.output_item.added`, `response.content_part.added`,
  its deltas (`response.reasoning_text.delta`, `response.output_text.delta`, or
  `response.function_call_arguments.delta`), their `.done`, `response.content_part.done`,
  `response.output_item.done`, and finally `response.completed` or `response.incomplete`, each with a
  `sequence_number`. Nothing is sent before it is certain: text that could begin `</think>` or a call
  waits for the next token.
- **Refused** (400, OpenAI error shape), rather than silently ignored: legacy `functions`, `n > 1`, audio and
  other content parts, images where the model reads none, tool types other than functions, and the Responses API's
  `conversation`.

### Chat templates

A model is served by the template it ships (`chat_template.jinja`, or `chat_template` in
`tokenizer_config.json`), run by AMQL's own Jinja interpreter. It keeps to transformers' rules (trimmed blocks,
`tojson`, `raise_exception`, `strftime_now`). On 23 models' templates, 966 conversations came out exactly as
transformers renders them. What a template raises is the request's 400.

No family is named in the code. When a model loads, the server renders probe conversations and compares what comes
out to learn:

| | from |
|---|---|
| What opens and closes thinking, and the empty block that stands for none | an assistant message with reasoning; the prompt with thinking on and off |
| What ends a turn, opens the next and opens a tool's result | conversations with one more message each |
| Whether tools, calls, results and images are written at all; what is not is refused | the same conversations with and without them |
| How calls are written | calls with no arguments, one, two and a number, and two calls in one turn |

Calls come in four forms:

| Form | Example | Models |
|---|---|---|
| JSON object | `<tool_call>{"name": "f", "arguments": {"k": "v"}}</tool_call>`, `[TOOL_CALLS]f[ARGS]{"k": "v"}` | Qwen 3, Granite, Mellum, Mistral |
| Parameters one by one | `<function=f><parameter=k>v</parameter></function>`, `<param name="k">v</param>`, `<arg_key>k</arg_key><arg_value>v</arg_value>` | Qwen 3.5 and later, Nemotron, MiniCPM, GLM |
| Python call | `<\|tool_call_start\|>[f(k='v')]<\|tool_call_end\|>` | LFM 2.5 |
| Object with marked strings | `<\|tool_call>call:f{k:<\|"\|>v<\|"\|>}<tool_call\|>` | Gemma 4 |

Each form is checked by writing calls with the template, reading them back and holding them to their tools: on the
models' own templates, and for Mistral's and GLM's on templates cut down to the same form. Live, on 32 tool tests
each (one call, several, loops, forced, streamed, with thinking and without): Qwen 3.5 4B 32, Gemma 4 26B 31,
MiniCPM5 2B 31.

The log says at load whether a model serves tools, and why not when it does not. Refused: a template that writes no
calls or no results (Gemma 3, Phi-3), and calls that start with no mark of their own (gpt-oss's, which start
in the middle of a header). LFM 2.5 1.2B's template writes strings between quotes with nothing escaped, so a call
whose text holds a quote before a comma or a bracket cannot be read for certain.

A conversation is rendered as transformers would render it, with three exceptions. A result that names no call
answers the calls in order. What was said along with a call is written before it where the template would put it
after the call's results. And a reply that goes on from an image in a tool result, with thinking off, starts with
the empty thinking block.

### Calls held to the tools

A small model writes calls the harness cannot run: a parameter the tool does not have (`edit_file` with
`content`), an array parameter whose JSON does not parse, a tool that does not exist. So once a reply opens
a call (outside thinking), the rest of it is held to the request's tools by the same constrained
decoding as structured output, until it closes. In the form Qwen 3.5 writes, and likewise in the others:

- `<function=NAME>` names a declared tool; each `<parameter=K>` is one of its parameters, not given yet;
  `</function>` comes only once every required parameter is there.
- A string parameter (or an untyped one) is free text up to `</parameter>`; a string enum is one of its values,
  bare; any other type (array, object, integer, number, boolean) is JSON its schema allows, then `</parameter>`.
- Calls the model writes are therefore always ones the harness can parse: on a random model forced to call a
  tool (`tool_choice: required`), every reply is a call with the declared name and typed, parseable arguments.

`--free-tool-calls` turns it off.

**What a schema says in words.** Some rules a schema states only in its descriptions, and a small model breaks
every one: Hydra's `edit_file` edits mark `find`, `find_lines`, `replace` and `replace_lines` "(replace_text
only)", and three pairs "set one or the other, never both", and in one session Qwen 3.5 0.8B sent `op: replace` with
`find`/`replace`, then `replace` beside `replace_lines`, and was refused each time. So in an object with a
required string enum (the edit's `op`), a property whose description says it is only for some of the enum's
values ("(replace_text only)", "insert_* only", "only for replace/delete") is offered only after the enum has one
of them, and of two properties a description says to set "X or Y, never both", only one may be written.
Descriptions without such words change nothing.

**Mentioned tools.** Told "write a new version from scratch with create_file (overwrite true), no need to read the
old one", the 0.8B read the file and patched it with `edit_file`, whose anchors it cannot write. So the first call
after the user's message, when the message names tools (as whole words, not in a clause that says not to: "no
need to", "don't use"), names the first of them, provided the model gives it 1e-4 of its probability. Later calls
are the model's own: there it is working from results, and a redirect turned a reply that meant to read the file
it had to fix into another build. Replayed, the step that had begun with `read_file` began with `create_file` and
a whole program. `--no-mention-rule` turns it off with the parameter rule.

**Tried and dropped: copying the task.** Given the code to write ("sum new FileInfo(f).Length over
Directory.EnumerateFiles(folder, …)"), the 0.8B wrote `new FileInfo(folder)` and an invented `.FileList`. A prior
that, where the reply ended with a stretch of the user's message, gave the stretch's next tokens a share of the
probability (kNN-LM style, λ 0.5) did not pay. At a 16-character match it never acted, because the reply leaves
the user's words before copying that much. At 10 characters with an absolute floor (0.1%) it pulled the user's
prose into the code (`foreach (…) print the name and bytes / 1048576.0 formatted with …`). With a floor relative to
the model's first choice it was safe but sampled programs built 2/8 times with it and 3/8 without. The mistake is
in the model's plan, not in drifting from text it is copying, so the code was removed.

**Mentioned parameters.** A small model plans in words and then writes the call by habit: asked to replace a file
that exists, Qwen 3.5 0.8B said "I'll overwrite it", the tool answered "Pass overwrite: true", the user said so
too, and still, measured, it chose `content` over `overwrite` at 0.995 to 0.005, and after `content` closed the
call at 0.999, so the call failed the same way every time. So where a call can open a parameter, an optional
parameter not given yet that the current task names, as a whole word, in the user's latest message, a tool
result since, or the reply's own thinking, is opened next, provided the model gives the way there at least
1e-4 of its probability. Its value is still the model's, under the grammar. In the agent session it took
`create_file … overwrite: true` from failing five times in a row to succeeding first time. A tool result counts
only where it tells the call what to pass ("Pass overwrite: true", "set 'force' to true"): results name many
fields (`"handle": "bg-95c9"`) and warn about some ("'handle' means poll or stop"), and those are not hints.
`--no-mention-rule` turns it off.

**Announced calls.** A reply with tools on offer that ends, outside thinking and outside a code block, on a colon
after saying it is about to act ("Now let me build it:", "I'll run the tests:") has promised a call; ending the
turn there leaves the harness nothing to run, and a small model then claims the build succeeded. So there the
call opens, when the model gives the way to `<tool_call>` at least 1e-4 of its probability. A colon after
anything else ("Here's what was done:") introduces what follows, so the reply may not end on it but no call is
forced: forcing one there sent the agent round again, re-running a program it had just run. `--no-announce-rule`
turns it off.

**Copied line decoration.** A harness may show a file with each line marked (`L42:a3f9c1│code`); a small model that
reads a file and writes it back copies the marks with the code, and every line fails to build (Hydra's agent did
exactly this, then copied the copy). So the server learns the decoration from the request's own tool results (a
line start up to a separator, with a varying number or hash, on 80%+ of a result's lines; no harness is named
in the code) and, in a call's string value where 80%+ of the lines carry it, removes it, along with the result's
header lines copied above it. The shorter forms the model also writes (`L42:code`, the hash dropped) and a copy
of a copy (marked twice) are removed too. Values that merely contain a mark or two are left alone. The log says
when it acts.

The model is also shown the results without the decoration: a 0.8B model read `L4:e3b0c4│` on a blank line as a
missing line, listed 18 lines of a sound file as missing and patched them one by one; shown the same file plain,
it restated the six requirements of the task and wrote the program. A JSON result is written back compactly with
its text unescaped; one with nothing decorated is sent as it came. The cost: anchors such as `L42:a3f9c1` are not
in view, so a tool that edits by them is out of the model's reach (it had not used one successfully).
`--keep-line-marks` shows the results as they came.

**Repeated calls.** At temperature 0 a small model copies its own previous call from the conversation: it ran the
program, got the output, and ran it again, and again. A call (tool and arguments, key order ignored) that has
already been made at least twice, whose latest result is one it returned before (numbers and hashes such as build
ids and durations ignored; not only the time before, since a harness may answer a re-read with "unchanged since
your last read", so the file and that note take turns, and the model read one file eight times), is not made
again: the reply is generated again from what it said before the call, with calls closed (the call's markup
cannot be written as text either), after a lead-in that says why ("That run_terminal call has already returned
the same result twice, so instead of making it again, here is where things stand:"). After a run the agent then
reports the output; after an edit, where the build stands. An earlier lead-in, "here is what it showed", led it
to paste an old version of the file as if the edit had returned it. `--allow-repeated-calls` turns it off.

**The user's turn.** The rules that read "the user's message" take the turn as beginning at the user's own
message: user messages straight after it (a harness's retrieved notes and hints) belong to it, and a user message
after a tool result (a harness's "[System] same action 3 times") is a note inside the turn, not a new one. When
Hydra moved its notes into a message of their own after the user's, the task the user gave had vanished from
"the newest user message", and the first call went back to reading the file. The mentioned-tool rule reads the
user's own message only, since the notes name tools too ("tool:edit_file (graph proximity 0.45)").

### Continuing a reply

`continue_final_message: true` (vLLM's name) on chat completions treats the last message, an assistant message
with text, as the start of the reply: the prompt is the conversation's generation prompt followed by that text,
and the model carries on from it. The reply holds only the continuation. With `logprobs`, it shows what a model
would write at any point of a reply, which is how the call habits above were measured.

### The thinking budget

A small reasoning model often has its answer long before it stops thinking, and a reply that runs out of tokens
while still thinking has no answer at all. `--thinking-budget N` (or `"thinking_budget"` in a model's config, or in
the request, top level or in `chat_template_kwargs`) closes the block after N thinking tokens, at the end of the
line, and lets the answer follow. It closes earlier when the reply's own `max_tokens` would leave too little room: a
quarter of the limit, at most 512 tokens, is kept. 0, the default, is no limit. On 80 held-out hard problems
Qwen 3.5 4B got 60 thinking without a limit and 80 with a budget of 1,536 (79 at 2,048 and 3,072; see
[Reasoning](entail-reasoning.md#the-4b)). It applies to free text only; structured output never thinks.

### The loop guard

A small model can fall into repeating itself: the same line of a file, again and again, until the token limit,
which at a long agent context (47k tokens, ~140 tok/s) is minutes of nothing visible, since text inside an
open call is only sent once the call closes. The server watches each reply's own tail: once one stretch (up to
256 tokens) has repeated back to back at least 4 times and 192 tokens in all, a reply outside thinking ends
there (`length` / `incomplete`; an open call comes back as text and is never run), and thinking is closed
instead so the answer can follow. Only the reply is read, never the prompt, so copying a path or command from
the conversation is never taken for a loop (what broke `no-copy`); lines that vary, even by one token, are not a
loop either. A counting loop is the exception: lines that differ only in a number or a hash (the model writing
`L2196:c74105│ //`, `L2197:c74105│ //`, … for 32,000 tokens) are a loop once that has held for 16 rounds and
512 tokens, far longer than a numbered list. A reply returned whole (not streamed) that the guard stopped comes back as its
first round, finished (`stop` / `completed`), since it said what it had to say once and a client that sees
`length` asks for more. The log says when it acts. Hydra's request that looped on one `this.Controls.Add(...)`
line for 32,000 tokens now ends after 424, in 4.7 s. `--no-loop-guard` turns it off; structured output is exempt
(its grammar bounds it).

### Structured output

`response_format` (chat: `{"type":"json_schema","json_schema":{"name","schema","strict"}}` or
`{"type":"json_object"}`) and `text.format` (Responses: `{"type":"json_schema","name","schema","strict"}` or
`{"type":"json_object"}`) hold the reply to JSON by constrained decoding: at every token only those that keep the
reply a value the schema allows can be drawn (the request's own temperature, top-p and top-k over just those),
and the reply can end only where the value is complete. So the reply always parses and follows the schema,
whatever the model would have written; a token limit can still cut a long one short (`length` / `incomplete`).

- **The schema subset enforced:** `type` and type lists (`["string","null"]`), `properties` in their declared order
  (an optional one may be left out, a required one may not), `required`, no undeclared properties, `items`,
  `enum`, `const`, `maxLength`, `anyOf`/`oneOf` (merged), and `$ref` into the schema's own `$defs`/`definitions`.
  Other keywords (`pattern`, `minimum`, `minItems` …) are not enforced; a remote `$ref` is a 400. `json_object`
  takes any object.
- **Bounded:** whitespace between tokens is limited (24 characters in a row) and a string without `maxLength` to
  8,000 characters, so a reply cannot spin forever on either.
- **The model is told:** the schema is appended to the system message ("Respond with a single JSON value … that
  follows this JSON Schema"), so a small model knows what it is writing; the constraint only decides what it may
  write. A structured reply has no thinking block and no tool calls.
- **Cost:** the 256 likeliest tokens are checked against the grammar each step, the whole vocabulary only when
  none fits; works in batches too. Hydra's 7,000-token transcript assessment comes back in 0.8 s.

### Tools (function calling)

Where the [template](#chat-templates) writes tools, calls and results, requests may carry `tools`: chat's `{"type":"function","function":{name, description, parameters}}`, rendered exactly as the
client sent them, or the Responses API's flat `{"type":"function", name, description, parameters}`.

- **Calls** in the reply become `tool_calls` (chat, `finish_reason: "tool_calls"`) or `function_call` output
  items (Responses), each with a `call_…` id and its arguments as a JSON object string. A parameter is read as
  the type its schema gives it — integer, number, boolean, object or array — and a value that does not read as
  that type stays a string for the client to report. Text the model writes after its last call is dropped, as
  the template asks; a call that never closed (the token limit cut it off) is returned as content, never as a
  call. In a stream, each call is sent whole once it closes.
- **History**: assistant messages with `tool_calls` and `tool` messages (chat), or `function_call` and
  `function_call_output` items (Responses), are rendered as the template renders them.
- **`tool_choice`**: `auto` (default); `none` leaves the tools out of the prompt; `required` writes
  a call's opening for the model, and a named function writes the whole opening of its call, so the
  model can only continue a call — after its thinking block when it thinks. A function the request does not
  offer is a 400.
- **`parallel_tool_calls: false`** keeps the first call, and stops generating once it closes.
- **`previous_response_id`**: responses are kept in memory (the newest 512, unless `"store": false`), so a
  request can continue one without resending it; the earlier response's `instructions` are not carried over, as
  in OpenAI's API. An unknown id is a 404 `previous_response_not_found`.

### Images

A chat model whose container carries a vision tower this build runs (Gemma 4's) reads images: `images` is true
in `/v1/models`. Chat's `{"type":"image_url","image_url":{"url":"data:image/png;base64,…","detail":"auto"}}` and the
Responses API's `{"type":"input_image","image_url":"data:…"}` go in user messages and tool results. PNG and JPEG are
read, and only `data:` URLs: the server fetches nothing on a client's behalf. A request may be 256 MB.
See [gemma-4.md](gemma-4.md#images).

### Entail adapters

`--adapter [id=]<dir>[,<dir>…]` runs [Entail](entail.md) adapters beside a chat model (`id=` picks one model; a
bare list goes to every chat model). Each rules on the next token as the reply is generated — the model's
weights are never changed — so a program such as `stop-circling` (close the thinking block once the model
repeats itself, see [entail-reasoning.md](entail-reasoning.md)) works through the API as it does in
`amql-cli entail bench`. While an adapter is speaking, `logprobs` still describe the model's own distribution. Not every program suits an agent: `no-copy`, which rules out the
token that would continue a run already written three times, broke Hydra's tool calls. The working directory
`D:\Temp\ModelTest` appeared three times in the conversation, so it ruled out `Test` and the model listed
`D:\Temp\Model`; replayed without it, the same request gives the right path. Serve agents with `stop-circling`
alone, which acts only inside the thinking block.

A container may carry its own adapters (`amql-cli attach <container> --adapter <dir>,…`, in `adapters/`). A
model given no adapters, by `--model` or by a `--config` entry with no `"adapters"`, runs those, in the order the
container lists them; `"adapters": []` runs none, and a name in `"adapters"` that is no folder is one of the
container's own (`"adapters": ["stop-circling"]`). Adapters read an image's tokens as tokens they do not know, so
images and adapters run together.

```bash
dotnet bin/win-x64/Release/amql-server.dll --model qwen=<qwen3.5-0.8b container> --think \
    --adapter qwen=<adapters>/stop-circling
```

## Verification

`tests/Amql.Tests/ServerTests.cs`, against goldens from transformers and the Von SDK:

- chat rendering equals HF `apply_chat_template` with Qwen3.5's real template (6 conversations, thinking on and
  off, a history turn with a think block, non-ASCII text);
- greedy generation equals transformers' `generate` token for token on a GatedDeltaNet + attention model;
- embeddings equal transformers' mean-pooled, normalised hidden states (1e-5);
- decisions return the SDK's answers, on `/v1` and in the jevai envelope;
- streamed and non-streamed chat produce the same text; the Responses event sequence; base64 embeddings;
  the refusals; the API key.

`tests/Amql.Tests/ServerToolTests.cs`, for tools and reasoning:

- prompts with tools, calls, parallel calls, tool results, reasoning and history turns equal HF
  `apply_chat_template` with Qwen3.5's template (6 conversations, from `fixtures/make_tool_chat_golden.py`);
- a reply splits into reasoning, content and typed calls, and fed one character at a time gives the same result
  without ever sending part of a marker; a cut-off call is content; without tools a `<tool_call>` is text;
- calls held to the tools (`ToolCallConstraintTests`, `ServerToolTests`): an agent harness's `create_file` and
  `edit_file` calls accepted however split, and refused for an unknown tool or parameter, a parameter given twice,
  a required one missing, a trailing comma or wrong type in a JSON value, a value outside a string enum, or
  anything but the close after a value; a random model's forced calls always name the tool with typed arguments;
  the way to an optional parameter from every point a parameter can open; `continue_final_message`'s prompt;
- structured output (`JsonConstraintTests`, `ServerToolTests`): Hydra's assessment schema accepted however a reply
  is split and refused for a skipped required property, an early close, an undeclared property, a value outside
  an enum, a wrong type, a bad escape, text after the value or a code fence; numbers end only when whole; a random
  model's replies, sampled at temperature 1 with thinking asked for, parse and follow the schema on chat and
  Responses, on the managed path and the GPU engine;
- batching: 14 concurrent requests per model (more than a batch, greedy and sampled, thinking on and off,
  with and without an Entail adapter) equal the same requests one at a time, on the managed path and on the GPU
  engine (`fixtures/qwen-lm-engine`, a tiny Qwen3.5 the engine runs); a client that goes away leaves the batch
  while the others finish; `GpuBatchTests` hold batched engine logits to each sequence alone on four families;
- several models: the listing and its defaults, `/v1/models/{id}`, per-model defaults and request overrides,
  answers from two models at once equal to their answers alone (this caught the shared device buffers), and
  the config file;
- over HTTP: tool turns reach the prompt token for token, a named `tool_choice` starts the call, thinking
  returns `reasoning_content`, Responses take function items and continue a stored response, and a streamed
  reasoning item opens and closes.
