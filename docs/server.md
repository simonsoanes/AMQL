# amql-server — HTTP endpoints for AMQL containers

`amql-server` loads one or more VINDEX3 containers and serves each on the endpoints its recorded facts say it
can answer. Nothing is guessed from a model's name: a container that cannot serve an endpoint is not offered
on it, and the startup log says why.

```bash
dotnet build -c Release src/Amql.Server
./bin/Release/amql-server --model von=<von-container> --model chat=<qwen-container> --model emb=<embedding-container> \
                          --urls http://127.0.0.1:8000 [--api-key KEY] [--max-tokens N] [--no-warm] \
                          [--adapter [id=]<dir>[,<dir>…]] [--think] [--temperature T] [--top-p P] [--top-k K] \
                          [--config <models.json>] [--no-batch] [--batch-bf16]
```

| Container | Endpoints |
|---|---|
| Von decision model (ModernBERT + option-marker head) | `POST /v1/decisions`, `POST /v1/systemone` (OpenJEV's path), `POST /api/v1/decisions` (jevai.org's `{ code, message, data }` envelope) |
| Embedding container (`convert-to-embedding`, an `embedding` surface) | `POST /v1/embeddings` |
| Generative decoder with a chat template this build renders | `POST /v1/chat/completions`, `POST /v1/responses` — streaming (SSE) or not; tools and reasoning for Qwen3.5-family templates |
| Every model | `GET /v1/models`, `GET /v1/models/{id}` (with `capabilities`, and for chat models `context_length`, `tools`, `thinking`, `adapters` and `defaults`), `GET /health` |

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
- `--api-key` (or `AMQL_API_KEY`) requires `Authorization: Bearer <key>` on `/v1/*` and `/api/*`; `/health` is
  open. Without a key the server logs that it is unauthenticated — bind it to loopback.
- Weights are warmed at startup (one short forward pass per model) so the first request does not pay for
  widening them; `--no-warm` skips that.

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
only the top-k survivors, with the same draws). What remains of sampling's cost is the softmax over the whole
vocabulary. The engine alone (`amql-cli bench <container> --batch 8`) does 668 tok/s at 1 sequence and 2,256 at 8,
3,577 with tensor-core batches. Prompt prefill on the engine runs at 30-50k tok/s, so reusing prompt prefixes
between requests would save little and is not done.

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

Messages are rendered with the container's own chat template, reproduced exactly for the families this build
knows — Qwen3/Qwen3.5 ChatML (trimmed contents, `<think>` handling, the empty think block unless thinking is on,
tools, calls and tool results), plain ChatML and Llama 3 — and tokenised without extra special tokens, as
`apply_chat_template` does. Any other template is refused at load rather than approximated.

- **Content**: strings or arrays of text parts (`text`, `input_text`, `output_text`); `developer` is a system
  message; the Responses API's `instructions` is a system message placed first.
- **Sampling**: `temperature` (0 = greedy; default 1), `top_p`, `top_k` (an extension; 0 = off), `seed`, `stop`
  (up to 4 strings, never streamed past), `max_tokens` / `max_completion_tokens` / `max_output_tokens` (default
  `--max-tokens`, bounded by the context). `--temperature`, `--top-p` and `--top-k` set the defaults for requests
  that leave them out. Generation stops at the tokens named by `generation_config.json`'s `eos_token_id`, the
  tokenizer's `eos_token`, and the template's end-of-turn marker. Other sampling fields (penalties, `min_p`) are
  ignored.
- **Text** is decoded from the accumulated bytes at every step, so a character split across tokens is sent once,
  whole.
- **Thinking** is on when `chat_template_kwargs.enable_thinking` is true; else when `reasoning_effort` (chat) or
  `reasoning.effort` (Responses) is anything but `none`/`minimal`; else when the server runs with `--think`. The
  thinking block comes back apart from the answer: `reasoning_content` on the chat message (and in stream
  deltas), a `reasoning` output item with `reasoning_text` in Responses. Sent back on a later turn
  (`reasoning_content` on an assistant message, or a `reasoning` input item), it is rendered as the template
  renders it.
- **Finish**: `stop`, `length` or `tool_calls` for chat; Responses reports `completed`, or `incomplete` with
  `incomplete_details.reason: "max_output_tokens"`.
- **Streaming**: chat sends `chat.completion.chunk` deltas (role first, `finish_reason` last, a `usage` chunk
  with `stream_options.include_usage`) and `data: [DONE]`. Responses sends `response.created`,
  `response.in_progress`, then for each output item `response.output_item.added`, `response.content_part.added`,
  its deltas (`response.reasoning_text.delta`, `response.output_text.delta`, or
  `response.function_call_arguments.delta`), their `.done`, `response.content_part.done`,
  `response.output_item.done`, and finally `response.completed` or `response.incomplete`, each with a
  `sequence_number`. Nothing is sent before it is certain: text that could begin `</think>` or `<tool_call>`
  waits for the next token.
- **Refused** (400, OpenAI error shape), rather than silently ignored: legacy `functions`, `n > 1`, audio, image
  or other non-text content parts, tool types other than functions, and the Responses API's `conversation`.

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

For Qwen3.5-family templates (they write calls as `<tool_call><function=NAME><parameter=K>…` XML), requests may
carry `tools`: chat's `{"type":"function","function":{name, description, parameters}}`, rendered exactly as the
client sent them, or the Responses API's flat `{"type":"function", name, description, parameters}`.

- **Calls** in the reply become `tool_calls` (chat, `finish_reason: "tool_calls"`) or `function_call` output
  items (Responses), each with a `call_…` id and its arguments as a JSON object string. A parameter is read as
  the type its schema gives it — integer, number, boolean, object or array — and a value that does not read as
  that type stays a string for the client to report. Text the model writes after its last call is dropped, as
  the template asks; a call that never closed (the token limit cut it off) is returned as content, never as a
  call. In a stream, each call is sent whole once its `</tool_call>` arrives.
- **History**: assistant messages with `tool_calls` and `tool` messages (chat), or `function_call` and
  `function_call_output` items (Responses), are rendered as the template renders them: calls in the assistant
  turn, results as `<tool_response>` blocks in a user turn.
- **`tool_choice`**: `auto` (default); `none` leaves the tools out of the prompt; `required` writes
  `<tool_call>` and `<function=` for the model, and a named function writes the whole opening of its call, so the
  model can only continue a call — after its thinking block when it thinks. A function the request does not
  offer is a 400.
- **`parallel_tool_calls: false`** keeps the first call, and stops generating once it closes.
- **`previous_response_id`**: responses are kept in memory (the newest 512, unless `"store": false`), so a
  request can continue one without resending it; the earlier response's `instructions` are not carried over, as
  in OpenAI's API. An unknown id is a 404 `previous_response_not_found`.

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

```bash
dotnet bin/Release/amql-server.dll --model qwen=<qwen3.5-0.8b container> --think \
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
