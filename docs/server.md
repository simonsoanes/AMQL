# amql-server — HTTP endpoints for AMQL containers

`amql-server` loads one or more VINDEX3 containers and serves each on the endpoints its recorded facts say it
can answer. Nothing is guessed from a model's name: a container that cannot serve an endpoint is not offered
on it, and the startup log says why.

```bash
dotnet build -c Release src/Amql.Server
./bin/Release/amql-server --model von=<von-container> --model chat=<qwen-container> --model emb=<embedding-container> \
                          --urls http://127.0.0.1:8000 [--api-key KEY] [--max-tokens 1024] [--no-warm]
```

| Container | Endpoints |
|---|---|
| Von decision model (ModernBERT + option-marker head) | `POST /v1/decisions`, `POST /v1/systemone` (OpenJEV's path), `POST /api/v1/decisions` (jevai.org's `{ code, message, data }` envelope) |
| Embedding container (`convert-to-embedding`, an `embedding` surface) | `POST /v1/embeddings` |
| Generative decoder with a chat template this build renders | `POST /v1/chat/completions`, `POST /v1/responses` — streaming (SSE) or not |
| Every model | `GET /v1/models` (with a `capabilities` list), `GET /health` |

A sequence classifier or a bare encoder loads but is offered no endpoint (the log notes `amql-cli classify`).

## Routing, concurrency, auth

- A request's `model` names a loaded model by id (`--model id=<dir>`, default: the directory name). An unknown
  id is a 404 `model_not_found`; a model that does not serve the endpoint is a 400 `model_not_supported`.
  Omitting `model` is allowed when exactly one loaded model serves the endpoint. For decisions, `model` names
  TypeSafe's hosted Jev (`openjev`, `typesafe-ai/jev`), so any unmatched name falls back to the sole Von model;
  the response reports the version actually served.
- The runtimes keep per-sequence state, so each model serves one request at a time; requests to different
  models run in parallel. Generation runs off the request thread, streams through a channel, and stops at the
  next token when the client disconnects.
- `--api-key` (or `AMQL_API_KEY`) requires `Authorization: Bearer <key>` on `/v1/*` and `/api/*`; `/health` is
  open. Without a key the server logs that it is unauthenticated — bind it to loopback.
- Weights are warmed at startup (one short forward pass per model) so the first request does not pay for
  widening them; `--no-warm` skips that.

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
knows — Qwen3/Qwen3.5 ChatML (trimmed contents, `<think>` handling, the empty think block unless
`chat_template_kwargs.enable_thinking` is true), plain ChatML and Llama 3 — and tokenised without extra special
tokens, as `apply_chat_template` does. Any other template is refused at load rather than approximated.

- **Content**: strings or arrays of text parts (`text`, `input_text`, `output_text`); `developer` is a system
  message; the Responses API's `instructions` is a system message placed first.
- **Sampling**: `temperature` (0 = greedy; default 1), `top_p`, `seed`, `stop` (up to 4 strings, never streamed
  past), `max_tokens` / `max_completion_tokens` / `max_output_tokens` (default `--max-tokens`, bounded by the
  context). Generation stops at the tokens named by `generation_config.json`'s `eos_token_id`, the tokenizer's
  `eos_token`, and the template's end-of-turn marker.
- **Text** is decoded from the accumulated bytes at every step, so a character split across tokens is sent once,
  whole.
- **Finish**: `stop` or `length` for chat; Responses reports `completed`, or `incomplete` with
  `incomplete_details.reason: "max_output_tokens"`.
- **Streaming**: chat sends `chat.completion.chunk` deltas (role first, `finish_reason` last, a `usage` chunk
  with `stream_options.include_usage`) and `data: [DONE]`. Responses sends `response.created`,
  `response.in_progress`, `response.output_item.added`, `response.content_part.added`,
  `response.output_text.delta`…, `response.output_text.done`, `response.content_part.done`,
  `response.output_item.done`, then `response.completed` or `response.incomplete`, each with a
  `sequence_number`.
- **Refused** (400, OpenAI error shape), rather than silently ignored: tools and function calling, `n > 1`,
  logprobs, audio, image or other non-text content parts, structured output (`response_format` /
  `text.format` other than text), and the Responses API's `previous_response_id` / `conversation` (the server
  keeps no state).

## Verification

`tests/Amql.Tests/ServerTests.cs`, against goldens from transformers and the Von SDK:

- chat rendering equals HF `apply_chat_template` with Qwen3.5's real template (6 conversations, thinking on and
  off, a history turn with a think block, non-ASCII text);
- greedy generation equals transformers' `generate` token for token on a GatedDeltaNet + attention model;
- embeddings equal transformers' mean-pooled, normalised hidden states (1e-5);
- decisions return the SDK's answers, on `/v1` and in the jevai envelope;
- streamed and non-streamed chat produce the same text; the Responses event sequence; base64 embeddings;
  the refusals; the API key.
