# Statistical Outputs — Logprobs, Entropy & Perplexity

**Status:** design · **Date:** 2026-10-03

## 1. Motivation

A user running inference wants to know how uncertain the model is about its
output — per-token log-probabilities, the entropy of the next-token distribution,
and aggregate sequence-level metrics. This is standard in the OpenAI API
(`logprobs`, `top_logprobs`) and essential for:

- **Calibration checks** — is the model's stated confidence well-calibrated?
- **Hallucination detection** — high-entropy steps often precede fabrications.
- **Contrastive evaluation** — running the same prompt against two models and
  comparing where each was uncertain.
- **Prompt engineering** — identifying which tokens in a prompt drive the model
  into low-confidence territory.

AMQL already computes the full softmax distribution internally (the Sampler picks
from it), already computes top-K candidates for display (`--logits K`), and
already computes Shannon entropy for trace recording — but none of it is
surfaced to the user as a first-class output.

## 2. What already exists

| Capability | Where | Surfaced? |
|---|---|---|
| Softmax over full vocab | `Sampler.ApplyTemperatureAndSample` | No — discarded after token selection |
| Top-K candidates (logit + prob) | `InferenceCommands.CandidatesFor` | CLI only (`--logits K`) |
| Shannon entropy (nats) | `InferenceCommands.SoftmaxEntropy` | Trace only (`--trace-json`) |
| Top-1 margin | `InferenceCommands` (inline) | Trace only |
| Negative log-likelihood | — | Not computed |
| Perplexity | `Amql.Merge/MoeIfy.cs` | MoE quality gate only |
| OpenAI `logprobs` parameter | `ServerApp.RefuseUnsupported` | Explicitly refused |
| `logprobs` in response | `ServerApp` | Hardcoded `null` |

The softmax is recomputed independently in four places (`Sampler.Sample`,
`CandidatesFor`, `SoftmaxEntropy`, `CausalTracer.SoftmaxProb`). There is no
single point of computation — every consumer re-derives it from raw logits.

## 3. Data model

### 3.1 `LogprobsResult` — per-token statistics

A new type in `Amql.Inference` that carries everything derivable from one
logits row in a single pass:

```csharp
/// <summary>Per-token statistical output from one logits row: log-probabilities,
/// entropy, and the top-K candidates with their tokens and probabilities.</summary>
public sealed class LogprobsResult
{
    /// <summary>The selected token's log-probability (natural log).</summary>
    public float TokenLogprob { get; init; }

    /// <summary>Shannon entropy (nats) of the full next-token distribution.</summary>
    public float Entropy { get; init; }

    /// <summary>The probability of the top-1 token minus the probability of the
    /// top-2 token — zero when the top two are equally likely, near 1.0 when
    /// the model is certain.</summary>
    public float Top1Margin { get; init; }

    /// <summary>Top-K candidates ordered by descending probability. The list
    /// always includes at least the selected token.</summary>
    public IReadOnlyList<TokenLogprob> TopLogprobs { get; init; }

    /// <summary>Raw vocab-size distribution — null unless the caller asked
    /// for it (large vocab × many steps = impractically large).</summary>
    public float[]? FullLogprobs { get; init; }
}

/// <summary>One token in the top-K: its id, log-probability, and softmax
/// probability, plus an optional text decoration.</summary>
public sealed record TokenLogprob(int Token, float Logprob, float Probability,
    string? Text = null);
```

### 3.2 Extended `StepOutcome`

The CLI's `StepOutcome` gains an optional `LogprobsResult` field:

```csharp
public sealed record StepOutcome(int Token, int Position,
    IReadOnlyList<Candidate>? Candidates,
    IReadOnlyList<LayerTraceLine>? Trace,
    LogprobsResult? Logprobs);   // NEW
```

### 3.3 Extended `GenerationResult`

The server's `GenerationResult` gains per-token logprobs plus aggregate stats:

```csharp
public sealed record GenerationResult(string Text, string FinishReason,
    int PromptTokens, int CompletionTokens)
{
    public IReadOnlyList<int> Tokens { get; init; } = Array.Empty<int>();
    public IReadOnlyList<LogprobsResult>? Logprobs { get; init; }  // NEW
    public float? Perplexity { get; init; }                          // NEW
    public float? MeanEntropy { get; init; }                         // NEW
}
```

Perplexity is `exp(average NLL)` over the completion tokens. Mean entropy is the
arithmetic mean of per-token entropies.

### 3.4 Why not store the full distribution by default

A full 151,936-entry float array (Qwen3.5) × 1,024 steps = ~600 MiB of logprobs
— impractical as a default. The top-K list (typically 5–20 entries) is enough
for calibration and uncertainty display. The full distribution (`FullLogprobs`)
is opt-in and only sensible for short generations or single-step inspection.

## 4. Where computation lives

### 4.1 Centralised softmax in `Sampler`

The refactored `Sampler` computes the softmax **once** and returns both the
selected token and the logprobs result. A new method:

```csharp
public static (int Token, LogprobsResult Logprobs) SampleWithLogprobs(
    ReadOnlySpan<float> logits, SamplingConfig config, Random rng, int topK);
```

This replaces the current `Sample` call in the generation loop. The existing
`Sample` / `ArgMax` remain for callers that don't want logprobs (zero overhead
for the non-logprobs path — no LogprobsResult allocation).

The softmax path inside `ApplyTemperatureAndSample` is refactored to produce
a `LogprobsResult` in one go:
1. Temperature-scale → exp → normalize → (top-k/top-p filter) → probabilities
2. Compute entropy: `H = -Σ p_i · ln(p_i)` over the filtered distribution
3. Compute top-1 margin: `p[0] - p[1]`
4. Select top-K by probability, with log-probability
5. Sample token, record its logprob
6. Return `(token, result)`

### 4.2 `CandidatesFor` and `SoftmaxEntropy` become consumers

After the refactor, `CandidatesFor` is replaced by reading `LogprobsResult.TopLogprobs`,
and `SoftmaxEntropy` is replaced by reading `LogprobsResult.Entropy`. The
duplicate softmax sites in `InferenceCommands.cs` are removed.

### 4.3 `CausalTracer.SoftmaxProb` is unchanged

It computes the probability of a single target token and runs inside a tight
replay loop — the overhead of a full `LogprobsResult` would dominate there.
It remains a standalone method.

## 5. API surface

### 5.1 CLI — `generate`

```
amql-cli generate <container> --prompt "..." --tokenizer <dir> \
    --logprobs [K]         # enable logprobs, optionally with top-K (default 5)
    --full-logprobs        # also return the full vocab distribution (verbose mode)
    --uncertainty          # print per-step entropy + perplexity summary
```

Output when `--logprobs` is active — each step line gains a statistical suffix:

```
   token 42 → "Paris"  (logprob -0.32, entropy 1.45, margin 0.87)
```

When `--uncertainty` is active, a summary line is printed at the end:

```
  perplexity 3.21  mean entropy 1.45 nats  mean margin 0.82
```

### 5.2 CLI — `inspect-token`

Already has `--logits K` which shows top-K logits. Gains `--logprobs` to show
log-probabilities and entropy for a single forward pass — useful for inspecting
how certain the model is about a specific continuation.

### 5.3 HTTP API — `/v1/chat/completions`

Accepts the OpenAI `logprobs` parameter:

```json
{
  "model": "qwen3.5",
  "messages": [...],
  "logprobs": true,
  "top_logprobs": 5
}
```

Response gains `logprobs` in each choice:

```json
{
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "..." },
    "logprobs": {
      "content": [
        {
          "token": "Paris",
          "logprob": -0.32,
          "bytes": [80, 97, 114, 105, 115],
          "top_logprobs": [
            { "token": "Paris", "logprob": -0.32, "bytes": [...] },
            { "token": "France", "logprob": -1.85, "bytes": [...] },
            ...
          ]
        },
        ...
      ]
    },
    "finish_reason": "stop"
  }],
  "usage": { ... }
}
```

Key decisions:
- `bytes` is the UTF-8 encoding of the token text, matching OpenAI's contract.
- `top_logprobs` count is capped at 20 (OpenAI's default max).
- When `logprobs` is false/absent, `"logprobs": null` as before — zero overhead.
- Streaming: each `chat.completion.chunk` carries the logprobs for its delta
  token, same as non-streaming but per-chunk.

### 5.4 HTTP API — `/v1/responses`

The Responses API does not have a `logprobs` field in OpenAI's spec, so we do
not add one. Users who want logprobs use `/v1/chat/completions`.

### 5.5 Non-goal: prompt logprobs

OpenAI's `echo` parameter returns logprobs for the prompt tokens too. This is
out of scope for this design — prompt logprobs require storing the prefill
logits for every position, which is a separate storage concern. The focus is
on completion token logprobs.

## 6. Phasing

### Phase 1 — Centralised logprobs computation (Amql.Inference)
- New `LogprobsResult` and `TokenLogprob` types in `Amql.Inference`.
- `Sampler.SampleWithLogprobs()` — one-pass softmax → logprobs + token.
- Remove duplicate softmax from `CandidatesFor` / `SoftmaxEntropy` in the CLI.

### Phase 2 — CLI surface (Amql.Cli)
- `StepOutcome` gains `LogprobsResult? Logprobs`.
- `--logprobs [K]` flag on `generate` — prints per-step stats.
- `--uncertainty` flag — prints perplexity + mean entropy summary.
- `InferenceRunner.Generate` wires logprobs through when requested.

### Phase 3 — HTTP API (Amql.Server)
- `GenerationResult` gains `Logprobs` and aggregate fields.
- `TextGenerator.Generate` returns logprobs when requested.
- Server accepts `logprobs` and `top_logprobs` in `/v1/chat/completions`.
- OpenAI-compatible response format with `bytes`, `logprob`, `top_logprobs`.

## 7. Risks & limitations

- **Large vocabularies**: A full-distribution logprobs request (`--full-logprobs`)
  on a 151k-vocab model generates ~600 KB per step. The feature is opt-in and
  documented as memory-heavy.
- **Temperature interaction**: When temperature ≠ 1.0, the "logprobs" returned
  are the log-probabilities of the temperature-scaled distribution, not the
  raw model logits. This matches OpenAI's behavior.
- **Top-k/top-p**: Filtered-out tokens have logprob = -∞. The entropy is
  computed over the filtered distribution (un-normalized remainder), matching
  what the sampler actually considered.
- **Greedy decoding (temperature ≤ 0)**: Logprobs are still well-defined —
  the argmax token has logprob 0.0 (ln(1.0)) and entropy is 0. The top-K
  list still shows the softmax over the unscaled logits for inspection.