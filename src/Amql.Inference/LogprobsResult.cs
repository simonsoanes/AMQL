namespace Amql.Inference;

/// <summary>Per-token statistical output from one logits row: log-probabilities,
/// entropy, and the top-K candidates with their tokens and probabilities.
/// Produced in a single pass by <see cref="Sampler.SampleWithLogprobs"/>.</summary>
public sealed class LogprobsResult
{
    /// <summary>The selected token's log-probability (natural log).</summary>
    public float TokenLogprob { get; init; }

    /// <summary>Shannon entropy (nats) of the full next-token distribution
    /// after temperature scaling and top-k/top-p filtering.</summary>
    public float Entropy { get; init; }

    /// <summary>The probability of the top-1 token minus the probability of the
    /// top-2 token — zero when the top two are equally likely, near 1.0 when
    /// the model is certain.</summary>
    public float Top1Margin { get; init; }

    /// <summary>Top-K candidates ordered by descending probability. The list
    /// always includes at least the selected token.</summary>
    public IReadOnlyList<TokenLogprob> TopLogprobs { get; init; }

    /// <summary>Raw vocab-size log-probability distribution — null unless the
    /// caller asked for it (large vocab × many steps = impractically large).</summary>
    public float[]? FullLogprobs { get; init; }

    public LogprobsResult(float tokenLogprob, float entropy, float top1Margin,
        IReadOnlyList<TokenLogprob> topLogprobs, float[]? fullLogprobs = null)
    {
        TokenLogprob = tokenLogprob;
        Entropy = entropy;
        Top1Margin = top1Margin;
        TopLogprobs = topLogprobs;
        FullLogprobs = fullLogprobs;
    }
}

/// <summary>One token in the top-K: its id, log-probability, and softmax
/// probability, plus an optional text decoration.</summary>
public sealed record TokenLogprob(int Token, float Logprob, float Probability,
    string? Text = null);