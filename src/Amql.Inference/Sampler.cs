namespace Amql.Inference;

/// <summary>Sampling configuration. Temperature 1.0 with no top-k/top-p
/// is the plain multinomial over the logits; top-k and top-p compose
/// (top-k first, then top-p over the survivors).</summary>
public sealed record SamplingConfig(
    int Seed = 42,
    float Temperature = 1.0f,
    int TopK = 0,
    float TopP = 0f);

/// <summary>Token selection from model logits: greedy arg-max or
/// temperature-scaled multinomial with optional top-k / top-p filtering.
/// Mirrors the reference's sampling step contract; beam/search strategies
/// are out of scope for this build.</summary>
public static class Sampler
{
    public static int ArgMax(Tensor2D logits)
    {
        var row = logits.FirstRow();
        int best = 0;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > row[best])
            {
                best = i;
            }
        }
        return best;
    }

    public static int Sample(Tensor2D logits, SamplingConfig config)
    {
        var row = logits.FirstRow();
        return ApplyTemperatureAndSample(row, config, new Random(config.Seed));
    }

    public static int Sample(ReadOnlySpan<float> logits, SamplingConfig config) =>
        ApplyTemperatureAndSample(logits, config, new Random(config.Seed));

    /// <summary>
    /// Sample with an explicit session RNG. Autoregressive loops MUST pass
    /// one Random created once per session: the convenience overloads above
    /// reseed from <c>config.Seed</c> on every call, which repeats the same
    /// draw sequence across steps otherwise.
    /// </summary>
    public static int Sample(ReadOnlySpan<float> logits, SamplingConfig config, Random rng) =>
        ApplyTemperatureAndSample(logits, config, rng);

    public static int Sample(Tensor2D logits, SamplingConfig config, Random rng) =>
        ApplyTemperatureAndSample(logits.FirstRow(), config, rng);

    private static int ApplyTemperatureAndSample(ReadOnlySpan<float> logits, SamplingConfig config, Random rng)
    {
        // Greedy for temperature ≤ 0.
        if (config.Temperature <= 0f)
        {
            return ArgMaxOf(logits);
        }

        var scaled = new float[logits.Length];
        float invTemp = 1f / config.Temperature;
        for (int i = 0; i < logits.Length; i++)
        {
            scaled[i] = logits[i] * invTemp;
        }

        // Stable softmax.
        float max = float.NegativeInfinity;
        foreach (var v in scaled)
        {
            if (v > max)
            {
                max = v;
            }
        }
        var probs = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(scaled[i] - max);
            sum += probs[i];
        }
        if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
        {
            return ArgMaxOf(logits);
        }
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= sum;
        }

        // Top-k filter.
        if (config.TopK > 0 && config.TopK < probs.Length)
        {
            var order = Enumerable.Range(0, probs.Length)
                .OrderByDescending(i => probs[i])
                .Take(config.TopK)
                .ToArray();
            float cutoff = probs[order[^1]];
            for (int i = 0; i < probs.Length; i++)
            {
                if (probs[i] < cutoff)
                {
                    probs[i] = 0f;
                }
            }
        }

        // Top-p filter (cumulative, descending).
        if (config.TopP is > 0f and < 1f)
        {
            var order = Enumerable.Range(0, probs.Length)
                .OrderByDescending(i => probs[i])
                .ToArray();
            float cumulative = 0f;
            var keep = new HashSet<int>();
            foreach (var i in order)
            {
                cumulative += probs[i];
                keep.Add(i);
                if (cumulative >= config.TopP)
                {
                    break;
                }
            }
            for (int i = 0; i < probs.Length; i++)
            {
                if (!keep.Contains(i))
                {
                    probs[i] = 0f;
                }
            }
        }

        double total = 0;
        foreach (var p in probs)
        {
            total += p;
        }
        if (total <= 0)
        {
            return ArgMaxOf(logits);
        }

        double draw = rng.NextDouble() * total;
        double acc = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            acc += probs[i];
            if (draw <= acc)
            {
                return i;
            }
        }
        return probs.Length - 1;
    }

    private static int ArgMaxOf(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }
        return best;
    }
}