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
    public static int ArgMax(Tensor2D logits) => ArgMax(logits.FirstRow());

    /// <summary>The index of the largest value — the first one when several
    /// tie — skipping NaNs unless the first value is one, exactly as a scalar
    /// <c>values[i] &gt; values[best]</c> scan. Vectorised: over a 248k-token
    /// vocabulary the scalar scan costs a noticeable share of a GPU decode
    /// step.</summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        int n = values.Length;
        int width = System.Numerics.Vector<float>.Count;
        if (n < 4 * width || float.IsNaN(values[0]))
        {
            return ScalarArgMax(values);
        }
        var negInf = new System.Numerics.Vector<float>(float.NegativeInfinity);
        var maxV = negInf;
        int i = 0;
        for (; i <= n - width; i += width)
        {
            var v = new System.Numerics.Vector<float>(values.Slice(i, width));
            // NaN never wins the scalar scan, so it never enters the max.
            v = System.Numerics.Vector.ConditionalSelect(System.Numerics.Vector.Equals(v, v), v, negInf);
            maxV = System.Numerics.Vector.Max(maxV, v);
        }
        float max = float.NegativeInfinity;
        for (int j = 0; j < width; j++)
        {
            max = Math.Max(max, maxV[j]);
        }
        for (; i < n; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
            }
        }
        var target = new System.Numerics.Vector<float>(max);
        for (i = 0; i <= n - width; i += width)
        {
            var eq = System.Numerics.Vector.Equals(new System.Numerics.Vector<float>(values.Slice(i, width)), target);
            if (!eq.Equals(System.Numerics.Vector<int>.Zero))
            {
                for (int j = i; j < i + width; j++)
                {
                    if (values[j] == max)
                    {
                        return j;
                    }
                }
            }
        }
        for (; i < n; i++)
        {
            if (values[i] == max)
            {
                return i;
            }
        }
        return ScalarArgMax(values);
    }

    private static int ScalarArgMax(ReadOnlySpan<float> values)
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

    /// <summary>
    /// Sample a token AND return per-token statistics in a single pass over
    /// the logits row. When <paramref name="logprobsTopK"/> is non-null the
    /// softmax is computed once and the result carries the selected token's
    /// log-probability, the distribution entropy, the top-1 margin, and the
    /// top-K candidates with their log-probabilities. When it is null this is
    /// equivalent to <see cref="Sample(ReadOnlySpan{float},SamplingConfig,Random)"/>.
    /// </summary>
    public static (int Token, LogprobsResult? Logprobs) SampleWithLogprobs(
        ReadOnlySpan<float> logits, SamplingConfig config, Random rng,
        int? logprobsTopK, bool fullLogprobs = false)
    {
        if (logprobsTopK is not { } topK || topK <= 0)
        {
            return (ApplyTemperatureAndSample(logits, config, rng), null);
        }

        return SampleWithStats(logits, config, rng, topK, fullLogprobs);
    }

    public static (int Token, LogprobsResult? Logprobs) SampleWithLogprobs(
        Tensor2D logits, SamplingConfig config, Random rng,
        int? logprobsTopK, bool fullLogprobs = false) =>
        SampleWithLogprobs(logits.FirstRow(), config, rng, logprobsTopK, fullLogprobs);

    private static int ApplyTemperatureAndSample(ReadOnlySpan<float> logits, SamplingConfig config, Random rng)
    {
        // Greedy for temperature ≤ 0.
        if (config.Temperature <= 0f)
        {
            return ArgMaxOf(logits);
        }

        var probs = SoftmaxAndFilter(logits, config, out _);
        if (probs is null)
        {
            return ArgMaxOf(logits);
        }

        return MultinomialDraw(probs, rng);
    }

    private static (int Token, LogprobsResult Logprobs) SampleWithStats(
        ReadOnlySpan<float> logits, SamplingConfig config, Random rng, int topK, bool fullLogprobs)
    {
        int vocab = logits.Length;

        // Greedy: the selected token is certain (logprob 0), but we still
        // compute a softmax over the raw logits so the caller can inspect the
        // top-K alternatives.
        if (config.Temperature <= 0f)
        {
            int token = ArgMaxOf(logits);
            var (gProbs, gSorted) = SoftmaxTopK(logits, 1f, topK);
            var greedyTopLogprobs = BuildTopLogprobs(gProbs, gSorted, topK, vocab);
            float greedyEntropy = EntropyOf(gProbs);
            float greedyMargin = gSorted.Length >= 2 ? gProbs[gSorted[0]] - gProbs[gSorted[1]] : gProbs[gSorted[0]];
            return (token, new LogprobsResult(0f, greedyEntropy, greedyMargin, greedyTopLogprobs));
        }

        var probs = SoftmaxAndFilter(logits, config, out int[]? sortedIndices, wantSorted: true);
        if (probs is null)
        {
            int fallback = ArgMaxOf(logits);
            return (fallback, new LogprobsResult(0f, 0f, 1f,
                new[] { new TokenLogprob(fallback, 0f, 1f) }));
        }

        // Entropy over the filtered (non-zero) distribution. Zero-probability
        // tokens contribute nothing to entropy (0 · ln 0 = 0).
        float entropy = EntropyOf(probs);

        // Build sorted indices if not already available from top-p filtering.
        if (sortedIndices is null)
        {
            sortedIndices = Enumerable.Range(0, vocab)
                .OrderByDescending(i => probs[i])
                .ToArray();
        }

        // Top-1 margin: difference between the two highest probabilities.
        float top1Margin = sortedIndices.Length >= 2
            ? probs[sortedIndices[0]] - probs[sortedIndices[1]]
            : probs[sortedIndices[0]];

        // Top-K with log-probabilities.
        var resultTopLogprobs = BuildTopLogprobs(probs, sortedIndices, topK, vocab);

        // Full logprobs if requested.
        float[]? full = null;
        if (fullLogprobs)
        {
            full = new float[vocab];
            for (int i = 0; i < vocab; i++)
            {
                full[i] = probs[i] > 0f ? MathF.Log(probs[i]) : float.NegativeInfinity;
            }
        }

        int sampled = MultinomialDraw(probs, rng);
        float tokenLogprob = probs[sampled] > 0f ? MathF.Log(probs[sampled]) : float.NegativeInfinity;

        return (sampled, new LogprobsResult(tokenLogprob, entropy, top1Margin, resultTopLogprobs, full));
    }

    /// <summary>Temperature-scale, softmax, and top-k/top-p filter.
    /// Returns the probability array (some entries zeroed by filtering) and,
    /// when top-p ran, the sorted indices it already computed.</summary>
    private static float[]? SoftmaxAndFilter(ReadOnlySpan<float> logits, SamplingConfig config,
        out int[]? sortedIndices, bool wantSorted = false)
    {
        sortedIndices = null;
        int n = logits.Length;

        float invTemp = 1f / config.Temperature;
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            float v = logits[i] * invTemp;
            if (v > max)
            {
                max = v;
            }
        }

        var probs = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            probs[i] = MathF.Exp(logits[i] * invTemp - max);
            sum += probs[i];
        }
        if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
        {
            return null;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < n; i++)
        {
            probs[i] *= invSum;
        }

        // Top-k filter.
        if (config.TopK > 0 && config.TopK < n)
        {
            // The k-th largest probability, from a min-heap of k: one pass over the
            // vocabulary instead of sorting it. Ties at the cutoff are all kept, as before.
            var heap = new PriorityQueue<float, float>(config.TopK + 1);
            for (int i = 0; i < n; i++)
            {
                if (heap.Count < config.TopK)
                {
                    heap.Enqueue(probs[i], probs[i]);
                }
                else if (probs[i] > heap.Peek())
                {
                    heap.EnqueueDequeue(probs[i], probs[i]);
                }
            }
            float cutoff = heap.Peek();
            for (int i = 0; i < n; i++)
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
            // Only non-zero entries can be kept, so only they are ranked: after a top-k
            // filter that is k entries rather than the whole vocabulary. The order is a
            // stable sort by descending probability (ties in index order).
            var candidates = new List<int>(config.TopK is > 0 and < 4096 ? config.TopK + 8 : 1024);
            for (int i = 0; i < n; i++)
            {
                if (probs[i] > 0f)
                {
                    candidates.Add(i);
                }
            }
            var ranked = candidates.ToArray();
            var keys = ranked.Select(i => -probs[i]).ToArray();
            Array.Sort(keys, ranked);   // not stable: restore index order among equal keys below
            for (int lo = 0; lo < ranked.Length;)
            {
                int hi = lo + 1;
                while (hi < ranked.Length && keys[hi] == keys[lo])
                {
                    hi++;
                }
                if (hi - lo > 1)
                {
                    Array.Sort(ranked, lo, hi - lo);
                }
                lo = hi;
            }
            // Keep the shortest prefix whose mass reaches top-p; zero the other candidates.
            float cumulative = 0f;
            int keep = ranked.Length;
            for (int r = 0; r < ranked.Length; r++)
            {
                cumulative += probs[ranked[r]];
                if (cumulative >= config.TopP)
                {
                    keep = r + 1;
                    break;
                }
            }
            for (int r = keep; r < ranked.Length; r++)
            {
                probs[ranked[r]] = 0f;
            }
            if (wantSorted)
            {
                // The whole vocabulary in that order, zeros after in index order, for logprobs.
                var sorted = new int[n];
                Array.Copy(ranked, sorted, ranked.Length);
                int at = ranked.Length;
                var isCandidate = new bool[n];
                foreach (int i in ranked)
                {
                    isCandidate[i] = true;
                }
                for (int i = 0; i < n; i++)
                {
                    if (!isCandidate[i])
                    {
                        sorted[at++] = i;
                    }
                }
                sortedIndices = sorted;
            }
        }

        return probs;
    }

    /// <summary>Softmax with unscaled (or pre-scaled) logits, returning
    /// probabilities and indices sorted by descending probability.</summary>
    private static (float[] Probs, int[] Sorted) SoftmaxTopK(
        ReadOnlySpan<float> logits, float temperature, int topK)
    {
        int n = logits.Length;
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            float v = temperature > 0f && temperature != 1f ? logits[i] / temperature : logits[i];
            if (v > max)
            {
                max = v;
            }
        }

        var probs = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            float v = temperature > 0f && temperature != 1f ? logits[i] / temperature : logits[i];
            probs[i] = MathF.Exp(v - max);
            sum += probs[i];
        }
        if (sum <= 0f)
        {
            // All-zero or degenerate: uniform fallback so the caller gets
            // something to display.
            for (int i = 0; i < n; i++)
            {
                probs[i] = 1f / n;
            }
            sum = 1f;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < n; i++)
        {
            probs[i] *= invSum;
        }

        int[] sorted = Enumerable.Range(0, n)
            .OrderByDescending(i => probs[i])
            .Take(Math.Min(topK, n))
            .ToArray();

        return (probs, sorted);
    }

    private static TokenLogprob[] BuildTopLogprobs(float[] probs, int[] sorted, int topK, int vocab)
    {
        int count = Math.Min(topK, sorted.Length);
        var result = new TokenLogprob[count];
        for (int j = 0; j < count; j++)
        {
            int idx = sorted[j];
            float p = probs[idx];
            result[j] = new TokenLogprob(idx, p > 0f ? MathF.Log(p) : float.NegativeInfinity, p);
        }
        return result;
    }

    private static float EntropyOf(float[] probs)
    {
        float entropy = 0f;
        foreach (float p in probs)
        {
            if (p > 0f)
            {
                entropy -= p * MathF.Log(p);
            }
        }
        return entropy;
    }

    private static int MultinomialDraw(float[] probs, Random rng)
    {
        double total = 0;
        foreach (float p in probs)
        {
            total += p;
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

    private static int ArgMaxOf(ReadOnlySpan<float> values) => ArgMax(values);
}