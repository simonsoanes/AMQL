using Amql.Inference;
using Amql.Inference.Tracing;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>A sampled generation step: the produced token plus (optionally)
/// the top-k candidate window with probabilities.</summary>
public sealed record StepOutcome(int Token, int Position, IReadOnlyList<Candidate>? Candidates,
    IReadOnlyList<LayerTraceLine>? Trace);

public sealed record LayerTraceLine(int Layer, float ResidualNorm, float DeltaNorm);

public sealed record Candidate(int Token, float Logit, float Probability);

public sealed record TensorTraceLine(string ObjectId, string TensorName, long[] Shape, bool CacheHit);

/// <summary>
/// Drives autoregressive generation against a VINDEX3 container: plan the
/// component, prefill the context, then sample step by step with greedy or
/// temperature/top-k/top-p decoding. The session RNG is created once per
/// run — repeated samples are draws, not replays.
/// </summary>
public static class InferenceRunner
{
    public sealed record GenerateOptions(
        bool Trace,
        bool TraceTensors,
        WeightWorkingSet? WeightWorkingSet = null,
        string? TraceJsonPath = null);

    /// <summary>How many top candidates a traced step records.</summary>
    private const int TraceTopK = 10;

    public static (int[] Prefill, List<StepOutcome> Steps) Generate(
        Vindex3Container container, string componentId, int[] tokens,
        int steps, SamplingConfig config, int? showTopK = null,
        WeightPatch? patch = null,
        GenerateOptions? options = null)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);

        var vocab = plan.Embedding?.VocabSize ?? plan.Output?.VocabSize ?? 0;
        foreach (var token in tokens)
        {
            if (token < 0 || token >= vocab)
            {
                throw new CliException(
                    $"token {token} is outside the vocabulary [0, {vocab})");
            }
        }

        var session = new DecodeSession(plan, store, patch, options?.WeightWorkingSet);
        var rng = new Random(config.Seed);
        bool tracing = options is { Trace: true } or { TraceTensors: true };
        bool tensorTrace = options is { TraceTensors: true };

        // Wire tensor-load trace for --trace-tensors.
        var tensorLoads = tensorTrace ? new List<TensorTraceLine>() : null;
        if (tensorLoads is not null)
        {
            session.Runtime.Weights.LoadTrace = (objId, tensor, shape, hit) =>
            {
                lock (tensorLoads) { tensorLoads.Add(new TensorTraceLine(objId, tensor, shape, hit)); }
            };
        }

        session.Prefill(tokens);

        // The operator trace is attached only after prefill: the recorder's
        // step buffer is what gives an observation its context, and prefill
        // runs before any step exists.
        var recorder = options?.TraceJsonPath is not null ? new TraceRecorder() : null;
        if (recorder is not null)
        {
            session.Runtime.OpTrace = recorder.Observe;
            session.Runtime.ExpertRoutingTrace = (_, experts) => recorder.ObserveExperts(experts);
        }

        var outcomes = new List<StepOutcome>(steps);
        for (int step = 0; step < steps; step++)
        {
            var logits = session.LastLogits;
            int token = config.Temperature <= 0f
                ? Sampler.ArgMax(logits)
                : Sampler.Sample(logits, config, rng);

            recorder?.BeginStep(session.Position);

            // Enable trace for the forward pass that produces the next logits.
            if (tracing) { session.Runtime.BeginTrace(); }

            session.Step(token);

            List<LayerTraceLine>? trace = null;
            if (tracing)
            {
                session.Runtime.EndTrace();
                trace = session.Runtime.Trace
                    .Select(t => new LayerTraceLine(t.Layer, t.R, t.D))
                    .ToList();
            }

            if (recorder is not null)
            {
                // The distribution recorded is the one this forward produced,
                // not the one the input token was drawn from: a step reads
                // "fed this token, these operators ran, this came out".
                var produced = session.LastLogits;
                var topK = CandidatesForTrace(produced);
                recorder.EndStep(token, topK, SoftmaxEntropy(produced),
                    topK.Count >= 2 ? topK[0].Probability - topK[1].Probability : topK[0].Probability);
            }

            outcomes.Add(new StepOutcome(
                token,
                session.Position,
                CandidatesFor(logits, showTopK),
                trace));

            // Dump tensor loads after the first step.
            if (tensorLoads is not null && step == 0)
            {
                // Collected during prefill + first step — report once.
            }
        }

        if (recorder is not null && options?.TraceJsonPath is { } tracePath)
        {
            session.Runtime.OpTrace = null;
            session.Runtime.ExpertRoutingTrace = null;
            TraceRecorder.WriteJson(
                recorder.ToRunTrace(
                    container.Index.Model, componentId, plan.HiddenSize, plan.Layers.Count, tokens,
                    $"temperature={config.Temperature} top_k={config.TopK} top_p={config.TopP} seed={config.Seed}",
                    (options?.WeightWorkingSet ?? WeightWorkingSetExtensions.FromEnv()).ToString()!),
                tracePath);
        }

        return (tokens, outcomes);
    }

    private static IReadOnlyList<TokenCandidate> CandidatesForTrace(Tensor2D logits)
        => (CandidatesFor(logits, TraceTopK) ?? Array.Empty<Candidate>())
            .Select(c => new TokenCandidate(c.Token, c.Logit, c.Probability))
            .ToArray();

    /// <summary>Shannon entropy (nats) of the next-token distribution. A low
    /// entropy step is one the model was certain about, which is where an
    /// edit to a weight is least likely to show up; a high one is where the
    /// map is worth reading.</summary>
    private static float SoftmaxEntropy(Tensor2D logits)
    {
        var row = logits.FirstRow();
        float max = float.NegativeInfinity;
        for (int i = 0; i < row.Length; i++)
        {
            if (row[i] > max)
            {
                max = row[i];
            }
        }
        double sum = 0.0, entropy = 0.0;
        for (int i = 0; i < row.Length; i++)
        {
            double e = Math.Exp(row[i] - max);
            sum += e;
            entropy += e * (row[i] - max);
        }
        if (sum <= 0.0)
        {
            return 0f;
        }
        // H = log(sum) - (1/sum) * Σ e_i * (x_i - max)
        return (float)(Math.Log(sum) - entropy / sum);
    }

    private static IReadOnlyList<Candidate>? CandidatesFor(Tensor2D logits, int? showTopK)
    {
        if (showTopK is not { } k || k <= 0)
        {
            return null;
        }
        var row = logits.FirstRow().ToArray();

        // Softmax over the full row for probabilities.
        float max = float.NegativeInfinity;
        for (int i = 0; i < row.Length; i++)
        {
            if (row[i] > max)
            {
                max = row[i];
            }
        }
        var probs = new float[row.Length];
        float sum = 0f;
        for (int i = 0; i < row.Length; i++)
        {
            probs[i] = MathF.Exp(row[i] - max);
            sum += probs[i];
        }
        float inv = 1f / sum;
        for (int i = 0; i < row.Length; i++)
        {
            probs[i] *= inv;
        }

        var order = Enumerable.Range(0, row.Length).ToArray();
        Array.Sort(order, (a, b) => row[b].CompareTo(row[a]));
        return order.Take(k).Select(i => new Candidate(i, row[i], probs[i])).ToArray();
    }
}

/// <summary>Inspects a specific token in vocabulary space: where it sits in
/// the embedding table, its numeric profile, and its nearest neighbours by
/// cosine similarity. When a context is given AND the component plans, the
/// model's own logit/rank for the token at the last position is reported
/// too.</summary>
public static class TokenInspector
{
    public const int DefaultNeighbors = 5;

    public sealed record EmbeddingProfile(
        int Token,
        int Vocab,
        int Dim,
        string StoredDtype,
        float[] Row,
        float Min,
        float Max,
        float Mean,
        double Norm,
        IReadOnlyList<Neighbor> Neighbors);

    public sealed record Neighbor(int Token, double Cosine);

    public sealed record LogitReport(int Token, int Rank, double Logit, double Probability,
        IReadOnlyList<Candidate> Top);

    public static EmbeddingProfile InspectEmbedding(Vindex3Container container, string componentId, int token, int neighborCount, WeightPatch? patch = null)
    {
        var graph = container.Graph ??
            throw new CliException("container records no system graph — cannot locate the embedding object");
        var embedding = graph.Objects.FirstOrDefault(o =>
            o.Component == componentId && o.Kind == ObjectKind.Embedding)
            ?? throw new CliException(
                $"component '{componentId}' owns no embedding object — nothing to inspect");

        using var store = container.CreateOperandStore();
        var resolution = store.ResolveWidened(embedding.Id, "weight");
        if (resolution.Shape.Length != 2)
        {
            throw new CliException(
                $"embedding '{embedding.Id}' resolves to shape [{string.Join("x", resolution.Shape)}], expected [vocab, dim]");
        }
        int vocab = checked((int)resolution.Shape[0]);
        int dim = checked((int)resolution.Shape[1]);
        if (token < 0 || token >= vocab)
        {
            throw new CliException($"token {token} is outside the vocabulary [0, {vocab})");
        }

        var table = resolution.Values;
        ApplyEmbeddingPatch(table, embedding.Id, patch);
        var row = new float[dim];
        Array.Copy(table, token * dim, row, 0, dim);

        float min = row.Min();
        float max = row.Max();
        float mean = (float)row.Average();
        double norm = Math.Sqrt(TensorOps.Dot(row, row));

        var neighbors = Neighbours(table, vocab, dim, token, neighborCount);

        return new EmbeddingProfile(token, vocab, dim, resolution.Dtype.Label(), row, min, max, mean, norm, neighbors);
    }

    /// <summary>Logit-space verdict for the same token at the end of
    /// <c>context</c>: its rank and probability among the vocabulary.</summary>
    public static LogitReport InspectLogits(Vindex3Container container, string componentId, int token, int[] context, int topK, WeightPatch? patch = null)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);
        var session = new DecodeSession(plan, store, patch);
        var logits = session.Prefill(context);
        var row = logits.FirstRow().ToArray();

        var order = Enumerable.Range(0, row.Length).ToArray();
        Array.Sort(order, (a, b) => row[b].CompareTo(row[a]));
        int rank = Array.IndexOf(order, token);
        var top = order.Take(Math.Max(1, topK)).Select(i => new Candidate(i, row[i], 0f)).ToArray();

        float max = float.NegativeInfinity;
        for (int i = 0; i < row.Length; i++)
        {
            if (row[i] > max)
            {
                max = row[i];
            }
        }
        double sum = 0;
        for (int i = 0; i < row.Length; i++)
        {
            sum += Math.Exp(row[i] - max);
        }
        double probability = Math.Exp(row[token] - max) / sum;
        return new LogitReport(token, rank, row[token], probability, top);
    }

    /// <summary>Merges a patched embedding delta into the widened
    /// embedding table — the inspection path reads the table directly
    /// (neighbours + profile) instead of through the weight loader.</summary>
    private static void ApplyEmbeddingPatch(float[] table, string embeddingId, WeightPatch? patch)
    {
        if (patch is null || !patch.TryGet(embeddingId, "weight", out var entry))
        {
            return;
        }
        if (entry.Delta.Length != table.Length)
        {
            throw new CliException(
                $"patch entry '{entry.Key}' holds {entry.Delta.Length} deltas but the embedding " +
                $"'{embeddingId}' has {table.Length} elements");
        }
        for (int i = 0; i < table.Length; i++)
        {
            table[i] += entry.Delta[i];
        }
    }

    private static List<Neighbor> Neighbours(float[] table, int vocab, int dim, int token, int count)
    {
        var target = table.AsSpan(token * dim, dim);
        double targetNorm = Math.Sqrt(TensorOps.Dot(target, target));

        var scored = new List<(int Token, double Cosine)>();
        for (int t = 0; t < vocab; t++)
        {
            if (t == token)
            {
                continue;
            }
            var other = table.AsSpan(t * dim, dim);
            double otherNorm = Math.Sqrt(TensorOps.Dot(other, other));
            if (otherNorm == 0 || targetNorm == 0)
            {
                continue;
            }
            scored.Add((t, TensorOps.Dot(target, other) / (targetNorm * otherNorm)));
        }

        return scored
            .OrderByDescending(p => p.Cosine)
            .Take(Math.Min(count, scored.Count))
            .Select(p => new Neighbor(p.Token, p.Cosine))
            .ToList();
    }
}