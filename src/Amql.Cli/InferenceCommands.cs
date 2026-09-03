using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>A sampled generation step: the produced token plus (optionally)
/// the top-k candidate window with probabilities.</summary>
public sealed record StepOutcome(int Token, int Position, IReadOnlyList<Candidate>? Candidates);

public sealed record Candidate(int Token, float Logit, float Probability);

/// <summary>
/// Drives autoregressive generation against a VINDEX3 container: plan the
/// component, prefill the context, then sample step by step with greedy or
/// temperature/top-k/top-p decoding. The session RNG is created once per
/// run — repeated samples are draws, not replays.
/// </summary>
public static class InferenceRunner
{
    public static (int[] Prefill, List<StepOutcome> Steps) Generate(
        Vindex3Container container, string componentId, int[] tokens,
        int steps, SamplingConfig config, int? showTopK = null)
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

        var session = new DecodeSession(plan, store);
        var rng = new Random(config.Seed);

        session.Prefill(tokens);
        var outcomes = new List<StepOutcome>(steps);
        for (int step = 0; step < steps; step++)
        {
            var logits = session.LastLogits;
            int token = config.Temperature <= 0f
                ? Sampler.ArgMax(logits)
                : Sampler.Sample(logits, config, rng);
            outcomes.Add(new StepOutcome(
                token,
                session.Position,
                CandidatesFor(logits, showTopK)));
            session.Step(token);
        }
        return (tokens, outcomes);
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

    public static EmbeddingProfile InspectEmbedding(Vindex3Container container, string componentId, int token, int neighborCount)
    {
        var graph = container.Graph ??
            throw new CliException("container records no system graph — cannot locate the embedding object");
        var embedding = graph.Objects.FirstOrDefault(o =>
            o.Component == componentId && o.Kind == ObjectKind.Embedding)
            ?? throw new CliException(
                $"component '{componentId}' owns no embedding object — nothing to inspect");

        using var store = container.CreateOperandStore();
        var resolution = store.Resolve(embedding.Id, "weight");
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
        if (!resolution.Dtype.IsWidenableToF32())
        {
            throw new CliException(
                $"embedding '{embedding.Id}' dtype {resolution.Dtype.Label()} has no f32 widening path");
        }

        var table = BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
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
    public static LogitReport InspectLogits(Vindex3Container container, string componentId, int token, int[] context, int topK)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);
        var session = new DecodeSession(plan, store);
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