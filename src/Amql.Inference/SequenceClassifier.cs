using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>One classified input: the raw head outputs, what the problem
/// type turns them into, and which token the pooling read.</summary>
public sealed record ClassificationResult(
    float[] Logits,
    double[] Scores,
    int PooledIndex,
    int Tokens);

/// <summary>
/// Serves a sequence-classification container (a decoder backbone plus a
/// <c>ClassifierHead</c>): forward the input, pool the post-final-norm hidden
/// state by the recorded rule, project through the <c>score</c> head, and
/// map the logits by the recorded <c>problem_type</c> — softmax for
/// single-label, per-label sigmoid for multi-label, identity for regression.
/// Every one of those steps is a fact read from the container; none is
/// defaulted, and a rule this build has not judged refuses by name.
/// </summary>
public sealed class SequenceClassifier
{
    private readonly DecodeSession _session;
    private readonly Tensor2D _score;
    private readonly float[]? _bias;

    public ClassifierSurface Surface { get; }

    private SequenceClassifier(DecodeSession session, ClassifierSurface surface, Tensor2D score, float[]? bias)
    {
        _session = session;
        Surface = surface;
        _score = score;
        _bias = bias;
    }

    public static SequenceClassifier Load(Vindex3Container container, OperandStore store, WeightPatch? patch = null,
        string componentId = "target")
    {
        var component = container.Graph?.Component(componentId)
            ?? throw new ContainerException("container records no system graph");
        var surface = component.Execution?.Classifier
            ?? throw new UnsupportedOperatorException(
                $"component '{componentId}' has no classifier surface — it is not a sequence-classification container");
        var headObject = container.Graph.Objects.FirstOrDefault(o => o.Component == componentId && o.Kind == ObjectKind.ClassifierHead)
            ?? throw new ContainerException($"component '{componentId}' declares a classifier surface but owns no classifier_head object");
        ClassifierPooling.Validate(surface.Pooling);
        _ = ClassifierPooling.Activate(surface.ProblemType, new float[surface.NumLabels]);  // refuses an unjudged problem type now

        var plan = Planner.Plan(container, componentId, store);
        var session = new DecodeSession(plan, store, patch);

        float[] Widen(string name, long[] expected)
        {
            var r = store.Resolve(headObject.Id, name);
            if (!r.Shape.SequenceEqual(expected))
            {
                throw new ContainerException(
                    $"{headObject.Id}/{name} is [{string.Join("x", r.Shape)}], expected [{string.Join("x", expected)}]");
            }
            var values = BitPattern.WidenToF32(r.Dtype, r.Payload);
            if (patch is not null && patch.TryGet(headObject.Id, name, out var entry))
            {
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] += entry.Delta[i];
                }
            }
            return values;
        }

        int hidden = component.HiddenSize;
        var score = new Tensor2D(Widen("weight", new long[] { surface.NumLabels, hidden }), surface.NumLabels, hidden);
        float[]? bias = surface.ScoreBias ? Widen("bias", new long[] { surface.NumLabels }) : null;
        return new SequenceClassifier(session, surface, score, bias);
    }

    /// <summary>Classifies one tokenised input (no padding: the input is one
    /// sequence, so its attention mask is all ones).</summary>
    public ClassificationResult Classify(int[] ids)
    {
        if (ids.Length == 0)
        {
            throw new ArgumentException("cannot classify an empty input", nameof(ids));
        }
        var mask = Enumerable.Repeat(1, ids.Length).ToArray();
        int pooled = ClassifierPooling.Index(Surface.Pooling, mask);

        _session.Reset();
        // The backbone is causal, so the hidden state at the pooled position
        // depends only on the tokens up to it: forwarding exactly that prefix
        // gives the same row a right-padded batch would, without computing
        // anything past it.
        var row = _session.PrefillHidden(ids[..(pooled + 1)]);
        var logits = new float[Surface.NumLabels];
        for (int k = 0; k < logits.Length; k++)
        {
            logits[k] = TensorOps.Dot(row.Row(0), _score.Row(k)) + (_bias?[k] ?? 0f);
        }
        return new ClassificationResult(logits, ClassifierPooling.Activate(Surface.ProblemType, logits), pooled, ids.Length);
    }
}

/// <summary>The pooling and output rules a classifier surface can record.</summary>
public static class ClassifierPooling
{
    /// <summary>Only last-token pooling is served: the other pooling kinds
    /// belong to embedding models, and a classifier recording one is refused
    /// rather than pooled some other way.</summary>
    public static void Validate(PoolingSurface pooling)
    {
        if (pooling.Kind != PoolingKind.Last)
        {
            throw new UnsupportedOperatorException(
                $"classifier pooling '{pooling.Kind}' is not served — this build pools the last token (causal classifiers)");
        }
    }

    /// <summary>The row the head reads, from an attention mask over a
    /// (right-padded) sequence: with <c>LastNonPad</c>, the last real token,
    /// <c>Σmask − 1</c>; without, the final row regardless of padding.</summary>
    public static int Index(PoolingSurface pooling, IReadOnlyList<int> attentionMask)
    {
        Validate(pooling);
        if (attentionMask.Count == 0)
        {
            throw new ArgumentException("empty attention mask", nameof(attentionMask));
        }
        if (!pooling.LastNonPad)
        {
            return attentionMask.Count - 1;
        }
        int real = attentionMask.Sum();
        if (real == 0)
        {
            throw new ArgumentException("the attention mask has no real tokens to pool", nameof(attentionMask));
        }
        return real - 1;
    }

    /// <summary>Maps logits by HF's <c>problem_type</c>.</summary>
    public static double[] Activate(string problemType, float[] logits) => problemType switch
    {
        "single_label_classification" => Softmax(logits),
        "multi_label_classification" => logits.Select(l => 1.0 / (1.0 + Math.Exp(-l))).ToArray(),
        "regression" => logits.Select(l => (double)l).ToArray(),
        _ => throw new UnsupportedOperatorException($"problem_type '{problemType}' has no output rule in this build"),
    };

    private static double[] Softmax(float[] logits)
    {
        if (logits.Length == 0)
        {
            return Array.Empty<double>();
        }
        double max = logits.Max();
        var e = logits.Select(l => Math.Exp(l - max)).ToArray();
        double sum = e.Sum();
        return e.Select(v => v / sum).ToArray();
    }
}
