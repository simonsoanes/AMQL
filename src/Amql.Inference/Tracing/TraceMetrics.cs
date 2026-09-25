namespace Amql.Inference.Tracing;

/// <summary>
/// One operator's statistics reduced over a whole run, plus the intensity used
/// to colour it. Kept free of any UI types so the analysis is testable without
/// a display: a single token says very little, and the question the map exists
/// to answer — which tensors carry this computation — is a reduction over all
/// of them.
/// </summary>
public sealed record NodeStats(
    int NodeId,
    int Layer,
    string Op,
    string? WeightObject,
    string? WeightTensor,
    float MeanL2,
    float MaxL2,
    float MeanAbs,
    double MeanMs,
    int Steps)
{
    /// <summary>Mean L2 relative to the busiest operator in the run, in [0, 1].
    /// A relative scale rather than an absolute one because magnitudes differ
    /// by orders of magnitude between a projection output and a norm, and an
    /// absolute ramp would leave most of the map one colour.</summary>
    public float Intensity { get; init; }

    /// <summary>True when this operator reads a single weight tensor, and so is
    /// something the user can select and edit.</summary>
    public bool HasWeight => WeightObject is not null;

    /// <summary>A display name for the weight, or null for a pure activation.</summary>
    public string? WeightLabel => WeightTensor;
}

/// <summary>
/// Reductions and layout order over a <see cref="RunTrace"/>. The visualiser
/// draws what this computes and holds no analysis of its own, so the numbers on
/// screen are the numbers the tests assert on.
/// </summary>
public static class TraceMetrics
{
    /// <summary>
    /// Canonical operator order within a layer, so a column reads top to bottom
    /// in the order the computation actually happens. Operators a given layer
    /// does not have are simply absent — a linear-attention layer has no
    /// attn_q/attn_k/attn_v, and a dense layer has no ffn_routed.
    /// </summary>
    public static readonly IReadOnlyList<string> OpOrder = new[]
    {
        "pre_attn_norm",
        "attn_q", "attn_k", "attn_v", "attn_context", "attn_output",
        "softmax_attn", "linear_attn", "conv",
        "pre_ffn_norm",
        "ffn_dense", "ffn_routed",
        "post_ffn_norm",
        "residual_out",
    };

    /// <summary>The row a given operator occupies in a layer column. Anything
    /// not in <see cref="OpOrder"/> is appended after it in first-seen order,
    /// so a newly instrumented operator still gets a stable slot instead of
    /// vanishing from the map.</summary>
    public static IReadOnlyDictionary<string, int> OpSlots(IReadOnlyList<OpNode> nodes)
    {
        var slots = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < OpOrder.Count; i++)
        {
            slots[OpOrder[i]] = i;
        }
        int next = OpOrder.Count;
        foreach (var node in nodes)
        {
            if (!slots.ContainsKey(node.Op))
            {
                slots[node.Op] = next++;
            }
        }
        return slots;
    }

    /// <summary>Reduces every node over the whole run and sets
    /// <see cref="NodeStats.Intensity"/> relative to the busiest one.</summary>
    public static IReadOnlyList<NodeStats> Reduce(RunTrace trace)
    {
        var acc = new Dictionary<int, (double SumL2, double MaxL2, double SumAbs, double Ms, int Count)>();
        foreach (var step in trace.Steps)
        {
            foreach (var op in step.Ops)
            {
                acc.TryGetValue(op.NodeId, out var a);
                acc[op.NodeId] = (a.SumL2 + op.L2, Math.Max(a.MaxL2, (double)op.L2),
                    a.SumAbs + op.MeanAbs, a.Ms + op.Ms, a.Count + 1);
            }
        }

        var stats = new List<NodeStats>(acc.Count);
        foreach (var (id, a) in acc)
        {
            var node = trace.Nodes[id];
            int count = Math.Max(1, a.Count);
            stats.Add(new NodeStats(
                id, node.Layer, node.Op, node.Weight?.ObjectId, node.Weight?.TensorName,
                (float)(a.SumL2 / count), (float)a.MaxL2, (float)(a.SumAbs / count),
                a.Ms / count, a.Count)
            {
                Intensity = 0f,
            });
        }

        float peak = stats.Count == 0 ? 0f : stats.Max(s => s.MeanL2);
        if (peak <= 0f)
        {
            return stats;
        }
        for (int i = 0; i < stats.Count; i++)
        {
            stats[i] = stats[i] with { Intensity = stats[i].MeanL2 / peak };
        }
        return stats;
    }

    /// <summary>Nodes ordered by mean activation magnitude, busiest first — the
    /// ranking panel, and the shortlist of tensors worth editing.</summary>
    public static IReadOnlyList<NodeStats> RankByMeanL2(RunTrace trace)
        => Reduce(trace).OrderByDescending(s => s.MeanL2).ThenBy(s => s.NodeId).ToArray();

    /// <summary>Nodes ordered by mean wall time, slowest first. Magnitude says
    /// what mattered; this says what cost anything.</summary>
    public static IReadOnlyList<NodeStats> RankByMeanMs(RunTrace trace)
        => Reduce(trace).OrderByDescending(s => s.MeanMs).ThenBy(s => s.NodeId).ToArray();

    /// <summary>Per-node L2 for one step, for the scrubber: selecting a step
    /// recolours the map with that step's values rather than the aggregate.</summary>
    public static IReadOnlyDictionary<int, float> StepValues(RunTrace trace, int stepIndex)
    {
        if (stepIndex < 0 || stepIndex >= trace.Steps.Count)
        {
            return new Dictionary<int, float>();
        }
        var step = trace.Steps[stepIndex];
        float peak = step.Ops.Count == 0 ? 0f : step.Ops.Max(o => o.L2);
        var result = new Dictionary<int, float>(step.Ops.Count);
        foreach (var op in step.Ops)
        {
            result[op.NodeId] = peak <= 0f ? 0f : op.L2 / peak;
        }
        return result;
    }

    /// <summary>The number of layer columns and the deepest operator row, which
    /// is all the renderer needs to size the map.</summary>
    public static (int Layers, int Rows) LayoutSize(RunTrace trace)
    {
        int layers = trace.Nodes.Count == 0 ? 0 : trace.Nodes.Max(n => n.Layer) + 1;
        return (Math.Max(layers, trace.Layers), OpSlots(trace.Nodes).Count);
    }
}
