using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Per-layer attribution for a token relationship: how much each layer's
/// residual stream at the source position contributes to the model's
/// propensity to predict <c>targetId</c> after the context. This is the
/// patch/LoRA targeting map — layers with large deltas are the tensors to
/// adjust to change the propensity.
/// </summary>
public sealed record CausalAttribution(
    float CleanProbability,
    float CorruptProbability,
    float[] RestoredProbability,
    float[] LayerDelta,
    float[] LayerShare)
{
    /// <summary>Total propensity lost by corrupting the source token.</summary>
    public float TotalEffect => CleanProbability - CorruptProbability;
}

/// <summary>
/// Causal tracing (ROME-style) over the managed runtime: run the context
/// clean; corrupt the source token's embedding; then, for each traced
/// layer in turn, restore the source position's clean residual after that
/// layer and re-measure P(target). The gain over the corrupt run is that
/// layer's share of the knowledge. One forward per traced layer (plus the
/// clean and corrupt runs).
/// </summary>
public static class CausalTracer
{
    /// <summary>
    /// Position-major forward over <c>ids</c>: one token through every
    /// layer before the next token (required for stateful/linear plans;
    /// equivalent for pure-softmax ones). Optionally injects a residual
    /// patch after <c>restore.Layer</c>, optionally captures the clean
    /// residuals of <c>sourceRow</c> after every layer, and optionally
    /// captures the final token's attention trace.
    /// </summary>
    public static Tensor2D RunPositionMajor(
        GenericRuntime rt,
        ComponentOpPlan plan,
        int[] ids,
        GenericRuntime.ResidualPatch? restore = null,
        float[][]? captureRows = null,
        int sourceRow = -1,
        bool captureTrace = false)
    {
        rt.Kv.Reset();
        rt.ResetSession();
        rt.ClearPatches();
        rt.AttentionTrace = captureTrace ? new List<GenericRuntime.LayerHeadAttention>() : null;

        Tensor2D? hidden = null;
        for (int p = 0; p < ids.Length; p++)
        {
            hidden = rt.Embed(new[] { ids[p] });
            var qpos = new[] { p };
            var kvpos = Enumerable.Range(0, p + 1).ToArray();
            for (int layer = 0; layer < plan.Layers.Count; layer++)
            {
                if (restore is { } r && r.Layer == layer)
                {
                    rt.SetPatch(r.Layer, r.Row, r.Values);
                }
                hidden = rt.RunLayerInternal(hidden, layer, qpos, kvpos, appendKv: true);
                if (captureRows is not null && p == sourceRow)
                {
                    captureRows[layer] = hidden.Row(0).ToArray();
                }
            }
        }

        rt.ClearPatches();
        return rt.FinalNormAndHead(hidden!);
    }

    /// <summary>Softmax probability of <c>targetId</c> under the LAST row of
    /// the logits matrix (max-subtracted, fp64 accumulation).</summary>
    public static float SoftmaxProb(Tensor2D logits, int targetId)
    {
        var row = logits.Row(logits.Rows - 1);
        if (targetId < 0 || targetId >= row.Length)
        {
            return 0f;
        }
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
        if (sum <= 0 || double.IsNaN(sum) || double.IsInfinity(sum))
        {
            return 0f;
        }
        return (float)(Math.Exp(row[targetId] - max) / sum);
    }

    /// <summary>Max softmax probability over the candidate target ids — the
    /// router scores both the standalone and the space-merged ("ĠParis")
    /// spellings.</summary>
    public static float SoftmaxProb(Tensor2D logits, IReadOnlyCollection<int> targetIds)
    {
        float best = 0f;
        foreach (var id in targetIds)
        {
            best = Math.Max(best, SoftmaxProb(logits, id));
        }
        return best;
    }

    /// <summary>
    /// Attribution of the A→B link: how much each traced layer's residual
    /// at <c>sourceRow</c> drives P(B). <c>corruptTokenId</c> replaces
    /// <c>sourceRow</c>'s embedding in the corrupted runs.
    /// </summary>
    public static CausalAttribution Trace(
        GenericRuntime rt,
        ComponentOpPlan plan,
        int[] contextIds,
        int sourceRow,
        int[] targetIds,
        int corruptTokenId,
        int layerStart,
        int layerEnd,
        Action? progress = null)
    {
        var corrupted = (int[])contextIds.Clone();
        corrupted[sourceRow] = corruptTokenId;

        // Clean run + per-layer residual capture at the source row.
        var cleanRows = new float[plan.Layers.Count][];
        var cleanLogits = RunPositionMajor(rt, plan, contextIds, captureRows: cleanRows, sourceRow: sourceRow);
        float clean = SoftmaxProb(cleanLogits, targetIds);

        var corruptLogits = RunPositionMajor(rt, plan, corrupted);
        float corruptProb = SoftmaxProb(corruptLogits, targetIds);

        int start = Math.Max(0, layerStart);
        int end = Math.Min(plan.Layers.Count, layerEnd);
        var restored = new float[plan.Layers.Count];
        for (int layer = start; layer < end; layer++)
        {
            var restore = new GenericRuntime.ResidualPatch(layer, sourceRow, cleanRows[layer]);
            var logits = RunPositionMajor(rt, plan, corrupted, restore);
            restored[layer] = SoftmaxProb(logits, targetIds);
            progress?.Invoke();
        }

        var delta = new float[plan.Layers.Count];
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            delta[l] = restored[l] - corruptProb;
        }
        float total = delta.Sum();
        var share = new float[plan.Layers.Count];
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            share[l] = total > 1e-9 ? delta[l] / total : 0f;
        }
        return new CausalAttribution(clean, corruptProb, restored, delta, share);
    }
}