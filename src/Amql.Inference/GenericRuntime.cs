using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The generic layer interpreter: embed each token, run the layer loop
/// (pre-norm → attention → residual → pre-ffn norm → FFN → residual),
/// final norm, output head. Position/span behaviour comes exclusively from
/// the per-layer policy table; no architecture branch exists here.
/// </summary>
public sealed class GenericRuntime
{
    private readonly ComponentOpPlan _plan;
    private readonly WeightLoader _weights;
    private readonly Dictionary<int, LayerKvGeometry> _kvGeometry = new();

    public GenericRuntime(ComponentOpPlan plan, OperandStore store)
    {
        _plan = plan;
        _weights = new WeightLoader(store);
    }

    public ComponentOpPlan Plan => _plan;
    public WeightLoader Weights => _weights;
    public RowKvCache Kv { get; } = new();

    public LayerKvGeometry GeometryFor(int layer)
    {
        if (_kvGeometry.TryGetValue(layer, out var geometry))
        {
            return geometry;
        }
        var attn = _plan.Layers[layer].Attention;
        geometry = new LayerKvGeometry(attn.KvDim, attn.Window);
        _kvGeometry[layer] = geometry;
        return geometry;
    }

    // ── Embedding ───────────────────────────────────────────────────────────

    public Tensor2D Embed(ReadOnlySpan<int> tokens)
    {
        var embed = _plan.Embedding
            ?? throw new UnsupportedOperatorException("plan carries no embedding op");
        var table = _weights.Matrix(embed.Table, embed.VocabSize, embed.HiddenSize);
        var hidden = TensorOps.GatherRows(table, tokens);
        if (embed.Norm is { } norm)
        {
            var ones = new float[hidden.Cols];
            for (int i = 0; i < ones.Length; i++)
            {
                ones[i] = 1f;
            }
            Norms.ApplyInPlace(hidden, norm.Kind, norm.Eps, ones, 0f);
        }
        if (embed.Scale is { } scale)
        {
            for (int i = 0; i < hidden.Data.Length; i++)
            {
                hidden.Data[i] = (float)(hidden.Data[i] * scale);
            }
        }
        return hidden;
    }

    // ── Layer loop ──────────────────────────────────────────────────────────

    /// <summary>Runs one layer over the given positions. When
    /// <c>appendKv</c> the freshly computed key/value rows are added to the
    /// cache before attention runs (prefill and decode both append; a
    /// caller replaying a layer never does).</summary>
    public Tensor2D RunLayerInternal(Tensor2D x, int layer, int[] queryPositions, int[] kvPositions, bool appendKv)
    {
        var layerPlan = _plan.Layers[layer];
        var attn = layerPlan.Attention;
        int hidden = _plan.HiddenSize;

        // Pre-attention norm.
        Tensor2D h = x;
        if (layerPlan.PreAttentionNorm is { } preAttnNorm)
        {
            h = x.Clone();
            var preAttnW = _weights.Vector(preAttnNorm.Weight, hidden);
            Norms.ApplyInPlace(h, preAttnNorm.Kind, preAttnNorm.Eps, preAttnW, preAttnNorm.WeightOffset);
        }

        // Projections.
        var q = TensorOps.MatMulTransposedB(h, _weights.Matrix(attn.QProj, attn.QDim, hidden));
        var k = TensorOps.MatMulTransposedB(h, _weights.Matrix(attn.KProj, attn.KvDim, hidden));
        var v = TensorOps.MatMulTransposedB(h, _weights.Matrix(attn.VProj, attn.KvDim, hidden));

        ApplyQkNorm(q, k, attn, hidden);

        if (attn.ParameterFreeQkNorm.V && !attn.VFromK)
        {
            var unit = new float[attn.HeadDim];
            Array.Fill(unit, 1f);
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(v, attn.NumKvHeads, attn.HeadDim, unit, attn.ParameterFreeQkNormEps);
            }
            else
            {
                ApplyPerRow(v, hidden, unit, attn.ParameterFreeQkNormEps);
            }
        }

        // RoPE — position policy from the per-layer table; the reference
        // guarantee that mutating one row provably changes execution.
        Rope.Apply(q, attn.NumQHeads, attn.HeadDim, attn.Position, queryPositions);

        // The freshly computed k rows (the appended slice of the cache)
        // rotate at their OWN absolute positions. In prefill the local rows
        // are the whole range; in decode they are the trailing rows, whose
        // positions are the trailing entries of the cache position array.
        int kvRows = k.Rows;
        var ropeKvPositions = kvRows == kvPositions.Length
            ? kvPositions
            : kvPositions[^kvRows..];
        Rope.Apply(k, attn.NumKvHeads, attn.HeadDim, attn.Position, ropeKvPositions);

        if (attn.VFromK)
        {
            v = k.Clone();
        }

        if (appendKv)
        {
            AppendRows(layer, k, v);
        }

        // Attention over the current cache (which includes the new rows).
        var kRows = Kv.Keys(layer);
        var vRows = Kv.Values(layer);
        int kvSeq = kRows.Count;
        var kMat = new Tensor2D(kRows.SelectMany(r => r).ToArray(),
            kvSeq, kRows.Count == 0 ? attn.KvDim : kRows[0].Length);
        var vMat = new Tensor2D(vRows.SelectMany(r => r).ToArray(),
            kvSeq, vRows.Count == 0 ? attn.KvDim : vRows[0].Length);

        float[]? sinks = null;
        var output = AttentionKernel.Execute(
            q, kMat, vMat,
            attn.NumQHeads, attn.NumKvHeads, attn.HeadDim,
            attn.ScoreScale, attn.LogitSoftcapping, sinks,
            attn.Window is { } windowValue ? checked((int)windowValue) : null,
            queryPositions, kvPositions);

        var o = TensorOps.MatMulTransposedB(output, _weights.Matrix(attn.OProj, hidden, attn.QDim));

        // Residual, with optional post-attention norm up front.
        if (layerPlan.PostAttentionNorm is { } postAttnNorm)
        {
            var postAttnW = _weights.Vector(postAttnNorm.Weight, hidden);
            Norms.ApplyInPlace(o, postAttnNorm.Kind, postAttnNorm.Eps, postAttnW, postAttnNorm.WeightOffset);
        }
        AddInPlace(x, o);

        // Pre-FFN norm.
        Tensor2D hf = x;
        if (layerPlan.PreFfnNorm is { } preFfnNorm)
        {
            hf = x.Clone();
            var preFfnW = _weights.Vector(preFfnNorm.Weight, hidden);
            Norms.ApplyInPlace(hf, preFfnNorm.Kind, preFfnNorm.Eps, preFfnW, preFfnNorm.WeightOffset);
        }

        // FFN.
        if (layerPlan.Ffn is { IsPresent: true } ffn)
        {
            Tensor2D ffnOut;
            if (ffn.Dense is { } dense)
            {
                var gate = dense.Gate is null
                    ? null
                    : _weights.Matrix(dense.Gate, dense.IntermediateSize, hidden);
                var up = _weights.Matrix(dense.Up, dense.IntermediateSize, hidden);
                var down = _weights.Matrix(dense.Down, hidden, dense.IntermediateSize);
                ffnOut = FfnKernel.Dense(hf, gate, up, down, dense.Activation, dense.IsGated);
            }
            else
            {
                var routed = ffn.Routed!;
                ffnOut = RunRoutedFfn(hf, routed);
            }

            if (layerPlan.PostFfnNorm is { } postFfnNorm)
            {
                var postFfnW = _weights.Vector(postFfnNorm.Weight, hidden);
                Norms.ApplyInPlace(ffnOut, postFfnNorm.Kind, postFfnNorm.Eps, postFfnW, postFfnNorm.WeightOffset);
            }
            AddInPlace(x, ffnOut);
        }

        return x;
    }

    private Tensor2D RunRoutedFfn(Tensor2D h, RoutedFfnOp routed)
    {
        int hidden = routed.HiddenSize;
        var router = _weights.Matrix(routed.Router, routed.NumExperts, hidden);

        // Gate/up/down are resolved through the operand store by explicit
        // per-expert names — the prefix fields disambiguate spellings.
        var gates = new Tensor2D[routed.NumExperts];
        var ups = new Tensor2D[routed.NumExperts];
        var downs = new Tensor2D[routed.NumExperts];
        string stackId = routed.Router.ObjectId;
        for (int e = 0; e < routed.NumExperts; e++)
        {
            var stem = $"{routed.ExpertGatePrefix}{e}.";
            gates[e] = _weights.Matrix(new OperandRef(stackId, stem + "gate_proj.weight"), routed.ExpertIntermediateSize, hidden);
            ups[e] = _weights.Matrix(new OperandRef(stackId, stem + "up_proj.weight"), routed.ExpertIntermediateSize, hidden);
            downs[e] = _weights.Matrix(new OperandRef(stackId, stem + "down_proj.weight"), hidden, routed.ExpertIntermediateSize);
        }

        return FfnKernel.Routed(h, router, gates, ups, downs,
            routed.TopK, routed.RoutingPolicy, routed.Activation);
    }

    private void ApplyQkNorm(Tensor2D q, Tensor2D k, AttentionOp attn, int hidden)
    {
        var pf = attn.ParameterFreeQkNorm;
        if (!pf.Q && !pf.K)
        {
            return;
        }
        // Weightless: RMS-normalise with unit weights. Scope decides the
        // reduction: per head-dim slice, or the full projection.
        var unit = new float[attn.HeadDim];
        Array.Fill(unit, 1f);
        if (pf.Q)
        {
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(q, attn.NumQHeads, attn.HeadDim, unit, attn.ParameterFreeQkNormEps);
            }
            else
            {
                ApplyPerRow(q, hidden, unit, attn.ParameterFreeQkNormEps);
            }
        }
        if (pf.K)
        {
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(k, attn.NumKvHeads, attn.HeadDim, unit, attn.ParameterFreeQkNormEps);
            }
            else
            {
                ApplyPerRow(k, hidden, unit, attn.ParameterFreeQkNormEps);
            }
        }
    }

    private static void ApplyPerHead(Tensor2D x, int heads, int headDim, float[] unit, double eps)
    {
        for (int r = 0; r < x.Rows; r++)
        {
            var row = x.Row(r);
            for (int h = 0; h < heads; h++)
            {
                Norms.ApplyRow(row.Slice(h * headDim, headDim), NormType.RmsNorm, eps, unit, 0f);
            }
        }
    }

    private static void ApplyPerRow(Tensor2D x, int width, float[] unit, double eps)
    {
        for (int r = 0; r < x.Rows; r++)
        {
            Norms.ApplyRow(x.Row(r), NormType.RmsNorm, eps, unit, 0f);
        }
    }

    private void AppendRows(int layer, Tensor2D k, Tensor2D v)
    {
        for (int r = 0; r < k.Rows; r++)
        {
            Kv.Append(layer, k.Row(r), v.Row(r));
        }
    }

    // ── Residual / head ─────────────────────────────────────────────────────

    public static void AddInPlace(Tensor2D target, Tensor2D addend)
    {
        if (target.Rows != addend.Rows || target.Cols != addend.Cols)
        {
            throw new ArgumentException($"residual shape mismatch: {target.Rows}x{target.Cols} vs {addend.Rows}x{addend.Cols}");
        }
        for (int i = 0; i < target.Data.Length; i++)
        {
            target.Data[i] += addend.Data[i];
        }
    }

    public Tensor2D FinalNormAndHead(Tensor2D hidden)
    {
        var finalNorm = _plan.FinalNorm;
        var w = _weights.Vector(finalNorm.Weight, hidden.Cols);
        Norms.ApplyInPlace(hidden, finalNorm.Kind, finalNorm.Eps, w, finalNorm.WeightOffset);

        if (_plan.Output is not { } output)
        {
            throw new UnsupportedOperatorException("plan carries no output head");
        }
        var head = _weights.Matrix(output.Projection, output.VocabSize, output.HiddenSize);
        var logits = TensorOps.MatMulTransposedB(hidden, head);
        if (output.Multiplier is { } multiplier)
        {
            for (int i = 0; i < logits.Data.Length; i++)
            {
                logits.Data[i] = (float)(logits.Data[i] * multiplier);
            }
        }
        if (output.LogitSoftcapping is { } cap)
        {
            for (int i = 0; i < logits.Data.Length; i++)
            {
                logits.Data[i] = (float)(cap * Math.Tanh(logits.Data[i] / cap));
            }
        }
        return logits;
    }
}