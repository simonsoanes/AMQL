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
        var attn = _plan.Layers[layer].Attention
            ?? throw new InvalidOperationException($"layer {layer} has no softmax attention geometry");
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
        int hidden = _plan.HiddenSize;

        // Pre-attention norm (shared by both operator families).
        Tensor2D h = x;
        if (layerPlan.PreAttentionNorm is { } preAttnNorm)
        {
            h = x.Clone();
            var preAttnW = _weights.Vector(preAttnNorm.Weight, hidden);
            Norms.ApplyInPlace(h, preAttnNorm.Kind, preAttnNorm.Eps, preAttnW, preAttnNorm.WeightOffset);
        }

        // Token mixer dispatch.
        if (layerPlan.LinearAttention is { } linear)
        {
            var mixerOut = RunLinearAttention(h, layer, linear);
            AddInPlace(x, mixerOut);
        }
        else
        {
            var mixerOut = RunSoftmaxAttention(h, layer, layerPlan, queryPositions, kvPositions, appendKv);
            AddInPlace(x, mixerOut);
        }

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

    // ── softmax attention mixer ─────────────────────────────────────────────

    private Tensor2D RunSoftmaxAttention(Tensor2D h, int layer, LayerPlan layerPlan, int[] queryPositions, int[] kvPositions, bool appendKv)
    {
        var attn = layerPlan.Attention!;
        int hidden = _plan.HiddenSize;

        // q projection — with the hard output gate, q_proj [hidden, 2×QDim]
        // interleaves per head [q_h | gate_h]; the reference's
        // view(…, -1, 2·head_dim).chunk(2) takes each block's half.
        var qRaw = TensorOps.MatMulTransposedB(h, _weights.Matrix(attn.QProj, attn.QProjWidth, hidden));
        Tensor2D? gate = null;
        Tensor2D q = attn.OutputGate ? ChunkBlocks(qRaw, attn.HeadDim, 0) : qRaw;
        if (attn.OutputGate)
        {
            gate = ChunkBlocks(qRaw, attn.HeadDim, 1);
        }
        var k = TensorOps.MatMulTransposedB(h, _weights.Matrix(attn.KProj, attn.KvDim, hidden));
        var v = TensorOps.MatMulTransposedB(h, _weights.Matrix(attn.VProj, attn.KvDim, hidden));

        // QK norm — weighted (learned q_norm/k_norm when present) or
        // parameter-free; reducer per head dim (the reference's "norm only
        // on the head dim").
        if (attn.QNorm is not null || attn.KNorm is not null)
        {
            ApplyWeightedQkNorm(q, k, attn, attn.QNorm, attn.KNorm, hidden);
        }
        ApplyParameterFreeQkNorm(q, k, v, attn, hidden);

        // RoPE — position policy from the per-layer table; partial rotary
        // (text mrope) rotates the first head_dim × factor dims.
        Rope.Apply(q, attn.NumQHeads, attn.HeadDim, attn.Position, queryPositions);

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

        var kRows2 = Kv.Keys(layer);
        var vRows = Kv.Values(layer);
        int kvSeq = kRows2.Count;
        var kMat = new Tensor2D(kRows2.SelectMany(r => r).ToArray(),
            kvSeq, kvSeq == 0 ? attn.KvDim : kRows2[0].Length);
        var vMat = new Tensor2D(vRows.SelectMany(r => r).ToArray(),
            kvSeq, kvSeq == 0 ? attn.KvDim : vRows[0].Length);

        var output = AttentionKernel.Execute(
            q, kMat, vMat,
            attn.NumQHeads, attn.NumKvHeads, attn.HeadDim,
            attn.ScoreScale, attn.LogitSoftcapping, null,
            attn.Window is { } windowValue ? checked((int)windowValue) : null,
            queryPositions, kvPositions);

        // Hard output gate: attention output × sigmoid(gate).
        if (gate is not null && output.Rows == gate.Rows)
        {
            for (int i = 0; i < output.Data.Length; i++)
            {
                output.Data[i] *= 1f / (1f + MathF.Exp(-gate.Data[i]));
            }
        }

        var o = TensorOps.MatMulTransposedB(output, _weights.Matrix(attn.OProj, hidden, attn.QDim));

        // Post-attention norm (four-norm placement) applies to the mixer
        // output before the residual add.
        if (layerPlan.PostAttentionNorm is { } postAttnNorm)
        {
            var postAttnW = _weights.Vector(postAttnNorm.Weight, hidden);
            Norms.ApplyInPlace(o, postAttnNorm.Kind, postAttnNorm.Eps, postAttnW, postAttnNorm.WeightOffset);
        }
        return o;
    }

    private void ApplyWeightedQkNorm(Tensor2D q, Tensor2D k, AttentionOp attn, NormOp? qNorm, NormOp? kNorm, int hidden)
    {
        if (qNorm is { } qn)
        {
            var w = _weights.Vector(qn.Weight, qn.Width);
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(q, attn.NumQHeads, attn.HeadDim, w, qn.Eps, qn.WeightOffset);
            }
            else
            {
                ApplyPerRows(q, hidden, w, qn.Eps, qn.WeightOffset);
            }
        }
        if (kNorm is { } kn)
        {
            var w = _weights.Vector(kn.Weight, kn.Width);
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(k, attn.NumKvHeads, attn.HeadDim, w, kn.Eps, kn.WeightOffset);
            }
            else
            {
                ApplyPerRows(k, hidden, w, kn.Eps, kn.WeightOffset);
            }
        }
    }

    private void ApplyParameterFreeQkNorm(Tensor2D q, Tensor2D k, Tensor2D v, AttentionOp attn, int hidden)
    {
        var pf = attn.ParameterFreeQkNorm;
        var unit = new float[attn.HeadDim];
        Array.Fill(unit, 1f);
        if (pf.Q)
        {
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(q, attn.NumQHeads, attn.HeadDim, unit, attn.ParameterFreeQkNormEps, 0);
            }
            else
            {
                ApplyPerRows(q, hidden, unit, attn.ParameterFreeQkNormEps, 0);
            }
        }
        if (pf.K)
        {
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(k, attn.NumKvHeads, attn.HeadDim, unit, attn.ParameterFreeQkNormEps, 0);
            }
            else
            {
                ApplyPerRows(k, hidden, unit, attn.ParameterFreeQkNormEps, 0);
            }
        }
        if (pf.V && !attn.VFromK)
        {
            if (attn.QkNormScope == QkNormScope.PerHead)
            {
                ApplyPerHead(v, attn.NumKvHeads, attn.HeadDim, unit, attn.ParameterFreeQkNormEps, 0);
            }
            else
            {
                ApplyPerRows(v, hidden, unit, attn.ParameterFreeQkNormEps, 0);
            }
        }
    }

    public static void ApplyPerHead(Tensor2D x, int heads, int headDim, float[] weight, double eps, float weightOffset)
    {
        for (int r = 0; r < x.Rows; r++)
        {
            var row = x.Row(r);
            for (int h = 0; h < heads; h++)
            {
                Norms.ApplyRow(row.Slice(h * headDim, headDim), NormType.RmsNorm, eps, weight, weightOffset);
            }
        }
    }

    private static void ApplyPerRows(Tensor2D x, int width, float[] weight, double eps, float weightOffset)
    {
        for (int r = 0; r < x.Rows; r++)
        {
            Norms.ApplyRow(x.Row(r), NormType.RmsNorm, eps, weight, weightOffset);
        }
    }

    public static Tensor2D ChunkBlocks(Tensor2D source, int headDim, int chunkIndex)
    {
        // Source rows contain heads of [headDim × 2]; take the chunkIndex-th
        // headDim-slice of every head (the reference's chunk(qkv, 2, -1)).
        int stride = 2 * headDim;
        int heads = source.Cols / stride;
        var result = new float[source.Rows * heads * headDim];
        for (int r = 0; r < source.Rows; r++)
        {
            for (int h = 0; h < heads; h++)
            {
                Array.Copy(source.Data, r * source.Cols + h * stride + chunkIndex * headDim,
                    result, r * heads * headDim + h * headDim, headDim);
            }
        }
        return new Tensor2D(result, source.Rows, heads * headDim);
    }

    public static Tensor2D ColRange(Tensor2D source, int start, int width)
    {
        var result = new float[source.Rows * width];
        for (int r = 0; r < source.Rows; r++)
        {
            Array.Copy(source.Data, r * source.Cols + start, result, r * width, width);
        }
        return new Tensor2D(result, source.Rows, width);
    }

    // ── linear attention (GatedDeltaNet) mixer ─────────────────────────────

    private readonly Dictionary<int, LinearAttentionState> _linearStates = new();

    /// <summary>Resets the per-layer recurrent state and the session
    /// position (fresh session). The caller coordinates with the KV cache
    /// reset.</summary>
    public void ResetSession()
    {
        _linearStates.Clear();
        SessionPosition = 0;
    }

    private LinearAttentionState StateFor(int layer, LinearAttentionOp op)
    {
        if (!_linearStates.TryGetValue(layer, out var state))
        {
            state = new LinearAttentionState(op.NumVHeads, op.HeadKDim, op.HeadVDim, op.ConvDim, op.ConvKernel);
            _linearStates[layer] = state;
        }
        return state;
    }

    private Tensor2D RunLinearAttention(Tensor2D h, int layer, LinearAttentionOp op)
    {
        int t = h.Rows;
        var state = StateFor(layer, op);

        var mixed = TensorOps.MatMulTransposedB(h, _weights.Matrix(op.InProjQkv, op.ConvDim, op.HiddenSize));
        var z = TensorOps.MatMulTransposedB(h, _weights.Matrix(op.InProjZ, op.ValueDim, op.HiddenSize));
        var b = TensorOps.MatMulTransposedB(h, _weights.Matrix(op.InProjB, op.NumVHeads, op.HiddenSize));
        var a = TensorOps.MatMulTransposedB(h, _weights.Matrix(op.InProjA, op.NumVHeads, op.HiddenSize));

        // Depthwise causal conv over the qkv stream (columns → channels),
        // SiLU activated. A single position advances the conv state.
        var stream = ToColumnMajor(mixed, t, op.ConvDim);
        var convWeight = _weights.Vector(op.Conv1d, op.ConvDim * op.ConvKernel);
        float[,] walked = t == 1
            ? GatedDeltaKernel.ConvStep(stream, convWeight, op.ConvKernel, silu: true, state)
            : GatedDeltaKernel.ConvForward(stream, convWeight, op.ConvKernel, silu: true, state);

        // Split q / k / v (the reference splits [key, key, value]) and
        // reshape into heads; expand key heads to the value-head count
        // when the ratio > 1 (repeat_interleave).
        var q = SliceFromChannels(walked, 0, op.KeyDim, t);
        var k = SliceFromChannels(walked, op.KeyDim, op.KeyDim, t);
        var v = SliceFromChannels(walked, 2 * op.KeyDim, op.ValueDim, t);
        int ratio = op.NumVHeads / op.NumKHeads;
        if (ratio > 1)
        {
            q = RepeatInterleave(q, op.NumKHeads, op.HeadKDim, ratio);
            k = RepeatInterleave(k, op.NumKHeads, op.HeadKDim, ratio);
        }

        // Decay g and gate β per value head.
        var aLog = _weights.Vector(op.ALog, op.NumVHeads);
        var dtBias = _weights.Vector(op.DtBias, op.NumVHeads);
        var g = new float[a.Data.Length];
        for (int i = 0; i < a.Data.Length; i++)
        {
            int head = i % op.NumVHeads;
            double x = a.Data[i] + dtBias[head];
            double softPlus = x > 20 ? x : Math.Log(1.0 + Math.Exp(x));
            g[i] = (float)(-Math.Exp(aLog[head]) * softPlus);
        }
        var beta = new float[b.Data.Length];
        for (int i = 0; i < beta.Length; i++)
        {
            beta[i] = 1f / (1f + MathF.Exp(-b.Data[i]));
        }

        var core = GatedDeltaKernel.Recurrent(
            q, k, v,
            new Tensor2D(g, t, op.NumVHeads),
            new Tensor2D(beta, t, op.NumVHeads),
            op.NumVHeads, op.HeadKDim, op.HeadVDim, state);

        // z-gated RMSNorm over the value head dim, then output projection.
        var normWeight = _weights.Vector(op.NormWeight, op.HeadVDim);
        int rows = t * op.NumVHeads;
        var gated = GatedDeltaKernel.GatedRMSNorm(
            new Tensor2D(core.Data, rows, op.HeadVDim),
            new Tensor2D(z.Data, rows, op.HeadVDim),
            normWeight, op.NormEps);
        var out2 = TensorOps.MatMulTransposedB(
            new Tensor2D(gated.Data, t, op.ValueDim),
            _weights.Matrix(op.OutProj, op.HiddenSize, op.ValueDim));
        return out2;
    }

    private static float[,] ToColumnMajor(Tensor2D matrix, int rows, int cols)
    {
        var result = new float[cols, rows];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result[c, r] = matrix.Data[r * cols + c];
            }
        }
        return result;
    }

    private static Tensor2D SliceFromChannels(float[,] channels, int startCol, int width, int rows)
    {
        var result = new float[rows * width];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < width; c++)
            {
                result[r * width + c] = channels[startCol + c, r];
            }
        }
        return new Tensor2D(result, rows, width);
    }

    private static Tensor2D RepeatInterleave(Tensor2D heads, int srcHeads, int headDim, int ratio)
    {
        var result = new float[heads.Rows * srcHeads * ratio * headDim];
        for (int r = 0; r < heads.Rows; r++)
        {
            for (int raw = 0; raw < srcHeads * headDim; raw++)
            {
                int srcHead = raw / headDim;
                int dim = raw % headDim;
                for (int rep = 0; rep < ratio; rep++)
                {
                    int dst = ((srcHead * ratio + rep) * headDim + dim);
                    result[r * result.Length / heads.Rows + dst] = heads.Data[r * heads.Cols + raw];
                }
            }
        }
        return new Tensor2D(result, heads.Rows, srcHeads * ratio * headDim);
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

    private void AppendRows(int layer, Tensor2D k, Tensor2D v)
    {
        for (int r = 0; r < k.Rows; r++)
        {
            Kv.Append(layer, k.Row(r), v.Row(r));
        }
    }

    // ── residual / head ─────────────────────────────────────────────────────

    /// <summary>One token through every layer at the current position —
    /// the decode step and the position-major prefill loop share this.
    /// Softmax layers append their KV rows; linear layers advance their
    /// recurrent state.</summary>
    public Tensor2D StepForward(int token)
    {
        int position = SessionPosition;
        var hidden = Embed(new[] { token });
        var queryPositions = new[] { position };
        var kvPositions = Enumerable.Range(0, position + 1).ToArray();

        for (int layer = 0; layer < _plan.Layers.Count; layer++)
        {
            hidden = RunLayerInternal(hidden, layer, queryPositions, kvPositions, appendKv: true);
        }
        SessionPosition++;
        return hidden;
    }

    /// <summary>The session's absolute token position — advances once per
    /// consumed token, independent of the KV cache shape (mixed plans may
    /// have no key/value rows at layer 0).</summary>
    public int SessionPosition { get; internal set; }

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