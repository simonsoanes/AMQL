using System.Diagnostics;
using Amql.Inference.Tracing;
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

    public GenericRuntime(ComponentOpPlan plan, OperandStore store, WeightPatch? patch = null, WeightWorkingSet? workingSet = null)
    {
        _plan = plan;
        _weights = new WeightLoader(store, patch, workingSet);
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
            long t = TraceStart;
            Norms.ApplyInPlace(h, preAttnNorm.Kind, preAttnNorm.Eps, preAttnW, preAttnNorm.WeightOffset);
            Report(layer, "pre_attn_norm", preAttnNorm.Weight, h, t);
        }

        // Token mixer dispatch.
        if (layerPlan.LinearAttention is { } linear)
        {
            long t = TraceStart;
            var mixerOut = RunLinearAttention(h, layer, linear);
            Report(layer, "linear_attn", null, mixerOut, t);
            AddResidualInPlace(x, mixerOut);
        }
        else if (layerPlan.Conv is { } conv)
        {
            long t = TraceStart;
            var mixerOut = RunConv(h, layer, conv);
            Report(layer, "conv", null, mixerOut, t);
            AddResidualInPlace(x, mixerOut);
        }
        else
        {
            long t = TraceStart;
            var mixerOut = RunSoftmaxAttention(h, layer, layerPlan, queryPositions, kvPositions, appendKv);
            Report(layer, "softmax_attn", null, mixerOut, t);
            AddResidualInPlace(x, mixerOut);
        }

        // Pre-FFN norm.
        Tensor2D hf = x;
        if (layerPlan.PreFfnNorm is { } preFfnNorm)
        {
            hf = x.Clone();
            var preFfnW = _weights.Vector(preFfnNorm.Weight, hidden);
            long t = TraceStart;
            Norms.ApplyInPlace(hf, preFfnNorm.Kind, preFfnNorm.Eps, preFfnW, preFfnNorm.WeightOffset);
            Report(layer, "pre_ffn_norm", preFfnNorm.Weight, hf, t);
        }

        // FFN.
        if (layerPlan.Ffn is { IsPresent: true } ffn)
        {
            FfnInputCapture?.Invoke(layer, hf);
            Tensor2D ffnOut;
            long t = TraceStart;
            if (ffn.Dense is { } dense)
            {
                var gate = dense.Gate is null
                    ? null
                    : _weights.Matrix(dense.Gate, dense.IntermediateSize, hidden);
                var up = _weights.Matrix(dense.Up, dense.IntermediateSize, hidden);
                var down = _weights.Matrix(dense.Down, hidden, dense.IntermediateSize);
                ffnOut = FfnKernel.Dense(hf, gate, up, down, dense.Activation, dense.IsGated);
                Report(layer, "ffn_dense", dense.Down, ffnOut, t);
            }
            else
            {
                var routed = ffn.Routed!;
                ffnOut = RunRoutedFfn(hf, layer, routed);
                Report(layer, "ffn_routed", routed.Router, ffnOut, t);
            }

            if (layerPlan.PostFfnNorm is { } postFfnNorm)
            {
                var postFfnW = _weights.Vector(postFfnNorm.Weight, hidden);
                long tn = TraceStart;
                Norms.ApplyInPlace(ffnOut, postFfnNorm.Kind, postFfnNorm.Eps, postFfnW, postFfnNorm.WeightOffset);
                Report(layer, "post_ffn_norm", postFfnNorm.Weight, ffnOut, tn);
            }
            AddResidualInPlace(x, ffnOut);
        }

        ApplyPatches(x, layer, queryPositions);
        // started 0 means "no timing" — the residual is the layer's result, not
        // an operation with a duration of its own.
        Report(layer, "residual_out", null, x, 0);
        return x;
    }

    /// <summary>Granite's decoder join is h' = h·residual_multiplier + branch
    /// at BOTH the mixer and the FFN add; families without the knob carry
    /// scale 1.0 and this is the plain add.</summary>
    private void AddResidualInPlace(Tensor2D x, Tensor2D branch)
    {
        double scale = _plan.ResidualScale;
        if (scale != 1.0)
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                x.Data[i] = (float)(x.Data[i] * scale);
            }
        }
        AddInPlace(x, branch);
    }

    // ── softmax attention mixer ─────────────────────────────────────────────

    private List<(int Head, float[] Weights)>? _lastCapture;

    private Tensor2D RunSoftmaxAttention(Tensor2D h, int layer, LayerPlan layerPlan, int[] queryPositions, int[] kvPositions, bool appendKv)
    {
        var attn = layerPlan.Attention!;
        int hidden = _plan.HiddenSize;

        // q projection — with the hard output gate, q_proj [hidden, 2×QDim]
        // interleaves per head [q_h | gate_h]; the reference's
        // view(…, -1, 2·head_dim).chunk(2) takes each block's half.
        var wQ = _weights.Matrix(attn.QProj, attn.QProjWidth, hidden);
        var wK = _weights.Matrix(attn.KProj, attn.KvDim, hidden);
        var wV = _weights.Matrix(attn.VProj, attn.KvDim, hidden);

        // GPU-batched path: q, k, v all share the same input h — launch
        // all three GEMMs on the device in one batch, then sync once.
        Tensor2D qRaw, k, v;
        if (CudaShim.Enabled &&
            wQ.DeviceWeightF16 != IntPtr.Zero &&
            wK.DeviceWeightF16 != IntPtr.Zero &&
            wV.DeviceWeightF16 != IntPtr.Zero)
        {
            var batch = TensorOps.MatMulTransposedBMulti(h, new[] { wQ, wK, wV });
            qRaw = batch[0];
            k = batch[1];
            v = batch[2];
        }
        else
        {
            qRaw = TensorOps.MatMulTransposedB(h, wQ);
            k = TensorOps.MatMulTransposedB(h, wK);
            v = TensorOps.MatMulTransposedB(h, wV);
        }

        // Sub-operator observations carry no duration: on the GPU path q, k and
        // v are a single batched launch, so a per-projection time would be
        // fiction. Timing lives on the layer-level operators instead.
        Report(layer, "attn_q", attn.QProj, qRaw, 0);
        Report(layer, "attn_k", attn.KProj, k, 0);
        Report(layer, "attn_v", attn.VProj, v, 0);

        Tensor2D? gate = null;
        Tensor2D q = attn.OutputGate ? ChunkBlocks(qRaw, attn.HeadDim, 0) : qRaw;
        if (attn.OutputGate)
        {
            gate = ChunkBlocks(qRaw, attn.HeadDim, 1);
        }

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

        _lastCapture = AttentionTrace is not null
            ? new List<(int Head, float[] Weights)>(attn.NumQHeads)
            : null;
        var output = AttentionKernel.Execute(
            q, kMat, vMat,
            attn.NumQHeads, attn.NumKvHeads, attn.HeadDim,
            attn.ScoreScale, attn.LogitSoftcapping, null,
            attn.Window is { } windowValue ? checked((int)windowValue) : null,
            queryPositions, kvPositions,
            _lastCapture);

        if (AttentionTrace is { } trace && _lastCapture is { Count: > 0 })
        {
            foreach (var (head, weights) in _lastCapture)
            {
                trace.Add(new LayerHeadAttention(layer, head, weights));
            }
        }
        _lastCapture = null;

        // Hard output gate: attention output × sigmoid(gate).
        if (gate is not null && output.Rows == gate.Rows)
        {
            for (int i = 0; i < output.Data.Length; i++)
            {
                output.Data[i] *= 1f / (1f + MathF.Exp(-gate.Data[i]));
            }
        }

        Report(layer, "attn_context", null, output, 0);

        var o = TensorOps.MatMulTransposedB(output, _weights.Matrix(attn.OProj, hidden, attn.QDim));
        Report(layer, "attn_output", attn.OProj, o, 0);

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
        if (_patches.Count > 0)
        {
            _patches.Clear();
        }
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

        // Three of every four layers in a Qwen3.5 model are linear-attention,
        // so reporting the mixer as one node left most of the map without an
        // editable tensor to point at. These are the projections a user would
        // actually want to turn down.
        Report(layer, "linear_qkv", op.InProjQkv, mixed, 0);
        Report(layer, "linear_z", op.InProjZ, z, 0);
        Report(layer, "linear_a", op.InProjA, a, 0);
        Report(layer, "linear_b", op.InProjB, b, 0);

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

        var decay = new Tensor2D(g, t, op.NumVHeads);
        var gate = new Tensor2D(beta, t, op.NumVHeads);
        Report(layer, "linear_decay", op.ALog, decay, 0);
        Report(layer, "linear_gate", op.DtBias, gate, 0);

        var core = GatedDeltaKernel.Recurrent(
            q, k, v,
            decay,
            gate,
            op.NumVHeads, op.HeadKDim, op.HeadVDim, state);
        Report(layer, "linear_core", null, core, 0);

        // z-gated RMSNorm over the value head dim, then output projection.
        var normWeight = _weights.Vector(op.NormWeight, op.HeadVDim);
        int rows = t * op.NumVHeads;
        var gated = GatedDeltaKernel.GatedRMSNorm(
            new Tensor2D(core.Data, rows, op.HeadVDim),
            new Tensor2D(z.Data, rows, op.HeadVDim),
            normWeight, op.NormEps);
        Report(layer, "linear_norm", op.NormWeight, gated, 0);
        var out2 = TensorOps.MatMulTransposedB(
            new Tensor2D(gated.Data, t, op.ValueDim),
            _weights.Matrix(op.OutProj, op.HiddenSize, op.ValueDim));
        Report(layer, "linear_out", op.OutProj, out2, 0);
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

    private Tensor2D RunRoutedFfn(Tensor2D h, int layer, RoutedFfnOp routed)
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

        // Only the last row's routing is reported, matching Report's convention
        // — it is the row that decides the next token, and capturing every
        // prefill row would dwarf the rest of the trace.
        int lastRow = h.Rows - 1;
        return FfnKernel.Routed(h, router, gates, ups, downs,
            routed.TopK, routed.RoutingPolicy, routed.Activation,
            onExpertsRouted: ExpertRoutingTrace is null
                ? null
                : (row, selected) =>
                {
                    if (row == lastRow)
                    {
                        ExpertRoutingTrace(layer, selected);
                    }
                });
    }

    // ── conv layer (LFM2.5 short-convolution) ───────────────────────────

    private readonly Dictionary<int, ConvState> _convStates = new();

    private sealed class ConvState
    {
        public readonly float[] Buffer; // [kernel-1, convDim] sliding window
        public int FillCount;
        public ConvState(int kernel, int convDim)
        {
            Buffer = new float[(kernel - 1) * convDim];
            FillCount = 0;
        }
    }

    private ConvState ConvStateFor(int layer, ConvOp op)
    {
        if (!_convStates.TryGetValue(layer, out var state))
        {
            state = new ConvState(op.KernelSize, op.ConvDim);
            _convStates[layer] = state;
        }
        return state;
    }

    private Tensor2D RunConv(Tensor2D h, int layer, ConvOp op)
    {
        int seqLen = h.Rows;
        int hidden = op.HiddenSize;
        int convDim = op.ConvDim;
        int kernel = op.KernelSize;

        // Input projection: hidden → conv_dim
        var inProj = _weights.Matrix(op.InProj, convDim, hidden);
        var projected = TensorOps.MatMulTransposedB(h, inProj); // [seqLen, convDim]

        // Get or create conv state
        var state = ConvStateFor(layer, op);

        // Depthwise causal conv1d with SiLU
        var convWeight = _weights.Matrix(op.ConvWeight, convDim, kernel); // [convDim, kernel]
        float[]? convBias = op.ConvBias is { } biasRef
            ? _weights.Vector(biasRef, convDim)
            : null;

        var convOut = new float[seqLen * convDim];
        for (int t = 0; t < seqLen; t++)
        {
            for (int c = 0; c < convDim; c++)
            {
                float sum = convBias?[c] ?? 0f;
                for (int k = 0; k < kernel; k++)
                {
                    // Causal: only look at positions <= t
                    int pos = t - (kernel - 1 - k);
                    float inputVal;
                    if (pos < 0)
                    {
                        // Use conv state buffer (padding from previous sequence)
                        int stateIdx = (kernel - 1 + pos) * convDim + c;
                        inputVal = stateIdx >= 0 && stateIdx < state.Buffer.Length
                            ? state.Buffer[stateIdx]
                            : 0f;
                    }
                    else if (pos < state.FillCount)
                    {
                        // From state buffer (prefill history)
                        int stateIdx = pos * convDim + c;
                        inputVal = stateIdx < state.Buffer.Length ? state.Buffer[stateIdx] : 0f;
                    }
                    else
                    {
                        // From current batch
                        int batchIdx = (pos - state.FillCount) * convDim + c;
                        inputVal = batchIdx < projected.Data.Length ? projected.Data[batchIdx] : 0f;
                    }
                    sum += inputVal * convWeight.Data[c * kernel + k];
                }
                // SiLU activation
                convOut[t * convDim + c] = sum * Sigmoid(sum);
            }
        }

        // Update conv state with the last (kernel-1) positions
        int newFill = Math.Min(state.FillCount + seqLen, kernel - 1);
        for (int i = 0; i < kernel - 1; i++)
        {
            int srcPos = seqLen - (kernel - 1) + i;
            if (srcPos >= 0)
            {
                Array.Copy(projected.Data, srcPos * convDim, state.Buffer, i * convDim, convDim);
            }
        }
        state.FillCount = newFill;

        // Output projection: conv_dim → hidden
        var convTensor = new Tensor2D(convOut, seqLen, convDim);
        var outProj = _weights.Matrix(op.OutProj, hidden, convDim);
        return TensorOps.MatMulTransposedB(convTensor, outProj);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

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
            var beforeLast = hidden.Row(hidden.Rows - 1);
            hidden = RunLayerInternal(hidden, layer, queryPositions, kvPositions, appendKv: true);
            var afterLast = hidden.Row(hidden.Rows - 1);
            float residualNorm = (float)Math.Sqrt(TensorOps.Dot(afterLast, afterLast));
            float deltaNorm = 0f;
            for (int i = 0; i < afterLast.Length; i++)
            {
                float d = afterLast[i] - beforeLast[i];
                deltaNorm += d * d;
            }
            LayerNormTrace?.Invoke(position, layer, residualNorm, (float)Math.Sqrt(deltaNorm));

            if (LogitLensTrace is { } lens)
            {
                // Clone before projecting — see LogitLensTrace. FinalNormAndHead
                // normalises in place and would otherwise eat the residual that
                // every remaining layer still needs.
                lens(layer, FinalNormAndHead(hidden.Clone()));
            }
        }
        SessionPosition++;
        return hidden;
    }

    /// <summary>The session's absolute token position — advances once per
    /// consumed token, independent of the KV cache shape (mixed plans may
    /// have no key/value rows at layer 0).</summary>
    public int SessionPosition { get; internal set; }

    /// <summary>One attention-weights row: a softmax layer's final query
    /// row, one head's post-softmax attention over its causal key span.</summary>
    public sealed record LayerHeadAttention(int Layer, int Head, float[] Weights);

    /// <summary>When set, softmax layers append their final-row attention
    /// weights here. The relationship router consumes this to name which
    /// (layer, head) tensors carry a token link.</summary>
    public List<LayerHeadAttention>? AttentionTrace { get; set; }

    /// <summary>When set, the pre-FFN (normed) layer input is handed out per
    /// layer as the layer runs — the MoE-ification sampler's activation
    /// seam (see <c>amql-cli moe-ify</c>).</summary>
    public Action<int, Tensor2D>? FfnInputCapture { get; set; }

    /// <summary>When set, every layer's attention-output and FFN-output
    /// L2 norms are reported per step for diagnostic tracing
    /// (<c>--trace</c>).</summary>
    public Action<int, int, float, float>? LayerNormTrace { get; set; }

    /// <summary>When set, each operator reports its output statistics as it
    /// runs — the per-operator seam the inference visualiser draws from, where
    /// <see cref="LayerNormTrace"/> only sees two scalars per layer. Left null
    /// the observation sites cost one null check and nothing else.</summary>
    public Action<OpObservation>? OpTrace { get; set; }

    /// <summary>When set, a routed FFN reports which experts it selected for
    /// the row that decides the next token, as (layer, expert ids). Genuinely
    /// sparse, and the most interpretable thing a MoE layer tells you.</summary>
    public Action<int, IReadOnlyList<int>>? ExpertRoutingTrace { get; set; }

    /// <summary>When set, the logit lens is on: after each layer the residual is
    /// projected through the final norm and output head, and the resulting
    /// logits handed out as (layer, logits). That shows at which depth the
    /// eventual prediction forms, which activation magnitude cannot.
    /// <para>
    /// The projection runs on a CLONE. <see cref="FinalNormAndHead"/> normalises
    /// in place, so projecting the live residual would corrupt every later layer
    /// and quietly change what the model generates — a debugging feature that
    /// alters the thing being debugged is worse than no feature. It also costs a
    /// full head GEMM per layer per step, so it is opt-in and impractical on a
    /// large model.
    /// </para>
    /// </summary>
    public Action<int, Tensor2D>? LogitLensTrace { get; set; }

    /// <summary>A timestamp for an observation about to be made, or 0 when
    /// nothing is listening so the untraced path never queries the clock.</summary>
    private long TraceStart => OpTrace is null ? 0 : Stopwatch.GetTimestamp();

    /// <summary>Measures an operator's output row and reports it. Only the last
    /// row is summarised: under decode that is the whole activation, and under
    /// prefill it is the row that feeds the next token, which is the one the
    /// map is being asked about.</summary>
    private void Report(int layer, string op, OperandRef? weight, Tensor2D output, long started)
    {
        if (OpTrace is not { } sink)
        {
            return;
        }
        var row = output.Row(output.Rows - 1);
        float sumSq = 0f, maxAbs = 0f, sumAbs = 0f;
        for (int i = 0; i < row.Length; i++)
        {
            float v = row[i];
            float a = MathF.Abs(v);
            sumSq += v * v;
            sumAbs += a;
            if (a > maxAbs)
            {
                maxAbs = a;
            }
        }
        double ms = started == 0
            ? 0.0
            : (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        sink(new OpObservation(layer, op, weight, MathF.Sqrt(sumSq), maxAbs,
            row.Length == 0 ? 0f : sumAbs / row.Length, ms));
    }

    /// <summary>Collected per-layer trace from the most recent forward pass.
    /// Cleared at the start of each StepForward; populated by
    /// LayerNormTrace during the layer loop.</summary>
    public readonly List<(int Layer, float R, float D)> Trace = new();

    /// <summary>Enables trace collection for the next forward pass.</summary>
    public void BeginTrace()
    {
        Trace.Clear();
        LayerNormTrace = (_, layer, r, d) => { Trace.Add((layer, r, d)); };
    }

    /// <summary>Stops trace collection.</summary>
    public void EndTrace()
    {
        LayerNormTrace = null;
    }

    /// <summary>
    /// A residual-stream override: after <c>Layer</c> completes (mixer +
    /// FFN + residual), the computed row at absolute position <c>Row</c> is
    /// overwritten with <c>Values</c> (residual-stream width). This is the
    /// activation patch seam — future patch applications and LoRA adapter
    /// injection build on it; the causal tracer uses it today.
    /// </summary>
    public sealed record ResidualPatch(int Layer, int Row, float[] Values);

    private List<ResidualPatch> _patches = new();

    /// <summary>Schedules a residual-stream patch (applied after the layer
    /// completes). Cleared per session reset.</summary>
    public void SetPatch(int layer, int row, float[] values) => _patches.Add(new ResidualPatch(layer, row, values));

    public void ClearPatches() => _patches.Clear();

    /// <summary>Applies any patches scheduled for this layer. The patch row
    /// is an absolute position, resolved against <c>queryPositions</c>
    /// (each x row's absolute position).</summary>
    private void ApplyPatches(Tensor2D x, int layer, int[] queryPositions)
    {
        if (_patches.Count == 0)
        {
            return;
        }
        foreach (var patch in _patches)
        {
            if (patch.Layer != layer)
            {
                continue;
            }
            for (int r = 0; r < x.Rows; r++)
            {
                if (queryPositions[r] == patch.Row)
                {
                    if (patch.Values.Length != x.Cols)
                    {
                        throw new ArgumentException(
                            $"patch at layer {layer} row {patch.Row}: {patch.Values.Length} values vs residual width {x.Cols}");
                    }
                    patch.Values.CopyTo(x.Data, r * x.Cols);
                    break;
                }
            }
        }
    }

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