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
public sealed partial class GenericRuntime
{
    private readonly ComponentOpPlan _plan;
    private readonly WeightLoader _weights;
    private readonly Dictionary<int, LayerKvGeometry> _kvGeometry = new();

    public GenericRuntime(ComponentOpPlan plan, OperandStore store, WeightPatch? patch = null, WeightWorkingSet? workingSet = null)
    {
        _plan = plan;
        _weights = new WeightLoader(store, patch, workingSet);
    }

    /// <summary>A runtime over weights another runtime already loaded, with its own
    /// KV cache and recurrent state. See <see cref="DecodeSession.Sibling"/>.</summary>
    internal GenericRuntime(ComponentOpPlan plan, WeightLoader weights)
    {
        _plan = plan;
        _weights = weights;
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
        // The lean working set reads each token's row instead of the table.
        var hidden = _weights.GatherRows(embed.Table, embed.VocabSize, embed.HiddenSize, tokens)
            ?? TensorOps.GatherRows(_weights.Matrix(embed.Table, embed.VocabSize, embed.HiddenSize), tokens);
        if (embed.Norm is { } norm)
        {
            float[] weight;
            if (embed.NormWeight is { } learned)
            {
                weight = _weights.Vector(learned, hidden.Cols);
            }
            else
            {
                weight = new float[hidden.Cols];
                Array.Fill(weight, 1f);
            }
            Norms.ApplyInPlace(hidden, norm.Kind, norm.Eps, weight, 0f);
        }
        if (embed.Scale is { } scale)
        {
            for (int i = 0; i < hidden.Data.Length; i++)
            {
                hidden.Data[i] = (float)(hidden.Data[i] * scale);
            }
        }
        _passTokens = tokens.ToArray();
        // Qwen4's gated residual carries every stream from the start: the
        // embedding repeated once per stream.
        return _plan.ResidualStreams > 1 ? RepeatStreams(hidden, _plan.ResidualStreams) : hidden;
    }

    // ── Layer loop ──────────────────────────────────────────────────────────

    /// <summary>Runs one layer over the given positions. When
    /// <c>appendKv</c> the freshly computed key/value rows are added to the
    /// cache before attention runs (prefill and decode both append; a
    /// caller replaying a layer never does).</summary>
    /// <para><paramref name="positionStreams"/> gives an MRoPE layer its
    /// per-row (t, h, w) positions; null means every stream is the row's
    /// position, as for text.</para>
    public Tensor2D RunLayerInternal(Tensor2D x, int layer, int[] queryPositions, int[] kvPositions, bool appendKv,
        int[][]? positionStreams = null)
    {
        if (_plan.RunsByVisit)
        {
            throw new UnsupportedOperatorException(
                "this plan runs by visit schedule (a depth loop, a block residual or a row-exact op): step it with StepForward, not layer by layer");
        }
        var layerPlan = _plan.Layers[layer];
        int hidden = _plan.HiddenSize;
        if (layerPlan.MixerHyperConnection is not null)
        {
            return RunHyperLayer(x, layer, layerPlan, queryPositions, kvPositions, appendKv, positionStreams);
        }

        // Pre-attention norm (shared by both operator families).
        Tensor2D h = x;
        if (layerPlan.PreAttentionNorm is { } preAttnNorm)
        {
            h = x.Clone();
            long t = TraceStart;
            ApplyNorm(h, preAttnNorm, hidden);
            Report(layer, "pre_attn_norm", preAttnNorm.Weight, h, t);
        }

        Tensor2D mixerOut = RunMixer(h, layer, layerPlan, queryPositions, kvPositions, appendKv, positionStreams);

        // Post-mixer norm (four-norm / sandwich placement) applies to the
        // mixer output before the residual add, whichever operator ran.
        if (layerPlan.PostAttentionNorm is { } postAttnNorm)
        {
            long tn = TraceStart;
            ApplyNorm(mixerOut, postAttnNorm, hidden);
            Report(layer, "post_attn_norm", postAttnNorm.Weight, mixerOut, tn);
        }
        // A parallel block's FFN reads the attention's own normalised input;
        // the attention branch joins the residual together with the FFN's.
        bool parallel = layerPlan.ParallelResidual;
        if (!parallel)
        {
            AddResidualInPlace(x, mixerOut);
        }

        // Pre-FFN norm.
        Tensor2D hf = parallel ? h : x;
        if (!parallel && layerPlan.PreFfnNorm is { } preFfnNorm)
        {
            hf = x.Clone();
            long t = TraceStart;
            ApplyNorm(hf, preFfnNorm, hidden);
            Report(layer, "pre_ffn_norm", preFfnNorm.Weight, hf, t);
        }

        // FFN.
        if (layerPlan.Ffn is { IsPresent: true } ffn)
        {
            Tensor2D ffnOut = RunFfn(hf, layer, ffn, x);

            if (layerPlan.PostFfnNorm is { } postFfnNorm)
            {
                long tn = TraceStart;
                ApplyNorm(ffnOut, postFfnNorm, hidden);
                Report(layer, "post_ffn_norm", postFfnNorm.Weight, ffnOut, tn);
            }
            if (parallel)
            {
                AddInPlace(mixerOut, ffnOut);
                AddResidualInPlace(x, mixerOut);
            }
            else
            {
                AddResidualInPlace(x, ffnOut);
            }
        }
        else if (parallel)
        {
            AddResidualInPlace(x, mixerOut);
        }

        if (layerPlan.OutputScale is { } outputScale)
        {
            float s = _weights.Vector(outputScale, 1)[0];
            for (int i = 0; i < x.Data.Length; i++)
            {
                x.Data[i] *= s;
            }
        }
        ApplyPatches(x, layer, queryPositions);
        // started 0 means "no timing" — the residual is the layer's result, not
        // an operation with a duration of its own.
        Report(layer, "residual_out", null, x, 0);
        return x;
    }

    /// <summary>The token mixer the layer carries: linear attention, a short
    /// conv or softmax attention.</summary>
    private Tensor2D RunMixer(Tensor2D h, int layer, LayerPlan layerPlan, int[] queryPositions, int[] kvPositions, bool appendKv,
        int[][]? positionStreams)
    {
        Tensor2D mixerOut;
        if (layerPlan.LinearAttention is { } linear)
        {
            long t = TraceStart;
            mixerOut = RunLinearAttention(h, layer, linear);
            Report(layer, "linear_attn", null, mixerOut, t);
        }
        else if (layerPlan.Conv is { } conv)
        {
            long t = TraceStart;
            mixerOut = RunConv(h, layer, conv);
            Report(layer, "conv", null, mixerOut, t);
        }
        else
        {
            long t = TraceStart;
            mixerOut = RunSoftmaxAttention(h, layer, layerPlan, queryPositions, kvPositions, appendKv, positionStreams);
            Report(layer, "softmax_attn", null, mixerOut, t);
        }
        return mixerOut;
    }

    /// <summary>The layer's FFN over its (normalised) input, with any adapter's
    /// contribution added.</summary>
    /// <param name="residual">The residual the FFN's norm read: what experts beside
    /// a dense FFN (<see cref="HybridFfn"/>) read themselves.</param>
    private Tensor2D RunFfn(Tensor2D hf, int layer, LayerFfn ffn, Tensor2D? residual = null)
    {
        int hidden = _plan.HiddenSize;
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
            ffnOut = FfnKernel.Dense(hf, gate, up, down, dense.Activation, dense.IsGated,
                OptionalVector(dense.GateBias, dense.IntermediateSize),
                OptionalVector(dense.UpBias, dense.IntermediateSize),
                OptionalVector(dense.DownBias, hidden));
            Report(layer, "ffn_dense", dense.Down, ffnOut, t);
        }
        else
        {
            var routed = ffn.Routed!;
            ffnOut = routed.Hybrid is { } hybrid
                ? RunHybridFfn(hf, residual ?? throw new InvalidOperationException("experts beside a dense FFN read the residual"), layer, routed, hybrid)
                : RunRoutedFfn(hf, layer, routed);
            Report(layer, "ffn_routed", routed.Router, ffnOut, t);
        }
        if (FfnAdapter?.Invoke(layer, hf) is { } extra)
        {
            AddInPlace(ffnOut, extra);
        }
        return ffnOut;
    }

    /// <summary>Applies a norm op in place: the normalisation, the learned
    /// weight (with its affine offset) and, for a LayerNorm that ships one, the
    /// additive bias.</summary>
    private void ApplyNorm(Tensor2D x, NormOp norm, int width)
    {
        var w = _weights.Vector(norm.Weight, width);
        Norms.ApplyInPlace(x, norm.Kind, norm.Eps, w, norm.WeightOffset);
        FfnKernel.AddRowBias(x, OptionalVector(norm.Bias, width));
    }

    private float[]? OptionalVector(OperandRef? operand, int length) =>
        operand is null ? null : _weights.Vector(operand, length);

    /// <summary>Granite's decoder join is h' = h + branch·residual_multiplier
    /// at BOTH the mixer and the FFN add (transformers' GraniteDecoderLayer:
    /// <c>residual + hidden_states * residual_multiplier</c>); families without
    /// the knob carry scale 1.0 and this is the plain add.</summary>
    private void AddResidualInPlace(Tensor2D x, Tensor2D branch)
    {
        double scale = _plan.ResidualScale;
        if (scale != 1.0)
        {
            for (int i = 0; i < branch.Data.Length; i++)
            {
                branch.Data[i] = (float)(branch.Data[i] * scale);
            }
        }
        AddInPlace(x, branch);
    }

    // ── softmax attention mixer ─────────────────────────────────────────────

    private List<(int Head, float[] Weights)>? _lastCapture;

    private Tensor2D RunSoftmaxAttention(Tensor2D h, int layer, LayerPlan layerPlan, int[] queryPositions, int[] kvPositions, bool appendKv,
        int[][]? positionStreams = null)
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
            var batch = TensorOps.MatMulTransposedBMany(h, wQ, wK, wV);
            qRaw = batch[0];
            k = batch[1];
            v = batch[2];
        }

        // Sub-operator observations carry no duration: on the GPU path q, k and
        // v are a single batched launch, so a per-projection time would be
        // fiction. Timing lives on the layer-level operators instead.
        FfnKernel.AddRowBias(qRaw, OptionalVector(attn.QBias, attn.QProjWidth));
        FfnKernel.AddRowBias(k, OptionalVector(attn.KBias, attn.KvDim));
        FfnKernel.AddRowBias(v, OptionalVector(attn.VBias, attn.KvDim));
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
        Rope.Apply(q, attn.NumQHeads, attn.HeadDim, attn.Position, queryPositions, attn.RopeInterleaved, positionStreams);

        int kvRows = k.Rows;
        var ropeKvPositions = kvRows == kvPositions.Length
            ? kvPositions
            : kvPositions[^kvRows..];
        Rope.Apply(k, attn.NumKvHeads, attn.HeadDim, attn.Position, ropeKvPositions, attn.RopeInterleaved, positionStreams);

        if (attn.VFromK)
        {
            v = k.Clone();
        }

        if (appendKv)
        {
            AppendRows(layer, k, v);
        }

        // The cached rows in place (contiguous per layer, read without a copy).
        var (kBuffer, kWidth, vBuffer, vWidth, kvSeq) = Kv.Buffers(layer);
        if (kvSeq == 0)
        {
            kWidth = attn.KvDim;
            vWidth = attn.KvDim;
        }

        _lastCapture = AttentionTrace is not null
            ? new List<(int Head, float[] Weights)>(attn.NumQHeads)
            : null;

        Tensor2D output;
        // QSA: the indexer picks each query's keys. Null means every row keeps
        // every visible key (no more complete blocks than its budget), and the
        // attention below is exactly the dense one.
        if (attn.Indexer is { } indexer &&
            SelectSparseKeys(h, layer, indexer, queryPositions, kvPositions, appendKv, kvSeq, positionStreams) is { } selection)
        {
            output = SparseAttention(q, kBuffer, kWidth, vBuffer, vWidth, attn, selection, queryPositions);
            goto attentionDone;
        }
        // GPU attention path — fused QK^T + causal softmax + PV on device.
        // Only for non-trivial prefills (≥ 8 query tokens) when the weights
        // are already on the device.  Single-token decode is too small for
        // the GPU launch overhead.
        if (CudaShim.Enabled && _lastCapture is null &&
            q.Rows >= 8 &&
            _weights.AnyDeviceWeight)
        {
            var softCap = attn.LogitSoftcapping ?? 0f;
            var window = attn.Window is { } wv ? checked((int)wv) : 0;
            var gpuOut = new float[q.Rows * attn.QDim];
            // The device path takes exact [kvSeq × width] arrays.
            var kExact = kBuffer.AsSpan(0, kvSeq * kWidth).ToArray();
            var vExact = vBuffer.AsSpan(0, kvSeq * vWidth).ToArray();
            if (CudaShim.TryAttentionGqa(
                    q.Data, q.Rows,
                    kExact, kvSeq,
                    vExact,
                    gpuOut,
                    attn.NumQHeads, attn.NumKvHeads, attn.HeadDim,
                    (float)attn.ScoreScale, softCap, window,
                    null,
                    attn.Causal ? queryPositions : MaskPositionsSeeingAll(q.Rows, kvPositions),
                    kvPositions))
            {
                output = new Tensor2D(gpuOut, q.Rows, attn.QDim);
                goto attentionDone;
            }
        }
        output = AttentionKernel.Execute(
            q, kBuffer, kWidth, vBuffer, vWidth, kvSeq,
            attn.NumQHeads, attn.NumKvHeads, attn.HeadDim,
            attn.ScoreScale, attn.LogitSoftcapping, null,
            attn.Window is { } windowValue ? checked((int)windowValue) : null,
            queryPositions, kvPositions,
            _lastCapture,
            attn.Causal);

        attentionDone:

        AttentionContextCapture?.Invoke(layer, output);

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
            var outData = output.Data;
            var gateData = gate.Data;
            int cols = output.Cols;
            void GateRow(int r)
            {
                for (int i = r * cols; i < (r + 1) * cols; i++)
                {
                    outData[i] *= 1f / (1f + MathF.Exp(-gateData[i]));
                }
            }
            if (output.Rows >= 8)
            {
                KernelPool.For(output.Rows, GateRow);
            }
            else
            {
                for (int r = 0; r < output.Rows; r++)
                {
                    GateRow(r);
                }
            }
        }

        Report(layer, "attn_context", null, output, 0);

        // Per-head output gate (LFM3): each head's context × sigmoid(gate_proj·h).
        if (attn.HeadGate is { } headGateRef)
        {
            var headGate = TensorOps.MatMulTransposedB(h, _weights.Matrix(headGateRef, attn.NumQHeads, hidden));
            Report(layer, "attn_head_gate", headGateRef, headGate, 0);
            for (int r = 0; r < output.Rows; r++)
            {
                for (int head = 0; head < attn.NumQHeads; head++)
                {
                    float g = Sigmoid(headGate.Data[r * attn.NumQHeads + head]);
                    int start = r * attn.QDim + head * attn.HeadDim;
                    for (int d = 0; d < attn.HeadDim; d++)
                    {
                        output.Data[start + d] *= g;
                    }
                }
            }
        }

        var o = TensorOps.MatMulTransposedB(output, _weights.Matrix(attn.OProj, hidden, attn.QDim));
        FfnKernel.AddRowBias(o, OptionalVector(attn.OBias, hidden));
        Report(layer, "attn_output", attn.OProj, o, 0);
        return o;
    }

    private void ApplyWeightedQkNorm(Tensor2D q, Tensor2D k, AttentionOp attn, NormOp? qNorm, NormOp? kNorm, int hidden)
    {
        if (qNorm is { } qn)
        {
            var w = _weights.Vector(qn.Weight, qn.Width);
            if (qn.Width == attn.HeadDim)
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
            if (kn.Width == attn.HeadDim)
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
        var data = x.Data;
        int cols = x.Cols;
        void Row(int r)
        {
            var row = data.AsSpan(r * cols, cols);
            for (int h = 0; h < heads; h++)
            {
                Norms.ApplyRow(row.Slice(h * headDim, headDim), NormType.RmsNorm, eps, weight, weightOffset);
            }
        }
        // Rows (and heads within them) normalise independently.
        if (x.Rows >= 8)
        {
            KernelPool.For(x.Rows, Row);
            return;
        }
        for (int r = 0; r < x.Rows; r++)
        {
            Row(r);
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
        var result = GC.AllocateUninitializedArray<float>(source.Rows * width);
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
        _convStates.Clear();
        ResetQwen4State();
        ResetVisitState();
        SessionPosition = 0;
        if (_patches.Count > 0)
        {
            _patches.Clear();
        }
    }

    /// <summary>Clears the KV cache and resets session state — reuse the
    /// same runtime for a new prompt without re-loading weights.</summary>
    public void ResetKvCache()
    {
        Kv.Reset();
        ResetSession();
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

        // The four in-projections share h: one fused pass over their rows.
        var projected = TensorOps.MatMulTransposedBMany(h,
            _weights.Matrix(op.InProjQkv, op.ConvDim, op.HiddenSize),
            _weights.Matrix(op.InProjZ, op.ValueDim, op.HiddenSize),
            _weights.Matrix(op.InProjB, op.NumVHeads, op.HiddenSize),
            _weights.Matrix(op.InProjA, op.NumVHeads, op.HiddenSize));
        var mixed = projected[0];
        var z = projected[1];
        var b = projected[2];
        var a = projected[3];

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
        var convWeight = _weights.Vector(op.Conv1d, op.ConvDim * op.ConvKernel);
        Tensor2D q, k, v;
        if (t == 1)
        {
            // One position: the conv step reads the projected row directly and
            // the q / k / v splits are contiguous ranges of its output — the
            // same values the column-major route below produces.
            var walkedRow = GatedDeltaKernel.ConvStepFlat(mixed.Data, convWeight, op.ConvKernel, silu: true, state);
            q = new Tensor2D(walkedRow.AsSpan(0, op.KeyDim).ToArray(), 1, op.KeyDim);
            k = new Tensor2D(walkedRow.AsSpan(op.KeyDim, op.KeyDim).ToArray(), 1, op.KeyDim);
            v = new Tensor2D(walkedRow.AsSpan(2 * op.KeyDim, op.ValueDim).ToArray(), 1, op.ValueDim);
        }
        else
        {
            // Row-major throughout: the conv reads the projection as laid out
            // and the q / k / v splits are column ranges of its output.
            var walked = new Tensor2D(
                GatedDeltaKernel.ConvForwardRows(mixed.Data, t, convWeight, op.ConvKernel, silu: true, state),
                t, op.ConvDim);

            // Split q / k / v (the reference splits [key, key, value]) and
            // reshape into heads.
            q = ColRange(walked, 0, op.KeyDim);
            k = ColRange(walked, op.KeyDim, op.KeyDim);
            v = ColRange(walked, 2 * op.KeyDim, op.ValueDim);
        }
        // Expand key heads to the value-head count when the ratio > 1
        // (repeat_interleave).
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
            normWeight, op.NormEps, op.SigmoidGate);
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

    private Tensor2D RunHybridFfn(Tensor2D normed, Tensor2D residual, int layer, RoutedFfnOp routed, HybridFfn hybrid)
    {
        int hidden = routed.HiddenSize;
        var d = hybrid.Dense;
        var dense = FfnKernel.Dense(normed,
            d.Gate is null ? null : _weights.Matrix(d.Gate, d.IntermediateSize, hidden),
            _weights.Matrix(d.Up, d.IntermediateSize, hidden),
            _weights.Matrix(d.Down, hidden, d.IntermediateSize), d.Activation, d.IsGated,
            OptionalVector(d.GateBias, d.IntermediateSize), OptionalVector(d.UpBias, d.IntermediateSize), OptionalVector(d.DownBias, hidden));
        if (hybrid.DenseOut is { } denseOut)
        {
            ApplyNorm(dense, denseOut, hidden);
        }
        var routedIn = residual.Clone();
        if (hybrid.RoutedIn is { } inNorm)
        {
            ApplyNorm(routedIn, inNorm, hidden);
        }
        var experts = RunRoutedFfn(routedIn, layer, routed, residual);
        if (hybrid.RoutedOut is { } outNorm)
        {
            ApplyNorm(experts, outNorm, hidden);
        }
        AddInPlace(dense, experts);
        return dense;
    }

    /// <summary>What the router scores: its input, normalised and scaled where the
    /// layer says so.</summary>
    private Tensor2D RouterInput(Tensor2D h, RoutedFfnOp routed)
    {
        if (routed.RouterInputNormEps is null && routed.RouterScale is null)
        {
            return h;
        }
        var input = h.Clone();
        int hidden = routed.HiddenSize;
        if (routed.RouterInputNormEps is { } normEps)
        {
            var ones = new float[hidden];
            Array.Fill(ones, 1f);
            Norms.ApplyInPlace(input, NormType.RmsNorm, normEps, ones, 0f);
        }
        if (routed.RouterScale is { } scaleOperand)
        {
            var scale = _weights.Vector(scaleOperand, hidden);
            float root = (float)(1.0 / Math.Sqrt(hidden));
            for (int r = 0; r < input.Rows; r++)
            {
                var row = input.Row(r);
                for (int c = 0; c < row.Length; c++)
                {
                    row[c] *= scale[c] * root;
                }
            }
        }
        return input;
    }

    /// <param name="routeBy">What the router reads, when it is not what the experts read.</param>
    private Tensor2D RunRoutedFfn(Tensor2D h, int layer, RoutedFfnOp routed, Tensor2D? routeBy = null)
    {
        int hidden = routed.HiddenSize;
        var router = _weights.Matrix(routed.Router, routed.NumExperts, hidden);
        _weights.BeginExpertLayer();

        // Gate/up/down are resolved through the operand store by explicit
        // per-expert names — the prefix fields disambiguate spellings — and
        // only for the experts the router picks.
        string stackId = routed.Router.ObjectId;
        (Tensor2D, Tensor2D, Tensor2D) Expert(int e) => (
            _weights.ExpertMatrix(new OperandRef(stackId, $"{routed.ExpertGatePrefix}{e}.{routed.ExpertGateStem}"), routed.ExpertIntermediateSize, hidden),
            _weights.ExpertMatrix(new OperandRef(stackId, $"{routed.ExpertUpPrefix}{e}.{routed.ExpertUpStem}"), routed.ExpertIntermediateSize, hidden),
            _weights.ExpertMatrix(new OperandRef(stackId, $"{routed.ExpertDownPrefix}{e}.{routed.ExpertDownStem}"), hidden, routed.ExpertIntermediateSize));

        // Only the last row's routing is reported, matching Report's convention
        // — it is the row that decides the next token, and capturing every
        // prefill row would dwarf the rest of the trace.
        int lastRow = h.Rows - 1;
        var routerInput = RouterInput(routeBy ?? h, routed);
        var mixture = FfnKernel.Routed(h, router, routed.NumExperts, hidden, Expert,
            routed.TopK, routed.RoutingPolicy, routed.Activation,
            onExpertsRouted: ExpertRoutingTrace is null
                ? null
                : (row, selected) =>
                {
                    if (row == lastRow)
                    {
                        ExpertRoutingTrace(layer, selected);
                    }
                },
            release: w =>
            {
                _weights.Release(w.Gate);
                _weights.Release(w.Up);
                _weights.Release(w.Down);
            },
            routerInput: ReferenceEquals(routerInput, h) ? null : routerInput,
            expertScale: OptionalVector(routed.ExpertScale, routed.NumExperts));
        if (routed.SharedExpert is { } shared)
        {
            // Every token also passes through the shared expert, scaled by
            // sigmoid(gate·x) where the layer carries that gate.
            var sharedOut = FfnKernel.Dense(h,
                shared.Gate is null ? null : _weights.Matrix(shared.Gate, shared.IntermediateSize, hidden),
                _weights.Matrix(shared.Up, shared.IntermediateSize, hidden),
                _weights.Matrix(shared.Down, hidden, shared.IntermediateSize),
                shared.Activation, shared.IsGated);
            float[]? gate = routed.SharedExpertGate is { } g ? _weights.Vector(g, hidden) : null;
            for (int r = 0; r < h.Rows; r++)
            {
                float scale = 1f;
                if (gate is not null)
                {
                    double dot = 0;
                    var row = h.Row(r);
                    for (int c = 0; c < hidden; c++)
                    {
                        dot += row[c] * gate[c];
                    }
                    scale = (float)(1 / (1 + Math.Exp(-dot)));
                }
                var outRow = mixture.Data.AsSpan(r * hidden, hidden);
                var add = sharedOut.Row(r);
                for (int c = 0; c < hidden; c++)
                {
                    outRow[c] += scale * add[c];
                }
            }
        }
        return mixture;
    }

    // ── conv layer (LFM2/LFM3 gated short convolution) ──────────────────

    /// <summary>Per conv layer: the last (kernel-1) rows of B⊙x, oldest
    /// first, zero before the session's first token.</summary>
    private readonly Dictionary<int, float[]> _convStates = new();

    private float[] ConvStateFor(int layer, ConvOp op)
    {
        if (!_convStates.TryGetValue(layer, out var state))
        {
            state = new float[(op.KernelSize - 1) * op.ConvDim];
            _convStates[layer] = state;
        }
        return state;
    }

    private Tensor2D RunConv(Tensor2D h, int layer, ConvOp op)
    {
        int seqLen = h.Rows;
        int hidden = op.HiddenSize;
        int d = op.ConvDim;
        int kernel = op.KernelSize;

        // in_proj → (B, C, x), each d wide, in that order along the row.
        var bcx = TensorOps.MatMulTransposedB(h, _weights.Matrix(op.InProj, 3 * d, hidden));
        Report(layer, "conv_in", op.InProj, bcx, 0);

        var bx = new float[seqLen * d];
        for (int r = 0; r < seqLen; r++)
        {
            int row = r * 3 * d;
            for (int c = 0; c < d; c++)
            {
                bx[r * d + c] = bcx.Data[row + c] * bcx.Data[row + 2 * d + c];
            }
        }

        // Depthwise causal conv over B⊙x, no activation: out[t] =
        // Σ_j w[c, j]·in[t - (K-1) + j], positions before the batch read the
        // carried history (row i of the history is position i - (K-1)).
        var history = ConvStateFor(layer, op);
        var w = _weights.Vector(op.ConvWeight, d * kernel);
        float[]? bias = op.ConvBias is { } biasRef ? _weights.Vector(biasRef, d) : null;
        var y = new float[seqLen * d];
        Parallel.For(0, seqLen, r =>
        {
            int row = r * 3 * d;
            for (int c = 0; c < d; c++)
            {
                float sum = bias?[c] ?? 0f;
                for (int j = 0; j < kernel; j++)
                {
                    int pos = r - (kernel - 1) + j;
                    float v = pos >= 0 ? bx[pos * d + c] : history[(pos + kernel - 1) * d + c];
                    sum += w[c * kernel + j] * v;
                }
                // Output gate C.
                y[r * d + c] = bcx.Data[row + d + c] * sum;
            }
        });

        // Advance the history to the last (K-1) rows of [history; B⊙x].
        int keep = kernel - 1;
        var next = new float[keep * d];
        for (int i = 0; i < keep; i++)
        {
            int pos = seqLen - keep + i; // position relative to this batch
            if (pos >= 0)
            {
                Array.Copy(bx, pos * d, next, i * d, d);
            }
            else
            {
                Array.Copy(history, (pos + keep) * d, next, i * d, d);
            }
        }
        _convStates[layer] = next;

        var gated = new Tensor2D(y, seqLen, d);
        Report(layer, "conv_gated", op.ConvWeight, gated, 0);
        var outProj = _weights.Matrix(op.OutProj, hidden, d);
        var result = TensorOps.MatMulTransposedB(gated, outProj);
        Report(layer, "conv_out", op.OutProj, result, 0);
        return result;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>The device kernel takes positions for its causal mask only (rope
    /// is already applied): a query placed at the last key position masks
    /// nothing, which is bidirectional attention without a second kernel. The
    /// planner refuses a window over a bidirectional stack, so the window test
    /// these positions also feed never fires.</summary>
    private static int[] MaskPositionsSeeingAll(int queries, int[] kvPositions)
    {
        var positions = new int[queries];
        Array.Fill(positions, kvPositions.Length == 0 ? 0 : kvPositions.Max());
        return positions;
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
        if (_plan.Diffusion is not null)
        {
            // A cached prefix is only valid when later tokens cannot change
            // earlier rows; in a bidirectional stack every new token does.
            throw new UnsupportedOperatorException(
                "step decoding is causal-only — a masked diffusion model generates with MaskedDiffusionSampler");
        }
        if (_plan.RunsByVisit)
        {
            return StepVisits(token);
        }
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

    /// <summary>When set, called with each layer's FFN input (normed, rows × hidden); a
    /// result is added to that layer's FFN output before any post-FFN norm. A LoRA trained
    /// here runs through it as its own low-rank path in f32, so the base weights, and any
    /// device copies of them, never change.</summary>
    public Func<int, Tensor2D, Tensor2D?>? FfnAdapter { get; set; }

    /// <summary>When set, each softmax-attention layer's per-head context
    /// output (rows × QDim, contiguous per-head <c>HeadDim</c> blocks, before
    /// the output gate and <c>o_proj</c>) is handed out as the layer runs —
    /// the Star-Elastic head-pruning importance seam.</summary>
    public Action<int, Tensor2D>? AttentionContextCapture { get; set; }

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

    /// <summary>Whether anything watches or alters the layer loop — a trace, a
    /// capture or a residual patch. The device engine runs the whole pass on
    /// the GPU and has none of these seams, so a session with any of them runs
    /// here instead.</summary>
    internal bool HasObservers =>
        OpTrace is not null || AttentionTrace is not null || FfnInputCapture is not null || FfnAdapter is not null ||
        AttentionContextCapture is not null || LayerNormTrace is not null || LogitLensTrace is not null ||
        ExpertRoutingTrace is not null || _patches.Count > 0;

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

    /// <summary>The final norm alone, in place — the hidden state a pooled
    /// head (a classifier's score linear) reads.</summary>
    public Tensor2D FinalNorm(Tensor2D hidden)
    {
        if (_plan.FinalMixer is { } mixer)
        {
            // Qwen4: the streams are read into one hidden state, not normalised in place.
            return HyperRead(hidden, mixer).Read;
        }
        if (_plan.RunsByVisit)
        {
            return RowFinalNorm(hidden);
        }
        ApplyNorm(hidden, _plan.FinalNorm!, hidden.Cols);
        return hidden;
    }

    public Tensor2D FinalNormAndHead(Tensor2D hidden)
    {
        hidden = FinalNorm(hidden);

        if (_plan.Output is not { } output)
        {
            throw new UnsupportedOperatorException("plan carries no output head");
        }
        if (_plan.RunsByVisit)
        {
            return RowHead(hidden);
        }
        // A head the lean working set cannot hold is used a block of rows at a time.
        var logits = _weights.StreamedProduct(hidden, output.Projection, output.StoredRows, output.HiddenSize)
            ?? TensorOps.MatMulTransposedB(hidden, _weights.Matrix(output.Projection, output.StoredRows, output.HiddenSize));
        if (output.TableRows is not null)
        {
            // A padded vocabulary: the rows past it are never scored.
            var scored = new float[hidden.Rows * output.VocabSize];
            for (int r = 0; r < hidden.Rows; r++)
            {
                Array.Copy(logits.Data, r * logits.Cols, scored, r * output.VocabSize, output.VocabSize);
            }
            logits = new Tensor2D(scored, hidden.Rows, output.VocabSize);
        }
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