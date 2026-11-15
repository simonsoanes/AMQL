using System.Runtime.InteropServices;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The device-resident decode engine: the whole forward pass of a planned
/// component on the GPU. Weights are uploaded once (BF16 as stored, f32 where
/// the container or a patch makes them f32), and activations, the KV cache and
/// the recurrent state never leave the device. A decode step is captured
/// once into a CUDA graph and replayed, so a token costs one graph launch plus
/// the token id going up and the logits coming back.
/// <para>
/// This is what <see cref="DecodeSession"/> runs when CUDA is available and
/// the plan is within reach (see <see cref="Unsupported"/>): dense FFNs,
/// softmax attention (GQA, output gate, weighted per-head QK norm, rotary,
/// sliding window, softcap), Gated DeltaNet linear attention, RMS/LayerNorm
/// in any placement, parallel residuals and residual scaling. Everything
/// else — routed experts, short convs, traces, captures, residual patches —
/// stays on the managed path.
/// </para>
/// <para>
/// Numerics: f32 activations and accumulation, the container's weights
/// exactly. The order of floating-point reductions differs from the managed
/// kernels, so results agree to rounding rather than bit for bit.
/// </para>
/// </summary>
public sealed class GpuEngine : IDisposable
{
    /// <summary>Rows a prefill pushes through the layers at once; longer
    /// prompts run in chunks (the recurrent state and KV cache carry over).</summary>
    public const int ChunkRows = 512;

    private const int BF16 = 0, F32 = 1;

    private readonly ComponentOpPlan _plan;
    private readonly IntPtr _engine;
    private readonly List<IntPtr> _allocations = new();
    private readonly Dictionary<(string, string), DeviceMatrix> _matrixCache = new();
    private readonly Dictionary<(string, string), IntPtr> _vectorCache = new();
    private readonly LayerDevice[] _layers;
    private readonly int _hidden;
    private readonly int _vocab;

    private readonly DeviceMatrix _embed;
    private readonly float _embedScale;
    private readonly bool _hasEmbedScale;
    private readonly DeviceMatrix _head;
    private readonly IntPtr _finalNormW, _finalNormB;
    private readonly float _finalNormOff, _finalNormEps;
    private readonly int _finalNormKind;

    // Activations, [ChunkRows × width] each.
    private readonly IntPtr _pos, _tokens, _x, _hn, _proj, _q, _gate, _ctx, _conv, _gdn, _qk, _gb, _ffn, _branch, _branch2, _raw, _logits;
    private IntPtr _attnPart;
    private int _attnSplits;
    private readonly int _maxHeads, _maxHeadDim;
    private readonly IntPtr _hostToken, _hostLogits;

    private int _kvCapacity;
    private int _position;
    private bool _graphReady;
    private bool _disposed;

    /// <summary>The <see cref="WeightLoader.Version"/> the uploaded weights
    /// were read at.</summary>
    internal int WeightsVersion { get; }

    public int Position => _position;

    private static readonly bool Timing = Environment.GetEnvironmentVariable("AMQL_GPU_ENGINE_TIMING") is "1";
    private long _launchTicks, _syncTicks, _copyTicks, _timedSteps;

    private GpuEngine(ComponentOpPlan plan, WeightLoader weights)
    {
        _plan = plan;
        _hidden = plan.HiddenSize;
        Check(Native.amqe_create(out _engine), "create");
        WeightsVersion = weights.Version;
        try
        {
            var embedOp = plan.Embedding!;
            _vocab = plan.Output!.VocabSize;
            _embed = Matrix(weights, embedOp.Table, embedOp.VocabSize, embedOp.HiddenSize);
            _hasEmbedScale = embedOp.Scale is not null;
            _embedScale = (float)(embedOp.Scale ?? 1.0);
            _head = Matrix(weights, plan.Output.Projection, plan.Output.VocabSize, plan.Output.HiddenSize);
            (_finalNormW, _finalNormB, _finalNormOff, _finalNormEps, _finalNormKind) = Norm(weights, plan.FinalNorm);

            _layers = new LayerDevice[plan.Layers.Count];
            int projWidth = _hidden, qWidth = 1, ctxWidth = 1, convWidth = 1, gdnWidth = 1, ffnWidth = 1, keyWidth = 1, vHeads = 1;
            for (int i = 0; i < plan.Layers.Count; i++)
            {
                var lp = plan.Layers[i];
                var ld = new LayerDevice { Plan = lp };
                ld.PreAttn = lp.PreAttentionNorm is { } n1 ? Norm(weights, n1) : null;
                ld.PostAttn = lp.PostAttentionNorm is { } n2 ? Norm(weights, n2) : null;
                ld.PreFfn = lp.PreFfnNorm is { } n3 ? Norm(weights, n3) : null;
                ld.PostFfn = lp.PostFfnNorm is { } n4 ? Norm(weights, n4) : null;

                if (lp.Attention is { } attn)
                {
                    ld.Qkv = Concat(weights, _hidden,
                        (attn.QProj, attn.QProjWidth), (attn.KProj, attn.KvDim), (attn.VProj, attn.KvDim));
                    ld.QkvBias = ConcatBias(weights,
                        (attn.QBias, attn.QProjWidth), (attn.KBias, attn.KvDim), (attn.VBias, attn.KvDim));
                    ld.O = Matrix(weights, attn.OProj, _hidden, attn.QDim);
                    ld.OBias = OptionalVector(weights, attn.OBias, _hidden);
                    if (attn.QNorm is { } qn)
                    {
                        ld.QNormW = Vector(weights, qn.Weight, qn.Width);
                        ld.QNormOff = qn.WeightOffset;
                        ld.QNormEps = (float)qn.Eps;
                        ld.QNormRow = qn.Width != attn.HeadDim;
                    }
                    if (attn.KNorm is { } kn)
                    {
                        ld.KNormW = Vector(weights, kn.Weight, kn.Width);
                        ld.KNormOff = kn.WeightOffset;
                        ld.KNormEps = (float)kn.Eps;
                        ld.KNormRow = kn.Width != attn.HeadDim;
                    }
                    var rope = RopeFor(attn);
                    if (rope is { } r)
                    {
                        ld.InvFreq = UploadFloats(r.InvFreq.Select(f => (float)f).ToArray());
                        ld.RopePairs = r.InvFreq.Length;
                        ld.RopeScale = r.Scale;
                    }
                    projWidth = Math.Max(projWidth, ld.Qkv.Rows);
                    qWidth = Math.Max(qWidth, attn.QDim);
                    _maxHeads = Math.Max(_maxHeads, attn.NumQHeads);
                    _maxHeadDim = Math.Max(_maxHeadDim, attn.HeadDim);
                    ctxWidth = Math.Max(ctxWidth, attn.QDim);
                }
                else if (lp.LinearAttention is { } lin)
                {
                    ld.In = Concat(weights, _hidden,
                        (lin.InProjQkv, lin.ConvDim), (lin.InProjZ, lin.ValueDim),
                        (lin.InProjB, lin.NumVHeads), (lin.InProjA, lin.NumVHeads));
                    ld.ConvW = Vector(weights, lin.Conv1d, lin.ConvDim * lin.ConvKernel);
                    ld.ALog = Vector(weights, lin.ALog, lin.NumVHeads);
                    ld.DtBias = Vector(weights, lin.DtBias, lin.NumVHeads);
                    ld.GdnNormW = Vector(weights, lin.NormWeight, lin.HeadVDim);
                    ld.Out = Matrix(weights, lin.OutProj, _hidden, lin.ValueDim);
                    ld.State = Alloc((long)lin.NumVHeads * lin.HeadKDim * lin.HeadVDim * 4);
                    ld.StateBytes = (long)lin.NumVHeads * lin.HeadKDim * lin.HeadVDim * 4;
                    ld.ConvHistory = Alloc((long)lin.ConvDim * (lin.ConvKernel - 1) * 4);
                    ld.ConvHistoryBytes = (long)lin.ConvDim * (lin.ConvKernel - 1) * 4;
                    projWidth = Math.Max(projWidth, ld.In.Rows);
                    convWidth = Math.Max(convWidth, lin.ConvDim);
                    gdnWidth = Math.Max(gdnWidth, lin.ValueDim);
                    keyWidth = Math.Max(keyWidth, lin.KeyDim);
                    vHeads = Math.Max(vHeads, lin.NumVHeads);
                }

                if (lp.Ffn?.Dense is { } dense)
                {
                    if (dense.IsGated)
                    {
                        ld.Up = Concat(weights, _hidden, (dense.Gate!, dense.IntermediateSize), (dense.Up, dense.IntermediateSize));
                        ld.UpBias = ConcatBias(weights, (dense.GateBias, dense.IntermediateSize), (dense.UpBias, dense.IntermediateSize));
                    }
                    else
                    {
                        ld.Up = Matrix(weights, dense.Up, dense.IntermediateSize, _hidden);
                        ld.UpBias = OptionalVector(weights, dense.UpBias, dense.IntermediateSize);
                    }
                    ld.Down = Matrix(weights, dense.Down, _hidden, dense.IntermediateSize);
                    ld.DownBias = OptionalVector(weights, dense.DownBias, _hidden);
                    ld.Act = dense.Activation switch
                    {
                        Activation.Silu => 1,
                        Activation.Gelu or Activation.GeluTanh => 2,
                        Activation.Relu => 3,
                        _ => throw new UnsupportedOperatorException($"activation {dense.Activation}"),
                    };
                    projWidth = Math.Max(projWidth, ld.Up.Rows);
                    ffnWidth = Math.Max(ffnWidth, dense.IntermediateSize);
                }
                _layers[i] = ld;
            }

            long rows = ChunkRows;
            _pos = Alloc(16);
            _tokens = Alloc(rows * 4);
            _x = Alloc(rows * _hidden * 4);
            _hn = Alloc(rows * _hidden * 4);
            _proj = Alloc(rows * projWidth * 4);
            _q = Alloc(rows * qWidth * 4);
            _gate = Alloc(rows * qWidth * 4);
            _ctx = Alloc(rows * ctxWidth * 4);
            _conv = Alloc(rows * convWidth * 4);
            _gdn = Alloc(rows * gdnWidth * 4);
            _qk = Alloc(rows * 2 * keyWidth * 4);
            _gb = Alloc(rows * 2 * vHeads * 4);
            _ffn = Alloc(rows * ffnWidth * 4);
            _branch = Alloc(rows * _hidden * 4);
            _branch2 = Alloc(rows * _hidden * 4);
            _raw = Alloc(rows * projWidth * 4);
            _logits = Alloc((long)_vocab * 4);
            Check(Native.amqe_host_alloc(out _hostToken, 16), "host alloc");
            Check(Native.amqe_host_alloc(out _hostLogits, (nuint)((long)_vocab * 4)), "host alloc");
            EnsureKv(1024);
            Reset();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The engine for this plan, or null with the reason it cannot
    /// run it. Building uploads every weight.</summary>
    public static GpuEngine? TryCreate(ComponentOpPlan plan, WeightLoader weights, out string? reason)
    {
        reason = Unsupported(plan);
        if (reason is not null)
        {
            return null;
        }
        try
        {
            return new GpuEngine(plan, weights);
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            reason = ex.Message;
            return null;
        }
    }

    /// <summary>Why the engine cannot run this plan, or null when it can.</summary>
    public static string? Unsupported(ComponentOpPlan plan)
    {
        if (plan.Diffusion is not null)
        {
            return "masked diffusion";
        }
        if (plan.Embedding is not { } embed || plan.Output is null)
        {
            return "no embedding or output head";
        }
        if (embed.Norm is not null)
        {
            return "embedding norm";
        }
        if (plan.Output.HiddenSize != plan.HiddenSize || embed.HiddenSize != plan.HiddenSize)
        {
            return "head width differs from the residual";
        }
        foreach (var layer in plan.Layers)
        {
            if (layer.Conv is not null)
            {
                return "short-conv layer";
            }
            if (layer.Ffn?.Routed is not null)
            {
                return "routed experts";
            }
            if (layer.Attention is { } attn)
            {
                if (!attn.Causal || attn.VFromK || attn.HeadGate is not null || attn.ParameterFreeQkNorm.Any)
                {
                    return "attention variant";
                }
                if (attn.HeadDim is < 2 or > 256 || attn.NumQHeads % attn.NumKvHeads != 0)
                {
                    return $"attention head dim {attn.HeadDim}";
                }
                if ((attn.QNorm is { } qn && qn.Width != attn.HeadDim && (qn.Width != attn.QDim || attn.OutputGate)) ||
                    (attn.KNorm is { } kn && kn.Width != attn.HeadDim && kn.Width != attn.KvDim))
                {
                    return "QK norm width";
                }
                if ((attn.QNorm is { Kind: not NormType.RmsNorm }) || (attn.KNorm is { Kind: not NormType.RmsNorm }))
                {
                    return "QK LayerNorm";
                }
                if (attn.Window is > int.MaxValue)
                {
                    return "window";
                }
                try
                {
                    RopeFor(attn);
                }
                catch (UnsupportedOperatorException ex)
                {
                    return ex.Message;
                }
            }
            else if (layer.LinearAttention is { } lin)
            {
                if (lin.HeadKDim is not (64 or 128) || lin.HeadVDim % 32 != 0 || lin.HeadVDim > 256 ||
                    lin.NumVHeads % lin.NumKHeads != 0 || lin.ConvKernel is < 1 or > 9)
                {
                    return "linear-attention geometry";
                }
            }
            else
            {
                return "layer without a token mixer";
            }
            if (layer.Ffn?.Dense is { } dense && dense.IsGated && dense.Activation == Activation.Relu)
            {
                return "gated ReLU";
            }
        }
        return null;
    }

    /// <summary>Rotary tables for a layer: the inverse frequencies and the
    /// cos/sin multiplier, or null for a NoPE layer. Length-dependent scaling
    /// (dynamic, longrope) would need the tables rebuilt as the sequence grows,
    /// so it is refused.</summary>
    private static (double[] InvFreq, double Scale)? RopeFor(AttentionOp attn)
    {
        if (attn.Position is PositionNone)
        {
            return null;
        }
        (double theta, int width, RopeScaling? scaling) = attn.Position switch
        {
            PositionRope r => (r.Theta, attn.HeadDim, r.Scaling),
            PositionPartialRope p => (p.Theta, p.RotaryWidth(attn.HeadDim), p.Scaling),
            _ => throw new UnsupportedOperatorException($"position policy {attn.Position.GetType().Name}"),
        };
        var shortSeq = RopeScaling.Frequencies(theta, width, scaling, 1);
        var longSeq = RopeScaling.Frequencies(theta, width, scaling, 1L << 22);
        if (shortSeq.AttentionScale != longSeq.AttentionScale || !shortSeq.InvFreq.SequenceEqual(longSeq.InvFreq))
        {
            throw new UnsupportedOperatorException("length-dependent rope scaling");
        }
        return (shortSeq.InvFreq, shortSeq.AttentionScale);
    }

    // ── session ───────────────────────────────────────────────────────────

    /// <summary>A fresh sequence: position 0, zero recurrent state.</summary>
    public void Reset()
    {
        Check(Native.amqe_memset(_engine, _pos, 0, 16), "reset");
        foreach (var l in _layers)
        {
            if (l.State != IntPtr.Zero)
            {
                Check(Native.amqe_memset(_engine, l.State, 0, (nuint)l.StateBytes), "reset");
                if (l.ConvHistoryBytes > 0)
                {
                    Check(Native.amqe_memset(_engine, l.ConvHistory, 0, (nuint)l.ConvHistoryBytes), "reset");
                }
            }
        }
        Check(Native.amqe_sync(_engine), "reset");
        _position = 0;
    }

    /// <summary>Runs the tokens through every layer from the current position
    /// and returns the last token's logits.</summary>
    public float[] Prefill(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0)
        {
            throw new ArgumentException("prefill requires at least one token", nameof(tokens));
        }
        EnsureKv(_position + tokens.Length);
        var ids = tokens.ToArray();
        for (int start = 0; start < ids.Length; start += ChunkRows)
        {
            int rows = Math.Min(ChunkRows, ids.Length - start);
            var handle = GCHandle.Alloc(ids, GCHandleType.Pinned);
            try
            {
                Check(Native.amqe_upload(_engine, _tokens, handle.AddrOfPinnedObject() + start * 4, (nuint)(rows * 4)), "tokens");
            }
            finally
            {
                handle.Free();
            }
            Forward(rows, head: start + rows == ids.Length);
            _position += rows;
        }
        Check(Native.amqe_sync(_engine), "prefill");
        return ReadLogits(fromDevice: true);
    }

    /// <summary>One token at the current position; returns its logits.</summary>
    public float[] Step(int token)
    {
        if (EnsureKv(_position + 1))
        {
            _graphReady = false;
        }
        Marshal.WriteInt32(_hostToken, token);
        if (!_graphReady)
        {
            Check(Native.amqe_capture_begin(_engine), "capture");
            try
            {
                Check(Native.amqe_copy_async(_engine, _tokens, _hostToken, 4, 1), "token");
                Forward(1, head: true);
                Check(Native.amqe_copy_async(_engine, _hostLogits, _logits, (nuint)((long)_vocab * 4), 2), "logits");
            }
            catch
            {
                Native.amqe_capture_end(_engine);
                Native.amqe_capture_drop(_engine);
                throw;
            }
            Check(Native.amqe_capture_end(_engine), "capture");
            _graphReady = true;
        }
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        Check(Native.amqe_graph_launch_async(_engine), "step");
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        Check(Native.amqe_sync(_engine), "step");
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        _position++;
        var logits = ReadLogits(fromDevice: false);
        if (Timing)
        {
            long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
            _launchTicks += t1 - t0;
            _syncTicks += t2 - t1;
            _copyTicks += t3 - t2;
            _timedSteps++;
        }
        return logits;
    }

    private float[] ReadLogits(bool fromDevice)
    {
        if (fromDevice)
        {
            Check(Native.amqe_download(_engine, _hostLogits, _logits, (nuint)((long)_vocab * 4)), "logits");
        }
        var logits = GC.AllocateUninitializedArray<float>(_vocab);
        Marshal.Copy(_hostLogits, logits, 0, _vocab);
        return logits;
    }

    /// <summary>Grows every KV cache to hold <paramref name="needed"/> rows;
    /// true when the buffers moved (a captured graph is then stale).</summary>
    private bool EnsureKv(int needed)
    {
        if (needed <= _kvCapacity)
        {
            return false;
        }
        int capacity = Math.Max(1024, _kvCapacity);
        while (capacity < needed)
        {
            capacity *= 2;
        }
        foreach (var l in _layers)
        {
            if (l.Plan.Attention is not { } attn)
            {
                continue;
            }
            long bytes = (long)capacity * attn.KvDim * 4;
            var k = Alloc(bytes);
            var v = Alloc(bytes);
            if (l.KCache != IntPtr.Zero)
            {
                long used = (long)_position * attn.KvDim * 4;
                if (used > 0)
                {
                    Check(Native.amqe_copy_async(_engine, k, l.KCache, (nuint)used, 3), "kv grow");
                    Check(Native.amqe_copy_async(_engine, v, l.VCache, (nuint)used, 3), "kv grow");
                }
                Check(Native.amqe_sync(_engine), "kv grow");
                Free(l.KCache);
                Free(l.VCache);
            }
            l.KCache = k;
            l.VCache = v;
        }
        _kvCapacity = capacity;
        // The decode attention's partial softmaxes: one slice per
        // amqe_attention_split_keys keys of capacity, for up to 8 query rows.
        if (_maxHeads > 0)
        {
            if (_attnPart != IntPtr.Zero)
            {
                Free(_attnPart);
            }
            int keys = Native.amqe_attention_split_keys();
            _attnSplits = (capacity + keys - 1) / keys;
            _attnPart = Alloc(8L * _maxHeads * _attnSplits * (_maxHeadDim + 2) * 4);
        }
        return true;
    }

    // ── the forward pass ────────────────────────────────────────────────────

    private void Forward(int rows, bool head)
    {
        int hidden = _hidden;
        float residualScale = (float)_plan.ResidualScale;
        Check(Native.amqe_embed(_engine, _embed.Ptr, _embed.Type, _tokens, rows, hidden, _embedScale, _hasEmbedScale ? 1 : 0, _x), "embed");

        for (int i = 0; i < _layers.Length; i++)
        {
            var l = _layers[i];
            var lp = l.Plan;
            bool parallel = lp.ParallelResidual;

            IntPtr hIn = _x;
            (IntPtr, IntPtr, float, float, int)? inNorm = null;
            if (l.PreAttn is { } pre)
            {
                if (!parallel && FusesNorm(rows))
                {
                    inNorm = pre;
                }
                else
                {
                    ApplyNorm(pre, _x, _hn, rows);
                    hIn = _hn;
                }
            }

            // The mixer's output joins the residual straight out of its last
            // projection unless a post-norm or a parallel block needs it apart.
            bool mixerDirect = l.PostAttn is null && !parallel;
            IntPtr mixerOut = mixerDirect ? _x : _branch;
            if (lp.Attention is { } attn)
            {
                int qkvWidth = l.Qkv!.Rows;
                Linear(l.Qkv!, hIn, hidden, rows, _proj, qkvWidth, l.QkvBias, norm: inNorm);
                if (l.QNormRow)
                {
                    Check(Native.amqe_norm(_engine, _proj, qkvWidth, _proj, qkvWidth, rows, attn.QDim,
                        l.QNormW, l.QNormOff, IntPtr.Zero, l.QNormEps, 0), "q norm");
                }
                if (l.KNormRow)
                {
                    IntPtr k = _proj + attn.QProjWidth * 4;
                    Check(Native.amqe_norm(_engine, k, qkvWidth, k, qkvWidth, rows, attn.KvDim,
                        l.KNormW, l.KNormOff, IntPtr.Zero, l.KNormEps, 0), "k norm");
                }
                Check(Native.amqe_attn_prep(_engine, _proj, qkvWidth, rows, attn.NumQHeads, attn.NumKvHeads, attn.HeadDim,
                    attn.OutputGate ? 1 : 0, _q, _gate, l.KCache, l.VCache, _pos,
                    l.QNormRow ? IntPtr.Zero : l.QNormW, l.QNormOff, l.QNormEps,
                    l.KNormRow ? IntPtr.Zero : l.KNormW, l.KNormOff, l.KNormEps,
                    l.InvFreq, l.RopePairs, l.RopeScale, attn.RopeInterleaved ? 1 : 0, l.InvFreq != IntPtr.Zero ? 1 : 0), "attn prep");
                Check(Native.amqe_attention(_engine, _q, l.KCache, l.VCache, attn.OutputGate ? _gate : IntPtr.Zero, _ctx, _pos,
                    rows, attn.NumQHeads, attn.NumKvHeads, attn.HeadDim, (float)attn.ScoreScale,
                    attn.LogitSoftcapping ?? 0f, attn.Window is { } w ? (int)w : 0, _attnPart, _attnSplits), "attention");
                Linear(l.O!, _ctx, attn.QDim, rows, mixerOut, hidden, l.OBias,
                    residual: mixerDirect ? _x : IntPtr.Zero, resScale: residualScale);
            }
            else
            {
                var lin = lp.LinearAttention!;
                int inWidth = l.In!.Rows;
                Linear(l.In!, hIn, hidden, rows, _proj, inWidth, IntPtr.Zero, norm: inNorm);
                Check(Native.amqe_gdn(_engine, _proj, inWidth, rows, lin.KeyDim, lin.ValueDim,
                    lin.NumKHeads, lin.NumVHeads, lin.HeadKDim, lin.HeadVDim, l.ConvW, lin.ConvKernel, l.ConvHistory,
                    l.ALog, l.DtBias, l.GdnNormW, (float)lin.NormEps, l.State, _conv, _qk, _gb, _gdn), "gated deltanet");
                Linear(l.Out!, _gdn, lin.ValueDim, rows, mixerOut, hidden, IntPtr.Zero,
                    residual: mixerDirect ? _x : IntPtr.Zero, resScale: residualScale);
            }

            if (!mixerDirect)
            {
                if (l.PostAttn is { } post)
                {
                    ApplyNorm(post, _branch, _branch, rows);
                }
                if (!parallel)
                {
                    Check(Native.amqe_add_scaled(_engine, _x, _branch, residualScale, (long)rows * hidden), "residual");
                }
            }

            if (l.Up is null)
            {
                if (parallel)
                {
                    Check(Native.amqe_add_scaled(_engine, _x, _branch, residualScale, (long)rows * hidden), "residual");
                }
                continue;
            }

            IntPtr fIn = parallel ? hIn : _x;
            (IntPtr, IntPtr, float, float, int)? ffnNorm = null;
            if (!parallel && l.PreFfn is { } preFfn)
            {
                if (FusesNorm(rows))
                {
                    ffnNorm = preFfn;
                }
                else
                {
                    ApplyNorm(preFfn, _x, _hn, rows);
                    fIn = _hn;
                }
            }
            var dense = lp.Ffn!.Dense!;
            int inter = dense.IntermediateSize;
            Linear(l.Up, fIn, hidden, rows, _ffn, inter, l.UpBias, act: l.Act, glu: dense.IsGated, n: inter, norm: ffnNorm);
            bool ffnDirect = l.PostFfn is null && !parallel;
            Linear(l.Down!, _ffn, inter, rows, ffnDirect ? _x : _branch2, hidden, l.DownBias,
                residual: ffnDirect ? _x : IntPtr.Zero, resScale: residualScale);
            if (!ffnDirect)
            {
                if (l.PostFfn is { } postFfn)
                {
                    ApplyNorm(postFfn, _branch2, _branch2, rows);
                }
                if (parallel)
                {
                    Check(Native.amqe_add_scaled(_engine, _branch, _branch2, 1f, (long)rows * hidden), "residual");
                    Check(Native.amqe_add_scaled(_engine, _x, _branch, residualScale, (long)rows * hidden), "residual");
                }
                else
                {
                    Check(Native.amqe_add_scaled(_engine, _x, _branch2, residualScale, (long)rows * hidden), "residual");
                }
            }
        }

        if (head)
        {
            // Only the last row goes through the final norm and the head.
            var finalNorm = (_finalNormW, _finalNormB, _finalNormOff, _finalNormEps, _finalNormKind);
            IntPtr last = _x + (rows - 1) * hidden * 4;
            if (FusesNorm(1))
            {
                Linear(_head, last, hidden, 1, _logits, _vocab, IntPtr.Zero, norm: finalNorm);
            }
            else
            {
                ApplyNorm(finalNorm, last, _hn, 1);
                Linear(_head, _hn, hidden, 1, _logits, _vocab, IntPtr.Zero);
            }
            var output = _plan.Output!;
            if (output.Multiplier is not null || output.LogitSoftcapping is not null)
            {
                Check(Native.amqe_logit_post(_engine, _logits, _vocab, (float)(output.Multiplier ?? 1.0),
                    output.Multiplier is null ? 0 : 1, output.LogitSoftcapping ?? 0f), "logit post");
            }
        }
        Check(Native.amqe_advance(_engine, _pos, rows), "advance");
    }

    /// <summary>y = epilogue(x · Wᵀ); with <paramref name="norm"/> the rows of
    /// x are normalised inside the projection (decode only: see
    /// <see cref="FusesNorm"/>).</summary>
    private void Linear(DeviceMatrix w, IntPtr x, int ldx, int rows, IntPtr y, int ldy, IntPtr bias,
        IntPtr residual = default, float resScale = 1f, int act = 0, bool glu = false, int n = -1,
        (IntPtr W, IntPtr B, float Off, float Eps, int Kind)? norm = null)
    {
        int outputs = n < 0 ? w.Rows : n;
        var nm = norm ?? default;
        Check(Native.amqe_linear(_engine, w.Ptr, w.Type, outputs, w.Cols, x, ldx, rows, y, ldy,
            bias, residual, ldy, resScale, act, glu ? 1 : 0, _raw,
            norm is null ? IntPtr.Zero : nm.W, nm.Off, nm.B, nm.Eps, nm.Kind), "linear");
    }

    /// <summary>Whether a norm can run inside the projection that reads it: a
    /// few rows (the GEMV path), whose normalised copy fits in shared memory.</summary>
    private bool FusesNorm(int rows) => rows <= 8 && (long)rows * _hidden * 4 <= 48 * 1024;

    private void ApplyNorm((IntPtr W, IntPtr B, float Off, float Eps, int Kind) norm, IntPtr x, IntPtr output, int rows)
    {
        Check(Native.amqe_norm(_engine, x, _hidden, output, _hidden, rows, _hidden, norm.W, norm.Off, norm.B, norm.Eps, norm.Kind), "norm");
    }

    // ── uploads ─────────────────────────────────────────────────────────────

    private sealed class DeviceMatrix
    {
        public required IntPtr Ptr { get; init; }
        public required int Type { get; init; }
        public required int Rows { get; init; }
        public required int Cols { get; init; }
    }

    private sealed class LayerDevice
    {
        public required LayerPlan Plan { get; init; }
        public (IntPtr W, IntPtr B, float Off, float Eps, int Kind)? PreAttn, PostAttn, PreFfn, PostFfn;
        public DeviceMatrix? Qkv, O, In, Out, Up, Down;
        public IntPtr QkvBias, OBias, UpBias, DownBias;
        public IntPtr QNormW, KNormW;
        public float QNormOff, QNormEps, KNormOff, KNormEps;

        /// <summary>QK norms over the whole projected row (OLMo 2) rather
        /// than per head: run as row norms on the projection, not in the
        /// per-head prep.</summary>
        public bool QNormRow, KNormRow;
        public IntPtr InvFreq;
        public int RopePairs;
        public double RopeScale = 1.0;
        public IntPtr KCache, VCache;
        public IntPtr ConvW, ALog, DtBias, GdnNormW, State, ConvHistory;
        public long StateBytes, ConvHistoryBytes;
        public int Act;
    }

    private DeviceMatrix Matrix(WeightLoader weights, OperandRef operand, int rows, int cols)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_matrixCache.TryGetValue(key, out var cached))
        {
            return cached;
        }
        var (bf16, f32) = weights.DeviceSource(operand, rows, cols);
        var matrix = bf16 is not null
            ? new DeviceMatrix { Ptr = Upload(bf16), Type = BF16, Rows = rows, Cols = cols }
            : new DeviceMatrix { Ptr = UploadFloats(f32!), Type = F32, Rows = rows, Cols = cols };
        _matrixCache[key] = matrix;
        return matrix;
    }

    /// <summary>Weights that share an input, stacked by rows into one device
    /// matrix so a single pass over them produces every output. BF16 when
    /// every part is BF16, otherwise all f32 (BF16 widens exactly).</summary>
    private DeviceMatrix Concat(WeightLoader weights, int cols, params (OperandRef Operand, int Rows)[] parts)
    {
        var sources = parts.Select(p => weights.DeviceSource(p.Operand, p.Rows, cols)).ToArray();
        int rows = parts.Sum(p => p.Rows);
        if (sources.All(s => s.Bf16 is not null))
        {
            var all = new ushort[(long)rows * cols];
            long offset = 0;
            foreach (var s in sources)
            {
                s.Bf16!.CopyTo(all, offset);
                offset += s.Bf16!.Length;
            }
            return new DeviceMatrix { Ptr = Upload(all), Type = BF16, Rows = rows, Cols = cols };
        }
        var f32 = new float[(long)rows * cols];
        long at = 0;
        foreach (var s in sources)
        {
            if (s.F32 is { } values)
            {
                values.CopyTo(f32, at);
                at += values.Length;
            }
            else
            {
                foreach (var bits in s.Bf16!)
                {
                    f32[at++] = BitConverter.Int32BitsToSingle(bits << 16);
                }
            }
        }
        return new DeviceMatrix { Ptr = UploadFloats(f32), Type = F32, Rows = rows, Cols = cols };
    }

    /// <summary>The biases of stacked weights, zeros where a part has none;
    /// null when no part has one.</summary>
    private IntPtr ConcatBias(WeightLoader weights, params (OperandRef? Operand, int Width)[] parts)
    {
        if (parts.All(p => p.Operand is null))
        {
            return IntPtr.Zero;
        }
        var all = new float[parts.Sum(p => p.Width)];
        int offset = 0;
        foreach (var (operand, width) in parts)
        {
            if (operand is not null)
            {
                weights.Vector(operand, width).CopyTo(all, offset);
            }
            offset += width;
        }
        return UploadFloats(all);
    }

    private (IntPtr W, IntPtr B, float Off, float Eps, int Kind) Norm(WeightLoader weights, NormOp norm) =>
        (Vector(weights, norm.Weight, norm.Width), OptionalVector(weights, norm.Bias, norm.Width),
         norm.WeightOffset, (float)norm.Eps, norm.Kind == NormType.LayerNorm ? 1 : 0);

    private IntPtr Vector(WeightLoader weights, OperandRef operand, int width)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (!_vectorCache.TryGetValue(key, out var ptr))
        {
            ptr = UploadFloats(weights.Vector(operand, width));
            _vectorCache[key] = ptr;
        }
        return ptr;
    }

    private IntPtr OptionalVector(WeightLoader weights, OperandRef? operand, int width) =>
        operand is null ? IntPtr.Zero : Vector(weights, operand, width);

    private IntPtr Upload(ushort[] data)
    {
        var ptr = Alloc((long)data.Length * 2);
        var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_upload(_engine, ptr, handle.AddrOfPinnedObject(), (nuint)((long)data.Length * 2)), "upload");
        }
        finally
        {
            handle.Free();
        }
        return ptr;
    }

    private IntPtr UploadFloats(float[] data)
    {
        var ptr = Alloc((long)data.Length * 4);
        var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_upload(_engine, ptr, handle.AddrOfPinnedObject(), (nuint)((long)data.Length * 4)), "upload");
        }
        finally
        {
            handle.Free();
        }
        return ptr;
    }

    private IntPtr Alloc(long bytes)
    {
        Check(Native.amqe_alloc(out var ptr, (nuint)Math.Max(16, bytes)), $"allocate {bytes} bytes");
        _allocations.Add(ptr);
        return ptr;
    }

    private void Free(IntPtr ptr)
    {
        _allocations.Remove(ptr);
        Native.amqe_free(ptr);
    }

    private static void Check(int status, string what)
    {
        if (status != 0)
        {
            throw new InvalidOperationException($"GPU engine: {what} failed (status {status})");
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        if (Timing && _timedSteps > 0)
        {
            double us(long t) => t * 1e6 / System.Diagnostics.Stopwatch.Frequency / _timedSteps;
            Console.Error.WriteLine($"gpu engine: {_timedSteps} steps, per step launch {us(_launchTicks):F1} us, " +
                                    $"wait {us(_syncTicks):F1} us, logits copy {us(_copyTicks):F1} us");
        }
        if (_engine != IntPtr.Zero)
        {
            Native.amqe_sync(_engine);
        }
        foreach (var ptr in _allocations)
        {
            Native.amqe_free(ptr);
        }
        _allocations.Clear();
        if (_hostToken != IntPtr.Zero)
        {
            Native.amqe_host_free(_hostToken);
        }
        if (_hostLogits != IntPtr.Zero)
        {
            Native.amqe_host_free(_hostLogits);
        }
        if (_engine != IntPtr.Zero)
        {
            Native.amqe_destroy(_engine);
        }
    }

    private static class Native
    {
        private const string Lib = "amql_cuda";

        [DllImport(Lib)] public static extern int amqe_create(out IntPtr engine);
        [DllImport(Lib)] public static extern void amqe_destroy(IntPtr engine);
        [DllImport(Lib)] public static extern int amqe_alloc(out IntPtr ptr, nuint bytes);
        [DllImport(Lib)] public static extern int amqe_free(IntPtr ptr);
        [DllImport(Lib)] public static extern int amqe_host_alloc(out IntPtr ptr, nuint bytes);
        [DllImport(Lib)] public static extern int amqe_host_free(IntPtr ptr);
        [DllImport(Lib)] public static extern int amqe_upload(IntPtr e, IntPtr dst, IntPtr src, nuint bytes);
        [DllImport(Lib)] public static extern int amqe_download(IntPtr e, IntPtr dst, IntPtr src, nuint bytes);
        [DllImport(Lib)] public static extern int amqe_copy_async(IntPtr e, IntPtr dst, IntPtr src, nuint bytes, int kind);
        [DllImport(Lib)] public static extern int amqe_memset(IntPtr e, IntPtr dst, int value, nuint bytes);
        [DllImport(Lib)] public static extern int amqe_sync(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_capture_begin(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_capture_end(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_capture_drop(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_graph_launch(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_graph_launch_async(IntPtr e);

        [DllImport(Lib)]
        public static extern int amqe_linear(IntPtr e, IntPtr w, int wType, int n, int k,
            IntPtr x, int ldx, int rows, IntPtr y, int ldy,
            IntPtr bias, IntPtr residual, int ldr, float resScale, int act, int glu, IntPtr raw,
            IntPtr normW, float normOff, IntPtr normBias, float normEps, int normKind);

        [DllImport(Lib)]
        public static extern int amqe_embed(IntPtr e, IntPtr table, int wType, IntPtr tokens, int rows, int hidden,
            float scale, int hasScale, IntPtr output);

        [DllImport(Lib)]
        public static extern int amqe_norm(IntPtr e, IntPtr x, long ldx, IntPtr output, long ldo, int rows, int cols,
            IntPtr w, float wOff, IntPtr bias, float eps, int kind);

        [DllImport(Lib)] public static extern int amqe_add_scaled(IntPtr e, IntPtr y, IntPtr b, float scale, long count);

        [DllImport(Lib)]
        public static extern int amqe_attn_prep(IntPtr e, IntPtr qkv, int ldqkv, int rows, int hq, int hkv, int d, int gated,
            IntPtr q, IntPtr gate, IntPtr kCache, IntPtr vCache, IntPtr pos,
            IntPtr qNormW, float qNormOff, float qNormEps,
            IntPtr kNormW, float kNormOff, float kNormEps,
            IntPtr invFreq, int pairs, double ropeScale, int interleaved, int rope);

        [DllImport(Lib)]
        public static extern int amqe_attention(IntPtr e, IntPtr q, IntPtr kCache, IntPtr vCache,
            IntPtr gate, IntPtr output, IntPtr pos, int rows, int hq, int hkv, int d,
            float scale, float softcap, int window, IntPtr part, int nsplit);

        [DllImport(Lib)]
        public static extern int amqe_gdn(IntPtr e, IntPtr proj, int ldp, int rows, int keyDim, int valueDim,
            int nk, int nv, int kd, int vd, IntPtr convW, int kernel, IntPtr history,
            IntPtr aLog, IntPtr dtBias, IntPtr normW, float normEps,
            IntPtr state, IntPtr conv, IntPtr qk, IntPtr gb, IntPtr output);

        [DllImport(Lib)]
        public static extern int amqe_logit_post(IntPtr e, IntPtr logits, long count, float mult, int hasMult, float cap);

        [DllImport(Lib)] public static extern int amqe_advance(IntPtr e, IntPtr pos, int by);
        [DllImport(Lib)] public static extern int amqe_attention_split_keys();
    }
}
