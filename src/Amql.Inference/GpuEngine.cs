using System.Runtime.InteropServices;
using Amql.Safetensors;
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
    private readonly IntPtr _tokens, _x, _hn, _proj, _q, _gate, _ctx, _conv, _gdn, _qk, _gb, _ffn, _branch, _branch2, _raw, _logits;
    private IntPtr _attnPart;
    private int _attnPartSplits;
    private int _maxHeads, _maxHeadDim;
    private readonly IntPtr _hostToken, _hostLogits, _hostPrefillLogits;

    // The MTP drafter (null without one): its layer, and scratch for its passes.
    private readonly MtpDevice? _mtp;
    private readonly IntPtr _hfin, _cat, _mtpH, _mtpLogits, _mtpTok, _drafts, _hostDrafts;

    /// <summary>The most drafts one <see cref="Draft"/> proposes; a verify is
    /// the chosen token and up to this many drafts, within the 8-row GEMV.</summary>
    public const int MaxDrafts = 7;

    /// <summary>Whether the model ships an MTP drafter the engine runs
    /// (<c>AMQL_MTP=0</c> turns it off).</summary>
    public bool HasDrafter => _mtp is not null;

    /// <summary>True makes every decode projection sum each row in the order
    /// one-row decode does, so a <see cref="Verify"/> row's logits equal a
    /// <see cref="Step(int)"/>'s exactly. On by default with a drafter.</summary>
    public bool RowInvariant
    {
        get => _rowInvariant;
        set
        {
            _rowInvariant = value;
            Check(Native.amqe_set_row_invariant(_engine, value ? 1 : 0), "row invariant");
            _graphReady = false;
        }
    }

    private bool _rowInvariant;

    /// <summary>The head rows a draft chooses among: the first ones (a BPE
    /// vocabulary runs roughly from common to rare), so a draft reads a slice
    /// of the head instead of all of it. A rarer token is never drafted, which
    /// costs a draft, not a token. <c>AMQL_MTP_VOCAB</c>; 32,768 by default: on
    /// Qwen 3.5 4B as many drafts held as with all 248k rows, at a third of the
    /// cost per draft.</summary>
    public int DraftVocab
    {
        get => _draftVocab;
        set => _draftVocab = Math.Clamp(value, 1, _vocab);
    }

    private int _draftVocab = int.TryParse(Environment.GetEnvironmentVariable("AMQL_MTP_VOCAB"), out int dv) && dv > 0 ? dv : 32768;

    /// <summary>The most sequences one <see cref="Step(IReadOnlyList{GpuSequence}, ReadOnlySpan{int})"/>
    /// advances together: the decode GEMV reads each weight once for up to 8 rows.</summary>
    public const int MaxBatch = 8;

    // The sequence the single-sequence API (Prefill, Step, Reset, Position) runs.
    private readonly GpuSequence _main;
    private readonly List<GpuSequence> _sequences = new();

    // The captured decode graph and the sequences it was captured for, in order.
    private GpuSequence[]? _graphFor;

    // A batch's per-sequence pointers, by layer: K caches, V caches, positions,
    // recurrent states, conv histories, MaxBatch of each (see BatchTable).
    private readonly IntPtr _batchTable;
    private const int TableKinds = 6;
    private bool _graphReady;
    private bool _disposed;

    /// <summary>The <see cref="WeightLoader.Version"/> the uploaded weights
    /// were read at.</summary>
    internal int WeightsVersion { get; }

    public int Position => _main.Position;

    private static readonly bool RowsAttention = Environment.GetEnvironmentVariable("AMQE_ATTN_ROWS") is not "0";
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
            _head = Matrix(weights, plan.Output.Projection, plan.Output.VocabSize, plan.Output.HiddenSize, project: QuantiseHead);
            (_finalNormW, _finalNormB, _finalNormOff, _finalNormEps, _finalNormKind) = Norm(weights, plan.RequireFinalNorm("the GPU engine"));

            _layers = new LayerDevice[plan.Layers.Count];
            _projWidth = _hidden;
            for (int i = 0; i < plan.Layers.Count; i++)
            {
                _layers[i] = UploadLayer(weights, plan.Layers[i]);
            }
            _mtp = Environment.GetEnvironmentVariable("AMQL_MTP") is "0" ? null : MtpDevice.TryUpload(this, weights, plan);
            if (_mtp is not null)
            {
                _projWidth = Math.Max(_projWidth, _mtp.Layer.Qkv!.Rows);
                _projWidth = Math.Max(_projWidth, _mtp.Layer.Up!.Rows);
            }
            int projWidth = _projWidth, qWidth = _qWidth, ctxWidth = _ctxWidth, convWidth = _convWidth, gdnWidth = _gdnWidth,
                ffnWidth = _ffnWidth, keyWidth = _keyWidth, vHeads = _vHeads;

            long rows = ChunkRows;
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
            _logits = Alloc((long)MaxBatch * _vocab * 4);
            _batchTable = Alloc((long)_layers.Length * TableKinds * MaxBatch * 8);
            Check(Native.amqe_host_alloc(out _hostToken, MaxBatch * 4), "host alloc");
            Check(Native.amqe_host_alloc(out _hostLogits, (nuint)((long)MaxBatch * _vocab * 4)), "host alloc");
            Check(Native.amqe_host_alloc(out _hostPrefillLogits, (nuint)((long)_vocab * 4)), "host alloc");
            if (_mtp is not null)
            {
                _hfin = Alloc(rows * _hidden * 4);
                _cat = Alloc(rows * 2 * _hidden * 4);
                _mtpH = Alloc(_hidden * 4);
                _mtpLogits = Alloc((long)_vocab * 4);
                _mtpTok = Alloc((MaxDrafts + 1) * 4);
                _drafts = Alloc(MaxDrafts * 4);
                Check(Native.amqe_host_alloc(out _hostDrafts, MaxDrafts * 4), "host alloc");
                RowInvariant = true;
                DraftVocab = _draftVocab;
            }
            ChooseKvFormat();
            _main = NewSequence();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    // Scratch widths, the widest any uploaded layer needs.
    private int _projWidth, _qWidth = 1, _ctxWidth = 1, _convWidth = 1, _gdnWidth = 1, _ffnWidth = 1, _keyWidth = 1, _vHeads = 1;

    private LayerDevice UploadLayer(WeightLoader weights, LayerPlan lp)
    {
        int projWidth = _projWidth, qWidth = _qWidth, ctxWidth = _ctxWidth, convWidth = _convWidth, gdnWidth = _gdnWidth,
            ffnWidth = _ffnWidth, keyWidth = _keyWidth, vHeads = _vHeads;
        {
            {
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
                    ld.O = Matrix(weights, attn.OProj, _hidden, attn.QDim, project: true);
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
                        ld.InvFreqLong = ld.InvFreq;
                        if (ExtensionOf(attn) is { } beyond && RopeFor(attn, extended: true) is { } longer)
                        {
                            ld.InvFreqLong = UploadFloats(longer.InvFreq.Select(f => (float)f).ToArray());
                            RopeExtendsBeyond = Math.Min(RopeExtendsBeyond ?? long.MaxValue, beyond);
                        }
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
                    ld.Out = Matrix(weights, lin.OutProj, _hidden, lin.ValueDim, project: true);
                    ld.StateBytes = (long)lin.NumVHeads * lin.HeadKDim * lin.HeadVDim * 4;
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
                        ld.Up = Matrix(weights, dense.Up, dense.IntermediateSize, _hidden, project: true);
                        ld.UpBias = OptionalVector(weights, dense.UpBias, dense.IntermediateSize);
                    }
                    ld.Down = Matrix(weights, dense.Down, _hidden, dense.IntermediateSize, project: true);
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
                (_projWidth, _qWidth, _ctxWidth, _convWidth, _gdnWidth, _ffnWidth, _keyWidth, _vHeads) =
                    (projWidth, qWidth, ctxWidth, convWidth, gdnWidth, ffnWidth, keyWidth, vHeads);
                return ld;
            }
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
        if (plan.FinalMixer is not null)
        {
            return "Qwen4 gated residual";
        }
        foreach (var layer in plan.Layers)
        {
            if (layer.Ngram is not null || layer.Attention?.Indexer is not null)
            {
                return "Qwen4 QSA / n-gram embedding";
            }
            if (layer.LinearAttention is { SigmoidGate: true })
            {
                return "sigmoid-gated linear attention";
            }
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

    private static long? ExtensionOf(AttentionOp attn) => attn.Position switch
    {
        PositionRope r => r.Scaling?.EngageBeyond,
        PositionPartialRope p => p.Scaling?.EngageBeyond,
        _ => null,
    };

    /// <summary>The length past which a sequence runs at the model's extended
    /// rotary frequencies (see <see cref="ContextExtension"/>), or null when
    /// every sequence rotates alike. Which it is is settled while the sequence
    /// is empty, from the length <see cref="Reserve"/> or the first prefill
    /// gives it, and holds until it is reset: keys are stored rotated.</summary>
    public long? RopeExtendsBeyond { get; private set; }

    /// <summary>Whether a sequence to reach <paramref name="positions"/> runs extended.</summary>
    public bool Extends(long positions) => RopeExtendsBeyond is { } beyond && positions > beyond;

    private void ChooseRope(GpuSequence seq, long positions)
    {
        if (seq.Position != 0 || seq.RopeChosen)
        {
            return;
        }
        seq.RopeChosen = true;
        bool extended = Extends(positions);
        if (seq.Extended != extended)
        {
            seq.Extended = extended;
            if (_graphFor is not null && _graphFor.Contains(seq))
            {
                _graphReady = false;   // a captured graph holds the frequencies' address
            }
        }
    }

    private static IntPtr InvFreq(LayerDevice l, GpuSequence seq) => seq.Extended ? l.InvFreqLong : l.InvFreq;

    /// <summary>Rotary tables for a layer: the inverse frequencies and the
    /// cos/sin multiplier, or null for a NoPE layer. Length-dependent scaling
    /// (dynamic, longrope) would need the tables rebuilt as the sequence grows,
    /// so it is refused.</summary>
    /// <param name="extended">The frequencies of a sequence past the length a
    /// context extension is for; without one, the same.</param>
    private static (double[] InvFreq, double Scale)? RopeFor(AttentionOp attn, bool extended = false)
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
        if (scaling?.EngageBeyond is not null && scaling.Type is "ntk" or "linear" && shortSeq.AttentionScale == longSeq.AttentionScale)
        {
            return extended ? longSeq : shortSeq;   // which of the two is settled as a sequence starts
        }
        if (shortSeq.AttentionScale != longSeq.AttentionScale || !shortSeq.InvFreq.SequenceEqual(longSeq.InvFreq))
        {
            throw new UnsupportedOperatorException("length-dependent rope scaling");
        }
        return (shortSeq.InvFreq, shortSeq.AttentionScale);
    }

    // ── session ───────────────────────────────────────────────────────────

    /// <summary>A fresh sequence: position 0, zero recurrent state.</summary>
    public void Reset() => Reset(_main);

    /// <summary>A sequence of its own beside the others: its KV cache,
    /// recurrent state and position, over the engine's one copy of the weights.
    /// Release it when done.</summary>
    public GpuSequence NewSequence()
    {
        var seq = new GpuSequence(_layers.Length) { Pos = Alloc(16) };
        for (int i = 0; i < _layers.Length; i++)
        {
            if (_layers[i].StateBytes > 0)
            {
                seq.State[i] = Alloc(_layers[i].StateBytes);
                seq.Conv[i] = _layers[i].ConvHistoryBytes > 0 ? Alloc(_layers[i].ConvHistoryBytes) : IntPtr.Zero;
            }
        }
        if (_mtp is not null)
        {
            seq.MtpPos = Alloc(16);
            seq.MtpPending = Alloc(_hidden * 4);
        }
        _sequences.Add(seq);
        if (_kvAllF16)
        {
            seq.ExactRows = AllRows;
            seq.ExactF16 = true;
        }
        EnsureKv(seq, 1024);
        Reset(seq);
        return seq;
    }

    /// <summary>Frees a sequence's state.</summary>
    public void Release(GpuSequence seq)
    {
        if (ReferenceEquals(seq, _main) || !_sequences.Remove(seq))
        {
            return;
        }
        Check(Native.amqe_sync(_engine), "release");
        if (_graphFor is not null && _graphFor.Contains(seq))
        {
            _graphReady = false;
        }
        foreach (var ptr in seq.K.Concat(seq.V).Concat(seq.State).Concat(seq.Conv).Concat(seq.MarkState ?? []).Concat(seq.MarkConv ?? [])
                     .Concat(seq.SnapState ?? []).Concat(seq.SnapConv ?? [])
                     .Concat([seq.Pos, seq.MtpK, seq.MtpV, seq.MtpPos, seq.MtpPending, seq.MarkMtpPending]))
        {
            if (ptr != IntPtr.Zero)
            {
                Free(ptr);
            }
        }
        seq.Released = true;
    }

    /// <summary>Starts a sequence again: position 0, zero recurrent state.</summary>
    public void Reset(GpuSequence seq)
    {
        Live(seq);
        // Position 0, the sequence's exact rows kept (see SetExactRows).
        UploadPos(seq.Pos, 0, ExactSpec(seq));
        if (seq.MtpPos != IntPtr.Zero)
        {
            UploadPos(seq.MtpPos, 0, ExactSpec(seq));
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            if (seq.State[i] != IntPtr.Zero)
            {
                Check(Native.amqe_memset(_engine, seq.State[i], 0, (nuint)_layers[i].StateBytes), "reset");
                if (seq.Conv[i] != IntPtr.Zero)
                {
                    Check(Native.amqe_memset(_engine, seq.Conv[i], 0, (nuint)_layers[i].ConvHistoryBytes), "reset");
                }
            }
        }
        Check(Native.amqe_sync(_engine), "reset");
        seq.Position = 0;
        seq.RopeChosen = false;
        seq.MarkedAt = -1;
        seq.MtpValid = _mtp is not null;
        seq.VerifyRows = 0;
    }

    /// <summary>Saves the sequence's recurrent state at its present position,
    /// so a later prompt that starts with the same tokens can <see cref="Rewind"/>
    /// here instead of prefilling them again. The KV cache needs no copy: its
    /// entries before the mark stay as they are until the sequence passes the
    /// mark again.</summary>
    public void Mark(GpuSequence seq)
    {
        Live(seq);
        seq.MarkState ??= new IntPtr[_layers.Length];
        seq.MarkConv ??= new IntPtr[_layers.Length];
        for (int i = 0; i < _layers.Length; i++)
        {
            if (seq.State[i] == IntPtr.Zero)
            {
                continue;
            }
            if (seq.MarkState[i] == IntPtr.Zero)
            {
                seq.MarkState[i] = Alloc(_layers[i].StateBytes);
            }
            Check(Native.amqe_copy_async(_engine, seq.MarkState[i], seq.State[i], (nuint)_layers[i].StateBytes, 3), "mark");
            if (seq.Conv[i] != IntPtr.Zero)
            {
                if (seq.MarkConv[i] == IntPtr.Zero)
                {
                    seq.MarkConv[i] = Alloc(_layers[i].ConvHistoryBytes);
                }
                Check(Native.amqe_copy_async(_engine, seq.MarkConv[i], seq.Conv[i], (nuint)_layers[i].ConvHistoryBytes, 3), "mark");
            }
        }
        if (_mtp is not null)
        {
            // The drafter's cache holds every pair before the mark; the hidden
            // state at the mark's last position waits for the token after it.
            if (seq.MarkMtpPending == IntPtr.Zero)
            {
                seq.MarkMtpPending = Alloc(_hidden * 4);
            }
            Check(Native.amqe_copy_async(_engine, seq.MarkMtpPending, seq.MtpPending, (nuint)(_hidden * 4), 3), "mark");
            seq.MarkMtpValid = seq.MtpValid;
        }
        Check(Native.amqe_sync(_engine), "mark");
        seq.MarkedAt = seq.Position;
    }

    /// <summary>Takes the sequence back to its <see cref="Mark"/>: the recurrent
    /// state saved there and its position, with the KV cache as it was.</summary>
    public void Rewind(GpuSequence seq)
    {
        Live(seq);
        if (seq.MarkedAt < 0 || seq.MarkState is null || seq.MarkConv is null)
        {
            throw new InvalidOperationException("the sequence has no mark");
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            if (seq.State[i] == IntPtr.Zero)
            {
                continue;
            }
            Check(Native.amqe_copy_async(_engine, seq.State[i], seq.MarkState[i], (nuint)_layers[i].StateBytes, 3), "rewind");
            if (seq.Conv[i] != IntPtr.Zero)
            {
                Check(Native.amqe_copy_async(_engine, seq.Conv[i], seq.MarkConv[i], (nuint)_layers[i].ConvHistoryBytes, 3), "rewind");
            }
        }
        if (_mtp is not null && seq.MarkMtpPending != IntPtr.Zero)
        {
            Check(Native.amqe_copy_async(_engine, seq.MtpPending, seq.MarkMtpPending, (nuint)(_hidden * 4), 3), "rewind");
            seq.MtpValid = seq.MarkMtpValid;
        }
        seq.VerifyRows = 0;
        UploadPos(seq.Pos, seq.MarkedAt, ExactSpec(seq));
        Check(Native.amqe_sync(_engine), "rewind");
        seq.Position = seq.MarkedAt;
    }

    // ── KV cache format ──────────────────────────────────────────────────

    /// <summary>The KV format engines created from now on use: <c>"fp16"</c>
    /// (the default), <c>"f32"</c> or <c>"nvfp4"</c> (also <c>AMQL_KV_FORMAT</c>).
    /// FP16 rows are each value rounded to nearest FP16: half f32's memory, and
    /// on Qwen 3.5 4B no measurable difference from f32. F32 rows are the values
    /// as computed.
    /// NVFP4 rows hold E2M1 codes, an E4M3 scale per 16 values and an f32
    /// scale per head: 592 bytes instead of 4,096 a position on Qwen 3.5 4B.
    /// Every path (prompt, decode, batch, verify, drafter) writes and reads the
    /// same rows, so a reply does not depend on which path ran it, but it
    /// differs from f32's.</summary>
    public static string RequestedKvFormat { get; set; } =
        Environment.GetEnvironmentVariable("AMQL_KV_FORMAT")?.Trim().ToLowerInvariant() switch
        {
            "nvfp4" or "fp4" => "nvfp4",
            "f32" or "fp32" => "f32",
            _ => "fp16",
        };

    /// <summary>The KV format this engine stores: <c>"f32"</c>, <c>"fp16"</c> or <c>"nvfp4"</c>.</summary>
    public string KvFormat => _kvAllF16 ? "fp16" : _kvFp4 ? "nvfp4" : "f32";

    /// <summary>Why the requested KV format was not used, or null.</summary>
    public string? KvFormatRefusal { get; private set; }

    // NVFP4 caches use the mixed layout (exact rows, then NVFP4 rows); an FP16
    // cache is that layout with every row exact at FP16.
    private bool _kvFp4, _kvAllF16;
    private const int AllRows = (1 << 30) - 1;
    private long[] _kvRow = Array.Empty<long>();   // bytes per cache row, by layer (0 without attention)
    private long[] _kvRowF32 = Array.Empty<long>();   // the same row at f32 (a sequence's exact rows)
    private long _mtpKvRow, _mtpKvRowF32;

    private void ChooseKvFormat()
    {
        var heads = _layers.Select(l => l.Plan.Attention).Append(_mtp?.Layer.Plan.Attention).OfType<AttentionOp>().ToList();
        if (RequestedKvFormat is "nvfp4" or "fp16")
        {
            if (heads.FirstOrDefault(a => a.HeadDim % 16 != 0) is { } odd)
            {
                KvFormatRefusal = $"{RequestedKvFormat.ToUpperInvariant()} KV needs heads a multiple of 16 wide, not {odd.HeadDim}";
            }
            else
            {
                _kvFp4 = true;
                _kvAllF16 = RequestedKvFormat == "fp16";
            }
        }
        int fmt = _kvFp4 ? 1 : 0;
        Check(Native.amqe_set_kv_format(_engine, fmt), "kv format");
        _kvRow = _layers.Select(l => l.Plan.Attention is { } a ? Native.amqe_kv_row_bytes(a.NumKvHeads, a.HeadDim, fmt) : 0L).ToArray();
        _mtpKvRow = _mtp?.Layer.Plan.Attention is { } m ? Native.amqe_kv_row_bytes(m.NumKvHeads, m.HeadDim, fmt) : 0L;
        _kvRowF32 = _layers.Select(l => l.Plan.Attention is { } a ? 4L * a.KvDim : 0L).ToArray();
        _mtpKvRowF32 = _mtp?.Layer.Plan.Attention is { } mf ? 4L * mf.KvDim : 0L;
    }

    /// <summary>Bytes a cache row (one position, K or V) takes in layer i.</summary>
    public long KvRowBytes(int layer) => _kvAllF16 ? _kvRowF32[layer] / 2 : _kvRow[layer];

    // Bytes of a layer's first `rows` cache rows when the first `exact` stay f32.
    private long KvBytes(int layer, long rows, GpuSequence seq) => MixedBytes(rows, seq, _kvRow[layer], _kvRowF32[layer]);
    private long MtpKvBytes(long rows, GpuSequence seq) => MixedBytes(rows, seq, _mtpKvRow, _mtpKvRowF32);
    private long MixedBytes(long rows, GpuSequence seq, long row, long rowF32) =>
        _kvFp4 ? Math.Min(rows, seq.ExactRows) * (seq.ExactF16 ? rowF32 / 2 : rowF32) + Math.Max(0, rows - seq.ExactRows) * row : rows * row;

    // The exact rows as the kernels read them: the count, and bit 30 for FP16.
    private static int ExactSpec(GpuSequence seq) => seq.ExactRows | (seq.ExactF16 ? 1 << 30 : 0);

    /// <summary>Keeps the sequence's first <paramref name="rows"/> positions
    /// at f32 in an NVFP4 cache (a system prompt and first instruction, say),
    /// and NVFP4 after them. Set while the sequence is empty; it holds until
    /// <see cref="Reset(GpuSequence)"/>, through rewinds, paging and batches.
    /// No effect on f32 caches. With <paramref name="f16"/> the exact rows are
    /// FP16 (half the memory, ~2e-4 relative rounding) rather than f32.</summary>
    public void SetExactRows(GpuSequence seq, int rows, bool f16 = false)
    {
        Live(seq);
        if (seq.Position != 0)
        {
            throw new InvalidOperationException("exact rows are set before the sequence holds any");
        }
        if (_kvAllF16)
        {
            return;   // every row is FP16 already
        }
        rows = _kvFp4 ? Math.Clamp(rows, 0, AllRows) : 0;
        f16 &= rows > 0;
        if (rows == seq.ExactRows && f16 == seq.ExactF16)
        {
            return;
        }
        Check(Native.amqe_sync(_engine), "exact rows");
        // The layout changes: buffers are laid out again from empty.
        seq.ExactRows = rows;
        seq.ExactF16 = f16;
        int capacity = seq.KvCapacity;
        seq.KvCapacity = 0;
        for (int i = 0; i < _layers.Length; i++)
        {
            if (seq.K[i] != IntPtr.Zero)
            {
                Free(seq.K[i]);
                Free(seq.V[i]);
                seq.K[i] = seq.V[i] = IntPtr.Zero;
            }
        }
        if (seq.MtpK != IntPtr.Zero)
        {
            Free(seq.MtpK);
            Free(seq.MtpV);
            seq.MtpK = seq.MtpV = IntPtr.Zero;
        }
        if (_graphFor is not null && _graphFor.Contains(seq))
        {
            _graphReady = false;
        }
        EnsureKv(seq, Math.Max(capacity, 1024));
        UploadPos(seq.Pos, 0, ExactSpec(seq));
        if (seq.MtpPos != IntPtr.Zero)
        {
            UploadPos(seq.MtpPos, 0, ExactSpec(seq));
        }
    }

    private void UploadPos(IntPtr pos, int position, int exact)
    {
        var values = new[] { position, exact, 0, 0 };
        var handle = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_upload(_engine, pos, handle.AddrOfPinnedObject(), 16), "position");
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>Research hook: when set, every softmax-attention layer of the
    /// stack hands out its query rows as a prompt chunk runs (layer, position
    /// of the chunk's first row, rows, [rows × heads × headDim] after norm and
    /// rope). Not called for decode steps, verifies or the drafter.</summary>
    public Action<int, int, int, float[]>? QueryProbe { get; set; }
    private int _probeFrom = -1;

    /// <summary>A layer's first <paramref name="rows"/> cache rows as stored
    /// (f32 values, or NVFP4 rows of <see cref="KvRowBytes"/> bytes).</summary>
    public byte[] DownloadKvRows(GpuSequence seq, int layer, bool values, int rows)
    {
        Live(seq);
        var host = new byte[KvBytes(layer, rows, seq)];
        var h = GCHandle.Alloc(host, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_download(_engine, h.AddrOfPinnedObject(), values ? seq.V[layer] : seq.K[layer], (nuint)host.LongLength), "kv download");
        }
        finally
        {
            h.Free();
        }
        return host;
    }

    /// <summary>Research hook: a layer's first <paramref name="rows"/> cached
    /// keys (or values) as f32, [rows × kvDim]. F32 caches only.</summary>
    public float[] DownloadKv(GpuSequence seq, int layer, bool values, int rows)
    {
        Live(seq);
        if (_kvFp4)
        {
            throw new InvalidOperationException("DownloadKv reads f32 caches");
        }
        var host = new float[rows * _kvRow[layer] / 4];
        var h = GCHandle.Alloc(host, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_download(_engine, h.AddrOfPinnedObject(), values ? seq.V[layer] : seq.K[layer], (nuint)(host.LongLength * 4)), "kv download");
        }
        finally
        {
            h.Free();
        }
        return host;
    }

    // ── paging a mark out and back ───────────────────────────────────────

    /// <summary>What a saved mark depends on besides the weights: each layer's
    /// cache and state sizes, the drafter's, and the prefill precision. A mark
    /// saved under one geometry is only restored under the same.</summary>
    public string MarkGeometry
    {
        get
        {
            var sb = new System.Text.StringBuilder($"h{_hidden};");
            foreach (var layer in _layers)
            {
                sb.Append(layer.Plan.Attention is { } attn ? $"a{attn.KvDim}" : "").Append($"s{layer.StateBytes}c{layer.ConvHistoryBytes};");
            }
            sb.Append(_mtp is null ? "mtp-" : $"mtp{_mtp.KvDim}").Append(_exactPrefill ? ";f32" : ";bf16").Append(";kv-").Append(KvFormat);
            // Keys are stored rotated, so state is only good under the rotation that made it
            // (a context extension changes it without changing the container).
            var rope = new System.Text.StringBuilder();
            foreach (var layer in _layers)
            {
                foreach (bool extended in new[] { false, true })
                if (layer.Plan.Attention is { } attn && RopeFor(attn, extended) is { } r)
                {
                    rope.Append(r.Scale.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                    foreach (var f in r.InvFreq)
                    {
                        rope.Append(',').Append(f.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    rope.Append(';');
                }
            }
            sb.Append(";rope-").Append(Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.ASCII.GetBytes(rope.ToString()))).AsSpan(0, 16));
            return sb.ToString();
        }
    }

    /// <summary>Device memory a sequence holds: its KV caches at their
    /// capacity, its recurrent state, the copy saved at its mark and a
    /// verify's snapshots.</summary>
    public long DeviceBytes(GpuSequence seq)
    {
        long bytes = 0;
        for (int i = 0; i < _layers.Length; i++)
        {
            if (_layers[i].Plan.Attention is { } attn)
            {
                bytes += 2L * KvBytes(i, seq.KvCapacity, seq);
            }
            if (seq.State[i] != IntPtr.Zero)
            {
                long state = _layers[i].StateBytes + _layers[i].ConvHistoryBytes;
                int copies = 1 + (seq.MarkState?[i] is { } m && m != IntPtr.Zero ? 1 : 0) + (seq.SnapState?[i] is { } s && s != IntPtr.Zero ? seq.SnapRows : 0);
                bytes += copies * state;
            }
        }
        if (_mtp is not null)
        {
            bytes += 2L * MtpKvBytes(seq.KvCapacity, seq) + 2L * _hidden * 4;
        }
        return bytes;
    }

    // A saved mark's header: its position, whether the drafter kept up, the exact rows.
    private const int MarkHeader = 16;

    /// <summary>Bytes <see cref="SaveMark"/> writes for the sequence.</summary>
    public long MarkBytes(GpuSequence seq)
    {
        long at = Math.Max(0, seq.MarkedAt);
        long bytes = MarkHeader;
        for (int i = 0; i < _layers.Length; i++)
        {
            if (_layers[i].Plan.Attention is { } attn)
            {
                bytes += 2 * KvBytes(i, at, seq);
            }
            if (seq.State[i] != IntPtr.Zero)
            {
                bytes += _layers[i].StateBytes + _layers[i].ConvHistoryBytes;
            }
        }
        if (_mtp is not null)
        {
            bytes += 2 * MtpKvBytes(at, seq) + _hidden * 4L;
        }
        return bytes;
    }

    /// <summary>Writes what <see cref="Rewind"/> needs to take the sequence
    /// back to its mark: the KV rows before the mark, the recurrent state
    /// saved there and the drafter's side. Rows after the mark are left out;
    /// <see cref="RestoreMark"/> rebuilds the sequence from what is written,
    /// bit for bit, under the same <see cref="MarkGeometry"/>.</summary>
    public void SaveMark(GpuSequence seq, Stream to)
    {
        Live(seq);
        if (seq.MarkedAt < 0 || seq.MarkState is null || seq.MarkConv is null)
        {
            throw new InvalidOperationException("the sequence has no mark");
        }
        Check(Native.amqe_sync(_engine), "save mark");
        int at = seq.MarkedAt;
        var head = new byte[MarkHeader];
        BitConverter.TryWriteBytes(head.AsSpan(0, 4), at);
        head[4] = (byte)(seq.MarkMtpValid ? 1 : 0);
        BitConverter.TryWriteBytes(head.AsSpan(8, 4), seq.ExactRows);
        head[5] = (byte)(seq.ExactF16 ? 1 : 0);
        head[6] = (byte)(seq.Extended ? 1 : 0);
        to.Write(head);
        var buffer = new byte[Math.Min(16 << 20, Math.Max(1 << 16, MarkBytes(seq)))];
        var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        try
        {
            void Out(IntPtr device, long bytes)
            {
                for (long done = 0; done < bytes;)
                {
                    int n = (int)Math.Min(buffer.Length, bytes - done);
                    Check(Native.amqe_download(_engine, handle.AddrOfPinnedObject(), device + (nint)done, (nuint)n), "save mark");
                    to.Write(buffer, 0, n);
                    done += n;
                }
            }
            for (int i = 0; i < _layers.Length; i++)
            {
                if (_layers[i].Plan.Attention is { } attn)
                {
                    Out(seq.K[i], KvBytes(i, at, seq));
                    Out(seq.V[i], KvBytes(i, at, seq));
                }
                if (seq.State[i] != IntPtr.Zero)
                {
                    Out(seq.MarkState[i], _layers[i].StateBytes);
                    if (_layers[i].ConvHistoryBytes > 0)
                    {
                        Out(seq.MarkConv[i], _layers[i].ConvHistoryBytes);
                    }
                }
            }
            if (_mtp is not null)
            {
                Out(seq.MtpK, MtpKvBytes(at, seq));
                Out(seq.MtpV, MtpKvBytes(at, seq));
                if (seq.MarkMtpPending != IntPtr.Zero)
                {
                    Out(seq.MarkMtpPending, _hidden * 4L);
                }
                else
                {
                    to.Write(new byte[_hidden * 4]);
                }
            }
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>A new sequence at the mark <see cref="SaveMark"/> wrote: its
    /// position at the mark, the mark saved there again, so it continues (and
    /// rewinds) exactly as the sequence that was saved.</summary>
    public GpuSequence RestoreMark(Stream from)
    {
        var head = new byte[MarkHeader];
        from.ReadExactly(head);
        int at = BitConverter.ToInt32(head, 0);
        if (at <= 0)
        {
            throw new InvalidDataException("a saved mark at position 0");
        }
        var seq = NewSequence();
        try
        {
            SetExactRows(seq, BitConverter.ToInt32(head, 8), head[5] != 0);
            seq.Extended = head[6] != 0;
            seq.RopeChosen = true;
            EnsureKv(seq, at + 1);
            seq.MarkState = new IntPtr[_layers.Length];
            seq.MarkConv = new IntPtr[_layers.Length];
            var buffer = new byte[16 << 20];
            var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                void In(IntPtr device, long bytes)
                {
                    for (long done = 0; done < bytes;)
                    {
                        int n = (int)Math.Min(buffer.Length, bytes - done);
                        from.ReadExactly(buffer, 0, n);
                        Check(Native.amqe_upload(_engine, device + (nint)done, handle.AddrOfPinnedObject(), (nuint)n), "restore mark");
                        done += n;
                    }
                }
                for (int i = 0; i < _layers.Length; i++)
                {
                    if (_layers[i].Plan.Attention is { } attn)
                    {
                        In(seq.K[i], KvBytes(i, at, seq));
                        In(seq.V[i], KvBytes(i, at, seq));
                    }
                    if (seq.State[i] != IntPtr.Zero)
                    {
                        seq.MarkState[i] = Alloc(_layers[i].StateBytes);
                        In(seq.MarkState[i], _layers[i].StateBytes);
                        if (_layers[i].ConvHistoryBytes > 0)
                        {
                            seq.MarkConv[i] = Alloc(_layers[i].ConvHistoryBytes);
                            In(seq.MarkConv[i], _layers[i].ConvHistoryBytes);
                        }
                    }
                }
                if (_mtp is not null)
                {
                    In(seq.MtpK, MtpKvBytes(at, seq));
                    In(seq.MtpV, MtpKvBytes(at, seq));
                    seq.MarkMtpPending = Alloc(_hidden * 4);
                    In(seq.MarkMtpPending, _hidden * 4L);
                    seq.MarkMtpValid = head[4] != 0;
                }
            }
            finally
            {
                handle.Free();
            }
            seq.MarkedAt = at;
            Rewind(seq);
            return seq;
        }
        catch
        {
            Release(seq);
            throw;
        }
    }

    private static void Live(GpuSequence seq)
    {
        if (seq.Released)
        {
            throw new InvalidOperationException("the sequence was released");
        }
    }

    /// <summary>True runs prompt projections as FP32 SGEMM over widened
    /// weights, keeping activations at f32; false (the default) rounds them to
    /// BF16 for tensor-core GEMMs, accumulating in f32. Decode steps are f32
    /// either way.</summary>
    public bool ExactPrefill
    {
        get => _exactPrefill;
        set
        {
            _exactPrefill = value;
            Check(Native.amqe_set_prefill_f32(_engine, value ? 1 : 0), "prefill precision");
        }
    }

    private bool _exactPrefill;

    /// <summary>Runs the tokens through every layer from the current position
    /// and returns the last token's logits.</summary>
    public float[] Prefill(ReadOnlySpan<int> tokens) => Prefill(_main, tokens);

    /// <summary>Runs the tokens through every layer from the sequence's
    /// position and returns the last token's logits.</summary>
    public float[] Prefill(GpuSequence seq, ReadOnlySpan<int> tokens)
    {
        Live(seq);
        if (tokens.Length == 0)
        {
            throw new ArgumentException("prefill requires at least one token", nameof(tokens));
        }
        ChooseRope(seq, tokens.Length);
        // Growing moves the KV buffers, and the captured decode graph holds their old
        // addresses: replaying it after a long prefill read freed memory (status -700).
        if (EnsureKv(seq, seq.Position + tokens.Length))
        {
            _graphReady = false;
        }
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
            bool drafting = _mtp is not null && seq.MtpValid;
            _probeFrom = QueryProbe is null ? -1 : seq.Position;
            try
            {
                if (_massOut != IntPtr.Zero)
                {
                    Check(Native.amqe_set_attention_mass(_engine, _massOut, _massScratch, _massSplits), "attention mass");
                }
                Forward(new[] { seq }, rows, head: start + rows == ids.Length, hidden: drafting);
            }
            finally
            {
                _probeFrom = -1;
                if (_massOut != IntPtr.Zero)
                {
                    Check(Native.amqe_set_attention_mass(_engine, IntPtr.Zero, IntPtr.Zero, 0), "attention mass");
                }
            }
            if (drafting)
            {
                MtpCatchUp(seq, seq.Position, rows);
            }
            seq.Position += rows;
        }
        Check(Native.amqe_sync(_engine), "prefill");
        seq.VerifyRows = 0;
        return ReadLogits(0, fromDevice: true);
    }

    // ── speculative decoding ────────────────────────────────────────────────

    /// <summary>
    /// The chosen next token and drafts after it, as rows at the sequence's
    /// position in one pass: row i's logits (<see cref="HostLogits"/>(i)) are
    /// the next token's after <paramref name="tokens"/>[0..i]. With
    /// <see cref="RowInvariant"/> each row equals what a <see cref="Step(int)"/>
    /// there gives, so drafts the model would have chosen are free tokens.
    /// <see cref="Accept"/> then keeps the rows whose drafts held.
    /// </summary>
    public void Verify(GpuSequence seq, ReadOnlySpan<int> tokens)
    {
        Live(seq);
        int n = tokens.Length;
        if (n < 1 || n > MaxDrafts + 1)
        {
            throw new ArgumentException($"a verify takes 1 to {MaxDrafts + 1} rows");
        }
        if (EnsureKv(seq, seq.Position + n + MaxDrafts))
        {
            _graphReady = false;
        }
        EnsureSnapshots(seq, n - 1);
        for (int i = 0; i < n; i++)
        {
            Marshal.WriteInt32(_hostToken, i * 4, tokens[i]);
        }
        bool drafting = _mtp is not null && seq.MtpValid;
        Check(Native.amqe_copy_async(_engine, _tokens, _hostToken, (nuint)(n * 4), 1), "tokens");
        Forward(new[] { seq }, n, head: true, allRows: true, hidden: drafting, snap: n > 1);
        Check(Native.amqe_copy_async(_engine, _hostLogits, _logits, (nuint)((long)n * _vocab * 4), 2), "logits");
        Check(Native.amqe_sync(_engine), "verify");
        seq.VerifyFrom = seq.Position;
        seq.VerifyRows = n;
        seq.Position += n;
        _lastRows = n;
    }

    /// <summary>Keeps the first <paramref name="rows"/> rows of the last
    /// <see cref="Verify"/>: the sequence stands after them, its recurrent
    /// state as it was there, and the drafter catches up on them.</summary>
    public void Accept(GpuSequence seq, int rows)
    {
        Live(seq);
        int n = seq.VerifyRows;
        if (n == 0 || rows < 1 || rows > n)
        {
            throw new ArgumentException($"accept 1 to {n} rows of the last verify");
        }
        if (rows < n)
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                if (seq.State[i] == IntPtr.Zero)
                {
                    continue;
                }
                long sb = _layers[i].StateBytes, cb = _layers[i].ConvHistoryBytes;
                Check(Native.amqe_copy_async(_engine, seq.State[i], seq.SnapState![i] + (nint)((rows - 1) * sb), (nuint)sb, 3), "accept");
                if (seq.Conv[i] != IntPtr.Zero)
                {
                    Check(Native.amqe_copy_async(_engine, seq.Conv[i], seq.SnapConv![i] + (nint)((rows - 1) * cb), (nuint)cb, 3), "accept");
                }
            }
            Check(Native.amqe_set_int(_engine, seq.Pos, seq.VerifyFrom + rows), "accept");
            seq.Position = seq.VerifyFrom + rows;
        }
        if (_mtp is not null && seq.MtpValid)
        {
            MtpCatchUp(seq, seq.VerifyFrom, rows);
        }
        seq.VerifyRows = 0;
    }

    /// <summary>
    /// Up to <paramref name="count"/> draft tokens to follow
    /// <paramref name="token"/> (the token chosen after the sequence's last
    /// position), from the MTP drafter, greedily, each fed back for the next.
    /// Empty without a drafter, or when the drafter has not kept up with the
    /// sequence (a batch step passed it by). The sequence itself is unchanged.
    /// </summary>
    public int[] Draft(GpuSequence seq, int token, int count)
    {
        Live(seq);
        count = Math.Min(count, MaxDrafts);
        if (_mtp is null || !seq.MtpValid || seq.Position < 1 || count < 1)
        {
            return Array.Empty<int>();
        }
        if (EnsureKv(seq, seq.Position + count + MaxDrafts + 1))
        {
            _graphReady = false;
        }
        Check(Native.amqe_set_int(_engine, _mtpTok, token), "draft");
        for (int i = 0; i < count; i++)
        {
            MtpPass(seq, 1, _mtpTok + i * 4, i == 0 ? seq.MtpPending : _mtpH, IntPtr.Zero, seq.Position - 1 + i,
                draftTo: _drafts + i * 4, nextToken: _mtpTok + (i + 1) * 4);
        }
        Check(Native.amqe_copy_async(_engine, _hostDrafts, _drafts, (nuint)(count * 4), 2), "draft");
        Check(Native.amqe_sync(_engine), "draft");
        var drafts = new int[count];
        Marshal.Copy(_hostDrafts, drafts, 0, count);
        return drafts;
    }

    /// <summary>The drafter over the main rows just run at positions
    /// <paramref name="from"/>.., whose tokens are in <c>_tokens</c> and whose
    /// final-normed hidden states are in <c>_hfin</c>: pairs (hidden at p − 1,
    /// token at p) fill its cache at p − 1, and the last row's hidden state
    /// waits for the token chosen after it.</summary>
    private void MtpCatchUp(GpuSequence seq, int from, int rows)
    {
        if (from == 0)
        {
            if (rows > 1)
            {
                MtpPass(seq, rows - 1, _tokens + 4, _hfin, _hfin + _hidden * 4, 0);
            }
        }
        else
        {
            MtpPass(seq, rows, _tokens, seq.MtpPending, _hfin, from - 1);
        }
        Check(Native.amqe_copy_async(_engine, seq.MtpPending, _hfin + (nint)((long)(rows - 1) * _hidden * 4), (nuint)(_hidden * 4), 3), "mtp");
    }

    /// <summary>One drafter pass over <paramref name="rows"/> pairs at
    /// positions <paramref name="pos"/>..: row r pairs the token at
    /// <paramref name="tokens"/>[r] with a hidden state, <paramref name="hidden0"/>
    /// for row 0 and <paramref name="hiddenRest"/>[r − 1] after. With
    /// <paramref name="draftTo"/>, the last row goes on through the drafter's norm
    /// and the head, and its argmax is written there and to <paramref name="nextToken"/>,
    /// its normed hidden state to <c>_mtpH</c>.</summary>
    private void MtpPass(GpuSequence seq, int rows, IntPtr tokens, IntPtr hidden0, IntPtr hiddenRest, int pos,
        IntPtr draftTo = default, IntPtr nextToken = default)
    {
        var m = _mtp!;
        int h = _hidden;
        long ld = 2L * h;
        Check(Native.amqe_embed(_engine, _embed.Ptr, _embed.Type, tokens, rows, h, _embedScale, _hasEmbedScale ? 1 : 0, _hn), "mtp embed");
        Check(Native.amqe_norm(_engine, _hn, h, _cat, ld, rows, h, m.PreEmbed.W, m.PreEmbed.Off, m.PreEmbed.B, m.PreEmbed.Eps, m.PreEmbed.Kind), "mtp norm");
        Check(Native.amqe_norm(_engine, hidden0, h, _cat + h * 4, ld, 1, h, m.PreHidden.W, m.PreHidden.Off, m.PreHidden.B, m.PreHidden.Eps, m.PreHidden.Kind), "mtp norm");
        if (rows > 1)
        {
            Check(Native.amqe_norm(_engine, hiddenRest, h, _cat + (nint)(ld * 4) + h * 4, ld, rows - 1, h,
                m.PreHidden.W, m.PreHidden.Off, m.PreHidden.B, m.PreHidden.Eps, m.PreHidden.Kind), "mtp norm");
        }
        Linear(m.Fc, _cat, (int)ld, rows, _x, h, IntPtr.Zero);
        Check(Native.amqe_set_int(_engine, seq.MtpPos, pos), "mtp position");
        Layer(m.Layer, -1, new[] { seq }, rows, new KvTarget(seq.MtpK, seq.MtpV, seq.MtpPos, pos), snap: false);
        if (draftTo != IntPtr.Zero)
        {
            Check(Native.amqe_norm(_engine, _x + (nint)((long)(rows - 1) * h * 4), h, _mtpH, h, 1, h,
                m.Norm.W, m.Norm.Off, m.Norm.B, m.Norm.Eps, m.Norm.Kind), "mtp norm");
            Linear(_head, _mtpH, h, 1, _mtpLogits, _vocab, IntPtr.Zero, n: DraftVocab);
            Check(Native.amqe_argmax(_engine, _mtpLogits, DraftVocab, draftTo, nextToken), "mtp argmax");
        }
    }

    /// <summary>Room for a verify's recurrent state after each of
    /// <paramref name="rows"/> rows.</summary>
    private void EnsureSnapshots(GpuSequence seq, int rows)
    {
        if (rows <= seq.SnapRows)
        {
            return;
        }
        seq.SnapState ??= new IntPtr[_layers.Length];
        seq.SnapConv ??= new IntPtr[_layers.Length];
        for (int i = 0; i < _layers.Length; i++)
        {
            if (seq.State[i] == IntPtr.Zero)
            {
                continue;
            }
            if (seq.SnapState[i] != IntPtr.Zero)
            {
                Free(seq.SnapState[i]);
            }
            seq.SnapState[i] = Alloc(rows * _layers[i].StateBytes);
            if (seq.Conv[i] != IntPtr.Zero)
            {
                if (seq.SnapConv[i] != IntPtr.Zero)
                {
                    Free(seq.SnapConv[i]);
                }
                seq.SnapConv[i] = Alloc(rows * _layers[i].ConvHistoryBytes);
            }
        }
        seq.SnapRows = rows;
    }

    /// <summary>One token at the current position; returns its logits.</summary>
    public float[] Step(int token) => Step(new[] { _main }, new[] { token })[0];

    /// <summary>
    /// One token for each of up to <see cref="MaxBatch"/> sequences, in one pass:
    /// the projections and the head read every weight once for all of them, while
    /// attention and the recurrence run per sequence over its own state. Returns
    /// each sequence's logits, in order. The pass is captured as a graph and
    /// replayed while the same sequences step together.
    /// </summary>
    public float[][] Step(IReadOnlyList<GpuSequence> seqs, ReadOnlySpan<int> tokens)
    {
        StepInPlace(seqs, tokens);
        var logits = new float[seqs.Count][];
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = HostLogits(i).ToArray();
        }
        return logits;
    }

    /// <summary>
    /// <see cref="Step(IReadOnlyList{GpuSequence}, ReadOnlySpan{int})"/> without copying the
    /// logits out: row i of the step is <see cref="HostLogits"/>(i), read where the device
    /// wrote it, until the next step on this engine. (A prefill writes elsewhere.)
    /// </summary>
    public void StepInPlace(IReadOnlyList<GpuSequence> seqs, ReadOnlySpan<int> tokens)
    {
        int b = seqs.Count;
        if (b < 1 || b > MaxBatch || tokens.Length != b)
        {
            throw new ArgumentException($"a step takes 1 to {MaxBatch} sequences, one token each");
        }
        if (seqs.Distinct().Count() != b)
        {
            throw new ArgumentException("a sequence can only step once per pass");
        }
        var batch = seqs.ToArray();
        foreach (var seq in batch)
        {
            Live(seq);
            if (EnsureKv(seq, seq.Position + 1))
            {
                _graphReady = false;
            }
        }
        if (_graphFor is null || !_graphFor.SequenceEqual(batch))
        {
            _graphReady = false;
        }
        for (int i = 0; i < b; i++)
        {
            Marshal.WriteInt32(_hostToken, i * 4, tokens[i]);
        }
        if (!_graphReady)
        {
            if (b > 1)
            {
                UploadBatchTable(batch);
            }
            Check(Native.amqe_capture_begin(_engine), "capture");
            try
            {
                Check(Native.amqe_copy_async(_engine, _tokens, _hostToken, (nuint)(b * 4), 1), "token");
                Forward(batch, b, head: true);
                Check(Native.amqe_copy_async(_engine, _hostLogits, _logits, (nuint)((long)b * _vocab * 4), 2), "logits");
            }
            catch
            {
                Native.amqe_capture_end(_engine);
                Native.amqe_capture_drop(_engine);
                throw;
            }
            Check(Native.amqe_capture_end(_engine), "capture");
            _graphReady = true;
            _graphFor = batch;
        }
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        Check(Native.amqe_graph_launch_async(_engine), "step");
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        Check(Native.amqe_sync(_engine), "step");
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        foreach (var seq in batch)
        {
            seq.Position++;
            // The drafter did not see this step: it drafts again after a reset or rewind.
            seq.MtpValid = false;
            seq.VerifyRows = 0;
        }
        _lastRows = b;
        if (Timing)
        {
            long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
            _launchTicks += t1 - t0;
            _syncTicks += t2 - t1;
            _copyTicks += t3 - t2;
            if (++_timedSteps % 500 == 0)
            {
                double us(long t) => t * 1e6 / System.Diagnostics.Stopwatch.Frequency / _timedSteps;
                Console.Error.WriteLine($"gpu engine: {_timedSteps} steps of {b}, per step launch {us(_launchTicks):F1} us, " +
                                        $"wait {us(_syncTicks):F1} us, logits copy {us(_copyTicks):F1} us");
                _launchTicks = _syncTicks = _copyTicks = _timedSteps = 0;
            }
        }
    }

    private int _lastRows;

    /// <summary>Row <paramref name="row"/> of the last step's logits, in the pinned host
    /// buffer the device copied them to; valid until the next step on this engine.</summary>
    public unsafe ReadOnlySpan<float> HostLogits(int row)
    {
        if ((uint)row >= (uint)_lastRows)
        {
            throw new ArgumentOutOfRangeException(nameof(row), $"the last step had {_lastRows} rows");
        }
        return new ReadOnlySpan<float>((float*)(_hostLogits + (nint)((long)row * _vocab * 4)), _vocab);
    }

    private float[] ReadLogits(int row, bool fromDevice)
    {
        IntPtr host = fromDevice ? _hostPrefillLogits : _hostLogits;
        if (fromDevice)
        {
            Check(Native.amqe_download(_engine, _hostPrefillLogits, _logits, (nuint)((long)_vocab * 4)), "logits");
        }
        var logits = GC.AllocateUninitializedArray<float>(_vocab);
        Marshal.Copy(host + row * _vocab * 4, logits, 0, _vocab);
        return logits;
    }

    /// <summary>For tests: the decode splits and prompt keys a long cache's
    /// attention takes at a time, and the scratch bytes past which a prompt is
    /// taken so; 0 is the default. Set before an engine grows its caches.</summary>
    public static void SetAttentionSegments(int splits, int keys, long budget) =>
        Check(Native.amqe_set_attention_segments(splits, keys, budget), "attention segments");

    /// <summary>Says how long an empty sequence is to get, which settles
    /// whether it runs at a context extension's frequencies (no memory is taken;
    /// see <see cref="RopeExtendsBeyond"/>).</summary>
    public void Plan(GpuSequence seq, long positions)
    {
        Live(seq);
        ChooseRope(seq, positions);
    }

    /// <summary>Positions in each window <see cref="ReadAttention"/> reports.</summary>
    public static int AttentionWindow => Native.amqe_attention_split_keys();

    /// <summary>
    /// Reads the tokens on, one at a time, and returns where the softmax layers'
    /// attention went as they were read: for each window of
    /// <see cref="AttentionWindow"/> positions from the start, the share of
    /// attention it took, summed over the tokens, layers and heads. The last
    /// token's logits come back too.
    /// </summary>
    public (float[] Mass, float[] Logits) ReadAttention(GpuSequence seq, ReadOnlySpan<int> tokens)
    {
        Live(seq);
        int end = seq.Position + tokens.Length;
        Reserve(seq, end);
        int splits = Splits(seq);
        var mass = Alloc(splits * 4L);
        IntPtr scratch = IntPtr.Zero;
        try
        {
            scratch = Alloc(_maxHeads * (2L * splits + 2) * 4);
            Check(Native.amqe_memset(_engine, mass, 0, (nuint)(splits * 4L)), "attention mass");
            _massOut = mass;
            _massScratch = scratch;
            _massSplits = splits;
            var logits = Array.Empty<float>();
            for (int i = 0; i < tokens.Length; i++)
            {
                logits = Prefill(seq, tokens.Slice(i, 1));
            }
            var host = new float[splits];
            var handle = GCHandle.Alloc(host, GCHandleType.Pinned);
            try
            {
                Check(Native.amqe_download(_engine, handle.AddrOfPinnedObject(), mass, (nuint)(splits * 4L)), "attention mass");
            }
            finally
            {
                handle.Free();
            }
            return (host, logits);
        }
        finally
        {
            _massOut = IntPtr.Zero;
            Check(Native.amqe_sync(_engine), "attention mass");
            Free(mass);
            if (scratch != IntPtr.Zero)
            {
                Free(scratch);
            }
        }
    }

    private IntPtr _massOut, _massScratch;
    private int _massSplits;

    /// <summary>Makes a sequence's KV caches long enough for
    /// <paramref name="positions"/> positions, so reading up to there does not
    /// grow them step by step. An empty sequence that is to go past the trained
    /// length of an extended model takes the extended rotary frequencies here.
    /// Throws when the device has no room.</summary>
    public void Reserve(GpuSequence seq, int positions)
    {
        Live(seq);
        ChooseRope(seq, positions);
        if (EnsureKv(seq, positions))
        {
            _graphReady = false;
        }
    }

    /// <summary>Grows a sequence's KV caches to hold <paramref name="needed"/> rows;
    /// true when buffers moved (a captured graph is then stale).</summary>
    private bool EnsureKv(GpuSequence seq, int needed)
    {
        if (needed <= seq.KvCapacity)
        {
            return false;
        }
        // Doubling to 128K rows, then a quarter at a time: at long contexts a
        // doubled cache would be gigabytes nothing fills.
        int capacity = Math.Max(1024, seq.KvCapacity);
        while (capacity < needed)
        {
            capacity = capacity < GrowStep ? capacity * 2 : checked(capacity + Math.Max(GrowStep, capacity / 4 / GrowStep * GrowStep));
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            if (_layers[i].Plan.Attention is not { } attn)
            {
                continue;
            }
            long bytes = KvBytes(i, capacity, seq);
            var (k, v) = AllocPair(bytes);
            if (seq.K[i] != IntPtr.Zero)
            {
                long used = KvBytes(i, seq.Position, seq);
                if (used > 0)
                {
                    Check(Native.amqe_copy_async(_engine, k, seq.K[i], (nuint)used, 3), "kv grow");
                    Check(Native.amqe_copy_async(_engine, v, seq.V[i], (nuint)used, 3), "kv grow");
                }
                Check(Native.amqe_sync(_engine), "kv grow");
                Free(seq.K[i]);
                Free(seq.V[i]);
            }
            seq.K[i] = k;
            seq.V[i] = v;
        }
        if (_mtp is not null)
        {
            long bytes = MtpKvBytes(capacity, seq);
            var (k, v) = AllocPair(bytes);
            if (seq.MtpK != IntPtr.Zero)
            {
                long used = MtpKvBytes(seq.Position, seq);
                if (used > 0)
                {
                    Check(Native.amqe_copy_async(_engine, k, seq.MtpK, (nuint)used, 3), "kv grow");
                    Check(Native.amqe_copy_async(_engine, v, seq.MtpV, (nuint)used, 3), "kv grow");
                }
                Check(Native.amqe_sync(_engine), "kv grow");
                Free(seq.MtpK);
                Free(seq.MtpV);
            }
            seq.MtpK = k;
            seq.MtpV = v;
        }
        seq.KvCapacity = capacity;
        // The decode attention's partial softmaxes: one slice per
        // amqe_attention_split_keys keys of capacity, for up to 8 query rows,
        // sized for the longest sequence (they run one after another).
        if (_maxHeads > 0 && PartSlots(seq) > _attnPartSplits)
        {
            Check(Native.amqe_sync(_engine), "kv grow");
            // The new one first: with no room for it the old one still stands.
            int slots = PartSlots(seq);
            var part = Alloc(8L * _maxHeads * slots * (_maxHeadDim + 2) * 4);
            if (_attnPart != IntPtr.Zero)
            {
                Free(_attnPart);
            }
            _attnPart = part;
            _attnPartSplits = slots;
        }
        return true;
    }

    /// <summary>Writes the batch's pointers where the batched kernels read them:
    /// for layer i and kind k, MaxBatch pointers at (i·TableKinds + k)·MaxBatch.
    /// The captured graph reads this table, so it is written before each capture.</summary>
    private void UploadBatchTable(GpuSequence[] batch)
    {
        var table = new long[_layers.Length * TableKinds * MaxBatch];
        for (int i = 0; i < _layers.Length; i++)
        {
            for (int r = 0; r < batch.Length; r++)
            {
                int at = i * TableKinds * MaxBatch + r;
                table[at] = batch[r].K[i];
                table[at + MaxBatch] = batch[r].V[i];
                table[at + 2 * MaxBatch] = batch[r].Pos;
                table[at + 3 * MaxBatch] = batch[r].State[i];
                table[at + 4 * MaxBatch] = batch[r].Conv[i];
                table[at + 5 * MaxBatch] = InvFreq(_layers[i], batch[r]);
            }
        }
        var handle = GCHandle.Alloc(table, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_upload(_engine, _batchTable, handle.AddrOfPinnedObject(), (nuint)(table.Length * 8)), "batch table");
        }
        finally
        {
            handle.Free();
        }
    }

    private IntPtr Table(int layer, int kind) => _batchTable + (layer * TableKinds + kind) * MaxBatch * 8;

    private const int GrowStep = 1 << 17;

    // Scratch slots the decode attention needs for a cache this long.
    private static int PartSlots(GpuSequence seq) => Native.amqe_attention_part_slots(Splits(seq));

    private static int Splits(GpuSequence seq)
    {
        int keys = Native.amqe_attention_split_keys();
        return (seq.KvCapacity + keys - 1) / keys;
    }

    // ── the forward pass ────────────────────────────────────────────────────

    /// <summary>
    /// The layers over <paramref name="rows"/> rows: one sequence's next
    /// <paramref name="rows"/> tokens, or (several sequences) one token each,
    /// row i belonging to <paramref name="seqs"/>[i]. Everything but attention
    /// and the recurrence treats the rows alike; those two run per sequence.
    /// The head reads the last row of one sequence, or every row of a batch.
    /// </summary>
    private void Forward(GpuSequence[] seqs, int rows, bool head, bool allRows = false, bool hidden = false, bool snap = false)
    {
        bool batch = seqs.Length > 1;
        Check(Native.amqe_embed(_engine, _embed.Ptr, _embed.Type, _tokens, rows, _hidden, _embedScale, _hasEmbedScale ? 1 : 0, _x), "embed");
        for (int i = 0; i < _layers.Length; i++)
        {
            Layer(_layers[i], i, seqs, rows, null, snap);
        }
        if (hidden)
        {
            // Every row's final-normed hidden state, for the drafter.
            ApplyNorm((_finalNormW, _finalNormB, _finalNormOff, _finalNormEps, _finalNormKind), _x, _hfin, rows);
        }

        if (head)
        {
            // One sequence: only its last row goes through the final norm and
            // the head. A batch: every row, each its own sequence's next token.
            var finalNorm = (_finalNormW, _finalNormB, _finalNormOff, _finalNormEps, _finalNormKind);
            int headRows = batch || allRows ? rows : 1;
            IntPtr first = batch || allRows ? _x : _x + (rows - 1) * _hidden * 4;
            if (FusesNorm(headRows, finalNorm))
            {
                Linear(_head, first, _hidden, headRows, _logits, _vocab, IntPtr.Zero, norm: finalNorm);
            }
            else
            {
                ApplyNorm(finalNorm, first, _hn, headRows);
                Linear(_head, _hn, _hidden, headRows, _logits, _vocab, IntPtr.Zero);
            }
            var output = _plan.Output!;
            if (output.Multiplier is not null || output.LogitSoftcapping is not null)
            {
                Check(Native.amqe_logit_post(_engine, _logits, (long)headRows * _vocab, (float)(output.Multiplier ?? 1.0),
                    output.Multiplier is null ? 0 : 1, output.LogitSoftcapping ?? 0f), "logit post");
            }
        }
        foreach (var seq in seqs)
        {
            Check(Native.amqe_advance(_engine, seq.Pos, batch ? 1 : rows), "advance");
        }
    }

    // ── calibration and 4-bit rounding ──────────────────────────────────────

    /// <summary>The inputs a layer's projections read, by what reads them: the
    /// mixer's in-projections (the normed residual), its out-projection (the
    /// attention context or Gated DeltaNet output), the FFN's gate and up (the
    /// normed residual again), and its down-projection (the gated activation).</summary>
    public const int GramMixerIn = 0, GramMixerOut = 1, GramFfnIn = 2, GramDownIn = 3;

    private int _gramLayer = -1;
    private readonly IntPtr[] _grams = new IntPtr[4];
    private readonly int[] _gramDims = new int[4];
    private long _gramRows;

    /// <summary>
    /// Starts collecting XᵀX of the inputs each projection of layer
    /// <paramref name="layer"/> reads, over every row prompts then run through
    /// it (<see cref="Prefill(GpuSequence, ReadOnlySpan{int})"/> with chunks of
    /// more than 8 rows, where norms run apart from the projections).
    /// <see cref="EndGrams"/> returns them.
    /// </summary>
    public void BeginGrams(int layer)
    {
        EndGrams(out _);
        _gramLayer = layer;
        _gramRows = 0;
    }

    /// <summary>The Grams collected since <see cref="BeginGrams"/> (k × k each,
    /// indexed by <see cref="GramMixerIn"/> and the rest; null where no row
    /// reached that input), and how many rows they sum.</summary>
    public float[]?[] EndGrams(out long rows)
    {
        var result = new float[]?[4];
        Check(Native.amqe_sync(_engine), "grams");
        for (int kind = 0; kind < 4; kind++)
        {
            if (_grams[kind] == IntPtr.Zero)
            {
                continue;
            }
            int k = _gramDims[kind];
            var h = new float[(long)k * k];
            Download(h, _grams[kind]);
            Native.amqe_free(_grams[kind]);
            _grams[kind] = IntPtr.Zero;
            result[kind] = h;
        }
        rows = _gramRows;
        _gramLayer = -1;
        return result;
    }

    private void Gram(int kind, IntPtr x, int ldx, int rows, int k)
    {
        if (_grams[kind] == IntPtr.Zero)
        {
            nuint bytes = (nuint)((long)k * k * 4);
            Check(Native.amqe_alloc(out _grams[kind], bytes), "gram");
            Check(Native.amqe_memset(_engine, _grams[kind], 0, bytes), "gram");
            _gramDims[kind] = k;
        }
        else if (_gramDims[kind] != k)
        {
            throw new InvalidOperationException($"gram {kind} is {_gramDims[kind]} wide, a projection read {k}");
        }
        Check(Native.amqe_gram(_engine, x, ldx, rows, k, _grams[kind]), "gram");
    }

    /// <summary>
    /// A weight ([rows × cols] f32, row-major) in 4 bits: FP4 codes two to a
    /// byte (low nibble first) and a scale byte per block (NVFP4: E4M3 per 16,
    /// times <paramref name="global"/>; MXFP4: E8M0 per 32).
    /// <para>With a Gram <paramref name="hessian"/> (cols × cols, from
    /// <see cref="EndGrams"/>) this is GPTQ: each column's rounding error is
    /// carried onto the columns after it, so the layer's output over those
    /// inputs moves least; <paramref name="damp"/> × the mean diagonal is added
    /// to keep H invertible. Without one, each element rounds to nearest.
    /// Either way each block takes the scale, among those near its peak's,
    /// that leaves the least (weighted) error.</para>
    /// </summary>
    public (byte[] Codes, byte[] Scales) QuantiseFp4(float[] weights, int rows, int cols, float[]? hessian,
        bool mx, float global, double damp = 0.01)
    {
        if (cols % 32 != 0 || weights.LongLength != (long)rows * cols)
        {
            throw new ArgumentException($"[{rows}, {cols}] needs a multiple of 32 columns and as many weights");
        }
        long kk = (long)cols * cols;
        var w = weights;
        var h = new float[kk];
        if (hessian is null)
        {
            for (int i = 0; i < cols; i++)
            {
                h[(long)i * cols + i] = 1f;
            }
        }
        else
        {
            if (hessian.LongLength != kk)
            {
                throw new ArgumentException($"a Gram for {cols} columns has {kk} entries");
            }
            Array.Copy(hessian, h, kk);
            double mean = 0;
            for (int i = 0; i < cols; i++)
            {
                mean += h[(long)i * cols + i];
            }
            mean /= cols;
            for (int i = 0; i < cols; i++)
            {
                long d = (long)i * cols + i;
                if (h[d] == 0f)
                {
                    // No input ever reached this column: its weight is never used.
                    h[d] = 1f;
                    if (w == weights)
                    {
                        w = (float[])weights.Clone();
                    }
                    for (int r = 0; r < rows; r++)
                    {
                        w[(long)r * cols + i] = 0f;
                    }
                }
                h[d] += (float)(damp * mean);
            }
        }
        int group = mx ? 32 : 16;
        var codes = new byte[(long)rows * cols / 2];
        var scales = new byte[(long)rows * cols / group];
        IntPtr dw = IntPtr.Zero, dh = IntPtr.Zero, du = IntPtr.Zero, dc = IntPtr.Zero, ds = IntPtr.Zero, de = IntPtr.Zero;
        try
        {
            Check(Native.amqe_alloc(out dh, (nuint)(kk * 4)), "gptq");
            Check(Native.amqe_alloc(out du, (nuint)(kk * 4)), "gptq");
            int rc = 1;
            for (int attempt = 0; attempt < 4 && rc != 0; attempt++)
            {
                if (attempt > 0)
                {
                    // Not positive definite in f32: more damping.
                    for (int i = 0; i < cols; i++)
                    {
                        h[(long)i * cols + i] *= 2f;
                    }
                }
                Upload(dh, h);
                rc = Native.amqe_gptq_prepare(_engine, dh, cols, du);
                if (rc < 0)
                {
                    Check(rc, "gptq prepare");
                }
            }
            if (rc != 0)
            {
                throw new InvalidOperationException($"the Gram for a [{rows}, {cols}] weight stayed singular after damping");
            }
            Native.amqe_free(dh);
            dh = IntPtr.Zero;
            Check(Native.amqe_alloc(out dw, (nuint)(w.LongLength * 4)), "gptq");
            Check(Native.amqe_alloc(out dc, (nuint)codes.LongLength), "gptq");
            Check(Native.amqe_alloc(out ds, (nuint)scales.LongLength), "gptq");
            Check(Native.amqe_alloc(out de, (nuint)((long)rows * 32 * 4)), "gptq");
            Upload(dw, w);
            Check(Native.amqe_gptq(_engine, dw, rows, cols, du, mx ? 1 : 0, global, dc, ds, de), "gptq");
            Download(codes, dc);
            Download(scales, ds);
        }
        finally
        {
            foreach (var p in new[] { dw, dh, du, dc, ds, de })
            {
                if (p != IntPtr.Zero)
                {
                    Native.amqe_free(p);
                }
            }
        }
        return (codes, scales);
    }

    private void Upload<T>(IntPtr device, T[] host) where T : unmanaged
    {
        var handle = GCHandle.Alloc(host, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_upload(_engine, device, handle.AddrOfPinnedObject(),
                (nuint)(host.LongLength * System.Runtime.CompilerServices.Unsafe.SizeOf<T>())), "upload");
        }
        finally
        {
            handle.Free();
        }
    }

    private void Download<T>(T[] host, IntPtr device) where T : unmanaged
    {
        var handle = GCHandle.Alloc(host, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_download(_engine, handle.AddrOfPinnedObject(), device,
                (nuint)(host.LongLength * System.Runtime.CompilerServices.Unsafe.SizeOf<T>())), "download");
        }
        finally
        {
            handle.Free();
        }
    }


    /// <summary>The drafter's attention target: its own KV cache and position
    /// in place of the sequence's layer <c>i</c>.</summary>
    private readonly record struct KvTarget(IntPtr K, IntPtr V, IntPtr Pos, int HostPos);

    /// <summary>One layer over the rows in <c>_x</c> (in place). With
    /// <paramref name="kv"/> its attention reads and writes that cache instead
    /// of the sequence's; with <paramref name="snap"/> each Gated DeltaNet keeps
    /// its state after every row but the last (a single sequence's verify).</summary>
    private void Layer(LayerDevice l, int i, GpuSequence[] seqs, int rows, KvTarget? kv, bool snap)
    {
        int hidden = _hidden;
        bool batch = seqs.Length > 1;
        float residualScale = (float)_plan.ResidualScale;
        var lp = l.Plan;
        bool parallel = lp.ParallelResidual;

        IntPtr hIn = _x;
        (IntPtr, IntPtr, float, float, int)? inNorm = null;
        if (l.PreAttn is { } pre)
        {
            if (!parallel && FusesNorm(rows, pre))
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
        bool gram = _gramLayer >= 0 && i == _gramLayer;
        if (gram)
        {
            if (inNorm is not null)
            {
                throw new InvalidOperationException("calibration needs prompt chunks (the norm runs apart)");
            }
            Gram(GramMixerIn, hIn, hidden, rows, hidden);
            _gramRows += rows;
        }
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
            float softcap = attn.LogitSoftcapping ?? 0f;
            int window = attn.Window is { } w ? (int)w : 0;
            long qRow = (long)attn.NumQHeads * attn.HeadDim * 4;
            if (batch)
            {
                // Every sequence in one launch each: row r reads its own cache and position.
                Check(Native.amqe_attn_prep_batch(_engine, _proj, qkvWidth, rows, attn.NumQHeads, attn.NumKvHeads, attn.HeadDim,
                    attn.OutputGate ? 1 : 0, _q, _gate, Table(i, 0), Table(i, 1), Table(i, 2),
                    l.QNormRow ? IntPtr.Zero : l.QNormW, l.QNormOff, l.QNormEps,
                    l.KNormRow ? IntPtr.Zero : l.KNormW, l.KNormOff, l.KNormEps,
                    Table(i, 5), l.RopePairs, l.RopeScale, attn.RopeInterleaved ? 1 : 0, l.InvFreq != IntPtr.Zero ? 1 : 0), "attn prep");
                Check(Native.amqe_attention_batch(_engine, _q, Table(i, 0), Table(i, 1), attn.OutputGate ? _gate : IntPtr.Zero, _ctx,
                    Table(i, 2), rows, attn.NumQHeads, attn.NumKvHeads, attn.HeadDim, (float)attn.ScoreScale, softcap, window,
                    _attnPart, seqs.Max(Splits)), "attention");
            }
            for (int s = 0; s < (batch ? 0 : seqs.Length); s++)
            {
                var seq = seqs[s];
                int n = batch ? 1 : rows;
                long at = batch ? s : 0;
                IntPtr q = _q + (nint)(at * qRow), gateRow = _gate + (nint)(at * qRow), ctx = _ctx + (nint)(at * qRow);
                Check(Native.amqe_attn_prep(_engine, _proj + (nint)(at * qkvWidth * 4), qkvWidth, n, attn.NumQHeads, attn.NumKvHeads,
                    attn.HeadDim, attn.OutputGate ? 1 : 0, q, gateRow, kv?.K ?? seq.K[i], kv?.V ?? seq.V[i], kv?.Pos ?? seq.Pos,
                    l.QNormRow ? IntPtr.Zero : l.QNormW, l.QNormOff, l.QNormEps,
                    l.KNormRow ? IntPtr.Zero : l.KNormW, l.KNormOff, l.KNormEps,
                    InvFreq(l, seq), l.RopePairs, l.RopeScale, attn.RopeInterleaved ? 1 : 0, l.InvFreq != IntPtr.Zero ? 1 : 0), "attn prep");
                if (_probeFrom >= 0 && kv is null && QueryProbe is { } probe)
                {
                    // Prompt chunks only: never inside a captured graph.
                    var host = new float[(long)n * attn.NumQHeads * attn.HeadDim];
                    var h = GCHandle.Alloc(host, GCHandleType.Pinned);
                    try
                    {
                        Check(Native.amqe_download(_engine, h.AddrOfPinnedObject(), q, (nuint)(host.LongLength * 4)), "query probe");
                    }
                    finally
                    {
                        h.Free();
                    }
                    probe(i, _probeFrom, n, host);
                }
                IntPtr gate = attn.OutputGate ? gateRow : IntPtr.Zero;
                // A prompt chunk goes through cuBLAS (its position is known on
                // the host: prompts are never captured); decode steps, and
                // chunks whose scores would pass the scratch budget, use the
                // attention kernels.
                int rc = n > 8 && RowsAttention
                    ? Native.amqe_attention_rows(_engine, q, kv?.K ?? seq.K[i], kv?.V ?? seq.V[i], gate, ctx, kv?.HostPos ?? seq.Position,
                        n, attn.NumQHeads, attn.NumKvHeads, attn.HeadDim, (float)attn.ScoreScale, softcap, window, ExactSpec(seq))
                    : -200;
                if (rc == -200)
                {
                    rc = Native.amqe_attention(_engine, q, kv?.K ?? seq.K[i], kv?.V ?? seq.V[i], gate, ctx, kv?.Pos ?? seq.Pos,
                        n, attn.NumQHeads, attn.NumKvHeads, attn.HeadDim, (float)attn.ScoreScale,
                        softcap, window, _attnPart, Splits(seq));
                }
                Check(rc, "attention");
            }
            if (gram)
            {
                Gram(GramMixerOut, _ctx, attn.QDim, rows, attn.QDim);
            }
            Linear(l.O!, _ctx, attn.QDim, rows, mixerOut, hidden, l.OBias,
                residual: mixerDirect ? _x : IntPtr.Zero, resScale: residualScale);
        }
        else
        {
            var lin = lp.LinearAttention!;
            int inWidth = l.In!.Rows;
            Linear(l.In!, hIn, hidden, rows, _proj, inWidth, IntPtr.Zero, norm: inNorm);
            if (batch)
            {
                Check(Native.amqe_gdn_batch(_engine, _proj, inWidth, rows, lin.KeyDim, lin.ValueDim,
                    lin.NumKHeads, lin.NumVHeads, lin.HeadKDim, lin.HeadVDim, l.ConvW, lin.ConvKernel, Table(i, 4),
                    l.ALog, l.DtBias, l.GdnNormW, (float)lin.NormEps, Table(i, 3), _conv, _qk, _gb, _gdn), "gated deltanet");
            }
            for (int s = 0; s < (batch ? 0 : seqs.Length); s++)
            {
                long at = batch ? s : 0;
                Check(Native.amqe_gdn(_engine, _proj + (nint)(at * inWidth * 4), inWidth, batch ? 1 : rows, lin.KeyDim, lin.ValueDim,
                    lin.NumKHeads, lin.NumVHeads, lin.HeadKDim, lin.HeadVDim, l.ConvW, lin.ConvKernel, seqs[s].Conv[i],
                    l.ALog, l.DtBias, l.GdnNormW, (float)lin.NormEps, seqs[s].State[i], _conv, _qk, _gb,
                    _gdn + (nint)(at * lin.ValueDim * 4),
                    snap ? seqs[s].SnapState![i] : IntPtr.Zero, snap ? seqs[s].SnapConv![i] : IntPtr.Zero), "gated deltanet");
            }
            if (gram)
            {
                Gram(GramMixerOut, _gdn, lin.ValueDim, rows, lin.ValueDim);
            }
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
            return;
        }

        IntPtr fIn = parallel ? hIn : _x;
        (IntPtr, IntPtr, float, float, int)? ffnNorm = null;
        if (!parallel && l.PreFfn is { } preFfn)
        {
            if (FusesNorm(rows, preFfn))
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
        if (gram)
        {
            if (ffnNorm is not null)
            {
                throw new InvalidOperationException("calibration needs prompt chunks (the norm runs apart)");
            }
            Gram(GramFfnIn, fIn, hidden, rows, hidden);
        }
        Linear(l.Up, fIn, hidden, rows, _ffn, inter, l.UpBias, act: l.Act, glu: dense.IsGated, n: inter, norm: ffnNorm);
        if (gram)
        {
            Gram(GramDownIn, _ffn, inter, rows, inter);
        }
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
            norm is null ? IntPtr.Zero : nm.W, nm.Off, nm.B, nm.Eps, nm.Kind, w.Scales, w.RowScale), "linear");
    }

    /// <summary>Whether a norm can run inside the projection that reads it: a
    /// few rows (the GEMV path), where an RMS norm without a bias is applied in
    /// the dot loop and any other norm's normalised copy must fit in shared memory.</summary>
    private bool FusesNorm(int rows, (IntPtr W, IntPtr B, float Off, float Eps, int Kind) norm) =>
        rows < GemmFrom && ((norm.Kind == 0 && norm.B == IntPtr.Zero) || (long)rows * _hidden * 4 <= 48 * 1024);

    // Rows from which a projection runs on cuBLAS rather than the GEMV (AMQE_GEMM_FROM, as the native side reads it).
    private static readonly int GemmFrom =
        int.TryParse(Environment.GetEnvironmentVariable("AMQE_GEMM_FROM"), out int g) && g >= 2 ? g : 9;

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

        /// <summary>4-bit weights (types 2, 3): the block scales, and each
        /// row's tensor scale (zero when every row's is 1).</summary>
        public IntPtr Scales { get; init; }
        public IntPtr RowScale { get; init; }
    }

    private const int NVFP4 = 2, MXFP4 = 3;

    /// <summary>A 4-bit weight on its way to the device: packed elements,
    /// block-scale bytes, and each row's tensor scale.</summary>
    private sealed record Fp4Source(byte[] Packed, byte[] Scales, float[] RowScale, int Type);

    /// <summary>The form the engine holds projections in. A container whose
    /// projections are stored as NVFP4 or MXFP4 (see <c>amql-cli quantise</c>)
    /// runs them as stored whatever this says; otherwise NVFP4 or MXFP4 here
    /// quantises them at load (<c>AMQL_GPU_FORMAT=nvfp4|mxfp4</c>). The
    /// embedding stays as stored, and so does the head unless
    /// <c>AMQL_GPU_FORMAT_HEAD=1</c>.</summary>
    public static GpuWeightFormat WeightFormat { get; set; } = ComputeBudget.GpuWeightFormat;

    public static bool QuantiseHead { get; set; } = Environment.GetEnvironmentVariable("AMQL_GPU_FORMAT_HEAD") is "1";

    /// <summary>Matrices the engine holds at 4 bits, and their device bytes.</summary>
    public int Fp4Matrices { get; private set; }
    public long Fp4Bytes { get; private set; }

    private Fp4Source? Fp4Of(WeightLoader weights, OperandRef operand, int rows, int cols, bool project)
    {
        if (weights.StoredFp4(operand) is { } stored)
        {
            if (stored.Rows != rows || stored.Cols != cols)
            {
                throw new ContainerException($"'{operand.ObjectId}/{operand.TensorName}' is [{stored.Rows}, {stored.Cols}], expected [{rows}, {cols}]");
            }
            return FromScaled(stored, rows);
        }
        if (!project || WeightFormat == GpuWeightFormat.Fp16 || cols % 32 != 0)
        {
            return null;
        }
        // From the stored values, not a BF16-resident copy: the same 4 bits a
        // compiled container holds.
        var values = weights.StoredValues(operand, rows, cols);
        if (WeightFormat == GpuWeightFormat.Mxfp4)
        {
            var mx = Mxfp4.Quantize(values, rows, cols);
            return new Fp4Source(mx.Packed, mx.BlockScales, Enumerable.Repeat(1f, rows).ToArray(), MXFP4);
        }
        return FromScaled(Nvfp4.Quantize(values, rows, cols), rows);
    }

    /// <summary>A stored or freshly quantised FP4 matrix in the engine's
    /// layout, or null when its blocks are neither NVFP4's nor MXFP4's.</summary>
    private static Fp4Source? FromScaled(ScaledMatrix m, int rows)
    {
        if (m.Element != Dtype.FP4 || m.BlockRows != 1)
        {
            return null;
        }
        if (m.IsNvfp4)
        {
            var bytes = new byte[m.Scales.Length];
            for (int i = 0; i < bytes.Length; i++)
            {
                bytes[i] = BitPattern.EncodeF8E4M3(m.Scales[i]);
            }
            return new Fp4Source(m.Elements, bytes, Enumerable.Repeat(m.Global, rows).ToArray(), NVFP4);
        }
        if (m.BlockCols == Mxfp4.BlockElements && m.ScaleCols * Mxfp4.BlockElements == m.Cols)
        {
            var bytes = new byte[m.Scales.Length];
            for (int i = 0; i < bytes.Length; i++)
            {
                float s = m.Scales[i];
                int bits = BitConverter.SingleToInt32Bits(s);
                if (s <= 0 || (bits & 0x007FFFFF) != 0)
                {
                    return null;   // not a power of two: not E8M0
                }
                bytes[i] = (byte)((bits >> 23) & 0xFF);
            }
            return new Fp4Source(m.Elements, bytes, Enumerable.Repeat(m.Global, rows).ToArray(), MXFP4);
        }
        return null;
    }

    private static float[] Widen(ushort[] bits)
    {
        var f = new float[bits.Length];
        for (int i = 0; i < f.Length; i++)
        {
            f[i] = BitConverter.Int32BitsToSingle(bits[i] << 16);
        }
        return f;
    }

    private DeviceMatrix UploadFp4(Fp4Source source, int rows, int cols)
    {
        var matrix = new DeviceMatrix
        {
            Ptr = UploadBytes(source.Packed),
            Type = source.Type,
            Rows = rows,
            Cols = cols,
            Scales = UploadBytes(source.Scales),
            RowScale = source.RowScale.All(v => v == 1f) ? IntPtr.Zero : UploadFloats(source.RowScale),
        };
        Fp4Matrices++;
        Fp4Bytes += source.Packed.LongLength + source.Scales.LongLength + (matrix.RowScale == IntPtr.Zero ? 0 : rows * 4L);
        return matrix;
    }

    private IntPtr UploadBytes(byte[] data)
    {
        var ptr = Alloc(Math.Max(16, data.LongLength));
        var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_upload(_engine, ptr, handle.AddrOfPinnedObject(), (nuint)data.LongLength), "upload");
        }
        finally
        {
            handle.Free();
        }
        return ptr;
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
        public IntPtr InvFreqLong;   // for sequences past the trained length; InvFreq without a context extension
        public int RopePairs;
        public double RopeScale = 1.0;
        public IntPtr ConvW, ALog, DtBias, GdnNormW;
        public long StateBytes, ConvHistoryBytes;
        public int Act;
    }

    /// <summary>
    /// The model's own multi-token-prediction drafter (Qwen 3.5, Qwen3-Next):
    /// one decoder layer shaped like the stack's full-attention layers, which
    /// reads the final-normed hidden state at position t beside the embedding
    /// of token t+1, <c>fc([norm_e(embed(t+1)) ; norm_h(h_t)])</c>, and through
    /// its own final norm and the shared head predicts token t+2. It keeps a KV
    /// cache of its own, one row per position.
    /// </summary>
    private sealed class MtpDevice
    {
        public required LayerDevice Layer;
        public required DeviceMatrix Fc;
        public required (IntPtr W, IntPtr B, float Off, float Eps, int Kind) PreEmbed, PreHidden, Norm;
        public required int KvDim;

        private const string Stack = "mtp.stack";

        public static MtpDevice? TryUpload(GpuEngine engine, WeightLoader weights, ComponentOpPlan plan)
        {
            var fc = new OperandRef(Stack, "fc.weight");
            if (!weights.Contains(fc) || plan.FinalNorm is not { } finalNorm)
            {
                return null;
            }
            // The drafter's layer has the shape of the stack's full-attention layers.
            int trunk = plan.Layers.FindLastIndex(l => l.Attention is not null && l.Ffn?.Dense is not null);
            if (trunk < 0)
            {
                return null;
            }
            var src = plan.Layers[trunk];
            string prefix = $"{trunk}.";
            OperandRef Rebind(OperandRef r) => r.TensorName.StartsWith(prefix, StringComparison.Ordinal)
                ? new OperandRef(Stack, "layers.0." + r.TensorName[prefix.Length..])
                : throw new UnsupportedOperatorException($"MTP: cannot rebind {r.ObjectId}/{r.TensorName}");
            OperandRef? RebindOpt(OperandRef? r) => r is null ? null : Rebind(r);
            NormOp? N(NormOp? n) => n is null ? null : new NormOp
            {
                Weight = Rebind(n.Weight), Bias = RebindOpt(n.Bias), Kind = n.Kind, Eps = n.Eps,
                WeightOffset = n.WeightOffset, Width = n.Width,
            };
            var a = src.Attention!;
            var d = src.Ffn!.Dense!;
            var layer = new LayerPlan
            {
                PreAttentionNorm = N(src.PreAttentionNorm),
                PostAttentionNorm = N(src.PostAttentionNorm),
                PreFfnNorm = N(src.PreFfnNorm),
                PostFfnNorm = N(src.PostFfnNorm),
                ParallelResidual = src.ParallelResidual,
                Attention = new AttentionOp
                {
                    NumQHeads = a.NumQHeads, NumKvHeads = a.NumKvHeads, HeadDim = a.HeadDim,
                    ScoreScale = a.ScoreScale, LogitSoftcapping = a.LogitSoftcapping, Window = a.Window,
                    Causal = a.Causal, Position = a.Position, VFromK = a.VFromK, OutputGate = a.OutputGate,
                    QkNormScope = a.QkNormScope, QkNormWeightOffset = a.QkNormWeightOffset,
                    ParameterFreeQkNorm = a.ParameterFreeQkNorm, ParameterFreeQkNormEps = a.ParameterFreeQkNormEps,
                    QNorm = N(a.QNorm), KNorm = N(a.KNorm),
                    QProj = Rebind(a.QProj), KProj = Rebind(a.KProj), VProj = Rebind(a.VProj), OProj = Rebind(a.OProj),
                    QBias = RebindOpt(a.QBias), KBias = RebindOpt(a.KBias), VBias = RebindOpt(a.VBias), OBias = RebindOpt(a.OBias),
                    RopeInterleaved = a.RopeInterleaved,
                },
                Ffn = new LayerFfn
                {
                    Dense = new DenseFfnOp
                    {
                        Up = Rebind(d.Up), Down = Rebind(d.Down), Gate = RebindOpt(d.Gate),
                        GateBias = RebindOpt(d.GateBias), UpBias = RebindOpt(d.UpBias), DownBias = RebindOpt(d.DownBias),
                        Activation = d.Activation, IntermediateSize = d.IntermediateSize, HiddenSize = d.HiddenSize,
                    },
                },
            };
            if (a.HeadGate is not null || a.Indexer is not null || a.VFromK)
            {
                return null;
            }
            foreach (var name in new[] { "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight", "layers.0.self_attn.q_proj.weight" })
            {
                if (!weights.Contains(new OperandRef(Stack, name)))
                {
                    return null;
                }
            }
            NormOp Top(string name) => new()
            {
                Weight = new OperandRef(Stack, name), Kind = finalNorm.Kind, Eps = finalNorm.Eps,
                WeightOffset = finalNorm.WeightOffset, Width = finalNorm.Width,
            };
            int hidden = plan.HiddenSize;
            return new MtpDevice
            {
                Layer = engine.UploadLayer(weights, layer),
                Fc = engine.Matrix(weights, fc, hidden, 2 * hidden, project: true),
                PreEmbed = engine.Norm(weights, Top("pre_fc_norm_embedding.weight")),
                PreHidden = engine.Norm(weights, Top("pre_fc_norm_hidden.weight")),
                Norm = engine.Norm(weights, Top("norm.weight")),
                KvDim = a.KvDim,
            };
        }
    }

    private DeviceMatrix Matrix(WeightLoader weights, OperandRef operand, int rows, int cols, bool project = false)
    {
        var fp4 = Fp4Of(weights, operand, rows, cols, project);
        var key = (operand.ObjectId, operand.TensorName + (fp4 is null ? "" : "#fp4"));
        if (_matrixCache.TryGetValue(key, out var cached))
        {
            return cached;
        }
        if (fp4 is not null)
        {
            return _matrixCache[key] = UploadFp4(fp4, rows, cols);
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
        int rows = parts.Sum(p => p.Rows);
        // 4-bit when every part is, in one format: elements and block scales
        // stack by rows, and each row keeps its own part's tensor scale.
        var fp4 = parts.Select(p => Fp4Of(weights, p.Operand, p.Rows, cols, project: true)).ToArray();
        if (fp4.All(f => f is not null) && fp4.Select(f => f!.Type).Distinct().Count() == 1)
        {
            var packed = fp4.SelectMany(f => f!.Packed).ToArray();
            var scales = fp4.SelectMany(f => f!.Scales).ToArray();
            var rowScale = fp4.SelectMany(f => f!.RowScale).ToArray();
            return UploadFp4(new Fp4Source(packed, scales, rowScale, fp4[0]!.Type), rows, cols);
        }
        var sources = parts.Select(p => weights.DeviceSource(p.Operand, p.Rows, cols)).ToArray();
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

    private (IntPtr K, IntPtr V) AllocPair(long bytes)
    {
        var k = Alloc(bytes);
        try
        {
            return (k, Alloc(bytes));
        }
        catch
        {
            Free(k);
            throw;
        }
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
        if (_hostPrefillLogits != IntPtr.Zero)
        {
            Native.amqe_host_free(_hostPrefillLogits);
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

        [DllImport(Lib)] public static extern int amqe_set_prefill_f32(IntPtr e, int f32);
        [DllImport(Lib)] public static extern int amqe_set_kv_format(IntPtr e, int fmt);
        [DllImport(Lib)] public static extern long amqe_kv_row_bytes(int hkv, int d, int fmt);
        [DllImport(Lib)] public static extern int amqe_capture_begin(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_capture_end(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_capture_drop(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_graph_launch(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_graph_launch_async(IntPtr e);

        [DllImport(Lib)]
        public static extern int amqe_linear(IntPtr e, IntPtr w, int wType, int n, int k,
            IntPtr x, int ldx, int rows, IntPtr y, int ldy,
            IntPtr bias, IntPtr residual, int ldr, float resScale, int act, int glu, IntPtr raw,
            IntPtr normW, float normOff, IntPtr normBias, float normEps, int normKind, IntPtr wScales, IntPtr rowScale);

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
            IntPtr state, IntPtr conv, IntPtr qk, IntPtr gb, IntPtr output, IntPtr stateSnap, IntPtr histSnap);

        [DllImport(Lib)] public static extern int amqe_set_int(IntPtr e, IntPtr p, int v);
        [DllImport(Lib)] public static extern int amqe_argmax(IntPtr e, IntPtr x, int n, IntPtr output, IntPtr output2);
        [DllImport(Lib)] public static extern int amqe_set_row_invariant(IntPtr e, int on);
        [DllImport(Lib)] public static extern int amqe_gram(IntPtr e, IntPtr x, int ldx, int rows, int k, IntPtr h);
        [DllImport(Lib)] public static extern int amqe_gptq_prepare(IntPtr e, IntPtr h, int k, IntPtr u);
        [DllImport(Lib)]
        public static extern int amqe_gptq(IntPtr e, IntPtr w, int n, int k, IntPtr u, int mx, float global,
            IntPtr packed, IntPtr scales, IntPtr err);

        [DllImport(Lib)]
        public static extern int amqe_logit_post(IntPtr e, IntPtr logits, long count, float mult, int hasMult, float cap);

        [DllImport(Lib)] public static extern int amqe_advance(IntPtr e, IntPtr pos, int by);

        [DllImport(Lib)]
        public static extern int amqe_attn_prep_batch(IntPtr e, IntPtr qkv, int ldqkv, int batch, int hq, int hkv, int d, int gated,
            IntPtr q, IntPtr gate, IntPtr kCaches, IntPtr vCaches, IntPtr poss,
            IntPtr qNormW, float qNormOff, float qNormEps,
            IntPtr kNormW, float kNormOff, float kNormEps,
            IntPtr invFreq, int pairs, double ropeScale, int interleaved, int rope);

        [DllImport(Lib)]
        public static extern int amqe_attention_batch(IntPtr e, IntPtr q, IntPtr kCaches, IntPtr vCaches,
            IntPtr gate, IntPtr output, IntPtr poss, int batch, int hq, int hkv, int d,
            float scale, float softcap, int window, IntPtr part, int nsplit);

        [DllImport(Lib)]
        public static extern int amqe_gdn_batch(IntPtr e, IntPtr proj, int ldp, int batch, int keyDim, int valueDim,
            int nk, int nv, int kd, int vd, IntPtr convW, int kernel, IntPtr histories,
            IntPtr aLog, IntPtr dtBias, IntPtr normW, float normEps,
            IntPtr states, IntPtr conv, IntPtr qk, IntPtr gb, IntPtr output);
        [DllImport(Lib)] public static extern int amqe_attention_split_keys();
        [DllImport(Lib)] public static extern int amqe_attention_part_slots(int nsplit);
        [DllImport(Lib)] public static extern int amqe_set_attention_mass(IntPtr e, IntPtr output, IntPtr scratch, int splits);
        [DllImport(Lib)] public static extern int amqe_set_attention_segments(int splits, int keys, long budget);

        [DllImport(Lib)]
        public static extern int amqe_attention_rows(IntPtr e, IntPtr q, IntPtr kCache, IntPtr vCache,
            IntPtr gate, IntPtr output, int pos0, int rows, int hq, int hkv, int d,
            float scale, float softcap, int window, int exact);
    }
}

/// <summary>One sequence on a <see cref="GpuEngine"/>: its KV caches, recurrent
/// state and position, beside the engine's other sequences and over its one
/// copy of the weights.</summary>
public sealed class GpuSequence
{
    internal GpuSequence(int layers)
    {
        K = new IntPtr[layers];
        V = new IntPtr[layers];
        State = new IntPtr[layers];
        Conv = new IntPtr[layers];
    }

    internal IntPtr Pos;
    internal readonly IntPtr[] K, V, State, Conv;
    internal IntPtr[]? MarkState, MarkConv;
    internal int KvCapacity;
    internal bool Released;

    // The MTP drafter's side: its KV cache and position, the final-normed hidden
    // state of the last position (which pairs with the token chosen after it),
    // and whether its cache has kept up with every position (a batch step does
    // not update it).
    internal IntPtr MtpK, MtpV, MtpPos, MtpPending, MarkMtpPending;
    internal bool MtpValid, MarkMtpValid;

    // A verify's recurrent state after each of its rows but the last, by layer,
    // so keeping only its first rows restores the state there.
    internal IntPtr[]? SnapState, SnapConv;
    internal int SnapRows;
    internal int VerifyFrom, VerifyRows;

    /// <summary>Positions kept at f32 in an NVFP4 cache (see <see cref="GpuEngine.SetExactRows"/>).</summary>
    public int ExactRows { get; internal set; }

    /// <summary>Whether the sequence runs at the model's extended rotary
    /// frequencies (see <see cref="GpuEngine.RopeExtendsBeyond"/>).</summary>
    public bool Extended { get; internal set; }
    internal bool RopeChosen;

    /// <summary>Whether the exact rows are FP16 rather than f32.</summary>
    public bool ExactF16 { get; internal set; }

    /// <summary>The position <see cref="GpuEngine.Mark"/> last saved, or -1:
    /// <see cref="GpuEngine.Rewind"/> takes the sequence back there.</summary>
    public int MarkedAt { get; internal set; } = -1;

    /// <summary>Tokens the sequence has consumed.</summary>
    public int Position { get; internal set; }
}
