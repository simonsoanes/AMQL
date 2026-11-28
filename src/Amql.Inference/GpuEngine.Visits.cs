using System.Runtime.InteropServices;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The engine's visit path: a plan that <see cref="ComponentOpPlan.RunsByVisit"/>
/// (a depth loop, a block residual, attention conditioned without a rotary
/// embedding, packed experts) run on the device, op for op what
/// <see cref="GenericRuntime"/>'s row-exact path runs on the CPU
/// (docs/depth-ops.md), over the primitives of <c>native/amql_cuda/amql_depth.cu</c>.
/// <para>
/// Each physical layer's weights go up once and every visit of it reads them:
/// the large matrices as stored (BF16) or f32, the small tensors f32, packed
/// experts in their stored formats (NVFP4 with its per-expert factor, MXFP8,
/// plain). A sequence holds every cache slot of the schedule: a windowed slot is a
/// ring of its window and a chunk more (f32 rows), a full slot every position in
/// the requested KV format (f32, fp16 or nvfp4), plus the slot's convolution
/// histories. A step reads its token and position from device memory, so its
/// launches are captured once per sequence as a CUDA graph and replayed; a prompt
/// runs in chunks of <see cref="DepthChunk"/> tokens.
/// </para>
/// <para>
/// What the path keeps fused: projections that share an input in one launch,
/// the q, k and v message gates' steps batched, the norm after a depth mix in the
/// mix's kernel, the router's top k in one warp, a step's experts on the
/// engine's decode kernels and a chunk's grouped on its tensor cores. It runs one
/// sequence at a time (no batching, marks or drafts).
/// </para>
/// </summary>
public sealed partial class GpuEngine
{
    /// <summary>Tokens a prompt chunk holds on the visit path.</summary>
    public const int DepthChunk = 256;

    // Key splits a prompt row's attention takes at most (a decode step takes more).
    private const int DepthRowSplits = 8;

    private const int KvF32 = 0, KvF16 = 1, KvFp4 = 2;
    private const int WNvfp4 = 2, WMxfp8 = 4;

    private VisitDevice? _visits;

    /// <summary>Why this engine runs one sequence at a time without marks, drafts
    /// or batching; null when it batches.</summary>
    public string? BatchRefusal => _visits is null ? null : "looped depth runs one sequence at a time on the GPU engine";

    /// <summary>False runs each visit-path step's launches directly instead of
    /// replaying its captured graph (<c>AMQL_GPU_GRAPHS=0</c>).</summary>
    public bool UseGraphs { get; set; } = Environment.GetEnvironmentVariable("AMQL_GPU_GRAPHS")?.Trim() is not "0";

    private static readonly bool TiledAttention = Environment.GetEnvironmentVariable("AMQL_TILED_ATTENTION")?.Trim() is not "0";

    // ── device structs (amql_depth.cu's Mat, Experts, KvView) ──────────────────

    [StructLayout(LayoutKind.Sequential)]
    private struct DepthMat
    {
        public IntPtr W;
        public int Type, Rows, Cols, Pad;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DepthExperts
    {
        public IntPtr W, Scales, Factor, RowScale;
        public int Type, Rows, Cols, Pad;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct DepthKv
    {
        public IntPtr K, V, Ks, Vs, Kg, Vg;
        public int Cap, Ring, Format, Pad;
    }

    /// <summary>One physical layer on the device, shared by every visit of it.</summary>
    private sealed class DepthLayer
    {
        public required LayerPlan Plan { get; init; }
        public DepthMat Wq, Wk, Wv, Wo, Router;
        public DepthMat? Rel, SinkProj, OutGate, KTaps, VTaps;
        public IntPtr QScale, RelScale, RelTable, SinkStatic;
        public IntPtr KConv, VConv;
        public int KMode, VMode, ConvKernel;
        public bool SeparateInputs;
        public IntPtr QNorm, KNorm, VNorm, FfnNorm;
        public float AttnEps, FfnEps;
        public IntPtr MixerQueries, FfnQueries;
        public int MixerCount;
        public float MixerMixEps, FfnMixEps;
        public IntPtr MixerConv, FfnConv;
        public int MixerConvKernel, FfnConvKernel;
        public bool OpensBlock;
        // Message gates, in the plan's order; their queries [gates x hidden] mix once a visit.
        public string[]? GateSites;
        public DepthMat[] Down = [], Forget = [], Gain = [];
        public IntPtr GateQueries;
        public float GateEps, GateMixEps;
        public int GateRank;
        // The attention's geometry and conditioning.
        public int Heads, KvHeads, HeadDim, Window, Extent, RelRank;
        public float Scale, QkClip, RelClip, VEps, Alpha;
        public int VNormOn, LogScale, Floor;
        // The experts.
        public IntPtr SelectionBias;
        public DepthExperts GateUp, DownExperts;
        public int Experts, TopK, Width;
        public float RouteScale, GateCap, UpCap;
    }

    /// <summary>One cache slot of a sequence: its rows and its convolutions' histories.</summary>
    internal sealed class DepthSlot
    {
        public DepthKv Kv;
        public IntPtr KHist, VHist, MixerHist, FfnHist;
        public int Layer, C;
        public bool Ring;
    }

    /// <summary>A sequence's state on the visit path.</summary>
    internal sealed class DepthSequence
    {
        public required DepthSlot[] Slots { get; init; }
        public required int Format { get; init; }
        public int Capacity;
        public IntPtr Part;     // the attention's split partials, sized for the capacity
        public IntPtr Graph;    // the step's captured pass, or zero
        public readonly List<IntPtr> Owned = new();
        public long Bytes;
    }

    /// <summary>What the visit path holds beside the layers: the model-wide
    /// tensors and one chunk's scratch.</summary>
    private sealed class VisitDevice
    {
        public required DepthLayer[] Layers { get; init; }
        public required IReadOnlyList<LayerVisit> Schedule { get; init; }
        public required int[] SlotLayer { get; init; }   // cache slot → physical layer
        public DepthMat Embed, Head;
        public IntPtr EmbedNorm;
        public bool EmbedNormed;
        public float EmbedEps;
        public IntPtr FinalNorm, OutputQuery;
        public float FinalEps, OutputMixEps;
        public float? Softcap;
        public int Vocab, MaxBank;
        // Scratch, DepthChunk rows each.
        public IntPtr[] H = [];
        public IntPtr Tokens, Q, Gate, Tmp, K, V, R, Sinks, RouterLogits, KDyn, VDyn, Mixed, Msg, Cat, GateVec,
            Attn, Xq, Xk, Xv, Nq, Nk, Nv, Res, FfnIn, FfnGated, Normed, Ffn, Act, ExpertPart, Logits, Wt, Rb, Every;
        public IntPtr Idx, Order, Offsets, GroupOrder, Groups;
        public IntPtr HostTokens, HostLogits;
        public int MaxHeads, MaxExtentHeads;
    }

    // ── the plan's ops the visit path takes ───────────────────────────────────

    /// <summary>Why the visit path cannot run this plan, or null when it can.</summary>
    private static string? VisitUnsupported(ComponentOpPlan plan)
    {
        if (GenericRuntime.RowPathRefusal(plan) is { } refusal)
        {
            return refusal;
        }
        if (plan.Embedding!.Scale is not null)
        {
            return "an embedding scale on the visit path";
        }
        if (plan.Output!.HiddenSize != plan.HiddenSize || plan.Embedding.HiddenSize != plan.HiddenSize)
        {
            return "head width differs from the residual";
        }
        if (plan.FinalNorm!.WeightOffset != 0f)
        {
            return "a final norm with a weight offset";
        }
        if (plan.BlockResidual is { } block && plan.Layers.Count / Math.Max(1, block.BlockSize) + 1 > 10)
        {
            return "a block residual of more than nine blocks";
        }
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            var layer = plan.Layers[l];
            var attn = layer.Attention!;
            foreach (var norm in new[] { layer.PreAttentionNorm, layer.QInputNorm, layer.KInputNorm, layer.VInputNorm, layer.PreFfnNorm })
            {
                if (norm is { WeightOffset: not 0f })
                {
                    return $"layer {l}: a norm with a weight offset";
                }
            }
            if ((layer.QInputNorm ?? layer.PreAttentionNorm) is null || layer.PreFfnNorm is null)
            {
                return $"layer {l}: an unnormalised attention or FFN input";
            }
            if (attn.HeadDim > 128 || attn.HeadDim % 4 != 0)
            {
                return $"layer {l}: attention head dim {attn.HeadDim} (the visit path takes up to 128, a multiple of 4)";
            }
            if (attn.RelativeBias is { Rank: > 32 })
            {
                return $"layer {l}: a relative bias of rank {attn.RelativeBias.Rank} (up to 32)";
            }
            if (attn.KeyConv is { } kc && attn.ValueConv is { } vc && kc.Kernel != vc.Kernel)
            {
                return $"layer {l}: key and value convolutions of different widths";
            }
            if (new[] { attn.KeyConv, attn.ValueConv, layer.MixerOutputConv, layer.FfnOutputConv }.Any(c => c is { Kernel: < 2 }))
            {
                return $"layer {l}: a residual convolution narrower than two taps";
            }
            if (attn.KeyConv is { Dynamic: true } dk && dk.Heads != attn.NumKvHeads ||
                attn.ValueConv is { Dynamic: true } dv && dv.Heads != attn.NumKvHeads)
            {
                return $"layer {l}: dynamic taps shared over other than the key/value heads";
            }
            if (attn.Window is > int.MaxValue / 2)
            {
                return $"layer {l}: window";
            }
            var routed = layer.Ffn!.Routed!;
            if (routed.NumExperts > 512 || routed.TopK > 32 || !routed.Packed!.InterleavedGateUp)
            {
                return $"layer {l}: packed experts beyond the visit path's router ({routed.NumExperts} experts, top {routed.TopK})";
            }
            if (layer.MessageGates is { } gates)
            {
                if (gates.Count > 4 || gates.Select(g => g.Input).Distinct().Count() != gates.Count ||
                    gates.Select(g => (g.Eps, g.MixEps, g.Rank)).Distinct().Count() > 1)
                {
                    return $"layer {l}: message gates other than one each at q, k, v and ffn, alike";
                }
            }
        }
        return null;
    }

    // ── upload ────────────────────────────────────────────────────────────────

    private DepthMat DepthMatrix(WeightLoader weights, OperandRef operand, int rows, int cols, int? readRows = null)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (!_matrixCache.TryGetValue(key, out var m))
        {
            var (bf16, f32) = weights.DeviceSource(operand, rows, cols);
            m = bf16 is not null
                ? new DeviceMatrix { Ptr = Upload(bf16), Type = BF16, Rows = rows, Cols = cols }
                : new DeviceMatrix { Ptr = UploadFloats(f32!), Type = F32, Rows = rows, Cols = cols };
            _matrixCache[key] = m;
        }
        if (m.Type is not (BF16 or F32) || m.Cols != cols)
        {
            throw new InvalidOperationException($"{operand.TensorName}: held as type {m.Type} [{m.Rows} x {m.Cols}], not BF16 or f32 [{rows} x {cols}]");
        }
        return new DepthMat { W = m.Ptr, Type = m.Type, Rows = readRows ?? rows, Cols = cols };
    }

    private DepthExperts DepthExpertStack(OperandStore store, PackedOperand operand, int experts, int rows, int cols)
    {
        var e = operand.Elements;
        var dtype = store.DtypeOf(e.ObjectId, e.TensorName);
        var shape = store.ShapeOf(e.ObjectId, e.TensorName);
        if (!shape.SequenceEqual(new long[] { experts, rows, cols }))
        {
            throw new ContainerException($"{e.ObjectId}/{e.TensorName}: expected [{experts}x{rows}x{cols}], found [{string.Join("x", shape)}]");
        }
        OperandResolution Part(OperandRef? r, string what) =>
            store.Resolve(r ?? throw new ContainerException($"{e.TensorName} is stored {dtype.Label()} but its '{what}' is not bound"));
        switch (dtype)
        {
            case Dtype.FP4:
            {
                if (cols % 16 != 0)
                {
                    throw new ContainerException($"{e.TensorName}: NVFP4 rows of {cols} are not whole blocks of 16");
                }
                var scale = Part(operand.Scale, "scale");
                var amax = BitPattern.WidenToF32(Dtype.F32, Part(operand.Amax, "amax").Payload);
                var den = BitPattern.WidenToF32(Dtype.F32, Part(operand.Denominator, "denominator").Payload);
                if (scale.Dtype != Dtype.F8_E4M3 || scale.Payload.LongLength != (long)experts * rows * cols / 16 || amax.Length != experts || den.Length != 1)
                {
                    throw new ContainerException($"{e.TensorName}: NVFP4 needs an E4M3 scale per 16, an amax per expert and one denominator");
                }
                var factor = new float[experts];
                for (int i = 0; i < experts; i++)
                {
                    factor[i] = amax[i] / den[0];
                }
                var w = UploadBytes(store.Resolve(e).Payload);
                var scales = UploadBytes(scale.Payload);
                var fac = UploadFloats(factor);
                long count = (long)experts * rows;
                var rowScale = Alloc(count * 4);
                Check(DepthNative.amqe_expert_row_scale(_engine, fac, rows, count, rowScale), "expert row scale");
                Check(Native.amqe_sync(_engine), "expert row scale");
                return new DepthExperts { W = w, Scales = scales, Factor = fac, RowScale = rowScale, Type = WNvfp4, Rows = rows, Cols = cols };
            }
            case Dtype.F8_E4M3:
            {
                if (cols % 32 != 0)
                {
                    throw new ContainerException($"{e.TensorName}: MXFP8 rows of {cols} are not whole blocks of 32");
                }
                var scale = Part(operand.Scale, "scale");
                if (scale.Dtype is not (Dtype.U8 or Dtype.F8_E8M0) || scale.Payload.LongLength != (long)experts * rows * cols / 32)
                {
                    throw new ContainerException($"{e.TensorName}: MXFP8 needs an E8M0 exponent per 32");
                }
                return new DepthExperts
                {
                    W = UploadBytes(store.Resolve(e).Payload), Scales = UploadBytes(scale.Payload), Type = WMxfp8, Rows = rows, Cols = cols,
                };
            }
            default:
            {
                var plain = store.Resolve(e);
                return plain.Dtype switch
                {
                    Dtype.BF16 => new DepthExperts { W = UploadBytes(plain.Payload), Type = BF16, Rows = rows, Cols = cols },
                    Dtype.F32 => new DepthExperts { W = UploadBytes(plain.Payload), Type = F32, Rows = rows, Cols = cols },
                    _ => new DepthExperts { W = UploadFloats(BitPattern.WidenToF32(plain.Dtype, plain.Payload)), Type = F32, Rows = rows, Cols = cols },
                };
            }
        }
    }

    private static float Clip(double? clip) => clip is { } c ? (float)c : float.PositiveInfinity;

    private DepthLayer UploadDepthLayer(WeightLoader weights, LayerPlan lp)
    {
        int d = _hidden;
        var attn = lp.Attention!;
        var routed = lp.Ffn!.Routed!;
        var policy = (ExpertGateSoftcappedGlu)routed.GatePolicy;
        int heads = attn.NumQHeads, kvHeads = attn.NumKvHeads, hd = attn.HeadDim;
        var L = new DepthLayer
        {
            Plan = lp,
            Heads = heads, KvHeads = kvHeads, HeadDim = hd,
            Window = attn.Window is { } w ? checked((int)w) : 0,
            // 1/sqrt(hd) as the kernels have always taken it, else the plan's.
            Scale = Math.Abs(attn.ScoreScale - 1.0 / Math.Sqrt(hd)) < 1e-12 ? 1f / MathF.Sqrt(hd) : (float)attn.ScoreScale,
            QkClip = Clip(attn.QkClip),
            VNormOn = attn.ParameterFreeQkNorm.V ? 1 : 0,
            VEps = (float)attn.ParameterFreeQkNormEps,
            SeparateInputs = lp.QInputNorm is not null,
            OpensBlock = lp.OpensBlock,
            Experts = routed.NumExperts, TopK = routed.TopK, Width = routed.ExpertIntermediateSize,
            RouteScale = (float)routed.RouteScale, GateCap = (float)policy.GateCap, UpCap = (float)policy.UpCap,
        };
        L.Wq = DepthMatrix(weights, attn.QProj, attn.QDim, d);
        L.Wk = DepthMatrix(weights, attn.KProj, attn.KvDim, d);
        L.Wv = DepthMatrix(weights, attn.VProj, attn.KvDim, d);
        L.Wo = DepthMatrix(weights, attn.OProj, d, attn.QDim);
        L.QScale = attn.QueryChannelScale is { } qs ? Vector(weights, qs, attn.QDim) : IntPtr.Zero;
        if (attn.LogScaling is { } log)
        {
            L.LogScale = 1;
            L.Floor = log.Floor;
            L.Alpha = (float)log.Alpha;
        }
        if (attn.RelativeBias is { } bias)
        {
            L.Rel = DepthMatrix(weights, bias.Projection, heads * bias.Rank, d);
            L.RelScale = Vector(weights, bias.ChannelScale, heads * bias.Rank);
            L.RelTable = Vector(weights, bias.Table, bias.Rank * bias.Extent);
            L.RelRank = bias.Rank;
            L.Extent = bias.Extent;
            L.RelClip = Clip(bias.Clip);
        }
        if (attn.Sink is { } sink)
        {
            if (sink.Projected)
            {
                L.SinkProj = DepthMatrix(weights, sink.Weight, heads, d);
            }
            else
            {
                L.SinkStatic = Vector(weights, sink.Weight, heads);
            }
        }
        if (attn.OutputGateProjection is { } gate)
        {
            L.OutGate = DepthMatrix(weights, gate, attn.QDim, d);
        }
        (L.KMode, L.KConv, L.KTaps) = ProjectionConv(attn.KeyConv);
        (L.VMode, L.VConv, L.VTaps) = ProjectionConv(attn.ValueConv);
        L.ConvKernel = attn.KeyConv?.Kernel ?? attn.ValueConv?.Kernel ?? 0;

        var qNorm = lp.QInputNorm ?? lp.PreAttentionNorm!;
        L.QNorm = Vector(weights, qNorm.Weight, qNorm.Width);
        L.KNorm = lp.KInputNorm is { } kn ? Vector(weights, kn.Weight, kn.Width) : L.QNorm;
        L.VNorm = lp.VInputNorm is { } vn ? Vector(weights, vn.Weight, vn.Width) : L.QNorm;
        L.AttnEps = (float)qNorm.Eps;
        if (new[] { lp.KInputNorm, lp.VInputNorm }.Any(n => n is not null && (float)n.Eps != L.AttnEps))
        {
            throw new InvalidOperationException("the attention's input norms differ in eps");
        }
        L.FfnNorm = Vector(weights, lp.PreFfnNorm!.Weight, lp.PreFfnNorm.Width);
        L.FfnEps = (float)lp.PreFfnNorm.Eps;
        if (lp.MixerInputMix is { } mm)
        {
            L.MixerQueries = Vector(weights, mm.Queries, mm.Count * d);
            L.MixerCount = mm.Count;
            L.MixerMixEps = (float)mm.Eps;
        }
        if (lp.FfnInputMix is { } fm)
        {
            L.FfnQueries = Vector(weights, fm.Queries, d);
            L.FfnMixEps = (float)fm.Eps;
        }
        if (lp.MixerOutputConv is { } mc)
        {
            L.MixerConv = Vector(weights, mc.Weight!, mc.Channels * mc.Kernel);
            L.MixerConvKernel = mc.Kernel;
        }
        if (lp.FfnOutputConv is { } fc)
        {
            L.FfnConv = Vector(weights, fc.Weight!, fc.Channels * fc.Kernel);
            L.FfnConvKernel = fc.Kernel;
        }
        if (lp.MessageGates is { Count: > 0 } gates)
        {
            L.GateSites = gates.Select(g => g.Input).ToArray();
            L.Down = gates.Select(g => DepthMatrix(weights, g.Down, g.Rank, 2 * d)).ToArray();
            L.Forget = gates.Select(g => DepthMatrix(weights, g.Forget, d, g.Rank)).ToArray();
            L.Gain = gates.Select(g => DepthMatrix(weights, g.Gain, d, g.Rank)).ToArray();
            var queries = new float[gates.Count * d];
            for (int i = 0; i < gates.Count; i++)
            {
                var g = gates[i];
                int rows = checked((int)weights.Rows.RowsOf(g.Query));
                weights.Vector(g.Query, rows * d).AsSpan(g.QueryRow * d, d).CopyTo(queries.AsSpan(i * d, d));
            }
            L.GateQueries = UploadFloats(queries);
            L.GateEps = (float)gates[0].Eps;
            L.GateMixEps = (float)gates[0].MixEps;
            L.GateRank = gates[0].Rank;
        }
        L.Router = DepthMatrix(weights, routed.Router, routed.NumExperts, d);
        L.SelectionBias = routed.SelectionBias is { } sb ? Vector(weights, sb, routed.NumExperts) : IntPtr.Zero;
        var store = weights.Store;
        L.GateUp = DepthExpertStack(store, routed.Packed!.GateUp, routed.NumExperts, 2 * routed.ExpertIntermediateSize, d);
        L.DownExperts = DepthExpertStack(store, routed.Packed.Down, routed.NumExperts, d, routed.ExpertIntermediateSize);
        return L;

        (int Mode, IntPtr Conv, DepthMat? Taps) ProjectionConv(ResidualConvOp? conv) => conv switch
        {
            null => (0, IntPtr.Zero, null),
            { Dynamic: true } c => (2, Vector(weights, c.StaticTaps!, c.Kernel * c.Channels),
                DepthMatrix(weights, c.TapProjection!, c.Heads * c.Kernel, d)),
            { } c => (1, Vector(weights, c.Weight!, c.Channels * c.Kernel), null),
        };
    }

    private void BuildVisitDevice(ComponentOpPlan plan, WeightLoader weights)
    {
        int d = _hidden;
        var layers = plan.Layers.Select(lp => UploadDepthLayer(weights, lp)).ToArray();
        var schedule = plan.Schedule;
        int slots = schedule.Max(v => v.CacheSlot) + 1;
        var slotLayer = new int[slots];
        foreach (var v in schedule)
        {
            slotLayer[v.CacheSlot] = v.Layer;
        }
        var embed = plan.Embedding!;
        var output = plan.Output!;
        var vd = new VisitDevice
        {
            Layers = layers,
            Schedule = schedule,
            SlotLayer = slotLayer,
            Embed = DepthMatrix(weights, embed.Table, embed.VocabSize, embed.HiddenSize),
            Head = DepthMatrix(weights, output.Projection, output.StoredRows, output.HiddenSize, readRows: output.VocabSize),
            EmbedNormed = embed.Norm is not null,
            EmbedEps = (float)(embed.Norm?.Eps ?? 0),
            EmbedNorm = embed.NormWeight is { } nw ? Vector(weights, nw, d) : IntPtr.Zero,
            FinalNorm = Vector(weights, plan.FinalNorm!.Weight, plan.FinalNorm.Width),
            FinalEps = (float)plan.FinalNorm.Eps,
            Softcap = output.LogitSoftcapping,
            Vocab = output.VocabSize,
            MaxBank = plan.BlockResidual is { } b ? plan.Layers.Count / b.BlockSize + 1 : 1,
        };
        if (plan.BlockResidual is { } block)
        {
            vd.OutputQuery = Vector(weights, block.OutputQuery.Queries, d);
            vd.OutputMixEps = (float)block.OutputQuery.Eps;
        }

        long R = DepthChunk;
        int maxQ = layers.Max(l => l.Heads * l.HeadDim), maxC = layers.Max(l => l.KvHeads * l.HeadDim);
        int maxHeads = layers.Max(l => l.Heads), maxKv = layers.Max(l => l.KvHeads);
        int maxRel = Math.Max(1, layers.Max(l => l.Heads * l.RelRank));
        int maxTaps = Math.Max(1, layers.Max(l => l.KvHeads * l.ConvKernel));
        int experts = layers.Max(l => l.Experts), topK = layers.Max(l => l.TopK), width = layers.Max(l => l.Width);
        int gates = Math.Max(1, layers.Max(l => l.GateSites?.Length ?? 0)), rank = Math.Max(1, layers.Max(l => l.GateRank));
        vd.MaxHeads = maxHeads;
        vd.MaxExtentHeads = Math.Max(1, layers.Max(l => l.Heads * Math.Max(l.Extent, 1)));
        vd.H = Enumerable.Range(0, schedule.Count + 1).Select(_ => Alloc(R * d * 4)).ToArray();
        vd.Tokens = Alloc((R + 1) * 4);
        vd.Q = Alloc(R * maxQ * 4);
        vd.Gate = Alloc(R * maxQ * 4);
        vd.Tmp = Alloc(R * maxQ * 4);
        vd.K = Alloc(R * maxC * 4);
        vd.V = Alloc(R * maxC * 4);
        vd.R = Alloc(R * maxRel * 4);
        vd.Sinks = Alloc(R * maxHeads * 4);
        vd.RouterLogits = Alloc(R * experts * 4);
        vd.KDyn = Alloc(R * maxTaps * 4);
        vd.VDyn = Alloc(R * maxTaps * 4);
        vd.Mixed = Alloc(R * 3 * d * 4);
        vd.Msg = Alloc(R * gates * d * 4);
        vd.Cat = Alloc(3 * R * 2 * d * 4);
        vd.GateVec = Alloc(3 * R * rank * 4);
        vd.Attn = Alloc(R * d * 4);
        vd.Xq = Alloc(R * d * 4);
        vd.Xk = Alloc(R * d * 4);
        vd.Xv = Alloc(R * d * 4);
        vd.Nq = Alloc(R * d * 4);
        vd.Nk = Alloc(R * d * 4);
        vd.Nv = Alloc(R * d * 4);
        vd.Res = Alloc(R * d * 4);
        vd.FfnIn = Alloc(R * d * 4);
        vd.FfnGated = Alloc(R * d * 4);
        vd.Normed = Alloc(R * d * 4);
        vd.Ffn = Alloc(R * d * 4);
        vd.Act = Alloc(R * topK * width * 4);
        vd.ExpertPart = Alloc(R * topK * d * 4);
        vd.Logits = Alloc((long)vd.Vocab * 4);
        vd.Wt = Alloc(R * topK * 4);
        vd.Idx = Alloc(R * topK * 4);
        vd.Order = Alloc(R * topK * 4);
        vd.Offsets = Alloc((experts + 1) * 4L);
        vd.GroupOrder = Alloc(3 * (R * topK + experts) * 4);
        vd.Groups = Alloc(3 * (R * topK + experts) * 4);
        vd.Rb = Alloc(R * vd.MaxExtentHeads * 4);
        Check(Native.amqe_host_alloc(out vd.HostTokens, (nuint)((R + 1) * 4)), "host alloc");
        Check(Native.amqe_host_alloc(out vd.HostLogits, (nuint)((long)vd.Vocab * 4)), "host alloc");
        _visits = vd;
    }

    // ── sequences ─────────────────────────────────────────────────────────────

    private static int KvFormatCode(string format) => format switch
    {
        "f32" => KvF32,
        "nvfp4" => KvFp4,
        _ => KvF16,
    };

    private static long DepthKvBytes(int rows, int c, int kvHeads, int format) => format switch
    {
        KvF32 => 2L * rows * c * 4,
        KvF16 => 2L * rows * c * 2,
        _ => 2L * ((long)rows * c / 2 + (long)rows * c / 16 + (long)rows * kvHeads * 4),
    };

    // A windowed slot holds its window and a chunk more, so the rows a chunk
    // writes never overwrite keys its earlier rows still read; a full slot every position.
    private static int DepthSlotRows(DepthLayer L, int capacity) =>
        L.Window > 0 ? Math.Min(capacity, L.Window + DepthChunk) : capacity;

    private IntPtr SeqAlloc(DepthSequence s, long bytes)
    {
        var p = Alloc(bytes);
        Check(Native.amqe_memset(_engine, p, 0, (nuint)Math.Max(16, bytes)), "zero");
        s.Owned.Add(p);
        s.Bytes += Math.Max(16, bytes);
        return p;
    }

    private void SeqFree(DepthSequence s, IntPtr p)
    {
        if (p == IntPtr.Zero)
        {
            return;
        }
        s.Owned.Remove(p);
        Free(p);
    }

    private DepthKv AllocDepthKv(DepthSequence s, int rows, int c, int kvHeads, int format, bool ring)
    {
        var kv = new DepthKv { Cap = rows, Ring = ring ? 1 : 0, Format = format };
        if (format is KvF32 or KvF16)
        {
            long el = format == KvF32 ? 4 : 2;
            kv.K = SeqAlloc(s, (long)rows * c * el);
            kv.V = SeqAlloc(s, (long)rows * c * el);
        }
        else
        {
            kv.K = SeqAlloc(s, (long)rows * c / 2);
            kv.V = SeqAlloc(s, (long)rows * c / 2);
            kv.Ks = SeqAlloc(s, (long)rows * c / 16);
            kv.Vs = SeqAlloc(s, (long)rows * c / 16);
            kv.Kg = SeqAlloc(s, (long)rows * kvHeads * 4);
            kv.Vg = SeqAlloc(s, (long)rows * kvHeads * 4);
        }
        return kv;
    }

    private long DepthPartBytes(DepthSequence s)
    {
        long most = 1;
        foreach (var slot in s.Slots)
        {
            int heads = _visits!.Layers[slot.Layer].Heads;
            int sp = DepthNative.amqe_depth_attention_splits(slot.Kv.Cap, slot.Kv.Ring);
            most = Math.Max(most, Math.Max((long)sp * heads, (long)DepthChunk * Math.Min(sp, DepthRowSplits) * heads));
        }
        return most * 130 * 4;
    }

    private DepthSequence NewDepthSequence(int capacity)
    {
        var vd = _visits!;
        int format = KvFormatCode(RequestedKvFormat);
        if (format != KvF32 && vd.Layers.Any(l => l.HeadDim % 16 != 0))
        {
            KvFormatRefusal = $"{RequestedKvFormat.ToUpperInvariant()} KV needs heads a multiple of 16 wide";
            format = KvF32;
        }
        var s = new DepthSequence { Slots = new DepthSlot[vd.SlotLayer.Length], Format = format, Capacity = capacity };
        try
        {
            for (int i = 0; i < s.Slots.Length; i++)
            {
                var L = vd.Layers[vd.SlotLayer[i]];
                int c = L.KvHeads * L.HeadDim;
                bool ring = L.Window > 0;
                // Windowed slots keep f32 rows of their window (a window is small); full slots the format asked for.
                var slot = new DepthSlot { Layer = vd.SlotLayer[i], C = c, Ring = ring };
                slot.Kv = AllocDepthKv(s, DepthSlotRows(L, capacity), c, L.KvHeads, ring ? KvF32 : format, ring);
                if (L.ConvKernel > 0)
                {
                    slot.KHist = SeqAlloc(s, 4L * (L.ConvKernel - 1) * c);
                    slot.VHist = SeqAlloc(s, 4L * (L.ConvKernel - 1) * c);
                }
                if (L.MixerConvKernel > 0)
                {
                    slot.MixerHist = SeqAlloc(s, 4L * (L.MixerConvKernel - 1) * _hidden);
                }
                if (L.FfnConvKernel > 0)
                {
                    slot.FfnHist = SeqAlloc(s, 4L * (L.FfnConvKernel - 1) * _hidden);
                }
                s.Slots[i] = slot;
            }
            s.Part = SeqAlloc(s, DepthPartBytes(s));
            Check(Native.amqe_sync(_engine), "sequence");
        }
        catch
        {
            ReleaseDepth(s);
            throw;
        }
        return s;
    }

    private void ReleaseDepth(DepthSequence s)
    {
        Native.amqe_sync(_engine);
        if (s.Graph != IntPtr.Zero)
        {
            Native.amqe_pass_free(s.Graph);
            s.Graph = IntPtr.Zero;
        }
        foreach (var p in s.Owned.ToArray())
        {
            SeqFree(s, p);
        }
        s.Bytes = 0;
    }

    private void DropDepthGraph(DepthSequence s)
    {
        if (s.Graph != IntPtr.Zero)
        {
            Native.amqe_sync(_engine);
            Native.amqe_pass_free(s.Graph);
            s.Graph = IntPtr.Zero;
        }
    }

    /// <summary>Room for <paramref name="positions"/>: each slot whose rows depend on
    /// the capacity moves to a larger allocation (its rows so far kept), and the
    /// step's graph is captured again.</summary>
    private void DepthRoom(GpuSequence seq, long positions)
    {
        var s = seq.Depth!;
        if (positions <= s.Capacity)
        {
            return;
        }
        int capacity = (int)Math.Min(Math.Max(positions, 2L * s.Capacity), int.MaxValue / 2);
        var vd = _visits!;
        Check(Native.amqe_sync(_engine), "grow");
        foreach (var slot in s.Slots)
        {
            var L = vd.Layers[slot.Layer];
            int rows = DepthSlotRows(L, capacity);
            if (rows <= slot.Kv.Cap)
            {
                continue;
            }
            // A ring that grows has not wrapped yet: it held fewer rows than the
            // positions so far would need only once the capacity exceeded it.
            var old = slot.Kv;
            var grown = AllocDepthKv(s, rows, slot.C, L.KvHeads, old.Format, slot.Ring);
            long n = Math.Min(seq.Position, old.Cap);
            void Copy(IntPtr to, IntPtr from, long bytes)
            {
                if (bytes > 0)
                {
                    Check(Native.amqe_copy_async(_engine, to, from, (nuint)bytes, 3), "grow");
                }
            }
            if (old.Format == KvFp4)
            {
                Copy(grown.K, old.K, n * slot.C / 2);
                Copy(grown.V, old.V, n * slot.C / 2);
                Copy(grown.Ks, old.Ks, n * slot.C / 16);
                Copy(grown.Vs, old.Vs, n * slot.C / 16);
                Copy(grown.Kg, old.Kg, n * L.KvHeads * 4);
                Copy(grown.Vg, old.Vg, n * L.KvHeads * 4);
            }
            else
            {
                long el = old.Format == KvF32 ? 4 : 2;
                Copy(grown.K, old.K, n * slot.C * el);
                Copy(grown.V, old.V, n * slot.C * el);
            }
            Check(Native.amqe_sync(_engine), "grow");
            foreach (var p in new[] { old.K, old.V, old.Ks, old.Vs, old.Kg, old.Vg })
            {
                SeqFree(s, p);
            }
            s.Bytes -= DepthKvBytes(old.Cap, slot.C, L.KvHeads, old.Format);
            slot.Kv = grown;
        }
        // A larger cache may take more splits than the partials were sized for.
        SeqFree(s, s.Part);
        s.Part = SeqAlloc(s, DepthPartBytes(s));
        Check(Native.amqe_sync(_engine), "grow");
        s.Capacity = capacity;
        DropDepthGraph(s);
    }

    // ── a forward pass ────────────────────────────────────────────────────────

    private static IntPtr At(IntPtr p, long floats) => p + (nint)(floats * 4);

    private void Proj(int T, params (DepthMat M, IntPtr X, IntPtr Y)[] list)
    {
        var m = list.Select(p => p.M).ToArray();
        var x = list.Select(p => p.X).ToArray();
        var y = list.Select(p => p.Y).ToArray();
        Check(DepthNative.amqe_projections(_engine, m, x, y, list.Length, T), "projections");
    }

    private void Norms(int T, float eps, params (IntPtr X, IntPtr W, IntPtr Out)[] list)
    {
        Check(DepthNative.amqe_rms_rows(_engine, list.Select(n => n.X).ToArray(), list.Select(n => n.W).ToArray(),
            list.Select(n => n.Out).ToArray(), list.Length, T, _hidden, eps), "norms");
    }

    private void Mix(List<IntPtr> bank, IntPtr current, IntPtr queries, int count, IntPtr output, int T, float eps,
        IntPtr[]? normW = null, IntPtr[]? normOut = null, float normEps = 0f)
    {
        var states = bank.Append(current).ToArray();
        Check(DepthNative.amqe_depth_mix(_engine, states, states.Length, queries, count, output, T, _hidden, eps, normW, normOut, normEps),
            "depth mix");
    }

    private void Gates(DepthLayer L, int[] which, IntPtr[] h, IntPtr messages, IntPtr[] output, int T)
    {
        var vd = _visits!;
        long td = (long)T * _hidden;
        var m = which.Select(g => At(messages, g * td)).ToArray();
        Check(DepthNative.amqe_message_gates(_engine, which.Length, h, m, output,
            which.Select(g => L.Down[g]).ToArray(), which.Select(g => L.Forget[g]).ToArray(), which.Select(g => L.Gain[g]).ToArray(),
            vd.Cat, vd.GateVec, DepthChunk, T, _hidden, L.GateRank, L.GateEps), "message gates");
    }

    /// <summary>One visit of a layer for T rows at the positions from the token
    /// buffer's: its inputs (depth mix, message gates, norms), the attention and
    /// its residual conv, the residual join (or a new block), then the FFN the same
    /// way. Mirrors GenericRuntime.RunRowLayer.</summary>
    private IntPtr DepthVisit(DepthLayer L, DepthSlot slot, IntPtr h, List<IntPtr> bank, IntPtr messages, IntPtr output, int T,
        int position)
    {
        var vd = _visits!;
        int d = _hidden;
        long td = (long)T * d;
        IntPtr pos = At(vd.Tokens, DepthChunk);
        int Gate(string site) => L.GateSites is { } sites && messages != IntPtr.Zero ? Array.IndexOf(sites, site) : -1;

        // The attention's inputs.
        IntPtr xq = h, xk = h, xv = h;
        bool normed = false;
        bool single = !L.SeparateInputs;
        if (L.MixerQueries != IntPtr.Zero)
        {
            bool fuse = messages == IntPtr.Zero && (L.MixerCount == 3 || single);
            IntPtr[]? w = fuse ? (L.MixerCount == 3 ? [L.QNorm, L.KNorm, L.VNorm] : [L.QNorm]) : null;
            IntPtr[]? o = fuse ? (L.MixerCount == 3 ? [vd.Nq, vd.Nk, vd.Nv] : [vd.Nq]) : null;
            Mix(bank, h, L.MixerQueries, L.MixerCount, vd.Mixed, T, L.MixerMixEps, w, o, L.AttnEps);
            normed = fuse;
            xq = vd.Mixed;
            xk = L.MixerCount == 3 ? At(vd.Mixed, td) : vd.Mixed;
            xv = L.MixerCount == 3 ? At(vd.Mixed, 2 * td) : vd.Mixed;
        }
        {
            var sites = new List<int>();
            var ins = new List<IntPtr>();
            var outs = new List<IntPtr>();
            foreach (var (site, x, to) in new[] { ("q", xq, vd.Xq), ("k", xk, vd.Xk), ("v", xv, vd.Xv) })
            {
                if (Gate(site) is >= 0 and var g)
                {
                    sites.Add(g);
                    ins.Add(x);
                    outs.Add(to);
                }
            }
            if (sites.Count > 0)
            {
                Gates(L, sites.ToArray(), ins.ToArray(), messages, outs.ToArray(), T);
                if (Gate("q") >= 0) xq = vd.Xq;
                if (Gate("k") >= 0) xk = vd.Xk;
                if (Gate("v") >= 0) xv = vd.Xv;
                normed = false;
            }
        }
        IntPtr qIn, kIn, vIn;
        if (normed)
        {
            qIn = vd.Nq;
            kIn = L.MixerCount == 3 ? vd.Nk : vd.Nq;
            vIn = L.MixerCount == 3 ? vd.Nv : vd.Nq;
        }
        else if (single && xq == xk && xq == xv)
        {
            Norms(T, L.AttnEps, (xq, L.QNorm, vd.Nq));
            qIn = kIn = vIn = vd.Nq;
        }
        else
        {
            Norms(T, L.AttnEps, (xq, L.QNorm, vd.Nq), (xk, L.KNorm, vd.Nk), (xv, L.VNorm, vd.Nv));
            (qIn, kIn, vIn) = (vd.Nq, vd.Nk, vd.Nv);
        }

        // The attention: projections sharing an input in one launch each.
        var qGroup = new List<(DepthMat, IntPtr, IntPtr)> { (L.Wq, qIn, vd.Q) };
        if (L.Rel is { } rel) qGroup.Add((rel, qIn, vd.R));
        if (L.SinkProj is { } sinkProj) qGroup.Add((sinkProj, qIn, vd.Sinks));
        if (L.OutGate is { } outGate) qGroup.Add((outGate, qIn, vd.Gate));
        Proj(T, qGroup.ToArray());
        var kvGroup = new List<(DepthMat, IntPtr, IntPtr)> { (L.Wk, kIn, vd.K), (L.Wv, vIn, vd.V) };
        if (L.KTaps is { } kTaps) kvGroup.Add((kTaps, kIn, vd.KDyn));
        if (L.VTaps is { } vTaps) kvGroup.Add((vTaps, vIn, vd.VDyn));
        Proj(T, kvGroup.ToArray());
        Check(DepthNative.amqe_head_prep(_engine, vd.Q, L.QScale, L.HeadDim, L.QkClip,
            L.Rel is null ? IntPtr.Zero : vd.R, L.RelScale, L.RelRank, L.RelClip, L.Heads, T, L.LogScale, pos, L.Floor, L.Alpha), "head prep");
        var kv = slot.Kv;
        Check(DepthNative.amqe_kv_prep(_engine, ref kv, vd.K, vd.V, vd.KDyn, vd.VDyn, L.KConv, L.VConv, slot.KHist, slot.VHist,
            L.KMode, L.VMode, slot.C, L.HeadDim, L.ConvKernel, L.KvHeads, T, L.QkClip, L.VNormOn, L.VEps, pos), "keys and values");

        int splits = DepthNative.amqe_depth_attention_splits(kv.Cap, kv.Ring);
        if (T > 1)
        {
            splits = Math.Min(splits, DepthRowSplits);
        }
        int hd = L.HeadDim;
        bool tiled = TiledAttention && T > 1 && !_exactPrefill && hd is 16 or 32 or 64 or 128 && L.Heads <= 64;
        if (tiled)
        {
            // A chunk on the tensor cores: splits enough to fill the device, a few hundred keys each at least.
            long keys = L.Window == 0 ? position + T : Math.Min(position + T, L.Window + 64);
            splits = (int)Math.Max(1, Math.Min(splits, keys / 512));
        }
        Check(DepthNative.amqe_depth_attention(_engine, vd.Q, vd.R, L.RelTable, vd.Rb, ref kv, DepthPartOf(),
            L.Heads, L.KvHeads, hd, L.RelRank, L.Extent, L.Window, splits, L.Scale, T, tiled ? 1 : 0, pos), "attention");
        IntPtr sinks = L.SinkProj is not null ? vd.Sinks : L.SinkStatic;
        int sinkStride = L.SinkProj is not null ? L.Heads : 0;
        Check(DepthNative.amqe_depth_attention_merge(_engine, DepthPartOf(), sinks, sinkStride, L.OutGate is null ? IntPtr.Zero : vd.Gate,
            vd.Tmp, L.Heads, splits, hd, T), "attention merge");
        Proj(T, (L.Wo, vd.Tmp, vd.Attn));

        // The attention's residual conv and the join: a block's first layer keeps
        // its input in the bank and restarts the residual from the mixer's output.
        if (L.OpensBlock)
        {
            bank.Add(h);
        }
        Check(DepthNative.amqe_residual_conv(_engine, vd.Attn, L.MixerConv, slot.MixerHist, L.OpensBlock ? IntPtr.Zero : h, vd.Res,
            d, L.MixerConvKernel, T, pos), "mixer conv");

        // The FFN's input.
        int ffnGate = Gate("ffn");
        IntPtr ffnIn = vd.Res;
        bool ffnNormed = false;
        if (L.FfnQueries != IntPtr.Zero)
        {
            bool fuse = ffnGate < 0;
            Mix(bank, vd.Res, L.FfnQueries, 1, vd.FfnIn, T, L.FfnMixEps, fuse ? [L.FfnNorm] : null, fuse ? [vd.Normed] : null, L.FfnEps);
            ffnNormed = fuse;
            ffnIn = vd.FfnIn;
        }
        if (ffnGate >= 0)
        {
            Gates(L, [ffnGate], [ffnIn], messages, [vd.FfnGated], T);
            ffnIn = vd.FfnGated;
        }
        if (!ffnNormed)
        {
            Norms(T, L.FfnEps, (ffnIn, L.FfnNorm, vd.Normed));
        }

        // The experts: sigmoid routing over the chosen, a softcapped GLU.
        Proj(T, (L.Router, vd.Normed, vd.RouterLogits));
        Check(DepthNative.amqe_route_sigmoid_selected(_engine, vd.RouterLogits, L.SelectionBias, L.Experts, L.TopK, L.RouteScale,
            vd.Idx, vd.Wt, T), "router");
        var gu = L.GateUp;
        var dn = L.DownExperts;
        Check(DepthNative.amqe_packed_experts(_engine, ref gu, ref dn, L.Experts, d, L.Width, L.TopK, L.GateCap, L.UpCap,
            vd.Normed, T, vd.Idx, vd.Wt, vd.Act, vd.ExpertPart, vd.Ffn, vd.Order, vd.Offsets, vd.GroupOrder, vd.Groups), "experts");
        Check(DepthNative.amqe_residual_conv(_engine, vd.Ffn, L.FfnConv, slot.FfnHist, vd.Res, output, d, L.FfnConvKernel, T, pos),
            "ffn conv");
        return output;
    }

    private DepthSequence? _depthNow;

    private IntPtr DepthPartOf() => _depthNow!.Part;

    /// <summary>T rows through the visit schedule, as launches on the engine's
    /// stream (a step's are captured into a graph): the last row's logits into the
    /// logits buffer, or every row's into <see cref="VisitDevice.Every"/>. Mirrors
    /// GenericRuntime.StepVisits.</summary>
    private void DepthRecord(DepthSequence s, int T, bool every, int position)
    {
        var vd = _visits!;
        int d = _hidden;
        long td = (long)T * d;
        _depthNow = s;
        int buf = 0;
        IntPtr h = vd.H[buf++];
        var embed = vd.Embed;
        Check(DepthNative.amqe_embed_rows(_engine, ref embed, vd.Tokens, vd.EmbedNorm, vd.EmbedNormed ? 1 : 0, vd.EmbedEps, h, T), "embed");
        var bank = new List<IntPtr>();
        var loop = _plan.DepthLoop;
        IntPtr saved = IntPtr.Zero, messageOut = IntPtr.Zero;
        List<IntPtr>? savedBank = null, messageBank = null;
        foreach (var visit in vd.Schedule)
        {
            if (loop is not null && visit.Layer == loop.Start)
            {
                if (visit.Pass == 0)
                {
                    saved = h;
                    savedBank = bank.ToList();
                }
                else
                {
                    // The previous pass's bank past the state the loop restarts
                    // from, and its output, are what this pass's gates read.
                    messageBank = bank.Skip(savedBank!.Count + loop.SkipBlocks).ToList();
                    messageOut = h;
                    h = saved;
                    bank = savedBank.ToList();
                }
            }
            var L = vd.Layers[visit.Layer];
            IntPtr messages = IntPtr.Zero;
            if (visit.Pass > 0 && L.GateSites is { } sites)
            {
                // Every gate's message, [gates][T][d], from one mix.
                Mix(messageBank!, messageOut, L.GateQueries, sites.Length, vd.Msg, T, L.GateMixEps);
                messages = vd.Msg;
            }
            h = DepthVisit(L, s.Slots[visit.CacheSlot], h, bank, messages, vd.H[buf++], T, position);
        }
        if (_plan.BlockResidual is not null)
        {
            Mix(bank, h, vd.OutputQuery, 1, vd.Mixed, T, vd.OutputMixEps, [vd.FinalNorm], [vd.Normed], vd.FinalEps);
        }
        else
        {
            Norms(T, vd.FinalEps, (h, vd.FinalNorm, vd.Normed));
        }
        if (every)
        {
            Proj(T, (vd.Head, vd.Normed, vd.Every));
            if (vd.Softcap is { } cap)
            {
                Check(DepthNative.amqe_softcap(_engine, vd.Every, (long)T * vd.Vocab, cap), "softcap");
            }
        }
        else
        {
            Proj(1, (vd.Head, At(vd.Normed, td - d), vd.Logits));
            if (vd.Softcap is { } cap)
            {
                Check(DepthNative.amqe_softcap(_engine, vd.Logits, vd.Vocab, cap), "softcap");
            }
        }
    }

    private void DepthTokens(ReadOnlySpan<int> tokens, int position)
    {
        var vd = _visits!;
        for (int i = 0; i < tokens.Length; i++)
        {
            if ((uint)tokens[i] >= (uint)_plan.Embedding!.VocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), $"token {tokens[i]} is outside the vocabulary of {_plan.Embedding.VocabSize}");
            }
            Marshal.WriteInt32(vd.HostTokens, i * 4, tokens[i]);
        }
        Marshal.WriteInt32(vd.HostTokens, DepthChunk * 4, position);
        Check(Native.amqe_copy_async(_engine, vd.Tokens, vd.HostTokens, (nuint)(tokens.Length * 4), 1), "tokens");
        Check(Native.amqe_copy_async(_engine, At(vd.Tokens, DepthChunk), At(vd.HostTokens, DepthChunk), 4, 1), "position");
    }

    /// <summary>One token through the schedule; the logits land in the pinned
    /// host buffer (<see cref="HostLogits"/> row 0).</summary>
    private void DepthStep(GpuSequence seq, int token, bool logits = true)
    {
        var s = seq.Depth!;
        var vd = _visits!;
        DepthRoom(seq, seq.Position + 1L);
        DepthTokens([token], seq.Position);
        if (!UseGraphs)
        {
            DepthRecord(s, 1, every: false, seq.Position);
        }
        else
        {
            if (s.Graph == IntPtr.Zero)
            {
                Check(Native.amqe_pass_begin(_engine), "capture");
                try
                {
                    DepthRecord(s, 1, every: false, seq.Position);
                }
                catch
                {
                    if (Native.amqe_pass_end(_engine, out var broken) == 0)
                    {
                        Native.amqe_pass_free(broken);
                    }
                    throw;
                }
                Check(Native.amqe_pass_end(_engine, out var pass), "capture");
                s.Graph = pass;
            }
            Check(Native.amqe_pass_run(_engine, s.Graph), "step");
        }
        if (logits)
        {
            Check(Native.amqe_copy_async(_engine, _hostLogits, vd.Logits, (nuint)((long)vd.Vocab * 4), 2), "logits");
        }
        Check(Native.amqe_sync(_engine), "step");
        seq.Position++;
        _lastRows = 1;
    }

    /// <summary>The tokens in chunks of up to <see cref="DepthChunk"/>: the last
    /// position's logits, or with <paramref name="every"/> each position's.</summary>
    private float[][] DepthPrefill(GpuSequence seq, ReadOnlySpan<int> tokens, bool every)
    {
        var s = seq.Depth!;
        var vd = _visits!;
        int vocab = vd.Vocab;
        DepthRoom(seq, seq.Position + (long)tokens.Length);
        if (every && vd.Every == IntPtr.Zero)
        {
            vd.Every = Alloc((long)DepthChunk * vocab * 4);
        }
        var result = new float[every ? tokens.Length : 1][];
        for (int done = 0; done < tokens.Length;)
        {
            int T = Math.Min(DepthChunk, tokens.Length - done);
            bool last = done + T == tokens.Length;
            if (T == 1)
            {
                DepthStep(seq, tokens[done], logits: every || last);
                if (every || last)
                {
                    result[every ? done : 0] = HostLogits(0).ToArray();
                }
                done++;
                continue;
            }
            DepthTokens(tokens.Slice(done, T), seq.Position);
            DepthRecord(s, T, every, seq.Position);
            if (every)
            {
                var flat = GC.AllocateUninitializedArray<float>(T * vocab);
                Check(Native.amqe_sync(_engine), "prefill");
                var handle = GCHandle.Alloc(flat, GCHandleType.Pinned);
                try
                {
                    Check(Native.amqe_download(_engine, handle.AddrOfPinnedObject(), vd.Every, (nuint)((long)T * vocab * 4)), "logits");
                }
                finally
                {
                    handle.Free();
                }
                for (int t = 0; t < T; t++)
                {
                    result[done + t] = flat.AsSpan(t * vocab, vocab).ToArray();
                }
            }
            else if (last)
            {
                Check(Native.amqe_copy_async(_engine, vd.HostLogits, vd.Logits, (nuint)((long)vocab * 4), 2), "logits");
            }
            Check(Native.amqe_sync(_engine), "prefill");
            if (!every && last)
            {
                var row = GC.AllocateUninitializedArray<float>(vocab);
                Marshal.Copy(vd.HostLogits, row, 0, vocab);
                result[0] = row;
            }
            seq.Position += T;
            done += T;
        }
        return result;
    }

    /// <summary>Every position's logits for <paramref name="tokens"/> fed from the
    /// sequence's position (T x vocabulary). Visit-path plans only.</summary>
    public float[][] PrefillAll(GpuSequence seq, ReadOnlySpan<int> tokens)
    {
        Live(seq);
        if (_visits is null)
        {
            throw new NotSupportedException("every position's logits are read on the visit path only");
        }
        if (tokens.Length == 0)
        {
            throw new ArgumentException("prefill requires at least one token", nameof(tokens));
        }
        return DepthPrefill(seq, tokens, every: true);
    }

    /// <summary>The output of visit <paramref name="visit"/> (in schedule order) for
    /// the first <paramref name="rows"/> rows of the last chunk the visit path ran:
    /// every visit of a prompt that fit in one chunk. Diagnostics.</summary>
    public float[][] VisitOutput(int visit, int rows)
    {
        var vd = _visits ?? throw new NotSupportedException("visit outputs are the visit path's");
        if (visit < 0 || visit >= vd.Schedule.Count || rows < 1 || rows > DepthChunk)
        {
            throw new ArgumentOutOfRangeException(nameof(visit));
        }
        Check(Native.amqe_sync(_engine), "visit output");
        var flat = new float[rows * _hidden];
        var handle = GCHandle.Alloc(flat, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_download(_engine, handle.AddrOfPinnedObject(), vd.H[visit + 1], (nuint)((long)rows * _hidden * 4)), "visit output");
        }
        finally
        {
            handle.Free();
        }
        return Enumerable.Range(0, rows).Select(r => flat.AsSpan(r * _hidden, _hidden).ToArray()).ToArray();
    }

    private InvalidOperationException NotOnVisits(string what) => new($"{what}: {BatchRefusal}");

    /// <summary>Device bytes a visit-path sequence of <paramref name="positions"/>
    /// would hold in <paramref name="format"/> (f32, fp16, nvfp4): every slot's rows
    /// and convolution histories.</summary>
    public long DepthSequenceBytes(int positions, string format)
    {
        var vd = _visits ?? throw new NotSupportedException("sequence sizes here are the visit path's");
        long total = 0;
        foreach (int layer in vd.SlotLayer)
        {
            var L = vd.Layers[layer];
            int c = L.KvHeads * L.HeadDim;
            total += DepthKvBytes(DepthSlotRows(L, positions), c, L.KvHeads, L.Window > 0 ? KvF32 : KvFormatCode(format));
            total += 4L * (Math.Max(0, L.ConvKernel - 1) * 2L * c + Math.Max(0, L.MixerConvKernel - 1) * (long)_hidden +
                           Math.Max(0, L.FfnConvKernel - 1) * (long)_hidden);
        }
        return total;
    }

    /// <summary>The device's free and total memory, in bytes.</summary>
    public static (long Free, long Total) DeviceMemory()
    {
        Check(DepthNative.amqe_mem_info(out var free, out var total), "cudaMemGetInfo");
        return ((long)free, (long)total);
    }

    private static class DepthNative
    {
        private const string Lib = "amql_cuda";

        [DllImport(Lib)] public static extern int amqe_projections(IntPtr e, DepthMat[] m, IntPtr[] x, IntPtr[] y, int n, int rows);
        [DllImport(Lib)] public static extern int amqe_rms_rows(IntPtr e, IntPtr[] x, IntPtr[] w, IntPtr[] output, int n, int rows, int d, float eps);
        [DllImport(Lib)] public static extern int amqe_depth_mix(IntPtr e, IntPtr[] states, int n, IntPtr queries, int count, IntPtr output,
            int rows, int d, float eps, IntPtr[]? normW, IntPtr[]? normOut, float normEps);
        [DllImport(Lib)] public static extern int amqe_message_gates(IntPtr e, int n, IntPtr[] h, IntPtr[] m, IntPtr[] output,
            DepthMat[] down, DepthMat[] forget, DepthMat[] gain, IntPtr cat, IntPtr g, int stride, int rows, int d, int rank, float eps);
        [DllImport(Lib)] public static extern int amqe_head_prep(IntPtr e, IntPtr q, IntPtr qScale, int qDim, float qClip,
            IntPtr r, IntPtr rScale, int rDim, float rClip, int heads, int rows, int logScale, IntPtr pos, int floor, float alpha);
        [DllImport(Lib)] public static extern int amqe_kv_prep(IntPtr e, ref DepthKv slot, IntPtr kRaw, IntPtr vRaw, IntPtr kDyn, IntPtr vDyn,
            IntPtr kConv, IntPtr vConv, IntPtr kHist, IntPtr vHist, int kMode, int vMode, int c, int hd, int k, int kvHeads, int rows,
            float clip, int vnorm, float eps, IntPtr pos);
        [DllImport(Lib)] public static extern int amqe_depth_attention_splits(int cap, int ring);
        [DllImport(Lib)] public static extern int amqe_depth_attention(IntPtr e, IntPtr q, IntPtr r, IntPtr relProj, IntPtr rb, ref DepthKv kv,
            IntPtr part, int heads, int kvHeads, int hd, int dRel, int extent, int window, int splits, float scale, int rows, int tiled, IntPtr pos);
        [DllImport(Lib)] public static extern int amqe_depth_attention_merge(IntPtr e, IntPtr part, IntPtr sinks, int sinkStride, IntPtr gate,
            IntPtr output, int heads, int splits, int hd, int rows);
        [DllImport(Lib)] public static extern int amqe_residual_conv(IntPtr e, IntPtr x, IntPtr w, IntPtr hist, IntPtr add, IntPtr output,
            int c, int k, int rows, IntPtr pos);
        [DllImport(Lib)] public static extern int amqe_route_sigmoid_selected(IntPtr e, IntPtr logits, IntPtr bias, int experts, int topK,
            float routeScale, IntPtr idx, IntPtr wt, int rows);
        [DllImport(Lib)] public static extern int amqe_packed_experts(IntPtr e, ref DepthExperts gateUp, ref DepthExperts down, int experts,
            int d, int width, int topK, float gateCap, float upCap, IntPtr x, int rows, IntPtr idx, IntPtr wt, IntPtr act, IntPtr part,
            IntPtr ffn, IntPtr order, IntPtr offsets, IntPtr groupOrder, IntPtr groups);
        [DllImport(Lib)] public static extern int amqe_expert_row_scale(IntPtr e, IntPtr factor, int rows, long count, IntPtr output);
        [DllImport(Lib)] public static extern int amqe_embed_rows(IntPtr e, ref DepthMat table, IntPtr tokens, IntPtr normW, int normed, float eps,
            IntPtr output, int rows);
        [DllImport(Lib)] public static extern int amqe_softcap(IntPtr e, IntPtr x, long n, float cap);
        [DllImport(Lib)] public static extern int amqe_mem_info(out nuint free, out nuint total);
    }
}
