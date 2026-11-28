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
/// What the path keeps fused: a layer's attention projections in one launch,
/// the q, k and v message gates batched, the norm after a depth mix in the mix's
/// kernel, a step's input norms in its GEMVs, the output projection's conv in
/// its epilogue, the attention's merge in its last split block, the routing in
/// the experts' first kernel and their weighted sum in the conv after them; a
/// step's experts on lane-group kernels and a chunk's grouped on the tensor
/// cores. A step brings the next matrices into L2 while its small kernels run.
/// It runs one sequence at a time (no batching, marks or drafts).
/// </para>
/// </summary>
public sealed partial class GpuEngine
{
    /// <summary>Tokens a prompt chunk holds on the visit path by default: 512, or
    /// <c>AMQL_DEPTH_CHUNK</c> (a multiple of 64 up to 1024, and then fixed). A
    /// longer prompt borrows a chunk of up to <see cref="MaxDepthChunk"/> rows while
    /// the device's free memory holds it (<see cref="BorrowScratch"/>): a larger chunk
    /// decodes each chosen expert once for more tokens, for more scratch and ring rows.</summary>
    public static readonly int DepthChunk =
        int.TryParse(Environment.GetEnvironmentVariable("AMQL_DEPTH_CHUNK"), out var c) && c is >= 64 and <= MaxDepthChunk && c % 64 == 0 ? c : 512;

    /// <summary>The rows a borrowed prompt chunk holds at most.</summary>
    public const int MaxDepthChunk = 1024;

    /// <summary>False runs every prompt in the engine's own chunk; a set
    /// <c>AMQL_DEPTH_CHUNK</c> starts it false.</summary>
    public bool BorrowChunks { get; set; } = Environment.GetEnvironmentVariable("AMQL_DEPTH_CHUNK") is not { Length: > 0 };

    /// <summary>The rows the last prompt's chunks held.</summary>
    public int LastPromptChunk { get; private set; }

    // Free device memory a borrowed chunk leaves at least.
    private const long BorrowReserve = 1L << 30;

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

    /// <summary>False leaves a step's next matrices to their GEMVs instead of
    /// bringing them into L2 while the small kernels before them run
    /// (<c>AMQL_GPU_PREFETCH=0</c>).</summary>
    public bool PrefetchWeights { get; set; } = Environment.GetEnvironmentVariable("AMQL_GPU_PREFETCH")?.Trim() is not "0";

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
        public float[] GateQueryRows = [];
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
        // The engine's own chunk scratch (DepthChunk rows), and the one in use: it
        // or a long prompt's borrowed one.
        public DepthScratch Base = null!, W = null!;
        public int ScratchQ, ScratchC, ScratchRel, ScratchTaps, ScratchExperts, ScratchTopK, ScratchWidth, ScratchGates, ScratchRank;
        public IntPtr Logits;
        // A step's message mixes for a whole pass of the loop in one launch: every
        // loop layer's gate queries (they share the pass's bank), each layer's first row.
        public IntPtr LoopQueries, LoopMessages;
        public int LoopQueryCount;
        public float LoopMixEps;
        public int[] LoopQueryRow = [];
        public IntPtr HostLogits;
        public int MaxHeads, MaxExtentHeads;
    }

    /// <summary>One chunk's scratch, <see cref="Rows"/> rows each.</summary>
    private sealed class DepthScratch
    {
        public int Rows;
        public IntPtr[] H = [];
        public IntPtr Tokens, Q, Gate, Tmp, K, V, R, Sinks, RouterLogits, KDyn, VDyn, Mixed, Msg, Cat, GateVec,
            Attn, Xq, Xk, Xv, Nq, Nk, Nv, Res, FfnIn, FfnGated, Normed, Act, ExpertPart, Wt, Rb, Every;
        public IntPtr Idx, Order, Offsets, GroupOrder, Groups;
        public IntPtr SplitCounts;  // [Rows x heads] ints, zero between launches
        public IntPtr HostTokens;   // Rows tokens, then the position
        public IntPtr Part;         // a borrowed chunk's attention partials; zero: the sequence's
        public readonly List<IntPtr> Owned = new();
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
            L.GateQueryRows = queries;
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
        if (plan.DepthLoop is { } loop)
        {
            var gated = Enumerable.Range(loop.Start, loop.End - loop.Start).Where(l => layers[l].GateSites is not null).ToArray();
            if (gated.Length > 0 && gated.Select(l => layers[l].GateMixEps).Distinct().Count() == 1)
            {
                vd.LoopQueryRow = Enumerable.Repeat(-1, layers.Length).ToArray();
                var rows = new List<float>();
                foreach (int l in gated)
                {
                    vd.LoopQueryRow[l] = rows.Count / d;
                    rows.AddRange(layers[l].GateQueryRows);
                }
                vd.LoopQueryCount = rows.Count / d;
                vd.LoopQueries = UploadFloats(rows.ToArray());
                vd.LoopMessages = Alloc((long)rows.Count * 4);
                vd.LoopMixEps = layers[gated[0]].GateMixEps;
            }
        }

        int maxQ = layers.Max(l => l.Heads * l.HeadDim), maxC = layers.Max(l => l.KvHeads * l.HeadDim);
        int maxHeads = layers.Max(l => l.Heads), maxKv = layers.Max(l => l.KvHeads);
        int maxRel = Math.Max(1, layers.Max(l => l.Heads * l.RelRank));
        int maxTaps = Math.Max(1, layers.Max(l => l.KvHeads * l.ConvKernel));
        int experts = layers.Max(l => l.Experts), topK = layers.Max(l => l.TopK), width = layers.Max(l => l.Width);
        int gates = Math.Max(1, layers.Max(l => l.GateSites?.Length ?? 0)), rank = Math.Max(1, layers.Max(l => l.GateRank));
        vd.MaxHeads = maxHeads;
        vd.MaxExtentHeads = Math.Max(1, layers.Max(l => l.Heads * Math.Max(l.Extent, 1)));
        vd.ScratchQ = maxQ;
        vd.ScratchC = maxC;
        vd.ScratchRel = maxRel;
        vd.ScratchTaps = maxTaps;
        vd.ScratchExperts = experts;
        vd.ScratchTopK = topK;
        vd.ScratchWidth = width;
        vd.ScratchGates = gates;
        vd.ScratchRank = rank;
        _visits = vd;
        vd.Base = vd.W = NewScratch(DepthChunk, part: false, dry: false, out _);
        vd.Logits = Alloc((long)vd.Vocab * 4);
        Check(Native.amqe_host_alloc(out vd.HostLogits, (nuint)((long)vd.Vocab * 4)), "host alloc");
        _visits = vd;
    }

    /// <summary>A chunk's scratch for <paramref name="rows"/> rows, with its own
    /// attention partials when <paramref name="part"/>; with <paramref name="dry"/>
    /// only its size; with <paramref name="every"/> room for every row's logits.
    /// <paramref name="bytes"/> is the device bytes it holds.</summary>
    private DepthScratch NewScratch(int rows, bool part, bool dry, out long bytes, bool every = false)
    {
        var vd = _visits!;
        int d = _hidden;
        long R = rows, total = 0;
        var w = new DepthScratch { Rows = rows };
        // One allocation carved in 256-byte steps, its size from a dry pass.
        IntPtr block = IntPtr.Zero;
        if (!dry)
        {
            NewScratch(rows, part, dry: true, out long size, every);
            block = Alloc(size);
            w.Owned.Add(block);
        }
        IntPtr A(long n)
        {
            var p = block == IntPtr.Zero ? IntPtr.Zero : block + (nint)total;
            total += (Math.Max(16, n) + 255) / 256 * 256;
            return p;
        }
        try
        {
            w.H = Enumerable.Range(0, vd.Schedule.Count + 1).Select(_ => A(R * d * 4)).ToArray();
            w.Tokens = A((R + 1) * 4);
            w.Q = A(R * vd.ScratchQ * 4);
            w.Gate = A(R * vd.ScratchQ * 4);
            w.Tmp = A(R * vd.ScratchQ * 4);
            w.K = A(R * vd.ScratchC * 4);
            w.V = A(R * vd.ScratchC * 4);
            w.R = A(R * vd.ScratchRel * 4);
            w.Sinks = A(R * vd.MaxHeads * 4);
            w.RouterLogits = A(R * vd.ScratchExperts * 4);
            w.KDyn = A(R * vd.ScratchTaps * 4);
            w.VDyn = A(R * vd.ScratchTaps * 4);
            w.Mixed = A(R * 3 * d * 4);
            w.Msg = A(R * vd.ScratchGates * d * 4);
            w.Cat = A(3 * R * 2 * d * 4);
            w.GateVec = A(3 * R * vd.ScratchRank * 4);
            w.Attn = A(R * d * 4);
            w.Xq = A(R * d * 4);
            w.Xk = A(R * d * 4);
            w.Xv = A(R * d * 4);
            w.Nq = A(R * d * 4);
            w.Nk = A(R * d * 4);
            w.Nv = A(R * d * 4);
            w.Res = A(R * d * 4);
            w.FfnIn = A(R * d * 4);
            w.FfnGated = A(R * d * 4);
            w.Normed = A(R * d * 4);
            w.Act = A(R * vd.ScratchTopK * vd.ScratchWidth * 4);
            w.ExpertPart = A(R * vd.ScratchTopK * d * 4);
            w.Wt = A(R * vd.ScratchTopK * 4);
            w.Idx = A(R * vd.ScratchTopK * 4);
            w.Order = A(R * vd.ScratchTopK * 4);
            w.Offsets = A((vd.ScratchExperts + 1) * 4L);
            w.GroupOrder = A(3 * (R * vd.ScratchTopK + vd.ScratchExperts) * 4);
            w.Groups = A(3 * (R * vd.ScratchTopK + vd.ScratchExperts) * 4);
            w.Rb = A(R * vd.MaxExtentHeads * 4);
            w.SplitCounts = A(R * vd.MaxHeads * 4);
            if (part)
            {
                w.Part = A(R * DepthRowSplits * vd.MaxHeads * 130 * 4);
            }
            if (every)
            {
                w.Every = A(R * vd.Vocab * 4);
            }
            if (!dry)
            {
                Check(Native.amqe_memset(_engine, w.SplitCounts, 0, (nuint)(R * vd.MaxHeads * 4)), "zero");
                Check(Native.amqe_host_alloc(out w.HostTokens, (nuint)((R + 1) * 4)), "host alloc");
            }
        }
        catch
        {
            FreeScratch(w);
            throw;
        }
        bytes = total;
        return w;
    }

    private void FreeScratch(DepthScratch w)
    {
        Native.amqe_sync(_engine);
        foreach (var p in w.Owned)
        {
            Free(p);
        }
        w.Owned.Clear();
        if (w.HostTokens != IntPtr.Zero)
        {
            Native.amqe_host_free(w.HostTokens);
            w.HostTokens = IntPtr.Zero;
        }
    }

    /// <summary>A larger chunk's scratch for a prompt of <paramref name="tokens"/>
    /// from the sequence's position, or null to run it in the engine's own: up to
    /// <see cref="MaxDepthChunk"/> rows (the prompt rounded up to 64), fewer while
    /// the scratch, the windowed slots' extra rows and <see cref="BorrowReserve"/>
    /// would not fit in the device's free memory. A windowed slot that has not
    /// wrapped grows to its window and the chunk; one that has bounds the chunk to
    /// the rows past its window, so a chunk never overwrites keys its earlier rows read.</summary>
    private DepthScratch? BorrowScratch(GpuSequence seq, int tokens, bool every)
    {
        var vd = _visits!;
        var s = seq.Depth!;
        int own = vd.Base.Rows;
        if (!BorrowChunks || tokens <= own)
        {
            return null;
        }
        long end = seq.Position + (long)tokens;
        int rows = Math.Min(MaxDepthChunk, (tokens + 63) / 64 * 64);
        foreach (var slot in s.Slots)
        {
            var L = vd.Layers[slot.Layer];
            if (slot.Ring && slot.Kv.Cap < end && seq.Position > slot.Kv.Cap)
            {
                rows = Math.Min(rows, (slot.Kv.Cap - L.Window) / 64 * 64);
            }
        }
        List<(DepthSlot Slot, int Rows)> Grows(int r)
        {
            var list = new List<(DepthSlot, int)>();
            foreach (var slot in s.Slots)
            {
                var L = vd.Layers[slot.Layer];
                int want = (int)Math.Min(s.Capacity, L.Window + (long)r);
                if (slot.Ring && slot.Kv.Cap < end && slot.Kv.Cap < want)
                {
                    list.Add((slot, want));
                }
            }
            return list;
        }
        long free = DeviceMemory().Free;
        for (; rows > own; rows -= 128)
        {
            NewScratch(rows, part: true, dry: true, out long need, every);
            foreach (var (slot, want) in Grows(rows))
            {
                need += DepthKvBytes(want, slot.C, vd.Layers[slot.Layer].KvHeads, slot.Kv.Format);
            }
            if (need + BorrowReserve <= free)
            {
                break;
            }
        }
        if (rows <= own)
        {
            return null;
        }
        try
        {
            var grows = Grows(rows);
            foreach (var (slot, want) in grows)
            {
                GrowSlot(s, slot, want, seq.Position);
            }
            if (grows.Count > 0)
            {
                DropDepthGraph(s);
            }
            return NewScratch(rows, part: true, dry: false, out _, every);
        }
        catch (InvalidOperationException)
        {
            // The memory went elsewhere meanwhile: the engine's own chunk.
            return null;
        }
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
            GrowSlot(s, slot, rows, seq.Position);
        }
        // A larger cache may take more splits than the partials were sized for.
        SeqFree(s, s.Part);
        s.Part = SeqAlloc(s, DepthPartBytes(s));
        Check(Native.amqe_sync(_engine), "grow");
        s.Capacity = capacity;
        DropDepthGraph(s);
    }

    /// <summary>A slot's rows moved to an allocation of <paramref name="rows"/>, its
    /// rows so far kept (a ring that grows has not wrapped).</summary>
    private void GrowSlot(DepthSequence s, DepthSlot slot, int rows, long position)
    {
        var L = _visits!.Layers[slot.Layer];
        // A ring that grows has not wrapped yet: it held fewer rows than the
        // positions so far would need only once the capacity exceeded it.
        var old = slot.Kv;
        var grown = AllocDepthKv(s, rows, slot.C, L.KvHeads, old.Format, slot.Ring);
        long n = Math.Min(position, old.Cap);
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

    // ── a forward pass ────────────────────────────────────────────────────────

    private static IntPtr At(IntPtr p, long floats) => p + (nint)(floats * 4);

    private void Proj(int T, params (DepthMat M, IntPtr X, IntPtr Y)[] list)
    {
        var m = list.Select(p => p.M).ToArray();
        var x = list.Select(p => p.X).ToArray();
        var y = list.Select(p => p.Y).ToArray();
        Check(DepthNative.amqe_projections(_engine, m, x, y, list.Length, T), "projections");
    }

    /// <summary>Projections of inputs each RMS-normed first with its gain W: a
    /// step's GEMVs norm from their own loads; a chunk's inputs are normed into Out.
    /// With <paramref name="keepFirst"/> the first input's normed row stays in its Out for a step too.</summary>
    private void ProjNormed(int T, float eps, bool keepFirst, params (DepthMat M, IntPtr X, IntPtr Y, IntPtr W, IntPtr Out)[] list)
    {
        Check(DepthNative.amqe_projections_normed(_engine, list.Select(p => p.M).ToArray(), list.Select(p => p.X).ToArray(),
            list.Select(p => p.Y).ToArray(), list.Length, T, list.Select(p => p.W).ToArray(), list.Select(p => p.Out).ToArray(), eps,
            keepFirst ? 1 : 0), "projections");
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
            vd.W.Cat, vd.W.GateVec, vd.W.Rows, T, _hidden, L.GateRank, L.GateEps), "message gates");
    }

    /// <summary>One visit of a layer for T rows at the positions from the token
    /// buffer's: its inputs (depth mix, message gates, norms), the attention and
    /// its residual conv, the residual join (or a new block), then the FFN the same
    /// way. Mirrors GenericRuntime.RunRowLayer.</summary>
    private IntPtr DepthVisit(DepthLayer L, DepthSlot slot, IntPtr h, List<IntPtr> bank, IntPtr messages, IntPtr output, int T,
        int position, DepthLayer? next)
    {
        var vd = _visits!;
        int d = _hidden;
        long td = (long)T * d;
        IntPtr pos = At(vd.W.Tokens, vd.W.Rows);
        int Gate(string site) => L.GateSites is { } sites && messages != IntPtr.Zero ? Array.IndexOf(sites, site) : -1;

        // The attention's inputs.
        IntPtr xq = h, xk = h, xv = h;
        bool normed = false;
        bool single = !L.SeparateInputs;
        if (L.MixerQueries != IntPtr.Zero)
        {
            bool fuse = messages == IntPtr.Zero && (L.MixerCount == 3 || single);
            IntPtr[]? w = fuse ? (L.MixerCount == 3 ? [L.QNorm, L.KNorm, L.VNorm] : [L.QNorm]) : null;
            IntPtr[]? o = fuse ? (L.MixerCount == 3 ? [vd.W.Nq, vd.W.Nk, vd.W.Nv] : [vd.W.Nq]) : null;
            Mix(bank, h, L.MixerQueries, L.MixerCount, vd.W.Mixed, T, L.MixerMixEps, w, o, L.AttnEps);
            normed = fuse;
            xq = vd.W.Mixed;
            xk = L.MixerCount == 3 ? At(vd.W.Mixed, td) : vd.W.Mixed;
            xv = L.MixerCount == 3 ? At(vd.W.Mixed, 2 * td) : vd.W.Mixed;
        }
        {
            var sites = new List<int>();
            var ins = new List<IntPtr>();
            var outs = new List<IntPtr>();
            foreach (var (site, x, to) in new[] { ("q", xq, vd.W.Xq), ("k", xk, vd.W.Xk), ("v", xv, vd.W.Xv) })
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
                if (Gate("q") >= 0) xq = vd.W.Xq;
                if (Gate("k") >= 0) xk = vd.W.Xk;
                if (Gate("v") >= 0) xv = vd.W.Xv;
                normed = false;
            }
        }
        // The attention's normed inputs: from the mix's kernel, else normed by the
        // projections themselves (a step's GEMVs from their own loads).
        IntPtr qIn, kIn, vIn;
        IntPtr qW = IntPtr.Zero, kW = IntPtr.Zero, vW = IntPtr.Zero;
        bool selfNorm = !normed;
        if (normed)
        {
            qIn = vd.W.Nq;
            kIn = L.MixerCount == 3 ? vd.W.Nk : vd.W.Nq;
            vIn = L.MixerCount == 3 ? vd.W.Nv : vd.W.Nq;
        }
        else if (single && xq == xk && xq == xv)
        {
            (qIn, kIn, vIn) = (xq, xq, xq);
            qW = kW = vW = L.QNorm;
        }
        else
        {
            (qIn, kIn, vIn) = (xq, xk, xv);
            (qW, kW, vW) = (L.QNorm, L.KNorm, L.VNorm);
        }
        IntPtr qOut = vd.W.Nq, kOut = kIn == qIn && kW == qW ? vd.W.Nq : vd.W.Nk, vOut = vIn == qIn && vW == qW ? vd.W.Nq : vIn == kIn && vW == kW ? kOut : vd.W.Nv;

        // The attention: projections sharing an input in one launch each.
        var qGroup = new List<(DepthMat, IntPtr, IntPtr)> { (L.Wq, qIn, vd.W.Q) };
        if (L.Rel is { } rel) qGroup.Add((rel, qIn, vd.W.R));
        if (L.SinkProj is { } sinkProj) qGroup.Add((sinkProj, qIn, vd.W.Sinks));
        if (L.OutGate is { } outGate) qGroup.Add((outGate, qIn, vd.W.Gate));
        var kvGroup = new List<(DepthMat, IntPtr, IntPtr, IntPtr, IntPtr)> { (L.Wk, kIn, vd.W.K, kW, kOut), (L.Wv, vIn, vd.W.V, vW, vOut) };
        if (L.KTaps is { } kTaps) kvGroup.Add((kTaps, kIn, vd.W.KDyn, kW, kOut));
        if (L.VTaps is { } vTaps) kvGroup.Add((vTaps, vIn, vd.W.VDyn, vW, vOut));
        // All of them in one launch (up to eight matrices), so one GEMV's ramp serves all.
        if (selfNorm)
        {
            ProjNormed(T, L.AttnEps, false, qGroup.Select(p => (p.Item1, p.Item2, p.Item3, qW, qOut)).Concat(kvGroup).ToArray());
        }
        else
        {
            Proj(T, qGroup.Concat(kvGroup.Select(p => (p.Item1, p.Item2, p.Item3))).ToArray());
        }
        // The output projection into L2 while the attention runs.
        Prefetch(T, L.Wo);

        // The queries' and coefficients' prep and the keys' and values' rows (one launch for a step).
        var kv = slot.Kv;
        Check(DepthNative.amqe_step_prep(_engine, vd.W.Q, L.QScale, L.HeadDim, L.QkClip,
            L.Rel is null ? IntPtr.Zero : vd.W.R, L.RelScale, L.RelRank, L.RelClip, L.Heads, L.LogScale, L.Floor, L.Alpha,
            ref kv, vd.W.K, vd.W.V, vd.W.KDyn, vd.W.VDyn, L.KConv, L.VConv, slot.KHist, slot.VHist,
            L.KMode, L.VMode, slot.C, L.HeadDim, L.ConvKernel, L.KvHeads, T, L.QkClip, L.VNormOn, L.VEps, pos), "queries, keys and values");

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
        // The splits merged with the sink and the output gate (by the last split block of each head, without tiles).
        IntPtr sinks = L.SinkProj is not null ? vd.W.Sinks : L.SinkStatic;
        int sinkStride = L.SinkProj is not null ? L.Heads : 0;
        Check(DepthNative.amqe_depth_attention_merged(_engine, vd.W.Q, vd.W.R, L.RelTable, vd.W.Rb, ref kv, DepthPartOf(),
            L.Heads, L.KvHeads, hd, L.RelRank, L.Extent, L.Window, splits, L.Scale, T, tiled ? 1 : 0, pos,
            vd.W.SplitCounts, sinks, sinkStride, L.OutGate is null ? IntPtr.Zero : vd.W.Gate, vd.W.Tmp), "attention");
        // The output projection, its residual conv and the join (a step's conv in
        // the GEMV's epilogue): a block's first layer keeps its input in the bank
        // and restarts the residual from the mixer's output.
        if (L.OpensBlock)
        {
            bank.Add(h);
        }
        var wo = L.Wo;
        Check(DepthNative.amqe_projection_conv(_engine, ref wo, vd.W.Tmp, vd.W.Attn, T, L.MixerConv, slot.MixerHist,
            L.OpensBlock ? IntPtr.Zero : h, vd.W.Res, L.MixerConvKernel, pos), "output projection and mixer conv");

        // The FFN's input.
        int ffnGate = Gate("ffn");
        IntPtr ffnIn = vd.W.Res;
        bool ffnNormed = false;
        if (L.FfnQueries != IntPtr.Zero)
        {
            bool fuse = ffnGate < 0;
            Mix(bank, vd.W.Res, L.FfnQueries, 1, vd.W.FfnIn, T, L.FfnMixEps, fuse ? [L.FfnNorm] : null, fuse ? [vd.W.Normed] : null, L.FfnEps);
            ffnNormed = fuse;
            ffnIn = vd.W.FfnIn;
        }
        if (ffnGate >= 0)
        {
            Gates(L, [ffnGate], [ffnIn], messages, [vd.W.FfnGated], T);
            ffnIn = vd.W.FfnGated;
        }

        // The experts: sigmoid routing over the chosen (inside the experts' first
        // kernel for a step), a softcapped GLU; their weighted sum in the conv.
        // Unless the mix normed it, the router's projection norms the FFN's input
        // and leaves it normed for the experts.
        if (ffnNormed)
        {
            Proj(T, (L.Router, vd.W.Normed, vd.W.RouterLogits));
        }
        else
        {
            ProjNormed(T, L.FfnEps, true, (L.Router, ffnIn, vd.W.RouterLogits, L.FfnNorm, vd.W.Normed));
        }
        var gu = L.GateUp;
        var dn = L.DownExperts;
        Check(DepthNative.amqe_packed_experts(_engine, ref gu, ref dn, L.Experts, d, L.Width, L.TopK, L.GateCap, L.UpCap,
            vd.W.Normed, T, vd.W.Idx, vd.W.Wt, vd.W.Act, vd.W.ExpertPart, IntPtr.Zero, vd.W.Order, vd.W.Offsets, vd.W.GroupOrder, vd.W.Groups,
            vd.W.RouterLogits, L.SelectionBias, L.RouteScale), "experts");
        // The next visit's attention projections into L2 while its inputs are made.
        if (next is not null)
        {
            Prefetch(T, AttentionMatrices(next));
        }
        Check(DepthNative.amqe_residual_conv_parts(_engine, vd.W.ExpertPart, vd.W.Wt, L.TopK, L.FfnConv, slot.FfnHist, vd.W.Res, output, d,
            L.FfnConvKernel, T, pos), "ffn conv");
        return output;
    }

    private DepthSequence? _depthNow;

    private IntPtr DepthPartOf() => _visits!.W.Part != IntPtr.Zero ? _visits.W.Part : _depthNow!.Part;

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
        IntPtr h = vd.W.H[buf++];
        var embed = vd.Embed;
        Check(DepthNative.amqe_embed_rows(_engine, ref embed, vd.W.Tokens, vd.EmbedNorm, vd.EmbedNormed ? 1 : 0, vd.EmbedEps, h, T), "embed");
        Prefetch(T, AttentionMatrices(vd.Layers[vd.Schedule[0].Layer]));
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
                    if (T == 1 && vd.LoopQueries != IntPtr.Zero)
                    {
                        // Every gate's message of the pass, [layer gates][d], from one mix.
                        Mix(messageBank, messageOut, vd.LoopQueries, vd.LoopQueryCount, vd.LoopMessages, 1, vd.LoopMixEps);
                    }
                }
            }
            var L = vd.Layers[visit.Layer];
            IntPtr messages = IntPtr.Zero;
            if (visit.Pass > 0 && L.GateSites is { } sites)
            {
                if (T == 1 && vd.LoopQueries != IntPtr.Zero && vd.LoopQueryRow[visit.Layer] >= 0)
                {
                    messages = At(vd.LoopMessages, (long)vd.LoopQueryRow[visit.Layer] * d);
                }
                else
                {
                    // Every gate's message, [gates][T][d], from one mix.
                    Mix(messageBank!, messageOut, L.GateQueries, sites.Length, vd.W.Msg, T, L.GateMixEps);
                    messages = vd.W.Msg;
                }
            }
            var nextVisit = buf < vd.Schedule.Count ? vd.Layers[vd.Schedule[buf].Layer] : null;
            h = DepthVisit(L, s.Slots[visit.CacheSlot], h, bank, messages, vd.W.H[buf++], T, position, nextVisit);
        }
        if (_plan.BlockResidual is not null)
        {
            Mix(bank, h, vd.OutputQuery, 1, vd.W.Mixed, T, vd.OutputMixEps, [vd.FinalNorm], [vd.W.Normed], vd.FinalEps);
        }
        else
        {
            Norms(T, vd.FinalEps, (h, vd.FinalNorm, vd.W.Normed));
        }
        if (every)
        {
            Proj(T, (vd.Head, vd.W.Normed, vd.W.Every));
            if (vd.Softcap is { } cap)
            {
                Check(DepthNative.amqe_softcap(_engine, vd.W.Every, (long)T * vd.Vocab, cap), "softcap");
            }
        }
        else
        {
            Proj(1, (vd.Head, At(vd.W.Normed, td - d), vd.Logits));
            if (vd.Softcap is { } cap)
            {
                Check(DepthNative.amqe_softcap(_engine, vd.Logits, vd.Vocab, cap), "softcap");
            }
        }
        Check(DepthNative.amqe_l2_join(_engine), "prefetch");
    }

    private static DepthMat[] AttentionMatrices(DepthLayer L) =>
        new DepthMat?[] { L.Wq, L.Rel, L.SinkProj, L.OutGate, L.Wk, L.Wv, L.KTaps, L.VTaps }.OfType<DepthMat>().ToArray();

    /// <summary>A step's matrices into L2 on a side branch (<see cref="PrefetchWeights"/>).</summary>
    private void Prefetch(int T, params DepthMat[] mats)
    {
        if (T != 1 || !PrefetchWeights || mats.Length == 0)
        {
            return;
        }
        Check(DepthNative.amqe_l2_prefetch(_engine, mats.Select(m => m.W).ToArray(),
            mats.Select(m => (long)m.Rows * m.Cols * (m.Type == F32 ? 4 : 2)).ToArray(), mats.Length), "prefetch");
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
            Marshal.WriteInt32(vd.W.HostTokens, i * 4, tokens[i]);
        }
        Marshal.WriteInt32(vd.W.HostTokens, vd.W.Rows * 4, position);
        Check(Native.amqe_copy_async(_engine, vd.W.Tokens, vd.W.HostTokens, (nuint)(tokens.Length * 4), 1), "tokens");
        Check(Native.amqe_copy_async(_engine, At(vd.W.Tokens, vd.W.Rows), At(vd.W.HostTokens, vd.W.Rows), 4, 1), "position");
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
        if (every && vd.W.Every == IntPtr.Zero)
        {
            vd.W.Every = Alloc((long)vd.W.Rows * vocab * 4);
            vd.W.Owned.Add(vd.W.Every);
        }
        var result = new float[every ? tokens.Length : 1][];
        var borrowed = BorrowScratch(seq, tokens.Length, every);
        _lastChunkBorrowed = borrowed is not null;
        LastPromptChunk = borrowed?.Rows ?? vd.Base.Rows;
        try
        {
            DepthChunks(seq, tokens, every, borrowed, result);
        }
        finally
        {
            vd.W = vd.Base;
            if (borrowed is not null)
            {
                FreeScratch(borrowed);
            }
        }
        return result;
    }

    // Whether the last prompt ran in a borrowed chunk (its visit outputs are freed).
    private bool _lastChunkBorrowed;

    private void DepthChunks(GpuSequence seq, ReadOnlySpan<int> tokens, bool every, DepthScratch? borrowed, float[][] result)
    {
        var s = seq.Depth!;
        var vd = _visits!;
        int vocab = vd.Vocab;
        int rows = borrowed?.Rows ?? vd.Base.Rows;
        for (int done = 0; done < tokens.Length;)
        {
            int T = Math.Min(rows, tokens.Length - done);
            bool last = done + T == tokens.Length;
            // A step runs (and its graph is captured) on the engine's own scratch.
            vd.W = T > 1 && borrowed is not null ? borrowed : vd.Base;
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
                    Check(Native.amqe_download(_engine, handle.AddrOfPinnedObject(), vd.W.Every, (nuint)((long)T * vocab * 4)), "logits");
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
        if (visit < 0 || visit >= vd.Schedule.Count || rows < 1 || rows > vd.Base.Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(visit));
        }
        if (_lastChunkBorrowed)
        {
            throw new InvalidOperationException("the last prompt ran in a borrowed chunk, whose visit outputs are freed");
        }
        Check(Native.amqe_sync(_engine), "visit output");
        var flat = new float[rows * _hidden];
        var handle = GCHandle.Alloc(flat, GCHandleType.Pinned);
        try
        {
            Check(Native.amqe_download(_engine, handle.AddrOfPinnedObject(), vd.W.H[visit + 1], (nuint)((long)rows * _hidden * 4)), "visit output");
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
        [DllImport(Lib)] public static extern int amqe_projections_normed(IntPtr e, DepthMat[] m, IntPtr[] x, IntPtr[] y, int n, int rows,
            IntPtr[] normW, IntPtr[] normOut, float eps, int keepFirst);
        [DllImport(Lib)] public static extern int amqe_rms_rows(IntPtr e, IntPtr[] x, IntPtr[] w, IntPtr[] output, int n, int rows, int d, float eps);
        [DllImport(Lib)] public static extern int amqe_depth_mix(IntPtr e, IntPtr[] states, int n, IntPtr queries, int count, IntPtr output,
            int rows, int d, float eps, IntPtr[]? normW, IntPtr[]? normOut, float normEps);
        [DllImport(Lib)] public static extern int amqe_l2_prefetch(IntPtr e, IntPtr[] ptrs, long[] bytes, int n);
        [DllImport(Lib)] public static extern int amqe_l2_join(IntPtr e);
        [DllImport(Lib)] public static extern int amqe_projection_conv(IntPtr e, ref DepthMat m, IntPtr x, IntPtr y, int rows,
            IntPtr w, IntPtr hist, IntPtr add, IntPtr output, int k, IntPtr pos);
        [DllImport(Lib)] public static extern int amqe_message_gates(IntPtr e, int n, IntPtr[] h, IntPtr[] m, IntPtr[] output,
            DepthMat[] down, DepthMat[] forget, DepthMat[] gain, IntPtr cat, IntPtr g, int stride, int rows, int d, int rank, float eps);
        [DllImport(Lib)] public static extern int amqe_step_prep(IntPtr e, IntPtr q, IntPtr qScale, int qDim, float qClip,
            IntPtr r, IntPtr rScale, int rDim, float rClip, int heads, int logScale, int floor, float alpha,
            ref DepthKv slot, IntPtr kRaw, IntPtr vRaw, IntPtr kDyn, IntPtr vDyn,
            IntPtr kConv, IntPtr vConv, IntPtr kHist, IntPtr vHist, int kMode, int vMode, int c, int hd, int k, int kvHeads, int rows,
            float clip, int vnorm, float eps, IntPtr pos);
        [DllImport(Lib)] public static extern int amqe_depth_attention_merged(IntPtr e, IntPtr q, IntPtr r, IntPtr relProj, IntPtr rb, ref DepthKv kv,
            IntPtr part, int heads, int kvHeads, int hd, int dRel, int extent, int window, int splits, float scale, int rows, int tiled, IntPtr pos,
            IntPtr counter, IntPtr sinks, int sinkStride, IntPtr gate, IntPtr output);
        [DllImport(Lib)] public static extern int amqe_depth_attention_splits(int cap, int ring);
        [DllImport(Lib)] public static extern int amqe_residual_conv(IntPtr e, IntPtr x, IntPtr w, IntPtr hist, IntPtr add, IntPtr output,
            int c, int k, int rows, IntPtr pos);
        [DllImport(Lib)] public static extern int amqe_residual_conv_parts(IntPtr e, IntPtr part, IntPtr wt, int topK, IntPtr w, IntPtr hist,
            IntPtr add, IntPtr output, int c, int k, int rows, IntPtr pos);
        [DllImport(Lib)] public static extern int amqe_packed_experts(IntPtr e, ref DepthExperts gateUp, ref DepthExperts down, int experts,
            int d, int width, int topK, float gateCap, float upCap, IntPtr x, int rows, IntPtr idx, IntPtr wt, IntPtr act, IntPtr part,
            IntPtr ffn, IntPtr order, IntPtr offsets, IntPtr groupOrder, IntPtr groups, IntPtr logits, IntPtr selectionBias, float routeScale);
        [DllImport(Lib)] public static extern int amqe_expert_row_scale(IntPtr e, IntPtr factor, int rows, long count, IntPtr output);
        [DllImport(Lib)] public static extern int amqe_embed_rows(IntPtr e, ref DepthMat table, IntPtr tokens, IntPtr normW, int normed, float eps,
            IntPtr output, int rows);
        [DllImport(Lib)] public static extern int amqe_softcap(IntPtr e, IntPtr x, long n, float cap);
        [DllImport(Lib)] public static extern int amqe_mem_info(out nuint free, out nuint total);
    }
}
