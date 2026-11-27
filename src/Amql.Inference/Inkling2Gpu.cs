using System.Runtime.InteropServices;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>A model fed one sequence a token at a time: what a text session
/// needs of a runtime that is not a planned stack (Inkling 2 looped).</summary>
public interface ITokenModel : IDisposable
{
    int VocabSize { get; }
    int Position { get; }
    /// <summary><c>gpu</c> or <c>cpu</c>.</summary>
    string Backend { get; }
    float[] Prefill(IReadOnlyList<int> tokens);
    float[] Step(int token);
    void Reset();
}

/// <summary>
/// Inkling 2 looped on the GPU (<c>native/amql_cuda/amql_inkling.cu</c>, built
/// into amql_cuda beside the engine, whose GEMMs and grouped experts run a
/// prompt's chunks). The weights go up as stored: the experts NVFP4 (elements,
/// E4M3 block scales and the per-expert factor amax / denominator) or MXFP8
/// (elements and E8M0 exponents) or plain, the large matrices BF16, the small
/// tensors f32. A step runs as one captured CUDA graph that reads its token
/// and position from device memory. Each sequence has every layer visit's
/// slot: sliding slots a ring of the window in f32, full slots growing rows in
/// the requested KV format (f32, fp16 or nvfp4).
/// </summary>
public sealed class Inkling2Gpu : IDisposable
{
    private static class Native
    {
        private const string Lib = "amql_cuda";
        [DllImport(Lib)] public static extern int ink_available();
        [DllImport(Lib)] public static extern int ink_mem_info(out long free, out long total);
        [DllImport(Lib)] public static extern int ink_create(ref Config cfg, out IntPtr ctx);
        [DllImport(Lib)] public static extern int ink_upload(IntPtr ctx, byte[] host, long bytes, out IntPtr dev);
        [DllImport(Lib)] public static extern int ink_upload(IntPtr ctx, float[] host, long bytes, out IntPtr dev);
        [DllImport(Lib)] public static extern int ink_set_geometry(IntPtr ctx, int layer, int full, int heads, int kvHeads, int headDim, int extent, int loop);
        [DllImport(Lib)] public static extern int ink_set_mat(IntPtr ctx, int layer, int which, IntPtr dev, int type, int rows, int cols);
        [DllImport(Lib)] public static extern int ink_set_vec(IntPtr ctx, int layer, int which, IntPtr dev);
        [DllImport(Lib)] public static extern int ink_set_experts(IntPtr ctx, int layer, int which, IntPtr w, IntPtr scales, IntPtr factor, int type, int rows, int cols);
        [DllImport(Lib)] public static extern int ink_set_global(IntPtr ctx, int which, IntPtr dev, int type, int rows, int cols);
        [DllImport(Lib)] public static extern long ink_device_bytes(IntPtr ctx);
        [DllImport(Lib)] public static extern int ink_seq_create(IntPtr ctx, int capacity, int format, out IntPtr seq);
        [DllImport(Lib)] public static extern int ink_seq_grow(IntPtr ctx, IntPtr seq, int capacity);
        [DllImport(Lib)] public static extern long ink_seq_bytes(IntPtr seq);
        [DllImport(Lib)] public static extern long ink_seq_estimate(IntPtr ctx, int capacity, int format);
        [DllImport(Lib)] public static extern int ink_seq_reset(IntPtr ctx, IntPtr seq);
        [DllImport(Lib)] public static extern int ink_seq_free(IntPtr ctx, IntPtr seq);
        [DllImport(Lib)] public static extern int ink_step(IntPtr ctx, IntPtr seq, int token, float[]? logits, int graphs);
        [DllImport(Lib)] public static extern int ink_prefill(IntPtr ctx, IntPtr seq, int[] tokens, int count, float[]? logits, int every, int graphs);
        [DllImport(Lib)] public static extern int ink_free(IntPtr ctx);
        [DllImport(Lib)] public static extern int ink_seq_visit(IntPtr ctx, IntPtr seq, int visit, float[] output, int rows);
        [DllImport(Lib)] public static extern IntPtr ink_error_string(int code);
        [DllImport(Lib)] public static extern int ink_set_prefill_f32(IntPtr ctx, int f32);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Config
    {
        public int D, Layers, Experts, TopK, Width, ConvK, Block, LoopStart, LoopEnd, Passes, Rank;
        public int PaddedVocab, Vocab, RelDim, RelExtent, Window, LogFloor;
        public float Alpha, QkClip, RelClip, Eps, GateCap, UpCap, FinalCap, RouteScale;
    }

    private const int WF32 = 0, WBf16 = 1, WNvfp4 = 2, WMxfp8 = 3;
    public const int KvF32 = 0, KvF16 = 1, KvFp4 = 2;

    private enum M { Wq, Wk, Wv, Wr, Wo, Gate, Sink, KProj, VProj, Router, Ctrl }
    private enum V { QScale, RelScale, RelProj, KConv, VConv, KStatic, VStatic, AttnNorm, QNorm, KNorm, VNorm, MlpNorm, AttnConv, MlpConv, RouterBias, AttnQuery, MlpQuery }
    private enum G { Embed, Unembed, EmbedNorm, FinalNorm, OutputQuery, Messages }

    private IntPtr _ctx;
    private readonly Inkling2Surface _f;

    public int VocabSize => _f.UnpaddedVocab;
    /// <summary>The positions a sequence may reach (the model's own context unless set: with no
    /// rotary, nothing in the arithmetic depends on it).</summary>
    public long ContextLength { get; set; }

    /// <summary>Device bytes the weights hold.</summary>
    public long WeightBytes => Native.ink_device_bytes(_ctx);

    /// <summary>The KV format new sequences use: <see cref="GpuEngine.RequestedKvFormat"/> unless set.</summary>
    public string KvFormat { get; set; } = GpuEngine.RequestedKvFormat;

    /// <summary>True: a prompt's chunks keep activations at f32 (f32 SGEMMs over
    /// widened weights, the experts on f32 kernels), which holds them within 1e-3
    /// of the managed runtime. False (the default): the engine's BF16 tensor
    /// cores, activations rounded to BF16 and products summed in f32, as
    /// <see cref="GpuEngine.ExactPrefill"/>. Steps are f32 either way.</summary>
    public bool ExactPrefill
    {
        get => _exactPrefill;
        set
        {
            Check(Native.ink_set_prefill_f32(_ctx, value ? 1 : 0), "prefill precision");
            _exactPrefill = value;
        }
    }

    private bool _exactPrefill;

    /// <summary>False runs each step's launches directly instead of replaying the captured graph.</summary>
    public bool UseGraphs { get; set; } = Environment.GetEnvironmentVariable("AMQL_INKLING_GRAPHS")?.Trim() is not "0";

    private Inkling2Gpu(IntPtr ctx, Inkling2Surface facts, long context)
    {
        _ctx = ctx;
        _f = facts;
        ContextLength = context;
    }

    private static bool? _available;

    /// <summary>Whether the library loads and sees a device.</summary>
    public static bool Available
    {
        get
        {
            if (_available is { } known)
            {
                return known;
            }
            try
            {
                _available = Native.ink_available() == 1;
            }
            catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
            {
                _available = false;
            }
            return _available.Value;
        }
    }

    /// <summary>The device's free and total memory, in bytes.</summary>
    public static (long Free, long Total) DeviceMemory()
    {
        Check(Native.ink_mem_info(out long free, out long total), "cudaMemGetInfo");
        return (free, total);
    }

    private static void Check(int rc, string what)
    {
        if (rc != 0)
        {
            string text = Marshal.PtrToStringAnsi(Native.ink_error_string(rc)) ?? "";
            throw new InvalidOperationException($"Inkling 2 GPU: {what} failed ({rc}: {text})");
        }
    }

    /// <summary>The engine for an Inkling 2 container, or null with the reason
    /// (no device, the library absent, <c>AMQL_GPU_ENGINE=0</c>, too little device memory).</summary>
    public static Inkling2Gpu? TryCreate(Vindex3Container container, OperandStore store, out string? reason, string componentId = "target")
    {
        string? mode = Environment.GetEnvironmentVariable("AMQL_GPU_ENGINE")?.Trim().ToLowerInvariant();
        if (mode is "0" or "off" or "false")
        {
            reason = "AMQL_GPU_ENGINE=0";
            return null;
        }
        if (!Available)
        {
            reason = "no CUDA device or amql_cuda";
            return null;
        }
        var component = (container.Graph ?? throw new ContainerException("container records no system graph")).Component(componentId);
        if (component.Execution?.Inkling2 is null)
        {
            reason = "not an Inkling 2 looped stack";
            return null;
        }
        long need = store.TensorNames("target.decoder_stack").Sum(n => PayloadBytes(store, "target.decoder_stack", n))
                    + new[] { "target.embedding", "target.final_norm", "target.output_head" }
                        .Sum(o => store.TensorNames(o).Sum(n => PayloadBytes(store, o, n)));
        Check(Native.ink_mem_info(out long free, out _), "cudaMemGetInfo");
        if (need + (1L << 30) > free)
        {
            reason = $"the weights need {need / (1 << 20)} MiB and the device has {free / (1 << 20)} MiB free";
            return null;
        }
        try
        {
            reason = null;
            return Load(container, store, component);
        }
        catch (InvalidOperationException e)
        {
            reason = e.Message;
            return null;
        }
    }

    private static long PayloadBytes(OperandStore store, string obj, string name)
    {
        var shape = store.ShapeOf(obj, name);
        long count = shape.Aggregate(1L, (a, b) => a * b);
        return store.DtypeOf(obj, name) switch
        {
            Dtype.FP4 => count / 2,
            Dtype.F8_E4M3 or Dtype.U8 or Dtype.F8_E8M0 => count,
            Dtype.BF16 or Dtype.F16 => count * 2,
            _ => count * 4,
        };
    }

    private static Inkling2Gpu Load(Vindex3Container container, OperandStore store, Component component)
    {
        var f = component.Execution!.Inkling2!;
        int d = component.HiddenSize, k = f.ConvKernel;
        var cfg = new Config
        {
            D = d, Layers = component.NumLayers, Experts = f.Experts, TopK = f.TopK, Width = f.ExpertWidth, ConvK = k,
            Block = f.BlockSize, LoopStart = f.LoopStart, LoopEnd = f.LoopEnd, Passes = f.LoopPasses, Rank = f.GateRank,
            PaddedVocab = f.PaddedVocab, Vocab = f.UnpaddedVocab, RelDim = f.RelativeDim, RelExtent = f.RelativeExtent,
            Window = f.SlidingWindow, LogFloor = f.LogScalingFloor,
            Alpha = (float)f.LogScalingAlpha, QkClip = (float)f.QkClip, RelClip = (float)f.RelativeClip, Eps = (float)f.Eps,
            GateCap = (float)f.GateSoftcap, UpCap = (float)f.UpSoftcap, FinalCap = (float)f.FinalSoftcap, RouteScale = (float)f.RouteScale,
        };
        Check(Native.ink_create(ref cfg, out var ctx), "create");
        var gpu = new Inkling2Gpu(ctx, f, component.Execution.ContextLength ?? int.MaxValue);
        try
        {
            gpu.Upload(store, component, f, d, k);
        }
        catch
        {
            gpu.Dispose();
            throw;
        }
        return gpu;
    }

    private IntPtr Up(byte[] bytes)
    {
        Check(Native.ink_upload(_ctx, bytes, bytes.LongLength, out var dev), "upload");
        return dev;
    }

    private IntPtr Up(float[] values)
    {
        Check(Native.ink_upload(_ctx, values, values.LongLength * 4, out var dev), "upload");
        return dev;
    }

    private void Upload(OperandStore store, Component component, Inkling2Surface f, int d, int k)
    {
        const string stack = "target.decoder_stack", mixing = "target.final_norm";

        OperandResolution R(string obj, string name, params long[] shape)
        {
            var r = store.Resolve(obj, name);
            if (!r.Shape.SequenceEqual(shape))
            {
                throw new ContainerException($"{obj}/{name}: expected [{string.Join("x", shape)}], found [{string.Join("x", r.Shape)}]");
            }
            return r;
        }
        IntPtr Vec(string obj, string name, params long[] shape)
        {
            var r = R(obj, name, shape);
            return Up(BitPattern.WidenToF32(r.Dtype, r.Payload));
        }
        (IntPtr Dev, int Type) MatOf(string obj, string name, long rows, long cols)
        {
            var r = R(obj, name, rows, cols);
            return r.Dtype switch
            {
                Dtype.BF16 => (Up(r.Payload), WBf16),
                Dtype.F32 => (Up(r.Payload), WF32),
                _ => (Up(BitPattern.WidenToF32(r.Dtype, r.Payload)), WF32),
            };
        }
        void Mat(int layer, int which, string name, long rows, long cols)
        {
            var (dev, type) = MatOf(stack, name, rows, cols);
            Check(Native.ink_set_mat(_ctx, layer, which, dev, type, checked((int)rows), checked((int)cols)), $"layer {layer} matrix {name}");
        }
        void SetVec(int layer, V which, IntPtr dev) => Check(Native.ink_set_vec(_ctx, layer, (int)which, dev), $"layer {layer} {which}");
        void Experts(int layer, int which, string name, int experts, int rows, int cols)
        {
            var dtype = store.DtypeOf(stack, name);
            if (!store.ShapeOf(stack, name).SequenceEqual(new long[] { experts, rows, cols }))
            {
                throw new ContainerException($"{stack}/{name}: expected [{experts}x{rows}x{cols}]");
            }
            IntPtr w, scales = IntPtr.Zero, factor = IntPtr.Zero;
            int type;
            switch (dtype)
            {
                case Dtype.FP4:
                {
                    var scale = R(stack, name + "_scale", experts, rows, cols / 16);
                    var amax = BitPattern.WidenToF32(Dtype.F32, R(stack, name + "_amax", experts).Payload);
                    var den = BitPattern.WidenToF32(Dtype.F32, R(stack, name + "_outer_scale_denominator", 1).Payload);
                    var fac = new float[experts];
                    for (int e = 0; e < experts; e++)
                    {
                        fac[e] = amax[e] / den[0];
                    }
                    w = Up(store.Resolve(stack, name).Payload);
                    scales = Up(scale.Payload);
                    factor = Up(fac);
                    type = WNvfp4;
                    break;
                }
                case Dtype.F8_E4M3:
                {
                    var scale = R(stack, name + "_scale", experts, rows, cols / 32);
                    w = Up(store.Resolve(stack, name).Payload);
                    scales = Up(scale.Payload);
                    type = WMxfp8;
                    break;
                }
                default:
                    var plain = R(stack, name, experts, rows, cols);
                    (w, type) = plain.Dtype switch
                    {
                        Dtype.BF16 => (Up(plain.Payload), WBf16),
                        Dtype.F32 => (Up(plain.Payload), WF32),
                        _ => (Up(BitPattern.WidenToF32(plain.Dtype, plain.Payload)), WF32),
                    };
                    break;
            }
            Check(Native.ink_set_experts(_ctx, layer, which, w, scales, factor, type, rows, cols), $"layer {layer} experts {name}");
        }

        for (int l = 0; l < component.NumLayers; l++)
        {
            bool full = f.FullLayers[l];
            int heads = full ? f.FullHeads : f.SlidingHeads;
            int kvHeads = full ? f.FullKvHeads : f.SlidingKvHeads;
            int hd = full ? f.FullHeadDim : f.SlidingHeadDim;
            int extent = full ? f.RelativeExtent : f.SlidingWindow;
            int c = kvHeads * hd;
            bool loop = l >= f.LoopStart && l < f.LoopEnd;
            string a = $"{l}.attn.";
            Check(Native.ink_set_geometry(_ctx, l, full ? 1 : 0, heads, kvHeads, hd, extent, loop ? 1 : 0), $"layer {l} geometry");
            Mat(l, (int)M.Wq, a + "wq_du.weight", heads * hd, d);
            Mat(l, (int)M.Wk, a + "wk_dv.weight", c, d);
            Mat(l, (int)M.Wv, a + "wv_dv.weight", c, d);
            Mat(l, (int)M.Wr, a + "wr_du.weight", heads * f.RelativeDim, d);
            Mat(l, (int)M.Wo, a + "wo_ud.weight", d, heads * hd);
            Mat(l, (int)M.Gate, a + "gate_attn.weight", heads * hd, d);
            Mat(l, (int)M.Sink, a + "attn_sink_proj.weight", heads, d);
            Mat(l, (int)M.Router, $"{l}.mlp.gate.weight", f.Experts, d);
            SetVec(l, V.QScale, Vec(stack, a + "q_l2_clip.scale", heads, hd));
            SetVec(l, V.RelScale, Vec(stack, a + "rel_l2_clip.scale", heads, f.RelativeDim));
            SetVec(l, V.RelProj, Vec(stack, a + "rel_logits_proj.proj", f.RelativeDim, extent));
            if (full)
            {
                Mat(l, (int)M.KProj, a + "k_sconv.weight_proj", kvHeads * k, d);
                Mat(l, (int)M.VProj, a + "v_sconv.weight_proj", kvHeads * k, d);
                SetVec(l, V.KStatic, Vec(stack, a + "k_sconv.static_w", k, c));
                SetVec(l, V.VStatic, Vec(stack, a + "v_sconv.static_w", k, c));
                SetVec(l, V.QNorm, Vec(stack, $"{l}.q_input_norm.weight", d));
                SetVec(l, V.KNorm, Vec(stack, $"{l}.k_input_norm.weight", d));
                SetVec(l, V.VNorm, Vec(stack, $"{l}.v_input_norm.weight", d));
            }
            else
            {
                SetVec(l, V.KConv, Vec(stack, a + "k_sconv.weight", c, 1, k));
                SetVec(l, V.VConv, Vec(stack, a + "v_sconv.weight", c, 1, k));
                SetVec(l, V.AttnNorm, Vec(stack, $"{l}.attn_norm.weight", d));
            }
            SetVec(l, V.MlpNorm, Vec(stack, $"{l}.mlp_norm.weight", d));
            SetVec(l, V.AttnConv, Vec(stack, $"{l}.attn_sconv.weight", d, 1, k));
            SetVec(l, V.MlpConv, Vec(stack, $"{l}.mlp_sconv.weight", d, 1, k));
            SetVec(l, V.RouterBias, Vec(stack, $"{l}.mlp.gate.bias", f.Experts));
            if (l > 0)
            {
                SetVec(l, V.AttnQuery, full
                    ? Vec(mixing, $"attn_res.attn_queries.{l - 1}", 3, d)
                    : Vec(mixing, $"attn_res.attn_queries.{l - 1}", d));
            }
            SetVec(l, V.MlpQuery, Vec(mixing, $"attn_res.mlp_queries.{l}", d));
            if (loop)
            {
                string[] gates = { "attn", "k", "v", "mlp" };
                for (int g = 0; g < 4; g++)
                {
                    Mat(l, (int)M.Ctrl + 3 * g, $"{l}.controllers.{gates[g]}.down_weight", f.GateRank, 2 * d);
                    Mat(l, (int)M.Ctrl + 3 * g + 1, $"{l}.controllers.{gates[g]}.forget_weight", d, f.GateRank);
                    Mat(l, (int)M.Ctrl + 3 * g + 2, $"{l}.controllers.{gates[g]}.gain_weight", d, f.GateRank);
                }
            }
            Experts(l, 0, $"{l}.mlp.experts.w13_weight", f.Experts, 2 * f.ExpertWidth, d);
            Experts(l, 1, $"{l}.mlp.experts.w2_weight", f.Experts, d, f.ExpertWidth);
        }

        // [q, mlp] and [k, v] per core layer in the checkpoint; q, k, v, mlp here.
        int core = f.LoopEnd - f.LoopStart;
        var qmr = R(mixing, "middle_cycle_cross_pass_queries", 2 * core, d);
        var kvr = R(mixing, "middle_cycle_cross_pass_kv_queries", 2 * core, d);
        var qm = BitPattern.WidenToF32(qmr.Dtype, qmr.Payload);
        var kv = BitPattern.WidenToF32(kvr.Dtype, kvr.Payload);
        var messages = new float[core * 4 * d];
        for (int i = 0; i < core; i++)
        {
            Array.Copy(qm, 2 * i * d, messages, (4 * i + 0) * d, d);
            Array.Copy(kv, 2 * i * d, messages, (4 * i + 1) * d, d);
            Array.Copy(kv, (2 * i + 1) * d, messages, (4 * i + 2) * d, d);
            Array.Copy(qm, (2 * i + 1) * d, messages, (4 * i + 3) * d, d);
        }
        void Global(G which, IntPtr dev, int type = WF32, int rows = 0, int cols = 0) =>
            Check(Native.ink_set_global(_ctx, (int)which, dev, type, rows, cols), $"global {which}");
        Global(G.Messages, Up(messages));
        var embed = MatOf("target.embedding", "embed.weight", f.PaddedVocab, d);
        Global(G.Embed, embed.Dev, embed.Type, f.PaddedVocab, d);
        var unembed = MatOf("target.output_head", "weight", f.PaddedVocab, d);
        Global(G.Unembed, unembed.Dev, unembed.Type, f.PaddedVocab, d);
        Global(G.EmbedNorm, Vec("target.embedding", "embed_norm.weight", d));
        Global(G.FinalNorm, Vec(mixing, "norm.weight", d));
        Global(G.OutputQuery, Vec(mixing, "attn_res.output_query", d));
    }

    public static int FormatCode(string format) => format switch
    {
        "f32" => KvF32,
        "nvfp4" => KvFp4,
        _ => KvF16,
    };

    /// <summary>Device bytes a sequence of <paramref name="positions"/> would hold in <paramref name="format"/>.</summary>
    public long SequenceBytes(int positions, string format) => Native.ink_seq_estimate(_ctx, positions, FormatCode(format));

    /// <summary>A new sequence, room for <paramref name="capacity"/> positions to start with (it grows).</summary>
    public Sequence NewSequence(int capacity = 4096) => new(this, capacity, FormatCode(KvFormat));

    public void Dispose()
    {
        if (_ctx != IntPtr.Zero)
        {
            Native.ink_free(_ctx);
            _ctx = IntPtr.Zero;
        }
    }

    /// <summary>One sequence's slots on the device.</summary>
    public sealed class Sequence : ITokenModel
    {
        private readonly Inkling2Gpu _gpu;
        private IntPtr _seq;
        private int _capacity;

        internal Sequence(Inkling2Gpu gpu, int capacity, int format)
        {
            _gpu = gpu;
            Format = format;
            _capacity = Math.Max(16, capacity);
            Check(Native.ink_seq_create(gpu._ctx, _capacity, format, out _seq), $"a sequence of {_capacity} positions");
        }

        public int Format { get; }
        public int VocabSize => _gpu.VocabSize;
        public int Position { get; private set; }
        public string Backend => "gpu";

        /// <summary>Device bytes this sequence holds (slots and scratch).</summary>
        public long DeviceBytes => Native.ink_seq_bytes(_seq);

        private void Room(int more)
        {
            long need = (long)Position + more;
            if (need > _gpu.ContextLength)
            {
                throw new InvalidOperationException($"{need} positions exceed the model's context of {_gpu.ContextLength}");
            }
            if (need <= _capacity)
            {
                return;
            }
            int grown = (int)Math.Min(Math.Max(need, (long)_capacity * 2), Math.Min(_gpu.ContextLength, int.MaxValue / 2));
            Check(Native.ink_seq_grow(_gpu._ctx, _seq, grown), $"growing to {grown} positions");
            _capacity = grown;
        }

        public float[] Prefill(IReadOnlyList<int> tokens)
        {
            Room(tokens.Count);
            var logits = new float[VocabSize];
            Check(Native.ink_prefill(_gpu._ctx, _seq, tokens.ToArray(), tokens.Count, logits, 0, _gpu.UseGraphs ? 1 : 0), "prefill");
            Position += tokens.Count;
            return logits;
        }

        /// <summary>Every position's logits (count x vocabulary).</summary>
        public float[][] PrefillAll(IReadOnlyList<int> tokens)
        {
            Room(tokens.Count);
            var flat = new float[(long)tokens.Count * VocabSize];
            Check(Native.ink_prefill(_gpu._ctx, _seq, tokens.ToArray(), tokens.Count, flat, 1, _gpu.UseGraphs ? 1 : 0), "prefill");
            Position += tokens.Count;
            return Enumerable.Range(0, tokens.Count).Select(t => flat.AsSpan(t * VocabSize, VocabSize).ToArray()).ToArray();
        }

        /// <summary>Each layer visit's output (visit, row, hidden) for the prompt
        /// just fed by <see cref="PrefillAll"/> or <see cref="Prefill"/>, when it fit in
        /// one chunk (256 tokens): the layer outputs a reference records.</summary>
        public float[][][] VisitOutputs(int visits, int rows, int hidden)
        {
            var result = new float[visits][][];
            var flat = new float[rows * hidden];
            for (int v = 0; v < visits; v++)
            {
                Check(Native.ink_seq_visit(_gpu._ctx, _seq, v, flat, rows), $"visit {v}");
                result[v] = Enumerable.Range(0, rows).Select(r => flat.AsSpan(r * hidden, hidden).ToArray()).ToArray();
            }
            return result;
        }

        public float[] Step(int token)
        {
            Room(1);
            var logits = new float[VocabSize];
            Check(Native.ink_step(_gpu._ctx, _seq, token, logits, _gpu.UseGraphs ? 1 : 0), "step");
            Position++;
            return logits;
        }

        public void Reset()
        {
            Check(Native.ink_seq_reset(_gpu._ctx, _seq), "reset");
            Position = 0;
        }

        public void Dispose()
        {
            if (_seq != IntPtr.Zero)
            {
                Native.ink_seq_free(_gpu._ctx, _seq);
                _seq = IntPtr.Zero;
            }
        }
    }
}

/// <summary>One sequence on the Inkling 2 GPU path that owns the engine as
/// well: what a single-request text session holds.</summary>
public sealed class Inkling2GpuSession : ITokenModel
{
    private readonly Inkling2Gpu _gpu;
    private readonly Inkling2Gpu.Sequence _seq;

    public Inkling2GpuSession(Inkling2Gpu gpu, int capacity = 4096)
    {
        _gpu = gpu;
        _seq = gpu.NewSequence(capacity);
    }

    public Inkling2Gpu Engine => _gpu;
    public int VocabSize => _seq.VocabSize;
    public int Position => _seq.Position;
    public string Backend => "gpu";
    public float[] Prefill(IReadOnlyList<int> tokens) => _seq.Prefill(tokens);
    public float[] Step(int token) => _seq.Step(token);
    public void Reset() => _seq.Reset();

    public void Dispose()
    {
        _seq.Dispose();
        _gpu.Dispose();
    }
}
