using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Inkling 2 looped (<c>inkling2_loop</c>) on the managed path, one token at a
/// time, transcribed from transformers' <c>modeling_inkling2_loop.py</c> (each
/// method names the module it mirrors). Every state that crosses positions
/// is causal (keys and values, and the histories of the short convolutions),
/// so feeding a prompt token by token is the reference's whole-sequence pass.
/// <code>
///   h = embed_norm(embed[id]);  blocks = []
///   layers [0, start)                       one pass
///   layers [start, end)                     pass 0
///   pass p &gt; 0:  bank = pass p-1's blocks after its first, and its output
///                h, blocks = the state after layer start-1
///                layers [start, end), each gated on messages read from bank
///   layers [end, n)
///   logits = softcap(unembed · norm(mix(output_query, blocks, h)))
/// </code>
/// A layer: <c>x = mix(attn_query, blocks, h)</c> (three mixes on full layers,
/// one each for q, k and v); attention with no rotary but a relative bias
/// over the last <c>extent</c> keys and a sink logit; a short convolution of
/// its output; at a block's first layer the input becomes a block and the
/// residual restarts, otherwise it is added; then <c>mix(mlp_query, …)</c>,
/// the experts and another short convolution. Each pass of the core has its
/// own key, value and convolution slots, as the reference's extra cache layers.
/// <para>
/// Memory: the expert stacks stay as stored (NVFP4 elements with their E4M3
/// block scales and per-expert factor, MXFP8 with its E8M0 exponents, or a
/// plain float type), and only the rows of the experts a token routes to are
/// decoded, one row at a time. Large 16-bit matrices (the embedding, the head
/// and the attention projections) stay 16-bit and are widened as they are
/// read; everything small is widened to f32 once. So the real Inkling-2-Mini
/// holds about 19 GB (15.7 GB of it experts), its 19.5 GB checkpoint less the
/// vision tower. A tensor stored F32 is read as F32, as before.
/// </para>
/// </summary>
public sealed class Inkling2Model : ITokenModel
{
    /// <summary>A 2-D weight held as f32, or as BF16/F16 bits widened as each
    /// row is read (which is exact, so it changes nothing but memory).</summary>
    private sealed class Mat
    {
        private readonly float[]? _f32;
        private readonly byte[]? _half;
        private readonly bool _isF16;

        private Mat(float[]? f32, byte[]? half, bool isF16, int rows, int cols)
        {
            _f32 = f32;
            _half = half;
            _isF16 = isF16;
            Rows = rows;
            Cols = cols;
        }

        public int Rows { get; }
        public int Cols { get; }

        /// <summary>16-bit payloads of at least this many elements stay 16-bit.</summary>
        public const long KeepHalfElements = 1L << 20;

        public static Mat From(Dtype dtype, byte[] payload, int rows, int cols)
        {
            if (dtype is Dtype.BF16 or Dtype.F16 && (long)rows * cols >= KeepHalfElements)
            {
                if (payload.LongLength != (long)rows * cols * 2)
                {
                    throw new ContainerException($"a {dtype.Label()} [{rows}x{cols}] matrix needs {(long)rows * cols * 2} bytes, not {payload.LongLength}");
                }
                return new Mat(null, payload, dtype == Dtype.F16, rows, cols);
            }
            var values = BitPattern.WidenToF32(dtype, payload);
            if (values.LongLength != (long)rows * cols)
            {
                throw new ContainerException($"a [{rows}x{cols}] matrix needs {(long)rows * cols} values, not {values.LongLength}");
            }
            return new Mat(values, null, false, rows, cols);
        }

        public double Dot(int row, ReadOnlySpan<float> x)
        {
            if (_f32 is { } f)
            {
                return Inkling2Model.Dot(f.AsSpan(checked(row * Cols), Cols), x);
            }
            var bits = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(
                _half.AsSpan(checked(row * Cols * 2), Cols * 2));
            double sum = 0;
            if (_isF16)
            {
                for (int i = 0; i < bits.Length; i++)
                {
                    sum += (double)(float)BitConverter.UInt16BitsToHalf(bits[i]) * x[i];
                }
            }
            else
            {
                for (int i = 0; i < bits.Length; i++)
                {
                    sum += (double)BitConverter.Int32BitsToSingle(bits[i] << 16) * x[i];
                }
            }
            return sum;
        }

        public void Row(int row, Span<float> target)
        {
            if (_f32 is { } f)
            {
                f.AsSpan(checked(row * Cols), Cols).CopyTo(target);
                return;
            }
            var bits = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(
                _half.AsSpan(checked(row * Cols * 2), Cols * 2));
            for (int i = 0; i < bits.Length; i++)
            {
                target[i] = _isF16 ? (float)BitConverter.UInt16BitsToHalf(bits[i]) : BitConverter.Int32BitsToSingle(bits[i] << 16);
            }
        }

        /// <summary><c>W·x</c>, rows spread over the budgeted cores when the
        /// matrix is large; each row's sum is the same either way.</summary>
        public float[] MatVec(float[] x)
        {
            var y = new float[Rows];
            if ((long)Rows * Cols < ParallelElements)
            {
                for (int r = 0; r < Rows; r++)
                {
                    y[r] = (float)Dot(r, x);
                }
                return y;
            }
            Parallel.For(0, Rows, Budget, r => y[r] = (float)Dot(r, x));
            return y;
        }

        public float[] MatVec(float[] x, int rows)
        {
            if (rows == Rows)
            {
                return MatVec(x);
            }
            var y = new float[rows];
            Parallel.For(0, rows, Budget, r => y[r] = (float)Dot(r, x));
            return y;
        }
    }

    /// <summary>
    /// One layer's experts, <c>[experts, rows, cols]</c>, as stored: plain, or
    /// NVFP4 (FP4 elements, an E4M3 scale per 16 and the factor
    /// <c>amax / outer_scale_denominator</c> per expert), or MXFP8 (E4M3
    /// elements and an E8M0 exponent per 32). A row decodes exactly as
    /// transformers' <c>Inkling2LoopDequantizeExperts</c> computes it in f32:
    /// <c>(element × block scale) × factor</c>. Every pass of the loop reads
    /// the same stack and the same factor.
    /// </summary>
    private sealed class ExpertStack
    {
        private readonly Mat? _plain;          // [experts·rows, cols]
        private readonly Dtype _element;       // FP4 or F8_E4M3 when scaled
        private readonly byte[]? _elements;
        private readonly byte[]? _scales;      // one byte per block
        private readonly bool _e8m0;
        private readonly float[]? _factor;     // NVFP4: per expert
        private readonly int _block;

        public ExpertStack(int experts, int rows, int cols, Mat plain)
        {
            Experts = experts;
            Rows = rows;
            Cols = cols;
            _plain = plain;
        }

        public ExpertStack(int experts, int rows, int cols, Dtype element, byte[] elements, byte[] scales, bool e8m0,
            int block, float[]? factor)
        {
            Experts = experts;
            Rows = rows;
            Cols = cols;
            long count = (long)experts * rows * cols;
            if (cols % block != 0 || elements.LongLength != (element == Dtype.FP4 ? count / 2 : count) ||
                scales.LongLength != count / block || factor is not null && factor.Length != experts)
            {
                throw new ContainerException(
                    $"expert stack [{experts}, {rows}, {cols}] of {element.Label()}: {elements.LongLength} element bytes, " +
                    $"{scales.LongLength} scales of one per {block}{(factor is null ? string.Empty : $", {factor.Length} factors")} do not fit");
            }
            _element = element;
            _elements = elements;
            _scales = scales;
            _e8m0 = e8m0;
            _block = block;
            _factor = factor;
        }

        public int Experts { get; }
        public int Rows { get; }
        public int Cols { get; }

        /// <summary>How the stack is held: plain, NVFP4 or MXFP8.</summary>
        public string Form => _plain is not null ? "plain" : _factor is not null ? "NVFP4" : "MXFP8";

        private static readonly float[] E4M3 = Enumerable.Range(0, 256).Select(i => BitPattern.DecodeF8E4M3((byte)i)).ToArray();
        private static readonly float[] E8M0 = Enumerable.Range(0, 256).Select(i => BitPattern.DecodeF8E8M0((byte)i)).ToArray();

        public void Row(int expert, int row, Span<float> target)
        {
            if (_plain is { } plain)
            {
                plain.Row(checked(expert * Rows + row), target);
                return;
            }
            long index = (long)expert * Rows + row;
            int rowBytes = _element == Dtype.FP4 ? Cols / 2 : Cols;
            int blocks = Cols / _block;
            Span<float> scale = blocks <= 512 ? stackalloc float[blocks] : new float[blocks];
            var scaleBytes = _scales.AsSpan(checked((int)(index * blocks)), blocks);
            var table = _e8m0 ? E8M0 : E4M3;
            for (int b = 0; b < blocks; b++)
            {
                scale[b] = table[scaleBytes[b]];
            }
            ScaledMatrix.DequantRowOf(_element, _elements.AsSpan(checked((int)(index * rowBytes)), rowBytes), Cols,
                scale, _block, _factor?[expert] ?? 1f, target);
        }
    }

    /// <summary>Matrices of at least this many elements spread their rows over cores.</summary>
    private const long ParallelElements = 1L << 18;

    private static readonly ParallelOptions Budget = new() { MaxDegreeOfParallelism = Math.Max(1, ComputeBudget.Cores) };

    private sealed class Gate
    {
        public required float[] Down { get; init; }   // [rank, 2d]
        public required float[] Forget { get; init; } // [d, rank]
        public required float[] Gain { get; init; }   // [d, rank]
    }

    private sealed class Layer
    {
        public required bool Full { get; init; }
        public required int Heads { get; init; }
        public required int KvHeads { get; init; }
        public required int HeadDim { get; init; }
        public required int Extent { get; init; }
        public required Mat Wq { get; init; }
        public required Mat Wk { get; init; }
        public required Mat Wv { get; init; }
        public required Mat Wr { get; init; }
        public required Mat Wo { get; init; }
        public required Mat OutGate { get; init; }
        public required float[] Sink { get; init; }
        public required float[] QScale { get; init; }
        public required float[] RelScale { get; init; }
        public required float[] RelProj { get; init; }   // [d_rel, extent]
        public float[]? KConv { get; init; }             // static: [C, K]
        public float[]? VConv { get; init; }
        public float[]? KStatic { get; init; }           // dynamic: [K, C]
        public float[]? VStatic { get; init; }
        public float[]? KProj { get; init; }             // dynamic: [kv_heads*K, d]
        public float[]? VProj { get; init; }
        public float[]? AttnNorm { get; init; }          // sliding layers
        public float[]? QNorm { get; init; }             // full layers
        public float[]? KNorm { get; init; }
        public float[]? VNorm { get; init; }
        public required float[] MlpNorm { get; init; }
        public required float[] AttnConv { get; init; }  // [d, K]
        public required float[] MlpConv { get; init; }
        public required float[] Router { get; init; }    // [E, d]
        public required float[] RouterBias { get; init; }
        public required ExpertStack W13 { get; init; }   // [E, 2I, d], gate and up rows interleaved
        public required ExpertStack W2 { get; init; }    // [E, d, I]
        public float[]? AttnQuery { get; init; }         // [1 or 3, d]; none on layer 0
        public required float[] MlpQuery { get; init; }
        public Gate[]? Gates { get; init; }              // q, k, v, mlp: loop layers only
    }

    /// <summary>One cache slot: a physical layer's keys, values and
    /// convolution histories in one pass.</summary>
    private sealed class Slot
    {
        public readonly List<float[]> Keys = new();
        public readonly List<float[]> Values = new();
        public readonly List<float[]> KHistory = new();
        public readonly List<float[]> VHistory = new();
        public readonly List<float[]> AttnHistory = new();
        public readonly List<float[]> MlpHistory = new();
    }

    private readonly Inkling2Surface _f;
    private readonly int _d;
    private readonly Mat _embed;
    private readonly float[] _embedNorm;
    private readonly float[] _finalNorm;
    private readonly float[] _outputQuery;
    private readonly Mat _unembed;
    private readonly float[] _messageQueries; // [core layers, 4, d]: q, k, v, mlp
    private readonly Layer[] _layers;
    private Slot[] _slots;

    public int Position { get; private set; }
    public int VocabSize => _f.UnpaddedVocab;
    public int NumLayers => _layers.Length;

    /// <summary>How each layer's experts are held: "plain", "NVFP4" or "MXFP8".</summary>
    public IReadOnlyList<string> ExpertForms => _layers.Select(l => l.W13.Form).ToList();
    public long ContextLength { get; }

    /// <summary>Called with each layer visit's input (visit index, hidden
    /// state), in the order transformers records <c>hidden_states</c>.</summary>
    public Action<int, float[]>? OnVisit { get; set; }

    /// <summary>Called with each layer visit's output (visit index, hidden
    /// state): what a forward hook on the decoder layer sees.</summary>
    public Action<int, float[]>? OnVisitOutput { get; set; }

    private Inkling2Model(Inkling2Surface facts, int hidden, Mat embed, float[] embedNorm, float[] finalNorm,
        float[] outputQuery, Mat unembed, float[] messageQueries, Layer[] layers, long context)
    {
        _f = facts;
        _d = hidden;
        _embed = embed;
        _embedNorm = embedNorm;
        _finalNorm = finalNorm;
        _outputQuery = outputQuery;
        _unembed = unembed;
        _messageQueries = messageQueries;
        _layers = layers;
        ContextLength = context;
        _slots = NewSlots();
    }

    private int CoreLayers => _f.LoopEnd - _f.LoopStart;

    private Slot[] NewSlots() =>
        Enumerable.Range(0, _layers.Length + (_f.LoopPasses - 1) * CoreLayers).Select(_ => new Slot()).ToArray();

    public static bool Serves(Vindex3Container container, string componentId = "target") =>
        container.Graph?.Components.FirstOrDefault(c => c.Id == componentId)?.Execution?.Inkling2 is not null;

    public static Inkling2Model Load(Vindex3Container container, OperandStore store, string componentId = "target")
    {
        var graph = container.Graph ?? throw new ContainerException("container records no system graph");
        var component = graph.Component(componentId);
        var f = component.Execution?.Inkling2
            ?? throw new UnsupportedOperatorException($"component '{componentId}' is not an Inkling 2 looped stack");
        int d = component.HiddenSize;
        int k = f.ConvKernel;
        const string stack = "target.decoder_stack";
        const string mixing = "target.final_norm";

        float[] T(string obj, string name, params long[] shape)
        {
            var r = store.Resolve(obj, name);
            if (!r.Shape.SequenceEqual(shape))
            {
                throw new ContainerException($"{obj}/{name}: expected [{string.Join("x", shape)}], found [{string.Join("x", r.Shape)}]");
            }
            return BitPattern.WidenToF32(r.Dtype, r.Payload);
        }

        Mat M(string obj, string name, long rows, long cols)
        {
            var r = store.Resolve(obj, name);
            if (!r.Shape.SequenceEqual(new[] { rows, cols }))
            {
                throw new ContainerException($"{obj}/{name}: expected [{rows}x{cols}], found [{string.Join("x", r.Shape)}]");
            }
            return Mat.From(r.Dtype, r.Payload, checked((int)rows), checked((int)cols));
        }

        // An expert stack as stored, with whichever companions its form has.
        ExpertStack E(string name, int experts, int rows, int cols)
        {
            var dtype = store.DtypeOf(stack, name);
            var shape = store.ShapeOf(stack, name);
            if (!shape.SequenceEqual(new long[] { experts, rows, cols }))
            {
                throw new ContainerException($"{stack}/{name}: expected [{experts}x{rows}x{cols}], found [{string.Join("x", shape)}]");
            }
            switch (dtype)
            {
                case Dtype.FP4:
                {
                    var scale = store.Resolve(stack, name + "_scale");
                    if (scale.Dtype != Dtype.F8_E4M3 || !scale.Shape.SequenceEqual(new long[] { experts, rows, cols / 16 }))
                    {
                        throw new ContainerException($"{stack}/{name}_scale: NVFP4 needs F8_E4M3 [{experts}x{rows}x{cols / 16}]");
                    }
                    var amax = T(stack, name + "_amax", experts);
                    var denominator = T(stack, name + "_outer_scale_denominator", 1);
                    var factor = new float[experts];
                    for (int e = 0; e < experts; e++)
                    {
                        // f32 division, as transformers' amax.float() / denominator.float().
                        factor[e] = amax[e] / denominator[0];
                    }
                    return new ExpertStack(experts, rows, cols, Dtype.FP4, store.Resolve(stack, name).Payload, scale.Payload,
                        e8m0: false, block: 16, factor);
                }
                case Dtype.F8_E4M3:
                {
                    var scale = store.Resolve(stack, name + "_scale");
                    if (scale.Dtype is not (Dtype.U8 or Dtype.F8_E8M0) || !scale.Shape.SequenceEqual(new long[] { experts, rows, cols / 32 }))
                    {
                        throw new ContainerException($"{stack}/{name}_scale: MXFP8 needs U8 exponents [{experts}x{rows}x{cols / 32}]");
                    }
                    return new ExpertStack(experts, rows, cols, Dtype.F8_E4M3, store.Resolve(stack, name).Payload, scale.Payload,
                        e8m0: true, block: 32, factor: null);
                }
                default:
                    return new ExpertStack(experts, rows, cols,
                        Mat.From(dtype, store.Resolve(stack, name).Payload, checked(experts * rows), cols));
            }
        }

        var layers = new Layer[component.NumLayers];
        for (int l = 0; l < layers.Length; l++)
        {
            bool full = f.FullLayers[l];
            int heads = full ? f.FullHeads : f.SlidingHeads;
            int kvHeads = full ? f.FullKvHeads : f.SlidingKvHeads;
            int hd = full ? f.FullHeadDim : f.SlidingHeadDim;
            int extent = full ? f.RelativeExtent : f.SlidingWindow;
            int c = kvHeads * hd;
            string a = $"{l}.attn.";
            bool loop = l >= f.LoopStart && l < f.LoopEnd;
            int experts = f.Experts;
            layers[l] = new Layer
            {
                Full = full,
                Heads = heads,
                KvHeads = kvHeads,
                HeadDim = hd,
                Extent = extent,
                Wq = M(stack, a + "wq_du.weight", heads * hd, d),
                Wk = M(stack, a + "wk_dv.weight", c, d),
                Wv = M(stack, a + "wv_dv.weight", c, d),
                Wr = M(stack, a + "wr_du.weight", heads * f.RelativeDim, d),
                Wo = M(stack, a + "wo_ud.weight", d, heads * hd),
                OutGate = M(stack, a + "gate_attn.weight", heads * hd, d),
                Sink = T(stack, a + "attn_sink_proj.weight", heads, d),
                QScale = T(stack, a + "q_l2_clip.scale", heads, hd),
                RelScale = T(stack, a + "rel_l2_clip.scale", heads, f.RelativeDim),
                RelProj = T(stack, a + "rel_logits_proj.proj", f.RelativeDim, extent),
                KConv = full ? null : T(stack, a + "k_sconv.weight", c, 1, k),
                VConv = full ? null : T(stack, a + "v_sconv.weight", c, 1, k),
                KStatic = full ? T(stack, a + "k_sconv.static_w", k, c) : null,
                VStatic = full ? T(stack, a + "v_sconv.static_w", k, c) : null,
                KProj = full ? T(stack, a + "k_sconv.weight_proj", kvHeads * k, d) : null,
                VProj = full ? T(stack, a + "v_sconv.weight_proj", kvHeads * k, d) : null,
                AttnNorm = full ? null : T(stack, $"{l}.attn_norm.weight", d),
                QNorm = full ? T(stack, $"{l}.q_input_norm.weight", d) : null,
                KNorm = full ? T(stack, $"{l}.k_input_norm.weight", d) : null,
                VNorm = full ? T(stack, $"{l}.v_input_norm.weight", d) : null,
                MlpNorm = T(stack, $"{l}.mlp_norm.weight", d),
                AttnConv = T(stack, $"{l}.attn_sconv.weight", d, 1, k),
                MlpConv = T(stack, $"{l}.mlp_sconv.weight", d, 1, k),
                Router = T(stack, $"{l}.mlp.gate.weight", experts, d),
                RouterBias = T(stack, $"{l}.mlp.gate.bias", experts),
                W13 = E($"{l}.mlp.experts.w13_weight", experts, 2 * f.ExpertWidth, d),
                W2 = E($"{l}.mlp.experts.w2_weight", experts, d, f.ExpertWidth),
                AttnQuery = l == 0 ? null
                    : full ? T(mixing, $"attn_res.attn_queries.{l - 1}", 3, d) : T(mixing, $"attn_res.attn_queries.{l - 1}", d),
                MlpQuery = T(mixing, $"attn_res.mlp_queries.{l}", d),
                Gates = loop
                    ? new[] { "attn", "k", "v", "mlp" }.Select(g => new Gate
                    {
                        Down = T(stack, $"{l}.controllers.{g}.down_weight", f.GateRank, 2 * d),
                        Forget = T(stack, $"{l}.controllers.{g}.forget_weight", d, f.GateRank),
                        Gain = T(stack, $"{l}.controllers.{g}.gain_weight", d, f.GateRank),
                    }).ToArray()
                    : null,
            };
        }

        // The checkpoint keeps [q, mlp] and [k, v] per core layer in two
        // tensors; the runtime reads them as q, k, v, mlp.
        int core = f.LoopEnd - f.LoopStart;
        var qm = T(mixing, "middle_cycle_cross_pass_queries", 2 * core, d);
        var kv = T(mixing, "middle_cycle_cross_pass_kv_queries", 2 * core, d);
        var messages = new float[core * 4 * d];
        for (int i = 0; i < core; i++)
        {
            Array.Copy(qm, (2 * i) * d, messages, (4 * i + 0) * d, d);
            Array.Copy(kv, (2 * i) * d, messages, (4 * i + 1) * d, d);
            Array.Copy(kv, (2 * i + 1) * d, messages, (4 * i + 2) * d, d);
            Array.Copy(qm, (2 * i + 1) * d, messages, (4 * i + 3) * d, d);
        }

        return new Inkling2Model(f, d,
            M("target.embedding", "embed.weight", f.PaddedVocab, d),
            T("target.embedding", "embed_norm.weight", d),
            T(mixing, "norm.weight", d),
            T(mixing, "attn_res.output_query", d),
            M("target.output_head", "weight", f.PaddedVocab, d),
            messages, layers, component.Execution!.ContextLength ?? long.MaxValue);
    }

    string ITokenModel.Backend => "cpu";

    float[] ITokenModel.Prefill(IReadOnlyList<int> tokens) => Prefill(tokens)[^1];

    void IDisposable.Dispose()
    {
    }

    public void Reset()
    {
        _slots = NewSlots();
        Position = 0;
    }

    /// <summary>Feeds the tokens in turn and returns every position's logits.</summary>
    public float[][] Prefill(IReadOnlyList<int> tokens) => tokens.Select(Step).ToArray();

    /// <summary>Feeds one token and returns the logits for the next.</summary>
    public float[] Step(int token)
    {
        if ((uint)token >= (uint)_f.PaddedVocab)
        {
            throw new ArgumentOutOfRangeException(nameof(token), $"token {token} is outside the vocabulary of {_f.PaddedVocab}");
        }
        int d = _d;
        int visit = 0;
        var row = new float[d];
        _embed.Row(token, row);
        var h = RmsNorm(row, _embedNorm);
        var blocks = new List<float[]>();

        for (int l = 0; l < _f.LoopStart; l++)
        {
            OnVisit?.Invoke(visit, h);
            visit++;
            h = RunLayer(l, _slots[l], h, blocks, null);
            OnVisitOutput?.Invoke(visit - 1, h);
        }
        var prelude = h;
        var preludeBlocks = blocks.ToList();
        int core = CoreLayers;
        for (int pass = 0; pass < _f.LoopPasses; pass++)
        {
            List<float[]>? bank = null;
            float[]? bankOut = null;
            if (pass > 0)
            {
                bank = blocks.Skip(preludeBlocks.Count + 1).ToList();
                bankOut = h;
                h = prelude;
                blocks = preludeBlocks.ToList();
            }
            for (int ip = _f.LoopStart; ip < _f.LoopEnd; ip++)
            {
                int slot = pass == 0 ? ip : _layers.Length + (pass - 1) * core + (ip - _f.LoopStart);
                OnVisit?.Invoke(visit, h);
                visit++;
                float[][]? messages = pass == 0 ? null : Messages(bank!, bankOut!, ip - _f.LoopStart);
                h = RunLayer(ip, _slots[slot], h, blocks, messages);
                OnVisitOutput?.Invoke(visit - 1, h);
            }
        }
        for (int l = _f.LoopEnd; l < _layers.Length; l++)
        {
            OnVisit?.Invoke(visit, h);
            visit++;
            h = RunLayer(l, _slots[l], h, blocks, null);
            OnVisitOutput?.Invoke(visit - 1, h);
        }

        h = RmsNorm(Mix(_outputQuery, 1, blocks, h)[0], _finalNorm);
        // The head's padded rows past the unpadded vocabulary are never read.
        var logits = _unembed.MatVec(h, _f.UnpaddedVocab);
        double cap = _f.FinalSoftcap;
        for (int v = 0; v < logits.Length; v++)
        {
            logits[v] = (float)(Math.Tanh(logits[v] / cap) * cap);
        }
        Position++;
        return logits;
    }

    // ── a layer (Inkling2Loop{Sliding,Full}[Loop]DecoderLayer) ───────────────

    private float[] RunLayer(int l, Slot slot, float[] h, List<float[]> blocks, float[][]? messages)
    {
        var layer = _layers[l];
        var blockResidual = h;
        float[] qIn, kIn, vIn;
        if (layer.Full)
        {
            float[] xq = h, xk = h, xv = h;
            if (layer.AttnQuery is { } query)
            {
                var mixed = Mix(query, 3, blocks, h);
                (xq, xk, xv) = (mixed[0], mixed[1], mixed[2]);
            }
            if (messages is not null)
            {
                xq = MessageGate(layer.Gates![0], xq, messages[0]);
                xk = MessageGate(layer.Gates![1], xk, messages[1]);
                xv = MessageGate(layer.Gates![2], xv, messages[2]);
            }
            qIn = RmsNorm(xq, layer.QNorm!);
            kIn = RmsNorm(xk, layer.KNorm!);
            vIn = RmsNorm(xv, layer.VNorm!);
        }
        else
        {
            var x = layer.AttnQuery is { } query ? Mix(query, 1, blocks, h)[0] : h;
            if (messages is not null)
            {
                qIn = RmsNorm(MessageGate(layer.Gates![0], x, messages[0]), layer.AttnNorm!);
                kIn = RmsNorm(MessageGate(layer.Gates![1], x, messages[1]), layer.AttnNorm!);
                vIn = RmsNorm(MessageGate(layer.Gates![2], x, messages[2]), layer.AttnNorm!);
            }
            else
            {
                qIn = kIn = vIn = RmsNorm(x, layer.AttnNorm!);
            }
        }

        var attn = Attention(layer, slot, qIn, kIn, vIn);
        attn = ShortConv(layer.AttnConv, slot.AttnHistory, attn);

        float[] residual;
        if (l % _f.BlockSize == 0)
        {
            blocks.Add(blockResidual);
            residual = attn;
        }
        else
        {
            residual = Add(blockResidual, attn);
        }

        var mlpIn = Mix(layer.MlpQuery, 1, blocks, residual)[0];
        if (messages is not null)
        {
            mlpIn = MessageGate(layer.Gates![3], mlpIn, messages[3]);
        }
        var ffn = Experts(layer, RmsNorm(mlpIn, layer.MlpNorm));
        ffn = ShortConv(layer.MlpConv, slot.MlpHistory, ffn);
        return Add(residual, ffn);
    }

    // ── attention (Inkling2Loop{Sliding,Full}Attention) ──────────────────────

    private float[] Attention(Layer layer, Slot slot, float[] qIn, float[] kIn, float[] vIn)
    {
        int d = _d, heads = layer.Heads, kvHeads = layer.KvHeads, hd = layer.HeadDim;
        int c = kvHeads * hd;
        int t = Position;
        double tau = layer.Full
            ? 1.0 + _f.LogScalingAlpha * Math.Log(Math.Max(1.0, (t + 1) / (double)_f.LogScalingFloor))
            : 1.0;

        // Queries: a learned scale per channel, clipped per head, then log-scaled.
        var q = layer.Wq.MatVec(qIn);
        for (int h = 0; h < heads; h++)
        {
            var head = q.AsSpan(h * hd, hd);
            for (int i = 0; i < hd; i++)
            {
                head[i] *= layer.QScale[h * hd + i];
            }
            Clip(head, _f.QkClip);
            Scale(head, tau);
        }

        // Keys and values: projected, through their short convolutions, then
        // the keys clipped per head and the values RMS-normed without a gain.
        var k = layer.Wk.MatVec(kIn);
        var v = layer.Wv.MatVec(vIn);
        if (layer.Full)
        {
            k = DynamicConv(layer.KStatic!, MatVec(layer.KProj!, kvHeads * _f.ConvKernel, d, kIn), kvHeads, slot.KHistory, k);
            v = DynamicConv(layer.VStatic!, MatVec(layer.VProj!, kvHeads * _f.ConvKernel, d, vIn), kvHeads, slot.VHistory, v);
        }
        else
        {
            k = ShortConv(layer.KConv!, slot.KHistory, k);
            v = ShortConv(layer.VConv!, slot.VHistory, v);
        }
        for (int h = 0; h < kvHeads; h++)
        {
            Clip(k.AsSpan(h * hd, hd), _f.QkClip);
            RmsInPlace(v.AsSpan(h * hd, hd), _f.Eps);
        }
        slot.Keys.Add(k);
        slot.Values.Add(v);

        // The relative bias (Inkling2LoopRelativePositionBias): per head, a
        // clipped mix of the trained profiles over backward distance.
        int dRel = _f.RelativeDim, extent = layer.Extent;
        var r = layer.Wr.MatVec(qIn);
        for (int h = 0; h < heads; h++)
        {
            var head = r.AsSpan(h * dRel, dRel);
            for (int i = 0; i < dRel; i++)
            {
                head[i] *= layer.RelScale[h * dRel + i];
            }
            Clip(head, _f.RelativeClip);
            Scale(head, tau);
        }
        var sinks = MatVec(layer.Sink, heads, d, qIn);

        int first = layer.Full ? 0 : Math.Max(0, t - _f.SlidingWindow + 1);
        var output = new float[heads * hd];
        double scale = 1.0 / Math.Sqrt(hd);
        int group = heads / kvHeads;
        var scores = new double[t - first + 1];
        for (int h = 0; h < heads; h++)
        {
            int kvh = h / group;
            var qh = q.AsSpan(h * hd, hd);
            double max = sinks[h];
            for (int j = first; j <= t; j++)
            {
                int dist = t - j;
                double s = Dot(qh, slot.Keys[j].AsSpan(kvh * hd, hd)) * scale;
                if (dist < extent)
                {
                    double bias = 0;
                    for (int i = 0; i < dRel; i++)
                    {
                        bias += r[h * dRel + i] * (double)layer.RelProj[i * extent + dist];
                    }
                    s += bias;
                }
                scores[j - first] = s;
                max = Math.Max(max, s);
            }
            double sum = Math.Exp(sinks[h] - max);
            for (int j = 0; j < scores.Length; j++)
            {
                scores[j] = Math.Exp(scores[j] - max);
                sum += scores[j];
            }
            var outHead = output.AsSpan(h * hd, hd);
            var acc = new double[hd];
            for (int j = first; j <= t; j++)
            {
                double p = scores[j - first] / sum;
                var vj = slot.Values[j].AsSpan(kvh * hd, hd);
                for (int i = 0; i < hd; i++)
                {
                    acc[i] += p * vj[i];
                }
            }
            for (int i = 0; i < hd; i++)
            {
                outHead[i] = (float)acc[i];
            }
        }

        // The output gate (Inkling2LoopGatedOutputProjection) reads the query input.
        var gate = layer.OutGate.MatVec(qIn);
        for (int i = 0; i < output.Length; i++)
        {
            output[i] *= (float)(1.0 / (1.0 + Math.Exp(-gate[i])));
        }
        return layer.Wo.MatVec(output);
    }

    // ── the short convolutions ──────────────────────────────────────────────

    /// <summary>Inkling2LoopShortConvolution: a depthwise causal convolution of
    /// width K over this slot's history, added to its input and scaled by
    /// 1/sqrt(K+1). Weights are [C, 1, K], the last tap on the current token.</summary>
    private float[] ShortConv(float[] weights, List<float[]> history, float[] x)
    {
        int kernel = _f.ConvKernel;
        int c = x.Length;
        var output = new float[c];
        double norm = 1.0 / Math.Sqrt(kernel + 1);
        for (int ch = 0; ch < c; ch++)
        {
            double sum = weights[ch * kernel + kernel - 1] * (double)x[ch];
            for (int lag = 1; lag < kernel; lag++)
            {
                int index = history.Count - lag;
                if (index >= 0)
                {
                    sum += weights[ch * kernel + kernel - 1 - lag] * (double)history[index][ch];
                }
            }
            output[ch] = (float)((sum + x[ch]) * norm);
        }
        Remember(history, x, kernel - 1);
        return output;
    }

    /// <summary>Inkling2LoopDynamicShortConvolution: taps that are a static
    /// weight per channel plus a weight per head projected from the layer's
    /// input, <c>static[lag, c] + dyn[head, lag]</c>, lag 0 on the current
    /// token; scaled by 1/sqrt(2K+1).</summary>
    private float[] DynamicConv(float[] statics, float[] dynamics, int heads, List<float[]> history, float[] x)
    {
        int kernel = _f.ConvKernel;
        int c = x.Length;
        int perHead = c / heads;
        var output = new float[c];
        double norm = 1.0 / Math.Sqrt(2 * kernel + 1);
        for (int ch = 0; ch < c; ch++)
        {
            int head = ch / perHead;
            double sum = 0;
            for (int lag = 0; lag < kernel; lag++)
            {
                float value;
                if (lag == 0)
                {
                    value = x[ch];
                }
                else
                {
                    int index = history.Count - lag;
                    if (index < 0)
                    {
                        continue;
                    }
                    value = history[index][ch];
                }
                sum += (statics[lag * c + ch] + (double)dynamics[head * kernel + lag]) * value;
            }
            output[ch] = (float)((sum + x[ch]) * norm);
        }
        Remember(history, x, kernel - 1);
        return output;
    }

    private static void Remember(List<float[]> history, float[] x, int keep)
    {
        history.Add(x);
        if (history.Count > keep)
        {
            history.RemoveAt(0);
        }
    }

    // ── the residual's mixes (Inkling2Loop{Block,Message}AttentionResidual) ──

    /// <summary>Attention over depth: the blocks and the current state, each
    /// RMS-normed without a gain, weighted by softmax(query·state/sqrt(d)).
    /// One output per query row.</summary>
    private float[][] Mix(float[] queries, int count, List<float[]> blocks, float[] partial)
    {
        int d = _d;
        var states = new float[blocks.Count + 1][];
        for (int i = 0; i < blocks.Count; i++)
        {
            states[i] = Unweighted(blocks[i], 1e-6);
        }
        states[^1] = Unweighted(partial, 1e-6);
        var result = new float[count][];
        double scale = 1.0 / Math.Sqrt(d);
        var weights = new double[states.Length];
        for (int q = 0; q < count; q++)
        {
            var query = queries.AsSpan(q * d, d);
            double max = double.NegativeInfinity;
            for (int i = 0; i < states.Length; i++)
            {
                weights[i] = Dot(query, states[i]) * scale;
                max = Math.Max(max, weights[i]);
            }
            double sum = 0;
            for (int i = 0; i < states.Length; i++)
            {
                weights[i] = Math.Exp(weights[i] - max);
                sum += weights[i];
            }
            var acc = new double[d];
            for (int i = 0; i < states.Length; i++)
            {
                double w = weights[i] / sum;
                for (int j = 0; j < d; j++)
                {
                    acc[j] += w * states[i][j];
                }
            }
            result[q] = acc.Select(x => (float)x).ToArray();
        }
        return result;
    }

    /// <summary>The messages a core layer reads from the previous pass: its
    /// q, k, v and mlp queries over that pass's blocks and output.</summary>
    private float[][] Messages(List<float[]> bank, float[] bankOut, int coreLayer) =>
        Mix(_messageQueries.AsSpan(coreLayer * 4 * _d, 4 * _d).ToArray(), 4, bank, bankOut);

    /// <summary>Inkling2LoopMessageGate: <c>(1 - tanh(F·g))·h + tanh(G·g)·m</c>,
    /// <c>g = down([norm(h); norm(m)])</c>.</summary>
    private float[] MessageGate(Gate gate, float[] h, float[] m)
    {
        int d = _d, rank = _f.GateRank;
        var cat = new float[2 * d];
        Unweighted(h, _f.Eps).CopyTo(cat, 0);
        Unweighted(m, _f.Eps).CopyTo(cat, d);
        var g = MatVec(gate.Down, rank, 2 * d, cat);
        var output = new float[d];
        for (int i = 0; i < d; i++)
        {
            double forget = Math.Tanh(Dot(gate.Forget.AsSpan(i * rank, rank), g));
            double gain = Math.Tanh(Dot(gate.Gain.AsSpan(i * rank, rank), g));
            output[i] = (float)((1 - forget) * h[i] + gain * m[i]);
        }
        return output;
    }

    // ── the experts (Inkling2LoopTopKRouter, Inkling2LoopExperts) ────────────

    private float[] Experts(Layer layer, float[] x)
    {
        int d = _d, experts = _f.Experts, width = _f.ExpertWidth;
        var scores = new double[experts];
        for (int e = 0; e < experts; e++)
        {
            scores[e] = 1.0 / (1.0 + Math.Exp(-Dot(layer.Router.AsSpan(e * d, d), x)));
        }
        var chosen = Enumerable.Range(0, experts)
            .OrderByDescending(e => scores[e] + layer.RouterBias[e])
            .Take(_f.TopK)
            .ToArray();
        double total = chosen.Sum(e => scores[e]) + 1e-20;

        var acc = new double[d];
        var act = new float[width];
        double gateCap = _f.GateSoftcap, upCap = _f.UpSoftcap;
        // Only the chosen experts' rows are decoded, a row at a time, into a
        // buffer per worker; each row's dot product is the same whichever
        // worker takes it.
        bool spread = (long)width * d >= ParallelElements;
        foreach (int e in chosen)
        {
            double weight = scores[e] / total * _f.RouteScale;
            var w13 = layer.W13;
            void Up(int j, float[] buffer)
            {
                // Gate and up are interleaved row by row in the checkpoint.
                w13.Row(e, 2 * j, buffer);
                double g = Dot(buffer, x);
                w13.Row(e, 2 * j + 1, buffer);
                double u = Dot(buffer, x);
                act[j] = (float)(Math.Tanh(g / gateCap) * gateCap / (1.0 + Math.Exp(-g)) * (Math.Tanh(u / upCap) * upCap));
            }
            var w2 = layer.W2;
            void Down(int i, float[] buffer)
            {
                w2.Row(e, i, buffer);
                acc[i] += weight * Dot(buffer, act);
            }
            if (spread)
            {
                Parallel.For(0, width, Budget, () => new float[d], (j, _, buffer) => { Up(j, buffer); return buffer; }, _ => { });
                Parallel.For(0, d, Budget, () => new float[width], (i, _, buffer) => { Down(i, buffer); return buffer; }, _ => { });
            }
            else
            {
                var upRow = new float[d];
                for (int j = 0; j < width; j++)
                {
                    Up(j, upRow);
                }
                var downRow = new float[width];
                for (int i = 0; i < d; i++)
                {
                    Down(i, downRow);
                }
            }
        }
        return acc.Select(v => (float)v).ToArray();
    }

    // ── arithmetic ──────────────────────────────────────────────────────────

    private float[] RmsNorm(ReadOnlySpan<float> x, float[] weight)
    {
        var y = Unweighted(x, _f.Eps);
        for (int i = 0; i < y.Length; i++)
        {
            y[i] *= weight[i];
        }
        return y;
    }

    private static float[] Unweighted(ReadOnlySpan<float> x, double eps)
    {
        double sum = 0;
        foreach (float v in x)
        {
            sum += (double)v * v;
        }
        double inv = 1.0 / Math.Sqrt(sum / x.Length + eps);
        var y = new float[x.Length];
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = (float)(x[i] * inv);
        }
        return y;
    }

    private static void RmsInPlace(Span<float> x, double eps)
    {
        var y = Unweighted(x, eps);
        y.CopyTo(x);
    }

    /// <summary>Inkling2LoopL2Clip: a vector longer than the threshold is scaled down to it.</summary>
    private static void Clip(Span<float> x, double threshold)
    {
        double sum = 0;
        foreach (float v in x)
        {
            sum += (double)v * v;
        }
        double factor = Math.Min(1.0, threshold / (Math.Sqrt(sum) + 1e-6));
        Scale(x, factor);
    }

    private static void Scale(Span<float> x, double factor)
    {
        if (factor == 1.0)
        {
            return;
        }
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)(x[i] * factor);
        }
    }

    private static float[] Add(float[] a, float[] b)
    {
        var y = new float[a.Length];
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = a[i] + b[i];
        }
        return y;
    }

    private static float[] MatVec(float[] w, int rows, int cols, ReadOnlySpan<float> x)
    {
        var y = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            y[r] = (float)Dot(w.AsSpan(r * cols, cols), x);
        }
        return y;
    }

    private static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }
        return sum;
    }
}
