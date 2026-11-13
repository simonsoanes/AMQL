using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Onnx;

public sealed record OnnxExportResult(string Path, string Model, int NodeCount, int InitializerCount);

/// <summary>
/// Builds an ONNX model from an AMQL VINDEX3 container by translating the
/// planner's op program, op for op, into standard ONNX operators — so the
/// graph computes exactly what <see cref="GenericRuntime"/> computes, for
/// any family the planner can run: every norm site (RMS or LayerNorm, with
/// bias and the 1+w offset), projection biases, weighted QK norms (per head
/// or whole projection), half-split / interleaved / partial / zero-tail
/// rotary, NoPE layers, sliding windows, logit softcaps, the scale knobs
/// (embedding, score, residual, output) and parallel blocks.
/// <para>Inputs <c>input_ids</c> and <c>attention_mask</c> are
/// [batch, seq]; positions are 0..seq-1 (no KV cache — one full pass).
/// Outputs <c>logits</c> [batch, seq, vocab], or a classifier's pooled
/// <c>logits</c> / an embedding model's <c>embeddings</c>.</para>
/// <para>Refused by name: stateful layers (linear attention, conv), routed
/// FFNs, attention output gates, learned sinks.</para>
/// </summary>
public static class OnnxExporter
{
    public static OnnxExportResult Export(
        Vindex3Container container,
        string componentId,
        string outPath,
        WeightPatch? patch = null,
        bool int4Weights = false)
    {
        if (patch is not null)
        {
            throw new NotSupportedException(
                "ONNX export reads the container's own weights — bake the patch first (amql-cli export --patch) and re-encode");
        }
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);
        var graph = container.Graph
            ?? throw new InvalidOperationException("container records no system graph");
        var component = graph.Components.First(c => c.Id == componentId);
        var surface = component.Execution
            ?? throw new InvalidOperationException($"component '{componentId}' has no execution surface");

        if (plan.Layers.Any(l => l.IsStateful))
        {
            throw new NotSupportedException(
                "ONNX export does not yet support stateful (linear-attention/GDN or conv) layers — the graph is a single stateless pass");
        }
        if (plan.Layers.Any(l => l.Ffn?.Routed is not null))
        {
            throw new NotSupportedException("ONNX export does not yet support routed (MoE) FFNs");
        }
        if (plan.Layers.Any(l => l.Attention is { } a && (a.OutputGate || a.HeadGate is not null)))
        {
            throw new NotSupportedException("ONNX export does not yet support attention output gates");
        }
        if (plan.Diffusion is not null)
        {
            throw new NotSupportedException("ONNX export of a masked diffusion (bidirectional) model is not supported");
        }

        var g = new GraphBuilder(store, int4Weights);
        int hidden = plan.HiddenSize;
        bool isClassifier = surface.Classifier is not null;
        bool isEmbedding = surface.Embedding is not null;

        var inputs = new List<OnnxVar>
        {
            new() { Name = "input_ids", ElementType = OnnxTypes.Int64,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Parametric("seq") } },
            new() { Name = "attention_mask", ElementType = OnnxTypes.Int64,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Parametric("seq") } },
        };
        var outputs = new List<OnnxVar>();

        // ── positions and masks, shared by every layer ──────────────────
        // seq = Shape(input_ids)[1]; positions 0..seq-1 as float and int64.
        string seq = g.Node("Gather", new[] { g.Node("Shape", "input_ids"), g.Int64s(1) }, ("axis", 0L));
        string pos = g.Node("Range", g.Int64s(0), seq, g.Int64s(1));
        string posF = g.Node("Cast", new[] { pos }, ("to", (long)OnnxTypes.Float));
        // Key padding: [batch, 1, 1, seq] additive (0 or -1e9).
        string keyMask = g.Node("Unsqueeze", new[] { "attention_mask", g.Int64s(1, 2) });
        string padAdd = g.Node("Mul",
            g.Node("Sub", g.Float(1f), g.Node("Cast", new[] { keyMask }, ("to", (long)OnnxTypes.Float))), g.Float(-1e9f));
        var masks = new Dictionary<long, string>(); // window (0 = full) → additive [batch,1,seq,seq]
        string MaskFor(long? window)
        {
            long key = window ?? 0;
            if (masks.TryGetValue(key, out var m))
            {
                return m;
            }
            // distance[i, j] = i - j; invalid where j > i, or i - j >= window.
            string dist = g.Node("Sub", g.Node("Unsqueeze", new[] { pos, g.Int64s(1) }), g.Node("Unsqueeze", new[] { pos, g.Int64s(0) }));
            string invalid = g.Node("Less", dist, g.Int64s(0));
            if (window is { } w)
            {
                invalid = g.Node("Or", invalid, g.Node("GreaterOrEqual", dist, g.Int64s(w)));
            }
            string causal = g.Node("Where", invalid, g.Float(-1e9f), g.Float(0f));
            return masks[key] = g.Node("Add", causal, padAdd);
        }

        // ── embedding ───────────────────────────────────────────────────
        var embed = plan.Embedding ?? throw new NotSupportedException("the plan carries no embedding op");
        string x = g.Node("Gather", new[] { g.Weight(embed.Table), "input_ids" }, ("axis", 0L));
        if (embed.Norm is not null)
        {
            throw new NotSupportedException("ONNX export of a weightless embedding norm is not supported");
        }
        if (embed.Scale is { } embScale)
        {
            x = g.Node("Mul", x, g.Float((float)embScale));
        }

        // ── layers ──────────────────────────────────────────────────────
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            var layer = plan.Layers[l];
            string h = layer.PreAttentionNorm is { } preAttn ? g.Norm(x, preAttn) : x;
            string mixer = Attention(g, h, layer.Attention!, MaskFor(layer.Attention!.Window), posF);
            if (layer.PostAttentionNorm is { } postAttn)
            {
                mixer = g.Norm(mixer, postAttn);
            }
            string Residual(string stream, string branch) =>
                g.Node("Add", stream, plan.ResidualScale != 1.0 ? g.Node("Mul", branch, g.Float((float)plan.ResidualScale)) : branch);

            if (layer.ParallelResidual)
            {
                string ffnOut = layer.Ffn?.Dense is { } pd ? Ffn(g, h, pd) : throw new NotSupportedException("a parallel block without an FFN");
                x = Residual(x, g.Node("Add", mixer, ffnOut));
                continue;
            }
            x = Residual(x, mixer);
            if (layer.Ffn?.Dense is { } dense)
            {
                string hf = layer.PreFfnNorm is { } preFfn ? g.Norm(x, preFfn) : x;
                string f = Ffn(g, hf, dense);
                if (layer.PostFfnNorm is { } postFfn)
                {
                    f = g.Norm(f, postFfn);
                }
                x = Residual(x, f);
            }
        }
        x = g.Norm(x, plan.FinalNorm);

        // ── output ──────────────────────────────────────────────────────
        if (isClassifier)
        {
            string pooled = g.Node("Squeeze", new[] { g.Node("Gather", new[] { x, g.Int64s(-1) }, ("axis", 1L)), g.Int64s(1) });
            g.Linear(pooled, new OperandRef("target.classifier_head", "weight"),
                surface.Classifier!.ScoreBias ? new OperandRef("target.classifier_head", "bias") : null, "logits");
            outputs.Add(new OnnxVar { Name = "logits", ElementType = OnnxTypes.Float,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Fixed(surface.Classifier.NumLabels) } });
        }
        else if (isEmbedding)
        {
            string pooled = g.Node("ReduceMean", new[] { x, g.Int64s(1) }, ("keepdims", 0L));
            string norm = g.Node("Sqrt", g.Node("ReduceSum", new[] { g.Node("Mul", pooled, pooled), g.Int64s(-1) }, ("keepdims", 1L)));
            g.NodeNamed("embeddings", "Div", pooled, norm);
            outputs.Add(new OnnxVar { Name = "embeddings", ElementType = OnnxTypes.Float,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Fixed(hidden) } });
        }
        else
        {
            var output = plan.Output ?? throw new NotSupportedException("the plan carries no output head");
            string logits = g.Linear(x, output.Projection, null);
            if (output.Multiplier is { } multiplier)
            {
                logits = g.Node("Mul", logits, g.Float((float)multiplier));
            }
            if (output.LogitSoftcapping is { } cap)
            {
                logits = g.Node("Mul", g.Node("Tanh", g.Node("Div", logits, g.Float(cap))), g.Float(cap));
            }
            g.NodeNamed("logits", "Identity", logits);
            outputs.Add(new OnnxVar { Name = "logits", ElementType = OnnxTypes.Float,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Parametric("seq"), OnnxDim.Fixed(plan.Output.VocabSize) } });
        }

        OnnxWriter.Write(outPath, new OnnxGraph
        {
            Name = container.Index.Model,
            Inputs = inputs,
            Outputs = outputs,
            Nodes = g.Nodes,
            Initializers = g.Initializers,
        }, outPath + ".data");
        return new OnnxExportResult(outPath, container.Index.Model, g.Nodes.Count, g.Initializers.Count);
    }

    /// <summary>Softmax attention, as <see cref="GenericRuntime"/> runs it.</summary>
    private static string Attention(GraphBuilder g, string h, AttentionOp attn, string mask, string posF)
    {
        int heads = attn.NumQHeads, kvHeads = attn.NumKvHeads, d = attn.HeadDim;
        string q = g.Linear(h, attn.QProj, attn.QBias);
        string k = g.Linear(h, attn.KProj, attn.KBias);
        string v = g.Linear(h, attn.VProj, attn.VBias);

        // QK norm: per head ([.., heads, d]) or over the whole projection.
        string Heads(string t, int n) => g.Node("Reshape", t, g.Int64s(0, 0, n, d));
        q = Heads(attn.QNorm is { } qn && qn.Width != d ? g.Norm(q, qn) : q, heads);
        k = Heads(attn.KNorm is { } kn && kn.Width != d ? g.Norm(k, kn) : k, kvHeads);
        if (attn.QNorm is { } qh && qh.Width == d)
        {
            q = g.Norm(q, qh);
        }
        if (attn.KNorm is { } kh && kh.Width == d)
        {
            k = g.Norm(k, kh);
        }
        if (attn.ParameterFreeQkNorm.Any)
        {
            throw new NotSupportedException("ONNX export of a parameter-free QK norm is not supported");
        }

        // [batch, seq, n, d] → [batch, n, seq, d]
        q = g.Node("Transpose", new[] { q }, ("perm", new long[] { 0, 2, 1, 3 }));
        k = g.Node("Transpose", new[] { k }, ("perm", new long[] { 0, 2, 1, 3 }));
        v = g.Node("Transpose", new[] { Heads(v, kvHeads) }, ("perm", new long[] { 0, 2, 1, 3 }));
        q = Rope(g, q, attn, posF);
        k = Rope(g, k, attn, posF);

        if (heads != kvHeads)
        {
            // GQA: repeat each kv head heads/kvHeads times, adjacent.
            string Repeat(string t) => g.Node("Reshape",
                g.Node("Tile", g.Node("Unsqueeze", new[] { t, g.Int64s(2) }), g.Int64s(1, 1, heads / kvHeads, 1, 1)),
                g.Int64s(0, heads, -1, d));
            k = Repeat(k);
            v = Repeat(v);
        }
        string scores = g.Node("Mul",
            g.Node("MatMul", q, g.Node("Transpose", new[] { k }, ("perm", new long[] { 0, 1, 3, 2 }))),
            g.Float((float)attn.ScoreScale));
        if (attn.LogitSoftcapping is { } cap)
        {
            scores = g.Node("Mul", g.Node("Tanh", g.Node("Div", scores, g.Float(cap))), g.Float(cap));
        }
        if (!attn.Causal)
        {
            throw new NotSupportedException("ONNX export of bidirectional attention is not supported");
        }
        string probs = g.Node("Softmax", new[] { g.Node("Add", scores, mask) }, ("axis", -1L));
        string context = g.Node("Transpose", new[] { g.Node("MatMul", probs, v) }, ("perm", new long[] { 0, 2, 1, 3 }));
        context = g.Node("Reshape", context, g.Int64s(0, 0, heads * d));
        return g.Linear(context, attn.OProj, attn.OBias);
    }

    /// <summary>Rotary embedding on [batch, n, seq, d], mirroring <c>Rope.Apply</c>.</summary>
    private static string Rope(GraphBuilder g, string x, AttentionOp attn, string posF)
    {
        int d = attn.HeadDim;
        int width;
        double[] invFreq;
        switch (attn.Position)
        {
            case PositionNone:
                return x;
            case PositionRope rope:
                width = d;
                invFreq = Enumerable.Range(0, width / 2).Select(i => 1.0 / Math.Pow(rope.Theta, 2.0 * i / width)).ToArray();
                break;
            case PositionPartialRope partial:
                width = partial.RotaryWidth(d);
                invFreq = Enumerable.Range(0, width / 2).Select(i => 1.0 / Math.Pow(partial.Theta, 2.0 * i / width)).ToArray();
                break;
            case PositionZeroTailRope zeroTail when !attn.RopeInterleaved:
                // Full-width half-split pairs; the tail frequencies are zero (identity).
                width = d;
                int rotatedPairs = zeroTail.RotatedPairs(d);
                invFreq = Enumerable.Range(0, d / 2).Select(i => i < rotatedPairs ? 1.0 / Math.Pow(zeroTail.Theta, 2.0 * i / d) : 0.0).ToArray();
                break;
            default:
                throw new NotSupportedException($"position policy '{attn.Position.GetType().Name}' has no ONNX rotary implementation");
        }
        int half = width / 2;
        // angles [seq, half]
        string angles = g.Node("Mul", g.Node("Unsqueeze", new[] { posF, g.Int64s(1) }),
            g.Constant(invFreq.Select(f => (float)f).ToArray(), 1, half));
        string cos = g.Node("Cos", angles), sin = g.Node("Sin", angles);
        string rot = width == d ? x : g.Slice(x, 0, width, 3);
        string rest = width == d ? null! : g.Slice(x, width, d, 3);
        string rotated;
        if (attn.RopeInterleaved)
        {
            // pairs (2i, 2i+1): [.., seq, half, 2]
            string pairs = g.Node("Reshape", rot, g.Int64s(0, 0, 0, half, 2));
            string a = g.Node("Squeeze", new[] { g.Slice(pairs, 0, 1, 4), g.Int64s(4) });
            string b = g.Node("Squeeze", new[] { g.Slice(pairs, 1, 2, 4), g.Int64s(4) });
            string ra = g.Node("Sub", g.Node("Mul", a, cos), g.Node("Mul", b, sin));
            string rb = g.Node("Add", g.Node("Mul", a, sin), g.Node("Mul", b, cos));
            string stacked = g.Node("Concat", new[] { g.Node("Unsqueeze", new[] { ra, g.Int64s(4) }), g.Node("Unsqueeze", new[] { rb, g.Int64s(4) }) }, ("axis", 4L));
            rotated = g.Node("Reshape", stacked, g.Int64s(0, 0, 0, width));
        }
        else
        {
            string a = g.Slice(rot, 0, half, 3), b = g.Slice(rot, half, width, 3);
            string ra = g.Node("Sub", g.Node("Mul", a, cos), g.Node("Mul", b, sin));
            string rb = g.Node("Add", g.Node("Mul", a, sin), g.Node("Mul", b, cos));
            rotated = g.Node("Concat", new[] { ra, rb }, ("axis", 3L));
        }
        return width == d ? rotated : g.Node("Concat", new[] { rotated, rest }, ("axis", 3L));
    }

    private static string Ffn(GraphBuilder g, string x, DenseFfnOp ffn)
    {
        string up = g.Linear(x, ffn.Up, ffn.UpBias);
        string Act(string t) => ffn.Activation switch
        {
            Activation.Silu => g.Node("Mul", t, g.Node("Sigmoid", t)),
            // GenericRuntime computes both GELUs with the tanh approximation.
            Activation.Gelu or Activation.GeluTanh => g.Node("Gelu", new[] { t }, ("approximate", "tanh")),
            Activation.Relu => g.Node("Relu", t),
            var other => throw new NotSupportedException($"activation {other} has no ONNX spelling"),
        };
        string inner = ffn.Gate is { } gate ? g.Node("Mul", Act(g.Linear(x, gate, ffn.GateBias)), up) : Act(up);
        return g.Linear(inner, ffn.Down, ffn.DownBias);
    }

    /// <summary>Node and initializer bookkeeping.</summary>
    private sealed class GraphBuilder(OperandStore store, bool int4)
    {
        public List<OnnxNode> Nodes { get; } = new();
        public List<OnnxInitializer> Initializers { get; } = new();
        private readonly Dictionary<string, string> _weights = new(StringComparer.Ordinal);
        private int _next;

        public string Node(string op, params string[] inputs) => Node(op, inputs, Array.Empty<(string, object)>());

        public string Node(string op, string[] inputs, params (string Name, object Value)[] attributes) =>
            NodeNamed($"n{_next++}", op, inputs, attributes);

        public string NodeNamed(string name, string op, params string[] inputs) => NodeNamed(name, op, inputs, Array.Empty<(string, object)>());

        public string NodeNamed(string name, string op, string[] inputs, params (string Name, object Value)[] attributes)
        {
            var node = new OnnxNode { OpType = op, Inputs = inputs, Outputs = new[] { name } };
            foreach (var (key, value) in attributes)
            {
                node.Attributes[key] = value;
            }
            Nodes.Add(node);
            return name;
        }

        public string Float(float value) => Constant(new[] { value });

        public string Constant(float[] values, params long[] dims)
        {
            string name = $"c{_next++}";
            Initializers.Add(new OnnxInitializer
            {
                Name = name, DataType = OnnxTypes.Float, Dims = dims.Length == 0 ? Array.Empty<long>() : dims,
                RawData = F32ToRaw(values),
            });
            return name;
        }

        public string Int64s(params long[] values)
        {
            string name = $"i{_next++}";
            var raw = new byte[values.Length * 8];
            Buffer.BlockCopy(values, 0, raw, 0, raw.Length);
            Initializers.Add(new OnnxInitializer { Name = name, DataType = OnnxTypes.Int64, Dims = new long[] { values.Length }, RawData = raw });
            return name;
        }

        public string Slice(string x, long start, long end, long axis) => Node("Slice", x, Int64s(start), Int64s(end), Int64s(axis));

        /// <summary>A weight as an F32 graph value (stored F32, or FP16/INT4 decoded in-graph).</summary>
        public string Weight(OperandRef operand)
        {
            string key = operand.ObjectId + "/" + operand.TensorName;
            if (_weights.TryGetValue(key, out var existing))
            {
                return existing;
            }
            var res = store.Resolve(operand);
            string name = $"w_{operand.ObjectId.Replace('.', '_')}_{operand.TensorName}";
            bool bf16 = res.Dtype == Dtype.BF16;
            string value;
            if (int4 && bf16 && res.Shape.Length == 2)
            {
                // Per-row symmetric INT4 from the BF16 payload, dequantised in-graph.
                int rows = checked((int)res.Shape[0]), cols = checked((int)res.Shape[1]);
                var scales = new float[rows];
                var packed = new byte[((long)rows * cols + 1) / 2];
                for (int r = 0; r < rows; r++)
                {
                    long rowOff = (long)r * cols * 2;
                    float maxAbs = 0f;
                    for (int c = 0; c < cols; c++)
                    {
                        maxAbs = MathF.Max(maxAbs, MathF.Abs(Bf16At(res.Payload, rowOff + c * 2L)));
                    }
                    float scale = maxAbs > 0f ? maxAbs / 7f : 1f;
                    scales[r] = scale;
                    long ti = (long)r * cols;
                    for (int c = 0; c < cols; c++, ti++)
                    {
                        byte nibble = (byte)((int)MathF.Round(Math.Clamp(Bf16At(res.Payload, rowOff + c * 2L) / scale, -8f, 7f)) & 0xF);
                        packed[ti >> 1] = (ti & 1) == 0 ? nibble : (byte)(packed[ti >> 1] | (nibble << 4));
                    }
                }
                Initializers.Add(new OnnxInitializer { Name = name, DataType = OnnxTypes.Int4, Dims = res.Shape, RawData = packed });
                Initializers.Add(new OnnxInitializer { Name = name + "_s", DataType = OnnxTypes.Float, Dims = new long[] { rows }, RawData = F32ToRaw(scales) });
                value = Node("DequantizeLinear", new[] { name, name + "_s" }, ("axis", 0L));
            }
            else if (bf16)
            {
                Initializers.Add(new OnnxInitializer { Name = name, DataType = OnnxTypes.Float16, Dims = res.Shape, RawData = Bf16ToFp16(res.Payload) });
                value = Node("Cast", new[] { name }, ("to", (long)OnnxTypes.Float));
            }
            else
            {
                Initializers.Add(new OnnxInitializer
                {
                    Name = name, DataType = OnnxTypes.Float, Dims = res.Shape,
                    RawData = F32ToRaw(BitPattern.WidenToF32(res.Dtype, res.Payload)),
                });
                value = name;
            }
            return _weights[key] = value;
        }

        /// <summary>x·Wᵀ (+ b) for a stored [out, in] weight.</summary>
        public string Linear(string x, OperandRef weight, OperandRef? bias, string? outName = null)
        {
            string wt = Node("Transpose", new[] { Weight(weight) }, ("perm", new long[] { 1, 0 }));
            string y = bias is null && outName is not null ? NodeNamed(outName, "MatMul", x, wt) : Node("MatMul", x, wt);
            if (bias is not null)
            {
                y = outName is null ? Node("Add", y, Weight(bias)) : NodeNamed(outName, "Add", y, Weight(bias));
            }
            return y;
        }

        /// <summary>A norm op over the last axis: RMS or LayerNorm, (w + offset), optional bias.</summary>
        public string Norm(string x, NormOp norm)
        {
            string w = Weight(norm.Weight);
            if (norm.WeightOffset != 0f)
            {
                w = Node("Add", w, Float(norm.WeightOffset));
            }
            string normalized;
            if (norm.Kind == NormType.LayerNorm)
            {
                string centred = Node("Sub", x, Node("ReduceMean", new[] { x, Int64s(-1) }, ("keepdims", 1L)));
                string variance = Node("ReduceMean", new[] { Node("Mul", centred, centred), Int64s(-1) }, ("keepdims", 1L));
                normalized = Node("Div", centred, Node("Sqrt", Node("Add", variance, Float((float)norm.Eps))));
            }
            else
            {
                string meanSquare = Node("ReduceMean", new[] { Node("Mul", x, x), Int64s(-1) }, ("keepdims", 1L));
                normalized = Node("Div", x, Node("Sqrt", Node("Add", meanSquare, Float((float)norm.Eps))));
            }
            string y = Node("Mul", normalized, w);
            return norm.Bias is { } bias ? Node("Add", y, Weight(bias)) : y;
        }
    }

    private static byte[] F32ToRaw(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float Bf16At(byte[] bf16, long offset)
        => BitConverter.Int32BitsToSingle((bf16[offset] | (bf16[offset + 1] << 8)) << 16);

    private static byte[] Bf16ToFp16(byte[] bf16)
    {
        int n = bf16.Length / 2;
        var fp16 = new byte[n * 2];
        for (int i = 0; i < n; i++)
        {
            ushort bits = (ushort)(bf16[i * 2] | (bf16[i * 2 + 1] << 8));
            ushort half = BitConverter.HalfToUInt16Bits((Half)BitConverter.Int32BitsToSingle(bits << 16));
            fp16[i * 2] = (byte)half;
            fp16[i * 2 + 1] = (byte)(half >> 8);
        }
        return fp16;
    }
}
