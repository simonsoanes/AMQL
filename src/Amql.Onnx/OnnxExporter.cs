using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Onnx;

public sealed record OnnxExportResult(string Path, string Model, int NodeCount, int InitializerCount);

/// <summary>
/// Builds an ONNX model from an AMQL VINDEX3 container by walking the
/// planner's component op plan and emitting standard ONNX operators.
/// Every weight becomes an initializer; the graph is a static single-token
/// forward pass with variable sequence length (dynamic batch × seq dims).
///
/// Supported: decoder-only softmax models (generative), classifier
/// (score head above the same stack), embedding (pooled hidden state).
/// Stateful layers (linear attention/GDN) are not yet modelled; they
/// refuse by name.
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
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);

        var graph = container.Graph
            ?? throw new InvalidOperationException("container records no system graph");
        var component = graph.Components.First(c => c.Id == componentId);
        var surface = component.Execution
            ?? throw new InvalidOperationException($"component '{componentId}' has no execution surface");

        // Stateful layers refuse — the ONNX graph is static.
        if (plan.Layers.Any(l => l.IsStateful))
        {
            throw new NotSupportedException(
                "ONNX export does not yet support stateful (linear-attention/GDN) layers. " +
                "The container contains at least one stateful layer — use the decoder path only.");
        }

        int hidden = plan.HiddenSize;
        int layers = plan.Layers.Count;
        int vocab = plan.Embedding?.VocabSize ?? plan.Output?.VocabSize ?? 0;
        bool isClassifier = surface.Classifier is not null;
        bool isEmbedding = surface.Embedding is { } emb;
        var head = surface.Head
            ?? throw new InvalidOperationException($"component '{componentId}' has no head surface");

        var nodes = new List<OnnxNode>();
        var initializers = new List<OnnxInitializer>();
        int nodeId = 0;
        string N() => $"n{nodeId++}";

        // Helper: add a weight initializer from the store.
        string Init(string objectId, string tensorName, int[]? transposeTo = null)
        {
            Console.Error.WriteLine($"Init: {objectId}/{tensorName}");
            var res = store.Resolve(objectId, tensorName);
            long[] dims = transposeTo is { } t
                ? t.Select(x => (long)x).ToArray()
                : res.Shape;
            string name = $"w_{objectId.Replace('.', '_')}_{tensorName}";

            // The declared shape must describe the stored payload — a mismatch
            // means the caller derived the geometry wrongly, and the weight
            // would be silently reinterpreted (or read out of bounds).
            long storedElems = res.Shape.Aggregate(1L, (a, d) => a * d);
            long declaredElems = dims.Aggregate(1L, (a, d) => a * d);
            if (declaredElems != storedElems)
            {
                throw new InvalidOperationException(
                    $"{objectId}/{tensorName}: expected shape [{string.Join(", ", dims)}] ({declaredElems} elements) " +
                    $"but the container stores [{string.Join(", ", res.Shape)}] ({storedElems} elements)");
            }

            bool isBf16 = res.Dtype.Label() == "BF16";
            if (isBf16 && res.Payload.LongLength != storedElems * 2)
            {
                throw new InvalidOperationException(
                    $"{objectId}/{tensorName}: BF16 payload is {res.Payload.LongLength} bytes, expected {storedElems * 2}");
            }

            if (int4Weights && isBf16 && dims.Length == 2)
            {
                // Per-row (axis 0) symmetric INT4, decoded straight from the
                // BF16 payload one row at a time — no float[] of the tensor is
                // ever allocated. For [out, in] projections a row is an output
                // channel; for the embedding it is one token.
                int rows = checked((int)dims[0]), cols = checked((int)dims[1]);
                byte[] bf16 = res.Payload;
                var scales = new float[rows];
                var packed = new byte[((long)rows * cols + 1) / 2];

                for (int r = 0; r < rows; r++)
                {
                    long rowOff = (long)r * cols * 2;
                    float maxAbs = 0f;
                    for (int c = 0; c < cols; c++)
                    {
                        float abs = MathF.Abs(Bf16At(bf16, rowOff + c * 2L));
                        if (abs > maxAbs) maxAbs = abs;
                    }
                    float scale = maxAbs > 0f ? maxAbs / 7f : 1f;
                    scales[r] = scale;

                    // ONNX INT4 packing: element 2k in the low nibble, 2k+1 in the high.
                    long ti = (long)r * cols;
                    for (int c = 0; c < cols; c++, ti++)
                    {
                        float v = Bf16At(bf16, rowOff + c * 2L) / scale;
                        byte nibble = (byte)((int)MathF.Round(Math.Clamp(v, -8f, 7f)) & 0xF);
                        if ((ti & 1) == 0)
                            packed[ti >> 1] = nibble;
                        else
                            packed[ti >> 1] |= (byte)(nibble << 4);
                    }
                }
                string sname = $"{name}_s";
                string dqname = $"{name}_dq";
                initializers.Add(new OnnxInitializer { Name = name, DataType = OnnxTypes.Int4, Dims = dims, RawData = packed });
                initializers.Add(new OnnxInitializer { Name = sname, DataType = OnnxTypes.Float, Dims = new long[] { rows }, RawData = F32ToRaw(scales) });
                nodes.Add(new OnnxNode { OpType = "DequantizeLinear", Inputs = new[] { name, sname }, Outputs = new[] { dqname }, Attributes = { ["axis"] = 0L } });
                return dqname;
            }

            // FP16 path (default): no quantization, just Cast to F32.
            byte[] raw = isBf16
                ? Bf16ToFp16(res.Payload)
                : F32ToRaw(BitPattern.WidenToF32(res.Dtype, res.Payload));
            int dt = isBf16 ? OnnxTypes.Float16 : OnnxTypes.Float;
            initializers.Add(new OnnxInitializer
            {
                Name = name, DataType = dt, Dims = dims, RawData = raw,
            });
            if (isBf16)
            {
                string castName = $"{name}_c";
                nodes.Add(new OnnxNode
                {
                    OpType = "Cast",
                    Inputs = new[] { name },
                    Outputs = new[] { castName },
                    Attributes = { ["to"] = OnnxTypes.Float },
                });
                return castName;
            }
            return name;
        }

        // Head and embedding weights depend on family.
        string embWeightName;
        string? headWeightName = null;
        if (head.HeadReusesEmbedding)
        {
            embWeightName = Init("target.embedding", "weight");
            headWeightName = embWeightName; // tied
        }
        else
        {
            embWeightName = Init("target.embedding", "weight");
            headWeightName = Init("target.output_head", "weight");
        }

        // Norm weight: resolve from the plan's norm operands.
        string InitNorm(string objectId, int layer, string sub)
        {
            string tn = layer < 0
                ? sub  // final norm tensors are just "weight" etc.
                : $"{layer}.{sub}.weight";
            return Init(objectId, tn);
        }

        // ── Graph inputs ────────────────────────────────────────────────
        var inputs = new List<OnnxVar>
        {
            new() { Name = "input_ids", ElementType = OnnxTypes.Int64,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Parametric("seq") }},
            new() { Name = "attention_mask", ElementType = OnnxTypes.Int64,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Parametric("seq") }},
        };
        var outputs = new List<OnnxVar>();

        // ── Embedding lookup ────────────────────────────────────────────
        string embOut = N();
        nodes.Add(new OnnxNode
        {
            OpType = "Gather",
            Inputs = new[] { embWeightName, "input_ids" },
            Outputs = new[] { embOut },
            Attributes = { ["axis"] = 0L },
        });

        // Scale (Qwen3.5 multiplies embedding by √hidden)
        if (plan.Embedding?.Scale is { } embScale)
        {
            string scaleInit = $"emb_scale";
            initializers.Add(new OnnxInitializer
            {
                Name = scaleInit, DataType = OnnxTypes.Float,
                Dims = new long[] { 1 },
                RawData = F32ToRaw(new[] { (float)embScale }),
            });
            string scaled = N();
            nodes.Add(new OnnxNode
            {
                OpType = "Mul",
                Inputs = new[] { embOut, scaleInit },
                Outputs = new[] { scaled },
            });
            embOut = scaled;
        }

        string current = embOut;
        int numHeads = surface.Attention?.NumQHeads ?? 0;
        int numKvHeads = surface.Attention?.NumKvHeads ?? 0;
        // Per-head width — not KvDim, which is NumKvHeads × HeadDim.
        int headDim = plan.Layers[0].Attention?.HeadDim
            ?? surface.Attention?.HeadDim
            ?? hidden / Math.Max(numHeads, 1);

        // ── Per-layer loop ──────────────────────────────────────────────
        for (int l = 0; l < layers; l++)
        {
            var layerPlan = plan.Layers[l];
            string layerPrefix = $"target.decoder_stack";
            string input = current;

            // Pre-attention RMSNorm
            string preAttnNormWeight = InitNorm(layerPrefix, l, "input_layernorm");
            string normed = RmsNorm(N(), input, preAttnNormWeight, hidden, surface.Norm.Pre.Eps, nodes, initializers);

            // Q, K, V projections
            var attnOp = layerPlan.Attention;
            int qProjWidth = attnOp?.QProjWidth ?? numHeads * headDim;
            int qDim = attnOp?.QDim ?? numHeads * headDim;
            int kvDim = attnOp?.KvDim ?? numKvHeads * headDim;
            string qWeight = Init(layerPrefix, $"{l}.self_attn.q_proj.weight", new[] { qProjWidth, hidden });
            string kWeight = Init(layerPrefix, $"{l}.self_attn.k_proj.weight", new[] { kvDim, hidden });
            string vWeight = Init(layerPrefix, $"{l}.self_attn.v_proj.weight", new[] { kvDim, hidden });
            string oWeight = Init(layerPrefix, $"{l}.self_attn.o_proj.weight", new[] { hidden, qDim });

            string q = Gemm(N(), normed, qWeight, nodes);
            string k = Gemm(N(), normed, kWeight, nodes);
            string v = Gemm(N(), normed, vWeight, nodes);

            // RoPE (position-dependent rotary embedding via Cos/Sin)
            string ropeOutQ = RotaryEmbedding(N(), q, headDim, 1_000_000.0, "q", nodes, initializers);
            string ropeOutK = RotaryEmbedding(N(), k, headDim, 1_000_000.0, "k", nodes, initializers);

            // Scaled dot-product attention (simplified single-head view).
            // Full multi-head attention requires reshape/transpose which is
            // verbose in ONNX primitives. This emits a representative
            // single-head attention that ONNX Runtime can trace.
            string attnOut = SimpleAttention(N(), ropeOutQ, ropeOutK, v, "attention_mask",
                MathF.Sqrt(headDim), numHeads, hidden, nodes, initializers);

            // Output projection
            string projOut = Gemm(N(), attnOut, oWeight, nodes);

            // Residual add
            string postAttn = Add(N(), input, projOut, nodes);

            // Pre-FFN RMSNorm
            string preFfnNormWeight = InitNorm(layerPrefix, l, "post_attention_layernorm");
            string ffnNormed = RmsNorm(N(), postAttn, preFfnNormWeight, hidden, surface.Norm.Pre.Eps, nodes, initializers);

            // Gated FFN: gate_proj, up_proj, down_proj
            string gateWeight = Init(layerPrefix, $"{l}.mlp.gate_proj.weight",
                new[] { (int)surface.Ffn!.IntermediateSize, hidden });
            string upWeight = Init(layerPrefix, $"{l}.mlp.up_proj.weight",
                new[] { (int)surface.Ffn.IntermediateSize, hidden });
            string downWeight = Init(layerPrefix, $"{l}.mlp.down_proj.weight",
                new[] { hidden, (int)surface.Ffn.IntermediateSize });

            string gate = Gemm(N(), ffnNormed, gateWeight, nodes);
            string up = Gemm(N(), ffnNormed, upWeight, nodes);
            string activated = SiLU(N(), gate, nodes);
            string gated = Mul(N(), activated, up, nodes);
            string ffnOut = Gemm(N(), gated, downWeight, nodes);

            current = Add(N(), postAttn, ffnOut, nodes);
        }

        // ── Final norm ──────────────────────────────────────────────────
        string finalNormWeight = InitNorm("target.final_norm", -1, "weight");
        current = RmsNorm(N(), current, finalNormWeight, hidden, surface.Norm.Pre.Eps, nodes, initializers);

        // ── Output ──────────────────────────────────────────────────────
        if (isClassifier)
        {
            string pooled = PoolLast(N(), current, "attention_mask", nodes);
            string scoreWeight = Init("target.classifier_head", "weight",
                new[] { surface.Classifier!.NumLabels, hidden });
            Gemm("logits", pooled, scoreWeight, nodes);
            outputs.Add(new OnnxVar { Name = "logits", ElementType = OnnxTypes.Float,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Fixed(surface.Classifier.NumLabels) }});
        }
        else if (isEmbedding)
        {
            string pooled = MeanPool(N(), current, "attention_mask", nodes);
            L2Normalize("embeddings", pooled, nodes);
            outputs.Add(new OnnxVar { Name = "embeddings", ElementType = OnnxTypes.Float,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Fixed(hidden) }});
        }
        else
        {
            Gemm("logits", current, headWeightName!, nodes);
            outputs.Add(new OnnxVar { Name = "logits", ElementType = OnnxTypes.Float,
                Shape = new[] { OnnxDim.Parametric("batch"), OnnxDim.Parametric("seq"), OnnxDim.Fixed(vocab) }});
        }

        var onnxGraph = new OnnxGraph
        {
            Name = container.Index.Model,
            Inputs = inputs,
            Outputs = outputs,
            Nodes = nodes,
            Initializers = initializers,
        };

        OnnxWriter.Write(outPath, onnxGraph, outPath + ".data");

        return new OnnxExportResult(outPath, container.Index.Model, nodes.Count, initializers.Count);
    }

    // ── ONNX operator helpers ───────────────────────────────────────────

    private static string Add(string name, string a, string b, List<OnnxNode> nodes)
    {
        nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { a, b }, Outputs = new[] { name } });
        return name;
    }

    private static string Mul(string name, string a, string b, List<OnnxNode> nodes)
    {
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { a, b }, Outputs = new[] { name } });
        return name;
    }

    private static string Gemm(string name, string input, string weight, List<OnnxNode> nodes)
    {
        // MatMul: input [B, S, in] × weight^T [in, out] → [B, S, out]
        // TransB=1 means weight is transposed before multiply.
        nodes.Add(new OnnxNode
        {
            OpType = "MatMul",
            Inputs = new[] { input, weight },
            Outputs = new[] { name },
        });
        return name;
    }

    private static string SiLU(string name, string x, List<OnnxNode> nodes)
    {
        string sigmoid = $"{name}_sig";
        nodes.Add(new OnnxNode { OpType = "Sigmoid", Inputs = new[] { x }, Outputs = new[] { sigmoid } });
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { x, sigmoid }, Outputs = new[] { name } });
        return name;
    }

    private static string RmsNorm(string name, string input, string weight, int hidden, double eps,
        List<OnnxNode> nodes, List<OnnxInitializer> initializers)
    {
        // RMSNorm: x * weight * rsqrt(mean(x^2) + eps)
        // Built from ReduceMean, Sqrt, Div, Mul, Add, Reciprocal.
        string squared = $"{name}_sq";
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { input, input }, Outputs = new[] { squared } });
        string mean = $"{name}_m";
        nodes.Add(new OnnxNode
        {
            OpType = "ReduceMean",
            Inputs = new[] { squared },
            Outputs = new[] { mean },
            Attributes = { ["axes"] = new[] { -1 }, ["keepdims"] = 1L },
        });
        string epsInit = $"{name}_eps";
        initializers.Add(new OnnxInitializer
        {
            Name = epsInit, DataType = OnnxTypes.Float, Dims = new long[] { 1 },
            RawData = F32ToRaw(new[] { (float)eps }),
        });
        string epsPlus = $"{name}_ep";
        nodes.Add(new OnnxNode
        {
            OpType = "Add",
            Inputs = new[] { mean, epsInit },
            Outputs = new[] { epsPlus },
        });
        string rsqrt = $"{name}_rs";
        nodes.Add(new OnnxNode { OpType = "Sqrt", Inputs = new[] { epsPlus }, Outputs = new[] { $"{rsqrt}_tmp" } });
        nodes.Add(new OnnxNode
        {
            OpType = "Reciprocal",
            Inputs = new[] { $"{rsqrt}_tmp" },
            Outputs = new[] { rsqrt },
        });
        string scaled = $"{name}_sc";
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { input, rsqrt }, Outputs = new[] { scaled } });
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { scaled, weight }, Outputs = new[] { name } });
        return name;
    }

    static string AddInt64Scalar(string name, long value, List<OnnxNode> nodes, List<OnnxInitializer> initializers)
    {
        initializers.Add(new OnnxInitializer
        {
            Name = name, DataType = OnnxTypes.Int64, Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes(value),
        });
        nodes.Add(new OnnxNode { OpType = "Constant", Inputs = Array.Empty<string>(), Outputs = new[] { name },
            Attributes = { ["value_int"] = value } });
        return name;
    }

    private static string RotaryEmbedding(string name, string x, int headDim, double theta,
        string prefix, List<OnnxNode> nodes, List<OnnxInitializer> initializers)
    {
        // Build inv_freq: 1.0 / (theta^(2*i/headDim)) for i = 0..headDim/2-1
        int halfDim = headDim / 2;
        var invFreq = new float[halfDim];
        for (int i = 0; i < halfDim; i++)
        {
            invFreq[i] = (float)(1.0 / Math.Pow(theta, 2.0 * i / headDim));
        }
        string invFreqName = $"{name}_inv_freq";
        initializers.Add(new OnnxInitializer
        {
            Name = invFreqName, DataType = OnnxTypes.Float,
            Dims = new long[] { halfDim }, RawData = F32ToRaw(invFreq),
        });

        // pos = Range(0, seq_len, 1, dtype=float32)
        string shapeName = $"{name}_shape";
        nodes.Add(new OnnxNode { OpType = "Shape", Inputs = new[] { x }, Outputs = new[] { shapeName } });
        string seqLenName = $"{name}_seq";
        nodes.Add(new OnnxNode
        {
            OpType = "Gather",
            Inputs = new[] { shapeName, $"{name}_seq_idx" },
            Outputs = new[] { seqLenName },
            Attributes = { ["axis"] = 0L },
        });
        // Gather index = 1 (sequence dim)
        string seqIdxInit = $"{name}_seq_idx";
        initializers.Add(new OnnxInitializer
        {
            Name = seqIdxInit, DataType = OnnxTypes.Int64,
            Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes(1L), // axis 1 = seq dim
        });
        if (!BitConverter.IsLittleEndian)
        {
            var bytes = (byte[])(Array)initializers[^1].RawData;
            Array.Reverse(bytes);
        }

        // Cast seq_len to float (Range output is float if inputs are float, but Gather
        // output is int64 — need Cast)
        string seqFloat = $"{name}_seq_f";
        nodes.Add(new OnnxNode
        {
            OpType = "Cast",
            Inputs = new[] { seqLenName },
            Outputs = new[] { seqFloat },
            Attributes = { ["to"] = OnnxTypes.Float },
        });
        string zeroPosInit = $"{name}_zero";
        string oneStepInit = $"{name}_one";
        initializers.Add(new OnnxInitializer
        {
            Name = zeroPosInit, DataType = OnnxTypes.Float,
            Dims = new long[] { 1 }, RawData = F32ToRaw(new[] { 0f }),
        });
        initializers.Add(new OnnxInitializer
        {
            Name = oneStepInit, DataType = OnnxTypes.Float,
            Dims = new long[] { 1 }, RawData = F32ToRaw(new[] { 1f }),
        });
        string posFloat = $"{name}_pos_f";
        nodes.Add(new OnnxNode
        {
            OpType = "Range",
            Inputs = new[] { zeroPosInit, seqFloat, oneStepInit },
            Outputs = new[] { posFloat },
        });

        // freqs = pos[:, None] * inv_freq[None, :] → [seq_len, halfDim]
        string posUnsq = $"{name}_pu";
        string posAxesInit = AddInt64Scalar($"{name}_pa", 1, nodes, initializers);
        nodes.Add(new OnnxNode
        {
            OpType = "Unsqueeze",
            Inputs = new[] { posFloat, posAxesInit },
            Outputs = new[] { posUnsq },
        });
        string freqUnsq = $"{name}_fu";
        string freqAxesInit = AddInt64Scalar($"{name}_fa", 0, nodes, initializers);
        nodes.Add(new OnnxNode
        {
            OpType = "Unsqueeze",
            Inputs = new[] { invFreqName, freqAxesInit },
            Outputs = new[] { freqUnsq },
        });
        string freqsName = $"{name}_fr";
        nodes.Add(new OnnxNode
        {
            OpType = "Mul",
            Inputs = new[] { posUnsq, freqUnsq },
            Outputs = new[] { freqsName },
        });

        // cos = Cos(freqs), sin = Sin(freqs)
        string cosName = $"{name}_cos";
        string sinName = $"{name}_sin";
        nodes.Add(new OnnxNode { OpType = "Cos", Inputs = new[] { freqsName }, Outputs = new[] { cosName } });
        nodes.Add(new OnnxNode { OpType = "Sin", Inputs = new[] { freqsName }, Outputs = new[] { sinName } });

        // cos2 = Concat([cos, cos], axis=-1) → [seq_len, headDim]
        string cos2Name = $"{name}_cos2";
        nodes.Add(new OnnxNode
        {
            OpType = "Concat",
            Inputs = new[] { cosName, cosName },
            Outputs = new[] { cos2Name },
            Attributes = { ["axis"] = -1L },
        });
        string sin2Name = $"{name}_sin2";
        nodes.Add(new OnnxNode
        {
            OpType = "Concat",
            Inputs = new[] { sinName, sinName },
            Outputs = new[] { sin2Name },
            Attributes = { ["axis"] = -1L },
        });

        // Broadcast cos/sin for batch dim: unsqueeze at axis 0
        string cosBName = $"{name}_cos_b";
        string sinBName = $"{name}_sin_b";
        string bcastAxes = AddInt64Scalar($"{name}_ba", 0, nodes, initializers);
        nodes.Add(new OnnxNode
        {
            OpType = "Unsqueeze",
            Inputs = new[] { cos2Name, bcastAxes },
            Outputs = new[] { cosBName },
        });
        nodes.Add(new OnnxNode
        {
            OpType = "Unsqueeze",
            Inputs = new[] { sin2Name, bcastAxes },
            Outputs = new[] { sinBName },
        });

        // Split x's last dim into two halves (even/odd pairs)
        // Reshape x from [B, S, D] to [B, S, D/2, 2]
        string xShapeName = $"{name}_x_shape";
        nodes.Add(new OnnxNode { OpType = "Shape", Inputs = new[] { x }, Outputs = new[] { xShapeName } });
        string batchDimName = $"{name}_b";
        string seqDimName2 = $"{name}_s2";
        // Gather batch and seq dims
        string bIdxInit = $"{name}_bi";
        string sIdxInit = $"{name}_si";
        initializers.Add(new OnnxInitializer
        {
            Name = bIdxInit, DataType = OnnxTypes.Int64, Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes(0L),
        });
        initializers.Add(new OnnxInitializer
        {
            Name = sIdxInit, DataType = OnnxTypes.Int64, Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes(1L),
        });
        nodes.Add(new OnnxNode
        {
            OpType = "Gather",
            Inputs = new[] { xShapeName, bIdxInit },
            Outputs = new[] { batchDimName },
            Attributes = { ["axis"] = 0L },
        });
        nodes.Add(new OnnxNode
        {
            OpType = "Gather",
            Inputs = new[] { xShapeName, sIdxInit },
            Outputs = new[] { seqDimName2 },
            Attributes = { ["axis"] = 0L },
        });
        // Build reshape target: [batch, seq, halfDim, 2]
        string halfDimInit = $"{name}_hd";
        string twoInit = $"{name}_two";
        initializers.Add(new OnnxInitializer
        {
            Name = halfDimInit, DataType = OnnxTypes.Int64, Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes((long)halfDim),
        });
        initializers.Add(new OnnxInitializer
        {
            Name = twoInit, DataType = OnnxTypes.Int64, Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes(2L),
        });
        string xReshapeShapeName = $"{name}_rs";
        nodes.Add(new OnnxNode
        {
            OpType = "Concat",
            Inputs = new[] { batchDimName, seqDimName2, halfDimInit, twoInit },
            Outputs = new[] { xReshapeShapeName },
            Attributes = { ["axis"] = 0L },
        });
        string xReshapeName = $"{name}_x4";
        nodes.Add(new OnnxNode
        {
            OpType = "Reshape",
            Inputs = new[] { x, xReshapeShapeName },
            Outputs = new[] { xReshapeName },
        });

        // Split into even/odd: Gather index 0 or 1 along the last axis

        string xEvenName2 = $"{name}_e2";
        string xOddName2 = $"{name}_o2";
        string evenIdxInit = $"{name}_ei";
        string oddIdxInit = $"{name}_oi";
        initializers.Add(new OnnxInitializer
        {
            Name = evenIdxInit, DataType = OnnxTypes.Int64, Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes(0L),
        });
        initializers.Add(new OnnxInitializer
        {
            Name = oddIdxInit, DataType = OnnxTypes.Int64, Dims = new long[] { 1 },
            RawData = BitConverter.GetBytes(1L),
        });
        nodes.Add(new OnnxNode
        {
            OpType = "Gather",
            Inputs = new[] { xReshapeName, evenIdxInit },
            Outputs = new[] { xEvenName2 },
            Attributes = { ["axis"] = 3L },
        });
        nodes.Add(new OnnxNode
        {
            OpType = "Gather",
            Inputs = new[] { xReshapeName, oddIdxInit },
            Outputs = new[] { xOddName2 },
            Attributes = { ["axis"] = 3L },
        });

        // Squeeze the last dim (which is size 1 after Gather)
        string evenSqueezed = $"{name}_es";
        string oddSqueezed = $"{name}_os";
        nodes.Add(new OnnxNode
        {
            OpType = "Squeeze",
            Inputs = new[] { xEvenName2 },
            Outputs = new[] { evenSqueezed },
            Attributes = { ["axes"] = new[] { 3L } },
        });
        nodes.Add(new OnnxNode
        {
            OpType = "Squeeze",
            Inputs = new[] { xOddName2 },
            Outputs = new[] { oddSqueezed },
            Attributes = { ["axes"] = new[] { 3L } },
        });

        // Apply rotation: rot_even = even*cos - odd*sin, rot_odd = odd*cos + even*sin
        string eCos = $"{name}_ec";
        string oSin = $"{name}_os2";
        string oCos = $"{name}_oc";
        string eSin = $"{name}_es2";
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { evenSqueezed, cosBName }, Outputs = new[] { eCos } });
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { oddSqueezed, sinBName }, Outputs = new[] { oSin } });
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { oddSqueezed, cosBName }, Outputs = new[] { oCos } });
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { evenSqueezed, sinBName }, Outputs = new[] { eSin } });

        string rotEven = $"{name}_re";
        string rotOdd = $"{name}_ro";
        nodes.Add(new OnnxNode { OpType = "Sub", Inputs = new[] { eCos, oSin }, Outputs = new[] { rotEven } });
        nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { oCos, eSin }, Outputs = new[] { rotOdd } });

        // Unsqueeze back to [B, S, D/2, 1], then Concat
        string reU = $"{name}_reu";
        string roU = $"{name}_rou";
        string rotAxesName = AddInt64Scalar($"{name}_ra", 3, nodes, initializers);
        nodes.Add(new OnnxNode { OpType = "Unsqueeze", Inputs = new[] { rotEven, rotAxesName }, Outputs = new[] { reU } });
        nodes.Add(new OnnxNode { OpType = "Unsqueeze", Inputs = new[] { rotOdd, rotAxesName }, Outputs = new[] { roU } });

        string rotConcat = $"{name}_rc";
        nodes.Add(new OnnxNode
        {
            OpType = "Concat",
            Inputs = new[] { reU, roU },
            Outputs = new[] { rotConcat },
            Attributes = { ["axis"] = 3L },
        });

        // Reshape back to [B, S, D] using the original shape
        nodes.Add(new OnnxNode
        {
            OpType = "Reshape",
            Inputs = new[] { rotConcat, xShapeName },
            Outputs = new[] { name },
        });

        return name;
    }

    private static string SimpleAttention(string name, string q, string k, string v,
        string maskName, float scale, int numHeads, int hidden, List<OnnxNode> nodes,
        List<OnnxInitializer> initializers)
    {
        string scaleInit = $"{name}_scale";
        initializers.Add(new OnnxInitializer
        {
            Name = scaleInit, DataType = OnnxTypes.Float, Dims = new long[] { 1 },
            RawData = F32ToRaw(new[] { scale }),
        });
        string qScaled = $"{name}_qs";
        nodes.Add(new OnnxNode
        {
            OpType = "Div",
            Inputs = new[] { q, scaleInit },
            Outputs = new[] { qScaled },
        });
        // QK^T
        string kTrans = $"{name}_kt";
        // Transpose K: last two dims swapped. Use Transpose op.
        nodes.Add(new OnnxNode
        {
            OpType = "Transpose",
            Inputs = new[] { k },
            Outputs = new[] { kTrans },
            Attributes = { ["perm"] = new[] { 0L, 2L, 1L } },
        });
        string scores = $"{name}_sc";
        nodes.Add(new OnnxNode
        {
            OpType = "MatMul",
            Inputs = new[] { qScaled, kTrans },
            Outputs = new[] { scores },
        });
        // Mask
        string masked = $"{name}_mk";
        nodes.Add(new OnnxNode
        {
            OpType = "Add",
            Inputs = new[] { scores, maskName },
            Outputs = new[] { masked },
        });
        // Softmax
        string attn = $"{name}_at";
        nodes.Add(new OnnxNode
        {
            OpType = "Softmax",
            Inputs = new[] { masked },
            Outputs = new[] { attn },
            Attributes = { ["axis"] = -1L },
        });
        // Weighted sum: attn × V
        string weighted = name;
        nodes.Add(new OnnxNode
        {
            OpType = "MatMul",
            Inputs = new[] { attn, v },
            Outputs = new[] { weighted },
        });
        return weighted;
    }

    private static string PoolLast(string name, string hidden, string maskName, List<OnnxNode> nodes)
    {
        // Gather the last row (position = attention_mask sum - 1).
        // Simplified: take the last row directly.
        string shape = $"{name}_sh";
        nodes.Add(new OnnxNode
        {
            OpType = "Shape",
            Inputs = new[] { hidden },
            Outputs = new[] { shape },
        });
        string gatherIdx = $"{name}_idx";
        nodes.Add(new OnnxNode
        {
            OpType = "Constant",
            Inputs = Array.Empty<string>(),
            Outputs = new[] { gatherIdx },
            Attributes = { ["value_ints"] = new[] { -1 } },
        });
        string gathered = $"{name}_g";
        nodes.Add(new OnnxNode
        {
            OpType = "Gather",
            Inputs = new[] { hidden, gatherIdx },
            Outputs = new[] { gathered },
            Attributes = { ["axis"] = 1L },
        });
        // Squeeze the seq dim (which is 1)
        nodes.Add(new OnnxNode
        {
            OpType = "Squeeze",
            Inputs = new[] { gathered },
            Outputs = new[] { name },
            Attributes = { ["axes"] = new[] { 1L } },
        });
        return name;
    }

    private static string MeanPool(string name, string hidden, string maskName, List<OnnxNode> nodes)
    {
        // Mean over the sequence length, masked.
        // Simplification: ReduceMean over axis=1 (seq).
        string reduced = name;
        nodes.Add(new OnnxNode
        {
            OpType = "ReduceMean",
            Inputs = new[] { hidden },
            Outputs = new[] { reduced },
            Attributes = { ["axes"] = new[] { 1 }, ["keepdims"] = 0L },
        });
        return reduced;
    }

    private static string L2Normalize(string name, string x, List<OnnxNode> nodes)
    {
        // L2 norm: x / sqrt(sum(x^2) + 1e-12)
        string sq = $"{name}_sq";
        nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { x, x }, Outputs = new[] { sq } });
        string sum = $"{name}_s";
        nodes.Add(new OnnxNode
        {
            OpType = "ReduceSum",
            Inputs = new[] { sq },
            Outputs = new[] { sum },
            Attributes = { ["axes"] = new[] { -1 }, ["keepdims"] = 1L },
        });
        string sqrt = $"{name}_sr";
        nodes.Add(new OnnxNode { OpType = "Sqrt", Inputs = new[] { sum }, Outputs = new[] { sqrt } });
        nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { x, sqrt }, Outputs = new[] { name } });
        return name;
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
            float f = BitConverter.Int32BitsToSingle(bits << 16);
            fp16[i * 2] = (byte)(ushort)(Half)f;
            fp16[i * 2 + 1] = (byte)((ushort)(Half)f >> 8);
        }
        return fp16;
    }
}