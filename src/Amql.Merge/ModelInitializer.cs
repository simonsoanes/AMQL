using System.Text.Json;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

public sealed record ModelSpec(
    string ModelName,
    string Family,
    int HiddenSize,
    int NumLayers,
    int NumQHeads,
    int NumKvHeads,
    int HeadDim,
    int IntermediateSize,
    int VocabSize,
    int ContextLength,
    double NormEps,
    bool TieEmbeddings,
    IReadOnlyList<string> LayerTypes)
{
    public static ModelSpec Default =>
        new(
            ModelName: "untitled-model",
            Family: "qwen3_5",
            HiddenSize: 768,
            NumLayers: 12,
            NumQHeads: 12,
            NumKvHeads: 2,
            HeadDim: 64,
            IntermediateSize: 2048,
            VocabSize: 32000,
            ContextLength: 2048,
            NormEps: 1e-5,
            TieEmbeddings: true,
            LayerTypes: Enumerable.Repeat("full_attention", 12).ToList()
        );
}

/// <summary>
/// Creates a new VINDEX3 container from scratch with randomly initialised
/// weights and a user-defined architecture. The resulting container is ready
/// for training via <c>fine-tune</c> and all other AMQL commands.
/// </summary>
public static class ModelInitializer
{
    public static string Create(ModelSpec spec, string outDir, int seed = 42)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"container output '{outDir}' already exists");
        }
        if (spec.LayerTypes.Count != spec.NumLayers)
        {
            throw new MergeException(
                $"layer_types count ({spec.LayerTypes.Count}) != num_layers ({spec.NumLayers})");
        }

        var rng = new Random(seed);
        var notes = new List<string>();

        int hidden = spec.HiddenSize;
        int layers = spec.NumLayers;
        int qDim = spec.NumQHeads * spec.HeadDim;
        int kvDim = spec.NumKvHeads * spec.HeadDim;
        int intermediate = spec.IntermediateSize;
        int vocab = spec.VocabSize;

        Directory.CreateDirectory(outDir);

        // ── 1. Create the weight tensors ────────────────────────────────
        var decoderTensors = new List<NamedTensorData>();
        void Matrix(List<NamedTensorData> list, string name, int rows, int cols)
        {
            var data = XavierInit(rng, rows, cols);
            list.Add(new NamedTensorData
            {
                Name = name, Dtype = Dtype.F32,
                Shape = new long[] { rows, cols },
                Data = FloatToBytes(data),
            });
        }

        void Vector(List<NamedTensorData> list, string name, int width)
        {
            var data = new float[width];
            for (int i = 0; i < width; i++)
            {
                // RMSNorm weights start at 1.0 (identity)
                data[i] = 1.0f;
            }
            list.Add(new NamedTensorData
            {
                Name = name, Dtype = Dtype.F32,
                Shape = new long[] { width },
                Data = FloatToBytes(data),
            });
        }

        // Embedding (always written, even when tied — it's the source)
        var embTensors = new List<NamedTensorData>();
        Matrix(embTensors, "weight", vocab, hidden);

        // Decoder stack: per-layer weights
        for (int l = 0; l < layers; l++)
        {
            string prefix = $"{l}.";
            Matrix(decoderTensors, $"{prefix}self_attn.q_proj.weight", hidden, qDim);
            Matrix(decoderTensors, $"{prefix}self_attn.k_proj.weight", hidden, kvDim);
            Matrix(decoderTensors, $"{prefix}self_attn.v_proj.weight", hidden, kvDim);
            Matrix(decoderTensors, $"{prefix}self_attn.o_proj.weight", hidden, qDim);
            Vector(decoderTensors, $"{prefix}input_layernorm.weight", hidden);
            Vector(decoderTensors, $"{prefix}post_attention_layernorm.weight", hidden);
            Matrix(decoderTensors, $"{prefix}mlp.gate_proj.weight", intermediate, hidden);
            Matrix(decoderTensors, $"{prefix}mlp.up_proj.weight", intermediate, hidden);
            Matrix(decoderTensors, $"{prefix}mlp.down_proj.weight", hidden, intermediate);
        }

        // Final norm
        var normTensors = new List<NamedTensorData>();
        Vector(normTensors, "weight", hidden);

        // Output head (only if untied)
        var headTensors = new List<NamedTensorData>();
        bool untied = !spec.TieEmbeddings;
        if (untied)
        {
            Matrix(headTensors, "weight", vocab, hidden);
        }

        // ── 2. Write segments ───────────────────────────────────────────
        string embSegmentPath = "segments/target.embedding.bin";
        var embResult = SegmentWriter.Write(
            Path.Combine(outDir, embSegmentPath), "target.embedding@F32", embTensors);

        string decoderSegmentPath = "segments/target.decoder_stack.bin";
        var decoderResult = SegmentWriter.Write(
            Path.Combine(outDir, decoderSegmentPath), "target.decoder_stack@F32", decoderTensors);

        string normSegmentPath = "segments/target.final_norm.bin";
        var normResult = SegmentWriter.Write(
            Path.Combine(outDir, normSegmentPath), "target.final_norm@F32", normTensors);

        string headSegmentPath = "segments/target.output_head.bin";
        var headResult = untied
            ? SegmentWriter.Write(Path.Combine(outDir, headSegmentPath), "target.output_head@F32", headTensors)
            : null;

        // ── 3. Build representations map ────────────────────────────────
        var representations = new Dictionary<string, RepresentationEntry>
        {
            ["target.embedding@F32"] = new()
            {
                Object = "target.embedding", Encoding = "F32",
                Segment = embSegmentPath,
                TensorCount = embTensors.Count,
                PayloadBytes = embResult.PayloadBytes,
                PayloadSha256 = embResult.PayloadSha256Hex,
                SegmentSha256 = embResult.SegmentSha256Hex,
            },
            ["target.decoder_stack@F32"] = new()
            {
                Object = "target.decoder_stack", Encoding = "F32",
                Segment = decoderSegmentPath,
                TensorCount = decoderTensors.Count,
                PayloadBytes = decoderResult.PayloadBytes,
                PayloadSha256 = decoderResult.PayloadSha256Hex,
                SegmentSha256 = decoderResult.SegmentSha256Hex,
            },
            ["target.final_norm@F32"] = new()
            {
                Object = "target.final_norm", Encoding = "F32",
                Segment = normSegmentPath,
                TensorCount = normTensors.Count,
                PayloadBytes = normResult.PayloadBytes,
                PayloadSha256 = normResult.PayloadSha256Hex,
                SegmentSha256 = normResult.SegmentSha256Hex,
            },
        };
        var segments = new Dictionary<string, int>
        {
            ["segments/target.embedding"] = 1,
            ["segments/target.decoder_stack"] = layers,
            ["segments/target.final_norm"] = 1,
        };

        if (untied && headResult is not null)
        {
            representations["target.output_head@F32"] = new()
            {
                Object = "target.output_head", Encoding = "F32",
                Segment = headSegmentPath,
                TensorCount = headTensors.Count,
                PayloadBytes = headResult.PayloadBytes,
                PayloadSha256 = headResult.PayloadSha256Hex,
                SegmentSha256 = headResult.SegmentSha256Hex,
            };
            segments["segments/target.output_head"] = 1;
        }

        // ── 4. Build the system graph ───────────────────────────────────
        var position = new PositionRope { Theta = 1_000_000.0 };
        var policies = new List<AttentionLayerPolicy>(layers);
        var objectSuffixes = new[] { "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight" };

        for (int l = 0; l < layers; l++)
        {
            string op = spec.LayerTypes[l] switch
            {
                "full_attention" => LayerOperators.Softmax,
                "linear_attention" => LayerOperators.LinearAttention,
                var other => throw new MergeException(
                    $"layer {l}: operator '{other}' is not supported by create-model. " +
                    "Use 'full_attention' or 'linear_attention'.")
            };
            policies.Add(new AttentionLayerPolicy
            {
                Operator = op,
                Span = AttentionSpan.Full,
                Position = position,
                Geometry = new HeadGeometry { HeadDim = spec.HeadDim, NumKvHeads = spec.NumKvHeads },
            });
        }

        var surface = new ExecutionSurface
        {
            ContextLength = spec.ContextLength,
            Attention = new AttentionSurface
            {
                NumQHeads = spec.NumQHeads,
                NumKvHeads = spec.NumKvHeads,
                HeadDim = spec.HeadDim,
                ScoreScale = 1.0 / Math.Sqrt(spec.HeadDim),
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = spec.IntermediateSize,
                Activation = Activation.Silu,
                FfnType = FfnType.Gated,
            },
            Norm = new NormSurface
            {
                Pre = new NormSpec { Kind = NormType.RmsNorm, Eps = spec.NormEps },
                Post = new NormSpec { Kind = NormType.RmsNorm, Eps = spec.NormEps },
                FinalNorm = new NormSpec { Kind = NormType.RmsNorm, Eps = spec.NormEps },
                Placement = NormPlacement.PreOnly,
            },
            Head = new HeadSurface
            {
                VocabSize = spec.VocabSize,
                HeadReusesEmbedding = spec.TieEmbeddings,
            },
        };

        var objects = new List<LogicalObject>
        {
            new()
            {
                Id = "target.embedding", Component = "target",
                Kind = ObjectKind.Embedding,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "model.embed_tokens", TensorPrefix = "model.embed_tokens",
                        Tensors = embTensors.Count, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = "F32", Fidelity = Fidelity.Canonical },
                },
            },
            new()
            {
                Id = "target.decoder_stack", Component = "target",
                Kind = ObjectKind.DecoderStack,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "model.layers", TensorPrefix = "model.layers",
                        Tensors = decoderTensors.Count, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = "F32", Fidelity = Fidelity.Canonical },
                },
            },
            new()
            {
                Id = "target.final_norm", Component = "target",
                Kind = ObjectKind.FinalNorm,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "model.norm", TensorPrefix = "model.norm",
                        Tensors = normTensors.Count, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = "F32", Fidelity = Fidelity.Canonical },
                },
            },
        };

        if (untied)
        {
            objects.Add(new LogicalObject
            {
                Id = "target.output_head", Component = "target",
                Kind = ObjectKind.OutputHead,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "lm_head", TensorPrefix = "lm_head",
                        Tensors = headTensors.Count, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = "F32", Fidelity = Fidelity.Canonical },
                },
            });
        }

        var components = new List<Component>
        {
            new()
            {
                Id = "target", Role = ComponentRole.PrimaryText,
                SourceArtifact = "model.layers",
                NumLayers = layers,
                HiddenSize = hidden,
                Attention = policies,
                Execution = surface,
            },
        };

        var graph = new SystemGraph
        {
            Schema = SystemGraph.CurrentSchema,
            Components = components,
            Objects = objects,
            Edges = new List<HiddenStateEdge>(),
        };

        File.WriteAllText(Path.Combine(outDir, "system_graph.json"),
            JsonSerializer.Serialize(graph, ViJson.Options));

        // ── 5. Write the index ──────────────────────────────────────────
        var index = new Vindex3Index
        {
            Version = Vindex3Index.CurrentSchema,
            Model = spec.ModelName,
            Family = spec.Family,
            HiddenSize = hidden,
            NumLayers = layers,
            SystemGraph = "system_graph.json",
            Representations = representations,
            Profiles = new List<Profile> { Profile.Exact() },
            Segments = segments,
            Authority = ContainerAuthority.Canonical,
        };

        File.WriteAllText(Path.Combine(outDir, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));

        notes.Add($"model: {spec.ModelName} ({spec.Family})");
        notes.Add($"architecture: {layers} layers × {hidden} hidden, {spec.NumQHeads}/{spec.NumKvHeads} heads × {spec.HeadDim} dim");
        notes.Add($"vocab: {vocab}, context: {spec.ContextLength}, tied: {spec.TieEmbeddings}");
        notes.Add($"weights: Xavier-uniform init, norm weights = 1.0, seed {seed}");
        notes.Add("the container is ready for training — use 'amql-cli fine-tune' to apply training data");

        return outDir;
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static float[] XavierInit(Random rng, int rows, int cols)
    {
        float scale = MathF.Sqrt(6.0f / (rows + cols)); // Xavier uniform
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        }
        return data;
    }

    private static byte[] FloatToBytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}