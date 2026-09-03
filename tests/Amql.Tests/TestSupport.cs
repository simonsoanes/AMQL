using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>Disposable temp directory for test fixtures.</summary>
public sealed class TempDir : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"amql-tests-{Guid.NewGuid():N}");

    public TempDir() => Directory.CreateDirectory(Path);

    public void Dispose()
    {
        if (Directory.Exists(Path))
        {
            Directory.Delete(Path, recursive: true);
        }
    }
}

/// <summary>Geometry of the synthetic model used by the inference tests.</summary>
public sealed record Dims(
    int Vocab = 9,
    int Hidden = 4,
    int NumQHeads = 2,
    int NumKvHeads = 1,
    int HeadDim = 2,
    int Layers = 2,
    int Intermediate = 6,
    double RopeTheta = 10_000.0,
    double NormEps = 1e-5,
    bool Rope = true,
    long? Window = null,
    bool MoE = false,
    int Experts = 2,
    int TopK = 1,
    bool WeightedQkNorm = false,
    bool OutputGate = false,
    bool LinearLayer0 = false)
{
    public int QDim => NumQHeads * HeadDim;
    public int KvDim => NumKvHeads * HeadDim;

    // Linear-attention geometry for the synthetic LinearLayer0 case.
    public const int LinKHeads = 2;
    public const int LinKHeadDim = 2;
    public const int LinVHeads = 2;
    public const int LinVHeadDim = 2;
    public const int LinConvKernel = 2;
    public int LinKeyDim => LinKHeads * LinKHeadDim;
    public int LinValueDim => LinVHeads * LinVHeadDim;
    public int LinConvDim => 2 * LinKeyDim + LinValueDim;
}

/// <summary>
/// Builds a deterministic synthetic Llama-shaped model as an in-memory
/// VINDEX3 spec. The same weight formula regenerates every tensor so the
/// naive reference in the tests is truly independent of the container.
/// </summary>
public static class SyntheticModel
{
    /// <summary>Deterministic weight value for (layer, row, col) with a
    /// per-tensor salt. Range [−0.5, 0.5).</summary>
    public static float W(int l, int row, int col, int salt) =>
        0.1f * ((l * 101 + row * 17 + col * 3 + salt * 5) % 11 - 5);

    /// <summary>Deterministic norm-scale value.</summary>
    public static float NormW(int i, int salt) => 1f + 0.1f * ((i * 7 + salt) % 3);

    public static ContainerSpec BuildSpec(Dims d)
    {
        var tensors = new Dictionary<string, (Dtype Dtype, long[] Shape, byte[] Data)>(StringComparer.Ordinal);

        void Matrix(string name, int rows, int cols, int salt, int l = 0)
        {
            var data = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    data[i * cols + j] = W(l, i, j, salt);
                }
            }
            tensors[name] = (Dtype.F32, new long[] { rows, cols }, ToBytes(data));
        }

        void Vector(string name, int width, int salt)
        {
            var data = new float[width];
            for (int i = 0; i < width; i++)
            {
                data[i] = NormW(i, salt);
            }
            tensors[name] = (Dtype.F32, new long[] { width }, ToBytes(data));
        }

        Matrix("embedding.weight", d.Vocab, d.Hidden, 1);
        Matrix("lm_head.weight", d.Vocab, d.Hidden, 9);
        Vector("final_norm.weight", d.Hidden, 3);

        for (int l = 0; l < d.Layers; l++)
        {
            bool linearLayer = d.LinearLayer0 && l == 0;
            if (linearLayer)
            {
                int qkvRows = d.LinConvDim;
                Matrix($"{l}.linear_attn.in_proj_qkv.weight", qkvRows, d.Hidden, 50, l);
                Matrix($"{l}.linear_attn.in_proj_z.weight", d.LinValueDim, d.Hidden, 51, l);
                Matrix($"{l}.linear_attn.in_proj_a.weight", Dims.LinVHeads, d.Hidden, 52, l);
                Matrix($"{l}.linear_attn.in_proj_b.weight", Dims.LinVHeads, d.Hidden, 53, l);
                Matrix($"{l}.linear_attn.out_proj.weight", d.Hidden, d.LinValueDim, 54, l);
                // conv1d weight [LinConvDim, 1, LinConvKernel] flattened.
                tensors[$"{l}.linear_attn.conv1d.weight"] = (
                    Dtype.F32,
                    new long[] { d.LinConvDim, 1, Dims.LinConvKernel },
                    ToBytes(Enumerable.Range(0, d.LinConvDim * Dims.LinConvKernel)
                        .Select(i => W(l, i, 0, 55)).ToArray()));
                // A_log and dt_bias are per-value-head vectors; A_log is a
                // deliberate F32 inside an otherwise F16 family in the real
                // model — here both are stored F32 for simplicity.
                tensors[$"{l}.linear_attn.A_log"] = (
                    Dtype.F32, new long[] { Dims.LinVHeads },
                    ToBytes(Enumerable.Range(0, Dims.LinVHeads).Select(i => W(l, i, 0, 56)).ToArray()));
                tensors[$"{l}.linear_attn.dt_bias"] = (
                    Dtype.F32, new long[] { Dims.LinVHeads },
                    ToBytes(Enumerable.Range(0, Dims.LinVHeads).Select(i => W(l, i, 0, 57)).ToArray()));
                tensors[$"{l}.linear_attn.norm.weight"] = (
                    Dtype.F32, new long[] { Dims.LinVHeadDim },
                    ToBytes(Enumerable.Range(0, Dims.LinVHeadDim).Select(i => W(l, i, 0, 58)).ToArray()));
            }
            else
            {
                Matrix($"{l}.self_attn.q_proj.weight", d.OutputGate ? 2 * d.QDim : d.QDim, d.Hidden, 2, l);
                Matrix($"{l}.self_attn.k_proj.weight", d.KvDim, d.Hidden, 3, l);
                Matrix($"{l}.self_attn.v_proj.weight", d.KvDim, d.Hidden, 4, l);
                Matrix($"{l}.self_attn.o_proj.weight", d.Hidden, d.QDim, 5, l);
            }
            Vector($"{l}.input_layernorm.weight", d.Hidden, 11);
            Vector($"{l}.post_attention_layernorm.weight", d.Hidden, 12);

            if (d.WeightedQkNorm && !linearLayer)
            {
                Vector($"{l}.self_attn.q_norm.weight", d.HeadDim, 40);
                Vector($"{l}.self_attn.k_norm.weight", d.HeadDim, 41);
            }

            if (d.MoE)
            {
                Matrix($"{l}.mlp.router.weight", d.Experts, d.Hidden, 13, l);
                for (int e = 0; e < d.Experts; e++)
                {
                    Matrix($"{l}.mlp.experts.{e}.gate_proj.weight", d.Intermediate, d.Hidden, 14 + e, l);
                    Matrix($"{l}.mlp.experts.{e}.up_proj.weight", d.Intermediate, d.Hidden, 20 + e, l);
                    Matrix($"{l}.mlp.experts.{e}.down_proj.weight", d.Hidden, d.Intermediate, 30 + e, l);
                }
            }
            else
            {
                Matrix($"{l}.mlp.gate_proj.weight", d.Intermediate, d.Hidden, 6, l);
                Matrix($"{l}.mlp.up_proj.weight", d.Intermediate, d.Hidden, 7, l);
                Matrix($"{l}.mlp.down_proj.weight", d.Hidden, d.Intermediate, 8, l);
            }
        }

        // Linear-attention surface facts (used when a linear layer exists).
        System.Text.Json.JsonElement? linearSurface = d.LinearLayer0
            ? System.Text.Json.JsonSerializer.SerializeToElement(new
            {
                key_heads = Dims.LinKHeads,
                key_head_dim = Dims.LinKHeadDim,
                value_heads = Dims.LinVHeads,
                value_head_dim = Dims.LinVHeadDim,
                conv_kernel = Dims.LinConvKernel,
                state_dtype = "float32",
            }, ViJson.Options)
            : null;

        var normSpec = new NormSpec { Kind = NormType.RmsNorm, Eps = d.NormEps, WeightOffset = 0 };
        var surface = new ExecutionSurface
        {
            ContextLength = 2048,
            LinearAttention = linearSurface,
            Attention = new AttentionSurface
            {
                NumQHeads = d.NumQHeads,
                NumKvHeads = d.NumKvHeads,
                HeadDim = d.HeadDim,
                ScoreScale = 1.0 / Math.Sqrt(d.HeadDim),
                OutputGate = d.OutputGate
                    ? System.Text.Json.JsonSerializer.SerializeToElement(new { attn_output_gate = true }, ViJson.Options)
                    : null,
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = d.Intermediate,
                Activation = Activation.Silu,
                FfnType = FfnType.Gated,
                Moe = d.MoE
                    ? new MoeSurface
                    {
                        Experts = d.Experts,
                        TopK = d.TopK,
                        ExpertIntermediateSize = d.Intermediate,
                        RoutingPolicy = ExpertRoutingPolicy.SoftmaxThenSelect,
                    }
                    : null,
            },
            Norm = new NormSurface
            {
                Pre = normSpec,
                Post = normSpec,
                FinalNorm = normSpec,
                Placement = NormPlacement.PreOnly,
            },
            Head = new HeadSurface { VocabSize = d.Vocab },
        };

        var policies = new List<AttentionLayerPolicy>();
        for (int l = 0; l < d.Layers; l++)
        {
            bool linearLayer = d.LinearLayer0 && l == 0;
            policies.Add(new AttentionLayerPolicy
            {
                Operator = linearLayer ? LayerOperators.LinearAttention : LayerOperators.Softmax,
                Span = d.Window is null ? AttentionSpan.Full : AttentionSpan.Sliding,
                Window = linearLayer ? null : d.Window,
                // Linear layers are position-free; softmax layers use rope.
                Position = linearLayer ? PositionPolicy.None : (d.Rope ? PositionPolicy.CreateRope(d.RopeTheta) : PositionPolicy.None),
                Geometry = new HeadGeometry { HeadDim = d.HeadDim, NumKvHeads = d.NumKvHeads },
            });
        }

        var graph = new SystemGraph
        {
            Schema = SystemGraph.CurrentSchema,
            Components = new List<Component>
            {
                new()
                {
                    Id = "target",
                    Role = ComponentRole.PrimaryText,
                    SourceArtifact = "synth",
                    NumLayers = d.Layers,
                    HiddenSize = d.Hidden,
                    Attention = policies,
                    Execution = surface,
                },
            },
            Objects = new List<LogicalObject>
            {
                Object("target.embedding", "target", ObjectKind.Embedding),
                Object("target.decoder_stack", "target", ObjectKind.DecoderStack),
                Object("target.final_norm", "target", ObjectKind.FinalNorm),
                Object("target.output_head", "target", ObjectKind.OutputHead),
            },
            Edges = new List<HiddenStateEdge>(),
        };

        var reps = new List<RepresentationSpec>
        {
            Rep("target.embedding", "weight", tensors.Where(t => t.Key == "embedding.weight")),
            Rep("target.decoder_stack", null, tensors.Where(t => t.Key != "embedding.weight" && t.Key != "lm_head.weight" && t.Key != "final_norm.weight")),
            Rep("target.final_norm", "weight", tensors.Where(t => t.Key == "final_norm.weight")),
            Rep("target.output_head", "weight", tensors.Where(t => t.Key == "lm_head.weight")),
        };

        return new ContainerSpec
        {
            Model = "synth-lm",
            Family = "synthetic",
            HiddenSize = d.Hidden,
            NumLayers = d.Layers,
            SystemGraph = graph,
            Representations = reps,
        };

        static LogicalObject Object(string id, string component, ObjectKind kind) => new()
        {
            Id = id,
            Component = component,
            Kind = kind,
            SourceBindings = new List<SourceBinding>
            {
                new() { Artifact = "synth", TensorPrefix = id, Tensors = 1, Bytes = 0 },
            },
            Representations = new List<Representation>
            {
                new() { Encoding = "F32", Fidelity = Fidelity.Canonical },
            },
        };

        static RepresentationSpec Rep(string objectId, string? relativeName, IEnumerable<KeyValuePair<string, (Dtype Dtype, long[] Shape, byte[] Data)>> source) => new()
        {
            ObjectId = objectId,
            Encoding = "F32",
            Tensors = source
                .OrderBy(t => t.Key, StringComparer.Ordinal)
                .Select(t => new NamedTensorData
                {
                    // Segment tensor names are object-relative and
                    // structural (the reference's "physical names bind an
                    // object but never define its identity"); single-tensor
                    // objects carry the plain "weight".
                    Name = relativeName ?? t.Key,
                    Dtype = t.Value.Dtype,
                    Shape = t.Value.Shape,
                    Data = t.Value.Data,
                })
                .ToList(),
        };
    }

    public static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public static float[] FromBytes(byte[] bytes)
    {
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}