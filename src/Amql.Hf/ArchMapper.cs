using System.Text.Json;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Hf;

/// <summary>
/// G1 resolution → G2 graph → G3 materialisation instruction, all in one
/// pass: the reference's invent/representability/plan pipeline, .NET-shaped.
/// The graph is built from <em>judged facts only</em> — an unseen
/// layer_type or activation refuses the checkpoint, it never defaults.
/// Unknown surface facts (partial MRoPE, output gates, the linear-attention
/// operator) are <em>carried verbatim in the graph</em>, exactly as the
/// runtime's fail-closed contract demands: recorded, never approximated.
/// </summary>
public static class ArchMapper
{
    public sealed record EncodeOptions
    {
        public bool IncludeVision { get; init; } = true;
        public bool IncludeMtp { get; init; } = true;
    }

    public static ContainerSpec MapToContainerSpec(string modelId, TextArchitectureFacts facts, HfInventory inventory, EncodeOptions options)
    {
        string prefix = DetectTextPrefix(inventory);
        if (facts.LayerTypes.Count != facts.NumLayers)
        {
            throw new ModelConfigException(
                $"layer_types declares {facts.LayerTypes.Count} layers but num_hidden_layers is {facts.NumLayers}");
        }
        if (facts.HiddenAct != "silu")
        {
            throw new ModelConfigException(
                $"hidden_act '{facts.HiddenAct}' has no judged FFN mapping (only 'silu')");
        }

        // ── per-layer policy table ────────────────────────────────────────
        // Position policy: plain default rope (no MRoPE sections, full
        // rotary factor, no frequency scaling) is SERVED as a standard
        // PositionRope; every other rope fact (partial factor, MRoPE
        // sections, scaled families) is carried verbatim as unresolved and
        // refused by the planner naming the kind.
        PositionPolicy position = JudgePosition(facts);
        var policies = new List<AttentionLayerPolicy>(facts.NumLayers);
        for (int l = 0; l < facts.NumLayers; l++)
        {
            bool linear = facts.LayerTypes[l] == "linear_attention";
            HeadGeometry geometry = linear && facts.LinearAttention is { } lin
                ? new HeadGeometry { HeadDim = lin.KeyHeadDim, NumKvHeads = lin.KeyHeads }
                : new HeadGeometry { HeadDim = facts.HeadDim, NumKvHeads = facts.NumKvHeads };
            policies.Add(new AttentionLayerPolicy
            {
                Operator = facts.LayerTypes[l] switch
                {
                    "full_attention" => LayerOperators.Softmax,
                    "linear_attention" => LayerOperators.LinearAttention,
                    var other => throw new ModelConfigException(
                        $"layer {l}: layer_type '{other}' has no judged operator mapping"),
                },
                Span = AttentionSpan.Full,
                Position = position,
                Geometry = geometry,
            });
        }

        // ── execution surface ────────────────────────────────────────────
        var normSpec = new NormSpec { Kind = NormType.RmsNorm, Eps = facts.RmsNormEps, WeightOffset = 0 };
        var surface = new ExecutionSurface
        {
            ContextLength = facts.MaxPositionEmbeddings,
            Attention = new AttentionSurface
            {
                NumQHeads = facts.NumQueryHeads,
                NumKvHeads = facts.NumKvHeads,
                HeadDim = facts.HeadDim,
                ScoreScale = 1.0 / Math.Sqrt(facts.HeadDim),
                QkNormScope = QkNormScope.PerHead,
                AttentionBias = facts.AttentionBias,
                OutputGate = facts.AttentionOutputGate
                    ? JsonSerializer.SerializeToElement(new { attn_output_gate = true }, ViJson.Options)
                    : null,
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = facts.IntermediateSize,
                Activation = Activation.Silu,
                FfnType = FfnType.Gated,
            },
            Norm = new NormSurface
            {
                Pre = normSpec,
                Post = normSpec,
                FinalNorm = normSpec,
                Placement = NormPlacement.PreOnly,
            },
            Head = new HeadSurface
            {
                VocabSize = facts.VocabSize,
                HeadReusesEmbedding = facts.TieWordEmbeddings,
            },
            LinearAttention = facts.LinearAttention is { } la
                ? JsonSerializer.SerializeToElement(new
                {
                    key_heads = la.KeyHeads,
                    key_head_dim = la.KeyHeadDim,
                    value_heads = la.ValueHeads,
                    value_head_dim = la.ValueHeadDim,
                    conv_kernel = la.ConvKernelDim,
                    state_dtype = "float32",
                }, ViJson.Options)
                : null,
        };

        // ── canonical encoding: judged from the stored dtype, never
        // assumed — the checkpoint is BF16 today; that is a fact of the
        // artifact, not of the family. The whole materialised stack must
        // share one encoding (canonical, unquantised in this build).
        string encoding = EncodingFor(inventory, $"{prefix}.layers.0.self_attn.q_proj.weight") ??
                          EncodingFor(inventory, $"{prefix}.embed_tokens.weight") ??
                          throw new ModelConfigException("no decoder stack tensors found in the inventory");
        Dtype encodingDtype = DtypeExtensions.FromLabel(encoding);

        // ── logical objects ──────────────────────────────────────────────
        var objects = new List<LogicalObject>
        {
            TextObject("target.embedding", ObjectKind.Embedding, prefix, "embed_tokens", encoding),
            TextObject("target.decoder_stack", ObjectKind.DecoderStack, prefix, "layers", encoding),
            TextObject("target.final_norm", ObjectKind.FinalNorm, prefix, "norm", encoding),
            new()
            {
                Id = "target.output_head",
                Component = "target",
                Kind = ObjectKind.OutputHead,
                SourceBindings = new List<SourceBinding>(),
                Representations = new List<Representation>(), // tied: no dedicated segment
            },
        };

        var components = new List<Component>
        {
            new()
            {
                Id = "target",
                Role = ComponentRole.PrimaryText,
                SourceArtifact = prefix,
                NumLayers = facts.NumLayers,
                HiddenSize = facts.HiddenSize,
                Attention = policies,
                Execution = surface,
            },
        };

        if (options.IncludeVision)
        {
            const string visionPrefix = "model.visual";
            objects.Add(new LogicalObject
            {
                Id = "vision.perception_tower",
                Component = "vision",
                Kind = ObjectKind.PerceptionTower,
                SourceBindings = new List<SourceBinding>
                {
                    new()
                    {
                        Artifact = visionPrefix,
                        TensorPrefix = visionPrefix + ".",
                        Tensors = inventory.CountUnder(visionPrefix + "."),
                        Bytes = inventory.BytesUnder(visionPrefix + "."),
                    },
                },
                Representations = new List<Representation>(), // carried, not materialised
            });
            components.Add(new Component
            {
                Id = "vision",
                Role = ComponentRole.Perception,
                SourceArtifact = visionPrefix,
                NumLayers = 12,
                HiddenSize = 768,
                Perception = JsonSerializer.SerializeToElement(
                    new { modality = "image", transform = new { kind = "encoder" } }, ViJson.Options),
            });
        }

        if (options.IncludeMtp)
        {
            const string mtpPrefix = "mtp";
            objects.Add(new LogicalObject
            {
                Id = "mtp.stack",
                Component = "mtp",
                Kind = ObjectKind.DecoderStack,
                SourceBindings = new List<SourceBinding>
                {
                    new()
                    {
                        Artifact = mtpPrefix,
                        TensorPrefix = mtpPrefix + ".",
                        Tensors = inventory.CountUnder(mtpPrefix + "."),
                        Bytes = inventory.BytesUnder(mtpPrefix + "."),
                    },
                },
                Representations = new List<Representation>(),
            });
            components.Add(new Component
            {
                Id = "mtp",
                Role = ComponentRole.Drafter,
                SourceArtifact = mtpPrefix,
                NumLayers = 1,
                HiddenSize = facts.HiddenSize,
            });
        }

        var graph = new SystemGraph
        {
            Schema = SystemGraph.CurrentSchema,
            Components = components,
            Objects = objects,
            Edges = new List<HiddenStateEdge>(),
        };

        // ── representation specs: bind actual shard tensors ──────────────
        var reps = new List<RepresentationSpec>
        {
            Rep("target.embedding", encoding,
                BindOne(inventory, prefix, "embed_tokens.weight")),
            Rep("target.decoder_stack", encoding,
                BindLayers(inventory, prefix)),
            Rep("target.final_norm", encoding,
                BindOne(inventory, prefix, "norm.weight")),
        };

        // Stored-precision policy: the canonical encoding is the stack
        // majority; tensors deliberately kept in another dtype (Qwen3.5
        // keeps A_log / the recurrent norm in F32 inside the BF16 stack)
        // are recorded as exceptions, never promoted. Segment headers carry
        // each tensor's own dtype verbatim either way.
        var exceptions = reps
            .SelectMany(r => r.Tensors)
            .Where(t => t.Dtype != encodingDtype)
            .Select(t => t.Name)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToList();

        return new ContainerSpec
        {
            Model = modelId,
            Family = facts.ModelType,
            HiddenSize = facts.HiddenSize,
            NumLayers = facts.NumLayers,
            SystemGraph = graph,
            Representations = reps,
            PrecisionMap = exceptions.Count == 0
                ? null
                : new PrecisionMap
                {
                    Name = $"{facts.ModelType}-composite",
                    Encoding = encoding,
                    Roles = new List<string> { "all" },
                    Exceptions = exceptions,
                },
        };
    }

    /// <summary>Judges whether the persisted rope facts are the served plain
    /// default rotary (PositionRope) or must be carried unresolved. The
    /// reference's rule is mirrored: a fact this build cannot serve is
    /// carried verbatim and refused by name, never approximated.</summary>
    private static PositionPolicy JudgePosition(TextArchitectureFacts facts)
    {
        var rope = facts.RopeParameters;
        bool hasMropeSections = rope.ValueKind == JsonValueKind.Object &&
                                rope.TryGetProperty("mrope_section", out var section) &&
                                section.ValueKind == JsonValueKind.Array;
        string? scaledType = rope.ValueKind == JsonValueKind.Object &&
                             rope.TryGetProperty("rope_type", out var rtype)
            ? rtype.GetString()
            : "default";

        if (!hasMropeSections && facts.PartialRotaryFactor >= 1.0 && scaledType == "default")
        {
            double theta = rope.ValueKind == JsonValueKind.Object &&
                           rope.TryGetProperty("rope_theta", out var t)
                ? t.GetDouble()
                : 10_000.0;
            return PositionPolicy.CreateRope(theta);
        }

        string kind = facts.PartialRotaryFactor < 1.0
            ? "partial_mrope"
            : hasMropeSections
                ? "mrope"
                : $"rope_scaling({scaledType})";
        return new PositionUnresolved { Kind = kind, Payload = facts.RopeParameters.Clone() };
    }

    /// <summary>Finds the tensor prefix the text decoder actually lives
    /// under ("model.language_model" for multimodal wrappers, "model" for
    /// bare text checkpoints). Never assumed.</summary>
    private static string DetectTextPrefix(HfInventory inventory)
    {
        foreach (var candidate in new[] { "model.language_model", "model", "language_model" })
        {
            if (inventory.CountUnder(candidate + ".layers.") > 0)
            {
                return candidate;
            }
        }
        throw new ModelConfigException(
            "no 'model.*.layers.N' decoder tensors found in the inventory — this build refuses to guess the text prefix");
    }

    private static string? EncodingFor(HfInventory inventory, string anyTensor)
    {
        return inventory.TryGet(anyTensor, out var info) ? info.Dtype.Label() : null;
    }

    private static LogicalObject TextObject(string id, ObjectKind kind, string prefix, string stem, string encoding)
    {
        var fullPrefix = $"{prefix}.{stem}";
        return new LogicalObject
        {
            Id = id,
            Component = "target",
            Kind = kind,
            SourceBindings = new List<SourceBinding>
            {
                new()
                {
                    Artifact = prefix,
                    TensorPrefix = fullPrefix,
                    Tensors = 1,
                    Bytes = 0,
                },
            },
            Representations = new List<Representation>
            {
                new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
            },
        };
    }

    /// <summary>Binds the layer tensors of the object to object-relative
    /// segment names: <c>layers.3.self_attn.q_proj.weight → 3.self_attn.q_proj.weight</c>.
    /// Payload bytes are copied verbatim (no widening at encode time).</summary>
    private static List<NamedTensorData> BindLayers(HfInventory inventory, string prefix)
    {
        var bound = new List<NamedTensorData>();
        var layersPrefix = $"{prefix}.layers.";
        foreach (var fullName in inventory.TensorNames)
        {
            if (!fullName.StartsWith(layersPrefix, StringComparison.Ordinal))
            {
                continue;
            }
            bound.Add(ToTensorData(inventory, fullName, fullName[layersPrefix.Length..]));
        }
        if (bound.Count == 0)
        {
            throw new ModelConfigException($"no tensors bound from '{layersPrefix}' — inventory mismatch");
        }
        return bound;
    }

    /// <summary>Binds one object-relative tensor:
    /// <c>embed_tokens.weight → weight</c>, <c>norm.weight → weight</c>.</summary>
    private static List<NamedTensorData> BindOne(HfInventory inventory, string prefix, string stem)
    {
        var fullName = $"{prefix}.{stem}";
        if (!inventory.TryGet(fullName, out _))
        {
            throw new ModelConfigException($"binding requires '{fullName}' but the inventory has no such tensor");
        }
        return new List<NamedTensorData> { ToTensorData(inventory, fullName, "weight") };
    }

    private static NamedTensorData ToTensorData(HfInventory inventory, string fullName, string relative)
    {
        var info = inventory.Get(fullName);
        return new NamedTensorData
        {
            Name = relative,
            Dtype = info.Dtype,
            Shape = info.Shape,
            Data = inventory.ReadBytes(fullName),
        };
    }

    private static RepresentationSpec Rep(string objectId, string encoding, List<NamedTensorData> tensors) => new()
    {
        ObjectId = objectId,
        Encoding = encoding,
        Tensors = tensors,
    };
}