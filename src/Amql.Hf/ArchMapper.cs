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
public static partial class ArchMapper
{
    public sealed record EncodeOptions
    {
        public bool IncludeVision { get; init; } = true;
        public bool IncludeMtp { get; init; } = true;
    }

    public static ContainerSpec MapToContainerSpec(string modelId, TextArchitectureFacts facts, HfInventory inventory, EncodeOptions options,
        ClassificationFacts? classification = null)
    {
        // ModernBERT keeps its layers at the checkpoint root ("layers.N"),
        // which the text-prefix probe below would reject.
        if (facts.ModelType == ModernBert.ModelType)
        {
            return MapModernBert(modelId, facts, inventory);
        }

        bool isNomicBert = facts.ModelType == "nomic_bert";
        string prefix = DetectTextPrefix(inventory);
        if (facts.LayerTypes.Count != facts.NumLayers)
        {
            throw new ModelConfigException(
                $"layer_types declares {facts.LayerTypes.Count} layers but num_hidden_layers is {facts.NumLayers}");
        }
        // Activation is validated uniformally inside MapActivation.

        if (isNomicBert)
        {
            return MapNomicBert(modelId, facts, inventory, options, classification, prefix);
        }

        // ── per-layer policy table ────────────────────────────────────────
        // Position policy: plain default rope (no MRoPE sections, full
        // rotary factor, no frequency scaling) is SERVED as a standard
        // PositionRope; every other rope fact (partial factor, MRoPE
        // sections, scaled families) is carried verbatim as unresolved and
        // refused by the planner naming the kind.
        bool isLfm3 = facts.ModelType == "lfm3";
        DiffusionSurface? diffusion = JudgeDiffusion(facts);
        PositionPolicy position = isLfm3 ? PositionPolicy.None : JudgePosition(facts);
        var policies = new List<AttentionLayerPolicy>(facts.NumLayers);
        for (int l = 0; l < facts.NumLayers; l++)
        {
            bool linear = facts.LayerTypes[l] == "linear_attention";
            bool sliding = facts.LayerTypes[l] == "sliding_attention";
            if (sliding && !isLfm3)
            {
                throw new ModelConfigException(
                    $"layer {l}: layer_type 'sliding_attention' is judged for lfm3 only (model_type '{facts.ModelType}')");
            }
            if (sliding && facts.SlidingWindow is null)
            {
                throw new ModelConfigException($"layer {l}: 'sliding_attention' declared but the config carries no sliding_window");
            }
            HeadGeometry geometry = linear && facts.LinearAttention is { } lin
                ? new HeadGeometry { HeadDim = lin.KeyHeadDim, NumKvHeads = lin.KeyHeads }
                : new HeadGeometry { HeadDim = facts.HeadDim, NumKvHeads = facts.NumKvHeads };
            policies.Add(new AttentionLayerPolicy
            {
                Operator = facts.LayerTypes[l] switch
                {
                    "full_attention" or "sliding_attention" => LayerOperators.Softmax,
                    "linear_attention" => LayerOperators.LinearAttention,
                    "conv" => LayerOperators.Conv,
                    var other => throw new ModelConfigException(
                        $"layer {l}: layer_type '{other}' has no judged operator mapping"),
                },
                Span = sliding ? AttentionSpan.Sliding : AttentionSpan.Full,
                Window = sliding ? facts.SlidingWindow : null,
                Position = isLfm3 ? JudgeLfm3Position(facts, facts.LayerTypes[l]) : position,
                Geometry = geometry,
            });
        }

        // ── execution surface ────────────────────────────────────────────
        // The 1+w affine convention is an architecture fact: Qwen3.5's
        // RMSNorm (layer norms AND the per-head QK norms) applies
        // RMS(x)·(1+w), i.e. weightOffset 1.0. The linear layers' internal
        // z-gated norm multiplies the weight directly (offset 0) — that is
        // an operator fact, not a surface fact.
        var normSpec = new NormSpec
        {
            Kind = NormType.RmsNorm,
            Eps = facts.RmsNormEps,
            WeightOffset = NormWeightOffsetFor(facts.ModelType),
        };
        var surface = new ExecutionSurface
        {
            ContextLength = facts.MaxPositionEmbeddings,
            Attention = new AttentionSurface
            {
                NumQHeads = facts.NumQueryHeads,
                NumKvHeads = facts.NumKvHeads,
                HeadDim = facts.HeadDim,
                // Granite replaces the usual 1/sqrt(head_dim) with an explicit
                // attention_multiplier; families that do not declare one get the
                // standard scale. Reading it here rather than defaulting is what
                // keeps a Granite import numerically faithful.
                ScoreScale = facts.AttentionMultiplier ?? 1.0 / Math.Sqrt(facts.HeadDim),
                QkNormScope = QkNormScope.PerHead,
                QkNormWeightOffset = NormWeightOffsetFor(facts.ModelType),
                AttentionBias = facts.AttentionBias,
                // LFM3 gates each head's output by sigmoid(gate_proj·x), one
                // scalar per head — a different operator from Qwen3.5's
                // elementwise gate folded into q_proj.
                OutputGate = facts.AttentionOutputGate
                    ? JsonSerializer.SerializeToElement(new { attn_output_gate = true }, ViJson.Options)
                    : HasPerHeadGate(inventory)
                        ? JsonSerializer.SerializeToElement(new { per_head_gate = true }, ViJson.Options)
                        : null,
                Bidirectional = diffusion is not null,
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = facts.IntermediateSize,
                Activation = MapActivation(facts.HiddenAct),
                FfnType = FfnType.Gated,
                Moe = facts.Moe is { } moe
                    ? new MoeSurface
                    {
                        Experts = moe.Experts,
                        TopK = moe.TopK,
                        ExpertIntermediateSize = moe.ExpertIntermediateSize,
                        RoutingPolicy = moe.RoutingPolicy,
                    }
                    : null,
            },
            Norm = new NormSurface
            {
                Pre = normSpec,
                Post = normSpec,
                FinalNorm = normSpec,
                Placement = facts.UseSandwichNorm ? NormPlacement.Sandwich : NormPlacement.PreOnly,
            },
            Head = new HeadSurface
            {
                VocabSize = facts.VocabSize,
                HeadReusesEmbedding = facts.TieWordEmbeddings,
                // Granite's embedding_multiplier scales every looked-up row; the
                // runtime applies EmbeddingOp.Scale, so carrying it here is what
                // makes the knob live rather than decorative.
                EmbedScale = facts.EmbeddingMultiplier,
                // Granite divides the logits by logits_scaling; the head op
                // multiplies, so the two are reciprocals.
                OutputMultiplier = facts.LogitsScaling is { } scaling && scaling != 0.0
                    ? 1.0 / scaling
                    : null,
            },
            // Granite multiplies the incoming stream by residual_multiplier at
            // BOTH decoder adds (h' = h*rm + branch); the runtime scales the
            // residual before each AddInPlace when this is set.
            ResidualScale = facts.ResidualMultiplier,
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
            Classifier = classification is { HasScoreTensor: true }
                ? new ClassifierSurface
                {
                    NumLabels = classification.NumLabels,
                    ProblemType = classification.ProblemType,
                    Pooling = new PoolingSurface
                    {
                        Kind = PoolingKind.Last,
                        LastNonPad = true,
                        MaskPadding = true,
                    },
                    Template = classification.Template,
                    Labels = classification.Labels,
                    ScoreBias = classification.BiasTensor is not null,
                    PadTokenId = classification.PadTokenId,
                }
                : null,
            Diffusion = diffusion,
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
        // Qwen3.5 ships the output head either tied to the embedding (no
        // lm_head tensor) or — as Qwen3.8 does — untied, with the 248k-row
        // lm_head named at the TOP level of the wrapper checkpoint, not
        // under the text prefix.
        bool untied = !facts.TieWordEmbeddings;
        var objects = new List<LogicalObject>
        {
            TextObject("target.embedding", ObjectKind.Embedding, prefix, "embed_tokens", encoding),
            TextObject("target.decoder_stack", ObjectKind.DecoderStack, prefix, "layers", encoding),
            // LFM2/LFM3 name the final norm "embedding_norm"; the binding has
            // to carry the real stem or export rebuilds a name nothing loads.
            TextObject("target.final_norm", ObjectKind.FinalNorm, prefix,
                inventory.TryGet($"{prefix}.embedding_norm.weight", out _) ? "embedding_norm" : "norm", encoding),
            new()
            {
                Id = "target.output_head",
                Component = "target",
                Kind = ObjectKind.OutputHead,
                SourceBindings = untied
                    ? new List<SourceBinding>
                    {
                        new() { Artifact = prefix, TensorPrefix = "lm_head", Tensors = 1, Bytes = 0 },
                    }
                    : new List<SourceBinding>(),
                Representations = untied
                    ? new List<Representation>
                    {
                        new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                    }
                    : new List<Representation>(), // tied: no dedicated segment
            },
        };

        // Classifier head: a separate score linear (not a vocabulary head).
        if (classification is { HasScoreTensor: true })
        {
            long scoreWidth = inventory.Get(classification.WeightTensor).Shape[1];
            if (scoreWidth != facts.HiddenSize)
            {
                throw new ModelConfigException(
                    $"'{classification.WeightTensor}' reads {scoreWidth} features but the stack's hidden size is {facts.HiddenSize}");
            }
            // The object's prefix is the checkpoint's own ("score" or
            // "model.score"), so export rebuilds the exact tensor names.
            string scorePrefix = classification.WeightTensor[..^".weight".Length];
            objects.Add(new LogicalObject
            {
                Id = "target.classifier_head",
                Component = "target",
                Kind = ObjectKind.ClassifierHead,
                SourceBindings = new List<SourceBinding>
                {
                    new()
                    {
                        Artifact = prefix,
                        TensorPrefix = scorePrefix,
                        Tensors = 1,
                        Bytes = 0,
                    },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                },
            });
        }

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

        // ── vision tower prefix detection ────────────────────────────────────
        // Qwen uses "model.visual.", LFM2.5-VL (SigLIP2) uses
        // "model.vision_tower.vision_model.". Detect which exists.
        // LFM3-VL keeps its own tower directly under "model.vision_tower.".
        string visionPrefix = inventory.CountUnder("model.visual.") > 0
            ? "model.visual"
            : inventory.CountUnder("model.vision_tower.vision_model.") > 0
                ? "model.vision_tower.vision_model"
                : inventory.CountUnder("model.vision_tower.") > 0
                    ? "model.vision_tower"
                    : "model.visual"; // fallback

        if (options.IncludeVision)
        {
            int visionTensors = inventory.CountUnder(visionPrefix + ".");
            bool visionMaterialised = visionTensors > 0;
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
                        TensorPrefix = visionPrefix,
                        Tensors = visionTensors,
                        Bytes = inventory.BytesUnder(visionPrefix + "."),
                    },
                },
                // The tower is part of the model: with tensors in the source
                // it is materialised so export can include it; without them
                // the object stays carried.
                Representations = visionMaterialised
                    ? new List<Representation>
                    {
                        new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                    }
                    : new List<Representation>(),
            });
            components.Add(new Component
            {
                Id = "vision",
                Role = ComponentRole.Perception,
                SourceArtifact = visionPrefix,
                NumLayers = facts.Vision?.NumLayers ?? 12,
                HiddenSize = facts.Vision?.HiddenSize ?? 768,
                // The wrapper config (vision tower, projector, media tokens)
                // rides along verbatim so a multimodal export can rebuild it.
                Perception = facts.WrapperConfig is { } wrapper
                    ? JsonSerializer.SerializeToElement(
                        new { modality = "image", transform = new { kind = "encoder" }, hf_wrapper = wrapper }, ViJson.Options)
                    : JsonSerializer.SerializeToElement(
                        new { modality = "image", transform = new { kind = "encoder" } }, ViJson.Options),
            });
        }

        // ── multi-modal projector (LFM2.5-VL, Qwen-VL, etc.) ────────────
        if (options.IncludeVision)
        {
            const string projectorPrefix = "model.multi_modal_projector";
            int projectorTensors = inventory.CountUnder(projectorPrefix + ".");
            if (projectorTensors > 0)
            {
                objects.Add(new LogicalObject
                {
                    Id = "vision.projector",
                    Component = "vision",
                    Kind = ObjectKind.FeatureProjector,
                    SourceBindings = new List<SourceBinding>
                    {
                        new()
                        {
                            Artifact = projectorPrefix,
                            TensorPrefix = projectorPrefix,
                            Tensors = projectorTensors,
                            Bytes = inventory.BytesUnder(projectorPrefix + "."),
                        },
                    },
                    Representations = new List<Representation>
                    {
                        new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                    },
                });
            }
        }

        if (options.IncludeMtp)
        {
            const string mtpPrefix = "mtp";
            int mtpTensors = inventory.CountUnder(mtpPrefix + ".");
            bool mtpMaterialised = mtpTensors > 0;
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
                        TensorPrefix = mtpPrefix,
                        Tensors = mtpTensors,
                        Bytes = inventory.BytesUnder(mtpPrefix + "."),
                    },
                },
                // With mtp tensors in the checkpoint, the drafter's module
                // is materialised into its own segment (the container
                // becomes self-contained for export-mtp). Without them, the
                // object stays carried: nothing to bind.
                Representations = mtpMaterialised
                    ? new List<Representation>
                    {
                        new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                    }
                    : new List<Representation>(),
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
                TryBindOne(inventory, prefix, "embedding_norm.weight")
                ?? BindOne(inventory, prefix, "norm.weight")),
        };
        if (untied)
        {
            reps.Add(Rep("target.output_head", encoding, BindOutputHead(inventory, prefix)));
        }
        if (classification is { HasScoreTensor: true })
        {
            reps.Add(Rep("target.classifier_head", encoding, BindScoreHead(inventory, classification)));
        }
        if (options.IncludeMtp && inventory.CountUnder("mtp.") > 0)
        {
            reps.Add(Rep("mtp.stack", encoding, BindUnder(inventory, "mtp.")));
        }
        if (options.IncludeVision && inventory.CountUnder(visionPrefix + ".") > 0)
        {
            reps.Add(Rep("vision.perception_tower", encoding, BindUnder(inventory, visionPrefix + ".")));
        }
        if (options.IncludeVision && inventory.CountUnder("model.multi_modal_projector.") > 0)
        {
            reps.Add(Rep("vision.projector", encoding, BindUnder(inventory, "model.multi_modal_projector.")));
        }

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

    /// <summary>The RMSNorm affine convention of a text family: Qwen3.5
    /// applies RMS(x)·(1+w) to its layer norms and per-head QK norms;
    /// other families this build maps default to plain RMS(x)·w.</summary>
    private static float NormWeightOffsetFor(string modelType) =>
        modelType is "qwen3_5_text" or "qwen3_5" ? 1.0f : 0f;

    /// <summary>dLLM's A2D ("autoregressive to diffusion") checkpoints are the
    /// base family's weights and config under <c>a2d-&lt;family&gt;</c>, with the
    /// causal mask replaced by a padding-only one and MDLM training on top.
    /// Only the Qwen3 conversion is judged: the published modeling code for it
    /// changes nothing but the mask, and its logits are read unshifted. Any other
    /// <c>a2d-*</c> refuses rather than run as its causal base.</summary>
    private static DiffusionSurface? JudgeDiffusion(TextArchitectureFacts facts)
    {
        if (!facts.ModelType.StartsWith("a2d-", StringComparison.Ordinal))
        {
            return null;
        }
        if (facts.ModelType != "a2d-qwen3")
        {
            throw new ModelConfigException(
                $"model_type '{facts.ModelType}' is an A2D diffusion conversion this build has not judged (only a2d-qwen3)");
        }
        if (facts.MaskTokenId is not { } mask)
        {
            throw new ModelConfigException(
                "a2d-qwen3 is a masked diffusion model but the checkpoint's tokenizer declares no mask_token with an id");
        }
        if (facts.SlidingWindow is not null && facts.LayerTypes.Any(t => t != "full_attention"))
        {
            throw new ModelConfigException("a2d-qwen3 with sliding-window layers is not judged");
        }
        return new DiffusionSurface
        {
            Objective = "masked_diffusion",
            MaskTokenId = mask,
            ModelType = facts.ModelType,
            Architecture = facts.Architecture ?? "A2DQwen3LMHeadModel",
            AutoMap = facts.AutoMap,
        };
    }

    /// <summary>Whether the attention layers carry LFM3's per-head output
    /// gate (<c>self_attn.gate_proj</c>, [heads, hidden]).</summary>
    private static bool HasPerHeadGate(HfInventory inventory) =>
        inventory.TensorNames.Any(n => n.EndsWith(".self_attn.gate_proj.weight", StringComparison.Ordinal));

    /// <summary>LFM3 keys its rope facts by layer type. The full-attention
    /// layers use the zero-tail partial rotary (the progression of the full
    /// head dim, trailing frequencies zeroed); a factor of 1 is plain rope.</summary>
    private static PositionPolicy JudgeLfm3Position(TextArchitectureFacts facts, string layerType)
    {
        if (layerType == "conv")
        {
            return PositionPolicy.None;
        }
        var rope = facts.RopeParameters;
        if (rope.ValueKind != JsonValueKind.Object || !rope.TryGetProperty(layerType, out var p) ||
            p.ValueKind != JsonValueKind.Object)
        {
            throw new ModelConfigException($"lfm3: rope_parameters carries no '{layerType}' entry");
        }
        string ropeType = p.TryGetProperty("rope_type", out var rt) ? rt.GetString() ?? "default" : "default";
        if (ropeType != "default")
        {
            return new PositionUnresolved { Kind = $"rope_scaling({ropeType})", Payload = p.Clone() };
        }
        double theta = p.TryGetProperty("rope_theta", out var t) ? t.GetDouble() : 10_000.0;
        double factor = p.TryGetProperty("partial_rotary_factor", out var f) ? f.GetDouble() : 1.0;
        return factor < 1.0
            ? new PositionZeroTailRope { Theta = theta, RotaryFactor = factor }
            : PositionPolicy.CreateRope(theta);
    }

    /// <summary>Judges whether the persisted rope facts are the served plain
    /// default rotary (PositionRope) or must be carried unresolved. The
    /// reference's rule is mirrored: a fact this build cannot serve is
    /// carried verbatim and refused by name, never approximated.</summary>
    private static PositionPolicy JudgePosition(TextArchitectureFacts facts)
    {
        var rope = facts.RopeParameters;
        string? scaledType = rope.ValueKind == JsonValueKind.Object &&
                             rope.TryGetProperty("rope_type", out var rtype)
            ? rtype.GetString()
            : "default";

        if (scaledType == "default")
        {
            double theta = rope.ValueKind == JsonValueKind.Object &&
                           rope.TryGetProperty("rope_theta", out var t)
                ? t.GetDouble()
                : 10_000.0;

            // Text-only MRoPE carries identical positions across the
            // streams, which collapses to standard rotary over the partial
            // width — served as PositionPartialRope. A partial factor is
            // exactly that; full factor is plain PositionRope.
            if (facts.PartialRotaryFactor < 1.0)
            {
                return new PositionPartialRope { Theta = theta, RotaryFactor = facts.PartialRotaryFactor };
            }
            return PositionPolicy.CreateRope(theta);
        }

        return new PositionUnresolved
        {
            Kind = $"rope_scaling({scaledType})",
            Payload = facts.RopeParameters.Clone(),
        };
    }

    /// <summary>Judges the persisted <c>hidden_act</c> string against the
    /// known activation set. Unknown activations refuse by name.</summary>
    private static Activation MapActivation(string hiddenAct) => hiddenAct switch
    {
        "silu" => Activation.Silu,
        "gelu" => Activation.Gelu,
        _ => throw new ModelConfigException(
            $"hidden_act '{hiddenAct}' is not a judged FFN activation — this build maps 'silu' and 'gelu' only"),
    };

    /// <summary>Finds the tensor prefix the text decoder actually lives
    /// under ("model.language_model" for multimodal wrappers, "model" for
    /// bare text checkpoints). Never assumed.</summary>
    private static string DetectTextPrefix(HfInventory inventory)
    {
        foreach (var candidate in new[] { "model.language_model", "model", "language_model", "encoder" })
        {
            if (inventory.CountUnder(candidate + ".layers.") > 0)
            {
                return candidate;
            }
        }
        // nomic-bert uses "encoder.layers." without a traditional prefix
        if (inventory.CountUnder("encoder.layers.") > 0)
        {
            return "encoder";
        }
        throw new ModelConfigException(
            "no '*.*.layers.N' decoder tensors found in the inventory — this build refuses to guess the text prefix");
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
        return TryBindOne(inventory, prefix, stem)
            ?? throw new ModelConfigException($"binding requires '{prefix}.{stem}' but the inventory has no such tensor");
    }

    /// <summary>Tries one object-relative tensor, returns null if not found.</summary>
    private static List<NamedTensorData>? TryBindOne(HfInventory inventory, string prefix, string stem)
    {
        var fullName = $"{prefix}.{stem}";
        if (!inventory.TryGet(fullName, out _))
        {
            return null;
        }
        return new List<NamedTensorData> { ToTensorData(inventory, fullName, "weight") };
    }

    /// <summary>Binds an untied output head. The Qwen3.5 multimodal wrapper
    /// ships <c>lm_head.weight</c> at the TOP level of the checkpoint (no
    /// text prefix); the in-prefix spelling covers bare-text checkpoints.</summary>
    private static List<NamedTensorData> BindOutputHead(HfInventory inventory, string prefix)
    {
        foreach (var fullName in new[] { "lm_head.weight", $"{prefix}.lm_head.weight" })
        {
            if (inventory.TryGet(fullName, out _))
            {
                return new List<NamedTensorData> { ToTensorData(inventory, fullName, "weight") };
            }
        }
        throw new ModelConfigException(
            "tie_word_embeddings is false but neither 'lm_head.weight' nor '<prefix>.lm_head.weight' is in the inventory");
    }

    /// <summary>Binds the classifier score head — its weight and, when the
    /// checkpoint ships one, its bias — as the object-relative tensors
    /// <c>weight</c> / <c>bias</c> of the ClassifierHead object.</summary>
    private static List<NamedTensorData> BindScoreHead(HfInventory inventory, ClassificationFacts classification)
    {
        var bound = new List<NamedTensorData> { ToTensorData(inventory, classification.WeightTensor, "weight") };
        if (classification.BiasTensor is { } bias)
        {
            bound.Add(ToTensorData(inventory, bias, "bias"));
        }
        return bound;
    }

    /// <summary>Binds a carried module (the MTP drafter's <c>mtp.</c> stem or
    /// the vision tower's <c>model.visual.</c> stem) verbatim: every tensor
    /// under the stem, as object-relative names — the stem IS the binding's
    /// tensor prefix, so export rebuilds the original checkpoint names.</summary>
    private static List<NamedTensorData> BindUnder(HfInventory inventory, string stem) =>
        inventory.TensorNames
            .Where(n => n.StartsWith(stem, StringComparison.Ordinal))
            .Select(n => ToTensorData(inventory, n, n[stem.Length..]))
            .ToList();

    private static NamedTensorData ToTensorData(HfInventory inventory, string fullName, string relative)
    {
        var info = inventory.Get(fullName);
        // Read from the checkpoint's memory map only when the segment is
        // written, a chunk at a time: encoding a checkpoint never holds more
        // than one chunk of it, however large the model.
        return new NamedTensorData
        {
            Name = relative,
            Dtype = info.Dtype,
            Shape = info.Shape,
            Source = (offset, count) => inventory.ReadBytes(fullName, offset, count),
            SourceLength = info.DataLength,
        };
    }

    private static RepresentationSpec Rep(string objectId, string encoding, List<NamedTensorData> tensors) => new()
    {
        ObjectId = objectId,
        Encoding = encoding,
        Tensors = tensors,
    };

    // ── nomic-bert encoder path ─────────────────────────────────────────

    private static ContainerSpec MapNomicBert(string modelId, TextArchitectureFacts facts, HfInventory inventory,
        EncodeOptions options, ClassificationFacts? classification, string prefix)
    {
        if (classification is not null)
        {
            // This path binds no ClassifierHead object, so accepting the
            // checkpoint would drop its score weights.
            throw new ModelConfigException(
                "a nomic-bert checkpoint with a sequence-classification head is not judged by this build — " +
                "refusing rather than dropping the head");
        }
        string encoding = EncodingFor(inventory, "encoder.layers.0.attn.Wqkv.weight") ??
                          EncodingFor(inventory, "embeddings.word_embeddings.weight") ??
                          "F32";
        Dtype encodingDtype = DtypeExtensions.FromLabel(encoding);

        // Per-layer policies: all layers are full bidirectional attention.
        var position = new PositionRope { Theta = 1000.0 };
        var policies = new List<AttentionLayerPolicy>(facts.NumLayers);
        for (int l = 0; l < facts.NumLayers; l++)
        {
            policies.Add(new AttentionLayerPolicy
            {
                Operator = LayerOperators.Softmax,
                Span = AttentionSpan.Full,
                Position = position,
                Geometry = new HeadGeometry { HeadDim = facts.HeadDim, NumKvHeads = facts.NumKvHeads },
            });
        }

        // Post-LayerNorm with bias (nomic-bert uses weight+bias LayerNorm
        // after the residual, not before).
        var normSpec = new NormSpec
        {
            Kind = NormType.LayerNorm,
            Eps = facts.RmsNormEps,
            WeightOffset = 0f,
        };
        var surface = new ExecutionSurface
        {
            ContextLength = facts.MaxPositionEmbeddings,
            Attention = new AttentionSurface
            {
                NumQHeads = facts.NumQueryHeads,
                NumKvHeads = facts.NumKvHeads,
                HeadDim = facts.HeadDim,
                // Granite replaces the usual 1/sqrt(head_dim) with an explicit
                // attention_multiplier; families that do not declare one get the
                // standard scale. Reading it here rather than defaulting is what
                // keeps a Granite import numerically faithful.
                ScoreScale = facts.AttentionMultiplier ?? 1.0 / Math.Sqrt(facts.HeadDim),
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = facts.IntermediateSize,
                Activation = MapActivation(facts.HiddenAct),
                FfnType = FfnType.Gated,
            },
            Norm = new NormSurface
            {
                Pre = normSpec,
                Post = normSpec,
                FinalNorm = normSpec,
                Placement = NormPlacement.PrePost,
            },
            Head = new HeadSurface
            {
                VocabSize = facts.VocabSize,
                HeadReusesEmbedding = true,
            },
        };

        // Logical objects: EncoderStack for the bidirectional layers,
        // with source bindings to the nomic-bert tensor names so export
        // rebuilds the original checkpoint.
        var objects = new List<LogicalObject>
        {
            new()
            {
                Id = "target.embedding",
                Component = "target",
                Kind = ObjectKind.Embedding,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "embeddings", TensorPrefix = "embeddings", Tensors = 1, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                },
            },
            new()
            {
                Id = "target.encoder_stack",
                Component = "target",
                Kind = ObjectKind.EncoderStack,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "encoder.layers", TensorPrefix = "encoder.layers", Tensors = 0, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                },
            },
            new()
            {
                Id = "target.final_norm",
                Component = "target",
                Kind = ObjectKind.FinalNorm,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "emb_ln", TensorPrefix = "emb_ln", Tensors = 1, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                },
            },
        };

        var components = new List<Component>
        {
            new()
            {
                Id = "target",
                Role = ComponentRole.PrimaryText,
                SourceArtifact = "encoder.layers",
                NumLayers = facts.NumLayers,
                HiddenSize = facts.HiddenSize,
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

        // Representation specs: bind the actual checkpoint tensors.
        var reps = new List<RepresentationSpec>
        {
            Rep("target.embedding", encoding, BindNomicEmbedding(inventory)),
            Rep("target.encoder_stack", encoding, BindNomicEncoder(inventory)),
            Rep("target.final_norm", encoding, BindNomicFinalNorm(inventory)),
        };

        return new ContainerSpec
        {
            Model = modelId,
            Family = facts.ModelType,
            HiddenSize = facts.HiddenSize,
            NumLayers = facts.NumLayers,
            SystemGraph = graph,
            Representations = reps,
        };
    }

    private static List<NamedTensorData> BindNomicEmbedding(HfInventory inventory)
    {
        var bound = new List<NamedTensorData>();
        foreach (var name in new[] { "word_embeddings.weight", "token_type_embeddings.weight" })
        {
            var fullName = $"embeddings.{name}";
            if (inventory.TryGet(fullName, out _))
            {
                bound.Add(ToTensorData(inventory, fullName, name));
            }
        }
        if (bound.Count == 0)
        {
            throw new ModelConfigException("no embedding tensors found under 'embeddings.'");
        }
        return bound;
    }

    private static List<NamedTensorData> BindNomicEncoder(HfInventory inventory)
    {
        var bound = new List<NamedTensorData>();
        var prefix = "encoder.layers.";
        foreach (var fullName in inventory.TensorNames)
        {
            if (fullName.StartsWith(prefix, StringComparison.Ordinal))
            {
                bound.Add(ToTensorData(inventory, fullName, fullName[prefix.Length..]));
            }
        }
        if (bound.Count == 0)
        {
            throw new ModelConfigException("no encoder tensors found under 'encoder.layers.'");
        }
        return bound;
    }

    private static List<NamedTensorData> BindNomicFinalNorm(HfInventory inventory)
    {
        var bound = new List<NamedTensorData>();
        foreach (var name in new[] { "weight", "bias" })
        {
            var fullName = $"emb_ln.{name}";
            if (inventory.TryGet(fullName, out _))
            {
                bound.Add(ToTensorData(inventory, fullName, name));
            }
        }
        if (bound.Count == 0)
        {
            throw new ModelConfigException("no final norm tensors found under 'emb_ln.'");
        }
        return bound;
    }
}