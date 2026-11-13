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

        // ── the stack's tensors and the role each plays ──────────────────
        // Bound first: which norms, biases, gates and fusions exist is read
        // from the tensors themselves, and every later judgement leans on it.
        var conventions = FamilyConventions.For(facts.ModelType);
        var (stackTensors, fused) = BindLayers(inventory, prefix, facts);
        LayerRoleMap roles;
        try
        {
            roles = TensorRoles.Judge(stackTensors.Select(t => t.Name), fused);
        }
        catch (ContainerException e)
        {
            throw new ModelConfigException(e.Message, e);
        }
        bool parallel = TensorRoles.IsParallelBlock(roles);
        bool gatedFfn = roles.Name(TensorRoles.Gate) is not null || facts.Moe is not null;

        // ── per-layer policy table ────────────────────────────────────────
        // Position policy: plain or partial rope (and per-layer-type rope
        // blocks) are served; every other rope fact (MRoPE sections, scaled
        // families) is carried verbatim as unresolved and refused by the
        // planner naming the kind.
        DiffusionSurface? diffusion = JudgeDiffusion(facts);
        PositionPolicy position = JudgePosition(facts, conventions);
        bool anySliding = false;
        var policies = new List<AttentionLayerPolicy>(facts.NumLayers);
        for (int l = 0; l < facts.NumLayers; l++)
        {
            string layerType = facts.LayerTypes[l];
            bool linear = layerType == "linear_attention";
            // A sliding window without a per-layer table applies to every
            // layer (Mistral, Phi-3, Starcoder2) unless the config switches it
            // off (Qwen2/3 carry a size with use_sliding_window: false).
            bool sliding = layerType == "sliding_attention" ||
                           (layerType == "full_attention" && !facts.LayerTypesDeclared &&
                            facts.SlidingWindow is not null && facts.UseSlidingWindow != false);
            if (sliding && facts.SlidingWindow is null)
            {
                throw new ModelConfigException($"layer {l}: 'sliding_attention' declared but the config carries no sliding_window");
            }
            anySliding |= sliding;
            HeadGeometry geometry = linear && facts.LinearAttention is { } lin
                ? new HeadGeometry { HeadDim = lin.KeyHeadDim, NumKvHeads = lin.KeyHeads }
                : new HeadGeometry { HeadDim = facts.HeadDim, NumKvHeads = facts.NumKvHeads };
            policies.Add(new AttentionLayerPolicy
            {
                Operator = layerType switch
                {
                    "full_attention" or "sliding_attention" => LayerOperators.Softmax,
                    "linear_attention" => LayerOperators.LinearAttention,
                    "conv" => LayerOperators.Conv,
                    var other => throw new ModelConfigException(
                        $"layer {l}: layer_type '{other}' has no judged operator mapping"),
                },
                Span = sliding ? AttentionSpan.Sliding : AttentionSpan.Full,
                Window = sliding ? facts.SlidingWindow : null,
                Position = LayerPosition(facts, conventions, l, layerType, position),
                Geometry = geometry,
            });
        }
        if (!conventions.RopeOnFullLayers && anySliding)
        {
            // EXAONE 4: with sliding layers present, the full layers are NoPE.
            for (int l = 0; l < policies.Count; l++)
            {
                if (policies[l].Operator == LayerOperators.Softmax && policies[l].Span == AttentionSpan.Full)
                {
                    policies[l] = new AttentionLayerPolicy
                    {
                        Operator = policies[l].Operator,
                        Span = policies[l].Span,
                        Window = policies[l].Window,
                        Position = PositionPolicy.None,
                        Geometry = policies[l].Geometry,
                    };
                }
            }
        }

        // ── execution surface ────────────────────────────────────────────
        // Norm kind is read from the checkpoint: a norm that ships a bias, or
        // an epsilon spelled layer_norm_eps, is a LayerNorm. The affine
        // convention (w or 1+w) is the one fact no checkpoint states; it comes
        // from the family's conventions entry and defaults to plain w.
        bool layerNorm = roles.Roles.Keys.Any(k => k.EndsWith("_norm.bias", StringComparison.Ordinal)) ||
                         facts.NormEpsKey is "layer_norm_eps" or "layer_norm_epsilon" ||
                         inventory.TryGet($"{prefix}.norm.bias", out _);
        var normSpec = new NormSpec
        {
            Kind = layerNorm ? NormType.LayerNorm : NormType.RmsNorm,
            Eps = facts.RmsNormEps,
            WeightOffset = conventions.NormWeightOffset,
        };
        bool sandwich = roles.Name(TensorRoles.AttnOut) is not null || roles.Name(TensorRoles.FfnOut) is not null;
        var surface = new ExecutionSurface
        {
            ContextLength = facts.MaxPositionEmbeddings,
            Attention = new AttentionSurface
            {
                NumQHeads = facts.NumQueryHeads,
                NumKvHeads = facts.NumKvHeads,
                HeadDim = facts.HeadDim,
                // Granite replaces the usual 1/sqrt(head_dim) with an explicit
                // attention_multiplier; Gemma scales by query_pre_attn_scalar;
                // everything else gets the standard scale.
                ScoreScale = facts.AttentionMultiplier
                             ?? (facts.QueryPreAttnScalar is { } qpas ? 1.0 / Math.Sqrt(qpas) : 1.0 / Math.Sqrt(facts.HeadDim)),
                LogitSoftcapping = facts.AttnLogitSoftcapping is { } attnCap ? (float)attnCap : null,
                QkNormScope = QkNormScope.PerHead,
                QkNormWeightOffset = conventions.NormWeightOffset,
                AttentionBias = roles.Name(TensorRoles.Q, "bias") is not null ? true : facts.AttentionBias ? true : null,
                // LFM3 gates each head's output by sigmoid(gate_proj·x), one
                // scalar per head — a different operator from Qwen3.5's
                // elementwise gate folded into q_proj.
                OutputGate = facts.AttentionOutputGate
                    ? JsonSerializer.SerializeToElement(new { attn_output_gate = true }, ViJson.Options)
                    : HasPerHeadGate(inventory)
                        ? JsonSerializer.SerializeToElement(new { per_head_gate = true }, ViJson.Options)
                        : null,
                Bidirectional = diffusion is not null,
                RopeInterleaved = conventions.RopeInterleaved,
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = facts.IntermediateSize,
                Activation = MapActivation(facts.HiddenAct),
                FfnType = gatedFfn ? FfnType.Gated : FfnType.Dense,
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
                Placement = sandwich ? NormPlacement.Sandwich : NormPlacement.PreOnly,
            },
            Head = new HeadSurface
            {
                VocabSize = facts.VocabSize,
                HeadReusesEmbedding = facts.TieWordEmbeddings,
                // Granite's embedding_multiplier scales every looked-up row;
                // Gemma's convention is sqrt(hidden_size).
                EmbedScale = facts.EmbeddingMultiplier
                             ?? (conventions.EmbedScaleSqrtHidden ? Math.Sqrt(facts.HiddenSize) : null),
                // Granite divides the logits by logits_scaling; Cohere
                // multiplies them by logit_scale; the head op multiplies.
                OutputMultiplier = facts.LogitsScaling is { } scaling && scaling != 0.0
                    ? 1.0 / scaling
                    : facts.LogitScale,
                FinalLogitSoftcapping = facts.FinalLogitSoftcapping is { } finalCap ? (float)finalCap : null,
            },
            // Granite multiplies each branch by residual_multiplier at BOTH
            // decoder adds (h' = h + branch*rm).
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
            Roles = roles,
            ParallelResidual = parallel,
            Provenance = conventions.Provenance.Count > 0 ? conventions.Provenance : null,
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
            Rep("target.decoder_stack", encoding, stackTensors),
            Rep("target.final_norm", encoding, BindFinalNorm(inventory, prefix)),
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

    /// <summary>One layer's position policy. Conv layers carry none; SmolLM3's
    /// <c>no_rope_layers</c> switches individual layers to NoPE; a rope block
    /// keyed by the layer's own type (Gemma 3, LFM3: <c>rope_parameters.sliding_attention</c>)
    /// wins over the stack-wide one.</summary>
    private static PositionPolicy LayerPosition(TextArchitectureFacts facts, FamilyConventions conventions, int layer,
        string layerType, PositionPolicy stackWide)
    {
        if (layerType == "conv")
        {
            return PositionPolicy.None;
        }
        if (facts.NoRopeLayers is { } noRope && layer < noRope.Count && noRope[layer] == 0)
        {
            return PositionPolicy.None;
        }
        var rope = facts.RopeParameters;
        if (rope.ValueKind == JsonValueKind.Object && rope.TryGetProperty(layerType, out var block) &&
            block.ValueKind == JsonValueKind.Object)
        {
            return RopeBlock(block, block.TryGetProperty("partial_rotary_factor", out var f) ? f.GetDouble() : 1.0, conventions);
        }
        return stackWide;
    }

    /// <summary>Judges one rope block (<c>rope_type</c>, <c>rope_theta</c>) at
    /// the given partial factor. A factor below 1 is served as the family's
    /// partial-rotary convention.</summary>
    private static PositionPolicy RopeBlock(JsonElement block, double factor, FamilyConventions conventions)
    {
        string ropeType = block.TryGetProperty("rope_type", out var rt) ? rt.GetString() ?? "default" : "default";
        if (ropeType != "default")
        {
            return new PositionUnresolved { Kind = $"rope_scaling({ropeType})", Payload = block.Clone() };
        }
        double theta = block.TryGetProperty("rope_theta", out var t) ? t.GetDouble() : 10_000.0;
        if (factor < 1.0)
        {
            return conventions.ZeroTailPartialRotary
                ? new PositionZeroTailRope { Theta = theta, RotaryFactor = factor }
                : new PositionPartialRope { Theta = theta, RotaryFactor = factor };
        }
        return PositionPolicy.CreateRope(theta);
    }

    /// <summary>Judges whether the persisted rope facts are the served plain
    /// default rotary (PositionRope) or must be carried unresolved. The
    /// reference's rule is mirrored: a fact this build cannot serve is
    /// carried verbatim and refused by name, never approximated.</summary>
    private static PositionPolicy JudgePosition(TextArchitectureFacts facts, FamilyConventions conventions)
    {
        var rope = facts.RopeParameters;
        if (rope.ValueKind == JsonValueKind.Object && !rope.TryGetProperty("rope_type", out _) &&
            !rope.TryGetProperty("rope_theta", out _))
        {
            // Keyed by layer type only (each layer resolves its own block);
            // the stack-wide policy is never consulted for those layers.
            return PositionPolicy.None;
        }
        // Text-only MRoPE carries identical positions across the streams,
        // which collapses to standard rotary over the partial width.
        return RopeBlock(rope, facts.PartialRotaryFactor, conventions);
    }

    /// <summary>Judges the persisted <c>hidden_act</c> string against the
    /// known activation set. Unknown activations refuse by name.</summary>
    private static Activation MapActivation(string hiddenAct) => hiddenAct switch
    {
        "silu" or "swish" => Activation.Silu,
        "gelu" => Activation.Gelu,
        "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" or "gelu_tanh" => Activation.GeluTanh,
        "relu" => Activation.Relu,
        _ => throw new ModelConfigException(
            $"hidden_act '{hiddenAct}' is not a judged FFN activation — this build maps silu, gelu, gelu_pytorch_tanh and relu"),
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
    /// Payload bytes are copied verbatim (no widening at encode time).
    /// <para>A contiguous fusion (Phi-3's <c>qkv_proj</c>, GLM's
    /// <c>gate_up_proj</c>) is bound as its parts: each part is a byte range of
    /// the fused tensor (row-major, so a block of rows is contiguous), recorded
    /// in the returned list so export can concatenate it back.</para></summary>
    private static (List<NamedTensorData> Tensors, List<FusedTensor> Fused) BindLayers(
        HfInventory inventory, string prefix, TextArchitectureFacts facts)
    {
        var bound = new List<NamedTensorData>();
        var fused = new Dictionary<string, FusedTensor>(StringComparer.Ordinal);
        var layersPrefix = $"{prefix}.layers.";
        foreach (var fullName in inventory.TensorNames)
        {
            if (!fullName.StartsWith(layersPrefix, StringComparison.Ordinal))
            {
                continue;
            }
            string relative = fullName[layersPrefix.Length..];
            if (SplitFusion(inventory, fullName, relative, facts) is { } split)
            {
                bound.AddRange(split.Parts);
                fused.TryAdd(split.Fused.Name, split.Fused);
                continue;
            }
            bound.Add(ToTensorData(inventory, fullName, relative));
        }
        if (bound.Count == 0)
        {
            throw new ModelConfigException($"no tensors bound from '{layersPrefix}' — inventory mismatch");
        }
        return (bound, fused.Values.ToList());
    }

    /// <summary>Splits a contiguous fusion into its role tensors, or returns null
    /// when <paramref name="relative"/> (<c>3.mlp.gate_up_proj.weight</c>) is not one.</summary>
    private static (List<NamedTensorData> Parts, FusedTensor Fused)? SplitFusion(
        HfInventory inventory, string fullName, string relative, TextArchitectureFacts facts)
    {
        int dot = relative.IndexOf('.');
        string suffix = relative[(dot + 1)..];
        int lastDot = suffix.LastIndexOf('.');
        string stem = suffix[..lastDot], part = suffix[(lastDot + 1)..];
        if (!TensorRoles.ContiguousFusions.TryGetValue(stem, out var partStems))
        {
            return null;
        }
        var info = inventory.Get(fullName);
        long rows = info.Shape[0];
        long[] partRows = partStems.Length == 3
            ? new long[] { (long)facts.NumQueryHeads * facts.HeadDim, (long)facts.NumKvHeads * facts.HeadDim, (long)facts.NumKvHeads * facts.HeadDim }
            : new long[] { rows / 2, rows / 2 };
        if (partRows.Sum() != rows || info.DataLength % rows != 0)
        {
            throw new ModelConfigException(
                $"'{fullName}' has {rows} rows but its parts ({string.Join(" + ", partStems)}) need {partRows.Sum()} — refusing to guess the split");
        }
        long bytesPerRow = info.DataLength / rows;
        string layer = relative[..dot];
        var parts = new List<NamedTensorData>();
        long rowStart = 0;
        for (int i = 0; i < partStems.Length; i++)
        {
            long byteStart = rowStart * bytesPerRow;
            var shape = (long[])info.Shape.Clone();
            shape[0] = partRows[i];
            parts.Add(new NamedTensorData
            {
                Name = $"{layer}.{partStems[i]}.{part}",
                Dtype = info.Dtype,
                Shape = shape,
                Source = (offset, count) => inventory.ReadBytes(fullName, byteStart + offset, count),
                SourceLength = partRows[i] * bytesPerRow,
            });
            rowStart += partRows[i];
        }
        return (parts, new FusedTensor
        {
            Name = $"{stem}.{part}",
            Parts = partStems.Select(s => $"{s}.{part}").ToList(),
        });
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

    /// <summary>The final norm's weight — and its bias, for a LayerNorm that
    /// ships one — as the object-relative <c>weight</c> / <c>bias</c>.</summary>
    private static List<NamedTensorData> BindFinalNorm(HfInventory inventory, string prefix)
    {
        string stem = inventory.TryGet($"{prefix}.embedding_norm.weight", out _) ? "embedding_norm" : "norm";
        var bound = BindOne(inventory, prefix, $"{stem}.weight");
        if (inventory.TryGet($"{prefix}.{stem}.bias", out _))
        {
            bound.Add(ToTensorData(inventory, $"{prefix}.{stem}.bias", "bias"));
        }
        return bound;
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