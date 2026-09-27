using System.Text.Json;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Hf;

/// <summary>ModernBERT facts that the shared <see cref="TextArchitectureFacts"/>
/// has no field for: the two RoPE bases, the local window, the bias switches
/// and the special-token ids.</summary>
public sealed record ModernBertFacts(
    double GlobalRopeTheta,
    double LocalRopeTheta,
    int LocalAttention,
    bool NormBias,
    bool MlpBias,
    bool AttentionBias,
    IReadOnlyDictionary<string, int> SpecialTokenIds);

/// <summary>What a Von <c>option_marker.pt</c> contributes on top of the
/// backbone: the scoring head's tensors and the run-time switches from
/// <c>marker_calibration.json</c>.</summary>
public sealed record OptionMarkerFacts(
    string CheckpointPath,
    int ScorerHiddenSize,
    bool IndependentOptions,
    bool DigitSplit);

/// <summary>
/// ModernBERT (<c>model_type: modernbert</c>) and the Von decision model built
/// on it (<see href="https://huggingface.co/wfzyx/von"/>).
/// <para>
/// Von ships two weight files. <c>model.safetensors</c> is the ModernBERT
/// backbone in bare <c>ModernBertModel</c> naming; <c>option_marker.pt</c> is
/// the full <c>OptionMarkerModel</c> state dict — the same backbone under
/// <c>encoder.</c> plus the eight-tensor scoring head under <c>scorer.</c> —
/// which the Von SDK loads with <c>strict=True</c> over the backbone. The
/// <c>.pt</c> therefore wins if the two ever disagree, so the import checks
/// that its <c>encoder.</c> copy is byte-identical to the safetensors backbone
/// and refuses otherwise, rather than silently importing weights Von does not
/// run.
/// </para>
/// </summary>
public static class ModernBert
{
    public const string ModelType = "modernbert";
    public const string Architecture = "ModernBertModel";
    public const string OptionMarkerFile = "option_marker.pt";
    public const string CalibrationFile = "marker_calibration.json";
    public const string ScorerPrefix = "scorer";
    public const string EncoderPrefixInPt = "encoder.";
    public const string MarkerToken = "[MASK]";

    /// <summary>The scoring head's tensors and the rank each must have.</summary>
    public static readonly IReadOnlyList<(string Name, int Rank)> ScorerTensors = new[]
    {
        ("input_norm.weight", 1), ("input_norm.bias", 1),
        ("dense.weight", 2), ("dense.bias", 1),
        ("norm.weight", 1), ("norm.bias", 1),
        ("out_proj.weight", 2), ("out_proj.bias", 1),
    };

    public static readonly IReadOnlyList<string> SpecialTokenKeys = new[]
    {
        "pad_token_id", "bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id",
    };

    public static ModernBertFacts ReadFacts(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = doc.RootElement;

        // transformers 5 writes per-layer-type rope_parameters; 4.x wrote
        // global_rope_theta / local_rope_theta. Either is authoritative, and
        // neither may be guessed — the two bases differ by 16x.
        double? global = RopeTheta(root, "full_attention") ?? Number(root, "global_rope_theta");
        double? local = RopeTheta(root, "sliding_attention") ?? Number(root, "local_rope_theta");
        if (global is null || local is null)
        {
            throw new ModelConfigException(
                "modernbert config declares no RoPE base for full and sliding layers " +
                "(rope_parameters.full_attention/sliding_attention.rope_theta or global_rope_theta/local_rope_theta)");
        }
        int window = root.TryGetProperty("local_attention", out var la)
            ? la.GetInt32()
            : throw new ModelConfigException("modernbert config declares no 'local_attention' window");

        var special = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string key in SpecialTokenKeys)
        {
            if (root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number)
            {
                special[key] = v.GetInt32();
            }
        }

        return new ModernBertFacts(
            global.Value,
            local.Value,
            window,
            NormBias: Bool(root, "norm_bias", fallback: false),
            MlpBias: Bool(root, "mlp_bias", fallback: false),
            AttentionBias: Bool(root, "attention_bias", fallback: false),
            special);
    }

    private static double? RopeTheta(JsonElement root, string layerType) =>
        root.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object &&
        rp.TryGetProperty(layerType, out var lt) && lt.ValueKind == JsonValueKind.Object &&
        lt.TryGetProperty("rope_theta", out var theta)
            ? theta.GetDouble()
            : null;

    private static double? Number(JsonElement root, string name) =>
        root.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;

    private static bool Bool(JsonElement root, string name, bool fallback) =>
        root.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean()
            : fallback;

    /// <summary>Reads the switches Von's SDK takes from
    /// <c>marker_calibration.json</c>. Absent means the SDK defaults: full
    /// cross-option attention (the 1.1-and-earlier mode) and no digit split.</summary>
    public static (bool IndependentOptions, bool DigitSplit) ReadCalibrationSwitches(string modelDir)
    {
        string path = Path.Combine(modelDir, CalibrationFile);
        if (!File.Exists(path))
        {
            return (false, false);
        }
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        return (Bool(root, "independent_options", fallback: false), Bool(root, "digit_split", fallback: false));
    }
}

public static partial class ArchMapper
{
    private static ContainerSpec MapModernBert(string modelId, TextArchitectureFacts facts, HfInventory inventory)
    {
        var mb = ModernBert.ReadFacts(Path.Combine(inventory.Root, "config.json"));
        if (facts.HiddenAct != "gelu")
        {
            throw new ModelConfigException(
                $"modernbert hidden_activation '{facts.HiddenAct}' — this build judges the GELU (GeGLU) encoder only");
        }
        if (mb.MlpBias || mb.AttentionBias || mb.NormBias)
        {
            throw new ModelConfigException(
                "modernbert with norm/mlp/attention bias enabled is not judged by this build — " +
                "the importer binds weight-only LayerNorms and projections");
        }

        string encoding = EncodingFor(inventory, "embeddings.tok_embeddings.weight")
            ?? throw new ModelConfigException(
                "modernbert checkpoint has no 'embeddings.tok_embeddings.weight' — expected the bare ModernBertModel layout " +
                "(a ModernBertForMaskedLM / ForSequenceClassification checkpoint nests the backbone under 'model.' and is not supported)");

        // ── per-layer table ───────────────────────────────────────────────
        // Global layers attend over the whole sequence with the large RoPE
        // base; sliding layers see ±local_attention/2 tokens with the small
        // one. The table is the authority — the runtime never re-derives the
        // every-third-layer pattern.
        int numKvHeads = facts.NumKvHeads == 0 ? facts.NumQueryHeads : facts.NumKvHeads;
        var geometry = new HeadGeometry { HeadDim = facts.HeadDim, NumKvHeads = numKvHeads };
        var policies = new List<AttentionLayerPolicy>(facts.NumLayers);
        for (int l = 0; l < facts.NumLayers; l++)
        {
            policies.Add(facts.LayerTypes[l] switch
            {
                "full_attention" => new AttentionLayerPolicy
                {
                    Operator = LayerOperators.Softmax,
                    Span = AttentionSpan.Full,
                    Position = new PositionRope { Theta = mb.GlobalRopeTheta },
                    Geometry = geometry,
                },
                "sliding_attention" => new AttentionLayerPolicy
                {
                    Operator = LayerOperators.Softmax,
                    Span = AttentionSpan.Sliding,
                    Window = mb.LocalAttention,
                    DeclaredSpan = $"bidirectional ±{mb.LocalAttention / 2} tokens (local_attention={mb.LocalAttention})",
                    Position = new PositionRope { Theta = mb.LocalRopeTheta },
                    Geometry = geometry,
                },
                var other => throw new ModelConfigException(
                    $"layer {l}: modernbert layer_type '{other}' has no judged operator mapping"),
            });
        }

        var norm = new NormSpec { Kind = NormType.LayerNorm, Eps = facts.RmsNormEps, WeightOffset = 0f };

        // ── option-marker head (Von) ──────────────────────────────────────
        List<NamedTensorData>? scorer = null;
        OptionMarkerSurface? optionMarker = null;
        string markerPath = Path.Combine(inventory.Root, ModernBert.OptionMarkerFile);
        if (File.Exists(markerPath))
        {
            (scorer, var scorerHidden) = BindOptionMarker(markerPath, inventory, facts.HiddenSize);
            var (independent, digitSplit) = ModernBert.ReadCalibrationSwitches(inventory.Root);
            optionMarker = new OptionMarkerSurface
            {
                ScorerHiddenSize = scorerHidden,
                NormEps = 1e-5,                 // nn.LayerNorm default; OptionMarkerScorer sets none
                Activation = Activation.Gelu,   // nn.GELU(), exact erf form
                MarkerToken = ModernBert.MarkerToken,
                IndependentOptions = independent,
                DigitSplit = digitSplit,
            };
        }

        var surface = new ExecutionSurface
        {
            ContextLength = facts.MaxPositionEmbeddings,
            Attention = new AttentionSurface
            {
                NumQHeads = facts.NumQueryHeads,
                NumKvHeads = numKvHeads,
                HeadDim = facts.HeadDim,
                ScoreScale = 1.0 / Math.Sqrt(facts.HeadDim),
                AttentionBias = false,
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = facts.IntermediateSize,
                Activation = Activation.Gelu,
                FfnType = FfnType.Gated,
            },
            Norm = new NormSurface
            {
                Pre = norm,
                FinalNorm = norm,
                Placement = NormPlacement.PreOnly,
            },
            Head = new HeadSurface
            {
                VocabSize = facts.VocabSize,
                HeadReusesEmbedding = facts.TieWordEmbeddings,
            },
            Encoder = new EncoderSurface
            {
                Architecture = ModernBert.Architecture,
                NormBias = mb.NormBias,
                MlpBias = mb.MlpBias,
                EmbeddingNorm = true,
                FirstLayerSkipsAttentionNorm = !inventory.TryGet("layers.0.attn_norm.weight", out _),
                FusedFfnLayout = "input_then_gate",
                SpecialTokenIds = mb.SpecialTokenIds.Count > 0 ? mb.SpecialTokenIds : null,
            },
            OptionMarker = optionMarker,
        };

        // ── objects ───────────────────────────────────────────────────────
        LogicalObject Obj(string id, ObjectKind kind, string artifact, string prefix) => new()
        {
            Id = id,
            Component = "target",
            Kind = kind,
            SourceBindings = new List<SourceBinding>
            {
                new() { Artifact = artifact, TensorPrefix = prefix, Tensors = 0, Bytes = 0 },
            },
            Representations = new List<Representation>
            {
                new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
            },
        };

        var embedding = BindUnder(inventory, "embeddings.");
        var stack = BindUnder(inventory, "layers.");
        var finalNorm = BindUnder(inventory, "final_norm.");
        if (stack.Count == 0 || finalNorm.Count == 0)
        {
            throw new ModelConfigException("modernbert checkpoint is missing 'layers.*' or 'final_norm.*' tensors");
        }
        int bound = embedding.Count + stack.Count + finalNorm.Count;
        var unbound = inventory.TensorNames
            .Where(n => !n.StartsWith("embeddings.", StringComparison.Ordinal) &&
                        !n.StartsWith("layers.", StringComparison.Ordinal) &&
                        !n.StartsWith("final_norm.", StringComparison.Ordinal))
            .ToList();
        if (unbound.Count > 0)
        {
            // An MLM or classification head this build does not model; dropping
            // it silently would export a checkpoint missing weights.
            throw new ModelConfigException(
                $"modernbert checkpoint carries {unbound.Count} tensor(s) outside the backbone that this build does not model: " +
                string.Join(", ", unbound.Take(5)) + (unbound.Count > 5 ? ", …" : string.Empty));
        }

        var objects = new List<LogicalObject>
        {
            Obj("target.embedding", ObjectKind.Embedding, "model.safetensors", "embeddings"),
            Obj("target.encoder_stack", ObjectKind.EncoderStack, "model.safetensors", "layers"),
            Obj("target.final_norm", ObjectKind.FinalNorm, "model.safetensors", "final_norm"),
        };
        var reps = new List<RepresentationSpec>
        {
            Rep("target.embedding", encoding, embedding),
            Rep("target.encoder_stack", encoding, stack),
            Rep("target.final_norm", encoding, finalNorm),
        };
        if (scorer is not null)
        {
            string scorerEncoding = scorer[0].Dtype.Label();
            var head = Obj("target.option_marker_head", ObjectKind.ClassifierHead, ModernBert.OptionMarkerFile,
                ModernBert.ScorerPrefix);
            head.Representations[0] = new Representation { Encoding = scorerEncoding, Fidelity = Fidelity.Canonical };
            objects.Add(head);
            reps.Add(Rep("target.option_marker_head", scorerEncoding, scorer));
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
                    SourceArtifact = "model.safetensors",
                    NumLayers = facts.NumLayers,
                    HiddenSize = facts.HiddenSize,
                    Attention = policies,
                    Execution = surface,
                },
            },
            Objects = objects,
            Edges = new List<HiddenStateEdge>(),
        };

        return new ContainerSpec
        {
            Model = modelId,
            Family = ModernBert.ModelType,
            HiddenSize = facts.HiddenSize,
            NumLayers = facts.NumLayers,
            SystemGraph = graph,
            Representations = reps,
        };
    }

    /// <summary>Binds the scorer tensors from <c>option_marker.pt</c> and proves
    /// its <c>encoder.</c> copy of the backbone is the safetensors backbone
    /// byte for byte. Returns the head's tensors and its hidden width.</summary>
    private static (List<NamedTensorData> Tensors, int ScorerHidden) BindOptionMarker(
        string path, HfInventory inventory, int hiddenSize)
    {
        using var pt = TorchCheckpoint.Open(path);

        var expectedEncoder = new HashSet<string>(inventory.TensorNames, StringComparer.Ordinal);
        var tensors = new List<NamedTensorData>();
        foreach (string name in pt.TensorNames)
        {
            var info = pt.Get(name);
            if (name.StartsWith(ModernBert.EncoderPrefixInPt, StringComparison.Ordinal))
            {
                string backbone = name[ModernBert.EncoderPrefixInPt.Length..];
                if (!expectedEncoder.Remove(backbone))
                {
                    throw new ModelConfigException(
                        $"'{ModernBert.OptionMarkerFile}' carries '{name}' but model.safetensors has no '{backbone}'");
                }
                var st = inventory.Get(backbone);
                if (st.Dtype != info.Dtype || !st.Shape.SequenceEqual(info.Shape) ||
                    !inventory.ReadBytes(backbone).AsSpan().SequenceEqual(pt.ReadBytes(name)))
                {
                    throw new ModelConfigException(
                        $"'{ModernBert.OptionMarkerFile}' holds a different '{backbone}' than model.safetensors. " +
                        "Von loads the .pt over the backbone, so these are the weights it runs — re-save " +
                        "model.safetensors from the .pt's encoder before importing, so the container holds what Von executes");
                }
                continue;
            }
            if (!name.StartsWith(ModernBert.ScorerPrefix + ".", StringComparison.Ordinal))
            {
                throw new ModelConfigException(
                    $"'{ModernBert.OptionMarkerFile}' carries '{name}', outside the encoder and the scorer — not an OptionMarkerModel state dict");
            }
            tensors.Add(new NamedTensorData
            {
                Name = name[(ModernBert.ScorerPrefix.Length + 1)..],
                Dtype = info.Dtype,
                Shape = info.Shape,
                Data = pt.ReadBytes(name),
            });
        }
        if (expectedEncoder.Count > 0)
        {
            throw new ModelConfigException(
                $"'{ModernBert.OptionMarkerFile}' is missing {expectedEncoder.Count} backbone tensor(s), e.g. 'encoder.{expectedEncoder.First()}' — " +
                "Von loads it with strict=True, so it must hold the whole backbone");
        }

        var byName = tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
        foreach (var (name, rank) in ModernBert.ScorerTensors)
        {
            if (!byName.TryGetValue(name, out var t) || t.Shape.Length != rank)
            {
                throw new ModelConfigException(
                    $"'{ModernBert.OptionMarkerFile}' has no rank-{rank} 'scorer.{name}' — not the Von option-marker head");
            }
        }
        if (byName.Count != ModernBert.ScorerTensors.Count)
        {
            var extra = byName.Keys.Except(ModernBert.ScorerTensors.Select(s => s.Name)).First();
            throw new ModelConfigException($"'{ModernBert.OptionMarkerFile}' has an unexpected scorer tensor 'scorer.{extra}'");
        }
        var dense = byName["dense.weight"].Shape;
        var outProj = byName["out_proj.weight"].Shape;
        if (dense[1] != hiddenSize || outProj[0] != 1 || outProj[1] != dense[0])
        {
            throw new ModelConfigException(
                $"scorer shapes do not chain: dense [{string.Join("x", dense)}], out_proj [{string.Join("x", outProj)}], hidden {hiddenSize}");
        }

        // Keep the reference's registration order so an export writes the
        // state dict in the order torch would.
        var order = ModernBert.ScorerTensors.Select(s => s.Name).ToList();
        tensors.Sort((a, b) => order.IndexOf(a.Name).CompareTo(order.IndexOf(b.Name)));
        return (tensors, (int)dense[0]);
    }
}
