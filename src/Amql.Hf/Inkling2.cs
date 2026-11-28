using System.Text.Json;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Hf;

/// <summary>
/// Inkling 2 looped (<c>inkling2_loop</c>, Thinking Machines). Its stack has
/// none of the usual roles: no rotary but a relative bias per head, sinks,
/// short causal convolutions around attention and the experts, a residual
/// that attends over blocks of earlier states, and a core of layers run
/// several times. This is its config adapter: it reads the checkpoint's keys
/// and names, and describes the stack with the generic surfaces the planner
/// reads (<see cref="DescribeSurface"/>) and a role map of its tensors; the
/// runtimes know the stack only by those surfaces.
/// <para>
/// Two spellings of the config are read: the checkpoint's own
/// (<c>text_config</c> with <c>loop_config</c>, <c>local_layer_ids</c>,
/// <c>sconv_kernel_size</c>, <c>intermediate_size</c>) and transformers'
/// (<c>layer_types</c>, <c>loop_layer_index_start</c>, …). The tensors keep
/// the checkpoint's names under <c>model.llm.</c> either way.
/// </para>
/// </summary>
public static class Inkling2
{
    public const string Prefix = "model.llm.";
    public const string Architecture = "Inkling2LoopForCausalLM";

    /// <summary>Where builds before the generic surfaces recorded this adapter's facts.</summary>
    public const string FactsKey = "inkling2";

    /// <summary>Whether <paramref name="configPath"/> describes an Inkling 2 looped model.</summary>
    public static bool Describes(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = doc.RootElement;
        string? type = Str(root, "model_type");
        string? textType = root.TryGetProperty("text_config", out var text) && text.ValueKind == JsonValueKind.Object
            ? Str(text, "model_type")
            : null;
        return type is "inkling2_loop" or "inkling2_loop_text" || textType is "inkling2_loop_text";
    }

    public static Inkling2Facts ReadFacts(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = doc.RootElement;
        var text = root.TryGetProperty("text_config", out var nested) && nested.ValueKind == JsonValueKind.Object ? nested : root;

        int layers = Int(text, "num_hidden_layers");
        var full = new bool[layers];
        if (text.TryGetProperty("layer_types", out var types) && types.ValueKind == JsonValueKind.Array)
        {
            var list = types.EnumerateArray().Select(t => t.GetString()).ToList();
            if (list.Count != layers)
            {
                throw new ModelConfigException($"layer_types lists {list.Count} layers, num_hidden_layers is {layers}");
            }
            for (int l = 0; l < layers; l++)
            {
                full[l] = list[l] switch
                {
                    "hybrid" => true,
                    "hybrid_sliding" => false,
                    var other => throw new ModelConfigException($"layer {l}: Inkling 2 layer type '{other}' is not judged"),
                };
            }
        }
        else if (text.TryGetProperty("local_layer_ids", out var local) && local.ValueKind == JsonValueKind.Array)
        {
            Array.Fill(full, true);
            foreach (var id in local.EnumerateArray())
            {
                full[id.GetInt32()] = false;
            }
        }
        else
        {
            throw new ModelConfigException("Inkling 2 config gives neither layer_types nor local_layer_ids");
        }

        // The loop: transformers' flat keys, or the checkpoint's loop_config.
        int loopStart, loopEnd, passes, rank;
        if (text.TryGetProperty("loop_layer_index_start", out _))
        {
            loopStart = Int(text, "loop_layer_index_start");
            loopEnd = Int(text, "loop_layer_index_end");
            passes = Int(text, "loop_num_passes");
            rank = Int(text, "loop_gate_rank");
        }
        else if (text.TryGetProperty("loop_config", out var loop) || root.TryGetProperty("loop_config", out loop))
        {
            loopStart = Int(loop, "core_start");
            loopEnd = Int(loop, "core_end");
            passes = Int(loop, "num_passes");
            rank = Int(loop, "gate_rank");
            if (loop.TryGetProperty("is_varloop", out var varloop) && varloop.ValueKind == JsonValueKind.True)
            {
                throw new ModelConfigException("a variable loop (is_varloop) is not judged by this build");
            }
        }
        else
        {
            throw new ModelConfigException("Inkling 2 looped config states no loop (loop_config or loop_layer_index_start)");
        }

        int block = Int(text, "n_layers_per_attn_res_block");
        if (passes < 2 || loopStart <= 0 || loopEnd <= loopStart || loopEnd > layers || loopStart % block != 0 || loopEnd % block != 0)
        {
            throw new ModelConfigException(
                $"Inkling 2 loop [{loopStart}, {loopEnd}) x{passes} over {layers} layers in blocks of {block} is not a loop this build runs");
        }

        string scoring = Str(text, "scoring_func") ?? "sigmoid";
        if (scoring != "sigmoid")
        {
            throw new ModelConfigException($"Inkling 2 router scoring '{scoring}' is not judged (sigmoid is)");
        }
        if (Num(text, "n_shared_experts") is > 0)
        {
            throw new ModelConfigException("Inkling 2 with shared experts is not judged by this build");
        }
        foreach (var (key, served) in new[] { ("q_bias", false), ("o_bias", false), ("use_value_norm_no_gamma", true),
                     ("attn_res_normalize_values", true), ("use_gate_bias", true) })
        {
            if (text.TryGetProperty(key, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False && v.GetBoolean() != served)
            {
                throw new ModelConfigException($"Inkling 2 with {key} = {v.GetBoolean()} is not judged by this build");
            }
        }

        int vocab = Int(text, "vocab_size");
        return new Inkling2Facts
        {
            FullLayers = full,
            FullHeads = Int(text, "num_attention_heads"),
            FullKvHeads = Int(text, "num_key_value_heads"),
            FullHeadDim = Int(text, "head_dim"),
            SlidingHeads = (int)(Num(text, "swa_num_attention_heads") ?? Int(text, "num_attention_heads")),
            SlidingKvHeads = (int)(Num(text, "swa_num_key_value_heads") ?? Int(text, "num_key_value_heads")),
            SlidingHeadDim = (int)(Num(text, "swa_head_dim") ?? Int(text, "head_dim")),
            SlidingWindow = Int(text, "sliding_window_size"),
            RelativeDim = Int(text, "d_rel"),
            RelativeExtent = Int(text, "rel_extent"),
            LogScalingFloor = Int(text, "log_scaling_n_floor"),
            LogScalingAlpha = Num(text, "log_scaling_alpha") ?? throw Missing("log_scaling_alpha"),
            QkClip = Num(text, "qk_l2_clip_threshold") ?? throw Missing("qk_l2_clip_threshold"),
            RelativeClip = Num(text, "rel_l2_clip_threshold") ?? throw Missing("rel_l2_clip_threshold"),
            ConvKernel = (int)(Num(text, "conv_kernel_size") ?? Num(text, "sconv_kernel_size") ?? throw Missing("conv_kernel_size")),
            BlockSize = block,
            Experts = Int(text, "n_routed_experts"),
            TopK = Int(text, "num_experts_per_tok"),
            ExpertWidth = (int)(Num(text, "moe_intermediate_size") ?? Num(text, "intermediate_size") ?? throw Missing("moe_intermediate_size")),
            RouteScale = Num(text, "route_scale") ?? throw Missing("route_scale"),
            GateSoftcap = Num(text, "mlp_gate_softcap") ?? throw Missing("mlp_gate_softcap"),
            UpSoftcap = Num(text, "mlp_up_softcap") ?? throw Missing("mlp_up_softcap"),
            FinalSoftcap = Num(text, "final_logit_softcapping") ?? throw Missing("final_logit_softcapping"),
            LoopStart = loopStart,
            LoopEnd = loopEnd,
            LoopPasses = passes,
            GateRank = rank,
            Eps = Num(text, "rms_norm_eps") ?? throw Missing("rms_norm_eps"),
            PaddedVocab = vocab,
            UnpaddedVocab = (int)(Num(text, "unpadded_vocab_size") ?? vocab),
        };
    }

    /// <summary>
    /// The execution surface for these facts: the generic surfaces the planner
    /// reads (depth loop, block residual, message gates, residual convolutions,
    /// the attention's conditioning and relative bias, the packed experts'
    /// routing and gate) with the layer role map that names this checkpoint's
    /// tensors. A fresh encode and <see cref="UpgradeGraph"/> both build it here,
    /// so the two agree.
    /// </summary>
    public static ExecutionSurface DescribeSurface(Inkling2Facts facts, long? contextLength)
    {
        if (facts.SlidingHeads != facts.FullHeads)
        {
            throw new ModelConfigException(
                $"Inkling 2 with {facts.SlidingHeads} sliding and {facts.FullHeads} full query heads is not described to the planner " +
                "(its attention surface holds one query-head count)");
        }
        int layers = facts.FullLayers.Count;
        var norm = new NormSpec { Kind = NormType.RmsNorm, Eps = facts.Eps, WeightOffset = 0f };
        var fullLayers = Enumerable.Range(0, layers).Where(l => facts.FullLayers[l]).ToList();
        return new ExecutionSurface
        {
            ContextLength = contextLength,
            Attention = new AttentionSurface
            {
                NumQHeads = facts.FullHeads,
                NumKvHeads = facts.FullKvHeads,
                HeadDim = facts.FullHeadDim,
                ScoreScale = 1.0 / Math.Sqrt(facts.FullHeadDim),
                AttentionBias = false,
                ParameterFreeQkNorm = new ParameterFreeQkNorm { V = true },
                QkClip = facts.QkClip,
                LogScaling = new PositionLogScaling { Floor = facts.LogScalingFloor, Alpha = facts.LogScalingAlpha, Layers = fullLayers },
                RelativeBias = new RelativeBiasSurface { Rank = facts.RelativeDim, Clip = facts.RelativeClip },
                SinkLogits = SinkLogitKind.Projected,
                SeparateOutputGate = true,
            },
            Ffn = new FfnSurface
            {
                IntermediateSize = facts.ExpertWidth,
                Activation = Activation.Silu,
                FfnType = FfnType.Gated,
                GatePolicy = new ExpertGateSoftcappedGlu { GateCap = facts.GateSoftcap, UpCap = facts.UpSoftcap },
                Moe = new MoeSurface
                {
                    Experts = facts.Experts,
                    TopK = facts.TopK,
                    ExpertIntermediateSize = facts.ExpertWidth,
                    RoutingPolicy = ExpertRoutingPolicy.SigmoidNormalisedOverSelected,
                    SelectionBias = true,
                    RouteScale = facts.RouteScale,
                    PackedExperts = true,
                    GateUpLayout = GateUpLayouts.InterleavedRows,
                    ExpertFormat = "as_stored",
                },
            },
            Norm = new NormSurface
            {
                Pre = norm, Post = norm, FinalNorm = norm, Placement = NormPlacement.PreOnly, FinalNormWeight = "norm.weight",
            },
            Head = new HeadSurface
            {
                VocabSize = facts.PaddedVocab,
                UnpaddedVocabSize = facts.UnpaddedVocab < facts.PaddedVocab ? facts.UnpaddedVocab : null,
                HeadReusesEmbedding = false,
                FinalLogitSoftcapping = (float)facts.FinalSoftcap,
                EmbeddingWeight = "embed.weight",
                EmbeddingNorm = new EmbeddingNorm { Kind = NormType.RmsNorm, Eps = facts.Eps, Weight = "embed_norm.weight" },
            },
            DepthLoop = new DepthLoopSurface { Start = facts.LoopStart, End = facts.LoopEnd, Passes = facts.LoopPasses },
            BlockResidual = new BlockResidualSurface
            {
                BlockSize = facts.BlockSize,
                Eps = 1e-6,
                QueryObject = "target.final_norm",
                MixerQueries = Enumerable.Range(0, layers).Select(l => l == 0 ? null : $"attn_res.attn_queries.{l - 1}").ToList(),
                FfnQueries = Enumerable.Range(0, layers).Select(l => $"attn_res.mlp_queries.{l}").ToList(),
                OutputQuery = "attn_res.output_query",
            },
            MessageGates = new MessageGateSurface
            {
                Rank = facts.GateRank,
                Eps = facts.Eps,
                SkipBlocks = 1,
                // The checkpoint keeps [q, ffn] and [k, v] per looped layer in two tensors.
                Sites = new List<MessageGateSite>
                {
                    new() { Input = "q", Query = "middle_cycle_cross_pass_queries", Stride = 2, Offset = 0 },
                    new() { Input = "k", Query = "middle_cycle_cross_pass_kv_queries", Stride = 2, Offset = 0 },
                    new() { Input = "v", Query = "middle_cycle_cross_pass_kv_queries", Stride = 2, Offset = 1 },
                    new() { Input = "ffn", Query = "middle_cycle_cross_pass_queries", Stride = 2, Offset = 1 },
                },
            },
            ResidualConv = new ResidualConvSurface { Kernel = facts.ConvKernel },
            Roles = Roles(),
            Provenance = new Dictionary<string, string>
            {
                ["block_residual.eps"] = "modeling_inkling2_loop: the depth mix normalises its states with a fixed 1e-6",
                ["message_gates.skip_blocks"] = "modeling_inkling2_loop: a pass's first block is the state the loop restarts from",
            },
        };
    }

    /// <summary>This checkpoint's layer tensors by role (layer-relative names).</summary>
    public static LayerRoleMap Roles()
    {
        var roles = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [$"{TensorRoles.AttnIn}.weight"] = "attn_norm.weight",
            [$"{TensorRoles.QInNorm}.weight"] = "q_input_norm.weight",
            [$"{TensorRoles.KInNorm}.weight"] = "k_input_norm.weight",
            [$"{TensorRoles.VInNorm}.weight"] = "v_input_norm.weight",
            [$"{TensorRoles.FfnIn}.weight"] = "mlp_norm.weight",
            [$"{TensorRoles.Q}.weight"] = "attn.wq_du.weight",
            [$"{TensorRoles.K}.weight"] = "attn.wk_dv.weight",
            [$"{TensorRoles.V}.weight"] = "attn.wv_dv.weight",
            [$"{TensorRoles.O}.weight"] = "attn.wo_ud.weight",
            [$"{TensorRoles.OutGateProj}.weight"] = "attn.gate_attn.weight",
            [$"{TensorRoles.SinkProj}.weight"] = "attn.attn_sink_proj.weight",
            [$"{TensorRoles.QChannelScale}.weight"] = "attn.q_l2_clip.scale",
            [$"{TensorRoles.RelProj}.weight"] = "attn.wr_du.weight",
            [$"{TensorRoles.RelChannelScale}.weight"] = "attn.rel_l2_clip.scale",
            [$"{TensorRoles.RelTable}.weight"] = "attn.rel_logits_proj.proj",
            [$"{TensorRoles.KConv}.weight"] = "attn.k_sconv.weight",
            [$"{TensorRoles.KConv}.static"] = "attn.k_sconv.static_w",
            [$"{TensorRoles.KConv}.tap_proj"] = "attn.k_sconv.weight_proj",
            [$"{TensorRoles.VConv}.weight"] = "attn.v_sconv.weight",
            [$"{TensorRoles.VConv}.static"] = "attn.v_sconv.static_w",
            [$"{TensorRoles.VConv}.tap_proj"] = "attn.v_sconv.weight_proj",
            [$"{TensorRoles.MixerOutConv}.weight"] = "attn_sconv.weight",
            [$"{TensorRoles.FfnOutConv}.weight"] = "mlp_sconv.weight",
            [$"{TensorRoles.Router}.weight"] = "mlp.gate.weight",
            [$"{TensorRoles.Router}.bias"] = "mlp.gate.bias",
        };
        foreach (var (role, stem) in new[] { (TensorRoles.PackedGateUp, "mlp.experts.w13_weight"), (TensorRoles.PackedDown, "mlp.experts.w2_weight") })
        {
            roles[$"{role}.weight"] = stem;
            roles[$"{role}.scale"] = stem + "_scale";
            roles[$"{role}.amax"] = stem + "_amax";
            roles[$"{role}.denominator"] = stem + "_outer_scale_denominator";
        }
        foreach (var (site, controller) in new[] { ("q", "attn"), ("k", "k"), ("v", "v"), ("ffn", "mlp") })
        {
            foreach (var part in new[] { "down", "forget", "gain" })
            {
                roles[$"{TensorRoles.MessageGate(site)}.{part}"] = $"controllers.{controller}.{part}_weight";
            }
        }
        return new LayerRoleMap { Roles = roles };
    }

    /// <summary>
    /// Rewrites a system graph written by an older build: every component that
    /// carries Inkling 2's facts (<c>execution.inkling2</c>, which this build no
    /// longer reads) gets <see cref="DescribeSurface"/> of those facts (its
    /// context length kept), and the facts are dropped. Tensors and every other
    /// component are untouched. Returns the components rewritten (none when the
    /// graph is already current).
    /// </summary>
    public static (SystemGraph Graph, IReadOnlyList<string> Upgraded) UpgradeGraph(SystemGraph graph)
    {
        var upgraded = new List<string>();
        var components = graph.Components.Select(c =>
        {
            if (c.Execution is not { Unread: { } unread } old || !unread.TryGetValue(FactsKey, out var recorded))
            {
                return c;
            }
            var facts = recorded.Deserialize<Inkling2Facts>(ViJson.Options)
                        ?? throw new ContainerException($"component '{c.Id}': '{FactsKey}' records no facts");
            upgraded.Add(c.Id);
            return new Component
            {
                Id = c.Id,
                Role = c.Role,
                SourceArtifact = c.SourceArtifact,
                NumLayers = c.NumLayers,
                HiddenSize = c.HiddenSize,
                Attention = c.Attention,
                Perception = c.Perception,
                Execution = DescribeSurface(facts, old.ContextLength),
            };
        }).ToList();
        return (new SystemGraph { Schema = graph.Schema, Components = components, Objects = graph.Objects, Edges = graph.Edges }, upgraded);
    }

    internal static int Hidden(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = doc.RootElement;
        var text = root.TryGetProperty("text_config", out var nested) && nested.ValueKind == JsonValueKind.Object ? nested : root;
        return Int(text, "hidden_size");
    }

    internal static long Context(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = doc.RootElement;
        var text = root.TryGetProperty("text_config", out var nested) && nested.ValueKind == JsonValueKind.Object ? nested : root;
        return (long)(Num(text, "max_position_embeddings") ?? Num(text, "model_max_length") ?? 0);
    }

    private static ModelConfigException Missing(string key) => new($"Inkling 2 config is missing '{key}'");

    private static int Int(JsonElement e, string key) =>
        e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : throw Missing(key);

    private static double? Num(JsonElement e, string key) =>
        e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;

    private static string? Str(JsonElement e, string key) =>
        e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
}

public static partial class ArchMapper
{
    /// <summary>Inkling 2 looped: four objects under the checkpoint's own
    /// <c>model.llm.</c> names, so export writes them back as they were. The
    /// residual's queries (<c>attn_res.*</c>, <c>middle_cycle_cross_pass_*</c>)
    /// travel with the final norm: they are the stack's mixing, not a layer's.</summary>
    public static ContainerSpec MapInkling2(string modelId, string configPath, HfInventory inventory, EncodeOptions options)
    {
        var facts = Inkling2.ReadFacts(configPath);
        int hidden = Inkling2.Hidden(configPath);
        const string p = Inkling2.Prefix;

        // The drafter ships in its own file (mtp.safetensors, unprefixed names
        // such as fuse_weight and layers.N.*) that the index does not list: it
        // is not the model's, and is never encoded. The vision tower is in the
        // model's own file and is refused unless left out with --no-vision.
        var own = OwnTensorNames(inventory.Root);
        bool SideFile(string n) => own is not null && !own.Contains(n);
        var unmodelled = inventory.TensorNames
            .Where(n => !SideFile(n) &&
                        !n.StartsWith(p, StringComparison.Ordinal) &&
                        !(n.StartsWith("model.visual.", StringComparison.Ordinal) && !options.IncludeVision) &&
                        !(n.StartsWith("mtp.", StringComparison.Ordinal) && !options.IncludeMtp))
            .ToList();
        if (unmodelled.Count > 0)
        {
            throw new ModelConfigException(
                $"Inkling 2 checkpoint carries {unmodelled.Count} tensor(s) this build does not model " +
                $"(the vision tower and the drafter are not served yet: encode with --no-vision): " +
                string.Join(", ", unmodelled.Take(4)) + (unmodelled.Count > 4 ? ", …" : string.Empty));
        }
        for (int l = 0; l < facts.FullLayers.Count; l++)
        {
            CheckExperts(inventory, $"{p}layers.{l}.mlp.experts.w13_weight", facts.Experts, 2L * facts.ExpertWidth, hidden);
            CheckExperts(inventory, $"{p}layers.{l}.mlp.experts.w2_weight", facts.Experts, hidden, facts.ExpertWidth);
        }

        string encoding = EncodingFor(inventory, p + "embed.weight")
            ?? throw new ModelConfigException($"Inkling 2 checkpoint has no '{p}embed.weight'");

        var embedding = new List<NamedTensorData>();
        var mixing = new List<NamedTensorData>();
        var head = new List<NamedTensorData>();
        foreach (string name in inventory.TensorNames.Where(n => n.StartsWith(p, StringComparison.Ordinal)))
        {
            string rel = name[p.Length..];
            if (rel.StartsWith("layers.", StringComparison.Ordinal))
            {
                continue;
            }
            var list = rel is "embed.weight" or "embed_norm.weight" ? embedding
                : rel.StartsWith("unembed.", StringComparison.Ordinal) ? head
                : mixing;
            list.Add(ToTensorData(inventory, name, list == head ? rel["unembed.".Length..] : rel));
        }
        var stack = BindUnder(inventory, p + "layers.");
        if (head.Count == 0)
        {
            throw new ModelConfigException($"Inkling 2 checkpoint has no '{p}unembed.weight'");
        }

        long Shape0(string name) => inventory.Get(p + name).Shape[0];
        if (Shape0("embed.weight") != facts.PaddedVocab || Shape0("unembed.weight") != facts.PaddedVocab)
        {
            throw new ModelConfigException($"Inkling 2 embedding or head rows differ from vocab_size {facts.PaddedVocab}");
        }
        int numLayers = facts.FullLayers.Count;
        for (int l = 0; l < numLayers; l++)
        {
            bool fullLayer = facts.FullLayers[l];
            int heads = fullLayer ? facts.FullHeads : facts.SlidingHeads;
            int headDim = fullLayer ? facts.FullHeadDim : facts.SlidingHeadDim;
            if (inventory.Get($"{p}layers.{l}.attn.wq_du.weight").Shape[0] != heads * headDim)
            {
                throw new ModelConfigException($"layer {l}: wq_du rows are not {heads} heads of {headDim}");
            }
            bool dynamic = inventory.TryGet($"{p}layers.{l}.attn.k_sconv.static_w", out _);
            if (dynamic != fullLayer)
            {
                throw new ModelConfigException($"layer {l}: the key convolution is {(dynamic ? "dynamic" : "static")} on a {(fullLayer ? "full" : "sliding")} layer");
            }
            bool loop = l >= facts.LoopStart && l < facts.LoopEnd;
            if (inventory.TryGet($"{p}layers.{l}.controllers.attn.down_weight", out _) != loop)
            {
                throw new ModelConfigException($"layer {l}: message gates {(loop ? "missing in" : "outside")} the loop");
            }
        }

        var policies = facts.FullLayers.Select(fullLayer => new AttentionLayerPolicy
        {
            Operator = LayerOperators.Softmax,
            Span = fullLayer ? AttentionSpan.Full : AttentionSpan.Sliding,
            Window = fullLayer ? null : facts.SlidingWindow,
            DeclaredSpan = fullLayer ? "full, relative bias over rel_extent" : "sliding window, relative bias over it",
            Position = PositionPolicy.None,
            Geometry = new HeadGeometry
            {
                HeadDim = fullLayer ? facts.FullHeadDim : facts.SlidingHeadDim,
                NumKvHeads = fullLayer ? facts.FullKvHeads : facts.SlidingKvHeads,
            },
        }).ToList();

        var surface = Inkling2.DescribeSurface(facts, Inkling2.Context(configPath) is > 0 and var context ? context : null);

        LogicalObject Obj(string id, ObjectKind kind, string prefix) => new()
        {
            Id = id,
            Component = "target",
            Kind = kind,
            SourceBindings = new List<SourceBinding> { new() { Artifact = "model.safetensors", TensorPrefix = prefix, Tensors = 0, Bytes = 0 } },
            Representations = new List<Representation> { new() { Encoding = encoding, Fidelity = Fidelity.Canonical } },
        };
        var objects = new List<LogicalObject>
        {
            Obj("target.embedding", ObjectKind.Embedding, "model.llm"),
            Obj("target.decoder_stack", ObjectKind.DecoderStack, "model.llm.layers"),
            Obj("target.final_norm", ObjectKind.FinalNorm, "model.llm"),
            Obj("target.output_head", ObjectKind.OutputHead, "model.llm.unembed"),
        };
        var reps = new List<RepresentationSpec>
        {
            Rep("target.embedding", encoding, embedding),
            Rep("target.decoder_stack", encoding, stack),
            Rep("target.final_norm", encoding, mixing),
            Rep("target.output_head", encoding, head),
        };

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
                    NumLayers = numLayers,
                    HiddenSize = hidden,
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
            Family = "inkling2_loop",
            HiddenSize = hidden,
            NumLayers = numLayers,
            SystemGraph = graph,
            Representations = reps,
        };
    }

    /// <summary>The tensors <c>model.safetensors.index.json</c> says are the
    /// model's, or null when there is no index (every shard is then the model's).</summary>
    private static HashSet<string>? OwnTensorNames(string modelDir)
    {
        string index = Path.Combine(modelDir, "model.safetensors.index.json");
        if (!File.Exists(index))
        {
            return null;
        }
        using var doc = JsonDocument.Parse(File.ReadAllBytes(index));
        return doc.RootElement.TryGetProperty("weight_map", out var map) && map.ValueKind == JsonValueKind.Object
            ? map.EnumerateObject().Select(e => e.Name).ToHashSet(StringComparer.Ordinal)
            : null;
    }

    /// <summary>
    /// An expert stack, <c>[experts, rows, cols]</c>, in one of the three forms
    /// Inkling 2 checkpoints use, each kept as stored:
    /// <list type="bullet">
    ///   <item>plain (F32, BF16, F16);</item>
    ///   <item>NVFP4: <c>U8 [E, rows, cols/2]</c> (bound as FP4 at the logical
    ///   shape), an E4M3 <c>_scale [E, rows, cols/16]</c>, and the per-expert
    ///   factor <c>_amax [E] / _outer_scale_denominator [1]</c>;</item>
    ///   <item>MXFP8: E4M3 <c>[E, rows, cols]</c> and a U8 E8M0
    ///   <c>_scale [E, rows, cols/32]</c>, with no amax.</item>
    /// </list>
    /// The factor is one per expert, shared by every pass of the loop: the
    /// checkpoint has no per-pass scales.
    /// </summary>
    private static void CheckExperts(HfInventory inventory, string name, long experts, long rows, long cols)
    {
        if (!inventory.TryGet(name, out var info))
        {
            throw new ModelConfigException($"Inkling 2 checkpoint has no '{name}'");
        }
        string Shape(TensorInfo t) => $"{t.Dtype.Label()} [{string.Join(", ", t.Shape)}]";
        bool Is(TensorInfo t, params long[] shape) => t.Shape.SequenceEqual(shape);
        bool hasScale = inventory.TryGet(name + "_scale", out var scale);
        bool hasAmax = inventory.TryGet(name + "_amax", out var amax);
        bool hasDenominator = inventory.TryGet(name + "_outer_scale_denominator", out var denominator);
        switch (info.Dtype)
        {
            case Dtype.F32 or Dtype.BF16 or Dtype.F16 when Is(info, experts, rows, cols) && !hasScale && !hasAmax:
                return;
            case Dtype.U8 when Is(info, experts, rows, cols / 2) && cols % 16 == 0:
                if (!hasScale || scale.Dtype != Dtype.F8_E4M3 || !Is(scale, experts, rows, cols / 16))
                {
                    throw new ModelConfigException(
                        $"'{name}' is NVFP4 but its '_scale' is {(hasScale ? Shape(scale) : "missing")}, not F8_E4M3 [{experts}, {rows}, {cols / 16}]");
                }
                if (!hasAmax || amax.Dtype != Dtype.F32 || !Is(amax, experts) ||
                    !hasDenominator || denominator.Dtype != Dtype.F32 || denominator.ElementCount != 1)
                {
                    throw new ModelConfigException(
                        $"'{name}' is NVFP4 but its per-expert factor is not F32 '_amax' [{experts}] over one F32 '_outer_scale_denominator'");
                }
                return;
            case Dtype.F8_E4M3 when Is(info, experts, rows, cols) && cols % 32 == 0:
                if (!hasScale || scale.Dtype is not (Dtype.U8 or Dtype.F8_E8M0) || !Is(scale, experts, rows, cols / 32))
                {
                    throw new ModelConfigException(
                        $"'{name}' is MXFP8 but its '_scale' is {(hasScale ? Shape(scale) : "missing")}, not U8 [{experts}, {rows}, {cols / 32}]");
                }
                if (hasAmax || hasDenominator)
                {
                    throw new ModelConfigException($"'{name}' is MXFP8 but carries an NVFP4 amax or denominator");
                }
                return;
            default:
                throw new ModelConfigException(
                    $"'{name}' is {Shape(info)}: not an expert stack of [{experts}, {rows}, {cols}] in F32/BF16/F16, NVFP4 or MXFP8");
        }
    }
}

/// <summary>
/// Inkling 2 (<c>inkling2_loop</c>): the config's facts, as the adapter reads
/// them before it describes the stack with the generic surfaces (see
/// <see cref="Inkling2.DescribeSurface"/>). Builds before those surfaces wrote
/// these facts into the graph as <c>execution.inkling2</c>;
/// <see cref="Inkling2.UpgradeGraph"/> reads them back from there.
/// Layers <see cref="LoopStart"/> to <see cref="LoopEnd"/> run
/// <see cref="LoopPasses"/> times; every pass after the first restarts from the
/// layers before the loop and is gated on messages from the pass before it,
/// with its own keys, values and convolution states.
/// </summary>
public sealed class Inkling2Facts
{
    /// <summary>Per layer: true for full attention (<c>hybrid</c>), false for a
    /// sliding window (<c>hybrid_sliding</c>).</summary>
    public required IReadOnlyList<bool> FullLayers { get; init; }

    public required int FullHeads { get; init; }
    public required int FullKvHeads { get; init; }
    public required int FullHeadDim { get; init; }
    public required int SlidingHeads { get; init; }
    public required int SlidingKvHeads { get; init; }
    public required int SlidingHeadDim { get; init; }
    public required int SlidingWindow { get; init; }

    /// <summary>Width per head of the states that mix the relative bias profiles.</summary>
    public required int RelativeDim { get; init; }

    /// <summary>How far back the full layers' relative bias reaches; a sliding
    /// layer's reaches its window.</summary>
    public required int RelativeExtent { get; init; }

    /// <summary>Full layers scale queries and relative states by
    /// <c>1 + alpha·ln(max(1, (pos+1)/floor))</c>.</summary>
    public required int LogScalingFloor { get; init; }
    public required double LogScalingAlpha { get; init; }

    public required double QkClip { get; init; }
    public required double RelativeClip { get; init; }
    public required int ConvKernel { get; init; }

    /// <summary>Layers per block of the attention residuals.</summary>
    public required int BlockSize { get; init; }

    public required int Experts { get; init; }
    public required int TopK { get; init; }
    public required int ExpertWidth { get; init; }
    public required double RouteScale { get; init; }
    public required double GateSoftcap { get; init; }
    public required double UpSoftcap { get; init; }
    public required double FinalSoftcap { get; init; }

    public required int LoopStart { get; init; }
    public required int LoopEnd { get; init; }
    public required int LoopPasses { get; init; }
    public required int GateRank { get; init; }

    public required double Eps { get; init; }

    /// <summary>Rows of the embedding and head; logits past
    /// <see cref="UnpaddedVocab"/> are padding.</summary>
    public required int PaddedVocab { get; init; }
    public required int UnpaddedVocab { get; init; }
}
