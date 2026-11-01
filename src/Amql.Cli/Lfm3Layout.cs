using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// The export side of LFM3: regenerates the <c>lfm3</c> text config — conv,
/// full and sliding layer types, the per-layer-type rope block, the sandwich
/// norm switch and the short-conv width — from the container's judged facts.
/// When the container carries the LFM3-VL wrapper (vision tower + projector),
/// the text config is nested back into the wrapper's own <c>config.json</c>,
/// which rides the perception component verbatim; without a materialised tower
/// the export is a plain <c>Lfm3ForCausalLM</c> whose tensors move from
/// <c>model.language_model.</c> to <c>model.</c>.
/// </summary>
internal static class Lfm3Layout
{
    public const string Family = "lfm3";
    private const string WrappedPrefix = "model.language_model.";

    public static bool Applies(Vindex3Container container) => container.Index.Family == Family;

    /// <summary>The wrapper config when the container exports as LFM3-VL
    /// (tower materialised, wrapper facts carried); null for text-only.</summary>
    public static JsonObject? Wrapper(Vindex3Container container)
    {
        var graph = container.Graph!;
        var vision = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.Perception);
        if (vision is null || !graph.Objects.Any(o => o.Component == vision.Id && o.Representations.Count > 0))
        {
            return null;
        }
        if (vision.Perception is not { ValueKind: JsonValueKind.Object } perception ||
            !perception.TryGetProperty("hf_wrapper", out var wrapper))
        {
            throw new CliException("lfm3 container materialises a vision tower but carries no wrapper config — cannot rebuild config.json");
        }
        return JsonNode.Parse(wrapper.GetRawText())!.AsObject();
    }

    /// <summary>Tensor names for a text-only export: the decoder leaves the
    /// VL wrapper's <c>language_model</c> scope.</summary>
    public static string RemapTensorName(string hfName, bool textOnly) =>
        textOnly && hfName.StartsWith(WrappedPrefix, StringComparison.Ordinal)
            ? "model." + hfName[WrappedPrefix.Length..]
            : hfName;

    public static string BuildConfigJson(Vindex3Container container, OperandStore store)
    {
        var graph = container.Graph
            ?? throw new CliException("container records no system graph — cannot regenerate config.json");
        var component = graph.Components.First(c => c.Role == ComponentRole.PrimaryText);
        var surface = component.Execution
            ?? throw new CliException("lfm3 component declares no execution surface — cannot regenerate config.json");
        var attention = surface.Attention!;
        var ffn = surface.Ffn!;
        var head = surface.Head!;
        var policies = component.Attention
            ?? throw new CliException("lfm3 component declares no per-layer attention table");
        var stack = graph.Objects.First(o => o.Component == component.Id && o.Kind == ObjectKind.DecoderStack);

        var layerTypes = new List<string>(policies.Count);
        var rope = new JsonObject();
        long? window = null;
        int? convWidth = null;
        bool convBias = false;
        for (int l = 0; l < policies.Count; l++)
        {
            var policy = policies[l];
            switch (policy.Operator)
            {
                case LayerOperators.Conv:
                    layerTypes.Add("conv");
                    var shape = store.ShapeOf(stack.Id, $"{l}.conv.conv.weight");
                    convWidth ??= checked((int)shape[^1]);
                    if (convWidth != shape[^1])
                    {
                        throw new CliException($"layer {l}: conv width {shape[^1]} differs from {convWidth} — lfm3 has one conv_L_cache");
                    }
                    convBias |= store.ContainsTensor(stack.Id, $"{l}.conv.conv.bias");
                    break;
                case LayerOperators.Softmax:
                    bool sliding = policy.Span == AttentionSpan.Sliding;
                    string type = sliding ? "sliding_attention" : "full_attention";
                    layerTypes.Add(type);
                    if (sliding)
                    {
                        window ??= policy.Window;
                        if (window != policy.Window)
                        {
                            throw new CliException($"layer {l}: sliding window {policy.Window} differs from {window} — lfm3 has one");
                        }
                    }
                    var entry = policy.Position switch
                    {
                        PositionZeroTailRope zt => new JsonObject
                        {
                            ["partial_rotary_factor"] = zt.RotaryFactor,
                            ["rope_theta"] = zt.Theta,
                            ["rope_type"] = "default",
                        },
                        PositionRope r => new JsonObject { ["rope_theta"] = r.Theta, ["rope_type"] = "default" },
                        var other => throw new CliException(
                            $"layer {l}: position policy '{other.GetType().Name}' has no lfm3 rope spelling"),
                    };
                    if (rope[type] is { } seen && seen.ToJsonString() != entry.ToJsonString())
                    {
                        throw new CliException($"layer {l}: '{type}' layers disagree on their rope facts — lfm3 keys rope by layer type");
                    }
                    rope[type] = entry;
                    break;
                default:
                    throw new CliException($"layer {l}: operator '{policy.Operator}' has no lfm3 layer_types spelling");
            }
        }

        var wrapper = Wrapper(container);
        var text = new JsonObject
        {
            ["model_type"] = Family,
            ["hidden_size"] = component.HiddenSize,
            ["intermediate_size"] = ffn.IntermediateSize,
            ["num_hidden_layers"] = component.NumLayers,
            ["num_attention_heads"] = attention.NumQHeads,
            ["num_key_value_heads"] = attention.NumKvHeads,
            ["head_dim"] = attention.HeadDim,
            ["layer_types"] = new JsonArray(layerTypes.Select(t => (JsonNode)JsonValue.Create(t)!).ToArray()),
            ["max_position_embeddings"] = surface.ContextLength,
            ["norm_eps"] = surface.Norm.Pre.Eps,
            ["vocab_size"] = head.VocabSize,
            ["tie_word_embeddings"] = head.HeadReusesEmbedding,
            ["use_sandwich_norm"] = surface.Norm.Placement == NormPlacement.Sandwich,
            ["rope_parameters"] = rope,
            ["use_cache"] = true,
        };
        if (window is { } w)
        {
            text["sliding_window"] = w;
        }
        if (convWidth is { } k)
        {
            text["conv_L_cache"] = k;
            text["conv_bias"] = convBias;
        }
        if (ffn.Moe is { } moe)
        {
            // AMQL's own block, as ExportConfig writes it: a re-encode judges
            // the FFNs routed. Stock LFM3 has no MoE spelling.
            text["moe"] = new JsonObject
            {
                ["experts"] = moe.Experts,
                ["top_k"] = moe.TopK,
                ["expert_intermediate_size"] = moe.ExpertIntermediateSize,
                ["routing_policy"] = moe.RoutingPolicy == ExpertRoutingPolicy.NormalisedOverSelected
                    ? "normalised_over_selected"
                    : "softmax_then_select",
            };
        }

        JsonObject config;
        if (wrapper is not null)
        {
            foreach (var key in new[] { "bos_token_id", "eos_token_id", "pad_token_id" })
            {
                if (wrapper[key] is { } id)
                {
                    text[key] = id.DeepClone();
                }
            }
            config = wrapper;
            config["text_config"] = text;
        }
        else
        {
            config = text;
            config["architectures"] = new JsonArray(JsonValue.Create("Lfm3ForCausalLM"));
        }
        return config.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }
}
