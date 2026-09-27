using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// The export side of ModernBERT / Von: regenerates a <c>ModernBertModel</c>
/// <c>config.json</c> from the container's judged facts. Kept apart from
/// <see cref="ExportConfig"/>, whose field set is the causal decoder's —
/// ModernBERT spells the same facts differently (<c>hidden_activation</c>,
/// <c>norm_eps</c>) and adds ones a decoder has no word for (the local window,
/// the two RoPE bases, the bias switches).
/// </summary>
internal static class ModernBertLayout
{
    public static bool Applies(Vindex3Container container) =>
        container.Index.Family == ModernBert.ModelType;

    public static string BuildConfigJson(Vindex3Container container)
    {
        var graph = container.Graph
            ?? throw new CliException("container records no system graph — cannot regenerate config.json");
        var component = graph.Components.First(c => c.Role == ComponentRole.PrimaryText);
        var surface = component.Execution
            ?? throw new CliException("modernbert component declares no execution surface — cannot regenerate config.json");
        var encoder = surface.Encoder
            ?? throw new CliException("modernbert component declares no encoder surface — cannot regenerate config.json");
        var attention = surface.Attention!;
        var ffn = surface.Ffn!;
        var policies = component.Attention
            ?? throw new CliException("modernbert component declares no per-layer attention table");

        var layerTypes = new List<string>(policies.Count);
        double? globalTheta = null, localTheta = null;
        long? window = null;
        foreach (var (policy, l) in policies.Select((p, i) => (p, i)))
        {
            if (policy.Operator != LayerOperators.Softmax || policy.Position is not PositionRope rope)
            {
                throw new CliException($"layer {l}: not a RoPE softmax layer — has no modernbert spelling");
            }
            if (policy.Span == AttentionSpan.Full)
            {
                layerTypes.Add("full_attention");
                globalTheta ??= rope.Theta;
                if (globalTheta != rope.Theta)
                {
                    throw new CliException($"layer {l}: global layers disagree on the RoPE base — modernbert has one");
                }
            }
            else if (policy.Span == AttentionSpan.Sliding && policy.Window is { } w)
            {
                layerTypes.Add("sliding_attention");
                localTheta ??= rope.Theta;
                window ??= w;
                if (localTheta != rope.Theta || window != w)
                {
                    throw new CliException($"layer {l}: sliding layers disagree on window or RoPE base — modernbert has one of each");
                }
            }
            else
            {
                throw new CliException($"layer {l}: span '{policy.Span}' has no modernbert spelling");
            }
        }

        // 4.x transformers reads the alternation as a period, not a table, so
        // the export only stays loadable there if the table IS periodic.
        int? period = null;
        for (int n = 1; n <= layerTypes.Count && period is null; n++)
        {
            if (layerTypes.Select((t, i) => (t == "full_attention") == (i % n == 0)).All(x => x))
            {
                period = n;
            }
        }
        if (period is null)
        {
            throw new CliException(
                "the layer table is not 'global every n layers' — no modernbert config can express it (was a layer pruned?)");
        }
        if (globalTheta is null)
        {
            throw new CliException("no global attention layer — modernbert requires layer 0 to be global");
        }

        string hiddenActivation = ffn.Activation switch
        {
            Activation.Gelu => "gelu",
            var other => throw new CliException($"FFN activation '{other}' has no modernbert spelling"),
        };

        var ropeParameters = new JsonObject
        {
            ["full_attention"] = new JsonObject { ["rope_theta"] = globalTheta, ["rope_type"] = "default" },
        };
        if (localTheta is not null)
        {
            ropeParameters["sliding_attention"] = new JsonObject { ["rope_theta"] = localTheta, ["rope_type"] = "default" };
        }

        string dtype = container.Index.Representations.Values.First().Encoding switch
        {
            "F32" => "float32",
            "BF16" => "bfloat16",
            "F16" => "float16",
            var other => throw new CliException($"encoding '{other}' has no torch dtype spelling"),
        };

        var config = new JsonObject
        {
            ["architectures"] = new JsonArray(JsonValue.Create(encoder.Architecture)!),
            ["model_type"] = ModernBert.ModelType,
            ["dtype"] = dtype,
            ["hidden_size"] = component.HiddenSize,
            ["num_hidden_layers"] = component.NumLayers,
            ["num_attention_heads"] = attention.NumQHeads,
            ["intermediate_size"] = ffn.IntermediateSize,
            ["hidden_activation"] = hiddenActivation,
            ["max_position_embeddings"] = surface.ContextLength,
            ["vocab_size"] = surface.Head!.VocabSize,
            ["norm_eps"] = surface.Norm.Pre.Eps,
            ["layer_norm_eps"] = surface.Norm.Pre.Eps,
            ["norm_bias"] = encoder.NormBias,
            ["mlp_bias"] = encoder.MlpBias,
            ["attention_bias"] = attention.AttentionBias == true,
            ["tie_word_embeddings"] = surface.Head.HeadReusesEmbedding,
            ["layer_types"] = new JsonArray(layerTypes.Select(t => JsonValue.Create(t)).ToArray<JsonNode?>()),
            ["rope_parameters"] = ropeParameters,
            // The 4.x spellings of the same facts, so either major version loads it.
            ["global_attn_every_n_layers"] = period,
            ["global_rope_theta"] = globalTheta,
            ["local_attention"] = window ?? 128,
            ["local_rope_theta"] = localTheta ?? globalTheta,
            ["position_embedding_type"] = "absolute",
        };
        foreach (var (key, id) in encoder.SpecialTokenIds ?? new Dictionary<string, int>())
        {
            config[key] = id;
        }
        return config.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>The <c>option_marker.pt</c> key order torch would write:
    /// module registration order, which for ModernBertModel is embeddings,
    /// then each layer's attn_norm, attn, mlp_norm, mlp, then final_norm.</summary>
    public static int EncoderKeyRank(string backboneName)
    {
        if (backboneName.StartsWith("embeddings.", StringComparison.Ordinal))
        {
            return backboneName.Contains("tok_embeddings") ? 0 : 1;
        }
        if (backboneName.StartsWith("final_norm.", StringComparison.Ordinal))
        {
            return int.MaxValue;
        }
        var parts = backboneName.Split('.');
        if (parts.Length >= 3 && parts[0] == "layers" &&
            int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int layer))
        {
            int within = Array.IndexOf(
                new[] { "attn_norm", "attn.Wqkv", "attn.Wo", "mlp_norm", "mlp.Wi", "mlp.Wo" },
                string.Join('.', parts.Skip(2).Take(parts.Length - 3)));
            return 2 + layer * 16 + (within < 0 ? 15 : within);
        }
        return int.MaxValue - 1;
    }
}
