using System.Text.Json;

using Amql.Vindex3;

namespace Amql.Hf;

/// <summary>Raised when an HF checkpoint's config declares facts this build
/// cannot faithfully map into a VINDEX3 graph. Mirrors the reference's
/// G1 stage: refuse the checkpoint, never approximate an architecture.</summary>
public sealed class ModelConfigException : Exception
{
    public ModelConfigException(string message) : base(message) { }

    public ModelConfigException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>Geometry of the persisted linear-attention surface.</summary>
public sealed record LinearAttentionFacts(
    int ConvKernelDim,
    int KeyHeads,
    int KeyHeadDim,
    int ValueHeads,
    int ValueHeadDim);

/// <summary>The <c>moe</c> config block an MoE-ified export carries — the
/// facts that make a re-encoded container judge its FFNs routed instead
/// of dense.</summary>
public sealed record MoeFacts(
    int Experts,
    int TopK,
    int ExpertIntermediateSize,
    ExpertRoutingPolicy RoutingPolicy);

/// <summary>The wrapper config's <c>vision_config</c> dimensionality — the
/// tower's judged facts (read from the checkpoint, never invented).</summary>
public sealed record VisionFacts(int HiddenSize, int NumLayers);

/// <summary>
/// G1 output: architecture facts lifted from <c>config.json</c> — the
/// read-only inputs the graph/surface builder turns into a system graph.
/// A multimodal wrapper config (<c>Qwen3_5ForConditionalGeneration</c>)
/// carries the text stack in <c>text_config</c>; that nesting is unwrapped
/// here so the mapper never sees it.
/// </summary>
public sealed record TextArchitectureFacts(
    string ModelType,
    int HiddenSize,
    int NumLayers,
    int NumQueryHeads,
    int NumKvHeads,
    int HeadDim,
    int IntermediateSize,
    string HiddenAct,
    double RmsNormEps,
    int VocabSize,
    bool TieWordEmbeddings,
    bool AttentionBias,
    bool AttentionOutputGate,
    long MaxPositionEmbeddings,
    IReadOnlyList<string> LayerTypes,
    JsonElement RopeParameters,
    LinearAttentionFacts? LinearAttention,
    double PartialRotaryFactor,
    MoeFacts? Moe,
    VisionFacts? Vision);

/// <summary>G1 reader: <c>config.json</c> → <see cref="TextArchitectureFacts"/>.</summary>
public static class ModelConfig
{
    public static TextArchitectureFacts ReadTextFacts(string configPath)
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        }
        catch (JsonException e)
        {
            throw new ModelConfigException($"'{configPath}' is not valid JSON: {e.Message}", e);
        }
        catch (IOException e)
        {
            throw new ModelConfigException($"cannot read '{configPath}': {e.Message}", e);
        }

        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new ModelConfigException($"'{configPath}': config root is not an object");
            }

            // Unwrap a multimodal wrapper: the text stack always lives in
            // text_config; a plain text config IS the text config.
            JsonElement text = root.TryGetProperty("text_config", out var nested) && nested.ValueKind == JsonValueKind.Object
                ? nested
                : root;

            var layerTypes = new List<string>();
            if (text.TryGetProperty("layer_types", out var types) && types.ValueKind == JsonValueKind.Array)
            {
                layerTypes.AddRange(types.EnumerateArray().Select(t => t.GetString() ?? string.Empty));
            }
            if (layerTypes.Count == 0)
            {
                throw new ModelConfigException(
                    "'layer_types' is absent or empty — this build refuses to guess a per-layer operator table");
            }

            JsonElement rope = text.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object
                ? rp.Clone()
                : JsonDocument.Parse("{\"rope_type\":\"default\"}").RootElement.Clone();

            LinearAttentionFacts? linear = null;
            if (text.TryGetProperty("linear_conv_kernel_dim", out var ck) &&
                text.TryGetProperty("linear_num_key_heads", out var khe) &&
                text.TryGetProperty("linear_key_head_dim", out var khd) &&
                text.TryGetProperty("linear_num_value_heads", out var vhe) &&
                text.TryGetProperty("linear_value_head_dim", out var vhd))
            {
                linear = new LinearAttentionFacts(
                    ck.GetInt32(), khe.GetInt32(), khd.GetInt32(), vhe.GetInt32(), vhd.GetInt32());
            }

            MoeFacts? moe = null;
            if (text.TryGetProperty("moe", out var moeElement) &&
                moeElement.ValueKind == JsonValueKind.Object &&
                moeElement.TryGetProperty("experts", out var expertsEl) &&
                moeElement.TryGetProperty("top_k", out var topKEl) &&
                moeElement.TryGetProperty("expert_intermediate_size", out var eisEl))
            {
                ExpertRoutingPolicy policy = moeElement.TryGetProperty("routing_policy", out var policyEl) &&
                    policyEl.GetString() == "normalised_over_selected"
                    ? ExpertRoutingPolicy.NormalisedOverSelected
                    : ExpertRoutingPolicy.SoftmaxThenSelect;
                moe = new MoeFacts(expertsEl.GetInt32(), topKEl.GetInt32(), eisEl.GetInt32(), policy);
            }

            VisionFacts? vision = null;
            if (root.TryGetProperty("vision_config", out var visionElement) &&
                visionElement.ValueKind == JsonValueKind.Object)
            {
                int? vHidden = visionElement.TryGetProperty("hidden_size", out var vh) ? vh.GetInt32() : null;
                // The Qwen3.5 vision config names its depth field "depth"
                // ("num_hidden_layers" is the standard HF spelling).
                int? vLayers = visionElement.TryGetProperty("num_hidden_layers", out var vl)
                    ? vl.GetInt32()
                    : visionElement.TryGetProperty("depth", out var vd)
                        ? vd.GetInt32()
                        : null;
                if (vHidden is int vh2 && vLayers is int vl2)
                {
                    vision = new VisionFacts(vh2, vl2);
                }
            }

            // Partial rotary factor, in the reference's precedence
            // (`PreTrainedConfig::standardize_rope_params`, transformers 5.x):
            //  1. top-level `partial_rotary_factor` — the legacy flat form,
            //     which the reference copies INTO rope_parameters;
            //  2. rope_parameters.full_attention.partial_rotary_factor — the
            //     per-layer-type form (Gemma 4);
            //  3. rope_parameters.partial_rotary_factor — the 5.x flat form.
            // Qwen3.8 writes forms 1 and 3 together (both 0.25); reading form
            // 1 first matches the reference exactly.
            double partialRotaryFactor = 1.0;
            if (text.TryGetProperty("partial_rotary_factor", out var topLevelPf))
            {
                partialRotaryFactor = topLevelPf.GetDouble();
            }
            else if (rope.TryGetProperty("full_attention", out var fa) &&
                     fa.ValueKind == JsonValueKind.Object &&
                     fa.TryGetProperty("partial_rotary_factor", out var faPf))
            {
                partialRotaryFactor = faPf.GetDouble();
            }
            else if (rope.TryGetProperty("partial_rotary_factor", out var flatPf))
            {
                partialRotaryFactor = flatPf.GetDouble();
            }

            return new TextArchitectureFacts(
                ModelType: text.GetProperty("model_type").GetString() ?? "unknown",
                HiddenSize: Int(text, "hidden_size"),
                NumLayers: Int(text, "num_hidden_layers"),
                NumQueryHeads: Int(text, "num_attention_heads"),
                NumKvHeads: Int(text, "num_key_value_heads", required: false),
                HeadDim: Int(text, "head_dim"),
                IntermediateSize: Int(text, "intermediate_size"),
                HiddenAct: text.TryGetProperty("hidden_act", out var act) ? act.GetString() ?? "silu" : "silu",
                RmsNormEps: text.TryGetProperty("rms_norm_eps", out var eps) ? eps.GetDouble() : 1e-6,
                VocabSize: Int(text, "vocab_size"),
                TieWordEmbeddings: text.TryGetProperty("tie_word_embeddings", out var tie) && tie.GetBoolean(),
                AttentionBias: text.TryGetProperty("attention_bias", out var ab) && ab.GetBoolean(),
                AttentionOutputGate: text.TryGetProperty("attn_output_gate", out var og) && og.GetBoolean(),
                MaxPositionEmbeddings: Long(text, "max_position_embeddings"),
                LayerTypes: layerTypes,
                RopeParameters: rope,
                LinearAttention: linear,
                PartialRotaryFactor: partialRotaryFactor,
                Moe: moe,
                Vision: vision);
        }
    }

    private static int Int(JsonElement obj, string name, bool required = true)
    {
        if (obj.TryGetProperty(name, out var value))
        {
            return value.GetInt32();
        }
        if (required)
        {
            throw new ModelConfigException($"config is missing required field '{name}'");
        }
        return 0;
    }

    private static long Long(JsonElement obj, string name)
    {
        if (obj.TryGetProperty(name, out var value))
        {
            return value.GetInt64();
        }
        throw new ModelConfigException($"config is missing required field '{name}'");
    }
}