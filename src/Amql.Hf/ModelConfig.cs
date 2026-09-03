using System.Text.Json;

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
    LinearAttentionFacts? LinearAttention)
{
    /// <summary>Rotary subspace: the partial factor (0.25 for this family)
    /// times the head dim — the width the MRoPE sections rotate.</summary>
    public double PartialRotaryFactor
    {
        get
        {
            if (RopeParameters.ValueKind == JsonValueKind.Object &&
                RopeParameters.TryGetProperty("partial_rotary_factor", out var pf))
            {
                return pf.GetDouble();
            }
            return 1.0;
        }
    }
}

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
                LinearAttention: linear);
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