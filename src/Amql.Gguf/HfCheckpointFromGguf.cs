using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Gguf;

/// <summary>
/// Converts a GGUF file into an HF checkpoint directory (config.json +
/// model.safetensors + tokenizer.json), dequantising quantised tensor
/// formats to F32 on the fly. The resulting checkpoint can then be
/// encoded into a VINDEX3 container via ModelToContainer.Encode.
/// </summary>
public static class HfCheckpointFromGguf
{
    /// <summary>
    /// Produce a temporary HF checkpoint from a GGUF file. Returns the
    /// checkpoint directory path (ready for ModelToContainer.Encode).
    /// </summary>
    public static string Convert(string ggufPath, string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        using var gguf = GgufReader.Open(ggufPath);

        string arch = gguf.Arch;
        if (string.IsNullOrEmpty(arch))
            throw new GgufException("GGUF file has no general.architecture key");

        bool qwen35 = arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3" || arch == "qwen3moe";
        if (!qwen35)
            throw new GgufException($"architecture '{arch}' is not yet supported by from-gguf. Supported: qwen35, qwen35moe, qwen3, qwen3moe");

        var facts = ReadFacts(gguf);
        WriteConfig(outputDir, facts);
        WriteTokenizer(outputDir, gguf);
        WriteSafetensors(outputDir, gguf, facts);
        return outputDir;
    }

    private sealed class ModelFacts
    {
        public string ModelType = "";
        public int HiddenSize, NumLayers, NumAttentionHeads, NumKvHeads, HeadDim;
        public int IntermediateSize, VocabSize, MaxPositionEmbeddings;
        public float RmsNormEps = 1e-6f;
        public bool AttnOutputGate;
        public float RopeTheta = 10_000_000f;
        public float PartialRotaryFactor;
        public readonly List<string> LayerTypes = new();
        public bool HasMoe;
        public int Experts, TopK, ExpertIntermediate;
        public int LinearKeyHeads, LinearKeyHeadDim, LinearValueHeads, LinearValueHeadDim, LinearConvKernel;
        public bool HasLinearLayers;
    }

    private static ModelFacts ReadFacts(GgufReader gguf)
    {
        var f = new ModelFacts();
        string arch = gguf.Arch;

        f.HiddenSize = GetI32(gguf, $"{arch}.embedding_length");
        f.NumLayers = GetI32(gguf, $"{arch}.block_count");
        f.NumAttentionHeads = GetI32(gguf, $"{arch}.attention.head_count");
        f.NumKvHeads = TryI32(gguf, $"{arch}.attention.head_count_kv") ?? f.NumAttentionHeads;
        f.HeadDim = f.HiddenSize / f.NumAttentionHeads;
        f.IntermediateSize = GetI32(gguf, $"{arch}.feed_forward_length");
        f.VocabSize = GetI32(gguf, $"{arch}.vocab_size");
        f.MaxPositionEmbeddings = TryI32(gguf, $"{arch}.context_length")
            ?? TryI32(gguf, $"{arch}.rope.dimension_count") ?? 32768;
        f.RmsNormEps = TryF32(gguf, $"{arch}.attention.layer_norm_epsilon") ?? 1e-6f;
        f.AttnOutputGate = TryBool(gguf, $"{arch}.attention.out_gate") ?? false;
        f.RopeTheta = TryF32(gguf, $"{arch}.rope.freq_base") ?? 10_000_000f;

        if (TryF32(gguf, $"{arch}.rope.partial_rotary_factor") is { } prf)
            f.PartialRotaryFactor = prf;

        // Detect layer types from the tensor name table: if a layer has
        // attn_qkv.weight, it's linear_attention; if attn_q.weight, full.
        f.LayerTypes.Clear();
        for (int i = 0; i < f.NumLayers; i++)
            f.LayerTypes.Add("full_attention");

        foreach (var t in gguf.Tensors)
        {
            if (!t.Name.StartsWith("blk.", StringComparison.Ordinal)) continue;
            string rest = t.Name[4..];
            int dot = rest.IndexOf('.');
            if (dot < 0) continue;
            if (!int.TryParse(rest[..dot], out int layer)) continue;
            string tail = rest[(dot + 1)..];
            if (tail.StartsWith("attn_qkv", StringComparison.Ordinal)
                || tail.StartsWith("attn_gate", StringComparison.Ordinal)
                || tail.StartsWith("ssm_", StringComparison.Ordinal))
            {
                f.LayerTypes[layer] = "linear_attention";
                f.HasLinearLayers = true;
            }
        }

        // MoE.
        if (TryI32(gguf, $"{arch}.expert_count") is { } experts and > 0)
        {
            f.HasMoe = true;
            f.Experts = experts;
            f.TopK = TryI32(gguf, $"{arch}.expert_feed_forward_count") ?? 2;
            f.ExpertIntermediate = TryI32(gguf, $"{arch}.expert_feed_forward_length") ?? f.IntermediateSize;
        }

        // Linear (GDN) head config.
        f.LinearKeyHeads = TryI32(gguf, $"{arch}.ssm.head_count_k") ?? 0;
        f.LinearKeyHeadDim = TryI32(gguf, $"{arch}.ssm.head_dim_k") ?? 0;
        f.LinearValueHeads = TryI32(gguf, $"{arch}.ssm.head_count_v") ?? 0;
        f.LinearValueHeadDim = TryI32(gguf, $"{arch}.ssm.head_dim_v") ?? 0;
        f.LinearConvKernel = TryI32(gguf, $"{arch}.ssm.conv_kernel") ?? 0;

        f.ModelType = "qwen3_5_text";
        return f;
    }

    private static int GetI32(GgufReader gguf, string key)
    {
        var v = gguf.Get(key);
        if (v.Kind == GgufValueType.Uint32 || v.Kind == GgufValueType.Int32)
            return (int)v.AsInt32();
        return (int)v.AsInt64();
    }

    private static int? TryI32(GgufReader gguf, string key)
    {
        if (!gguf.TryGet(key, out var v)) return null;
        if (v.Kind == GgufValueType.Uint32 || v.Kind == GgufValueType.Int32)
            return (int)v.AsInt32();
        return (int)v.AsInt64();
    }

    private static float? TryF32(GgufReader gguf, string key) =>
        gguf.TryGet(key, out var v) ? v.AsFloat() : null;

    private static bool? TryBool(GgufReader gguf, string key) =>
        gguf.TryGet(key, out var v) ? v.AsBool() : null;

    private static void WriteConfig(string dir, ModelFacts f)
    {
        var config = new Dictionary<string, object>
        {
            ["model_type"] = f.ModelType,
            ["hidden_size"] = f.HiddenSize,
            ["num_hidden_layers"] = f.NumLayers,
            ["num_attention_heads"] = f.NumAttentionHeads,
            ["num_key_value_heads"] = f.NumKvHeads,
            ["head_dim"] = f.HeadDim,
            ["intermediate_size"] = f.IntermediateSize,
            ["vocab_size"] = f.VocabSize,
            ["max_position_embeddings"] = f.MaxPositionEmbeddings,
            ["rms_norm_eps"] = f.RmsNormEps,
            ["attn_output_gate"] = f.AttnOutputGate,
            ["layer_types"] = f.LayerTypes,
            ["rope_parameters"] = new Dictionary<string, object>
            {
                ["rope_theta"] = f.RopeTheta,
                ["partial_rotary_factor"] = f.PartialRotaryFactor,
            },
            ["tie_word_embeddings"] = false,
            ["torch_dtype"] = "float32",
        };

        if (f.LinearKeyHeads > 0 || f.HasLinearLayers)
        {
            // The GGUF metadata may not carry the separate linear-attention
            // head dimensions.  Use the architecture-level defaults that
            // match real Qwen3.5 configs: key heads match KV heads, value
            // heads default to 2, both at headDim (or 128 if headDim < 128).
            int lkHeads = f.LinearKeyHeads > 0 ? f.LinearKeyHeads : f.NumKvHeads;
            int lkHeadDim = f.LinearKeyHeadDim > 0 ? f.LinearKeyHeadDim
                : (f.HeadDim >= 128 ? f.HeadDim : 128);
            int lvHeads = f.LinearValueHeads > 0 ? f.LinearValueHeads : Math.Min(2, f.NumKvHeads);
            int lvHeadDim = f.LinearValueHeadDim > 0 ? f.LinearValueHeadDim : lkHeadDim;
            int lcKernel = f.LinearConvKernel > 0 ? f.LinearConvKernel : 4;
            config["linear_num_key_heads"] = lkHeads;
            config["linear_key_head_dim"] = lkHeadDim;
            config["linear_num_value_heads"] = lvHeads;
            config["linear_value_head_dim"] = lvHeadDim;
            config["linear_conv_kernel_dim"] = lcKernel;
        }

        if (f.HasMoe)
        {
            config["moe"] = new Dictionary<string, object>
            {
                ["experts"] = f.Experts,
                ["top_k"] = f.TopK,
                ["expert_intermediate_size"] = f.ExpertIntermediate,
            };
        }

        File.WriteAllText(Path.Combine(dir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void WriteTokenizer(string dir, GgufReader gguf)
    {
        // Extract tokenizer from GGUF tokenizer metadata.
        var tokenizer = new Dictionary<string, object>
        {
            ["version"] = "1.0",
        };

        string modelType = "BPE";
        if (gguf.TryGet("tokenizer.ggml.model", out var tm))
        {
            string m = tm.AsString();
            if (m == "gpt2") modelType = "BPE";
            else modelType = m;
        }

        // Build the vocab from GGUF tokenizer arrays.
        var vocab = new Dictionary<string, int>();
        if (gguf.TryGet("tokenizer.ggml.tokens", out var tokens))
        {
            var (_, items) = tokens.AsArray();
            for (int i = 0; i < items.Count; i++)
            {
                string tok = (string)items[i];
                vocab[tok] = i;
            }
        }

        var modelObj = new Dictionary<string, object>
        {
            ["type"] = modelType,
            ["vocab"] = vocab,
            ["merges"] = Array.Empty<object>(),
        };

        // Merges for BPE tokenizers.
        if (gguf.TryGet("tokenizer.ggml.merges", out var mergesArr))
        {
            var merges = new List<string>();
            var (_, mergeItems) = mergesArr.AsArray();
            foreach (var item in mergeItems)
                merges.Add((string)item);
            modelObj["merges"] = merges;
        }

        tokenizer["model"] = modelObj;

        // Special tokens.
        var added = new List<object>();
        if (gguf.TryGet("tokenizer.ggml.bos_token_id", out var bos) && bos.AsInt32() is int bosId)
            added.Add(new Dictionary<string, object> { ["id"] = bosId, ["content"] = "<|beginoftext|>", ["special"] = true });
        if (gguf.TryGet("tokenizer.ggml.eos_token_id", out var eos) && eos.AsInt32() is int eosId)
            added.Add(new Dictionary<string, object> { ["id"] = eosId, ["content"] = "<|endoftext|>", ["special"] = true });
        tokenizer["added_tokens"] = added;

        File.WriteAllText(Path.Combine(dir, "tokenizer.json"),
            JsonSerializer.Serialize(tokenizer, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void WriteSafetensors(string dir, GgufReader gguf, ModelFacts facts)
    {
        string arch = gguf.Arch;
        var tensors = new List<(string Name, float[] Data, long[] Shape)>();

        foreach (var t in gguf.Tensors)
        {
            string name = t.Name;
            long[] shape = t.Dims;

            // GGUF shapes are reversed (fastest-varying first). HF convention
            // is slowest-varying first, so reverse the shape.
            long[] hfShape = shape.Reverse().ToArray();

            float[] data = ReadAndDequant(gguf, name);

            // Transpose 2-D tensors: GGUF is [out, in], HF is [in, out].
            if (hfShape.Length == 2 && hfShape[0] != hfShape[1])
            {
                int rows = (int)hfShape[0];
                int cols = (int)hfShape[1];
                var transposed = new float[data.Length];
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        transposed[c * rows + r] = data[r * cols + c];
                data = transposed;
            }

            // Map GGUF name → HF name.
            string? hfName = MapTensorName(name, facts);
            if (hfName is null)
            {
                Console.WriteLine($"warning: skipping unrecognised tensor '{name}'");
                continue;
            }

            tensors.Add((hfName, data, hfShape));
        }

        // Write safetensors.
        var header = new Dictionary<string, object>();
        long offset = 0;
        foreach (var (name, data, shape) in tensors)
        {
            int byteLen = data.Length * 4;
            header[name] = new Dictionary<string, object>
            {
                ["dtype"] = "F32",
                ["shape"] = shape,
                ["data_offsets"] = new[] { offset, offset + (long)byteLen },
            };
            offset += byteLen;
        }

        var headerJson = JsonSerializer.Serialize(header);
        byte[] headerBytes = System.Text.Encoding.UTF8.GetBytes(headerJson);
        // Safetensors convention: JSON header must be space-padded to a
        // multiple of 8 bytes. The header-length field is the padded length.
        int pad = (8 - headerBytes.Length % 8) % 8;
        if (pad > 0)
        {
            var padded = new byte[headerBytes.Length + pad];
            Array.Copy(headerBytes, padded, headerBytes.Length);
            for (int i = 0; i < pad; i++) padded[headerBytes.Length + i] = (byte)' ';
            headerBytes = padded;
        }
        long headerLen = headerBytes.Length;

        using var fs = new FileStream(Path.Combine(dir, "model.safetensors"), FileMode.Create, FileAccess.Write, FileShare.None);
        fs.Write(BitConverter.GetBytes(headerLen));
        fs.Write(headerBytes);

        foreach (var (_, data, _) in tensors)
        {
            byte[] bytes = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            fs.Write(bytes);
        }
    }

    private static float[] ReadAndDequant(GgufReader gguf, string name)
    {
        var t = gguf.GetTensor(name);
        byte[] packed = gguf.ReadBytes(name);
        long elements = 1;
        foreach (long d in t.Dims) elements *= d;
        return GgmlDequant.Dequant(t.Type, packed, (int)elements);
    }

    private static string? MapTensorName(string ggufName, ModelFacts facts)
    {
        // Embedding.
        if (ggufName == "token_embd.weight")
            return "model.language_model.embed_tokens.weight";

        // Output norm.
        if (ggufName == "output_norm.weight")
            return "model.language_model.norm.weight";

        // Output head.
        if (ggufName == "output.weight")
            return "lm_head.weight";

        // Layer-scoped tensors.
        foreach (var prefix in new[] { "blk." })
        {
            if (!ggufName.StartsWith(prefix, StringComparison.Ordinal)) continue;
            string rest = ggufName[prefix.Length..];
            int dot = rest.IndexOf('.');
            if (dot < 0) continue;
            if (!int.TryParse(rest[..dot], out int layer)) continue;
            string tail = rest[(dot + 1)..];
            string b = $"model.language_model.layers.{layer}.";

            // Detect layer type from the tensor name itself: linear layers
            // have attn_qkv / ssm_* tensors; full-attention layers have
            // attn_q / attn_k / attn_v.
            bool looksLinear = tail.StartsWith("attn_qkv", StringComparison.Ordinal)
                            || tail.StartsWith("attn_gate", StringComparison.Ordinal)
                            || tail.StartsWith("ssm_", StringComparison.Ordinal);

            if (looksLinear) return MapLinearTensor(b, tail);
            else return MapFullTensor(b, tail);
        }

        return null;
    }

    private static string? MapFullTensor(string prefix, string tail) => tail switch
    {
        "attn_norm.weight" => prefix + "input_layernorm.weight",
        "attn_q.weight" => prefix + "self_attn.q_proj.weight",
        "attn_k.weight" => prefix + "self_attn.k_proj.weight",
        "attn_v.weight" => prefix + "self_attn.v_proj.weight",
        "attn_q_norm.weight" => prefix + "self_attn.q_norm.weight",
        "attn_k_norm.weight" => prefix + "self_attn.k_norm.weight",
        "attn_output.weight" => prefix + "self_attn.o_proj.weight",
        "ffn_norm.weight" or "post_attention_norm.weight" => prefix + "post_attention_layernorm.weight",
        "ffn_gate.weight" => prefix + "mlp.gate_proj.weight",
        "ffn_up.weight" => prefix + "mlp.up_proj.weight",
        "ffn_down.weight" => prefix + "mlp.down_proj.weight",
        _ => null,
    };

    private static string? MapLinearTensor(string prefix, string tail) => tail switch
    {
        "attn_norm.weight" => prefix + "input_layernorm.weight",
        "attn_qkv.weight" => prefix + "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => prefix + "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => prefix + "linear_attn.in_proj_a.weight",
        "ssm_a" => prefix + "linear_attn.A_log",
        "ssm_beta.weight" or "ssm_b" => prefix + "linear_attn.in_proj_b.weight",
        "ssm_out.weight" => prefix + "linear_attn.out_proj.weight",
        "ssm_conv1d.weight" => prefix + "linear_attn.conv1d.weight",
        "ssm_dt.bias" => prefix + "linear_attn.dt_bias",
        "ssm_norm.weight" => prefix + "linear_attn.norm.weight",
        "post_attention_norm.weight" => prefix + "post_attention_layernorm.weight",
        "ffn_gate.weight" => prefix + "mlp.gate_proj.weight",
        "ffn_up.weight" => prefix + "mlp.up_proj.weight",
        "ffn_down.weight" => prefix + "mlp.down_proj.weight",
        _ => null,
    };
}