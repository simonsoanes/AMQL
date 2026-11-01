using System.Text.Json;

namespace Amql.Gguf;

/// <summary>
/// dLLM A2D-Qwen3 (masked diffusion Qwen3) checkpoint → GGUF for llama.cpp's
/// <c>rnd1</c> architecture, which <c>llama-diffusion-cli</c> runs. RND1 is a
/// Qwen3-MoE stack with non-causal attention; a dense Qwen3 FFN is written as
/// its one-expert special case — gate/up/down become <c>*_exps</c> tensors
/// with an expert axis of 1 and the router is a zero row. The softmax over a
/// single router logit is exactly 1, so the routed FFN computes the dense one
/// bit for bit, whatever the router holds.
/// <para>
/// <c>diffusion.shift_logits</c> is written false (the A2D logits predict the
/// token at their own position; llama.cpp assumes a shift when the key is
/// absent) and the mask token id comes from the tokenizer files.
/// </para>
/// </summary>
public static class A2dQwen3Gguf
{
    public const string Arch = "rnd1";
    public const string ModelType = "a2d-qwen3";

    public static bool Applies(string checkpointDir)
    {
        string configPath = Path.Combine(checkpointDir, "config.json");
        if (!File.Exists(configPath))
        {
            return false;
        }
        using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
        return doc.RootElement.TryGetProperty("model_type", out var mt) && mt.GetString() == ModelType;
    }

    public static GgufConversion Convert(string checkpointDir, string outFile, string quantization = "none")
    {
        if (!Lfm3Gguf.Quantizations.Contains(quantization))
        {
            throw new GgufException(
                $"unknown quantization '{quantization}' for {ModelType} — expected one of: {string.Join(", ", Lfm3Gguf.Quantizations)}");
        }
        return Convert(checkpointDir, outFile, t => quantization switch
        {
            "none" => GgufType.BF16,
            "f16" => GgufType.F16,
            _ when t.Name == "token_embd.weight" => GgufType.BF16,
            "q8_0" => GgufType.Q8_0,
            _ => GgufType.Mxfp4,
        }, quantization);
    }

    private static GgufConversion Convert(string checkpointDir, string outFile,
        Func<(string Name, long[] Shape), GgufType> typeFor, string label)
    {
        using var configDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpointDir, "config.json")));
        var root = configDoc.RootElement;
        int hidden = Int(root, "hidden_size");
        int layers = Int(root, "num_hidden_layers");
        int heads = Int(root, "num_attention_heads");
        int kvHeads = Int(root, "num_key_value_heads");
        int headDim = root.TryGetProperty("head_dim", out var hd) && hd.ValueKind == JsonValueKind.Number ? hd.GetInt32() : hidden / heads;
        int intermediate = Int(root, "intermediate_size");
        int vocab = Int(root, "vocab_size");
        int context = Int(root, "max_position_embeddings");
        float eps = root.GetProperty("rms_norm_eps").GetSingle();
        double theta = root.TryGetProperty("rope_theta", out var rt) && rt.ValueKind == JsonValueKind.Number
            ? rt.GetDouble()
            : root.TryGetProperty("rope_parameters", out var rp) && rp.TryGetProperty("rope_theta", out var rpt)
                ? rpt.GetDouble()
                : throw new GgufException($"{ModelType} config carries no rope_theta");
        if (root.TryGetProperty("rope_scaling", out var scaling) && scaling.ValueKind == JsonValueKind.Object)
        {
            throw new GgufException($"{ModelType} with rope_scaling has no judged GGUF spelling");
        }
        int maskId = MaskTokenId(checkpointDir);

        using var shards = new Lfm3Gguf.ShardSet(checkpointDir);
        var plan = new List<Lfm3Gguf.Entry>();
        void Weight(string hf, string gguf, long[]? shape = null)
        {
            shape ??= shards.Info(hf).Shape;
            var type = typeFor((gguf, shape));
            if (GgufTypeSizing.IsBlockQuantized(type) && shape[^1] % GgufTypeSizing.BlockElements(type) != 0)
            {
                type = GgufType.BF16;
            }
            plan.Add(new Lfm3Gguf.Entry(gguf, shape, type, hf, null));
        }
        void Norm(string hf, string gguf) =>
            plan.Add(new Lfm3Gguf.Entry(gguf, shards.Info(hf).Shape, GgufType.F32, hf, null));

        Weight("model.embed_tokens.weight", "token_embd.weight");
        Norm("model.norm.weight", "output_norm.weight");
        // A tied checkpoint may still store lm_head (this one does, as a copy of
        // the embedding); llama.cpp falls back to the embedding when it is absent.
        bool tied = root.TryGetProperty("tie_word_embeddings", out var tie) && tie.ValueKind == JsonValueKind.True;
        if (!tied && shards.Contains("lm_head.weight"))
        {
            Weight("lm_head.weight", "output.weight");
        }
        for (int l = 0; l < layers; l++)
        {
            string p = $"model.layers.{l}.";
            string b = $"blk.{l}.";
            Norm(p + "input_layernorm.weight", b + "attn_norm.weight");
            Weight(p + "self_attn.q_proj.weight", b + "attn_q.weight");
            Weight(p + "self_attn.k_proj.weight", b + "attn_k.weight");
            Weight(p + "self_attn.v_proj.weight", b + "attn_v.weight");
            Weight(p + "self_attn.o_proj.weight", b + "attn_output.weight");
            Norm(p + "self_attn.q_norm.weight", b + "attn_q_norm.weight");
            Norm(p + "self_attn.k_norm.weight", b + "attn_k_norm.weight");
            Norm(p + "post_attention_layernorm.weight", b + "ffn_norm.weight");
            // One expert: the same bytes as the dense matrix under a leading axis of 1.
            plan.Add(new Lfm3Gguf.Entry(b + "ffn_gate_inp.weight", new long[] { 1, hidden }, GgufType.F32, null, new float[hidden]));
            Weight(p + "mlp.gate_proj.weight", b + "ffn_gate_exps.weight", new long[] { 1, intermediate, hidden });
            Weight(p + "mlp.up_proj.weight", b + "ffn_up_exps.weight", new long[] { 1, intermediate, hidden });
            Weight(p + "mlp.down_proj.weight", b + "ffn_down_exps.weight", new long[] { 1, hidden, intermediate });
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outFile))!);
        using var stream = new FileStream(outFile, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 20);
        using var writer = new GgufWriter(stream);
        writer.Kv("general.architecture", GgufValue.String(Arch));
        writer.Kv("general.type", GgufValue.String("model"));
        writer.Kv("general.name", GgufValue.String(Path.GetFileName(Path.GetFullPath(checkpointDir))));
        writer.Kv("general.quantization_version", GgufValue.Uint32(2));
        writer.Kv($"{Arch}.block_count", GgufValue.Uint32((uint)layers));
        writer.Kv($"{Arch}.context_length", GgufValue.Uint32((uint)context));
        writer.Kv($"{Arch}.embedding_length", GgufValue.Uint32((uint)hidden));
        writer.Kv($"{Arch}.feed_forward_length", GgufValue.Uint32((uint)intermediate));
        writer.Kv($"{Arch}.attention.head_count", GgufValue.Uint32((uint)heads));
        writer.Kv($"{Arch}.attention.head_count_kv", GgufValue.Uint32((uint)kvHeads));
        writer.Kv($"{Arch}.attention.key_length", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{Arch}.attention.value_length", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{Arch}.rope.freq_base", GgufValue.Float32((float)theta));
        writer.Kv($"{Arch}.attention.layer_norm_rms_epsilon", GgufValue.Float32(eps));
        writer.Kv($"{Arch}.expert_count", GgufValue.Uint32(1));
        writer.Kv($"{Arch}.expert_used_count", GgufValue.Uint32(1));
        writer.Kv($"{Arch}.expert_feed_forward_length", GgufValue.Uint32((uint)intermediate));
        writer.Kv($"{Arch}.attention.causal", GgufValue.Bool(false));
        writer.Kv("diffusion.shift_logits", GgufValue.Bool(false));
        Lfm3Gguf.WriteTokenizer(writer, checkpointDir, vocab, root, root, pre: "qwen2");
        writer.Kv("tokenizer.ggml.mask_token_id", GgufValue.Uint32((uint)maskId));
        writer.Kv("general.file_type", GgufValue.Uint32(Lfm3Gguf.FileType(plan)));

        foreach (var entry in plan)
        {
            writer.Tensor(entry.GgufName, entry.Type, entry.Shape.Reverse().ToArray());
        }
        writer.WriteHeader();
        for (int i = 0; i < plan.Count; i++)
        {
            var entry = plan[i];
            writer.SeekTensor(i);
            byte[] payload;
            if (entry.Values is { } values)
            {
                payload = Lfm3Gguf.Encode(entry.Type, values, (int)entry.Shape[^1]);
            }
            else if (entry.Type == GgufType.BF16 && shards.Info(entry.Source!).Dtype == Amql.Safetensors.Dtype.BF16)
            {
                payload = shards.ReadBytes(entry.Source!);
            }
            else
            {
                payload = Lfm3Gguf.Encode(entry.Type, shards.ReadFloats(entry.Source!), (int)entry.Shape[^1]);
            }
            writer.Data.Write(payload);
            writer.FinishTensor(i);
        }

        var counts = plan.GroupBy(e => e.Type).OrderBy(g => g.Key).Select(g => $"{g.Count()}×{g.Key}");
        return new GgufConversion
        {
            OutputPath = outFile,
            Architecture = Arch,
            TensorsWritten = plan.Count,
            OutputBytes = new FileInfo(outFile).Length,
            VocabSize = vocab,
            Notes = new[]
            {
                $"{Arch} ({label}): {string.Join(", ", counts)}",
                "dense FFNs written as one-expert MoE (router softmax over one expert is exactly 1)",
                $"non-causal attention, diffusion.shift_logits=false, mask token {maskId} — run with llama-diffusion-cli",
            },
        };
    }

    /// <summary>The mask token's id from <c>tokenizer_config.json</c> +
    /// <c>added_tokens.json</c> / <c>tokenizer.json</c>.</summary>
    private static int MaskTokenId(string dir)
    {
        string tokenizerConfig = Path.Combine(dir, "tokenizer_config.json");
        if (!File.Exists(tokenizerConfig))
        {
            throw new GgufException($"'{tokenizerConfig}' missing — the mask token is named there");
        }
        using var tc = JsonDocument.Parse(File.ReadAllText(tokenizerConfig));
        string name = tc.RootElement.TryGetProperty("mask_token", out var m)
            ? m.ValueKind == JsonValueKind.Object ? m.GetProperty("content").GetString()! : m.GetString()!
            : throw new GgufException("tokenizer_config.json declares no mask_token — a masked diffusion GGUF needs one");
        string added = Path.Combine(dir, "added_tokens.json");
        if (File.Exists(added))
        {
            using var at = JsonDocument.Parse(File.ReadAllText(added));
            if (at.RootElement.TryGetProperty(name, out var id))
            {
                return id.GetInt32();
            }
        }
        using var tj = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "tokenizer.json")));
        foreach (var t in tj.RootElement.GetProperty("added_tokens").EnumerateArray())
        {
            if (t.GetProperty("content").GetString() == name)
            {
                return t.GetProperty("id").GetInt32();
            }
        }
        throw new GgufException($"mask token '{name}' has no id in the tokenizer files");
    }

    private static int Int(JsonElement obj, string name) =>
        obj.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetInt32()
            : throw new GgufException($"{ModelType} config is missing '{name}'");
}
