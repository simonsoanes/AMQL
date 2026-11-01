using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Gguf;

/// <summary>
/// LFM3 checkpoint → GGUF for llama.cpp's <c>lfm3</c> architecture. Mirrors
/// <c>convert_hf_to_gguf.py</c>'s <c>LFM3Model</c>: per-layer
/// <c>head_count_kv</c> (0 on conv layers), the sliding-window pattern, the
/// short-conv width, a <c>rope_freqs</c> tensor that masks the full-attention
/// layers' trailing rotary pairs (factor 1e30) to express LFM3's zero-tail
/// partial rope, and the conv kernel squeezed to [K, hidden]. The vision tower
/// and projector are left out (llama.cpp loads them from a separate mmproj).
/// <para>
/// Weights are written per <see cref="Quantizations"/>: <c>none</c> keeps the
/// stored BF16, the others quantise every 2-D projection while embeddings
/// (unless asked), norms and the conv kernel stay as llama-quantize leaves them.
/// </para>
/// </summary>
public static class Lfm3Gguf
{
    public const string Arch = "lfm3";

    public static readonly IReadOnlyList<string> Quantizations = new[] { "none", "f16", "q8_0", "mxfp4" };

    /// <summary>Whether <paramref name="checkpointDir"/> holds an LFM3 (or
    /// LFM3-VL) checkpoint.</summary>
    public static bool Applies(string checkpointDir)
    {
        string configPath = Path.Combine(checkpointDir, "config.json");
        if (!File.Exists(configPath))
        {
            return false;
        }
        using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
        return TextConfig(doc.RootElement).TryGetProperty("model_type", out var mt) && mt.GetString() == Arch;
    }

    private static JsonElement TextConfig(JsonElement root) =>
        root.TryGetProperty("text_config", out var text) && text.ValueKind == JsonValueKind.Object ? text : root;

    /// <summary>One tensor of the output plan: its GGUF name, logical shape,
    /// the type it is written as and where its values come from.</summary>
    internal sealed record Entry(string GgufName, long[] Shape, GgufType Type, string? Source, float[]? Values);

    public static GgufConversion Convert(string checkpointDir, string outFile, string quantization = "none")
    {
        if (!Quantizations.Contains(quantization))
        {
            throw new GgufException(
                $"unknown quantization '{quantization}' for lfm3 — expected one of: {string.Join(", ", Quantizations)}");
        }
        return Convert(checkpointDir, outFile, entry => DefaultType(entry, quantization), quantization);
    }

    /// <summary>The converter proper, with the per-tensor type decided by
    /// <paramref name="typeFor"/> (given the GGUF name and logical shape of
    /// every quantisable 2-D weight).</summary>
    internal static GgufConversion Convert(string checkpointDir, string outFile,
        Func<(string Name, long[] Shape), GgufType> typeFor, string label)
    {
        using var configDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpointDir, "config.json")));
        var root = configDoc.RootElement;
        var text = TextConfig(root);

        int hidden = Int(text, "hidden_size");
        int layers = Int(text, "num_hidden_layers");
        int heads = Int(text, "num_attention_heads");
        int kvHeads = Int(text, "num_key_value_heads");
        int headDim = Int(text, "head_dim");
        int intermediate = Int(text, "intermediate_size");
        int vocab = Int(text, "vocab_size");
        int context = Int(text, "max_position_embeddings");
        int window = Int(text, "sliding_window");
        int convWidth = Int(text, "conv_L_cache");
        float eps = text.GetProperty("norm_eps").GetSingle();
        var layerTypes = text.GetProperty("layer_types").EnumerateArray().Select(e => e.GetString()!).ToList();
        if (layerTypes.Count != layers)
        {
            throw new GgufException($"layer_types has {layerTypes.Count} entries for {layers} layers");
        }
        var rope = text.GetProperty("rope_parameters");
        var fullRope = rope.GetProperty("full_attention");
        var swaRope = rope.GetProperty("sliding_attention");
        double partial = fullRope.TryGetProperty("partial_rotary_factor", out var pf) ? pf.GetDouble() : 1.0;
        bool hasMoe = text.TryGetProperty("moe", out var moe) && moe.ValueKind == JsonValueKind.Object;
        if (hasMoe)
        {
            throw new GgufException("llama.cpp's lfm3 architecture has dense FFNs only — a routed (moe) lfm3 checkpoint has no GGUF spelling");
        }

        using var shards = new ShardSet(checkpointDir);
        string prefix = shards.Contains("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";

        // ── tensor plan ─────────────────────────────────────────────────
        var plan = new List<Entry>();
        void Weight(string hf, string gguf)
        {
            var info = shards.Info(hf);
            var shape = info.Shape;
            plan.Add(new Entry(gguf, shape, typeFor((gguf, shape)), hf, null));
        }
        void Norm(string hf, string gguf) =>
            plan.Add(new Entry(gguf, shards.Info(hf).Shape, GgufType.F32, hf, null));

        // Zero-tail partial rope: the first factor·head_dim/2 pairs rotate,
        // the remainder are divided by 1e30 (frequency ≈ 0).
        int rotated = (int)(partial * headDim) / 2;
        var ropeFreqs = Enumerable.Range(0, headDim / 2).Select(i => i < rotated ? 1f : 1e30f).ToArray();
        plan.Add(new Entry("rope_freqs.weight", new long[] { headDim / 2 }, GgufType.F32, null, ropeFreqs));

        Weight(prefix + "embed_tokens.weight", "token_embd.weight");
        Norm(prefix + "embedding_norm.weight", "output_norm.weight");
        bool tied = !shards.Contains("lm_head.weight");
        if (!tied)
        {
            Weight("lm_head.weight", "output.weight");
        }
        for (int l = 0; l < layers; l++)
        {
            string p = $"{prefix}layers.{l}.";
            string b = $"blk.{l}.";
            Norm(p + "operator_norm.weight", b + "attn_norm.weight");
            Norm(p + "ffn_norm.weight", b + "ffn_norm.weight");
            if (shards.Contains(p + "post_operator_norm.weight"))
            {
                Norm(p + "post_operator_norm.weight", b + "post_attention_norm.weight");
                Norm(p + "post_ffn_norm.weight", b + "post_ffw_norm.weight");
            }
            Weight(p + "feed_forward.w1.weight", b + "ffn_gate.weight");
            Weight(p + "feed_forward.w3.weight", b + "ffn_up.weight");
            Weight(p + "feed_forward.w2.weight", b + "ffn_down.weight");
            if (layerTypes[l] == "conv")
            {
                // [hidden, 1, K] squeezed to [hidden, K]; ggml's ssm_conv wants F32.
                var convShape = shards.Info(p + "conv.conv.weight").Shape;
                plan.Add(new Entry(b + "shortconv.conv.weight", new[] { convShape[0], convShape[^1] }, GgufType.F32,
                    p + "conv.conv.weight", null));
                Weight(p + "conv.in_proj.weight", b + "shortconv.in_proj.weight");
                Weight(p + "conv.out_proj.weight", b + "shortconv.out_proj.weight");
            }
            else
            {
                Weight(p + "self_attn.q_proj.weight", b + "attn_q.weight");
                Weight(p + "self_attn.k_proj.weight", b + "attn_k.weight");
                Weight(p + "self_attn.v_proj.weight", b + "attn_v.weight");
                Weight(p + "self_attn.o_proj.weight", b + "attn_output.weight");
                Weight(p + "self_attn.gate_proj.weight", b + "attn_gate.weight");
                Norm(p + "self_attn.q_norm.weight", b + "attn_q_norm.weight");
                Norm(p + "self_attn.k_norm.weight", b + "attn_k_norm.weight");
            }
        }

        // ── metadata ────────────────────────────────────────────────────
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outFile))!);
        using var stream = new FileStream(outFile, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 20);
        using var writer = new GgufWriter(stream);
        writer.Kv("general.architecture", GgufValue.String(Arch));
        writer.Kv("general.type", GgufValue.String("model"));
        writer.Kv("general.name", GgufValue.String(Path.GetFileName(Path.GetFullPath(checkpointDir))));
        writer.Kv("general.quantization_version", GgufValue.Uint32(2));
        writer.Kv("general.file_type", GgufValue.Uint32(FileType(plan)));
        writer.Kv($"{Arch}.block_count", GgufValue.Uint32((uint)layers));
        writer.Kv($"{Arch}.context_length", GgufValue.Uint32((uint)context));
        writer.Kv($"{Arch}.embedding_length", GgufValue.Uint32((uint)hidden));
        writer.Kv($"{Arch}.feed_forward_length", GgufValue.Uint32((uint)intermediate));
        writer.Kv($"{Arch}.attention.head_count", GgufValue.Uint32((uint)heads));
        writer.Kv($"{Arch}.attention.head_count_kv",
            GgufValue.Int32Array(layerTypes.Select(t => t == "conv" ? 0 : kvHeads).ToArray()));
        writer.Kv($"{Arch}.rope.freq_base", GgufValue.Float32(fullRope.GetProperty("rope_theta").GetSingle()));
        writer.Kv($"{Arch}.rope.freq_base_swa", GgufValue.Float32(swaRope.GetProperty("rope_theta").GetSingle()));
        writer.Kv($"{Arch}.attention.layer_norm_rms_epsilon", GgufValue.Float32(eps));
        writer.Kv($"{Arch}.attention.key_length", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{Arch}.attention.value_length", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{Arch}.rope.dimension_count", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{Arch}.attention.sliding_window_pattern",
            GgufValue.BoolArray(layerTypes.Select(t => t == "sliding_attention").ToArray()));
        writer.Kv($"{Arch}.attention.sliding_window", GgufValue.Uint32((uint)window));
        writer.Kv($"{Arch}.shortconv.l_cache", GgufValue.Uint32((uint)convWidth));
        WriteTokenizer(writer, checkpointDir, vocab, text, root);

        foreach (var entry in plan)
        {
            // GGUF dims are the logical shape reversed (fastest-varying first).
            writer.Tensor(entry.GgufName, entry.Type, entry.Shape.Reverse().ToArray());
        }
        writer.WriteHeader();

        // ── payloads ────────────────────────────────────────────────────
        for (int i = 0; i < plan.Count; i++)
        {
            var entry = plan[i];
            writer.SeekTensor(i);
            byte[] payload = entry.Values is { } values
                ? Encode(entry.Type, values, (int)entry.Shape[^1])
                : EncodeSource(shards, entry);
            writer.Data.Write(payload);
            writer.FinishTensor(i);
        }

        var counts = plan.GroupBy(e => e.Type).OrderBy(g => g.Key).Select(g => $"{g.Count()}×{g.Key}");
        return new GgufConversion
        {
            OutputPath = outFile,
            Architecture = Arch,
            TensorsWritten = plan.Count,
            TensorsSkippedVision = shards.Names.Count(n => n.StartsWith("model.vision_tower.", StringComparison.Ordinal) ||
                                                           n.StartsWith("model.multi_modal_projector.", StringComparison.Ordinal)),
            OutputBytes = new FileInfo(outFile).Length,
            VocabSize = vocab,
            Notes = new[]
            {
                $"lfm3 ({label}): {string.Join(", ", counts)}",
                $"zero-tail rope as rope_freqs: {rotated} of {headDim / 2} pairs rotate on full-attention layers",
                "vision tower and projector left out — llama.cpp reads them from a separate mmproj GGUF",
            },
        };
    }

    private static GgufType DefaultType((string Name, long[] Shape) tensor, string quantization) => quantization switch
    {
        "none" => GgufType.BF16,
        "f16" => GgufType.F16,
        // llama-quantize keeps the embedding at its source precision unless asked.
        _ when tensor.Name == "token_embd.weight" => GgufType.BF16,
        "q8_0" => GgufType.Q8_0,
        "mxfp4" => GgufType.Mxfp4,
        _ => throw new GgufException($"unknown quantization '{quantization}'"),
    };

    internal static uint FileType(List<Entry> plan)
    {
        // The dominant weight type names the file, as llama-quantize does
        // (expert banks are 3-D, so the weight is the full element count);
        // a file with no weight below F32 is ALL_F32 (0).
        var dominant = plan.Where(e => e.Shape.Length >= 2 && e.Type != GgufType.F32)
            .GroupBy(e => e.Type)
            .OrderByDescending(g => g.Sum(e => e.Shape.Aggregate(1L, (a, b) => a * b)))
            .Select(g => (GgufType?)g.Key)
            .FirstOrDefault();
        return dominant switch
        {
            GgufType.F16 => 1,
            GgufType.Q4_0 => 2,
            GgufType.Q8_0 => 7,
            GgufType.BF16 => 32,
            GgufType.Mxfp4 => 38,
            _ => 0,
        };
    }

    internal static byte[] Encode(GgufType type, float[] values, int cols) => type switch
    {
        GgufType.F32 => ToBytes(values),
        GgufType.F16 => ToF16(values),
        GgufType.BF16 => ToBf16(values),
        _ => GgmlQuant.Encode(type, values, cols),
    };

    private static byte[] EncodeSource(ShardSet shards, Entry entry)
    {
        var info = shards.Info(entry.Source!);
        if (entry.Type == GgufType.BF16 && info.Dtype == Dtype.BF16)
        {
            return shards.ReadBytes(entry.Source!); // verbatim, no widening
        }
        return Encode(entry.Type, shards.ReadFloats(entry.Source!), (int)entry.Shape[^1]);
    }

    private static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] ToF16(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), BitConverter.HalfToUInt16Bits((Half)values[i]));
        }
        return bytes;
    }

    private static byte[] ToBf16(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]);
            // Round to nearest even, as torch does.
            uint rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
            bytes[i * 2] = (byte)rounded;
            bytes[i * 2 + 1] = (byte)(rounded >> 8);
        }
        return bytes;
    }

    // ── tokenizer ───────────────────────────────────────────────────────

    /// <summary>The BPE vocabulary as llama.cpp's converter writes it for
    /// LFM (and, with <paramref name="pre"/> <c>qwen2</c>, for Qwen): ids the tokenizer does not define become <c>[PADn]</c> (UNUSED),
    /// added tokens are CONTROL when special or when they look special
    /// (<c>&lt;|…|&gt;</c>), USER_DEFINED otherwise.</summary>
    internal static void WriteTokenizer(GgufWriter writer, string checkpointDir, int vocab, JsonElement text, JsonElement root,
        string pre = "lfm2")
    {
        string path = Path.Combine(checkpointDir, "tokenizer.json");
        if (!File.Exists(path))
        {
            throw new GgufException($"'{path}' missing — a tokenizer.json is required for the GGUF tokenizer block");
        }
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var model = doc.RootElement.GetProperty("model");
        var tokens = new string?[vocab];
        var types = new int[vocab];
        foreach (var kv in model.GetProperty("vocab").EnumerateObject())
        {
            int id = kv.Value.GetInt32();
            if (id >= 0 && id < vocab)
            {
                tokens[id] = kv.Name;
                types[id] = 1;
            }
        }
        foreach (var added in doc.RootElement.GetProperty("added_tokens").EnumerateArray())
        {
            int id = added.GetProperty("id").GetInt32();
            if (id < 0 || id >= vocab)
            {
                continue;
            }
            string content = added.GetProperty("content").GetString()!;
            bool special = added.TryGetProperty("special", out var sp) && sp.ValueKind == JsonValueKind.True;
            bool looksSpecial = content.StartsWith("<|", StringComparison.Ordinal) && content.EndsWith("|>", StringComparison.Ordinal);
            tokens[id] = content;
            types[id] = special || looksSpecial ? 3 : 4;
        }
        for (int i = 0; i < vocab; i++)
        {
            if (tokens[i] is null)
            {
                tokens[i] = $"[PAD{i}]";
                types[i] = 5;
            }
        }

        var merges = new List<string>();
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
        {
            merges.Add(merge.ValueKind == JsonValueKind.Array
                ? $"{merge[0].GetString()} {merge[1].GetString()}"
                : merge.GetString()!);
        }

        writer.Kv("tokenizer.ggml.model", GgufValue.String("gpt2"));
        writer.Kv("tokenizer.ggml.pre", GgufValue.String(pre));
        writer.Kv("tokenizer.ggml.tokens", GgufValue.StringArray(tokens!));
        writer.Kv("tokenizer.ggml.token_type", GgufValue.Int32Array(types));
        writer.Kv("tokenizer.ggml.merges", GgufValue.StringArray(merges));
        foreach (var (key, field) in new[]
                 {
                     ("tokenizer.ggml.bos_token_id", "bos_token_id"),
                     ("tokenizer.ggml.eos_token_id", "eos_token_id"),
                     ("tokenizer.ggml.padding_token_id", "pad_token_id"),
                 })
        {
            if ((text.TryGetProperty(field, out var id) || root.TryGetProperty(field, out id)) && id.ValueKind == JsonValueKind.Number)
            {
                writer.Kv(key, GgufValue.Uint32((uint)id.GetInt32()));
            }
        }
        // The chat template writes the BOS itself.
        writer.Kv("tokenizer.ggml.add_bos_token", GgufValue.Bool(false));
        writer.Kv("tokenizer.ggml.add_eos_token", GgufValue.Bool(false));
        string templatePath = Path.Combine(checkpointDir, "chat_template.jinja");
        if (File.Exists(templatePath))
        {
            writer.Kv("tokenizer.chat_template", GgufValue.String(File.ReadAllText(templatePath)));
        }
    }

    private static int Int(JsonElement obj, string name) =>
        obj.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetInt32()
            : throw new GgufException($"lfm3 config is missing '{name}'");

    /// <summary>Every safetensors shard of a checkpoint, by tensor name.</summary>
    internal sealed class ShardSet : IDisposable
    {
        private readonly List<SafetensorsFile> _files = new();
        private readonly Dictionary<string, SafetensorsFile> _owner = new(StringComparer.Ordinal);

        public ShardSet(string dir)
        {
            foreach (var path in Directory.GetFiles(dir, "*.safetensors").Order(StringComparer.Ordinal))
            {
                var file = SafetensorsFile.Open(path);
                _files.Add(file);
                foreach (var name in file.Tensors.Keys)
                {
                    _owner[name] = file;
                }
            }
            if (_files.Count == 0)
            {
                throw new GgufException($"'{dir}' holds no .safetensors shards");
            }
        }

        public IEnumerable<string> Names => _owner.Keys;
        public bool Contains(string name) => _owner.ContainsKey(name);

        public TensorInfo Info(string name) => _owner.TryGetValue(name, out var file)
            ? file.Tensors[name]
            : throw new GgufException($"checkpoint has no tensor '{name}'");

        public byte[] ReadBytes(string name) => _owner[name].ReadBytes(name);

        public float[] ReadFloats(string name)
        {
            var info = Info(name);
            var bytes = ReadBytes(name);
            long count = info.Shape.Aggregate(1L, (a, b) => a * b);
            var values = new float[count];
            for (long i = 0; i < count; i++)
            {
                values[i] = info.Dtype switch
                {
                    Dtype.F32 => BitConverter.ToSingle(bytes, (int)(i * 4)),
                    Dtype.BF16 => BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(bytes, (int)(i * 2)) << 16),
                    Dtype.F16 => (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(bytes, (int)(i * 2))),
                    var other => throw new GgufException($"'{name}': source dtype {other} has no float widening here"),
                };
            }
            return values;
        }

        public void Dispose()
        {
            foreach (var file in _files)
            {
                file.Dispose();
            }
        }
    }
}
