using System.Buffers.Binary;
using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Gguf;

/// <summary>Result summary of a conversion run.</summary>
public sealed class GgufConversion
{
    public string OutputPath { get; init; } = string.Empty;
    public string Architecture { get; init; } = string.Empty;
    public int TensorsWritten { get; init; }
    public int TensorsSkippedVision { get; init; }
    public long OutputBytes { get; init; }
    public int VocabSize { get; init; }
    public string[] Notes { get; init; } = Array.Empty<string>();
}

/// <summary>
/// Converts an AMQL-exported HF checkpoint directory (config.json +
/// model.safetensors + tokenizer.json) into a GGUF v3 file following
/// llama.cpp's tensor/metadata conventions for the Qwen3-Next family
/// (hybrid linear/full attention, MoE FFN, partial rotary). Weights are
/// written as F16 (BF16 sources convert losslessly in the normal range);
/// F32 tensors (the routers) stay F32. Everything transforms to the
/// llama.cpp layout: projections are transposed [out,in] → [in,out],
/// embeddings/head transposed to [hidden, vocab], the per-expert FFN
/// tensors are stacked into one 3-D tensor per layer, and the vision
/// tower is skipped (an LLM-only GGUF). The MTP drafter companion
/// (mtp.safetensors + mtp.config.json) is not embedded — it remains a
/// separate shard beside the checkpoint.
/// </summary>
public static class GgufConverter
{
    private const string HybridArch = "qwen3nextmoe";
    private const int TransposeBlockRows = 8192;
    private const long CopyChunkBytes = 1 << 20;

    public static GgufConversion Convert(string checkpointDir, string outFile)
    {
        // ── config ─────────────────────────────────────────────────────────
        string configPath = Path.Combine(checkpointDir, "config.json");
        if (!File.Exists(configPath))
        {
            throw new GgufException($"'{checkpointDir}' has no config.json — not an exported checkpoint");
        }
        using var config = JsonDocument.Parse(File.ReadAllText(configPath));
        var model = config.RootElement;

        int hidden = GetInt32(model, "hidden_size");
        int layers = GetInt32(model, "num_hidden_layers");
        int heads = GetInt32(model, "num_attention_heads");
        int kvHeads = GetInt32(model, "num_key_value_heads");
        int headDim = GetInt32(model, "head_dim");
        int intermediate = GetInt32(model, "intermediate_size");
        int vocab = GetInt32(model, "vocab_size");
        int maxPos = GetInt32(model, "max_position_embeddings");
        float rmsEps = model.TryGetProperty("rms_norm_eps", out var re) ? re.GetSingle() : 1e-6f;
        bool attnOutputGate = model.TryGetProperty("attn_output_gate", out var og) && og.GetBoolean();
        bool hasMoe = model.TryGetProperty("moe", out var moe);
        int experts = hasMoe ? GetInt32(moe, "experts") : 0;
        int topK = hasMoe ? GetInt32(moe, "top_k") : 0;
        int expertIntermediate = hasMoe ? GetInt32(moe, "expert_intermediate_size") : intermediate;

        var layerTypes = new List<string>();
        if (model.TryGetProperty("layer_types", out var lt) && lt.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in lt.EnumerateArray())
            {
                layerTypes.Add(item.GetString() ?? "full_attention");
            }
        }

        float ropeTheta = 10_000_000f;
        float partialRotary = 0;
        if (model.TryGetProperty("rope_parameters", out var rope))
        {
            if (rope.TryGetProperty("rope_theta", out var rth) && rth.ValueKind == JsonValueKind.Number)
            {
                ropeTheta = rth.GetSingle();
            }
            if (rope.TryGetProperty("partial_rotary_factor", out var prf))
            {
                partialRotary = prf.GetSingle();
            }
        }

        bool anyLinear = layerTypes.Any(t => t.Contains("linear", StringComparison.OrdinalIgnoreCase));
        string arch = anyLinear
            ? "qwen3nextmoe"
            : hasMoe ? "qwen3moe" : "qwen3";

        // ── sources ────────────────────────────────────────────────────────
        using var directory = ModelDirectory.Open(checkpointDir);
        var names = directory.TensorNames.ToHashSet(StringComparer.Ordinal);
        if (names.Count == 0)
        {
            throw new GgufException($"no tensors found in '{checkpointDir}'");
        }

        // ── tensor plan ─────────────────────────────────────────────────────
        var plan = new List<PlanEntry>();
        int skippedVision = names.Where(n => n.StartsWith("model.visual.", StringComparison.Ordinal)).Count();

        void Add(string sourceName, string ggufName, string? transform = null)
        {
            if (!names.Contains(sourceName))
            {
                throw new GgufException($"source tensor '{sourceName}' missing from the checkpoint");
            }
            var (file, info) = directory.Get(sourceName);
            plan.Add(new PlanEntry(new Source(file, info), ggufName, transform, null));
        }

        Add("model.language_model.embed_tokens.weight", "token_embd.weight", "transpose");
        Add("model.language_model.norm.weight", "output_norm.weight");
        Add("lm_head.weight", "output.weight", "transpose");

        for (int layer = 0; layer < layers; layer++)
        {
            string b = $"model.language_model.layers.{layer}.";
            string blk = $"blk.{layer}.";
            string kind = layer < layerTypes.Count ? layerTypes[layer] : "full_attention";
            bool linear = kind.Contains("linear", StringComparison.OrdinalIgnoreCase);

            Add(b + "input_layernorm.weight", blk + "attn_norm.weight");
            Add(b + "post_attention_layernorm.weight", blk + "ffn_norm.weight");

            if (linear)
            {
                Add(b + "linear_attn.in_proj_qkv.weight", blk + "linear_attn.in_proj_qkv.weight", "transpose");
                Add(b + "linear_attn.in_proj_a.weight", blk + "linear_attn.in_proj_a.weight", "transpose");
                Add(b + "linear_attn.in_proj_b.weight", blk + "linear_attn.in_proj_b.weight", "transpose");
                Add(b + "linear_attn.in_proj_z.weight", blk + "linear_attn.in_proj_z.weight", "transpose");
                Add(b + "linear_attn.out_proj.weight", blk + "linear_attn.out_proj.weight", "transpose");
                Add(b + "linear_attn.conv1d.weight", blk + "linear_attn.conv1d.weight");
                Add(b + "linear_attn.A_log", blk + "linear_attn.A_log");
                Add(b + "linear_attn.dt_bias", blk + "linear_attn.dt_bias");
                Add(b + "linear_attn.norm.weight", blk + "linear_attn.norm.weight");
            }
            else
            {
                Add(b + "self_attn.q_proj.weight", blk + "attn_q.weight", "transpose");
                Add(b + "self_attn.k_proj.weight", blk + "attn_k.weight", "transpose");
                Add(b + "self_attn.v_proj.weight", blk + "attn_v.weight", "transpose");
                Add(b + "self_attn.o_proj.weight", blk + "attn_output.weight", "transpose");
                Add(b + "self_attn.q_norm.weight", blk + "attn_q_norm.weight");
                Add(b + "self_attn.k_norm.weight", blk + "attn_k_norm.weight");
            }

            if (hasMoe && experts > 0)
            {
                Add(b + "mlp.router.weight", blk + "ffn_gate_inp.weight", "transpose");
                Stack(b + "mlp.experts.", "gate_proj", blk + "ffn_gate_exps.weight", plan, experts, directory, names);
                Stack(b + "mlp.experts.", "up_proj", blk + "ffn_up_exps.weight", plan, experts, directory, names);
                Stack(b + "mlp.experts.", "down_proj", blk + "ffn_down_exps.weight", plan, experts, directory, names);
            }
        }

        // ── metadata ────────────────────────────────────────────────────────
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outFile))!);
        using var writer = new GgufWriter(new FileStream(outFile, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20));

        writer.Kv("general.architecture", GgufValue.String(arch));
        writer.Kv("general.name", GgufValue.String(Path.GetFileName(checkpointDir)));
        writer.Kv("general.file_type", GgufValue.Uint32(1));
        writer.Kv("general.alignment", GgufValue.Uint32(32));
        writer.Kv($"{arch}.block_count", GgufValue.Uint32((uint)layers));
        writer.Kv($"{arch}.context_length", GgufValue.Uint32((uint)maxPos));
        writer.Kv($"{arch}.embedding_length", GgufValue.Uint32((uint)hidden));
        writer.Kv($"{arch}.feed_forward_length", GgufValue.Uint32((uint)expertIntermediate));
        writer.Kv($"{arch}.attention.head_count", GgufValue.Uint32((uint)heads));
        writer.Kv($"{arch}.attention.head_count_kv", GgufValue.Uint32((uint)kvHeads));
        writer.Kv($"{arch}.attention.key_length", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{arch}.attention.value_length", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{arch}.attention.norm_rms_epsilon", GgufValue.Float32(rmsEps));
        writer.Kv($"{arch}.attention.gate_use_softmax", GgufValue.Bool(attnOutputGate));
        writer.Kv($"{arch}.rope.freq_base", GgufValue.Float32(ropeTheta));
        if (partialRotary > 0)
        {
            writer.Kv($"{arch}.rope.partial_rotary_factor", GgufValue.Float32(partialRotary));
        }
        if (hasMoe && experts > 0)
        {
            writer.Kv($"{arch}.expert_count", GgufValue.Uint32((uint)experts));
            writer.Kv($"{arch}.expert_used_count", GgufValue.Uint32((uint)topK));
        }
        if (model.TryGetProperty("linear_num_key_heads", out var lkh))
        {
            writer.Kv($"{arch}.linear_attention.key_head_count", GgufValue.Uint32((uint)lkh.GetInt32()));
            writer.Kv($"{arch}.linear_attention.key_head_dim", GgufValue.Uint32((uint)model.GetProperty("linear_key_head_dim").GetInt32()));
            writer.Kv($"{arch}.linear_attention.value_head_count", GgufValue.Uint32((uint)model.GetProperty("linear_num_value_heads").GetInt32()));
            writer.Kv($"{arch}.linear_attention.value_head_dim", GgufValue.Uint32((uint)model.GetProperty("linear_value_head_dim").GetInt32()));
            writer.Kv($"{arch}.linear_attention.conv_kernel", GgufValue.Uint32((uint)model.GetProperty("linear_conv_kernel_dim").GetInt32()));
        }

        var (tokens, scores, tokenTypes, merges) = LoadTokenizer(Path.Combine(checkpointDir, "tokenizer.json"), vocab);
        writer.Kv("tokenizer.ggml.model", GgufValue.String("gpt2"));
        writer.Kv("tokenizer.ggml.tokens", GgufValue.StringArray(tokens));
        writer.Kv("tokenizer.ggml.scores", GgufValue.FloatArray(scores));
        writer.Kv("tokenizer.ggml.token_type", GgufValue.Uint32Array(tokenTypes));
        if (merges.Length > 0)
        {
            writer.Kv("tokenizer.ggml.merges", GgufValue.StringArray(merges));
        }
        writer.Kv($"{arch}.vocab_size", GgufValue.Uint32((uint)vocab));

        // ── register + stream ───────────────────────────────────────────────
        foreach (var entry in plan)
        {
            writer.Tensor(entry.Name, GgufTypeFor(entry), GgufDims(entry));
        }
        writer.WriteHeader();

        for (int i = 0; i < plan.Count; i++)
        {
            WritePayload(writer, plan[i], i);
        }

        long bytes = new FileInfo(outFile).Length;
        string[] notes =
        {
            skippedVision > 0 ? $"skipped vision tower: {skippedVision} model.visual.* tensors (LLM-only GGUF)" : "",
            hasMoe && experts > 0 ? $"stacked {experts} experts per layer into 3-D MoE tensors (top-{topK})" : "",
            "MTP drafter (mtp.safetensors + mtp.config.json) stays a separate companion shard — not embedded",
            "weights written as F16 (BF16 sources; lossless in the normal range); F32 routers kept",
        };

        return new GgufConversion
        {
            OutputPath = outFile,
            Architecture = arch,
            TensorsWritten = plan.Count,
            TensorsSkippedVision = skippedVision,
            OutputBytes = bytes,
            VocabSize = vocab,
            Notes = notes.Where(n => n.Length > 0).ToArray(),
        };
    }

    private sealed record Source(SafetensorsFile File, TensorInfo Info);

    private sealed record PlanEntry(
        Source Source,
        string Name,
        string? Transform,
        IReadOnlyList<Source>? StackSlices);

    private static void Stack(
        string expertPrefix, string projKind, string ggufName,
        List<PlanEntry> plan, int experts, ModelDirectory directory, HashSet<string> names)
    {
        var slices = new List<Source>();
        for (int e = 0; e < experts; e++)
        {
            string sourceName = $"{expertPrefix}{e}.{projKind}.weight";
            if (!names.Contains(sourceName))
            {
                throw new GgufException($"source tensor '{sourceName}' missing from the checkpoint");
            }
            var (file, info) = directory.Get(sourceName);
            slices.Add(new Source(file, info));
        }
        plan.Add(new PlanEntry(slices[0], ggufName, $"stack:{projKind}", slices));
    }

    // ── descriptors ────────────────────────────────────────────────────────

    private static GgufType GgufTypeFor(PlanEntry entry)
        => entry.Source.Info.Dtype switch
        {
            Dtype.F32 => GgufType.F32,
            Dtype.BF16 or Dtype.F16 => GgufType.F16,
            _ => throw new GgufException($"source tensor '{entry.Source.Info.Name}' dtype {entry.Source.Info.Dtype.Label()} is not convertible"),
        };

    private static long[] GgufDims(PlanEntry entry)
    {
        var shape = entry.Source.Info.Shape;
        if (entry.Transform == "transpose")
        {
            if (shape.Length != 2)
            {
                throw new GgufException($"cannot transpose non-2-D tensor '{entry.Source.Info.Name}'");
            }
            return new[] { shape[0], shape[1] }; // gguf order = reversed logical [in,out]
        }
        if (entry.Transform != null && entry.Transform.StartsWith("stack:", StringComparison.Ordinal))
        {
            // logical [experts, mid, emb] → gguf dims [emb, mid, experts]
            return new[] { shape[1], shape[0], (long)entry.StackSlices!.Count };
        }
        return shape.Reverse().ToArray();
    }

    // ── payload streaming ──────────────────────────────────────────────────

    private static void WritePayload(GgufWriter writer, PlanEntry entry, int index)
    {
        var info = entry.Source.Info;
        writer.SeekTensor(index);

        if (entry.Transform is null)
        {
            CopyRaw(entry.Source, writer.Data);
        }
        else if (entry.Transform == "transpose")
        {
            WriteTransposed(writer, entry, index);
        }
        else
        {
            foreach (var slice in entry.StackSlices!)
            {
                CopyRaw(slice, writer.Data);
            }
        }

        writer.FinishTensor(index);
    }

    private static void CopyRaw(Source source, Stream output)
    {
        bool convert = source.Info.Dtype == Dtype.BF16;
        var tmp = new byte[CopyChunkBytes];
        long remaining = source.Info.DataLength;
        long offset = 0;
        while (remaining > 0)
        {
            int take = (int)Math.Min(remaining, CopyChunkBytes);
            byte[] chunk = source.File.ReadBytes(source.Info, offset, take);
            if (convert)
            {
                ConvertBf16ToF16(chunk, tmp.AsSpan(0, take));
                output.Write(tmp, 0, take);
            }
            else
            {
                output.Write(chunk, 0, take);
            }
            offset += take;
            remaining -= take;
        }
    }

    private static void WriteTransposed(GgufWriter writer, PlanEntry entry, int index)
    {
        var info = entry.Source.Info;
        long rows = info.Shape[0], cols = info.Shape[1];
        int elem = info.Dtype == Dtype.F32 ? 4 : 2;
        long basePosition = (long)writer.TensorOffset(index);
        long spanBytes = rows * cols * elem;
        if (spanBytes <= 512L << 20)
        {
            WriteTransposedMemory(writer, entry, basePosition, rows, cols, elem);
        }
        else
        {
            WriteTransposedBlocked(writer, entry, basePosition, rows, cols);
        }
    }

    private static void WriteTransposedMemory(GgufWriter writer, PlanEntry entry, long basePosition, long rows, long cols, int elem)
    {
        var info = entry.Source.Info;
        byte[] bytes = entry.Source.File.ReadBytes(info);
        byte[] converted = info.Dtype == Dtype.BF16 ? new byte[bytes.Length] : bytes;
        if (info.Dtype == Dtype.BF16)
        {
            ConvertBf16ToF16(bytes, converted);
        }
        var colRun = new byte[rows * elem];
        for (long c = 0; c < cols; c++)
        {
            for (long r = 0; r < rows; r++)
            {
                long src = (r * cols + c) * elem;
                for (int b = 0; b < elem; b++)
                {
                    colRun[(int)(r * elem) + b] = converted[(int)src + b];
                }
            }
            writer.SeekAbsolute(basePosition + c * rows * elem);
            writer.Data.Write(colRun, 0, (int)(rows * elem));
        }
    }

    private static void WriteTransposedBlocked(GgufWriter writer, PlanEntry entry, long basePosition, long rows, long cols)
    {
        var info = entry.Source.Info;
        if (info.Dtype != Dtype.BF16)
        {
            // only BF16 sources reach the >512 MiB blocked path
            throw new GgufException($"blocked transpose of non-BF16 tensor '{info.Name}' is unsupported");
        }
        int blockRows = (int)Math.Min(TransposeBlockRows, rows);
        long rowBytes = cols * 2;
        var block = new byte[blockRows * cols * 2];
        var colRun = new byte[blockRows * 2];

        for (long r0 = 0; r0 < rows; r0 += blockRows)
        {
            int b = (int)Math.Min(blockRows, rows - r0);
            for (long r = 0; r < b; r++)
            {
                byte[] row = entry.Source.File.ReadBytes(info, (r0 + r) * rowBytes, (int)rowBytes);
                row.CopyTo(block, (int)(r * cols * 2));
            }
            ConvertBf16ToF16(block.AsSpan(0, b * (int)cols * 2), block.AsSpan(0, b * (int)cols * 2));
            for (long c = 0; c < cols; c++)
            {
                for (long r = 0; r < b; r++)
                {
                    colRun[(int)(r * 2)] = block[(int)((r * cols + c) * 2)];
                    colRun[(int)(r * 2) + 1] = block[(int)((r * cols + c) * 2) + 1];
                }
                writer.SeekAbsolute(basePosition + (c * rows + r0) * 2);
                writer.Data.Write(colRun, 0, b * 2);
            }
        }
    }

    private static void ConvertBf16ToF16(Span<byte> src, Span<byte> dst)
    {
        if (src.Length != dst.Length)
        {
            throw new ArgumentException("src/dst length mismatch");
        }
        for (int i = 0; i < src.Length; i += 2)
        {
            ushort bf = (ushort)(src[i] | (src[i + 1] << 8));
            float f = BitConverter.UInt32BitsToSingle((uint)bf << 16);
            ushort f16 = BitConverter.HalfToUInt16Bits((Half)f);
            dst[i] = (byte)(f16 & 0xFF);
            dst[i + 1] = (byte)(f16 >> 8);
        }
    }

    // ── tokenizer ──────────────────────────────────────────────────────────

    private static (string[] Tokens, float[] Scores, uint[] Types, string[] Merges) LoadTokenizer(string tokenizerPath, int vocab)
    {
        if (!File.Exists(tokenizerPath))
        {
            throw new GgufException($"'{tokenizerPath}' missing — a tokenizer.json is required for the GGUF tokenizer block");
        }
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerPath));
        var model = doc.RootElement.GetProperty("model");

        var idToToken = new (int Id, string Token)[vocab];
        var vocabObj = model.GetProperty("vocab");
        foreach (var kv in vocabObj.EnumerateObject())
        {
            int id = kv.Value.GetInt32();
            if (id >= 0 && id < vocab)
            {
                idToToken[id] = (id, kv.Name);
            }
        }

        var tokens = new string[vocab];
        var scores = new float[vocab];
        var types = new uint[vocab];
        for (int i = 0; i < vocab; i++)
        {
            tokens[i] = idToToken[i].Token ?? $"<unk_{i}>";
            types[i] = 1; // normal
        }

        if (model.TryGetProperty("scores", out var scoresJson) && scoresJson.ValueKind == JsonValueKind.Array)
        {
            foreach (var pair in scoresJson.EnumerateArray())
            {
                if (pair.ValueKind != JsonValueKind.Array || pair.GetArrayLength() < 2)
                {
                    continue;
                }
                int id = pair[0].GetInt32();
                float score = pair[1].GetSingle();
                if (id >= 0 && id < vocab)
                {
                    scores[id] = score;
                }
            }
        }

        var merges = new List<string>();
        if (model.TryGetProperty("merges", out var mergesJson) && mergesJson.ValueKind == JsonValueKind.Array)
        {
            foreach (var entry in mergesJson.EnumerateArray())
            {
                merges.Add(entry.GetString() ?? string.Empty);
            }
        }

        return (tokens, scores, types, merges.ToArray());
    }

    private static int GetInt32(JsonElement obj, string name)
    {
        if (obj.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number)
        {
            return value.GetInt32();
        }
        throw new GgufException($"config.json is missing '{name}'");
    }
}