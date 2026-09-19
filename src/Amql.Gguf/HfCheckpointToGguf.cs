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
/// model.safetensors + tokenizer.json) into a GGUF v3 file for llama.cpp.
///
/// Qwen3.5-family checkpoints (model_type qwen3_5_text, hybrid linear +
/// full attention, optional MoE) are emitted as the qwen35 / qwen35moe
/// architectures following llama.cpp's convert_hf_to_gguf.py exactly:
///   - linear-attention tensors map to attn_qkv (the fused in_proj_qkv),
///     attn_gate (in_proj_z), ssm_a (A_log, value = -exp(A_log)),
///     ssm_dt (.dt_proj.bias from dt_bias), ssm_conv1d (squeezed conv1d),
///     ssm_ba / ssm_beta (in_proj_a / in_proj_b) and ssm_out (out_proj)
///   - V heads are reordered from grouped (by K head) to tiled order in
///     in_proj_qkv rows, in_proj_z, in_proj_a/b, A_log/dt and the conv's
///     V-channel rows, and out_proj's columns — the broadcast layout
///     llama.cpp expects when num_k_heads != num_v_heads
///   - every norm except linear_attn.norm is written +1 (the Qwen3-Next
///     checkpoint convention llama.cpp's converter applies)
///   - MoE FFNs stack per-expert gate/up/down into 3-D ffn_gate_exps /
///     ffn_up_exps / ffn_down_exps and transpose the router to
///     ffn_gate_inp; the full-attention layers keep attn_q/k/v/output
///     (+ q/k norms)
///   - metadata: qwen35moe.* hparams incl. the REQUIRED
///     rope.dimension_sections [11,11,10,0], the ssm head config,
///     attention.recurrent_layers, full_attention_interval, and the MoE
///     expert counts
/// Everything is transposed to the [in, out] GGUF layout, weights are F16
/// (BF16 sources; lossless in the normal range), the vision tower is
/// skipped and the MTP drafter companion stays a separate shard.
/// </summary>
public static class GgufConverter
{
    private const long TransposeBlockRows = 8192;
    private const long CopyChunkBytes = 1 << 20;

    private enum Transform
    {
        Copy,
        Transpose,
        Stack,
        Zeros,
        Q35Qkv,     // in_proj_qkv: transpose + tiled V-row reorder
        Q35Z,       // in_proj_z:   transpose + row reorder
        Q35A,       // in_proj_a:   transpose + row reorder (head dim 1)
        Q35B,       // in_proj_b:   transpose + row reorder (head dim 1)
        Q35ALog,    // A_log:       1-D reorder + value = -exp
        Q35Dt,      // dt_bias:     1-D reorder + rename
        Q35Conv,    // conv1d:      squeeze + V-channel row reorder
        Q35Out,     // out_proj:    transpose + column reorder
        Q35Norm,    // norms:       copy with +1 (Qwen3-Next convention)
    }

    public static GgufConversion Convert(string checkpointDir, string outFile)
    {
        string configPath = Path.Combine(checkpointDir, "config.json");
        if (!File.Exists(configPath))
        {
            throw new GgufException($"'{checkpointDir}' has no config.json — not an exported checkpoint");
        }
        using var config = JsonDocument.Parse(File.ReadAllText(configPath));
        var model = config.RootElement;

        string modelType = model.TryGetProperty("model_type", out var mt) ? mt.GetString() ?? "" : "";
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

        bool anyLinear = layerTypes.Any(t => t.Contains("linear", StringComparison.OrdinalIgnoreCase));
        bool qwen35 = modelType.Contains("qwen3_5", StringComparison.OrdinalIgnoreCase) || anyLinear;
        string arch = qwen35
            ? (hasMoe && experts > 0 ? "qwen35moe" : "qwen35")
            : (hasMoe && experts > 0 ? "qwen3moe" : "qwen3");

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

        int linearKeyHeads = GetInt32Or(model, "linear_num_key_heads", 0);
        int linearKeyHeadDim = GetInt32Or(model, "linear_key_head_dim", 0);
        int linearValueHeads = GetInt32Or(model, "linear_num_value_heads", 0);
        int linearValueHeadDim = GetInt32Or(model, "linear_value_head_dim", 0);
        int linearConvKernel = GetInt32Or(model, "linear_conv_kernel_dim", 0);
        bool reorderLinear = anyLinear && linearKeyHeads > 0 && linearValueHeads > 0 && linearKeyHeads != linearValueHeads;
        int vPerK = linearKeyHeads > 0 ? linearValueHeads / linearKeyHeads : 0;

        using var directory = ModelDirectory.Open(checkpointDir);
        var names = directory.TensorNames.ToHashSet(StringComparer.Ordinal);
        if (names.Count == 0)
        {
            throw new GgufException($"no tensors found in '{checkpointDir}'");
        }

        var plan = new List<PlanEntry>();
        int skippedVision = names.Where(n => n.StartsWith("model.visual.", StringComparison.Ordinal)).Count();

        void Add(string sourceName, string ggufName, Transform transform, bool addOne = false, List<Source>? stack = null)
        {
            if (!names.Contains(sourceName))
            {
                throw new GgufException($"source tensor '{sourceName}' missing from the checkpoint");
            }
            var (file, info) = directory.Get(sourceName);
            plan.Add(new PlanEntry(new Source(file, info), ggufName, transform, addOne, stack, null));
        }

        void AddZeros(string ggufName, params long[] neDims)
            => plan.Add(new PlanEntry(null, ggufName, Transform.Zeros, false, null, neDims));

        void Stack(string expertPrefix, string projKind, string ggufName)
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
            Add($"{expertPrefix}0.{projKind}.weight", ggufName, Transform.Stack, stack: slices);
        }

        Add("model.language_model.embed_tokens.weight", "token_embd.weight", Transform.Transpose);
        Add("model.language_model.norm.weight", "output_norm.weight", qwen35 ? Transform.Q35Norm : Transform.Copy);
        Add("lm_head.weight", "output.weight", Transform.Transpose);

        for (int layer = 0; layer < layers; layer++)
        {
            string b = $"model.language_model.layers.{layer}.";
            string blk = $"blk.{layer}.";
            string kind = layer < layerTypes.Count ? layerTypes[layer] : "full_attention";
            bool linear = kind.Contains("linear", StringComparison.OrdinalIgnoreCase);

            Add(b + "input_layernorm.weight", blk + "attn_norm.weight", qwen35 ? Transform.Q35Norm : Transform.Copy);
            Add(b + "post_attention_layernorm.weight",
                qwen35 ? blk + "post_attention_norm.weight" : blk + "ffn_norm.weight",
                qwen35 ? Transform.Q35Norm : Transform.Copy);

            if (linear)
            {
                Add(b + "linear_attn.in_proj_qkv.weight", blk + "attn_qkv.weight",
                    reorderLinear ? Transform.Q35Qkv : Transform.Transpose);
                Add(b + "linear_attn.in_proj_z.weight", blk + "attn_gate.weight",
                    reorderLinear ? Transform.Q35Z : Transform.Transpose);
                Add(b + "linear_attn.in_proj_a.weight", blk + "ssm_alpha.weight",
                    reorderLinear ? Transform.Q35A : Transform.Transpose);
                Add(b + "linear_attn.in_proj_b.weight", blk + "ssm_beta.weight",
                    reorderLinear ? Transform.Q35B : Transform.Transpose);
                Add(b + "linear_attn.out_proj.weight", blk + "ssm_out.weight",
                    reorderLinear ? Transform.Q35Out : Transform.Transpose);
                Add(b + "linear_attn.conv1d.weight", blk + "ssm_conv1d.weight",
                    reorderLinear ? Transform.Q35Conv : Transform.Transpose);
                Add(b + "linear_attn.A_log", blk + "ssm_a",
                    reorderLinear ? Transform.Q35ALog : Transform.Q35ALog);
                Add(b + "linear_attn.dt_bias", blk + "ssm_dt.bias",
                    reorderLinear ? Transform.Q35Dt : Transform.Q35Dt);
                Add(b + "linear_attn.norm.weight", blk + "ssm_norm.weight", Transform.Copy);
            }
            else
            {
                Add(b + "self_attn.q_proj.weight", blk + "attn_q.weight", Transform.Transpose);
                Add(b + "self_attn.k_proj.weight", blk + "attn_k.weight", Transform.Transpose);
                Add(b + "self_attn.v_proj.weight", blk + "attn_v.weight", Transform.Transpose);
                Add(b + "self_attn.o_proj.weight", blk + "attn_output.weight", Transform.Transpose);
                Add(b + "self_attn.q_norm.weight", blk + "attn_q_norm.weight", qwen35 ? Transform.Q35Norm : Transform.Copy);
                Add(b + "self_attn.k_norm.weight", blk + "attn_k_norm.weight", qwen35 ? Transform.Q35Norm : Transform.Copy);
            }

            if (hasMoe && experts > 0)
            {
                Add(b + "mlp.router.weight", blk + "ffn_gate_inp.weight", Transform.Transpose);
                Stack(b + "mlp.experts.", "gate_proj", blk + "ffn_gate_exps.weight");
                Stack(b + "mlp.experts.", "up_proj", blk + "ffn_up_exps.weight");
                Stack(b + "mlp.experts.", "down_proj", blk + "ffn_down_exps.weight");
                if (qwen35)
                {
                    // the qwen35moe arch carries shared experts in its tensor
                    // set unconditionally; this model has none, so emit inert
                    // (zero) shared-expert tensors to satisfy the loader —
                    // a zero shared-expert contribution is neutral in the
                    // routed sum. One shared "expert" of expert width.
                    AddZeros(blk + "ffn_gate_inp_shexp.weight", hidden, 1);
                    AddZeros(blk + "ffn_gate_shexp.weight", hidden, expertIntermediate);
                    AddZeros(blk + "ffn_up_shexp.weight", hidden, expertIntermediate);
                    AddZeros(blk + "ffn_down_shexp.weight", expertIntermediate, hidden);
                }
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
        if (hasMoe && experts > 0)
        {
            writer.Kv($"{arch}.expert_feed_forward_length", GgufValue.Uint32((uint)expertIntermediate));
        }
        writer.Kv($"{arch}.attention.head_count", GgufValue.Uint32((uint)heads));
        writer.Kv($"{arch}.attention.head_count_kv", GgufValue.Uint32((uint)kvHeads));
        writer.Kv($"{arch}.attention.key_length", GgufValue.Uint32((uint)headDim));
        writer.Kv($"{arch}.attention.value_length", GgufValue.Uint32((uint)headDim));
        // the qwen35 family reads layer_norm_rms_epsilon (the newer key
        // convention); the qwen3-family fallback keeps norm_rms_epsilon
        writer.Kv(qwen35 ? $"{arch}.attention.layer_norm_rms_epsilon" : $"{arch}.attention.norm_rms_epsilon",
            GgufValue.Float32(rmsEps));
        writer.Kv($"{arch}.attention.gate_use_softmax", GgufValue.Bool(attnOutputGate));
        writer.Kv($"{arch}.rope.freq_base", GgufValue.Float32(ropeTheta));
        if (qwen35)
        {
            writer.Kv($"{arch}.rope.dimension_sections", GgufValue.Uint32Array(new uint[] { 11, 11, 10, 0 }));
        }
        writer.Kv($"{arch}.rope.dimension_count", GgufValue.Uint32((uint)Math.Round(headDim * (partialRotary > 0 ? partialRotary : 1.0f))));

        if (anyLinear && linearKeyHeads > 0)
        {
            writer.Kv($"{arch}.ssm.conv_kernel", GgufValue.Uint32((uint)linearConvKernel));
            writer.Kv($"{arch}.ssm.state_size", GgufValue.Uint32((uint)linearKeyHeadDim));
            writer.Kv($"{arch}.ssm.group_count", GgufValue.Uint32((uint)linearKeyHeads));
            writer.Kv($"{arch}.ssm.time_step_rank", GgufValue.Uint32((uint)linearValueHeads));
            writer.Kv($"{arch}.ssm.inner_size", GgufValue.Uint32((uint)(linearValueHeadDim * linearValueHeads)));
            writer.Kv($"{arch}.attention.recurrent_layers",
                GgufValue.BoolArray(layerTypes.Select(t => t.Contains("linear", StringComparison.OrdinalIgnoreCase)).ToList()));
            writer.Kv($"{arch}.attention.full_attention_interval", GgufValue.Uint32(4));
        }

        if (hasMoe && experts > 0)
        {
            writer.Kv($"{arch}.expert_count", GgufValue.Uint32((uint)experts));
            writer.Kv($"{arch}.expert_used_count", GgufValue.Uint32((uint)topK));
            writer.Kv($"{arch}.expert_shared_count", GgufValue.Uint32(1));
        }

        var (tokens, scores, tokenTypes, merges) = LoadTokenizer(Path.Combine(checkpointDir, "tokenizer.json"), vocab);
        writer.Kv("tokenizer.ggml.model", GgufValue.String("gpt2"));
        writer.Kv("tokenizer.ggml.pre", GgufValue.String("default"));
        writer.Kv("tokenizer.ggml.tokens", GgufValue.StringArray(tokens));
        writer.Kv("tokenizer.ggml.scores", GgufValue.FloatArray(scores));
        writer.Kv("tokenizer.ggml.token_type", GgufValue.Int32Array(tokenTypes.Select(t => (int)t).ToArray()));
        if (merges.Length > 0)
        {
            writer.Kv("tokenizer.ggml.merges", GgufValue.StringArray(merges));
        }
        writer.Kv($"{arch}.vocab_size", GgufValue.Uint32((uint)vocab));

        foreach (var entry in plan)
        {
            writer.Tensor(entry.Name, GgufTypeFor(entry), GgufDims(entry, reorderLinear));
        }
        writer.WriteHeader();

        for (int i = 0; i < plan.Count; i++)
        {
            WritePayload(writer, plan[i], i, reorderLinear, vPerK, linearValueHeads, linearKeyHeads, linearValueHeadDim);
        }

        long bytes = new FileInfo(outFile).Length;
        string[] notes =
        {
            skippedVision > 0 ? $"skipped vision tower: {skippedVision} model.visual.* tensors (LLM-only GGUF)" : "",
            hasMoe && experts > 0 ? $"stacked {experts} experts per layer into 3-D MoE tensors (top-{topK})" : "",
            anyLinear && reorderLinear
                ? $"linear-attention V heads reordered grouped → tiled ({linearValueHeads} value / {linearKeyHeads} key heads) per llama.cpp's Qwen3.5 convention" : "",
            qwen35 ? "Qwen3-Next value transforms applied: A_log = -exp(A_log), norms +1, conv1d squeezed" : "",
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
        Source? Source,
        string Name,
        Transform Transform,
        bool AddOne,
        IReadOnlyList<Source>? StackSlices,
        long[]? ExplicitDims);

    // ── descriptors ────────────────────────────────────────────────────────

    private static GgufType GgufTypeFor(PlanEntry entry)
        => entry.Transform switch
        {
            Transform.Zeros => GgufType.F16,
            // llama.cpp's Qwen3-Next reference keeps the ssm decay/delta
            // tensors in float32 (they are float in the HF checkpoints);
            // the graph's binary ops mix them with f32 states, so writing
            // them f16 trips ggml's "unsupported types" in build.
            Transform.Q35ALog or Transform.Q35Dt => GgufType.F32,
            _ => entry.Source!.Info.Dtype switch
            {
                Dtype.F32 => GgufType.F32,
                Dtype.BF16 or Dtype.F16 => GgufType.F16,
                _ => throw new GgufException($"source tensor '{entry.Source.Info.Name}' dtype {entry.Source.Info.Dtype.Label()} is not convertible"),
            },
        };

    private static long[] GgufDims(PlanEntry entry, bool reorderLinear)
    {
        var shape = entry.Source?.Info.Shape ?? Array.Empty<long>();
        switch (entry.Transform)
        {
            case Transform.Zeros:
                return entry.ExplicitDims!;
            case Transform.Stack:
                // expert slices stacked as [experts, s0, s1]; the file dims
                // are llama.cpp's ne[] order: [s1, s0, experts]
                return new[] { shape[1], shape[0], (long)entry.StackSlices!.Count };
            case Transform.Transpose:
            case Transform.Q35Qkv:
            case Transform.Q35Z:
            case Transform.Q35A:
            case Transform.Q35B:
            case Transform.Q35Out:
            case Transform.Q35Conv:
                // conv1d [C, 1, K] is squeezed: file dims [K, C] (ne order)
                if (shape.Length == 3)
                {
                    return new[] { shape[2], shape[0] };
                }
                return new[] { shape[1], shape[0] };
            default:
                return shape.ToArray();
        }
    }

    // ── payload streaming ──────────────────────────────────────────────────

    private static void WritePayload(GgufWriter writer, PlanEntry entry, int index, bool reorderLinear,
        int vPerK, int numVHeads, int numKHeads, int headVDim)
    {
        writer.SeekTensor(index);

        switch (entry.Transform)
        {
            case Transform.Zeros:
                long elements = 1;
                foreach (long d in entry.ExplicitDims!)
                {
                    elements *= d;
                }
                byte[] zeroBuf = new byte[Math.Min(elements * 2, 1 << 20)];
                long zeroLeft = elements * 2;
                while (zeroLeft > 0)
                {
                    int take = (int)Math.Min(zeroLeft, zeroBuf.Length);
                    writer.Data.Write(zeroBuf, 0, take);
                    zeroLeft -= take;
                }
                break;
            case Transform.Copy:
                CopyRaw(entry.Source, writer.Data, entry.AddOne);
                break;
            case Transform.Stack:
                foreach (var slice in entry.StackSlices!)
                {
                    CopyRaw(slice, writer.Data, addOne: false);
                }
                break;
            case Transform.Transpose:
                WriteTransposed(writer, entry, index, addOne: entry.AddOne, columnReorder: null);
                break;
            case Transform.Q35Norm:
                CopyRaw(entry.Source, writer.Data, addOne: true);
                break;
            case Transform.Q35Qkv:
                // split rows into q | k | v, reorder v tiled, concatenate, transpose
                var info = entry.Source.Info;
                long qDim = (long)numKHeads * headVDim;
                long kDim = (long)numKHeads * headVDim;
                long vDim = (long)numVHeads * headVDim;
                ReorderRowsAndTranspose(writer, entry, index, qDim, kDim, vDim, vPerK, numKHeads, headVDim);
                break;
            case Transform.Q35Z:
                ReorderRowsAndTranspose(writer, entry, index, 0, 0, entry.Source.Info.Shape[0], vPerK, numKHeads, headVDim);
                break;
            case Transform.Q35A:
            case Transform.Q35B:
                ReorderRowsAndTranspose(writer, entry, index, 0, 0, entry.Source.Info.Shape[0], vPerK, numKHeads, 1);
                break;
            case Transform.Q35ALog:
                // 1-D per-value-head decay: reorder, value = -exp
                ReorderScalar1D(writer, entry, index, negateExp: true, vPerK, numKHeads);
                break;
            case Transform.Q35Dt:
                ReorderScalar1D(writer, entry, index, negateExp: false, vPerK, numKHeads);
                break;
            case Transform.Q35Conv:
                // squeeze [C,1,K] → [C,K]; reorder the V-channel rows
                ReorderConvRows(writer, entry, index, numKHeads, headVDim, vPerK);
                break;
            case Transform.Q35Out:
                // transpose + reorder columns (input dim) tiled
                WriteTransposed(writer, entry, index, addOne: false, columnReorder: (numVHeads, numKHeads, headVDim, vPerK));
                break;
            default:
                throw new GgufException($"unhandled transform {entry.Transform}");
        }

        writer.FinishTensor(index);
    }

    // ── copy paths ─────────────────────────────────────────────────────────

    private static void CopyRaw(Source source, Stream output, bool addOne)
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
                if (addOne)
                {
                    AddOneToF16(tmp.AsSpan(0, take));
                }
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

    private static void WriteTransposed(GgufWriter writer, PlanEntry entry, int index, bool addOne,
        (int NumVHeads, int NumKHeads, int HeadVDim, int VPerK)? columnReorder = null)
    {
        var info = entry.Source.Info;
        long rows = info.Shape[0], cols = info.Shape[1];
        int elem = info.Dtype == Dtype.F32 ? 4 : 2;
        long basePosition = (long)writer.TensorOffset(index);

        if (rows * cols * elem <= 512L << 20)
        {
            byte[] bytes = entry.Source.File.ReadBytes(info);
            byte[] converted = info.Dtype == Dtype.BF16 ? new byte[bytes.Length] : bytes;
            if (info.Dtype == Dtype.BF16)
            {
                ConvertBf16ToF16(bytes, converted);
                if (addOne)
                {
                    AddOneToF16(converted);
                }
            }
            if (columnReorder is { } cr)
            {
                converted = ReorderColumns(converted, rows, cols, elem, cr.VPerK, cr.NumKHeads, cr.HeadVDim);
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
        else
        {
            WriteTransposedBlocked(writer, entry, basePosition, rows, cols, addOne);
        }
    }

    private static void WriteTransposedBlocked(GgufWriter writer, PlanEntry entry, long basePosition, long rows, long cols, bool addOne)
    {
        var info = entry.Source.Info;
        if (info.Dtype != Dtype.BF16)
        {
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
            if (addOne)
            {
                AddOneToF16(block.AsSpan(0, b * (int)cols * 2));
            }
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

    // ── Qwen3.5 linear-attention transforms ────────────────────────────────

    /// <summary>Reads the source 2-D matrix, applies the V-row tiled reorder
    /// over the given row window, and writes it transposed.</summary>
    private static void ReorderRowsAndTranspose(GgufWriter writer, PlanEntry entry, int index,
        long qRows, long kRows, long vRows, int vPerK, int numKHeads, int headVDim)
    {
        var info = entry.Source.Info;
        long rows = info.Shape[0], cols = info.Shape[1];
        byte[] bytes = entry.Source.File.ReadBytes(info);
        byte[] converted = new byte[bytes.Length];
        ConvertBf16ToF16(bytes, converted);
        byte[] reordered = new byte[converted.Length];

        long dest = 0;
        long copy = qRows + kRows; // the untouched head of the matrix (q + k, or empty)
        if (copy > 0)
        {
            Buffer.BlockCopy(converted, 0, reordered, 0, (int)(copy * cols * 2));
            dest = copy;
        }
        // v window: rows [qRows+kRows, rows), reorder tiled
        long vStart = qRows + kRows;
        long vCount = rows - vStart;
        long group = (long)numKHeads * headVDim;     // how many rows per K group
        long vPerKRows = (long)vPerK * headVDim;     // rows per K-group's V block
        for (long vRow = 0; vRow < vCount; vRow++)
        {
            long k = vRow / vPerKRows;
            long inner = vRow % vPerKRows;
            long vp = inner / headVDim;
            long headRow = inner % headVDim;
            long tiledK = vp * numKHeads + k;        // tiled order
            long srcRow = vStart + vRow;
            long dstRow = dest + tiledK * headVDim + headRow;
            Buffer.BlockCopy(converted, (int)(srcRow * cols * 2), reordered, (int)(dstRow * cols * 2), (int)(cols * 2));
        }

        var colRun = new byte[rows * 2];
        for (long c = 0; c < cols; c++)
        {
            for (long r = 0; r < rows; r++)
            {
                colRun[(int)(r * 2)] = reordered[(int)((r * cols + c) * 2)];
                colRun[(int)(r * 2) + 1] = reordered[(int)((r * cols + c) * 2) + 1];
            }
            writer.SeekAbsolute((long)writer.TensorOffset(index) + c * rows * 2);
            writer.Data.Write(colRun, 0, (int)(rows * 2));
        }
    }

    private static void ReorderScalar1D(GgufWriter writer, PlanEntry entry, int index, bool negateExp, int vPerK, int numKHeads)
    {
        var info = entry.Source.Info;
        byte[] bytes = entry.Source.File.ReadBytes(info);
        int n = info.Shape[0] > 0 ? (int)info.Shape[0] : (int)(bytes.Length / 2);
        var outBytes = new byte[n * 4];
        for (int i = 0; i < n; i++)
        {
            ushort bf = (ushort)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
            float f = BitConverter.UInt32BitsToSingle((uint)bf << 16);
            if (negateExp)
            {
                f = -MathF.Exp(f);
            }
            BinaryPrimitives.WriteSingleLittleEndian(outBytes.AsSpan(i * 4), f);
        }
        // reorder: grouped → tiled over the value heads
        var reordered = new byte[outBytes.Length];
        for (int v = 0; v < n; v++)
        {
            int k = v / vPerK;
            int vp = v % vPerK;
            int tiled = vp * numKHeads + k;
            Buffer.BlockCopy(outBytes, v * 4, reordered, tiled * 4, 4);
        }
        writer.Data.Write(reordered, 0, reordered.Length);
    }

    private static void ReorderConvRows(GgufWriter writer, PlanEntry entry, int index, int numKHeads, int headVDim, int vPerK)
    {
        var info = entry.Source.Info;
        // squeeze [C, 1, K] → [C, K]
        long c = info.Shape[0], k = info.Shape[2];
        byte[] bytes = entry.Source.File.ReadBytes(info);
        var conv = new byte[c * k * 2];
        for (long i = 0; i < c; i++)
        {
            for (long j = 0; j < k; j++)
            {
                long srcIdx = (i * k + j) * 2;
                conv[(int)((i * k + j) * 2)] = bytes[(int)((i * k + j) * 2)];
                conv[(int)((i * k + j) * 2) + 1] = bytes[(int)((i * k + j) * 2) + 1];
            }
        }
        // squash: the payload is already [C, K] contiguous from [C,1,K]
        byte[] converted = new byte[conv.Length];
        ConvertBf16ToF16(conv, converted);

        long qkChannels = (long)numKHeads * headVDim * 2;
        long vChannels = c - qkChannels;
        if (vChannels <= 0)
        {
            // nothing to reorder — write as-is (GGUF dims [K, C] reversed)
            writer.Data.Write(converted, 0, converted.Length);
            return;
        }
        long group = (long)numKHeads * headVDim;      // rows per K-group
        long vPerKRows = (long)vPerK * headVDim;
        var reordered = new byte[converted.Length];
        Buffer.BlockCopy(converted, 0, reordered, 0, (int)(qkChannels * k * 2));
        for (long vRow = 0; vRow < vChannels; vRow++)
        {
            long kk = vRow / vPerKRows;
            long inner = vRow % vPerKRows;
            long vp = inner / headVDim;
            long headRow = inner % headVDim;
            long tiledK = vp * numKHeads + kk;
            long srcRow = qkChannels + vRow;
            long dstRow = qkChannels + tiledK * headVDim + headRow;
            Buffer.BlockCopy(converted, (int)(srcRow * k * 2), reordered, (int)(dstRow * k * 2), (int)(k * 2));
        }
        writer.Data.Write(reordered, 0, reordered.Length);
    }

    private static byte[] ReorderColumns(byte[] converted, long rows, long cols, int elem, int vPerK, int numKHeads, int headVDim)
    {
        // out_proj [rows = hidden, cols = num_v_heads*head_v_dim]: tiled reorder of columns
        var reordered = new byte[converted.Length];
        long group = (long)numKHeads * headVDim;
        long vPerKCols = (long)vPerK * headVDim;
        for (long c = 0; c < cols; c++)
        {
            long k = c / vPerKCols;
            long inner = c % vPerKCols;
            long vp = inner / headVDim;
            long headCol = inner % headVDim;
            long tiled = vp * numKHeads + k;
            long srcCol = tiled * headVDim + headCol;
            for (long r = 0; r < rows; r++)
            {
                long si = (r * cols + srcCol) * elem;
                long di = (r * cols + c) * elem;
                for (int b = 0; b < elem; b++)
                {
                    reordered[di + b] = converted[si + b];
                }
            }
        }
        return reordered;
    }

    private static void AddOneToF16(Span<byte> data)
    {
        for (int i = 0; i < data.Length; i += 2)
        {
            ushort bits = (ushort)(data[i] | (data[i + 1] << 8));
            float f = (float)BitConverter.UInt16BitsToHalf(bits);
            ushort plus = BitConverter.HalfToUInt16Bits((Half)(f + 1.0f));
            data[i] = (byte)(plus & 0xFF);
            data[i + 1] = (byte)(plus >> 8);
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
            types[i] = 1;
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

    private static int GetInt32Or(JsonElement obj, string name, int fallback)
        => obj.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : fallback;
}