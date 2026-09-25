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

    /// <summary>The <c>--quant</c> modes this converter can emit: F16, Q4_0,
    /// all-MXFP4, and llama.cpp's MXFP4_MOE recipe (expert stacks MXFP4,
    /// every other quantizable weight Q8_0).</summary>
    public static readonly IReadOnlyList<string> Quantizations = new[] { "none", "q4_0", "mxfp4", "mxfp4_moe" };

    public static GgufConversion Convert(string checkpointDir, string outFile, string quantization = "none")
    {
        if (!Quantizations.Contains(quantization))
        {
            throw new GgufException(
                $"unknown quantization '{quantization}' — expected one of: {string.Join(", ", Quantizations)}");
        }

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
        // A tied head has no lm_head tensor in the checkpoint. llama.cpp
        // infers tying from the ABSENCE of output.weight, so emitting a copy
        // of the embedding would both waste space and misreport the model.
        if (names.Contains("lm_head.weight"))
        {
            Add("lm_head.weight", "output.weight", Transform.Transpose);
        }

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
                // conv1d is [C, 1, K] in HF, which in ggml's ne order is
                // already [ne0=K, ne1=C] — a straight copy, not a transpose.
                // Transposing here reads the shape as [C, 1] and writes a
                // quarter of the payload.
                Add(b + "linear_attn.conv1d.weight", blk + "ssm_conv1d.weight",
                    reorderLinear ? Transform.Q35Conv : Transform.Copy);
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
            else
            {
                // Dense FFN. Without this branch a non-MoE checkpoint exports
                // with no feed-forward weights at all, and llama.cpp fails at
                // load with "tensor 'blk.N.ffn_gate.weight' not found".
                Add(b + "mlp.gate_proj.weight", blk + "ffn_gate.weight", Transform.Transpose);
                Add(b + "mlp.up_proj.weight", blk + "ffn_up.weight", Transform.Transpose);
                Add(b + "mlp.down_proj.weight", blk + "ffn_down.weight", Transform.Transpose);
            }
        }

        // ── metadata ────────────────────────────────────────────────────────
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outFile))!);
        using var writer = new GgufWriter(new FileStream(outFile, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20));

        writer.Kv("general.architecture", GgufValue.String(arch));
        writer.Kv("general.name", GgufValue.String(Path.GetFileName(checkpointDir)));
        // ggml ftype: 1 = mostly F16, 2 = mostly Q4_0, 38 = mostly MXFP4_MOE.
        // Declaring F16 over a file whose weights are block-quantized
        // misreports the model to every consumer. There is no plain
        // MOSTLY_MXFP4 in llama.cpp's enum, so an all-MXFP4 file declares the
        // MXFP4_MOE ftype too — the per-tensor types in the header are what
        // the loader actually reads.
        uint fileType = quantization switch
        {
            "q4_0" => 2u,
            "mxfp4" or "mxfp4_moe" => 38u,
            _ => 1u,
        };
        writer.Kv("general.file_type", GgufValue.Uint32(fileType));
        if (quantization != "none")
        {
            writer.Kv("general.quantization_version", GgufValue.Uint32(2));
        }
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

        if (qwen35 && linearKeyHeads > 0)
        {
            writer.Kv($"{arch}.ssm.conv_kernel", GgufValue.Uint32((uint)linearConvKernel));
            writer.Kv($"{arch}.ssm.state_size", GgufValue.Uint32((uint)linearKeyHeadDim));
            writer.Kv($"{arch}.ssm.group_count", GgufValue.Uint32((uint)linearKeyHeads));
            writer.Kv($"{arch}.ssm.time_step_rank", GgufValue.Uint32((uint)linearValueHeads));
            writer.Kv($"{arch}.ssm.inner_size", GgufValue.Uint32((uint)(linearValueHeadDim * linearValueHeads)));
            writer.Kv($"{arch}.attention.recurrent_layers",
                GgufValue.BoolArray(layerTypes.Select(t => t.Contains("linear", StringComparison.OrdinalIgnoreCase)).ToList()));
            int fullCount = layerTypes.Count(t => !t.Contains("linear", StringComparison.OrdinalIgnoreCase));
            writer.Kv($"{arch}.full_attention_interval",
                GgufValue.Uint32((uint)(fullCount > 0 ? layers / fullCount : layers)));
        }

        if (hasMoe && experts > 0)
        {
            writer.Kv($"{arch}.expert_count", GgufValue.Uint32((uint)experts));
            writer.Kv($"{arch}.expert_used_count", GgufValue.Uint32((uint)topK));
            writer.Kv($"{arch}.expert_shared_count", GgufValue.Uint32(1));
        }

        var tok = LoadTokenizer(Path.Combine(checkpointDir, "tokenizer.json"), vocab);
        writer.Kv("tokenizer.ggml.model", GgufValue.String("gpt2"));
        // The pretokenizer regex is arch-specific; "default" mis-splits Qwen
        // text. Matches the lmstudio-community reference for this family.
        writer.Kv("tokenizer.ggml.pre", GgufValue.String(qwen35 ? "qwen35" : "default"));
        writer.Kv("tokenizer.ggml.tokens", GgufValue.StringArray(tok.Tokens));
        if (tok.Scores is { } scores)
        {
            writer.Kv("tokenizer.ggml.scores", GgufValue.FloatArray(scores));
        }
        writer.Kv("tokenizer.ggml.token_type", GgufValue.Int32Array(tok.Types.Select(t => (int)t).ToArray()));
        if (tok.Merges.Length > 0)
        {
            writer.Kv("tokenizer.ggml.merges", GgufValue.StringArray(tok.Merges));
        }
        if (tok.EosTokenId is { } eosId)
        {
            writer.Kv("tokenizer.ggml.eos_token_id", GgufValue.Uint32((uint)eosId));
        }
        if (tok.PaddingTokenId is { } padId)
        {
            // Qwen keeps bos == pad and never actually prepends a BOS.
            writer.Kv("tokenizer.ggml.bos_token_id", GgufValue.Uint32((uint)padId));
            writer.Kv("tokenizer.ggml.padding_token_id", GgufValue.Uint32((uint)padId));
            writer.Kv("tokenizer.ggml.add_bos_token", GgufValue.Bool(false));
        }
        // llama.cpp reads the chat template out of the GGUF itself, not from a
        // sidecar file. Without this key the runtime can only do raw
        // completions, so the checkpoint directory has to have carried
        // chat_template.jinja through the container for it to land here.
        string? chatTemplate = LoadChatTemplate(checkpointDir);
        if (chatTemplate is not null)
        {
            writer.Kv("tokenizer.chat_template", GgufValue.String(chatTemplate));
        }
        writer.Kv($"{arch}.vocab_size", GgufValue.Uint32((uint)vocab));

        foreach (var entry in plan)
        {
            writer.Tensor(entry.Name, GgufTypeFor(entry, quantization), GgufDims(entry, reorderLinear));
        }
        writer.WriteHeader();

        for (int i = 0; i < plan.Count; i++)
        {
            WritePayload(writer, plan[i], i, reorderLinear, vPerK, linearKeyHeads, linearValueHeadDim, quantization);
        }

        long bytes = new FileInfo(outFile).Length;
        string[] notes =
        {
            skippedVision > 0 ? $"skipped vision tower: {skippedVision} model.visual.* tensors (LLM-only GGUF)" : "",
            hasMoe && experts > 0 ? $"stacked {experts} experts per layer into 3-D MoE tensors (top-{topK})" : "",
            anyLinear && reorderLinear
                ? $"linear-attention V heads reordered grouped → tiled ({linearValueHeads} value / {linearKeyHeads} key heads) per llama.cpp's Qwen3.5 convention" : "",
            qwen35 ? "Qwen3-Next value transforms applied: A_log = -exp(A_log), norms +1, conv1d squeezed" : "",
            chatTemplate is null
                ? "no chat template found (chat_template.jinja or tokenizer_config.json) — the GGUF is completion-only"
                : $"chat template embedded ({chatTemplate.Length} chars)",
            "MTP drafter (mtp.safetensors + mtp.config.json) stays a separate companion shard — not embedded",
            quantization switch
            {
                "q4_0" => "Q4_0 quantization applied to weight matrices"
                    + (hasMoe && experts > 0
                        ? " including the stacked 3-D MoE expert tensors (every expert slice quantized)"
                        : "")
                    + "; norms, embeddings, output head and routers kept full precision",
                "mxfp4" => "MXFP4 quantization applied to weight matrices (ggml type 39: 32 E2M1 values "
                    + "sharing one E8M0 exponent per 17-byte block)"
                    + (hasMoe && experts > 0 ? " including the stacked 3-D MoE expert tensors" : "")
                    + "; norms, embeddings, output head and routers kept full precision",
                "mxfp4_moe" => hasMoe && experts > 0
                    ? "MXFP4_MOE recipe (llama.cpp ftype 38): 3-D MoE expert stacks in MXFP4, every other "
                      + "quantizable weight in Q8_0; norms, embeddings, output head and routers kept full precision"
                    : "MXFP4_MOE recipe (llama.cpp ftype 38) requested for a dense model — there are no expert "
                      + "stacks, so every quantizable weight is Q8_0",
                _ => "weights written as F16 (BF16 sources; lossless in the normal range); F32 routers kept",
            },
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
        long[]? ExplicitDims)
    {
        /// <summary>The source tensor for transforms that require one, with a
        /// typed error instead of a null dereference when the plan is malformed.</summary>
        public Source RequiredSource => Source ?? throw new GgufException(
            $"transform {Transform} for '{Name}' has no source tensor");
    }

    // ── Q4_0 quantization ────────────────────────────────────────────────

    /// <summary>Q4_0 block size: 32 elements per block.</summary>
    private const int Q4_0BlockSize = 32;

    /// <summary>Quantizes a float array to ggml's Q4_0 format: one 18-byte
    /// block per 32 values, a float16 scale followed by 16 bytes of nibbles.
    /// Byte j carries element j in its low nibble and element j+16 in its high
    /// nibble — packing even/odd pairs instead permutes every row, and the
    /// result still loads and passes any check that shares the same wrong
    /// assumption while the model generates garbage. Mirrors
    /// quantize_row_q4_0_ref in ggml-quants.c, including the scale and the
    /// truncating rounding, so the bytes match llama.cpp's own quantizer.</summary>
    private static byte[] QuantizeQ4_0(float[] values)
    {
        if (values.Length % Q4_0BlockSize != 0)
            throw new GgufException($"cannot Q4_0-quantize {values.Length} values: not a multiple of {Q4_0BlockSize}");

        int numBlocks = values.Length / Q4_0BlockSize;
        var result = new byte[numBlocks * 18];

        for (int block = 0; block < numBlocks; block++)
        {
            int start = block * Q4_0BlockSize;

            // The signed value with the largest magnitude, not the magnitude:
            // its sign makes the stored scale negative so that the decoded
            // range [-8d, 7d] straddles the block instead of sitting above it.
            float amax = 0f, max = 0f;
            for (int i = 0; i < Q4_0BlockSize; i++)
            {
                float v = values[start + i];
                float abs = MathF.Abs(v);
                if (amax < abs) { amax = abs; max = v; }
            }

            float d = max / -8f;
            float id = d != 0f ? 1f / d : 0f;

            ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)d);
            result[block * 18] = (byte)(scaleBits & 0xFF);
            result[block * 18 + 1] = (byte)(scaleBits >> 8);

            int nibbles = block * 18 + 2;
            for (int j = 0; j < Q4_0BlockSize / 2; j++)
            {
                byte lo = QuantizeNibble(values[start + j], id);
                byte hi = QuantizeNibble(values[start + Q4_0BlockSize / 2 + j], id);
                result[nibbles + j] = (byte)(lo | (hi << 4));
            }
        }

        return result;
    }

    /// <summary>Maps one value to its stored 4-bit code. ggml truncates
    /// x*id + 8.5 toward zero and clamps to [0, 15] instead of rounding to
    /// nearest, so the top of the range saturates one step early; matching
    /// that keeps the bytes identical to llama-quantize's.</summary>
    private static byte QuantizeNibble(float value, float inverseScale)
    {
        float q = MathF.Truncate(value * inverseScale + 8.5f);
        return (byte)Math.Clamp((int)q, 0, 15);
    }

    /// <summary>Writes a block-quantized tensor payload: loads the source as
    /// floats, applies any additive transform, encodes the blocks and writes
    /// them. No physical transpose — GGUF reverses the declared dims but keeps
    /// the source row-major byte order.</summary>
    private static void WriteQuantizedPayload(GgufWriter writer, PlanEntry entry, int index, GgufType type)
    {
        // Stacked MoE experts: the header declares all N slices, so every
        // slice must be quantized and concatenated. Writing only slice 0
        // would leave FinishTensor zero-padding the rest — a structurally
        // valid file whose expert weights are mostly zeros.
        if (entry.Transform == Transform.Stack)
        {
            foreach (var slice in entry.StackSlices!)
            {
                var sliceInfo = slice.Info;
                float[] sliceValues = LoadFloats(slice, sliceInfo.Shape[0] * sliceInfo.Shape[1]);
                // ne0 is sliceCols in GGUF order; slices are written in source
                // row-major order, which already matches [ne0=sliceCols, ne1=sliceRows].
                writer.Data.Write(EncodeBlocks(type, sliceValues));
            }
            writer.FinishTensor(index);
            return;
        }

        var source = entry.RequiredSource;
        var info = source.Info;
        float[] values = LoadFloats(source, info.Shape[0] * info.Shape[1]);

        if (entry.AddOne)
        {
            for (long i = 0; i < values.Length; i++)
            {
                values[i] += 1.0f;
            }
        }

        writer.Data.Write(EncodeBlocks(type, values));
        writer.FinishTensor(index);
    }

    /// <summary>Encodes a float array into the given block type.</summary>
    private static byte[] EncodeBlocks(GgufType type, float[] values) => type switch
    {
        GgufType.Q4_0 => QuantizeQ4_0(values),
        _ => GgmlQuant.Encode(type, values),
    };

    // ── descriptors ────────────────────────────────────────────────────────

    private static GgufType GgufTypeFor(PlanEntry entry, string quantization)
    {
        // Block quantization for eligible weight tensors.
        if (ShouldQuantize(entry))
        {
            switch (quantization)
            {
                case "q4_0":
                    return GgufType.Q4_0;
                case "mxfp4":
                    return GgufType.Mxfp4;
                case "mxfp4_moe":
                    // llama.cpp's MXFP4_MOE recipe (ftype 38): the 3-D expert
                    // stacks become MXFP4 and every other quantizable weight
                    // becomes Q8_0. Keeping the shared attention path at 8 bit
                    // is the point of the recipe — 4-bit there costs more
                    // quality than the expert stacks do, and on a dense model
                    // with no expert stacks this degrades to plain Q8_0.
                    return entry.Transform == Transform.Stack ? GgufType.Mxfp4 : GgufType.Q8_0;
            }
        }

        // ggml requires F32 for every norm weight and for the SSM depthwise
        // conv in the qwen35 hybrid path. Writing these as F16 (which is what
        // a BF16 source maps to) makes the loader die during graph build —
        // verified against a working qwen35 GGUF, which keeps all *_norm.weight
        // and ssm_conv1d in F32.
        if (entry.Name.EndsWith("_norm.weight", StringComparison.Ordinal)
            || entry.Name.Contains("ssm_conv1d", StringComparison.Ordinal)
            || entry.Name == "ssm_a"
            || entry.Name.EndsWith("ssm_dt.bias", StringComparison.Ordinal))
        {
            return GgufType.F32;
        }

        return entry.Transform switch
        {
            Transform.Zeros => GgufType.F16,
            // The per-value-head scalar params (A_log/dt/in_proj_a/in_proj_b)
            // and the depthwise conv stay float32 — they are float in the HF
            // checkpoints and ggml mixes them with f32 recurrent state, where
            // an f16 operand trips the mixed-type binary ops in build.
            // in_proj_qkv / in_proj_z / out_proj are ordinary projection
            // matrices that merely need their V heads permuted first, so they
            // follow the normal dtype/quantization policy like every other
            // weight; forcing them to F32 was costing 4 bytes per element on
            // three quarters of a hybrid model's layers.
            Transform.Q35ALog or Transform.Q35Dt or Transform.Q35A or Transform.Q35B
                or Transform.Q35Conv => GgufType.F32,
            _ => entry.Source!.Info.Dtype switch
            {
                Dtype.F32 => GgufType.F32,
                Dtype.BF16 or Dtype.F16 => GgufType.F16,
                _ => throw new GgufException($"source tensor '{entry.Source.Info.Name}' dtype {entry.Source.Info.Dtype.Label()} is not convertible"),
            },
        };
    }

    /// <summary>Determines if a tensor is eligible for block quantization, in
    /// any of the supported types. Norms, embeddings, output heads, MoE
    /// routers and small tensors stay full precision — matching the skip list
    /// in llama.cpp's <c>tensor_allows_quantization</c>, since 4-bit routers
    /// measurably degrade expert selection and its 1-D tensors are never
    /// quantized at all.</summary>
    private static bool ShouldQuantize(PlanEntry entry)
    {
        if (entry.Source is null) return false;
        var name = entry.Source.Info.Name;
        var shape = entry.Source.Info.Shape;

        // Skip norms, embeddings, output heads and MoE routers. The router
        // reaches GGUF as ffn_gate_inp but its source name carries "router".
        if (name.Contains("norm") || name.Contains("embed") || name.Contains("lm_head")
            || name.Contains("router"))
            return false;

        // Skip small tensors (< 256 elements)
        long totalElements = 1;
        foreach (var dim in shape) totalElements *= dim;
        if (totalElements < 256) return false;

        // Skip non-2D tensors (conv weights, stacked MoE experts, etc.)
        if (shape.Length != 2) return false;

        // ggml blocks along the contiguous dimension, so anything that is not
        // a whole number of blocks has no valid encoding in any of these types
        // and must stay full precision rather than be padded.
        if (totalElements % GgmlQuant.BlockElements != 0) return false;

        // Skip the transforms that must stay F32 (see GgufTypeFor). The
        // reordering projections are not in this list: their head permutation
        // happens inside their own write path before the values are blocked.
        if (entry.Transform is Transform.Q35ALog or Transform.Q35Dt or Transform.Q35A
            or Transform.Q35B or Transform.Q35Conv or Transform.Q35Norm)
            return false;

        return true;
    }

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
            case Transform.Copy:
                // A depthwise conv1d arrives as [C, 1, K], which in ggml's ne
                // order is already [ne0=K, ne1=C]: the payload needs no
                // rearrangement, but the declared dims must be squeezed or
                // llama.cpp rejects the tensor as 3-D. Anything else copies
                // through with its own shape.
                if (shape.Length == 3)
                {
                    return new[] { shape[2], shape[0] };
                }
                return shape.ToArray();
            default:
                return shape.ToArray();
        }
    }

    // ── payload streaming ──────────────────────────────────────────────────

    private static void WritePayload(GgufWriter writer, PlanEntry entry, int index, bool reorderLinear,
        int vPerK, int numKHeads, int headVDim, string quantization)
    {
        writer.SeekTensor(index);

        // Block quantization: load floats, encode, write. The reordering
        // projections are excluded here because their V-head permutation has
        // to happen before the values are cut into blocks; they encode inside
        // their own write path instead.
        var declared = GgufTypeFor(entry, quantization);
        if (GgufTypeSizing.IsBlockQuantized(declared) && !ReordersHeads(entry.Transform))
        {
            WriteQuantizedPayload(writer, entry, index, declared);
            return;
        }

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
                if (GgufTypeFor(entry, quantization) == GgufType.F32 && entry.RequiredSource.Info.Dtype != Dtype.F32)
                {
                    CopyAsF32(entry.RequiredSource, writer.Data, entry.AddOne);
                }
                else
                {
                    CopyRaw(entry.RequiredSource, writer.Data, entry.AddOne);
                }
                break;
            case Transform.Stack:
                foreach (var slice in entry.StackSlices!)
                {
                    CopyRaw(slice, writer.Data, addOne: false);
                }
                break;
            case Transform.Transpose:
                // GGUF stores ne[] as the HF shape REVERSED but keeps the
                // bytes in row-major source order — gguf-py does shape[::-1]
                // and writes the buffer unchanged. Physically transposing here
                // double-transposes every nn.Linear weight ([out,in] in HF),
                // which loads fine and then generates garbage.
                if (GgufTypeFor(entry, quantization) == GgufType.F32 && entry.RequiredSource.Info.Dtype != Dtype.F32)
                {
                    CopyAsF32(entry.RequiredSource, writer.Data, entry.AddOne);
                }
                else
                {
                    CopyRaw(entry.RequiredSource, writer.Data, entry.AddOne);
                }
                break;
            case Transform.Q35Norm:
                // Norms must land as F32; the source is usually BF16, so widen
                // rather than emit the half-width payload CopyRaw would write.
                if (entry.RequiredSource.Info.Dtype != Dtype.F32)
                {
                    CopyAsF32(entry.RequiredSource, writer.Data, addOne: NormPlusOne);
                }
                else
                {
                    CopyRaw(entry.RequiredSource, writer.Data, addOne: NormPlusOne);
                }
                break;
            case Transform.Q35Qkv:
                // rows are q | k | v; only the V window is permuted
                long qDim = (long)numKHeads * headVDim;
                WriteRowReordered(writer, entry, index, 2 * qDim, numKHeads, headVDim, vPerK, quantization);
                break;
            case Transform.Q35Z:
                WriteRowReordered(writer, entry, index, 0, numKHeads, headVDim, vPerK, quantization);
                break;
            case Transform.Q35A:
            case Transform.Q35B:
                // one row per value head, so the "head dim" of the permutation is 1
                WriteRowReordered(writer, entry, index, 0, numKHeads, 1, vPerK, quantization);
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
                // the input dim carries the tiled V-head states
                WriteColReordered(writer, entry, index, numKHeads, headVDim, vPerK, quantization);
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

    /// <summary>Copies a source tensor widened to F32. Used where ggml
    /// requires float32 (norm weights, the SSM depthwise conv) but the
    /// checkpoint stores BF16/F16 — writing the narrow type would leave the
    /// payload half the size the header declares.</summary>
    private static void CopyAsF32(Source source, Stream output, bool addOne)
    {
        var dtype = source.Info.Dtype;
        int srcElem = dtype.ElementSize();
        long elements = source.Info.DataLength / srcElem;
        var chunk = new byte[Math.Min(elements, 1 << 18) * srcElem];
        var outBuf = new byte[chunk.Length / srcElem * 4];
        long done = 0;
        while (done < elements)
        {
            long take = Math.Min(elements - done, chunk.Length / srcElem);
            byte[] read = source.File.ReadBytes(source.Info, done * srcElem, (int)(take * srcElem));
            var span = outBuf.AsSpan(0, (int)take * 4);
            for (int i = 0; i < take; i++)
            {
                float v = dtype switch
                {
                    Dtype.F32 => BitConverter.ToSingle(read, i * 4),
                    Dtype.BF16 => BitConverter.UInt32BitsToSingle((uint)BitConverter.ToUInt16(read, i * 2) << 16),
                    Dtype.F16 => (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(read, i * 2)),
                    _ => throw new GgufException($"cannot widen dtype {dtype.Label()} to F32"),
                };
                if (addOne)
                {
                    v += 1.0f;
                }
                BitConverter.TryWriteBytes(span.Slice(i * 4, 4), v);
            }
            output.Write(span);
            done += take;
        }
    }

    /// <summary>Whether norm weights are stored as (w - 1) in the source and
    /// need +1 on the way out. Qwen3-Next does this for some checkpoints and
    /// not others, so it is switchable for A/B verification against a runtime:
    /// set AMQL_GGUF_NORM_PLUS_ONE=0 to write the weights unchanged.</summary>
    private static readonly bool NormPlusOne =
        Environment.GetEnvironmentVariable("AMQL_GGUF_NORM_PLUS_ONE") != "0";

    /// <summary>True for the transforms that permute V heads, which must run
    /// their reorder before any quantization blocking.</summary>
    private static bool ReordersHeads(Transform transform)
        => transform is Transform.Q35Qkv or Transform.Q35Z or Transform.Q35Out;

    /// <summary>Permutes the V-head rows of a linear-attention projection from
    /// HF's grouped-by-K-head order into ggml's tiled order and writes the
    /// result verbatim. Only rows move — the array keeps its [rows, cols]
    /// shape — so the payload is the permuted buffer unchanged while the
    /// header declares the reversed dims. Transposing it as well would
    /// double-transpose the weight, the same failure mode Transform.Transpose
    /// had. Mirrors _reorder_v_heads in llama.cpp's converter: reshape the V
    /// window to [num_k_heads, num_v_per_k, head_dim], swap the first two
    /// axes, flatten back. Rows before <paramref name="vWindowStart"/> are the
    /// q and k window of in_proj_qkv and keep their place.</summary>
    private static void WriteRowReordered(GgufWriter writer, PlanEntry entry, int index,
        long vWindowStart, int numKHeads, int headVDim, int vPerK, string quantization)
    {
        var source = entry.RequiredSource;
        var info = source.Info;
        long rows = info.Shape[0], cols = info.Shape[1];
        byte[] bytes = source.File.ReadBytes(info);

        long vPerKRows = (long)vPerK * headVDim;
        var reordered = new float[rows * cols];
        for (long r = 0; r < rows; r++)
        {
            long dst = r;
            if (r >= vWindowStart)
            {
                long rr = r - vWindowStart;
                long k = rr / vPerKRows;
                long inner = rr % vPerKRows;
                long vp = inner / headVDim;
                long headRow = inner % headVDim;
                dst = vWindowStart + (vp * numKHeads + k) * headVDim + headRow;
            }

            long srcBase = r * cols, dstBase = dst * cols;
            for (long c = 0; c < cols; c++)
            {
                reordered[dstBase + c] = ReadFloatAt(bytes, info.Dtype, srcBase + c);
            }
        }

        WriteTypedFloats(writer, entry, quantization, reordered);
    }

    /// <summary>Permutes the columns of out_proj, whose input dimension carries
    /// the tiled V-head states. Same rule as the row case: a permutation of the
    /// existing shape, written verbatim, with a grouped source column landing
    /// at its tiled destination. The direction matters — the permutation is
    /// only self-inverse when num_v_per_k equals num_k_heads, so a reversed
    /// mapping passes on a small fixture and on Qwen3.5-0.8B (v_per_k 1) and
    /// scrambles out_proj on the 27B (16 key heads, v_per_k 3).</summary>
    private static void WriteColReordered(GgufWriter writer, PlanEntry entry, int index,
        int numKHeads, int headVDim, int vPerK, string quantization)
    {
        var source = entry.RequiredSource;
        var info = source.Info;
        long rows = info.Shape[0], cols = info.Shape[1];
        byte[] bytes = source.File.ReadBytes(info);

        long vPerKCols = (long)vPerK * headVDim;
        var reordered = new float[rows * cols];
        for (long c = 0; c < cols; c++)
        {
            // c is the grouped source column (k, vp, d); it lands at the tiled
            // column (vp, k, d).
            long k = c / vPerKCols;
            long inner = c % vPerKCols;
            long vp = inner / headVDim;
            long headCol = inner % headVDim;
            long dstCol = (vp * numKHeads + k) * headVDim + headCol;
            for (long r = 0; r < rows; r++)
            {
                reordered[r * cols + dstCol] = ReadFloatAt(bytes, info.Dtype, r * cols + c);
            }
        }

        WriteTypedFloats(writer, entry, quantization, reordered);
    }

    /// <summary>Emits floats in whatever type the header already declared for
    /// this tensor, so a permuted projection can be block-quantized, F16 or
    /// F32 without the reorder needing to know which. Does not call
    /// FinishTensor; the caller's switch does.</summary>
    private static void WriteTypedFloats(GgufWriter writer, PlanEntry entry, string quantization, float[] values)
    {
        var type = GgufTypeFor(entry, quantization);
        if (GgufTypeSizing.IsBlockQuantized(type))
        {
            writer.Data.Write(EncodeBlocks(type, values));
            return;
        }
        switch (type)
        {
            case GgufType.F32:
                writer.Data.Write(MakeF32Bytes(values));
                break;
            default:
                var f16 = new byte[values.Length * 2];
                for (int i = 0; i < values.Length; i++)
                {
                    BinaryPrimitives.WriteUInt16LittleEndian(f16.AsSpan(i * 2),
                        BitConverter.HalfToUInt16Bits((Half)values[i]));
                }
                writer.Data.Write(f16);
                break;
        }
    }

    /// <summary>Reads one value, widening BF16/F16 to float. Guessing which
    /// half-width flavour a tensor uses silently halves the exponent field —
    /// that is how A_log came out as -exp(0) = -1 in every even head slot.</summary>
    private static float ReadFloatAt(byte[] bytes, Dtype dtype, long index) => dtype switch
    {
        Dtype.F32 => BitConverter.Int32BitsToSingle(
            BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan((int)index * 4))),
        Dtype.BF16 => BitConverter.UInt32BitsToSingle((uint)(ushort)(
            bytes[(int)index * 2] | (bytes[(int)index * 2 + 1] << 8)) << 16),
        Dtype.F16 => (float)BitConverter.UInt16BitsToHalf((ushort)(
            bytes[(int)index * 2] | (bytes[(int)index * 2 + 1] << 8))),
        _ => throw new GgufException($"cannot read dtype {dtype.Label()} as float"),
    };

    private static float[] LoadFloats(Source source, long count)
    {
        byte[] bytes = source.File.ReadBytes(source.Info);
        var floats = new float[count];
        for (long i = 0; i < count; i++)
        {
            floats[i] = ReadFloatAt(bytes, source.Info.Dtype, i);
        }
        return floats;
    }

    // ── Qwen3.5 linear-attention transforms ────────────────────────────────

    private static void ReorderScalar1D(GgufWriter writer, PlanEntry entry, int index, bool negateExp, int vPerK, int numKHeads)
    {
        var source = entry.RequiredSource;
        var info = source.Info;
        byte[] bytes = source.File.ReadBytes(info);
        int n = info.Shape[0] > 0 ? (int)info.Shape[0] : (int)(bytes.Length / info.Dtype.ElementSize());
        // The source dtype varies: A_log is F32 in Qwen3.5 checkpoints but
        // BF16 in others. Reading F32 bytes as BF16 pairs takes the low half
        // of each float (zero for BF16-origin values) for even slots and the
        // real high half for odd ones — producing -exp(0) = -1 in half the
        // head slots and silently dropping the other half of the heads.
        var outBytes = new byte[n * 4];
        for (int i = 0; i < n; i++)
        {
            float f = info.Dtype switch
            {
                Dtype.F32 => BitConverter.UInt32BitsToSingle(
                    BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(i * 4))),
                Dtype.BF16 => BitConverter.UInt32BitsToSingle(
                    (uint)(ushort)(bytes[i * 2] | (bytes[i * 2 + 1] << 8)) << 16),
                Dtype.F16 => (float)BitConverter.UInt16BitsToHalf(
                    (ushort)(bytes[i * 2] | (bytes[i * 2 + 1] << 8))),
                _ => throw new GgufException(
                    $"cannot read '{info.Name}' dtype {info.Dtype.Label()} as a scalar 1-D tensor"),
            };
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
        var source = entry.RequiredSource;
        var info = source.Info;
        // squeeze [C, 1, K] → [C, K]
        long c = info.Shape[0], k = info.Shape[2];
        byte[] bytes = source.File.ReadBytes(info);
        var conv = new float[c * k];
        for (long i = 0; i < c * k; i++)
        {
            conv[i] = info.Dtype == Dtype.F32
                ? BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan((int)(i * 4))))
                : BitConverter.UInt32BitsToSingle((uint)((ushort)(bytes[(int)(i * 2)] | (bytes[(int)(i * 2) + 1] << 8)) << 16));
        }

        long qkChannels = (long)numKHeads * headVDim * 2;
        long vChannels = c - qkChannels;
        if (vChannels <= 0)
        {
            writer.Data.Write(MakeF32Bytes(conv));
            return;
        }
        long vPerKRows = (long)vPerK * headVDim;
        var reordered = new float[conv.Length];
        Buffer.BlockCopy(conv, 0, reordered, 0, (int)qkChannels * (int)k * 4);
        for (long vRow = 0; vRow < vChannels; vRow++)
        {
            long kk = vRow / vPerKRows;
            long inner = vRow % vPerKRows;
            long vp = inner / headVDim;
            long headRow = inner % headVDim;
            long tiledK = vp * numKHeads + kk;
            long srcRow = qkChannels + vRow;
            long dstRow = qkChannels + tiledK * headVDim + headRow;
            Buffer.BlockCopy(conv, (int)(srcRow * k * 4), reordered, (int)(dstRow * k * 4), (int)(k * 4));
        }
        writer.Data.Write(MakeF32Bytes(reordered));
    }

    private static byte[] MakeF32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        }
        return bytes;
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

    /// <summary>Finds the chat template for a checkpoint. HF moved it out of
    /// tokenizer_config.json into a standalone chat_template.jinja, and older
    /// repos keep it inside either JSON file, so all three are tried in that
    /// order. Returns null when the checkpoint ships none, which makes the
    /// GGUF completion-only rather than failing the conversion — a base model
    /// legitimately has no template.</summary>
    private static string? LoadChatTemplate(string checkpointDir)
    {
        var jinja = Path.Combine(checkpointDir, "chat_template.jinja");
        if (File.Exists(jinja))
        {
            string text = File.ReadAllText(jinja).Trim();
            if (text.Length > 0)
            {
                return text;
            }
        }

        foreach (string name in new[] { "tokenizer_config.json", "tokenizer.json" })
        {
            var path = Path.Combine(checkpointDir, name);
            if (!File.Exists(path))
            {
                continue;
            }
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                if (doc.RootElement.ValueKind == JsonValueKind.Object
                    && doc.RootElement.TryGetProperty("chat_template", out var template)
                    && template.ValueKind == JsonValueKind.String
                    && template.GetString() is { Length: > 0 } value)
                {
                    return value;
                }
            }
            catch (JsonException)
            {
                // a malformed sidecar should not fail the whole conversion
            }
        }

        return null;
    }

    /// <summary>Reads the GGUF tokenizer block out of an HF tokenizer.json.
    /// <c>added_tokens</c> carries the special tokens at ids above
    /// <c>model.vocab</c>; ignoring it truncates the vocabulary and leaves
    /// the table shorter than the ids the tokenizer hands out, which reads
    /// past the end of the embedding at load time.</summary>
    private static TokenizerBlock LoadTokenizer(string tokenizerPath, int vocab)
    {
        if (!File.Exists(tokenizerPath))
        {
            throw new GgufException($"'{tokenizerPath}' missing — a tokenizer.json is required for the GGUF tokenizer block");
        }
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerPath));
        var model = doc.RootElement.GetProperty("model");

        var idToToken = new string?[vocab];
        var vocabObj = model.GetProperty("vocab");
        foreach (var kv in vocabObj.EnumerateObject())
        {
            int id = kv.Value.GetInt32();
            if (id >= 0 && id < vocab)
            {
                idToToken[id] = kv.Name;
            }
        }

        var tokens = new string[vocab];
        var types = new uint[vocab];
        for (int i = 0; i < vocab; i++)
        {
            types[i] = 1; // NORMAL
        }

        // ggml token types: 1 NORMAL, 3 CONTROL, 4 USER_DEFINED.
        var specials = new Dictionary<string, int>(StringComparer.Ordinal);
        if (doc.RootElement.TryGetProperty("added_tokens", out var addedTokens) &&
            addedTokens.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in addedTokens.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object ||
                    !item.TryGetProperty("id", out var idElement) ||
                    !idElement.TryGetInt32(out int id) ||
                    id < 0 || id >= vocab)
                {
                    continue;
                }
                string? content = item.TryGetProperty("content", out var contentElement)
                    ? contentElement.GetString()
                    : null;
                if (content is null)
                {
                    continue;
                }
                bool isSpecial = item.TryGetProperty("special", out var sp) &&
                                 sp.ValueKind == JsonValueKind.True;
                idToToken[id] = content;
                types[id] = isSpecial ? 3u : 4u;
                specials[content] = id;
            }
        }

        for (int i = 0; i < vocab; i++)
        {
            tokens[i] = idToToken[i] ?? $"<unk_{i}>";
        }

        // BPE tokenizers carry no scores; emitting an all-zero array is worse
        // than omitting it, so only real scores are returned.
        float[]? scores = null;
        if (model.TryGetProperty("scores", out var scoresJson) && scoresJson.ValueKind == JsonValueKind.Array)
        {
            var parsed = new float[vocab];
            bool any = false;
            foreach (var pair in scoresJson.EnumerateArray())
            {
                if (pair.ValueKind != JsonValueKind.Array || pair.GetArrayLength() < 2)
                {
                    continue;
                }
                int id = pair[0].GetInt32();
                if (id >= 0 && id < vocab)
                {
                    parsed[id] = pair[1].GetSingle();
                    any = true;
                }
            }
            if (any)
            {
                scores = parsed;
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

        // Qwen convention (matches the lmstudio-community reference):
        // <|endoftext|> is bos and pad, <|im_end|> is eos, and no BOS is
        // actually prepended.
        int? eos = specials.TryGetValue("<|im_end|>", out int imEnd) ? imEnd
            : specials.TryGetValue("<|endoftext|>", out int eot0) ? eot0
            : null;
        int? pad = specials.TryGetValue("<|endoftext|>", out int eot) ? eot : null;

        return new TokenizerBlock(tokens, scores, types, merges.ToArray(), eos, pad);
    }

    private sealed record TokenizerBlock(
        string[] Tokens,
        float[]? Scores,
        uint[] Types,
        string[] Merges,
        int? EosTokenId,
        int? PaddingTokenId);

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