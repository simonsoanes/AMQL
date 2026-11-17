using System.Text.Json;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Hf;

/// <summary>
/// Qwen4 (transformers <c>qwen4_exp</c>, llama.cpp <c>qwen4exp</c>; the open
/// preview is Qwen3.8-Flash-Next): a Qwen3.5-style hybrid of Gated DeltaNet
/// and gated softmax attention, with three additions this file judges from the
/// checkpoint — the Gated Residual (hyper-connections) that replaces every
/// norm, the QSA indexer on each softmax layer, and the hashed n-gram
/// embedding (PLE) on the layers <c>ple_layer_ids</c> names. Every layer's FFN
/// is routed experts plus a gated shared expert, stored packed
/// (<c>mlp.experts.gate_up_proj</c> [E, 2I, H], <c>mlp.experts.down_proj</c>
/// [E, H, I]); those are split into per-expert tensors on import and packed
/// back on export.
/// </summary>
public static partial class ArchMapper
{
    public static bool IsQwen4Exp(TextArchitectureFacts facts) =>
        facts.ModelType is "qwen4_exp" or "qwen4_exp_text";

    /// <summary>Families whose packed experts are known to be laid out as
    /// transformers' <c>Qwen3_5MoeExperts</c> / <c>Qwen4ExpTextExperts</c>:
    /// <c>gate_up_proj</c> [E, 2I, H] with each expert's gate rows before its
    /// up rows (<c>chunk(2)</c>), <c>down_proj</c> [E, H, I]. GPT-OSS and others
    /// pack differently and stay refused.</summary>
    private static readonly HashSet<string> GateThenUpPackedExperts = new(StringComparer.Ordinal)
    {
        "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text",
    };

    private const string PackedGateUp = "mlp.experts.gate_up_proj";
    private const string PackedDown = "mlp.experts.down_proj";

    /// <summary>Splits a packed expert tensor (<c>3.mlp.experts.gate_up_proj</c>)
    /// into per-expert tensors, each a byte range of it, or returns null when
    /// <paramref name="relative"/> is not one.</summary>
    private static (List<NamedTensorData> Parts, FusedTensor Fused)? SplitPackedExperts(
        HfInventory inventory, string fullName, string relative, TextArchitectureFacts facts)
    {
        int dot = relative.IndexOf('.');
        string suffix = relative[(dot + 1)..];
        if (suffix is not (PackedGateUp or PackedDown))
        {
            return null;
        }
        if (!GateThenUpPackedExperts.Contains(facts.ModelType))
        {
            return null; // refused by name when the roles are judged
        }
        var info = inventory.Get(fullName);
        if (info.Shape.Length != 3)
        {
            throw new ModelConfigException($"'{fullName}' is not a packed [experts, rows, cols] tensor (shape [{string.Join(", ", info.Shape)}])");
        }
        long experts = info.Shape[0], rows = info.Shape[1], cols = info.Shape[2];
        if (facts.Moe is { } moe && moe.Experts != experts)
        {
            throw new ModelConfigException($"'{fullName}' packs {experts} experts but the config states {moe.Experts}");
        }
        bool gateUp = suffix == PackedGateUp;
        if (gateUp && rows % 2 != 0)
        {
            throw new ModelConfigException($"'{fullName}' has {rows} rows per expert — a gate/up pair needs an even count");
        }
        long bytesPerExpert = info.DataLength / experts;
        long bytesPerRow = bytesPerExpert / rows;
        string layer = relative[..dot];
        var parts = new List<NamedTensorData>();
        var partNames = new List<string>();
        for (long e = 0; e < experts; e++)
        {
            var stems = gateUp
                ? new[] { ($"mlp.experts.{e}.gate_proj.weight", 0L, rows / 2), ($"mlp.experts.{e}.up_proj.weight", rows / 2, rows / 2) }
                : new[] { ($"mlp.experts.{e}.down_proj.weight", 0L, rows) };
            foreach (var (stem, rowStart, partRows) in stems)
            {
                long byteStart = e * bytesPerExpert + rowStart * bytesPerRow;
                parts.Add(new NamedTensorData
                {
                    Name = $"{layer}.{stem}",
                    Dtype = info.Dtype,
                    Shape = new[] { partRows, cols },
                    Source = (offset, count) => inventory.ReadBytes(fullName, byteStart + offset, count),
                    SourceLength = partRows * bytesPerRow,
                });
                partNames.Add(stem);
            }
        }
        return (parts, new FusedTensor
        {
            Name = suffix,
            Parts = partNames,
            LeadingShape = new List<long> { experts, rows },
        });
    }

    /// <summary>The text config of a (possibly wrapped) checkpoint config.</summary>
    private static JsonElement TextConfig(TextArchitectureFacts facts)
    {
        var root = facts.SourceConfig ?? throw new ModelConfigException("the checkpoint config was not carried");
        return root.TryGetProperty("text_config", out var text) && text.ValueKind == JsonValueKind.Object ? text : root;
    }

    private static int RequiredInt(JsonElement text, string name) =>
        text.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetInt32()
            : throw new ModelConfigException($"qwen4_exp config is missing '{name}'");

    private static int? OptionalInt(JsonElement text, string name) =>
        text.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;

    /// <summary>The facts Qwen4 adds over Qwen3.5, judged against the bound
    /// layer tensors.</summary>
    private sealed record Qwen4Facts(
        HyperConnectionSurface HyperConnections,
        SparseAttentionSurface? SparseAttention,
        NgramEmbeddingSurface? NgramEmbedding,
        string LinearGateActivation);

    private static Qwen4Facts JudgeQwen4(TextArchitectureFacts facts, HfInventory inventory, string prefix,
        IReadOnlyCollection<NamedTensorData> stackTensors)
    {
        var text = TextConfig(facts);
        var names = stackTensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
        int hidden = facts.HiddenSize;

        // ── gated residual ────────────────────────────────────────────────
        int streams = RequiredInt(text, "hc_count");
        int lowrank = RequiredInt(text, "hc_lowrank");
        if (streams < 2)
        {
            throw new ModelConfigException($"qwen4_exp needs hc_count > 1, the config states {streams}");
        }
        for (int l = 0; l < facts.NumLayers; l++)
        {
            foreach (var site in new[] { "attn_hyper_connection", "mlp_hyper_connection" })
            {
                ExpectShape(names, $"{l}.{site}.hc_norm.weight", streams * (long)hidden);
                ExpectShape(names, $"{l}.{site}.input_mix_weight_down.weight", lowrank, streams * (long)hidden);
                ExpectShape(names, $"{l}.{site}.input_mix_weight_up.weight", streams * (long)hidden, lowrank);
                ExpectShape(names, $"{l}.{site}.block_inject_weight.weight", streams, streams * (long)hidden);
            }
            if (names.ContainsKey($"{l}.input_layernorm.weight") || names.ContainsKey($"{l}.post_attention_layernorm.weight"))
            {
                throw new ModelConfigException($"layer {l} carries layer norms beside its gated residual — qwen4_exp has none");
            }
        }
        string mixer = $"{prefix}.hyper_connection_mixer";
        foreach (var (stem, shape) in new[]
                 {
                     ("hc_norm.weight", new[] { streams * (long)hidden }),
                     ("input_mix_weight_down.weight", new[] { (long)lowrank, streams * (long)hidden }),
                     ("input_mix_weight_up.weight", new[] { streams * (long)hidden, lowrank }),
                 })
        {
            if (!inventory.TryGet($"{mixer}.{stem}", out var info) || !info.Shape.SequenceEqual(shape))
            {
                throw new ModelConfigException($"qwen4_exp needs '{mixer}.{stem}' of shape [{string.Join(", ", shape)}]");
            }
        }

        // ── attention: the gate rides in q_proj, and QSA picks the keys ────
        SparseAttentionSurface? sparse = null;
        var softmaxLayers = Enumerable.Range(0, facts.NumLayers).Where(l => facts.LayerTypes[l] != "linear_attention").ToList();
        foreach (int l in softmaxLayers)
        {
            ExpectShape(names, $"{l}.self_attn.q_proj.weight", 2L * facts.NumQueryHeads * facts.HeadDim, hidden);
        }
        if (OptionalInt(text, "indexer_n_heads") is { } indexHeads)
        {
            sparse = new SparseAttentionSurface
            {
                Heads = indexHeads,
                KvHeads = RequiredInt(text, "indexer_kv_heads"),
                HeadDim = RequiredInt(text, "indexer_head_dim"),
                Budget = RequiredInt(text, "indexer_budget"),
                CompressRatio = RequiredInt(text, "indexer_compress_ratio"),
            };
            if (sparse.KvHeads != 1)
            {
                throw new ModelConfigException($"QSA needs indexer_kv_heads = 1, the config states {sparse.KvHeads}");
            }
            if (sparse.Heads <= 0 || sparse.HeadDim <= 0 || sparse.CompressRatio <= 0 || sparse.Budget % sparse.CompressRatio != 0)
            {
                throw new ModelConfigException(
                    $"QSA geometry is not judged: {sparse.Heads} heads of {sparse.HeadDim}, budget {sparse.Budget}, ratio {sparse.CompressRatio}");
            }
            int rotary = (int)(facts.HeadDim * facts.PartialRotaryFactor);
            if (rotary > sparse.HeadDim)
            {
                throw new ModelConfigException($"the attention's rotary width {rotary} does not fit the QSA head of {sparse.HeadDim}");
            }
            foreach (int l in softmaxLayers)
            {
                ExpectShape(names, $"{l}.self_attn.indexer.index_qk_proj.weight", (long)(sparse.Heads + 1) * sparse.HeadDim, hidden);
                ExpectShape(names, $"{l}.self_attn.indexer.q_layernorm.weight", sparse.HeadDim);
                ExpectShape(names, $"{l}.self_attn.indexer.k_layernorm.weight", sparse.HeadDim);
            }
        }
        else if (softmaxLayers.Any(l => names.ContainsKey($"{l}.self_attn.indexer.index_qk_proj.weight")))
        {
            throw new ModelConfigException("the layers carry QSA indexers but the config states no indexer geometry");
        }

        // ── the n-gram embedding ─────────────────────────────────────────
        NgramEmbeddingSurface? ngram = null;
        var pleIds = text.TryGetProperty("ple_layer_ids", out var ids) && ids.ValueKind == JsonValueKind.Array
            ? ids.EnumerateArray().Select(e => e.GetInt32()).Distinct().OrderBy(i => i).ToList()
            : new List<int>();
        if (pleIds.Count > 0)
        {
            if (pleIds.Any(id => id < 1 || id > facts.NumLayers))
            {
                throw new ModelConfigException($"ple_layer_ids are one-based layer numbers; [{string.Join(", ", pleIds)}] reach outside 1..{facts.NumLayers}");
            }
            int eos = text.TryGetProperty("eos_token_id", out var e) ? e.ValueKind switch
            {
                JsonValueKind.Number => e.GetInt32(),
                JsonValueKind.Array when e.GetArrayLength() > 0 => e[0].GetInt32(),
                _ => throw new ModelConfigException("PLE needs an eos_token_id"),
            } : throw new ModelConfigException("PLE needs an eos_token_id");
            var layers = pleIds.Select(id => id - 1).ToList();
            int shards = CountShards(names, layers[0]);
            ngram = new NgramEmbeddingSurface
            {
                Layers = layers,
                NgramSize = OptionalInt(text, "ngram_size") ?? 3,
                HeadsPerNgram = OptionalInt(text, "heads_per_ngram") ?? 8,
                VocabSizeBase = text.TryGetProperty("ngram_vocab_size_base", out var vb) ? vb.GetInt64() : 20_000_000,
                VocabSizeDivisor = OptionalInt(text, "make_ngram_vocab_size_divisible_by") ?? 128,
                Seed = text.TryGetProperty("seed", out var seed) ? seed.GetInt64() : 1234,
                EmbedDim = OptionalInt(text, "ple_embed_dim") ?? hidden,
                ConvKernel = OptionalInt(text, "ple_conv_kernel_size") ?? 4,
                EosTokenId = eos,
                VocabSize = facts.VocabSize,
                Shards = shards,
            };
            if (ngram.NgramSize < 2 || ngram.HeadsPerNgram < 1 || ngram.EmbedDim % ngram.NgramHeads != 0)
            {
                throw new ModelConfigException(
                    $"PLE embedding width {ngram.EmbedDim} does not split over {ngram.NgramHeads} n-gram heads");
            }
            long hcWidth = streams * (long)hidden;
            int headDim = ngram.EmbedDim / ngram.NgramHeads;
            for (int i = 0; i < layers.Count; i++)
            {
                int l = layers[i];
                if (CountShards(names, l) != shards)
                {
                    throw new ModelConfigException($"layer {l}: the n-gram table has {CountShards(names, l)} shards, layer {layers[0]} has {shards}");
                }
                ExpectShape(names, $"{l}.ple.key_proj.weight", hcWidth, ngram.EmbedDim);
                ExpectShape(names, $"{l}.ple.value_proj.weight", hidden, ngram.EmbedDim);
                ExpectShape(names, $"{l}.ple.norm_key.weight", hcWidth);
                ExpectShape(names, $"{l}.ple.norm_query.weight", hcWidth);
                ExpectShape(names, $"{l}.ple.norm_conv.weight", hcWidth);
                ExpectShape(names, $"{l}.ple.conv1d.weight", hcWidth, 1, ngram.ConvKernel);
                var layout = NgramHashing.For(ngram, i);
                long rows = 0;
                foreach (var shard in ShardNames(l, shards))
                {
                    var t = names[shard];
                    if (t.Shape.Length != 2 || t.Shape[1] != headDim)
                    {
                        throw new ModelConfigException($"'{shard}' is not a [rows, {headDim}] slice of the n-gram table");
                    }
                    rows += t.Shape[0];
                }
                if (rows != layout.PaddedRows)
                {
                    throw new ModelConfigException(
                        $"layer {l}: the n-gram shards hold {rows} rows but the config's hash heads need {layout.PaddedRows}");
                }
                // The checkpoint ships the hash's constants it was trained with
                // (transformers saves them as buffers); they must be the ones the
                // config derives, or every lookup reads the wrong rows. The
                // runtime recomputes them, so they are checked here, once.
                ExpectLongs(names, l, "layer_multipliers", layout.Multipliers);
                ExpectLongs(names, l, "ngram_heads_vocab_sizes", layout.HeadVocabSizes);
                ExpectLongs(names, l, "ngram_heads_offsets", layout.HeadOffsets);
            }
            var unexpected = names.Keys.Where(n => n.Contains(".ple.", StringComparison.Ordinal) &&
                                                   !layers.Contains(int.Parse(n[..n.IndexOf('.')]))).ToList();
            if (unexpected.Count > 0)
            {
                throw new ModelConfigException($"PLE tensors on layers ple_layer_ids does not name: {string.Join(", ", unexpected.Take(3))}");
            }
        }
        else if (names.Keys.Any(n => n.Contains(".ple.", StringComparison.Ordinal)))
        {
            throw new ModelConfigException("the layers carry PLE tensors but ple_layer_ids names none");
        }

        string gate = text.TryGetProperty("output_gate_type", out var g) && g.ValueKind == JsonValueKind.String
            ? g.GetString()!
            : facts.HiddenAct;
        if (gate is not ("silu" or "sigmoid"))
        {
            throw new ModelConfigException($"output_gate_type '{gate}' is not judged (silu or sigmoid)");
        }
        return new Qwen4Facts(new HyperConnectionSurface { Streams = streams, Lowrank = lowrank }, sparse, ngram, gate);
    }

    /// <summary>The n-gram table's shard tensors on a layer, in shard order.</summary>
    public static IEnumerable<string> ShardNames(int layer, int shards) =>
        Enumerable.Range(0, shards).Select(i => $"{layer}.ple.ple_embedding.ngram_embedding.shard_{i}.weight");

    private static int CountShards(Dictionary<string, NamedTensorData> names, int layer)
    {
        if (names.ContainsKey($"{layer}.ple.ple_embedding.ngram_embedding.weight"))
        {
            throw new ModelConfigException(
                $"layer {layer}: the n-gram table is stored whole; this build reads the sharded checkpoint layout (ngram_embedding.shard_i)");
        }
        int shards = 0;
        while (names.ContainsKey($"{layer}.ple.ple_embedding.ngram_embedding.shard_{shards}.weight"))
        {
            shards++;
        }
        if (shards == 0)
        {
            throw new ModelConfigException($"layer {layer} is a PLE layer but carries no n-gram table shards");
        }
        int stray = names.Keys.Count(n => n.StartsWith($"{layer}.ple.ple_embedding.ngram_embedding.shard_", StringComparison.Ordinal));
        if (stray != shards)
        {
            throw new ModelConfigException($"layer {layer}: the n-gram shards are not numbered 0..{shards - 1} without gaps");
        }
        return shards;
    }

    /// <summary>Checks a saved I64 hash buffer, when the checkpoint has one,
    /// against the values the config derives.</summary>
    private static void ExpectLongs(Dictionary<string, NamedTensorData> names, int layer, string buffer, long[] expected)
    {
        if (!names.TryGetValue($"{layer}.ple.ple_embedding.{buffer}", out var stored))
        {
            return;
        }
        if (stored.Dtype != Dtype.I64 || stored.Shape.Length != 1 || stored.Shape[0] != expected.Length)
        {
            throw new ModelConfigException($"layer {layer}: {buffer} is not an I64 vector of {expected.Length}");
        }
        var values = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, long>(stored.Data.AsSpan()).ToArray();
        if (!values.SequenceEqual(expected))
        {
            throw new ModelConfigException(
                $"layer {layer}: the checkpoint's n-gram {buffer} [{string.Join(", ", values)}] are not the ones its config gives " +
                $"[{string.Join(", ", expected)}]");
        }
    }

    private static void ExpectShape(Dictionary<string, NamedTensorData> names, string name, params long[] shape)
    {
        if (!names.TryGetValue(name, out var t))
        {
            throw new ModelConfigException($"qwen4_exp needs layer tensor '{name}'");
        }
        if (!t.Shape.SequenceEqual(shape))
        {
            throw new ModelConfigException(
                $"'{name}' has shape [{string.Join(", ", t.Shape)}], qwen4_exp needs [{string.Join(", ", shape)}]");
        }
    }

    /// <summary>The final mixer stands where the final norm does: its three
    /// tensors, object-relative.</summary>
    private static List<NamedTensorData> BindFinalMixer(HfInventory inventory, string prefix) =>
        new[] { "hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight" }
            .Select(stem => ToTensorData(inventory, $"{prefix}.hyper_connection_mixer.{stem}", stem))
            .ToList();
}
