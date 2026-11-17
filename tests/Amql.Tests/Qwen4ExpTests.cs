using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Qwen4 (transformers <c>qwen4_exp</c>; Qwen3.8-Flash-Next is its open
/// preview) on the managed path: the Gated Residual, QSA's selected keys, the
/// hashed n-gram embedding with its sharded table, and packed experts.
/// <para><c>fixtures/arch-zoo/qwen4_exp-tiny</c> (made by
/// <c>make_qwen4_exp_tiny.py</c> with transformers' own qwen4_exp) is sized so
/// each of those changes the answer; the arch-zoo tests also encode it, match
/// its logits and round-trip it through HF export.</para>
/// </summary>
public sealed class Qwen4ExpTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");
    private static string Tiny => Path.Combine(Zoo, "qwen4_exp-tiny");
    private static readonly ArchMapper.EncodeOptions TextOnly = new() { IncludeVision = false, IncludeMtp = false };

    private static string Encode(TempDir dir, string source, string name = "c")
    {
        string containerDir = Path.Combine(dir.Path, name);
        ModelToContainer.Encode(source, containerDir, "qwen4-tiny", TextOnly);
        return containerDir;
    }

    [Fact]
    public void Tiny_Fixture_Selects_Keys_And_Crosses_A_Document_Boundary()
    {
        // The golden comparison only proves QSA and the n-gram hash if the
        // fixture leans on them: positions past budget/ratio blocks attend to a
        // subset, and an end-of-sequence token sits inside the sequence.
        var (ids, _) = ArchZooTests.Golden("qwen4_exp-tiny");
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Tiny, "config.json")));
        var root = config.RootElement;
        int blocks = root.GetProperty("indexer_budget").GetInt32() / root.GetProperty("indexer_compress_ratio").GetInt32();
        int ratio = root.GetProperty("indexer_compress_ratio").GetInt32();
        Assert.True(ids.Length / ratio > blocks + 4, "most positions must exceed the QSA budget");
        int eos = root.GetProperty("eos_token_id").GetInt32();
        Assert.Contains(eos, ids[1..^1]);
        Assert.NotEmpty(root.GetProperty("ple_layer_ids").EnumerateArray());
        Assert.Equal("sigmoid", root.GetProperty("output_gate_type").GetString());
    }

    [Fact]
    public void Imports_The_Gated_Residual_Qsa_And_Sharded_Ngram_Table()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir, Tiny));
        var surface = container.Graph!.Component("target").Execution!;
        Assert.Equal(4, surface.HyperConnections!.Streams);
        Assert.Equal(16, surface.HyperConnections.Lowrank);
        Assert.Equal(2, surface.SparseAttention!.BlockTopK);
        var ngram = surface.NgramEmbedding!;
        Assert.Equal(new[] { 1 }, ngram.Layers);
        Assert.Equal(4, ngram.Shards);
        Assert.Equal(2, ngram.EosTokenId);

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.Null(plan.FinalNorm);
        Assert.NotNull(plan.FinalMixer);
        Assert.Equal(4 * 64, plan.ResidualWidth);
        Assert.NotNull(plan.Layers[1].Ngram);
        Assert.Equal(new long[] { 0, 60, 120, 180 }, plan.Layers[1].Ngram!.ShardStarts);
        Assert.True(plan.Layers[0].LinearAttention!.SigmoidGate);
        Assert.NotNull(plan.Layers[2].Attention!.Indexer);
        Assert.NotNull(plan.Layers[3].Ffn!.Routed!.SharedExpertGate);
        Assert.NotNull(GpuEngine.Unsupported(plan));
    }

    /// <summary>Prefill part of the sequence, then step through the rest: the
    /// n-gram history and dilated conv, the indexer's key and block caches and
    /// the recurrent state all carry over, so each step's logits are the golden
    /// row for that position.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(13)]
    public void Decoding_Step_By_Step_Matches_The_Full_Forward(int prefix)
    {
        var (ids, golden) = ArchZooTests.Golden("qwen4_exp-tiny");
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir, Tiny));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        using var session = new DecodeSession(plan, store) { AllowGpuEngine = false };

        double worst = Diff(session.Prefill(ids[..prefix]).Row(0), golden[prefix - 1]);
        for (int t = prefix; t < ids.Length; t++)
        {
            worst = Math.Max(worst, Diff(session.Step(ids[t]).Row(0), golden[t]));
        }
        Assert.True(worst < 2e-3, $"worst logit diff stepping from {prefix}: {worst:G4}");

        // A reset session starts the n-gram history and caches afresh.
        session.Reset();
        Assert.True(Diff(session.Prefill(ids).Row(0), golden[^1]) < 2e-3);
    }

    /// <summary>The published checkpoint packs each layer's experts into two
    /// 3-D tensors; transformers saves them one per expert. Both import to the
    /// same model, and export writes the packing back.</summary>
    [Fact]
    public void Packed_Experts_Import_And_Export_Packed()
    {
        var (ids, golden) = ArchZooTests.Golden("qwen4_exp-tiny");
        using var dir = new TempDir();
        string packed = Path.Combine(dir.Path, "packed");
        WritePacked(Tiny, packed);
        string containerDir = Encode(dir, packed);
        Assert.True(ArchZooTests.WorstDiff(ArchZooTests.RunPrefill(containerDir, ids), golden) < 2e-3);

        string exported = Path.Combine(dir.Path, "hf");
        using (var container = Vindex3Container.Open(containerDir))
        {
            ModelExporter.Export(container, exported, patch: null);
        }
        using var file = SafetensorsFile.Open(Path.Combine(exported, "model.safetensors"));
        Assert.Equal(new long[] { 4, 64, 64 }, file.GetTensor("model.layers.0.mlp.experts.gate_up_proj").Shape);
        Assert.Equal(new long[] { 4, 64, 32 }, file.GetTensor("model.layers.0.mlp.experts.down_proj").Shape);
        Assert.False(file.Contains("model.layers.0.mlp.experts.0.gate_proj.weight"));
        using var source = SafetensorsFile.Open(Path.Combine(packed, "model.safetensors"));
        Assert.Equal(source.ReadBytes("model.layers.3.mlp.experts.gate_up_proj"), file.ReadBytes("model.layers.3.mlp.experts.gate_up_proj"));

        string again = Encode(dir, exported, "c2");
        Assert.True(ArchZooTests.WorstDiff(ArchZooTests.RunPrefill(again, ids), golden) < 2e-3);
    }

    [Fact]
    public void A_Missing_Ngram_Shard_Refuses_The_Checkpoint()
    {
        using var dir = new TempDir();
        string broken = Path.Combine(dir.Path, "broken");
        Rewrite(Tiny, broken, tensors => tensors.Where(t => !t.Name.EndsWith("ngram_embedding.shard_3.weight", StringComparison.Ordinal)));
        var e = Assert.ThrowsAny<Exception>(() => Encode(dir, broken));
        Assert.Contains("n-gram", e.Message);
    }

    [Fact]
    public void A_Wrong_Hash_Seed_Refuses_The_Checkpoint()
    {
        // The saved multipliers are what the table was trained against: a
        // config whose seed derives others would look up the wrong rows.
        using var dir = new TempDir();
        string reseeded = Path.Combine(dir.Path, "reseeded");
        Rewrite(Tiny, reseeded, tensors => tensors);
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(reseeded, "config.json")))!.AsObject();
        config["seed"] = 4321;
        File.WriteAllText(Path.Combine(reseeded, "config.json"), config.ToJsonString());
        var e = Assert.ThrowsAny<Exception>(() => Encode(dir, reseeded));
        Assert.Contains("layer_multipliers", e.Message);
    }

    /// <summary>The hash constants at Qwen3.8-Flash-Next's own size, from
    /// transformers' <c>_build_layer_multipliers</c> and
    /// <c>_find_nth_prime_after</c>: 16 hash heads of consecutive primes after
    /// 20,000,000, padded to 128 — 2,500,012 rows in each of its 128 shards.</summary>
    [Fact]
    public void Flash_Next_Hash_Layout_Matches_Transformers()
    {
        var surface = new NgramEmbeddingSurface
        {
            Layers = new[] { 1 }, NgramSize = 3, HeadsPerNgram = 8, VocabSizeBase = 20_000_000, VocabSizeDivisor = 128,
            Seed = 1234, EmbedDim = 2560, ConvKernel = 4, EosTokenId = 248044, VocabSize = 248320, Shards = 128,
        };
        var layout = NgramHashing.For(surface, 0);
        Assert.Equal(new[] { 23703573157769L, 20109073645365L, 8052911324071L }, layout.Multipliers);
        Assert.Equal(new long[]
        {
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
        }, layout.HeadVocabSizes);
        Assert.Equal(320001536L, layout.PaddedRows);
        Assert.Equal(0L, layout.PaddedRows % surface.Shards);
    }

    [Fact]
    public void Ngram_Ids_Do_Not_Reach_Back_Across_End_Of_Sequence()
    {
        var surface = new NgramEmbeddingSurface
        {
            Layers = new[] { 0 }, NgramSize = 3, HeadsPerNgram = 1, VocabSizeBase = 50, VocabSizeDivisor = 1,
            Seed = 1234, EmbedDim = 2, ConvKernel = 4, EosTokenId = 2, VocabSize = 128, Shards = 1,
        };
        var layout = NgramHashing.For(surface, 0);
        Span<long> afterEos = stackalloc long[2], fresh = stackalloc long[2];
        // "9 2 | 5": the 2- and 3-grams of 5 read the boundary, not the 9.
        NgramHashing.Rows(layout, surface, new[] { 9, 2 }, 5, afterEos);
        // The start of the sequence reads as end-of-sequence tokens.
        NgramHashing.Rows(layout, surface, new[] { 2, 2 }, 5, fresh);
        Assert.Equal(fresh.ToArray(), afterEos.ToArray());

        // "7 9 5" and "2 | 9 5": the same 2-gram (9 5); the 3-gram of the second
        // stops at the boundary.
        Span<long> inside = stackalloc long[2], across = stackalloc long[2];
        NgramHashing.Rows(layout, surface, new[] { 7, 9 }, 5, inside);
        NgramHashing.Rows(layout, surface, new[] { 2, 9 }, 5, across);
        Assert.Equal(inside[0], across[0]);
        Assert.NotEqual(inside[1], across[1]);
    }

    /// <summary>A Qwen3.5 model exported as Qwen4 (the exact rewrite: norms
    /// folded into the gated residual, the dense MLP as one routed expert, a
    /// QSA budget covering the context) imports and runs to the source model's
    /// transformers logits.</summary>
    [Fact]
    public void Qwen35_Exported_As_Qwen4_Runs_To_The_Source_Logits()
    {
        var (ids, golden) = ArchZooTests.Golden("qwen3_5-gdn");
        using var dir = new TempDir();
        string source = Path.Combine(dir.Path, "source");
        ModelToContainer.Encode(Path.Combine(Zoo, "qwen3_5-gdn"), source, "qwen35-gdn", TextOnly);
        string exported = Path.Combine(dir.Path, "qwen4");
        using (var container = Vindex3Container.Open(source))
        {
            ModelExporter.Export(container, exported, patch: null, arch: Qwen4NextLayout.Arch);
        }
        string again = Encode(dir, exported, "again");
        using (var container = Vindex3Container.Open(again))
        {
            var surface = container.Graph!.Component("target").Execution!;
            Assert.NotNull(surface.HyperConnections);
            Assert.NotNull(surface.SparseAttention);
        }
        double worst = ArchZooTests.WorstDiff(ArchZooTests.RunPrefill(again, ids), golden);
        Assert.True(worst < 2e-3, $"qwen3.5 → qwen4 export → import: worst logit diff {worst:G4}");
    }

    private static double Diff(ReadOnlySpan<float> mine, float[] reference)
    {
        double worst = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(mine[i] - reference[i]));
        }
        return worst;
    }

    /// <summary>Copies a checkpoint directory, passing its tensors through <paramref name="edit"/>.</summary>
    private static void Rewrite(string from, string to, Func<IEnumerable<TensorPayload>, IEnumerable<TensorPayload>> edit)
    {
        Directory.CreateDirectory(to);
        foreach (var file in Directory.EnumerateFiles(from).Where(f => !f.EndsWith(".safetensors", StringComparison.Ordinal)))
        {
            File.Copy(file, Path.Combine(to, Path.GetFileName(file)));
        }
        List<TensorPayload> tensors;
        using (var source = SafetensorsFile.Open(Path.Combine(from, "model.safetensors")))
        {
            tensors = source.TensorNames.Select(n =>
            {
                var info = source.GetTensor(n);
                return new TensorPayload { Name = n, Dtype = info.Dtype, Shape = info.Shape, Data = source.ReadBytes(n) };
            }).ToList();
        }
        SafetensorsWriter.Write(Path.Combine(to, "model.safetensors"), edit(tensors));
    }

    /// <summary>The fixture with its experts packed the way the published
    /// checkpoint stores them: <c>gate_up_proj</c> [E, 2I, H] (each expert's gate
    /// rows, then its up rows) and <c>down_proj</c> [E, H, I].</summary>
    private static void WritePacked(string from, string to) => Rewrite(from, to, tensors =>
    {
        var list = tensors.ToList();
        var byName = list.ToDictionary(t => t.Name, StringComparer.Ordinal);
        var result = list.Where(t => !t.Name.Contains(".mlp.experts.", StringComparison.Ordinal)).ToList();
        for (int l = 0; l < 4; l++)
        {
            string p = $"model.layers.{l}.mlp.experts";
            var gateUp = Enumerable.Range(0, 4).SelectMany(e =>
                byName[$"{p}.{e}.gate_proj.weight"].Data.Concat(byName[$"{p}.{e}.up_proj.weight"].Data)).ToArray();
            var down = Enumerable.Range(0, 4).SelectMany(e => byName[$"{p}.{e}.down_proj.weight"].Data).ToArray();
            result.Add(new TensorPayload { Name = $"{p}.gate_up_proj", Dtype = Dtype.F32, Shape = new long[] { 4, 64, 64 }, Data = gateUp });
            result.Add(new TensorPayload { Name = $"{p}.down_proj", Dtype = Dtype.F32, Shape = new long[] { 4, 64, 32 }, Data = down });
        }
        return result;
    });
}
