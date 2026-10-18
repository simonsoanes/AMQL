using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for <c>prune --approach star-elastic</c> — Star-Elastic
/// structured FFN-channel width pruning: rank the intermediate channels by
/// activation magnitude on one forward pass, slice gate/up/down to the top
/// channels on every dense layer, and emit a smaller complete container.
///
/// The second test is the independent-oracle check: the score is recomputed
/// here with a hand-rolled matmul (not <c>TensorOps.MatMulTransposedB</c>)
/// and the slice is re-derived with float-level row/column gathers (not the
/// production byte-level gather), so the ranking and the kept values are
/// verified against a path that shares no assumptions with the pruner.
/// </summary>
public class StarElasticPruneTests
{
    private static readonly string[] Tokens5 = { "a", "b", "c", "d", "e" };

    /// <summary>Wide FFN with hidden 32 and intermediate 64 — enough channel
    /// headroom that dropping one channel is deterministic and the ranking is
    /// over a real multi-channel set.</summary>
    private static Dims SeDims(int vocab = 9) => new(Vocab: vocab, Hidden: 32, Layers: 2, Intermediate: 64);

    private static string BuildContainer(TempDir dir, string name, Dims dims, IReadOnlyList<string> tokens)
    {
        var spec = SyntheticModel.BuildSpec(dims);
        var renamed = new ContainerSpec
        {
            Model = name,
            Family = spec.Family,
            HiddenSize = spec.HiddenSize,
            NumLayers = spec.NumLayers,
            SystemGraph = spec.SystemGraph,
            Representations = spec.Representations,
            PrecisionMap = spec.PrecisionMap,
        };
        var path = Path.Combine(dir.Path, name);
        ContainerEncoder.Encode(path, renamed);
        WriteBpeTokenizer(path, tokens);
        return path;
    }

    private static void WriteBpeTokenizer(string dir, IReadOnlyList<string> tokens)
    {
        var vocab = new JsonObject();
        for (int i = 0; i < tokens.Count; i++)
        {
            vocab[tokens[i]] = i;
        }
        var root = new JsonObject
        {
            ["version"] = "1.0.0",
            ["added_tokens"] = new JsonArray(),
            ["normalizer"] = new JsonObject { ["type"] = "NFC" },
            ["pre_tokenizer"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JsonArray
                {
                    new JsonObject
                    {
                        ["type"] = "Split",
                        ["pattern"] = new JsonObject { ["Regex"] = "(?i:[a-e?])| +|[\\s\\S]" },
                        ["behavior"] = "Isolated",
                        ["invert"] = false,
                    },
                    new JsonObject
                    {
                        ["type"] = "ByteLevel",
                        ["add_prefix_space"] = false,
                        ["trim_offsets"] = false,
                        ["use_regex"] = false,
                    },
                },
            },
            ["decoder"] = new JsonObject
            {
                ["type"] = "ByteLevel",
                ["add_prefix_space"] = false,
                ["trim_offsets"] = false,
                ["use_regex"] = false,
            },
            ["model"] = new JsonObject
            {
                ["type"] = "BPE",
                ["unk_token"] = null,
                ["ignore_merges"] = false,
                ["vocab"] = vocab,
                ["merges"] = new JsonArray(),
            },
        };
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), root.ToJsonString(ViJson.Options));
    }

    private static long DirBytes(string dir) =>
        Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories)
            .Sum(f => new FileInfo(f).Length);

    // ── end to end: valid, smaller, budgeted, planable ─────────────────────

    [Fact]
    public void StarElastic_Prunes_Ffn_Width_And_Produces_A_Valid_Model()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "se", SeDims(), Tokens5);
        long input = DirBytes(container);

        // Dropping one gated channel frees 3 tensors × hidden(32) × 4 bytes ×
        // 2 layers; target anything below the full size forces a drop.
        long target = input - 1;

        var outDir = Path.Combine(dir.Path, "pruned");
        var report = StarElasticPruner.Prune(container, outDir,
            new StarElasticOptions(TargetBytes: target, CorpusText: "abcdeabcde"));

        Assert.True(report.KeptIntermediate < report.OriginalIntermediate,
            "the prune must drop at least one channel");
        Assert.Equal(64, report.OriginalIntermediate);
        Assert.Equal(2, report.Layers);
        Assert.Equal(2, report.DenseLayers);
        Assert.True(report.FinalBytes <= target, $"final {report.FinalBytes} exceeds target {target}");
        Assert.True(DirBytes(outDir) <= target, "the output container exceeds the budget");

        using var pruned = Vindex3Container.Open(outDir);
        Assert.True(pruned.VerifyIntegrity().Ok);

        // The planner agrees on the reduced homogeneous FFN width, and the
        // sliced tensors carry the new shapes.
        using var store = pruned.CreateOperandStore();
        var plan = Planner.Plan(pruned, "target", store);
        Assert.Equal(2, plan.Layers.Count);
        foreach (var layer in plan.Layers)
        {
            Assert.Equal(report.KeptIntermediate, layer.Ffn!.Dense!.IntermediateSize);
        }
        foreach (var name in new[] { "0.mlp.gate_proj.weight", "0.mlp.up_proj.weight", "0.mlp.down_proj.weight" })
        {
            var t = store.Resolve("target.decoder_stack", name);
            string shape = string.Join("x", t.Shape);
            if (name.EndsWith(".down_proj.weight", StringComparison.Ordinal))
            {
                Assert.Equal($"32x{report.KeptIntermediate}", shape);
            }
            else
            {
                Assert.Equal($"{report.KeptIntermediate}x32", shape);
            }
        }

        // The prune provenance is recorded in the index.
        var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, "index.json")));
        var prune = doc.RootElement.GetProperty("prune");
        Assert.Equal("star-elastic", prune.GetProperty("approach").GetString());
        Assert.Equal("ffn_channels", prune.GetProperty("axis").GetString());
        Assert.Equal(report.KeptIntermediate, prune.GetProperty("to_intermediate").GetInt32());
    }

    // ── the independent-oracle check ───────────────────────────────────────

    [Fact]
    public void StarElastic_Ranking_And_Slice_Match_An_Independent_Oracle()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "se", SeDims(), Tokens5);
        long input = DirBytes(container);
        long target = input - 3 * 32 * 4 * 2 * 8; // drop ~8 channels worth

        var outDir = Path.Combine(dir.Path, "pruned");
        var report = StarElasticPruner.Prune(container, outDir,
            new StarElasticOptions(TargetBytes: target, CorpusText: "abcdeabcde"));
        int kept = report.KeptIntermediate;
        Assert.True(kept < 64 && kept > 0);

        // 1. Recompute the score with a hand-rolled matmul (independent of
        //    TensorOps.MatMulTransposedB) and verify the pruner kept the same
        //    top-K channels.
        using (var origContainer = Vindex3Container.Open(container))
        using (var origStore = origContainer.CreateOperandStore())
        {
            var plan = Planner.Plan(origContainer, "target", origStore);
            int hidden = plan.HiddenSize;
            int inter = 64;
            var score = new double[inter];
            var runtime = new GenericRuntime(plan, origStore);
            runtime.FfnInputCapture = (layer, hf) =>
            {
                var up = runtime.Weights.Matrix(plan.Layers[layer].Ffn!.Dense!.Up, inter, hidden);
                for (int r = 0; r < hf.Rows; r++)
                {
                    int hOff = r * hidden;
                    for (int j = 0; j < inter; j++)
                    {
                        double s = 0;
                        for (int k = 0; k < hidden; k++)
                        {
                            s += hf.Data[hOff + k] * up.Data[j * hidden + k];
                        }
                        score[j] += Math.Abs(s);
                    }
                }
            };
            foreach (var token in EncodeTokens(origContainer, "abcdeabcde"))
            {
                runtime.StepForward(token);
            }

            var expected = Enumerable.Range(0, inter)
                .OrderByDescending(j => score[j])
                .ThenBy(j => j)
                .Take(kept)
                .OrderBy(j => j)
                .ToArray();
            Assert.Equal(expected, report.KeptChannels.ToArray());
        }

        // 2. Verify the sliced tensors are byte-exact: the pruned gate/up/down
        //    equal the original gathered (float-level, widened) at the kept
        //    channels — independent of the pruner's byte-granular gather.
        using (var orig = Vindex3Container.Open(container))
        using (var pruned = Vindex3Container.Open(outDir))
        {
            var keptSet = report.KeptChannels.ToArray();
            foreach (var l in new[] { 0, 1 })
            {
                foreach (var kind in new[] { "gate_proj", "up_proj", "down_proj" })
                {
                    string name = $"{l}.mlp.{kind}.weight";
                    float[] original = Widen(orig, name);
                    float[] prunedVals = Widen(pruned, name);
                    float[] expected = kind == "down_proj"
                        ? KeepCols(original, rows: 32, cols: 64, keptSet)
                        : KeepRows(original, rows: 64, cols: 32, keptSet);
                    Assert.Equal(expected, prunedVals);
                }
            }
        }
    }

    private static float[] Widen(Vindex3Container container, string tensorName)
    {
        using var store = container.CreateOperandStore();
        var r = store.Resolve("target.decoder_stack", tensorName);
        return BitPattern.WidenToF32(r.Dtype, r.Payload);
    }

    private static float[] KeepRows(float[] src, int rows, int cols, int[] kept)
    {
        var dst = new float[kept.Length * cols];
        for (int i = 0; i < kept.Length; i++)
        {
            Array.Copy(src, kept[i] * cols, dst, i * cols, cols);
        }
        return dst;
    }

    private static float[] KeepCols(float[] src, int rows, int cols, int[] kept)
    {
        var dst = new float[rows * kept.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int i = 0; i < kept.Length; i++)
            {
                dst[r * kept.Length + i] = src[r * cols + kept[i]];
            }
        }
        return dst;
    }

    private static int[] EncodeTokens(Vindex3Container container, string text)
    {
        return Amql.Hf.HfTokenizer.FromModelDir(container.Root).EncodeToIds(text).ToArray();
    }

    // ── fail-closed behaviour ──────────────────────────────────────────────

    [Fact]
    public void StarElastic_Refuses_When_Target_Is_Below_The_Floor()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "se", SeDims(), Tokens5);
        var e = Assert.Throws<MergeException>(
            () => StarElasticPruner.Prune(container, Path.Combine(dir.Path, "p"),
                new StarElasticOptions(TargetBytes: 1, CorpusText: "abcde")));
        Assert.Contains("cannot reach", e.Message);
    }

    [Fact]
    public void StarElastic_Requires_A_Corpus()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "se", SeDims(), Tokens5);
        var e = Assert.Throws<MergeException>(
            () => StarElasticPruner.Prune(container, Path.Combine(dir.Path, "p"),
                new StarElasticOptions(TargetBytes: 1024)));
        Assert.Contains("--text", e.Message);
    }

    [Fact]
    public void StarElastic_Refuses_Routed_Only_Models()
    {
        using var dir = new TempDir();
        var routed = BuildContainer(dir, "moe", SeDims() with { MoE = true, Experts = 2, TopK = 1 }, Tokens5);
        var e = Assert.Throws<MergeException>(
            () => StarElasticPruner.Prune(routed, Path.Combine(dir.Path, "p"),
                new StarElasticOptions(TargetBytes: 1024, CorpusText: "abcde")));
        Assert.Contains("dense FFN layer", e.Message);
    }
}