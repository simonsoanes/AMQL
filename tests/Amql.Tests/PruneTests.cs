using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for the layer prune (<c>amql-cli prune</c>): dropping whole
/// decoder layers to meet a byte budget, the three ranking approaches
/// (provenance / corpus / random), the fail-closed floor, and the
/// acceptance scenario — merge two models, prune, then merge a smaller
/// model into the pruned container to prove it is still a complete,
/// mergeable model.
/// </summary>
public class PruneTests
{
    // ── fixtures ──────────────────────────────────────────────────────────

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
        WriteTokenizer(path, tokens);
        return path;
    }

    private static void WriteTokenizer(string containerOrModelDir, IReadOnlyList<string> tokens)
    {
        var vocab = new Dictionary<string, int>();
        for (int i = 0; i < tokens.Count; i++)
        {
            vocab[tokens[i]] = i;
        }
        var root = new JsonObject { ["model"] = new JsonObject { ["vocab"] = JsonSerializer.SerializeToNode(vocab)!.AsObject() } };
        File.WriteAllText(Path.Combine(containerOrModelDir, "tokenizer.json"), root.ToJsonString(ViJson.Options));
    }

    private static long DirBytes(string dir) =>
        Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories)
            .Sum(f => new FileInfo(f).Length);

    private static long[] LayerPayloadBytes(string containerPath, int layers)
    {
        using var container = Vindex3Container.Open(containerPath);
        string repId = container.CanonicalRepresentationId("target.decoder_stack");
        var entry = container.Index.Representations[repId];
        using var segment = SegmentFile.Open(Path.Combine(container.Root, entry.Segment));
        var bytes = new long[layers];
        foreach (var tensor in segment.Header.Tensors)
        {
            int dot = tensor.Name.IndexOf('.');
            if (dot > 0 && int.TryParse(tensor.Name.AsSpan(0, dot), out int l) && l >= 0 && l < layers)
            {
                bytes[l] += tensor.Len;
            }
        }
        return bytes;
    }

    private static List<string> StackTensorNames(string containerPath)
    {
        using var container = Vindex3Container.Open(containerPath);
        string repId = container.CanonicalRepresentationId("target.decoder_stack");
        var entry = container.Index.Representations[repId];
        using var segment = SegmentFile.Open(Path.Combine(container.Root, entry.Segment));
        return segment.Header.Tensors.Select(t => t.Name).OrderBy(n => n, StringComparer.Ordinal).ToList();
    }

    private static void AssertSingleLayerPlan(string containerPath, int expectedLayers)
    {
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.Equal(expectedLayers, plan.Layers.Count);
    }

    private static readonly string[] Tokens5 = { "a", "b", "c", "d", "e" };

    /// <summary>Layers large enough that a layer's payload dwarfs the
    /// rebuilt segment header, so dropping exactly the budgeted number of
    /// layers is deterministic in the tests.</summary>
    private static Dims BigDims(int vocab, int layers) => new(Vocab: vocab, Hidden: 64, Layers: layers, Intermediate: 128);

    /// <summary>Merge-friendly dims: the alignment fit needs more shared
    /// anchors than the width (anchors are ≫ 8 here), while the layer
    /// payload still dwarfs the rebuilt header so the drop count stays
    /// deterministic.</summary>
    private static Dims MergeDims(int vocab, int layers) => new(Vocab: vocab, Hidden: 8, Layers: layers, Intermediate: 16);

    private static string[] Tokens(string prefix, int from, int to) =>
        Enumerable.Range(from, to - from).Select(i => $"{prefix}{i}").ToArray();

    /// <summary>A real BPE tokenizer.json over a token list (the pruner's
    /// corpus approach needs HfTokenizer, which only serves BPE files);
    /// ids 0..n-1 match the embedding rows. Plain letters only — the
    /// split regex isolates a..j.</summary>
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
                        ["pattern"] = new JsonObject { ["Regex"] = "(?i:[a-j?])| +|[\\s\\S]" },
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

    // ── provenance approach (position descending when no token-map) ───────

    [Fact]
    public void Prune_Drops_Exactly_The_Top_Layers_To_Hit_The_Byte_Budget()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "four", BigDims(vocab: 5, layers: 4), Tokens5);
        var lb = LayerPayloadBytes(container, 4);
        long input = DirBytes(container);
        long target = input - lb[3] - lb[2] - 64; // provenance ties across layers → position descending → 3, 2

        var outDir = Path.Combine(dir.Path, "pruned");
        var report = LayerPruner.Prune(container, outDir, new PruneOptions(target));

        Assert.Equal(new[] { 3, 2 }, report.DroppedLayers);
        Assert.Equal(2, report.KeptLayers);
        Assert.Equal(4, report.OriginalLayers);
        Assert.True(report.FinalBytes <= target, $"final {report.FinalBytes} exceeds target {target}");
        Assert.True(DirBytes(outDir) <= target, "the output container exceeds the budget");

        using (var pruned = Vindex3Container.Open(outDir))
        {
            Assert.Equal(2, pruned.Index.NumLayers);
            Assert.True(pruned.VerifyIntegrity().Ok);

            // The kept layers are renumbered 0..1; nothing of 2/3 remains.
            var names = StackTensorNames(outDir);
            Assert.Contains(names, n => n.StartsWith("0.", StringComparison.Ordinal));
            Assert.Contains(names, n => n.StartsWith("1.", StringComparison.Ordinal));
            Assert.DoesNotContain(names, n => n.StartsWith("2.", StringComparison.Ordinal));
            Assert.DoesNotContain(names, n => n.StartsWith("3.", StringComparison.Ordinal));

            // The prune provenance is recorded in the index.
            var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, "index.json")));
            var prune = doc.RootElement.GetProperty("prune");
            Assert.Equal(2, prune.GetProperty("to_layers").GetInt32());
            Assert.Equal(2, prune.GetProperty("dropped_layers").GetArrayLength());
        }

        AssertSingleLayerPlan(outDir, 2);
    }

    // ── fail-closed floor ─────────────────────────────────────────────────

    [Fact]
    public void Prune_Refuses_When_The_Target_Is_Below_The_Floor()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "two", new Dims(Vocab: 5, Layers: 2), Tokens5);
        var outDir = Path.Combine(dir.Path, "pruned");

        var e = Assert.Throws<MergeException>(
            () => LayerPruner.Prune(container, outDir, new PruneOptions(TargetBytes: 1)));
        Assert.Contains("cannot reach", e.Message);
        Assert.False(Directory.Exists(outDir), "a failed prune must not leave an output tree");
    }

    [Fact]
    public void Prune_Refuses_When_Min_Layers_Exceeds_The_Stack()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "two", new Dims(Vocab: 5, Layers: 2), Tokens5);
        var e = Assert.Throws<MergeException>(
            () => LayerPruner.Prune(container, Path.Combine(dir.Path, "pruned"),
                new PruneOptions(TargetBytes: 1024, MinLayers: 3)));
        Assert.Contains("cannot keep", e.Message);
    }

    // ── random approach: seeded determinism ───────────────────────────────

    [Fact]
    public void Prune_Random_Is_Deterministic_For_A_Seed()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "four", new Dims(Vocab: 5, Layers: 4), Tokens5);
        var lb = LayerPayloadBytes(container, 4);
        long input = DirBytes(container);
        long target = input - lb[0] - lb[1] - lb[2] - 64; // any two-layer drop lands under the budget

        var one = LayerPruner.Prune(container, Path.Combine(dir.Path, "r1"),
            new PruneOptions(target, Approach: PruneApproach.Random, Seed: 999));
        var two = LayerPruner.Prune(container, Path.Combine(dir.Path, "r2"),
            new PruneOptions(target, Approach: PruneApproach.Random, Seed: 999));

        Assert.Equal(2, one.DroppedLayers.Count);
        Assert.Equal(one.DroppedLayers, two.DroppedLayers);
        Assert.Equal(one.FinalBytes, two.FinalBytes);

        // Same seed → the same layers land in the output, byte for byte.
        var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir.Path, "r1", "index.json")));
        Assert.Equal(
            JsonSerializer.Serialize(doc.RootElement.GetProperty("prune").GetProperty("dropped_layers")),
            JsonSerializer.Serialize(
                JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir.Path, "r2", "index.json")))
                    .RootElement.GetProperty("prune").GetProperty("dropped_layers")));
    }

    // ── corpus approach: one forward pass over the corpus ─────────────────

    [Fact]
    public void Prune_Corpus_Drops_A_Layer_Measured_On_The_Corpus()
    {
        using var dir = new TempDir();
        var container = BuildContainer(dir, "synth", BigDims(vocab: 5, layers: 2), Tokens5);
        WriteBpeTokenizer(container, Tokens5); // the corpus pass needs a BPE tokenizer beside the container

        var lb = LayerPayloadBytes(container, 2);
        long input = DirBytes(container);
        long target = input - lb.Min() - 64; // exactly one layer droppable under the budget

        var outDir = Path.Combine(dir.Path, "pruned");
        var report = LayerPruner.Prune(container, outDir,
            new PruneOptions(target, Approach: PruneApproach.Corpus, CorpusText: "abcde"));

        Assert.Single(report.DroppedLayers);
        Assert.Equal(1, report.KeptLayers);
        Assert.True(report.FinalBytes <= target, $"final {report.FinalBytes} exceeds target {target}");

        using var pruned = Vindex3Container.Open(outDir);
        Assert.Equal(1, pruned.Index.NumLayers);
        Assert.True(pruned.VerifyIntegrity().Ok);
        AssertSingleLayerPlan(outDir, 1);
    }

    // ── the acceptance scenario: merge → prune → merge a smaller model ────

    [Fact]
    public void Prune_Merged_Container_Then_Import_A_Smaller_Model_Is_A_Complete_Model()
    {
        using var dir = new TempDir();
        var baseContainer = BuildContainer(dir, "base", MergeDims(40, 4), Tokens("t", 0, 40));
        var importedContainer = BuildContainer(dir, "imported", MergeDims(32, 4), Tokens("t", 8, 40));
        var merged = Path.Combine(dir.Path, "merged");
        ModelMerger.Import(baseContainer, importedContainer, merged, importedIsContainer: true);

        // Both models are same-shaped, so no grown tensors: the provenance
        // approach falls back to position descending.
        var lb = LayerPayloadBytes(merged, 4);
        long input = DirBytes(merged);
        long target = input - lb[3] - lb[2] - 64;
        var pruned = Path.Combine(dir.Path, "pruned");
        var report = LayerPruner.Prune(merged, pruned, new PruneOptions(target));
        Assert.Equal(new[] { 3, 2 }, report.DroppedLayers);
        Assert.True(report.FinalBytes <= target);

        // The acceptance: the pruned container merges like any complete
        // model — a smaller model imports into it and the result is a
        // valid, planable container.
        var small = BuildContainer(dir, "small", MergeDims(24, 2), Tokens("t", 16, 40));
        var result = Path.Combine(dir.Path, "result");
        ModelMerger.Import(pruned, small, result, importedIsContainer: true);

        using var opened = Vindex3Container.Open(result);
        Assert.Equal(2, opened.Index.NumLayers); // the pruned (2-layer) container scaffolds the result
        Assert.True(opened.VerifyIntegrity().Ok);
        AssertSingleLayerPlan(result, 2);
    }
}