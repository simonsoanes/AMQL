using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Hf;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Tests for the model import/merge (<c>amql-cli import</c>): tokenization
/// mapping, anchored-alignment blending, vocabulary growth with shape
/// evolution, stack scaffolding with per-layer provenance, and the export
/// round trip of a merged container.
/// </summary>
public class MergeTests
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

    private static (float[] Rows, Dtype Dtype) Table(string containerPath, string objectId, int width)
    {
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var resolution = store.Resolve(objectId, "weight");
        var values = BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
        int rows = values.Length / (int)resolution.Shape[1];
        var padded = new float[rows * width];
        for (int r = 0; r < rows; r++)
        {
            Array.Copy(values, r * (int)resolution.Shape[1], padded, r * width, (int)resolution.Shape[1]);
        }
        return (padded, resolution.Dtype);
    }

    private static byte[] ResolvedPayload(string containerPath, string objectId, string tensor)
    {
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        return store.Resolve(objectId, tensor).Payload;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-3f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tolerance * (1 + MathF.Abs(expected[i])),
                $"element {i}: expected {expected[i]}, got {actual[i]}");
        }
    }

    /// <summary>The M·row the merger computes for a blended/new token.</summary>
    private static float[] Mapped(float[] map, float[] importedRows, int importedId, int width, float[]? blendBaseRow = null)
    {
        var result = new float[width];
        for (int r = 0; r < width; r++)
        {
            double sum = 0;
            for (int c = 0; c < width; c++)
            {
                sum += map[r * width + c] * importedRows[importedId * width + c];
            }
            result[r] = (float)sum;
            if (blendBaseRow is not null)
            {
                result[r] = 0.5f * (blendBaseRow[r] + result[r]);
            }
        }
        return result;
    }

    // ── the flagship merge: union vocab, blend, shape evolution, head ─────

    [Fact]
    public void Import_Blends_Appends_And_Evolves_To_The_Larger_Shape()
    {
        using var dir = new TempDir();
        // Base: hidden 4, 2 layers, vocab 9 — the smaller model.
        var baseTokens = new[] { "Ġ", "a", "b", "c", "d", "e", "f", "g", "h" };
        var importedTokens = new[] { "a", "b", "c", "x", "y", "z", "u", "v", "w", "q", "r", "s" };
        var basePath = BuildContainer(dir, "synth-a",
            new Dims(Vocab: 9, Hidden: 4, Layers: 2), baseTokens);
        var importedPath = BuildContainer(dir, "synth-b",
            new Dims(Vocab: 12, Hidden: 8, NumQHeads: 4, NumKvHeads: 2, HeadDim: 2, Layers: 3, Intermediate: 12),
            importedTokens);
        var outDir = Path.Combine(dir.Path, "merged");

        var report = ModelMerger.Import(basePath, importedPath, outDir, importedIsContainer: true);

        // Shared with base: a, b, c (3). Base-only: Ġ, d..h (6).
        // Imported-only: x, y, z, u, v, w, q, r, s (9). Merged 9 + 9 = 18.
        Assert.Equal("synth-a+synth-b", report.ResultModel);
        Assert.Equal("synth-b", report.Scaffold);
        Assert.Equal(9, report.BaseVocab);
        Assert.Equal(12, report.ImportedVocab);
        Assert.Equal(18, report.MergedVocab);
        Assert.Equal(3, report.Blended);
        Assert.Equal(6, report.BaseOnly);
        Assert.Equal(9, report.AlignedNew);
        Assert.Equal(8, report.HiddenSize);
        Assert.Equal(3, report.Layers);
        Assert.False(report.HeadTied);
        Assert.Equal(3, report.PreservedSegments);

        // ── index / graph ─────────────────────────────────────────────
        using (var merged = Vindex3Container.Open(outDir))
        {
            Assert.Equal("synth-a+synth-b", merged.Index.Model);
            Assert.Equal(8, merged.Index.HiddenSize);
            Assert.Equal(3, merged.Index.NumLayers);
            Assert.Equal(ContainerAuthority.Derived, merged.Index.Authority);
            Assert.Equal("synth-a", merged.Index.DerivedFromModel);
            Assert.Equal(ModelMerger.ManifestName, merged.Index.TokenMap);

            var target = merged.Graph!.Components.Single(c => c.Role == ComponentRole.PrimaryText);
            Assert.Equal(8, target.HiddenSize);
            Assert.Equal(3, target.NumLayers);
            Assert.Equal(3, target.Attention!.Count);
            Assert.Equal(18, target.Execution!.Head!.VocabSize);
            Assert.False(target.Execution.Head.HeadReusesEmbedding);

            Assert.Contains(merged.Index.Representations.Keys, k => k == "target.embedding@F32");
            Assert.Contains(merged.Index.Representations.Keys, k => k == "target.output_head@F32");
            Assert.Contains(merged.Index.Representations.Keys, k => k == "target.decoder_stack@F32");
        }

        // ── the tokenization mapping layer ─────────────────────────────
        using (var manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, ModelMerger.ManifestName))))
        {
            var vocab = manifest.RootElement.GetProperty("vocab").EnumerateArray().ToArray();
            Assert.Equal(18, vocab.Length);
            Assert.Equal("base_only", vocab[0].GetProperty("kind").GetString());
            Assert.Equal(0, vocab[0].GetProperty("base_id").GetInt32());
            Assert.Equal("blended", vocab[1].GetProperty("kind").GetString());
            Assert.Equal(1, vocab[1].GetProperty("base_id").GetInt32());
            Assert.Equal(0, vocab[1].GetProperty("imported_id").GetInt32());
            Assert.Equal("aligned_new", vocab[9].GetProperty("kind").GetString());
            Assert.Equal(3, vocab[9].GetProperty("imported_id").GetInt32());
            Assert.Equal("synth-b", manifest.RootElement.GetProperty("scaffold").GetProperty("model").GetString());
            Assert.Equal("synth-a", manifest.RootElement.GetProperty("preserved").GetProperty("model").GetString());
        }

        // ── merged embedding rows ──────────────────────────────────────
        int width = 8;
        var (baseRows, _) = Table(basePath, "target.embedding", width);
        var (importedRows, _) = Table(importedPath, "target.embedding", width);
        var importedByToken = importedTokens
            .Select((t, i) => (t, i))
            .ToDictionary(x => x.t, x => x.i);
        // Anchors: shared tokens a (base 1 / imported 0), b (2/1), c (3/2).
        var anchorBase = new[] { 1, 2, 3 };
        var anchorImported = anchorBase.Select(baseId => importedByToken[baseTokens[baseId]]).ToArray();
        var a = new float[anchorBase.Length * width];
        var b = new float[anchorBase.Length * width];
        for (int k = 0; k < anchorBase.Length; k++)
        {
            Array.Copy(baseRows, anchorBase[k] * width, a, k * width, width);
            Array.Copy(importedRows, anchorImported[k] * width, b, k * width, width);
        }
        var map = LeastSquares.Fit(a, b, anchorBase.Length, width);

        var (mergedRows, mergedDtype) = Table(outDir, "target.embedding", width);
        Assert.Equal(Dtype.F32, mergedDtype);
        // base-only Ġ keeps the base row, zero-extended.
        var padG = new float[width];
        Array.Copy(baseRows, 0, padG, 0, width);
        AssertClose(padG, Row(mergedRows, 0, width));
        Assert.Equal(0f, padG[4]);
        Assert.Equal(0f, padG[7]);
        // blended "a" = 0.5 · (base + M·imported).
        var expectA = Mapped(map, importedRows, anchorImported[0], width, blendBaseRow: Row(baseRows, 1, width));
        AssertClose(expectA, Row(mergedRows, 1, width));
        // aligned-new "x" = M·imported row (imported id 3).
        var expectX = Mapped(map, importedRows, importedByToken["x"], width);
        AssertClose(expectX, Row(mergedRows, 9, width));

        // ── merged head is fitted separately (untied) ──────────────────
        var (baseHeadRows, _) = Table(basePath, "target.output_head", width);
        var (importedHeadRows, _) = Table(importedPath, "target.output_head", width);
        var aHead = new float[anchorBase.Length * width];
        var bHead = new float[anchorBase.Length * width];
        for (int k = 0; k < anchorBase.Length; k++)
        {
            Array.Copy(baseHeadRows, anchorBase[k] * width, aHead, k * width, width);
            Array.Copy(importedHeadRows, anchorImported[k] * width, bHead, k * width, width);
        }
        var headMap = LeastSquares.Fit(aHead, bHead, anchorBase.Length, width);
        var (mergedHeadRows, _) = Table(outDir, "target.output_head", width);
        var expectHeadA = Mapped(headMap, importedHeadRows, anchorImported[0], width,
            blendBaseRow: Row(baseHeadRows, 1, width));
        AssertClose(expectHeadA, Row(mergedHeadRows, 1, width));

        // ── the stack is the scaffold's, byte-identical ────────────────
        foreach (var tensor in new[]
                 {
                     "0.self_attn.q_proj.weight",
                     "2.self_attn.o_proj.weight",
                     "1.mlp.down_proj.weight",
                     "2.input_layernorm.weight",
                 })
        {
            Assert.Equal(
                ResolvedPayload(importedPath, "target.decoder_stack", tensor),
                ResolvedPayload(outDir, "target.decoder_stack", tensor));
        }

        // ── preserved source stack ─────────────────────────────────────
        Assert.True(File.Exists(Path.Combine(outDir, "segments", "source", "synth-a", "target.embedding.bin")));
        Assert.True(File.Exists(Path.Combine(outDir, "segments", "source", "synth-a", "target.decoder_stack.bin")));
        Assert.True(File.Exists(Path.Combine(outDir, "segments", "source", "synth-a", "target.final_norm.bin")));

        // ── the merged tokenizer is the union ──────────────────────────
        var tokenizer = TokenVocabulary.ReadFromFile(Path.Combine(outDir, "tokenizer.json"));
        Assert.Equal(18, tokenizer.Count);
        Assert.Equal(0, tokenizer.ByToken["Ġ"]);
        Assert.Equal(9, tokenizer.ByToken["x"]);
    }

    private static float[] Row(float[] table, int row, int width)
    {
        var result = new float[width];
        Array.Copy(table, row * width, result, 0, width);
        return result;
    }

    // ── tied head + same-shape merge through the real checkpoint path ─────

    [Fact]
    public void Import_Tied_Head_Through_Checkpoint_Path_Keeps_Head_Tied()
    {
        using var dir = new TempDir();
        var modelA = Path.Combine(dir.Path, "synth-a");
        SyntheticCheckpoint.Write(modelA);
        var containerA = Path.Combine(dir.Path, "container-a");
        ModelToContainer.Encode(modelA, containerA, "synth-a");

        // A second synthetic checkpoint with a partially overlapping
        // vocabulary, imported as a plain checkpoint directory.
        var modelB = Path.Combine(dir.Path, "synth-b");
        SyntheticCheckpoint.Write(modelB);
        WriteTokenizer(modelB, new[] { "x", "y", "z", "a", "b", "c", "d", "e", "f", "w", "v", "u" });
        var outDir = Path.Combine(dir.Path, "merged");

        var report = ModelMerger.Import(containerA, modelB, outDir, importedIsContainer: false);

        Assert.True(report.HeadTied);
        Assert.Equal("synth-a+synth-b", report.ResultModel);
        Assert.Equal("synth-a", report.Scaffold); // equal sizes → the base scaffolds
        // Shared a..f (6); base-only Ġ,g,h,i,j,? (6); imported-only x,y,z,w,v,u (6).
        Assert.Equal(18, report.MergedVocab);
        Assert.Equal(6, report.Blended);

        using (var merged = Vindex3Container.Open(outDir))
        {
            Assert.False(merged.Index.Representations.ContainsKey("target.output_head@F32"));
            var head = merged.Graph!.Objects.Single(o => o.Kind == ObjectKind.OutputHead);
            Assert.Empty(head.Representations);
            Assert.True(merged.Graph.Components.Single(c => c.Role == ComponentRole.PrimaryText)
                .Execution!.Head!.HeadReusesEmbedding);
        }
    }

    // ── export round trip of a merged container ───────────────────────────

    [Fact]
    public void Export_Of_Merged_Container_Round_Trips()
    {
        using var dir = new TempDir();
        var modelA = Path.Combine(dir.Path, "synth-a");
        SyntheticCheckpoint.Write(modelA);
        var containerA = Path.Combine(dir.Path, "container-a");
        ModelToContainer.Encode(modelA, containerA, "synth-a");
        var modelB = Path.Combine(dir.Path, "synth-b");
        SyntheticCheckpoint.Write(modelB);
        WriteTokenizer(modelB, new[] { "x", "y", "z", "a", "b", "c", "d", "e", "f", "w", "v", "u" });
        var merged = Path.Combine(dir.Path, "merged");
        ModelMerger.Import(containerA, modelB, merged, importedIsContainer: false);

        var exported = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(merged))
        {
            ModelExporter.Export(container, exported, patch: null);
        }

        // The exported checkpoint is a plain model of the merged shape.
        var facts = ModelConfig.ReadTextFacts(Path.Combine(exported, "config.json"));
        Assert.Equal(18, facts.VocabSize);
        Assert.Equal(2, facts.NumLayers);
        Assert.True(facts.TieWordEmbeddings);

        // Re-encode → the merged rows come back verbatim.
        var reEncoded = Path.Combine(dir.Path, "re-encoded");
        ModelToContainer.Encode(exported, reEncoded, "re-merged");
        var (before, _) = Table(merged, "target.embedding", 4);
        var (after, _) = Table(reEncoded, "target.embedding", 4);
        AssertClose(before, after, tolerance: 0f);

        // The union tokenizer travelled with the export.
        Assert.Equal(18, TokenVocabulary.ReadFromFile(Path.Combine(exported, "tokenizer.json")).Count);
    }

    // ── the base scaffolds when it is the larger model ────────────────────

    [Fact]
    public void Import_Keeps_Base_As_Scaffold_When_Larger_And_Preserves_Imported()
    {
        using var dir = new TempDir();
        var basePath = BuildContainer(dir, "synth-big",
            new Dims(Vocab: 12, Hidden: 8, NumQHeads: 4, NumKvHeads: 2, HeadDim: 2, Layers: 3, Intermediate: 12),
            new[] { "a", "b", "c", "x", "y", "z", "u", "v", "w", "q", "r", "s" });
        var importedPath = BuildContainer(dir, "synth-small",
            new Dims(Vocab: 9, Hidden: 4, Layers: 2),
            new[] { "Ġ", "a", "b", "c", "d", "e", "f", "g", "h" });
        var outDir = Path.Combine(dir.Path, "merged");

        var report = ModelMerger.Import(basePath, importedPath, outDir, importedIsContainer: true);

        Assert.Equal("synth-big", report.Scaffold);
        Assert.Equal(3, report.PreservedSegments); // the imported small stack is preserved
        Assert.Equal(
            ResolvedPayload(basePath, "target.decoder_stack", "2.self_attn.o_proj.weight"),
            ResolvedPayload(outDir, "target.decoder_stack", "2.self_attn.o_proj.weight"));
        Assert.True(File.Exists(Path.Combine(outDir, "segments", "source", "synth-small", "target.decoder_stack.bin")));
    }

    // ── a tensor kind the scaffold lacks is grown in, with provenance ─────

    [Fact]
    public void Import_Grows_Tensor_Kinds_The_Scaffold_Lacks_With_Provenance()
    {
        using var dir = new TempDir();
        // Scaffold (larger) has NO q_norm; the imported model does.
        var basePath = BuildContainer(dir, "synth-a",
            new Dims(Vocab: 9, Hidden: 8, NumQHeads: 4, NumKvHeads: 2, HeadDim: 2, Layers: 2, Intermediate: 12),
            new[] { "Ġ", "a", "b", "c", "d", "e", "f", "g", "h" });
        var importedPath = BuildContainer(dir, "synth-b",
            new Dims(Vocab: 9, Hidden: 4, Layers: 2, WeightedQkNorm: true),
            new[] { "a", "b", "c", "x", "y", "z", "u", "v", "w" });
        var outDir = Path.Combine(dir.Path, "merged");

        ModelMerger.Import(basePath, importedPath, outDir, importedIsContainer: true);

        // q_norm appears in the merged stack with the imported model's
        // bytes, and the manifest marks it grown.
        Assert.Equal(
            ResolvedPayload(importedPath, "target.decoder_stack", "1.self_attn.q_norm.weight"),
            ResolvedPayload(outDir, "target.decoder_stack", "1.self_attn.q_norm.weight"));
        using var manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, ModelMerger.ManifestName)));
        var layer1 = manifest.RootElement.GetProperty("layer_table").EnumerateArray()
            .Single(l => l.GetProperty("layer").GetInt32() == 1);
        Assert.Equal("grown", layer1.GetProperty("tensors")
            .GetProperty("1.self_attn.q_norm.weight").GetProperty("operation").GetString());
        Assert.Equal("copied", layer1.GetProperty("tensors")
            .GetProperty("1.mlp.gate_proj.weight").GetProperty("operation").GetString());
    }

    // ── fail closed: no shared tokens, no alignment ───────────────────────

    [Fact]
    public void Import_Refuses_When_The_Vocabularies_Share_No_Tokens()
    {
        using var dir = new TempDir();
        var basePath = BuildContainer(dir, "synth-a",
            new Dims(Vocab: 6, Hidden: 4, Layers: 2),
            new[] { "a", "b", "c", "d", "e", "f" });
        var importedPath = BuildContainer(dir, "synth-b",
            new Dims(Vocab: 6, Hidden: 4, Layers: 2),
            new[] { "u", "v", "w", "x", "y", "z" });

        var ex = Assert.ThrowsAny<Exception>(() =>
            ModelMerger.Import(basePath, importedPath, Path.Combine(dir.Path, "out"), importedIsContainer: true));
        Assert.Contains("no anchors", ex.Message);
        Assert.False(Directory.Exists(Path.Combine(dir.Path, "out")));
    }
}