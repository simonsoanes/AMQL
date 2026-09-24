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

    private static string BuildContainer(TempDir dir, string name, Dims dims, IReadOnlyList<string> tokens,
        Func<float, float>? tableTransform = null)
    {
        var spec = SyntheticModel.BuildSpec(dims);
        if (tableTransform is not null)
        {
            // Mutate the token-interface tables (embedding + untied head)
            // only — the stack stays intact.
            var mutated = spec.Representations.Select(rep => new RepresentationSpec
            {
                ObjectId = rep.ObjectId,
                Encoding = rep.Encoding,
                Tensors = rep.Tensors.Select(t =>
                {
                    if (t.Name != "weight" ||
                        (rep.ObjectId != "target.embedding" && rep.ObjectId != "target.output_head"))
                    {
                        return t;
                    }
                    var values = SyntheticModel.FromBytes(t.Data);
                    for (int i = 0; i < values.Length; i++)
                    {
                        values[i] = tableTransform(values[i]);
                    }
                    return new NamedTensorData
                    {
                        Name = t.Name,
                        Dtype = t.Dtype,
                        Shape = t.Shape,
                        Data = SyntheticModel.ToBytes(values),
                    };
                }).ToList(),
            }).ToList();
            spec = new ContainerSpec
            {
                Model = spec.Model,
                Family = spec.Family,
                HiddenSize = spec.HiddenSize,
                NumLayers = spec.NumLayers,
                SystemGraph = spec.SystemGraph,
                Representations = mutated,
                PrecisionMap = spec.PrecisionMap,
            };
        }
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

    /// <summary>Replicates the merger's agreement gate: if cos(scaffoldRow,
    /// W·otherRow) clears the threshold, blend and restore the scaffold's
    /// energy; otherwise defer to the scaffold row verbatim.</summary>
    private static float[] AgreeOrDefer(float[] scaffoldRows, int scaffoldId, float[] mappedRow, int width)
    {
        double c = Cosine(mappedRow, 0, scaffoldRows, scaffoldId * width, width);
        if (!double.IsFinite(c) || c < ModelMerger.AgreementThreshold)
        {
            return Row(scaffoldRows, scaffoldId, width);
        }
        var blend = new float[width];
        for (int i = 0; i < width; i++)
        {
            blend[i] = 0.5f * (scaffoldRows[scaffoldId * width + i] + mappedRow[i]);
        }
        float scale = Norm(scaffoldRows, scaffoldId * width, width) / Norm(blend, 0, width);
        for (int i = 0; i < width; i++)
        {
            blend[i] *= scale;
        }
        return blend;
    }

    private static double Cosine(float[] a, int aOff, float[] b, int bOff, int width)
    {
        double dot = 0, na = 0, nb = 0;
        for (int c = 0; c < width; c++)
        {
            float av = a[aOff + c];
            float bv = b[bOff + c];
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if (na == 0 || nb == 0)
        {
            return double.NaN;
        }
        return dot / Math.Sqrt(na * nb);
    }

    private static float Norm(float[] row, int off, int width)
    {
        double sum = 0;
        for (int c = 0; c < width; c++)
        {
            sum += (double)row[off + c] * row[off + c];
        }
        return (float)Math.Sqrt(sum);
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
            var anchors = manifest.RootElement.GetProperty("anchors");
            Assert.Equal("synth-b", anchors.GetProperty("space").GetString());
            Assert.True(Math.Abs(anchors.GetProperty("agreement_threshold").GetDouble() - ModelMerger.AgreementThreshold) < 1e-9);
        }
        Assert.Contains(report.Notes, n => n.Contains("consensus"));

        // ── merged embedding rows (scaffold = imported, hidden 8) ─────
        int width = 8;
        var (scaffoldRows, _) = Table(importedPath, "target.embedding", width); // B, the scaffold
        var (otherRows, _) = Table(basePath, "target.embedding", width);        // A, the other (padded)
        var importedByToken = importedTokens
            .Select((t, i) => (t, i))
            .ToDictionary(x => x.t, x => x.i);
        // Anchors: shared tokens a (base 1 / imported 0), b (2/1), c (3/2).
        var anchorBase = new[] { 1, 2, 3 };
        var anchorScaffold = anchorBase.Select(baseId => importedByToken[baseTokens[baseId]]).ToArray();
        // Fit A_pad → B: map minimising ‖B − M·A_pad‖ over the anchors.
        var fitA = new float[anchorBase.Length * width]; // scaffold (B) anchor rows
        var fitB = new float[anchorBase.Length * width]; // other (A_pad) anchor rows
        for (int k = 0; k < anchorBase.Length; k++)
        {
            Array.Copy(scaffoldRows, anchorScaffold[k] * width, fitA, k * width, width);
            Array.Copy(otherRows, anchorBase[k] * width, fitB, k * width, width);
        }
        var map = LeastSquares.Fit(fitA, fitB, anchorBase.Length, width);

        var (mergedRows, mergedDtype) = Table(outDir, "target.embedding", width);
        Assert.Equal(Dtype.F32, mergedDtype);
        // base-only Ġ (only in A, the other) arrives through the map: W·A_pad.
        AssertClose(Mapped(map, otherRows, 0, width), Row(mergedRows, 0, width));
        // shared "a": agreement-gated — either the scaffold row verbatim or
        // the blended row restored to the scaffold's energy.
        var wbA = Mapped(map, otherRows, 1, width);
        var expectedA = AgreeOrDefer(scaffoldRows, anchorScaffold[0], wbA, width);
        AssertClose(expectedA, Row(mergedRows, 1, width));
        // aligned-new "x" (only in B, the scaffold): the scaffold row verbatim.
        AssertClose(Row(scaffoldRows, importedByToken["x"], width), Row(mergedRows, 9, width), tolerance: 0f);

        // ── merged head is fitted separately (untied) ──────────────────
        var (scaffoldHeadRows, _) = Table(importedPath, "target.output_head", width);
        var (otherHeadRows, _) = Table(basePath, "target.output_head", width);
        var fitAHead = new float[anchorBase.Length * width];
        var fitBHead = new float[anchorBase.Length * width];
        for (int k = 0; k < anchorBase.Length; k++)
        {
            Array.Copy(scaffoldHeadRows, anchorScaffold[k] * width, fitAHead, k * width, width);
            Array.Copy(otherHeadRows, anchorBase[k] * width, fitBHead, k * width, width);
        }
        var headMap = LeastSquares.Fit(fitAHead, fitBHead, anchorBase.Length, width);
        var (mergedHeadRows, _) = Table(outDir, "target.output_head", width);
        var wbHeadA = Mapped(headMap, otherHeadRows, 1, width);
        var expectedHeadA = AgreeOrDefer(scaffoldHeadRows, anchorScaffold[0], wbHeadA, width);
        AssertClose(expectedHeadA, Row(mergedHeadRows, 1, width));

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

    // ── consensus reinforcement: agree → blend at full energy ─────────────

    [Fact]
    public void Import_Agreement_Keeps_Identical_Rows_At_Full_Energy()
    {
        using var dir = new TempDir();
        var tokens = new[] { "a", "b", "c", "d", "e", "f", "g", "h", "i" };
        var dims = new Dims(Vocab: 9, Hidden: 4, Layers: 2);
        var basePath = BuildContainer(dir, "synth-a", dims, tokens);
        var importedPath = BuildContainer(dir, "synth-b", dims, tokens); // identical token tables
        var outDir = Path.Combine(dir.Path, "merged");

        var report = ModelMerger.Import(basePath, importedPath, outDir, importedIsContainer: true);

        // Equal shapes → the base scaffolds; every shared token agrees
        // (cosine ≈ 1), so the merged rows equal the scaffold's at full
        // energy — the blend must not shrink them.
        Assert.Equal("synth-a", report.Scaffold);
        Assert.Contains(report.Notes, n => n.Contains("consensus") && n.Contains("100.0%"));
        var (expected, _) = Table(basePath, "target.embedding", 4);
        var (actual, _) = Table(outDir, "target.embedding", 4);
        AssertClose(expected, actual, tolerance: 1e-4f);
        var (expectedHead, _) = Table(basePath, "target.output_head", 4);
        var (actualHead, _) = Table(outDir, "target.output_head", 4);
        AssertClose(expectedHead, actualHead, tolerance: 1e-4f);
    }

    [Fact]
    public void Import_Agreement_Disagreement_Defers_To_The_Scaffold()
    {
        using var dir = new TempDir();
        var tokens = new[] { "a", "b", "c", "d", "e", "f", "g", "h", "i" };
        var dims = new Dims(Vocab: 9, Hidden: 4, Layers: 2);
        var basePath = BuildContainer(dir, "synth-a", dims, tokens);
        // The imported model's token tables are zeroed: its aligned rows
        // cannot agree with the scaffold (cosine undefined), so every row
        // must defer to the scaffold verbatim instead of averaging toward
        // zero — the failures the 50/50 blend produces.
        var importedPath = BuildContainer(dir, "synth-b", dims, tokens, tableTransform: _ => 0f);
        var outDir = Path.Combine(dir.Path, "merged");

        ModelMerger.Import(basePath, importedPath, outDir, importedIsContainer: true);

        var (expected, _) = Table(basePath, "target.embedding", 4);
        var (actual, _) = Table(outDir, "target.embedding", 4);
        AssertClose(expected, actual, tolerance: 1e-5f);
        var (expectedHead, _) = Table(basePath, "target.output_head", 4);
        var (actualHead, _) = Table(outDir, "target.output_head", 4);
        AssertClose(expectedHead, actualHead, tolerance: 1e-5f);
    }

    // ── special tokens and vocabulary padding ────────────────────────────

    /// <summary>Writes a tokenizer.json whose special tokens live in the
    /// top-level <c>added_tokens</c> array above <c>model.vocab</c> — the HF
    /// layout, and the one the merge used to drop.</summary>
    private static void WriteTokenizerWithSpecials(string containerDir, string[] baseTokens, string[] specials)
    {
        var added = specials.Select((content, i) => new
        {
            id = baseTokens.Length + i,
            content,
            special = true,
            single_word = false,
            lstrip = false,
            rstrip = false,
            normalized = false,
        }).ToArray();

        var root = new JsonObject
        {
            ["model"] = new JsonObject
            {
                ["type"] = "bpe",
                ["vocab"] = new JsonObject(
                    baseTokens.Select((t, i) => new KeyValuePair<string, JsonNode?>(t, JsonValue.Create(i))).ToArray()),
                ["merges"] = new JsonArray(),
            },
            ["added_tokens"] = JsonSerializer.SerializeToNode(added)!.AsArray(),
        };
        File.WriteAllText(Path.Combine(containerDir, "tokenizer.json"),
            root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    [Fact]
    public void TokenVocabulary_Includes_Added_Tokens_Above_The_Base_Vocab()
    {
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "tokenizer.json");
        File.WriteAllText(path, """
        {
          "model": {
            "type": "bpe",
            "vocab": { "a": 0, "b": 1, "c": 2 },
            "merges": []
          },
          "added_tokens": [
            { "id": 3, "content": "<|fim_prefix|>", "special": true },
            { "id": 4, "content": "<|fim_middle|>", "special": true },
            { "id": 5, "content": "<|fim_suffix|>", "special": true }
          ]
        }
        """);

        var vocab = TokenVocabulary.ReadFromFile(path);

        // The special tokens are part of the vocabulary, not a tail to drop.
        Assert.Equal(6, vocab.Count);
        Assert.Equal(3, vocab.ByToken["<|fim_prefix|>"]);
        Assert.Equal(5, vocab.ByToken["<|fim_suffix|>"]);
        Assert.Equal("<|fim_middle|>", vocab.Ordered[4]);
    }

    [Fact]
    public void TokenVocabulary_Rejects_Conflicting_Ids_For_The_Same_Token()
    {
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "tokenizer.json");
        File.WriteAllText(path, """
        {
          "model": { "type": "bpe", "vocab": { "a": 0, "dup": 1 }, "merges": [] },
          "added_tokens": [ { "id": 7, "content": "dup", "special": true } ]
        }
        """);

        var ex = Assert.Throws<MergeException>(() => TokenVocabulary.ReadFromFile(path));
        Assert.Contains("dup", ex.Message);
    }

    [Fact]
    public void Import_Preserves_Special_Token_And_Padding_Vocab_Rows()
    {
        using var dir = new TempDir();
        // Embedding has 12 rows; the tokenizer defines 8 base tokens plus 3
        // specials at ids 8-10, leaving row 11 as published vocab padding.
        // The merge must not size the output from model.vocab alone.
        var baseTokens = new[] { "Ġ", "a", "b", "c", "d", "e", "f", "g" };
        var specials = new[] { "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>" };
        var dims = new Dims(Vocab: 12, Hidden: 4, Layers: 2);

        var basePath = BuildContainer(dir, "synth-a", dims, baseTokens);
        WriteTokenizerWithSpecials(basePath, baseTokens, specials);
        var importedPath = BuildContainer(dir, "synth-b", dims, baseTokens);
        WriteTokenizerWithSpecials(importedPath, baseTokens, specials);
        var outDir = Path.Combine(dir.Path, "merged");

        var report = ModelMerger.Import(basePath, importedPath, outDir, importedIsContainer: true);

        // 11 real tokens (8 base + 3 special), and the table keeps all 12 rows.
        Assert.Equal(11, report.MergedVocab);

        var (mergedEmbedding, _) = Table(outDir, "target.embedding", 4);
        Assert.Equal(12 * 4, mergedEmbedding.Length);

        // The trailing padding row is carried across from the scaffold rather
        // than dropped or zeroed.
        var (scaffoldEmbedding, _) = Table(basePath, "target.embedding", 4);
        for (int c = 0; c < 4; c++)
        {
            Assert.Equal(scaffoldEmbedding[11 * 4 + c], mergedEmbedding[11 * 4 + c]);
        }

        // The merged tokenizer still defines the specials at their original
        // ids (8 base tokens occupy 0-7), and the graph's vocab_size agrees
        // with the table that was actually written.
        var mergedTokenizer = TokenVocabulary.ReadFromFile(Path.Combine(outDir, "tokenizer.json"));
        Assert.Equal(8, mergedTokenizer.ByToken["<|fim_prefix|>"]);
        Assert.Equal(10, mergedTokenizer.ByToken["<|fim_suffix|>"]);

        using var container = Vindex3Container.Open(outDir);
        var component = container.Graph!.Components.First(c => c.Role == ComponentRole.PrimaryText);
        Assert.Equal(12, component.Execution!.Head!.VocabSize);
    }
}