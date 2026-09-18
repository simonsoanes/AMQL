using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Tests for the checkpoint export (<c>amql-cli export</c>): the inverse
/// of encode. An unpatched export must regenerate byte-identical tensors
/// (re-encode round trip), a patched export must bake the f32 deltas into
/// the stored dtype of each touched tensor, config.json must read back
/// through <see cref="ModelConfig"/> and re-encode, and an unjudged layer
/// operator must refuse the export by name.
/// </summary>
public class ExportTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 4)]
    [InlineData(10, 10)]  // under the budget: full count
    public void WorkerCount_Caps_At_The_Compute_Budget(int cores, int expected)
    {
        Assert.Equal(expected, ModelExporter.WorkerCount(cores));
    }

    [Fact]
    public void WorkerCount_Never_Exceeds_Budget_Or_Requested_Cores()
    {
        // The old "leave two spare over ten" heuristic is superseded by
        // the process compute budget: the budget reserves a core margin
        // for the machine, and a single call cannot exceed either the
        // budget or the requested count.
        Assert.Equal(ComputeBudget.Cores, ModelExporter.WorkerCount(ComputeBudget.Cores + 64));
        Assert.Equal(ComputeBudget.Cores, ModelExporter.WorkerCount(int.MaxValue));
        Assert.True(ModelExporter.WorkerCount(1) >= 1);
    }
    private static string WriteSynthContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-export");
        return containerPath;
    }

    // ── untied head: Qwen3.8 ships lm_head.weight at the top level ────────

    [Fact]
    public void Encode_Untied_Head_Materialises_The_LmHead()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);

        // Untie: flip tie_word_embeddings in the wrapper AND the text
        // config, then drop a top-level lm_head shard (the Qwen3.8-27B
        // convention: "lm_head.weight" without any text prefix).
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(modelDir, "config.json")))!.AsObject();
        config["tie_word_embeddings"] = false;
        config["text_config"]!["tie_word_embeddings"] = false;
        File.WriteAllText(Path.Combine(modelDir, "config.json"), config.ToJsonString(ViJson.Options));

        var lmHead = new float[SyntheticCheckpoint.Vocab * SyntheticCheckpoint.Hidden];
        for (int i = 0; i < lmHead.Length; i++)
        {
            lmHead[i] = 0.01f * (i % 37);
        }
        SafetensorsWriter.Write(Path.Combine(modelDir, "lm_head.safetensors"), new[]
        {
            new TensorPayload
            {
                Name = "lm_head.weight",
                Dtype = Dtype.F32,
                Shape = new long[] { SyntheticCheckpoint.Vocab, SyntheticCheckpoint.Hidden },
                Data = SyntheticModel.ToBytes(lmHead),
            },
        });

        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-untied");

        using (var container = Vindex3Container.Open(containerPath))
        {
            Assert.Contains(container.Index.Representations.Keys, k => k == "target.output_head@F32");
            var head = container.Graph!.Objects.Single(o => o.Kind == ObjectKind.OutputHead);
            Assert.Single(head.Representations);
            Assert.Equal("lm_head", Assert.Single(head.SourceBindings).TensorPrefix);

            using var store = container.CreateOperandStore();
            Assert.Equal(
                SyntheticModel.ToBytes(lmHead),
                store.Resolve("target.output_head", "weight").Payload);
        }

        // And the export round-trips the untied head back to "lm_head.weight".
        var exported = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, exported, patch: null);
        }
        using var file = SafetensorsFile.Open(Path.Combine(exported, "model.safetensors"));
        Assert.Equal(
            SyntheticModel.ToBytes(lmHead),
            file.ReadBytes("lm_head.weight"));
        using var exportedConfig = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(exported, "config.json")));
        // Falsy facts are omitted — the encoder default is false, so the
        // untied head must re-encode as untied.
        Assert.False(exportedConfig.RootElement.TryGetProperty("tie_word_embeddings", out var tie) &&
                     tie.GetBoolean());
    }

    private static byte[] Payload(string containerPath, string objectId, string tensor)
    {
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        return store.Resolve(objectId, tensor).Payload;
    }

    private static float Cell32(string containerPath, string objectId, string tensor, long cell)
    {
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var resolution = store.Resolve(objectId, tensor);
        return BitPattern.WidenToF32(resolution.Dtype, resolution.Payload)[cell];
    }

    /// <summary>Every (object, tensor) pair a container materialises.</summary>
    private static IEnumerable<(string ObjectId, string Tensor)> Materialised(Vindex3Container container)
    {
        using var store = container.CreateOperandStore();
        foreach (var obj in container.Graph!.Objects.Where(o => o.Representations.Count > 0))
        {
            string? path = store.SegmentPathFor(obj.Id);
            if (path is null)
            {
                continue;
            }
            using var segment = SegmentFile.Open(Path.Combine(container.Root, path));
            foreach (var tensor in segment.Header.Tensors)
            {
                yield return (obj.Id, tensor.Name);
            }
        }
    }

    private static float EmbeddingCell(string containerPath, long cell) =>
        Cell32(containerPath, "target.embedding", "weight", cell);

    // ── unpatched export round trips ──────────────────────────────────────

    [Fact]
    public void Export_Without_Patch_RoundTrips_Byte_Identical_Through_Encode()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outDir = Path.Combine(dir.Path, "exported");
        var reEncoded = Path.Combine(dir.Path, "re-encoded");

        ExportReport report;
        using (var container = Vindex3Container.Open(containerPath))
        {
            report = ModelExporter.Export(container, outDir, patch: null);
        }

        Assert.Equal("synth-export", report.Model);
        Assert.Equal("model.safetensors, config.json, tokenizer.json", "model.safetensors, config.json" +
            (File.Exists(Path.Combine(outDir, "tokenizer.json")) ? ", tokenizer.json" : string.Empty));
        var note = Assert.Single(report.Notes, n => n.Contains("target.output_head"));
        Assert.Contains("carried only", note);

        // The exported checkpoint is a plain model directory: re-encode it
        // and every tensor must come back byte-identical.
        ModelToContainer.Encode(outDir, reEncoded, "synth-re-export");
        using var original = Vindex3Container.Open(containerPath);
        using var rebuilt = Vindex3Container.Open(reEncoded);
        foreach (var (objectId, tensor) in Materialised(original))
        {
            Assert.Equal(
                Payload(containerPath, objectId, tensor),
                Payload(reEncoded, objectId, tensor));
        }
    }

    // ── patched export bakes deltas in ────────────────────────────────────

    [Fact]
    public void Export_With_Patch_Bakes_Deltas_Into_The_Shard()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outDir = Path.Combine(dir.Path, "exported");

        long cell = 1 * 4 + 1; // embedding [12, 4], row 1 col 1
        IReadOnlyList<WeightPatchEntry> entries;
        string model;
        using (var container = Vindex3Container.Open(containerPath))
        {
            model = container.Index.Model;
            entries = TensorPatchTools.ApplyEdit(container, "target.embedding", "weight",
                TensorEditOp.Set, -0.5f, cell, Array.Empty<WeightPatchEntry>()).Entries;
        }
        var patch = WeightPatch.FromEntries(entries, model);

        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, outDir, patch);
        }

        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        // The patch is baked in — the shard carries no patch tensor names.
        Assert.False(file.Contains("target.embedding/weight"));

        // The edited cell is the patched value; every other cell is the
        // base value.
        var embedding = file.DecodeF32("model.embed_tokens.weight");
        Assert.Equal(4, file.GetTensor("model.embed_tokens.weight").Shape[1]);
        Assert.Equal(-0.5f, embedding[cell], precision: 6);
        using var container1 = Vindex3Container.Open(containerPath);
        using var store = container1.CreateOperandStore();
        var baseResolution = store.Resolve("target.embedding", "weight");
        var baseTable = BitPattern.WidenToF32(baseResolution.Dtype, baseResolution.Payload);
        for (int i = 0; i < baseTable.Length; i++)
        {
            if (i == cell)
            {
                continue;
            }
            Assert.Equal(baseTable[i], embedding[i]);
        }

        // And the patched checkpoint re-encodes to the patched value.
        var reEncoded = Path.Combine(dir.Path, "re-encoded");
        ModelToContainer.Encode(outDir, reEncoded, "synth-re-export");
        Assert.Equal(-0.5f, EmbeddingCell(reEncoded, cell), precision: 6);
    }

    // ── config regeneration ───────────────────────────────────────────────

    [Fact]
    public void Export_Writes_Config_That_Reads_Back_As_Text_Facts()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outDir = Path.Combine(dir.Path, "exported");

        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, outDir, patch: null);
        }

        var facts = ModelConfig.ReadTextFacts(Path.Combine(outDir, "config.json"));
        Assert.Equal("qwen3_5_text", facts.ModelType);
        Assert.Equal(SyntheticCheckpoint.Hidden, facts.HiddenSize);
        Assert.Equal(SyntheticCheckpoint.Layers, facts.NumLayers);
        Assert.Equal(SyntheticCheckpoint.NumQueryHeads, facts.NumQueryHeads);
        Assert.Equal(SyntheticCheckpoint.NumKvHeads, facts.NumKvHeads);
        Assert.Equal(SyntheticCheckpoint.HeadDim, facts.HeadDim);
        Assert.Equal(SyntheticCheckpoint.Intermediate, facts.IntermediateSize);
        Assert.Equal(SyntheticCheckpoint.Vocab, facts.VocabSize);
        Assert.True(facts.TieWordEmbeddings);
        Assert.False(facts.AttentionBias);
        Assert.Equal(new[] { "full_attention", "full_attention" }, facts.LayerTypes);
        Assert.Equal(2048, facts.MaxPositionEmbeddings);
    }

    // ── fail-closed: unjudged operators refuse the export ─────────────────

    [Fact]
    public void Export_Refuses_Unjudged_Layer_Operator_By_Name()
    {
        var spec = SyntheticModel.BuildSpec(new Dims());
        spec.SystemGraph.Components[0].Attention![0].SetOperator(LayerOperators.Kda);

        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "c");
        ContainerEncoder.Encode(containerPath, spec);

        using var container = Vindex3Container.Open(containerPath);
        var ex = Assert.ThrowsAny<Exception>(() => ModelExporter.Export(container, Path.Combine(dir.Path, "out"), patch: null));
        Assert.Contains("kda", ex.Message);
        Assert.Contains("layer_types", ex.Message);
        Assert.False(Directory.Exists(Path.Combine(dir.Path, "out")),
            "a refused export must not leave a partial checkpoint behind");
    }

    // ── dtype regeneration: BF16 tensors ──────────────────────────────────

    [Fact]
    public void Export_Applies_Patch_To_Bf16_Tensor_And_Copies_Others_Verbatim()
    {
        // A BF16-stored container: same synthetic weights, storage dtype
        // re-encoded to BF16 (mirrors the real stack's canonical dtype).
        var spec = SyntheticModel.BuildSpec(new Dims());
        var bf16 = new ContainerSpec
        {
            Model = spec.Model,
            Family = spec.Family,
            HiddenSize = spec.HiddenSize,
            NumLayers = spec.NumLayers,
            SystemGraph = spec.SystemGraph,
            Representations = spec.Representations.Select(rep => new RepresentationSpec
            {
                ObjectId = rep.ObjectId,
                Encoding = rep.Encoding,
                Tensors = rep.Tensors.Select(t => new NamedTensorData
                {
                    Name = t.Name,
                    Dtype = Dtype.BF16,
                    Shape = t.Shape,
                    Data = ToBf16Bytes(t.Data),
                }).ToList(),
            }).ToList(),
        };

        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "c");
        ContainerEncoder.Encode(containerPath, bf16);

        long cell = 1 * 4 + 1;
        IReadOnlyList<WeightPatchEntry> entries;
        using (var container = Vindex3Container.Open(containerPath))
        {
            entries = TensorPatchTools.ApplyEdit(container, "target.embedding", "weight",
                TensorEditOp.Set, 2.5f, cell, Array.Empty<WeightPatchEntry>()).Entries;
        }

        var outDir = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, outDir, WeightPatch.FromEntries(entries));
        }

        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        // The patched BF16 tensor re-encodes the merged value.
        var patchedInfo = file.GetTensor("target.embedding.weight");
        Assert.Equal("BF16", patchedInfo.Dtype.Label());
        var patched = file.DecodeF32("target.embedding.weight");
        Assert.True(MathF.Abs(patched[cell] - 2.5f) <= 0.01f,
            $"patched BF16 cell must decode to 2.5, got {patched[cell]}");

        // An unpatched tensor is copied verbatim — byte-identical to the
        // container's stored payload.
        var raw = file.ReadBytes("target.decoder_stack.0.self_attn.q_proj.weight");
        Assert.Equal(Payload(containerPath, "target.decoder_stack", "0.self_attn.q_proj.weight"), raw);
    }

    private static byte[] ToBf16Bytes(byte[] f32Bytes)
    {
        var values = SyntheticModel.FromBytes(f32Bytes);
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BitPattern.EncodeBf16(values[i]);
            bytes[2 * i] = (byte)(bits & 0xFF);
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }
}