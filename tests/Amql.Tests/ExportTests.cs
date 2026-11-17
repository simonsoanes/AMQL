using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Gguf;
using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Onnx;
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

    // ── ancillary config survives the round trip ──────────────────────────

    /// <summary>
    /// The chat template and the processor configs are not recoverable from
    /// the tensors, so the container is the only place that can keep them.
    /// Without this a model re-exported from a container can tokenize but
    /// cannot format a conversation, and a multi-modal model silently loses
    /// its image/video processor settings. Repo metadata is not model config
    /// and must not be dragged along with it.
    /// </summary>
    [Fact]
    public void Encode_And_Export_Carry_The_Chat_Template_And_Processor_Configs()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);

        const string template = "{{ '<|im_start|>' + role }}";
        File.WriteAllText(Path.Combine(modelDir, "chat_template.jinja"), template);
        File.WriteAllText(Path.Combine(modelDir, "preprocessor_config.json"), "{\"image_mean\":[0.5]}");
        File.WriteAllText(Path.Combine(modelDir, "generation_config.json"), "{\"eos_token_id\":1}");
        File.WriteAllText(Path.Combine(modelDir, "README.md"), "repo metadata, not model config");

        var containerPath = Path.Combine(dir.Path, "container");
        var report = ModelToContainer.Encode(modelDir, containerPath, "synth-ancillary");

        Assert.Equal(
            // source_config.json is the checkpoint's config.json, kept as the
            // base export writes the judged facts over.
            new[] { "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "source_config.json" },
            report.AncillaryCopied.OrderBy(name => name, StringComparer.Ordinal).ToArray());
        Assert.Equal(template, File.ReadAllText(Path.Combine(containerPath, "chat_template.jinja")));
        Assert.False(File.Exists(Path.Combine(containerPath, "README.md")));

        var outDir = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, outDir, patch: null);
        }

        Assert.Equal(template, File.ReadAllText(Path.Combine(outDir, "chat_template.jinja")));
        Assert.Equal("{\"image_mean\":[0.5]}", File.ReadAllText(Path.Combine(outDir, "preprocessor_config.json")));
        Assert.Equal("{\"eos_token_id\":1}", File.ReadAllText(Path.Combine(outDir, "generation_config.json")));
        Assert.False(File.Exists(Path.Combine(outDir, "README.md")));
    }

    // ── ternary export ─────────────────────────────────────────────────────

    /// <summary>
    /// PTQ1_0 packs 26 bytes per 128-weight block and PQ2_0 packs 32, so the
    /// two exports must differ in size — that is the discriminator which catches
    /// a dispatcher defaulting to one packing whatever the flag says, which is
    /// exactly what it did. The packed payload is stored as an opaque U8 byte
    /// matrix because safetensors has no dtype whose element size matches a
    /// ternary packing (FP4, the only sub-byte tag, declares half a byte per
    /// element and the writer rejected the payload outright), and the logical
    /// shape travels in __metadata__ so a decoder can invert the rotation.
    /// </summary>
    [Fact]
    public void Ternary_Export_Honours_The_Chosen_Packing()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var ptq1Dir = Path.Combine(dir.Path, "ptq1");
        var pq2Dir = Path.Combine(dir.Path, "pq2");
        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, ptq1Dir, patch: null, quantizeTernary: "ptq1");
            ModelExporter.Export(container, pq2Dir, patch: null, quantizeTernary: "pq2");
        }

        using var ptq1 = SafetensorsFile.Open(Path.Combine(ptq1Dir, "model.safetensors"));
        using var pq2 = SafetensorsFile.Open(Path.Combine(pq2Dir, "model.safetensors"));

        Assert.Equal("PTQ1_0", ptq1.Metadata!["amql.ternary.packing"]);
        Assert.Equal("PQ2_0", pq2.Metadata!["amql.ternary.packing"]);
        Assert.Equal(Ternary.BlockElements.ToString(), ptq1.Metadata!["amql.ternary.block_elements"]);

        // any quantized tensor: the one with a ".scales" companion
        string name = ptq1.Tensors.Keys
            .First(k => !k.EndsWith(".scales") && ptq1.Tensors.ContainsKey(k + ".scales"));

        var a = ptq1.Tensors[name];
        var b = pq2.Tensors[name];
        Assert.Equal(Dtype.U8, a.Dtype);
        Assert.Equal(Dtype.U8, b.Dtype);
        Assert.Equal(2, a.Shape.Length);
        Assert.Equal(a.Shape[0], b.Shape[0]);                       // same block count
        Assert.Equal(Ternary.Ptq1BytesPerBlock, a.Shape[1]);        // 26
        Assert.Equal(Ternary.Pq2BytesPerBlock, b.Shape[1]);         // 32

        var packedA = ptq1.ReadBytes(name);
        var packedB = pq2.ReadBytes(name);
        Assert.Equal(a.Shape[0] * Ternary.Ptq1BytesPerBlock, packedA.Length);
        Assert.Equal(b.Shape[0] * Ternary.Pq2BytesPerBlock, packedB.Length);
        Assert.True(packedB.Length > packedA.Length,
            "PQ2_0 is the denser packing; if both came from one encoder the sizes would match");

        // the logical shape a decoder needs is recoverable from the header
        var shapes = ptq1.Metadata!["amql.ternary.shapes"]
            .Split(';', StringSplitOptions.RemoveEmptyEntries)
            .Select(pair => pair.Split('='))
            .ToDictionary(p => p[0], p => p[1]);
        Assert.True(shapes.TryGetValue(name, out var dims), $"no logical shape recorded for '{name}'");
        var rc = dims.Split('x');
        Assert.Equal(a.Shape[0], Ternary.BlockScaleCount(int.Parse(rc[0]) * int.Parse(rc[1])));
    }

    /// <summary>
    /// Both ternary codecs must be self-consistent, and must disagree in size:
    /// PTQ1_0 packs 26 bytes per 128-weight block, PQ2_0 packs 32. That is the
    /// discriminator for the exporter's dispatcher, which used to call
    /// EncodePq2 whatever the flag said. DecodePtq1 and DecodePq2 previously
    /// had no callers anywhere in the repo, so neither direction had ever been
    /// executed by anything.
    /// <para>
    /// Numeric fidelity is deliberately not asserted. <c>ComputeScale</c> uses
    /// the block's mean absolute value, and the Bonsai whitepaper specifies
    /// only that there is one FP16 scale per 128 weights — not how it is
    /// derived. The consequence is measurable: a uniform block of value c comes
    /// back as c/128, because its rotated form is a single spike of √n·c whose
    /// mean-abs is √n·c/n, and the reconstruction is then limited to that
    /// scale. Whether mean-abs is the PrismML reference's intent or a defect is
    /// an open question, and changing it would alter the on-disk numbers, so it
    /// is left alone rather than guessed at.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData("ptq1")]
    [InlineData("pq2")]
    public void Ternary_Codec_Is_Self_Consistent_For_Both_Packings(string packing)
    {
        const int rows = 8, cols = 128;                 // 1024 = one Hadamard block
        var values = new float[rows * cols];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.37f) * (1f + (i % 7) * 0.1f);
        }

        int bytesPerBlock = packing == "ptq1" ? Ternary.Ptq1BytesPerBlock : Ternary.Pq2BytesPerBlock;
        Assert.Equal(128, Ternary.BlockElements);
        Assert.True(bytesPerBlock > 0);

        // EncodePtq1/EncodePq2 rotate the input IN PLACE, so each call needs
        // its own copy — encoding twice from one array would rotate twice.
        var (packed, scales) = packing == "ptq1"
            ? Ternary.EncodePtq1((float[])values.Clone(), rows, cols)
            : Ternary.EncodePq2((float[])values.Clone(), rows, cols);

        int blocks = Ternary.BlockScaleCount(values.Length);
        Assert.Equal(blocks * bytesPerBlock, packed.Length);
        Assert.Equal(blocks * 2, scales.Length);        // one FP16 scale per block

        // deterministic: same input, same bytes
        var (packed2, scales2) = packing == "ptq1"
            ? Ternary.EncodePtq1((float[])values.Clone(), rows, cols)
            : Ternary.EncodePq2((float[])values.Clone(), rows, cols);
        Assert.Equal(packed, packed2);
        Assert.Equal(scales, scales2);

        var decoded = packing == "ptq1"
            ? Ternary.DecodePtq1(packed, scales, values.Length, rows, cols)
            : Ternary.DecodePq2(packed, scales, values.Length, rows, cols);
        Assert.Equal(values.Length, decoded.Length);
        Assert.All(decoded, v => Assert.True(float.IsFinite(v), "decoded a non-finite weight"));
        Assert.NotEqual(0f, decoded.Max(v => MathF.Abs(v)));   // not an all-zero decode

        // and the two packings really do differ on the same input
        var (other, _) = packing == "ptq1"
            ? Ternary.EncodePq2((float[])values.Clone(), rows, cols)
            : Ternary.EncodePtq1((float[])values.Clone(), rows, cols);
        Assert.NotEqual(packed.Length, other.Length);
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

    // ── Qwen4 (qwen4_exp) export ────────────────────────────────────────
    // Source: the arch-zoo Qwen3.5 gated-delta-net fixture (GDN + gated,
    // QK-normed attention, dense MLP, untied head). The export must compute
    // the source function exactly; these pin the tensor relations that make
    // it so (identities checked against the transformers qwen4_exp code).

    private static readonly string Qwen35Gdn =
        Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo", "qwen3_5-gdn");

    private static string ExportQwen4(TempDir dir)
    {
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(Qwen35Gdn, containerPath, "qwen35-gdn");
        var outDir = Path.Combine(dir.Path, "exported");
        using var container = Vindex3Container.Open(containerPath);
        ModelExporter.Export(container, outDir, patch: null, arch: Qwen4NextLayout.Arch);
        return outDir;
    }

    [Fact]
    public void Export_With_Arch_Qwen4Next_Writes_Qwen4Exp_Config()
    {
        using var dir = new TempDir();
        var outDir = ExportQwen4(dir);

        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, "config.json")));
        var root = config.RootElement;
        Assert.Equal("qwen4_exp", root.GetProperty("model_type").GetString());
        Assert.Equal("Qwen4ExpForCausalLM", root.GetProperty("architectures")[0].GetString());
        Assert.Equal("qwen4_exp_vision", root.GetProperty("vision_config").GetProperty("model_type").GetString());

        var text = root.GetProperty("text_config");
        Assert.Equal("qwen4_exp_text", text.GetProperty("model_type").GetString());
        Assert.Equal(64, text.GetProperty("hidden_size").GetInt32());
        Assert.Equal(4, text.GetProperty("num_hidden_layers").GetInt32());
        Assert.Equal(new[] { "linear_attention", "full_attention", "linear_attention", "full_attention" },
            text.GetProperty("layer_types").EnumerateArray().Select(e => e.GetString()).ToArray());
        Assert.True(text.GetProperty("attn_output_gate").GetBoolean());

        // Gated Residual, and Qwen3.5's SiLU z-gate (not Flash-Next's sigmoid).
        Assert.Equal(4, text.GetProperty("hc_count").GetInt32());
        Assert.Equal(320, text.GetProperty("hc_lowrank").GetInt32());
        Assert.Equal("silu", text.GetProperty("output_gate_type").GetString());

        // QSA: one key head, budget covering the context (256) — dense and exact,
        // budget / ratio = 2048 as vLLM requires, rotary dims (16·0.25) fit the index head.
        Assert.Equal(4, text.GetProperty("indexer_n_heads").GetInt32());
        Assert.Equal(1, text.GetProperty("indexer_kv_heads").GetInt32());
        Assert.Equal(128, text.GetProperty("indexer_head_dim").GetInt32());
        int ratio = text.GetProperty("indexer_compress_ratio").GetInt32();
        int budget = text.GetProperty("indexer_budget").GetInt32();
        Assert.Equal(4, ratio);
        Assert.Equal(2048, budget / ratio);
        Assert.True(budget >= text.GetProperty("max_position_embeddings").GetInt32());

        // A dense MLP is one routed expert; the shared expert is zero.
        Assert.Equal(1, text.GetProperty("num_experts").GetInt32());
        Assert.Equal(1, text.GetProperty("num_experts_per_tok").GetInt32());
        Assert.Equal(96, text.GetProperty("moe_intermediate_size").GetInt32());
        Assert.True(text.GetProperty("norm_topk_prob").GetBoolean());
        Assert.Equal(Qwen4NextLayout.ZeroSharedExpertWidth, text.GetProperty("shared_expert_intermediate_size").GetInt32());

        // PLE off.
        Assert.Equal(0, text.GetProperty("ple_layer_ids").GetArrayLength());
        Assert.Equal(0.25, text.GetProperty("rope_parameters").GetProperty("partial_rotary_factor").GetDouble(), 6);
    }

    [Theory]
    [InlineData(2048, 4, 8192)]
    [InlineData(8193, 8, 16384)]
    [InlineData(262144, 128, 262144)]
    public void Qwen4Next_Qsa_Budget_Covers_The_Context(long context, int ratio, int budget)
    {
        Assert.Equal(ratio, Qwen4NextLayout.CompressRatioFor(context));
        Assert.Equal(budget, Qwen4NextLayout.IndexerBlockTopK * ratio);
    }

    [Fact]
    public void Export_With_Arch_Qwen4Next_Folds_Norms_Into_Gated_Residual()
    {
        using var dir = new TempDir();
        var outDir = ExportQwen4(dir);
        using var source = SafetensorsFile.Open(Path.Combine(Qwen35Gdn, "model.safetensors"));
        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));

        // qwen4_exp has no layer or final norms: they live in hc_norm.
        for (int l = 0; l < 4; l++)
        {
            Assert.False(file.Contains($"model.layers.{l}.input_layernorm.weight"));
            Assert.False(file.Contains($"model.layers.{l}.post_attention_layernorm.weight"));
        }
        Assert.False(file.Contains("model.norm.weight"));

        // Read gate is sigmoid(0) = ½ with zero mix projections, so
        // ½·(1 + hc_norm) must equal the source gain (1 + w): hc_norm = 1 + 2w, per stream.
        void AssertFolded(string sourceNorm, string hcPrefix, bool combine)
        {
            float[] w = source.DecodeF32(sourceNorm);
            float[] hc = file.DecodeF32(hcPrefix + ".hc_norm.weight");
            Assert.Equal(4 * w.Length, hc.Length);
            for (int s = 0; s < 4; s++)
            {
                for (int i = 0; i < w.Length; i++)
                {
                    Assert.Equal(1f + w[i], 0.5f * (1f + hc[s * w.Length + i]), 5);
                }
            }
            Assert.All(file.DecodeF32(hcPrefix + ".input_mix_weight_down.weight"), v => Assert.Equal(0f, v));
            Assert.All(file.DecodeF32(hcPrefix + ".input_mix_weight_up.weight"), v => Assert.Equal(0f, v));
            Assert.Equal(new long[] { 320, 256 }, file.GetTensor(hcPrefix + ".input_mix_weight_down.weight").Shape);
            Assert.Equal(new long[] { 256, 320 }, file.GetTensor(hcPrefix + ".input_mix_weight_up.weight").Shape);
            Assert.Equal(combine, file.Contains(hcPrefix + ".block_inject_weight.weight"));
            if (combine)
            {
                // Zero inject logits: write gate 2·sigmoid(0) = 1 on every stream.
                Assert.All(file.DecodeF32(hcPrefix + ".block_inject_weight.weight"), v => Assert.Equal(0f, v));
            }
        }
        for (int l = 0; l < 4; l++)
        {
            AssertFolded($"model.layers.{l}.input_layernorm.weight", $"model.layers.{l}.attn_hyper_connection", combine: true);
            AssertFolded($"model.layers.{l}.post_attention_layernorm.weight", $"model.layers.{l}.mlp_hyper_connection", combine: true);
        }
        AssertFolded("model.norm.weight", "model.hyper_connection_mixer", combine: false);

        // Untouched: Qwen3.5's (1 + w) QK norms are qwen4_exp's convention already.
        Assert.Equal(source.ReadBytes("model.layers.1.self_attn.q_norm.weight"),
                     file.ReadBytes("model.layers.1.self_attn.q_norm.weight"));
        Assert.Equal(source.ReadBytes("model.layers.0.linear_attn.in_proj_qkv.weight"),
                     file.ReadBytes("model.layers.0.linear_attn.in_proj_qkv.weight"));
    }

    [Fact]
    public void Export_With_Arch_Qwen4Next_Adds_Indexers_And_Routes_The_Dense_Mlp_As_One_Expert()
    {
        using var dir = new TempDir();
        var outDir = ExportQwen4(dir);
        using var source = SafetensorsFile.Open(Path.Combine(Qwen35Gdn, "model.safetensors"));
        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));

        // QSA indexers on the full-attention layers only.
        foreach (int l in new[] { 1, 3 })
        {
            string ix = $"model.layers.{l}.self_attn.indexer.";
            Assert.Equal(new long[] { 5 * 128, 64 }, file.GetTensor(ix + "index_qk_proj.weight").Shape);
            Assert.Equal(new long[] { 128 }, file.GetTensor(ix + "q_layernorm.weight").Shape);
            Assert.Equal(new long[] { 128 }, file.GetTensor(ix + "k_layernorm.weight").Shape);
        }
        Assert.False(file.Contains("model.layers.0.self_attn.indexer.index_qk_proj.weight"));

        for (int l = 0; l < 4; l++)
        {
            string p = $"model.layers.{l}.mlp.";
            Assert.False(file.Contains(p + "gate_proj.weight"));
            // gate_up_proj [1, 2I, H] = gate rows then up rows (the experts chunk(2) the output).
            Assert.Equal(new long[] { 1, 192, 64 }, file.GetTensor(p + "experts.gate_up_proj").Shape);
            Assert.Equal(source.ReadBytes(p + "gate_proj.weight").Concat(source.ReadBytes(p + "up_proj.weight")).ToArray(),
                         file.ReadBytes(p + "experts.gate_up_proj"));
            Assert.Equal(new long[] { 1, 64, 96 }, file.GetTensor(p + "experts.down_proj").Shape);
            Assert.Equal(source.ReadBytes(p + "down_proj.weight"), file.ReadBytes(p + "experts.down_proj"));

            // Zero router over one expert: softmax weight 1.
            Assert.Equal(new long[] { 1, 64 }, file.GetTensor(p + "gate.weight").Shape);
            Assert.All(file.DecodeF32(p + "gate.weight"), v => Assert.Equal(0f, v));

            // Zero shared expert, and its gate.
            int w = Qwen4NextLayout.ZeroSharedExpertWidth;
            Assert.Equal(new long[] { 64, w }, file.GetTensor(p + "shared_expert.down_proj.weight").Shape);
            Assert.All(file.DecodeF32(p + "shared_expert.down_proj.weight"), v => Assert.Equal(0f, v));
            Assert.Equal(new long[] { 1, 64 }, file.GetTensor(p + "shared_expert_gate.weight").Shape);
        }
    }

    [Fact]
    public void Export_With_Arch_Qwen4Next_Contains_Safetensors_Metadata()
    {
        using var dir = new TempDir();
        var outDir = ExportQwen4(dir);

        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        Assert.NotNull(file.Metadata);
        Assert.Equal(ModelExporter.ExportFormat, file.Metadata!["format"]);
        Assert.Equal(Qwen4NextLayout.Arch, file.Metadata!["arch"]);
    }

    [Fact]
    public void Export_With_Arch_Qwen4Next_Refuses_A_Source_Without_The_Attention_Output_Gate()
    {
        // The synthetic checkpoint is Llama-shaped: no output gate, so no exact qwen4_exp form.
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outDir = Path.Combine(dir.Path, "exported");

        using var container = Vindex3Container.Open(containerPath);
        var ex = Assert.ThrowsAny<Exception>(() =>
            ModelExporter.Export(container, outDir, patch: null, arch: Qwen4NextLayout.Arch));
        Assert.Contains("output gate", ex.Message);
        Assert.False(Directory.Exists(outDir));
    }

    [Fact]
    public void Export_With_Arch_Qwen4Next_Refuses_Unjudged_Layer_Operator()
    {
        var spec = SyntheticModel.BuildSpec(new Dims());
        spec.SystemGraph.Components[0].Attention![0].SetOperator(LayerOperators.Kda);

        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "c");
        ContainerEncoder.Encode(containerPath, spec);

        using var container = Vindex3Container.Open(containerPath);
        var ex = Assert.ThrowsAny<Exception>(() =>
            ModelExporter.Export(container, Path.Combine(dir.Path, "out"), patch: null, arch: Qwen4NextLayout.Arch));
        Assert.Contains("kda", ex.Message);
        Assert.Contains("layer_types", ex.Message);
        Assert.False(Directory.Exists(Path.Combine(dir.Path, "out")));
    }

    // ── Model type conversion ────────────────────────────────────────────

    [Fact]
    public void ConvertToClassifier_Creates_ScoreHead_And_ClassifierSurface()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outDir = Path.Combine(dir.Path, "classifier");

        var report = ModelConverter.ConvertToClassifier(containerPath, outDir, numLabels: 3);

        Assert.Equal(outDir, report.OutDir);
        Assert.Contains("classifier", report.Model);

        // Container can be opened
        using var container = Vindex3Container.Open(outDir);
        var graph = container.Graph!;
        var component = graph.Components.First(c => c.Role == ComponentRole.PrimaryText);

        // ClassifierSurface
        var surface = component.Execution!;
        Assert.NotNull(surface.Classifier);
        Assert.Equal(3, surface.Classifier!.NumLabels);
        Assert.Equal("single_label_classification", surface.Classifier.ProblemType);
        Assert.Equal(PoolingKind.Last, surface.Classifier.Pooling.Kind);
        Assert.True(surface.Classifier.Pooling.LastNonPad);

        // ClassifierHead object
        var classifierObj = graph.Objects.First(o => o.Kind == ObjectKind.ClassifierHead);
        Assert.Equal("target.classifier_head", classifierObj.Id);
        Assert.Equal("model.score", classifierObj.SourceBindings![0].TensorPrefix);

        // Score tensor exists and is shaped [numLabels, hidden]
        using var store = container.CreateOperandStore();
        var resolution = store.Resolve("target.classifier_head", "weight");
        Assert.Equal(Dtype.F32, resolution.Dtype);
        Assert.Equal(2, resolution.Shape.Length);
        Assert.Equal(3, resolution.Shape[0]); // numLabels
        Assert.Equal(SyntheticCheckpoint.Hidden, resolution.Shape[1]);

        // Original decoder tensors are still present
        store.Resolve("target.decoder_stack", "0.self_attn.q_proj.weight");
    }

    [Fact]
    public void ConvertToClassifier_Refuses_Already_Classifier()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outDir1 = Path.Combine(dir.Path, "classifier");

        ModelConverter.ConvertToClassifier(containerPath, outDir1, numLabels: 3);

        var ex = Assert.ThrowsAny<Exception>(() =>
            ModelConverter.ConvertToClassifier(outDir1, Path.Combine(dir.Path, "classifier2"), numLabels: 3));
        Assert.Contains("ClassifierSurface", ex.Message);
    }

    [Fact]
    public void ConvertToEmbedding_Adds_PoolingSurface_No_New_Tensors()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outDir = Path.Combine(dir.Path, "embedding");

        var report = ModelConverter.ConvertToEmbedding(containerPath, outDir);

        Assert.Equal(outDir, report.OutDir);
        Assert.Contains("embedding", report.Model);

        using var container = Vindex3Container.Open(outDir);
        var graph = container.Graph!;
        var component = graph.Components.First(c => c.Role == ComponentRole.PrimaryText);
        var surface = component.Execution!;

        // Embedding surface facts
        Assert.True(surface.Embedding.HasValue);
        var emb = surface.Embedding!.Value;
        Assert.Equal("mean", emb.GetProperty("pooling").GetProperty("kind").GetString());
        Assert.True(emb.GetProperty("pooling").GetProperty("l2_normalise").GetBoolean());

        // No new objects added (decoder stack stays as-is)
        Assert.DoesNotContain(graph.Objects, o => o.Kind == ObjectKind.EncoderStack);

        // Original tensors present
        using var store = container.CreateOperandStore();
        store.Resolve("target.decoder_stack", "0.self_attn.q_proj.weight");
    }

    [Fact]
    public void ConvertToEmbedding_Exports_And_Roundtrips()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var embeddingDir = Path.Combine(dir.Path, "embedding");

        ModelConverter.ConvertToEmbedding(containerPath, embeddingDir);

        // Export the embedding container
        var exportDir = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(embeddingDir))
        {
            ModelExporter.Export(container, exportDir, patch: null);
        }

        // Config.json is present and is a standard checkpoint
        var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(exportDir, "config.json")));
        Assert.Equal("qwen3_5_text", config.RootElement.GetProperty("model_type").GetString());

        // Model weights are present
        using var file = SafetensorsFile.Open(Path.Combine(exportDir, "model.safetensors"));
        Assert.True(file.Contains("model.embed_tokens.weight"));
    }

    [Fact]
    public void ConvertToClassifier_Exports_With_ScoreHead()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var classifierDir = Path.Combine(dir.Path, "classifier");

        ModelConverter.ConvertToClassifier(containerPath, classifierDir, numLabels: 5);

        var exportDir = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(classifierDir))
        {
            ModelExporter.Export(container, exportDir, patch: null);
        }

        // Config.json has classifier architectures
        var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(exportDir, "config.json")));
        Assert.Equal("Qwen3_5ForSequenceClassification",
            config.RootElement.GetProperty("architectures")[0].GetString());

        // Score tensor is exported
        using var file = SafetensorsFile.Open(Path.Combine(exportDir, "model.safetensors"));
        Assert.True(file.Contains("model.score.weight"));
    }

    // ── ONNX export ───────────────────────────────────────────────────────

    [Fact]
    public void ExportOnnx_GenerativeModel_Produces_Valid_File()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var outPath = Path.Combine(dir.Path, "model.onnx");

        using (var container = Vindex3Container.Open(containerPath))
        {
            var result = OnnxExporter.Export(container, "target", outPath);
            Assert.Equal(outPath, result.Path);
            Assert.True(result.NodeCount > 0);
            Assert.True(result.InitializerCount > 0);
        }

        Assert.True(File.Exists(outPath));
        Assert.True(new FileInfo(outPath).Length > 100); // at least some bytes
    }

    [Fact]
    public void ExportOnnx_ClassifierModel_Produces_Pooling_Graph()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var classifierDir = Path.Combine(dir.Path, "classifier");
        ModelConverter.ConvertToClassifier(containerPath, classifierDir, numLabels: 3);

        var outPath = Path.Combine(dir.Path, "classifier.onnx");
        using (var container = Vindex3Container.Open(classifierDir))
        {
            var result = OnnxExporter.Export(container, "target", outPath);
            Assert.True(result.NodeCount > 0);
        }
        Assert.True(File.Exists(outPath));
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