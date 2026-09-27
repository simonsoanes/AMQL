using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// ModernBERT and the Von option-marker decision model: the PyTorch
/// checkpoint reader/writer that <c>option_marker.pt</c> needs, and the
/// encode → export → encode round trip on a synthetic checkpoint laid out
/// exactly like <c>wfzyx/von</c> (bare <c>ModernBertModel</c> names, layer 0
/// without an attention norm, a <c>.pt</c> that repeats the backbone under
/// <c>encoder.</c> and adds the eight-tensor scorer).
/// <para>
/// The two <c>torch_*.pt</c> fixtures were written by <c>torch.save</c>
/// (torch 2.x) so the reader is tested against the real format, not against
/// this repository's own writer.
/// </para>
/// </summary>
public sealed class ModernBertTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, name);

    // ── torch checkpoint reader ─────────────────────────────────────────

    [Fact]
    public void Reads_A_Real_TorchSave_State_Dict()
    {
        using var pt = TorchCheckpoint.Open(Fixture("torch_state_dict.pt"));

        Assert.Equal(
            new[] { "steps", "lin.weight", "lin.bias", "norm.weight", "norm.bias", "view_a", "view_b", "scalar" },
            pt.TensorNames);

        Assert.Equal(new float[] { 0f, 0.5f, 1f, 1.5f, 2f, 2.5f }, Floats(pt.ReadBytes("lin.weight")));
        Assert.Equal(new long[] { 2, 3 }, pt.Get("lin.weight").Shape);
        Assert.Equal(new float[] { -1f, 2.5f }, Floats(pt.ReadBytes("lin.bias")));

        var norm = pt.Get("norm.weight");
        Assert.Equal(Dtype.BF16, norm.Dtype);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, BitPattern.WidenToF32(Dtype.BF16, pt.ReadBytes("norm.weight")));

        Assert.Equal(Dtype.I64, pt.Get("steps").Dtype);
        var steps = pt.ReadBytes("steps");
        Assert.Equal(new long[] { 7, -3, 40_000_000_000 },
            Enumerable.Range(0, 3).Select(i => BitConverter.ToInt64(steps, i * 8)).ToArray());

        // Two views of one storage: the offset has to be honoured.
        Assert.Equal(pt.Get("view_a").StorageKey, pt.Get("view_b").StorageKey);
        Assert.Equal(new float[] { 2, 3, 4, 5 }, BitPattern.WidenToF32(Dtype.F16, pt.ReadBytes("view_a")));
        Assert.Equal(new float[] { 6, 7, 8, 9 }, BitPattern.WidenToF32(Dtype.F16, pt.ReadBytes("view_b")));

        Assert.Empty(pt.Get("scalar").Shape);
        Assert.Equal(new[] { 3.25f }, Floats(pt.ReadBytes("scalar")));
    }

    [Fact]
    public void Refuses_A_NonContiguous_Tensor()
    {
        var e = Assert.Throws<SafetensorsException>(() => TorchCheckpoint.Open(Fixture("torch_noncontiguous.pt")));
        Assert.Contains("non-contiguous", e.Message);
    }

    [Fact]
    public void Refuses_A_Pickle_That_Names_An_Arbitrary_Global()
    {
        using var dir = new TempDir();
        string path = Path.Combine(dir.Path, "evil.pt");
        using (var zip = ZipFile.Open(path, ZipArchiveMode.Create))
        {
            using var s = zip.CreateEntry("archive/data.pkl").Open();
            // os.system("echo pwned") — must be refused before anything is called.
            var pickle = new List<byte> { 0x80, 0x02 };
            pickle.AddRange(Encoding.ASCII.GetBytes("cos\nsystem\nX"));
            pickle.AddRange(BitConverter.GetBytes(10));
            pickle.AddRange(Encoding.ASCII.GetBytes("echo pwned"));
            pickle.AddRange(new byte[] { 0x85, (byte)'R', (byte)'.' });
            s.Write(pickle.ToArray());
        }
        var e = Assert.Throws<SafetensorsException>(() => TorchCheckpoint.Open(path));
        Assert.Contains("os.system", e.Message);
    }

    [Fact]
    public void Writer_Round_Trips_Through_The_Reader_With_Aligned_Storages()
    {
        using var dir = new TempDir();
        string path = Path.Combine(dir.Path, "w.pt");
        var tensors = new[]
        {
            new TensorPayload { Name = "b.weight", Dtype = Dtype.F32, Shape = new long[] { 2, 2 }, Data = Bytes(1f, 2f, 3f, 4f) },
            new TensorPayload { Name = "a.bias", Dtype = Dtype.BF16, Shape = new long[] { 3 }, Data = new byte[] { 1, 2, 3, 4, 5, 6 } },
            new TensorPayload { Name = "count", Dtype = Dtype.I64, Shape = Array.Empty<long>(), Data = BitConverter.GetBytes(70_000L) },
        };
        TorchCheckpointWriter.Write(path, tensors, "state");

        using var pt = TorchCheckpoint.Open(path);
        Assert.Equal(new[] { "b.weight", "a.bias", "count" }, pt.TensorNames);   // order kept, not sorted
        foreach (var t in tensors)
        {
            Assert.Equal(t.Dtype, pt.Get(t.Name).Dtype);
            Assert.Equal(t.Shape, pt.Get(t.Name).Shape);
            Assert.Equal(t.Data, pt.ReadBytes(t.Name));
        }

        // Storage records start on 64-byte boundaries, as torch.save writes them.
        byte[] raw = File.ReadAllBytes(path);
        using var zip = ZipFile.OpenRead(path);
        foreach (var entry in zip.Entries.Where(e => e.FullName.StartsWith("state/data/", StringComparison.Ordinal)))
        {
            int header = IndexOf(raw, Encoding.ASCII.GetBytes(entry.FullName)) - 30;
            int nameLen = BitConverter.ToUInt16(raw, header + 26);
            int extraLen = BitConverter.ToUInt16(raw, header + 28);
            Assert.Equal(0, (header + 30 + nameLen + extraLen) % 64);
        }
    }

    // ── Von import / export ─────────────────────────────────────────────

    [Fact]
    public void Von_Encodes_With_Its_Judged_Facts()
    {
        using var dir = new TempDir();
        string model = WriteSyntheticVon(dir.Path);
        string containerPath = Path.Combine(dir.Path, "c");
        var report = ModelToContainer.Encode(model, containerPath, "von-synth");

        Assert.Contains("marker_calibration.json", report.AncillaryCopied);
        using var container = Vindex3Container.Open(containerPath);
        Assert.Equal("modernbert", container.Index.Family);
        var component = container.Graph!.Components.Single();
        var surface = component.Execution!;

        Assert.Equal(new[] { AttentionSpan.Full, AttentionSpan.Sliding, AttentionSpan.Sliding, AttentionSpan.Full },
            component.Attention!.Select(p => p.Span!.Value));
        Assert.Equal(new[] { 160_000.0, 10_000.0, 10_000.0, 160_000.0 },
            component.Attention!.Select(p => ((PositionRope)p.Position).Theta));
        Assert.Equal(4, component.Attention![1].Window);

        Assert.Equal(Activation.Gelu, surface.Ffn!.Activation);
        Assert.Equal(NormType.LayerNorm, surface.Norm.Pre.Kind);
        Assert.True(surface.Encoder!.FirstLayerSkipsAttentionNorm);
        Assert.Equal("input_then_gate", surface.Encoder.FusedFfnLayout);
        Assert.Equal(3, surface.Encoder.SpecialTokenIds!["pad_token_id"]);

        var marker = surface.OptionMarker!;
        Assert.True(marker.IndependentOptions);
        Assert.Equal(Scorer, marker.ScorerHiddenSize);
        Assert.Equal("[MASK]", marker.MarkerToken);

        var head = container.Graph.Objects.Single(o => o.Kind == ObjectKind.ClassifierHead);
        Assert.Equal(ModernBert.OptionMarkerFile, head.SourceBindings[0].Artifact);
        Assert.Equal(Backbone(Hidden, Layers).Count + 8, report.Tensors);

        // Encoders are refused by the runtime by name, not with a malformed-container error.
        using var store = container.CreateOperandStore();
        var e = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(container, "target", store));
        Assert.Contains("encoder_stack", e.Message);
    }

    [Fact]
    public void Von_Exports_Byte_Identical_And_Reencodes_To_The_Same_Container()
    {
        using var dir = new TempDir();
        string model = WriteSyntheticVon(dir.Path);
        string c1 = Path.Combine(dir.Path, "c1");
        var first = ModelToContainer.Encode(model, c1, "von-synth");

        string exported = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(c1))
        {
            ModelExporter.Export(container, exported, patch: null);
        }

        // The backbone shard: every tensor, same bytes.
        using (var a = SafetensorsFile.Open(Path.Combine(model, "model.safetensors")))
        using (var b = SafetensorsFile.Open(Path.Combine(exported, "model.safetensors")))
        {
            Assert.Equal(a.TensorNames.OrderBy(n => n), b.TensorNames.OrderBy(n => n));
            foreach (string name in a.TensorNames)
            {
                Assert.Equal(a.ReadBytes(name), b.ReadBytes(name));
            }
        }

        // option_marker.pt: same keys in the same (registration) order, same bytes.
        using (var a = TorchCheckpoint.Open(Path.Combine(model, ModernBert.OptionMarkerFile)))
        using (var b = TorchCheckpoint.Open(Path.Combine(exported, ModernBert.OptionMarkerFile)))
        {
            Assert.Equal(a.TensorNames, b.TensorNames);
            foreach (string name in a.TensorNames)
            {
                Assert.Equal(a.ReadBytes(name), b.ReadBytes(name));
            }
        }

        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(exported, "config.json")))!;
        Assert.Equal("ModernBertModel", config["architectures"]![0]!.GetValue<string>());
        Assert.Equal("gelu", config["hidden_activation"]!.GetValue<string>());
        Assert.Equal(3, config["global_attn_every_n_layers"]!.GetValue<int>());
        Assert.Equal(4, config["local_attention"]!.GetValue<int>());
        Assert.Equal(10_000.0, config["rope_parameters"]!["sliding_attention"]!["rope_theta"]!.GetValue<double>());
        Assert.Equal(50, config["sep_token_id"]!.GetValue<int>());
        Assert.Equal(File.ReadAllBytes(Path.Combine(model, "marker_calibration.json")),
            File.ReadAllBytes(Path.Combine(exported, "marker_calibration.json")));

        // Fixed point: the export encodes back to identical payloads.
        var second = ModelToContainer.Encode(exported, Path.Combine(dir.Path, "c2"), "von-synth");
        Assert.Equal(first.Segments.ToDictionary(s => s.Key, s => s.Value.PayloadSha256Hex),
            second.Segments.ToDictionary(s => s.Key, s => s.Value.PayloadSha256Hex));
    }

    [Fact]
    public void Von_Import_Refuses_A_Pt_Whose_Backbone_Differs()
    {
        using var dir = new TempDir();
        string model = WriteSyntheticVon(dir.Path, tamperPtBackbone: true);
        var e = Assert.Throws<ModelConfigException>(() =>
            ModelToContainer.Encode(model, Path.Combine(dir.Path, "c"), "von-synth"));
        Assert.Contains("different", e.Message);
    }

    [Fact]
    public void Plain_ModernBert_Without_A_Head_Encodes_And_Exports()
    {
        using var dir = new TempDir();
        string model = WriteSyntheticVon(dir.Path, withHead: false);
        string c = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(model, c, "modernbert-synth");
        using var container = Vindex3Container.Open(c);
        Assert.Null(container.Graph!.Components[0].Execution!.OptionMarker);

        string exported = Path.Combine(dir.Path, "out");
        ModelExporter.Export(container, exported, patch: null);
        Assert.False(File.Exists(Path.Combine(exported, ModernBert.OptionMarkerFile)));
        Assert.True(File.Exists(Path.Combine(exported, "model.safetensors")));
    }

    [Fact]
    public void Von_Export_Refuses_Quantisation()
    {
        using var dir = new TempDir();
        string model = WriteSyntheticVon(dir.Path);
        string c = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(model, c, "von-synth");
        using var container = Vindex3Container.Open(c);
        var e = Assert.ThrowsAny<Exception>(() =>
            ModelExporter.Export(container, Path.Combine(dir.Path, "out"), patch: null, quantizeMxfp4: true));
        Assert.Contains("--quantize", e.Message);
    }

    // ── synthetic checkpoint ────────────────────────────────────────────

    private const int Hidden = 8;
    private const int Layers = 4;
    private const int Intermediate = 6;
    private const int Vocab = 64;
    private const int Scorer = 4;

    /// <summary>The backbone tensors in ModernBertModel naming and
    /// registration order — layer 0 has no attn_norm, as in the real model.</summary>
    private static List<(string Name, long[] Shape)> Backbone(int hidden, int layers)
    {
        var list = new List<(string, long[])>
        {
            ("embeddings.tok_embeddings.weight", new long[] { Vocab, hidden }),
            ("embeddings.norm.weight", new long[] { hidden }),
        };
        for (int l = 0; l < layers; l++)
        {
            if (l > 0)
            {
                list.Add(($"layers.{l}.attn_norm.weight", new long[] { hidden }));
            }
            list.Add(($"layers.{l}.attn.Wqkv.weight", new long[] { 3 * hidden, hidden }));
            list.Add(($"layers.{l}.attn.Wo.weight", new long[] { hidden, hidden }));
            list.Add(($"layers.{l}.mlp_norm.weight", new long[] { hidden }));
            list.Add(($"layers.{l}.mlp.Wi.weight", new long[] { 2 * Intermediate, hidden }));
            list.Add(($"layers.{l}.mlp.Wo.weight", new long[] { hidden, Intermediate }));
        }
        list.Add(("final_norm.weight", new long[] { hidden }));
        return list;
    }

    private static string WriteSyntheticVon(string root, bool withHead = true, bool tamperPtBackbone = false)
    {
        string dir = Path.Combine(root, "von");
        Directory.CreateDirectory(dir);

        int seed = 1;
        TensorPayload Make(string name, long[] shape)
        {
            long n = shape.Aggregate(1L, (a, b) => a * b);
            var values = new float[n];
            for (long i = 0; i < n; i++)
            {
                values[i] = MathF.Sin(seed * 0.37f + i * 0.11f) * 0.1f;
            }
            seed++;
            return new TensorPayload { Name = name, Dtype = Dtype.F32, Shape = shape, Data = Bytes(values) };
        }

        var backbone = Backbone(Hidden, Layers).Select(t => Make(t.Name, t.Shape)).ToList();
        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), backbone,
            new Dictionary<string, string> { ["format"] = "pt" });

        var config = new JsonObject
        {
            ["architectures"] = new JsonArray("ModernBertModel"),
            ["model_type"] = "modernbert",
            ["hidden_size"] = Hidden,
            ["num_hidden_layers"] = Layers,
            ["num_attention_heads"] = 2,
            ["intermediate_size"] = Intermediate,
            ["hidden_activation"] = "gelu",
            ["max_position_embeddings"] = 64,
            ["vocab_size"] = Vocab,
            ["norm_eps"] = 1e-5,
            ["layer_norm_eps"] = 1e-5,
            ["norm_bias"] = false,
            ["mlp_bias"] = false,
            ["attention_bias"] = false,
            ["local_attention"] = 4,
            ["global_attn_every_n_layers"] = 3,
            ["tie_word_embeddings"] = true,
            ["pad_token_id"] = 3,
            ["sep_token_id"] = 50,
            ["layer_types"] = new JsonArray("full_attention", "sliding_attention", "sliding_attention", "full_attention"),
            ["rope_parameters"] = new JsonObject
            {
                ["full_attention"] = new JsonObject { ["rope_theta"] = 160000.0, ["rope_type"] = "default" },
                ["sliding_attention"] = new JsonObject { ["rope_theta"] = 10000.0, ["rope_type"] = "default" },
            },
        };
        File.WriteAllText(Path.Combine(dir, "config.json"), config.ToJsonString());

        if (withHead)
        {
            var pt = backbone.Select(t => t with { Name = "encoder." + t.Name }).ToList();
            if (tamperPtBackbone)
            {
                var data = (byte[])pt[3].Data.Clone();
                data[0] ^= 0x40;
                pt[3] = pt[3] with { Data = data };
            }
            pt.Add(Make("scorer.input_norm.weight", new long[] { Hidden }));
            pt.Add(Make("scorer.input_norm.bias", new long[] { Hidden }));
            pt.Add(Make("scorer.dense.weight", new long[] { Scorer, Hidden }));
            pt.Add(Make("scorer.dense.bias", new long[] { Scorer }));
            pt.Add(Make("scorer.norm.weight", new long[] { Scorer }));
            pt.Add(Make("scorer.norm.bias", new long[] { Scorer }));
            pt.Add(Make("scorer.out_proj.weight", new long[] { 1, Scorer }));
            pt.Add(Make("scorer.out_proj.bias", new long[] { 1 }));
            TorchCheckpointWriter.Write(Path.Combine(dir, ModernBert.OptionMarkerFile), pt, "option_marker");
            File.WriteAllText(Path.Combine(dir, "marker_calibration.json"),
                "{\"model_type\":\"option_marker\",\"independent_options\":true,\"temperature\":2.2}");
        }
        return dir;
    }

    private static byte[] Bytes(params float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] Floats(byte[] bytes)
    {
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static int IndexOf(byte[] haystack, byte[] needle) =>
        haystack.AsSpan().IndexOf(needle);
}
