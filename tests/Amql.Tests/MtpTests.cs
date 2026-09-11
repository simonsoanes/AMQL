using System.Text.Json;
using Amql.Cli;
using Amql.Hf;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// The MTP drafter + vision tower export path: the encode materialises the
/// carried modules when the source checkpoint holds them, the main export
/// rides the vision tower inside the model shard while the MTP drafter
/// exports automatically as a companion (mtp.safetensors + mtp.config.json),
/// and a source without the modules leaves them carried.
/// </summary>
public class MtpTests
{
    private static string EncodeSynth(TempDir dir, string name, bool withModules)
    {
        var modelDir = Path.Combine(dir.Path, name + "-model");
        SyntheticCheckpoint.Write(modelDir);
        if (withModules)
        {
            // Mirror the real checkpoint layout: a carried mtp module and a
            // model.visual tower, as their own shards.
            SafetensorsWriter.Write(Path.Combine(modelDir, "mtp.safetensors"), new[]
            {
                Fp("mtp.fc.weight", new long[] { 4, 8 }),
                Fp("mtp.layers.0.mlp.gate_proj.weight", new long[] { 8, 4 }),
                Fp("mtp.layers.0.self_attn.q_proj.weight", new long[] { 4, 4 }),
                Fp("mtp.norm.weight", new long[] { 4 }),
                Fp("mtp.pre_fc_norm_embedding.weight", new long[] { 4 }),
                Fp("mtp.pre_fc_norm_hidden.weight", new long[] { 4 }),
            });
            SafetensorsWriter.Write(Path.Combine(modelDir, "visual.safetensors"), new[]
            {
                Fp("model.visual.blocks.0.attn.q_proj.weight", new long[] { 8, 8 }),
                Fp("model.visual.blocks.0.mlp.up_proj.weight", new long[] { 16, 8 }),
            });
        }
        var containerPath = Path.Combine(dir.Path, name);
        ModelToContainer.Encode(modelDir, containerPath, name);
        return containerPath;
    }

    private static TensorPayload Fp(string name, long[] shape)
    {
        long elements = shape.Aggregate(1L, (a, b) => a * b);
        var data = new float[elements];
        for (long i = 0; i < elements; i++)
        {
            data[i] = (float)(i % 7) * 0.25f;
        }
        return new TensorPayload { Name = name, Dtype = Dtype.F32, Shape = shape, Data = SyntheticModel.ToBytes(data) };
    }

    [Fact]
    public void Encode_Materialises_The_Mtp_Drafter_And_Vision_Tower()
    {
        using var dir = new TempDir();
        var containerPath = EncodeSynth(dir, "demo", withModules: true);

        using var container = Vindex3Container.Open(containerPath);
        Assert.Contains(container.Index.Representations.Keys, k => k == "mtp.stack@F32");
        Assert.Contains(container.Index.Representations.Keys, k => k == "vision.perception_tower@F32");
        Assert.Single(container.Graph!.Objects.Single(o => o.Component == "mtp").Representations);
        Assert.Single(container.Graph.Objects.Single(o => o.Component == "vision").Representations);
    }

    [Fact]
    public void Export_Rides_The_Vision_Tower_And_Companions_The_Drafter()
    {
        using var dir = new TempDir();
        var containerPath = EncodeSynth(dir, "demo", withModules: true);
        var outDir = Path.Combine(dir.Path, "exported");

        using (var container = Vindex3Container.Open(containerPath))
        {
            var report = ModelExporter.Export(container, outDir, patch: null);
            Assert.Contains(report.Notes, n => n.Contains("MTP drafter exported alongside"));
        }

        // The main shard carries the vision tower under model.visual.*, and
        // no mtp tensors or double-dot reconstructions.
        using var shard = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        var names = shard.TensorNames;
        Assert.Contains(names, n => n.Contains("model.visual."));
        Assert.DoesNotContain(names, n => n.StartsWith("mtp.", StringComparison.Ordinal));
        Assert.DoesNotContain(names, n => n.Contains(".."));

        // The drafter rides alongside: module + shared embedding + head.
        Assert.True(File.Exists(Path.Combine(outDir, "mtp.safetensors")));
        Assert.True(File.Exists(Path.Combine(outDir, "mtp.config.json")));
        using var drafter = SafetensorsFile.Open(Path.Combine(outDir, "mtp.safetensors"));
        Assert.Contains("mtp.fc.weight", drafter.TensorNames);
        Assert.Contains("mtp.layers.0.mlp.gate_proj.weight", drafter.TensorNames);
        Assert.Contains("model.embed_tokens.weight", drafter.TensorNames);
        Assert.Contains("lm_head.weight", drafter.TensorNames);

        // The main config documents the vision tower's judged facts.
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, "config.json")));
        Assert.Equal("image", config.RootElement.GetProperty("vision_config").GetProperty("modality").GetString());
        using var mtpConfig = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, "mtp.config.json")));
        Assert.Equal(1, mtpConfig.RootElement.GetProperty("num_hidden_layers").GetInt32());
        Assert.False(mtpConfig.RootElement.GetProperty("mtp").GetProperty("use_dedicated_embeddings").GetBoolean());
    }

    [Fact]
    public void Without_A_Source_Module_The_Encode_Leaves_It_Carried()
    {
        using var dir = new TempDir();
        // No mtp / no vision tensors in the source: the objects exist but
        // stay carried — no companions, no vision lesions, export works.
        var containerPath = EncodeSynth(dir, "bare", withModules: false);
        var outDir = Path.Combine(dir.Path, "exported");

        using (var container = Vindex3Container.Open(containerPath))
        {
            Assert.DoesNotContain(container.Index.Representations.Keys, k => k.StartsWith("mtp.stack", StringComparison.Ordinal));
            Assert.DoesNotContain(container.Index.Representations.Keys, k => k.StartsWith("vision.", StringComparison.Ordinal));
            var report = ModelExporter.Export(container, outDir, patch: null);
            Assert.DoesNotContain(report.Notes, n => n.Contains("MTP drafter"));
        }
        Assert.False(File.Exists(Path.Combine(outDir, "mtp.safetensors")));
        using var shard = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        Assert.DoesNotContain(shard.TensorNames, n => n.Contains("mtp.") || n.Contains("model.visual."));
    }
}