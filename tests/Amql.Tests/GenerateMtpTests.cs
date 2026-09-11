using Amql.Cli;
using Amql.Hf;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// generate-mtp: a bootstrapped MTP drafter for a dense model that has
/// none — the trunk copies the last full-attention layer, the norms copy
/// the final norm, the fc projector is the mean-combination boot, the
/// module materialises as mtp.stack and export emits the companion.
/// </summary>
public class GenerateMtpTests
{
    private static string DemoContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "demo-mtp");
        return containerPath;
    }

    private static int[] DemoTokens(string containerPath) =>
        HfTokenizer.FromModelDir(containerPath).EncodeToIds("bcdefghij").ToArray();

    [Fact]
    public void GenerateMtp_Bootstraps_From_The_Model_Itself()
    {
        using var dir = new TempDir();
        var containerPath = DemoContainer(dir);
        var outDir = Path.Combine(dir.Path, "mtp");

        var report = GenerateMtp.Transform(containerPath, outDir, DemoTokens(containerPath));

        Assert.Equal(1, report.TrunkLayer); // the demo's last layer is full-attention
        Assert.True(double.IsFinite(report.DraftAcceptance));
        Assert.InRange(report.DraftAcceptance, 0.0, 1.0);

        // The module materialised: every tensor derived verbatim from the
        // source model, and the projector is the mean-combination boot.
        using var source = Vindex3Container.Open(containerPath);
        using var sourceStore = source.CreateOperandStore();
        using (var generated = Vindex3Container.Open(outDir))
        {
            Assert.Contains(generated.Index.Representations.Keys, k => k == "mtp.stack@F32");

            var mtp = generated.Index.Representations["mtp.stack@F32"];
            using var segment = SegmentFile.Open(Path.Combine(generated.Root, mtp.Segment));
            Assert.Equal(
                sourceStore.Resolve("target.decoder_stack", "1.self_attn.q_proj.weight").Payload,
                segment.ReadBytes("layers.0.self_attn.q_proj.weight"));
            Assert.Equal(
                sourceStore.Resolve("target.decoder_stack", "1.mlp.down_proj.weight").Payload,
                segment.ReadBytes("layers.0.mlp.down_proj.weight"));
            Assert.Equal(
                sourceStore.Resolve("target.final_norm", "weight").Payload,
                segment.ReadBytes("norm.weight"));
            Assert.Equal(
                sourceStore.Resolve("target.final_norm", "weight").Payload,
                segment.ReadBytes("pre_fc_norm_hidden.weight"));

            var fc = segment.GetTensor("fc.weight");
            Assert.Equal(new long[] { 4, 8 }, fc.Shape);
            var fcValues = BitPattern.WidenToF32(Dtype.F32, segment.ReadBytes("fc.weight"));
            // 0.5·I in both halves of the [h, 2h] projector.
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(0.5f, fcValues[i * 8 + i], precision: 5);
                Assert.Equal(0.5f, fcValues[i * 8 + 4 + i], precision: 5);
            }
            Assert.Equal(0f, fcValues[0 * 8 + 6]);
        }

        // The companion export works off the generated container.
        var exported = Path.Combine(dir.Path, "exported");
        using (var generated = Vindex3Container.Open(outDir))
        {
            var exportReport = ModelExporter.Export(generated, exported, patch: null);
            Assert.Contains(exportReport.Notes, n => n.Contains("MTP drafter exported alongside"));
        }
        Assert.True(File.Exists(Path.Combine(exported, "mtp.safetensors")));
    }

    [Fact]
    public void GenerateMtp_Without_A_Corpus_Skips_Only_The_Gate()
    {
        using var dir = new TempDir();
        var containerPath = DemoContainer(dir);
        var outDir = Path.Combine(dir.Path, "mtp");

        var report = GenerateMtp.Transform(containerPath, outDir, Array.Empty<int>());

        Assert.Equal(1, report.TrunkLayer);
        Assert.True(double.IsNaN(report.DraftAcceptance));
        Assert.Contains(report.Notes, n => n.Contains("acceptance gate was skipped"));
        using (var generated = Vindex3Container.Open(outDir))
        {
            Assert.Contains(generated.Index.Representations.Keys, k => k == "mtp.stack@F32");
        }
    }
}