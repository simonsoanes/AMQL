using Amql.Cli;
using Amql.Hf;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Phase 2 of the calculated MTP drafter — the K-projector mixture: the
/// routed free block evaluates exactly (argmax router selects the expert),
/// the sweep {1,4,8} reports a table and ships the acceptance winner, two
/// fits are byte-identical, the shipped block lands in the container and
/// the export retains the routed structure, and a container without a
/// materialised drafter (or with fewer pairs than clusters) is refused.
/// </summary>
public class MtpKProjectorsTests
{
    private static string BootContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "demo-k");
        var bootDir = Path.Combine(dir.Path, "boot");
        GenerateMtp.Transform(containerPath, bootDir, Array.Empty<int>());
        return bootDir;
    }

    private static int[] DemoTokens(string bootDir) =>
        HfTokenizer.FromModelDir(bootDir)
            .EncodeToIds("abcdefghijabcdefghijabcdefghijabcd")
            .ToArray();

    [Fact]
    public void Routed_Block_Routes_And_Projects_Exactly()
    {
        int hidden = 3, dIn = 6;
        var random = new Random(11);
        float[] Map()
        {
            var m = new float[hidden * dIn];
            for (int i = 0; i < m.Length; i++)
            {
                m[i] = (float)(random.NextDouble() * 2 - 1);
            }
            return m;
        }
        var w0 = Map();
        var w1 = Map();
        // r0 pulls +x0 rows to expert 0, r1 pulls -x0 rows to expert 1.
        var router = new float[2 * dIn];
        router[0] = 1f;
        router[dIn] = -1f;
        var block = DraftProjector.Routed(router, new[] { w0, w1 }, hidden);

        int rows = 20;
        var input = new float[rows * dIn];
        var expected = new float[rows * hidden];
        for (int t = 0; t < rows; t++)
        {
            int expert = t % 2;
            var map = expert == 0 ? w0 : w1;
            input[t * dIn] = (expert == 0 ? 1f : -1f) * (1f + 0.05f * t);
            for (int c = 1; c < dIn; c++)
            {
                input[t * dIn + c] = (float)(random.NextDouble() * 0.2 - 0.1);
            }
            for (int o = 0; o < hidden; o++)
            {
                double sum = 0;
                for (int c = 0; c < dIn; c++)
                {
                    sum += map[o * dIn + c] * input[t * dIn + c];
                }
                expected[t * hidden + o] = (float)sum;
            }
        }

        var output = block.Project(input, rows);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(output[i] - expected[i]) < 1e-3f,
                $"projected[{i}] = {output[i]}, expected {expected[i]}");
        }
    }

    [Fact]
    public void Sweep_Reports_Table_And_Materialises_Winner()
    {
        using var dir = new TempDir();
        var bootDir = BootContainer(dir);
        var tokens = DemoTokens(bootDir);
        var pairs = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(bootDir, pairs, tokens, fitPositions: 16, gatePositions: 8);

        var report = MtpKProjectors.SweepAndFit(bootDir, pairs, new[] { 1, 4, 8 });

        Assert.Equal(new[] { 1, 4, 8 }, report.Sweep.Select(e => e.Clusters));
        foreach (var entry in report.Sweep)
        {
            Assert.InRange(entry.GateAcceptance, 0.0, 1.0);
            Assert.True(double.IsFinite(entry.R2));
            Assert.Equal(64, entry.WeightsHash.Length);
        }
        Assert.Contains(report.ShippedClusters, new[] { 1, 4, 8 });
        Assert.True(double.IsFinite(report.GateAcceptanceBefore));
        Assert.True(double.IsFinite(report.GateAcceptanceAfter));
        Assert.Equal(64, report.WeightsHashHex.Length);
        Assert.Equal(report.ShippedClusters, report.Sweep
            .OrderByDescending(e => e.GateAcceptance)
            .ThenBy(e => e.Clusters)
            .First().Clusters);

        // The shipped block is in the container and scores identically
        // through the container path as it did in-memory.
        Assert.Equal(report.GateAcceptanceAfter,
            report.Sweep.First(e => e.Clusters == report.ShippedClusters).GateAcceptance, 3);

        // Export retains the free block's routed (or single) structure.
        using var container = Vindex3Container.Open(bootDir);
        var exported = Path.Combine(dir.Path, "exported");
        ModelExporter.Export(container, exported, patch: null);
        using var drafter = SafetensorsFile.Open(Path.Combine(exported, "mtp.safetensors"));
        if (report.ShippedClusters > 1)
        {
            Assert.True(drafter.Contains("mtp.fc.router.weight"));
            Assert.True(drafter.Contains("mtp.fc.experts.0.weight"));
            Assert.False(drafter.Contains("mtp.fc.weight"));
        }
        else
        {
            Assert.True(drafter.Contains("mtp.fc.weight"));
        }
    }

    [Fact]
    public void Sweep_Is_Byte_Deterministic()
    {
        using var dir = new TempDir();
        var bootDir = BootContainer(dir);
        var tokens = DemoTokens(bootDir);
        var pairs = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(bootDir, pairs, tokens, 16, 8);

        var first = MtpKProjectors.SweepAndFit(bootDir, pairs, new[] { 1, 4, 8 });
        var copy = Path.Combine(dir.Path, "copy");
        CopyDirectory(bootDir, copy);
        var second = MtpKProjectors.SweepAndFit(copy, pairs, new[] { 1, 4, 8 });

        Assert.Equal(first.ShippedClusters, second.ShippedClusters);
        Assert.Equal(first.WeightsHashHex, second.WeightsHashHex);
        Assert.Equal(
            first.Sweep.Select(e => e.WeightsHash),
            second.Sweep.Select(e => e.WeightsHash));
    }

    [Fact]
    public void Sweep_Refuses_Without_A_Materialised_Drafter()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "plain-k");
        var tokens = HfTokenizer.FromModelDir(containerPath).EncodeToIds("abcdefghijabcdefghij").ToArray();
        var pairs = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(containerPath, pairs, tokens, 8, 4);

        var ex = Assert.ThrowsAny<Exception>(() => MtpKProjectors.SweepAndFit(containerPath, pairs, new[] { 1, 4 }));
        Assert.Contains("generate-mtp", ex.Message);
    }

    [Fact]
    public void Sweep_Refuses_Clusters_Beyond_The_Pair_Count()
    {
        using var dir = new TempDir();
        var bootDir = BootContainer(dir);
        var tokens = DemoTokens(bootDir);
        var pairs = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(bootDir, pairs, tokens, 8, 4);

        var ex = Assert.ThrowsAny<Exception>(() => MtpKProjectors.SweepAndFit(bootDir, pairs, new[] { 16 }));
        Assert.Contains("balanced", ex.Message);
    }

    private static void CopyDirectory(string source, string dest)
    {
        Directory.CreateDirectory(dest);
        foreach (var file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(source, file);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(dest, relative))!);
            File.Copy(file, Path.Combine(dest, relative));
        }
    }
}