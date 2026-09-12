using Amql.Cli;
using Amql.Hf;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Phase 1 of the calculated MTP drafter — the ridge projector fit and
/// its assembly into the bootstrapped skeleton: the closed-form solve
/// recovers an exact linear map, two fits are byte-identical, the held-out
/// acceptance is reported through the runtime path, and the fitted block
/// lands in the container and exports.
/// </summary>
public class MtpFitterTests
{
    private static string BootContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "demo-fit");
        var bootDir = Path.Combine(dir.Path, "boot");
        GenerateMtp.Transform(containerPath, bootDir, Array.Empty<int>());
        return bootDir;
    }

    private static int[] DemoTokens(string bootDir) =>
        HfTokenizer.FromModelDir(bootDir)
            .EncodeToIds("abcdefghijabcdefghijabcdefghij")
            .ToArray();

    [Fact]
    public void FitProjection_Recovers_An_Exact_Linear_Map()
    {
        var random = new Random(3);
        int n = 24, dIn = 6, dOut = 3;
        var x = new float[n * dIn];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)(random.NextDouble() * 2 - 1);
        }
        var a = new float[dOut * dIn];
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)(random.NextDouble() * 2 - 1);
        }
        var y = new float[n * dOut];
        for (int i = 0; i < n; i++)
        {
            for (int o = 0; o < dOut; o++)
            {
                double sum = 0;
                for (int j = 0; j < dIn; j++)
                {
                    sum += a[o * dIn + j] * x[i * dIn + j];
                }
                y[i * dOut + o] = (float)sum;
            }
        }

        var w = LeastSquares.FitProjection(y, x, n, dOut, dIn);

        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(MathF.Abs(w[i] - a[i]) < 1e-3f, $"recovered W[{i}] = {w[i]}, expected {a[i]}");
        }
    }

    [Fact]
    public void Fit_Assembles_And_Reports_Held_Out_Acceptance()
    {
        using var dir = new TempDir();
        var bootDir = BootContainer(dir);
        var tokens = DemoTokens(bootDir);
        var pairs = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(bootDir, pairs, tokens, fitPositions: 8, gatePositions: 4);

        var report = MtpFitter.FitAndAssemble(bootDir, pairs);

        Assert.Equal(8, report.FitCount);
        Assert.Equal(4, report.Hidden);
        Assert.True(double.IsFinite(report.ResidualR2) && report.ResidualR2 is <= 1.001 and >= -0.001);
        Assert.True(double.IsFinite(report.GateAcceptanceBefore));
        Assert.True(double.IsFinite(report.GateAcceptanceAfter));
        Assert.InRange(report.GateAcceptanceAfter, 0.0, 1.0);
        Assert.Equal(64, report.WeightsHashHex.Length);

        // The fitted block replaced the boot's fc (its bytes differ), the
        // container still opens, and the export emits the companion.
        using var container = Vindex3Container.Open(bootDir);
        using var store = container.CreateOperandStore();
        var fc = store.Resolve("mtp.stack", "fc.weight").Payload;
        Assert.NotEqual(F32Bytes(new float[4 * 8]), fc); // fitted content, not zeros
        var exported = Path.Combine(dir.Path, "exported");
        var exportReport = ModelExporter.Export(container, exported, patch: null);
        Assert.Contains(exportReport.Notes, n => n.Contains("MTP drafter exported alongside"));
        Assert.True(File.Exists(Path.Combine(exported, "mtp.safetensors")));
    }

    private static byte[] F32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    [Fact]
    public void Fit_Is_Byte_Deterministic()
    {
        using var dir = new TempDir();
        var bootDir = BootContainer(dir);
        var tokens = DemoTokens(bootDir);
        var pairs = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(bootDir, pairs, tokens, 8, 4);

        var first = MtpFitter.FitAndAssemble(bootDir, pairs);
        // Reopen and refit on a byte-identical copy of the container.
        var copy = Path.Combine(dir.Path, "copy");
        CopyDirectory(bootDir, copy);
        var second = MtpFitter.FitAndAssemble(copy, pairs);

        Assert.Equal(first.WeightsHashHex, second.WeightsHashHex);
        Assert.Equal(first.ResidualR2, second.ResidualR2);
    }

    [Fact]
    public void Fit_Refuses_Without_A_Materialised_Drafter()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "plain");
        var tokens = HfTokenizer.FromModelDir(containerPath).EncodeToIds("abcdefghijabcdefghij").ToArray();
        var pairs = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(containerPath, pairs, tokens, 8, 4);

        var ex = Assert.ThrowsAny<Exception>(() => MtpFitter.FitAndAssemble(containerPath, pairs));
        Assert.Contains("generate-mtp", ex.Message);
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