using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Phase 0 of the calculated MTP drafter — the data contract: one
/// deterministic forward pass collects the fit pairs (x_t, y_t) and a
/// held-out split; the contract is shape-exact, byte-deterministic, and
/// y_t is independently verified to be the model's pre-final-norm
/// residual hidden at t+1.
/// </summary>
public class PairCollectorTests
{
    private static string DemoContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "demo-pairs");
        return containerPath;
    }

    private static int[] DemoTokens(string containerPath) =>
        HfTokenizer.FromModelDir(containerPath)
            .EncodeToIds("abcdefghijabcdefghijabcdefghij") // 30 chars, vocab a–j only
            .ToArray();

    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    [Fact]
    public void Collects_The_Contract_With_Exact_Shapes_And_Split()
    {
        using var dir = new TempDir();
        var containerPath = DemoContainer(dir);
        var tokens = DemoTokens(containerPath);
        var sink = Path.Combine(dir.Path, "pairs");

        var data = PairCollector.Collect(containerPath, sink, tokens, fitPositions: 8, gatePositions: 4);

        Assert.Equal(8, data.FitCount);
        Assert.Equal(4, data.GateCount);
        Assert.Equal(4, data.Hidden);
        Assert.Equal(8 * 2 * 4, ReadF32(Path.Combine(sink, "fit.x.bin")).Length);
        Assert.Equal(8 * 4, ReadF32(Path.Combine(sink, "fit.y.bin")).Length);

        using var manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(sink, "manifest.json")));
        Assert.Equal(8, manifest.RootElement.GetProperty("fit_count").GetInt32());
        Assert.Equal(4, manifest.RootElement.GetProperty("gate_count").GetInt32());
        Assert.Equal(8, manifest.RootElement.GetProperty("x_columns").GetInt32()); // 2h
        var split = manifest.RootElement.GetProperty("split");
        Assert.Equal(0, split[0]!.GetInt32());
        Assert.Equal(8, split[1]!.GetInt32());
        Assert.Equal(12, split[2]!.GetInt32());
    }

    [Fact]
    public void Collection_Is_Byte_Deterministic()
    {
        using var dir = new TempDir();
        var containerPath = DemoContainer(dir);
        var tokens = DemoTokens(containerPath);

        var a = Path.Combine(dir.Path, "pairs-a");
        var b = Path.Combine(dir.Path, "pairs-b");
        var dataA = PairCollector.Collect(containerPath, a, tokens, 8, 4);
        var dataB = PairCollector.Collect(containerPath, b, tokens, 8, 4);

        Assert.Equal(dataA.DeterminismHashHex, dataB.DeterminismHashHex);
        Assert.Equal(
            File.ReadAllBytes(Path.Combine(a, "fit.x.bin")),
            File.ReadAllBytes(Path.Combine(b, "fit.x.bin")));
        Assert.Equal(
            File.ReadAllBytes(Path.Combine(a, "fit.y.bin")),
            File.ReadAllBytes(Path.Combine(b, "fit.y.bin")));
        Assert.Equal(
            File.ReadAllBytes(Path.Combine(a, "tokens.bin")),
            File.ReadAllBytes(Path.Combine(b, "tokens.bin")));
    }

    [Fact]
    public void Y_Target_Is_The_Models_Residual_At_T_Plus_One()
    {
        using var dir = new TempDir();
        var containerPath = DemoContainer(dir);
        var tokens = DemoTokens(containerPath);
        var sink = Path.Combine(dir.Path, "pairs");
        PairCollector.Collect(containerPath, sink, tokens, 8, 4);

        var ys = ReadF32(Path.Combine(sink, "fit.y.bin"));

        // Independent forward over the SAME sequence: y[t] must equal the
        // raw post-layer residual (pre-final-norm) hidden at POSITION t+1
        // after consuming tokens[0..t+1].
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var runtime = new GenericRuntime(plan, store);
        float[] residual = Array.Empty<float>();
        var expected = new float[8 * 4];
        var normed = new float[8 * 4];
        for (int t = 0; t <= 8; t++)
        {
            residual = runtime.StepForward(tokens[t]).Data;
            if (t >= 1 && t - 1 < 8)
            {
                Array.Copy(residual, 0, expected, (t - 1) * 4, 4);
            }
            if (t < 8)
            {
                // x[t]’s hidden half = pre_fc_norm(final_norm(h_t)); the
                // demo’s carried mtp makes the collector use the final-norm
                // weights for both applications.
                var n = (float[])residual.Clone();
                var finalNorm = BitPattern.WidenToF32(
                    store.Resolve(plan.FinalNorm.Weight.ObjectId, plan.FinalNorm.Weight.TensorName).Dtype,
                    store.Resolve(plan.FinalNorm.Weight.ObjectId, plan.FinalNorm.Weight.TensorName).Payload);
                Norms.ApplyInPlace(new Tensor2D(n, 1, 4), plan.FinalNorm.Kind, plan.FinalNorm.Eps,
                    finalNorm, plan.FinalNorm.WeightOffset);
                Norms.ApplyInPlace(new Tensor2D(n, 1, 4), plan.FinalNorm.Kind, plan.FinalNorm.Eps,
                    finalNorm, plan.FinalNorm.WeightOffset);
                Array.Copy(n, 0, normed, t * 4, 4);
            }
        }

        Assert.Equal(expected, ys);

        // And x[t]'s embedding half is pre_fc_norm(e_{t+1}).
        var xs = ReadF32(Path.Combine(sink, "fit.x.bin"));
        for (int t = 0; t < 8; t++)
        {
            Assert.Equal(normed.Skip(t * 4).Take(4).ToArray(), xs.Skip(t * 8).Take(4).ToArray());
            var e = (float[])TensorOps.GatherRows(
                runtime.Weights.Matrix(plan.Embedding!.Table, plan.Embedding.VocabSize, plan.Embedding.HiddenSize),
                new[] { tokens[t + 1] }).Data.Clone();
            var finalNorm = BitPattern.WidenToF32(
                store.Resolve(plan.FinalNorm.Weight.ObjectId, plan.FinalNorm.Weight.TensorName).Dtype,
                store.Resolve(plan.FinalNorm.Weight.ObjectId, plan.FinalNorm.Weight.TensorName).Payload);
            Norms.ApplyInPlace(new Tensor2D(e, 1, 4), plan.FinalNorm.Kind, plan.FinalNorm.Eps,
                finalNorm, plan.FinalNorm.WeightOffset);
            Assert.Equal(e, xs.Skip(t * 8 + 4).Take(4).ToArray());
        }
    }

    [Fact]
    public void Sink_Refuses_Too_Small_A_Corpus()
    {
        using var dir = new TempDir();
        var containerPath = DemoContainer(dir);
        var tokens = DemoTokens(containerPath); // 30 tokens

        // 20 fit + 10 gate + 2 = 32 > 30 → refused.
        var ex = Assert.ThrowsAny<Exception>(() =>
            PairCollector.Collect(containerPath, Path.Combine(dir.Path, "pairs"), tokens, 20, 10));
        Assert.Contains("corpus tokens", ex.Message);
    }
}