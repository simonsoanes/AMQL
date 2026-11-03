using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for <c>amql-cli bake-predicate</c>: the closed-form context
/// distillation that writes an in-context predicate into FFN weights. The
/// bake can reach the GPU (solver, device weights), so it shares the
/// collection whose tests reset the process-wide CUDA state.
/// </summary>
[Collection("gpu")]
public class PredicateBakeTests
{
    [Fact]
    public void MinimumNormUpdate_Hits_Targets_And_Leaves_Orthogonal_Inputs_Alone()
    {
        // Three keys in the first three axes of a 6-dim space; targets in 2-d.
        int n = 3, dIn = 6, dOut = 2;
        var keys = new float[n * dIn];
        keys[0 * dIn + 0] = 1f;
        keys[1 * dIn + 1] = 2f;
        keys[2 * dIn + 0] = 1f;
        keys[2 * dIn + 2] = 1f;
        var targets = new float[] { 1f, 0f, 0f, 1f, -1f, 2f };

        var (delta, explained, _) = LeastSquares.FitMinimumNormUpdate(targets, keys, n, dOut, dIn, ridgeRel: 1e-9);

        Assert.True(explained > 0.999, $"explained {explained}");
        for (int i = 0; i < n; i++)
        {
            for (int o = 0; o < dOut; o++)
            {
                double fitted = 0;
                for (int j = 0; j < dIn; j++)
                    fitted += delta[o * dIn + j] * keys[i * dIn + j];
                Assert.Equal(targets[i * dOut + o], fitted, 3);
            }
        }
        // Minimum norm: no weight on the axes no key touches.
        for (int o = 0; o < dOut; o++)
        {
            for (int j = 3; j < dIn; j++)
                Assert.Equal(0f, delta[o * dIn + j]);
        }
    }

    private static string WriteContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.WriteWithDims(modelDir, vocab: 12, hidden: 16, layers: 4,
            numQHeads: 2, numKvHeads: 1, headDim: 8, intermediate: 64);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-predicate");
        return containerPath;
    }

    /// <summary>The "predicate" is a fixed token preamble the teacher sees;
    /// the bare prompts differ, so the model has to carry the preamble's
    /// effect itself once patched.</summary>
    private static List<PredicateProbe> Probes()
    {
        int[] predicate = { 9, 10, 9, 10, 11 };
        int[][] prompts = { new[] { 1, 2 }, new[] { 3, 4, 5 }, new[] { 6 }, new[] { 7, 8, 1 }, new[] { 2, 5, 3 } };
        return prompts.Select(p => new PredicateProbe(predicate.Concat(p).ToArray(), p)).ToList();
    }

    [Fact]
    public void Bake_Moves_The_Bare_Model_Toward_The_Predicated_One()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        var patchPath = Path.Combine(dir.Path, "predicate.safetensors");

        using var container = Vindex3Container.Open(containerPath);
        var report = PredicateBaker.Bake(container, "target", Probes(), patchPath,
            new PredicateBakeOptions { MaxNewTokens = 6, Layers = new[] { 1, 2 }, Ridge = 1e-3 });

        Assert.Equal(new[] { 1, 2 }, report.Layers.Select(l => l.Layer));
        Assert.Equal(5 * 6, report.AlignedRows);
        Assert.True(report.KlBefore > 0, $"teacher and bare model should differ, KL {report.KlBefore}");
        Assert.True(report.KlAfter < report.KlBefore * 0.5,
            $"KL should at least halve: {report.KlBefore:F5} → {report.KlAfter:F5}");
        Assert.True(report.Top1After >= report.Top1Before);

        var patch = WeightPatch.Load(patchPath);
        patch.ValidateAgainst(container);
        Assert.Equal(2, patch.Entries.Count);
        Assert.All(patch.Entries, e => Assert.Equal(new long[] { 16, 64 }, e.Shape));
    }

    [Fact]
    public void Bake_Picks_Middle_Layers_By_Default()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        using var container = Vindex3Container.Open(containerPath);

        var report = PredicateBaker.Bake(container, "target", Probes(), Path.Combine(dir.Path, "p.safetensors"),
            new PredicateBakeOptions { MaxNewTokens = 4, LayerCount = 2 });

        // 4 layers → the middle half is [1, 3).
        Assert.Equal(new[] { 1, 2 }, report.Layers.Select(l => l.Layer));
    }

    [Fact]
    public void Bake_Refuses_Out_Of_Range_Layers_And_No_Probes()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        using var container = Vindex3Container.Open(containerPath);
        string patch = Path.Combine(dir.Path, "p.safetensors");

        Assert.Throws<MergeException>(() => PredicateBaker.Bake(container, "target", Probes(), patch,
            new PredicateBakeOptions { Layers = new[] { 7 } }));
        Assert.Throws<MergeException>(() => PredicateBaker.Bake(container, "target",
            Array.Empty<PredicateProbe>(), patch));
    }

    [Fact]
    public void MinimumNormUpdate_Gpu_Matches_Cpu()
    {
        int n = 200, dIn = 512, dOut = 64;
        var rng = new Random(7);
        var keys = Enumerable.Range(0, n * dIn).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        var targets = Enumerable.Range(0, n * dOut).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        try
        {
            CudaShim.Reset();
            CudaShim.ForceDisable();
            var cpu = LeastSquares.FitMinimumNormUpdate(targets, keys, n, dOut, dIn);
            Assert.False(cpu.Gpu);

            CudaShim.Reset();
            if (!CudaShim.Enabled)
            {
                return; // no device here — the CPU path above is the whole test
            }
            var gpu = LeastSquares.FitMinimumNormUpdate(targets, keys, n, dOut, dIn);
            Assert.True(gpu.Gpu);

            double diff = 0, norm = 0;
            for (int i = 0; i < cpu.Delta.Length; i++)
            {
                diff += Math.Pow(cpu.Delta[i] - gpu.Delta[i], 2);
                norm += Math.Pow(cpu.Delta[i], 2);
            }
            Assert.True(Math.Sqrt(diff / norm) < 1e-4, $"relative difference {Math.Sqrt(diff / norm):g3}");
            Assert.Equal(cpu.Explained, gpu.Explained, 4);
        }
        finally
        {
            CudaShim.Reset();
        }
    }

    [Fact]
    public void AddDelta_Matches_A_Loaded_Patch_And_Undoes_Exactly()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var down = plan.Layers[1].Ffn!.Dense!.Down;
        var rng = new Random(3);
        var delta = Enumerable.Range(0, 16 * 64).Select(_ => (float)(rng.NextDouble() - 0.5) * 0.2f).ToArray();
        int[] tokens = { 1, 2, 3, 4 };

        float[] Logits(GenericRuntime rt)
        {
            rt.ResetKvCache();
            var hidden = rt.Embed(tokens);
            var positions = Enumerable.Range(0, tokens.Length).ToArray();
            for (int l = 0; l < plan.Layers.Count; l++)
                hidden = rt.RunLayerInternal(hidden, l, positions, positions, appendKv: true);
            return rt.FinalNormAndHead(hidden).Data;
        }

        var live = new GenericRuntime(plan, store);
        var baseLogits = Logits(live);
        live.Weights.AddDelta(down, 16, 64, delta);
        var liveLogits = Logits(live);

        var patched = new GenericRuntime(plan, store, WeightPatch.FromEntries(new[]
        {
            new WeightPatchEntry(down.ObjectId, down.TensorName, new long[] { 16, 64 }, delta),
        }));
        Assert.Equal(Logits(patched), liveLogits);
        Assert.NotEqual(baseLogits, liveLogits);

        live.Weights.AddDelta(down, 16, 64, delta.Select(v => -v).ToArray());
        // w + d − d is not bit-exact in f32; within rounding of the logit scale.
        var undone = Logits(live);
        float scale = baseLogits.Max(MathF.Abs);
        for (int i = 0; i < undone.Length; i++)
            Assert.True(MathF.Abs(baseLogits[i] - undone[i]) <= 1e-5f * scale,
                $"logit {i}: {baseLogits[i]} vs {undone[i]} after undo");
    }
}
