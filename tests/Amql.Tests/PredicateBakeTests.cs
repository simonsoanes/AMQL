using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for <c>amql-cli bake-predicate</c>: the closed-form context
/// distillation that writes an in-context predicate into FFN weights.
/// </summary>
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

        var (delta, explained) = LeastSquares.FitMinimumNormUpdate(targets, keys, n, dOut, dIn, ridgeRel: 1e-9);

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
}
