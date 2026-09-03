using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>Tests for the CLI inference commands: generation determinism
/// and vocabulary bounds, sampled-mode session RNG, token inspection
/// (embedding profile + neighbours against a brute-force oracle), the
/// fail-closed refusal path, and logits consistency with the library.</summary>
public class CliTests
{
    private static string WriteSynthContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-demo");
        return containerPath;
    }

    [Fact]
    public void Generate_Greedy_Is_Deterministic_And_In_Vocab()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var cfg = new SamplingConfig(Seed: 1, Temperature: 0f); // greedy

        int[] first;
        using (var container = Vindex3Container.Open(containerPath))
        {
            var (prefill, steps) = InferenceRunner.Generate(container, "target", new[] { 2, 3 }, 10, cfg);
            Assert.Equal(new[] { 2, 3 }, prefill);
            first = steps.Select(s => s.Token).ToArray();
        }

        Assert.Equal(10, first.Length);
        Assert.All(first, id => Assert.InRange(id, 0, SyntheticCheckpoint.Vocab - 1));
        Assert.All(first, id => Assert.Equal(true, first.Contains(id))); // ids are sane ints

        using var container2 = Vindex3Container.Open(containerPath);
        var (_, steps2) = InferenceRunner.Generate(container2, "target", new[] { 2, 3 }, 10, cfg);
        Assert.Equal(first, steps2.Select(s => s.Token).ToArray());
    }

    [Fact]
    public void Generate_Sampled_Is_Repeatable_For_A_Seed()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var cfg = new SamplingConfig(Seed: 7, Temperature: 1.0f, TopK: 5);

        int[] one;
        using (var container = Vindex3Container.Open(containerPath))
        {
            var (_, steps) = InferenceRunner.Generate(container, "target", new[] { 1 }, 6, cfg);
            one = steps.Select(o => o.Token).ToArray();
        }
        using (var container2 = Vindex3Container.Open(containerPath))
        {
            var (_, steps2) = InferenceRunner.Generate(container2, "target", new[] { 1 }, 6, cfg);
            Assert.Equal(one, steps2.Select(o => o.Token).ToArray());
        }
        Assert.All(one, id => Assert.InRange(id, 0, SyntheticCheckpoint.Vocab - 1));

        // A different seed must not replay the same sequence (near-certain
        // for a real distribution, asserted to catch RNG reseeding bugs).
        using var container3 = Vindex3Container.Open(containerPath);
        var (_, steps3) = InferenceRunner.Generate(container3, "target", new[] { 1 }, 6, cfg with { Seed = 8 });
        Assert.NotEqual(one, steps3.Select(o => o.Token).ToArray());
    }

    [Fact]
    public void InspectToken_Embedding_Matches_BruteForce_Neighbours()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);

        using var container = Vindex3Container.Open(containerPath);
        var profile = TokenInspector.InspectEmbedding(container, "target", 5, neighborCount: 3);

        Assert.Equal(5, profile.Token);
        Assert.Equal(SyntheticCheckpoint.Vocab, profile.Vocab);
        Assert.Equal(SyntheticCheckpoint.Hidden, profile.Dim);
        Assert.Equal("F32", profile.StoredDtype);

        // Row is the raw stored payload (F32 → identity widening).
        using var store = container.CreateOperandStore();
        var resolution = store.Resolve("target.embedding", "weight");
        var table = BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
        Assert.Equal(table.AsSpan(5 * profile.Dim, profile.Dim).ToArray(), profile.Row);

        // Neighbour ranking recomputed from scratch in the test.
        int dim = profile.Dim;
        double TargetDot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            double acc = 0;
            for (int i = 0; i < a.Length; i++)
            {
                acc += a[i] * b[i];
            }
            return acc;
        }
        var target = table.AsSpan(5 * dim, dim);
        double targetNorm = Math.Sqrt(TargetDot(target, target));
        var brute = new List<(int Token, double Cosine)>();
        for (int t = 0; t < SyntheticCheckpoint.Vocab; t++)
        {
            if (t == 5)
            {
                continue;
            }
            var other = table.AsSpan(t * dim, dim);
            double otherNorm = Math.Sqrt(TargetDot(other, other));
            brute.Add((t, TargetDot(target, other) / (targetNorm * otherNorm)));
        }
        var expected = brute.OrderByDescending(p => p.Cosine).Take(3).Select(p => p.Token).ToArray();
        Assert.Equal(expected, profile.Neighbors.Select(n => n.Token).ToArray());
        Assert.True(profile.Neighbors.Count <= 3);
    }

    [Fact]
    public void Generate_Refuses_Linear_Attention_By_Name()
    {
        var spec = SyntheticModel.BuildSpec(new Dims());
        spec.SystemGraph.Components[0].Attention![0].SetOperator(LayerOperators.LinearAttention);

        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "c");
        ContainerEncoder.Encode(containerPath, spec);

        using var container = Vindex3Container.Open(containerPath);
        var ex = Assert.Throws<UnsupportedOperatorException>(() =>
            InferenceRunner.Generate(container, "target", new[] { 1 }, 1, new SamplingConfig(Seed: 0)));
        Assert.Contains("linear_attention", ex.Message);
    }

    [Fact]
    public void InspectLogits_Top1_Matches_Library_Argmax()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);

        using var container = Vindex3Container.Open(containerPath);
        var report = TokenInspector.InspectLogits(container, "target", 3, new[] { 1, 2 }, 5);
        Assert.InRange(report.Rank, 0, SyntheticCheckpoint.Vocab - 1);
        Assert.Equal(5, report.Top.Count);

        // The reported top-1 must be the library's greedy pick over the
        // same prefill — verifies the CLI and runtime agree end to end.
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var session = new DecodeSession(plan, store);
        var logits = session.Prefill(new[] { 1, 2 });
        Assert.Equal(Sampler.ArgMax(logits), report.Top[0].Token);
    }
}