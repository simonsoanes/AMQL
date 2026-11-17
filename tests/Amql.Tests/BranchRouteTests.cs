using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Branch-adapt-route: topic LoRAs as experts beside the base FFN, a per-layer
/// router, and the merged routed container. The merged container must compute the
/// same function the bench mixes through the adapter seam.
/// </summary>
[Collection("gpu")]
public class BranchRouteTests
{
    private static string WriteContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.WriteWithDims(modelDir, vocab: 12, hidden: 16, layers: 4,
            numQHeads: 2, numKvHeads: 1, headDim: 8, intermediate: 64);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-bar");
        return containerPath;
    }

    private static LoraExpert RandomExpert(ComponentOpPlan plan, string name, int seed, params int[] layers)
    {
        var rng = new Random(seed);
        var byLayer = new Dictionary<int, LoraLayer>();
        foreach (var l in layers)
        {
            var dense = plan.Layers[l].Ffn!.Dense!;
            int rank = 2, inter = dense.IntermediateSize, hidden = plan.HiddenSize;
            float R() => (float)(rng.NextDouble() * 2 - 1);
            byLayer[l] = new LoraLayer
            {
                Layer = l, Down = dense.Down, Hidden = hidden, Inter = inter, Rank = rank, Scale = 1.0,
                A = Enumerable.Range(0, rank * inter).Select(_ => R() * 0.3f).ToArray(),
                B = Enumerable.Range(0, hidden * rank).Select(_ => R() * 0.3f).ToArray(),
            };
        }
        return new LoraExpert { Name = name, Layers = byLayer };
    }

    private static readonly LoraExample[] Data =
    {
        new("a0", new[] { 1, 2, 3 }, new[] { 4, 5, 6 }),
        new("a1", new[] { 1, 3, 2 }, new[] { 4, 6, 5 }),
        new("b0", new[] { 8, 9, 10 }, new[] { 7, 11, 7 }),
        new("b1", new[] { 9, 8, 10 }, new[] { 11, 7, 11 }),
    };

    private static readonly string[] Topics = { "a", "a", "b", "b" };

    private static MoeBench Bench(Vindex3Container container, Func<ComponentOpPlan, IReadOnlyList<LoraExpert>> experts) =>
        new(container, "target", experts);

    private static IReadOnlyList<LoraExpert> TwoExperts(ComponentOpPlan plan) =>
        new[] { RandomExpert(plan, "a", 1, 2, 3), RandomExpert(plan, "b", 2, 3) };

    [Fact]
    public void Fixed_Mixtures_Are_Linear_In_The_Experts()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(WriteContainer(dir));
        using var bench = Bench(container, TwoExperts);
        Assert.Equal(new[] { "anchor", "a", "b" }, bench.Names);
        Assert.Equal(new[] { 2, 3 }, bench.Layers);
        var e = Data[0];
        var prefix = bench.Prefix(e);
        double baseLoss = bench.Loss(e, prefix, Mixing.Of());
        double anchor = bench.Loss(e, prefix, Mixing.Of(1, 0, 0));
        double onlyA = bench.Loss(e, prefix, Mixing.Of(0, 1, 0));
        Assert.Equal(baseLoss, anchor, 6);
        Assert.NotEqual(baseLoss, onlyA, 3);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void The_Merged_Container_Matches_The_Bench(int topK)
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        var routerPath = Path.Combine(dir.Path, "router.safetensors");
        var moePath = Path.Combine(dir.Path, "moe");
        IReadOnlyList<LoraExpert> experts;
        var expected = new List<double>();
        MoeRouter router;
        using (var container = Vindex3Container.Open(containerPath))
        using (var bench = Bench(container, TwoExperts))
        {
            experts = bench.Experts;
            var rng = new Random(5);
            router = MoeRouter.Zero(bench.Names, bench.Layers, bench.Plan.HiddenSize, holdPrompt: false);
            foreach (var w in router.Weights.Values)
                for (int i = 0; i < w.Length; i++)
                    w[i] = (float)(rng.NextDouble() * 2 - 1);
            router.Save(routerPath);
            router = MoeRouter.Load(routerPath);
            foreach (var e in Data)
                expected.Add(bench.Loss(e, bench.Prefix(e), Mixing.Routed(router, topK)));
        }

        MoeBench.WriteMoe(containerPath, moePath, experts, router, topK);

        using var moe = Vindex3Container.Open(moePath);
        using var merged = Bench(moe, _ => Array.Empty<LoraExpert>());
        Assert.True(merged.Plan.Layers[2].Ffn!.Routed is not null && merged.Plan.Layers[3].Ffn!.Routed is not null);
        Assert.True(merged.Plan.Layers[0].Ffn!.Dense is not null && merged.Plan.Layers[1].Ffn!.Dense is not null);
        for (int i = 0; i < Data.Length; i++)
        {
            double actual = merged.Loss(Data[i], merged.Prefix(Data[i]), Mixing.Of());
            Assert.True(Math.Abs(actual - expected[i]) < 1e-4 * Math.Max(1, expected[i]),
                $"{Data[i].Id} top-{topK}: bench {expected[i]}, merged {actual}");
        }
    }

    [Fact]
    public void Loss_Training_Lowers_The_Routing_Set_Loss()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(WriteContainer(dir));
        // Expert a is trained-looking for topic a: here, simply a strong delta.
        using var bench = Bench(container, plan => new[] { RandomExpert(plan, "a", 3, 3), RandomExpert(plan, "b", 4, 3) });
        var prefixes = Data.Select(bench.Prefix).ToList();
        var steps = new List<double>();
        var router = bench.TrainByLoss(Data, prefixes, new RouterTrainOptions { Steps = 25, LearningRate = 5e-2 },
            entry =>
            {
                var loss = entry.GetType().GetProperty("loss")?.GetValue(entry);
                if (loss is double d)
                    steps.Add(d);
            });
        Assert.Equal(25, steps.Count);
        Assert.True(steps[^1] < steps[0] - 1e-3, $"routing loss {steps[0]} -> {steps[^1]}");
        Assert.Equal("loss", router.Method);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void The_Router_Gradient_Matches_Finite_Differences_At_The_Last_Layer(bool hold)
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(WriteContainer(dir));
        // Experts on the last layer only, where the residual gradient is exact.
        using var bench = Bench(container, plan => new[] { RandomExpert(plan, "a", 3, 3), RandomExpert(plan, "b", 4, 3) });
        var prefixes = Data.Select(bench.Prefix).ToList();
        var rng = new Random(9);
        var router = MoeRouter.Zero(bench.Names, bench.Layers, bench.Plan.HiddenSize, hold);
        foreach (var w in router.Weights.Values)
            for (int i = 0; i < w.Length; i++)
                w[i] = (float)(rng.NextDouble() - 0.5);
        var (_, grads, usage) = bench.RouterLossGradient(Data, prefixes, router);
        Assert.Equal(1.0, usage[3].Sum(), 5);
        var g = grads[3];
        var w3 = router.Weights[3];
        double gn = Math.Sqrt(g.Sum(x => (double)x * x));
        Assert.True(gn > 1e-6, "the router gradient vanished");
        var saved = (float[])w3.Clone();
        double h = 1e-2;
        double Shifted(double sign)
        {
            for (int i = 0; i < w3.Length; i++)
                w3[i] = saved[i] + (float)(sign * h * g[i] / gn);
            return bench.RouterLossGradient(Data, prefixes, router).Loss;
        }
        double measured = (Shifted(+1) - Shifted(-1)) / (2 * h);
        Array.Copy(saved, w3, saved.Length);
        Assert.True(Math.Abs(measured - gn) < 2e-3 * gn + 1e-6, $"predicted {gn}, measured {measured}");
    }

    [Fact]
    public void Label_Training_Separates_Two_Topics()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(WriteContainer(dir));
        using var bench = Bench(container, TwoExperts);
        var prefixes = Data.Select(bench.Prefix).ToList();
        var fits = new List<double>();
        var router = bench.TrainByLabel(Data, prefixes, Topics,
            new RouterTrainOptions { Steps = 300, LearningRate = 5e-2, RowStride = 1, HoldPrompt = true },
            entry =>
            {
                if (entry.GetType().GetProperty("accuracy")?.GetValue(entry) is double a)
                    fits.Add(a);
            });
        Assert.Equal(2, fits.Count);
        Assert.All(fits, a => Assert.Equal(1.0, a));
        Assert.True(router.HoldPrompt);
    }

    [Fact]
    public void Extending_A_Router_Keeps_Rows_By_Name()
    {
        var router = MoeRouter.Zero(new[] { "anchor", "a", "b" }, new[] { 2, 3 }, hidden: 4, holdPrompt: false);
        for (int e = 0; e < 3; e++)
            for (int h = 0; h < 4; h++)
                router.Weights[3][e * 4 + h] = e + 1;
        var wider = router.Extend(new[] { "anchor", "b", "c", "a" }, new[] { 3, 4 }, holdPrompt: false);
        Assert.Equal(new float[] { 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 2, 2, 2, 2 }, wider.Weights[3]);
        Assert.All(wider.Weights[4], w => Assert.Equal(0f, w));
        Assert.False(wider.Weights.ContainsKey(2));
    }

    [Fact]
    public void A_Prompt_Held_Router_Is_Not_Materialised()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        IReadOnlyList<LoraExpert> experts;
        MoeRouter router;
        using (var container = Vindex3Container.Open(containerPath))
        using (var bench = Bench(container, TwoExperts))
        {
            experts = bench.Experts;
            router = MoeRouter.Zero(bench.Names, bench.Layers, bench.Plan.HiddenSize, holdPrompt: true);
        }
        var ex = Assert.Throws<MergeException>(() =>
            MoeBench.WriteMoe(containerPath, Path.Combine(dir.Path, "moe"), experts, router, 0));
        Assert.Contains("prompt-held", ex.Message);
    }
}
