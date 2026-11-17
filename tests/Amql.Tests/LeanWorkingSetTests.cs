using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// The lean working set: stored bits, routed experts read only when picked,
/// embedding rows read one at a time, everything else bounded by a cache.
/// It must give the resident path's logits exactly, whatever the cache holds,
/// and a decode step must read no expert the router did not pick.
/// </summary>
public sealed class LeanWorkingSetTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    public static TheoryData<string> Families => new() { "qwen4_exp-tiny", "qwen3_moe", "olmoe", "qwen3_5-gdn", "llama" };

    private static string Encode(TempDir dir, string family)
    {
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        return containerDir;
    }

    /// <summary>Prefill half the golden ids, then decode the rest one at a
    /// time; every step's logits.</summary>
    private static List<float[]> Run(string containerDir, int[] ids, WeightWorkingSet mode, long? leanCache,
        Action<WeightLoader>? inspect = null)
    {
        string previous = Environment.GetEnvironmentVariable("AMQL_GPU") ?? string.Empty;
        try
        {
            // The managed path: the device GEMM's FP16 is not what is compared.
            Environment.SetEnvironmentVariable("AMQL_GPU", "0");
            CudaShim.Reset();
            using var container = Vindex3Container.Open(containerDir);
            using var store = container.CreateOperandStore();
            var plan = Planner.Plan(container, "target", store);
            var session = new DecodeSession(plan, store, workingSet: mode);
            if (leanCache is { } bytes)
            {
                session.Runtime.Weights.LeanCacheBytes = bytes;
                // Small blocks, so a head that is not cached streams in several.
                session.Runtime.Weights.StreamBlockBytes = 4096;
            }
            int split = Math.Max(1, ids.Length / 2);
            var logits = new List<float[]> { session.Prefill(ids[..split]).FirstRow().ToArray() };
            for (int i = split; i < ids.Length; i++)
            {
                logits.Add(session.Step(ids[i]).FirstRow().ToArray());
            }
            inspect?.Invoke(session.Runtime.Weights);
            return logits;
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", previous);
            CudaShim.Reset();
        }
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void Lean_Matches_Resident_Exactly(string family)
    {
        var (ids, _) = ArchZooTests.Golden(family);
        ids = ids[..Math.Min(ids.Length, 12)];
        using var dir = new TempDir();
        string containerDir = Encode(dir, family);

        var resident = Run(containerDir, ids, WeightWorkingSet.ResidentF32, null);
        // Nothing cached: every matrix streams, every step.
        var streamed = Run(containerDir, ids, WeightWorkingSet.Lean, 0, weights =>
        {
            // The head went through the block-streamed product.
            Assert.True(weights.BlockStreamedProducts > 0);
            Assert.Equal(0, weights.LoadedMatrixCount);
            Assert.Equal(0, weights.Lean.DenseBytes + weights.Lean.ExpertBytes);
            Assert.True(weights.Lean.Reads > 0);
        });
        // Everything fits.
        var cached = Run(containerDir, ids, WeightWorkingSet.Lean, 1L << 30);
        // Room for some: the dense weights and a few experts compete.
        var tight = Run(containerDir, ids, WeightWorkingSet.Lean, 96L << 10, weights =>
            Assert.True(weights.Lean.DenseBytes + weights.Lean.ExpertBytes <= 96L << 10));

        for (int s = 0; s < resident.Count; s++)
        {
            Assert.Equal(resident[s], streamed[s]);
            Assert.Equal(resident[s], cached[s]);
            Assert.Equal(resident[s], tight[s]);
        }
    }

    [Theory]
    [InlineData("qwen4_exp-tiny")]
    [InlineData("qwen3_moe")]
    [InlineData("olmoe")]
    public void A_Decode_Step_Reads_Only_The_Picked_Experts(string family)
    {
        var (ids, _) = ArchZooTests.Golden(family);
        using var dir = new TempDir();
        string containerDir = Encode(dir, family);

        string previous = Environment.GetEnvironmentVariable("AMQL_GPU") ?? string.Empty;
        try
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", "0");
            CudaShim.Reset();
            using var container = Vindex3Container.Open(containerDir);
            using var store = container.CreateOperandStore();
            var plan = Planner.Plan(container, "target", store);
            var session = new DecodeSession(plan, store, workingSet: WeightWorkingSet.Lean);
            var weights = session.Runtime.Weights;
            weights.LeanCacheBytes = 0;
            session.Prefill(ids[..4]);

            var loads = new List<string>();
            var shapes = new List<(OperandRef Operand, long[] Shape)>();
            weights.LoadTrace = (objectId, tensor, shape, _) =>
            {
                loads.Add(tensor);
                shapes.Add((new OperandRef(objectId, tensor), shape));
            };
            session.Step(ids[4]);

            for (int layer = 0; layer < plan.Layers.Count; layer++)
            {
                if (plan.Layers[layer].Ffn?.Routed is not { } routed)
                {
                    continue;
                }
                string prefix = routed.ExpertGatePrefix;
                var experts = loads
                    .Where(t => t.StartsWith(prefix, StringComparison.Ordinal))
                    .Select(t => t[prefix.Length..].Split('.')[0])
                    .Distinct()
                    .Count();
                Assert.InRange(experts, 1, routed.TopK);
                Assert.True(routed.NumExperts > routed.TopK, $"{family}: the fixture must route to a subset");
            }

            // The embedding is read a row at a time, never as the whole table.
            var embed = plan.Embedding!;
            if (plan.Output!.Projection != embed.Table)
            {
                Assert.All(shapes.Where(l => l.Operand == embed.Table), l => Assert.Equal(1, l.Shape[0]));
            }
            Assert.Contains(shapes, l => l.Operand == embed.Table);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", previous);
            CudaShim.Reset();
        }
    }

    [Fact]
    public void Routing_By_Expert_Matches_Routing_Row_By_Row()
    {
        // The kernel groups rows by expert and runs each picked expert once;
        // the answer must be the per-row one, bit for bit.
        var rng = new Random(7);
        Tensor2D Random(int rows, int cols)
        {
            var data = new float[rows * cols];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            }
            return new Tensor2D(data, rows, cols);
        }
        const int hidden = 48, inter = 40, experts = 6, topK = 2, rows = 9;
        var x = Random(rows, hidden);
        var router = Random(experts, hidden);
        var gates = Enumerable.Range(0, experts).Select(_ => Random(inter, hidden)).ToArray();
        var ups = Enumerable.Range(0, experts).Select(_ => Random(inter, hidden)).ToArray();
        var downs = Enumerable.Range(0, experts).Select(_ => Random(hidden, inter)).ToArray();

        foreach (var policy in new[] { ExpertRoutingPolicy.SoftmaxThenSelect, ExpertRoutingPolicy.NormalisedOverSelected })
        {
            var fetched = new List<int>();
            var batched = FfnKernel.Routed(x, router, experts, hidden,
                e => { fetched.Add(e); return (gates[e], ups[e], downs[e]); },
                topK, policy, Activation.Silu);

            var expected = new float[rows * hidden];
            var routedRows = new HashSet<int>();
            for (int r = 0; r < rows; r++)
            {
                var one = new Tensor2D(x.Row(r).ToArray(), 1, hidden);
                var single = FfnKernel.Routed(one, router, gates, ups, downs, topK, policy, Activation.Silu,
                    (_, picked) => routedRows.UnionWith(picked));
                single.Row(0).CopyTo(expected.AsSpan(r * hidden, hidden));
            }
            Assert.Equal(expected, batched.Data);
            // Each picked expert fetched once, none that was not picked.
            Assert.Equal(fetched.Count, fetched.Distinct().Count());
            Assert.True(fetched.ToHashSet().SetEquals(routedRows));
        }
    }
}
