using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;
using Xunit.Abstractions;

namespace Amql.Tests;

/// <summary>
/// The generic depth and conditioning ops — looped depth, the block residual,
/// cross-pass message gates, residual short convolutions, the attention's
/// conditioning (separate inputs, query scale and clip, log-scaling, a relative
/// bias, sinks, a separate output gate) and packed experts with sigmoid routing
/// and a softcapped GLU — planned from the generic surfaces and run by
/// GenericRuntime's row-exact path, on the tiny looped fixture, its experts
/// stored plain, NVFP4 and MXFP8. (Its logits are held to transformers' in
/// ArchZooTests, the quantised experts to their decoded values in
/// LoopedCheckpointTests, and the device engine to this path in DepthGpuTests.)
/// </summary>
public sealed class DepthOpsTests
{
    private readonly ITestOutputHelper _output;

    public DepthOpsTests(ITestOutputHelper output) => _output = output;

    private static readonly ArchMapper.EncodeOptions NoVision = new() { IncludeVision = false, IncludeMtp = false };

    /// <summary>The tiny fixture encoded with its experts quantised as the real checkpoint stores them.</summary>
    private static string Encode(TempDir dir)
    {
        string quantised = Path.Combine(dir.Path, "q"), decoded = Path.Combine(dir.Path, "dq");
        LoopedCheckpointTests.WriteCheckpoints(quantised, decoded);
        string container = Path.Combine(dir.Path, "qc");
        ModelToContainer.Encode(quantised, container, "looped-q", NoVision);
        return container;
    }

    private static int ArgMax(float[] x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] > x[best])
            {
                best = i;
            }
        }
        return best;
    }

    private static double Worst(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0;
        for (int i = 0; i < a.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        }
        return worst;
    }

    [Fact]
    public void LoopedDepth_Planner_Unrolls_Visits_With_Distinct_Cache_Slots()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);

        // 12 layers, [4, 8) three times: 4 + 3·4 + 4 visits over 12 + 2·4 slots.
        Assert.NotNull(plan.DepthLoop);
        Assert.Equal((4, 8, 3), (plan.DepthLoop!.Start, plan.DepthLoop.End, plan.DepthLoop.Passes));
        var visits = plan.Schedule;
        Assert.Equal(20, visits.Count);
        Assert.Equal(20, visits.Select(v => v.CacheSlot).Distinct().Count());
        Assert.Equal(Enumerable.Range(0, 8).Concat(Enumerable.Range(4, 4)).Concat(Enumerable.Range(4, 8)), visits.Select(v => v.Layer));
        Assert.All(visits.Where(v => v.Pass > 0), v => Assert.True(v.CacheSlot >= 12));
        // One weight list: the looped layers' plans are shared by their visits.
        Assert.Equal(12, plan.Layers.Count);
        Assert.True(plan.RunsByVisit);
        Assert.Equal(128, plan.Output!.VocabSize);
        Assert.Equal(136, plan.Output.TableRows);
    }

    [Fact]
    public void BlockResidual_And_Messages_Are_Planned_Per_Layer()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.NotNull(plan.BlockResidual);
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            var layer = plan.Layers[l];
            Assert.Equal(l % 2 == 0, layer.OpensBlock);
            Assert.Equal(l == 0, layer.MixerInputMix is null);
            Assert.NotNull(layer.FfnInputMix);
            bool full = l % 4 == 3;
            Assert.Equal(full, layer.Attention!.SeparateInputs);
            Assert.Equal(full && l > 0 ? 3 : l == 0 ? 0 : 1, layer.MixerInputMix?.Count ?? 0);
            Assert.Equal(l is >= 4 and < 8, layer.MessageGates is not null);
            Assert.NotNull(layer.MixerOutputConv);
            Assert.NotNull(layer.FfnOutputConv);
            Assert.Equal(full, layer.Attention.KeyConv!.Dynamic);
            Assert.NotNull(layer.Attention.RelativeBias);
            Assert.Equal(4, layer.Attention.RelativeBias!.Extent);
            Assert.True(layer.Attention.Sink is { Projected: true });
            Assert.NotNull(layer.Attention.OutputGateProjection);
            Assert.Equal(full, layer.Attention.LogScaling is not null);
            Assert.NotNull(layer.Ffn!.Routed!.Packed);
        }
        Assert.Equal(new[] { "q", "k", "v", "ffn" }, plan.Layers[5].MessageGates!.Select(g => g.Input));
        // Looped layer 5 is the loop's second: q reads row 2, k row 2, v row 3, ffn row 3.
        Assert.Equal(new[] { 2, 2, 3, 3 }, plan.Layers[5].MessageGates!.Select(g => g.QueryRow));
    }

    /// <summary>The planned path through a DecodeSession — prompt prefill
    /// (position by position), steps, and a reset — gives exactly what stepping
    /// the runtime token by token gives, every visit's output included.</summary>
    [Fact]
    public void LoopedDepth_DecodeSession_Prefills_And_Steps_Like_The_Runtime()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var runtime = new GenericRuntime(plan, store);
        var stepped = new List<float[]>();
        runtime.VisitOutputCapture = (_, h) => stepped.Add(h);
        using var session = new DecodeSession(plan, store) { AllowGpuEngine = false };
        var sessionVisits = new List<float[]>();
        session.Runtime.VisitOutputCapture = (_, h) => sessionVisits.Add(h);
        Assert.Equal("cpu", session.Backend);
        var (ids, _) = ArchZooTests.Golden("inkling2_loop");
        float[] expected = null!;
        foreach (int token in ids)
        {
            expected = runtime.FinalNormAndHead(runtime.StepForward(token)).Row(0).ToArray();
        }
        var continuation = new List<float[]>();
        var tokens = new List<int>();
        for (int i = 0; i < 4; i++)
        {
            tokens.Add(ArgMax(i == 0 ? expected : continuation[^1]));
            continuation.Add(runtime.FinalNormAndHead(runtime.StepForward(tokens[^1])).Row(0).ToArray());
        }
        for (int round = 0; round < 2; round++)
        {
            session.Reset();
            sessionVisits.Clear();
            Assert.Equal(expected, session.Prefill(ids).Row(0).ToArray());
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(continuation[i], session.Step(tokens[i]).Row(0).ToArray());
            }
            Assert.Equal(stepped.Count, sessionVisits.Count);
            for (int v = 0; v < stepped.Count; v++)
            {
                Assert.Equal(stepped[v], sessionVisits[v]);
            }
        }
    }

    [Fact]
    public void LoopedDepth_Plan_Runs_By_Visit_On_Both_Paths()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.Null(GpuEngine.Unsupported(plan));
        Assert.Null(GenericRuntime.RowPathRefusal(plan));
        var runtime = new GenericRuntime(plan, store);
        Assert.Throws<UnsupportedOperatorException>(() =>
            runtime.RunLayerInternal(runtime.Embed(new[] { 1 }), 0, new[] { 0 }, new[] { 0 }, appendKv: true));
    }

    /// <summary>
    /// A graph written by an older build (only the config adapter's facts, under a
    /// key this build does not read) is refused by the planner naming the key and
    /// the upgrade; the upgrade gives exactly the graph a fresh encode writes, and
    /// the planner then plans it.
    /// </summary>
    [Fact]
    public void GraphUpgrade_Rebuilds_The_Generic_Surfaces_From_The_Recorded_Facts()
    {
        using var dir = new TempDir();
        string containerDir = Encode(dir);
        string graphPath = Path.Combine(containerDir, "system_graph.json");
        var fresh = File.ReadAllBytes(graphPath);
        var current = JsonSerializer.Deserialize<SystemGraph>(fresh, ViJson.Options)!;

        // The graph as the converter wrote it before: only the facts.
        var component = current.Components.Single();
        var facts = Inkling2.ReadFacts(Path.Combine(containerDir, ModelToContainer.SourceConfigFile));
        var norm = new NormSpec { Kind = NormType.RmsNorm, Eps = facts.Eps, WeightOffset = 0f };
        var old = new SystemGraph
        {
            Schema = current.Schema,
            Objects = current.Objects,
            Edges = current.Edges,
            Components = new List<Component>
            {
                new()
                {
                    Id = component.Id, Role = component.Role, SourceArtifact = component.SourceArtifact,
                    NumLayers = component.NumLayers, HiddenSize = component.HiddenSize, Attention = component.Attention,
                    Execution = new ExecutionSurface
                    {
                        ContextLength = component.Execution!.ContextLength,
                        Attention = new AttentionSurface
                        {
                            NumQHeads = facts.FullHeads, NumKvHeads = facts.FullKvHeads, HeadDim = facts.FullHeadDim,
                            ScoreScale = 1.0 / Math.Sqrt(facts.FullHeadDim), AttentionBias = false,
                        },
                        Ffn = new FfnSurface { IntermediateSize = facts.ExpertWidth, Activation = Activation.Silu, FfnType = FfnType.Gated },
                        Norm = new NormSurface { Pre = norm, Post = norm, FinalNorm = norm, Placement = NormPlacement.PreOnly },
                        Head = new HeadSurface { VocabSize = facts.PaddedVocab, FinalLogitSoftcapping = (float)facts.FinalSoftcap },
                        Unread = new Dictionary<string, JsonElement> { [Inkling2.FactsKey] = JsonSerializer.SerializeToElement(facts, ViJson.Options) },
                    },
                },
            },
        };
        File.WriteAllBytes(graphPath, JsonSerializer.SerializeToUtf8Bytes(old, ViJson.Options));
        using (var before = Vindex3Container.Open(containerDir))
        using (var store = before.CreateOperandStore())
        {
            var refusal = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(before, "target", store));
            Assert.Contains("graph upgrade", refusal.Message);
            Assert.Contains(Inkling2.FactsKey, refusal.Message);
        }

        var (upgraded, names) = GraphUpgrade.Apply(JsonSerializer.Deserialize<SystemGraph>(File.ReadAllBytes(graphPath), ViJson.Options)!);
        Assert.Equal(new[] { "target" }, names);
        Assert.Equal(System.Text.Encoding.UTF8.GetString(fresh),
            System.Text.Encoding.UTF8.GetString(JsonSerializer.SerializeToUtf8Bytes(upgraded, ViJson.Options)));
        Assert.Empty(GraphUpgrade.Apply(upgraded).Upgraded);

        File.WriteAllBytes(graphPath, JsonSerializer.SerializeToUtf8Bytes(upgraded, ViJson.Options));
        using var after = Vindex3Container.Open(containerDir);
        using var afterStore = after.CreateOperandStore();
        Assert.True(Planner.Plan(after, "target", afterStore).RunsByVisit);
    }
}
