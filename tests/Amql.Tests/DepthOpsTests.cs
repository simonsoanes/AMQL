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
/// GenericRuntime's row-exact path, against Inkling2Model (a transcription of
/// the reference) on the tiny looped fixture, its experts stored plain, NVFP4
/// and MXFP8.
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
        Inkling2Tests.WriteCheckpoints(quantised, decoded);
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

    /// <summary>
    /// The planned row-exact path against Inkling2Model, token by token: the
    /// prompt, then a greedy continuation. Every logit of every step must be the
    /// same f32 value, and every visit's output the same vector.
    /// </summary>
    [Fact]
    public void LoopedDepth_PlannedRuntime_Matches_The_Oracle_Bit_For_Bit()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var oracle = Inkling2Model.Load(container, store);
        var plan = Planner.Plan(container, "target", store);
        var runtime = new GenericRuntime(plan, store);

        var oracleVisits = new List<float[]>();
        var plannedVisits = new List<float[]>();
        oracle.OnVisitOutput = (_, h) => oracleVisits.Add(h);
        runtime.VisitOutputCapture = (_, h) => plannedVisits.Add(h);

        var (ids, _) = ArchZooTests.Golden("inkling2_loop");
        double worst = 0;
        int steps = 0;
        float[] Compare(int token)
        {
            var expected = oracle.Step(token);
            var actual = runtime.FinalNormAndHead(runtime.StepForward(token)).Row(0).ToArray();
            worst = Math.Max(worst, Worst(actual, expected));
            steps++;
            return expected;
        }
        float[] last = null!;
        foreach (int token in ids)
        {
            last = Compare(token);
        }
        for (int i = 0; i < 8; i++)
        {
            last = Compare(ArgMax(last));
        }
        _output.WriteLine($"{steps} steps ({ids.Length} prompt, {steps - ids.Length} greedy), {plannedVisits.Count} visits: worst logit diff {worst:G4}");
        Assert.Equal(oracleVisits.Count, plannedVisits.Count);
        for (int v = 0; v < oracleVisits.Count; v++)
        {
            Assert.Equal(oracleVisits[v], plannedVisits[v]);
        }
        Assert.Equal(0.0, worst);
    }

    /// <summary>The planned path through a DecodeSession: prompt prefill (position by
    /// position), steps, and a reset, against the oracle.</summary>
    [Fact]
    public void LoopedDepth_DecodeSession_Prefills_And_Steps_Like_The_Oracle()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var oracle = Inkling2Model.Load(container, store);
        using var session = new DecodeSession(Planner.Plan(container, "target", store), store);
        Assert.Equal("cpu", session.Backend);
        var (ids, _) = ArchZooTests.Golden("inkling2_loop");
        for (int round = 0; round < 2; round++)
        {
            oracle.Reset();
            session.Reset();
            var expected = oracle.Prefill(ids)[^1];
            var actual = session.Prefill(ids).Row(0).ToArray();
            Assert.Equal(expected, actual);
            for (int i = 0; i < 4; i++)
            {
                int next = ArgMax(expected);
                expected = oracle.Step(next);
                Assert.Equal(expected, session.Step(next).Row(0).ToArray());
            }
        }
    }

    [Fact]
    public void GpuEngine_Refuses_The_Depth_Ops_By_Name()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.Equal("looped depth (a range of layers run several times)", GpuEngine.Unsupported(plan));
        Assert.Null(GenericRuntime.RowPathRefusal(plan));
        var runtime = new GenericRuntime(plan, store);
        Assert.Throws<UnsupportedOperatorException>(() =>
            runtime.RunLayerInternal(runtime.Embed(new[] { 1 }), 0, new[] { 0 }, new[] { 0 }, appendKv: true));
    }

    /// <summary>
    /// A graph written before the generic surfaces (only the dedicated session's
    /// facts) is refused by the planner with the upgrade named; the upgrade gives
    /// exactly the graph a fresh encode writes, and the planner then plans it.
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
        var facts = component.Execution!.Inkling2!;
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
                        ContextLength = component.Execution.ContextLength,
                        Attention = new AttentionSurface
                        {
                            NumQHeads = facts.FullHeads, NumKvHeads = facts.FullKvHeads, HeadDim = facts.FullHeadDim,
                            ScoreScale = 1.0 / Math.Sqrt(facts.FullHeadDim), AttentionBias = false,
                        },
                        Ffn = new FfnSurface { IntermediateSize = facts.ExpertWidth, Activation = Activation.Silu, FfnType = FfnType.Gated },
                        Norm = new NormSurface { Pre = norm, Post = norm, FinalNorm = norm, Placement = NormPlacement.PreOnly },
                        Head = new HeadSurface { VocabSize = facts.PaddedVocab, FinalLogitSoftcapping = (float)facts.FinalSoftcap },
                        Inkling2 = facts,
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
            // The dedicated session still reads the old graph.
            Assert.True(Inkling2Model.Serves(before));
        }

        var (upgraded, names) = Inkling2.UpgradeGraph(JsonSerializer.Deserialize<SystemGraph>(File.ReadAllBytes(graphPath), ViJson.Options)!);
        Assert.Equal(new[] { "target" }, names);
        Assert.Equal(System.Text.Encoding.UTF8.GetString(fresh),
            System.Text.Encoding.UTF8.GetString(JsonSerializer.SerializeToUtf8Bytes(upgraded, ViJson.Options)));
        Assert.Empty(Inkling2.UpgradeGraph(upgraded).Upgraded);

        File.WriteAllBytes(graphPath, JsonSerializer.SerializeToUtf8Bytes(upgraded, ViJson.Options));
        using var after = Vindex3Container.Open(containerDir);
        using var afterStore = after.CreateOperandStore();
        Assert.True(Planner.Plan(after, "target", afterStore).RunsByVisit);
    }
    /// <summary>
    /// The real checkpoint on the CPU (<c>AMQL_TEST_CONTAINER</c> naming a looped
    /// container whose graph carries the generic surfaces, e.g. one rewritten by
    /// <c>amql-cli graph upgrade</c>): a prompt and a greedy continuation on the
    /// planned row-exact path, then the same tokens on Inkling2Model, one model
    /// in memory at a time. Every logit of every step must agree exactly.
    /// <c>AMQL_TEST_STEPS</c> sets the continuation (4), <c>AMQL_TEST_PROMPT</c> the prompt.
    /// </summary>
    [Fact]
    public void LoopedDepth_Real_Checkpoint_Planned_Cpu_Matches_The_Oracle_When_Available()
    {
        if (RealCheckpoint.Container(c => c.Graph?.Components.FirstOrDefault(x => x.Id == "target")?.Execution is
                { DepthLoop: not null, Inkling2: not null }) is not { } containerDir)
        {
            return;
        }
        int steps = int.TryParse(RealCheckpoint.Setting("STEPS"), out var n) ? n : 4;
        var tokenizer = HfTokenizer.FromModelDir(containerDir);
        var ids = tokenizer.Encode(RealCheckpoint.Setting("PROMPT") ?? "The capital of France is").Ids.ToArray();
        var clock = System.Diagnostics.Stopwatch.StartNew();

        var planned = new List<float[]>();
        var fed = ids.ToList();
        using (var container = Vindex3Container.Open(containerDir))
        using (var store = container.CreateOperandStore())
        {
            var plan = Planner.Plan(container, "target", store);
            _output.WriteLine($"planned: {plan.Schedule.Count} visits over {plan.Layers.Count} layers; GPU engine: {GpuEngine.Unsupported(plan)}");
            var runtime = new GenericRuntime(plan, store);
            for (int i = 0; i < fed.Count; i++)
            {
                planned.Add(runtime.FinalNormAndHead(runtime.StepForward(fed[i])).Row(0).ToArray());
                if (i == fed.Count - 1 && fed.Count < ids.Length + steps)
                {
                    fed.Add(ArgMax(planned[^1]));
                }
            }
        }
        _output.WriteLine($"planned path: {fed.Count} tokens in {clock.Elapsed.TotalSeconds:F0}s (load included)");
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        clock.Restart();
        var oracleLogits = new List<float[]>();
        using (var container = Vindex3Container.Open(containerDir))
        using (var store = container.CreateOperandStore())
        {
            var oracle = Inkling2Model.Load(container, store);
            foreach (int token in fed)
            {
                oracleLogits.Add(oracle.Step(token));
            }
        }
        _output.WriteLine($"oracle: {fed.Count} tokens in {clock.Elapsed.TotalSeconds:F0}s (load included)");

        double worst = 0;
        for (int t = 0; t < fed.Count; t++)
        {
            worst = Math.Max(worst, Worst(planned[t], oracleLogits[t]));
        }
        string text = tokenizer.Decode(fed.Skip(ids.Length).ToList());
        _output.WriteLine($"ids [{string.Join(",", ids)}] greedy [{string.Join(",", fed.Skip(ids.Length))}] '{text}'; worst logit diff over {fed.Count} steps {worst:G4}");
        Assert.Equal(0.0, worst);
    }
}
