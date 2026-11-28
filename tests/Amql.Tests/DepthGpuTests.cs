using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;
using Xunit.Abstractions;

namespace Amql.Tests;

/// <summary>
/// The device engine's visit path (GpuEngine.Visits.cs over amql_depth.cu) on
/// the tiny looped fixture: looped depth, the block residual, message gates,
/// residual convolutions, the attention's conditioning and packed experts,
/// against the planned CPU path (GenericRuntime's row-exact path, the oracle).
/// Skips without a CUDA device.
/// </summary>
[Collection("gpu")]
public sealed class DepthGpuTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _kvFormat = GpuEngine.RequestedKvFormat;

    public DepthGpuTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => GpuEngine.RequestedKvFormat = _kvFormat;

    private static readonly ArchMapper.EncodeOptions NoVision = new() { IncludeVision = false, IncludeMtp = false };
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    private static bool GpuAvailable()
    {
        CudaShim.Reset();
        return CudaShim.Enabled;
    }

    /// <summary>Every position's logits on the planned CPU path: a token at a time.</summary>
    private static float[][] CpuLogits(ComponentOpPlan plan, OperandStore store, int[] tokens)
    {
        using var cpu = new DecodeSession(plan, store) { AllowGpuEngine = false };
        var rows = new float[tokens.Length][];
        rows[0] = cpu.Prefill([tokens[0]]).FirstRow().ToArray();
        for (int t = 1; t < tokens.Length; t++)
        {
            rows[t] = cpu.Step(tokens[t]).FirstRow().ToArray();
        }
        return rows;
    }

    private static double Worst(float[] actual, float[] expected, out double scale)
    {
        Assert.Equal(expected.Length, actual.Length);
        scale = 1.0;
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs((double)actual[i] - expected[i]));
        }
        return worst;
    }

    private static void AssertClose(float[] actual, float[] expected, string where, double tolerance = 1e-3)
    {
        double worst = Worst(actual, expected, out double scale);
        Assert.True(worst <= tolerance * scale, $"{where}: worst logit diff {worst:G4} (largest logit {scale:G4})");
    }

    private static int ArgMax(float[] x) => Sampler.ArgMax(x);

    /// <summary>
    /// The plain fixture (f32 experts) and the quantised one (NVFP4 with amax /
    /// denominator, MXFP8 and plain experts mixed inside the loop): every position
    /// of a 3-, 21- and 200-token prompt (the window is 4, so the windowed rings
    /// wrap and the full slots grow), then greedy steps, against the planned CPU
    /// path. With f32 rows and f32 prompts the two differ only in summation order
    /// (1e-3 of the largest logit); FP16 rows round every key and value (2e-2);
    /// NVFP4 rows are held to the decision. The replayed step graph equals its
    /// direct launches to the bit, a prompt in chunks agrees with the same tokens
    /// stepped, and a reset sequence repeats itself.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoopedDepth_Gpu_Matches_The_Planned_Cpu_Path(bool quantised)
    {
        if (!GpuAvailable())
        {
            return;
        }
        using var dir = new TempDir();
        string source = Path.Combine(Zoo, "inkling2_loop");
        if (quantised)
        {
            source = Path.Combine(dir.Path, "q");
            LoopedCheckpointTests.WriteCheckpoints(source, Path.Combine(dir.Path, "dq"));
        }
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(source, containerDir, "looped", NoVision);
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.True(plan.RunsByVisit);
        Assert.Null(GpuEngine.Unsupported(plan));
        using var gpu = GpuEngine.TryCreate(plan, new WeightLoader(store), out var reason);
        Assert.True(gpu is not null, reason);
        Assert.NotNull(gpu.BatchRefusal);
        gpu.ExactPrefill = true;
        var (ids, _) = ArchZooTests.Golden("inkling2_loop");
        foreach (int length in new[] { 3, 21, 200 })
        {
            var prompt = Enumerable.Range(0, length).Select(i => ids[i % ids.Length]).ToArray();
            var all = prompt.ToList();
            var reference = CpuLogits(plan, store, prompt).ToList();
            using (var cpu = new DecodeSession(plan, store) { AllowGpuEngine = false })
            {
                foreach (int t in prompt)
                {
                    if (cpu.Position == 0) cpu.Prefill([t]); else cpu.Step(t);
                }
                for (int s = 0; s < 4; s++)
                {
                    all.Add(ArgMax(reference[^1]));
                    reference.Add(cpu.Step(all[^1]).FirstRow().ToArray());
                }
            }
            // NVFP4 rows on the quantised experts compound two 4-bit roundings in a
            // random model of near ties: held to the decision on the plain fixture.
            var formats = new[] { ("f32", 1e-3), ("fp16", 2e-2), ("nvfp4", double.NaN) }.Where(f => !quantised || f.Item1 != "nvfp4");
            foreach (var (format, tolerance) in formats)
            {
                GpuEngine.RequestedKvFormat = format;
                var seq = gpu.NewSequence();
                try
                {
                    var mine = gpu.PrefillAll(seq, prompt).ToList();
                    for (int s = 0; s < 4; s++)
                    {
                        mine.Add(gpu.Step([seq], [all[length + s]])[0]);
                    }
                    string where = $"{(quantised ? "quantised" : "plain")}, {length}-token prompt, {format} rows";
                    // The last step's logits are after all[^1]; compare positions 0 .. length + 3.
                    if (double.IsNaN(tolerance))
                    {
                        int agree = Enumerable.Range(0, length).Count(t => ArgMax(mine[t]) == ArgMax(reference[t]));
                        Assert.True(agree >= length * 0.8, $"{where}: argmax agrees at {agree}/{length}");
                        continue;
                    }
                    double worst = 0;
                    for (int t = 0; t < mine.Count; t++)
                    {
                        AssertClose(mine[t], reference[t], $"{where}, position {t}", tolerance);
                        worst = Math.Max(worst, Worst(mine[t], reference[t], out _));
                    }
                    _output.WriteLine($"{where}: worst logit diff {worst:G3}");
                }
                finally
                {
                    gpu.Release(seq);
                }
            }
        }

        // Graphs against direct launches, chunks against steps, a reset.
        GpuEngine.RequestedKvFormat = "f32";
        var tokens = Enumerable.Range(0, 600).Select(i => ids[i % ids.Length]).ToArray();
        var direct = gpu.NewSequence();
        var graphed = gpu.NewSequence();
        var chunked = gpu.NewSequence();
        try
        {
            gpu.UseGraphs = false;
            var a = tokens.Select(t => gpu.Step([direct], [t])[0]).ToList();
            gpu.UseGraphs = true;
            var b = tokens.Select(t => gpu.Step([graphed], [t])[0]).ToList();
            for (int t = 0; t < tokens.Length; t++)
            {
                Assert.True(a[t].SequenceEqual(b[t]), $"graphed step {t} differs from its direct launches");
            }
            var c = gpu.PrefillAll(chunked, tokens);
            for (int t = 0; t < tokens.Length; t++)
            {
                AssertClose(c[t], b[t], $"chunked prompt against steps, position {t}");
            }
            _output.WriteLine($"600-token prompt in chunks of {gpu.LastPromptChunk}");
            gpu.Reset(chunked);
            Assert.Equal(c[^1], gpu.PrefillAll(chunked, tokens)[^1]);
            // The engine's own chunk against the borrowed one (or the same when memory is short).
            gpu.BorrowChunks = false;
            gpu.Reset(chunked);
            var own = gpu.PrefillAll(chunked, tokens);
            Assert.Equal(GpuEngine.DepthChunk, gpu.LastPromptChunk);
            for (int t = 0; t < tokens.Length; t++)
            {
                AssertClose(own[t], b[t], $"own-chunk prompt against steps, position {t}");
            }
            // Steps after a borrowed prompt, on the rings it grew.
            gpu.BorrowChunks = true;
            var tail = gpu.NewSequence();
            try
            {
                gpu.PrefillAll(tail, tokens.AsSpan(0, 580));
                for (int t = 580; t < tokens.Length; t++)
                {
                    AssertClose(gpu.Step([tail], [tokens[t]])[0], b[t], $"step after a borrowed prompt, position {t}");
                }
            }
            finally
            {
                gpu.Release(tail);
            }
        }
        finally
        {
            gpu.UseGraphs = true;
            gpu.BorrowChunks = true;
            gpu.Release(direct);
            gpu.Release(graphed);
            gpu.Release(chunked);
        }
    }

    /// <summary>The default prompt path (the engine's BF16 tensor cores, the
    /// quantised experts decoded once per chunk, tiled attention) against f32
    /// prompts, visit by visit, on the quantised fixture: within 2% through the
    /// prelude and the core's first pass (all three expert forms). Past that a
    /// random router's near ties flip under BF16 rounding, so the logits are the
    /// real checkpoint's to judge.</summary>
    [Fact]
    public void PackedExperts_Bf16_Prompt_Tracks_The_F32_Prompt_Visit_By_Visit()
    {
        if (!GpuAvailable())
        {
            return;
        }
        using var dir = new TempDir();
        string quantised = Path.Combine(dir.Path, "q"), containerDir = Path.Combine(dir.Path, "c");
        LoopedCheckpointTests.WriteCheckpoints(quantised, Path.Combine(dir.Path, "dq"));
        ModelToContainer.Encode(quantised, containerDir, "looped-q", NoVision);
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        GpuEngine.RequestedKvFormat = "f32";
        using var gpu = GpuEngine.TryCreate(plan, new WeightLoader(store), out var reason);
        Assert.True(gpu is not null, reason);
        var (ids, _) = ArchZooTests.Golden("inkling2_loop");
        var prompt = Enumerable.Range(0, 40).Select(i => ids[i % ids.Length]).ToArray();
        const int Visits = 7;   // the prelude (4) and the first pass's first three
        gpu.ExactPrefill = true;
        var seq = gpu.NewSequence();
        gpu.PrefillAll(seq, prompt);
        var exact = Enumerable.Range(0, Visits).Select(v => gpu.VisitOutput(v, prompt.Length)).ToArray();
        gpu.ExactPrefill = false;
        gpu.Reset(seq);
        gpu.PrefillAll(seq, prompt);
        for (int v = 0; v < Visits; v++)
        {
            var fast = gpu.VisitOutput(v, prompt.Length);
            AssertClose(fast.SelectMany(r => r).ToArray(), exact[v].SelectMany(r => r).ToArray(), $"BF16 prompt, visit {v}", 2e-2);
        }
        gpu.Release(seq);
    }
}
