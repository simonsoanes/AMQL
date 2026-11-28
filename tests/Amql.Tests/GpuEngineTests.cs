using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// The device engine against the managed path, over every arch-zoo family it
/// takes on: the same prompt (short enough for the GEMV path, long enough for
/// the batched GEMM path) and greedy steps, logits agreeing to rounding. The
/// families it refuses must say why. Skips without a CUDA device.
/// </summary>
[Collection("gpu")]
public class GpuEngineTests : IDisposable
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    // These hold the kernels to the managed f32 path, so the KV cache stays f32
    // (the default FP16 cache is held to f32 in KvFormatTests).
    private readonly string _kvFormat = GpuEngine.RequestedKvFormat;
    public GpuEngineTests() => GpuEngine.RequestedKvFormat = "f32";
    public void Dispose() => GpuEngine.RequestedKvFormat = _kvFormat;

    /// <summary>Families whose plans need an operator the engine leaves to the
    /// managed path (a shared expert, length-dependent rope, Qwen4's gated
    /// residual, QSA and n-gram embedding).</summary>
    private static readonly HashSet<string> Refused = new(StringComparer.Ordinal)
    {
        "qwen2_moe", "llama-dynamic", "phi3-longrope-long", "phi3-longrope-short",
        "qwen4_exp-tiny", "qwen4_exp-nvfp4-tiny",
    };

    private static bool GpuAvailable()
    {
        CudaShim.Reset();
        return CudaShim.Enabled;
    }

    [Theory]
    [MemberData(nameof(ArchZooTests.Families), MemberType = typeof(ArchZooTests))]
    public void Engine_Matches_Managed_Path(string family)
    {
        if (!GpuAvailable())
        {
            return;
        }
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        if (Inkling2Model.Serves(container))
        {
            // Inkling 2's looped stack plans (see DepthOpsTests), but the engine
            // does not run its depth ops yet and refuses them by name; its own
            // GPU path (amql_inkling.cu in amql_cuda) is held to its managed runtime.
            Assert.Equal("looped depth (a range of layers run several times)",
                GpuEngine.Unsupported(Planner.Plan(container, "target", store)));
            InklingMatchesManaged(container, store, family);
            return;
        }
        var plan = Planner.Plan(container, "target", store);

        string? refusal = GpuEngine.Unsupported(plan);
        if (Refused.Contains(family))
        {
            Assert.NotNull(refusal);
            return;
        }
        Assert.True(refusal is null, $"{family}: engine refused — {refusal}");

        var (ids, _) = ArchZooTests.Golden(family);
        // Exact prompts agree with the managed path to f32 rounding; the
        // default rounds prompt activations to BF16 for the tensor cores (a
        // 3-token prompt takes the GEMV path, f32 either way). 800 tokens run
        // as 512 + 288 rows, so a Gated DeltaNet takes its chunked form in
        // both, with a partial last chunk and the state carried between them.
        // A random router scores its experts close together, so a BF16-rounded prompt
        // can send a token to another expert and that token's logits move by whole
        // units (and only that token's): those are held to exact prompts.
        foreach (bool exact in family == "gemma4-moe" ? new[] { true } : new[] { true, false })
        foreach (int length in new[] { 3, 21, 800 })
        {
            var prompt = Enumerable.Range(0, length).Select(i => ids[i % ids.Length]).ToArray();
            using var gpu = new DecodeSession(plan, store) { ExactGpuPrefill = exact };
            using var cpu = new DecodeSession(plan, store) { AllowGpuEngine = false };
            var a = gpu.Prefill(prompt).FirstRow().ToArray();
            var b = cpu.Prefill(prompt).FirstRow().ToArray();
            Assert.Equal("gpu", gpu.Backend);
            Assert.Equal("cpu", cpu.Backend);
            double tolerance = exact ? 1e-3 : 2e-2;
            for (int step = 0; ; step++)
            {
                AssertClose(a, b, $"{family}, {length}-token prompt, {(exact ? "exact" : "bf16")}, step {step}", tolerance);
                if (step == 4)
                {
                    break;
                }
                int next = Sampler.ArgMax(b);
                a = gpu.Step(next).FirstRow().ToArray();
                b = cpu.Step(next).FirstRow().ToArray();
            }
        }
    }

    /// <summary>The tiny Inkling 2 with its experts NVFP4 (amax / denominator),
    /// MXFP8 and plain, mixed inside the loop: the GPU path's expert readers
    /// against the managed runtime's, f32 rows, to 1e-3 of the largest logit with
    /// prompts at f32. Then the default prompt path, the engine's BF16 tensor cores
    /// over the same experts decoded once per chunk, against the f32 one visit by
    /// visit: within 2% through the prelude and the core's first pass (all three
    /// expert forms). Past that a random router's near ties flip under BF16
    /// rounding (visit 7 sends four tokens to other experts), and the sliding
    /// window spreads it, so the logits are the real checkpoint's to judge.</summary>
    [Fact]
    public void Inkling2_Quantised_Experts_Match_Managed_Path()
    {
        if (!Inkling2Gpu.Available)
        {
            return;
        }
        using var dir = new TempDir();
        string quantised = Path.Combine(dir.Path, "q"), containerDir = Path.Combine(dir.Path, "c");
        Inkling2Tests.WriteCheckpoints(quantised, Path.Combine(dir.Path, "dq"));
        ModelToContainer.Encode(quantised, containerDir, "inkling2-q",
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        using var gpu = Inkling2Gpu.TryCreate(container, store, out var reason);
        Assert.True(gpu is not null, reason);
        gpu.KvFormat = "f32";
        gpu.ExactPrefill = true;
        var managed = Inkling2Model.Load(container, store);
        Assert.Contains("NVFP4", managed.ExpertForms);
        Assert.Contains("MXFP8", managed.ExpertForms);
        var (ids, _) = ArchZooTests.Golden("inkling2_loop");
        var prompt = Enumerable.Range(0, 40).Select(i => ids[i % ids.Length]).ToArray();
        var reference = managed.Prefill(prompt);
        using var seq = gpu.NewSequence(64);
        var mine = seq.PrefillAll(prompt);
        for (int t = 0; t < prompt.Length; t++)
        {
            AssertClose(mine[t], reference[t], $"quantised inkling2, position {t}");
        }

        int hidden = container.Graph!.Component("target").HiddenSize;
        const int Visits = 7;   // the prelude (4) and the first pass's first three
        var exact = seq.VisitOutputs(Visits, prompt.Length, hidden);
        gpu.ExactPrefill = false;
        using var tensor = gpu.NewSequence(64);
        tensor.PrefillAll(prompt);
        var fast = tensor.VisitOutputs(Visits, prompt.Length, hidden);
        for (int v = 0; v < Visits; v++)
        {
            // Measured against the visit's largest value: a row of small values rounds alike.
            AssertClose(fast[v].SelectMany(r => r).ToArray(), exact[v].SelectMany(r => r).ToArray(),
                $"quantised inkling2, BF16 prompt, visit {v}", 2e-2);
        }
    }

    /// <summary>
    /// Inkling 2 looped on its GPU path against <see cref="Inkling2Model"/>: every
    /// position of a 3-, 21- and 200-token prompt (the tiny model's window is 4,
    /// so the sliding slots' rings wrap 50 times and its full slots grow past
    /// their first allocation), then greedy steps. With f32 rows the two differ
    /// only in summation order (f32 warp trees against the managed path's double
    /// sums) through 20 layer visits, so the logits agree to 1e-3 of the largest;
    /// FP16 rows round every key and value (2e-2). NVFP4 rows are 4-bit, so they
    /// are held to the decision instead: the same greedy token at most positions.
    /// The replayed CUDA graph is the direct launches, to the bit.
    /// </summary>
    private static void InklingMatchesManaged(Vindex3Container container, OperandStore store, string family)
    {
        using var gpu = Inkling2Gpu.TryCreate(container, store, out var reason);
        Assert.True(gpu is not null, $"{family}: Inkling 2 GPU path refused — {reason}");
        gpu.ExactPrefill = true;
        var managed = Inkling2Model.Load(container, store);
        var (ids, _) = ArchZooTests.Golden(family);
        foreach (int length in new[] { 3, 21, 200 })
        {
            var prompt = Enumerable.Range(0, length).Select(i => ids[i % ids.Length]).ToArray();
            managed.Reset();
            var reference = managed.Prefill(prompt);
            var steps = new List<(int Token, float[] Logits)>();
            for (int s = 0; s < 4; s++)
            {
                int next = Sampler.ArgMax(s == 0 ? reference[^1] : steps[^1].Logits);
                steps.Add((next, managed.Step(next)));
            }
            foreach (var (format, tolerance) in new[] { ("f32", 1e-3), ("fp16", 2e-2), ("nvfp4", double.NaN) })
            {
                gpu.KvFormat = format;
                using var seq = gpu.NewSequence(capacity: 64);
                var mine = seq.PrefillAll(prompt);
                var after = steps.Select(s => seq.Step(s.Token)).ToList();
                string where = $"{family}, {length}-token prompt, {format} rows";
                if (double.IsNaN(tolerance))
                {
                    int agree = Enumerable.Range(0, length).Count(t => Sampler.ArgMax(mine[t]) == Sampler.ArgMax(reference[t]));
                    Assert.True(agree >= length * 0.8, $"{where}: argmax agrees at {agree}/{length}");
                    continue;
                }
                for (int t = 0; t < length; t++)
                {
                    AssertClose(mine[t], reference[t], $"{where}, position {t}", tolerance);
                }
                for (int s = 0; s < steps.Count; s++)
                {
                    AssertClose(after[s], steps[s].Logits, $"{where}, step {s}", tolerance);
                }
            }
        }

        // A step's replayed graph is its direct launches, to the bit; a prompt fed
        // a token at a time (the step kernels) agrees with the same prompt in one
        // chunk (the batched kernels) to f32 rounding; a reset sequence repeats itself.
        gpu.KvFormat = "f32";
        gpu.ContextLength = 1024;   // past the tiny model's 256, so the prompt takes three chunks
        var tokens = Enumerable.Range(0, 600).Select(i => ids[i % ids.Length]).ToArray();
        try
        {
            gpu.UseGraphs = false;
            using var direct = gpu.NewSequence(capacity: 16);
            var a = tokens.Select(direct.Step).ToList();
            gpu.UseGraphs = true;
            using var graphed = gpu.NewSequence(capacity: 16);
            var b = tokens.Select(graphed.Step).ToList();
            for (int t = 0; t < tokens.Length; t++)
            {
                Assert.True(a[t].SequenceEqual(b[t]), $"{family}: graphed step {t} differs from its direct launches");
            }
            using var chunked = gpu.NewSequence(capacity: 16);
            var c = chunked.PrefillAll(tokens);
            for (int t = 0; t < tokens.Length; t++)
            {
                AssertClose(c[t], b[t], $"{family}, chunked prompt against steps, position {t}");
            }
            chunked.Reset();
            Assert.Equal(c[^1], chunked.PrefillAll(tokens)[^1]);
        }
        finally
        {
            gpu.UseGraphs = true;
        }
    }

    /// <summary>A reset sequence on the engine reproduces the first one, and a
    /// sequence long enough to grow the KV cache (and recapture the decode
    /// graph) stays in step with the managed path.</summary>
    [Fact]
    public void Engine_Resets_And_Grows_The_Cache()
    {
        if (!GpuAvailable())
        {
            return;
        }
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, "qwen3"), containerDir, "qwen3",
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden("qwen3");
        var prompt = Enumerable.Range(0, 1000).Select(i => ids[i % ids.Length]).ToArray();

        // Exact prompts, so the 512-row chunks (the second starting past
        // position 0) are held to f32 agreement.
        using var gpu = new DecodeSession(plan, store) { ExactGpuPrefill = true };
        var first = gpu.Prefill(prompt).FirstRow().ToArray();
        gpu.Reset();
        var again = gpu.Prefill(prompt).FirstRow().ToArray();
        Assert.Equal(first, again);

        using var cpu = new DecodeSession(plan, store) { AllowGpuEngine = false };
        var b = cpu.Prefill(prompt).FirstRow().ToArray();
        var a = again;
        // 1000 + 40 crosses the initial 1024-row cache.
        for (int step = 0; step < 40; step++)
        {
            int next = Sampler.ArgMax(b);
            a = gpu.Step(next).FirstRow().ToArray();
            b = cpu.Step(next).FirstRow().ToArray();
        }
        AssertClose(a, b, "after growing the cache");
    }

    /// <summary>A short request captures the decode graph; the next, after a reset, is
    /// long enough that its prefill grows the cache. The graph must be captured again,
    /// or its first step reads the freed buffers (CUDA status 700).</summary>
    [Fact]
    public void Engine_Recaptures_After_A_Prefill_Grows_The_Cache()
    {
        if (!GpuAvailable())
        {
            return;
        }
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, "qwen3"), containerDir, "qwen3",
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden("qwen3");

        using var gpu = new DecodeSession(plan, store) { ExactGpuPrefill = true };
        var a = gpu.Prefill(ids.Take(5).ToArray()).FirstRow().ToArray();
        for (int step = 0; step < 3; step++)
        {
            a = gpu.Step(Sampler.ArgMax(a)).FirstRow().ToArray();
        }
        gpu.Reset();
        var prompt = Enumerable.Range(0, 1500).Select(i => ids[i % ids.Length]).ToArray();
        a = gpu.Prefill(prompt).FirstRow().ToArray();

        using var cpu = new DecodeSession(plan, store) { AllowGpuEngine = false };
        var b = cpu.Prefill(prompt).FirstRow().ToArray();
        for (int step = 0; step < 3; step++)
        {
            int next = Sampler.ArgMax(b);
            a = gpu.Step(next).FirstRow().ToArray();
            b = cpu.Step(next).FirstRow().ToArray();
        }
        AssertClose(a, b, "after a prefill grew the cache");
    }

    private static void AssertClose(float[] actual, float[] expected, string where, double tolerance = 1e-3)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 1.0;
        foreach (var v in expected)
        {
            scale = Math.Max(scale, Math.Abs(v));
        }
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs((double)actual[i] - expected[i]));
        }
        Assert.True(worst <= tolerance * scale, $"{where}: worst logit diff {worst:G4} (largest logit {scale:G4})");
    }
}
