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
    /// managed path (routed experts, length-dependent rope, Qwen4's gated
    /// residual, QSA and n-gram embedding).</summary>
    private static readonly HashSet<string> Refused = new(StringComparer.Ordinal)
    {
        "mixtral", "olmoe", "qwen2_moe", "qwen3_moe", "llama-dynamic", "phi3-longrope-long", "phi3-longrope-short",
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
        foreach (bool exact in new[] { true, false })
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
