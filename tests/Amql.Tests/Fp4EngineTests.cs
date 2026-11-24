using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// 4-bit projections. <c>QuantiseContainer</c> compiles a container's stack to
/// NVFP4 or MXFP4 once; the GPU engine then runs those weights as stored and
/// agrees with the managed path decoding the same weights, as it does at full
/// precision. Quantising at load instead gives the very same weights, so the
/// engine's logits are bit for bit the compiled container's. Speculative
/// decoding stays exact over 4-bit weights. Skip without a CUDA device (the
/// container tests need none).
/// </summary>
[Collection("gpu")]
public class Fp4EngineTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    private static string Encode(TempDir dir, string family)
    {
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        return containerDir;
    }

    [Theory]
    [InlineData(Fp4Format.Nvfp4)]
    [InlineData(Fp4Format.Mxfp4)]
    public void A_Quantised_Container_Holds_Its_Projections_At_4_Bits(Fp4Format format)
    {
        using var dir = new TempDir();
        string source = Encode(dir, "qwen3_5-gdn");
        string quantised = Path.Combine(dir.Path, "q");
        var report = QuantiseContainer.Transform(source, quantised, format);
        Assert.True(report.Quantised > 0);
        Assert.True(report.BytesAfter < report.BytesBefore);

        using var container = Vindex3Container.Open(quantised);
        using var store = container.CreateOperandStore();
        using var original = Vindex3Container.Open(source);
        using var originalStore = original.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var attention = plan.Layers.First(l => l.Attention is not null).Attention!;
        var scaled = store.ResolveScaled(attention.QProj);
        Assert.NotNull(scaled);
        Assert.Equal(Dtype.FP4, scaled!.Element);
        Assert.Equal(format == Fp4Format.Nvfp4 ? 16 : 32, scaled.BlockCols);

        // Each weight is the 4-bit code of the original: within a block's step.
        var before = originalStore.ResolveWidened(attention.QProj).Values;
        var after = store.ResolveWidened(attention.QProj).Values;
        double peak = before.Max(v => Math.Abs(v));
        for (int i = 0; i < before.Length; i++)
        {
            Assert.True(Math.Abs(before[i] - after[i]) <= peak / 4 + 1e-6, $"element {i}: {before[i]} became {after[i]}");
        }
        // Norms are not projections and stay as they were.
        var norm = plan.Layers[0].PreAttentionNorm!.Weight;
        Assert.Null(store.ResolveScaled(norm));
        Assert.Equal(originalStore.ResolveWidened(norm).Values, store.ResolveWidened(norm).Values);
    }

    [Theory]
    [InlineData("qwen3_5-gdn", Fp4Format.Nvfp4)]
    [InlineData("qwen3_5-gdn", Fp4Format.Mxfp4)]
    [InlineData("qwen3", Fp4Format.Nvfp4)]
    public void Engine_Runs_4_Bit_Weights_As_The_Managed_Path_Does(string family, Fp4Format format)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        string quantised = Path.Combine(dir.Path, "q");
        QuantiseContainer.Transform(Encode(dir, family), quantised, format);
        using var container = Vindex3Container.Open(quantised);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        foreach (int length in new[] { 3, 21, 600 })
        {
            var prompt = Enumerable.Range(0, length).Select(i => ids[i % ids.Length]).ToArray();
            using var gpu = new DecodeSession(plan, store) { ExactGpuPrefill = true };
            using var cpu = new DecodeSession(plan, store) { AllowGpuEngine = false };
            var a = gpu.Prefill(prompt).FirstRow().ToArray();
            var b = cpu.Prefill(prompt).FirstRow().ToArray();
            Assert.Equal("gpu", gpu.Backend);
            Assert.True(gpu.DeviceEngine()!.Fp4Matrices > 0);
            for (int step = 0; ; step++)
            {
                AssertClose(b, a, $"{family} {format}, {length}-token prompt, step {step}", 1e-3);
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

    [Theory]
    [InlineData(Fp4Format.Nvfp4)]
    [InlineData(Fp4Format.Mxfp4)]
    public void Quantising_At_Load_Is_The_Compiled_Container(Fp4Format format)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        const string family = "qwen3_5-gdn";
        using var dir = new TempDir();
        string source = Encode(dir, family);
        string quantised = Path.Combine(dir.Path, "q");
        QuantiseContainer.Transform(source, quantised, format);
        var (ids, _) = ArchZooTests.Golden(family);
        var prompt = Enumerable.Range(0, 40).Select(i => ids[i % ids.Length]).ToArray();

        float[][] Run(string containerDir, GpuWeightFormat load)
        {
            var saved = GpuEngine.WeightFormat;
            GpuEngine.WeightFormat = load;
            try
            {
                using var container = Vindex3Container.Open(containerDir);
                using var store = container.CreateOperandStore();
                var plan = Planner.Plan(container, "target", store);
                using var session = new DecodeSession(plan, store);
                var all = new List<float[]> { session.Prefill(prompt).FirstRow().ToArray() };
                for (int i = 0; i < 6; i++)
                {
                    all.Add(session.Step(Sampler.ArgMax(all[^1])).FirstRow().ToArray());
                }
                Assert.True(session.DeviceEngine()!.Fp4Matrices > 0);
                return all.ToArray();
            }
            finally
            {
                GpuEngine.WeightFormat = saved;
            }
        }

        var compiled = Run(quantised, GpuWeightFormat.Fp16);
        var atLoad = Run(source, format == Fp4Format.Nvfp4 ? GpuWeightFormat.Nvfp4 : GpuWeightFormat.Mxfp4);
        for (int i = 0; i < compiled.Length; i++)
        {
            Assert.Equal(compiled[i], atLoad[i]);
        }
    }

    [Fact]
    public void Verify_Rows_Equal_Steps_Over_4_Bit_Weights()
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        const string family = "qwen3_5-gdn";
        using var dir = new TempDir();
        string quantised = Path.Combine(dir.Path, "q");
        QuantiseContainer.Transform(Encode(dir, family), quantised, Fp4Format.Nvfp4);
        using var container = Vindex3Container.Open(quantised);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        var prompt = Enumerable.Range(0, 19).Select(i => ids[i % ids.Length]).ToArray();
        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;
        engine.RowInvariant = true;

        var plain = engine.NewSequence();
        var logits = new List<float[]> { engine.Prefill(plain, prompt) };
        var tokens = new List<int>();
        for (int i = 0; i < 8; i++)
        {
            tokens.Add(Sampler.ArgMax(logits[^1]));
            logits.Add(engine.Step(new[] { plain }, new[] { tokens[^1] })[0]);
        }
        var spec = engine.NewSequence();
        engine.Prefill(spec, prompt);
        engine.Verify(spec, tokens.ToArray());
        for (int r = 0; r < tokens.Count; r++)
        {
            Assert.Equal(logits[r + 1], engine.HostLogits(r).ToArray());
        }
        engine.Accept(spec, 5);
        Assert.Equal(logits[6], engine.Step(new[] { spec }, new[] { tokens[5] })[0]);
    }

    private static void AssertClose(float[] expected, float[] actual, string what, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0, scale = 1e-6;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
            scale = Math.Max(scale, Math.Abs(expected[i]));
        }
        Assert.True(worst <= tolerance * Math.Max(1, scale), $"{what}: logits differ by up to {worst} (largest {scale})");
    }
}
