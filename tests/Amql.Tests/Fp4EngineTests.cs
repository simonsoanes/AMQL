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
public class Fp4EngineTests : IDisposable
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    // These hold 4-bit weights to the managed path, so the KV cache stays f32
    // (the default FP16 cache is held to f32 in KvFormatTests).
    private readonly string _kvFormat = GpuEngine.RequestedKvFormat;
    public Fp4EngineTests() => GpuEngine.RequestedKvFormat = "f32";
    public void Dispose() => GpuEngine.RequestedKvFormat = _kvFormat;

    /// <summary>Rounding to nearest: what quantising as a weight loads does, value for value.</summary>
    private static QuantiseOptions Nearest(Fp4Format format) => new(format) { Rounding = Fp4Rounding.Nearest };

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
        var report = QuantiseContainer.Transform(source, quantised, Nearest(format));
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
    [InlineData("qwen3_5-gdn")] // q_norm read as x·(1 + w)
    [InlineData("qwen3")] // q_norm read as x·w
    public void Kept_Projections_Stay_Reused_Ones_Are_Copied_And_Scaled_Tensors_Are_Multiplied(string family)
    {
        using var dir = new TempDir();
        string source = Encode(dir, family);
        string first = Path.Combine(dir.Path, "q"), second = Path.Combine(dir.Path, "k");
        QuantiseContainer.Transform(source, first, Nearest(Fp4Format.Nvfp4));
        AttentionOp attention;
        using (var c = Vindex3Container.Open(source))
        using (var st = c.CreateOperandStore())
        {
            attention = Planner.Plan(c, "target", st).Layers.First(l => l.Attention is not null).Attention!;
        }
        string layer = attention.QProj.TensorName[..attention.QProj.TensorName.IndexOf('.')];
        var report = QuantiseContainer.Transform(source, second, new QuantiseOptions(Fp4Format.Nvfp4)
        {
            Reuse = first,
            Keep = new System.Text.RegularExpressions.Regex($@"^{layer}\.self_attn\.[qkv]_proj\."),
            Scale = [(new System.Text.RegularExpressions.Regex($@"^{layer}\.self_attn\.q_norm\."), 2f)],
        });
        Assert.True(report.Quantised > 0);

        using var original = Vindex3Container.Open(source);
        using var originalStore = original.CreateOperandStore();
        using var reused = Vindex3Container.Open(first);
        using var reusedStore = reused.CreateOperandStore();
        using var container = Vindex3Container.Open(second);
        using var store = container.CreateOperandStore();
        // The kept projection is the source's, the others the reused container's codes.
        Assert.Null(store.ResolveScaled(attention.QProj));
        Assert.Equal(originalStore.ResolveWidened(attention.QProj).Values, store.ResolveWidened(attention.QProj).Values);
        Assert.NotNull(store.ResolveScaled(attention.OProj));
        Assert.Equal(reusedStore.ResolveWidened(attention.OProj).Values, store.ResolveWidened(attention.OProj).Values);
        // What the query norm applies, offset + w, is doubled, to BF16's rounding.
        var norm = attention.QNorm!.Weight;
        float offset = attention.QNorm.WeightOffset;
        Assert.Equal(family == "qwen3" ? 0f : 1f, offset);
        var was = originalStore.ResolveWidened(norm).Values;
        var now = store.ResolveWidened(norm).Values;
        Assert.Equal(was.Length, now.Length);
        for (int i = 0; i < was.Length; i++)
        {
            float want = 2 * (offset + was[i]);
            Assert.True(Math.Abs(offset + now[i] - want) <= 1e-2f * Math.Max(1f, Math.Abs(want)),
                $"{family} q_norm[{i}]: {offset + now[i]} against {want}");
        }
        string index = File.ReadAllText(Path.Combine(second, "index.json"));
        Assert.Contains("\"reused\"", index);
        Assert.Contains("\"scaled\"", index);
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
        QuantiseContainer.Transform(Encode(dir, family), quantised, Nearest(format));
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
        QuantiseContainer.Transform(source, quantised, Nearest(format));
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
        QuantiseContainer.Transform(Encode(dir, family), quantised, Nearest(Fp4Format.Nvfp4));
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

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gptq_Keeps_A_Layers_Output_Closer_Than_Rounding_To_Nearest(bool mx)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir, "qwen3_5-gdn"));
        using var store = container.CreateOperandStore();
        using var host = new DecodeSession(Planner.Plan(container, "target", store), store);
        host.Prefill(new[] { 0 });
        var engine = host.DeviceEngine()!;

        // Inputs as activations are: correlated columns, a few far louder than the rest.
        const int rows = 96, cols = 256, samples = 2048, rank = 24;
        var rng = new Random(5);
        double Normal() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        var basis = Enumerable.Range(0, cols * rank).Select(_ => Normal()).ToArray();
        var loud = Enumerable.Range(0, cols).Select(c => c % 37 == 0 ? 8.0 : 1.0).ToArray();
        var x = new double[samples * cols];
        for (int s = 0; s < samples; s++)
        {
            var z = Enumerable.Range(0, rank).Select(_ => Normal()).ToArray();
            for (int c = 0; c < cols; c++)
            {
                double v = 0.1 * Normal();
                for (int j = 0; j < rank; j++)
                {
                    v += z[j] * basis[c * rank + j];
                }
                x[s * cols + c] = v * loud[c];
            }
        }
        var h = new double[cols * cols];
        Parallel.For(0, cols, i =>
        {
            for (int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int s = 0; s < samples; s++)
                {
                    sum += x[s * cols + i] * x[s * cols + j];
                }
                h[i * cols + j] = sum;
            }
        });
        var w = Enumerable.Range(0, rows * cols).Select(_ => (float)(0.05 * Normal())).ToArray();
        float global = mx ? 1f : w.Max(MathF.Abs) / (BitPattern.Fp4MaxValue * 448f);
        int group = mx ? 32 : 16;

        float[] Decode(byte[] codes, byte[] scaleBytes) => new ScaledMatrix(Dtype.FP4, codes, rows, cols,
            scaleBytes.Select(b => mx ? BitPattern.DecodeF8E8M0(b) : BitPattern.DecodeF8E4M3(b)).ToArray(),
            rows, cols / group, global).Dequant();
        // The layer's output error over the inputs: Σ‖X (W − Q)ᵀ‖² = Σ_r d_r H d_rᵀ.
        double OutputError(float[] q)
        {
            double total = 0;
            for (int r = 0; r < rows; r++)
            {
                var d = Enumerable.Range(0, cols).Select(c => (double)w[r * cols + c] - q[r * cols + c]).ToArray();
                for (int i = 0; i < cols; i++)
                {
                    double hd = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        hd += h[i * cols + j] * d[j];
                    }
                    total += d[i] * hd;
                }
            }
            return total;
        }
        double WeightError(float[] q) => w.Zip(q).Sum(p => (double)(p.First - p.Second) * (p.First - p.Second));

        float[] nearest;
        if (mx)
        {
            var m = Mxfp4.Quantize(w, rows, cols);
            nearest = Decode(m.Packed, m.BlockScales);
        }
        else
        {
            nearest = Nvfp4.Quantize(w, rows, cols).Dequant();
        }
        var (sc, ss) = engine.QuantiseFp4(w, rows, cols, null, mx, global);
        var searched = Decode(sc, ss);
        var (gc, gs) = engine.QuantiseFp4(w, rows, cols, h.Select(v => (float)v).ToArray(), mx, global);
        var gptq = Decode(gc, gs);

        // Searching the scale can only help the weights; GPTQ trades weight
        // error for output error, and wins on the output by a wide margin.
        Assert.True(WeightError(searched) <= WeightError(nearest) * 1.0001,
            $"searched scales {WeightError(searched)} vs nearest {WeightError(nearest)}");
        double eNear = OutputError(nearest), eGptq = OutputError(gptq);
        Assert.True(eGptq < 0.7 * eNear, $"GPTQ output error {eGptq} vs nearest {eNear}");
    }

    /// <summary>Each routed expert is rounded against the rows sent to it, read by a
    /// 4-bit copy of the model as a model too large for the GPU would be.</summary>
    [Fact]
    public void Experts_Are_Calibrated_On_What_The_Router_Sends_Them()
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        const string family = "gemma4-moe";
        using var dir = new TempDir();
        string source = Encode(dir, family), nearest = Path.Combine(dir.Path, "near"), gptq = Path.Combine(dir.Path, "gptq");
        var (ids, _) = ArchZooTests.Golden(family);
        var calibration = Enumerable.Range(0, 2048).Select(i => ids[(i * 7 + i / 5) % ids.Length]).ToArray();
        QuantiseContainer.Transform(source, nearest, Nearest(Fp4Format.Nvfp4));
        var report = QuantiseContainer.Transform(source, gptq, new QuantiseOptions(Fp4Format.Nvfp4)
        {
            Rounding = Fp4Rounding.Gptq,
            CalibrationIds = calibration,
            SequenceTokens = 128,
            CalibrateWith = nearest,
        });
        // 6 layers of 8 experts, three projections each, and the dense and attention ones.
        Assert.True(report.Quantised >= 6 * 8 * 3 + 6 * 6, $"{report.Quantised} projections");
        Assert.Contains("\"calibrated_with\":\"near\"", File.ReadAllText(Path.Combine(gptq, "index.json")));

        var prompt = Enumerable.Range(0, 200).Select(i => ids[(i * 5 + i / 3) % ids.Length]).ToArray();
        float[] Logits(string path, bool gpu)
        {
            using var container = Vindex3Container.Open(path);
            using var store = container.CreateOperandStore();
            using var session = new DecodeSession(Planner.Plan(container, "target", store), store) { AllowGpuEngine = gpu, ExactGpuPrefill = true };
            return session.Prefill(prompt).FirstRow().ToArray();
        }
        AssertClose(Logits(gptq, gpu: false), Logits(gptq, gpu: true), "gptq experts on both paths", 1e-3);
    }

    [Theory]
    [InlineData(Fp4Format.Nvfp4)]
    [InlineData(Fp4Format.Mxfp4)]
    public void A_Gptq_Container_Runs_On_The_Engine_As_The_Managed_Path_Does(Fp4Format format)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        const string family = "qwen3_5-gdn";
        using var dir = new TempDir();
        string quantised = Path.Combine(dir.Path, "q");
        var (ids, _) = ArchZooTests.Golden(family);
        var calibration = Enumerable.Range(0, 256).Select(i => ids[(i * 7) % ids.Length]).ToArray();
        var report = QuantiseContainer.Transform(Encode(dir, family), quantised, new QuantiseOptions(format)
        {
            Rounding = Fp4Rounding.Gptq,
            CalibrationIds = calibration,
            SequenceTokens = 64,
        });
        Assert.True(report.Quantised > 0);
        Assert.Contains("\"rounding\":\"gptq\"", File.ReadAllText(Path.Combine(quantised, "index.json")));

        using var container = Vindex3Container.Open(quantised);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var prompt = Enumerable.Range(0, 30).Select(i => ids[i % ids.Length]).ToArray();
        using var gpu = new DecodeSession(plan, store) { ExactGpuPrefill = true };
        using var cpu = new DecodeSession(plan, store) { AllowGpuEngine = false };
        var a = gpu.Prefill(prompt).FirstRow().ToArray();
        var b = cpu.Prefill(prompt).FirstRow().ToArray();
        Assert.True(gpu.DeviceEngine()!.Fp4Matrices > 0);
        AssertClose(b, a, $"gptq {format}", 1e-3);
    }

    [Fact]
    public void Running_At_Four_Bits_Picks_The_Gptq_Copy_Beside_The_Container()
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        const string family = "qwen3_5-gdn";
        using var dir = new TempDir();
        string source = Path.GetFullPath(Encode(dir, family)).TrimEnd('/', '\\');
        string copy = source + "-nvfp4-gptq";
        var (ids, _) = ArchZooTests.Golden(family);
        QuantiseContainer.Transform(source, copy, new QuantiseOptions(Fp4Format.Nvfp4)
        {
            CalibrationIds = Enumerable.Range(0, 128).Select(i => ids[(i * 7) % ids.Length]).ToArray(),
            SequenceTokens = 64,
        });
        Assert.Contains("\"rounding\":\"gptq\"", File.ReadAllText(Path.Combine(copy, "index.json")));

        // The copy is newer than its source, so it is reused, not rebuilt.
        var written = File.GetLastWriteTimeUtc(Path.Combine(copy, "index.json"));
        Assert.Equal(copy, Fp4Cache.Resolve(source, Fp4Format.Nvfp4));
        Assert.Equal(written, File.GetLastWriteTimeUtc(Path.Combine(copy, "index.json")));
        // A container already at 4 bits is run as it is.
        Assert.Equal(copy, Fp4Cache.Resolve(copy, Fp4Format.Nvfp4));
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
