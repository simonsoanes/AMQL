using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Quantised checkpoints in the layout of NVIDIA's Qwen3.8-Flash-Next-NVFP4-QAD
/// (ModelOpt MIXED_PRECISION): NVFP4 experts, MXFP8 projections and an FP8
/// n-gram table, each kept as stored and decoded through its scales.
/// <c>fixtures/arch-zoo/qwen4_exp-nvfp4-tiny</c> (made by
/// <c>make_qwen4_exp_nvfp4_tiny.py</c>) carries transformers' logits for the
/// dequantised model; the arch-zoo tests match them and round-trip the export.
/// </summary>
public sealed class QuantisedCheckpointTests
{
    internal static string Fixture => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo", "qwen4_exp-nvfp4-tiny");

    internal static string Encode(TempDir dir)
    {
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Fixture, containerDir, "nvfp4-tiny",
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        return containerDir;
    }

    [Fact]
    public void Each_Format_Is_Kept_As_Stored_And_Read_Through_Its_Scales()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        const string stack = "target.decoder_stack";

        // NVFP4: U8 [32, 32] in the checkpoint, FP4 [32, 64] in the container.
        Assert.Equal(Dtype.FP4, store.DtypeOf(stack, "3.mlp.experts.0.gate_proj.weight"));
        Assert.Equal(new long[] { 32, 64 }, store.ShapeOf(stack, "3.mlp.experts.0.gate_proj.weight"));
        var expert = store.ResolveScaled(stack, "3.mlp.experts.0.gate_proj.weight")!;
        Assert.True(expert.IsNvfp4);
        Assert.Equal((1, 16), (expert.BlockRows, expert.BlockCols));
        Assert.NotEqual(1f, expert.Global);

        // MXFP8: E4M3 elements, E8M0 exponents per 32 columns.
        var q = store.ResolveScaled(stack, "3.self_attn.q_proj.weight")!;
        Assert.Equal(Dtype.F8_E4M3, q.Element);
        Assert.Equal((1, 32), (q.BlockRows, q.BlockCols));
        Assert.True(q.IsMxfp8);
        Assert.All(q.Scales, s => Assert.Equal(s, MathF.Pow(2, MathF.Round(MathF.Log2(s)))));

        // FP8 n-gram shards share the table's one scale.
        var shard = store.ResolveScaled(stack, "1.ple.ple_embedding.ngram_embedding.shard_2.weight")!;
        Assert.Equal((1, 1), (shard.ScaleRows, shard.ScaleCols));

        // Plain tensors are not scaled; widening a scaled one decodes it.
        Assert.False(store.IsScaled(stack, "3.mlp.gate.weight"));
        var widened = store.ResolveWidened(stack, "3.mlp.experts.0.gate_proj.weight").Values;
        Assert.Equal(expert.Dequant(), widened);
    }

    /// <summary>Prefill half, decode the rest; every step's logits.</summary>
    internal static List<float[]> Run(string containerDir, int[] ids, WeightWorkingSet mode, Action<WeightLoader>? setup = null,
        Action<WeightLoader>? inspect = null)
    {
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var session = new DecodeSession(plan, store, workingSet: mode);
        setup?.Invoke(session.Runtime.Weights);
        int split = ids.Length / 2;
        var logits = new List<float[]> { session.Prefill(ids[..split]).FirstRow().ToArray() };
        for (int i = split; i < ids.Length; i++)
        {
            logits.Add(session.Step(ids[i]).FirstRow().ToArray());
        }
        inspect?.Invoke(session.Runtime.Weights);
        return logits;
    }

    [Fact]
    public void Lean_Holds_Quantised_Weights_At_Their_Stored_Size_And_Matches_Resident()
    {
        var (ids, _) = ArchZooTests.Golden("qwen4_exp-nvfp4-tiny");
        using var dir = new TempDir();
        string containerDir = Encode(dir);
        string previous = Environment.GetEnvironmentVariable("AMQL_GPU") ?? string.Empty;
        try
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", "0");
            CudaShim.Reset();
            var resident = Run(containerDir, ids, WeightWorkingSet.ResidentF32);
            var streamed = Run(containerDir, ids, WeightWorkingSet.Lean, w => w.LeanCacheBytes = 0);
            long expertBytes = 0;
            var cached = Run(containerDir, ids, WeightWorkingSet.Lean, w => w.LeanCacheBytes = 1L << 30,
                w => expertBytes = w.Lean.ExpertBytes);
            for (int s = 0; s < resident.Count; s++)
            {
                // Decoding a row block into the ordinary kernel is the widened product exactly.
                Assert.Equal(resident[s], streamed[s]);
                Assert.Equal(resident[s], cached[s]);
            }
            // 4 experts x 4 layers x 3 matrices of 2048 elements, held as NVFP4:
            // 0.5 bytes an element plus one f32 scale per 16, not 4 bytes.
            Assert.InRange(expertBytes, 1, 48 * 2048 * 3 / 4);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", previous);
            CudaShim.Reset();
        }
    }

    [Fact]
    public void Mxfp8_Codec_Keeps_Each_Element_Within_E4M3_Precision()
    {
        var rng = new Random(5);
        const int rows = 16, cols = 128;
        var values = Enumerable.Range(0, rows * cols).Select(_ => (float)(rng.NextDouble() * 2 - 1) * 1e-3f).ToArray();
        var packed = Mxfp8.Quantize(values, rows, cols);
        Assert.True(packed.IsMxfp8);
        var back = packed.Dequant();
        for (int i = 0; i < values.Length; i++)
        {
            float scale = packed.Scales[i / 32];
            // E4M3 keeps 3 mantissa bits (relative step 1/8); below its smallest
            // normal the step is the subnormal 2^-9, times the block scale.
            float allowed = MathF.Max(MathF.Abs(values[i]) / 16f, scale * MathF.Pow(2, -10));
            Assert.True(MathF.Abs(back[i] - values[i]) <= allowed * 1.0001f, $"element {i}: {values[i]} came back as {back[i]}");
        }
    }

    [Fact]
    public void Nvfp4_Codec_Keeps_Each_Element_Within_Half_A_Grid_Step()
    {
        var rng = new Random(3);
        const int rows = 24, cols = 96;
        var values = Enumerable.Range(0, rows * cols).Select(_ => (float)(rng.NextDouble() * 2 - 1) * (rng.Next(4) + 1)).ToArray();
        var packed = Nvfp4.Quantize(values, rows, cols);
        Assert.True(packed.IsNvfp4);
        var back = packed.Dequant();
        for (int i = 0; i < values.Length; i++)
        {
            float step = packed.Scales[i / 16] * packed.Global;
            // The E2M1 grid's widest gap is 2 (between 4 and 6), so half a gap is one step.
            Assert.True(MathF.Abs(back[i] - values[i]) <= step * 1.0001f + 1e-7f,
                $"element {i}: {values[i]} came back as {back[i]} (scale step {step})");
        }
    }
}

/// <summary>The NVFP4 and MXFP8 device paths: weights resident on the GPU as stored, products from them.</summary>
[Collection("gpu")]
public sealed class Nvfp4DeviceTests
{
    private static bool GpuAvailable()
    {
        CudaShim.Reset();
        return CudaShim.Enabled;
    }

    [Theory]
    [InlineData(37, 48, 1)]      // decode GEMV, rows not a multiple of the 8 a block covers
    [InlineData(300, 2560, 3)]   // Flash-Next's hidden width
    [InlineData(129, 640, 8)]    // the GEMV's largest batch
    [InlineData(257, 640, 17)]   // a GEMV in three passes
    [InlineData(257, 640, 45)]   // prefill, odd shapes
    [InlineData(640, 2560, 32)]  // prefill: an expert's gate/up
    [InlineData(2560, 640, 128)] // prefill: an expert's down
    public void Device_Product_Matches_The_Host(int n, int k, int m)
    {
        if (!GpuAvailable())
        {
            return;
        }
        var rng = new Random(n * 31 + k);
        var w = Nvfp4.Quantize(Enumerable.Range(0, n * k).Select(_ => (float)(rng.NextDouble() * 2 - 1) * 0.05f).ToArray(), n, k);
        var x = new Tensor2D(Enumerable.Range(0, m * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(), m, k);
        var host = TensorOps.MatMulTransposedB(x, Tensor2D.FromScaled(w));

        var device = CudaShim.UploadNvfp4(w);
        Assert.NotEqual(IntPtr.Zero, device);
        try
        {
            Assert.True(CudaShim.TryGemmNvfp4(x.Data, m, k, device, n, w.Global, out var result));
            float[] want = host.Data;
            double tolerance;
            if (m <= 32)
            {
                tolerance = 1e-4; // the GEMV: FP32 activations, FP32 accumulation
            }
            else if (CudaShim.Nvfp4UsedTensorCores(m, n, k))
            {
                // W4A4 on the tensor cores: the reference quantises the
                // activations with the same NVFP4 recipe, then multiplies.
                var quantisedX = Nvfp4.Quantize(x.Data, m, k).Dequant();
                want = TensorOps.MatMulTransposedB(new Tensor2D(quantisedX, m, k), Tensor2D.FromScaled(w)).Data;
                tolerance = 2e-3; // rounding ties may fall either way
            }
            else
            {
                tolerance = 5e-3; // FP16 activations
            }
            double scale = want.Max(v => Math.Abs(v));
            for (int i = 0; i < want.Length; i++)
            {
                Assert.True(Math.Abs(result![i] - want[i]) <= tolerance * scale,
                    $"[{i / n}, {i % n}]: device {result[i]} vs host {want[i]}");
            }
        }
        finally
        {
            CudaShim.FreeNvfp4(device, n, k);
        }
    }

    [Theory]
    [InlineData(40, false)] // prefill-sized expert batches, FP16 tiles
    [InlineData(40, true)]  // ... on the FP4 tensor cores (W4A4)
    [InlineData(1, true)]   // decode: the GEMV
    public void Fused_Ffn_Matches_The_Host_And_Moves_Activations_Once(int rows, bool tensorCores)
    {
        if (!GpuAvailable())
        {
            return;
        }
        const int hidden = 256, inter = 64, experts = 8, topK = 2;
        var rng = new Random(rows + (tensorCores ? 1 : 0));
        float[] Values(int count, float scale) =>
            Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();
        var gates = Enumerable.Range(0, experts).Select(_ => Nvfp4.Quantize(Values(inter * hidden, 0.1f), inter, hidden)).ToArray();
        var ups = Enumerable.Range(0, experts).Select(_ => Nvfp4.Quantize(Values(inter * hidden, 0.1f), inter, hidden)).ToArray();
        var downs = Enumerable.Range(0, experts).Select(_ => Mxfp8.Quantize(Values(hidden * inter, 0.1f), hidden, inter)).ToArray();
        var router = new Tensor2D(Values(experts * hidden, 0.2f), experts, hidden);
        var x = new Tensor2D(Values(rows * hidden, 1f), rows, hidden);

        var host = FfnKernel.Routed(x, router, experts, hidden,
            e => (Tensor2D.FromScaled(gates[e]), Tensor2D.FromScaled(ups[e]), Tensor2D.FromScaled(downs[e])),
            topK, ExpertRoutingPolicy.NormalisedOverSelected, Activation.Silu);

        var uploads = new List<(IntPtr Pointer, int Rows, int Cols, bool Fp4)>();
        Tensor2D Fp4(ScaledMatrix w)
        {
            var p = CudaShim.UploadNvfp4(w);
            uploads.Add((p, w.Rows, w.Cols, true));
            return Tensor2D.OnDeviceNvfp4(w.Rows, w.Cols, p, w.Global, () => Tensor2D.FromScaled(w));
        }
        Tensor2D Fp8(ScaledMatrix w)
        {
            var p = CudaShim.UploadMxfp8(w);
            uploads.Add((p, w.Rows, w.Cols, false));
            return Tensor2D.OnDeviceMxfp8(w.Rows, w.Cols, p, () => Tensor2D.FromScaled(w));
        }
        var held = Enumerable.Range(0, experts).Select(e => (Fp4(gates[e]), Fp4(ups[e]), Fp8(downs[e]))).ToArray();
        CudaShim.SetFp4Prefill(tensorCores);
        try
        {
            long calls = CudaShim.FfnCalls;
            var device = FfnKernel.Routed(x, router, experts, hidden, e => held[e],
                topK, ExpertRoutingPolicy.NormalisedOverSelected, Activation.Silu);
            Assert.Equal(calls + 1, CudaShim.FfnCalls); // every expert in one call

            // W4A4 quantises the expert inputs and the hidden units; the rest
            // differs from the host by FP16 activations at most.
            double tolerance = rows > 8 && tensorCores ? 0.15 : 1e-2;
            double scale = host.Data.Max(v => Math.Abs(v));
            for (int i = 0; i < host.Data.Length; i++)
            {
                Assert.True(Math.Abs(device.Data[i] - host.Data[i]) <= tolerance * scale,
                    $"[{i / hidden}, {i % hidden}]: device {device.Data[i]} vs host {host.Data[i]}");
            }

            // A dense FFN over the same weights is one call too.
            var dense = FfnKernel.Dense(x, held[0].Item1, held[0].Item2, held[0].Item3, Activation.Silu, true);
            var denseHost = FfnKernel.Dense(x, Tensor2D.FromScaled(gates[0]), Tensor2D.FromScaled(ups[0]),
                Tensor2D.FromScaled(downs[0]), Activation.Silu, true);
            Assert.Equal(calls + 2, CudaShim.FfnCalls);
            double denseScale = denseHost.Data.Max(v => Math.Abs(v));
            for (int i = 0; i < denseHost.Data.Length; i++)
            {
                Assert.True(Math.Abs(dense.Data[i] - denseHost.Data[i]) <= tolerance * denseScale);
            }
        }
        finally
        {
            CudaShim.SetFp4Prefill(true);
            foreach (var (pointer, r, c, fp4) in uploads)
            {
                if (fp4) CudaShim.FreeNvfp4(pointer, r, c); else CudaShim.FreeMxfp8(pointer, r, c);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(40)]
    public void Shared_Input_Products_Share_One_Upload(int m)
    {
        if (!GpuAvailable())
        {
            return;
        }
        const int k = 256;
        var rng = new Random(m);
        float[] Values(int count, float scale) =>
            Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();
        // As in a GDN layer: two large projections on the device (NVFP4,
        // MXFP8) and a small one left on the host.
        var big4 = Nvfp4.Quantize(Values(300 * k, 0.05f), 300, k);
        var big8 = Mxfp8.Quantize(Values(128 * k, 0.05f), 128, k);
        var small = Mxfp8.Quantize(Values(48 * k, 0.05f), 48, k);
        var x = new Tensor2D(Values(m * k, 1f), m, k);
        var p4 = CudaShim.UploadNvfp4(big4);
        var p8 = CudaShim.UploadMxfp8(big8);
        CudaShim.SetFp4Prefill(false);
        try
        {
            var weights = new[]
            {
                Tensor2D.OnDeviceNvfp4(300, k, p4, big4.Global, () => Tensor2D.FromScaled(big4)),
                Tensor2D.FromScaled(small),
                Tensor2D.OnDeviceMxfp8(128, k, p8, () => Tensor2D.FromScaled(big8)),
            };
            long calls = CudaShim.MultiCalls;
            var products = TensorOps.MatMulTransposedBMany(x, weights);
            Assert.Equal(calls + 1, CudaShim.MultiCalls);
            var hosts = new[] { big4, small, big8 }.Select(w => TensorOps.MatMulTransposedB(x, Tensor2D.FromScaled(w))).ToArray();
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal((m, hosts[i].Cols), (products[i].Rows, products[i].Cols));
                double scale = hosts[i].Data.Max(v => Math.Abs(v));
                double tolerance = i == 1 ? 0 : m <= 32 ? 1e-4 : 5e-3;
                for (int j = 0; j < hosts[i].Data.Length; j++)
                {
                    Assert.True(Math.Abs(products[i].Data[j] - hosts[i].Data[j]) <= tolerance * scale,
                        $"product {i} [{j}]: {products[i].Data[j]} vs {hosts[i].Data[j]}");
                }
            }
        }
        finally
        {
            CudaShim.SetFp4Prefill(true);
            CudaShim.FreeNvfp4(p4, 300, k);
            CudaShim.FreeMxfp8(p8, 128, k);
        }
    }

    [Fact]
    public void Prefill_Runs_On_The_Fp4_Tensor_Cores_At_Flash_Next_Shapes()
    {
        if (!GpuAvailable())
        {
            return;
        }
        foreach (var (n, k) in new[] { (640, 2560), (2560, 640) })
        {
            var rng = new Random(n);
            var w = Nvfp4.Quantize(Enumerable.Range(0, n * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(), n, k);
            var x = Enumerable.Range(0, 64 * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            var device = CudaShim.UploadNvfp4(w);
            try
            {
                Assert.True(CudaShim.TryGemmNvfp4(x, 64, k, device, n, w.Global, out _));
                Assert.True(CudaShim.Nvfp4UsedTensorCores(64, n, k), $"[{n} x {k}] fell back to FP16 tiles");
            }
            finally
            {
                CudaShim.FreeNvfp4(device, n, k);
            }
        }
    }

    [Theory]
    [InlineData(37, 64, 1)]
    [InlineData(300, 2560, 4)]   // Flash-Next's hidden width
    [InlineData(129, 6144, 8)]   // its o_proj / out_proj input width
    [InlineData(257, 640, 17)]   // a GEMV in three passes
    [InlineData(257, 640, 45)]   // prefill: tiles decoded to FP16, cuBLAS
    public void Mxfp8_Device_Product_Matches_The_Host(int n, int k, int m)
    {
        if (!GpuAvailable())
        {
            return;
        }
        var rng = new Random(n * 17 + k);
        // Rows of different magnitudes, so the per-block exponents vary.
        var values = Enumerable.Range(0, n * k).Select(i => (float)(rng.NextDouble() * 2 - 1) * MathF.Pow(2, (i / k) % 7 - 6)).ToArray();
        var w = Mxfp8.Quantize(values, n, k);
        Assert.True(w.IsMxfp8);
        var x = new Tensor2D(Enumerable.Range(0, m * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(), m, k);
        var host = TensorOps.MatMulTransposedB(x, Tensor2D.FromScaled(w));

        var device = CudaShim.UploadMxfp8(w);
        Assert.NotEqual(IntPtr.Zero, device);
        try
        {
            Assert.True(CudaShim.TryGemmMxfp8(x.Data, m, k, device, n, out var result));
            double tolerance = m <= 8 ? 1e-4 : 5e-3;
            for (int r = 0; r < m; r++)
            {
                for (int j = 0; j < n; j++)
                {
                    // Per output row of the weight: its magnitude sets the scale.
                    double scale = Enumerable.Range(0, m).Max(i => Math.Abs(host.Data[i * n + j])) + 1e-30;
                    double diff = Math.Abs(result![r * n + j] - host.Data[r * n + j]);
                    Assert.True(diff <= tolerance * scale, $"[{r}, {j}]: device {result[r * n + j]} vs host {host.Data[r * n + j]}");
                }
            }
        }
        finally
        {
            CudaShim.FreeMxfp8(device, n, k);
        }
    }

    [Fact]
    public void Lean_Runs_Nvfp4_Experts_On_The_Gpu_And_Rotates_Them()
    {
        if (!GpuAvailable())
        {
            return;
        }
        var (ids, golden) = ArchZooTests.Golden("qwen4_exp-nvfp4-tiny");
        using var dir = new TempDir();
        string containerDir = QuantisedCheckpointTests.Encode(dir);

        CudaShim.SetFp4Prefill(false);
        try
        {
            RunBoth(containerDir, ids, golden);
        }
        finally
        {
            CudaShim.SetFp4Prefill(true);
        }
    }

    private static void RunBoth(string containerDir, int[] ids, float[][] golden)
    {
        int roomy = 0, tight = 0;
        long deviceBytes = 0;
        var all = QuantisedCheckpointTests.Run(containerDir, ids, WeightWorkingSet.Lean,
            w => { w.LeanNvfp4MinimumElements = 1; w.LeanDeviceBytes = 64L << 20; },
            w => (roomy, deviceBytes) = (w.Lean.DeviceExperts, w.Lean.DeviceBytes));
        // Every expert matrix (gate, up [32, 64]; down [64, 32]) takes the same bytes.
        long perExpertMatrix = CudaShim.Nvfp4DeviceBytes(32, 64);
        long dense = deviceBytes - roomy * perExpertMatrix;
        Assert.True(dense > 0, "the MXFP8 projections and NVFP4 shared experts go to the GPU too");
        // Room for a few experts only: they are evicted and uploaded again as
        // routing moves, and anyone holding an evicted one falls back to the host.
        var rotating = QuantisedCheckpointTests.Run(containerDir, ids, WeightWorkingSet.Lean,
            // The dense weights (never displaced) and room for 3 expert matrices.
            w => { w.LeanNvfp4MinimumElements = 1; w.LeanDeviceBytes = dense + 3 * perExpertMatrix; },
            w => tight = w.Lean.DeviceExperts);

        Assert.True(roomy > tight && tight > 0, $"experts on the device: {roomy} with room, {tight} without");
        // The golden is the dequantised model, W4A16: the runs above took the
        // FP16-tile prefill (the caller turns the tensor cores off); W4A4 is
        // checked against its own host reference in Device_Product_Matches_The_Host.
        int split = ids.Length / 2;
        foreach (var run in new[] { all, rotating })
        {
            // Step 0 is the prefill's last row; the rest are decode steps.
            for (int s = 0; s < run.Count; s++)
            {
                var want = golden[split - 1 + s];
                double worst = want.Zip(run[s], (a, b) => Math.Abs(a - b)).Max();
                Assert.True(worst < 2e-2, $"step {s}: worst logit diff {worst:G4}");
            }
        }
    }
}
