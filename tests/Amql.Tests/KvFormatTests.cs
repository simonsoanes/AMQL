using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Server;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// NVFP4 KV caches: every path writes and reads the same rows, so the
/// guarantees f32 caches give still hold among the paths (batched steps equal
/// each sequence alone, a rewound or restored sequence carries on exactly),
/// rows take ~7x less memory, and the logits stay close to f32 caches'.
/// Skips without a CUDA device.
/// </summary>
[Collection("gpu")]
public class KvFormatTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    private sealed class Fp4Kv : IDisposable
    {
        private readonly string _saved = GpuEngine.RequestedKvFormat;
        public Fp4Kv(string format) => GpuEngine.RequestedKvFormat = format;
        public void Dispose() => GpuEngine.RequestedKvFormat = _saved;
    }

    private static (DecodeSession Host, GpuEngine Engine, int[] Ids, ComponentOpPlan Plan, IDisposable Owned) Open(TempDir dir, string family, string format)
    {
        string containerDir = Path.Combine(dir.Path, "c-" + format);
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        var container = Vindex3Container.Open(containerDir);
        var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        using (new Fp4Kv(format))
        {
            var host = new DecodeSession(plan, store);
            host.Prefill(new[] { ids[0] });
            var engine = host.DeviceEngine()!;
            engine.ExactPrefill = true;
            return (host, engine, ids, plan, new Owned(host, store, container));
        }
    }

    private sealed class Owned(params IDisposable[] items) : IDisposable
    {
        public void Dispose()
        {
            foreach (var i in items)
            {
                i.Dispose();
            }
        }
    }

    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    public void Nvfp4_Rows_Are_Small_And_Every_Path_Agrees(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        var (_, engine, ids, plan, owned) = Open(dir, family, "nvfp4");
        using var _o = owned;
        Assert.Equal("nvfp4", engine.KvFormat);
        var prompt = Enumerable.Range(0, 700).Select(i => ids[i % ids.Length]).ToArray();
        int layer = Enumerable.Range(0, 64).First(i => engine.KvRowBytes(i) > 0);

        // Rewind and restore carry on exactly, as with f32 rows.
        var seq = engine.NewSequence();
        engine.Prefill(seq, prompt.AsSpan(0, 600));
        engine.Mark(seq);
        var first = engine.Prefill(seq, prompt.AsSpan(600));
        var step = engine.Step(new[] { seq }, new[] { Sampler.ArgMax(first) })[0];
        engine.Rewind(seq);
        Assert.Equal(first, engine.Prefill(seq, prompt.AsSpan(600)));
        Assert.Equal(step, engine.Step(new[] { seq }, new[] { Sampler.ArgMax(first) })[0]);

        var state = new ChunkedBuffer();
        engine.SaveMark(seq, state);
        Assert.Equal(engine.MarkBytes(seq), state.Length);
        state.Position = 0;
        var restored = engine.RestoreMark(state);
        Assert.Equal(first, engine.Prefill(restored, prompt.AsSpan(600)));

        // A batch step equals each sequence's step alone (row invariance aside, within f32 rounding).
        var other = engine.NewSequence();
        var o = engine.Prefill(other, prompt.AsSpan(0, 333));
        var alone = engine.Step(new[] { other }, new[] { Sampler.ArgMax(o) })[0];
        engine.Rewind(restored);
        var r0 = engine.Prefill(restored, prompt.AsSpan(600));
        var other2 = engine.NewSequence();
        engine.Prefill(other2, prompt.AsSpan(0, 333));
        var both = engine.Step(new[] { restored, other2 }, new[] { Sampler.ArgMax(r0), Sampler.ArgMax(o) });
        AssertClose(alone, both[1], 1e-4);
        AssertClose(step, both[0], 1e-4);

        // The row: codes, block scales, an f32 per head, padded to 16 bytes (592 on Qwen 3.5 4B's 4 × 256).
        var a = plan.Layers[layer].Attention!;
        Assert.Equal(RowBytes(a.KvDim, a.NumKvHeads), engine.KvRowBytes(layer));
        Assert.Equal(592, RowBytes(1024, 4));
    }

    private static long RowBytes(int kvDim, int heads) => ((((kvDim / 2 + kvDim / 16) + 3) & ~3) + 4 * heads + 15) & ~15;

    /// <summary>Decodes an NVFP4 row as the kernels do.</summary>
    private static float[] Decode(ReadOnlySpan<byte> row, int kvDim, int headDim, out float[] blockScale)
    {
        int headOff = ((kvDim / 2 + kvDim / 16) + 3) & ~3;
        var values = new float[kvDim];
        blockScale = new float[kvDim / 16];
        float[] e2m1 = [0, 0.5f, 1, 1.5f, 2, 3, 4, 6];
        for (int c = 0; c < kvDim; c++)
        {
            int code = (row[c >> 1] >> ((c & 1) * 4)) & 15;
            float head = BitConverter.ToSingle(row.Slice(headOff + 4 * (c / headDim), 4));
            float s = E4m3(row[kvDim / 2 + (c >> 4)]) * head;
            blockScale[c >> 4] = s;
            values[c] = ((code & 8) != 0 ? -1 : 1) * e2m1[code & 7] * s;
        }
        return values;
    }

    private static float E4m3(byte b)
    {
        int sign = b >> 7, exp = (b >> 3) & 15, man = b & 7;
        float v = exp == 0 ? man / 8f * MathF.Pow(2, -6) : (1 + man / 8f) * MathF.Pow(2, exp - 7);
        return sign != 0 ? -v : v;
    }

    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    public void The_Encoder_Writes_Each_Value_Within_Its_Block_Step(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        var (_, f32, ids, plan, o1) = Open(dir, family, "f32");
        using var _1 = o1;
        var (_, fp4, _, _, o2) = Open(dir, family, "nvfp4");
        using var _2 = o2;
        var prompt = Enumerable.Range(0, 200).Select(i => ids[(i * 3 + 1) % ids.Length]).ToArray();
        // The first softmax layer's keys and values come from layers that read no cache,
        // so both engines compute them alike; the NVFP4 engine then rounds them.
        int layer = Enumerable.Range(0, plan.Layers.Count).First(i => plan.Layers[i].Attention is not null);
        var a = plan.Layers[layer].Attention!;
        var s1 = f32.NewSequence();
        var s2 = fp4.NewSequence();
        f32.Prefill(s1, prompt);
        fp4.Prefill(s2, prompt);
        double errSq = 0, refSq = 0;
        foreach (bool values in new[] { false, true })
        {
            var exact = f32.DownloadKv(s1, layer, values, prompt.Length);
            var rows = fp4.DownloadKvRows(s2, layer, values, prompt.Length);
            long rb = fp4.KvRowBytes(layer);
            for (int r = 0; r < prompt.Length; r++)
            {
                var got = Decode(rows.AsSpan((int)(r * rb), (int)rb), a.KvDim, a.HeadDim, out var step);
                for (int c = 0; c < a.KvDim; c++)
                {
                    float x = exact[r * a.KvDim + c];
                    // E2M1's widest gap is 2 (4 to 6): rounding moves a value at most one step of its block's scale,
                    // with f32 rounding of the activations between the engines on top.
                    Assert.True(Math.Abs(x - got[c]) <= step[c >> 4] * 1.0001 + 1e-3 * Math.Abs(x) + 1e-6,
                        $"{(values ? "V" : "K")} row {r} col {c}: {x} stored as {got[c]} (block step {step[c >> 4]})");
                    errSq += (x - got[c]) * (double)(x - got[c]);
                    refSq += x * (double)x;
                }
            }
        }
        double rel = Math.Sqrt(errSq / refSq);
        Assert.InRange(rel, 0.01, 0.2);   // NVFP4's ~9% on real Qwen 3.5 4B rows
    }

    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    public void Nvfp4_Logits_Stay_Close_To_F32(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        var (_, f32, ids, _, o1) = Open(dir, family, "f32");
        using var _1 = o1;
        var (_, fp4, _, _, o2) = Open(dir, family, "nvfp4");
        using var _2 = o2;
        Assert.Equal("f32", f32.KvFormat);
        Assert.Equal("nvfp4", fp4.KvFormat);
        var prompt = Enumerable.Range(0, 300).Select(i => ids[(i * 5 + 1) % ids.Length]).ToArray();
        var a = f32.NewSequence();
        var b = fp4.NewSequence();
        var la = f32.Prefill(a, prompt);
        var lb = fp4.Prefill(b, prompt);
        double worst = 0;
        for (int i = 0; i < 8; i++)
        {
            worst = Math.Max(worst, Rel(la, lb));
            int t = Sampler.ArgMax(la);
            la = f32.Step(new[] { a }, new[] { t })[0];
            lb = fp4.Step(new[] { b }, new[] { t })[0];
        }
        Assert.NotEqual(0, worst);       // the rows really are quantised
        // Random toy weights with 16-wide heads amplify 4-bit error; this only rules out garbage.
        // Quality is measured on real models (quantise compare <dir> <dir>@kv=nvfp4).
        Assert.True(worst < 0.5, $"NVFP4 KV moved logits by {worst:P1}");
    }

    [Theory]
    [InlineData("qwen3", false)]
    [InlineData("qwen3_5-gdn", false)]
    [InlineData("qwen3", true)]
    [InlineData("qwen3_5-gdn", true)]
    public void Exact_Rows_Stay_Exact_And_Every_Path_Still_Agrees(string family, bool f16)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        var (_, f32, ids, _, o1) = Open(dir, family, "f32");
        using var _1 = o1;
        var (_, fp4, _, plan, o2) = Open(dir, family, "nvfp4");
        using var _2 = o2;
        var prompt = Enumerable.Range(0, 700).Select(i => ids[(i * 3 + 2) % ids.Length]).ToArray();
        const int exact = 300;

        // Before the boundary everything reads only exact rows: with f32 the logits are an f32 cache's,
        // bit for bit; with FP16 within its rounding.
        var a = f32.NewSequence();
        var b = fp4.NewSequence();
        fp4.SetExactRows(b, exact, f16);
        Assert.Equal(exact, b.ExactRows);
        Assert.Equal(f16, b.ExactF16);
        var la = f32.Prefill(a, prompt.AsSpan(0, exact));
        var lb = fp4.Prefill(b, prompt.AsSpan(0, exact));
        if (f16)
        {
            Assert.InRange(Rel(la, lb), 1e-9, 1e-2);
        }
        else
        {
            Assert.Equal(la, lb);
        }
        // The first softmax layer's exact rows are the f32 values, or those values rounded to FP16.
        int layer = Enumerable.Range(0, plan.Layers.Count).First(i => plan.Layers[i].Attention is not null);
        int kvDim = plan.Layers[layer].Attention!.KvDim;
        var want = f32.DownloadKv(a, layer, false, exact);
        var got = fp4.DownloadKvRows(b, layer, false, exact);
        for (int i = 0; i < exact * kvDim; i++)
        {
            float stored = f16 ? (float)BitConverter.ToHalf(got, 2 * i) : BitConverter.ToSingle(got, 4 * i);
            Assert.Equal(f16 ? (float)(Half)want[i] : want[i], stored);
        }
        // After it the rows are NVFP4: the logits move.
        Assert.NotEqual(f32.Prefill(a, prompt.AsSpan(exact)), fp4.Prefill(b, prompt.AsSpan(exact)));
        long exactRow = (f16 ? 2L : 4L) * kvDim;
        Assert.Equal(exact * exactRow + (prompt.Length - exact) * fp4.KvRowBytes(layer),
            fp4.DownloadKvRows(b, layer, false, prompt.Length).LongLength);

        // Rewind, a decode step, save and restore, and a batch, with the boundary inside the mark.
        var seq = fp4.NewSequence();
        fp4.SetExactRows(seq, exact, f16);
        fp4.Prefill(seq, prompt.AsSpan(0, 600));
        fp4.Mark(seq);
        var first = fp4.Prefill(seq, prompt.AsSpan(600));
        var step = fp4.Step(new[] { seq }, new[] { Sampler.ArgMax(first) })[0];
        fp4.Rewind(seq);
        Assert.Equal(first, fp4.Prefill(seq, prompt.AsSpan(600)));
        Assert.Equal(step, fp4.Step(new[] { seq }, new[] { Sampler.ArgMax(first) })[0]);
        var state = new ChunkedBuffer();
        fp4.SaveMark(seq, state);
        Assert.Equal(fp4.MarkBytes(seq), state.Length);
        state.Position = 0;
        var restored = fp4.RestoreMark(state);
        Assert.Equal(exact, restored.ExactRows);
        Assert.Equal(f16, restored.ExactF16);
        Assert.Equal(first, fp4.Prefill(restored, prompt.AsSpan(600)));
        var other = fp4.NewSequence();
        var o = fp4.Prefill(other, prompt.AsSpan(0, 333));
        var alone = fp4.Step(new[] { other }, new[] { Sampler.ArgMax(o) })[0];
        var other2 = fp4.NewSequence();
        fp4.Prefill(other2, prompt.AsSpan(0, 333));
        var both = fp4.Step(new[] { restored, other2 }, new[] { Sampler.ArgMax(first), Sampler.ArgMax(o) });
        AssertClose(step, both[0], 1e-4);
        AssertClose(alone, both[1], 1e-4);

        // Reset keeps the boundary; it changes only while the sequence is empty.
        fp4.Reset(seq);
        Assert.Equal(exact, seq.ExactRows);
        fp4.Prefill(seq, prompt.AsSpan(0, 10));
        Assert.Throws<InvalidOperationException>(() => fp4.SetExactRows(seq, 5));
        // An f32 cache has no boundary to keep.
        var plain = f32.NewSequence();
        f32.SetExactRows(plain, exact);
        Assert.Equal(0, plain.ExactRows);
    }

    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    public void Fp16_Rows_Are_Rounded_F32_And_Every_Path_Agrees(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        var (_, f32, ids, _, o1) = Open(dir, family, "f32");
        using var _1 = o1;
        var (_, fp16, _, plan, o2) = Open(dir, family, "fp16");
        using var _2 = o2;
        Assert.Equal("fp16", fp16.KvFormat);
        var prompt = Enumerable.Range(0, 700).Select(i => ids[(i * 3 + 2) % ids.Length]).ToArray();
        int layer = Enumerable.Range(0, plan.Layers.Count).First(i => plan.Layers[i].Attention is not null);
        int kvDim = plan.Layers[layer].Attention!.KvDim;
        Assert.Equal(2L * kvDim, fp16.KvRowBytes(layer));

        // Every row is the f32 value rounded to FP16; the logits stay within FP16's rounding of f32's.
        var a = f32.NewSequence();
        var b = fp16.NewSequence();
        fp16.SetExactRows(b, 10);   // no effect: every row is FP16 already
        var la = f32.Prefill(a, prompt.AsSpan(0, 300));
        var lb = fp16.Prefill(b, prompt.AsSpan(0, 300));
        Assert.InRange(Rel(la, lb), 1e-9, 1e-2);
        var want = f32.DownloadKv(a, layer, true, 300);
        var got = fp16.DownloadKvRows(b, layer, true, 300);
        Assert.Equal(300L * 2 * kvDim, got.LongLength);
        for (int i = 0; i < want.Length; i++)
        {
            Assert.Equal((float)(Half)want[i], (float)BitConverter.ToHalf(got, 2 * i));
        }

        // Rewind, a step, save and restore, and a batch.
        var seq = fp16.NewSequence();
        fp16.Prefill(seq, prompt.AsSpan(0, 600));
        fp16.Mark(seq);
        var first = fp16.Prefill(seq, prompt.AsSpan(600));
        var step = fp16.Step(new[] { seq }, new[] { Sampler.ArgMax(first) })[0];
        fp16.Rewind(seq);
        Assert.Equal(first, fp16.Prefill(seq, prompt.AsSpan(600)));
        Assert.Equal(step, fp16.Step(new[] { seq }, new[] { Sampler.ArgMax(first) })[0]);
        var state = new ChunkedBuffer();
        fp16.SaveMark(seq, state);
        Assert.Equal(fp16.MarkBytes(seq), state.Length);
        state.Position = 0;
        var restored = fp16.RestoreMark(state);
        Assert.Equal(first, fp16.Prefill(restored, prompt.AsSpan(600)));
        var other = fp16.NewSequence();
        var o = fp16.Prefill(other, prompt.AsSpan(0, 333));
        var alone = fp16.Step(new[] { other }, new[] { Sampler.ArgMax(o) })[0];
        var other2 = fp16.NewSequence();
        fp16.Prefill(other2, prompt.AsSpan(0, 333));
        var both = fp16.Step(new[] { restored, other2 }, new[] { Sampler.ArgMax(first), Sampler.ArgMax(o) });
        AssertClose(step, both[0], 1e-4);
        AssertClose(alone, both[1], 1e-4);
    }

    [Fact]
    public void Auto_Keeps_Through_The_First_User_Message()
    {
        const int T = 900, U = 901, A = 902;   // <|im_start|>, "user", "assistant"
        int[] prompt = [T, 5, 6, 7, 8, T, U, 9, 10, 11, T, A, 12, 13, T, U, 14, T, A];
        Assert.Equal(10, PromptMark.Exact(prompt, -1, T, U));     // the turn after the first user message
        Assert.Equal(4, PromptMark.Exact(prompt, 4, T, U));       // a set count
        Assert.Equal(0, PromptMark.Exact(prompt, 0, T, U));
        Assert.Equal(prompt.Length, PromptMark.Exact(prompt, 999, T, U));
        Assert.Equal(0, PromptMark.Exact([T, 5, 6, T, A], -1, T, U));   // no user turn
        Assert.Equal(5, PromptMark.Exact([T, U, 7, 8, 9], -1, T, U));   // an unfinished first message
        Assert.Equal(0, PromptMark.Exact(prompt, -1, null, U));
    }

    private static double Rel(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            num += (a[i] - b[i]) * (double)(a[i] - b[i]);
            den += a[i] * (double)a[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static void AssertClose(float[] expected, float[] actual, double tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i])));
        }
        Assert.True(worst <= tol, $"worst {worst}");
    }
}
