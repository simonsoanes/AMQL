using Amql.Inference;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// The CPU kernels were rewritten for speed on the promise that every result
/// is bit-identical to the plain scalar code they replaced. These tests hold
/// them to it: each compares the fast path against a reference that is the
/// replaced code (or the scalar <see cref="TensorOps.Dot"/>), with
/// <c>Assert.Equal</c> on the bit patterns — no tolerance. Shapes are chosen
/// to hit every tail: widths that are not a whole number of vectors (or of
/// BF16 vector pairs), row counts that are not multiples of four, batches
/// large enough to take the parallel and tiled paths.
/// </summary>
public class KernelExactnessTests
{
    private static float[] Values(int count, int seed, float scale = 1f)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1) * scale;
        }
        return values;
    }

    /// <summary>BF16 bits and the f32 values they widen to.</summary>
    private static (ushort[] Bits, float[] Widened) Bf16Values(int count, int seed)
    {
        var f = Values(count, seed);
        var bits = new ushort[count];
        var widened = new float[count];
        for (int i = 0; i < count; i++)
        {
            bits[i] = (ushort)(BitConverter.SingleToInt32Bits(f[i]) >>> 16);
            widened[i] = BitConverter.Int32BitsToSingle(bits[i] << 16);
        }
        return (bits, widened);
    }

    private static void AssertBitEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                Assert.Fail($"{what}[{i}]: {actual[i]:R} is not bit-identical to {expected[i]:R}");
            }
        }
    }

    /// <summary>x @ Wᵀ by one <see cref="TensorOps.Dot"/> per output — the
    /// definition the GEMV/GEMM kernels promise to reproduce exactly.</summary>
    private static float[] ReferenceMatMul(float[] x, int m, int k, float[] w, int n)
    {
        var result = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i * n + j] = TensorOps.Dot(x.AsSpan(i * k, k), w.AsSpan(j * k, k));
            }
        }
        return result;
    }

    // k: under one vector, a vector plus a tail, an odd number of vectors (the
    // BF16 kernel's single-run path), whole BF16 pairs, a real width.
    public static TheoryData<int, int, int> Shapes => new()
    {
        { 1, 5, 7 },
        { 1, 13, 3 },
        { 1, 24, 9 },
        { 1, 1024, 3584 },
        { 3, 40, 17 },
        { 4, 64, 64 },
        { 7, 48, 130 },
        { 9, 1024, 1027 },
        { 37, 256, 515 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void MatMulTransposedB_F32_Is_Bit_Identical_To_Dot(int m, int k, int n)
    {
        var x = Values(m * k, 1);
        var w = Values(n * k, 2);
        var result = TensorOps.MatMulTransposedB(new Tensor2D(x, m, k), new Tensor2D(w, n, k));
        AssertBitEqual(ReferenceMatMul(x, m, k, w, n), result.Data, "f32");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void MatMulTransposedB_Bf16_Is_Bit_Identical_To_Widened_F32(int m, int k, int n)
    {
        var x = Values(m * k, 3);
        var (bits, widened) = Bf16Values(n * k, 4);
        var weight = Tensor2D.FromBf16(bits, n, k);
        Assert.NotNull(weight.Bf16Bits);
        var result = TensorOps.MatMulTransposedB(new Tensor2D(x, m, k), weight);
        Assert.NotNull(weight.Bf16Bits); // the kernel streamed the bits, it did not widen the tensor
        AssertBitEqual(ReferenceMatMul(x, m, k, widened, n), result.Data, "bf16");
    }

    [Fact]
    public void MatMulTransposedBMany_Matches_One_Call_Per_Weight()
    {
        foreach (int m in new[] { 1, 6, 70 })
        {
            int k = 96;
            var x = new Tensor2D(Values(m * k, 5), m, k);
            var (bits, _) = Bf16Values(300 * k, 6);
            var weights = new[]
            {
                new Tensor2D(Values(200 * k, 7), 200, k),
                Tensor2D.FromBf16(bits, 300, k),
                new Tensor2D(Values(3 * k, 8), 3, k),
            };
            var fused = TensorOps.MatMulTransposedBMany(x, weights);
            for (int i = 0; i < weights.Length; i++)
            {
                AssertBitEqual(TensorOps.MatMulTransposedB(x, weights[i]).Data, fused[i].Data, $"m={m} weight {i}");
            }
        }
    }

    [Fact]
    public void Bf16_Tensor_Widens_Exactly_And_Edits_Through_Data_Are_Seen()
    {
        int n = 33, k = 40;
        var (bits, widened) = Bf16Values(n * k, 9);
        var weight = Tensor2D.FromBf16(bits, n, k);
        var row = new float[k];
        weight.CopyRowTo(5, row);
        AssertBitEqual(widened.AsSpan(5 * k, k), row, "row");
        Assert.NotNull(weight.Bf16Bits);

        var clone = weight.Clone();
        Assert.NotNull(clone.Bf16Bits);

        AssertBitEqual(widened, weight.Data, "data");
        Assert.Null(weight.Bf16Bits); // touched: the f32 array is the tensor now

        // An in-place edit (as WeightLoader.AddDelta makes) must reach the kernel.
        weight.Data[0] += 1f;
        widened[0] += 1f;
        var x = new Tensor2D(Values(2 * k, 10), 2, k);
        AssertBitEqual(ReferenceMatMul(x.Data, 2, k, widened, n), TensorOps.MatMulTransposedB(x, weight).Data, "edited");

        // The clone was taken before the edit and still holds the original bits.
        AssertBitEqual(Bf16Values(n * k, 9).Widened, clone.Data, "clone");
    }

    [Fact]
    public void RowKvCache_Keeps_Rows_In_Order_Across_Growth()
    {
        var cache = new RowKvCache();
        int width = 6;
        for (int r = 0; r < 100; r++)
        {
            var key = Enumerable.Range(0, width).Select(c => r * 10f + c).ToArray();
            var value = key.Select(v => -v).ToArray();
            cache.Append(3, key, value);
        }
        var (keys, keyWidth, values, valueWidth, count) = cache.Buffers(3);
        Assert.Equal(100, count);
        Assert.Equal(width, keyWidth);
        Assert.Equal(width, valueWidth);
        for (int r = 0; r < count; r++)
        {
            for (int c = 0; c < width; c++)
            {
                Assert.Equal(r * 10f + c, keys[r * width + c]);
                Assert.Equal(-(r * 10f + c), values[r * width + c]);
            }
        }
        Assert.Equal(100, cache.Keys(3).Count);
        Assert.Equal(0, cache.Count(0));
        Assert.Equal(0, cache.Position);
        Assert.Throws<ArgumentException>(() => cache.Append(3, new float[width + 1], new float[width]));
        cache.Reset();
        Assert.Equal(0, cache.Count(3));
    }

    [Theory]
    [InlineData(1, 4)]
    [InlineData(2, 4)]
    [InlineData(5, 4)]
    [InlineData(40, 4)]
    [InlineData(3, 1)]
    public void ConvForwardRows_And_ConvStepFlat_Match_The_Column_Major_Conv(int t, int kernel)
    {
        int channels = 700;
        var weight = Values(channels * kernel, 11);
        var history = Values(channels * Math.Max(0, kernel - 1), 12);
        var expectState = NewConvState(channels, kernel, history);
        var actualState = NewConvState(channels, kernel, history);

        for (int round = 0; round < 3; round++)
        {
            var rows = Values(t * channels, 13 + round);
            var columns = new float[channels, t];
            for (int pos = 0; pos < t; pos++)
            {
                for (int c = 0; c < channels; c++)
                {
                    columns[c, pos] = rows[pos * channels + c];
                }
            }
            float[,] expected;
            float[] actual;
            if (t == 1)
            {
                expected = GatedDeltaKernel.ConvStep(columns, weight, kernel, silu: true, expectState);
                actual = GatedDeltaKernel.ConvStepFlat(rows, weight, kernel, silu: true, actualState);
            }
            else
            {
                expected = GatedDeltaKernel.ConvForward(columns, weight, kernel, silu: true, expectState);
                actual = GatedDeltaKernel.ConvForwardRows(rows, t, weight, kernel, silu: true, actualState);
            }
            var expectedRows = new float[t * channels];
            for (int pos = 0; pos < t; pos++)
            {
                for (int c = 0; c < channels; c++)
                {
                    expectedRows[pos * channels + c] = expected[c, pos];
                }
            }
            AssertBitEqual(expectedRows, actual, $"round {round} output");
            AssertBitEqual(Flatten(expectState.ConvHistory), Flatten(actualState.ConvHistory), $"round {round} history");
        }
    }

    private static LinearAttentionState NewConvState(int channels, int kernel, float[] history)
    {
        var state = new LinearAttentionState(1, 1, 1, channels, kernel);
        int len = Math.Max(0, kernel - 1);
        for (int c = 0; c < channels; c++)
        {
            for (int j = 0; j < len; j++)
            {
                state.ConvHistory[c, j] = history[c * len + j];
            }
        }
        return state;
    }

    private static float[] Flatten(float[,] a) => a.Cast<float>().ToArray();

    [Theory]
    [InlineData(1, 4, 32, 32)]
    [InlineData(3, 2, 16, 24)]
    [InlineData(12, 16, 32, 64)]
    [InlineData(20, 4, 16, 128)]
    public void Recurrent_Is_Bit_Identical_To_The_Scalar_Reference(int t, int heads, int kDim, int vDim)
    {
        var q = new Tensor2D(Values(t * heads * kDim, 20), t, heads * kDim);
        var k = new Tensor2D(Values(t * heads * kDim, 21), t, heads * kDim);
        var v = new Tensor2D(Values(t * heads * vDim, 22), t, heads * vDim);
        var g = new Tensor2D(Values(t * heads, 23, 0.5f), t, heads);
        var beta = new Tensor2D(Values(t * heads, 24).Select(b => 0.5f + b / 2).ToArray(), t, heads);

        var expectState = new LinearAttentionState(heads, kDim, vDim, 1, 1);
        var actualState = new LinearAttentionState(heads, kDim, vDim, 1, 1);
        // Two calls, so the second starts from a carried state.
        for (int round = 0; round < 2; round++)
        {
            var expected = ScalarRecurrent(q, k, v, g, beta, heads, kDim, vDim, expectState);
            var actual = GatedDeltaKernel.Recurrent(q, k, v, g, beta, heads, kDim, vDim, actualState);
            AssertBitEqual(expected.Data, actual.Data, $"round {round} output");
            for (int h = 0; h < heads; h++)
            {
                AssertBitEqual(Flatten(expectState.S[h]), Flatten(actualState.S[h]), $"round {round} state {h}");
            }
        }
    }

    /// <summary>The recurrence as it was before vectorisation, verbatim.</summary>
    private static Tensor2D ScalarRecurrent(Tensor2D q, Tensor2D k, Tensor2D v, Tensor2D g, Tensor2D beta,
        int heads, int kDim, int vDim, LinearAttentionState state)
    {
        int t = q.Rows;
        double scale = 1.0 / Math.Sqrt(kDim);
        var qn = new float[q.Data.Length];
        var kn = new float[k.Data.Length];
        void L2(float[] data, int offset, float[] target)
        {
            double sumSquares = 0;
            for (int i = 0; i < kDim; i++)
            {
                sumSquares += (double)data[offset + i] * data[offset + i];
            }
            double inv = 1.0 / Math.Sqrt(sumSquares + GatedDeltaKernel.QkL2NormEps);
            for (int i = 0; i < kDim; i++)
            {
                target[offset + i] = (float)(data[offset + i] * inv);
            }
        }
        for (int row = 0; row < t; row++)
        {
            for (int h = 0; h < heads; h++)
            {
                L2(q.Data, row * q.Cols + h * kDim, qn);
                L2(k.Data, row * k.Cols + h * kDim, kn);
            }
        }
        var S = state.S;
        var output = new float[t * heads * vDim];
        for (int pos = 0; pos < t; pos++)
        {
            for (int h = 0; h < heads; h++)
            {
                double decay = Math.Exp(g.Data[pos * g.Cols + h]);
                float betaV = beta.Data[pos * beta.Cols + h];
                int kBase = pos * k.Cols + h * kDim;
                int vBase = pos * v.Cols + h * vDim;
                for (int kk = 0; kk < kDim; kk++)
                {
                    for (int d = 0; d < vDim; d++)
                    {
                        S[h][kk, d] = (float)(S[h][kk, d] * decay);
                    }
                }
                var kvMem = new float[vDim];
                for (int d = 0; d < vDim; d++)
                {
                    double acc = 0;
                    for (int kk = 0; kk < kDim; kk++)
                    {
                        acc += S[h][kk, d] * kn[kBase + kk];
                    }
                    kvMem[d] = (float)acc;
                }
                for (int kk = 0; kk < kDim; kk++)
                {
                    float kVal = kn[kBase + kk];
                    for (int d = 0; d < vDim; d++)
                    {
                        float delta = (v.Data[vBase + d] - kvMem[d]) * betaV;
                        S[h][kk, d] += kVal * delta;
                    }
                }
                for (int d = 0; d < vDim; d++)
                {
                    double acc = 0;
                    for (int kk = 0; kk < kDim; kk++)
                    {
                        acc += S[h][kk, d] * qn[pos * q.Cols + h * kDim + kk];
                    }
                    output[pos * heads * vDim + h * vDim + d] = (float)(acc * scale);
                }
            }
        }
        return new Tensor2D(output, t, heads * vDim);
    }

    [Theory]
    [InlineData(1, 1, 8, 2, 64, null)]
    [InlineData(1, 37, 8, 2, 64, null)]
    [InlineData(16, 16, 4, 4, 40, null)]
    [InlineData(24, 30, 6, 3, 32, 7)]
    public void Attention_On_Cache_Buffers_Is_Bit_Identical_To_The_Scalar_Reference(
        int seq, int kvSeq, int qHeads, int kvHeads, int headDim, int? window)
    {
        var q = new Tensor2D(Values(seq * qHeads * headDim, 30), seq, qHeads * headDim);
        int kvWidth = kvHeads * headDim;
        var k = new Tensor2D(Values(kvSeq * kvWidth, 31), kvSeq, kvWidth);
        var v = new Tensor2D(Values(kvSeq * kvWidth, 32), kvSeq, kvWidth);
        var kvPositions = Enumerable.Range(0, kvSeq).ToArray();
        var queryPositions = Enumerable.Range(kvSeq - seq, seq).ToArray();
        double scale = 1.0 / Math.Sqrt(headDim);

        var expected = ScalarAttention(q, k, v, qHeads, kvHeads, headDim, scale, window, queryPositions, kvPositions);

        // The cache hands the kernel buffers with spare capacity past the rows.
        var cache = new RowKvCache();
        for (int r = 0; r < kvSeq; r++)
        {
            cache.Append(0, k.Row(r), v.Row(r));
        }
        var (keys, keyWidth, values, valueWidth, count) = cache.Buffers(0);
        var actual = AttentionKernel.Execute(q, keys, keyWidth, values, valueWidth, count,
            qHeads, kvHeads, headDim, scale, null, null, window, queryPositions, kvPositions);
        AssertBitEqual(expected, actual.Data, "buffers");

        var viaTensors = AttentionKernel.Execute(q, k, v, qHeads, kvHeads, headDim, scale, null, null, window,
            queryPositions, kvPositions);
        AssertBitEqual(expected, viaTensors.Data, "tensors");
    }

    /// <summary>Causal GQA attention as it was before the rewrite (no softcap,
    /// sinks or capture), verbatim apart from those branches.</summary>
    private static float[] ScalarAttention(Tensor2D q, Tensor2D k, Tensor2D v, int numQHeads, int numKvHeads,
        int headDim, double scoreScale, int? window, int[] queryPositions, int[] kvPositions)
    {
        int seq = q.Rows, kvSeq = k.Rows, headsRep = numQHeads / numKvHeads;
        var output = Tensor2D.Zeros(seq, numQHeads * headDim);
        var weights = new float[kvSeq];
        for (int qi = 0; qi < seq; qi++)
        {
            int qPos = queryPositions[qi];
            int lastValid = -1;
            for (int h = 0; h < numQHeads; h++)
            {
                int kvHead = h / headsRep;
                var qSlice = q.Row(qi).Slice(h * headDim, headDim);
                var outSlice = output.Row(qi).Slice(h * headDim, headDim);
                for (int j = 0; j < kvSeq; j++)
                {
                    int s = qPos - kvPositions[j];
                    if (s < 0)
                    {
                        break;
                    }
                    if (window is { } w && s >= w)
                    {
                        weights[j] = float.NegativeInfinity;
                        continue;
                    }
                    var kSlice = k.Row(j).Slice(kvHead * headDim, headDim);
                    double score = TensorOps.Dot(qSlice, kSlice) * scoreScale;
                    weights[j] = (float)score;
                    lastValid = j;
                }
                int spanEnd = lastValid + 1;
                if (spanEnd == 0)
                {
                    outSlice.Clear();
                    continue;
                }
                AttentionKernel.SoftmaxInPlace(weights.AsSpan(0, spanEnd));
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < spanEnd; j++)
                    {
                        float w = weights[j];
                        if (w == 0f)
                        {
                            continue;
                        }
                        acc += w * v.Row(j)[kvHead * headDim + d];
                    }
                    outSlice[d] = acc;
                }
            }
        }
        return output.Data;
    }

    [Fact]
    public void KernelPool_Runs_Every_Index_Once_And_Propagates_Failures()
    {
        for (int round = 0; round < 50; round++)
        {
            int count = 1 + round * 7;
            var hits = new int[count];
            KernelPool.For(count, i => Interlocked.Increment(ref hits[i]));
            Assert.All(hits, h => Assert.Equal(1, h));
        }

        // Nested loops fall back to Parallel.For and still complete.
        var nested = new int[16 * 16];
        KernelPool.For(16, i => KernelPool.For(16, j => Interlocked.Increment(ref nested[i * 16 + j])));
        Assert.All(nested, h => Assert.Equal(1, h));

        var error = Assert.Throws<AggregateException>(() =>
            KernelPool.For(64, i => { if (i == 13) throw new InvalidOperationException("thirteen"); }));
        Assert.Contains("thirteen", error.InnerException!.Message);

        // The pool is usable after a failure.
        int sum = 0;
        KernelPool.For(100, i => Interlocked.Add(ref sum, i));
        Assert.Equal(4950, sum);
    }
}
