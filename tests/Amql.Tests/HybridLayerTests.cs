using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Numeric tests for the Qwen3.5 hybrid operator paths now served by the
/// runtime: the weighted per-head QK norm, the hard output gate, and the
/// linear-attention (GatedDeltaNet) layer with position-major stateful
/// prefill. Each runtime run is compared against an independent naive
/// implementation written from the reference formulas.
/// </summary>
public class HybridLayerTests
{
    private static float[] RunPrefillLogits(ContainerSpec spec, Dims d, int[] tokens)
    {
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "c");
        ContainerEncoder.Encode(containerPath, spec);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var session = new DecodeSession(plan, store);
        return session.Prefill(tokens).FirstRow().ToArray();
    }

    private static void AssertClose(float[] actual, float[] expected, Dims d, int tokenCount, double tolerance = 2e-3)
    {
        // The naives return the full T×vocab matrix; the runtime returns
        // the last position's row (some naives already return just that).
        var lastRow = expected.Length == d.Vocab
            ? expected
            : expected.Skip((tokenCount - 1) * d.Vocab).Take(d.Vocab).ToArray();
        Assert.Equal(actual.Length, lastRow.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = Math.Abs(actual[i] - lastRow[i]);
            double scale = Math.Max(1.0, Math.Abs(lastRow[i]));
            Assert.True(diff <= tolerance * scale,
                $"index {i}: actual {actual[i]} vs expected {lastRow[i]} (diff {diff})");
        }
    }

    // ── weighted per-head QK norm ───────────────────────────────────────────

    [Fact]
    public void WeightedQkNorm_Matches_Naive()
    {
        var d = new Dims(Layers: 1, WeightedQkNorm: true, Rope: false);
        var tokens = new[] { 1, 3, 5 };
        var spec = SyntheticModel.BuildSpec(d);

        var actual = RunPrefillLogits(spec, d, tokens);
        var expected = Naive.WeightedQkFull(d, tokens);
        AssertClose(actual, expected, d, tokens.Length);
    }

    // ── hard output gate ────────────────────────────────────────────────────

    [Fact]
    public void OutputGate_Matches_Naive()
    {
        var d = new Dims(Layers: 1, OutputGate: true, Rope: false);
        var tokens = new[] { 2, 6 };
        var spec = SyntheticModel.BuildSpec(d);

        var actual = RunPrefillLogits(spec, d, tokens);
        var expected = Naive.OutputGate(d, tokens);
        AssertClose(actual, expected, d, tokens.Length);
    }

    // ── linear attention (GatedDeltaNet), position-major prefill ───────────

    [Fact]
    public void LinearAttention_Layer_Matches_Reference()
    {
        // Layer 0 linear (stateful), layer 1 softmax — the position-major
        // prefill path — verified against the reference oracle
        // (synth_oracle.json: an independent python port of the Qwen3.5
        // math over the same synthetic weights).
        var d = new Dims(Layers: 2, LinearLayer0: true, Rope: false);
        var tokens = new[] { 1, 4, 2 };
        var spec = SyntheticModel.BuildSpec(d);

        var actual = RunPrefillLogits(spec, d, tokens);
        AssertAgainstOracle(actual, "mixed_logits", 5e-3);
    }

    [Fact]
    public void LinearLayer_Only_Matches_Reference()
    {
        var d = new Dims(Layers: 1, LinearLayer0: true, Rope: false);
        var tokens = new[] { 1, 4, 2 };
        var spec = SyntheticModel.BuildSpec(d);

        var actual = RunPrefillLogits(spec, d, tokens);
        AssertAgainstOracle(actual, "linear_logits", 5e-3);
    }

    private static void AssertAgainstOracle(float[] actual, string key, double tolerance)
    {
        var file = Path.Combine(AppContext.BaseDirectory, "synth_oracle.json");
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(file));
        var arr = doc.RootElement.GetProperty(key);
        var oracle = arr.EnumerateArray().Select(e => e.GetSingle()).ToArray();
        Assert.Equal(actual.Length, oracle.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = Math.Abs(actual[i] - oracle[i]);
            double scale = Math.Max(1.0, Math.Abs(oracle[i]));
            Assert.True(diff <= tolerance * scale,
                $"index {i}: actual {actual[i]} vs oracle {oracle[i]} (diff {diff})");
        }
    }

    [Fact]
    public void LinearAttention_Plans_And_Serves()
    {
        var d = new Dims(Layers: 2, LinearLayer0: true);
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "c");
        ContainerEncoder.Encode(containerPath, SyntheticModel.BuildSpec(d));
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.True(plan.Layers[0].IsStateful);
        Assert.NotNull(plan.Layers[0].LinearAttention);
        Assert.False(plan.Layers[1].IsStateful);
        Assert.NotNull(plan.Layers[1].Attention);
    }

    [Fact]
    public void LinearLayer_Only_Matches_Naive()
    {
        var d = new Dims(Layers: 1, LinearLayer0: true, Rope: false);
        var tokens = new[] { 1, 4, 2 };
        var spec = SyntheticModel.BuildSpec(d);

        var actual = RunPrefillLogits(spec, d, tokens);
        var expected = Naive.LinearOnly(d, tokens);
        AssertClose(actual, expected, d, tokens.Length, 3e-3);
    }
}

/// <summary>Independent naive implementations of the hybrid paths, written
/// straight from the reference formulas; weights are regenerated with the
/// same deterministic generator the synthetic checkpoint uses.</summary>
internal static class Naive
{
    private static float[] Wm(int rows, int cols, int salt, int layer) =>
        Build(rows, cols, salt, layer);

    private static float[] Build(int rows, int cols, int salt, int layer)
    {
        var m = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                m[i * cols + j] = SyntheticModel.W(layer, i, j, salt);
            }
        }
        return m;
    }

    private static float[] Wv(int width, int salt) =>
        Enumerable.Range(0, width).Select(i => SyntheticModel.NormW(i, salt)).ToArray();

    private static float[] W1(int width, int salt, int layer) =>
        Enumerable.Range(0, width).Select(i => SyntheticModel.W(layer, i, 0, salt)).ToArray();

    private static float[] MatMulTransposedB(float[] x, int xRows, int xCols, float[] w, int wRows)
    {
        var c = new float[xRows * wRows];
        for (int i = 0; i < xRows; i++)
        {
            for (int j = 0; j < wRows; j++)
            {
                float acc = 0;
                for (int k = 0; k < xCols; k++)
                {
                    acc += x[i * xCols + k] * w[j * xCols + k];
                }
                c[i * wRows + j] = acc;
            }
        }
        return c;
    }

    private static void Rms(Span<float> row, float[] weight, float offset, double eps)
    {
        float sum = 0;
        for (int i = 0; i < row.Length; i++)
        {
            sum += row[i] * row[i];
        }
        float inv = (float)(1.0 / Math.Sqrt(sum / row.Length + eps));
        for (int i = 0; i < row.Length; i++)
        {
            row[i] = row[i] * inv * (weight[i] + offset);
        }
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float Dot(float[] q, int qi, int head, int j, int kvHead, float[] k, Dims d)
    {
        float acc = 0;
        for (int dd = 0; dd < d.HeadDim; dd++)
        {
            acc += q[qi * d.QDim + head * d.HeadDim + dd] * k[j * d.KvDim + kvHead * d.HeadDim + dd];
        }
        return acc;
    }

    /// <summary>Weighted per-head QK norm applied to q and k (no rope).</summary>
    public static float[] WeightedQkFull(Dims d, int[] tokens)
    {
        int hidden = d.Hidden;
        int qDim = d.QDim;
        int kvDim = d.KvDim;
        int T = tokens.Length;
        float eps = (float)d.NormEps;
        float scale = (float)(1.0 / Math.Sqrt(d.HeadDim));

        var embed = Wm(d.Vocab, hidden, 1, 0);
        var x = new float[T * hidden];
        for (int i = 0; i < T; i++)
        {
            Array.Copy(embed, tokens[i] * hidden, x, i * hidden, hidden);
        }

        var h = (float[])x.Clone();
        for (int r = 0; r < T; r++)
        {
            Rms(h.AsSpan(r * hidden, hidden), Wv(hidden, 11), 0, eps);
        }

        var q = MatMulTransposedB(h, T, hidden, Wm(qDim, hidden, 2, 0), qDim);
        var k = MatMulTransposedB(h, T, hidden, Wm(kvDim, hidden, 3, 0), kvDim);
        var v = MatMulTransposedB(h, T, hidden, Wm(kvDim, hidden, 4, 0), kvDim);

        // Weighted per-head RMS on Q and K (the reference's "only on the
        // head dim"), offset 0 in the synthetic family.
        var qw = Wv(d.HeadDim, 40);
        var kw = Wv(d.HeadDim, 41);
        for (int r = 0; r < T; r++)
        {
            for (int hh = 0; hh < d.NumQHeads; hh++)
            {
                Rms(q.AsSpan(r * qDim + hh * d.HeadDim, d.HeadDim), qw, 0, eps);
            }
            for (int hh = 0; hh < d.NumKvHeads; hh++)
            {
                Rms(k.AsSpan(r * kvDim + hh * d.HeadDim, d.HeadDim), kw, 0, eps);
            }
        }

        var ao = Attention(q, k, v, d, T, scale);
        var o = MatMulTransposedB(ao, T, qDim, Wm(hidden, qDim, 5, 0), hidden);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += o[i];
        }

        var hf = (float[])x.Clone();
        for (int r = 0; r < T; r++)
        {
            Rms(hf.AsSpan(r * hidden, hidden), Wv(hidden, 12), 0, eps);
        }
        var g = MatMulTransposedB(hf, T, hidden, Wm(d.Intermediate, hidden, 6, 0), d.Intermediate);
        var u = MatMulTransposedB(hf, T, hidden, Wm(d.Intermediate, hidden, 7, 0), d.Intermediate);
        var up = new float[T * d.Intermediate];
        for (int i = 0; i < up.Length; i++)
        {
            up[i] = Silu(g[i]) * u[i];
        }
        var down = MatMulTransposedB(up, T, d.Intermediate, Wm(hidden, d.Intermediate, 8, 0), hidden);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += down[i];
        }
        for (int r = 0; r < T; r++)
        {
            Rms(x.AsSpan(r * hidden, hidden), Wv(hidden, 3), 0, eps);
        }
        return MatMulTransposedB(x, T, hidden, Wm(d.Vocab, hidden, 9, 0), d.Vocab);
    }

    public static float[] OutputGate(Dims d, int[] tokens)
    {
        int hidden = d.Hidden;
        int qDim = d.QDim;
        int kvDim = d.KvDim;
        int T = tokens.Length;
        float eps = (float)d.NormEps;
        float scale = (float)(1.0 / Math.Sqrt(d.HeadDim));

        var embed = Wm(d.Vocab, hidden, 1, 0);
        var x = new float[T * hidden];
        for (int i = 0; i < T; i++)
        {
            Array.Copy(embed, tokens[i] * hidden, x, i * hidden, hidden);
        }

        var h = (float[])x.Clone();
        for (int r = 0; r < T; r++)
        {
            Rms(h.AsSpan(r * hidden, hidden), Wv(hidden, 11), 0, eps);
        }

        // q_proj is 2×QDim interleaved per head [q_h | gate_h]; the second
        // half of each head block gates the attention output.
        var qRaw = MatMulTransposedB(h, T, hidden, Wm(2 * qDim, hidden, 2, 0), 2 * qDim);
        var q = new float[T * qDim];
        var gate = new float[T * qDim];
        int headDim = d.HeadDim;
        for (int i = 0; i < T; i++)
        {
            for (int hh = 0; hh < d.NumQHeads; hh++)
            {
                Array.Copy(qRaw, i * 2 * qDim + hh * 2 * headDim, q, i * qDim + hh * headDim, headDim);
                Array.Copy(qRaw, i * 2 * qDim + hh * 2 * headDim + headDim, gate, i * qDim + hh * headDim, headDim);
            }
        }
        var k = MatMulTransposedB(h, T, hidden, Wm(kvDim, hidden, 3, 0), kvDim);
        var v = MatMulTransposedB(h, T, hidden, Wm(kvDim, hidden, 4, 0), kvDim);

        var ao = Attention(q, k, v, d, T, scale);
        for (int i = 0; i < ao.Length; i++)
        {
            ao[i] *= Sigmoid(gate[i]); // hard gate: × σ(gate)
        }
        var o = MatMulTransposedB(ao, T, qDim, Wm(hidden, qDim, 5, 0), hidden);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += o[i];
        }

        var hf = (float[])x.Clone();
        for (int r = 0; r < T; r++)
        {
            Rms(hf.AsSpan(r * hidden, hidden), Wv(hidden, 12), 0, eps);
        }
        var g = MatMulTransposedB(hf, T, hidden, Wm(d.Intermediate, hidden, 6, 0), d.Intermediate);
        var u = MatMulTransposedB(hf, T, hidden, Wm(d.Intermediate, hidden, 7, 0), d.Intermediate);
        var up = new float[T * d.Intermediate];
        for (int i = 0; i < up.Length; i++)
        {
            up[i] = Silu(g[i]) * u[i];
        }
        var down = MatMulTransposedB(up, T, d.Intermediate, Wm(hidden, d.Intermediate, 8, 0), hidden);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += down[i];
        }
        for (int r = 0; r < T; r++)
        {
            Rms(x.AsSpan(r * hidden, hidden), Wv(hidden, 3), 0, eps);
        }
        return MatMulTransposedB(x, T, hidden, Wm(d.Vocab, hidden, 9, 0), d.Vocab);
    }

    // ── linear layer 0 → softmax layer 1, position-major ────────────────────

    public static float[] LinearOnly(Dims d, int[] tokens)
    {
        int hidden = d.Hidden;
        double eps = d.NormEps;
        int kd = Dims.LinKHeads * Dims.LinKHeadDim;
        int vd = Dims.LinVHeads * Dims.LinVHeadDim;
        int convDim = 2 * kd + vd;

        var embed = Wm(d.Vocab, hidden, 1, 0);
        var wQkv = Wm(convDim, hidden, 50, 0);
        var wZ = Wm(vd, hidden, 51, 0);
        var wA = Wm(Dims.LinVHeads, hidden, 52, 0);
        var wB = Wm(Dims.LinVHeads, hidden, 53, 0);
        var wOut = Wm(hidden, vd, 54, 0);
        var conv = W1(convDim * Dims.LinConvKernel, 55, 0);
        var aLog = W1(Dims.LinVHeads, 56, 0);
        var dtBias = W1(Dims.LinVHeads, 57, 0);
        var normW = W1(Dims.LinVHeadDim, 58, 0);
        var wNPre = Wv(hidden, 11);
        var wNPost = Wv(hidden, 12);
        var wNFinal = Wv(hidden, 3);
        var wG = Wm(d.Intermediate, hidden, 6, 0);
        var wU = Wm(d.Intermediate, hidden, 7, 0);
        var wD = Wm(hidden, d.Intermediate, 8, 0);
        var lmHead = Wm(d.Vocab, hidden, 9, 0);

        var s = new float[Dims.LinVHeads][,];
        var convHist = new float[convDim];
        var x = new float[hidden];

        for (int p = 0; p < tokens.Length; p++)
        {
            for (int i = 0; i < hidden; i++)
            {
                x[i] = embed[tokens[p] * hidden + i];
            }
            var h = (float[])x.Clone();
            Rms(h, wNPre, 0, eps);

            var mixedRow = MatMulTransposedB(h, 1, hidden, wQkv, convDim);
            var zRow = MatMulTransposedB(h, 1, hidden, wZ, vd);
            var aRow = MatMulTransposedB(h, 1, hidden, wA, Dims.LinVHeads);
            var bRow = MatMulTransposedB(h, 1, hidden, wB, Dims.LinVHeads);

            var walked = new float[convDim];
            for (int c = 0; c < convDim; c++)
            {
                walked[c] = Silu(conv[c * 2] * convHist[c] + conv[c * 2 + 1] * mixedRow[c]);
                convHist[c] = mixedRow[c];
            }

            var qL = new float[kd];
            var kL = new float[kd];
            var vL = new float[vd];
            Array.Copy(walked, 0, qL, 0, kd);
            Array.Copy(walked, kd, kL, 0, kd);
            Array.Copy(walked, 2 * kd, vL, 0, vd);

            var gL = new float[Dims.LinVHeads];
            var betaL = new float[Dims.LinVHeads];
            for (int hh = 0; hh < Dims.LinVHeads; hh++)
            {
                double zz = aRow[hh] + dtBias[hh];
                double softPlus = zz > 20 ? zz : Math.Log(1.0 + Math.Exp(zz));
                gL[hh] = (float)(-Math.Exp(aLog[hh]) * softPlus);
                betaL[hh] = Sigmoid(bRow[hh]);
            }

            var core = new float[Dims.LinVHeads * Dims.LinVHeadDim];
            for (int hh = 0; hh < Dims.LinVHeads; hh++)
            {
                var qh = new float[Dims.LinKHeadDim];
                var kh = new float[Dims.LinKHeadDim];
                double qDot = 0, kDot = 0;
                for (int i = 0; i < Dims.LinKHeadDim; i++)
                {
                    qh[i] = qL[hh * Dims.LinKHeadDim + i];
                    kh[i] = kL[hh * Dims.LinKHeadDim + i];
                    qDot += (double)qh[i] * qh[i];
                    kDot += (double)kh[i] * kh[i];
                }
                double qInv = 1.0 / Math.Sqrt(qDot + 1e-6);
                double kInv = 1.0 / Math.Sqrt(kDot + 1e-6);
                double qScale = 1.0 / Math.Sqrt(Dims.LinKHeadDim);
                for (int i = 0; i < Dims.LinKHeadDim; i++)
                {
                    qh[i] = (float)(qh[i] * qInv * qScale);
                    kh[i] = (float)(kh[i] * kInv);
                }

                var sH = s[hh] ??= new float[Dims.LinKHeadDim, Dims.LinVHeadDim];
                double decay = Math.Exp(gL[hh]);
                float betaV = betaL[hh];
                var kvMem = new float[Dims.LinVHeadDim];
                for (int dd = 0; dd < Dims.LinVHeadDim; dd++)
                {
                    double acc = 0;
                    for (int i = 0; i < Dims.LinKHeadDim; i++)
                    {
                        acc += sH[i, dd] * kh[i];
                    }
                    kvMem[dd] = (float)acc;
                }
                for (int i = 0; i < Dims.LinKHeadDim; i++)
                {
                    for (int dd = 0; dd < Dims.LinVHeadDim; dd++)
                    {
                        float delta = (vL[hh * Dims.LinVHeadDim + dd] - kvMem[dd]) * betaV;
                        sH[i, dd] = (float)(sH[i, dd] * decay) + kh[i] * delta;
                    }
                }
                for (int dd = 0; dd < Dims.LinVHeadDim; dd++)
                {
                    double acc = 0;
                    for (int i = 0; i < Dims.LinKHeadDim; i++)
                    {
                        acc += sH[i, dd] * qh[i];
                    }
                    core[hh * Dims.LinVHeadDim + dd] = (float)acc;
                }
            }

            var normed = new float[Dims.LinVHeads * Dims.LinVHeadDim];
            for (int c = 0; c < core.Length; c++)
            {
                int hh = c / Dims.LinVHeadDim;
                int dd = c % Dims.LinVHeadDim;
                double sumSq = 0;
                for (int i = 0; i < Dims.LinVHeadDim; i++)
                {
                    sumSq += (double)core[hh * Dims.LinVHeadDim + i] * core[hh * Dims.LinVHeadDim + i];
                }
                double inv = 1.0 / Math.Sqrt(sumSq / Dims.LinVHeadDim + eps);
                normed[c] = (float)(core[c] * inv * normW[dd] * Silu(zRow[c]));
            }
            var mixerOut = MatMulTransposedB(normed, 1, vd, wOut, hidden);
            for (int i = 0; i < hidden; i++)
            {
                x[i] += mixerOut[i];
            }
            
            var hf = (float[])x.Clone();
            Rms(hf, wNPost, 0, eps);
            var g2 = MatMulTransposedB(hf, 1, hidden, wG, d.Intermediate);
            var u2 = MatMulTransposedB(hf, 1, hidden, wU, d.Intermediate);
            for (int i = 0; i < d.Intermediate; i++)
            {
                g2[i] = Silu(g2[i]) * u2[i];
            }
            var down = MatMulTransposedB(g2, 1, d.Intermediate, wD, hidden);
            for (int i = 0; i < hidden; i++)
            {
                x[i] += down[i];
            }
                    }

        Rms(x, wNFinal, 0, eps);
        return MatMulTransposedB(x, 1, hidden, lmHead, d.Vocab);
    }

    // ── shared attention over q/k/v ─────────────────────────────────────────

    private static float[] Attention(float[] q, float[] k, float[] v, Dims d, int T, float scale)
    {
        int qDim = d.QDim;
        int kvDim = d.KvDim;
        var ao = new float[T * qDim];
        for (int qi = 0; qi < T; qi++)
        {
            for (int hh = 0; hh < d.NumQHeads; hh++)
            {
                int kvHead = hh / (d.NumQHeads / d.NumKvHeads);
                var scores = new float[qi + 1];
                for (int j = 0; j <= qi; j++)
                {
                    float acc = 0;
                    for (int dd = 0; dd < d.HeadDim; dd++)
                    {
                        acc += q[qi * qDim + hh * d.HeadDim + dd] * k[j * kvDim + kvHead * d.HeadDim + dd];
                    }
                    scores[j] = acc * scale;
                }
                float max = scores.Max();
                float sum = scores.Sum(s => MathF.Exp(s - max));
                for (int dd = 0; dd < d.HeadDim; dd++)
                {
                    float acc = 0;
                    for (int j = 0; j <= qi; j++)
                    {
                        acc += MathF.Exp(scores[j] - max) / sum * v[j * kvDim + kvHead * d.HeadDim + dd];
                    }
                    ao[qi * qDim + hh * d.HeadDim + dd] = acc;
                }
            }
        }
        return ao;
    }
}
