using System.Text.Json;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

public class InferenceTests
{
    // ── naive reference implementation ─────────────────────────────────────
    // Plain loops over freshly regenerated weights — the independent oracle
    // for the runtime's orchestration (residual placement, cache ordering,
    // GQA grouping, RoPE positions, span masking).

    private static float[] NaiveForward(Dims d, int[] tokens, bool applyRope, long? window)
    {
        int hidden = d.Hidden;
        int qDim = d.QDim;
        int kvDim = d.KvDim;
        int T = tokens.Length;
        float eps = (float)d.NormEps;
        float scale = (float)(1.0 / Math.Sqrt(d.HeadDim));

        float[] Wm(int rows, int cols, int salt, int l)
        {
            var m = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m[i * cols + j] = SyntheticModel.W(l, i, j, salt);
                }
            }
            return m;
        }

        float[] Wv(int width, int salt)
        {
            var v = new float[width];
            for (int i = 0; i < width; i++)
            {
                v[i] = SyntheticModel.NormW(i, salt);
            }
            return v;
        }

        float[] MatMulTransposedB(float[] x, int xRows, int xCols, float[] w, int wRows)
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

        void RmsNorm(Span<float> row, float[] w, float weightOffset)
        {
            float sum = 0;
            for (int i = 0; i < row.Length; i++)
            {
                sum += row[i] * row[i];
            }
            float inv = (float)(1.0 / Math.Sqrt(sum / row.Length + eps));
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = row[i] * inv * (w[i] + weightOffset);
            }
        }

        void NormRows(float[] m, int rows, int cols, float[] w, float weightOffset)
        {
            for (int r = 0; r < rows; r++)
            {
                RmsNorm(m.AsSpan(r * cols, cols), w, weightOffset);
            }
        }

        void ApplyRope(float[] m, int rows, int heads, int headDim, double theta)
        {
            int pairs = headDim / 2;
            var inv = new double[pairs];
            for (int i = 0; i < pairs; i++)
            {
                inv[i] = 1.0 / Math.Pow(theta, 2.0 * i / headDim);
            }
            for (int p = 0; p < rows; p++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int i = 0; i < pairs; i++)
                    {
                        double angle = p * inv[i];
                        double cos = Math.Cos(angle);
                        double sin = Math.Sin(angle);
                        int a = p * heads * headDim + h * headDim + 2 * i;
                        double x1 = m[a];
                        double x2 = m[a + 1];
                        m[a] = (float)(x1 * cos - x2 * sin);
                        m[a + 1] = (float)(x1 * sin + x2 * cos);
                    }
                }
            }
        }

        // Embedding lookup.
        var embed = Wm(d.Vocab, hidden, 1, 0);
        var x = new float[T * hidden];
        for (int i = 0; i < T; i++)
        {
            Array.Copy(embed, tokens[i] * hidden, x, i * hidden, hidden);
        }

        int headRep = d.NumQHeads / d.NumKvHeads;
        for (int l = 0; l < d.Layers; l++)
        {
            // Pre-attention norm.
            var preW = Wv(hidden, 11);
            var h = (float[])x.Clone();
            NormRows(h, T, hidden, preW, 0);

            var q = MatMulTransposedB(h, T, hidden, Wm(hidden, qDim, 2, l), qDim);
            var k = MatMulTransposedB(h, T, hidden, Wm(hidden, kvDim, 3, l), kvDim);
            var v = MatMulTransposedB(h, T, hidden, Wm(hidden, kvDim, 4, l), kvDim);

            if (applyRope)
            {
                ApplyRope(q, T, d.NumQHeads, d.HeadDim, d.RopeTheta);
                ApplyRope(k, T, d.NumKvHeads, d.HeadDim, d.RopeTheta);
            }

            // Attention (causal, optionally windowed).
            var ao = new float[T * qDim];
            var scores = new float[T];
            for (int qi = 0; qi < T; qi++)
            {
                for (int hh = 0; hh < d.NumQHeads; hh++)
                {
                    int kvHead = hh / headRep;
                    for (int j = 0; j <= qi; j++)
                    {
                        scores[j] = window is { } w && qi - j >= w
                            ? float.NegativeInfinity
                            : Dot(q.AsSpan(qi * qDim + hh * d.HeadDim, d.HeadDim),
                                  k.AsSpan(j * kvDim + kvHead * d.HeadDim, d.HeadDim)) * scale;
                    }
                    float max = float.NegativeInfinity;
                    for (int j = 0; j <= qi; j++)
                    {
                        max = Math.Max(max, scores[j]);
                    }
                    if (!float.IsNegativeInfinity(max))
                    {
                        float sum = 0;
                        for (int j = 0; j <= qi; j++)
                        {
                            scores[j] = MathF.Exp(scores[j] - max);
                            sum += scores[j];
                        }
                        for (int j = 0; j <= qi; j++)
                        {
                            scores[j] /= sum;
                        }
                        for (int dd = 0; dd < d.HeadDim; dd++)
                        {
                            float acc = 0;
                            for (int j = 0; j <= qi; j++)
                            {
                                if (window is { } w2 && qi - j >= w2)
                                {
                                    continue;
                                }
                                acc += scores[j] * v[j * kvDim + kvHead * d.HeadDim + dd];
                            }
                            ao[qi * qDim + hh * d.HeadDim + dd] = acc;
                        }
                    }
                }
            }

            var o = MatMulTransposedB(ao, T, qDim, Wm(hidden, qDim, 5, l), hidden);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += o[i];
            }

            // Pre-FFN norm.
            var postW = Wv(hidden, 12);
            var hf = (float[])x.Clone();
            NormRows(hf, T, hidden, postW, 0);

            var g = MatMulTransposedB(hf, T, hidden, Wm(d.Intermediate, hidden, 6, l), d.Intermediate);
            var u = MatMulTransposedB(hf, T, hidden, Wm(d.Intermediate, hidden, 7, l), d.Intermediate);
            var up = new float[T * d.Intermediate];
            for (int i = 0; i < up.Length; i++)
            {
                float gi = g[i];
                up[i] = gi / (1f + MathF.Exp(-gi)) * u[i]; // SiLU(gate) * up
            }
            var down = MatMulTransposedB(up, T, d.Intermediate, Wm(hidden, d.Intermediate, 8, l), hidden);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += down[i];
            }
        }

        // Final norm + head.
        var finalW = Wv(hidden, 3);
        NormRows(x, T, hidden, finalW, 0);
        var lmHead = Wm(d.Vocab, hidden, 9, 0);
        return MatMulTransposedB(x, T, hidden, lmHead, d.Vocab);

        static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            float acc = 0;
            for (int i = 0; i < a.Length; i++)
            {
                acc += a[i] * b[i];
            }
            return acc;
        }
    }

    // ── shared setup ───────────────────────────────────────────────────────

    private static string EncodeTo(ContainerSpec spec, TempDir dir)
    {
        var containerPath = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerPath, spec);
        return containerPath;
    }

    private static float[] RunRuntime(string containerPath, Dims d, int[] tokens)
    {
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var session = new DecodeSession(plan, store);
        return session.Prefill(tokens).FirstRow().ToArray();
    }

    /// <summary>The runtime session returns the logits of the last position
    /// (the token-level contract); the naive reference returns the whole
    /// T×vocab matrix. Equality is asserted on the last row.</summary>
    private static float[] LastRowOf(float[] fullMatrix, Dims d) =>
        fullMatrix.Skip((fullMatrix.Length / d.Vocab - 1) * d.Vocab).Take(d.Vocab).ToArray();

    private static void AssertClose(float[] actual, float[] expected, double tolerance = 1e-3)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);
            double scale = Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(diff <= tolerance * scale,
                $"index {i}: actual {actual[i]} vs expected {expected[i]} (diff {diff})");
        }
    }

    // ── tests ──────────────────────────────────────────────────────────────

    [Fact]
    public void Prefill_Matches_NaiveReference()
    {
        var d = new Dims();
        var tokens = new[] { 1, 3, 5 };
        using var dir = new TempDir();
        var containerPath = EncodeTo(SyntheticModel.BuildSpec(d), dir);

        var actual = RunRuntime(containerPath, d, tokens);
        var expected = NaiveForward(d, tokens, applyRope: true, window: null);

        AssertClose(actual, LastRowOf(expected, d), 2e-3);
    }

    [Fact]
    public void NoRope_Matches_NaiveReference()
    {
        var d = new Dims(Rope: false);
        var tokens = new[] { 1, 3, 5, 7 };
        using var dir = new TempDir();
        var containerPath = EncodeTo(SyntheticModel.BuildSpec(d), dir);

        var actual = RunRuntime(containerPath, d, tokens);
        var expected = NaiveForward(d, tokens, applyRope: false, window: null);

        AssertClose(actual, LastRowOf(expected, d), 2e-3);
    }

    [Fact]
    public void SlidingWindow_Matches_NaiveReference()
    {
        var d = new Dims(Window: 2);
        var tokens = new[] { 0, 2, 4, 6, 8 };
        using var dir = new TempDir();
        var containerPath = EncodeTo(SyntheticModel.BuildSpec(d), dir);

        var actual = RunRuntime(containerPath, d, tokens);
        var expected = NaiveForward(d, tokens, applyRope: true, window: 2);

        AssertClose(actual, LastRowOf(expected, d), 2e-3);
    }

    [Fact]
    public void Decode_Step_Continues_Prefill_Exactly()
    {
        var d = new Dims();
        var tokens = new[] { 1, 3, 5 };
        using var dir = new TempDir();
        var containerPath = EncodeTo(SyntheticModel.BuildSpec(d), dir);

        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var session = new DecodeSession(plan, store);
        session.Prefill(tokens);
        Assert.Equal(3, session.Position);

        var stepped = session.Step(4).FirstRow().ToArray();
        var expected = RunRuntime(containerPath, d, tokens.Concat(new[] { 4 }).ToArray());

        AssertClose(stepped, expected, 2e-3);
        Assert.Equal(4, session.Position);
    }

    [Fact]
    public void Greedy_Sampling_Selects_ArgMax()
    {
        var logits = new Tensor2D(new[] { -1f, 2f, 0.5f, -3f }, 1, 4);
        Assert.Equal(1, Sampler.ArgMax(logits));
        Assert.Equal(1, Sampler.Sample(logits, new SamplingConfig(Seed: 0, Temperature: 0)));
    }

    [Fact]
    public void Routed_MoE_Matches_Hand_Computed_Reference()
    {
        // Top-1 router over two experts must equal the selected expert's
        // dense output scaled by the router probability — checked against
        // a hand-computed forward pass, not the runtime itself.
        var d = new Dims(Layers: 1, MoE: true, TopK: 1, Rope: false);
        var tokens = new[] { 2, 6 };
        using var dir = new TempDir();
        var containerPath = EncodeTo(SyntheticModel.BuildSpec(d), dir);

        var actual = RunRuntime(containerPath, d, tokens);

        // ── reference: plain loops, weights regenerated ──────────────────
        int hidden = d.Hidden;
        int qDim = d.QDim;
        int kvDim = d.KvDim;
        float eps = (float)d.NormEps;
        float scale = (float)(1.0 / Math.Sqrt(d.HeadDim));

        float[] Wm(int rows, int cols, int salt, int l)
        {
            var m = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m[i * cols + j] = SyntheticModel.W(l, i, j, salt);
                }
            }
            return m;
        }

        float[] Wv(int width, int salt) =>
            Enumerable.Range(0, width).Select(i => SyntheticModel.NormW(i, salt)).ToArray();

        static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            float acc = 0;
            for (int i = 0; i < a.Length; i++)
            {
                acc += a[i] * b[i];
            }
            return acc;
        }

        float[] M2(float[] x, int xRows, int xCols, float[] w, int wRows)
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

        void Rms(Span<float> row, float[] w)
        {
            float sum = 0;
            for (int i = 0; i < row.Length; i++)
            {
                sum += row[i] * row[i];
            }
            float inv = (float)(1.0 / Math.Sqrt(sum / row.Length + eps));
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = row[i] * inv * w[i];
            }
        }

        // Embedding (raw residual) + pre-attention norm (separate tensor).
        var embed = Wm(d.Vocab, hidden, 1, 0);
        var rawX = new float[2 * hidden];
        var h = new float[2 * hidden];
        for (int i = 0; i < 2; i++)
        {
            Array.Copy(embed, tokens[i] * hidden, rawX, i * hidden, hidden);
        }
        Array.Copy(rawX, h, rawX.Length);
        for (int i = 0; i < 2; i++)
        {
            Rms(h.AsSpan(i * hidden, hidden), Wv(hidden, 11));
        }

        // Attention (causal, no rope).
        var q = M2(h, 2, hidden, Wm(hidden, qDim, 2, 0), qDim);
        var k = M2(h, 2, hidden, Wm(hidden, kvDim, 3, 0), kvDim);
        var v = M2(h, 2, hidden, Wm(hidden, kvDim, 4, 0), kvDim);
        var ao = new float[2 * qDim];
        for (int qi = 0; qi < 2; qi++)
        {
            for (int hh = 0; hh < d.NumQHeads; hh++)
            {
                int kvHead = hh / (d.NumQHeads / d.NumKvHeads);
                var scores = new float[qi + 1];
                for (int j = 0; j <= qi; j++)
                {
                    scores[j] = Dot(q.AsSpan(qi * qDim + hh * d.HeadDim, d.HeadDim),
                                    k.AsSpan(j * kvDim + kvHead * d.HeadDim, d.HeadDim)) * scale;
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
        var o = M2(ao, 2, qDim, Wm(hidden, qDim, 5, 0), hidden);
        for (int i = 0; i < rawX.Length; i++)
        {
            rawX[i] += o[i];
        }
        // Pre-FFN norm applies to a copy; the raw residual stays the base
        // the final norm sees (residual += ffn later).
        var postX = (float[])rawX.Clone();
        for (int i = 0; i < 2; i++)
        {
            Rms(postX.AsSpan(i * hidden, hidden), Wv(hidden, 12));
        }

        // Router softmax + top-1 expert, then residual + final norm + head.
        var router = Wm(d.Experts, hidden, 13, 0);
        var lmHead = Wm(d.Vocab, hidden, 9, 0);
        var finalNormW = Wv(hidden, 3);
        var expected = new float[2 * d.Vocab];
        for (int i = 0; i < 2; i++)
        {
            var row = postX.AsSpan(i * hidden, hidden);
            float r0 = Dot(row, router.AsSpan(0, hidden));
            float r1 = Dot(row, router.AsSpan(hidden, hidden));
            float m = Math.Max(r0, r1);
            float p0 = MathF.Exp(r0 - m);
            float p1 = MathF.Exp(r1 - m);
            float sum = p0 + p1;
            p0 /= sum;
            p1 /= sum;
            int selected = p0 >= p1 ? 0 : 1;
            float weight = selected == 0 ? p0 : p1;

            var gate = Wm(d.Intermediate, hidden, 14 + selected, 0);
            var up = Wm(d.Intermediate, hidden, 20 + selected, 0);
            var down = Wm(hidden, d.Intermediate, 30 + selected, 0);
            var hiddenVec = new float[d.Intermediate];
            for (int j = 0; j < d.Intermediate; j++)
            {
                float g = Dot(row, gate.AsSpan(j * hidden, hidden));
                float u = Dot(row, up.AsSpan(j * hidden, hidden));
                hiddenVec[j] = g / (1f + MathF.Exp(-g)) * u;
            }

            // residual (raw) += weight * expert(x); final norm; head.
            var finalRow = new float[hidden];
            float sumSquares = 0;
            for (int r2 = 0; r2 < hidden; r2++)
            {
                float acc = 0;
                for (int j = 0; j < d.Intermediate; j++)
                {
                    acc += hiddenVec[j] * down[r2 * d.Intermediate + j];
                }
                finalRow[r2] = rawX[i * hidden + r2] + weight * acc;
                sumSquares += finalRow[r2] * finalRow[r2];
            }
            float inv = (float)(1.0 / Math.Sqrt(sumSquares / hidden + eps));
            for (int r2 = 0; r2 < hidden; r2++)
            {
                finalRow[r2] *= inv * finalNormW[r2];
            }

            for (int vIdx = 0; vIdx < d.Vocab; vIdx++)
            {
                expected[i * d.Vocab + vIdx] = Dot(finalRow, lmHead.AsSpan(vIdx * hidden, hidden));
            }
        }

        AssertClose(actual, LastRowOf(expected, d), 2e-3);
    }

    // ── fail-closed refusals ───────────────────────────────────────────────

    [Fact]
    public void Unsupported_Operator_Refuses_At_Plan()
    {
        var spec = SyntheticModel.BuildSpec(new Dims());
        spec.SystemGraph.Components[0].Attention![0].SetOperator(LayerOperators.Mamba2);

        using var dir = new TempDir();
        var containerPath = EncodeTo(spec, dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var ex = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(container, "target", store));
        Assert.Contains("mamba2", ex.Message);
    }

    [Fact]
    public void Linear_Attention_Operator_Refuses_At_Plan()
    {
        var spec = SyntheticModel.BuildSpec(new Dims());
        spec.SystemGraph.Components[0].Attention![0].SetOperator(LayerOperators.LinearAttention);

        using var dir = new TempDir();
        var containerPath = EncodeTo(spec, dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var ex = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(container, "target", store));
        Assert.Contains("linear_attention", ex.Message);
    }

    [Fact]
    public void Gated_Attention_Refuses_At_Plan()
    {
        // A persisted output gate must refuse, never be silently skipped.
        using var dir = new TempDir();
        var containerPath = EncodeTo(SyntheticModel.BuildSpec(new Dims(OutputGate: true)), dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var ex = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(container, "target", store));
        Assert.Contains("output gate", ex.Message);
    }

    [Fact]
    public void Weighted_QkNorm_Refuses_At_Plan()
    {
        // q_norm/k_norm weight tensors present ⇒ weighted QK norm is part
        // of the program; the managed executor serves only the
        // parameter-free variant and must refuse instead of skipping it.
        using var dir = new TempDir();
        var containerPath = EncodeTo(SyntheticModel.BuildSpec(new Dims(WeightedQkNorm: true)), dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var ex = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(container, "target", store));
        Assert.Contains("weighted QK norm", ex.Message);
    }

    [Fact]
    public void Unresolved_Position_Refuses_At_Plan()
    {
        var spec = SyntheticModel.BuildSpec(new Dims());
        spec.SystemGraph.Components[0].Attention![0].SetPosition(
            new PositionUnresolved { Kind = "yarn", Payload = JsonDocument.Parse("{\"kind\":\"yarn\",\"theta\":500000}").RootElement.Clone() });

        using var dir = new TempDir();
        var containerPath = EncodeTo(spec, dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var ex = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(container, "target", store));
        Assert.Contains("yarn", ex.Message);
    }
}

// Reflection helpers to mutate init-only graph properties in tests.
internal static class GraphMutation
{
    public static void SetOperator(this AttentionLayerPolicy policy, string op) =>
        typeof(AttentionLayerPolicy).GetProperty("Operator")!.SetValue(policy, op);

    public static void SetPosition(this AttentionLayerPolicy policy, PositionPolicy position) =>
        typeof(AttentionLayerPolicy).GetProperty("Position")!.SetValue(policy, position);
}