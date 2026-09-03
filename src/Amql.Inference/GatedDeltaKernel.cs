namespace Amql.Inference;

/// <summary>
/// The caller-owned per-layer state of a linear-attention (GatedDeltaNet)
/// layer: the per-(key)-head recurrence state S ∈ R^{headKDim × headVDim}
/// and the causal-conv history window. Advances position by position —
/// prefill of a stateful plan therefore runs position-major.
/// </summary>
public sealed class LinearAttentionState
{
    /// <summary>[head][kDim, vDim] — the delta-rule state.</summary>
    public float[][,] S { get; }

    /// <summary>[channel, kernel-1] — the causal conv history (zero-padded,
    /// index 0 = the most recent input).</summary>
    public float[,] ConvHistory { get; }

    public LinearAttentionState(int heads, int kDim, int vDim, int channels, int convKernel)
    {
        S = new float[heads][,];
        for (int h = 0; h < heads; h++)
        {
            S[h] = new float[kDim, vDim];
        }
        ConvHistory = new float[channels, Math.Max(0, convKernel - 1)];
    }
}

/// <summary>
/// The managed port of the reference <c>Qwen3_5GatedDeltaNet</c>: a
/// depthwise causal conv1d (kernel K, left pad K-1, SiLU activation) over
/// the qkv stream, then the gated delta-rule recurrence — per key head,
/// <c>S ← exp(g)·S + k ⊗ (β·(v − Sᵀk))</c>, output <c>Sᵀq</c> — with Q/K
/// L2-normalised (eps 1e-6), a z-gated RMSNorm over the value head dim,
/// and the output projection. All arithmetic in f32, mirroring the
/// reference's float32 computation. No positional rotation is applied by
/// this operator.
/// </summary>
public static class GatedDeltaKernel
{
    public const double QkL2NormEps = 1e-6;

    // ── causal conv over the qkv stream ─────────────────────────────────────

    /// <summary>Runs the depthwise causal conv over a stream
    /// [channels × T] with weight [channels × kernel] (channel-major,
    /// kernel innermost), zero-padding the left by kernel-1, optionally
    /// SiLU-activated. The conv history is advanced to the tail of the
    /// sequence so a following step continues causally.</summary>
    public static float[,] ConvForward(float[,] stream, float[] convWeight, int kernel, bool silu, LinearAttentionState state)
    {
        int channels = stream.GetLength(0);
        int t = stream.GetLength(1);
        int historyLen = kernel - 1;
        var output = new float[channels, t];

        for (int c = 0; c < channels; c++)
        {
            for (int pos = 0; pos < t; pos++)
            {
                float acc = 0f;
                for (int k = 0; k < kernel; k++)
                {
                    int idx = pos - (kernel - 1) + k; // relative input index
                    float x;
                    if (idx < 0)
                    {
                        // History slot 0 = the most recent past input (idx -1).
                        x = state.ConvHistory[c, -idx - 1];
                    }
                    else
                    {
                        x = stream[c, idx];
                    }
                    acc += convWeight[c * kernel + k] * x;
                }
                output[c, pos] = silu ? Silu(acc) : acc;
            }
        }

        // History becomes the last (kernel-1) inputs, most recent first.
        for (int c = 0; c < channels; c++)
        {
            for (int k = 0; k < historyLen; k++)
            {
                int source = t - 1 - k;
                state.ConvHistory[c, k] = source >= 0 ? stream[c, source] : 0f;
            }
        }
        return output;
    }

    /// <summary>Single-position causal conv (decode step); stream is
    /// [channels × 1] and the history window is rotated in place.</summary>
    public static float[,] ConvStep(float[,] stream, float[] convWeight, int kernel, bool silu, LinearAttentionState state)
    {
        int channels = stream.GetLength(0);
        int historyLen = kernel - 1;
        var output = new float[channels, 1];

        for (int c = 0; c < channels; c++)
        {
            float acc = 0f;
            for (int k = 0; k < kernel; k++)
            {
                int idx = k - (kernel - 1); // relative input index (≤ 0 at decode)
                float x;
                if (idx < 0)
                {
                    x = state.ConvHistory[c, -idx - 1];
                }
                else
                {
                    x = stream[c, 0];
                }
                acc += convWeight[c * kernel + k] * x;
            }
            output[c, 0] = silu ? Silu(acc) : acc;
        }

        for (int c = 0; c < channels; c++)
        {
            for (int k = historyLen - 1; k >= 1; k--)
            {
                state.ConvHistory[c, k] = state.ConvHistory[c, k - 1];
            }
            state.ConvHistory[c, 0] = stream[c, 0];
        }
        return output;
    }

    // ── gated delta-rule recurrence ─────────────────────────────────────────

    /// <summary>
    /// The reference <c>torch_recurrent_gated_delta_rule</c>: a position
    /// loop over the sequence. q/k/v are [T, vHeads, headK/headV dim]
    /// (key heads already expanded to the value-head count); g and beta
    /// are [T, vHeads] with <c>g = -exp(A_log)·softplus(a + dt_bias)</c> and
    /// <c>beta = sigmoid(b)</c>. Q/K are L2-normalised (eps 1e-6), q
    /// scaled by 1/√headKDim. The caller-owned state advances in place;
    /// output rows are [T, vHeads × headVDim] (head-major). f32.
    /// </summary>
    public static Tensor2D Recurrent(
        Tensor2D q,
        Tensor2D k,
        Tensor2D v,
        Tensor2D g,
        Tensor2D beta,
        int heads,
        int kDim,
        int vDim,
        LinearAttentionState? state)
    {
        int t = q.Rows;
        double scale = 1.0 / Math.Sqrt(kDim);

        // L2-normalise q and k per row-head slice (the reference applies
        // l2norm before transposing into heads).
        var qn = new float[q.Data.Length];
        var kn = new float[k.Data.Length];
        for (int row = 0; row < t; row++)
        {
            for (int h = 0; h < heads; h++)
            {
                L2NormRow(q.Data, row * q.Cols + h * kDim, kDim, qn, row * q.Cols + h * kDim);
                L2NormRow(k.Data, row * k.Cols + h * kDim, kDim, kn, row * k.Cols + h * kDim);
            }
        }

        var S = state?.S ?? new float[heads][,];
        var output = new float[t * heads * vDim];

        for (int pos = 0; pos < t; pos++)
        {
            for (int h = 0; h < heads; h++)
            {
                double decay = Math.Exp(g.Data[pos * g.Cols + h]);
                float betaV = beta.Data[pos * beta.Cols + h];

                int kBase = pos * k.Cols + h * kDim;
                int vBase = pos * v.Cols + h * vDim;

                // kv_mem = S^T k — the state's prediction of v.
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

                // S = decay·S + k ⊗ (β·(v − kvMem))
                for (int kk = 0; kk < kDim; kk++)
                {
                    float kVal = kn[kBase + kk];
                    for (int d = 0; d < vDim; d++)
                    {
                        float delta = (v.Data[vBase + d] - kvMem[d]) * betaV;
                        S[h][kk, d] = (float)(S[h][kk, d] * decay) + kVal * delta;
                    }
                }

                // out = S^T q · scale
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

    // ── z-gated RMSNorm over the value head dim ─────────────────────────────

    /// <summary>The reference <c>Qwen3_5RMSNormGated</c> over rows of
    /// width <c>headVDim</c> (positions × heads flattened):
    /// <c>x̂ × weight × silu(z)</c>. The weight applies directly — no
    /// 1+w offset — unlike the layer norms.</summary>
    public static Tensor2D GatedRMSNorm(Tensor2D x, Tensor2D z, float[] weight, double eps)
    {
        int rows = x.Rows;
        int width = x.Cols;
        var output = new float[x.Data.Length];
        for (int r = 0; r < rows; r++)
        {
            double sumSquares = 0;
            for (int i = 0; i < width; i++)
            {
                sumSquares += (double)x.Data[r * width + i] * x.Data[r * width + i];
            }
            double inv = 1.0 / Math.Sqrt(sumSquares / width + eps);
            for (int i = 0; i < width; i++)
            {
                output[r * width + i] = (float)(x.Data[r * width + i] * inv * weight[i] * Silu(z.Data[r * width + i]));
            }
        }
        return new Tensor2D(output, rows, width);
    }

    // ── helpers ────────────────────────────────────────────────────────────

    private static void L2NormRow(float[] data, int offset, int width, float[] target, int targetOffset)
    {
        double sumSquares = 0;
        for (int i = 0; i < width; i++)
        {
            sumSquares += (double)data[offset + i] * data[offset + i];
        }
        double inv = 1.0 / Math.Sqrt(sumSquares + QkL2NormEps);
        for (int i = 0; i < width; i++)
        {
            target[targetOffset + i] = (float)(data[offset + i] * inv);
        }
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));
}