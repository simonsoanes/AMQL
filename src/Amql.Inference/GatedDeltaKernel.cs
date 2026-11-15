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
/// <c>S ← exp(g)·S</c>, then <c>S ← S + k ⊗ (β·(v − Sᵀk))</c>, output
/// <c>Sᵀq</c> — with Q/K
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
    /// <summary>
    /// <see cref="ConvStep"/> over a flat single-position stream (one value
    /// per channel, as the in-projection produced it), returning a flat
    /// per-channel output. Each channel's arithmetic and history update are
    /// <see cref="ConvStep"/>'s exactly; channels are independent, so wide
    /// streams split across the kernel pool with identical results.
    /// </summary>
    public static float[] ConvStepFlat(ReadOnlySpan<float> stream, float[] convWeight, int kernel, bool silu, LinearAttentionState state)
    {
        int channels = state.ConvHistory.GetLength(0);
        if (stream.Length < channels)
        {
            throw new ArgumentException($"stream holds {stream.Length} values for {channels} channels", nameof(stream));
        }
        var input = stream.Slice(0, channels).ToArray();
        var output = new float[channels];
        var history = state.ConvHistory;
        int historyLen = kernel - 1;

        void Channels(int c0, int c1)
        {
            var hist = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
                ref System.Runtime.CompilerServices.Unsafe.As<byte, float>(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(history)),
                channels * historyLen);
            for (int c = c0; c < c1; c++)
            {
                var h = hist.Slice(c * historyLen, historyLen);
                float acc = 0f;
                for (int k = 0; k < kernel; k++)
                {
                    int idx = k - (kernel - 1); // relative input index (≤ 0 at decode)
                    float x = idx < 0 ? h[-idx - 1] : input[c];
                    acc += convWeight[c * kernel + k] * x;
                }
                output[c] = silu ? Silu(acc) : acc;
                for (int k = historyLen - 1; k >= 1; k--)
                {
                    h[k] = h[k - 1];
                }
                if (historyLen > 0)
                {
                    h[0] = input[c];
                }
            }
        }

        const int ChannelsPerTask = 512;
        int tasks = (channels + ChannelsPerTask - 1) / ChannelsPerTask;
        if (tasks > 1)
        {
            KernelPool.For(tasks, t => Channels(t * ChannelsPerTask, Math.Min(channels, (t + 1) * ChannelsPerTask)));
        }
        else
        {
            Channels(0, channels);
        }
        return output;
    }

    /// <summary>
    /// <see cref="ConvForward"/> over a row-major [t × channels] stream (the
    /// in-projection's layout), returning row-major [t × channels]. Each
    /// (channel, position) sum and the history update are
    /// <see cref="ConvForward"/>'s exactly — including its history rule for a
    /// batch shorter than the kernel — and channels are independent, so they
    /// split across the kernel pool with identical results. Skips the
    /// column-major copies in and out.
    /// </summary>
    public static float[] ConvForwardRows(float[] stream, int t, float[] convWeight, int kernel, bool silu, LinearAttentionState state)
    {
        int channels = state.ConvHistory.GetLength(0);
        if (stream.Length < (long)t * channels)
        {
            throw new ArgumentException($"stream holds {stream.Length} values for {t}x{channels}", nameof(stream));
        }
        var output = GC.AllocateUninitializedArray<float>(t * channels);
        var history = state.ConvHistory;
        int historyLen = kernel - 1;

        void Channels(int c0, int c1)
        {
            var hist = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
                ref System.Runtime.CompilerServices.Unsafe.As<byte, float>(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(history)),
                channels * historyLen);
            for (int c = c0; c < c1; c++)
            {
                var h = hist.Slice(c * historyLen, historyLen);
                for (int pos = 0; pos < t; pos++)
                {
                    float acc = 0f;
                    for (int k = 0; k < kernel; k++)
                    {
                        int idx = pos - (kernel - 1) + k; // relative input index
                        // History slot 0 = the most recent past input (idx -1).
                        float x = idx < 0 ? h[-idx - 1] : stream[idx * channels + c];
                        acc += convWeight[c * kernel + k] * x;
                    }
                    output[pos * channels + c] = silu ? Silu(acc) : acc;
                }
                // History becomes the last (kernel-1) inputs, most recent first.
                for (int k = 0; k < historyLen; k++)
                {
                    int source = t - 1 - k;
                    h[k] = source >= 0 ? stream[source * channels + c] : 0f;
                }
            }
        }

        const int ChannelsPerTask = 256;
        int tasks = (channels + ChannelsPerTask - 1) / ChannelsPerTask;
        if (tasks > 1 && (long)t * channels >= 8192)
        {
            KernelPool.For(tasks, task => Channels(task * ChannelsPerTask, Math.Min(channels, (task + 1) * ChannelsPerTask)));
        }
        else
        {
            Channels(0, channels);
        }
        return output;
    }

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
        // q and k are L2-normalised per (row, head) slice inside each head's
        // task — the slices are disjoint, so this is the same values as a
        // separate pass over every row first.
        var qn = new float[q.Data.Length];
        var kn = new float[k.Data.Length];

        var S = state?.S ?? NewState(heads, kDim, vDim);
        var output = new float[t * heads * vDim];

        // Heads are independent, and within a head so is every value-dim
        // column of the state: column d's prediction, delta, update and output
        // read only column d. A long batch therefore splits each head into
        // column groups too, so a 16-head layer keeps more than 16 threads
        // busy through its sequential positions. Each element's arithmetic is
        // unchanged, so the result is identical however it is split.
        int fw = System.Numerics.Vector<float>.Count;
        int groups = 1;
        if (t >= 8)
        {
            int wanted = Math.Max(1, (KernelPool.Parallelism + heads - 1) / Math.Max(1, heads));
            foreach (int candidate in new[] { 8, 4, 2 })
            {
                if (candidate <= wanted && vDim % (candidate * fw) == 0)
                {
                    groups = candidate;
                    break;
                }
            }
        }
        if (groups > 1)
        {
            KernelPool.For(heads, h => NormaliseHead(h, t, kDim, q, qn, k, kn));
            int width = vDim / groups;
            KernelPool.For(heads * groups, task =>
            {
                int h = task / groups;
                int d0 = (task % groups) * width;
                RecurrentHead(h, t, kDim, vDim, heads, scale, S[h], qn, q.Cols, kn, k.Cols, v, g, beta, output, d0, d0 + width);
            });
        }
        else if (heads > 1 && (long)t * kDim * vDim * heads >= 64L << 10)
        {
            KernelPool.For(heads, h =>
            {
                NormaliseHead(h, t, kDim, q, qn, k, kn);
                RecurrentHead(h, t, kDim, vDim, heads, scale, S[h], qn, q.Cols, kn, k.Cols, v, g, beta, output, 0, vDim);
            });
        }
        else
        {
            for (int h = 0; h < heads; h++)
            {
                NormaliseHead(h, t, kDim, q, qn, k, kn);
                RecurrentHead(h, t, kDim, vDim, heads, scale, S[h], qn, q.Cols, kn, k.Cols, v, g, beta, output, 0, vDim);
            }
        }

        return new Tensor2D(output, t, heads * vDim);
    }

    private static void NormaliseHead(int h, int t, int kDim, Tensor2D q, float[] qn, Tensor2D k, float[] kn)
    {
        for (int row = 0; row < t; row++)
        {
            L2NormRow(q.Data, row * q.Cols + h * kDim, kDim, qn, row * q.Cols + h * kDim);
            L2NormRow(k.Data, row * k.Cols + h * kDim, kDim, kn, row * k.Cols + h * kDim);
        }
    }

    private static float[][,] NewState(int heads, int kDim, int vDim)
    {
        var state = new float[heads][,];
        for (int h = 0; h < heads; h++)
        {
            state[h] = new float[kDim, vDim];
        }
        return state;
    }

    /// <summary>
    /// One head of the recurrence over every position, for the value-dim
    /// columns [d0, d1), vectorised along those columns. The operation
    /// sequence per element is the scalar reference's exactly — decay in
    /// double then rounded to float; the S·k and S·q reductions as float
    /// products accumulated in double, in increasing key index; the update as
    /// a float multiply then add — so the result is bit-identical to it. The
    /// two reductions are fused into the row sweeps: kv_mem reads each state
    /// row just after it decays, and the output reads each row just after its
    /// (single) update. Column ranges never interact, so a head can be split
    /// across tasks by columns.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    private static void RecurrentHead(int h, int t, int kDim, int vDim, int heads, double scale,
        float[,] state, float[] qn, int qCols, float[] kn, int kCols,
        Tensor2D v, Tensor2D g, Tensor2D beta, float[] output, int d0, int d1)
    {
        var sAll = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref System.Runtime.CompilerServices.Unsafe.As<byte, float>(
                ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(state)),
            kDim * vDim);
        int width = d1 - d0;
        // A column group works on a private, contiguous copy of its columns:
        // two groups of one head on different cores would otherwise write
        // neighbouring floats of the same state rows (false sharing on every
        // row of every step). Copied in once, back once — values unchanged.
        float[]? local = null;
        int stride = vDim;
        int colOffset = d0;
        if (width != vDim)
        {
            local = new float[kDim * width];
            for (int kk = 0; kk < kDim; kk++)
            {
                sAll.Slice(kk * vDim + d0, width).CopyTo(local.AsSpan(kk * width, width));
            }
            sAll = local;
            stride = width;
            colOffset = 0;
        }
        int fw = System.Numerics.Vector<float>.Count;
        int vecEnd = width - width % fw;
        int half = fw / 2;
        Span<double> acc = width <= 512 ? stackalloc double[width] : new double[width];
        Span<float> delta = width <= 512 ? stackalloc float[width] : new float[width];

        for (int pos = 0; pos < t; pos++)
        {
            double decay = Math.Exp(g.Data[pos * g.Cols + h]);
            float betaV = beta.Data[pos * beta.Cols + h];
            int kBase = pos * kCols + h * kDim;
            int qBase = pos * qCols + h * kDim;
            int vBase = pos * v.Cols + h * vDim + d0;
            var decayV = new System.Numerics.Vector<double>(decay);

            // S ← exp(g)·S, and kv_mem = Sᵀk over the decayed rows.
            acc.Clear();
            for (int kk = 0; kk < kDim; kk++)
            {
                var row = sAll.Slice(kk * stride + colOffset, width);
                float kVal = kn[kBase + kk];
                var kV = new System.Numerics.Vector<float>(kVal);
                int d = 0;
                for (; d < vecEnd; d += fw)
                {
                    var sv = new System.Numerics.Vector<float>(row.Slice(d, fw));
                    System.Numerics.Vector.Widen(sv, out var lo, out var hi);
                    sv = System.Numerics.Vector.Narrow(lo * decayV, hi * decayV);
                    sv.CopyTo(row.Slice(d, fw));
                    System.Numerics.Vector.Widen(sv * kV, out var plo, out var phi);
                    var accSpan = acc.Slice(d, fw);
                    (new System.Numerics.Vector<double>(accSpan.Slice(0, half)) + plo).CopyTo(accSpan.Slice(0, half));
                    (new System.Numerics.Vector<double>(accSpan.Slice(half, half)) + phi).CopyTo(accSpan.Slice(half, half));
                }
                for (; d < width; d++)
                {
                    row[d] = (float)(row[d] * decay);
                    acc[d] += row[d] * kVal;
                }
            }

            // δ = β·(v − kv_mem)
            for (int d = 0; d < width; d++)
            {
                delta[d] = (v.Data[vBase + d] - (float)acc[d]) * betaV;
            }

            // S ← S + k ⊗ δ, and out = Sᵀq over the updated rows.
            acc.Clear();
            for (int kk = 0; kk < kDim; kk++)
            {
                var row = sAll.Slice(kk * stride + colOffset, width);
                float kVal = kn[kBase + kk];
                float qVal = qn[qBase + kk];
                var kV = new System.Numerics.Vector<float>(kVal);
                var qV = new System.Numerics.Vector<float>(qVal);
                int d = 0;
                for (; d < vecEnd; d += fw)
                {
                    var sv = new System.Numerics.Vector<float>(row.Slice(d, fw));
                    var dv = new System.Numerics.Vector<float>(delta.Slice(d, fw));
                    sv += kV * dv;
                    sv.CopyTo(row.Slice(d, fw));
                    System.Numerics.Vector.Widen(sv * qV, out var plo, out var phi);
                    var accSpan = acc.Slice(d, fw);
                    (new System.Numerics.Vector<double>(accSpan.Slice(0, half)) + plo).CopyTo(accSpan.Slice(0, half));
                    (new System.Numerics.Vector<double>(accSpan.Slice(half, half)) + phi).CopyTo(accSpan.Slice(half, half));
                }
                for (; d < width; d++)
                {
                    row[d] += kVal * delta[d];
                    acc[d] += row[d] * qVal;
                }
            }

            int oBase = pos * heads * vDim + h * vDim + d0;
            for (int d = 0; d < width; d++)
            {
                output[oBase + d] = (float)(acc[d] * scale);
            }
        }

        if (local is not null)
        {
            var shared = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
                ref System.Runtime.CompilerServices.Unsafe.As<byte, float>(
                    ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(state)),
                kDim * vDim);
            for (int kk = 0; kk < kDim; kk++)
            {
                local.AsSpan(kk * width, width).CopyTo(shared.Slice(kk * vDim + d0, width));
            }
        }
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
        var output = GC.AllocateUninitializedArray<float>(x.Data.Length);
        var xData = x.Data;
        var zData = z.Data;
        void Row(int r)
        {
            double sumSquares = 0;
            for (int i = 0; i < width; i++)
            {
                sumSquares += (double)xData[r * width + i] * xData[r * width + i];
            }
            double inv = 1.0 / Math.Sqrt(sumSquares / width + eps);
            for (int i = 0; i < width; i++)
            {
                output[r * width + i] = (float)(xData[r * width + i] * inv * weight[i] * Silu(zData[r * width + i]));
            }
        }
        // Rows are independent; the pool only changes which thread runs each.
        if (rows >= 8 && (long)rows * width >= 4096)
        {
            KernelPool.For(rows, Row);
        }
        else
        {
            for (int r = 0; r < rows; r++)
            {
                Row(r);
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