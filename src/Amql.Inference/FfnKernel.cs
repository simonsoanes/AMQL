using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// FFN operations. Dense: single projection <c>activation(x@up)</c> or the
/// gate/up/down shape <c>activation(x@gate) ⊙ (x@up)</c>, activated by the
/// surface's <c>Activation</c>. Routed: top-k softmax router over
/// per-expert gate/up/down projections, with the routing policy deciding
/// whether the selected weights are renormalised to sum to 1.
/// </summary>
public static class FfnKernel
{
    /// <summary>Dense FFN forward. When <c>gate</c> is null the FFN is the
    /// plain single-projection form.</summary>
    public static Tensor2D Dense(Tensor2D x, Tensor2D? gate, Tensor2D up, Tensor2D down,
        Activation activation, bool gated,
        float[]? gateBias = null, float[]? upBias = null, float[]? downBias = null)
    {
        var act = Activations.For(activation, gated);
        if (gated && gate is not null && activation == Activation.Silu && gateBias is null && upBias is null &&
            downBias is null && CudaShim.Enabled && CudaShim.CanFuse(gate, up, down))
        {
            // All three on the GPU: one call, the activations up and the
            // output down once, nothing in between.
            var all = new int[x.Rows];
            var ones = new float[x.Rows];
            for (int r = 0; r < all.Length; r++)
            {
                all[r] = r;
                ones[r] = 1f;
            }
            if (CudaShim.TryFfn(x.Data, x.Rows, x.Cols, down.Rows,
                    new[] { new CudaShim.FfnJob(all, ones, gate, up, down) }, out var fused))
            {
                return new Tensor2D(fused!, x.Rows, down.Rows);
            }
        }
        Tensor2D hidden;
        if (gated && gate is not null)
        {
            // GPU-batched path: gate and up share the same input — launch
            // both GEMMs on the device in one batch, then sync once.
            Tensor2D[] pair;
            if (CudaShim.Enabled &&
                gate.DeviceWeightF16 != IntPtr.Zero &&
                up.DeviceWeightF16 != IntPtr.Zero)
            {
                pair = TensorOps.MatMulTransposedBMulti(x, new[] { gate, up });
            }
            else
            {
                pair = TensorOps.MatMulTransposedBMany(x, gate, up);
            }
            var g = pair[0];
            var u = pair[1];
            AddRowBias(g, gateBias);
            AddRowBias(u, upBias);
            hidden = Tensor2D.Zeros(x.Rows, g.Cols);
            var gData = g.Data;
            var uData = u.Data;
            var hData = hidden.Data;
            // Element-wise act(g)·u: independent elements, split across the
            // pool when wide. The same scalar function per element.
            const int Span = 1024;
            int spans = (hData.Length + Span - 1) / Span;
            if (spans > 1)
            {
                KernelPool.For(spans, sIdx =>
                {
                    int end = Math.Min(hData.Length, (sIdx + 1) * Span);
                    for (int i = sIdx * Span; i < end; i++)
                    {
                        hData[i] = act(gData[i]) * uData[i];
                    }
                });
            }
            else
            {
                for (int i = 0; i < hData.Length; i++)
                {
                    hData[i] = act(gData[i]) * uData[i];
                }
            }
        }
        else
        {
            hidden = TensorOps.MatMulTransposedB(x, up);
            AddRowBias(hidden, upBias);
            for (int r = 0; r < hidden.Rows; r++)
            {
                for (int c = 0; c < hidden.Cols; c++)
                {
                    hidden.Set(r, c, act(hidden[r, c]));
                }
            }
        }
        var result = TensorOps.MatMulTransposedB(hidden, down);
        AddRowBias(result, downBias);
        return result;
    }

    /// <summary>Adds a per-column bias to every row, in place; no-op for null.</summary>
    public static void AddRowBias(Tensor2D x, float[]? bias)
    {
        if (bias is null)
        {
            return;
        }
        for (int r = 0; r < x.Rows; r++)
        {
            var row = x.Row(r);
            for (int c = 0; c < row.Length; c++)
            {
                row[c] += bias[c];
            }
        }
    }

    /// <summary>Routed MoE forward. <c>expertGate/Up/Down</c> are per-expert
    /// matrices indexed by expert; <c>router</c> is
    /// [numExperts, hidden].</summary>
    public static Tensor2D Routed(
        Tensor2D x,
        Tensor2D router,
        IReadOnlyList<Tensor2D> expertGate,
        IReadOnlyList<Tensor2D> expertUp,
        IReadOnlyList<Tensor2D> expertDown,
        int topK,
        ExpertRoutingPolicy routingPolicy,
        Activation activation,
        Action<int, IReadOnlyList<int>>? onExpertsRouted = null) =>
        Routed(x, router, expertGate.Count, expertDown[0].Rows,
            e => (expertGate[e], expertUp[e], expertDown[e]),
            topK, routingPolicy, activation, onExpertsRouted);

    /// <summary>How many f32 bytes of per-expert outputs one pass of
    /// <see cref="Routed(Tensor2D, Tensor2D, int, int, Func{int, ValueTuple{Tensor2D, Tensor2D, Tensor2D}}, int, ExpertRoutingPolicy, Activation, Action{int, IReadOnlyList{int}}?)"/>
    /// holds; a longer prompt is routed in row chunks that fit.</summary>
    private const long RoutedChunkBytes = 64L << 20;

    /// <summary>
    /// Routed MoE forward that asks for an expert's weights only when the
    /// router has picked it. Every row is routed first; then each picked
    /// expert is fetched once and runs over all the rows that chose it, and
    /// the rows add their experts' outputs back in the order they picked
    /// them. A per-row dot does not depend on how many rows share the GEMM,
    /// so the result matches routing one row at a time, while a model whose
    /// experts do not fit in memory only ever reads the ones in use.
    /// </summary>
    public static Tensor2D Routed(
        Tensor2D x,
        Tensor2D router,
        int numExperts,
        int outputWidth,
        Func<int, (Tensor2D Gate, Tensor2D Up, Tensor2D Down)> expert,
        int topK,
        ExpertRoutingPolicy routingPolicy,
        Activation activation,
        Action<int, IReadOnlyList<int>>? onExpertsRouted = null,
        Action<(Tensor2D Gate, Tensor2D Up, Tensor2D Down)>? release = null)
    {
        if (topK > numExperts)
        {
            throw new UnsupportedOperatorException(
                $"routed FFN top-k ({topK}) exceeds expert count ({numExperts})");
        }

        // Router scores: [rows, numExperts].
        var scores = TensorOps.MatMulTransposedB(x, router);
        var output = Tensor2D.Zeros(x.Rows, outputWidth);
        var selected = new int[x.Rows * topK];
        var weights = new float[x.Rows * topK];

        for (int r = 0; r < x.Rows; r++)
        {
            // Softmax over all experts.
            var row = scores.Row(r).ToArray();
            float max = float.NegativeInfinity;
            for (int e = 0; e < numExperts; e++)
            {
                if (row[e] > max)
                {
                    max = row[e];
                }
            }
            float sum = 0f;
            for (int e = 0; e < numExperts; e++)
            {
                row[e] = MathF.Exp(row[e] - max);
                sum += row[e];
            }
            float inv = 1f / sum;
            for (int e = 0; e < numExperts; e++)
            {
                row[e] *= inv;
            }

            // Select the top-k: highest first, an earlier expert first among
            // equals (what a stable descending sort gives).
            var picked = new int[topK];
            int found = 0;
            for (int e = 0; e < numExperts; e++)
            {
                int at = found;
                while (at > 0 && row[e] > row[picked[at - 1]])
                {
                    at--;
                }
                if (at >= topK)
                {
                    continue;
                }
                int last = Math.Min(found, topK - 1);
                for (int i = last; i > at; i--)
                {
                    picked[i] = picked[i - 1];
                }
                picked[at] = e;
                found = Math.Min(found + 1, topK);
            }
            onExpertsRouted?.Invoke(r, picked);

            if (routingPolicy == ExpertRoutingPolicy.NormalisedOverSelected)
            {
                float selectedSum = 0f;
                foreach (var e in picked)
                {
                    selectedSum += row[e];
                }
                if (selectedSum > 0f)
                {
                    foreach (var e in picked)
                    {
                        row[e] /= selectedSum;
                    }
                }
            }
            for (int k = 0; k < topK; k++)
            {
                selected[r * topK + k] = picked[k];
                weights[r * topK + k] = row[picked[k]];
            }
        }

        int cols = x.Cols;
        int chunkRows = (int)Math.Clamp(RoutedChunkBytes / Math.Max(1L, (long)topK * outputWidth * 4), 1, Math.Max(1, x.Rows));
        for (int first = 0; first < x.Rows; first += chunkRows)
        {
            int rows = Math.Min(chunkRows, x.Rows - first);
            // (row, slot) pairs per expert, in row order.
            var users = new SortedDictionary<int, List<int>>();
            for (int i = first * topK; i < (first + rows) * topK; i++)
            {
                if (weights[i] == 0f)
                {
                    continue;
                }
                if (!users.TryGetValue(selected[i], out var list))
                {
                    users[selected[i]] = list = new List<int>();
                }
                list.Add(i);
            }

            // Per-slot outputs of the experts run here; not needed (or made)
            // when every picked expert runs on the device.
            float[]? slotOut = null;
            var hostSlot = new bool[rows * topK];
            // Experts held on the GPU run together in one fused call (their
            // rows go up once, their weighted sum comes down once); the rest
            // run here as they come.
            var onDevice = new List<(List<int> Slots, (Tensor2D Gate, Tensor2D Up, Tensor2D Down) Weights)>();
            foreach (var (e, slots) in users)
            {
                if (activation == Activation.Silu && CudaShim.Enabled)
                {
                    var candidate = expert(e);
                    if (CudaShim.CanFuse(candidate.Gate, candidate.Up, candidate.Down))
                    {
                        onDevice.Add((slots, candidate));
                        continue;
                    }
                    ComputeOnHost(e, slots, candidate);
                    continue;
                }
                ComputeOnHost(e, slots, null);
            }
            float[]? deviceSum = onDevice.Count == 0 ? null : FusedExperts(x, first, rows, topK, weights, onDevice, outputWidth);
            if (deviceSum is null)
            {
                foreach (var (slots, held) in onDevice)
                {
                    ComputeOnHost(-1, slots, held); // the fused call failed: run them here
                }
            }
            else
            {
                foreach (var (_, held) in onDevice)
                {
                    release?.Invoke(held);
                }
            }

            void ComputeOnHost(int e, List<int> slots, (Tensor2D Gate, Tensor2D Up, Tensor2D Down)? held)
            {
                var input = new float[slots.Count * cols];
                for (int s = 0; s < slots.Count; s++)
                {
                    x.Row(slots[s] / topK).CopyTo(input.AsSpan(s * cols, cols));
                }
                var weightsOf = held ?? expert(e);
                var (gate, up, down) = weightsOf;
                var expertOut = Dense(new Tensor2D(input, slots.Count, cols), gate, up, down, activation, true);
                release?.Invoke(weightsOf);
                for (int s = 0; s < slots.Count; s++)
                {
                    long at = (long)(slots[s] - first * topK) * outputWidth;
                    slotOut ??= new float[(long)rows * topK * outputWidth];
                    expertOut.Data.AsSpan(s * outputWidth, outputWidth).CopyTo(slotOut.AsSpan((int)at, outputWidth));
                    hostSlot[slots[s] - first * topK] = true;
                }
            }

            for (int r = first; r < first + rows; r++)
            {
                var outRow = output.Data.AsSpan(r * outputWidth, outputWidth);
                for (int k = 0; k < topK; k++)
                {
                    float weight = weights[r * topK + k];
                    if (weight == 0f || !hostSlot[(r - first) * topK + k])
                    {
                        continue;
                    }
                    var add = slotOut!.AsSpan((r - first) * topK * outputWidth + k * outputWidth, outputWidth);
                    for (int c = 0; c < outputWidth; c++)
                    {
                        outRow[c] += weight * add[c];
                    }
                }
                if (deviceSum is not null)
                {
                    var add = deviceSum.AsSpan((r - first) * outputWidth, outputWidth);
                    for (int c = 0; c < outputWidth; c++)
                    {
                        outRow[c] += add[c];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>The weighted sum, for rows <c>[first, first + rows)</c>, of the
    /// experts held on the GPU, run as one fused device call; null if it failed.</summary>
    private static float[]? FusedExperts(Tensor2D x, int first, int rows, int topK, float[] weights,
        List<(List<int> Slots, (Tensor2D Gate, Tensor2D Up, Tensor2D Down) Weights)> onDevice, int outputWidth)
    {
        var chunk = x.Data.AsSpan(first * x.Cols, rows * x.Cols).ToArray();
        var jobs = new List<CudaShim.FfnJob>(onDevice.Count);
        foreach (var (slots, held) in onDevice)
        {
            var jobRows = new int[slots.Count];
            var jobWeights = new float[slots.Count];
            for (int s = 0; s < slots.Count; s++)
            {
                jobRows[s] = slots[s] / topK - first;
                jobWeights[s] = weights[slots[s]];
            }
            jobs.Add(new CudaShim.FfnJob(jobRows, jobWeights, held.Gate, held.Up, held.Down));
        }
        return CudaShim.TryFfn(chunk, rows, x.Cols, outputWidth, jobs, out var sum) ? sum : null;
    }
}
