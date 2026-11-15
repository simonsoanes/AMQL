using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Rotary position embedding. The general entry computes
/// <c>inv_freq[i] = 1 / theta^(2i/rotary_dim)</c> and rotates the two
/// halves of each rotary pair by <c>angle = position · inv_freq[i]</c>.
/// The rotation covers the first <c>rotaryWidth</c> dims of each head:
/// full-width for <see cref="PositionRope"/>, the partial fraction for
/// <see cref="PositionPartialRope"/>. Scaled frequencies come from
/// <see cref="RopeScaling.Frequencies"/>; the length-dependent kinds
/// (longrope, dynamic) read the sequence length as the largest position in
/// the call plus one, as transformers does. MRoPE takes each pair's position
/// from its stream when streams are given; text tokens carry one position in
/// every stream, which is plain rotary.
/// </summary>
public static class Rope
{
    /// <summary>
    /// Applies rotary embedding to a tensor of shape
    /// [rows, numHeads * headDim], in place, using the given absolute
    /// position per row. When <c>positionPolicy</c> is
    /// <see cref="PositionNone"/> this is a no-op.
    /// </summary>
    public static void Apply(Tensor2D x, int numHeads, int headDim,
        PositionPolicy positionPolicy, ReadOnlySpan<int> positions, bool interleaved = false,
        int[][]? streams = null)
    {
        int rotaryWidth;
        double theta;
        RopeScaling? scaling = null;
        RopeMultimodal? mrope = null;
        switch (positionPolicy)
        {
            case PositionNone:
                return;
            case PositionZeroTailRope zeroTail when !interleaved:
                ApplyZeroTail(x, numHeads, headDim, zeroTail, positions);
                return;
            case PositionRope rope:
                rotaryWidth = headDim;
                theta = rope.Theta;
                scaling = rope.Scaling;
                mrope = rope.Mrope;
                break;
            case PositionPartialRope partial:
                rotaryWidth = partial.RotaryWidth(headDim);
                theta = partial.Theta;
                scaling = partial.Scaling;
                mrope = partial.Mrope;
                break;
            case PositionZeroTailRope:
                throw new UnsupportedOperatorException("zero-tail rope with interleaved pairs has no managed implementation");
            default:
                throw new UnsupportedOperatorException(
                    $"position policy '{((PositionUnresolved)positionPolicy).Kind}' has no managed rotary implementation");
        }

        if (rotaryWidth > headDim)
        {
            throw new ArgumentException($"rotary width {rotaryWidth} exceeds head dim {headDim}");
        }

        int pairCount = rotaryWidth / 2;
        if (streams is not null && (mrope is null || streams.Length != mrope.Section.Length ||
                                    streams.Any(s => s.Length != x.Rows)))
        {
            throw new ArgumentException(mrope is null
                ? "position streams were given to a layer without MRoPE"
                : $"MRoPE needs {mrope.Section.Length} position streams of {x.Rows} rows");
        }
        long sequenceLength = 0;
        foreach (int p in positions)
        {
            sequenceLength = Math.Max(sequenceLength, p + 1L);
        }
        if (streams is not null)
        {
            sequenceLength = streams.SelectMany(s => s).Select(p => p + 1L).DefaultIfEmpty(0).Max();
        }
        // Scaled families rescale the frequencies; yarn and longrope also
        // multiply cos/sin by an attention factor.
        var (invFreq, attentionScale) = RopeScaling.Frequencies(theta, rotaryWidth, scaling, sequenceLength);
        var streamOf = streams is null ? null : Enumerable.Range(0, pairCount).Select(mrope!.StreamOf).ToArray();

        // A pair's angle depends on the row and the pair, not the head, so
        // its cos/sin are computed once per row and reused by every head — the
        // same values the per-head computation produced. Rows are
        // independent and split across the pool when there are many.
        var data = x.Data;
        int xCols = x.Cols;
        var positionsCopy = positions.ToArray();
        void RotateRow(int r, Span<double> cosRow, Span<double> sinRow)
        {
            double position = positionsCopy[r];
            for (int i = 0; i < pairCount; i++)
            {
                double angle = (streamOf is null ? position : streams![streamOf[i]][r]) * invFreq[i];
                cosRow[i] = Math.Cos(angle) * attentionScale;
                sinRow[i] = Math.Sin(angle) * attentionScale;
            }
            int rowStart = r * xCols;
            for (int h = 0; h < numHeads; h++)
            {
                int head = h * headDim;
                // The reference pairs the two halves of the rotary window:
                // dims i and i + rotaryWidth/2 rotate together. Interleaved
                // (GPT-J layout) pairs neighbours: 2i and 2i + 1.
                for (int i = 0; i < pairCount; i++)
                {
                    int a = interleaved ? rowStart + head + 2 * i : rowStart + head + i;
                    int b = interleaved ? a + 1 : a + pairCount;
                    double cos = cosRow[i];
                    double sin = sinRow[i];
                    double x1 = data[a];
                    double x2 = data[b];
                    data[a] = (float)(x1 * cos - x2 * sin);
                    data[b] = (float)(x1 * sin + x2 * cos);
                }
            }
        }
        if (x.Rows >= 8)
        {
            KernelPool.For(x.Rows, r =>
            {
                Span<double> c = pairCount <= 256 ? stackalloc double[pairCount] : new double[pairCount];
                Span<double> sn = pairCount <= 256 ? stackalloc double[pairCount] : new double[pairCount];
                RotateRow(r, c, sn);
            });
        }
        else
        {
            Span<double> c = pairCount <= 256 ? stackalloc double[pairCount] : new double[pairCount];
            Span<double> sn = pairCount <= 256 ? stackalloc double[pairCount] : new double[pairCount];
            for (int r = 0; r < x.Rows; r++)
            {
                RotateRow(r, c, sn);
            }
        }
    }

    /// <summary>LFM3's zero-tail partial rotary: pairs (i, i + head_dim/2)
    /// with <c>inv_freq[i] = theta^(-2i/head_dim)</c> for the first
    /// <c>RotatedPairs</c> pairs; the remaining pairs have frequency zero and
    /// pass through unrotated.</summary>
    private static void ApplyZeroTail(Tensor2D x, int numHeads, int headDim,
        PositionZeroTailRope policy, ReadOnlySpan<int> positions)
    {
        int half = headDim / 2;
        int rotated = policy.RotatedPairs(headDim);
        if (rotated > half)
        {
            throw new ArgumentException($"zero-tail rope rotates {rotated} pairs but head dim {headDim} has {half}");
        }
        var invFreq = new double[rotated];
        for (int i = 0; i < rotated; i++)
        {
            invFreq[i] = 1.0 / Math.Pow(policy.Theta, 2.0 * i / headDim);
        }
        for (int r = 0; r < x.Rows; r++)
        {
            double position = positions[r];
            int rowStart = r * x.Cols;
            for (int h = 0; h < numHeads; h++)
            {
                int head = rowStart + h * headDim;
                for (int i = 0; i < rotated; i++)
                {
                    int a = head + i;
                    int b = a + half;
                    double angle = position * invFreq[i];
                    double cos = Math.Cos(angle);
                    double sin = Math.Sin(angle);
                    double x1 = x.Data[a];
                    double x2 = x.Data[b];
                    x.Data[a] = (float)(x1 * cos - x2 * sin);
                    x.Data[b] = (float)(x1 * sin + x2 * cos);
                }
            }
        }
    }
}

/// <summary>
/// GQA softmax attention with the generic parameterisation the surface
/// carries: score scale, optional logit softcapping
/// (<c>tanh(scores/cap)·cap</c>), optional per-head sinks added to
/// position-0 scores, and an optional sliding window (rows outside the
/// window are masked to −∞ before softmax — never evicted, masked).
/// Causal unless the caller says otherwise (a bidirectional stack).
/// </summary>
public static class AttentionKernel
{
    public static void SoftmaxInPlace(Span<float> scores)
    {
        float max = float.NegativeInfinity;
        foreach (var s in scores)
        {
            if (s > max)
            {
                max = s;
            }
        }
        if (float.IsNegativeInfinity(max))
        {
            // Fully masked row — leave zeros.
            scores.Clear();
            return;
        }
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = MathF.Exp(scores[i] - max);
            sum += scores[i];
        }
        float inv = 1f / sum;
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] *= inv;
        }
    }

    /// <summary>
    /// Runs causal GQA attention over queries [seq, numQHeads*headDim] and
    /// key/value rows [kvSeq, numKvHeads*headDim], producing
    /// [seq, numQHeads*headDim]. Every query attends only to rows already
    /// in the KV cache (prefill passes all its rows; decode passes the
    /// cache plus the new row).
    /// </summary>
    public static Tensor2D Execute(
        Tensor2D q,
        Tensor2D k,
        Tensor2D v,
        int numQHeads,
        int numKvHeads,
        int headDim,
        double scoreScale,
        float? logitSoftcapping,
        float[]? sinks,
        int? window,
        int[] queryPositions,
        int[] kvPositions,
        List<(int Head, float[] Weights)>? lastRowCapture = null,
        bool causal = true) =>
        Execute(q, k.Data, k.Cols, v.Data, v.Cols, k.Rows,
            numQHeads, numKvHeads, headDim, scoreScale, logitSoftcapping, sinks, window,
            queryPositions, kvPositions, lastRowCapture, causal);

    /// <summary>
    /// <see cref="Execute(Tensor2D,Tensor2D,Tensor2D,int,int,int,double,float?,float[],int?,int[],int[],List{ValueTuple{int,float[]}},bool)"/>
    /// over key/value rows held in buffers: row j of the keys is
    /// <c>keys[j·keyWidth ..]</c>, for the first <paramref name="kvSeq"/> rows
    /// (the buffers may be longer — a KV cache's spare capacity).
    /// <para>
    /// Every (query row, head) pair is independent, so they run in parallel
    /// unless the last row's weights are being captured. Within a pair the
    /// arithmetic is the sequential order: each score is one <see cref="TensorOps.Dot"/>,
    /// and each output dim accumulates <c>w·v</c> over keys in increasing
    /// order (vectorised across dims, multiply then add), so the result is
    /// bit-identical however the pairs are scheduled.
    /// </para>
    /// </summary>
    public static Tensor2D Execute(
        Tensor2D q,
        float[] keys,
        int keyWidth,
        float[] values,
        int valueWidth,
        int kvSeq,
        int numQHeads,
        int numKvHeads,
        int headDim,
        double scoreScale,
        float? logitSoftcapping,
        float[]? sinks,
        int? window,
        int[] queryPositions,
        int[] kvPositions,
        List<(int Head, float[] Weights)>? lastRowCapture = null,
        bool causal = true)
    {
        int seq = q.Rows;
        int headsRep = numQHeads / numKvHeads;
        var output = Tensor2D.Zeros(seq, numQHeads * headDim);
        int pairs = seq * numQHeads;

        void Pair(int index, float[] weights)
        {
            int qi = index / numQHeads;
            int h = index % numQHeads;
            int qPos = queryPositions[qi];
            int kvHead = h / headsRep;
            var qSlice = q.Data.AsSpan(qi * q.Cols + h * headDim, headDim);
            var outSlice = output.Data.AsSpan(qi * output.Cols + h * headDim, headDim);

            // The causal span is [0, lastValid]: keys are position-ordered,
            // so everything after the first future position is out of the
            // row's reach. The buffer is refilled from scratch each row —
            // no stale values survive into the softmax.
            int lastValid = -1;
            for (int j = 0; j < kvSeq; j++)
            {
                int s = qPos - kvPositions[j];
                if (s < 0 && causal)
                {
                    break; // keys are position-ordered; past the window nothing is valid
                }
                if (window is { } w && s >= w)
                {
                    weights[j] = float.NegativeInfinity;
                    continue;
                }
                var kSlice = keys.AsSpan(j * keyWidth + kvHead * headDim, headDim);
                double score = TensorOps.Dot(qSlice, kSlice) * scoreScale;
                if (logitSoftcapping is { } cap && !double.IsInfinity(score))
                {
                    // softcap: tanh(score/cap) * cap
                    double c = cap;
                    score = Math.Tanh(score / c) * c;
                }
                if (sinks is not null && kvPositions[j] == 0 && sinks.Length > h)
                {
                    score += sinks[h];
                }
                weights[j] = (float)score;
                lastValid = j;
            }
            int spanEnd = lastValid + 1;
            if (spanEnd == 0)
            {
                outSlice.Clear();
                return;
            }

            SoftmaxInPlace(weights.AsSpan(0, spanEnd));
            if (lastRowCapture is not null && qi == seq - 1)
            {
                // Post-softmax attention of the final query row over all
                // its causal keys — the observable A→B link tensors.
                lastRowCapture.Add((h, weights.AsSpan(0, spanEnd).ToArray()));
            }
            // out[d] = Σ_j w_j·v[j, d], j increasing. outSlice starts at +0
            // (the output is zero-initialised and each slice is written by one
            // pair), exactly like the scalar accumulator it replaces.
            int vw = System.Numerics.Vector<float>.Count;
            int vecEnd = headDim - headDim % vw;
            for (int j = 0; j < spanEnd; j++)
            {
                float wj = weights[j];
                if (wj == 0f)
                {
                    continue;
                }
                var vRow = values.AsSpan(j * valueWidth + kvHead * headDim, headDim);
                var wv = new System.Numerics.Vector<float>(wj);
                int d = 0;
                for (; d < vecEnd; d += vw)
                {
                    var acc = new System.Numerics.Vector<float>(outSlice.Slice(d, vw));
                    (acc + wv * new System.Numerics.Vector<float>(vRow.Slice(d, vw))).CopyTo(outSlice.Slice(d, vw));
                }
                for (; d < headDim; d++)
                {
                    outSlice[d] += wj * vRow[d];
                }
            }
        }

        long work = (long)pairs * kvSeq * headDim;
        if (lastRowCapture is null && pairs > 1 && work >= 32L << 10)
        {
            KernelPool.For(pairs, index => Pair(index, ScoreBuffer(kvSeq)));
        }
        else
        {
            var weights = new float[kvSeq];
            for (int index = 0; index < pairs; index++)
            {
                Pair(index, weights);
            }
        }
        return output;
    }

    [ThreadStatic]
    private static float[]? _scoreBuffer;

    private static float[] ScoreBuffer(int length)
    {
        var buffer = _scoreBuffer;
        if (buffer is null || buffer.Length < length)
        {
            _scoreBuffer = buffer = new float[Math.Max(length, 256)];
        }
        return buffer;
    }
}