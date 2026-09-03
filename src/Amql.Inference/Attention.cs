using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Rotary position embedding. The general entry computes
/// <c>inv_freq[i] = 1 / theta^(2i/rotary_dim)</c> and rotates the two
/// halves of each rotary pair by <c>angle = position · inv_freq[i]</c>.
/// The rotation covers the first <c>rotaryWidth</c> dims of each head:
/// full-width for <see cref="PositionRope"/>, the partial fraction for
/// <see cref="PositionPartialRope"/> (text-only MRoPE collapses to this).
/// Scaled variants are refused — the reference refuses to serve an
/// unknown position policy rather than approximate one.
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
        PositionPolicy positionPolicy, ReadOnlySpan<int> positions)
    {
        int rotaryWidth;
        double theta;
        switch (positionPolicy)
        {
            case PositionNone:
                return;
            case PositionRope rope:
                rotaryWidth = headDim;
                theta = rope.Theta;
                break;
            case PositionPartialRope partial:
                rotaryWidth = partial.RotaryWidth(headDim);
                theta = partial.Theta;
                break;
            default:
                throw new UnsupportedOperatorException(
                    $"position policy '{((PositionUnresolved)positionPolicy).Kind}' has no managed rotary implementation");
        }

        if (rotaryWidth > headDim)
        {
            throw new ArgumentException($"rotary width {rotaryWidth} exceeds head dim {headDim}");
        }

        int pairCount = rotaryWidth / 2;
        var invFreq = new double[pairCount];
        for (int i = 0; i < pairCount; i++)
        {
            invFreq[i] = 1.0 / Math.Pow(theta, 2.0 * i / rotaryWidth);
        }

        for (int r = 0; r < x.Rows; r++)
        {
            double position = positions[r];
            int rowStart = r * x.Cols;
            for (int h = 0; h < numHeads; h++)
            {
                int head = h * headDim;
                // The reference pairs the two halves of the rotary window:
                // dims i and i + rotaryWidth/2 rotate together.
                for (int i = 0; i < pairCount; i++)
                {
                    int a = rowStart + head + i;
                    int b = a + pairCount;
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
/// Causal by construction.
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
        List<(int Head, float[] Weights)>? lastRowCapture = null)
    {
        int seq = q.Rows;
        int kvSeq = k.Rows;
        int headsRep = numQHeads / numKvHeads;
        var output = Tensor2D.Zeros(seq, numQHeads * headDim);
        var weights = new float[kvSeq];

        for (int qi = 0; qi < seq; qi++)
        {
            int qPos = queryPositions[qi];
            // The causal span is [0, lastValid]: keys are position-ordered,
            // so everything after the first future position is out of the
            // row's reach. The buffer is refilled from scratch each row —
            // no stale values survive into the softmax.
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
                        break; // keys are position-ordered; past the window nothing is valid
                    }
                    if (window is { } w && s >= w)
                    {
                        weights[j] = float.NegativeInfinity;
                        continue;
                    }
                    var kSlice = k.Row(j).Slice(kvHead * headDim, headDim);
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
                    continue;
                }

                SoftmaxInPlace(weights.AsSpan(0, spanEnd));
                if (lastRowCapture is not null && qi == seq - 1)
                {
                    // Post-softmax attention of the final query row over all
                    // its causal keys — the observable A→B link tensors.
                    lastRowCapture.Add((h, weights.AsSpan(0, spanEnd).ToArray()));
                }
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
        return output;
    }
}