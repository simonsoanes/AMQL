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
        Activation activation, bool gated)
    {
        var act = Activations.For(activation, gated);
        Tensor2D hidden;
        if (gated && gate is not null)
        {
            var g = TensorOps.MatMulTransposedB(x, gate);
            var u = TensorOps.MatMulTransposedB(x, up);
            hidden = Tensor2D.Zeros(x.Rows, g.Cols);
            for (int r = 0; r < hidden.Rows; r++)
            {
                for (int c = 0; c < hidden.Cols; c++)
                {
                    hidden.Set(r, c, act(g[r, c]) * u[r, c]);
                }
            }
        }
        else
        {
            hidden = TensorOps.MatMulTransposedB(x, up);
            for (int r = 0; r < hidden.Rows; r++)
            {
                for (int c = 0; c < hidden.Cols; c++)
                {
                    hidden.Set(r, c, act(hidden[r, c]));
                }
            }
        }
        return TensorOps.MatMulTransposedB(hidden, down);
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
        Activation activation)
    {
        int numExperts = expertGate.Count;
        if (topK > numExperts)
        {
            throw new UnsupportedOperatorException(
                $"routed FFN top-k ({topK}) exceeds expert count ({numExperts})");
        }

        // Router scores: [rows, numExperts].
        var scores = TensorOps.MatMulTransposedB(x, router);
        var output = Tensor2D.Zeros(x.Rows, expertDown[0].Rows);

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

            // Select the top-k.
            var selected = Enumerable.Range(0, numExperts)
                .OrderByDescending(e => row[e])
                .Take(topK)
                .ToArray();

            if (routingPolicy == ExpertRoutingPolicy.NormalisedOverSelected)
            {
                float selectedSum = 0f;
                foreach (var e in selected)
                {
                    selectedSum += row[e];
                }
                if (selectedSum > 0f)
                {
                    foreach (var e in selected)
                    {
                        row[e] /= selectedSum;
                    }
                }
            }

            foreach (var e in selected)
            {
                float weight = row[e];
                if (weight == 0f)
                {
                    continue;
                }
                var expertInput = new Tensor2D(x.Row(r).ToArray(), 1, x.Cols);
                var expertOut = Dense(expertInput, expertGate[e], expertUp[e], expertDown[e], activation, true);
                for (int c = 0; c < output.Cols; c++)
                {
                    output.Data[r * output.Cols + c] += weight * expertOut.Data[c];
                }
            }
        }
        return output;
    }
}