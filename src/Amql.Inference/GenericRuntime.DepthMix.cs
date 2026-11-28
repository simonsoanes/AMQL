namespace Amql.Inference;

/// <summary>
/// Depth mixing: attention over the residual's bank of block inputs (the
/// block residual's sub-block inputs and the final norm's), and the message
/// gates a looped layer's inputs take from the previous pass.
/// </summary>
public sealed partial class GenericRuntime
{
    /// <summary>
    /// The depth mix of <paramref name="bank"/> and the current state: every
    /// state RMS-normalised without a weight, weighted by
    /// <c>softmax(query · state / √hidden)</c>; one output per query row.
    /// </summary>
    private float[][] DepthMix(DepthMixOp mix, List<float[]> bank, float[] current)
    {
        int d = current.Length;
        var queries = Rows.Vector(mix.Queries, mix.Count * d);
        return DepthMix(queries, mix.Count, mix.Eps, bank, current);
    }

    private static float[][] DepthMix(float[] queries, int count, double eps, List<float[]> bank, float[] current)
    {
        int d = current.Length;
        var states = new float[bank.Count + 1][];
        for (int i = 0; i < bank.Count; i++)
        {
            states[i] = RowMath.Unweighted(bank[i], eps);
        }
        states[^1] = RowMath.Unweighted(current, eps);
        var result = new float[count][];
        double scale = 1.0 / Math.Sqrt(d);
        var weights = new double[states.Length];
        for (int q = 0; q < count; q++)
        {
            var query = queries.AsSpan(q * d, d);
            double max = double.NegativeInfinity;
            for (int i = 0; i < states.Length; i++)
            {
                weights[i] = RowMath.Dot(query, states[i]) * scale;
                max = Math.Max(max, weights[i]);
            }
            double sum = 0;
            for (int i = 0; i < states.Length; i++)
            {
                weights[i] = Math.Exp(weights[i] - max);
                sum += weights[i];
            }
            var acc = new double[d];
            for (int i = 0; i < states.Length; i++)
            {
                double w = weights[i] / sum;
                for (int j = 0; j < d; j++)
                {
                    acc[j] += w * states[i][j];
                }
            }
            var row = new float[d];
            for (int j = 0; j < d; j++)
            {
                row[j] = (float)acc[j];
            }
            result[q] = row;
        }
        return result;
    }

    /// <summary>
    /// The layer's message gate on one input site, when it has one:
    /// <c>(1 − tanh(F·g))·h + tanh(G·g)·m</c>, <c>g = D·[rms(h); rms(m)]</c>, with m
    /// the depth mix of the site's query over the previous pass's bank and output.
    /// </summary>
    private float[] GateInput(LayerPlan layer, string site, float[] h, (List<float[]> Bank, float[] Output) previous)
    {
        if (layer.MessageGates!.FirstOrDefault(g => g.Input == site) is not { } gate)
        {
            return h;
        }
        int d = h.Length, rank = gate.Rank;
        var query = Rows.VectorRow(gate.Query, checked((int)(QueryRows(gate))), d, gate.QueryRow);
        var message = DepthMix(query, 1, gate.MixEps, previous.Bank, previous.Output)[0];

        var cat = new float[2 * d];
        RowMath.Unweighted(h, gate.Eps).CopyTo(cat, 0);
        RowMath.Unweighted(message, gate.Eps).CopyTo(cat, d);
        var g = Rows.Matrix(gate.Down, rank, 2 * d).MatVec(cat);
        var forget = Rows.Matrix(gate.Forget, d, rank);
        var gain = Rows.Matrix(gate.Gain, d, rank);
        var output = new float[d];
        for (int i = 0; i < d; i++)
        {
            double f = Math.Tanh(forget.Dot(i, g));
            double a = Math.Tanh(gain.Dot(i, g));
            output[i] = (float)((1 - f) * h[i] + a * message[i]);
        }
        return output;
    }

    private long QueryRows(MessageGateOp gate) => _weights.Rows.RowsOf(gate.Query);
}
