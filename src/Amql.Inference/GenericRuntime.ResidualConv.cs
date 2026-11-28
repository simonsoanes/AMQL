namespace Amql.Inference;

/// <summary>
/// Residual short convolutions (see <see cref="ResidualConvOp"/>): a depthwise
/// causal convolution over a cache slot's history of the last K − 1 inputs,
/// added to its input and scaled by <c>1/√(n + 1)</c>.
/// </summary>
public sealed partial class GenericRuntime
{
    /// <param name="tapInput">What a dynamic convolution's taps are projected
    /// from (the projection's input); unused by a static one.</param>
    private float[] ResidualConv(ResidualConvOp op, List<float[]> history, float[] x, float[]? tapInput)
    {
        int kernel = op.Kernel;
        int c = x.Length;
        if (c != op.Channels)
        {
            throw new InvalidOperationException($"a residual conv of {op.Channels} channels was given {c}");
        }
        var output = new float[c];
        if (op.Dynamic)
        {
            // Taps static[lag, c] + dyn[head, lag], lag 0 on the current token.
            var statics = Rows.Vector(op.StaticTaps!, kernel * c);
            var input = tapInput ?? throw new InvalidOperationException("a dynamic residual conv needs the input its taps are projected from");
            var dynamics = Rows.Matrix(op.TapProjection!, op.Heads * kernel, input.Length).MatVec(input);
            int perHead = c / op.Heads;
            double norm = 1.0 / Math.Sqrt(2 * kernel + 1);
            for (int ch = 0; ch < c; ch++)
            {
                int head = ch / perHead;
                double sum = 0;
                for (int lag = 0; lag < kernel; lag++)
                {
                    float value;
                    if (lag == 0)
                    {
                        value = x[ch];
                    }
                    else
                    {
                        int index = history.Count - lag;
                        if (index < 0)
                        {
                            continue;
                        }
                        value = history[index][ch];
                    }
                    sum += (statics[lag * c + ch] + (double)dynamics[head * kernel + lag]) * value;
                }
                output[ch] = (float)((sum + x[ch]) * norm);
            }
        }
        else
        {
            // Weights [C, 1, K], the last tap on the current token.
            var weights = Rows.Vector(op.Weight!, c * kernel);
            double norm = 1.0 / Math.Sqrt(kernel + 1);
            for (int ch = 0; ch < c; ch++)
            {
                double sum = weights[ch * kernel + kernel - 1] * (double)x[ch];
                for (int lag = 1; lag < kernel; lag++)
                {
                    int index = history.Count - lag;
                    if (index >= 0)
                    {
                        sum += weights[ch * kernel + kernel - 1 - lag] * (double)history[index][ch];
                    }
                }
                output[ch] = (float)((sum + x[ch]) * norm);
            }
        }
        history.Add(x);
        if (history.Count > kernel - 1)
        {
            history.RemoveAt(0);
        }
        return output;
    }
}
