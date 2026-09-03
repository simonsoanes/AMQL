using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>Activation functions used by the FFN ops. GELU is served by the
/// tanh approximation, exactly as the reference does (no exact-GELU kernel
/// exists there either); ReLU has no gated kernel and refuses.</summary>
public static class Activations
{
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    public static float GeluTanh(float x) =>
        0.5f * x * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.044715f * x * x * x)));

    public static float Relu(float x) => x > 0f ? x : 0f;

    /// <summary>Dispatch table for FFN operations. <c>gelu</c> and
    /// <c>gelu_tanh</c> share the tanh kernel; <c>relu</c> refuses on a
    /// gated path (no kernel, matching the reference's panic).</summary>
    public static Func<float, float> For(Activation activation, bool gated) => activation switch
    {
        Activation.Silu => Silu,
        Activation.Gelu => GeluTanh,
        Activation.GeluTanh => GeluTanh,
        Activation.Relu => gated
            ? throw new UnsupportedOperatorException("Activation Relu has no gate/up FFN kernel")
            : Relu,
        _ => throw new UnsupportedOperatorException($"unknown activation {(int)activation}"),
    };
}

/// <summary>Normalisation kernels. The affine convention is
/// <c>normalise(x) * (weight + weight_offset)</c> — upstream's centred
/// variant is this type with <c>weight_offset = 1.0</c>.</summary>
public static class Norms
{
    public static void ApplyInPlace(Tensor2D x, NormType kind, double eps,
        ReadOnlySpan<float> weight, float weightOffset)
    {
        for (int r = 0; r < x.Rows; r++)
        {
            var row = x.Row(r);
            ApplyRow(row, kind, eps, weight, weightOffset);
        }
    }

    /// <summary>Normalises a row of length n in place, weight applied
    /// element-wise (weight length must equal n).</summary>
    public static void ApplyRow(Span<float> row, NormType kind, double eps,
        ReadOnlySpan<float> weight, float weightOffset)
    {
        int n = row.Length;
        if (kind == NormType.RmsNorm)
        {
            double sumSquares = 0;
            for (int i = 0; i < n; i++)
            {
                sumSquares += (double)row[i] * row[i];
            }
            double inv = 1.0 / Math.Sqrt(sumSquares / n + eps);
            for (int i = 0; i < n; i++)
            {
                row[i] = (float)(row[i] * inv * (weight[i] + weightOffset));
            }
        }
        else
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
            {
                mean += row[i];
            }
            mean /= n;
            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double d = row[i] - mean;
                variance += d * d;
            }
            variance /= n;
            double inv = 1.0 / Math.Sqrt(variance + eps);
            for (int i = 0; i < n; i++)
            {
                row[i] = (float)((row[i] - mean) * inv * (weight[i] + weightOffset));
            }
        }
    }
}