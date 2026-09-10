namespace Amql.Safetensors;

/// <summary>
/// The NVFP4 weight codec: FP4 elements (see
/// <see cref="BitPattern.Fp4PositiveGrid"/>) packed two per byte with a
/// scale hierarchy — one FP8 (E4M3) block scale per consecutive pair of
/// elements, plus one FP32 global scale per tensor. Dequantisation is
/// <c>x ≈ DecodeFp4(q) × DecodeE4M3(blockScale) × globalScale</c>. The
/// tensor layout follows the NVFP4 checkpoints published for the DeepSeek
/// lineage (per-2 FP8 scales, global scale, <c>_scale</c>/<c>_global_scale</c>
/// suffixes).
/// </summary>
public static class Nvfp4
{
    /// <summary>How many consecutive elements share one FP8 block scale.</summary>
    public const int BlockElements = 2;

    public const string ScaleSuffix = "_scale";
    public const string GlobalScaleSuffix = "_global_scale";

    /// <summary>The largest finite magnitude of the FP4 grid.</summary>
    public static float Fp4MaxValue => BitPattern.Fp4MaxValue;

    /// <summary>Stored byte length of a packed FP4 tensor.</summary>
    public static long PackedLength(long elements) => (elements + 1) / 2;

    public sealed record Quantized(byte[] Packed, byte[] BlockScales, float GlobalScale);

    /// <summary>
    /// Quantises in the scaffold's scale hierarchy: a tensor-level global
    /// scale maps the largest magnitude onto the top of the E2M1 range,
    /// then each 2-element block's local peak is captured by its own E4M3
    /// scale so the fp4 codes stay dense. Both values and scales round
    /// to nearest; zero blocks produce zero codes and zero scales.
    /// </summary>
    public static Quantized Quantize(float[] values, long elements)
    {
        if (elements < 0 || elements != values.Length)
        {
            throw new SafetensorsException(
                $"NVFP4 quantise: elements {elements} do not match the value count {values.Length}");
        }

        double maxAbs = 0;
        for (long i = 0; i < elements; i++)
        {
            double a = Math.Abs((double)values[i]);
            if (a > maxAbs)
            {
                maxAbs = a;
            }
        }

        float globalScale = maxAbs == 0 ? 0f : (float)(maxAbs / Fp4MaxValue);
        long blocks = (elements + 1) / 2;
        var packed = new byte[blocks];
        var scales = new byte[blocks];

        for (long g = 0; g < blocks; g++)
        {
            int first = (int)(2 * g);
            int second = first + 1 < elements ? first + 1 : -1;
            double blockPeak = 0;
            double v0 = 0, v1 = 0;
            if (globalScale != 0)
            {
                v0 = values[first] / globalScale;
                blockPeak = Math.Abs(v0);
                if (second >= 0)
                {
                    v1 = values[second] / globalScale;
                    double a1 = Math.Abs(v1);
                    if (a1 > blockPeak)
                    {
                        blockPeak = a1;
                    }
                }
            }

            byte blockScale = BitPattern.EncodeF8E4M3((float)blockPeak);
            scales[g] = blockScale;
            float s = blockPeak == 0 ? 0f : (float)(1.0 / blockPeak);
            byte low = blockPeak == 0 ? (byte)0 : BitPattern.EncodeFp4((float)(v0 * s));
            byte high = second < 0 || blockPeak == 0 ? (byte)0 : BitPattern.EncodeFp4((float)(v1 * s));
            packed[g] = (byte)(low | (high << 4));
        }

        return new Quantized(packed, scales, globalScale);
    }

    /// <summary>Dequantises a packed FP4 tensor back to f32.</summary>
    public static float[] Dequant(byte[] packed, byte[] blockScales, float globalScale, long elements)
    {
        var result = new float[elements];
        for (long i = 0; i < elements; i++)
        {
            long g = i / 2;
            byte nibble = (i & 1) == 0
                ? (byte)(packed[g] & 0x0F)
                : (byte)((packed[g] >> 4) & 0x0F);
            float s = BitPattern.DecodeF8E4M3(blockScales[g]);
            result[i] = BitPattern.DecodeFp4(nibble) * s * globalScale;
        }
        return result;
    }
}