namespace Amql.Safetensors;

/// <summary>
/// The MXFP4 weight codec — the OCP microscaling (MX) standard: FP4 E2M1
/// elements (see <see cref="BitPattern.Fp4PositiveGrid"/>) packed two per
/// byte, with one FP8-E8M0 scale (a pure shared exponent) per 32-element
/// block. Blocks run along the columns, one scale per row of 32 — the
/// layout MX consumers (MLX's "mxfp4" mode, GGML) use. Dequantisation is
/// <c>x ≈ DecodeFp4(q) × DecodeF8E8M0(scale)</c>; there is no per-tensor
/// global scale — the E8M0 blocks carry the whole magnitude range.
/// </summary>
public static class Mxfp4
{
    /// <summary>How many consecutive elements share one E8M0 scale (the MX
    /// block size).</summary>
    public const int BlockElements = 32;

    public const string ScaleSuffix = "_scale";

    /// <summary>The largest finite magnitude of the E2M1 grid.</summary>
    public static float Fp4MaxValue => BitPattern.Fp4MaxValue;

    /// <summary>Stored byte length of a packed FP4 tensor.</summary>
    public static long PackedLength(long elements) => (elements + 1) / 2;

    /// <summary>Blocks per row, and the number of scale bytes.</summary>
    public static long BlocksPerRow(long columns) => (columns + BlockElements - 1) / BlockElements;

    public sealed record Quantized(byte[] Packed, byte[] BlockScales);

    /// <summary>
    /// Quantises a row-major [rows × columns] tensor: per 32-element block
    /// the E8M0 scale is chosen so the block's peak magnitude maps onto the
    /// top of the E2M1 grid (scale ≈ peak / 6), then each element is coded
    /// against the *decoded* scale so round-tripping cancels the E8M0
    /// rounding. Zero blocks carry scale 0 and all-zero codes.
    /// </summary>
    public static Quantized Quantize(float[] values, long rows, long columns)
    {
        if (rows <= 0 || columns <= 0 || rows * columns != values.Length)
        {
            throw new SafetensorsException(
                $"MXFP4 quantize: shape [{rows}, {columns}] does not match the value count {values.Length}");
        }

        try
        {
            return QuantizeCore(values, rows, columns);
        }
        catch (IndexOutOfRangeException ex)
        {
            throw new SafetensorsException(
                $"MXFP4 Quantize failed for [{rows}×{columns}] (elements={values.Length}): {ex.Message}", ex);
        }
    }

    private static Quantized QuantizeCore(float[] values, long rows, long columns)
    {

        long blocksPerRow = BlocksPerRow(columns);
        long totalBlocks = rows * blocksPerRow;
        var packed = new byte[(values.Length + 1) / 2];
        var scales = new byte[totalBlocks];

        for (long row = 0; row < rows; row++)
        {
            long rowBase = row * columns;
            for (long b = 0; b < blocksPerRow; b++)
            {
                long blockStart = rowBase + b * BlockElements;
                long elementsInBlock = Math.Min(BlockElements, columns - b * BlockElements);
                double peak = 0;
                for (long k = 0; k < elementsInBlock; k++)
                {
                    double a = Math.Abs((double)values[blockStart + k]);
                    if (a > peak)
                    {
                        peak = a;
                    }
                }

                byte scaleByte = peak == 0 ? (byte)0 : BitPattern.EncodeF8E8M0((float)(peak / Fp4MaxValue));
                scales[row * blocksPerRow + b] = scaleByte;
                float scale = BitPattern.DecodeF8E8M0(scaleByte);
                // E8M0 can't represent true zero; 0x00 → 2⁻¹²⁷ ≈ 5.88e-39.
                // Guard against both true-zero blocks and empty trailing blocks.
                bool zero = scale < 1e-12f || elementsInBlock == 0;

                for (long k = 0; k < elementsInBlock; k++)
                {
                    long index = blockStart + k;
                    byte nibble = zero ? (byte)0 : BitPattern.EncodeFp4(values[index] / scale);
                    if ((index & 1) == 0)
                    {
                        packed[index / 2] = nibble;
                    }
                    else
                    {
                        packed[index / 2] |= (byte)(nibble << 4);
                    }
                }
            }
        }

        return new Quantized(packed, scales);
    }

    /// <summary>Dequantises a packed MXFP4 tensor back to f32. The E8M0
    /// scale is decoded once per 32-element block (not once per element —
    /// a power-2 evaluation per element was the hot-loop tax).</summary>
    public static float[] Dequant(byte[] packed, byte[] blockScales, long rows, long columns)
    {
        long blocksPerRow = BlocksPerRow(columns);
        var result = new float[rows * columns];
        for (long row = 0; row < rows; row++)
        {
            long rowBase = row * columns;
            long scaleBase = row * blocksPerRow;
            for (long b = 0; b < blocksPerRow; b++)
            {
                float scale = BitPattern.DecodeF8E8M0(blockScales[scaleBase + b]);
                long blockStart = rowBase + b * BlockElements;
                long blockEnd = Math.Min(blockStart + BlockElements, rowBase + columns);
                for (long index = blockStart; index < blockEnd; index++)
                {
                    byte nibble = (index & 1) == 0
                        ? (byte)(packed[index / 2] & 0x0F)
                        : (byte)((packed[index / 2] >> 4) & 0x0F);
                    result[index] = BitPattern.DecodeFp4(nibble) * scale;
                }
            }
        }
        return result;
    }
}