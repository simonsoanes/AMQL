namespace Amql.Safetensors;

/// <summary>
/// Ternary weight quantization (Bonsai-style): weights are quantised to
/// {-1, 0, +1} per block of <see cref="BlockElements"/> elements, each
/// block scaled by a float16 factor. Four ternary values pack into one
/// byte at 2 bits each.
/// </summary>
public static class Ternary
{
    public const int BlockElements = 128;          // standard Bonsai block size
    public const int ValuesPerByte = 4;            // 2 bits × 4 = 1 byte
    public const int BlockPackedBytes = BlockElements / ValuesPerByte; // 32 bytes per block
    public const int GridSize = 3;                 // {-1, 0, +1}

    /// <summary>The three representable ternary values.</summary>
    public static readonly float[] Grid = new[] { -1f, 0f, 1f };

    /// <summary>The broadcast scale dtype — FP16 for Bonsai.</summary>
    public static readonly Dtype ScaleDtype = Dtype.F16;

    public static int BlockScaleCount(int elementCount) =>
        (elementCount + BlockElements - 1) / BlockElements;

    // ── Encode helpers ──────────────────────────────────────────────────

    /// <summary>Encode a float array to packed ternary + FP16 scale.</summary>
    public static (byte[] Packed, byte[] Scales) Encode(float[] values)
    {
        int elements = values.Length;
        int blocks = BlockScaleCount(elements);

        var packed = new byte[blocks * BlockPackedBytes];
        var scales = new byte[blocks * 2]; // FP16 = 2 bytes

        for (int b = 0; b < blocks; b++)
        {
            int start = b * BlockElements;
            int count = Math.Min(BlockElements, elements - start);

            // Find optimal scale for this block: the median of absolute values
            float scale = ComputeTernaryScale(values.AsSpan(start, count));

            // Write scale as FP16
            ushort scaleBits = Float16ToBits(scale);
            scales[b * 2] = (byte)scaleBits;
            scales[b * 2 + 1] = (byte)(scaleBits >> 8);

            // Encode each ternary value
            for (int j = 0; j < BlockElements; j++)
            {
                int byteIdx = b * BlockPackedBytes + j / ValuesPerByte;
                int bitShift = (j % ValuesPerByte) * 2;

                int ternaryVal;
                if (j < count)
                {
                    float scaled = values[start + j] / scale;
                    // Find nearest tern: -1, 0, or +1
                    if (scaled < -0.5f) ternaryVal = 0; // -1 → bits 00
                    else if (scaled > 0.5f) ternaryVal = 2; // +1 → bits 10
                    else ternaryVal = 1; // 0 → bits 01
                }
                else
                {
                    ternaryVal = 1; // padding → 0
                }

                // Mask out old bits, set new value
                int mask = 0x3 << bitShift;
                packed[byteIdx] = (byte)((packed[byteIdx] & ~mask) | (ternaryVal << bitShift));
            }
        }

        return (packed, scales);
    }

    // ── Decode ──────────────────────────────────────────────────────────

    /// <summary>Decode packed ternary back to float values.</summary>
    public static float[] Decode(byte[] packed, byte[] scales, int elementCount)
    {
        var result = new float[elementCount];
        for (int i = 0; i < elementCount; i++)
        {
            int block = i / BlockElements;
            int byteIdx = block * BlockPackedBytes + (i % BlockElements) / ValuesPerByte;
            int bitShift = ((i % BlockElements) % ValuesPerByte) * 2;

            int code = (packed[byteIdx] >> bitShift) & 0x3;
            float ternaryVal = code switch
            {
                0 => -1f,  // bits 00 → -1
                2 => 1f,   // bits 10 → +1
                _ => 0f,   // bits 01 → 0
            };

            // Decode scale
            int scaleIdx = block * 2;
            ushort scaleBits = (ushort)(scales[scaleIdx] | (scales[scaleIdx + 1] << 8));
            float scale = Float16FromBits(scaleBits);

            result[i] = ternaryVal * scale;
        }
        return result;
    }

    // ── Internal helpers ────────────────────────────────────────────────

    private static float ComputeTernaryScale(Span<float> values)
    {
        // Bonsai scale: use the average of the top-k absolute values
        // Simplified: mean of absolute values (empirically close to optimal)
        float sum = 0f;
        int n = values.Length;
        for (int i = 0; i < n; i++)
        {
            sum += MathF.Abs(values[i]);
        }
        return sum / n;
    }

    internal static ushort Float16ToBits(float value)
    {
        // IEEE 754 float16 conversion
        uint f32 = BitConverter.SingleToUInt32Bits(value);
        uint sign = (f32 >> 16) & 0x8000;
        int exp32 = (int)((f32 >> 23) & 0xFF) - 127;
        uint mant = f32 & 0x7FFFFF;

        if (exp32 > 15) return (ushort)(sign | 0x7BFF); // Inf
        if (exp32 < -14) return (ushort)sign;            // zero
        if (exp32 == -127) return 0;                      // true zero

        int exp16 = exp32 + 15;
        uint mant16 = mant >> 13;
        return (ushort)(sign | ((uint)exp16 << 10) | mant16);
    }

    private static float Float16FromBits(ushort bits)
    {
        uint sign = (uint)(bits & 0x8000) << 16;
        int exp16 = (bits >> 10) & 0x1F;
        uint mant16 = (uint)(bits & 0x3FF);

        if (exp16 == 0) return mant16 == 0 ? 0f : BitConverter.UInt32BitsToSingle(sign | (mant16 << 13));
        if (exp16 == 31) return mant16 == 0 ? float.PositiveInfinity : float.NaN;

        int exp32 = exp16 - 15 + 127;
        uint mant32 = mant16 << 13;
        return BitConverter.UInt32BitsToSingle(sign | ((uint)exp32 << 23) | mant32);
    }
}