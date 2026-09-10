namespace Amql.Safetensors;

/// <summary>Bit-pattern ↔ f32 conversions for the storage dtypes safetensors
/// and the VINDEX3 segment format carry. Formulas follow the reference
/// implementations in <c>larql-models/src/quant/half.rs</c> and
/// <c>loading/safetensors/dtype.rs</c>.</summary>
public static class BitPattern
{
    // ── F16 (IEEE half) ────────────────────────────────────────────────────

    public static float DecodeF16(ushort h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        float v;
        if (exp == 0)
        {
            v = mant == 0 ? 0f : mant * MathF.Pow(2f, -24f);
        }
        else if (exp == 31)
        {
            v = mant == 0 ? float.PositiveInfinity : float.NaN;
        }
        else
        {
            v = (1f + mant / 1024f) * MathF.Pow(2f, exp - 15);
        }
        return sign == 1 ? -v : v;
    }

    /// <summary>f32 → F16 bits, round-to-nearest-even.</summary>
    public static ushort EncodeF16(float value)
    {
        int bits = BitConverter.SingleToInt32Bits(value);
        int sign = (bits >> 16) & 0x8000;
        int exp = (bits >> 23) & 0xFF;
        int mant = bits & 0x7FFFFF;

        if (exp == 0xFF)
        {
            // Inf/NaN: preserve payload crudely (top mantissa bits).
            return (ushort)(sign | 0x7C00 | (mant != 0 ? 0x0200 | (mant >> 13) : 0));
        }

        if (exp >= 0x8F)
        {
            return (ushort)(sign | 0x7C00); // overflow → ±Inf
        }

        if (exp <= 0x70)
        {
            return (ushort)sign; // underflow → ±0
        }

        int halfExp = exp - 127 + 15;
        if (halfExp <= 0)
        {
            // Subnormal half.
            int shift = 14 - halfExp;
            if (shift > 24)
            {
                return (ushort)sign;
            }
            int halfMant = (0x800000 | mant) >> shift;
            return (ushort)(sign | halfMant);
        }

        int hm = mant >> 13;
        int rem = mant & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (hm & 1) == 1))
        {
            hm++;
            if (hm == 0x400)
            {
                hm = 0;
                halfExp++;
            }
        }
        return (ushort)(sign | (halfExp << 10) | hm);
    }

    // ── BF16 ───────────────────────────────────────────────────────────────

    public static float DecodeBf16(ushort h) =>
        BitConverter.Int32BitsToSingle(unchecked((int)((uint)h << 16)));

    /// <summary>f32 → BF16 bits, round-to-nearest-even (the classic
    /// add-0x7FFF-then-truncate trick).</summary>
    public static ushort EncodeBf16(float value)
    {
        int bits = BitConverter.SingleToInt32Bits(value);
        if (float.IsNaN(value))
        {
            return (ushort)((uint)bits >> 16);
        }
        uint rounded = (uint)bits + 0x7FFFu + (((uint)bits >> 16) & 1u);
        return (ushort)(rounded >> 16);
    }

    // ── FP8 (Open Compute encodings) ───────────────────────────────────────

    /// <summary>E4M3: 1 sign + 4 exponent (bias 7) + 3 mantissa. NaN at
    /// 0x7F / 0xFF.</summary>
    public static float DecodeF8E4M3(byte b)
    {
        int sign = (b >> 7) & 1;
        int expBits = (b >> 3) & 0x0F;
        int mantBits = b & 0x07;
        float v;
        if (expBits == 0)
        {
            v = mantBits / 8f * MathF.Pow(2f, 1 - 7);
        }
        else if (expBits == 0x0F && mantBits == 0x07)
        {
            v = float.NaN;
        }
        else
        {
            v = (1f + mantBits / 8f) * MathF.Pow(2f, expBits - 7);
        }
        return sign == 1 ? -v : v;
    }

    /// <summary>E5M2: 1 sign + 5 exponent (bias 15) + 2 mantissa. Exponent
    /// 0x1F is ±Inf (mantissa 0) or NaN.</summary>
    public static float DecodeF8E5M2(byte b)
    {
        int sign = (b >> 7) & 1;
        int expBits = (b >> 2) & 0x1F;
        int mantBits = b & 0x03;
        float v;
        if (expBits == 0)
        {
            v = mantBits / 4f * MathF.Pow(2f, 1 - 15);
        }
        else if (expBits == 0x1F)
        {
            v = mantBits == 0 ? float.PositiveInfinity : float.NaN;
        }
        else
        {
            v = (1f + mantBits / 4f) * MathF.Pow(2f, expBits - 15);
        }
        return sign == 1 ? -v : v;
    }

    /// <summary>E8M0 (microscaling MX scale): 8 exponent bits, no sign or
    /// mantissa. Value = 2^(byte − 127); 0xFF is NaN.</summary>
    public static float DecodeF8E8M0(byte b) => b == 0xFF ? float.NaN : MathF.Pow(2f, b - 127);

    /// <summary>I8: sign-extend.</summary>
    public static float DecodeI8(byte b) => (sbyte)b;

    // ── FP4 (NVFP4 element grid) ─────────────────────────────────────────────

    /// <summary>The positive magnitudes of the NVFP4 element grid, indexed
    /// by the low three bits of the nibble (bit 3 is the sign). This is
    /// the value grid the published NVFP4 conversions carry:
    /// <c>{0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}</c>, so the largest magnitude
    /// is 3.0. The ecosystem labels this family both "E1M2" and "E2M1"
    /// depending on vendor — the grid, serialised into the exported
    /// config's <c>element_grid</c>, is the contract; a consumer with a
    /// different table can swap this one value table.</summary>
    public static readonly float[] Fp4PositiveGrid = { 0f, 0.25f, 0.5f, 0.75f, 1f, 1.5f, 2f, 3f };

    public const float Fp4MaxValue = 3f;

    /// <summary>Decode one FP4 nibble (low 4 bits of the byte).</summary>
    public static float DecodeFp4(byte nibble)
    {
        float v = Fp4PositiveGrid[nibble & 0x07];
        return (nibble & 0x08) != 0 ? -v : v;
    }

    /// <summary>Nearest FP4 nibble for a value (ties to the smaller
    /// magnitude); anything outside the ±3.0 grid clamps.</summary>
    public static byte EncodeFp4(float value)
    {
        float a = MathF.Abs(value);
        byte index = 0;
        float best = float.PositiveInfinity;
        for (byte i = 0; i < Fp4PositiveGrid.Length; i++)
        {
            float diff = MathF.Abs(a - Fp4PositiveGrid[i]);
            if (diff < best)
            {
                best = diff;
                index = i;
            }
        }
        // Sign bit via the raw pattern so −0 stays distinguishable.
        bool negative = BitConverter.SingleToInt32Bits(value) < 0;
        return negative ? (byte)(index | 0x08) : index;
    }

    /// <summary>f32 → E4M3 (OCP, bias 7) with round-to-nearest-even; the
    /// inverse of <see cref="DecodeF8E4M3"/>. Used for the NVFP4 block
    /// scales. The decoder in this file treats exponent field 15 as a
    /// valid exponent (NaN only with mantissa 7), so the largest finite
    /// magnitude here is <c>1.75 × 2⁸ = 448</c> (bits 0x7E).</summary>
    public static byte EncodeF8E4M3(float value)
    {
        if (float.IsNaN(value))
        {
            return 0xFF;
        }
        int sign = value < 0 ? 0x80 : 0;
        float a = MathF.Abs(value);
        if (a == 0)
        {
            return (byte)sign;
        }

        // Largest finite E4M3 under this decoder is 448 (exp field 15,
        // mantissa 6); exponent field 15 with mantissa 7 is NaN.
        const float maxValue = 448f;
        if (a >= maxValue)
        {
            return (byte)(sign | 0x7E);
        }

        double mantissa = a;
        int exp = 0;
        while (mantissa >= 2.0)
        {
            mantissa /= 2.0;
            exp++;
        }
        while (mantissa < 1.0)
        {
            mantissa *= 2.0;
            exp--;
        }
        // mantissa ∈ [1, 2), exp such that value ≈ mantissa × 2^exp,
        // e4m3 exponent field = exp + 7.
        int field = exp + 7;
        if (field <= 0)
        {
            // Subnormal: value = m/8 × 2⁻⁶ = m × 2⁻⁹.
            long m = (long)Math.Round(a * 512.0, MidpointRounding.ToEven);
            return (byte)(sign | (int)Math.Clamp(m, 0, 7));
        }

        long mant = (long)Math.Round((mantissa - 1.0) * 8.0, MidpointRounding.ToEven);
        if (mant == 8)
        {
            mant = 0;
            field++;
        }
        if (field > 15 || (field == 15 && mant >= 7))
        {
            return (byte)(sign | 0x7E);
        }
        return (byte)(sign | (field << 3) | (int)mant);
    }

    // ── Bulk widening ──────────────────────────────────────────────────────

    /// <summary>Widen a per-tensor byte payload to f32, matching the
    /// reference's <c>tensor_to_f32</c> dispatch.</summary>
    public static float[] WidenToF32(Dtype dtype, ReadOnlySpan<byte> bytes)
    {
        switch (dtype)
        {
            case Dtype.F32:
            {
                if (bytes.Length % 4 != 0)
                {
                    throw new SafetensorsException($"F32 payload length {bytes.Length} is not a multiple of 4");
                }
                var result = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes.ToArray(), 0, result, 0, bytes.Length);
                return result;
            }
            case Dtype.F16:
            {
                var result = new float[bytes.Length / 2];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = DecodeF16((ushort)(bytes[2 * i] | (bytes[2 * i + 1] << 8)));
                }
                return result;
            }
            case Dtype.BF16:
            {
                var result = new float[bytes.Length / 2];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = DecodeBf16((ushort)(bytes[2 * i] | (bytes[2 * i + 1] << 8)));
                }
                return result;
            }
            case Dtype.I8:
            {
                var result = new float[bytes.Length];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = (sbyte)bytes[i];
                }
                return result;
            }
            case Dtype.F8_E4M3:
            {
                var result = new float[bytes.Length];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = DecodeF8E4M3(bytes[i]);
                }
                return result;
            }
            case Dtype.F8_E5M2:
            {
                var result = new float[bytes.Length];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = DecodeF8E5M2(bytes[i]);
                }
                return result;
            }
            case Dtype.F8_E8M0:
            {
                var result = new float[bytes.Length];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = DecodeF8E8M0(bytes[i]);
                }
                return result;
            }
            default:
                throw new SafetensorsException(
                    $"dtype {dtype.Label()} cannot be widened to f32 — unsupported dtype");
        }
    }
}