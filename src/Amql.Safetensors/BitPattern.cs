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