using Amql.Safetensors;

namespace Amql.Gguf;

/// <summary>
/// Dequantization routines for GGUF block-quantised types, producing
/// float arrays from the packed byte representations. Each mirrors the
/// corresponding quantize_row_*_ref in ggml-quants.c.
/// </summary>
public static class GgmlDequant
{
    private const int BlockElements = 32;

    /// <summary>ggml's MXFP4 grid: the 16 E2M1 values doubled, matching
    /// <c>kvalues</c> in llama.cpp. Negative-zero duplicate at index 8.</summary>
    private static readonly float[] Fp4Grid =
        { 0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, 0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f };

    /// <summary>Dequantize MXFP4 (ggml type 39): 17 bytes per 32 elements —
    /// one E8M0 scale followed by 16 nibble pairs, low nibble first.</summary>
    public static float[] DequantMxfp4(byte[] packed, int elements)
    {
        int blocks = elements / BlockElements;
        var result = new float[elements];
        for (int b = 0; b < blocks; b++)
        {
            int o = b * 17;
            float scale = MathF.ScaleB(1f, packed[o] - 128);
            for (int j = 0; j < 16; j++)
            {
                byte pair = packed[o + 1 + j];
                result[b * 32 + j] = Fp4Grid[pair & 0xF] * scale;
                result[b * 32 + 16 + j] = Fp4Grid[pair >> 4] * scale;
            }
        }
        return result;
    }

    /// <summary>Dequantize Q4_0 (ggml type 2): 18 bytes per 32 elements —
    /// two-byte FP16 scale followed by 16 nibble bytes (low nibble first,
    /// then high nibble). Each nibble encodes (value * 16 + 8) so the
    /// decoded range is [-8, 7] in steps of 1.</summary>
    public static float[] DequantQ4_0(byte[] packed, int elements)
    {
        int blocks = elements / BlockElements;
        var result = new float[elements];
        for (int b = 0; b < blocks; b++)
        {
            int o = b * 18;
            float d = (float)BitConverter.Int16BitsToHalf(BitConverter.ToInt16(packed, o));
            for (int j = 0; j < 16; j++)
            {
                byte pair = packed[o + 2 + j];
                result[b * 32 + j] = ((pair & 0xF) - 8) * d;
                result[b * 32 + 16 + j] = ((pair >> 4) - 8) * d;
            }
        }
        return result;
    }

    /// <summary>Dequantize Q8_0 (ggml type 8): 34 bytes per 32 elements —
    /// two-byte FP16 scale followed by 32 signed bytes.</summary>
    public static float[] DequantQ8_0(byte[] packed, int elements)
    {
        int blocks = elements / BlockElements;
        var result = new float[elements];
        for (int b = 0; b < blocks; b++)
        {
            int o = b * 34;
            float d = (float)BitConverter.Int16BitsToHalf(BitConverter.ToInt16(packed, o));
            for (int i = 0; i < 32; i++)
            {
                result[b * 32 + i] = (sbyte)packed[o + 2 + i] * d;
            }
        }
        return result;
    }

    /// <summary>Dequantize a tensor from its GGUF packed bytes to float32.</summary>
    public static float[] Dequant(GgufType type, byte[] packed, int elements) => type switch
    {
        GgufType.F32  => FloatFromBytes(packed, elements, 4, BitConverter.ToSingle),
        GgufType.F16  => HalfToFloat(packed, elements),
        GgufType.BF16 => Bf16ToFloat(packed, elements),
        GgufType.Q8_0 => DequantQ8_0(packed, elements),
        GgufType.Q4_0 => DequantQ4_0(packed, elements),
        GgufType.Mxfp4 => DequantMxfp4(packed, elements),
        _ => throw new GgufException($"dequantization of {type} is not implemented in this build"),
    };

    private static float[] FloatFromBytes(byte[] packed, int elements, int stride, Func<byte[], int, float> convert)
    {
        var result = new float[elements];
        for (int i = 0; i < elements; i++)
            result[i] = convert(packed, i * stride);
        return result;
    }

    private static float[] HalfToFloat(byte[] packed, int elements)
    {
        var result = new float[elements];
        for (int i = 0; i < elements; i++)
            result[i] = (float)BitConverter.Int16BitsToHalf(BitConverter.ToInt16(packed, i * 2));
        return result;
    }

    private static float[] Bf16ToFloat(byte[] packed, int elements)
    {
        var result = new float[elements];
        for (int i = 0; i < elements; i++)
        {
            int bits = packed[i * 2] | (packed[i * 2 + 1] << 8);
            result[i] = BitConverter.Int32BitsToSingle(bits << 16);
        }
        return result;
    }
}