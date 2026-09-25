using Amql.Safetensors;

namespace Amql.Gguf;

/// <summary>
/// ggml block encoders for the types this converter can emit beyond Q4_0.
/// Each mirrors the corresponding <c>quantize_row_*_ref</c> in ggml-quants.c
/// exactly — scale formula, rounding mode and nibble order included — so the
/// bytes match what llama.cpp's own quantizer produces and can be checked
/// against gguf-py's reference implementation rather than against this code.
/// <para>
/// All of these pack element <c>j</c> into byte <c>j</c>'s low nibble and
/// element <c>j+16</c> into its high nibble. Pairing even with odd instead is
/// a silent permutation of every row: the file still loads and any check
/// written from the same wrong assumption still passes.
/// </para>
/// </summary>
public static class GgmlQuant
{
    /// <summary>Elements sharing one scale block (QK_MXFP4, QK4_0, QK8_0).</summary>
    public const int BlockElements = 32;

    /// <summary>Stored bytes per block: 1 E8M0 scale byte + 16 nibble bytes.</summary>
    public const int Mxfp4BlockBytes = 17;

    /// <summary>Stored bytes per block: 2 F16 scale bytes + 32 int8.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>The E2M1 grid as ggml stores it: doubled and signed, so a value
    /// decodes as <c>kvalues[code] * 2^(e-128)</c>. Indices 0-7 are the
    /// positive codes, 8 is negative zero and 9-15 the negatives — the
    /// duplicate zero is why a tie must resolve to the lower index.</summary>
    private static readonly int[] Fp4KValues =
        { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

    /// <summary>Quantizes to ggml's MXFP4: one 17-byte block per 32 values, a
    /// single E8M0 shared exponent followed by 16 bytes of E2M1 codes.
    /// Mirrors <c>quantize_row_mxfp4_ref</c>.</summary>
    public static byte[] QuantizeMxfp4(float[] values)
    {
        int numBlocks = RequireWholeBlocks(values.Length);
        var result = new byte[numBlocks * Mxfp4BlockBytes];

        for (int block = 0; block < numBlocks; block++)
        {
            int start = block * BlockElements;
            int o = block * Mxfp4BlockBytes;

            float amax = MaxAbs(values, start);

            // ggml picks the exponent so the block's peak lands two E2M1 steps
            // below the top of the grid, and stores 0 rather than an undefined
            // log2 for an all-zero block.
            byte e = amax > 0f
                ? (byte)Math.Clamp((int)MathF.Floor(MathF.Log2(amax)) - 2 + 127, 0, 255)
                : (byte)0;
            result[o] = e;

            float scale = E8M0ToHalfScale(e);
            for (int j = 0; j < BlockElements / 2; j++)
            {
                byte lo = BestFp4Code(values[start + j], scale);
                byte hi = BestFp4Code(values[start + BlockElements / 2 + j], scale);
                result[o + 1 + j] = (byte)(lo | (hi << 4));
            }
        }

        return result;
    }

    /// <summary>Quantizes to ggml's Q8_0: one 34-byte block per 32 values, an
    /// F16 scale followed by 32 signed bytes. Mirrors
    /// <c>quantize_row_q8_0</c>, including rounding half away from zero — this
    /// is the type llama.cpp's MXFP4_MOE recipe uses for every non-expert
    /// weight.</summary>
    public static byte[] QuantizeQ8_0(float[] values)
    {
        int numBlocks = RequireWholeBlocks(values.Length);
        var result = new byte[numBlocks * Q8_0BlockBytes];

        for (int block = 0; block < numBlocks; block++)
        {
            int start = block * BlockElements;
            int o = block * Q8_0BlockBytes;

            float d = MaxAbs(values, start) / 127f;
            float id = d != 0f ? 1f / d : 0f;

            ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)d);
            result[o] = (byte)(scaleBits & 0xFF);
            result[o + 1] = (byte)(scaleBits >> 8);

            for (int i = 0; i < BlockElements; i++)
            {
                // |values| <= amax, so the product cannot exceed 127 in exact
                // arithmetic; the clamp only absorbs float rounding at the top
                // of the range rather than letting an sbyte cast wrap silently.
                float q = MathF.Round(values[start + i] * id, MidpointRounding.AwayFromZero);
                result[o + 2 + i] = unchecked((byte)(sbyte)Math.Clamp((int)q, -127, 127));
            }
        }

        return result;
    }

    /// <summary>Encodes one block of floats with the given type.
    /// <paramref name="cols"/> is the row width of the logical matrix: the
    /// ternary packings rotate row-wise in 1024-column blocks, so they cannot
    /// be computed from a flat array alone.</summary>
    public static byte[] Encode(GgufType type, float[] values, int cols) => type switch
    {
        GgufType.Mxfp4 => QuantizeMxfp4(values),
        GgufType.Q8_0 => QuantizeQ8_0(values),
        GgufType.Ptq1_0 => QuantizeTernaryGguf(values, cols, ptq1: true),
        GgufType.Pq2_0 => QuantizeTernaryGguf(values, cols, ptq1: false),
        _ => throw new GgufException($"no ggml block encoder for {type} in this build"),
    };

    /// <summary>Encodes to the Bonsai fork's GGUF ternary block: per 128
    /// elements, the FP16 scale followed by the packed trits. The safetensors
    /// codec keeps scales and packed data in two separate arrays; a GGUF block
    /// interleaves them, scale first, matching every other ggml block type.
    /// <para>
    /// That ordering is ggml's convention, not something the Bonsai whitepaper
    /// specifies — the paper gives the rates (26 and 32 packed bytes per 128)
    /// but no byte order — and no reference file exists to check it against.
    /// A GGUF carrying type 142/143 loads only in the fork, and whether the
    /// fork agrees on the interleaving is unverified here.
    /// </para>
    /// </summary>
    public static byte[] QuantizeTernaryGguf(float[] values, int cols, bool ptq1)
    {
        if (cols <= 0 || values.Length % cols != 0)
        {
            throw new GgufException(
                $"ternary encode needs a row width that divides {values.Length} values, got {cols}");
        }
        int rows = values.Length / cols;

        // EncodePtq1/EncodePq2 rotate their argument in place, so hand them a
        // copy and keep the caller's array intact.
        var (packed, scales) = ptq1
            ? Ternary.EncodePtq1((float[])values.Clone(), rows, cols)
            : Ternary.EncodePq2((float[])values.Clone(), rows, cols);

        int packedPerBlock = ptq1 ? Ternary.Ptq1BytesPerBlock : Ternary.Pq2BytesPerBlock;
        int blocks = scales.Length / 2;
        int blockBytes = 2 + packedPerBlock;
        if (packed.Length != blocks * packedPerBlock)
        {
            throw new GgufException(
                $"ternary encode produced {packed.Length} packed bytes for {blocks} blocks, "
                + $"expected {blocks * packedPerBlock}");
        }

        var result = new byte[blocks * blockBytes];
        for (int b = 0; b < blocks; b++)
        {
            int o = b * blockBytes;
            result[o] = scales[b * 2];
            result[o + 1] = scales[b * 2 + 1];
            Array.Copy(packed, b * packedPerBlock, result, o + 2, packedPerBlock);
        }
        return result;
    }

    /// <summary>ggml's <c>GGML_E8M0_TO_FP32_HALF</c>. Because
    /// <see cref="Fp4KValues"/> holds the grid doubled, the scale that pairs
    /// with it is halved: 2^(e-128) rather than the plain E8M0 decode
    /// 2^(e-127). Computed by exponent shift so the subnormal results for
    /// e = 0 and e = 1 stay exact.</summary>
    private static float E8M0ToHalfScale(byte e) => MathF.ScaleB(1f, e - 128);

    /// <summary>Nearest point on the scaled grid, with the lower index winning
    /// a tie — the strict comparison is what makes ggml's
    /// <c>best_index_mxfp4</c> return code 0 rather than negative zero for a
    /// small negative value.</summary>
    private static byte BestFp4Code(float value, float scale)
    {
        byte best = 0;
        float bestError = MathF.Abs(Fp4KValues[0] * scale - value);
        for (int i = 1; i < Fp4KValues.Length; i++)
        {
            float error = MathF.Abs(Fp4KValues[i] * scale - value);
            if (error < bestError)
            {
                best = (byte)i;
                bestError = error;
            }
        }
        return best;
    }

    private static float MaxAbs(float[] values, int start)
    {
        float amax = 0f;
        for (int i = 0; i < BlockElements; i++)
        {
            float a = MathF.Abs(values[start + i]);
            if (amax < a)
            {
                amax = a;
            }
        }
        return amax;
    }

    private static int RequireWholeBlocks(int elements)
    {
        if (elements % BlockElements != 0)
        {
            throw new GgufException(
                $"cannot block-quantize {elements} values: not a multiple of {BlockElements}");
        }
        return elements / BlockElements;
    }
}
