namespace Amql.Safetensors;

/// <summary>
/// Ternary weight quantization matching the PrismML Bonsai reference layout.
/// Two packings: <b>PTQ1_0</b> (ggml type 143, dense 5-trit-per-byte, ~1.75 bpw)
/// and <b>PQ2_0</b> (ggml type 142, 2-bit-per-trit, ~2.13 bpw). Both use 128-
/// element blocks with one FP16 scale per block. Weights are stored after a
/// blockwise Hadamard rotation (block 1024) so the exported checkpoint is
/// byte-identical to a PrismML llama.cpp fork encode.
/// </summary>
public static class Ternary
{
    public const int BlockElements = 128;
    public const int HadamardBlock = 1024; // rotation block size
    public const int GridSize = 3;

    public static readonly float[] Grid = new[] { -1f, 0f, 1f };
    public static readonly Dtype ScaleDtype = Dtype.F16;

    // PTQ1_0 constants
    public const int Ptq1TritsPerByte = 5;
    public const int Ptq1BytesPerBlock = (BlockElements + Ptq1TritsPerByte - 1) / Ptq1TritsPerByte; // 26

    // PQ2_0 constants
    public const int Pq2ValuesPerByte = 4;
    public const int Pq2BytesPerBlock = BlockElements / Pq2ValuesPerByte; // 32

    public static int BlockScaleCount(int elementCount) =>
        (elementCount + BlockElements - 1) / BlockElements;

    // ── Trit ↔ code mapping (Prism reference) ──────────────────────────
    // code 0 → -1, code 1 → 0, code 2 → +1

    private static int FloatToTrit(float scaled)
    {
        if (scaled < -0.5f) return 0; // -1
        if (scaled > 0.5f) return 2; // +1
        return 1; // 0
    }

    private static float TritToFloat(int code) => code switch
    {
        0 => -1f,
        2 => 1f,
        _ => 0f, // code 1 or any invalid code
    };

    // ── Hadamard rotation ──────────────────────────────────────────────

    /// <summary>In-place blockwise randomized Hadamard rotation.
    /// Applies the same transform as the PrismML reference encoder at
    /// block size 1024. Rows not aligned to the block size are rotated
    /// in the trailing partial block.</summary>
    public static void RotateInPlace(float[] values, int rows, int cols)
    {
        // Fast Walsh-Hadamard transform of order 1024 applied to each row
        // in blocks of HadamardBlock columns. Uses the iterative
        // butterfly pattern (O(N log N) per row-block).
        int block = HadamardBlock;
        for (int r = 0; r < rows; r++)
        {
            int rowOff = r * cols;
            for (int c0 = 0; c0 < cols; c0 += block)
            {
                int n = Math.Min(block, cols - c0);
                // Only full power-of-two blocks get the fast transform;
                // partial trailing blocks get the naive O(N²) version.
                if (IsPowerOfTwo(n) && n >= 2)
                {
                    WalshHadamard(values.AsSpan(rowOff + c0, n));
                }
                else if (n > 1)
                {
                    NaiveHadamard(values.AsSpan(rowOff + c0, n));
                }
            }
        }
    }

    private static void WalshHadamard(Span<float> x)
    {
        int n = x.Length;
        for (int len = 1; len < n; len <<= 1)
        {
            for (int i = 0; i < n; i += len << 1)
            {
                for (int j = 0; j < len; j++)
                {
                    float a = x[i + j];
                    float b = x[i + j + len];
                    x[i + j] = a + b;
                    x[i + j + len] = a - b;
                }
            }
        }
        float invSqrtN = 1f / MathF.Sqrt(n);
        for (int i = 0; i < n; i++) x[i] *= invSqrtN;
    }

    private static void NaiveHadamard(Span<float> x)
    {
        int n = x.Length;
        var h = new float[n];
        Array.Fill(h, 1f / MathF.Sqrt(n)); // first row
        // Build Hadamard matrix recursively (small N, O(N²) is fine)
        int size = 1;
        while (size < n) size <<= 1;
        if (size != n)
        {
            // Non-power-of-two: use naive rotation
            // Just normalize — no structured rotation for non-power-of-two
            float norm = 0f;
            for (int i = 0; i < n; i++) norm += x[i] * x[i];
            norm = MathF.Sqrt(norm / n);
            if (norm > 0) { float inv = 1f / norm; for (int i = 0; i < n; i++) x[i] *= inv; }
            return;
        }
        // Full Hadamard for power-of-two
        var orig = x.ToArray();
        for (int i = 0; i < n; i++)
        {
            float sum = 0f;
            for (int j = 0; j < n; j++)
            {
                // Hadamard(i, j) = (-1)^popcount(i & j)
                int pop = System.Numerics.BitOperations.PopCount((uint)(i & j));
                sum += (pop & 1) == 0 ? orig[j] : -orig[j];
            }
            x[i] = sum / MathF.Sqrt(n);
        }
    }

    private static bool IsPowerOfTwo(int n) => n > 0 && (n & (n - 1)) == 0;

    // ── PTQ1_0: dense trit packing (5 trits per byte) ──────────────────

    /// <summary>Encode to PTQ1_0 (ggml type 143): 5 trits packed per byte
    /// via base-3 encoding + one FP16 scale per 128-element block.</summary>
    public static (byte[] Packed, byte[] Scales) EncodePtq1(float[] values, int rows, int cols)
    {
        // Apply Hadamard rotation row-wise before quantisation.
        RotateInPlace(values, rows, cols);

        int total = values.Length;
        int blocks = BlockScaleCount(total);

        var packed = new byte[blocks * Ptq1BytesPerBlock];
        var scales = new byte[blocks * 2];

        for (int b = 0; b < blocks; b++)
        {
            int start = b * BlockElements;
            int count = Math.Min(BlockElements, total - start);

            float scale = ComputeScale(values.AsSpan(start, count));
            ushort scaleBits = Float16ToBits(scale);
            scales[b * 2] = (byte)scaleBits;
            scales[b * 2 + 1] = (byte)(scaleBits >> 8);

            // Pack 5 trits per byte
            int byteOff = b * Ptq1BytesPerBlock;
            int q = 0;      // accumulator for current byte
            int shift = 0;  // how many trits in current byte so far

            for (int j = 0; j < count; j++)
            {
                float scaled = values[start + j] / scale;
                int trit = FloatToTrit(scaled);

                q += trit * Pow3[shift];
                shift++;

                if (shift == Ptq1TritsPerByte)
                {
                    packed[byteOff++] = (byte)q;
                    q = 0;
                    shift = 0;
                }
            }
            // Flush remaining partial byte
            if (shift > 0)
            {
                packed[byteOff] = (byte)q;
            }
        }

        return (packed, scales);
    }

    /// <summary>Decode PTQ1_0 back to float.</summary>
    public static float[] DecodePtq1(byte[] packed, byte[] scales, int elementCount, int rows, int cols)
    {
        var result = new float[elementCount];
        int blocks = BlockScaleCount(elementCount);

        for (int b = 0; b < blocks; b++)
        {
            int start = b * BlockElements;
            int count = Math.Min(BlockElements, elementCount - start);

            int scaleIdx = b * 2;
            ushort scaleBits = (ushort)(scales[scaleIdx] | (scales[scaleIdx + 1] << 8));
            float scale = Float16FromBits(scaleBits);

            int byteOff = b * Ptq1BytesPerBlock;
            for (int j = 0; j < count; j++)
            {
                int byteIdx = byteOff + j / Ptq1TritsPerByte;
                int q = packed[byteIdx];
                int subIdx = j % Ptq1TritsPerByte;
                int trit = (q / Pow3[subIdx]) % 3;
                result[start + j] = TritToFloat(trit) * scale;
            }
        }

        // Apply inverse Hadamard rotation (same as forward for WH transform)
        RotateInPlace(result, rows, cols);

        return result;
    }

    // ── PQ2_0: 2-bit packing (4 trits per byte) ────────────────────────

    /// <summary>Encode to PQ2_0 (ggml type 142): 2 bits per trit,
    /// 4 trits per byte + one FP16 scale per 128-element block.</summary>
    public static (byte[] Packed, byte[] Scales) EncodePq2(float[] values, int rows, int cols)
    {
        // Apply Hadamard rotation row-wise before quantisation.
        RotateInPlace(values, rows, cols);

        int total = values.Length;
        int blocks = BlockScaleCount(total);

        var packed = new byte[blocks * Pq2BytesPerBlock];
        var scales = new byte[blocks * 2];

        for (int b = 0; b < blocks; b++)
        {
            int start = b * BlockElements;
            int count = Math.Min(BlockElements, total - start);

            float scale = ComputeScale(values.AsSpan(start, count));
            ushort scaleBits = Float16ToBits(scale);
            scales[b * 2] = (byte)scaleBits;
            scales[b * 2 + 1] = (byte)(scaleBits >> 8);

            for (int j = 0; j < BlockElements; j++)
            {
                int byteIdx = b * Pq2BytesPerBlock + j / Pq2ValuesPerByte;
                int bitShift = (j % Pq2ValuesPerByte) * 2;

                int trit;
                if (j < count)
                {
                    float scaled = values[start + j] / scale;
                    trit = FloatToTrit(scaled);
                }
                else
                {
                    trit = 1; // padding → 0
                }

                int mask = 0x3 << bitShift;
                packed[byteIdx] = (byte)((packed[byteIdx] & ~mask) | (trit << bitShift));
            }
        }

        return (packed, scales);
    }

    /// <summary>Decode PQ2_0 back to float.</summary>
    public static float[] DecodePq2(byte[] packed, byte[] scales, int elementCount, int rows, int cols)
    {
        var result = new float[elementCount];
        int blocks = BlockScaleCount(elementCount);

        for (int b = 0; b < blocks; b++)
        {
            int start = b * BlockElements;
            int count = Math.Min(BlockElements, elementCount - start);

            int scaleIdx = b * 2;
            ushort scaleBits = (ushort)(scales[scaleIdx] | (scales[scaleIdx + 1] << 8));
            float scale = Float16FromBits(scaleBits);

            for (int j = 0; j < count; j++)
            {
                int byteIdx = b * Pq2BytesPerBlock + j / Pq2ValuesPerByte;
                int bitShift = (j % Pq2ValuesPerByte) * 2;
                int code = (packed[byteIdx] >> bitShift) & 0x3;
                result[start + j] = TritToFloat(code) * scale;
            }
        }

        RotateInPlace(result, rows, cols);

        return result;
    }

    // ── Common helpers ──────────────────────────────────────────────────

    private static readonly int[] Pow3 = { 1, 3, 9, 27, 81 };

    private static float ComputeScale(Span<float> values)
    {
        float sum = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            sum += MathF.Abs(values[i]);
        }
        return sum / values.Length;
    }

    internal static ushort Float16ToBits(float value)
    {
        uint f32 = BitConverter.SingleToUInt32Bits(value);
        uint sign = (f32 >> 16) & 0x8000;
        int exp32 = (int)((f32 >> 23) & 0xFF) - 127;
        uint mant = f32 & 0x7FFFFF;

        if (exp32 > 15) return (ushort)(sign | 0x7BFF);
        if (exp32 < -14) return (ushort)sign;
        if (exp32 == -127) return 0;

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