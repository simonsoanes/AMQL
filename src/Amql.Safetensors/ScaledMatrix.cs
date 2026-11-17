namespace Amql.Safetensors;

/// <summary>
/// A 2-D weight stored as low-precision elements times block scales: the
/// layouts quantised checkpoints ship (NVIDIA ModelOpt, DeepSeek/Qwen FP8,
/// OCP MX). The value of element (r, c) is
/// <c>element(r, c) × scale[r / BlockRows, c / BlockCols] × Global</c>.
/// <list type="bullet">
///   <item>NVFP4: FP4 E2M1 elements, an E4M3 scale per 16 columns, and an F32
///   tensor scale (<c>weight_scale_2</c>).</item>
///   <item>MXFP4: FP4 elements and an E8M0 scale per 32 columns.</item>
///   <item>MXFP8: E4M3 elements and an E8M0 scale per 32 columns.</item>
///   <item>FP8 block scales: E4M3 elements and a scale per 128×128 block
///   (<c>weight_scale_inv</c>, which despite the name multiplies).</item>
///   <item>FP8 per tensor: E4M3 elements and one scale.</item>
/// </list>
/// The elements stay as stored (FP4 two to a byte, low nibble first), so a
/// matrix held this way costs what the checkpoint does. Scales are kept
/// decoded, which for NVFP4 adds a quarter of a byte per element.
/// </summary>
public sealed class ScaledMatrix
{
    public ScaledMatrix(Dtype element, byte[] elements, int rows, int cols,
        float[] scales, int scaleRows, int scaleCols, float global)
    {
        if (element is not (Dtype.FP4 or Dtype.F8_E4M3 or Dtype.F8_E5M2 or Dtype.I8))
        {
            throw new SafetensorsException($"{element.Label()} is not a scaled element type");
        }
        if (element == Dtype.FP4 && cols % 2 != 0)
        {
            throw new SafetensorsException($"FP4 rows must hold an even number of elements, not {cols}");
        }
        long expected = element == Dtype.FP4 ? (long)rows * cols / 2 : (long)rows * cols;
        if (elements.LongLength != expected)
        {
            throw new SafetensorsException(
                $"{element.Label()} [{rows}, {cols}] needs {expected} bytes, the payload holds {elements.LongLength}");
        }
        if (scaleRows <= 0 || scaleCols <= 0 || scales.LongLength != (long)scaleRows * scaleCols)
        {
            throw new SafetensorsException($"scale grid [{scaleRows}, {scaleCols}] does not match {scales.LongLength} scales");
        }
        BlockRows = (rows + scaleRows - 1) / scaleRows;
        BlockCols = (cols + scaleCols - 1) / scaleCols;
        if ((rows + BlockRows - 1) / BlockRows != scaleRows || (cols + BlockCols - 1) / BlockCols != scaleCols)
        {
            throw new SafetensorsException(
                $"a [{scaleRows}, {scaleCols}] scale grid does not tile a [{rows}, {cols}] matrix in equal blocks");
        }
        Element = element;
        Elements = elements;
        Rows = rows;
        Cols = cols;
        Scales = scales;
        ScaleRows = scaleRows;
        ScaleCols = scaleCols;
        Global = global;
    }

    public Dtype Element { get; }
    public byte[] Elements { get; }
    public int Rows { get; }
    public int Cols { get; }
    public float[] Scales { get; }
    public int ScaleRows { get; }
    public int ScaleCols { get; }
    public int BlockRows { get; }
    public int BlockCols { get; }
    public float Global { get; }

    /// <summary>NVFP4 exactly: FP4 elements, one scale per 16 columns of a row,
    /// each an E4M3 value. Only these go to the GPU as packs.</summary>
    public bool IsNvfp4 => _isNvfp4 ??=
        Element == Dtype.FP4 && BlockRows == 1 && BlockCols == Nvfp4.BlockElements && ScaleCols * Nvfp4.BlockElements == Cols &&
        Scales.All(s => BitPattern.DecodeF8E4M3(BitPattern.EncodeF8E4M3(s)) == s);

    private bool? _isNvfp4;

    /// <summary>MXFP8 exactly: E4M3 elements, one power-of-two scale per 32
    /// columns of a row, no tensor scale. These go to the GPU as stored.</summary>
    public bool IsMxfp8 => _isMxfp8 ??=
        Element == Dtype.F8_E4M3 && BlockRows == 1 && BlockCols == Mxfp8.BlockElements &&
        ScaleCols * Mxfp8.BlockElements == Cols && Global == 1f &&
        Scales.All(s => s > 0 && BitPattern.DecodeF8E8M0(BitPattern.EncodeF8E8M0(s)) == s);

    private bool? _isMxfp8;

    /// <summary>MXFP8 block scales as E8M0 exponent bytes (exact: they are powers of two).</summary>
    public byte[] Mxfp8ScaleBytes()
    {
        var bytes = new byte[Scales.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = BitPattern.EncodeF8E8M0(Scales[i]);
        }
        return bytes;
    }

    /// <summary>Bytes held: elements plus decoded scales.</summary>
    public long Bytes => Elements.LongLength + Scales.LongLength * 4;

    private static readonly float[] Fp4Table = Enumerable.Range(0, 16).Select(i => BitPattern.DecodeFp4((byte)i)).ToArray();
    private static readonly float[] E4M3Table = Enumerable.Range(0, 256).Select(i => BitPattern.DecodeF8E4M3((byte)i)).ToArray();
    private static readonly float[] E5M2Table = Enumerable.Range(0, 256).Select(i => BitPattern.DecodeF8E5M2((byte)i)).ToArray();

    /// <summary>Row <paramref name="row"/> as f32. Each value is computed as
    /// <c>(element × scale) × Global</c> in f32, the one definition every
    /// reader of a scaled tensor shares (see <see cref="DequantRowOf"/>).</summary>
    public void DequantRow(int row, Span<float> target)
    {
        long rowBytes = Element == Dtype.FP4 ? Cols / 2 : Cols;
        DequantRowOf(Element, Elements.AsSpan((int)(row * rowBytes), (int)rowBytes), Cols,
            Scales.AsSpan(row / BlockRows * ScaleCols, ScaleCols), BlockCols, Global, target);
    }

    /// <summary>
    /// Decodes one stored row: <paramref name="elements"/> are its bytes,
    /// <paramref name="scaleRow"/> the scales of its row block. Shared by the
    /// held matrix and by readers that take a few rows of a table straight
    /// from storage, so both produce the same values.
    /// </summary>
    public static void DequantRowOf(Dtype element, ReadOnlySpan<byte> elements, int cols,
        ReadOnlySpan<float> scaleRow, int blockCols, float global, Span<float> target)
    {
        if (target.Length < cols)
        {
            throw new ArgumentException($"row of {cols} values needs a larger target");
        }
        int blocks = (cols + blockCols - 1) / blockCols;
        for (int block = 0; block < blocks; block++)
        {
            float scale = scaleRow[block];
            int first = block * blockCols;
            int last = Math.Min(cols, first + blockCols);
            switch (element)
            {
                case Dtype.FP4:
                    for (int c = first; c < last; c++)
                    {
                        byte b = elements[c / 2];
                        int nibble = (c & 1) == 0 ? b & 0x0F : b >> 4;
                        target[c] = Fp4Table[nibble] * scale * global;
                    }
                    break;
                case Dtype.F8_E4M3:
                    for (int c = first; c < last; c++)
                    {
                        target[c] = E4M3Table[elements[c]] * scale * global;
                    }
                    break;
                case Dtype.F8_E5M2:
                    for (int c = first; c < last; c++)
                    {
                        target[c] = E5M2Table[elements[c]] * scale * global;
                    }
                    break;
                default:
                    for (int c = first; c < last; c++)
                    {
                        target[c] = (sbyte)elements[c] * scale * global;
                    }
                    break;
            }
        }
    }

    /// <summary>Rows <c>[first, first + count)</c> as f32 into <paramref name="target"/>,
    /// several rows per core.</summary>
    public void DequantRows(int first, int count, float[] target)
    {
        if (target.LongLength < (long)count * Cols)
        {
            throw new ArgumentException($"{count} rows of {Cols} need a larger target");
        }
        const int RowsPerTask = 16;
        int tasks = (count + RowsPerTask - 1) / RowsPerTask;
        Parallel.For(0, tasks, t =>
        {
            int end = Math.Min(count, (t + 1) * RowsPerTask);
            for (int r = t * RowsPerTask; r < end; r++)
            {
                DequantRow(first + r, target.AsSpan(r * Cols, Cols));
            }
        });
    }

    /// <summary>The whole matrix as f32.</summary>
    public float[] Dequant()
    {
        var values = GC.AllocateUninitializedArray<float>(checked(Rows * Cols));
        DequantRows(0, Rows, values);
        return values;
    }

    /// <summary>NVFP4 block scales re-encoded as E4M3 bytes (exact: they were
    /// E4M3 to begin with), the form the GPU kernel reads.</summary>
    public byte[] Nvfp4ScaleBytes()
    {
        var bytes = new byte[Scales.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = BitPattern.EncodeF8E4M3(Scales[i]);
        }
        return bytes;
    }
}

/// <summary>
/// The NVFP4 weight codec, as NVIDIA ModelOpt writes it: FP4 E2M1 elements
/// two to a byte (the even column in the low nibble), one E4M3 scale per 16
/// columns, and an F32 tensor scale chosen so the largest block scale lands
/// at E4M3's top (448). Used to put a BF16 or F32 weight on the GPU at four
/// bits (<c>--gpu-format nvfp4</c>) and to build test checkpoints;
/// a checkpoint that is already NVFP4 is read as it is.
/// </summary>
public static class Nvfp4
{
    public const int BlockElements = 16;

    /// <summary>Quantises a row-major [rows × cols] matrix (cols a multiple of 16).</summary>
    public static ScaledMatrix Quantize(ReadOnlySpan<float> values, int rows, int cols)
    {
        if (cols % BlockElements != 0)
        {
            throw new SafetensorsException($"NVFP4 needs a multiple of {BlockElements} columns, not {cols}");
        }
        if (values.Length != (long)rows * cols)
        {
            throw new SafetensorsException($"NVFP4 quantize: {values.Length} values for [{rows}, {cols}]");
        }
        float amax = 0f;
        foreach (var v in values)
        {
            amax = MathF.Max(amax, MathF.Abs(v));
        }
        float global = amax > 0 ? amax / (BitPattern.Fp4MaxValue * 448f) : 1f;
        int blocksPerRow = cols / BlockElements;
        var scales = new float[(long)rows * blocksPerRow];
        var packed = new byte[(long)rows * cols / 2];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int first = r * cols + b * BlockElements;
                float blockMax = 0f;
                for (int i = 0; i < BlockElements; i++)
                {
                    blockMax = MathF.Max(blockMax, MathF.Abs(values[first + i]));
                }
                float scale = BitPattern.DecodeF8E4M3(BitPattern.EncodeF8E4M3(blockMax / BitPattern.Fp4MaxValue / global));
                scales[r * blocksPerRow + b] = scale;
                float step = scale * global;
                for (int i = 0; i < BlockElements; i += 2)
                {
                    byte lo = step > 0 ? BitPattern.EncodeFp4(values[first + i] / step) : (byte)0;
                    byte hi = step > 0 ? BitPattern.EncodeFp4(values[first + i + 1] / step) : (byte)0;
                    packed[first / 2 + i / 2] = (byte)(lo | (hi << 4));
                }
            }
        }
        return new ScaledMatrix(Dtype.FP4, packed, rows, cols, scales, rows, blocksPerRow, global);
    }
}

/// <summary>
/// The MXFP8 weight codec (OCP MX, as ModelOpt writes it): E4M3 elements and
/// one E8M0 scale per 32 columns, the smallest power of two that keeps the
/// block's peak within E4M3's 448. Used to build test weights; a checkpoint
/// that is already MXFP8 is read as it is.
/// </summary>
public static class Mxfp8
{
    public const int BlockElements = 32;

    public static ScaledMatrix Quantize(ReadOnlySpan<float> values, int rows, int cols)
    {
        if (cols % BlockElements != 0 || values.Length != (long)rows * cols)
        {
            throw new SafetensorsException($"MXFP8 needs [{rows}, {cols}] with a multiple of {BlockElements} columns");
        }
        int blocksPerRow = cols / BlockElements;
        var scales = new float[(long)rows * blocksPerRow];
        var elements = new byte[(long)rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int first = r * cols + b * BlockElements;
                float peak = 0f;
                for (int i = 0; i < BlockElements; i++)
                {
                    peak = MathF.Max(peak, MathF.Abs(values[first + i]));
                }
                float scale = peak > 0 ? MathF.Pow(2f, MathF.Ceiling(MathF.Log2(peak / 448f))) : 1f;
                scales[r * blocksPerRow + b] = scale;
                for (int i = 0; i < BlockElements; i++)
                {
                    elements[first + i] = BitPattern.EncodeF8E4M3(values[first + i] / scale);
                }
            }
        }
        return new ScaledMatrix(Dtype.F8_E4M3, elements, rows, cols, scales, rows, blocksPerRow, 1f);
    }
}
