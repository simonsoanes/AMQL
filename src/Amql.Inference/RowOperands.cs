using System.Runtime.InteropServices;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// A 2-D weight read a row at a time: held as f32, or as its stored BF16/F16
/// bits (widened as each row is read, which is exact) when it is large. Every
/// product accumulates in double, one row at a time, so its result does not
/// depend on how the rows are spread over cores. The row-exact path's matrices.
/// </summary>
public sealed class RowMatrix
{
    private readonly float[]? _f32;
    private readonly byte[]? _half;
    private readonly bool _isF16;

    private RowMatrix(float[]? f32, byte[]? half, bool isF16, int rows, int cols)
    {
        _f32 = f32;
        _half = half;
        _isF16 = isF16;
        Rows = rows;
        Cols = cols;
    }

    public int Rows { get; }
    public int Cols { get; }

    /// <summary>16-bit payloads of at least this many elements stay 16-bit.</summary>
    public const long KeepHalfElements = 1L << 20;

    /// <summary>Matrices of at least this many elements spread their rows over cores.</summary>
    public const long ParallelElements = 1L << 18;

    internal static readonly ParallelOptions Budget = new() { MaxDegreeOfParallelism = Math.Max(1, ComputeBudget.Cores) };

    public static RowMatrix From(Dtype dtype, byte[] payload, int rows, int cols)
    {
        if (dtype is Dtype.BF16 or Dtype.F16 && (long)rows * cols >= KeepHalfElements)
        {
            if (payload.LongLength != (long)rows * cols * 2)
            {
                throw new ContainerException($"a {dtype.Label()} [{rows}x{cols}] matrix needs {(long)rows * cols * 2} bytes, not {payload.LongLength}");
            }
            return new RowMatrix(null, payload, dtype == Dtype.F16, rows, cols);
        }
        var values = BitPattern.WidenToF32(dtype, payload);
        if (values.LongLength != (long)rows * cols)
        {
            throw new ContainerException($"a [{rows}x{cols}] matrix needs {(long)rows * cols} values, not {values.LongLength}");
        }
        return new RowMatrix(values, null, false, rows, cols);
    }

    public static RowMatrix FromF32(float[] values, int rows, int cols) => new(values, null, false, rows, cols);

    public double Dot(int row, ReadOnlySpan<float> x)
    {
        if (_f32 is { } f)
        {
            return RowMath.Dot(f.AsSpan(checked(row * Cols), Cols), x);
        }
        var bits = MemoryMarshal.Cast<byte, ushort>(_half.AsSpan(checked(row * Cols * 2), Cols * 2));
        double sum = 0;
        if (_isF16)
        {
            for (int i = 0; i < bits.Length; i++)
            {
                sum += (double)(float)BitConverter.UInt16BitsToHalf(bits[i]) * x[i];
            }
        }
        else
        {
            for (int i = 0; i < bits.Length; i++)
            {
                sum += (double)BitConverter.Int32BitsToSingle(bits[i] << 16) * x[i];
            }
        }
        return sum;
    }

    public void Row(int row, Span<float> target)
    {
        if (_f32 is { } f)
        {
            f.AsSpan(checked(row * Cols), Cols).CopyTo(target);
            return;
        }
        var bits = MemoryMarshal.Cast<byte, ushort>(_half.AsSpan(checked(row * Cols * 2), Cols * 2));
        for (int i = 0; i < bits.Length; i++)
        {
            target[i] = _isF16 ? (float)BitConverter.UInt16BitsToHalf(bits[i]) : BitConverter.Int32BitsToSingle(bits[i] << 16);
        }
    }

    /// <summary><c>W·x</c> over the first <paramref name="rows"/> rows (all by default).</summary>
    public float[] MatVec(ReadOnlySpan<float> x, int rows = -1)
    {
        rows = rows < 0 ? Rows : rows;
        var y = new float[rows];
        if ((long)rows * Cols < ParallelElements)
        {
            for (int r = 0; r < rows; r++)
            {
                y[r] = (float)Dot(r, x);
            }
            return y;
        }
        var input = x.ToArray();
        Parallel.For(0, rows, Budget, r => y[r] = (float)Dot(r, input));
        return y;
    }
}

/// <summary>
/// One layer's experts, <c>[experts, rows, cols]</c>, as stored: plain, NVFP4
/// (FP4 elements, an E4M3 scale per 16 and the factor
/// <c>amax / denominator</c> per expert) or MXFP8 (E4M3 elements and an E8M0
/// exponent per 32). A row decodes in f32 as <c>(element × block scale) ×
/// factor</c>; only the rows a token routes to are decoded.
/// </summary>
public sealed class PackedExpertStack
{
    private readonly RowMatrix? _plain;
    private readonly Dtype _element;
    private readonly byte[]? _elements;
    private readonly byte[]? _scales;
    private readonly bool _e8m0;
    private readonly float[]? _factor;
    private readonly int _block;

    private static readonly float[] E4M3 = Enumerable.Range(0, 256).Select(i => BitPattern.DecodeF8E4M3((byte)i)).ToArray();
    private static readonly float[] E8M0 = Enumerable.Range(0, 256).Select(i => BitPattern.DecodeF8E8M0((byte)i)).ToArray();

    private PackedExpertStack(int experts, int rows, int cols, RowMatrix plain)
    {
        Experts = experts;
        Rows = rows;
        Cols = cols;
        _plain = plain;
    }

    private PackedExpertStack(int experts, int rows, int cols, Dtype element, byte[] elements, byte[] scales, bool e8m0, int block,
        float[]? factor)
    {
        Experts = experts;
        Rows = rows;
        Cols = cols;
        long count = (long)experts * rows * cols;
        if (cols % block != 0 || elements.LongLength != (element == Dtype.FP4 ? count / 2 : count) ||
            scales.LongLength != count / block || factor is not null && factor.Length != experts)
        {
            throw new ContainerException(
                $"expert stack [{experts}, {rows}, {cols}] of {element.Label()}: {elements.LongLength} element bytes, " +
                $"{scales.LongLength} scales of one per {block}{(factor is null ? string.Empty : $", {factor.Length} factors")} do not fit");
        }
        _element = element;
        _elements = elements;
        _scales = scales;
        _e8m0 = e8m0;
        _block = block;
        _factor = factor;
    }

    public int Experts { get; }
    public int Rows { get; }
    public int Cols { get; }

    /// <summary>How the stack is held: "plain", "NVFP4" or "MXFP8".</summary>
    public string Form => _plain is not null ? "plain" : _factor is not null ? "NVFP4" : "MXFP8";

    /// <summary>Resolves a packed operand in whichever format it is stored.</summary>
    public static PackedExpertStack Load(OperandStore store, PackedOperand operand, int experts, int rows, int cols)
    {
        var e = operand.Elements;
        var dtype = store.DtypeOf(e.ObjectId, e.TensorName);
        var shape = store.ShapeOf(e.ObjectId, e.TensorName);
        if (!shape.SequenceEqual(new long[] { experts, rows, cols }))
        {
            throw new ContainerException($"{e.ObjectId}/{e.TensorName}: expected [{experts}x{rows}x{cols}], found [{string.Join("x", shape)}]");
        }
        switch (dtype)
        {
            case Dtype.FP4:
            {
                var scale = Companion(operand.Scale, "scale");
                var amaxRef = operand.Amax ?? throw Missing("amax");
                var denominatorRef = operand.Denominator ?? throw Missing("denominator");
                if (scale.Dtype != Dtype.F8_E4M3 || !scale.Shape.SequenceEqual(new long[] { experts, rows, cols / 16 }))
                {
                    throw new ContainerException($"{e.TensorName}: NVFP4 needs an F8_E4M3 scale [{experts}x{rows}x{cols / 16}]");
                }
                var amax = Vector(store, amaxRef, experts);
                var denominator = Vector(store, denominatorRef, 1);
                var factor = new float[experts];
                for (int i = 0; i < experts; i++)
                {
                    factor[i] = amax[i] / denominator[0];
                }
                return new PackedExpertStack(experts, rows, cols, Dtype.FP4, store.Resolve(e).Payload, scale.Payload, e8m0: false, block: 16, factor);
            }
            case Dtype.F8_E4M3:
            {
                var scale = Companion(operand.Scale, "scale");
                if (scale.Dtype is not (Dtype.U8 or Dtype.F8_E8M0) || !scale.Shape.SequenceEqual(new long[] { experts, rows, cols / 32 }))
                {
                    throw new ContainerException($"{e.TensorName}: MXFP8 needs U8 exponents [{experts}x{rows}x{cols / 32}]");
                }
                return new PackedExpertStack(experts, rows, cols, Dtype.F8_E4M3, store.Resolve(e).Payload, scale.Payload, e8m0: true, block: 32, factor: null);
            }
            default:
                return new PackedExpertStack(experts, rows, cols, RowMatrix.From(dtype, store.Resolve(e).Payload, checked(experts * rows), cols));
        }

        OperandResolution Companion(OperandRef? r, string part) => store.Resolve(r ?? throw Missing(part));
        ContainerException Missing(string part) => new($"{e.TensorName} is stored {dtype.Label()} but its '{part}' is not bound");
    }

    private static float[] Vector(OperandStore store, OperandRef r, int length)
    {
        var resolved = store.Resolve(r);
        var values = BitPattern.WidenToF32(resolved.Dtype, resolved.Payload);
        if (values.Length != length)
        {
            throw new ContainerException($"{r.TensorName}: expected {length} values, found {values.Length}");
        }
        return values;
    }

    public void Row(int expert, int row, Span<float> target)
    {
        if (_plain is { } plain)
        {
            plain.Row(checked(expert * Rows + row), target);
            return;
        }
        long index = (long)expert * Rows + row;
        int rowBytes = _element == Dtype.FP4 ? Cols / 2 : Cols;
        int blocks = Cols / _block;
        Span<float> scale = blocks <= 512 ? stackalloc float[blocks] : new float[blocks];
        var scaleBytes = _scales.AsSpan(checked((int)(index * blocks)), blocks);
        var table = _e8m0 ? E8M0 : E4M3;
        for (int b = 0; b < blocks; b++)
        {
            scale[b] = table[scaleBytes[b]];
        }
        ScaledMatrix.DequantRowOf(_element, _elements.AsSpan(checked((int)(index * rowBytes)), rowBytes), Cols,
            scale, _block, _factor?[expert] ?? 1f, target);
    }
}

/// <summary>
/// The row-exact path's operands, resolved once from the store and kept for
/// the session: matrices as <see cref="RowMatrix"/>, small tensors widened to
/// f32, expert stacks as stored. Shared by sibling runtimes over the same weights.
/// </summary>
public sealed class RowOperands
{
    private readonly OperandStore _store;
    private readonly Dictionary<OperandRef, RowMatrix> _matrices = new();
    private readonly Dictionary<OperandRef, float[]> _vectors = new();
    private readonly Dictionary<OperandRef, PackedExpertStack> _experts = new();
    private readonly object _lock = new();

    public RowOperands(OperandStore store) => _store = store;

    /// <summary>A 2-D (or [rows, 1, cols]) weight of exactly rows × cols elements.</summary>
    public RowMatrix Matrix(OperandRef operand, int rows, int cols)
    {
        lock (_lock)
        {
            if (_matrices.TryGetValue(operand, out var cached))
            {
                return cached;
            }
            var r = _store.Resolve(operand);
            long elements = r.Shape.Aggregate(1L, (a, b) => a * b);
            if (elements != (long)rows * cols)
            {
                throw new ContainerException(
                    $"{operand.ObjectId}/{operand.TensorName}: [{string.Join("x", r.Shape)}] is not [{rows}x{cols}]");
            }
            var m = RowMatrix.From(r.Dtype, r.Payload, rows, cols);
            _matrices[operand] = m;
            return m;
        }
    }

    /// <summary>A tensor of exactly <paramref name="length"/> elements, widened to f32.</summary>
    public float[] Vector(OperandRef operand, int length)
    {
        lock (_lock)
        {
            if (_vectors.TryGetValue(operand, out var cached))
            {
                return cached;
            }
            var r = _store.Resolve(operand);
            var values = BitPattern.WidenToF32(r.Dtype, r.Payload);
            if (values.Length != length)
            {
                throw new ContainerException($"{operand.ObjectId}/{operand.TensorName}: {values.Length} values, expected {length}");
            }
            _vectors[operand] = values;
            return values;
        }
    }

    /// <summary>The leading dimension of a tensor as stored.</summary>
    public long RowsOf(OperandRef operand) => _store.ShapeOf(operand.ObjectId, operand.TensorName)[0];

    /// <summary>Row <paramref name="row"/> of a [n, width] tensor.</summary>
    public float[] VectorRow(OperandRef operand, int rows, int width, int row) =>
        Vector(operand, rows * width).AsSpan(row * width, width).ToArray();

    public PackedExpertStack Experts(PackedOperand operand, int experts, int rows, int cols)
    {
        lock (_lock)
        {
            if (!_experts.TryGetValue(operand.Elements, out var stack))
            {
                stack = PackedExpertStack.Load(_store, operand, experts, rows, cols);
                _experts[operand.Elements] = stack;
            }
            return stack;
        }
    }
}

/// <summary>The row-exact path's arithmetic: double accumulation, f32 results.</summary>
internal static class RowMath
{
    public static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }
        return sum;
    }

    /// <summary>RMS-normalises without a weight: <c>x / √(mean(x²) + eps)</c>.</summary>
    public static float[] Unweighted(ReadOnlySpan<float> x, double eps)
    {
        double sum = 0;
        foreach (float v in x)
        {
            sum += (double)v * v;
        }
        double inv = 1.0 / Math.Sqrt(sum / x.Length + eps);
        var y = new float[x.Length];
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = (float)(x[i] * inv);
        }
        return y;
    }

    /// <summary>The weighted RMS norm: the unweighted one, times (weight + offset).</summary>
    public static float[] RmsNorm(ReadOnlySpan<float> x, float[] weight, double eps, float offset = 0f)
    {
        var y = Unweighted(x, eps);
        for (int i = 0; i < y.Length; i++)
        {
            y[i] *= offset == 0f ? weight[i] : weight[i] + offset;
        }
        return y;
    }

    /// <summary>Scales a vector longer than the threshold down to it.</summary>
    public static void Clip(Span<float> x, double threshold)
    {
        double sum = 0;
        foreach (float v in x)
        {
            sum += (double)v * v;
        }
        Scale(x, Math.Min(1.0, threshold / (Math.Sqrt(sum) + 1e-6)));
    }

    public static void Scale(Span<float> x, double factor)
    {
        if (factor == 1.0)
        {
            return;
        }
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)(x[i] * factor);
        }
    }

    public static float[] Add(float[] a, float[] b)
    {
        var y = new float[a.Length];
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = a[i] + b[i];
        }
        return y;
    }
}
