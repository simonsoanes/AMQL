using System.Numerics;

namespace Amql.Inference;

/// <summary>A dense row-major 2-D float tensor. The minimal substrate the
/// generic ops need; deliberately no external numerics dependency.</summary>
public sealed class Tensor2D
{
    public float[] Data { get; }

    public int Rows { get; }
    public int Cols { get; }

    public Tensor2D(float[] data, int rows, int cols)
    {
        if (data.Length != checked(rows * cols))
        {
            throw new ArgumentException($"data length {data.Length} != {rows}x{cols}", nameof(data));
        }
        Data = data;
        Rows = rows;
        Cols = cols;
    }

    public static Tensor2D Zeros(int rows, int cols) => new(new float[rows * cols], rows, cols);

    public static Tensor2D FromRowMajor(IEnumerable<float> values, int rows, int cols) =>
        new(values.ToArray(), rows, cols);

    public float this[int row, int col] => Data[row * Cols + col];

    public void Set(int row, int col, float value) => Data[row * Cols + col] = value;

    public Span<float> Row(int row) => Data.AsSpan(row * Cols, Cols);

    public Tensor2D Clone() => new((float[])Data.Clone(), Rows, Cols);

    /// <summary>First row (used for single-row outputs like logits).</summary>
    public ReadOnlySpan<float> FirstRow() => Row(0);

    public override string ToString() => $"Tensor2D[{Rows}x{Cols}]";
}

public static class TensorOps
{
    /// <summary>c = a @ b. Row-major, i-k-j accumulation for cache
    /// locality; `Vector<float>`-accelerated along j when the inner dim
    /// is large enough to amortise the overhead.</summary>
    public static Tensor2D MatMul(Tensor2D a, Tensor2D b)
    {
        if (a.Cols != b.Rows)
        {
            throw new ArgumentException($"matmul shape mismatch: {a.Rows}x{a.Cols} @ {b.Rows}x{b.Cols}");
        }
        int m = a.Rows, k = a.Cols, n = b.Cols;
        var result = new float[m * n];
        int vectorWidth = Vector<float>.Count;

        if (k >= 16)
        {
            // i-k-j with vectorised j runs.
            for (int i = 0; i < m; i++)
            {
                var aRow = a.Data.AsSpan(i * k, k);
                var cRow = result.AsSpan(i * n, n);
                for (int j = 0; j < n; j++)
                {
                    cRow[j] = 0f;
                }
                for (int kk = 0; kk < k; kk++)
                {
                    float av = aRow[kk];
                    if (av == 0f)
                    {
                        continue;
                    }
                    var bRow = b.Data.AsSpan(kk * n, n);
                    int j = 0;
                    for (; j + vectorWidth <= n; j += vectorWidth)
                    {
                        var acc = new Vector<float>(cRow.Slice(j, vectorWidth));
                        var bv = new Vector<float>(bRow.Slice(j, vectorWidth));
                        (acc + Vector<float>.One * av * bv).CopyTo(cRow.Slice(j, vectorWidth));
                    }
                    for (; j < n; j++)
                    {
                        cRow[j] += av * bRow[j];
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float acc = 0f;
                    for (int kk = 0; kk < k; kk++)
                    {
                        acc += a.Data[i * k + kk] * b.Data[kk * n + j];
                    }
                    result[i * n + j] = acc;
                }
            }
        }
        return new Tensor2D(result, m, n);
    }

    /// <summary>x @ W^T where W is [out, in] — the weight convention
    /// (rows are the output space).</summary>
    public static Tensor2D MatMulTransposedB(Tensor2D x, Tensor2D w)
    {
        if (x.Cols != w.Cols)
        {
            throw new ArgumentException($"shape mismatch: x {x.Rows}x{x.Cols}, weight {w.Rows}x{w.Cols} (weight rows are output)");
        }
        return MatMul(x, Transpose(w));
    }

    public static Tensor2D Transpose(Tensor2D a)
    {
        var result = new float[a.Cols * a.Rows];
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Cols; j++)
            {
                result[j * a.Rows + i] = a.Data[i * a.Cols + j];
            }
        }
        return new Tensor2D(result, a.Cols, a.Rows);
    }

    /// <summary>Row-wise dot product of a with b (same length).</summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float acc = 0f;
        int vectorWidth = Vector<float>.Count;
        int i = 0;
        if (a.Length >= vectorWidth)
        {
            var vAcc = Vector<float>.Zero;
            for (; i + vectorWidth <= a.Length; i += vectorWidth)
            {
                vAcc += new Vector<float>(a.Slice(i, vectorWidth)) * new Vector<float>(b.Slice(i, vectorWidth));
            }
            acc = Vector.Dot(vAcc, Vector<float>.One);
        }
        for (; i < a.Length; i++)
        {
            acc += a[i] * b[i];
        }
        return acc;
    }

    /// <summary>Gathers the <c>indices.Length</c> rows of <c>table</c> into
    /// a dense matrix — embedding lookup.</summary>
    public static Tensor2D GatherRows(Tensor2D table, ReadOnlySpan<int> indices)
    {
        var result = new float[indices.Length * table.Cols];
        for (int r = 0; r < indices.Length; r++)
        {
            table.Row(indices[r]).CopyTo(result.AsSpan(r * table.Cols, table.Cols));
        }
        return new Tensor2D(result, indices.Length, table.Cols);
    }
}