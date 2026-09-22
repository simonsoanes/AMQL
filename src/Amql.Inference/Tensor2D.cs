using System.Numerics;

namespace Amql.Inference;

/// <summary>A dense row-major 2-D float tensor. The minimal substrate the
/// generic ops need; deliberately no external numerics dependency.</summary>
public sealed class Tensor2D
{
    public float[] Data { get; }

    public int Rows { get; }
    public int Cols { get; }

    /// <summary>When non-zero, a device-resident FP16 copy of this tensor
    /// (a weight the MXFP4 working set uploaded once) exists and
    /// <see cref="TensorOps.MatMulTransposedB"/> routes the GEMM through
    /// CUDA. Zero (the default) keeps everything on the managed path.</summary>
    public IntPtr DeviceWeightF16 { get; set; }

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
    /// <summary>The dot-product parallelism floor: below this many
    /// multiply-accumulates the matmul runs single-threaded (thread
    /// setup would dominate).</summary>
    private const long ParallelFloorMacs = 4L << 20; // 4M MACs

    /// <summary>The CUDA dispatch floor: below this many
    /// multiply-accumulates the GEMM stays on the managed path (launch
    /// and copy-back would dominate the work).</summary>
    private const long CudaMinimumMacs = 8L << 20; // 8M MACs

    /// <summary>c = a @ b. Row-major, i-k-j accumulation for cache
    /// locality; `Vector<float>`-accelerated along j when the inner dim
    /// is large enough to amortise the overhead. Output rows parallelise
    /// across cores when the work is worth it.</summary>
    public static Tensor2D MatMul(Tensor2D a, Tensor2D b)
    {
        if (a.Cols != b.Rows)
        {
            throw new ArgumentException($"matmul shape mismatch: {a.Rows}x{a.Cols} @ {b.Rows}x{b.Cols}");
        }
        int m = a.Rows, k = a.Cols, n = b.Cols;
        long macs = (long)m * k * n;
        if (macs < ParallelFloorMacs || k < 16)
        {
            return MatMulCore(a, b, 0, m);
        }

        int workers = ParallelWorkerCount(m);
        if (workers <= 1)
        {
            return MatMulCore(a, b, 0, m);
        }
        var result = new float[m * n];
        int perWorker = (m + workers - 1) / workers;
        Parallel.For(0, workers, w =>
        {
            int i0 = w * perWorker;
            int i1 = Math.Min(m, i0 + perWorker);
            if (i0 < i1)
            {
                MatMulCoreInto(a, b, result, i0, i1);
            }
        });
        return new Tensor2D(result, m, n);
    }

    /// <summary>Sequential i-k-j matmul over rows [i0, i1) — the shared
    /// core the parallel wrapper splits.</summary>
    private static Tensor2D MatMulCore(Tensor2D a, Tensor2D b, int i0, int i1)
    {
        int m = a.Rows, k = a.Cols, n = b.Cols;
        var result = new float[(i1 - i0) * n];
        MatMulCoreInto(a, b, result, i0, i1);
        return new Tensor2D(result, i1 - i0, n);
    }

    private static void MatMulCoreInto(Tensor2D a, Tensor2D b, float[] result, int i0, int i1)
    {
        int k = a.Cols, n = b.Cols;
        int vectorWidth = Vector<float>.Count;
        int rows = i1 - i0;

        for (int r = 0; r < rows; r++)
        {
            int i = i0 + r;
            var aRow = a.Data.AsSpan(i * k, k);
            var cRow = result.AsSpan(r * n, n);
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

    /// <summary>
    /// x @ W^T where W is [out, in] — the weight convention (rows are the
    /// output space). Computed directly as per-output-element dot products
    /// against the weight rows: no transposed copy is materialised (a
    /// weight can be hundreds of MB on the 27B, and the naive
    /// transpose-every-call path copies it once per projection per token),
    /// and the output columns parallelise across cores — a single-row
    /// decode still fans out over the full output width.
    /// </summary>
    public static Tensor2D MatMulTransposedB(Tensor2D x, Tensor2D w)
    {
        if (x.Cols != w.Cols)
        {
            throw new ArgumentException($"shape mismatch: x {x.Rows}x{x.Cols}, weight {w.Rows}x{w.Cols} (weight rows are output)");
        }
        int m = x.Rows, k = x.Cols, n = w.Rows;
        long macs = (long)m * k * n;

        // CUDA Try-gate: when the weight carries a device-resident FP16
        // copy (MXFP4 working set + AMQL_GPU) and the GEMM is big enough
        // to amortise the launch, run it on the device. Any failure falls
        // through to the managed kernels below — the device is never a
        // correctness dependency.
        if (w.DeviceWeightF16 != IntPtr.Zero && macs >= CudaMinimumMacs &&
            CudaShim.TryGemmTransposedB(x.Data, m, k, w.DeviceWeightF16, n, out var gpuResult))
        {
            return new Tensor2D(gpuResult!, m, n);
        }

        if (macs < ParallelFloorMacs)
        {
            var result = new float[m * n];
            for (int i = 0; i < m; i++)
            {
                var xRow = x.Data.AsSpan(i * k, k);
                var cRow = result.AsSpan(i * n, n);
                for (int j = 0; j < n; j++)
                {
                    cRow[j] = Dot(xRow, w.Data.AsSpan(j * k, k));
                }
            }
            return new Tensor2D(result, m, n);
        }

        int workers = Math.Min(ComputeBudget.Cores, Math.Max(1, n / 32));
        if (workers <= 1)
        {
            var result = new float[m * n];
            for (int i = 0; i < m; i++)
            {
                var xRow = x.Data.AsSpan(i * k, k);
                var cRow = result.AsSpan(i * n, n);
                for (int j = 0; j < n; j++)
                {
                    cRow[j] = Dot(xRow, w.Data.AsSpan(j * k, k));
                }
            }
            return new Tensor2D(result, m, n);
        }

        var output = new float[m * n];
        int colsPerWorker = (n + workers - 1) / workers;
        Parallel.For(0, workers, worker =>
        {
            int j0 = worker * colsPerWorker;
            int j1 = Math.Min(n, j0 + colsPerWorker);
            if (j0 >= j1)
            {
                return;
            }
            for (int i = 0; i < m; i++)
            {
                var xRow = x.Data.AsSpan(i * k, k);
                var cRow = output.AsSpan(i * n, n);
                for (int j = j0; j < j1; j++)
                {
                    cRow[j] = Dot(xRow, w.Data.AsSpan(j * k, k));
                }
            }
        });
        return new Tensor2D(output, m, n);
    }

    /// <summary>
    /// GPU-batched transposed-B GEMM: a single activation <c>x</c> against
    /// <c>weights</c> weight matrices, all on the CUDA stream, synced once.
    /// Returns the results in the same order as the weight span.  Falls
    /// back to the CPU path when the GPU is not available or any weight
    /// lacks a device pointer.
    /// </summary>
    public static Tensor2D[] MatMulTransposedBMulti(
        Tensor2D x, ReadOnlySpan<Tensor2D> weights)
    {
        int m = x.Rows, k = x.Cols;
        if (weights.Length == 0)
        {
            return Array.Empty<Tensor2D>();
        }

        // All weights must carry a device pointer and be above the MACs
        // floor for the GPU fast path.
        bool canGpu = CudaShim.Enabled;
        if (canGpu)
        {
            foreach (var w in weights)
            {
                if (w.DeviceWeightF16 == IntPtr.Zero)
                {
                    canGpu = false;
                    break;
                }
            }
        }

        if (canGpu)
        {
            // GPU fast path: upload the activation once, launch all GEMMs
            // async, sync once, download all results.
            IntPtr aDev = CudaShim.UploadActivationF16(x.Data, m, k);
            if (aDev != IntPtr.Zero)
            {
                var results = new Tensor2D[weights.Length];
                var outputBuffers = new float[weights.Length][];
                bool allLaunched = true;
                for (int i = 0; i < weights.Length; i++)
                {
                    int n = weights[i].Rows;
                    var c = new float[m * n];
                    outputBuffers[i] = c;
                    if (!CudaShim.LaunchGemmDeviceA(aDev, weights[i].DeviceWeightF16, c, m, k, n))
                    {
                        allLaunched = false;
                        break;
                    }
                }
                if (allLaunched && CudaShim.Sync())
                {
                    for (int i = 0; i < weights.Length; i++)
                    {
                        results[i] = new Tensor2D(outputBuffers[i], m, weights[i].Rows);
                    }
                    CudaShim.FreeActivation(aDev);
                    return results;
                }
                CudaShim.FreeActivation(aDev);
                // Fall through to CPU on any GPU failure.
            }
        }

        // CPU fallback: run each GEMM independently.
        var cpuResults = new Tensor2D[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            cpuResults[i] = MatMulTransposedB(x, weights[i]);
        }
        return cpuResults;
    }

    /// <summary>How many cores to split a matmul's output rows across —
    /// capped by the row count and the process-wide compute budget (the
    /// machine is usually running other work).</summary>
    private static int ParallelWorkerCount(int rows) =>
        Math.Min(ComputeBudget.Cores, Math.Max(1, rows));

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