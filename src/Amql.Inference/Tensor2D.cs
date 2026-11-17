using System.Numerics;
using Amql.Safetensors;

namespace Amql.Inference;

/// <summary>A dense row-major 2-D float tensor. The minimal substrate the
/// generic ops need; deliberately no external numerics dependency.</summary>
public sealed class Tensor2D
{
    private float[]? _data;
    private ushort[]? _bf16;
    private ScaledMatrix? _scaled;

    /// <summary>The values as f32, row-major. A tensor built over BF16 bits
    /// (<see cref="FromBf16"/>) widens them here on first access — exact, BF16
    /// being the top half of an f32 — and from then on this array is the
    /// tensor: <see cref="Bf16Bits"/> turns null so the kernels read it too,
    /// and an in-place edit through it is seen by every later operation.</summary>
    public float[] Data => _data ?? Materialize();

    /// <summary>For a weight that lives on the GPU only (the lean working
    /// set): reads the host values again when a kernel needs them, which only
    /// happens when the device GEMM is not used. Null for an ordinary tensor.</summary>
    private Func<Tensor2D>? _reload;

    /// <summary>The BF16 bit patterns this tensor is held as, while
    /// <see cref="Data"/> has not been touched; null otherwise. The GEMV
    /// kernels widen these in registers, streaming half the bytes of the f32
    /// array for the same (bit-identical) result.</summary>
    public ushort[]? Bf16Bits
    {
        get
        {
            if (_reload is not null && _bf16 is null && _scaled is null && Volatile.Read(ref _data) is null)
            {
                Reload();
            }
            var bits = _bf16;
            return Volatile.Read(ref _data) is null ? bits : null;
        }
    }

    /// <summary>The low-precision elements and block scales this tensor is held
    /// as (a quantised checkpoint's weight, kept at its stored size), while
    /// <see cref="Data"/> has not been touched; null otherwise. Products with
    /// it decode a row block at a time (<see cref="TensorOps.MatMulTransposedB"/>).</summary>
    public ScaledMatrix? Scaled
    {
        get
        {
            if (_reload is not null && _bf16 is null && _scaled is null && Volatile.Read(ref _data) is null)
            {
                Reload();
            }
            return Volatile.Read(ref _data) is null ? _scaled : null;
        }
    }

    public int Rows { get; }
    public int Cols { get; }

    /// <summary>When non-zero, this weight is on the GPU as NVFP4 packs
    /// (elements, E4M3 block scales) with <see cref="DeviceNvfp4Global"/> as
    /// its tensor scale, and products with it run on the device.</summary>
    public IntPtr DeviceNvfp4 { get; set; }

    /// <summary>The tensor scale that goes with <see cref="DeviceNvfp4"/>.</summary>
    public float DeviceNvfp4Global { get; set; } = 1f;

    /// <summary>When non-zero, this weight is on the GPU as MXFP8 (E4M3
    /// elements, E8M0 block exponents), as the checkpoint stored it.</summary>
    public IntPtr DeviceMxfp8 { get; set; }

    /// <summary>Whether this weight lives on the GPU with no host copy kept
    /// (the lean working set's device weights), or as packs only the device
    /// multiplies (NVFP4, MXFP8): either way its products belong on the device.</summary>
    public bool DeviceFirst => (_reload is not null && OnGpu) || DeviceNvfp4 != IntPtr.Zero || DeviceMxfp8 != IntPtr.Zero;

    /// <summary>Whether a device copy of this weight exists, in any format.</summary>
    public bool OnGpu => DeviceWeightF16 != IntPtr.Zero || DeviceNvfp4 != IntPtr.Zero || DeviceMxfp8 != IntPtr.Zero;

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
        _data = data;
        Rows = rows;
        Cols = cols;
    }

    private Tensor2D(ushort[] bf16, int rows, int cols)
    {
        if (bf16.Length != checked(rows * cols))
        {
            throw new ArgumentException($"data length {bf16.Length} != {rows}x{cols}", nameof(bf16));
        }
        _bf16 = bf16;
        Rows = rows;
        Cols = cols;
    }

    private Tensor2D(int rows, int cols, IntPtr device, Func<Tensor2D> reload)
    {
        Rows = rows;
        Cols = cols;
        DeviceWeightF16 = device;
        _reload = reload;
    }

    private Tensor2D(ScaledMatrix scaled)
    {
        _scaled = scaled;
        Rows = scaled.Rows;
        Cols = scaled.Cols;
    }

    /// <summary>A tensor held as low-precision elements and block scales
    /// (see <see cref="Scaled"/>).</summary>
    public static Tensor2D FromScaled(ScaledMatrix scaled) => new(scaled);

    /// <summary>A weight held on the GPU as NVFP4 packs only, read back from
    /// <paramref name="reload"/> if a kernel needs it on the host.</summary>
    public static Tensor2D OnDeviceNvfp4(int rows, int cols, IntPtr device, float global, Func<Tensor2D> reload) =>
        new(rows, cols, IntPtr.Zero, reload) { DeviceNvfp4 = device, DeviceNvfp4Global = global };

    /// <summary>A weight held on the GPU as MXFP8 only.</summary>
    public static Tensor2D OnDeviceMxfp8(int rows, int cols, IntPtr device, Func<Tensor2D> reload) =>
        new(rows, cols, IntPtr.Zero, reload) { DeviceMxfp8 = device };

    /// <summary>Scaled tensors whose values had to be widened whole, because
    /// something read <see cref="Data"/> rather than multiplying.</summary>
    public static long ScaledWidenings => Interlocked.Read(ref _scaledWidenings);
    private static long _scaledWidenings;

    /// <summary>A weight held on the GPU only: <see cref="DeviceWeightF16"/> is
    /// set and nothing is kept on the host. A kernel that needs the host values
    /// after all (the device GEMM failed, or the product is too small for it)
    /// gets them from <paramref name="reload"/>, once; <see cref="HostReloads"/>
    /// counts how often that happens.</summary>
    public static Tensor2D OnDevice(int rows, int cols, IntPtr device, Func<Tensor2D> reload) =>
        new(rows, cols, device, reload);

    /// <summary>Host copies <see cref="OnDevice"/> tensors have had to read back.</summary>
    public static long HostReloads => Interlocked.Read(ref _hostReloads);
    private static long _hostReloads;

    private void Reload()
    {
        lock (_reload!)
        {
            if (_bf16 is not null || _data is not null || _scaled is not null)
            {
                return;
            }
            var host = _reload();
            Interlocked.Increment(ref _hostReloads);
            if (host.Bf16Bits is { } bits)
            {
                _bf16 = bits;
            }
            else if (host.Scaled is { } scaled)
            {
                _scaled = scaled;
            }
            else
            {
                Volatile.Write(ref _data, host.Data);
            }
        }
    }

    /// <summary>A tensor held as BF16 bit patterns (see <see cref="Bf16Bits"/>).</summary>
    public static Tensor2D FromBf16(ushort[] bits, int rows, int cols) => new(bits, rows, cols);

    private float[] Materialize()
    {
        if (_reload is not null && _bf16 is null && _scaled is null)
        {
            Reload();
            if (_data is { } loaded)
            {
                return loaded;
            }
        }
        if (_scaled is { } scaled)
        {
            var values = scaled.Dequant();
            Interlocked.Increment(ref _scaledWidenings);
            var won = Interlocked.CompareExchange(ref _data, values, null) ?? values;
            _scaled = null;
            return won;
        }
        var bits = _bf16!;
        var widened = GC.AllocateUninitializedArray<float>(bits.Length);
        BitPattern.WidenInto(Dtype.BF16,
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(bits.AsSpan()), widened);
        var winner = Interlocked.CompareExchange(ref _data, widened, null) ?? widened;
        _bf16 = null;
        return winner;
    }

    /// <summary>Copies one row as f32 into <paramref name="target"/> without
    /// widening the whole tensor (an embedding lookup over a BF16 table).</summary>
    public void CopyRowTo(int row, Span<float> target)
    {
        if (Scaled is { } scaled)
        {
            scaled.DequantRow(row, target);
            return;
        }
        var bits = Bf16Bits;
        if (bits is null)
        {
            Data.AsSpan(row * Cols, Cols).CopyTo(target);
            return;
        }
        BitPattern.WidenInto(Dtype.BF16,
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(bits.AsSpan(row * Cols, Cols)), target.Slice(0, Cols));
    }

    public static Tensor2D Zeros(int rows, int cols) => new(new float[rows * cols], rows, cols);

    public static Tensor2D FromRowMajor(IEnumerable<float> values, int rows, int cols) =>
        new(values.ToArray(), rows, cols);

    public float this[int row, int col] => Data[row * Cols + col];

    public void Set(int row, int col, float value) => Data[row * Cols + col] = value;

    public Span<float> Row(int row) => Data.AsSpan(row * Cols, Cols);

    public Tensor2D Clone() => Scaled is { } scaled
        ? FromScaled(scaled) // immutable: shared
        : Bf16Bits is { } bits
            ? FromBf16((ushort[])bits.Clone(), Rows, Cols)
            : new((float[])Data.Clone(), Rows, Cols);

    /// <summary>First row (used for single-row outputs like logits).</summary>
    public ReadOnlySpan<float> FirstRow() => Row(0);

    public override string ToString() => $"Tensor2D[{Rows}x{Cols}]";
}

public static class TensorOps
{
    /// <summary>The dot-product parallelism floor: below this many
    /// multiply-accumulates the matmul runs single-threaded (thread
    /// setup would dominate). <see cref="MatMul"/> keeps the old 4M floor:
    /// it still runs on <see cref="Parallel.For(int,int,Action{int})"/>.</summary>
    private const long ParallelFloorMacs = 4L << 20; // 4M MACs

    /// <summary>The floor for <see cref="MatMulTransposedB"/>, which runs on
    /// <see cref="KernelPool"/>: a loop there costs a few microseconds, so a
    /// projection is worth splitting once its weights are about a megabyte
    /// (one thread streams that in roughly 30 µs). Below the old 4M floor sat
    /// every FFN and linear-attention projection of a 1B-class model, which
    /// therefore ran on one core.</summary>
    private const long PoolFloorMacs = 256L << 10; // 256K MACs

    /// <summary>The CUDA dispatch floor: below this many
    /// multiply-accumulates the GEMM stays on the managed path (launch
    /// and copy-back would dominate the work).</summary>
    internal const long CudaMinimumMacs = 8L << 20; // 8M MACs

    /// <summary>Activation rows per GEMM tile (see <see cref="MatMulTransposedBCpu"/>):
    /// one micro-kernel's height. Four rows of activation sit in L1 while a
    /// long run of weight rows streams past, which measured fastest on a
    /// Zen 5 Threadripper (1.6 TFLOP/s at 512 rows, against 0.45 for
    /// weight-only splitting) — and the most tiles for load balance.</summary>
    private const int RowTile = 4;

    /// <summary>Weight rows per cache block inside a tile, in bytes.</summary>
    private const int BlockBytes = 128 * 1024;

    /// <summary>Tasks per thread a GEMV/GEMM is split into, so uneven progress
    /// (a core busy with other work) balances out. Finer splits measured slower.</summary>
    private const int ChunksPerThread = 4;

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
        if (w.DeviceNvfp4 != IntPtr.Zero &&
            CudaShim.TryGemmNvfp4(x.Data, m, k, w.DeviceNvfp4, n, w.DeviceNvfp4Global, out var fp4Result))
        {
            return new Tensor2D(fp4Result!, m, n);
        }
        if (w.DeviceMxfp8 != IntPtr.Zero &&
            CudaShim.TryGemmMxfp8(x.Data, m, k, w.DeviceMxfp8, n, out var fp8Result))
        {
            return new Tensor2D(fp8Result!, m, n);
        }
        if (w.DeviceWeightF16 != IntPtr.Zero && macs >= CudaMinimumMacs &&
            CudaShim.TryGemmTransposedB(x.Data, m, k, w.DeviceWeightF16, n, out var gpuResult))
        {
            return new Tensor2D(gpuResult!, m, n);
        }
        if (w.Scaled is { } scaled)
        {
            return ScaledProduct(x, scaled);
        }

        return MatMulTransposedBCpu(x, new[] { w })[0];
    }

    /// <summary>
    /// x @ Wᵢᵀ for several weights that share the input — a layer's q/k/v,
    /// an FFN's gate and up, the linear-attention in-projections. On the CPU
    /// the row chunks of every weight go out as ONE pool loop, so a decode
    /// step pays one dispatch and one ramp-up instead of one per projection;
    /// every output is the same per-row dot as <see cref="MatMulTransposedB"/>
    /// computes, so the results are bit-identical to calling it per weight.
    /// A weight with a device copy (the CUDA path) is computed on its own.
    /// </summary>
    public static Tensor2D[] MatMulTransposedBMany(Tensor2D x, params Tensor2D[] weights)
    {
        // Weights that belong on the device (see Tensor2D.DeviceFirst) share
        // one upload of x and one download of their products; any others (the
        // linear attention's small a/b projections stay on the host) run here.
        if (weights.Length > 1 && CudaShim.Enabled)
        {
            var onDevice = Enumerable.Range(0, weights.Length)
                .Where(i => weights[i].DeviceFirst && weights[i].Cols == x.Cols).ToArray();
            if (onDevice.Length > 1 &&
                CudaShim.TryMulti(x.Data, x.Rows, x.Cols, onDevice.Select(i => weights[i]).ToArray(), out var shared))
            {
                var results = new Tensor2D[weights.Length];
                for (int j = 0; j < onDevice.Length; j++)
                {
                    results[onDevice[j]] = shared![j];
                }
                var rest = Enumerable.Range(0, weights.Length).Where(i => results[i] is null).ToArray();
                if (rest.Length > 0)
                {
                    var hostResults = MatMulTransposedBMany(x, rest.Select(i => weights[i]).ToArray());
                    for (int j = 0; j < rest.Length; j++)
                    {
                        results[rest[j]] = hostResults[j];
                    }
                }
                return results;
            }
        }
        foreach (var w in weights)
        {
            if (x.Cols != w.Cols)
            {
                throw new ArgumentException($"shape mismatch: x {x.Rows}x{x.Cols}, weight {w.Rows}x{w.Cols} (weight rows are output)");
            }
            if (w.OnGpu || w.Scaled is not null)
            {
                var single = new Tensor2D[weights.Length];
                for (int i = 0; i < weights.Length; i++)
                {
                    single[i] = MatMulTransposedB(x, weights[i]);
                }
                return single;
            }
        }
        return MatMulTransposedBCpu(x, weights);
    }

    [ThreadStatic] private static Dictionary<int, float[]>? _scaledScratch;

    /// <summary>
    /// x times a scaled weight's transpose: the weight is decoded into a
    /// reusable f32 scratch matrix and multiplied by the ordinary kernel, so
    /// the product is exactly the one with the weight widened up front, while
    /// only the stored size stays held between products.
    /// </summary>
    private static Tensor2D ScaledProduct(Tensor2D x, ScaledMatrix w)
    {
        int length = checked(w.Rows * w.Cols);
        var scratch = _scaledScratch ??= new Dictionary<int, float[]>();
        if (!scratch.TryGetValue(length, out var values))
        {
            values = GC.AllocateUninitializedArray<float>(length);
            // Keep the decoded buffers of the few shapes a model has, up to
            // 256 MiB a thread; a larger one is used once and dropped.
            if (scratch.Values.Sum(v => (long)v.Length) + length <= (64L << 20))
            {
                scratch[length] = values;
            }
        }
        w.DequantRows(0, w.Rows, values);
        return MatMulTransposedBCpu(x, new[] { new Tensor2D(values, w.Rows, w.Cols) })[0];
    }

    /// <summary>The managed GEMV/GEMM over one or more weights (see
    /// <see cref="MatMulTransposedBMany"/>). A BF16-held weight streams its
    /// bits and widens in registers; an f32 one (or one whose Data has been
    /// touched) streams the floats. The two kernels are bit-identical.</summary>
    private static Tensor2D[] MatMulTransposedBCpu(Tensor2D x, Tensor2D[] weights)
    {
        int m = x.Rows, k = x.Cols;
        int count = weights.Length;
        var outputs = new float[count][];
        var bits = new ushort[]?[count];
        var data = new float[]?[count];
        long macs = 0;
        int totalRows = 0;
        for (int wi = 0; wi < count; wi++)
        {
            var w = weights[wi];
            // Every element is written by exactly one dot below.
            outputs[wi] = GC.AllocateUninitializedArray<float>(m * w.Rows);
            bits[wi] = w.Bf16Bits;
            data[wi] = bits[wi] is null ? w.Data : null;
            macs += (long)m * k * w.Rows;
            totalRows += w.Rows;
        }
        var xData = x.Data;

        if (macs < PoolFloorMacs)
        {
            for (int wi = 0; wi < count; wi++)
            {
                RowRange(xData, 0, m, k, weights[wi].Rows, bits[wi], data[wi], outputs[wi], 0, weights[wi].Rows, weights[wi].Rows);
            }
            return Wrap(outputs, weights, m);
        }

        // Output columns (weight rows) split into chunks, several per thread so
        // uneven progress balances out; a multiple of the kernel's four-row group.
        int threads = Math.Min(ComputeBudget.Cores, KernelPool.Parallelism);
        int per = ChunksPerThread;
        // A many-row batch is also split by activation rows (see below), so the
        // weight chunks only need to supply the rest of the parallelism — and
        // the wider each chunk, the more weight rows reuse an activation tile.
        int rowTile = m <= RowTile ? m : RowTile;
        int rowTiles = (m + rowTile - 1) / rowTile;
        int targetChunks = Math.Max(1, (threads * per + rowTiles - 1) / rowTiles);
        int chunkRows = Math.Max(4, (totalRows + targetChunks - 1) / targetChunks);
        chunkRows = (chunkRows + 3) & ~3;
        // chunkStart[wi] = index of weight wi's first chunk.
        var chunkStart = new int[count + 1];
        for (int wi = 0; wi < count; wi++)
        {
            chunkStart[wi + 1] = chunkStart[wi] + (weights[wi].Rows + chunkRows - 1) / chunkRows;
        }
        // Weight rows outer, in cache-sized blocks, activation rows inner: a
        // many-row GEMM (prefill, an encoder pass) then streams each weight
        // block from memory once instead of once per activation row. Every
        // output is still the same single Dot, so the result is bit-identical
        // to the untiled order.
        var blocks = new int[count];
        for (int wi = 0; wi < count; wi++)
        {
            int elementBytes = bits[wi] is null ? sizeof(float) : sizeof(ushort);
            blocks[wi] = Math.Max(4, (BlockBytes / (k * elementBytes)) & ~3);
        }
        // A many-row batch also splits its activation rows: each task is an
        // [rowTile × chunkRows] tile whose activation rows stay in L2 while the
        // weight rows stream past, instead of every task re-reading the whole
        // activation matrix. Which tile a thread computes does not change any
        // output value.
        int weightChunks = chunkStart[count];
        KernelPool.For(weightChunks * rowTiles, task =>
        {
            int chunk = task % weightChunks;
            int tile = task / weightChunks;
            int wi = 0;
            while (chunk >= chunkStart[wi + 1])
            {
                wi++;
            }
            int n = weights[wi].Rows;
            int j0 = (chunk - chunkStart[wi]) * chunkRows;
            int j1 = Math.Min(n, j0 + chunkRows);
            int i0 = tile * rowTile;
            int i1 = Math.Min(m, i0 + rowTile);
            RowRange(xData, i0, i1, k, n, bits[wi], data[wi], outputs[wi], j0, j1, blocks[wi]);
        });
        return Wrap(outputs, weights, m);
    }

    /// <summary>Weight rows [j0, j1) against every activation row, in blocks
    /// of <paramref name="block"/> weight rows.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    private static void RowRange(float[] xData, int i0, int i1, int k, int n, ushort[]? bits, float[]? data,
        float[] output, int j0, int j1, int block)
    {
        int vw = Vector<float>.Count;
        bool micro = i1 - i0 >= 4 && k >= 2 * vw && k % (bits is not null ? 2 * vw : vw) == 0;
        for (int jb = j0; jb < j1; jb += block)
        {
            int jEnd = Math.Min(j1, jb + block);
            int i = i0;
            if (micro)
            {
                for (; i + 4 <= i1; i += 4)
                {
                    Micro4x4(xData, i, k, bits, data, jb, jEnd, output, n);
                }
            }
            for (; i < i1; i++)
            {
                if (bits is not null)
                {
                    DotRowsBf16(xData.AsSpan(i * k, k), bits, k, jb, jEnd, output.AsSpan(i * n + jb, jEnd - jb));
                }
                else
                {
                    DotRows(xData.AsSpan(i * k, k), data!, k, jb, jEnd, output.AsSpan(i * n + jb, jEnd - jb));
                }
            }
        }
    }

    /// <summary>
    /// Four activation rows × four weight rows at once — sixteen accumulators,
    /// each weight run loaded (and for BF16 widened) once per four activation
    /// rows instead of once per row. Every accumulator is still one (i, j)
    /// dot accumulated in increasing k, multiply then add, folded with
    /// <see cref="Vector.Dot{T}"/>: bit-identical to <see cref="Dot"/>. Needs
    /// k to be a whole number of vector runs (2·Count for BF16), which every
    /// transformer width is; callers fall back to <see cref="DotRows"/>
    /// otherwise. Writes output[(i0+ii)·n + j] for weight rows [j0, j1).
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    private static void Micro4x4(float[] xData, int i0, int k, ushort[]? bits, float[]? data,
        int j0, int j1, float[] output, int n)
    {
        int vw = Vector<float>.Count;
        ref float x0 = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(xData);
        x0 = ref System.Runtime.CompilerServices.Unsafe.Add(ref x0, (nint)i0 * k);
        ref float x1 = ref System.Runtime.CompilerServices.Unsafe.Add(ref x0, k);
        ref float x2 = ref System.Runtime.CompilerServices.Unsafe.Add(ref x1, k);
        ref float x3 = ref System.Runtime.CompilerServices.Unsafe.Add(ref x2, k);
        int j = j0;
        for (; j + 4 <= j1; j += 4)
        {
            Vector<float> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector<float> a10 = default, a11 = default, a12 = default, a13 = default;
            Vector<float> a20 = default, a21 = default, a22 = default, a23 = default;
            Vector<float> a30 = default, a31 = default, a32 = default, a33 = default;
            if (bits is not null)
            {
                ref ushort wb = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(bits);
                ref ushort w0 = ref System.Runtime.CompilerServices.Unsafe.Add(ref wb, (nint)j * k);
                ref ushort w1 = ref System.Runtime.CompilerServices.Unsafe.Add(ref w0, k);
                ref ushort w2 = ref System.Runtime.CompilerServices.Unsafe.Add(ref w1, k);
                ref ushort w3 = ref System.Runtime.CompilerServices.Unsafe.Add(ref w2, k);
                for (int i = 0; i < k; i += 2 * vw)
                {
                    var x0l = Vector.LoadUnsafe(ref x0, (nuint)i);
                    var x1l = Vector.LoadUnsafe(ref x1, (nuint)i);
                    var x2l = Vector.LoadUnsafe(ref x2, (nuint)i);
                    var x3l = Vector.LoadUnsafe(ref x3, (nuint)i);
                    var x0h = Vector.LoadUnsafe(ref x0, (nuint)(i + vw));
                    var x1h = Vector.LoadUnsafe(ref x1, (nuint)(i + vw));
                    var x2h = Vector.LoadUnsafe(ref x2, (nuint)(i + vw));
                    var x3h = Vector.LoadUnsafe(ref x3, (nuint)(i + vw));
                    Widen(Vector.LoadUnsafe(ref w0, (nuint)i), out var lo, out var hi);
                    a00 += x0l * lo; a10 += x1l * lo; a20 += x2l * lo; a30 += x3l * lo;
                    a00 += x0h * hi; a10 += x1h * hi; a20 += x2h * hi; a30 += x3h * hi;
                    Widen(Vector.LoadUnsafe(ref w1, (nuint)i), out lo, out hi);
                    a01 += x0l * lo; a11 += x1l * lo; a21 += x2l * lo; a31 += x3l * lo;
                    a01 += x0h * hi; a11 += x1h * hi; a21 += x2h * hi; a31 += x3h * hi;
                    Widen(Vector.LoadUnsafe(ref w2, (nuint)i), out lo, out hi);
                    a02 += x0l * lo; a12 += x1l * lo; a22 += x2l * lo; a32 += x3l * lo;
                    a02 += x0h * hi; a12 += x1h * hi; a22 += x2h * hi; a32 += x3h * hi;
                    Widen(Vector.LoadUnsafe(ref w3, (nuint)i), out lo, out hi);
                    a03 += x0l * lo; a13 += x1l * lo; a23 += x2l * lo; a33 += x3l * lo;
                    a03 += x0h * hi; a13 += x1h * hi; a23 += x2h * hi; a33 += x3h * hi;
                }
            }
            else
            {
                ref float wb = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(data!);
                ref float w0 = ref System.Runtime.CompilerServices.Unsafe.Add(ref wb, (nint)j * k);
                ref float w1 = ref System.Runtime.CompilerServices.Unsafe.Add(ref w0, k);
                ref float w2 = ref System.Runtime.CompilerServices.Unsafe.Add(ref w1, k);
                ref float w3 = ref System.Runtime.CompilerServices.Unsafe.Add(ref w2, k);
                for (int i = 0; i < k; i += vw)
                {
                    var xv0 = Vector.LoadUnsafe(ref x0, (nuint)i);
                    var xv1 = Vector.LoadUnsafe(ref x1, (nuint)i);
                    var xv2 = Vector.LoadUnsafe(ref x2, (nuint)i);
                    var xv3 = Vector.LoadUnsafe(ref x3, (nuint)i);
                    var wv = Vector.LoadUnsafe(ref w0, (nuint)i);
                    a00 += xv0 * wv; a10 += xv1 * wv; a20 += xv2 * wv; a30 += xv3 * wv;
                    wv = Vector.LoadUnsafe(ref w1, (nuint)i);
                    a01 += xv0 * wv; a11 += xv1 * wv; a21 += xv2 * wv; a31 += xv3 * wv;
                    wv = Vector.LoadUnsafe(ref w2, (nuint)i);
                    a02 += xv0 * wv; a12 += xv1 * wv; a22 += xv2 * wv; a32 += xv3 * wv;
                    wv = Vector.LoadUnsafe(ref w3, (nuint)i);
                    a03 += xv0 * wv; a13 += xv1 * wv; a23 += xv2 * wv; a33 += xv3 * wv;
                }
            }
            var one = Vector<float>.One;
            int r0 = i0 * n + j, r1 = r0 + n, r2 = r1 + n, r3 = r2 + n;
            output[r0] = Vector.Dot(a00, one); output[r0 + 1] = Vector.Dot(a01, one); output[r0 + 2] = Vector.Dot(a02, one); output[r0 + 3] = Vector.Dot(a03, one);
            output[r1] = Vector.Dot(a10, one); output[r1 + 1] = Vector.Dot(a11, one); output[r1 + 2] = Vector.Dot(a12, one); output[r1 + 3] = Vector.Dot(a13, one);
            output[r2] = Vector.Dot(a20, one); output[r2 + 1] = Vector.Dot(a21, one); output[r2 + 2] = Vector.Dot(a22, one); output[r2 + 3] = Vector.Dot(a23, one);
            output[r3] = Vector.Dot(a30, one); output[r3 + 1] = Vector.Dot(a31, one); output[r3 + 2] = Vector.Dot(a32, one); output[r3 + 3] = Vector.Dot(a33, one);
        }
        if (j < j1)
        {
            for (int ii = 0; ii < 4; ii++)
            {
                var xRow = xData.AsSpan((i0 + ii) * k, k);
                var dst = output.AsSpan((i0 + ii) * n + j, j1 - j);
                if (bits is not null)
                {
                    DotRowsBf16(xRow, bits, k, j, j1, dst);
                }
                else
                {
                    DotRows(xRow, data!, k, j, j1, dst);
                }
            }
        }
    }

    private static Tensor2D[] Wrap(float[][] outputs, Tensor2D[] weights, int m)
    {
        var result = new Tensor2D[outputs.Length];
        for (int wi = 0; wi < outputs.Length; wi++)
        {
            result[wi] = new Tensor2D(outputs[wi], m, weights[wi].Rows);
        }
        return result;
    }

    /// <summary>
    /// <c>dst[j - j0] = Dot(x, w[j·k .. j·k+k))</c> for every weight row j in
    /// [j0, j1). Four rows run interleaved, each with its own accumulator, so
    /// the add latency is hidden and the activation is loaded once per four
    /// rows. Each row's arithmetic — one <see cref="Vector{T}"/> accumulator,
    /// multiply then add, <see cref="Vector.Dot{T}"/> fold, scalar tail — is
    /// exactly <see cref="Dot"/>'s, so the results are bit-identical to it.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    public static void DotRows(ReadOnlySpan<float> x, float[] w, int k, int j0, int j1, Span<float> dst)
    {
        int vw = Vector<float>.Count;
        int vecEnd = k >= vw ? k - k % vw : 0;
        ref float xRef = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x);
        ref float wBase = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(w);
        if ((long)j1 * k > w.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(j1));
        }
        int j = j0;
        for (; j + 4 <= j1; j += 4)
        {
            ref float r0 = ref System.Runtime.CompilerServices.Unsafe.Add(ref wBase, (nint)j * k);
            ref float r1 = ref System.Runtime.CompilerServices.Unsafe.Add(ref r0, k);
            ref float r2 = ref System.Runtime.CompilerServices.Unsafe.Add(ref r1, k);
            ref float r3 = ref System.Runtime.CompilerServices.Unsafe.Add(ref r2, k);
            var a0 = Vector<float>.Zero;
            var a1 = Vector<float>.Zero;
            var a2 = Vector<float>.Zero;
            var a3 = Vector<float>.Zero;
            for (int i = 0; i < vecEnd; i += vw)
            {
                var xv = Vector.LoadUnsafe(ref xRef, (nuint)i);
                a0 += xv * Vector.LoadUnsafe(ref r0, (nuint)i);
                a1 += xv * Vector.LoadUnsafe(ref r1, (nuint)i);
                a2 += xv * Vector.LoadUnsafe(ref r2, (nuint)i);
                a3 += xv * Vector.LoadUnsafe(ref r3, (nuint)i);
            }
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
            if (k >= vw)
            {
                s0 = Vector.Dot(a0, Vector<float>.One);
                s1 = Vector.Dot(a1, Vector<float>.One);
                s2 = Vector.Dot(a2, Vector<float>.One);
                s3 = Vector.Dot(a3, Vector<float>.One);
            }
            for (int i = vecEnd; i < k; i++)
            {
                float xs = System.Runtime.CompilerServices.Unsafe.Add(ref xRef, i);
                s0 += xs * System.Runtime.CompilerServices.Unsafe.Add(ref r0, i);
                s1 += xs * System.Runtime.CompilerServices.Unsafe.Add(ref r1, i);
                s2 += xs * System.Runtime.CompilerServices.Unsafe.Add(ref r2, i);
                s3 += xs * System.Runtime.CompilerServices.Unsafe.Add(ref r3, i);
            }
            dst[j - j0] = s0;
            dst[j - j0 + 1] = s1;
            dst[j - j0 + 2] = s2;
            dst[j - j0 + 3] = s3;
        }
        for (; j < j1; j++)
        {
            dst[j - j0] = Dot(x.Slice(0, k), w.AsSpan(j * k, k));
        }
    }

    /// <summary>
    /// <see cref="DotRows"/> over a BF16-held weight: each run of
    /// 2·<c>Vector&lt;float&gt;.Count</c> bit patterns is loaded once, widened to
    /// two f32 vectors (a zero-extend and a 16-bit shift, which is exact), and
    /// fed to the accumulator in element order. The accumulator therefore sees
    /// the same products in the same order as <see cref="Dot"/> over the
    /// widened row: bit-identical, from half the memory traffic.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    public static void DotRowsBf16(ReadOnlySpan<float> x, ushort[] w, int k, int j0, int j1, Span<float> dst)
    {
        int vw = Vector<float>.Count;
        int pair = 2 * vw;
        int vecEnd = k >= vw ? k - k % vw : 0;
        int pairEnd = vecEnd - vecEnd % pair;
        if ((long)j1 * k > w.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(j1));
        }
        ref float xRef = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x);
        ref ushort wBase = ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(w);
        Span<float> rowTail = stackalloc float[vw];
        int j = j0;
        for (; j + 4 <= j1; j += 4)
        {
            ref ushort r0 = ref System.Runtime.CompilerServices.Unsafe.Add(ref wBase, (nint)j * k);
            ref ushort r1 = ref System.Runtime.CompilerServices.Unsafe.Add(ref r0, k);
            ref ushort r2 = ref System.Runtime.CompilerServices.Unsafe.Add(ref r1, k);
            ref ushort r3 = ref System.Runtime.CompilerServices.Unsafe.Add(ref r2, k);
            var a0 = Vector<float>.Zero;
            var a1 = Vector<float>.Zero;
            var a2 = Vector<float>.Zero;
            var a3 = Vector<float>.Zero;
            for (int i = 0; i < pairEnd; i += pair)
            {
                var xLo = Vector.LoadUnsafe(ref xRef, (nuint)i);
                var xHi = Vector.LoadUnsafe(ref xRef, (nuint)(i + vw));
                Widen(Vector.LoadUnsafe(ref r0, (nuint)i), out var w0Lo, out var w0Hi);
                Widen(Vector.LoadUnsafe(ref r1, (nuint)i), out var w1Lo, out var w1Hi);
                Widen(Vector.LoadUnsafe(ref r2, (nuint)i), out var w2Lo, out var w2Hi);
                Widen(Vector.LoadUnsafe(ref r3, (nuint)i), out var w3Lo, out var w3Hi);
                a0 += xLo * w0Lo;
                a1 += xLo * w1Lo;
                a2 += xLo * w2Lo;
                a3 += xLo * w3Lo;
                a0 += xHi * w0Hi;
                a1 += xHi * w1Hi;
                a2 += xHi * w2Hi;
                a3 += xHi * w3Hi;
            }
            if (pairEnd < vecEnd)
            {
                // One single-width run left (k is an odd multiple of Count).
                var xv = Vector.LoadUnsafe(ref xRef, (nuint)pairEnd);
                a0 += xv * WidenRun(ref r0, pairEnd, rowTail);
                a1 += xv * WidenRun(ref r1, pairEnd, rowTail);
                a2 += xv * WidenRun(ref r2, pairEnd, rowTail);
                a3 += xv * WidenRun(ref r3, pairEnd, rowTail);
            }
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
            if (k >= vw)
            {
                s0 = Vector.Dot(a0, Vector<float>.One);
                s1 = Vector.Dot(a1, Vector<float>.One);
                s2 = Vector.Dot(a2, Vector<float>.One);
                s3 = Vector.Dot(a3, Vector<float>.One);
            }
            for (int i = vecEnd; i < k; i++)
            {
                float xs = System.Runtime.CompilerServices.Unsafe.Add(ref xRef, i);
                s0 += xs * Bf16(System.Runtime.CompilerServices.Unsafe.Add(ref r0, i));
                s1 += xs * Bf16(System.Runtime.CompilerServices.Unsafe.Add(ref r1, i));
                s2 += xs * Bf16(System.Runtime.CompilerServices.Unsafe.Add(ref r2, i));
                s3 += xs * Bf16(System.Runtime.CompilerServices.Unsafe.Add(ref r3, i));
            }
            dst[j - j0] = s0;
            dst[j - j0 + 1] = s1;
            dst[j - j0 + 2] = s2;
            dst[j - j0 + 3] = s3;
        }
        for (; j < j1; j++)
        {
            // The single-row form of the loop above, in the same order.
            ref ushort r = ref System.Runtime.CompilerServices.Unsafe.Add(ref wBase, (nint)j * k);
            var a = Vector<float>.Zero;
            for (int i = 0; i < pairEnd; i += pair)
            {
                Widen(Vector.LoadUnsafe(ref r, (nuint)i), out var lo, out var hi);
                a += Vector.LoadUnsafe(ref xRef, (nuint)i) * lo;
                a += Vector.LoadUnsafe(ref xRef, (nuint)(i + vw)) * hi;
            }
            if (pairEnd < vecEnd)
            {
                a += Vector.LoadUnsafe(ref xRef, (nuint)pairEnd) * WidenRun(ref r, pairEnd, rowTail);
            }
            float sum = k >= vw ? Vector.Dot(a, Vector<float>.One) : 0f;
            for (int i = vecEnd; i < k; i++)
            {
                sum += System.Runtime.CompilerServices.Unsafe.Add(ref xRef, i) * Bf16(System.Runtime.CompilerServices.Unsafe.Add(ref r, i));
            }
            dst[j - j0] = sum;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static void Widen(Vector<ushort> bits, out Vector<float> lo, out Vector<float> hi)
    {
        Vector.Widen(bits, out Vector<uint> l, out Vector<uint> h);
        lo = Vector.AsVectorSingle(Vector.ShiftLeft(l, 16));
        hi = Vector.AsVectorSingle(Vector.ShiftLeft(h, 16));
    }

    private static Vector<float> WidenRun(ref ushort row, int offset, Span<float> scratch)
    {
        for (int e = 0; e < scratch.Length; e++)
        {
            scratch[e] = Bf16(System.Runtime.CompilerServices.Unsafe.Add(ref row, offset + e));
        }
        return new Vector<float>(scratch);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float Bf16(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);

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
            table.CopyRowTo(indices[r], result.AsSpan(r * table.Cols, table.Cols));
        }
        return new Tensor2D(result, indices.Length, table.Cols);
    }
}