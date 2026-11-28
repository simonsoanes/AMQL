using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Resolves <see cref="OperandRef"/>s from the operand store once, widens
/// them to f32 (the executor's dtype widening, mirroring the reference's
/// <c>load_weight</c>) and caches the matrices for the session lifetime.
/// An optional weight patch merges its f32 deltas into every loaded
/// tensor, so every consumer of the loader sees the patched weights.
///
/// Three working-set modes (chosen by <c>AMQL_WEIGHTS</c> at construction,
/// so every runtime consumer — generate, route, path, moe-ify sampling,
/// MTP collection, prune corpus scoring — inherits it automatically):
///   <list type="bullet">
///     <item><see cref="WeightWorkingSet.ResidentF32"/> (default): every
///     tensor stays resident widened to f32 for the session. Byte-exact
///     and simplest; the working set doubles the container payload (a
///     47 GiB BF16 27B becomes ~94 GiB of f32).</item>
///     <item><see cref="WeightWorkingSet.OnDemandBf16"/>: stack
///     projections stay resident as their stored bytes (~2 B/element) and
///     are widened into a bounded f32 LRU on access — same numerics as
///     the resident path (BF16→f32 widening is lossless), ~2× the memory
///     saving. The right CPU mode when the f32 working set would thrash
///     but bit-exact output matters.</item>
///     <item><see cref="WeightWorkingSet.Mxfp4"/>: stack projections are
///     stored resident as MXFP4 packs (~0.53 B/element ≈ 13% of f32) and
///     dequantised into the bounded f32 LRU on access. The full 27B stack
///     drops from ~90 GiB of f32 to ~12.5 GiB of packs, and the packs are
///     the exact input Blackwell FP4 tensor cores consume (the CUDA
///     phase). Dequantisation is deterministic (the same codec export
///     uses), so results differ from the f32 path only by the codec's
///     quantisation error — within tolerance, never bit-exact, and small
///     models degrade more than large ones (FP4 is a large-model
///     technique).</item>
///     <item><see cref="WeightWorkingSet.Disk"/>: nothing is held for the
///     session. Every matrix (embedding and head included) is widened to
///     f32 from the container's memory map when it is used and kept only
///     while <see cref="ComputeBudget.DiskCacheBytes"/> has room; the rest
///     is read again on each pass. The resident path's exact numerics, on
///     the CPU, for a model whose f32 weights exceed RAM (a graft of a
///     large host) — paid for in reads, not memory.</item>
///   </list>
/// Embeddings, the output head, norms and other 1-D tensors stay
/// f32-resident in every mode but disk (precision-sensitive, small by
/// comparison); disk keeps only the 1-D tensors.
/// </summary>
public sealed class WeightLoader
{
    private readonly OperandStore _store;
    private readonly WeightPatch? _patch;
    private readonly WeightWorkingSet _workingSet;
    private readonly long _f32CacheBytes;
    private readonly Dictionary<(string ObjectId, string Tensor), Tensor2D> _matrices = new();
    private readonly Dictionary<(string ObjectId, string Tensor), float[]> _vectors = new();
    private readonly Dictionary<(string ObjectId, string Tensor), OperandResolution> _raw = new();
    private readonly Dictionary<(string ObjectId, string Tensor), Mxfp4.Quantized> _packs = new();
    private readonly LinkedList<(string ObjectId, string Tensor)> _lruOrder = new();
    private readonly Dictionary<(string ObjectId, string Tensor), LinkedListNode<(string ObjectId, string Tensor)>> _lruNodes = new();
    private long _lruBytes;

    /// <summary>Total f32 bytes this loader has widened from storage this
    /// session — the working-set driver for the resident f32 mode.</summary>
    private long _sessionF32Bytes;

    public WeightLoader(OperandStore store, WeightPatch? patch = null, WeightWorkingSet? workingSet = null)
    {
        _store = store;
        _patch = patch;
        _workingSet = workingSet ?? WeightWorkingSetExtensions.FromEnv();
        _f32CacheBytes = ComputeBudget.ResidentCacheBytes;
    }

    public WeightWorkingSet WorkingSet => _workingSet;

    /// <summary>The store the weights are read from (the device engine reads
    /// packed expert stacks from it as stored).</summary>
    internal OperandStore Store => _store;

    private RowOperands? _rows;

    /// <summary>The row-exact path's operands over the same store (resolved on
    /// first use and kept, whatever the working set): see <see cref="RowOperands"/>.</summary>
    public RowOperands Rows => _rows ??= new RowOperands(_store);

    /// <summary>Whether the store holds this tensor (a module the plan does
    /// not run, such as an MTP drafter, may be absent).</summary>
    public bool Contains(OperandRef operand)
    {
        try
        {
            return _store.ContainsTensor(operand.ObjectId, operand.TensorName);
        }
        catch (Amql.Vindex3.ContainerException)
        {
            return false;   // the graph has no such object
        }
    }

    /// <summary>True when at least one weight tensor has been uploaded to
    /// the device via the CUDA path — gates GPU-only ops like attention.</summary>
    public bool AnyDeviceWeight { get; private set; }

    /// <summary>When set, every tensor load reports its (objectId, tensorName),
    /// shape, and whether it was already cached (hit). The generate command's
    /// <c>--trace-tensors</c> flag wires it.</summary>
    public Action<string, string, long[], bool>? LoadTrace { get; set; }

    /// <summary>How many elements a 2-D stack tensor must have before it is
    /// worth compacting (norms and tiny projections stay f32).</summary>
    private const long CompressMinimumElements = 1L << 20; // 1M

    /// <summary>Charge the working set for one tensor, refusing when the
    /// process budget would be exceeded (resident f32 mode only — the
    /// compact modes are already bounded by the LRU cap).</summary>
    private void ChargeWidened(OperandRef operand, long elements)
    {
        long bytes = checked(elements * 4L);
        long projected = checked(_sessionF32Bytes + bytes);
        if (projected > ComputeBudget.MemoryBudgetBytes)
        {
            throw new ContainerException(
                $"refusing to widen '{operand.ObjectId}/{operand.TensorName}': the session would hold " +
                $"{FormatBytes(projected)} of f32 weights, over the {FormatBytes(ComputeBudget.MemoryBudgetBytes)} " +
                $"budget (set AMQL_MEMORY_GB for a larger job or use the compact working sets " +
                $"(AMQL_WEIGHTS=bf16|mxfp4))");
        }
        _sessionF32Bytes = projected;
    }

    /// <summary>Whether this operand belongs to the decoder stack and is a
    /// big enough 2-D tensor to compact. Embeddings, head, norms and every
    /// 1-D tensor stay f32-resident (precision-sensitive, small).</summary>
    private static bool Compactable(OperandRef operand, long[] shape)
    {
        if (!string.Equals(operand.ObjectId, "target.decoder_stack", StringComparison.Ordinal))
        {
            return false;
        }
        if (shape.Length != 2)
        {
            return false;
        }
        long elements = shape[0] * shape[1];
        return elements >= CompressMinimumElements;
    }

    /// <summary>Loads a 2-D weight as [rows, cols] (weight convention:
    /// rows are the output space). The resolved element count must match
    /// the requested shape exactly — a shape relabel over identical bytes
    /// is a defect, not a convenience.</summary>
    public Tensor2D Matrix(OperandRef operand, int rows, int cols)
    {
        if (_workingSet == WeightWorkingSet.Lean)
        {
            return MatrixLean(operand, rows, cols, expert: false);
        }
        if (_workingSet == WeightWorkingSet.Disk)
        {
            return MatrixDisk(operand, rows, cols);
        }
        var key = (operand.ObjectId, operand.TensorName);
        if (_matrices.TryGetValue(key, out var cached))
        {
            TouchLru(key);
            LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, new long[] { rows, cols }, true);
            return cached;
        }

        var shape = new[] { (long)rows, cols };
        Tensor2D matrix;
        if (_workingSet != WeightWorkingSet.ResidentF32 && Compactable(operand, shape) &&
            !_store.IsScaled(operand.ObjectId, operand.TensorName))
        {
            matrix = _workingSet == WeightWorkingSet.OnDemandBf16
                ? MatrixOnDemandBf16(operand, rows, cols)
                : MatrixMxfp4(operand, rows, cols);
            CacheBounded(key, matrix);
        }
        else if (HoldAsBf16(operand))
        {
            // Stored BF16 and nothing to add to it: keep the bits. The GEMV
            // kernels widen them in registers (bit-identical results, half the
            // bytes streamed per token); Data widens on first touch.
            ChargeWidened(operand, ((long)rows * cols + 1) / 2);
            var (bf16Shape, bits) = _store.ResolveBf16(operand.ObjectId, operand.TensorName);
            if (bf16Shape.Length == 0 || ElementCount(bf16Shape) != (long)rows * cols)
            {
                throw new ContainerException(
                    $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                    $"[{string.Join(",", bf16Shape)}] — expected [{rows} x {cols}]");
            }
            matrix = Tensor2D.FromBf16(bits, rows, cols);
            _matrices[key] = matrix;
        }
        else
        {
            // Charge before the widened array exists so an over-budget load
            // never allocates: the caller's shape IS the element count.
            ChargeWidened(operand, (long)rows * cols);

            var resolution = _store.ResolveWidened(operand);
            if (resolution.Shape.Length == 0 || ElementCount(resolution.Shape) != (long)rows * cols)
            {
                throw new ContainerException(
                    $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                    $"[{string.Join(",", resolution.Shape)}] — expected [{rows} x {cols}]");
            }

            var data = resolution.Values;
            ApplyPatch(operand, data);
            matrix = new Tensor2D(data, rows, cols);
            _matrices[key] = matrix;
        }
        LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, shape, false);
        return matrix;
    }

    /// <summary>A routed expert's matrix. The same as <see cref="Matrix"/>
    /// except in the lean working set, where experts are cached apart from the
    /// dense weights: least recently used first out, in whatever room the dense
    /// weights leave.</summary>
    public Tensor2D ExpertMatrix(OperandRef operand, int rows, int cols) =>
        _workingSet == WeightWorkingSet.Lean
            ? MatrixLean(operand, rows, cols, expert: true)
            : Matrix(operand, rows, cols);

    /// <summary>On-demand path: the tensor's stored bytes stay resident
    /// (BF16: 2 B/element, halving the f32 working set; also serves F16, I8
    /// and FP8 dtypes) and the f32 view comes from the LRU, widened from
    /// the stored payload on a miss. Widening is lossless for BF16/F16, so
    /// this is the resident path's numerics with a bounded working set.
    /// When the CUDA backend is available the widened f32 weights are also
    /// uploaded to the device as FP16 so GEMMs run on the GPU.</summary>
    private Tensor2D MatrixOnDemandBf16(OperandRef operand, int rows, int cols)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (!_raw.TryGetValue(key, out var stored))
        {
            stored = _store.Resolve(operand);
            if (stored.Shape.Length == 0 || ElementCount(stored.Shape) != (long)rows * cols)
            {
                throw new ContainerException(
                    $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                    $"[{string.Join(",", stored.Shape)}] — expected [{rows} x {cols}]");
            }
            _raw[key] = stored;
        }

        var widened = BitPattern.WidenToF32(stored.Dtype, stored.Payload);
        ApplyPatch(operand, widened);
        var matrix = new Tensor2D(widened, rows, cols);
        if (CudaShim.Enabled)
        {
            string deviceKey = $"{operand.ObjectId}/{operand.TensorName}";
            if (PatchedDeviceKey(operand) is { } patchedKey)
            {
                matrix.DeviceWeightF16 = CudaShim.UploadRawF16(patchedKey, BitPattern.F32ToFp16Bytes(widened), rows, cols);
            }
            else if (!CudaShim.TryGetDeviceWeight(operand.ObjectId, operand.TensorName, out _, out _))
            {
                var fp16 = BitPattern.F32ToFp16Bytes(widened);
                matrix.DeviceWeightF16 = CudaShim.UploadRawF16(deviceKey, fp16, rows, cols);
            }
            else
            {
                CudaShim.TryGetDeviceWeight(operand.ObjectId, operand.TensorName, out var ptr, out _);
                matrix.DeviceWeightF16 = ptr;
            }
            AnyDeviceWeight = matrix.DeviceWeightF16 != IntPtr.Zero;
        }
        return matrix;
    }

    /// <summary>MXFP4 path: the tensor's projection is resident as a
    /// packed block-scale set (~13% of the f32 size); the f32 view comes
    /// from the LRU (dequantised on a miss). Only what the current
    /// forward actually touches is ever dequantised, and the pack itself
    /// is the input Blackwell FP4 tensor cores consume (CUDA phase).</summary>
    private Tensor2D MatrixMxfp4(OperandRef operand, int rows, int cols)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (!_packs.TryGetValue(key, out var pack))
        {
            // Build the pack from a transient f32 widen (one tensor at a
            // time — freed after packing), so the pack is the only
            // resident copy of the projection.
            var resolution = _store.ResolveWidened(operand);
            if (resolution.Shape.Length == 0 || ElementCount(resolution.Shape) != (long)rows * cols)
            {
                throw new ContainerException(
                    $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                    $"[{string.Join(",", resolution.Shape)}] — expected [{rows} x {cols}]");
            }
            // The pack holds the base weight only: a patch delta is usually
            // far below FP4 resolution and would be lost in the codec, so it
            // is added at full precision to each dequantised view instead.
            _packs[key] = Mxfp4.Quantize(resolution.Values, rows, cols);
        }

        var dequant = Mxfp4.Dequant(_packs[key].Packed, _packs[key].BlockScales, rows, cols);
        ApplyPatch(operand, dequant);
        var matrix = new Tensor2D(dequant, rows, cols);
        // The device dequantises the shared base pack, which carries no
        // patch; a patched tensor uploads its patched view under its own key.
        matrix.DeviceWeightF16 = PatchedDeviceKey(operand) is { } patchedKey
            ? CudaShim.UploadRawF16(patchedKey, BitPattern.F32ToFp16Bytes(dequant), rows, cols)
            : CudaShim.UploadWeightF16(
                operand.ObjectId, operand.TensorName, _packs[key].Packed, _packs[key].BlockScales, rows, cols);
        AnyDeviceWeight = matrix.DeviceWeightF16 != IntPtr.Zero;
        return matrix;
    }

    private readonly object _diskLock = new();
    private readonly Dictionary<(string ObjectId, string Tensor), WeakReference<Tensor2D>> _streamed = new();

    /// <summary>The disk working set's entry point. Sibling sessions on
    /// other threads share this loader, so the cache and the operand store
    /// are only touched under a lock. A matrix that streamed through is
    /// remembered weakly: while any session still holds it, another gets
    /// the same array instead of reading it again.</summary>
    private Tensor2D MatrixDisk(OperandRef operand, int rows, int cols)
    {
        var key = (operand.ObjectId, operand.TensorName);
        lock (_diskLock)
        {
            bool hit = _matrices.TryGetValue(key, out var matrix) ||
                       (_streamed.TryGetValue(key, out var weak) && weak.TryGetTarget(out matrix));
            if (!hit)
            {
                matrix = MatrixFromDisk(operand, rows, cols);
                if (!CacheWhileRoom(key, matrix))
                {
                    _streamed[key] = new WeakReference<Tensor2D>(matrix);
                    ReclaimStreamed(matrix.Data.LongLength * 4L);
                }
            }
            LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, new long[] { rows, cols }, hit);
            return matrix!;
        }
    }

    private long _streamedSinceCollect;

    /// <summary>A matrix that streams through is garbage as soon as its layer
    /// has run, but each is a large-object allocation, and with RAM to spare
    /// the GC lets tens of GiB of them pile up before a full collection. The
    /// cache is what is meant to bound this working set, so once a quarter of
    /// it (between 256 MiB and 2 GiB) has streamed through, they are
    /// collected. Weight arrays hold no references, so the collection is
    /// cheap next to the reads that made them.</summary>
    private void ReclaimStreamed(long bytes)
    {
        _streamedSinceCollect += bytes;
        long threshold = _workingSet == WeightWorkingSet.Lean
            // The lean set has a RAM target to keep: collect well inside the
            // share of it that is held back for things other than the cache.
            ? Math.Clamp(ComputeBudget.LeanReserveBytes(ComputeBudget.LeanRamBytes) / 4, 64L << 20, 512L << 20)
            : Math.Clamp(DiskCacheBytes / 4, 256L << 20, 2L << 30);
        if (_streamedSinceCollect >= threshold)
        {
            _streamedSinceCollect = 0;
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        }
    }

    /// <summary>Disk path: widened from the container's memory map on every
    /// miss, nothing else retained — the stored bytes stay on disk (and in
    /// the OS page cache while there is room for them). Exact f32, CPU only:
    /// no device copy is made, so GPU memory is not a limit either.</summary>
    private Tensor2D MatrixFromDisk(OperandRef operand, int rows, int cols)
    {
        var resolution = _store.ResolveWidened(operand);
        if (resolution.Shape.Length == 0 || ElementCount(resolution.Shape) != (long)rows * cols)
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                $"[{string.Join(",", resolution.Shape)}] — expected [{rows} x {cols}]");
        }
        ApplyPatch(operand, resolution.Values);
        DiskReads++;
        return new Tensor2D(resolution.Values, rows, cols);
    }

    /// <summary>f32 bytes the disk working set keeps in memory; defaults to
    /// <see cref="ComputeBudget.DiskCacheBytes"/>. Zero streams every matrix.</summary>
    public long DiskCacheBytes { get; set; } = ComputeBudget.DiskCacheBytes;

    /// <summary>Matrices the disk working set has read from the container
    /// (cache misses), for reporting how much of a model streams.</summary>
    public long DiskReads { get; private set; }

    /// <summary>The disk working set's cache: a matrix stays while it fits,
    /// and once the cache is full new ones stream through without
    /// displacing it. A forward pass visits the layers in the same order
    /// every time, and under LRU a model larger than the cache would evict
    /// each matrix just before it is needed again; keeping the first ones
    /// resident makes the hit rate the share of the model that fits.</summary>
    private bool CacheWhileRoom((string ObjectId, string Tensor) key, Tensor2D matrix)
    {
        long bytes = matrix.Data.LongLength * 4L;
        if (_lruBytes + bytes > DiskCacheBytes)
        {
            return false;
        }
        _matrices[key] = matrix;
        _lruBytes += bytes;
        return true;
    }

    private void CacheBounded((string ObjectId, string Tensor) key, Tensor2D matrix)
    {
        long bytes = matrix.Data.LongLength * 4L;
        _matrices[key] = matrix;
        _lruBytes += bytes;
        _lruOrder.AddLast(key);
        _lruNodes[key] = _lruOrder.Last!;
        while (_lruBytes > _f32CacheBytes && _lruOrder.First is { } first)
        {
            if (first.Value == key)
            {
                break; // never evict the entry we just cached
            }
            if (_matrices.TryGetValue(first.Value, out var evicted))
            {
                _lruBytes -= evicted.Data.LongLength * 4L;
            }
            _matrices.Remove(first.Value);
            _lruOrder.RemoveFirst();
            _lruNodes.Remove(first.Value);
        }
    }

    private void TouchLru((string ObjectId, string Tensor) key)
    {
        if (!_lruNodes.TryGetValue(key, out var node))
        {
            return;
        }
        _lruOrder.Remove(node);
        _lruOrder.AddLast(node);
    }

    // ── Lean working set ────────────────────────────────────────────────────

    private readonly LinkedList<(string ObjectId, string Tensor)> _expertOrder = new();
    private readonly Dictionary<(string ObjectId, string Tensor), LinkedListNode<(string ObjectId, string Tensor)>> _expertNodes = new();
    private long _leanDenseBytes;
    private long _leanExpertBytes;
    private long _leanDeviceBytes;
    private long _leanBytesRead;
    private long? _leanDeviceBudget;
    private long? _resolvedDeviceBudget;

    /// <summary>Host bytes the lean working set may spend on cached weights:
    /// <see cref="ComputeBudget.LeanCacheBytes"/> at construction. Zero
    /// streams every matrix.</summary>
    public long LeanCacheBytes { get; set; } = ComputeBudget.LeanCacheBytes(CudaShim.Enabled);

    /// <summary>Device bytes the lean working set may fill with FP16 copies of
    /// dense matrices. Null (the default) means the device's free memory at
    /// first use, less a 1 GiB margin; zero keeps every weight on the host.
    /// Set it before the first matrix loads.</summary>
    public long? LeanDeviceBytes
    {
        get => _leanDeviceBudget ?? ComputeBudget.GpuMemoryBytes;
        set
        {
            _leanDeviceBudget = value;
            _resolvedDeviceBudget = null;
        }
    }

    private long DeviceBudget()
    {
        if (_resolvedDeviceBudget is { } known)
        {
            return known;
        }
        long budget = 0;
        if (CudaShim.Enabled)
        {
            if (LeanDeviceBytes is { } set)
            {
                budget = set;
            }
            else if (CudaShim.TryMemoryInfo(out long free, out _))
            {
                // Leave room for the GEMMs' activations, cuBLAS workspace and
                // anything else sharing the device.
                budget = Math.Max(0, free - (1L << 30));
            }
        }
        _resolvedDeviceBudget = budget;
        return budget;
    }

    /// <summary>What the lean working set holds and has read, for reporting.</summary>
    public LeanStats Lean
    {
        get
        {
            lock (_diskLock)
            {
                return new LeanStats(_leanDenseBytes, _leanExpertBytes, _expertNodes.Count,
                    _leanDeviceBytes, DiskReads, _leanBytesRead, Tensor2D.HostReloads, LeanCacheBytes, DeviceBudget(),
                    _deviceExpertNodes.Count, Tensor2D.ScaledWidenings);
            }
        }
    }

    /// <summary>
    /// The lean working set's entry point. A matrix comes from the cache, or
    /// is read from the container as its stored bits (a BF16 tensor is never
    /// widened: the kernels widen in registers). A dense matrix big enough for
    /// the device GEMM goes to the GPU while its budget has room, and then only
    /// the device copy is kept. Otherwise a dense matrix is cached while the
    /// host budget has room (displacing experts if it must: every pass needs
    /// every dense weight, but only some experts) and an expert joins the
    /// expert LRU. Whatever does not fit streams through and is read again
    /// next time. Locked like the disk set, so sibling sessions can share it.
    /// </summary>
    private long _heapFloor;

    // Streamed experts' BF16 buffers, recycled: a decode step streams every
    // picked expert that is not cached, and as fresh large-object arrays they
    // would be garbage the GC has to be made to collect, step after step.
    private readonly HashSet<Tensor2D> _recyclable = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<(long Elements, bool Bf16), Stack<Array>> _pool = new();
    private long _pooledBytes;

    private T[]? Rent<T>(long elements)
    {
        bool bf16 = typeof(T) == typeof(ushort);
        if (_pool.TryGetValue((elements, bf16), out var stack) && stack.TryPop(out var array))
        {
            _pooledBytes -= elements * (bf16 ? 2 : 4);
            return (T[])array;
        }
        return null;
    }

    /// <summary>
    /// Hands back a matrix from <see cref="ExpertMatrix"/> once it has been
    /// used. A streamed expert's buffer is then reused for the next one read;
    /// a cached matrix, or one from another working set, is left alone. The
    /// caller must not touch the matrix afterwards.
    /// </summary>
    public void Release(Tensor2D matrix)
    {
        if (_workingSet != WeightWorkingSet.Lean)
        {
            return;
        }
        lock (_diskLock)
        {
            if (!_recyclable.Remove(matrix) || matrix.Scaled is not null)
            {
                return; // a scaled matrix's arrays are its own
            }
            Array array = (Array?)matrix.Bf16Bits ?? matrix.Data;
            bool bf16 = array is ushort[];
            long bytes = array.LongLength * (bf16 ? 2 : 4);
            // Room for a few layers' worth of experts, within the reserve.
            long cap = ComputeBudget.LeanReserveBytes(ComputeBudget.LeanRamBytes) / 4;
            if (_pooledBytes + bytes > cap)
            {
                return;
            }
            if (!_pool.TryGetValue((array.LongLength, bf16), out var stack))
            {
                _pool[(array.LongLength, bf16)] = stack = new Stack<Array>();
            }
            stack.Push(array);
            _pooledBytes += bytes;
        }
    }

    /// <summary>
    /// Keeps the managed heap near the RAM target. Streamed matrices and the
    /// activations made from them are garbage as soon as a layer has run, but
    /// most are large-object allocations, which only a full collection frees,
    /// and with RAM to spare the GC lets gigabytes of them build up first. So
    /// once the heap passes the target less a margin, it is collected. If what
    /// is live is itself over that line, the next collection waits until the
    /// heap has grown another 256 MiB, rather than collecting on every load.
    /// </summary>
    private void KeepHeapNearTarget()
    {
        long ram = ComputeBudget.LeanRamBytes;
        long line = Math.Max(_heapFloor, ram - Math.Min(512L << 20, ram / 8));
        if (GC.GetTotalMemory(false) <= line)
        {
            return;
        }
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        _streamedSinceCollect = 0;
        _heapFloor = GC.GetTotalMemory(false) + (256L << 20);
    }

    private Tensor2D MatrixLean(OperandRef operand, int rows, int cols, bool expert)
    {
        var key = (operand.ObjectId, operand.TensorName);
        lock (_diskLock)
        {
            KeepHeapNearTarget();
            if (_matrices.TryGetValue(key, out var cached) ||
                (_streamed.TryGetValue(key, out var weak) && weak.TryGetTarget(out cached)))
            {
                if (_expertNodes.TryGetValue(key, out var node))
                {
                    _expertOrder.Remove(node);
                    _expertOrder.AddLast(node);
                }
                if (_deviceExpertNodes.TryGetValue(key, out var deviceNode))
                {
                    _deviceExpertOrder.Remove(deviceNode);
                    _deviceExpertOrder.AddLast(deviceNode);
                    _deviceExpertGeneration[key] = _expertGeneration;
                }
                LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, new long[] { rows, cols }, true);
                return cached!;
            }

            Tensor2D matrix;
            if (TryLeanDevice(operand, rows, cols, expert, out var onDevice))
            {
                matrix = onDevice;
                _matrices[key] = matrix;
            }
            else
            {
                matrix = LeanFromStore(operand, rows, cols, expert);
                long bytes = HostBytes(matrix);
                bool kept = expert ? CacheExpert(key, matrix, bytes) : CacheDense(key, matrix, bytes);
                if (!kept && expert && matrix.Scaled is null)
                {
                    // Handed back through Release once its expert has run.
                    _recyclable.Add(matrix);
                }
                else if (!kept)
                {
                    _streamed[key] = new WeakReference<Tensor2D>(matrix);
                    ReclaimStreamed(bytes);
                }
            }
            LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, new long[] { rows, cols }, false);
            return matrix;
        }
    }

    /// <summary>A matrix as the container stores it: BF16 bits when the
    /// tensor is BF16 and unpatched, otherwise widened to f32 with any patch
    /// and added delta applied.</summary>
    private Tensor2D LeanFromStore(OperandRef operand, int rows, int cols, bool recycled = false)
    {
        DiskReads++;
        long count = (long)rows * cols;
        if (!Patched(operand) && _store.ResolveScaled(operand) is { } scaled)
        {
            // A quantised checkpoint's weight: held at its stored size.
            if (scaled.Rows != rows || scaled.Cols != cols)
            {
                throw new ContainerException(
                    $"operand '{operand.ObjectId}/{operand.TensorName}' is [{scaled.Rows} x {scaled.Cols}] — expected [{rows} x {cols}]");
            }
            _leanBytesRead += scaled.Elements.LongLength;
            return Tensor2D.FromScaled(scaled);
        }
        if (HoldAsBf16Always(operand))
        {
            var (shape, bits) = _store.ResolveBf16(operand.ObjectId, operand.TensorName,
                recycled ? Rent<ushort>(count) : null);
            if (shape.Length == 0 || ElementCount(shape) != (long)rows * cols)
            {
                throw new ContainerException(
                    $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                    $"[{string.Join(",", shape)}] — expected [{rows} x {cols}]");
            }
            _leanBytesRead += bits.LongLength * 2;
            return Tensor2D.FromBf16(bits, rows, cols);
        }
        var resolution = _store.ResolveWidened(operand.ObjectId, operand.TensorName,
            recycled ? Rent<float>(count) : null);
        if (resolution.Shape.Length == 0 || ElementCount(resolution.Shape) != (long)rows * cols)
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                $"[{string.Join(",", resolution.Shape)}] — expected [{rows} x {cols}]");
        }
        ApplyPatch(operand, resolution.Values);
        _leanBytesRead += resolution.Values.LongLength * 4;
        return new Tensor2D(resolution.Values, rows, cols);
    }

    private bool Patched(OperandRef operand) =>
        _added.ContainsKey((operand.ObjectId, operand.TensorName)) ||
        (_patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _));

    /// <summary>Smallest matrix worth putting on the GPU as NVFP4: below it the
    /// launch and the activation's round trip cost more than the host product.</summary>
    public long LeanNvfp4MinimumElements { get; set; } = 1L << 18;

    // Experts on the GPU, least recently used first: the device budget's
    // share that dense weights leave, refilled as routing moves on.
    private readonly LinkedList<(string ObjectId, string Tensor)> _deviceExpertOrder = new();
    private readonly Dictionary<(string ObjectId, string Tensor), LinkedListNode<(string ObjectId, string Tensor)>> _deviceExpertNodes = new();
    private long _leanDeviceExpertBytes;

    /// <summary>
    /// Puts a matrix on the GPU when the device budget has room, keeping no
    /// host copy. A weight the checkpoint stores as NVFP4 goes as its packs
    /// (0.5625 bytes an element), routed experts included, which then share
    /// the device in least-recently-used order. With
    /// <see cref="ComputeBudget.GpuWeightFormat"/> NVFP4, other weights are
    /// quantised to NVFP4 on the way (lossy). Otherwise a dense matrix big
    /// enough for the device GEMM goes as FP16, decoded exactly from what is
    /// stored. Dense weights may displace experts, never the other way round.
    /// </summary>
    private bool TryLeanDevice(OperandRef operand, int rows, int cols, bool expert, out Tensor2D matrix)
    {
        matrix = null!;
        if (!CudaShim.Enabled || DeviceBudget() <= 0)
        {
            return false;
        }
        long elements = (long)rows * cols;
        var stored = _store.DtypeOf(operand.ObjectId, operand.TensorName);
        bool patched = Patched(operand);
        // A checkpoint's MXFP8 goes as stored (exact, 1.03 bytes an element);
        // its NVFP4 too; anything else as NVFP4 only when asked for.
        bool mxfp8 = stored == Dtype.F8_E4M3 && cols % Mxfp8.BlockElements == 0 &&
                     elements >= LeanNvfp4MinimumElements && !patched;
        bool nvfp4 = !mxfp8 && cols % Nvfp4.BlockElements == 0 && elements >= LeanNvfp4MinimumElements && !patched &&
                     (stored == Dtype.FP4 || ComputeBudget.GpuWeightFormat == GpuWeightFormat.Nvfp4);
        bool fp16 = !mxfp8 && !nvfp4 && !expert && elements >= TensorOps.CudaMinimumMacs;
        if (!mxfp8 && !nvfp4 && !fp16)
        {
            return false;
        }
        long bytes = mxfp8 ? CudaShim.Mxfp8DeviceBytes(rows, cols)
            : nvfp4 ? CudaShim.Nvfp4DeviceBytes(rows, cols)
            : elements * 2;
        if (!MakeDeviceRoom(bytes, expert))
        {
            return false;
        }
        var host = LeanFromStore(operand, rows, cols);
        Func<Tensor2D> reload = () =>
        {
            lock (_diskLock)
            {
                return LeanFromStore(operand, rows, cols);
            }
        };
        if (mxfp8)
        {
            if (host.Scaled is not { IsMxfp8: true } stored8)
            {
                return false; // FP8 with other scales (per tensor, 128x128 blocks): host
            }
            var device = CudaShim.UploadMxfp8(stored8);
            if (device == IntPtr.Zero)
            {
                return false;
            }
            matrix = Tensor2D.OnDeviceMxfp8(rows, cols, device, reload);
        }
        else if (nvfp4)
        {
            var packs = host.Scaled is { IsNvfp4: true } exact
                ? exact
                : ComputeBudget.GpuWeightFormat == GpuWeightFormat.Nvfp4
                    ? Nvfp4.Quantize(host.Scaled?.Dequant() ?? host.Data, rows, cols)
                    : null;
            var device = packs is null ? IntPtr.Zero : CudaShim.UploadNvfp4(packs);
            if (device == IntPtr.Zero)
            {
                return false;
            }
            matrix = Tensor2D.OnDeviceNvfp4(rows, cols, device, packs!.Global, reload);
        }
        else
        {
            string deviceKey = PatchedDeviceKey(operand) ?? $"{operand.ObjectId}/{operand.TensorName}";
            if (_added.ContainsKey((operand.ObjectId, operand.TensorName)))
            {
                deviceKey += $"#added{Version}";
            }
            var device = CudaShim.UploadRawF16(deviceKey,
                host.Scaled is { } scaled ? BitPattern.F32ToFp16Bytes(scaled.Dequant()) : Fp16Of(host), rows, cols);
            if (device == IntPtr.Zero)
            {
                return false;
            }
            matrix = Tensor2D.OnDevice(rows, cols, device, reload);
        }
        _leanDeviceBytes += bytes;
        AnyDeviceWeight = true;
        if (expert)
        {
            var key = (operand.ObjectId, operand.TensorName);
            _deviceExpertNodes[key] = _deviceExpertOrder.AddLast(key);
            _deviceExpertGeneration[key] = _expertGeneration;
            _leanDeviceExpertBytes += bytes;
        }
        return true;
    }

    /// <summary>Frees device experts, oldest first, until <paramref name="bytes"/>
    /// more fit the budget. An expert can only displace other experts.</summary>
    /// <summary>Frees a matrix's NVFP4 or MXFP8 device copy, leaving anyone who
    /// still holds the matrix to fall back to the host; the bytes it took.</summary>
    private static long FreeScaledDevice(Tensor2D matrix)
    {
        if (matrix.DeviceNvfp4 is var fp4 && fp4 != IntPtr.Zero)
        {
            matrix.DeviceNvfp4 = IntPtr.Zero;
            CudaShim.FreeNvfp4(fp4, matrix.Rows, matrix.Cols);
            return CudaShim.Nvfp4DeviceBytes(matrix.Rows, matrix.Cols);
        }
        if (matrix.DeviceMxfp8 is var fp8 && fp8 != IntPtr.Zero)
        {
            matrix.DeviceMxfp8 = IntPtr.Zero;
            CudaShim.FreeMxfp8(fp8, matrix.Rows, matrix.Cols);
            return CudaShim.Mxfp8DeviceBytes(matrix.Rows, matrix.Cols);
        }
        return 0;
    }

    private bool MakeDeviceRoom(long bytes, bool expert)
    {
        long budget = DeviceBudget();
        if (bytes > budget || (!expert && _leanDeviceBytes - _leanDeviceExpertBytes + bytes > budget))
        {
            return false;
        }
        // Experts fetched for the layer running now are waiting for its fused
        // device call: they are skipped, never evicted from under it.
        var candidate = _deviceExpertOrder.First;
        while (_leanDeviceBytes + bytes > budget && candidate is not null)
        {
            var next = candidate.Next;
            if (_deviceExpertGeneration.GetValueOrDefault(candidate.Value) != _expertGeneration)
            {
                EvictDeviceExpert(candidate.Value);
            }
            candidate = next;
        }
        return _leanDeviceBytes + bytes <= budget;
    }

    private long _expertGeneration;
    private readonly Dictionary<(string ObjectId, string Tensor), long> _deviceExpertGeneration = new();

    /// <summary>Starts a routed layer: device experts it fetches stay on the
    /// device until the next layer starts, so its fused call can use them all.</summary>
    public void BeginExpertLayer()
    {
        lock (_diskLock)
        {
            _expertGeneration++;
        }
    }

    private void EvictDeviceExpert((string ObjectId, string Tensor) key)
    {
        _deviceExpertGeneration.Remove(key);
        if (_deviceExpertNodes.Remove(key, out var node))
        {
            _deviceExpertOrder.Remove(node);
        }
        if (_matrices.Remove(key, out var evicted))
        {
            long bytes = FreeScaledDevice(evicted);
            _leanDeviceBytes -= bytes;
            _leanDeviceExpertBytes -= bytes;
        }
    }

    /// <summary>FP16 bytes of a host matrix, converted a slice at a time so a
    /// BF16 matrix is never widened whole on its way to the GPU.</summary>
    private static byte[] Fp16Of(Tensor2D host)
    {
        const int Slice = 1 << 20;
        long elements = (long)host.Rows * host.Cols;
        var fp16 = GC.AllocateUninitializedArray<byte>(checked((int)(elements * 2)));
        var bits = host.Bf16Bits;
        var widened = bits is null ? null : new float[Math.Min(Slice, elements)];
        for (long first = 0; first < elements; first += Slice)
        {
            int n = (int)Math.Min(Slice, elements - first);
            ReadOnlySpan<float> values;
            if (bits is not null)
            {
                BitPattern.WidenInto(Dtype.BF16,
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(bits.AsSpan((int)first, n)), widened.AsSpan(0, n));
                values = widened.AsSpan(0, n);
            }
            else
            {
                values = host.Data.AsSpan((int)first, n);
            }
            BitPattern.F32ToFp16Bytes(values).CopyTo(fp16.AsSpan((int)(first * 2), n * 2));
        }
        return fp16;
    }

    private static long HostBytes(Tensor2D matrix) =>
        matrix.Scaled is { } scaled ? scaled.Bytes
        : matrix.Bf16Bits is { } bits ? bits.LongLength * 2
        : matrix.Data.LongLength * 4;

    private bool CacheDense((string ObjectId, string Tensor) key, Tensor2D matrix, long bytes)
    {
        if (_leanDenseBytes + bytes > LeanCacheBytes)
        {
            return false;
        }
        while (_leanDenseBytes + _leanExpertBytes + bytes > LeanCacheBytes && _expertOrder.First is { } oldest)
        {
            EvictExpert(oldest.Value);
        }
        _matrices[key] = matrix;
        _leanDenseBytes += bytes;
        return true;
    }

    private bool CacheExpert((string ObjectId, string Tensor) key, Tensor2D matrix, long bytes)
    {
        if (_leanDenseBytes + bytes > LeanCacheBytes)
        {
            return false;
        }
        while (_leanDenseBytes + _leanExpertBytes + bytes > LeanCacheBytes && _expertOrder.First is { } oldest)
        {
            EvictExpert(oldest.Value);
        }
        _matrices[key] = matrix;
        _leanExpertBytes += bytes;
        _expertNodes[key] = _expertOrder.AddLast(key);
        return true;
    }

    private void EvictExpert((string ObjectId, string Tensor) key)
    {
        if (_matrices.Remove(key, out var evicted))
        {
            _leanExpertBytes -= HostBytes(evicted);
        }
        if (_expertNodes.Remove(key, out var node))
        {
            _expertOrder.Remove(node);
        }
    }

    /// <summary>f32 bytes of a matrix block <see cref="StreamedProduct"/> holds at once.</summary>
    public long StreamBlockBytes { get; set; } = 64L << 20;

    /// <summary>Products <see cref="StreamedProduct"/> has computed a block at a time.</summary>
    public long BlockStreamedProducts { get; private set; }

    /// <summary>
    /// x times W-transposed for a matrix the lean working set cannot hold: the
    /// matrix is read a block of rows at a time and each block's columns of the
    /// product computed from it, so no more than a block is ever in memory.
    /// Null when the matrix is held (cached or on the GPU), would be held if
    /// loaded, is patched, or the working set is not lean: the caller then
    /// multiplies by <see cref="Matrix"/>. Each output is the same dot over the
    /// same values, so the product is the whole-matrix one.
    /// </summary>
    public Tensor2D? StreamedProduct(Tensor2D x, OperandRef operand, int rows, int cols)
    {
        if (_workingSet != WeightWorkingSet.Lean || x.Cols != cols ||
            (_patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _)) ||
            _added.ContainsKey((operand.ObjectId, operand.TensorName)))
        {
            return null;
        }
        long blockRows = Math.Max(1, StreamBlockBytes / ((long)cols * 4));
        lock (_diskLock)
        {
            var key = (operand.ObjectId, operand.TensorName);
            long elements = (long)rows * cols;
            long hostBytes = elements * (_store.DtypeOf(operand.ObjectId, operand.TensorName) == Dtype.BF16 ? 2 : 4);
            bool held = _matrices.ContainsKey(key) ||
                        (_streamed.TryGetValue(key, out var weak) && weak.TryGetTarget(out _));
            bool wouldHold = _leanDenseBytes + hostBytes <= LeanCacheBytes ||
                             (CudaShim.Enabled && elements >= TensorOps.CudaMinimumMacs &&
                              _leanDeviceBytes + elements * 2 <= DeviceBudget());
            if (held || wouldHold || rows <= blockRows)
            {
                return null;
            }
            DiskReads++;
            BlockStreamedProducts++;
            _leanBytesRead += hostBytes;
            LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, new long[] { rows, cols }, false);
        }

        // A BF16 matrix streams as its bits, which the kernel widens in
        // registers: half the bytes read, and the same dots.
        bool bf16 = _store.DtypeOf(operand.ObjectId, operand.TensorName) == Dtype.BF16;
        var result = new float[(long)x.Rows * rows];
        var bitsBlock = bf16 ? new ushort[blockRows * cols] : null;
        var floatBlock = bf16 ? null : new float[blockRows * cols];
        for (long first = 0; first < rows; first += blockRows)
        {
            int n = (int)Math.Min(blockRows, rows - first);
            Tensor2D w;
            lock (_diskLock)
            {
                if (bitsBlock is not null)
                {
                    var target = n == blockRows ? bitsBlock : new ushort[n * cols];
                    _store.ReadRowRange(operand, first, n, target);
                    w = Tensor2D.FromBf16(target, n, cols);
                }
                else
                {
                    var target = n == blockRows ? floatBlock! : new float[n * cols];
                    _store.ReadRowRange(operand, first, n, target);
                    w = new Tensor2D(target, n, cols);
                }
            }
            var part = TensorOps.MatMulTransposedB(x, w);
            for (int r = 0; r < x.Rows; r++)
            {
                part.Data.AsSpan(r * n, n).CopyTo(result.AsSpan((int)(r * (long)rows + first), n));
            }
        }
        return new Tensor2D(result, x.Rows, rows);
    }

    /// <summary>
    /// Gathers rows of an embedding table without holding the table: in the
    /// lean working set each token's row is read straight from the container.
    /// Null in the other working sets, when the table is patched (a row read
    /// would miss the patch), or when the table is already held (it is also
    /// the output head) — the caller then gathers from <see cref="Matrix"/>.
    /// </summary>
    public Tensor2D? GatherRows(OperandRef table, int vocab, int hidden, ReadOnlySpan<int> tokens)
    {
        if (_workingSet != WeightWorkingSet.Lean ||
            (_patch is not null && _patch.TryGet(table.ObjectId, table.TensorName, out _)) ||
            _added.ContainsKey((table.ObjectId, table.TensorName)))
        {
            return null;
        }
        lock (_diskLock)
        {
            // A table held on the host (also the output head) is gathered from;
            // one held only on the GPU is not read back for a few rows.
            if (_matrices.TryGetValue((table.ObjectId, table.TensorName), out var held) && !held.OnGpu)
            {
                return null;
            }
            var ids = new long[tokens.Length];
            for (int i = 0; i < tokens.Length; i++)
            {
                if ((uint)tokens[i] >= (uint)vocab)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokens), $"token {tokens[i]} is outside the {vocab}-row embedding");
                }
                ids[i] = tokens[i];
            }
            var values = new float[(long)tokens.Length * hidden];
            _store.ReadRowsWidened(table, ids, values);
            _leanBytesRead += (long)tokens.Length * hidden * 2;
            LoadTrace?.Invoke(table.ObjectId, table.TensorName, new long[] { tokens.Length, hidden }, false);
            return new Tensor2D(values, tokens.Length, hidden);
        }
    }

    /// <summary>Like <see cref="HoldAsBf16"/>, but not switched off by
    /// <c>AMQL_RESIDENT_BF16</c>: the lean set never widens what it need not.</summary>
    private bool HoldAsBf16Always(OperandRef operand)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_added.ContainsKey(key) || (_patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _)))
        {
            return false;
        }
        return _store.DtypeOf(operand.ObjectId, operand.TensorName) == Dtype.BF16;
    }

    /// <summary>Reads chosen rows of a table straight from storage, widened to
    /// f32 — nothing cached, nothing charged to the working set. For tables too
    /// large to load (Qwen4's n-gram embedding). A patch on the table refuses:
    /// a row read would silently miss it.</summary>
    public void ReadRows(OperandRef operand, ReadOnlySpan<long> rows, Span<float> target)
    {
        if ((_patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _)) ||
            _added.ContainsKey((operand.ObjectId, operand.TensorName)))
        {
            throw new UnsupportedOperatorException(
                $"'{operand.ObjectId}/{operand.TensorName}' is read a row at a time, and a weight patch on it would be missed");
        }
        _store.ReadRowsWidened(operand, rows, target);
    }

    /// <summary>Loads a 1-D weight (norm scales).</summary>
    public float[] Vector(OperandRef operand, int width)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_vectors.TryGetValue(key, out var cached))
        {
            LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, new long[] { width }, true);
            return cached;
        }

        ChargeWidened(operand, width);

        var resolution = _store.ResolveWidened(operand);
        if (ElementCount(resolution.Shape) != width)
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to " +
                $"{ElementCount(resolution.Shape)} elements — expected {width}");
        }

        var vector = resolution.Values;
        ApplyPatch(operand, vector);
        _vectors[key] = vector;
        LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, resolution.Shape, false);
        return vector;
    }

    public int LoadedMatrixCount => _matrices.Count;
    public int LoadedVectorCount => _vectors.Count;

    /// <summary>Deltas added in memory this session (<see cref="AddDelta"/>),
    /// on top of the construction-time patch.</summary>
    private readonly Dictionary<(string ObjectId, string Tensor), float[]> _added = new();

    /// <summary>
    /// Adds <paramref name="delta"/> to a matrix in memory — the live
    /// session's weight changes at once, without rebuilding the runtime or
    /// widening the tensor again. The resident f32 view is updated in place
    /// and, when the tensor has a device FP16 copy, that copy is rewritten
    /// from the updated values, so GPU GEMMs see the change too. The delta
    /// is also remembered, so a compact working set that evicts and
    /// re-widens the tensor still includes it. Adding the negated delta
    /// undoes it.
    /// </summary>
    public void AddDelta(OperandRef operand, int rows, int cols, float[] delta)
    {
        Version++;
        if (delta.Length != (long)rows * cols)
        {
            throw new ArgumentException(
                $"delta for '{operand.ObjectId}/{operand.TensorName}' holds {delta.Length} values — expected {rows}x{cols}");
        }
        // Load first, so the cached view does not already carry the delta
        // when it is added below.
        var matrix = Matrix(operand, rows, cols);
        var key = (operand.ObjectId, operand.TensorName);
        if (!_added.TryGetValue(key, out var total))
        {
            total = new float[delta.Length];
            _added[key] = total;
        }
        var data = matrix.Data;
        for (int i = 0; i < delta.Length; i++)
        {
            data[i] += delta[i];
            total[i] += delta[i];
        }
        if (matrix.DeviceWeightF16 != IntPtr.Zero)
        {
            var fp16 = BitPattern.F32ToFp16Bytes(data);
            if (CudaShim.CopyHostToDevice(matrix.DeviceWeightF16, fp16, fp16.Length) != 0)
            {
                // A device copy that no longer matches the host weight would
                // silently compute with the old values; drop to managed GEMMs.
                matrix.DeviceWeightF16 = IntPtr.Zero;
            }
        }
        if (matrix.DeviceNvfp4 != IntPtr.Zero || matrix.DeviceMxfp8 != IntPtr.Zero)
        {
            // Four or eight bits cannot carry a small delta: the weight leaves
            // the GPU and runs from its updated host values.
            lock (_diskLock)
            {
                long bytes = FreeScaledDevice(matrix);
                _leanDeviceBytes -= bytes;
                if (_deviceExpertNodes.Remove(key, out var node))
                {
                    _deviceExpertOrder.Remove(node);
                    _leanDeviceExpertBytes -= bytes;
                }
            }
        }
    }

    /// <summary>The process-wide device cache is keyed by tensor name, so a
    /// tensor this loader's patch changes gets a key of its own — otherwise
    /// it would reuse (or leave behind) another runtime's device copy.
    /// Null when the patch does not touch the tensor.</summary>
    private string? PatchedDeviceKey(OperandRef operand) =>
        _patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _)
            ? $"{operand.ObjectId}/{operand.TensorName}#patch{_patch.Id}"
            : null;

    /// <summary>Merges the patch delta for this operand into the widened
    /// weight, in place. A delta whose element count differs from the
    /// widened tensor cannot be the same weight — the patch was made for a
    /// different container or tensor.</summary>
    /// <summary>
    /// Whether a resident matrix can stay as its stored BF16 bits: the
    /// storage is BF16 and no patch or in-memory delta changes it (a delta is
    /// rarely BF16-representable). Any code that reads
    /// <see cref="Tensor2D.Data"/> still gets the exact f32 values — the
    /// tensor widens itself on first touch — so this only changes what the
    /// GEMV kernels stream. <c>AMQL_RESIDENT_BF16=0</c> turns it off, widening
    /// every matrix up front as before.
    /// </summary>
    /// <summary>Bumped by every <see cref="AddDelta"/>, so a copy of the weights
    /// held elsewhere (the device engine's) can tell it has gone stale.</summary>
    internal int Version { get; private set; }

    /// <summary>A matrix's values for a copy held outside this loader (the
    /// device engine uploads them once): the stored BF16 bits when the matrix
    /// would be held as BF16 here, otherwise the f32 values with any patch and
    /// added delta applied. Nothing is cached or charged to the session's
    /// budget, so the host does not keep a second copy of what the device
    /// holds.</summary>
    internal (ushort[]? Bf16, float[]? F32) DeviceSource(OperandRef operand, int rows, int cols)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_matrices.TryGetValue(key, out var cached) && cached.Bf16Bits is null)
        {
            return (null, cached.Data);
        }
        if (HoldAsBf16(operand))
        {
            var (shape, bits) = _store.ResolveBf16(operand.ObjectId, operand.TensorName);
            if (shape.Length == 0 || ElementCount(shape) != (long)rows * cols)
            {
                throw new ContainerException(
                    $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                    $"[{string.Join(",", shape)}] — expected [{rows} x {cols}]");
            }
            return (bits, null);
        }
        var resolution = _store.ResolveWidened(operand);
        if (resolution.Shape.Length == 0 || ElementCount(resolution.Shape) != (long)rows * cols)
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                $"[{string.Join(",", resolution.Shape)}] — expected [{rows} x {cols}]");
        }
        var data = resolution.Values;
        ApplyPatch(operand, data);
        return (null, data);
    }

    /// <summary>The operand as stored when it is FP4 (an NVFP4 or MXFP4
    /// container, or a quantised checkpoint), and nothing patches it; else null.</summary>
    internal ScaledMatrix? StoredFp4(OperandRef operand)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_added.ContainsKey(key) || (_patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _)))
        {
            return null;
        }
        try
        {
            if (!_store.IsScaled(operand.ObjectId, operand.TensorName))
            {
                return null;
            }
        }
        catch (Amql.Vindex3.ContainerException)
        {
            return null;
        }
        return _store.ResolveScaled(operand) is { Element: Dtype.FP4 } m ? m : null;
    }

    /// <summary>The operand's stored NVFP4 or MXFP4 bytes, undecoded; null when it is
    /// stored any other way or a patch covers it.</summary>
    internal (byte[] Elements, byte[] Scales, float Global, bool Mxfp4, int Rows, int Cols)? StoredRawFp4(OperandRef operand)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_added.ContainsKey(key) || (_patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _)))
        {
            return null;
        }
        return _store.ContainsTensor(operand.ObjectId, operand.TensorName) ? _store.ResolveRawFp4(operand.ObjectId, operand.TensorName) : null;
    }

    /// <summary>The operand's values as stored, widened to f32 (patched, if a
    /// patch covers it), never through a BF16-resident copy.</summary>
    internal float[] StoredValues(OperandRef operand, int rows, int cols)
    {
        var resolution = _store.ResolveWidened(operand);
        if (ElementCount(resolution.Shape) != (long)rows * cols)
        {
            throw new ContainerException(
                $"operand '{operand.ObjectId}/{operand.TensorName}' resolves to shape " +
                $"[{string.Join(",", resolution.Shape)}] — expected [{rows} x {cols}]");
        }
        var data = resolution.Values;
        ApplyPatch(operand, data);
        return data;
    }

    private bool HoldAsBf16(OperandRef operand)
    {
        if (!ResidentBf16Enabled)
        {
            return false;
        }
        var key = (operand.ObjectId, operand.TensorName);
        if (_added.ContainsKey(key) || (_patch is not null && _patch.TryGet(operand.ObjectId, operand.TensorName, out _)))
        {
            return false;
        }
        return _store.DtypeOf(operand.ObjectId, operand.TensorName) == Dtype.BF16;
    }

    private static readonly bool ResidentBf16Enabled =
        Environment.GetEnvironmentVariable("AMQL_RESIDENT_BF16")?.Trim() is not ("0" or "false" or "off");

    private void ApplyPatch(OperandRef operand, float[] widened)
    {
        if (_added.TryGetValue((operand.ObjectId, operand.TensorName), out var added) &&
            added.Length == widened.Length)
        {
            for (int i = 0; i < widened.Length; i++)
            {
                widened[i] += added[i];
            }
        }
        if (_patch is null || !_patch.TryGet(operand.ObjectId, operand.TensorName, out var entry))
        {
            return;
        }
        if (entry.Delta.Length != widened.Length)
        {
            throw new ContainerException(
                $"patch entry '{entry.Key}' holds {entry.Delta.Length} deltas but the tensor " +
                $"'{operand.ObjectId}/{operand.TensorName}' has {widened.Length} elements");
        }
        for (int i = 0; i < widened.Length; i++)
        {
            widened[i] += entry.Delta[i];
        }
    }

    private static long ElementCount(long[] shape)
    {
        long n = 1;
        foreach (var dim in shape)
        {
            n = checked(n * dim);
        }
        return n;
    }

    private static string FormatBytes(long bytes) =>
        bytes >= 1L << 30
            ? $"{bytes / (double)(1L << 30):0.0} GiB"
            : $"{bytes / (double)(1L << 20):0.0} MiB";
}

/// <summary>What a lean working set holds and has read.</summary>
/// <param name="DenseBytes">Host bytes of dense weights cached.</param>
/// <param name="ExpertBytes">Host bytes of routed experts cached.</param>
/// <param name="ExpertsCached">Expert matrices cached.</param>
/// <param name="DeviceBytes">FP16 bytes of dense weights on the GPU.</param>
/// <param name="Reads">Matrices read from the container.</param>
/// <param name="BytesRead">Bytes those reads (and embedding rows) produced.</param>
/// <param name="HostReloads">GPU-held matrices whose host values had to be read back.</param>
/// <param name="CacheBudget">The host cache budget.</param>
/// <param name="DeviceBudget">The GPU budget.</param>
/// <param name="DeviceExperts">Expert matrices on the GPU.</param>
/// <param name="ScaledWidenings">Quantised matrices that had to be widened whole.</param>
public sealed record LeanStats(long DenseBytes, long ExpertBytes, int ExpertsCached, long DeviceBytes,
    long Reads, long BytesRead, long HostReloads, long CacheBudget, long DeviceBudget,
    int DeviceExperts = 0, long ScaledWidenings = 0);
