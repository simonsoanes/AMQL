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
///   </list>
/// Embeddings, the output head, norms and other 1-D tensors stay
/// f32-resident in every mode (precision-sensitive, small by comparison).
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
        var key = (operand.ObjectId, operand.TensorName);
        if (_matrices.TryGetValue(key, out var cached))
        {
            TouchLru(key);
            LoadTrace?.Invoke(operand.ObjectId, operand.TensorName, new long[] { rows, cols }, true);
            return cached;
        }

        var shape = new[] { (long)rows, cols };
        Tensor2D matrix;
        if (_workingSet != WeightWorkingSet.ResidentF32 && Compactable(operand, shape))
        {
            matrix = _workingSet == WeightWorkingSet.OnDemandBf16
                ? MatrixOnDemandBf16(operand, rows, cols)
                : MatrixMxfp4(operand, rows, cols);
            CacheBounded(key, matrix);
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

    /// <summary>BF16 on-demand path: the tensor's stored bytes stay
    /// resident (2 B/element; halving the f32 working set) and the f32
    /// view comes from the LRU, widened from the stored payload on a
    /// miss. Widening is lossless, so this is the resident path's
    /// numerics with a bounded working set.</summary>
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
        return new Tensor2D(widened, rows, cols);
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
            var values = resolution.Values;
            ApplyPatch(operand, values);
            _packs[key] = Mxfp4.Quantize(values, rows, cols);
        }

        var dequant = Mxfp4.Dequant(_packs[key].Packed, _packs[key].BlockScales, rows, cols);
        ApplyPatch(operand, dequant); // re-apply deltas on the dequantised view
        var matrix = new Tensor2D(dequant, rows, cols);
        matrix.DeviceWeightF16 = CudaShim.UploadWeightF16(
            operand.ObjectId, operand.TensorName, _packs[key].Packed, _packs[key].BlockScales, rows, cols);
        return matrix;
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

    /// <summary>Loads a 1-D weight (norm scales).</summary>
    public float[] Vector(OperandRef operand, int width)
    {
        var key = (operand.ObjectId, operand.TensorName);
        if (_vectors.TryGetValue(key, out var cached))
        {
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
        return vector;
    }

    public int LoadedMatrixCount => _matrices.Count;
    public int LoadedVectorCount => _vectors.Count;

    /// <summary>Merges the patch delta for this operand into the widened
    /// weight, in place. A delta whose element count differs from the
    /// widened tensor cannot be the same weight — the patch was made for a
    /// different container or tensor.</summary>
    private void ApplyPatch(OperandRef operand, float[] widened)
    {
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