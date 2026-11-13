using Amql.Safetensors;

namespace Amql.Vindex3;

/// <summary>A resolved operand: storage dtype, logical shape and the raw
/// payload bytes from the segment. The execution layer owns widening to
/// f32 — the format layer does not know numerics.</summary>
public sealed record OperandResolution(Dtype Dtype, long[] Shape, byte[] Payload);

/// <summary>An operand widened to f32 in one buffer: the storage dtype,
/// logical shape and the whole tensor as <c>float[]</c>. Tensors beyond
/// the 2 GiB raw-payload ceiling (Qwen3.8-27B's 2.37 GiB embedding and
/// lm_head) are read chunked and widened chunk-by-chunk — the widened
/// element count is below the array ceiling even when the storage bytes
/// are not, so a single buffer is legal.</summary>
public sealed record WidenedResolution(Dtype Dtype, long[] Shape, float[] Values);

/// <summary>
/// Operand resolution for execution: <c>object id → representation →
/// segment → tensor table entry → payload bytes</c>. Segments are opened
/// lazily and cached per path; resolution never consults an HF checkpoint
/// or a safetensors shard — the container is the only authority, and the
/// runtime's deletion invariant holds by construction.
/// </summary>
public sealed class OperandStore : IDisposable
{
    private readonly Vindex3Container _container;
    private readonly Dictionary<string, SegmentFile> _open = new(StringComparer.Ordinal);
    private readonly HashSet<string> _touched = new(StringComparer.Ordinal);
    private bool _disposed;

    internal OperandStore(Vindex3Container container)
    {
        _container = container;
    }

    /// <summary>Logical objects this store has resolved at least one
    /// operand from (the reference's "touched" accounting).</summary>
    public IReadOnlyCollection<string> TouchedObjects => _touched;

    /// <summary>Number of tensor payloads read out of segments.</summary>
    public long Loads { get; private set; }

    public int OpenSegmentCount => _open.Count;

    /// <summary>The segment path a representation lives at, or null when
    /// the object has no directory entry (never assumed).</summary>
    public string? SegmentPathFor(string objectId)
    {
        if (_container.Index.Representations.TryGetValue(_container.CanonicalRepresentationId(objectId), out var entry))
        {
            return entry.Segment;
        }
        return null;
    }

    /// <summary>A tensor's shape from its segment header alone — no payload read.</summary>
    public long[] ShapeOf(string objectId, string tensorName)
    {
        ThrowIfDisposed();
        var representationId = _container.CanonicalRepresentationId(objectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{objectId}' has no representation entry '{representationId}' in index.representations");
        }
        return OpenSegment(entry.Segment).GetTensor(tensorName).Shape;
    }

    public OperandResolution Resolve(string objectId, string tensorName)
    {
        ThrowIfDisposed();

        var representationId = _container.CanonicalRepresentationId(objectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{objectId}' has no representation entry '{representationId}' in index.representations");
        }

        var segment = OpenSegment(entry.Segment);
        var tensor = segment.GetTensor(tensorName);
        var payload = segment.ReadBytes(tensorName);

        _touched.Add(objectId);
        Loads++;

        return new OperandResolution(DtypeExtensions.FromLabel(tensor.Dtype), tensor.Shape, payload);
    }

    /// <summary>Reads a tensor payload as chunked buffers (each up to
    /// <paramref name="chunkBytes"/> long, concatenated in order) — a
    /// single operand can exceed the 2 GiB array ceiling (Qwen3.8-27B's
    /// 2.37 GiB embedding), and chunked consumers never materialise it as
    /// one buffer.</summary>
    public IReadOnlyList<byte[]> ReadPayloadChunks(
        string objectId, string tensorName, int chunkBytes = 512 * 1024 * 1024)
    {
        ThrowIfDisposed();

        var representationId = _container.CanonicalRepresentationId(objectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{objectId}' has no representation entry '{representationId}' in index.representations");
        }

        var segment = OpenSegment(entry.Segment);
        var tensor = segment.GetTensor(tensorName);
        long remaining = tensor.Len;
        var chunks = new List<byte[]>();
        for (long done = 0; done < tensor.Len; done += chunkBytes)
        {
            int count = (int)Math.Min(chunkBytes, remaining);
            chunks.Add(segment.ReadBytes(tensorName, done, count));
            remaining -= count;
        }

        _touched.Add(objectId);
        Loads++;
        return chunks;
    }

    /// <summary>Whether the object's segment carries the named tensor.
    /// Operand-closure probing: the planner binds what actually exists and
    /// refuses what is missing — never guesses a spelling.</summary>
    public bool ContainsTensor(string objectId, string tensorName)
    {
        ThrowIfDisposed();
        if (SegmentPathFor(objectId) is not { } path)
        {
            return false;
        }
        return OpenSegment(path).Contains(tensorName);
    }

    /// <summary>Every tensor name the object's segment carries (header only).</summary>
    public IReadOnlyList<string> TensorNames(string objectId)
    {
        ThrowIfDisposed();
        if (SegmentPathFor(objectId) is not { } path)
        {
            return Array.Empty<string>();
        }
        return OpenSegment(path).Header.Tensors.Select(t => t.Name).ToList();
    }

    public OperandResolution Resolve(OperandRef operand) => Resolve(operand.ObjectId, operand.TensorName);

    /// <summary>Resolves an operand widened to f32 in a single buffer.
    /// Storage payloads over the array ceiling are read chunked and
    /// widened chunk-by-chunk — consumers that only need the f32 values
    /// (weight loaders, merge alignment, inspection) never materialise a
    /// raw byte[] that cannot exist.</summary>
    public WidenedResolution ResolveWidened(string objectId, string tensorName)
    {
        ThrowIfDisposed();

        var representationId = _container.CanonicalRepresentationId(objectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{objectId}' has no representation entry '{representationId}' in index.representations");
        }

        var segment = OpenSegment(entry.Segment);
        var tensor = segment.GetTensor(tensorName);
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        if (!dtype.IsWidenableToF32())
        {
            throw new ContainerException(
                $"operand '{objectId}/{tensorName}' dtype {dtype.Label()} has no f32 widening path");
        }

        long elements = 1;
        foreach (var dim in tensor.Shape)
        {
            elements = checked(elements * dim);
        }
        int size = BitPattern.WidenedElementBytes(dtype);
        if (tensor.Len != elements * size)
        {
            throw new ContainerException(
                $"operand '{objectId}/{tensorName}' holds {tensor.Len} bytes, not {elements} {dtype.Label()} values");
        }
        // Widened a chunk at a time, straight from the map into the result
        // and on every core: no whole-tensor staging copy, nothing zeroed
        // that is about to be overwritten. Each chunk lands in its own slice,
        // so the values are the same however the chunks are scheduled.
        var values = GC.AllocateUninitializedArray<float>(checked((int)elements));
        long chunks = (elements + WidenChunkElements - 1) / WidenChunkElements;
        Parallel.For(0L, chunks, chunk =>
        {
            long first = chunk * WidenChunkElements;
            int count = (int)Math.Min(WidenChunkElements, elements - first);
            var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(count * size);
            try
            {
                segment.ReadInto(tensorName, first * size, buffer, count * size);
                BitPattern.WidenInto(dtype, buffer.AsSpan(0, count * size), values.AsSpan((int)first, count));
            }
            finally
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
            }
        });

        _touched.Add(objectId);
        Loads++;
        return new WidenedResolution(dtype, tensor.Shape, values);
    }

    public WidenedResolution ResolveWidened(OperandRef operand) =>
        ResolveWidened(operand.ObjectId, operand.TensorName);

    /// <summary>Elements widened per chunk (4 MiB of f32).</summary>
    private const long WidenChunkElements = 1L << 20;

    private SegmentFile OpenSegment(string relativePath)
    {
        if (_open.TryGetValue(relativePath, out var cached))
        {
            return cached;
        }
        var opened = _container.OpenSegment(relativePath);
        _open[relativePath] = opened;
        return opened;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(OperandStore));
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        foreach (var segment in _open.Values)
        {
            segment.Dispose();
        }
        _open.Clear();
    }
}

/// <summary>An operand reference in an operation plan: a logical object id
/// plus a segment-relative tensor name. Never a raw HF tensor name — those
/// were stripped at encode time.</summary>
public sealed record OperandRef(string ObjectId, string TensorName);