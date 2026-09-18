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
        var values = new float[elements];
        if (tensor.Len <= ChunkedReadThresholdBytes)
        {
            var widened = BitPattern.WidenToF32(dtype, segment.ReadBytes(tensorName));
            Array.Copy(widened, values, widened.Length);
        }
        else
        {
            int offset = 0;
            for (long done = 0; done < tensor.Len; done += ChunkedReadBytes)
            {
                int count = (int)Math.Min(ChunkedReadBytes, tensor.Len - done);
                var widened = BitPattern.WidenToF32(dtype, segment.ReadBytes(tensorName, done, count));
                Array.Copy(widened, 0, values, offset, widened.Length);
                offset += widened.Length;
            }
        }

        _touched.Add(objectId);
        Loads++;
        return new WidenedResolution(dtype, tensor.Shape, values);
    }

    public WidenedResolution ResolveWidened(OperandRef operand) =>
        ResolveWidened(operand.ObjectId, operand.TensorName);

    /// <summary>Raw payloads up to this size resolve as one buffer; larger
    /// tensors go through the chunked widened path. Below the 2 GiB array
    /// ceiling, with headroom for the widened allocation.</summary>
    private const long ChunkedReadThresholdBytes = 1L << 30; // 1 GiB
    private const long ChunkedReadBytes = 512L * 1024 * 1024; // 512 MiB

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