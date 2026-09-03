using Amql.Safetensors;

namespace Amql.Vindex3;

/// <summary>A resolved operand: storage dtype, logical shape and the raw
/// payload bytes from the segment. The execution layer owns widening to
/// f32 — the format layer does not know numerics.</summary>
public sealed record OperandResolution(Dtype Dtype, long[] Shape, byte[] Payload);

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