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

    // ── Scaled (block-quantised) tensors ────────────────────────────────────

    /// <summary>A scaled tensor's decoded scale grid and tensor scale, cached
    /// per segment tensor (they are small next to the elements).</summary>
    private sealed record ScaleGrid(float[] Scales, int Rows, int Cols, float Global);

    private readonly Dictionary<(string Segment, string Tensor), ScaleGrid?> _scaleGrids = new();

    private static readonly System.Text.RegularExpressions.Regex ShardName =
        new(@"^(?<table>.+)\.shard_\d+\.weight$", System.Text.RegularExpressions.RegexOptions.CultureInvariant);

    /// <summary>
    /// The scale grid of a tensor stored as low-precision elements times block
    /// scales, or null for a plain tensor. The scales are a companion tensor in
    /// the same segment, as quantised checkpoints ship them and encode copies
    /// them: <c>{name}_scale</c> (NVFP4, MXFP4, MXFP8, per-tensor FP8) or
    /// <c>{name}_scale_inv</c> (FP8 block scales), with an optional
    /// <c>{name}_scale_2</c> tensor scale (NVFP4). A row-sharded table's shards
    /// share <c>{table}.weight_scale</c>. U8 scales are E8M0 exponents.
    /// </summary>
    private ScaleGrid? ScaleGridOf(string segmentPath, SegmentFile segment, SegmentTensor tensor)
    {
        if (_scaleGrids.TryGetValue((segmentPath, tensor.Name), out var known))
        {
            return known;
        }
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        ScaleGrid? grid = null;
        if (dtype is Dtype.FP4 or Dtype.F8_E4M3 or Dtype.F8_E5M2 && tensor.Shape.Length is 1 or 2)
        {
            string? scaleName = new[] { tensor.Name + "_scale", tensor.Name + "_scale_inv" }
                .FirstOrDefault(segment.Contains);
            if (scaleName is null && ShardName.Match(tensor.Name) is { Success: true } shard &&
                segment.Contains(shard.Groups["table"].Value + ".weight_scale"))
            {
                scaleName = shard.Groups["table"].Value + ".weight_scale";
            }
            if (scaleName is not null)
            {
                long rows = tensor.Shape.Length == 2 ? tensor.Shape[0] : 1;
                var scale = segment.GetTensor(scaleName);
                var scales = ReadSmallWidened(segment, scale);
                int scaleRows, scaleCols;
                if (scale.Shape.Length == 2)
                {
                    (scaleRows, scaleCols) = (checked((int)scale.Shape[0]), checked((int)scale.Shape[1]));
                }
                else if (scales.Length == 1)
                {
                    (scaleRows, scaleCols) = (1, 1);
                }
                else if (scales.Length == rows)
                {
                    (scaleRows, scaleCols) = (scales.Length, 1); // one scale per output row
                }
                else
                {
                    throw new ContainerException(
                        $"scale '{scaleName}' [{string.Join(",", scale.Shape)}] does not fit '{tensor.Name}' [{string.Join(",", tensor.Shape)}]");
                }
                float global = 1f;
                if (segment.Contains(tensor.Name + "_scale_2"))
                {
                    var second = ReadSmallWidened(segment, segment.GetTensor(tensor.Name + "_scale_2"));
                    if (second.Length != 1)
                    {
                        throw new ContainerException($"'{tensor.Name}_scale_2' should hold one value, not {second.Length}");
                    }
                    global = second[0];
                }
                grid = new ScaleGrid(scales, scaleRows, scaleCols, global);
            }
            else if (dtype == Dtype.FP4)
            {
                throw new ContainerException(
                    $"'{tensor.Name}' is FP4 but its segment has no '{tensor.Name}_scale' to decode it with");
            }
        }
        _scaleGrids[(segmentPath, tensor.Name)] = grid;
        return grid;
    }

    private static float[] ReadSmallWidened(SegmentFile segment, SegmentTensor tensor)
    {
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        if (dtype == Dtype.U8)
        {
            dtype = Dtype.F8_E8M0; // MX block scales are stored as raw exponent bytes
        }
        var bytes = segment.ReadBytes(tensor.Name);
        var values = new float[bytes.Length / BitPattern.WidenedElementBytes(dtype)];
        BitPattern.WidenInto(dtype, bytes, values);
        return values;
    }

    /// <summary>Whether the operand is stored scaled (see <see cref="ResolveScaled"/>).</summary>
    public bool IsScaled(string objectId, string tensorName)
    {
        var (path, segment, tensor) = Locate(objectId, tensorName);
        return ScaleGridOf(path, segment, tensor) is not null;
    }

    /// <summary>
    /// A 2-D operand stored as low-precision elements times block scales,
    /// as it is stored: the elements are not widened. Null for a plain tensor.
    /// </summary>
    public ScaledMatrix? ResolveScaled(string objectId, string tensorName)
    {
        var (path, segment, tensor) = Locate(objectId, tensorName);
        if (ScaleGridOf(path, segment, tensor) is not { } grid)
        {
            return null;
        }
        if (tensor.Shape.Length != 2)
        {
            throw new ContainerException($"scaled operand '{objectId}/{tensorName}' is not 2-D");
        }
        var elements = segment.ReadBytes(tensorName);
        _touched.Add(objectId);
        Loads++;
        return new ScaledMatrix(DtypeExtensions.FromLabel(tensor.Dtype), elements,
            checked((int)tensor.Shape[0]), checked((int)tensor.Shape[1]),
            grid.Scales, grid.Rows, grid.Cols, grid.Global);
    }

    public ScaledMatrix? ResolveScaled(OperandRef operand) => ResolveScaled(operand.ObjectId, operand.TensorName);

    private (string Path, SegmentFile Segment, SegmentTensor Tensor) Locate(string objectId, string tensorName)
    {
        ThrowIfDisposed();
        var representationId = _container.CanonicalRepresentationId(objectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{objectId}' has no representation entry '{representationId}' in index.representations");
        }
        var segment = OpenSegment(entry.Segment);
        return (entry.Segment, segment, segment.GetTensor(tensorName));
    }

    /// <summary>Rows of a scaled table decoded straight from storage.</summary>
    private void ReadScaledRows(SegmentFile segment, SegmentTensor tensor, ScaleGrid grid,
        ReadOnlySpan<long> rows, Span<float> target)
    {
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        long height = tensor.Shape.Length == 2 ? tensor.Shape[0] : 1;
        int cols = checked((int)tensor.Shape[^1]);
        int rowBytes = dtype == Dtype.FP4 ? cols / 2 : cols;
        int blockRows = (int)((height + grid.Rows - 1) / grid.Rows);
        int blockCols = (cols + grid.Cols - 1) / grid.Cols;
        var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(rowBytes);
        try
        {
            for (int i = 0; i < rows.Length; i++)
            {
                long row = rows[i];
                if (row < 0 || row >= height)
                {
                    throw new ContainerException($"row {row} is outside '{tensor.Name}' ({height} rows)");
                }
                segment.ReadInto(tensor.Name, row * rowBytes, buffer, rowBytes);
                ScaledMatrix.DequantRowOf(dtype, buffer.AsSpan(0, rowBytes), cols,
                    grid.Scales.AsSpan((int)(row / blockRows) * grid.Cols, grid.Cols), blockCols, grid.Global,
                    target.Slice(i * cols, cols));
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>Resolves an operand widened to f32 in a single buffer.
    /// Storage payloads over the array ceiling are read chunked and
    /// widened chunk-by-chunk — consumers that only need the f32 values
    /// (weight loaders, merge alignment, inspection) never materialise a
    /// raw byte[] that cannot exist.</summary>
    public WidenedResolution ResolveWidened(string objectId, string tensorName) =>
        ResolveWidened(objectId, tensorName, null);

    /// <summary>As <see cref="ResolveWidened(string,string)"/>, filling
    /// <paramref name="reuse"/> when it is exactly the tensor's length.</summary>
    public WidenedResolution ResolveWidened(string objectId, string tensorName, float[]? reuse)
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
        if (ScaleGridOf(entry.Segment, segment, tensor) is { } grid)
        {
            // Stored as elements times block scales: decoded through them.
            long height = tensor.Shape.Length == 2 ? tensor.Shape[0] : 1;
            var decoded = reuse is not null && reuse.LongLength == height * tensor.Shape[^1]
                ? reuse
                : GC.AllocateUninitializedArray<float>(checked((int)(height * tensor.Shape[^1])));
            var all = new long[height];
            for (long r = 0; r < height; r++)
            {
                all[r] = r;
            }
            ReadScaledRows(segment, tensor, grid, all, decoded);
            _touched.Add(objectId);
            Loads++;
            return new WidenedResolution(dtype, tensor.Shape, decoded);
        }
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
        var values = reuse is not null && reuse.LongLength == elements
            ? reuse
            : GC.AllocateUninitializedArray<float>(checked((int)elements));
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

    /// <summary>A tensor's storage dtype from its segment header alone — no payload read.</summary>
    public Dtype DtypeOf(string objectId, string tensorName)
    {
        ThrowIfDisposed();
        var representationId = _container.CanonicalRepresentationId(objectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{objectId}' has no representation entry '{representationId}' in index.representations");
        }
        return DtypeExtensions.FromLabel(OpenSegment(entry.Segment).GetTensor(tensorName).Dtype);
    }

    /// <summary>A BF16 tensor's stored bit patterns as one <c>ushort[]</c>, read
    /// a chunk at a time on every core — the same reading as
    /// <see cref="ResolveWidened(string,string)"/> without the widening, so a
    /// kernel that widens in registers can stream half the bytes. Element
    /// counts up to the array ceiling are legal even when the payload exceeds
    /// 2 GiB.</summary>
    public (long[] Shape, ushort[] Bits) ResolveBf16(string objectId, string tensorName) =>
        ResolveBf16(objectId, tensorName, null);

    /// <summary>As <see cref="ResolveBf16(string,string)"/>, filling
    /// <paramref name="reuse"/> when it is exactly the tensor's length (a buffer
    /// recycled by the caller) instead of allocating.</summary>
    public (long[] Shape, ushort[] Bits) ResolveBf16(string objectId, string tensorName, ushort[]? reuse)
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
        if (DtypeExtensions.FromLabel(tensor.Dtype) != Dtype.BF16)
        {
            throw new ContainerException($"operand '{objectId}/{tensorName}' is {tensor.Dtype}, not BF16");
        }
        long elements = 1;
        foreach (var dim in tensor.Shape)
        {
            elements = checked(elements * dim);
        }
        if (tensor.Len != elements * 2)
        {
            throw new ContainerException(
                $"operand '{objectId}/{tensorName}' holds {tensor.Len} bytes, not {elements} BF16 values");
        }
        var bits = reuse is not null && reuse.LongLength == elements
            ? reuse
            : GC.AllocateUninitializedArray<ushort>(checked((int)elements));
        long chunks = (elements + WidenChunkElements - 1) / WidenChunkElements;
        Parallel.For(0L, chunks, chunk =>
        {
            long first = chunk * WidenChunkElements;
            int count = (int)Math.Min(WidenChunkElements, elements - first);
            var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(count * 2);
            try
            {
                segment.ReadInto(tensorName, first * 2, buffer, count * 2);
                System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(buffer.AsSpan(0, count * 2))
                    .CopyTo(bits.AsSpan((int)first, count));
            }
            finally
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
            }
        });

        _touched.Add(objectId);
        Loads++;
        return (tensor.Shape, bits);
    }

    /// <summary>
    /// Reads chosen rows of a 2-D tensor, widened to f32, without touching the
    /// rest of it: row <c>rows[i]</c> lands at <c>target[i·cols ..]</c>. This is
    /// how a table too large to hold (Qwen4's ~51B-parameter n-gram embedding)
    /// is looked up — a few rows per token, straight from the segment.
    /// </summary>
    public void ReadRowsWidened(OperandRef operand, ReadOnlySpan<long> rows, Span<float> target)
    {
        ThrowIfDisposed();
        var representationId = _container.CanonicalRepresentationId(operand.ObjectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{operand.ObjectId}' has no representation entry '{representationId}' in index.representations");
        }
        var segment = OpenSegment(entry.Segment);
        var tensor = segment.GetTensor(operand.TensorName);
        if (tensor.Shape.Length != 2)
        {
            throw new ContainerException($"operand '{operand.ObjectId}/{operand.TensorName}' is not a 2-D table");
        }
        if (ScaleGridOf(entry.Segment, segment, tensor) is { } scaled)
        {
            if (target.Length < rows.Length * tensor.Shape[1])
            {
                throw new ArgumentException($"{rows.Length} rows of {tensor.Shape[1]} need a larger target");
            }
            ReadScaledRows(segment, tensor, scaled, rows, target);
            _touched.Add(operand.ObjectId);
            return;
        }
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        long height = tensor.Shape[0];
        int cols = checked((int)tensor.Shape[1]);
        int rowBytes = cols * BitPattern.WidenedElementBytes(dtype);
        if (target.Length < rows.Length * cols)
        {
            throw new ArgumentException($"{rows.Length} rows of {cols} need {rows.Length * cols} values, the target holds {target.Length}");
        }
        var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(rowBytes);
        try
        {
            for (int i = 0; i < rows.Length; i++)
            {
                long row = rows[i];
                if (row < 0 || row >= height)
                {
                    throw new ContainerException(
                        $"row {row} is outside '{operand.ObjectId}/{operand.TensorName}' ({height} rows)");
                }
                segment.ReadInto(operand.TensorName, row * rowBytes, buffer, rowBytes);
                BitPattern.WidenInto(dtype, buffer.AsSpan(0, rowBytes), target.Slice(i * cols, cols));
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }
        _touched.Add(operand.ObjectId);
    }

    /// <summary>
    /// Reads rows <c>[first, first + count)</c> of a 2-D tensor into
    /// <paramref name="target"/>: widened to f32, or as raw BF16 bit patterns
    /// when <paramref name="bf16"/> is set (the tensor must then be BF16). One
    /// contiguous slice of the segment, read a chunk per core, so a matrix too
    /// large to hold can be used a block of rows at a time.
    /// </summary>
    public void ReadRowRange(OperandRef operand, long first, int count, float[] target) =>
        ReadRowRange(operand, first, count, target.Length, null, target);

    /// <summary>BF16 rows into an array, unwidened.</summary>
    public void ReadRowRange(OperandRef operand, long first, int count, ushort[] target) =>
        ReadRowRange(operand, first, count, target.Length, target, null);

    private void ReadRowRange(OperandRef operand, long first, int count, long targetLength,
        ushort[]? bits, float[]? values)
    {
        ThrowIfDisposed();
        var representationId = _container.CanonicalRepresentationId(operand.ObjectId);
        if (!_container.Index.Representations.TryGetValue(representationId, out var entry))
        {
            throw new ContainerException(
                $"object '{operand.ObjectId}' has no representation entry '{representationId}' in index.representations");
        }
        var segment = OpenSegment(entry.Segment);
        var tensor = segment.GetTensor(operand.TensorName);
        if (tensor.Shape.Length != 2)
        {
            throw new ContainerException($"operand '{operand.ObjectId}/{operand.TensorName}' is not 2-D");
        }
        if (values is not null && ScaleGridOf(entry.Segment, segment, tensor) is { } scaled)
        {
            var ids = new long[count];
            for (int i = 0; i < count; i++)
            {
                ids[i] = first + i;
            }
            ReadScaledRows(segment, tensor, scaled, ids, values);
            _touched.Add(operand.ObjectId);
            return;
        }
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        if (bits is not null && dtype != Dtype.BF16)
        {
            throw new ContainerException($"operand '{operand.ObjectId}/{operand.TensorName}' is {tensor.Dtype}, not BF16");
        }
        int cols = checked((int)tensor.Shape[1]);
        if (first < 0 || count < 0 || first + count > tensor.Shape[0])
        {
            throw new ContainerException(
                $"rows [{first}, {first + count}) are outside '{operand.ObjectId}/{operand.TensorName}' ({tensor.Shape[0]} rows)");
        }
        long elements = (long)count * cols;
        if (targetLength < elements)
        {
            throw new ArgumentException($"{count} rows of {cols} need {elements} values, the target holds {targetLength}");
        }
        int size = BitPattern.WidenedElementBytes(dtype);
        long start = first * cols;
        long chunks = (elements + WidenChunkElements - 1) / WidenChunkElements;
        Parallel.For(0L, chunks, chunk =>
        {
            long done = chunk * WidenChunkElements;
            int n = (int)Math.Min(WidenChunkElements, elements - done);
            var buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(n * size);
            try
            {
                segment.ReadInto(operand.TensorName, (start + done) * size, buffer, n * size);
                if (bits is not null)
                {
                    System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(buffer.AsSpan(0, n * 2))
                        .CopyTo(bits.AsSpan((int)done, n));
                }
                else
                {
                    BitPattern.WidenInto(dtype, buffer.AsSpan(0, n * size), values.AsSpan((int)done, n));
                }
            }
            finally
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
            }
        });
        _touched.Add(operand.ObjectId);
    }

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