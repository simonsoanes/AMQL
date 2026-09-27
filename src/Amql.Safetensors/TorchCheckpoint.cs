using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace Amql.Safetensors;

/// <summary>One tensor of a PyTorch state dict, as the pickle describes it:
/// which storage record holds its bytes, where in that storage it starts,
/// and its logical shape.</summary>
public sealed record TorchTensorInfo(
    string Name,
    Dtype Dtype,
    long[] Shape,
    string StorageKey,
    long StorageOffsetElements)
{
    public long Elements => Shape.Aggregate(1L, (a, b) => a * b);

    public long DataLength => Elements * Dtype.ElementSize();
}

/// <summary>
/// Reads a PyTorch zip checkpoint (<c>torch.save</c> of a state dict, the
/// format since torch 1.6) without Python: a zip archive holding
/// <c>&lt;archive&gt;/data.pkl</c>, the pickled dict, and one raw
/// little-endian storage per tensor under <c>&lt;archive&gt;/data/&lt;key&gt;</c>.
/// <para>
/// The pickle is interpreted by a small stack machine that accepts only the
/// globals a plain state dict needs — the same allow-list idea as
/// <c>torch.load(weights_only=True)</c>. A pickle that names any other global
/// is refused by name rather than executed, so opening an untrusted
/// checkpoint cannot run code. Non-contiguous views are refused too: their
/// bytes are not a straight slice of the storage, and gathering them would be
/// a transformation this reader is not in a position to judge.
/// </para>
/// </summary>
public sealed class TorchCheckpoint : IDisposable
{
    private readonly ZipArchive _zip;
    private readonly string _archive;
    private readonly Dictionary<string, TorchTensorInfo> _tensors;

    private TorchCheckpoint(string path, ZipArchive zip, string archive, List<TorchTensorInfo> ordered)
    {
        Path = path;
        _zip = zip;
        _archive = archive;
        TensorNames = ordered.Select(t => t.Name).ToArray();
        _tensors = ordered.ToDictionary(t => t.Name, StringComparer.Ordinal);
    }

    public string Path { get; }

    /// <summary>Tensor names in state-dict order (module registration order).</summary>
    public IReadOnlyList<string> TensorNames { get; }

    public static TorchCheckpoint Open(string path)
    {
        var stream = File.OpenRead(path);
        ZipArchive zip;
        try
        {
            zip = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: false);
        }
        catch (InvalidDataException e)
        {
            stream.Dispose();
            throw new SafetensorsException(
                $"'{path}' is not a zip-format PyTorch checkpoint (the legacy pre-1.6 torch.save format is not supported)", e);
        }

        try
        {
            var pkl = zip.Entries.FirstOrDefault(e =>
                e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal) &&
                e.FullName.IndexOf('/') == e.FullName.Length - "/data.pkl".Length);
            if (pkl is null)
            {
                throw new SafetensorsException($"'{path}': no '<archive>/data.pkl' record — not a torch.save checkpoint");
            }
            string archive = pkl.FullName[..^"/data.pkl".Length];

            var byteOrder = zip.GetEntry($"{archive}/byteorder");
            if (byteOrder is not null)
            {
                using var reader = new StreamReader(byteOrder.Open());
                string order = reader.ReadToEnd().Trim();
                if (order != "little")
                {
                    throw new SafetensorsException($"'{path}': storages are '{order}'-endian; only little-endian is supported");
                }
            }

            byte[] pickle;
            using (var s = pkl.Open())
            using (var ms = new MemoryStream())
            {
                s.CopyTo(ms);
                pickle = ms.ToArray();
            }

            object? root = new StateDictUnpickler(pickle, path).Load();
            if (root is not PickleDict dict)
            {
                throw new SafetensorsException($"'{path}': the pickled root is not a dict — not a state dict");
            }

            var tensors = new List<TorchTensorInfo>();
            foreach (var (key, value) in dict.Items)
            {
                if (key is not string name)
                {
                    throw new SafetensorsException($"'{path}': a state-dict key is not a string");
                }
                if (value is not TensorRef t)
                {
                    throw new SafetensorsException(
                        $"'{path}': entry '{name}' is not a tensor — only flat state dicts of tensors are supported");
                }
                tensors.Add(new TorchTensorInfo(name, t.Dtype, t.Shape, t.StorageKey, t.StorageOffset));
            }

            foreach (var t in tensors)
            {
                var entry = zip.GetEntry($"{archive}/data/{t.StorageKey}")
                    ?? throw new SafetensorsException($"'{path}': tensor '{t.Name}' names storage '{t.StorageKey}' but the archive has no such record");
                long needed = (t.StorageOffsetElements + t.Elements) * t.Dtype.ElementSize();
                if (entry.Length < needed)
                {
                    throw new SafetensorsException(
                        $"'{path}': tensor '{t.Name}' needs {needed} bytes of storage '{t.StorageKey}' but the record holds {entry.Length}");
                }
            }

            return new TorchCheckpoint(path, zip, archive, tensors);
        }
        catch
        {
            zip.Dispose();
            throw;
        }
    }

    public bool TryGet(string name, out TorchTensorInfo info) => _tensors.TryGetValue(name, out info!);

    public TorchTensorInfo Get(string name) =>
        _tensors.TryGetValue(name, out var info)
            ? info
            : throw new SafetensorsException($"'{Path}' has no tensor '{name}'");

    /// <summary>The tensor's payload, exactly as stored: little-endian
    /// elements in row-major order, the same bytes a safetensors file holds.</summary>
    public byte[] ReadBytes(string name)
    {
        var info = Get(name);
        long length = info.DataLength;
        if (length > Array.MaxLength)
        {
            throw new SafetensorsException($"tensor '{name}' is {length} bytes — beyond a single buffer");
        }
        var entry = _zip.GetEntry($"{_archive}/data/{info.StorageKey}")!;
        using var stream = entry.Open();
        Skip(stream, info.StorageOffsetElements * info.Dtype.ElementSize());
        var buffer = new byte[length];
        stream.ReadExactly(buffer);
        return buffer;
    }

    private static void Skip(Stream stream, long count)
    {
        var scratch = new byte[Math.Min(count, 1 << 20)];
        while (count > 0)
        {
            int n = stream.Read(scratch, 0, (int)Math.Min(count, scratch.Length));
            if (n == 0)
            {
                throw new EndOfStreamException();
            }
            count -= n;
        }
    }

    public void Dispose() => _zip.Dispose();

    // ── storage class names ──────────────────────────────────────────────

    internal static Dtype DtypeForStorage(string storageClass) => storageClass switch
    {
        "FloatStorage" => Dtype.F32,
        "DoubleStorage" => Dtype.F64,
        "HalfStorage" => Dtype.F16,
        "BFloat16Storage" => Dtype.BF16,
        "LongStorage" => Dtype.I64,
        "IntStorage" => Dtype.I32,
        "ShortStorage" => Dtype.I16,
        "CharStorage" => Dtype.I8,
        "ByteStorage" => Dtype.U8,
        "BoolStorage" => Dtype.BOOL,
        _ => throw new SafetensorsException($"storage class 'torch.{storageClass}' has no safetensors dtype mapping"),
    };

    internal static string StorageFor(Dtype dtype) => dtype switch
    {
        Dtype.F32 => "FloatStorage",
        Dtype.F64 => "DoubleStorage",
        Dtype.F16 => "HalfStorage",
        Dtype.BF16 => "BFloat16Storage",
        Dtype.I64 => "LongStorage",
        Dtype.I32 => "IntStorage",
        Dtype.I16 => "ShortStorage",
        Dtype.I8 => "CharStorage",
        Dtype.U8 => "ByteStorage",
        Dtype.BOOL => "BoolStorage",
        _ => throw new SafetensorsException($"dtype {dtype.Label()} has no PyTorch storage class"),
    };

    // ── pickle machine ───────────────────────────────────────────────────

    private sealed record Global(string Module, string Name);

    private sealed record TensorRef(Dtype Dtype, long[] Shape, string StorageKey, long StorageOffset);

    private sealed record StorageRef(Dtype Dtype, string Key);

    private sealed class PickleDict
    {
        public List<(object? Key, object? Value)> Items { get; } = new();
    }

    private sealed class Mark
    {
        public static readonly Mark Instance = new();
    }

    /// <summary>Just enough of the pickle protocol (0–5) to read what
    /// <c>torch.save(state_dict)</c> writes. Everything a state dict does
    /// not need is refused by opcode or by global name.</summary>
    private sealed class StateDictUnpickler
    {
        private readonly byte[] _data;
        private readonly string _path;
        private int _pos;
        private readonly List<object?> _stack = new();
        private readonly Dictionary<long, object?> _memo = new();

        public StateDictUnpickler(byte[] data, string path)
        {
            _data = data;
            _path = path;
        }

        private static readonly HashSet<(string, string)> Allowed = new()
        {
            ("collections", "OrderedDict"),
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch._utils", "_rebuild_parameter"),
            ("builtins", "dict"),
        };

        public object? Load()
        {
            while (true)
            {
                if (_pos >= _data.Length)
                {
                    throw Fail("pickle ended without STOP");
                }
                byte op = _data[_pos++];
                switch (op)
                {
                    case 0x80: _pos++; break;                           // PROTO
                    case 0x95: _pos += 8; break;                        // FRAME
                    case (byte)'.': return Pop();                       // STOP
                    case (byte)'(': _stack.Add(Mark.Instance); break;   // MARK
                    case (byte)'}': _stack.Add(new PickleDict()); break; // EMPTY_DICT
                    case (byte)']': _stack.Add(new List<object?>()); break; // EMPTY_LIST
                    case (byte)')': _stack.Add(Array.Empty<object?>()); break; // EMPTY_TUPLE
                    case (byte)'N': _stack.Add(null); break;
                    case 0x88: _stack.Add(true); break;                 // NEWTRUE
                    case 0x89: _stack.Add(false); break;                // NEWFALSE
                    case (byte)'K': _stack.Add((long)_data[_pos++]); break; // BININT1
                    case (byte)'M': _stack.Add((long)BinaryPrimitives.ReadUInt16LittleEndian(Take(2))); break;
                    case (byte)'J': _stack.Add((long)BinaryPrimitives.ReadInt32LittleEndian(Take(4))); break;
                    case 0x8a:                                          // LONG1
                    {
                        int n = _data[_pos++];
                        _stack.Add(LittleEndianSigned(Take(n)));
                        break;
                    }
                    case (byte)'G': _stack.Add(BinaryPrimitives.ReadDoubleBigEndian(Take(8))); break; // BINFLOAT
                    case (byte)'X': _stack.Add(Utf8(BinaryPrimitives.ReadInt32LittleEndian(Take(4)))); break; // BINUNICODE
                    case 0x8c: _stack.Add(Utf8(_data[_pos++])); break;  // SHORT_BINUNICODE
                    case (byte)'U': _stack.Add(Latin1(_data[_pos++])); break; // SHORT_BINSTRING
                    case (byte)'T': _stack.Add(Latin1(BinaryPrimitives.ReadInt32LittleEndian(Take(4)))); break; // BINSTRING
                    case (byte)'c':                                     // GLOBAL
                    {
                        string module = ReadLine();
                        string name = ReadLine();
                        _stack.Add(ResolveGlobal(module, name));
                        break;
                    }
                    case 0x93:                                          // STACK_GLOBAL
                    {
                        string name = (string)Pop()!;
                        string module = (string)Pop()!;
                        _stack.Add(ResolveGlobal(module, name));
                        break;
                    }
                    case (byte)'q': _memo[_data[_pos++]] = Peek(); break;          // BINPUT
                    case (byte)'r': _memo[BinaryPrimitives.ReadUInt32LittleEndian(Take(4))] = Peek(); break; // LONG_BINPUT
                    case 0x94: _memo[_memo.Count] = Peek(); break;                   // MEMOIZE
                    case (byte)'h': _stack.Add(Memo(_data[_pos++])); break;          // BINGET
                    case (byte)'j': _stack.Add(Memo(BinaryPrimitives.ReadUInt32LittleEndian(Take(4)))); break; // LONG_BINGET
                    case (byte)'t': _stack.Add(PopToMark().ToArray()); break;       // TUPLE
                    case 0x85: _stack.Add(new[] { Pop() }); break;                  // TUPLE1
                    case 0x86: { var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b }); break; }
                    case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b, c }); break; }
                    case (byte)'a': { var v = Pop(); ((List<object?>)Peek()!).Add(v); break; } // APPEND
                    case (byte)'e': { var items = PopToMark(); ((List<object?>)Peek()!).AddRange(items); break; } // APPENDS
                    case (byte)'s':                                     // SETITEM
                    {
                        var v = Pop();
                        var k = Pop();
                        AsDict(Peek()).Items.Add((k, v));
                        break;
                    }
                    case (byte)'u':                                     // SETITEMS
                    {
                        var items = PopToMark();
                        var dict = AsDict(Peek());
                        for (int i = 0; i + 1 < items.Count; i += 2)
                        {
                            dict.Items.Add((items[i], items[i + 1]));
                        }
                        break;
                    }
                    case (byte)'Q': _stack.Add(PersistentLoad(Pop())); break;       // BINPERSID
                    case (byte)'R':                                     // REDUCE
                    {
                        var args = (object?[])Pop()!;
                        var callable = Pop();
                        _stack.Add(Reduce(callable, args));
                        break;
                    }
                    case (byte)'b':                                     // BUILD
                        // The only state torch attaches is the OrderedDict's
                        // _metadata (per-module version numbers); a state dict
                        // needs none of it, so it is dropped.
                        Pop();
                        break;
                    default:
                        throw Fail($"pickle opcode 0x{op:x2} is not supported by the state-dict reader");
                }
            }
        }

        private object ResolveGlobal(string module, string name)
        {
            if (module == "torch" && name.EndsWith("Storage", StringComparison.Ordinal))
            {
                return DtypeForStorage(name);
            }
            if (!Allowed.Contains((module, name)))
            {
                throw Fail($"pickle references '{module}.{name}', which a plain state dict never needs — refusing to load it");
            }
            return new Global(module, name);
        }

        private object PersistentLoad(object? pid)
        {
            // ('storage', <storage class>, key, location, numel)
            if (pid is object?[] { Length: >= 5 } t && t[0] is "storage" && t[1] is Dtype dtype && t[2] is string key)
            {
                return new StorageRef(dtype, key);
            }
            throw Fail("unrecognised persistent id — expected ('storage', class, key, location, numel)");
        }

        private object? Reduce(object? callable, object?[] args)
        {
            if (callable is not Global g)
            {
                throw Fail("REDUCE on something that is not an allowed global");
            }
            switch (g.Name)
            {
                case "OrderedDict":
                case "dict":
                    return new PickleDict();
                case "_rebuild_parameter":
                    return args[0];
                case "_rebuild_tensor_v2":
                {
                    // (storage, storage_offset, size, stride, requires_grad, backward_hooks, [metadata])
                    if (args.Length < 4 || args[0] is not StorageRef storage)
                    {
                        throw Fail("_rebuild_tensor_v2 called with unexpected arguments");
                    }
                    long offset = (long)args[1]!;
                    long[] shape = ((object?[])args[2]!).Select(x => (long)x!).ToArray();
                    long[] stride = ((object?[])args[3]!).Select(x => (long)x!).ToArray();
                    long expected = 1;
                    for (int i = shape.Length - 1; i >= 0; i--)
                    {
                        if (shape[i] != 1 && stride[i] != expected)
                        {
                            throw Fail($"a tensor of shape [{string.Join(",", shape)}] has non-contiguous strides " +
                                       $"[{string.Join(",", stride)}] — save it with .contiguous() first");
                        }
                        expected *= shape[i];
                    }
                    return new TensorRef(storage.Dtype, shape, storage.Key, offset);
                }
                default:
                    throw Fail($"'{g.Module}.{g.Name}' cannot be called");
            }
        }

        private static PickleDict AsDict(object? o) =>
            o as PickleDict ?? throw new SafetensorsException("SETITEM(S) on something that is not a dict");

        private ReadOnlySpan<byte> Take(int n)
        {
            if (_pos + n > _data.Length)
            {
                throw Fail("pickle truncated");
            }
            var span = _data.AsSpan(_pos, n);
            _pos += n;
            return span;
        }

        private string Utf8(int n) => Encoding.UTF8.GetString(Take(n));

        private string Latin1(int n) => Encoding.Latin1.GetString(Take(n));

        private string ReadLine()
        {
            int end = Array.IndexOf(_data, (byte)'\n', _pos);
            if (end < 0)
            {
                throw Fail("pickle truncated inside GLOBAL");
            }
            string s = Encoding.ASCII.GetString(_data, _pos, end - _pos);
            _pos = end + 1;
            return s;
        }

        private static long LittleEndianSigned(ReadOnlySpan<byte> bytes)
        {
            if (bytes.Length == 0)
            {
                return 0;
            }
            if (bytes.Length > 8)
            {
                throw new SafetensorsException("pickle integer wider than 64 bits");
            }
            long value = 0;
            for (int i = bytes.Length - 1; i >= 0; i--)
            {
                value = (value << 8) | bytes[i];
            }
            if (bytes.Length < 8 && (bytes[^1] & 0x80) != 0)
            {
                value -= 1L << (8 * bytes.Length);
            }
            return value;
        }

        private object? Memo(long key) =>
            _memo.TryGetValue(key, out var v) ? v : throw Fail($"memo key {key} read before it was written");

        private object? Peek() => _stack.Count > 0 ? _stack[^1] : throw Fail("pickle stack underflow");

        private object? Pop()
        {
            var v = Peek();
            _stack.RemoveAt(_stack.Count - 1);
            return v;
        }

        private List<object?> PopToMark()
        {
            int mark = _stack.LastIndexOf(Mark.Instance);
            if (mark < 0)
            {
                throw Fail("pickle has no MARK to pop to");
            }
            var items = _stack.GetRange(mark + 1, _stack.Count - mark - 1);
            _stack.RemoveRange(mark, _stack.Count - mark);
            return items;
        }

        private SafetensorsException Fail(string message) => new($"'{_path}': {message}");
    }
}

/// <summary>
/// Writes a state dict in the format <c>torch.save</c> produces, so the file
/// loads with <c>torch.load(path, weights_only=True)</c> and
/// <c>load_state_dict(strict=True)</c>: a stored (uncompressed) zip with a
/// protocol-2 pickle, one storage record per tensor with its data aligned to
/// 64 bytes, and CPU as every storage's location. ZIP64 records are emitted
/// when the archive passes 4 GiB, as PyTorch's own writer does.
/// </summary>
public static class TorchCheckpointWriter
{
    private const int Alignment = 64;

    /// <summary>Writes <paramref name="tensors"/> in the order given — the
    /// order becomes the state dict's key order.</summary>
    public static void Write(string path, IEnumerable<TensorPayload> tensors, string archiveName = "archive")
    {
        var list = tensors.ToList();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var t in list)
        {
            t.Validate();
            if (!names.Add(t.Name))
            {
                throw new SafetensorsException($"duplicate tensor '{t.Name}' in a torch checkpoint write");
            }
            _ = TorchCheckpoint.StorageFor(t.Dtype);
        }

        using var file = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 1 << 20);
        var zip = new StoredZipWriter(file);
        zip.Add($"{archiveName}/data.pkl", new[] { BuildPickle(list) });
        zip.Add($"{archiveName}/.format_version", new[] { "1"u8.ToArray() });
        zip.Add($"{archiveName}/.storage_alignment", new[] { "64"u8.ToArray() });
        zip.Add($"{archiveName}/byteorder", new[] { "little"u8.ToArray() });
        for (int i = 0; i < list.Count; i++)
        {
            var t = list[i];
            zip.Add($"{archiveName}/data/{i}", t.Chunks ?? new[] { t.Data });
        }
        zip.Add($"{archiveName}/version", new[] { "3\n"u8.ToArray() });
        zip.Finish();
    }

    // ── pickle ───────────────────────────────────────────────────────────

    private static byte[] BuildPickle(IReadOnlyList<TensorPayload> tensors)
    {
        var p = new MemoryStream();
        p.Write([0x80, 0x02]);                                  // PROTO 2
        Global(p, "collections", "OrderedDict");
        p.WriteByte((byte)')');                                 // EMPTY_TUPLE
        p.WriteByte((byte)'R');                                 // REDUCE → OrderedDict()
        p.WriteByte((byte)'(');                                 // MARK
        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            Unicode(p, t.Name);
            Global(p, "torch._utils", "_rebuild_tensor_v2");
            p.WriteByte((byte)'(');                             // MARK (args)
            p.WriteByte((byte)'(');                             // MARK (persistent id)
            Unicode(p, "storage");
            Global(p, "torch", TorchCheckpoint.StorageFor(t.Dtype));
            Unicode(p, i.ToString(System.Globalization.CultureInfo.InvariantCulture));
            Unicode(p, "cpu");
            Int(p, t.PayloadLength / t.Dtype.ElementSize());
            p.WriteByte((byte)'t');                             // TUPLE
            p.WriteByte((byte)'Q');                             // BINPERSID
            Int(p, 0);                                          // storage offset
            Tuple(p, t.Shape);
            var stride = new long[t.Shape.Length];
            long step = 1;
            for (int d = t.Shape.Length - 1; d >= 0; d--)
            {
                stride[d] = step;
                step *= t.Shape[d];
            }
            Tuple(p, stride);
            p.WriteByte(0x89);                                  // requires_grad = False
            Global(p, "collections", "OrderedDict");
            p.WriteByte((byte)')');
            p.WriteByte((byte)'R');                             // backward_hooks = OrderedDict()
            p.WriteByte((byte)'t');                             // TUPLE (args)
            p.WriteByte((byte)'R');                             // REDUCE
        }
        p.WriteByte((byte)'u');                                 // SETITEMS
        p.WriteByte((byte)'.');                                 // STOP
        return p.ToArray();
    }

    private static void Global(Stream p, string module, string name)
    {
        p.WriteByte((byte)'c');
        p.Write(Encoding.ASCII.GetBytes($"{module}\n{name}\n"));
    }

    private static void Unicode(Stream p, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        p.WriteByte((byte)'X');
        Span<byte> len = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(len, bytes.Length);
        p.Write(len);
        p.Write(bytes);
    }

    private static void Int(Stream p, long v)
    {
        Span<byte> buf = stackalloc byte[8];
        if (v is >= 0 and < 256)
        {
            p.WriteByte((byte)'K');
            p.WriteByte((byte)v);
        }
        else if (v is >= 0 and < 65536)
        {
            p.WriteByte((byte)'M');
            BinaryPrimitives.WriteUInt16LittleEndian(buf, (ushort)v);
            p.Write(buf[..2]);
        }
        else if (v is >= int.MinValue and <= int.MaxValue)
        {
            p.WriteByte((byte)'J');
            BinaryPrimitives.WriteInt32LittleEndian(buf, (int)v);
            p.Write(buf[..4]);
        }
        else
        {
            p.WriteByte(0x8a);                                  // LONG1, 8 bytes
            p.WriteByte(8);
            BinaryPrimitives.WriteInt64LittleEndian(buf, v);
            p.Write(buf);
        }
    }

    private static void Tuple(Stream p, long[] values)
    {
        if (values.Length == 0)
        {
            p.WriteByte((byte)')');
            return;
        }
        p.WriteByte((byte)'(');
        foreach (long v in values)
        {
            Int(p, v);
        }
        p.WriteByte((byte)'t');
    }

    // ── zip ──────────────────────────────────────────────────────────────

    /// <summary>A minimal stored-only zip writer. System.IO.Compression
    /// cannot pad an entry's data to an alignment boundary, which PyTorch's
    /// reader expects (and which lets torch memory-map the storages), so the
    /// handful of records are written by hand.</summary>
    private sealed class StoredZipWriter
    {
        private readonly FileStream _file;
        private readonly List<(byte[] Name, uint Crc, long Size, long Offset)> _entries = new();

        public StoredZipWriter(FileStream file) => _file = file;

        public void Add(string name, IReadOnlyList<byte[]> chunks)
        {
            byte[] nameBytes = Encoding.UTF8.GetBytes(name);
            long size = chunks.Sum(c => (long)c.Length);
            long offset = _file.Position;
            bool zip64 = size >= uint.MaxValue || offset >= uint.MaxValue;

            // Local header: 30 fixed bytes + name + extra. The extra field
            // carries the ZIP64 sizes when needed, then "FB" padding so the
            // data starts on an alignment boundary.
            int zip64Extra = zip64 ? 20 : 0;
            long dataStart = offset + 30 + nameBytes.Length + zip64Extra + 4;
            int pad = (int)((Alignment - dataStart % Alignment) % Alignment);
            int extraLength = zip64Extra + 4 + pad;

            var header = new byte[30 + nameBytes.Length + extraLength];
            var h = header.AsSpan();
            BinaryPrimitives.WriteUInt32LittleEndian(h, 0x04034b50);
            BinaryPrimitives.WriteUInt16LittleEndian(h[4..], zip64 ? (ushort)45 : (ushort)20);
            BinaryPrimitives.WriteUInt16LittleEndian(h[6..], 0x0800);      // UTF-8 names
            BinaryPrimitives.WriteUInt16LittleEndian(h[8..], 0);           // stored
            BinaryPrimitives.WriteUInt32LittleEndian(h[10..], 0);          // time/date
            BinaryPrimitives.WriteUInt32LittleEndian(h[14..], 0);          // crc (patched)
            BinaryPrimitives.WriteUInt32LittleEndian(h[18..], zip64 ? uint.MaxValue : (uint)size);
            BinaryPrimitives.WriteUInt32LittleEndian(h[22..], zip64 ? uint.MaxValue : (uint)size);
            BinaryPrimitives.WriteUInt16LittleEndian(h[26..], (ushort)nameBytes.Length);
            BinaryPrimitives.WriteUInt16LittleEndian(h[28..], (ushort)extraLength);
            nameBytes.CopyTo(h[30..]);
            var extra = h[(30 + nameBytes.Length)..];
            if (zip64)
            {
                BinaryPrimitives.WriteUInt16LittleEndian(extra, 0x0001);
                BinaryPrimitives.WriteUInt16LittleEndian(extra[2..], 16);
                BinaryPrimitives.WriteUInt64LittleEndian(extra[4..], (ulong)size);
                BinaryPrimitives.WriteUInt64LittleEndian(extra[12..], (ulong)size);
                extra = extra[20..];
            }
            extra[0] = (byte)'F';
            extra[1] = (byte)'B';
            BinaryPrimitives.WriteUInt16LittleEndian(extra[2..], (ushort)pad);
            _file.Write(header);

            uint crc = 0;
            foreach (var chunk in chunks)
            {
                crc = Crc32.Append(crc, chunk);
                _file.Write(chunk);
            }

            long end = _file.Position;
            _file.Position = offset + 14;
            Span<byte> crcBytes = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(crcBytes, crc);
            _file.Write(crcBytes);
            _file.Position = end;

            _entries.Add((nameBytes, crc, size, offset));
        }

        public void Finish()
        {
            long cdStart = _file.Position;
            foreach (var (name, crc, size, offset) in _entries)
            {
                bool sizes64 = size >= uint.MaxValue;
                bool offset64 = offset >= uint.MaxValue;
                int extraLength = sizes64 || offset64 ? 4 + (sizes64 ? 16 : 0) + (offset64 ? 8 : 0) : 0;
                var rec = new byte[46 + name.Length + extraLength];
                var r = rec.AsSpan();
                BinaryPrimitives.WriteUInt32LittleEndian(r, 0x02014b50);
                BinaryPrimitives.WriteUInt16LittleEndian(r[4..], extraLength > 0 ? (ushort)45 : (ushort)20);
                BinaryPrimitives.WriteUInt16LittleEndian(r[6..], extraLength > 0 ? (ushort)45 : (ushort)20);
                BinaryPrimitives.WriteUInt16LittleEndian(r[8..], 0x0800);
                BinaryPrimitives.WriteUInt16LittleEndian(r[10..], 0);
                BinaryPrimitives.WriteUInt32LittleEndian(r[12..], 0);
                BinaryPrimitives.WriteUInt32LittleEndian(r[16..], crc);
                BinaryPrimitives.WriteUInt32LittleEndian(r[20..], sizes64 ? uint.MaxValue : (uint)size);
                BinaryPrimitives.WriteUInt32LittleEndian(r[24..], sizes64 ? uint.MaxValue : (uint)size);
                BinaryPrimitives.WriteUInt16LittleEndian(r[28..], (ushort)name.Length);
                BinaryPrimitives.WriteUInt16LittleEndian(r[30..], (ushort)extraLength);
                BinaryPrimitives.WriteUInt32LittleEndian(r[42..], offset64 ? uint.MaxValue : (uint)offset);
                name.CopyTo(r[46..]);
                if (extraLength > 0)
                {
                    var e = r[(46 + name.Length)..];
                    BinaryPrimitives.WriteUInt16LittleEndian(e, 0x0001);
                    BinaryPrimitives.WriteUInt16LittleEndian(e[2..], (ushort)(extraLength - 4));
                    int at = 4;
                    if (sizes64)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(e[at..], (ulong)size);
                        BinaryPrimitives.WriteUInt64LittleEndian(e[(at + 8)..], (ulong)size);
                        at += 16;
                    }
                    if (offset64)
                    {
                        BinaryPrimitives.WriteUInt64LittleEndian(e[at..], (ulong)offset);
                    }
                }
                _file.Write(rec);
            }
            long cdEnd = _file.Position;
            long cdSize = cdEnd - cdStart;
            bool needs64 = cdStart >= uint.MaxValue || cdEnd >= uint.MaxValue || _entries.Count >= ushort.MaxValue;

            if (needs64)
            {
                var z = new byte[56 + 20];
                var s = z.AsSpan();
                BinaryPrimitives.WriteUInt32LittleEndian(s, 0x06064b50);          // ZIP64 EOCD record
                BinaryPrimitives.WriteUInt64LittleEndian(s[4..], 44);
                BinaryPrimitives.WriteUInt16LittleEndian(s[12..], 45);
                BinaryPrimitives.WriteUInt16LittleEndian(s[14..], 45);
                BinaryPrimitives.WriteUInt64LittleEndian(s[24..], (ulong)_entries.Count);
                BinaryPrimitives.WriteUInt64LittleEndian(s[32..], (ulong)_entries.Count);
                BinaryPrimitives.WriteUInt64LittleEndian(s[40..], (ulong)cdSize);
                BinaryPrimitives.WriteUInt64LittleEndian(s[48..], (ulong)cdStart);
                BinaryPrimitives.WriteUInt32LittleEndian(s[56..], 0x07064b50);    // ZIP64 EOCD locator
                BinaryPrimitives.WriteUInt64LittleEndian(s[64..], (ulong)cdEnd);
                BinaryPrimitives.WriteUInt32LittleEndian(s[72..], 1);
                _file.Write(z);
            }

            var eocd = new byte[22];
            var q = eocd.AsSpan();
            BinaryPrimitives.WriteUInt32LittleEndian(q, 0x06054b50);
            ushort count = needs64 ? ushort.MaxValue : (ushort)_entries.Count;
            BinaryPrimitives.WriteUInt16LittleEndian(q[8..], count);
            BinaryPrimitives.WriteUInt16LittleEndian(q[10..], count);
            BinaryPrimitives.WriteUInt32LittleEndian(q[12..], needs64 ? uint.MaxValue : (uint)cdSize);
            BinaryPrimitives.WriteUInt32LittleEndian(q[16..], needs64 ? uint.MaxValue : (uint)cdStart);
            _file.Write(eocd);
        }
    }

    /// <summary>Table-driven CRC-32 (IEEE 802.3), sliced by 8. Zip readers
    /// verify it, and the BCL only ships one in a separate package.</summary>
    private static class Crc32
    {
        private static readonly uint[] Table = BuildTable();

        private static uint[] BuildTable()
        {
            var t = new uint[8 * 256];
            for (uint i = 0; i < 256; i++)
            {
                uint c = i;
                for (int k = 0; k < 8; k++)
                {
                    c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                }
                t[i] = c;
            }
            for (int i = 0; i < 256; i++)
            {
                for (int s = 1; s < 8; s++)
                {
                    t[s * 256 + i] = (t[(s - 1) * 256 + i] >> 8) ^ t[t[(s - 1) * 256 + i] & 0xFF];
                }
            }
            return t;
        }

        public static uint Append(uint crc, ReadOnlySpan<byte> data)
        {
            uint c = ~crc;
            int i = 0;
            for (; i + 8 <= data.Length; i += 8)
            {
                uint lo = c ^ BinaryPrimitives.ReadUInt32LittleEndian(data[i..]);
                uint hi = BinaryPrimitives.ReadUInt32LittleEndian(data[(i + 4)..]);
                c = Table[7 * 256 + (lo & 0xFF)] ^ Table[6 * 256 + ((lo >> 8) & 0xFF)] ^
                    Table[5 * 256 + ((lo >> 16) & 0xFF)] ^ Table[4 * 256 + (lo >> 24)] ^
                    Table[3 * 256 + (hi & 0xFF)] ^ Table[2 * 256 + ((hi >> 8) & 0xFF)] ^
                    Table[1 * 256 + ((hi >> 16) & 0xFF)] ^ Table[hi >> 24];
            }
            for (; i < data.Length; i++)
            {
                c = Table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
            }
            return ~c;
        }
    }
}
