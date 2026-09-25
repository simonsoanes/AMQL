using System.Buffers.Binary;
using System.Text;

namespace Amql.Gguf;

/// <summary>Raised while parsing a GGUF file.</summary>
public sealed class GgufException : Exception
{
    public GgufException(string message) : base(message) { }
}

/// <summary>
/// A parsed GGUF file, backed by a read stream. Parses the header,
/// metadata and tensor-info table so callers can inspect the architecture,
/// the tokenizer arrays and every tensor's type/dims/offset, and copy
/// payloads out by name (verification tool, tests, and future readers).
/// </summary>
public sealed class GgufReader : IDisposable
{
    private readonly Stream _input;
    private readonly Dictionary<string, GgufValue> _metadata = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (GgufTensor Tensor, long DataBytes)> _tensors = new(StringComparer.Ordinal);
    private readonly List<GgufTensor> _ordered = new();
    private long _dataSectionStart;

    private GgufReader(Stream input)
    {
        _input = input;
    }

    public IReadOnlyDictionary<string, GgufValue> Metadata => _metadata;
    public IReadOnlyList<GgufTensor> Tensors => _ordered;

    public static GgufReader Open(string path)
    {
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.RandomAccess);
        var reader = new GgufReader(stream);
        reader.Parse();
        return reader;
    }

    private void Parse()
    {
        Span<byte> scratch = stackalloc byte[8];
        byte[] magic = new byte[4];
        _input.ReadExactly(magic);
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
        {
            throw new GgufException("not a GGUF file (bad magic)");
        }
        _input.ReadExactly(scratch[..4]);
        uint version = BinaryPrimitives.ReadUInt32LittleEndian(scratch);
        if (version < 3)
        {
            throw new GgufException($"unsupported GGUF version {version} (need >= 3)");
        }
        _input.ReadExactly(scratch);
        ulong tensorCount = BinaryPrimitives.ReadUInt64LittleEndian(scratch);
        _input.ReadExactly(scratch);
        ulong kvCount = BinaryPrimitives.ReadUInt64LittleEndian(scratch);

        for (ulong i = 0; i < kvCount; i++)
        {
            string key = ReadString();
            GgufValue value = ReadValue();
            _metadata[key] = value;
        }

        // No padding between the metadata and the tensor-info table —
        // llama.cpp parses the infos immediately after the kv section and
        // only aligns before the data section.

        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadString();
            _input.ReadExactly(scratch[..4]);
            uint nDims = BinaryPrimitives.ReadUInt32LittleEndian(scratch);
            var dims = new long[nDims];
            for (uint d = 0; d < nDims; d++)
            {
                _input.ReadExactly(scratch);
                dims[d] = unchecked((long)BinaryPrimitives.ReadUInt64LittleEndian(scratch));
            }
            _input.ReadExactly(scratch[..4]);
            var type = (GgufType)BinaryPrimitives.ReadUInt32LittleEndian(scratch);
            _input.ReadExactly(scratch);
            ulong offset = BinaryPrimitives.ReadUInt64LittleEndian(scratch);

            long dataBytes = GgufTypeSizing.DataBytes(type, dims);
            var tensor = new GgufTensor { Name = name, Type = type, Dims = dims, Offset = offset };
            _ordered.Add(tensor);
            _tensors[name] = (tensor, dataBytes);
        }

        // Offsets are stored relative to the data-section start (llama.cpp
        // semantics: first tensor 0, contiguous aligned); the data section
        // begins at the aligned end of the metadata + tensor-info block.
        uint alignment = 32;
        if (_metadata.TryGetValue("general.alignment", out var a) && a.Kind == GgufValueType.Uint32)
        {
            alignment = a.AsUInt32();
        }
        _dataSectionStart = (long)(((ulong)_input.Position + alignment - 1) / alignment * alignment);
    }

    public bool Contains(string name) => _tensors.ContainsKey(name);
    public GgufTensor GetTensor(string name) => _tensors[name].Tensor;

    public string Arch
        => _metadata.TryGetValue("general.architecture", out var v) ? v.AsString() : string.Empty;

    public GgufValue Get(string key)
        => _metadata.TryGetValue(key, out var value) ? value : throw new GgufException($"metadata '{key}' not present");

    public bool TryGet(string key, out GgufValue value) => _metadata.TryGetValue(key, out value!);

    /// <summary>Copies a tensor's payload out of the file.</summary>
    public byte[] ReadBytes(string name)
    {
        var (tensor, dataBytes) = _tensors[name];
        var buffer = new byte[dataBytes];
        _input.Position = _dataSectionStart + (long)tensor.Offset;
        _input.ReadExactly(buffer);
        return buffer;
    }

    private string ReadString()
    {
        Span<byte> len = stackalloc byte[8];
        _input.ReadExactly(len);
        ulong length = BinaryPrimitives.ReadUInt64LittleEndian(len);
        if (length > 1 << 28)
        {
            throw new GgufException($"suspicious string length {length}");
        }
        var bytes = new byte[length];
        _input.ReadExactly(bytes);
        return Encoding.UTF8.GetString(bytes);
    }

    private GgufValue ReadValue()
    {
        Span<byte> scratch = stackalloc byte[8];
        _input.ReadExactly(scratch[..4]);
        var kind = (GgufValueType)BinaryPrimitives.ReadUInt32LittleEndian(scratch);
        return kind switch
        {
            GgufValueType.Bool => GgufValue.Bool(_input.ReadByte() != 0),
            GgufValueType.String => GgufValue.String(ReadString()),
            GgufValueType.Uint32 => ReadScalar(4, s => GgufValue.Uint32(BinaryPrimitives.ReadUInt32LittleEndian(s))),
            GgufValueType.Uint64 => ReadScalar(8, s => GgufValue.Uint64(BinaryPrimitives.ReadUInt64LittleEndian(s))),
            GgufValueType.Int32 => ReadScalar(4, s => GgufValue.Int32(BinaryPrimitives.ReadInt32LittleEndian(s))),
            GgufValueType.Int64 => ReadScalar(8, s => GgufValue.Int64(BinaryPrimitives.ReadInt64LittleEndian(s))),
            GgufValueType.Float32 => ReadScalar(4, s => GgufValue.Float32(BinaryPrimitives.ReadSingleLittleEndian(s))),
            GgufValueType.Float64 => ReadScalar(8, s => GgufValue.Float64(BinaryPrimitives.ReadDoubleLittleEndian(s))),
            GgufValueType.Uint8 => ReadScalar(1, s => GgufValue.Uint32(s[0])),
            GgufValueType.Int8 => ReadScalar(1, s => GgufValue.Int32((sbyte)s[0])),
            GgufValueType.Uint16 => ReadScalar(2, s => GgufValue.Uint32(BinaryPrimitives.ReadUInt16LittleEndian(s))),
            GgufValueType.Int16 => ReadScalar(2, s => GgufValue.Int32(BinaryPrimitives.ReadInt16LittleEndian(s))),
            GgufValueType.Array => ReadArray(),
            _ => throw new GgufException($"unsupported metadata value type {kind}"),
        };
    }

    private GgufValue ReadScalar(int width, Func<Span<byte>, GgufValue> convert)
    {
        Span<byte> scratch = stackalloc byte[8];
        _input.ReadExactly(scratch[..width]);
        return convert(scratch);
    }

    private GgufValue ReadArray()
    {
        Span<byte> scratch = stackalloc byte[8];
        _input.ReadExactly(scratch[..4]);
        var elementKind = (GgufValueType)BinaryPrimitives.ReadUInt32LittleEndian(scratch);
        _input.ReadExactly(scratch);
        ulong count = BinaryPrimitives.ReadUInt64LittleEndian(scratch);

        var items = new List<object>((int)Math.Min(count, 1 << 20));
        for (ulong i = 0; i < count; i++)
        {
            switch (elementKind)
            {
                case GgufValueType.String:
                    items.Add(ReadString());
                    break;
                case GgufValueType.Uint32:
                    _input.ReadExactly(scratch[..4]);
                    items.Add(BinaryPrimitives.ReadUInt32LittleEndian(scratch));
                    break;
                case GgufValueType.Int32:
                    _input.ReadExactly(scratch[..4]);
                    items.Add(BinaryPrimitives.ReadInt32LittleEndian(scratch));
                    break;
                case GgufValueType.Float32:
                    _input.ReadExactly(scratch[..4]);
                    items.Add(BinaryPrimitives.ReadSingleLittleEndian(scratch));
                    break;
                case GgufValueType.Bool:
                    items.Add(_input.ReadByte() != 0);
                    break;
                default:
                    throw new GgufException($"unsupported array element type {elementKind}");
            }
        }
        return GgufValue.ArrayOf(elementKind, items);
    }

    public void Dispose() => _input.Dispose();
}