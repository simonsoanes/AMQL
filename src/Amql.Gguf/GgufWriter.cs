using System.Buffers.Binary;
using System.Text;

namespace Amql.Gguf;

/// <summary>
/// A streaming GGUF v3 writer. The header, metadata and tensor-info table
/// are written first (payload offsets pre-computed and 32-byte aligned),
/// then each tensor's payload is streamed through <see cref="Data"/> at
/// its recorded offset. Payloads are written in registration order; the
/// zero-pad between them is written by the writer so the file has no
/// unwritten holes.
/// </summary>
public sealed class GgufWriter : IDisposable
{
    private const uint DefaultVersion = 3;
    private const uint DefaultAlignment = 32;

    private readonly Stream _out;
    private readonly List<KeyValuePair<string, GgufValue>> _kv = new();
    private readonly List<(string Name, GgufType Type, long[] Dims, long DataBytes)> _tensors = new();
    private readonly List<ulong> _offsets = new();
    private bool _headerWritten;
    private int _nextTensor;

    public GgufWriter(Stream output)
    {
        _out = output;
    }

    public void Kv(string key, GgufValue value) => _kv.Add(new(key, value));

    /// <summary>Registers a tensor. <paramref name="dims"/> must already be
    /// in GGUF order (logical order reversed — fastest-varying first). The
    /// payload length is derived from the dims and the type's element
    /// size, so <paramref name="dims"/> and the streamed payload must
    /// agree.</summary>
    public void Tensor(string name, GgufType type, long[] dims)
    {
        long elements = 1;
        foreach (long d in dims)
        {
            elements = checked(elements * d);
        }
        _tensors.Add((name, type, dims, checked(elements * ElementSize(type))));
    }

    private static int ElementSize(GgufType type) => type switch
    {
        GgufType.F32 or GgufType.I32 => 4,
        GgufType.F16 or GgufType.BF16 or GgufType.I16 => 2,
        GgufType.I8 => 1,
        _ => throw new NotSupportedException($"gguf type {type} is not a plain dtype"),
    };

    private static ulong AlignUp(ulong value, uint alignment)
        => (value + alignment - 1) / alignment * alignment;

    /// <summary>The tensor data can be streamed here once
    /// <see cref="WriteHeader"/> has positioned it on the first payload.
    /// Call <see cref="FinishTensor"/> after each payload.</summary>
    public Stream Data => _out;

    public ulong TensorOffset(int index) => _offsets[index];

    public void WriteHeader()
    {
        if (_headerWritten)
        {
            throw new InvalidOperationException("header already written");
        }
        _headerWritten = true;

        // Serialise the metadata block into memory so its length is known
        // exactly before the offsets are computed.
        var kvBytes = new MemoryStream();
        foreach (var (key, value) in _kv)
        {
            WriteString(kvBytes, key);
            WriteValue(kvBytes, value);
        }

        long infoBytes = 0;
        foreach (var (name, _, dims, _) in _tensors)
        {
            infoBytes += StringSize(name) + 4 + 8L * dims.Length + 4 + 8;
        }

        const int headerBytes = 4 + 4 + 8 + 8;
        ulong dataStart = AlignUp((ulong)(headerBytes + kvBytes.Length + infoBytes), DefaultAlignment);
        ulong cursor = dataStart;
        foreach (var (_, _, _, dataBytes) in _tensors)
        {
            _offsets.Add(cursor);
            cursor += AlignUp((ulong)dataBytes, DefaultAlignment);
        }

        var scratch = new byte[8];
        _out.Write(Encoding.ASCII.GetBytes("GGUF"));
        BinaryPrimitives.WriteUInt32LittleEndian(scratch, DefaultVersion);
        _out.Write(scratch, 0, 4);
        BinaryPrimitives.WriteUInt64LittleEndian(scratch, (ulong)_tensors.Count);
        _out.Write(scratch, 0, 8);
        BinaryPrimitives.WriteUInt64LittleEndian(scratch, (ulong)_kv.Count);
        _out.Write(scratch, 0, 8);

        _out.Write(kvBytes.ToArray());
        PadTo(DefaultAlignment);

        for (int i = 0; i < _tensors.Count; i++)
        {
            var (name, type, dims, _) = _tensors[i];
            WriteString(_out, name);
            BinaryPrimitives.WriteUInt32LittleEndian(scratch, (uint)dims.Length);
            _out.Write(scratch, 0, 4);
            foreach (long d in dims)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(scratch, unchecked((ulong)d));
                _out.Write(scratch, 0, 8);
            }
            BinaryPrimitives.WriteUInt32LittleEndian(scratch, (uint)type);
            _out.Write(scratch, 0, 4);
            BinaryPrimitives.WriteUInt64LittleEndian(scratch, _offsets[i]);
            _out.Write(scratch, 0, 8);
        }

        PadTo(DefaultAlignment);
        SeekTensor(0);
    }

    /// <summary>Positions the data stream at tensor <paramref name="index"/>'s
    /// payload start. Tensors must be written in order; inside a tensor the
    /// caller may seek freely (strided transposes) as long as every byte of
    /// the payload region ends up written before
    /// <see cref="FinishTensor"/>.</summary>
    public void SeekTensor(int index)
    {
        if (index != _nextTensor)
        {
            throw new InvalidOperationException($"tensors must be written in order (next is {_nextTensor})");
        }
        _out.Position = (long)_offsets[index];
    }

    /// <summary>Absolute seek inside the data region (strided transpose
    /// writes). Must stay within the current tensor's payload span.</summary>
    public void SeekAbsolute(long position) => _out.Position = position;

    /// <summary>Marks tensor <paramref name="index"/> complete: zero-fills
    /// from the current position up to the tensor's aligned region end, so
    /// the file has no unwritten holes and the next tensor starts cleanly
    /// at a 32-byte boundary.</summary>
    public void FinishTensor(int index)
    {
        if (index != _nextTensor)
        {
            throw new InvalidOperationException($"tensor {_nextTensor} is pending, not {index}");
        }
        _nextTensor++;
        ulong end = _offsets[index] + AlignUp((ulong)_tensors[index].DataBytes, DefaultAlignment);
        while ((ulong)_out.Position < end)
        {
            int take = (int)Math.Min(end - (ulong)_out.Position, 1 << 16);
            _out.Write(new byte[take], 0, take);
        }
        _out.Position = (long)end;
    }

    /// <summary>Number of bytes the padded region of tensor <paramref name="index"/>
    /// occupies (payload rounded up to the 32-byte alignment).</summary>
    public ulong TensorRegionBytes(int index)
        => AlignUp((ulong)_tensors[index].DataBytes, DefaultAlignment);

    public void Dispose()
    {
        _out.Flush();
        _out.Dispose();
    }

    // ── primitives ────────────────────────────────────────────────────────

    private static long StringSize(string s) => 8 + Encoding.UTF8.GetByteCount(s);

    private static void WriteString(Stream s, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        Span<byte> len = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)bytes.Length);
        s.Write(len);
        s.Write(bytes);
    }

    private static void WriteValue(Stream s, GgufValue value)
    {
        Span<byte> scratch = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(scratch, (uint)value.Kind);
        s.Write(scratch[..4]);
        switch (value.Kind)
        {
            case GgufValueType.Bool:
                s.WriteByte((byte)((bool)value.Boxed ? 1 : 0));
                break;
            case GgufValueType.String:
                WriteString(s, (string)value.Boxed);
                break;
            case GgufValueType.Uint32:
                BinaryPrimitives.WriteUInt32LittleEndian(scratch, (uint)value.Boxed);
                s.Write(scratch[..4]);
                break;
            case GgufValueType.Uint64:
                BinaryPrimitives.WriteUInt64LittleEndian(scratch, (ulong)value.Boxed);
                s.Write(scratch);
                break;
            case GgufValueType.Int32:
                BinaryPrimitives.WriteInt32LittleEndian(scratch, (int)value.Boxed);
                s.Write(scratch[..4]);
                break;
            case GgufValueType.Int64:
                BinaryPrimitives.WriteInt64LittleEndian(scratch, (long)value.Boxed);
                s.Write(scratch);
                break;
            case GgufValueType.Float32:
                BinaryPrimitives.WriteSingleLittleEndian(scratch, (float)value.Boxed);
                s.Write(scratch[..4]);
                break;
            case GgufValueType.Array:
                var (elementKind, items) = value.AsArray();
                BinaryPrimitives.WriteUInt32LittleEndian(scratch, (uint)elementKind);
                s.Write(scratch[..4]);
                BinaryPrimitives.WriteUInt64LittleEndian(scratch, (ulong)items.Count);
                s.Write(scratch);
                foreach (object item in items)
                {
                    switch (item)
                    {
                        case string str:
                            WriteString(s, str);
                            break;
                        case uint u:
                            BinaryPrimitives.WriteUInt32LittleEndian(scratch, u);
                            s.Write(scratch[..4]);
                            break;
                        case int i:
                            BinaryPrimitives.WriteInt32LittleEndian(scratch, i);
                            s.Write(scratch[..4]);
                            break;
                        case float f:
                            BinaryPrimitives.WriteSingleLittleEndian(scratch, f);
                            s.Write(scratch[..4]);
                            break;
                        case long l:
                            BinaryPrimitives.WriteInt64LittleEndian(scratch, l);
                            s.Write(scratch);
                            break;
                        case ulong ul:
                            BinaryPrimitives.WriteUInt64LittleEndian(scratch, ul);
                            s.Write(scratch);
                            break;
                        case bool b:
                            s.WriteByte((byte)(b ? 1 : 0));
                            break;
                        default:
                            throw new NotSupportedException($"array element {item.GetType().Name}");
                    }
                }
                break;
            default:
                throw new NotSupportedException($"gguf value type {value.Kind}");
        }
    }

    private void PadTo(uint alignment)
    {
        long remaining = (long)(AlignUp((ulong)_out.Position, alignment) - (ulong)_out.Position);
        while (remaining > 0)
        {
            int take = (int)Math.Min(remaining, 1 << 16);
            _out.Write(new byte[take], 0, take);
            remaining -= take;
        }
    }
}