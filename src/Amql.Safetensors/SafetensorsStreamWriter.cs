using System.Runtime.InteropServices;
using System.Text.Json;

namespace Amql.Safetensors;

/// <summary>
/// Writes a safetensors file one tensor at a time, so a checkpoint far larger
/// than memory can be produced while holding only the tensor (or the chunk of
/// it) being written. <see cref="SafetensorsWriter.Write"/> needs every
/// payload up front because the header, which lists every offset, comes
/// first; this writer reserves room for the header, streams payloads after
/// it, and fills the header in on <see cref="Complete"/>.
/// <para>Tensors are laid out in the order they are written, not sorted by
/// name: the format locates each tensor by its <c>data_offsets</c>, and
/// readers (this one and the reference crate) require only that the payload
/// regions are contiguous. If the header outgrows the reservation the file
/// is rewritten once with the right header, so a low estimate costs time,
/// never correctness.</para>
/// <para>A writer disposed without <see cref="Complete"/> deletes its file:
/// a failed build never leaves a checkpoint that looks finished.</para>
/// </summary>
public sealed class SafetensorsStreamWriter : IDisposable
{
    private const int CopyChunkBytes = 64 * 1024 * 1024;

    private readonly string _path;
    private readonly FileStream _file;
    private readonly long _reserved;
    private readonly List<(string Name, Dtype Dtype, long[] Shape, long From, long To)> _entries = new();
    private readonly HashSet<string> _names = new(StringComparer.Ordinal);
    private long _cursor;
    private bool _completed;

    /// <param name="reservedHeaderBytes">Room kept for the JSON header;
    /// about 160 bytes per tensor is plenty for typical names.</param>
    public SafetensorsStreamWriter(string path, long reservedHeaderBytes)
    {
        _path = path;
        // Header length field + header together end on the alignment.
        _reserved = RoundUp(Math.Max(reservedHeaderBytes, 64) + SafetensorsFile.HeaderLengthBytes,
            SafetensorsFile.HeaderByteAlignment) - SafetensorsFile.HeaderLengthBytes;
        _file = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 20);
        _file.Position = SafetensorsFile.HeaderLengthBytes + _reserved;
    }

    /// <summary>Bytes of payload written so far.</summary>
    public long PayloadBytes => _cursor;

    public int TensorCount => _entries.Count;

    /// <summary>Writes a tensor whose payload <paramref name="write"/>
    /// streams into the file. The byte count is checked against the dtype
    /// and shape.</summary>
    public void Write(string name, Dtype dtype, long[] shape, Action<Stream> write)
    {
        if (_completed)
        {
            throw new InvalidOperationException("the file is already complete");
        }
        if (!_names.Add(name))
        {
            throw new SafetensorsException($"tensor '{name}' written twice");
        }
        long from = _cursor;
        write(_file);
        long to = _file.Position - SafetensorsFile.HeaderLengthBytes - _reserved;
        new TensorInfo { Name = name, Dtype = dtype, Shape = shape, DataStart = from, DataLength = to - from }.Validate();
        _entries.Add((name, dtype, shape, from, to));
        _cursor = to;
    }

    public void Write(TensorPayload tensor) => Write(tensor.Name, tensor.Dtype, tensor.Shape, stream =>
    {
        if (tensor.Chunks is { } chunks)
        {
            foreach (var chunk in chunks)
            {
                stream.Write(chunk);
            }
        }
        else
        {
            stream.Write(tensor.Data);
        }
    });

    public void WriteF32(string name, float[] values, params long[] shape) =>
        Write(name, Dtype.F32, shape, stream => stream.Write(MemoryMarshal.AsBytes(values.AsSpan())));

    /// <summary>Writes the header and closes the file.</summary>
    public void Complete(IReadOnlyDictionary<string, string>? metadata = null)
    {
        if (_completed)
        {
            return;
        }
        var header = BuildHeader(metadata);
        if (header.Length <= _reserved)
        {
            WriteHeader(_file, header, _reserved);
            _file.Flush(flushToDisk: false);
            _file.Dispose();
        }
        else
        {
            Relocate(header);
        }
        _completed = true;
    }

    /// <summary>The header did not fit: copy the payload behind a header of
    /// the right size.</summary>
    private void Relocate(byte[] header)
    {
        long stored = RoundUp(header.Length + SafetensorsFile.HeaderLengthBytes, SafetensorsFile.HeaderByteAlignment)
            - SafetensorsFile.HeaderLengthBytes;
        string temp = _path + ".relocate";
        using (var target = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        {
            WriteHeader(target, header, stored);
            _file.Position = SafetensorsFile.HeaderLengthBytes + _reserved;
            var buffer = new byte[(int)Math.Min(CopyChunkBytes, Math.Max(_cursor, 1))];
            long left = _cursor;
            while (left > 0)
            {
                int read = _file.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
                if (read <= 0)
                {
                    throw new SafetensorsException($"'{_path}': payload ended early while relocating");
                }
                target.Write(buffer, 0, read);
                left -= read;
            }
        }
        _file.Dispose();
        File.Move(temp, _path, overwrite: true);
    }

    private static void WriteHeader(Stream stream, byte[] header, long stored)
    {
        stream.Position = 0;
        Span<byte> length = stackalloc byte[SafetensorsFile.HeaderLengthBytes];
        BitConverter.TryWriteBytes(length, (ulong)stored);
        stream.Write(length);
        stream.Write(header);
        var filler = new byte[stored - header.Length];
        Array.Fill(filler, (byte)0x20); // JSON whitespace after the closing brace
        stream.Write(filler);
    }

    private byte[] BuildHeader(IReadOnlyDictionary<string, string>? metadata)
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            foreach (var (name, dtype, shape, from, to) in _entries)
            {
                writer.WriteStartObject(name);
                writer.WriteString("dtype", dtype.Label());
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (var dim in shape)
                {
                    writer.WriteNumberValue(dim);
                }
                writer.WriteEndArray();
                writer.WritePropertyName("data_offsets");
                writer.WriteStartArray();
                writer.WriteNumberValue(from);
                writer.WriteNumberValue(to);
                writer.WriteEndArray();
                writer.WriteEndObject();
            }
            if (metadata is { Count: > 0 })
            {
                writer.WriteStartObject("__metadata__");
                foreach (var (key, value) in metadata)
                {
                    writer.WriteString(key, value);
                }
                writer.WriteEndObject();
            }
            writer.WriteEndObject();
        }
        return buffer.ToArray();
    }

    private static long RoundUp(long value, long unit) => (value + unit - 1) / unit * unit;

    public void Dispose()
    {
        if (_completed)
        {
            return;
        }
        _file.Dispose();
        try
        {
            File.Delete(_path);
        }
        catch (IOException)
        {
            // best effort: the caller is already handling the failure
        }
    }
}
