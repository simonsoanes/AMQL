using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.Json;

namespace Amql.Safetensors;

/// <summary>
/// A parsed safetensors file. The file is memory-mapped; only the header is
/// parsed eagerly (G0-style inventory: no payload I/O), tensor payloads are
/// read on demand. Mirrors the reference's usage of the Rust
/// <c>safetensors</c> crate plus its mmap-backed weight sources.
/// </summary>
public sealed class SafetensorsFile : IDisposable
{
    public const int HeaderLengthBytes = 8;
    public const int HeaderByteAlignment = 8;

    private readonly MemoryMappedFile _mapped;
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly Dictionary<string, TensorInfo> _tensors;

    private SafetensorsFile(
        string path,
        long headerLength,
        long payloadStart,
        Dictionary<string, TensorInfo> tensors,
        IReadOnlyDictionary<string, string>? metadata,
        MemoryMappedFile mapped,
        MemoryMappedViewAccessor accessor)
    {
        Path = path;
        HeaderLength = headerLength;
        PayloadStart = payloadStart;
        _tensors = tensors;
        Metadata = metadata;
        _mapped = mapped;
        _accessor = accessor;
    }

    /// <summary>Absolute path of the opened file.</summary>
    public string Path { get; }

    /// <summary>Byte length of the JSON header.</summary>
    public long HeaderLength { get; }

    /// <summary>File offset where the payload region begins
    /// (<c>8 + headerLength</c>).</summary>
    public long PayloadStart { get; }

    public IReadOnlyDictionary<string, TensorInfo> Tensors => _tensors;

    public IReadOnlyCollection<string> TensorNames => _tensors.Keys;

    /// <summary>File-level <c>__metadata__</c> entries, when the header
    /// carries any (null when absent).</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; }

    public static SafetensorsFile Open(string path)
    {
        var mapped = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var accessor = mapped.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        try
        {
            long fileLength = new FileInfo(path).Length;
            if (fileLength < HeaderLengthBytes)
            {
                throw new SafetensorsException($"'{path}' is shorter than the 8-byte safetensors header length field");
            }

            long headerLength = accessor.ReadInt64(0);
            if (headerLength < 0)
            {
                throw new SafetensorsException($"'{path}': negative header length {headerLength}");
            }

            long payloadStart = checked(HeaderLengthBytes + headerLength);
            if (payloadStart > fileLength)
            {
                throw new SafetensorsException(
                    $"'{path}': header claims {headerLength} bytes but the file is only {fileLength} bytes");
            }

            // The safetensors serialiser pads the header so the payload is
            // aligned; deserialisation refuses misaligned files.
            if ((payloadStart % HeaderByteAlignment) != 0)
            {
                throw new SafetensorsException(
                    $"'{path}': 8 + header length ({payloadStart}) is not a multiple of {HeaderByteAlignment} — " +
                    "header is not space-padded per the safetensors convention");
            }

            var headerBytes = new byte[headerLength];
            accessor.ReadArray(HeaderLengthBytes, headerBytes, 0, checked((int)headerLength));
            var (tensors, metadata) = ParseHeader(path, headerBytes);

            // Validate payload bounds eagerly (header-only read; payload
            // bytes themselves stay untouched until requested).
            foreach (var info in tensors.Values)
            {
                if (info.DataStart + info.DataLength > fileLength - payloadStart)
                {
                    throw new SafetensorsException(
                        $"'{path}': tensor '{info.Name}' payload [{info.DataStart}, {info.DataStart + info.DataLength}) " +
                        "exceeds the file payload region");
                }
            }

            return new SafetensorsFile(path, headerLength, payloadStart, tensors, metadata, mapped, accessor);
        }
        catch
        {
            accessor.Dispose();
            mapped.Dispose();
            throw;
        }
    }

    private static (Dictionary<string, TensorInfo> Tensors, IReadOnlyDictionary<string, string>? Metadata) ParseHeader(string path, byte[] headerBytes)
    {
        var result = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(headerBytes);
        }
        catch (JsonException e)
        {
            throw new SafetensorsException($"'{path}': malformed safetensors header JSON: {e.Message}", e);
        }

        Dictionary<string, string>? metadata = null;
        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw new SafetensorsException($"'{path}': safetensors header must be a JSON object");
            }

            foreach (var property in doc.RootElement.EnumerateObject())
            {
                if (property.Name == "__metadata__" && property.Value.ValueKind == JsonValueKind.Object)
                {
                    metadata = new Dictionary<string, string>(StringComparer.Ordinal);
                    foreach (var metaProperty in property.Value.EnumerateObject())
                    {
                        metadata[metaProperty.Name] = metaProperty.Value.GetString() ?? string.Empty;
                    }
                    continue; // file-level metadata, not a tensor
                }

                var entry = property.Value;
                if (entry.ValueKind != JsonValueKind.Object)
                {
                    throw new SafetensorsException($"'{path}': tensor '{property.Name}' entry is not an object");
                }

                var dataOffsets = entry.GetProperty("data_offsets");
                long from = dataOffsets[0].GetInt64();
                long to = dataOffsets[1].GetInt64();

                var shape = new List<long>();
                foreach (var dim in entry.GetProperty("shape").EnumerateArray())
                {
                    shape.Add(dim.GetInt64());
                }

                var info = new TensorInfo
                {
                    Name = property.Name,
                    Dtype = DtypeExtensions.FromLabel(entry.GetProperty("dtype").GetString() ?? string.Empty),
                    Shape = shape.ToArray(),
                    DataStart = from,
                    DataLength = to - from,
                };
                info.Validate();
                result[property.Name] = info;
            }
        }

        return (result, metadata);
    }

    public bool Contains(string name) => _tensors.ContainsKey(name);

    public TensorInfo GetTensor(string name)
    {
        if (_tensors.TryGetValue(name, out var info))
        {
            return info;
        }
        throw new SafetensorsException($"'{Path}' has no tensor '{name}'");
    }

    /// <summary>Copies a tensor's raw payload bytes out of the mapping.</summary>
    public byte[] ReadBytes(string name)
    {
        var info = GetTensor(name);
        return ReadBytes(info);
    }

    public byte[] ReadBytes(TensorInfo info)
    {
        var buffer = new byte[info.DataLength];
        if (info.DataLength > 0)
        {
            _accessor.ReadArray(checked(PayloadStart + info.DataStart), buffer, 0, checked((int)info.DataLength));
        }
        return buffer;
    }

    /// <summary>Widens a tensor's payload to f32. Refuses dtypes that have
    /// no widening path instead of guessing.</summary>
    public float[] DecodeF32(string name)
    {
        var info = GetTensor(name);
        return DecodeF32(info);
    }

    public float[] DecodeF32(TensorInfo info)
    {
        if (!info.Dtype.IsWidenableToF32())
        {
            throw new SafetensorsException(
                $"tensor '{info.Name}' has unsupported dtype {info.Dtype.Label()} for f32 widening");
        }
        return BitPattern.WidenToF32(info.Dtype, ReadBytes(info));
    }

    /// <summary>Reads only the header of a file; the payload is never
    /// touched. The returned object is fully functional for payload reads
    /// too — this is an inventory-oriented entry point.</summary>
    public void Dispose()
    {
        _accessor.Dispose();
        _mapped.Dispose();
    }
}