using System.IO.MemoryMappedFiles;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Vindex3;

public sealed class SegmentFormat
{
    public const uint CurrentSchema = 1;
    public const int HeaderLengthBytes = 8;
    public const int PayloadAlignment = 16;
}

/// <summary>One tensor table entry of a segment header: name relative to
/// the object ("3.self_attn.q_proj.weight"), storage dtype label (the
/// safetensors uppercase spelling passes through), shape, and the byte
/// range inside the payload region.</summary>
public sealed record SegmentTensor(string Name, string Dtype, long[] Shape, long Offset, long Len);

/// <summary>The JSON header framing a segment payload. Mirrors the
/// safetensors file framing with 16-byte payload alignment.</summary>
public sealed record SegmentHeader(int Schema, string Representation, List<SegmentTensor> Tensors);

/// <summary>Result of writing one segment: the hashes the index records
/// (payload-only and whole-file), plus the payload byte count.</summary>
public sealed record SegmentWriteResult(long PayloadBytes, string PayloadSha256Hex, string SegmentSha256Hex);

/// <summary>
/// Writes a VINDEX3 <c>.bin</c> segment — the reference's
/// <c>encode/segment.rs</c> layout:
///
/// <code>
/// [8 bytes u64 LE: header length]
/// [header JSON, space-padded so 8 + headerLen ≡ 0 (mod 16)]
/// [payload bytes, in tensor-table order]
/// </code>
///
/// Table order is deterministic — sorted by name — payload written in the
/// same order, offsets still recorded relative to the payload start. Two
/// SHA-256s are computed in the single write pass.
/// </summary>
public static class SegmentWriter
{
    public static SegmentWriteResult Write(
        string path,
        string representationId,
        IEnumerable<NamedTensorData> tensors)
    {
        var ordered = tensors.OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        foreach (var tensor in ordered)
        {
            tensor.Validate();
        }

        var entries = new List<(NamedTensorData Tensor, long From)>();
        long cursor = 0;
        foreach (var tensor in ordered)
        {
            entries.Add((tensor, cursor));
            cursor += tensor.Data.Length;
        }

        // ── header JSON ──────────────────────────────────────────────────
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteNumber("schema", SegmentFormat.CurrentSchema);
            writer.WriteString("representation", representationId);
            writer.WritePropertyName("tensors");
            writer.WriteStartArray();
            foreach (var (tensor, from) in entries)
            {
                writer.WriteStartObject();
                writer.WriteString("name", tensor.Name);
                writer.WriteString("dtype", tensor.Dtype.Label());
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (var dim in tensor.Shape)
                {
                    writer.WriteNumberValue(dim);
                }
                writer.WriteEndArray();
                writer.WriteNumber("offset", from);
                writer.WriteNumber("len", tensor.Data.Length);
                writer.WriteEndObject();
            }
            writer.WriteEndArray();
            writer.WriteEndObject();
            writer.Flush();
        }

        var headerJson = buffer.ToArray();
        int pad = (SegmentFormat.PayloadAlignment -
                   ((SegmentFormat.HeaderLengthBytes + headerJson.Length) % SegmentFormat.PayloadAlignment)) %
                  SegmentFormat.PayloadAlignment;

        // The stored length is the PADDED header length, so the reader's
        // `payload = 8 + storedLen` lands exactly on the aligned boundary —
        // same convention as the safetensors framing.
        int storedLength = headerJson.Length + pad;
        long total = SegmentFormat.HeaderLengthBytes + storedLength + cursor;
        var file = new byte[total];
        BitConverter.TryWriteBytes(file.AsSpan(0, 8), (ulong)storedLength);
        headerJson.CopyTo(file, 8);
        file.AsSpan(8 + headerJson.Length, pad).Fill(0x20);

        long payloadOffset = SegmentFormat.HeaderLengthBytes + headerJson.Length + pad;
        foreach (var (tensor, from) in entries)
        {
            tensor.Data.CopyTo(file, payloadOffset + from);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, file);

        var payloadSpan = file.AsSpan((int)payloadOffset);
        return new SegmentWriteResult(
            PayloadBytes: cursor,
            PayloadSha256Hex: Convert.ToHexStringLower(SHA256.HashData(payloadSpan)),
            SegmentSha256Hex: Convert.ToHexStringLower(SHA256.HashData(file)));
    }
}

/// <summary>A named tensor's raw payload being written somewhere (a
/// safetensors file or a VINDEX3 segment).</summary>
public sealed record NamedTensorData
{
    public required string Name { get; init; }
    public required Dtype Dtype { get; init; }
    public required long[] Shape { get; init; }
    public required byte[] Data { get; init; }

    /// <summary>Shape/dtype/payload agreement — the reference's tensor
    /// fact validation.</summary>
    public void Validate()
    {
        long elements = 1;
        foreach (var dim in Shape)
        {
            if (dim < 0)
            {
                throw new ContainerException($"tensor '{Name}': negative shape dimension");
            }
            elements = checked(elements * dim);
        }
        long expected = checked(elements * Dtype.ElementSize());
        if (Data.Length != expected)
        {
            throw new ContainerException(
                $"tensor '{Name}': payload has {Data.Length} bytes but dtype {Dtype.Label()} " +
                $"shape [{string.Join(",", Shape)}] requires {expected}");
        }
    }
}

/// <summary>
/// A memory-mapped segment providing random access to tensor payloads
/// (payload start from the header, offsets relative to it). The reader
/// parses the JSON header only; payload bytes are read on demand.
/// </summary>
public sealed class SegmentFile : IDisposable
{
    private readonly MemoryMappedFile _mapped;
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly Dictionary<string, SegmentTensor> _byName;

    private SegmentFile(
        string path,
        SegmentHeader header,
        long payloadStart,
        long fileLength,
        MemoryMappedFile mapped,
        MemoryMappedViewAccessor accessor)
    {
        Path = path;
        Header = header;
        PayloadStart = payloadStart;
        FileLength = fileLength;
        _mapped = mapped;
        _accessor = accessor;
        _byName = header.Tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
    }

    public string Path { get; }
    public SegmentHeader Header { get; }
    public long PayloadStart { get; }
    public long FileLength { get; }

    public static SegmentFile Open(string path)
    {
        var mapped = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var accessor = mapped.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        try
        {
            long fileLength = new FileInfo(path).Length;
            long headerLength = accessor.ReadInt64(0);
            long payloadStart = checked(SegmentFormat.HeaderLengthBytes + headerLength);
            if (payloadStart > fileLength)
            {
                throw new ContainerException($"segment '{path}': header claims {headerLength} bytes, file is {fileLength}");
            }
            if ((payloadStart % SegmentFormat.PayloadAlignment) != 0)
            {
                throw new ContainerException(
                    $"segment '{path}': 8 + header length ({payloadStart}) is not a multiple of {SegmentFormat.PayloadAlignment}");
            }

            var headerBytes = new byte[headerLength];
            accessor.ReadArray(SegmentFormat.HeaderLengthBytes, headerBytes, 0, checked((int)headerLength));

            SegmentHeader header;
            try
            {
                header = JsonSerializer.Deserialize<SegmentHeader>(headerBytes, ViJson.Options)
                         ?? throw new ContainerException($"segment '{path}': empty header");
            }
            catch (JsonException e)
            {
                throw new ContainerException($"segment '{path}': malformed header JSON: {e.Message}", e);
            }

            if (header.Schema != SegmentFormat.CurrentSchema)
            {
                throw new ContainerException(
                    $"segment '{path}': schema {header.Schema} is not supported (expected {SegmentFormat.CurrentSchema})");
            }

            foreach (var tensor in header.Tensors)
            {
                if (tensor.Name.Length == 0 || tensor.Offset < 0 || tensor.Len < 0)
                {
                    throw new ContainerException($"segment '{path}': invalid tensor table entry");
                }
            }

            return new SegmentFile(path, header, payloadStart, fileLength, mapped, accessor);
        }
        catch
        {
            accessor.Dispose();
            mapped.Dispose();
            throw;
        }
    }

    public bool Contains(string name) => _byName.ContainsKey(name);

    public SegmentTensor GetTensor(string name)
    {
        if (_byName.TryGetValue(name, out var tensor))
        {
            return tensor;
        }
        throw new ContainerException($"segment '{Path}' has no tensor '{name}'");
    }

    /// <summary>Copies a tensor payload region out of the mapping. The
    /// caller owns widening; this returns storage bytes.</summary>
    public byte[] ReadBytes(string name)
    {
        var tensor = GetTensor(name);
        var buffer = new byte[tensor.Len];
        if (tensor.Len > 0)
        {
            _accessor.ReadArray(checked(PayloadStart + tensor.Offset), buffer, 0, checked((int)tensor.Len));
        }
        return buffer;
    }

    /// <summary>All payload bytes, contiguous, in table order (offset 0 is
    /// the first entry). Used by integrity verification.</summary>
    public byte[] ReadPayload()
    {
        long payloadEnd = PayloadStart;
        foreach (var tensor in Header.Tensors)
        {
            payloadEnd = Math.Max(payloadEnd, PayloadStart + tensor.Offset + tensor.Len);
        }
        long length = payloadEnd - PayloadStart;
        var buffer = new byte[length];
        if (length > 0)
        {
            _accessor.ReadArray(PayloadStart, buffer, 0, checked((int)length));
        }
        return buffer;
    }

    public void Dispose()
    {
        _accessor.Dispose();
        _mapped.Dispose();
    }
}