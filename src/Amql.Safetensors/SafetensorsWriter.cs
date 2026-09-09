using System.Text;
using System.Text.Json;

namespace Amql.Safetensors;

/// <summary>One tensor to serialise into a safetensors file.</summary>
public sealed record TensorPayload
{
    public required string Name { get; init; }

    public required Dtype Dtype { get; init; }

    public required long[] Shape { get; init; }

    public required byte[] Data { get; init; }

    public void Validate()
    {
        var info = new TensorInfo
        {
            Name = Name,
            Dtype = Dtype,
            Shape = Shape,
            DataStart = 0,
            DataLength = Data.Length,
        };
        info.Validate();
    }
}

/// <summary>
/// Serialises tensors into a safetensors file. Mirror of the Rust
/// <c>safetensors::tensor::serialize</c>: entries are written in sorted
/// (ordinal) name order, offsets accumulate without inter-tensor padding,
/// and the JSON header is space-padded so <c>8 + headerLen ≡ 0 (mod 8)</c>.
/// Files are streamed (a single tensor can approach the 2 GiB array
/// ceiling, and a whole checkpoint far exceeds it).
/// </summary>
public static class SafetensorsWriter
{
    public static void Write(string path, IEnumerable<TensorPayload> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var ordered = tensors.OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        foreach (var tensor in ordered)
        {
            tensor.Validate();
        }

        var (headerJson, pad) = BuildHeader(ordered, metadata);

        // ── file: [header length u64 LE][padded header][payload] ────────
        // The stored length is the PADDED header length, so a reader can
        // compute `payload = 8 + storedLen` and rely on alignment — the
        // convention the reference (safetensors crate) writes and reads.
        int storedLength = headerJson.Length + pad;
        using var file = File.Create(path);
        Span<byte> length = stackalloc byte[SafetensorsFile.HeaderLengthBytes];
        BitConverter.TryWriteBytes(length, (ulong)storedLength);
        file.Write(length);
        file.Write(headerJson);
        if (pad > 0)
        {
            var filler = new byte[pad];
            Array.Fill(filler, (byte)0x20); // legal JSON whitespace after the closing brace
            file.Write(filler);
        }
        foreach (var tensor in ordered)
        {
            file.Write(tensor.Data);
        }
    }

    public static byte[] Build(IEnumerable<TensorPayload> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var ordered = tensors.OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        foreach (var tensor in ordered)
        {
            tensor.Validate();
        }

        var (headerJson, pad) = BuildHeader(ordered, metadata);
        long cursor = 0;
        foreach (var tensor in ordered)
        {
            cursor += tensor.Data.Length;
        }

        int storedLength = headerJson.Length + pad;
        long total = SafetensorsFile.HeaderLengthBytes + storedLength + cursor;
        var file = new byte[total];
        BitConverter.TryWriteBytes(file.AsSpan(0, 8), (ulong)storedLength);
        headerJson.CopyTo(file, 8);
        file.AsSpan(8 + headerJson.Length, pad).Fill(0x20);
        long payloadOffset = SafetensorsFile.HeaderLengthBytes + headerJson.Length + pad;
        foreach (var tensor in ordered)
        {
            tensor.Data.CopyTo(file, payloadOffset);
        }
        return file;
    }

    private static (byte[] HeaderJson, int Pad) BuildHeader(
        IReadOnlyList<TensorPayload> ordered,
        IReadOnlyDictionary<string, string>? metadata)
    {
        // Payload offsets accumulate across tensors in sorted order.
        var entries = new Dictionary<string, (long From, long To)>();
        long cursor = 0;
        foreach (var tensor in ordered)
        {
            entries[tensor.Name] = (cursor, cursor + tensor.Data.Length);
            cursor += tensor.Data.Length;
        }

        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            foreach (var tensor in ordered)
            {
                var (from, to) = entries[tensor.Name];
                writer.WriteStartObject(tensor.Name);
                writer.WriteString("dtype", tensor.Dtype.Label());
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (var dim in tensor.Shape)
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
            writer.Flush();
        }

        var headerJson = buffer.ToArray();
        int pad = (SafetensorsFile.HeaderByteAlignment -
                   ((SafetensorsFile.HeaderLengthBytes + headerJson.Length) %
                    SafetensorsFile.HeaderByteAlignment)) %
                  SafetensorsFile.HeaderByteAlignment;
        return (headerJson, pad);
    }
}