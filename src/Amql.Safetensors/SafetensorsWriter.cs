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
/// </summary>
public static class SafetensorsWriter
{
    public static void Write(string path, IEnumerable<TensorPayload> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        WriteBytes(path, Build(tensors, metadata));
    }

    public static byte[] Build(IEnumerable<TensorPayload> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var ordered = tensors.OrderBy(t => t.Name, StringComparer.Ordinal).ToArray();
        foreach (var tensor in ordered)
        {
            tensor.Validate();
        }

        // ── header JSON ──────────────────────────────────────────────────
        var entries = new Dictionary<string, (Dtype Dtype, long[] Shape, long From, long To)>();
        long cursor = 0;
        foreach (var tensor in ordered)
        {
            entries[tensor.Name] = (tensor.Dtype, tensor.Shape, cursor, cursor + tensor.Data.Length);
            cursor += tensor.Data.Length;
        }

        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            foreach (var tensor in ordered)
            {
                var (dtype, shape, from, to) = entries[tensor.Name];
                writer.WriteStartObject(tensor.Name);
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
            writer.Flush();
        }

        var headerJson = buffer.ToArray();
        int pad = (SafetensorsFile.HeaderByteAlignment -
                   ((SafetensorsFile.HeaderLengthBytes + headerJson.Length) %
                    SafetensorsFile.HeaderByteAlignment)) %
                  SafetensorsFile.HeaderByteAlignment;

        // ── file: [header length u64 LE][padded header][payload] ────────
        // The stored length is the PADDED header length, so a reader can
        // compute `payload = 8 + storedLen` and rely on alignment — the
        // convention the reference (safetensors crate) writes and reads.
        int storedLength = headerJson.Length + pad;
        long total = SafetensorsFile.HeaderLengthBytes + storedLength + cursor;
        var file = new byte[total];
        BitConverter.TryWriteBytes(file.AsSpan(0, 8), (ulong)storedLength);
        headerJson.CopyTo(file, 8);
        // Pad with spaces (0x20) per the safetensors convention so the JSON
        // parser sees only legal whitespace after the closing brace.
        file.AsSpan(8 + headerJson.Length, pad).Fill(0x20);
        long payloadOffset = SafetensorsFile.HeaderLengthBytes + headerJson.Length + pad;
        foreach (var tensor in ordered)
        {
            var (_, _, from, _) = entries[tensor.Name];
            tensor.Data.CopyTo(file, payloadOffset + from);
        }
        return file;
    }

    private static void WriteBytes(string path, byte[] file) => File.WriteAllBytes(path, file);
}