namespace Amql.Safetensors;

/// <summary>An exception raised while parsing or decoding safetensors
/// files. Mirrors the reference's <c>ModelError</c> (parse failures are
/// typed, never silent).</summary>
public sealed class SafetensorsException : Exception
{
    public SafetensorsException(string message) : base(message) { }

    public SafetensorsException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>One tensor as described by a safetensors or VINDEX3 segment
/// header: identity (name), storage dtype, logical shape and the byte
/// range of its payload (relative to the file's payload start).</summary>
public sealed record TensorInfo
{
    public required string Name { get; init; }

    public required Dtype Dtype { get; init; }

    public required long[] Shape { get; init; }

    /// <summary>Byte offset of the payload, relative to payload start.</summary>
    public required long DataStart { get; init; }

    /// <summary>Payload length in bytes.</summary>
    public required long DataLength { get; init; }

    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (var dim in Shape)
            {
                n = checked(n * dim);
            }
            return n;
        }
    }

    public long PayloadBytes => checked(Dtype.ElementSize() * ElementCount);

    public void Validate()
    {
        if (Shape.Any(d => d < 0))
        {
            throw new SafetensorsException($"tensor '{Name}': negative shape dimension");
        }
        if (DataStart < 0 || DataLength < 0)
        {
            throw new SafetensorsException($"tensor '{Name}': negative data offset/length");
        }
        if (DataLength != PayloadBytes)
        {
            throw new SafetensorsException(
                $"tensor '{Name}': payload length {DataLength} does not match dtype {Dtype.Label()} " +
                $"shape [{string.Join(",", Shape)}] which requires {PayloadBytes} bytes");
        }
    }
}