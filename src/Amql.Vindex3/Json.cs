using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Vindex3;

/// <summary>
/// Shared System.Text.Json configuration for everything serialised to disk.
/// Field names round-trip with the reference Rust serde spellings —
/// <c>snake_case</c> properties and <c>snake_case</c> enum values — so
/// containers written by this library are byte-compatible with LARQL and
/// vice versa.
/// </summary>
public static class ViJson
{
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) },
    };
}

/// <summary>Raised on malformed or inconsistent VINDEX3 containers. The
/// reference treats every container defect as a typed error, never a
/// guess; this exception is the .NET form of that policy.</summary>
public sealed class ContainerException : Exception
{
    public ContainerException(string message) : base(message) { }

    public ContainerException(string message, Exception inner) : base(message, inner) { }
}