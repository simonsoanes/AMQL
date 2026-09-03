using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Vindex3;

/// <summary>
/// How a layer encodes position. Serialises with the tagged form the
/// reference uses — <c>#[serde(tag = "kind")]</c> — i.e.
/// <c>{"kind":"none"}</c> or <c>{"kind":"rope","theta":10000.0}</c>.
///
/// The runtime implements <see cref="PositionRope"/> (plain rotary) and
/// <see cref="PositionNone"/> (NoPE). Every other variant the reference
/// carries (<c>llama3</c>, <c>yarn</c>, <c>partial_rope</c>, <c>mrope</c>,
/// relative position) is preserved verbatim as
/// <see cref="PositionUnresolved"/> — carried, never dropped, and refused
/// by the planner naming the kind.
/// </summary>
[JsonConverter(typeof(PositionPolicyConverter))]
public abstract class PositionPolicy
{
    public static PositionPolicy None { get; } = new PositionNone();

    public static PositionPolicy CreateRope(double theta) => new PositionRope { Theta = theta };
}

/// <summary>No positional rotation — an intentional per-layer execution
/// property, not theta = 0 (the number 0 must never circulate where
/// "none" is meant).</summary>
public sealed class PositionNone : PositionPolicy
{
}

/// <summary>Rotary position embedding at the given base frequency.</summary>
public sealed class PositionRope : PositionPolicy
{
    public double Theta { get; init; } = 10_000.0;
}

/// <summary>A position policy this build has not judged. The payload is
/// kept verbatim so the fact is never silently lost; planning refuses it.</summary>
public sealed class PositionUnresolved : PositionPolicy
{
    public required string Kind { get; init; }
    public required JsonElement Payload { get; init; }
}

public sealed class PositionPolicyConverter : JsonConverter<PositionPolicy>
{
    public override PositionPolicy Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        if (!root.TryGetProperty("kind", out var kindProp))
        {
            throw new JsonException("position policy is missing its 'kind' discriminator");
        }
        string kind = kindProp.GetString() ?? string.Empty;

        switch (kind)
        {
            case "none":
                return new PositionNone();
            case "rope":
                double theta = root.GetProperty("theta").GetDouble();
                return new PositionRope { Theta = theta };
            default:
                // Unknown/pending variant — carry verbatim, refuse at plan time.
                return new PositionUnresolved { Kind = kind, Payload = root.Clone() };
        }
    }

    public override void Write(Utf8JsonWriter writer, PositionPolicy value, JsonSerializerOptions options)
    {
        switch (value)
        {
            case PositionNone:
                writer.WriteStartObject();
                writer.WriteString("kind", "none");
                writer.WriteEndObject();
                break;
            case PositionRope rope:
                writer.WriteStartObject();
                writer.WriteString("kind", "rope");
                writer.WriteNumber("theta", rope.Theta);
                writer.WriteEndObject();
                break;
            case PositionUnresolved unresolved:
                // Re-emit the original object exactly, kind included.
                writer.WriteStartObject();
                foreach (var property in unresolved.Payload.EnumerateObject())
                {
                    property.WriteTo(writer);
                }
                writer.WriteEndObject();
                break;
            default:
                throw new JsonException($"cannot serialise {value.GetType().Name}");
        }
    }
}