using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Vindex3;

/// <summary>
/// The sole root authority of a VINDEX3 container: format version, model
/// identity, the representation directory (representation → segment file)
/// and the physical segment census. Unknown fields round-trip verbatim
/// (mirrors the reference's <c>#[serde(flatten)] extra</c>).
/// </summary>
public sealed class Vindex3Index
{
    public const int CurrentSchema = 4;

    /// <summary>Oldest schema this build will read (3..=4).</summary>
    public const int MinReadableSchema = 3;

    public required int Version { get; init; }
    public required string Model { get; init; }
    public required string Family { get; init; }
    public required int HiddenSize { get; init; }
    public required int NumLayers { get; init; }

    /// <summary>Routed (LYRW v2) containers only — not produced by this
    /// build.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? MoeManifest { get; init; }

    /// <summary>Relative path of the system graph. Absence means "no graph
    /// recorded", never "single-component assumed".</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? SystemGraph { get; init; }

    /// <summary>Representation directory: representation id
    /// <c>{object}@{encoding}</c> → its segment entry.</summary>
    public Dictionary<string, RepresentationEntry> Representations { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Selection profiles; never empty ("exact" at minimum).</summary>
    public List<Profile> Profiles { get; init; } = new();

    /// <summary>Segment census: path stem → physical file count.</summary>
    public Dictionary<string, int> Segments { get; init; } = new(StringComparer.Ordinal);

    public ContainerAuthority Authority { get; init; } = ContainerAuthority.Canonical;

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? DerivedFromModel { get; init; }

    [JsonExtensionData]
    public Dictionary<string, JsonElement>? Extra { get; set; }
}

/// <summary>A container's provenance class: canonical (primary authority)
/// or derived (built from another container's representations).</summary>
public enum ContainerAuthority
{
    Canonical,
    Derived,
}

/// <summary>One directory entry of <c>index.representations</c>: which
/// logical object a segment realises, with what encoding, and the hashes
/// G4 (byte-equivalence verification) recomputes.</summary>
public sealed class RepresentationEntry
{
    public required string Object { get; init; }
    public required string Encoding { get; init; }

    /// <summary>Segment path relative to the container root.</summary>
    public required string Segment { get; init; }

    public required int TensorCount { get; init; }
    public required long PayloadBytes { get; init; }

    /// <summary>SHA-256 over the payload region only.</summary>
    public required string PayloadSha256 { get; init; }

    /// <summary>SHA-256 over the whole segment file (header included).</summary>
    public required string SegmentSha256 { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? CompiledFrom { get; init; }
}

/// <summary>A selection profile: name plus the representation id selected
/// per object. Old spellings may carry a bare string ("exact") instead of
/// the object form; both are accepted (untagged).</summary>
[JsonConverter(typeof(ProfileConverter))]
public sealed class Profile
{
    public string Name { get; init; } = string.Empty;
    public Dictionary<string, string> Selects { get; init; } = new(StringComparer.Ordinal);

    public static Profile Exact() => new() { Name = "exact" };
}

/// <summary>Untagged deserialisation for <see cref="Profile"/>: a bare
/// string becomes "exact"; an object is read field-wise. Reading is manual
/// (the type-level converter must not re-enter itself).</summary>
public sealed class ProfileConverter : JsonConverter<Profile>
{
    public override Profile Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        if (doc.RootElement.ValueKind == JsonValueKind.String)
        {
            return new Profile { Name = doc.RootElement.GetString() ?? string.Empty };
        }

        var root = doc.RootElement;
        var profile = new Profile
        {
            Name = root.TryGetProperty("name", out var nameProp) ? nameProp.GetString() ?? string.Empty : string.Empty,
        };
        if (root.TryGetProperty("selects", out var selectsProp) && selectsProp.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in selectsProp.EnumerateObject())
            {
                profile.Selects[entry.Name] = entry.Value.GetString() ?? string.Empty;
            }
        }
        return profile;
    }

    public override void Write(Utf8JsonWriter writer, Profile value, JsonSerializerOptions options)
    {
        if (value.Selects.Count == 0)
        {
            writer.WriteStringValue(value.Name);
            return;
        }
        writer.WriteStartObject();
        writer.WriteString("name", value.Name);
        writer.WritePropertyName("selects");
        writer.WriteStartObject();
        foreach (var (k, v) in value.Selects)
        {
            writer.WriteString(k, v);
        }
        writer.WriteEndObject();
        writer.WriteEndObject();
    }
}