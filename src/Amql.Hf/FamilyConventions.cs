using System.Text.Json;

namespace Amql.Hf;

/// <summary>
/// The modelling conventions a checkpoint leaves unstated (see
/// <c>family-conventions.json</c>, embedded): the residue left after every fact
/// the config and the tensors do state has been read. Looked up by
/// <c>model_type</c>; a family with no entry follows the plain defaults
/// (RMSNorm·w, unscaled embeddings, half-split rotary pairs).
/// </summary>
public sealed record FamilyConventions
{
    public float NormWeightOffset { get; init; }
    public bool EmbedScaleSqrtHidden { get; init; }
    public bool RopeInterleaved { get; init; }
    public bool ZeroTailPartialRotary { get; init; }
    public bool RopeOnFullLayers { get; init; } = true;

    /// <summary>Which of the facts above came from an entry (fact → source).</summary>
    public Dictionary<string, string> Provenance { get; init; } = new(StringComparer.Ordinal);

    public static readonly FamilyConventions None = new();

    private static readonly Lazy<Dictionary<string, JsonElement>> Table = new(Load);

    /// <summary>The conventions for a family, or <see cref="None"/>.</summary>
    public static FamilyConventions For(string modelType)
    {
        if (!Table.Value.TryGetValue(modelType, out var entry))
        {
            return None;
        }
        string source = $"family-conventions.json[{modelType}]";
        var provenance = new Dictionary<string, string>(StringComparer.Ordinal);
        float offset = 0f;
        bool sqrtHidden = false, interleaved = false, zeroTail = false, ropeOnFull = true;
        foreach (var fact in entry.EnumerateObject())
        {
            switch (fact.Name)
            {
                case "norm_weight_offset":
                    offset = (float)fact.Value.GetDouble();
                    break;
                case "embed_scale":
                    sqrtHidden = fact.Value.GetString() == "sqrt_hidden"
                        ? true
                        : throw new ModelConfigException($"{source}: embed_scale '{fact.Value}' is not a known convention");
                    break;
                case "rope_interleaved":
                    interleaved = fact.Value.GetBoolean();
                    break;
                case "partial_rotary":
                    zeroTail = fact.Value.GetString() == "zero_tail"
                        ? true
                        : throw new ModelConfigException($"{source}: partial_rotary '{fact.Value}' is not a known convention");
                    break;
                case "rope_on_full_layers":
                    ropeOnFull = fact.Value.GetBoolean();
                    break;
                default:
                    throw new ModelConfigException($"{source}: unknown convention '{fact.Name}' — refusing rather than ignoring it");
            }
            provenance[fact.Name] = source;
        }
        return new FamilyConventions
        {
            NormWeightOffset = offset,
            EmbedScaleSqrtHidden = sqrtHidden,
            RopeInterleaved = interleaved,
            ZeroTailPartialRotary = zeroTail,
            RopeOnFullLayers = ropeOnFull,
            Provenance = provenance,
        };
    }

    private static Dictionary<string, JsonElement> Load()
    {
        var table = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        using (var stream = typeof(FamilyConventions).Assembly.GetManifestResourceStream("Amql.Hf.family-conventions.json")
                            ?? throw new InvalidOperationException("embedded family-conventions.json is missing"))
        {
            Merge(table, JsonDocument.Parse(stream).RootElement);
        }
        if (Environment.GetEnvironmentVariable("AMQL_CONVENTIONS") is { Length: > 0 } overridePath)
        {
            Merge(table, JsonDocument.Parse(File.ReadAllBytes(overridePath)).RootElement);
        }
        return table;

        static void Merge(Dictionary<string, JsonElement> into, JsonElement root)
        {
            foreach (var family in root.EnumerateObject())
            {
                if (!family.Name.StartsWith('_') && family.Value.ValueKind == JsonValueKind.Object)
                {
                    into[family.Name] = family.Value.Clone();
                }
            }
        }
    }
}
