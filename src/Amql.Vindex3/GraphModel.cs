using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Vindex3;

// ── Enumerations ───────────────────────────────────────────────────────────
// All serialise snake_case, matching the reference `#[serde(rename_all =
// "snake_case")]` spellings.

/// <summary>What part of the model system a component is — evidence-derived
/// in the reference (a nested '*_config' is perception, a declared
/// 'target_layer_ids' is a drafter, otherwise primary text).</summary>
public enum ComponentRole
{
    PrimaryText,
    Perception,
    Drafter,
}

/// <summary>The architectural vocabulary of logical objects. Identity is
/// conceptual (<c>{component}.{kind}</c>); physical tensor names bind an
/// object but never define it.</summary>
public enum ObjectKind
{
    Embedding,
    DecoderStack,
    FinalNorm,
    OutputHead,
    PerceptionTower,
    PerceptionAdapter,
    FeatureProjector,
    ExpertBank,
}

/// <summary>Fidelity of a materialisation: canonical or approximate
/// (quantised / derived).</summary>
public enum Fidelity
{
    Canonical,
    Approximate,
}

/// <summary>Attention span policy from the per-layer table. Positions
/// beyond a sliding window's reach are architecturally dead for that
/// layer before any semantic analysis runs.</summary>
public enum AttentionSpan
{
    Sliding,
    Full,
    Windowed,
}

/// <summary>
/// The layer operators the graph can declare. Kept as strings on purpose:
/// the container is the authority, and an operator this build has not
/// judged must <em>refuse</em> at plan time naming the primitive — never
/// fall back to a fabricated default. Compare the reference's exhaustive
/// match which fails compilation where the planner here throws.
/// </summary>
public static class LayerOperators
{
    public const string Softmax = "softmax";
    public const string GatedDelta = "gated_delta";
    public const string Kda = "kda";
    public const string Mamba2 = "mamba2";
    public const string Recurrent = "recurrent";
    public const string Mla = "mla";
    public const string ConvQkvAttention = "conv_qkv_attention";

    public static readonly IReadOnlyList<string> Known =
        new[] { Softmax, GatedDelta, Kda, Mamba2, Recurrent, Mla, ConvQkvAttention };

    public static bool IsKnown(string op) => Known.Contains(op);
}

// ── System graph ───────────────────────────────────────────────────────────

/// <summary>The semantic IR of a model system: components, logical objects
/// and hidden-state edges. Schema-gated: a graph of another schema is
/// refused, not upgraded.</summary>
public sealed class SystemGraph
{
    public const int CurrentSchema = 6;

    public required int Schema { get; init; }
    public required List<Component> Components { get; init; }
    public required List<LogicalObject> Objects { get; init; }
    public required List<HiddenStateEdge> Edges { get; init; }

    public Component Component(string id) =>
        Components.FirstOrDefault(c => c.Id == id)
        ?? throw new ContainerException($"system graph has no component '{id}'");

    public LogicalObject Object(string id) =>
        Objects.FirstOrDefault(o => o.Id == id)
        ?? throw new ContainerException($"system graph has no logical object '{id}'");
}

public sealed class Component
{
    public required string Id { get; init; }
    public required ComponentRole Role { get; init; }
    public required string SourceArtifact { get; init; }
    public required int NumLayers { get; init; }
    public required int HiddenSize { get; init; }

    /// <summary>Per-layer attention policy table. Absent means "no per-layer
    /// resolution recorded", which is honest only for components that do not
    /// carry one (perception towers today).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<AttentionLayerPolicy>? Attention { get; init; }

    /// <summary>The resolved execution surface the generic runtime needs —
    /// every value judged at build time; an executor never defaults.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ExecutionSurface? Execution { get; init; }

    /// <summary>Perception facts (modality/transform). Carried verbatim as
    /// opaque JSON; not executed by this build.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Perception { get; init; }
}

/// <summary>One layer's attention policy row — the sole authority for
/// span/window/position. The runtime reads this table; it never recomputes
/// layer patterns.</summary>
public sealed class AttentionLayerPolicy
{
    /// <summary>One of <see cref="LayerOperators"/>.</summary>
    public required string Operator { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public AttentionSpan? Span { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? Window { get; init; }

    public required PositionPolicy Position { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public HeadGeometry? Geometry { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool VFromK { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? DeclaredSpan { get; init; }
}

public sealed class HeadGeometry
{
    public required int HeadDim { get; init; }
    public required int NumKvHeads { get; init; }
}

/// <summary>A logical object: conceptual identity plus the physical traces
/// that bind it (source bindings) and the materialisations that realise it
/// (representations).</summary>
public sealed class LogicalObject
{
    public required string Id { get; init; }
    public required string Component { get; init; }
    public required ObjectKind Kind { get; init; }
    public List<SourceBinding> SourceBindings { get; init; } = new();
    public required List<Representation> Representations { get; init; }
}

/// <summary>The physical trace of an object: which artifact, under which
/// tensor prefix, how many tensors and payload bytes.</summary>
public sealed class SourceBinding
{
    public required string Artifact { get; init; }
    public required string TensorPrefix { get; init; }
    public required int Tensors { get; init; }
    public required long Bytes { get; init; }
}

public sealed class Representation
{
    public required string Encoding { get; init; }
    public required Fidelity Fidelity { get; init; }
}

/// <summary>Logical flow of residual states across a component boundary.
/// The edge is not the tensor: the fusion projector implementing its
/// consumer side is a separate tensor object.</summary>
public sealed class HiddenStateEdge
{
    public required string ProducerComponent { get; init; }
    public required List<int> ProducerLayers { get; init; }
    public required string ConsumerComponent { get; init; }
    public required string ConsumerObject { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? BlockSize { get; init; }
}