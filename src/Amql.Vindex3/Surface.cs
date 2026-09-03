using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Vindex3;

// ── Norms ──────────────────────────────────────────────────────────────────

public enum NormType
{
    RmsNorm,
    LayerNorm,
}

/// <summary>One normalisation operation, complete. Facts are per-site:
/// there is deliberately no "the model's norm" anywhere in this type.
/// <c>weight_offset</c> carries the affine convention (RMSNorm(x, eps) *
/// (weight + offset)); upstream's centred variant is offset 1.0.</summary>
public sealed class NormSpec
{
    public required NormType Kind { get; init; }
    public required double Eps { get; init; }
    public float WeightOffset { get; init; }
}

/// <summary>A weightless normalisation applied to the embedding output
/// (RMS-normalises every looked-up row, no learned weights ship).</summary>
public sealed class EmbeddingNorm
{
    public required NormType Kind { get; init; }
    public required double Eps { get; init; }
}

/// <summary>The norm placement on a stack. Established from operand
/// topology evidence when unambiguous; the surface pins it otherwise.</summary>
public enum NormPlacement
{
    PreOnly,
    PrePost,
}

public sealed class NormSurface
{
    public required NormSpec Pre { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public NormSpec? Post { get; init; }

    public required NormSpec FinalNorm { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public NormPlacement? Placement { get; init; }
}

// ── Attention surface ──────────────────────────────────────────────────────

public enum QkNormScope
{
    PerHead,
    FullProjection,
}

/// <summary>Parameter-free QK norm: RMS-normalise Q/K/V per head with no
/// learned weight tensors. Distinct from a weighted QK norm whose weights
/// exist in the stack.</summary>
public sealed class ParameterFreeQkNorm
{
    public bool Q { get; init; }
    public bool K { get; init; }
    public bool V { get; init; }

    public bool Any => Q || K || V;
}

public sealed class AttentionSurface
{
    public required int NumQHeads { get; init; }
    public required int NumKvHeads { get; init; }
    public required int HeadDim { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? QueryScale { get; init; }

    public double ScoreScale { get; init; } = 1.0;

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float? LogitSoftcapping { get; init; }

    public QkNormScope QkNormScope { get; init; } = QkNormScope.PerHead;

    public float QkNormWeightOffset { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public ParameterFreeQkNorm ParameterFreeQkNorm { get; init; } = new();

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? OutputGate { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Sinks { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? AttentionBias { get; init; }
}

// ── FFN surface ────────────────────────────────────────────────────────────

public enum FfnType
{
    Dense,
    Gated,
}

public enum Activation
{
    Silu,
    Gelu,
    GeluTanh,
    Relu,
}

/// <summary>How an expert router's selected weights are normalised. The
/// two observable behaviours differ by whether the selected weights sum
/// to 1.</summary>
public enum ExpertRoutingPolicy
{
    SoftmaxThenSelect,
    NormalisedOverSelected,
}

/// <summary>Expert gate arithmetic. <c>Gated</c> is the plain
/// activation(gate) * up used by Mixtral/Gemma/OLMoE;
/// <c>ClampedGlu</c> is GPT-OSS's clamped GLU (clamps both halves,
/// scales the sigmoid argument, adds one to the up branch).</summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "kind")]
[JsonDerivedType(typeof(ExpertGateGated), "gated")]
[JsonDerivedType(typeof(ExpertGateClampedGlu), "clamped_glu")]
public abstract class ExpertGatePolicy
{
    public static ExpertGatePolicy GatedDefault { get; } = new ExpertGateGated();
}

public sealed class ExpertGateGated : ExpertGatePolicy
{
}

public sealed class ExpertGateClampedGlu : ExpertGatePolicy
{
    public float Limit { get; init; }
    public float Alpha { get; init; }
}

public sealed class MoeSurface
{
    public required int Experts { get; init; }
    public required int TopK { get; init; }
    public required int ExpertIntermediateSize { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? RouterKind { get; init; }

    public ExpertRoutingPolicy RoutingPolicy { get; init; } = ExpertRoutingPolicy.SoftmaxThenSelect;

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? RouterBias { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ExpertFormat { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? GateUpLayout { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? SharedExperts { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? BranchScale { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? DensePrefixLayers { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Hybrid { get; init; }
}

public sealed class FfnSurface
{
    public required int IntermediateSize { get; init; }
    public required Activation Activation { get; init; }
    public required FfnType FfnType { get; init; }

    public ExpertGatePolicy GatePolicy { get; init; } = ExpertGatePolicy.GatedDefault;

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public MoeSurface? Moe { get; init; }
}

// ── Head surface ───────────────────────────────────────────────────────────

public sealed class HeadSurface
{
    public required int VocabSize { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public EmbeddingNorm? EmbeddingNorm { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? EmbedScale { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? OutputMultiplier { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float? FinalLogitSoftcapping { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool HeadReusesEmbedding { get; init; }
}

// ── The surface proper ─────────────────────────────────────────────────────

/// <summary>
/// What the generic operations need to run a component: grouped by op
/// (attention, ffn, norm, optional head), every value fully resolved at
/// build time. Presence means presence in schema ≥ 6 (older graphs refused
/// rather than upgraded). The completeness contract derives from the
/// objects a component owns — a decoder stack implies attention + ffn +
/// norm; an embedding/head implies a head surface.
/// </summary>
public sealed class ExecutionSurface
{
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? ContextLength { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public AttentionSurface? Attention { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public FfnSurface? Ffn { get; init; }

    public required NormSurface Norm { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public HeadSurface? Head { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? ResidualScale { get; init; }

    // Operators without a managed implementation in this build are carried
    // verbatim; the planner refuses them naming the primitive.

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? LinearAttention { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Kda { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? KdaGateLowerBound { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Mla { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Mamba2 { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? ConvQkv { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public bool? ResidualInFp32 { get; init; }
}