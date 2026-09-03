using Amql.Vindex3;

namespace Amql.Inference;

// ── The operation program ──────────────────────────────────────────────────
// Built by the planner from the container alone; consumed by the runtime.
// Every operand is an OperandRef (logical object + segment-relative tensor
// name) — the original HF tensor names were stripped at encode time.

public sealed class EmbeddingOp
{
    public required OperandRef Table { get; init; }
    public required int VocabSize { get; init; }
    public required int HiddenSize { get; init; }

    /// <summary>Weightless normalisation applied per looked-up row.</summary>
    public EmbeddingNorm? Norm { get; init; }

    public double? Scale { get; init; }
}

public sealed class NormOp
{
    public required OperandRef Weight { get; init; }
    public required NormType Kind { get; init; }
    public required double Eps { get; init; }
    public float WeightOffset { get; init; }
    public required int Width { get; init; }

    public static NormOp From(OperandRef weight, NormSpec spec, int width) => new()
    {
        Weight = weight,
        Kind = spec.Kind,
        Eps = spec.Eps,
        WeightOffset = spec.WeightOffset,
        Width = width,
    };
}

public sealed class AttentionOp
{
    public const string Operator = "softmax";

    public required int NumQHeads { get; init; }
    public required int NumKvHeads { get; init; }
    public required int HeadDim { get; init; }

    public double ScoreScale { get; init; } = 1.0;
    public float? LogitSoftcapping { get; init; }

    /// <summary>Sliding window in positions; null means full span.</summary>
    public long? Window { get; init; }

    /// <summary>Position policy resolved from the per-layer table.</summary>
    public required PositionPolicy Position { get; init; }

    public bool VFromK { get; init; }

    /// <summary>Hard attention output gate (Qwen3.5): <c>q_proj</c> is
    /// [hidden, 2×QDim], the second half gates the attention output
    /// elementwise — <c>o_proj(attn_out ⊙ sigmoid(gate))</c>.</summary>
    public bool OutputGate { get; init; }

    public QkNormScope QkNormScope { get; init; } = QkNormScope.PerHead;
    public float QkNormWeightOffset { get; init; }
    public ParameterFreeQkNorm ParameterFreeQkNorm { get; init; } = new();

    /// <summary>Weighted QK norm operands (learned q_norm/k_norm per
    /// head) — distinct from the parameter-free variant. The eps and the
    /// 1+w affine convention come from the declared norm surface.</summary>
    public NormOp? QNorm { get; init; }
    public NormOp? KNorm { get; init; }

    /// <summary>Epsilon for the parameter-free QK norm (weightless; the
    /// planner takes it from the declared norm surface — an executor never
    /// defaults one).</summary>
    public double ParameterFreeQkNormEps { get; init; } = 1e-6;

    public required OperandRef QProj { get; init; }
    public required OperandRef KProj { get; init; }
    public required OperandRef VProj { get; init; }
    public required OperandRef OProj { get; init; }

    /// <summary>Width of the q_proj operand: QDim when ungated, 2×QDim
    /// when the output gate is declared.</summary>
    public int QProjWidth => OutputGate ? 2 * QDim : QDim;

    public int KvDim => NumKvHeads * HeadDim;
    public int QDim => NumQHeads * HeadDim;
}

public sealed class DenseFfnOp
{
    public required OperandRef Up { get; init; }
    public required OperandRef Down { get; init; }
    public OperandRef? Gate { get; init; }
    public required Activation Activation { get; init; }
    public required int IntermediateSize { get; init; }
    public required int HiddenSize { get; init; }

    public bool IsGated => Gate is not null;
}

public sealed class RoutedFfnOp
{
    public required OperandRef Router { get; init; }
    public required int NumExperts { get; init; }
    public required int TopK { get; init; }
    public required int ExpertIntermediateSize { get; init; }
    public required int HiddenSize { get; init; }
    public required Activation Activation { get; init; }
    public required ExpertRoutingPolicy RoutingPolicy { get; init; }

    /// <summary>Per-expert operand names (gate is null when the expert
    /// shape is ungated — not typical).</summary>
    public required string ExpertGatePrefix { get; init; }
    public required string ExpertUpPrefix { get; init; }
    public required string ExpertDownPrefix { get; init; }
}

public sealed class LayerFfn
{
    public DenseFfnOp? Dense { get; init; }
    public RoutedFfnOp? Routed { get; init; }
    public bool IsPresent => Dense is not null || Routed is not null;
}

/// <summary>
/// The Qwen3.5-style linear-attention operator (GatedDeltaNet): a
/// depthwise causal conv over the qkv stream, then a gated delta-rule
/// recurrence with per-(key)-head state, a z-gated RMSNorm on the value
/// head dim, and the output projection. Position-free (NoPE). The state
/// the recurrence carries is caller-owned per layer.
/// </summary>
public sealed class LinearAttentionOp
{
    public required OperandRef InProjQkv { get; init; }
    public required OperandRef InProjZ { get; init; }
    public required OperandRef InProjA { get; init; }
    public required OperandRef InProjB { get; init; }
    public required OperandRef Conv1d { get; init; }
    public required OperandRef ALog { get; init; }
    public required OperandRef DtBias { get; init; }
    public required OperandRef NormWeight { get; init; }
    public required OperandRef OutProj { get; init; }

    public required int NumKHeads { get; init; }
    public required int HeadKDim { get; init; }
    public required int NumVHeads { get; init; }
    public required int HeadVDim { get; init; }
    public required int ConvKernel { get; init; }
    public required int HiddenSize { get; init; }
    public required double NormEps { get; init; }

    public int KeyDim => NumKHeads * HeadKDim;
    public int ValueDim => NumVHeads * HeadVDim;

    /// <summary>The conv stream width — qkv concatenated.</summary>
    public int ConvDim => 2 * KeyDim + ValueDim;
}

public sealed class LayerPlan
{
    /// <summary>The softmax attention op (present on full-attention
    /// layers).</summary>
    public AttentionOp? Attention { get; init; }

    /// <summary>The linear-attention op (present on linear layers).</summary>
    public LinearAttentionOp? LinearAttention { get; init; }

    /// <summary>Norm sites; which of these are bound follows the
    /// placement evidence: PreOnly = pre-attn + pre-ffn (the reference's
    /// "under two norms, post_attention_layernorm IS the pre-FFN norm"),
    /// PrePost = pre+post on both sides.</summary>
    public NormOp? PreAttentionNorm { get; init; }
    public NormOp? PostAttentionNorm { get; init; }
    public NormOp? PreFfnNorm { get; init; }
    public NormOp? PostFfnNorm { get; init; }

    public LayerFfn? Ffn { get; init; }

    /// <summary>Whether this layer carries a stateful operator (its states
    /// must advance position by position — prefill cannot batch rows).</summary>
    public bool IsStateful => LinearAttention is not null;
}

public sealed class OutputOp
{
    public required OperandRef Projection { get; init; }
    public required int VocabSize { get; init; }
    public required int HiddenSize { get; init; }

    /// <summary>True when the output projection reuses the embedding
    /// table (tied weights) — no separate tensor is loaded.</summary>
    public bool ReusesEmbedding { get; init; }

    public double? Multiplier { get; init; }
    public float? LogitSoftcapping { get; init; }
}

/// <summary>The complete generic operation program for one component —
/// the .NET analogue of the reference's <c>ComponentOpPlan</c>. The
/// runtime executes exactly this; nothing else about the model family
/// exists on the execution path.</summary>
public sealed class ComponentOpPlan
{
    public required string ComponentId { get; init; }
    public EmbeddingOp? Embedding { get; init; }
    public required List<LayerPlan> Layers { get; init; }
    public required NormOp FinalNorm { get; init; }
    public OutputOp? Output { get; init; }

    public int HiddenSize => FinalNorm.Width;
}