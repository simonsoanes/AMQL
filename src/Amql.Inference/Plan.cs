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

    /// <summary>LayerNorm's additive shift, when the checkpoint ships one.</summary>
    public OperandRef? Bias { get; init; }

    public static NormOp From(OperandRef weight, NormSpec spec, int width, OperandRef? bias = null) => new()
    {
        Weight = weight,
        Bias = bias,
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

    /// <summary>False for a bidirectional stack (masked diffusion): every
    /// query sees every key, before and after it.</summary>
    public bool Causal { get; init; } = true;

    /// <summary>Position policy resolved from the per-layer table.</summary>
    public required PositionPolicy Position { get; init; }

    public bool VFromK { get; init; }

    /// <summary>Hard attention output gate (Qwen3.5): <c>q_proj</c> is
    /// [hidden, 2×QDim], the second half gates the attention output
    /// elementwise — <c>o_proj(attn_out ⊙ sigmoid(gate))</c>.</summary>
    public bool OutputGate { get; init; }

    /// <summary>LFM3's per-head output gate: <c>gate_proj</c> [heads, hidden]
    /// on the mixer input, each head's context scaled by
    /// <c>sigmoid(gate_proj·x)[head]</c> before <c>o_proj</c>.</summary>
    public OperandRef? HeadGate { get; init; }

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

    /// <summary>Additive projection biases (Qwen2, StableLM, Starcoder2…);
    /// null where the checkpoint has none.</summary>
    public OperandRef? QBias { get; init; }
    public OperandRef? KBias { get; init; }
    public OperandRef? VBias { get; init; }
    public OperandRef? OBias { get; init; }

    /// <summary>Rotary pairs are adjacent dims (2i, 2i+1) rather than the
    /// two halves of the rotary window.</summary>
    public bool RopeInterleaved { get; init; }

    /// <summary>QSA (Qwen4): the indexer that selects which keys each query
    /// may attend to; null attends to every visible key.</summary>
    public SparseIndexerOp? Indexer { get; init; }

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
    public OperandRef? GateBias { get; init; }
    public OperandRef? UpBias { get; init; }
    public OperandRef? DownBias { get; init; }
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

    /// <summary>Per-expert operand names: expert <c>e</c>'s up projection is
    /// <c>{ExpertUpPrefix}{e}.{ExpertUpStem}</c>. The stems are the
    /// checkpoint's own (<c>up_proj.weight</c>, Mixtral's <c>w3.weight</c>).</summary>
    public required string ExpertGatePrefix { get; init; }
    public required string ExpertUpPrefix { get; init; }
    public required string ExpertDownPrefix { get; init; }
    public string ExpertGateStem { get; init; } = "gate_proj.weight";
    public string ExpertUpStem { get; init; } = "up_proj.weight";
    public string ExpertDownStem { get; init; } = "down_proj.weight";

    /// <summary>An expert every token passes through, added to the routed
    /// mixture (Qwen2-MoE); null when the layer has none.</summary>
    public DenseFfnOp? SharedExpert { get; init; }

    /// <summary>[1, hidden]: the shared expert's output is scaled by
    /// <c>sigmoid(gate·x)</c>; null adds it unscaled.</summary>
    public OperandRef? SharedExpertGate { get; init; }
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

    /// <summary>The gated norm's activation on z: SiLU in Qwen3.5, sigmoid
    /// where Qwen4 says so (<c>output_gate_type</c>).</summary>
    public bool SigmoidGate { get; init; }

    public int KeyDim => NumKHeads * HeadKDim;
    public int ValueDim => NumVHeads * HeadVDim;

    /// <summary>The conv stream width — qkv concatenated.</summary>
    public int ConvDim => 2 * KeyDim + ValueDim;
}

/// <summary>
/// LFM2/LFM3 gated short convolution: <c>in_proj</c> [3·hidden, hidden]
/// yields (B, C, x); the depthwise causal conv runs over <c>B ⊙ x</c> with no
/// activation, its output is gated by <c>C</c>, then <c>out_proj</c>. The
/// conv state is the last (kernel-1) rows of <c>B ⊙ x</c>.
/// </summary>
public sealed class ConvOp
{
    public required OperandRef InProj { get; init; }
    public required OperandRef ConvWeight { get; init; }
    public required OperandRef? ConvBias { get; init; }
    public required OperandRef OutProj { get; init; }

    public required int ConvDim { get; init; }
    public required int KernelSize { get; init; }
    public required int HiddenSize { get; init; }
}

public sealed class LayerPlan
{
    /// <summary>The softmax attention op (present on full-attention
    /// layers).</summary>
    public AttentionOp? Attention { get; init; }

    /// <summary>The linear-attention op (present on linear layers).</summary>
    public LinearAttentionOp? LinearAttention { get; init; }

    /// <summary>The short-convolution op (present on LFM2.5 conv layers).</summary>
    public ConvOp? Conv { get; init; }

    /// <summary>Norm sites; which of these are bound follows the
    /// placement evidence: PreOnly = pre-attn + pre-ffn (the reference's
    /// "under two norms, post_attention_layernorm IS the pre-FFN norm"),
    /// PrePost = pre+post on both sides.</summary>
    public NormOp? PreAttentionNorm { get; init; }
    public NormOp? PostAttentionNorm { get; init; }
    public NormOp? PreFfnNorm { get; init; }
    public NormOp? PostFfnNorm { get; init; }

    public LayerFfn? Ffn { get; init; }

    /// <summary>Qwen4's Gated Residual around the mixer and the FFN (present
    /// together, and then the layer carries no norms).</summary>
    public HyperConnectionOp? MixerHyperConnection { get; init; }
    public HyperConnectionOp? FfnHyperConnection { get; init; }

    /// <summary>Qwen4's n-gram embedding, added to the residual streams before
    /// the mixer reads them.</summary>
    public NgramEmbeddingOp? Ngram { get; init; }

    /// <summary>The FFN reads the attention's input (the pre-attention norm
    /// output) and both branches add to the residual together.</summary>
    public bool ParallelResidual { get; init; }

    /// <summary>Whether this layer carries a stateful operator (its states
    /// must advance position by position — prefill cannot batch rows).</summary>
    public bool IsStateful => LinearAttention is not null || Conv is not null || Ngram is not null;
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

    /// <summary>The final norm; null when the component ends in a
    /// <see cref="FinalMixer"/> instead (Qwen4's Gated Residual).</summary>
    public NormOp? FinalNorm { get; init; }

    /// <summary>Qwen4: the residual is <see cref="HyperConnectionOp.Streams"/>
    /// streams wide, and this read (without a write) narrows it to the hidden
    /// state the head reads, where other families apply the final norm.</summary>
    public HyperConnectionOp? FinalMixer { get; init; }

    public OutputOp? Output { get; init; }

    /// <summary>Multiplier on each branch at every decoder add (Granite:
    /// h' = h + branch·scale, at both the mixer and the FFN join). 1.0 is the
    /// plain add every other family uses.</summary>
    public double ResidualScale { get; init; } = 1.0;

    /// <summary>Set for a masked diffusion LM: the stack is bidirectional and
    /// generation unmasks <see cref="MaskedDiffusionOp.MaskTokenId"/> runs
    /// rather than stepping one token at a time.</summary>
    public MaskedDiffusionOp? Diffusion { get; init; }

    public int HiddenSize => FinalMixer?.HiddenSize ?? FinalNorm!.Width;

    /// <summary>Parallel residual streams: 1, or the gated residual's count.</summary>
    public int ResidualStreams => FinalMixer?.Streams ?? 1;

    /// <summary>Width of the residual the layers pass along.</summary>
    public int ResidualWidth => ResidualStreams * HiddenSize;

    /// <summary>The final norm, or a refusal naming the operator that stands in
    /// for it — for consumers that read a plain norm's weight.</summary>
    public NormOp RequireFinalNorm(string consumer) => FinalNorm ?? throw new UnsupportedOperatorException(
        $"{consumer} reads a final norm, but this component ends in a gated-residual mixer (Qwen4)");
}

/// <summary>
/// One Gated Residual site (Qwen4's hyper-connections) over a residual of
/// <see cref="Streams"/> × <see cref="HiddenSize"/>:
/// <code>
/// n      = rmsnorm per stream (1+w)
/// read   = mean_s( sigmoid(Up · silu(Down · n / S)) ⊙ n )
/// inject = 2 · sigmoid(Inject · n / S)          (absent on the final mixer)
/// x'     = x + out ⊗ inject
/// </code>
/// </summary>
public sealed class HyperConnectionOp
{
    public required OperandRef Norm { get; init; }
    public required OperandRef Down { get; init; }
    public required OperandRef Up { get; init; }
    public OperandRef? Inject { get; init; }
    public required int Streams { get; init; }
    public required int Lowrank { get; init; }
    public required int HiddenSize { get; init; }
    public required double Eps { get; init; }
    public float NormWeightOffset { get; init; }

    public int Width => Streams * HiddenSize;
}

/// <summary>
/// The QSA indexer of one softmax layer (see <c>SparseAttentionSurface</c>):
/// <c>QkProj</c> yields <see cref="Heads"/> query heads and one key head; the
/// keys are pooled per block of <see cref="CompressRatio"/> tokens, normalised
/// and rotated at the block's first position, and each query keeps its
/// <see cref="BlockTopK"/> best blocks plus the incomplete tail.
/// </summary>
public sealed class SparseIndexerOp
{
    public required OperandRef QkProj { get; init; }
    public required OperandRef QNorm { get; init; }
    public required OperandRef KNorm { get; init; }
    public required int Heads { get; init; }
    public required int HeadDim { get; init; }
    public required int Budget { get; init; }
    public required int CompressRatio { get; init; }
    public required double Eps { get; init; }
    public float NormWeightOffset { get; init; }

    /// <summary>The attention's rotary embedding over this head width (the
    /// same frequencies on the same leading dims).</summary>
    public required PositionPolicy Position { get; init; }

    public bool RopeInterleaved { get; init; }

    public int BlockTopK => Budget / CompressRatio;
}

/// <summary>
/// Qwen4's n-gram embedding (PLE) on one layer: hashed n-gram rows from a
/// sharded table, projected to a value and one key per residual stream; the
/// normalised streams gate the value, and a dilated depthwise conv (dilation =
/// n-gram size) adds local context. The result is added to every stream.
/// </summary>
public sealed class NgramEmbeddingOp
{
    public required NgramEmbeddingSurface Surface { get; init; }

    /// <summary>Position of this layer among the PLE layers (the hash seed).</summary>
    public required int PleIndex { get; init; }

    /// <summary>The table's row-shards, in order, and the first row each holds.</summary>
    public required IReadOnlyList<OperandRef> Shards { get; init; }
    public required IReadOnlyList<long> ShardStarts { get; init; }

    public required OperandRef KeyProj { get; init; }
    public required OperandRef ValueProj { get; init; }
    public required OperandRef NormKey { get; init; }
    public required OperandRef NormQuery { get; init; }
    public required OperandRef NormConv { get; init; }
    public required OperandRef Conv { get; init; }
    public required int Streams { get; init; }
    public required int HiddenSize { get; init; }
    public required double Eps { get; init; }
    public float NormWeightOffset { get; init; }

    /// <summary>Width of one hash head's row.</summary>
    public int RowWidth => Surface.EmbedDim / Surface.NgramHeads;

    public int Width => Streams * HiddenSize;
}

public sealed class MaskedDiffusionOp
{
    public required int MaskTokenId { get; init; }
}