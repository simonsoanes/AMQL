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

    /// <summary>Normalisation applied per looked-up row.</summary>
    public EmbeddingNorm? Norm { get; init; }

    /// <summary>The embedding norm's learned weight; null for the weightless norm.</summary>
    public OperandRef? NormWeight { get; init; }

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

    // ── conditioning (see AttentionSurface) ────────────────────────────────

    /// <summary>q, k and v are projected from three inputs (the layer's
    /// <see cref="LayerPlan.QInputNorm"/>, …), not one.</summary>
    public bool SeparateInputs { get; init; }

    /// <summary>[heads·head_dim]: what each query channel is multiplied by, before the clip.</summary>
    public OperandRef? QueryChannelScale { get; init; }

    /// <summary>L2 length each query and key head is clipped to.</summary>
    public double? QkClip { get; init; }

    /// <summary>Queries and relative states scale by
    /// <c>1 + Alpha·ln(max(1, (t+1)/Floor))</c> on this layer.</summary>
    public PositionLogScaleOp? LogScaling { get; init; }

    public RelativeBiasOp? RelativeBias { get; init; }

    public AttentionSinkOp? Sink { get; init; }

    /// <summary>[QDim, hidden]: the context is multiplied by
    /// <c>sigmoid(OutputGateProjection · q_input)</c> before <see cref="OProj"/>.</summary>
    public OperandRef? OutputGateProjection { get; init; }

    /// <summary>Residual short convolutions on the key and value projections.</summary>
    public ResidualConvOp? KeyConv { get; init; }
    public ResidualConvOp? ValueConv { get; init; }

    /// <summary>Whether anything here needs the row-exact path (see
    /// <see cref="ComponentOpPlan.RunsByVisit"/>).</summary>
    public bool IsConditioned =>
        SeparateInputs || QueryChannelScale is not null || QkClip is not null || LogScaling is not null ||
        RelativeBias is not null || Sink is not null || OutputGateProjection is not null || KeyConv is not null || ValueConv is not null;
}

public sealed class PositionLogScaleOp
{
    public required int Floor { get; init; }
    public required double Alpha { get; init; }
}

/// <summary>A learned relative score bias (see <c>RelativeBiasSurface</c>):
/// <see cref="Projection"/> [heads·Rank, hidden] on the query input,
/// <see cref="ChannelScale"/> [heads, Rank], <see cref="Table"/> [Rank, Extent].</summary>
public sealed class RelativeBiasOp
{
    public required OperandRef Projection { get; init; }
    public required OperandRef ChannelScale { get; init; }
    public required OperandRef Table { get; init; }
    public required int Rank { get; init; }
    public required int Extent { get; init; }
    public required double Clip { get; init; }
}

/// <summary>A sink logit per head: <see cref="Weight"/> [heads] as it is, or
/// [heads, hidden] projected from the query input.</summary>
public sealed class AttentionSinkOp
{
    public required OperandRef Weight { get; init; }
    public required bool Projected { get; init; }
}

/// <summary>
/// A short causal depthwise convolution with a residual, over a history of its
/// last <c>Kernel − 1</c> inputs per cache slot: <c>(Σ taps·x + x) / √(n + 1)</c>.
/// Static: <see cref="Weight"/> [Channels, 1, Kernel], the last tap on the current
/// token (n = Kernel). Dynamic: taps <c>StaticTaps[lag, c] + (TapProjection ·
/// input)[head, lag]</c>, lag 0 the current token (n = 2·Kernel).
/// </summary>
public sealed class ResidualConvOp
{
    public OperandRef? Weight { get; init; }
    public OperandRef? StaticTaps { get; init; }
    public OperandRef? TapProjection { get; init; }
    public required int Channels { get; init; }
    public required int Kernel { get; init; }

    /// <summary>Heads the dynamic taps are shared over (the channels split evenly).</summary>
    public int Heads { get; init; } = 1;

    public bool Dynamic => TapProjection is not null;
}

/// <summary>A depth mix's queries at one site (see <c>BlockResidualSurface</c>):
/// <see cref="Count"/> rows of <see cref="Queries"/>, each giving one input.</summary>
public sealed class DepthMixOp
{
    public required OperandRef Queries { get; init; }
    public required int Count { get; init; }
    public required double Eps { get; init; }
}

/// <summary>One message gate on a looped layer's input (see <c>MessageGateSurface</c>).</summary>
public sealed class MessageGateOp
{
    /// <summary><c>q</c>, <c>k</c>, <c>v</c> or <c>ffn</c>.</summary>
    public required string Input { get; init; }

    public required OperandRef Down { get; init; }
    public required OperandRef Forget { get; init; }
    public required OperandRef Gain { get; init; }

    /// <summary>The message's depth-mix query: row <see cref="QueryRow"/> of <see cref="Query"/>.</summary>
    public required OperandRef Query { get; init; }
    public required int QueryRow { get; init; }
    public required int Rank { get; init; }
    public required double Eps { get; init; }
    public required double MixEps { get; init; }
}

/// <summary>A layer's experts stored one tensor per projection, each in its
/// stored format (see <c>MoeSurface.PackedExperts</c>).</summary>
public sealed class PackedExpertsOp
{
    public required PackedOperand GateUp { get; init; }
    public required PackedOperand Down { get; init; }

    /// <summary>Gate and up rows alternate (2j gate, 2j+1 up); otherwise the
    /// gate rows come first.</summary>
    public required bool InterleavedGateUp { get; init; }
}

/// <summary>A packed expert tensor and its format's companions, when stored.</summary>
public sealed class PackedOperand
{
    public required OperandRef Elements { get; init; }
    public OperandRef? Scale { get; init; }
    public OperandRef? Amax { get; init; }
    public OperandRef? Denominator { get; init; }
}

/// <summary>One run of a physical layer: which pass of a depth loop it is,
/// and the cache slot its keys, values and convolution histories live in.</summary>
public sealed record LayerVisit(int Layer, int Pass, int CacheSlot);

/// <summary>Layers <c>[Start, End)</c> run <see cref="Passes"/> times; a pass
/// after the first restarts from the state saved before <see cref="Start"/>,
/// and its message gates read the previous pass's bank past its first
/// <see cref="SkipBlocks"/> blocks.</summary>
public sealed class DepthLoopOp
{
    public required int Start { get; init; }
    public required int End { get; init; }
    public required int Passes { get; init; }
    public int SkipBlocks { get; init; }
}

/// <summary>The residual's bank of block inputs (see <c>BlockResidualSurface</c>):
/// the final norm reads the mix of <see cref="OutputQuery"/>.</summary>
public sealed class BlockResidualOp
{
    public required int BlockSize { get; init; }
    public required DepthMixOp OutputQuery { get; init; }
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

    /// <summary>The router reads its input RMS-normalised without a weight (null:
    /// as it is), then scaled by <see cref="RouterScale"/> / √hidden when the layer
    /// carries that tensor.</summary>
    public double? RouterInputNormEps { get; init; }
    public OperandRef? RouterScale { get; init; }

    /// <summary>[experts]: what each selected expert's weight is multiplied by.</summary>
    public OperandRef? ExpertScale { get; init; }

    /// <summary>The dense FFN the experts run beside; null when they are the FFN.</summary>
    public HybridFfn? Hybrid { get; init; }

    /// <summary>[1, hidden]: the shared expert's output is scaled by
    /// <c>sigmoid(gate·x)</c>; null adds it unscaled.</summary>
    public OperandRef? SharedExpertGate { get; init; }

    /// <summary>The experts stored one tensor per projection; the per-expert
    /// prefixes and stems are then unused.</summary>
    public PackedExpertsOp? Packed { get; init; }

    /// <summary>[experts]: added to the scores only to choose the experts.</summary>
    public OperandRef? SelectionBias { get; init; }

    /// <summary>What every selected expert's weight is multiplied by.</summary>
    public double RouteScale { get; init; } = 1.0;

    /// <summary>The expert gate arithmetic (the plain gated activation unless stated).</summary>
    public ExpertGatePolicy GatePolicy { get; init; } = ExpertGatePolicy.GatedDefault;
}

/// <summary>
/// A dense FFN beside routed experts in one layer (Gemma 4's MoE). With x the
/// residual after attention:
/// <code>
/// d   = DenseOut(Dense(pre_ffn_norm(x)))
/// r   = RoutedOut(experts(RoutedIn(x)))      routed by the router's read of x
/// out = post_ffn_norm(d + r)
/// </code>
/// </summary>
public sealed class HybridFfn
{
    public required DenseFfnOp Dense { get; init; }
    public NormOp? DenseOut { get; init; }
    public NormOp? RoutedIn { get; init; }
    public NormOp? RoutedOut { get; init; }
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

    /// <summary>[1]: what the residual is multiplied by as it leaves the layer.</summary>
    public OperandRef? OutputScale { get; init; }

    /// <summary>The FFN reads the attention's input (the pre-attention norm
    /// output) and both branches add to the residual together.</summary>
    public bool ParallelResidual { get; init; }

    /// <summary>Whether this layer carries a stateful operator (its states
    /// must advance position by position — prefill cannot batch rows).</summary>
    public bool IsStateful => LinearAttention is not null || Conv is not null || Ngram is not null ||
                              MixerOutputConv is not null || FfnOutputConv is not null;

    // ── separate attention inputs, the depth bank, messages, residual convs ──

    /// <summary>The attention's three inputs' norms, when it reads separate
    /// q, k and v inputs (in place of <see cref="PreAttentionNorm"/>); the same
    /// norm may stand at all three.</summary>
    public NormOp? QInputNorm { get; init; }
    public NormOp? KInputNorm { get; init; }
    public NormOp? VInputNorm { get; init; }

    /// <summary>Under a block residual: the mixer's input mix (null reads the
    /// state itself), the FFN's, and whether this layer opens a block (its input
    /// joins the bank and the residual restarts from the mixer's output).</summary>
    public DepthMixOp? MixerInputMix { get; init; }
    public DepthMixOp? FfnInputMix { get; init; }
    public bool OpensBlock { get; init; }

    /// <summary>On a looped layer: the gates its inputs take from the previous pass.</summary>
    public IReadOnlyList<MessageGateOp>? MessageGates { get; init; }

    /// <summary>Residual short convolutions on the mixer's and the FFN's outputs.</summary>
    public ResidualConvOp? MixerOutputConv { get; init; }
    public ResidualConvOp? FfnOutputConv { get; init; }

    /// <summary>Whether anything here needs the row-exact path.</summary>
    public bool IsRowOnly =>
        QInputNorm is not null || MixerInputMix is not null || FfnInputMix is not null || OpensBlock ||
        MessageGates is not null || MixerOutputConv is not null || FfnOutputConv is not null ||
        Attention is { IsConditioned: true } || Ffn?.Routed is { Packed: not null } or { SelectionBias: not null } ||
        Ffn?.Routed?.GatePolicy is ExpertGateSoftcappedGlu ||
        Ffn?.Routed?.RoutingPolicy == ExpertRoutingPolicy.SigmoidNormalisedOverSelected;
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

    /// <summary>Rows the projection stores when more than <see cref="VocabSize"/>
    /// (a padded vocabulary): only the first VocabSize are scored.</summary>
    public int? TableRows { get; init; }

    public int StoredRows => TableRows ?? VocabSize;
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

    /// <summary>The container's drafter over this component's caches, when it has one.</summary>
    public DrafterPlan? Drafter { get; init; }

    /// <summary>Set for a masked diffusion LM: the stack is bidirectional and
    /// generation unmasks <see cref="MaskedDiffusionOp.MaskTokenId"/> runs
    /// rather than stepping one token at a time.</summary>
    public MaskedDiffusionOp? Diffusion { get; init; }

    /// <summary>A range of layers run several times (null: each layer once).</summary>
    public DepthLoopOp? DepthLoop { get; init; }

    /// <summary>The order layers run in and the cache slot of each run; null is
    /// each layer once, in order, in its own slot.</summary>
    public IReadOnlyList<LayerVisit>? Visits { get; init; }

    /// <summary>The residual's bank of block inputs, when it attends over depth.</summary>
    public BlockResidualOp? BlockResidual { get; init; }

    /// <summary>
    /// Whether this plan runs one token at a time through its visit schedule on
    /// the runtime's row-exact path (GenericRuntime.StepVisits): a depth loop, a
    /// block residual, or a layer op only that path implements. Such a plan has
    /// no layer-major entry (<see cref="GenericRuntime.RunLayerInternal"/> refuses
    /// it) and prefills position by position.
    /// </summary>
    public bool RunsByVisit => _runsByVisit ??= DepthLoop is not null || BlockResidual is not null || Layers.Any(l => l.IsRowOnly);

    private bool? _runsByVisit;

    /// <summary>The visit schedule, explicit or the default one.</summary>
    public IReadOnlyList<LayerVisit> Schedule => Visits ?? Layers.Select((_, l) => new LayerVisit(l, 0, l)).ToList();

    internal ComponentOpPlan With(DrafterPlan drafter) => new()
    {
        ComponentId = ComponentId, Embedding = Embedding, Layers = Layers, FinalNorm = FinalNorm, FinalMixer = FinalMixer,
        Output = Output, ResidualScale = ResidualScale, Diffusion = Diffusion, Drafter = drafter,
        DepthLoop = DepthLoop, Visits = Visits, BlockResidual = BlockResidual,
    };

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