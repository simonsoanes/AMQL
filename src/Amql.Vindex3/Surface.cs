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

/// <summary>A normalisation applied to the embedding output: every looked-up
/// row RMS-normalised, without a learned weight unless <see cref="Weight"/>
/// names one.</summary>
public sealed class EmbeddingNorm
{
    public required NormType Kind { get; init; }
    public required double Eps { get; init; }

    /// <summary>The norm's learned weight, a tensor of the embedding object;
    /// null for the weightless norm.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Weight { get; init; }
}

/// <summary>The norm placement on a stack. Established from operand
/// topology evidence when unambiguous; the surface pins it otherwise.</summary>
public enum NormPlacement
{
    PreOnly,
    PrePost,

    /// <summary>Sandwich (as LFM3 has): a pre norm into each sub-block and a post
    /// norm on each sub-block's output before its residual add
    /// (operator_norm / post_operator_norm, ffn_norm / post_ffn_norm).</summary>
    Sandwich,
}

public sealed class NormSurface
{
    public required NormSpec Pre { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public NormSpec? Post { get; init; }

    public required NormSpec FinalNorm { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public NormPlacement? Placement { get; init; }

    /// <summary>The final norm's weight in the final-norm object, when it is
    /// not simply <c>weight</c> (the object also holds other tensors).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinalNormWeight { get; init; }
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

    /// <summary>Every position attends to every other — no causal mask. A
    /// masked diffusion LM converted from a causal checkpoint keeps the same
    /// weights and changes only this, so it must be recorded: run causally,
    /// the same tensors give a different (and wrong) distribution.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Bidirectional { get; init; }

    /// <summary>Rotary pairs are adjacent dims (2i, 2i+1) — the GPT-J layout
    /// Cohere and GLM use — rather than the two halves of the rotary window.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool RopeInterleaved { get; init; }

    /// <summary>Each query head and key head is scaled down to this L2 length
    /// when longer: <c>x · min(1, clip / (|x| + 1e-6))</c>. Queries first take
    /// their learned per-channel scale (role <c>q_channel_scale</c>, when the
    /// layer carries it).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? QkClip { get; init; }

    /// <summary>Queries (and relative-bias states) are multiplied by
    /// <c>1 + alpha·ln(max(1, (t+1)/floor))</c> at position t, on the listed layers.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public PositionLogScaling? LogScaling { get; init; }

    /// <summary>A learned score bias over backward distance, in place of a
    /// rotary embedding (see <see cref="RelativeBiasSurface"/>).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public RelativeBiasSurface? RelativeBias { get; init; }

    /// <summary>A sink logit per head joins every softmax and takes probability
    /// mass without contributing a value. Not the legacy <see cref="Sinks"/>
    /// field, which stays refused.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public SinkLogitKind? SinkLogits { get; init; }

    /// <summary>The attention context is gated elementwise by
    /// <c>sigmoid(G · q_input)</c>, G a projection of its own
    /// (role <c>out_gate_proj</c>), before the output projection.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool SeparateOutputGate { get; init; }
}

/// <summary>Where an attention sink's logit comes from.</summary>
public enum SinkLogitKind
{
    /// <summary>One learned logit per head (role <c>sink</c>, [heads]).</summary>
    Static,

    /// <summary>Projected from the query input (role <c>sink_proj</c>, [heads, hidden]).</summary>
    Projected,
}

/// <summary>Position log-scaling of queries on long contexts.</summary>
public sealed class PositionLogScaling
{
    public required int Floor { get; init; }
    public required double Alpha { get; init; }

    /// <summary>The layers (0-based) that scale; the rest leave queries as they are.</summary>
    public required IReadOnlyList<int> Layers { get; init; }
}

/// <summary>
/// A learned relative score bias. Per head, a state
/// <c>r = scale ⊙ (R · q_input)</c> of width <see cref="Rank"/> is L2-clipped to
/// <see cref="Clip"/> (and log-scaled where the layer's queries are); the bias
/// on a key at backward distance <c>δ &lt; extent</c> is <c>r · T[:, δ]</c>, T the
/// layer's table <c>[rank, extent]</c> (role <c>rel_table</c>; its width is the
/// extent). R is role <c>rel_proj</c> [heads·rank, hidden], the scale
/// <c>rel_channel_scale</c> [heads, rank].
/// </summary>
public sealed class RelativeBiasSurface
{
    public required int Rank { get; init; }
    public required double Clip { get; init; }
}

// ── Depth: looped layers, block residuals, cross-pass messages ────────────

/// <summary>
/// Layers <c>[Start, End)</c> run <see cref="Passes"/> times. Each pass has
/// its own key, value and convolution cache slots; every pass after the first
/// restarts from the state (and residual bank) saved after layer
/// <c>Start − 1</c>. Weights are shared between passes.
/// </summary>
public sealed class DepthLoopSurface
{
    public required int Start { get; init; }
    public required int End { get; init; }
    public required int Passes { get; init; }
}

/// <summary>
/// A residual that attends over depth. The stream keeps a bank of block inputs:
/// at a layer whose index is a multiple of <see cref="BlockSize"/>, the layer's
/// input joins the bank and the residual restarts from the attention output
/// instead of adding to it. A sub-block's input is a depth mix:
/// <code>
/// s_i   = rms(state_i)            (no weight, eps <see cref="Eps"/>), states = bank ++ [current]
/// w     = softmax_i(query · s_i / √hidden)
/// input = Σ_i w_i s_i
/// </code>
/// Queries are tensors of <see cref="QueryObject"/>: per layer one for the
/// mixer (<see cref="MixerQueries"/>, [hidden] or [3, hidden] — three give the
/// attention separate q, k and v inputs; null reads the state itself) and one
/// for the FFN; the final norm reads the mix of <see cref="OutputQuery"/>.
/// </summary>
public sealed class BlockResidualSurface
{
    public required int BlockSize { get; init; }
    public required double Eps { get; init; }
    public required string QueryObject { get; init; }
    public required IReadOnlyList<string?> MixerQueries { get; init; }
    public required IReadOnlyList<string> FfnQueries { get; init; }
    public required string OutputQuery { get; init; }
}

/// <summary>
/// A gate on a looped layer's inputs from the pass before it (passes after the
/// first). For each site the message is a depth mix (as
/// <see cref="BlockResidualSurface"/>) over the previous pass's bank — the
/// blocks it opened after its first <see cref="SkipBlocks"/> — and its output:
/// <code>
/// g  = Down · [rms(h); rms(m)]        (no weights, eps <see cref="Eps"/>)
/// h' = (1 − tanh(Forget · g)) ⊙ h + tanh(Gain · g) ⊙ m
/// </code>
/// Down [rank, 2·hidden], Forget and Gain [hidden, rank] are the layer's roles
/// <c>message_gate_{site}</c> (parts <c>down</c>, <c>forget</c>, <c>gain</c>).
/// </summary>
public sealed class MessageGateSurface
{
    public required int Rank { get; init; }
    public required double Eps { get; init; }
    public required int SkipBlocks { get; init; }
    public required IReadOnlyList<MessageGateSite> Sites { get; init; }
}

/// <summary>One gated input. The site's query for looped layer i (counted
/// from the loop's start) is row <c>Offset + Stride·i</c> of
/// <see cref="Query"/>, a tensor of the residual's query object.</summary>
public sealed class MessageGateSite
{
    /// <summary><c>q</c>, <c>k</c>, <c>v</c> (the attention's inputs) or <c>ffn</c>.</summary>
    public required string Input { get; init; }

    public required string Query { get; init; }
    public required int Stride { get; init; }
    public required int Offset { get; init; }
}

/// <summary>
/// Short causal depthwise convolutions with a residual, of width
/// <see cref="Kernel"/>. Each site's output is <c>(Σ taps·x + x) / √(n + 1)</c>,
/// n the number of tap terms. Sites, by layer role:
/// <list type="bullet">
///   <item><c>mixer_out_conv</c>, <c>ffn_out_conv</c>: on a sub-block's output before its
///   residual join; weights [C, 1, K], the last tap on the current token (n = K).</item>
///   <item><c>k_conv</c>, <c>v_conv</c> on the key and value projections: static weights
///   [C, 1, K] as above, or dynamic taps <c>static[lag, c] + dyn[head, lag]</c>,
///   lag 0 the current token, with parts <c>static</c> [K, C] and <c>tap_proj</c>
///   [kv_heads·K, hidden] read from the projection's input (n = 2K).</item>
/// </list>
/// The key and value convolutions run before the key clip and the value norm.
/// </summary>
public sealed class ResidualConvSurface
{
    public required int Kernel { get; init; }
}

// ── Masked diffusion surface ─────────────────────────────────────────────

/// <summary>
/// A masked diffusion language model (MDLM): generation starts from a run of
/// mask tokens and unmasks them over several full-sequence passes, instead of
/// appending one token per pass. Logits at position i predict the token AT i
/// (no next-token shift). The HF identity is carried so export can restore the
/// custom class the checkpoint loads through.
/// </summary>
public sealed class DiffusionSurface
{
    /// <summary><c>masked_diffusion</c> is the only objective judged.</summary>
    public required string Objective { get; init; }

    public required int MaskTokenId { get; init; }

    /// <summary>The checkpoint's <c>model_type</c>, e.g. <c>a2d-qwen3</c>.</summary>
    public required string ModelType { get; init; }

    /// <summary>The checkpoint's <c>architectures</c> entry.</summary>
    public required string Architecture { get; init; }

    /// <summary>The checkpoint's <c>auto_map</c> (remote-code class map), verbatim.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? AutoMap { get; init; }
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

    /// <summary>Scores are <c>sigmoid(router · x)</c>; the experts are chosen by
    /// score plus the router's bias (when <see cref="MoeSurface.SelectionBias"/>),
    /// which does not enter the weights; the chosen scores are normalised to sum
    /// to one (over their sum + 1e-20) and multiplied by
    /// <see cref="MoeSurface.RouteScale"/>.</summary>
    SigmoidNormalisedOverSelected,
}

/// <summary>Expert gate arithmetic. <c>Gated</c> is the plain
/// activation(gate) * up used by Mixtral/Gemma/OLMoE;
/// <c>ClampedGlu</c> is GPT-OSS's clamped GLU (clamps both halves,
/// scales the sigmoid argument, adds one to the up branch);
/// <c>SoftcappedGlu</c> softcaps both halves.</summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "kind")]
[JsonDerivedType(typeof(ExpertGateGated), "gated")]
[JsonDerivedType(typeof(ExpertGateClampedGlu), "clamped_glu")]
[JsonDerivedType(typeof(ExpertGateSoftcappedGlu), "softcapped_glu")]
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

/// <summary>A SwiGLU whose halves are softcapped:
/// <c>cap(g, GateCap) · sigmoid(g) · cap(u, UpCap)</c>, <c>cap(x, c) = c·tanh(x/c)</c>
/// (the sigmoid reads the raw gate).</summary>
public sealed class ExpertGateSoftcappedGlu : ExpertGatePolicy
{
    public double GateCap { get; init; }
    public double UpCap { get; init; }
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

    /// <summary>The routed experts run beside a dense FFN in the same layer:
    /// <c>out(dense_out(dense(in(x))) + routed_out(experts(routed_in(x))))</c>, the
    /// router reading x itself.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Hybrid { get; init; }

    /// <summary>The router reads its input RMS-normalised without a weight.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool RouterInputNorm { get; init; }

    /// <summary>The router's bias (role <c>moe_router.bias</c>) is added to the
    /// scores only to choose the experts, not to weight them.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool SelectionBias { get; init; }

    /// <summary>What every selected expert's weight is multiplied by.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? RouteScale { get; init; }

    /// <summary>All of a layer's experts are stored in one tensor per projection
    /// (roles <c>packed_experts_gate_up</c> [E, 2I, hidden] and
    /// <c>packed_experts_down</c> [E, hidden, I]), each in its stored format —
    /// plain, NVFP4 (FP4 elements with parts <c>scale</c> (E4M3 per 16),
    /// <c>amax</c> [E] and <c>denominator</c> [1]) or MXFP8 (E4M3 with an E8M0
    /// <c>scale</c> per 32) — decoded a row at a time. <see cref="GateUpLayout"/>
    /// says how gate and up rows sit in the first.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool PackedExperts { get; init; }
}

/// <summary>Values of <see cref="MoeSurface.GateUpLayout"/> for packed experts.</summary>
public static class GateUpLayouts
{
    /// <summary>Rows alternate gate, up: row 2j is gate j, row 2j+1 up j.</summary>
    public const string InterleavedRows = "interleaved_rows";
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

    /// <summary>The embedding table's tensor in the embedding object, when it
    /// is not simply <c>weight</c>.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? EmbeddingWeight { get; init; }

    /// <summary>The vocabulary the logits cover, when the embedding and head
    /// carry more rows (<see cref="VocabSize"/>) than that: the rows past it are
    /// padding, never scored.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? UnpaddedVocabSize { get; init; }
}

// ── Pooling surface (shared by embedding and classifier families) ────────

public enum PoolingKind
{
    Mean,
    MeanSqrtLen,
    Max,
    Cls,
    Last,
    WeightedMean,
}

/// <summary>How hidden states are pooled into one vector. For <c>Last</c>,
/// <see cref="LastNonPad"/> distinguishes the final row from the last
/// non-padding token (attention-mask-aware).</summary>
public sealed class PoolingSurface
{
    public required PoolingKind Kind { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool LastNonPad { get; init; }

    public bool MaskPadding { get; init; } = true;

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool IncludeSpecialTokens { get; init; } = true;

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public NormSpec? PreTruncationNorm { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<int>? MatryoshkaDims { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool L2Normalise { get; init; }
}

// ── Classifier surface ────────────────────────────────────────────────────

/// <summary>The head of a sequence-classification model: a <c>score</c>
/// linear that maps the pooled hidden state → label logits. The backbone
/// is an ordinary decoder (or encoder); this surface records the
/// classification-specific facts above it.</summary>
public sealed class ClassifierSurface
{
    public required int NumLabels { get; init; }
    public required string ProblemType { get; init; }
    public required PoolingSurface Pooling { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Template { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool ScoreHeadReusesEmbeddingLayout { get; init; }

    /// <summary>The label table, <c>Labels[i]</c> naming logit <c>i</c> — the
    /// checkpoint's <c>id2label</c>, carried so output reads "entailment" and
    /// not 1, and so export regenerates the same map. Absent means the
    /// checkpoint named none; export then writes <c>LABEL_i</c>, as HF does.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyList<string>? Labels { get; init; }

    /// <summary>The score head carries a bias (<c>score.bias</c>).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool ScoreBias { get; init; }

    /// <summary>The pad token the checkpoint pads (and pools) with.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? PadTokenId { get; init; }
}

// ── Bidirectional encoder surface (ModernBERT) ───────────────────────────

/// <summary>
/// Facts a ModernBERT-style encoder carries beyond the shared attention /
/// FFN / norm surface. Each one changes the arithmetic, so none may be
/// defaulted: LayerNorms and projections without bias, a weighted
/// LayerNorm straight after the embedding lookup, layer 0 skipping its
/// attention norm (the embedding norm already did that work), and a GeGLU
/// whose fused <c>Wi</c> emits the activated half first and the gate second.
/// </summary>
public sealed class EncoderSurface
{
    /// <summary>The HF class this stack came from, e.g. <c>ModernBertModel</c>.</summary>
    public required string Architecture { get; init; }

    public required bool NormBias { get; init; }

    public required bool MlpBias { get; init; }

    /// <summary>A weighted LayerNorm is applied to the looked-up embeddings.</summary>
    public required bool EmbeddingNorm { get; init; }

    /// <summary>Layer 0 has no attention pre-norm (an identity in the reference).</summary>
    public required bool FirstLayerSkipsAttentionNorm { get; init; }

    /// <summary>How the fused FFN input projection splits: <c>input_then_gate</c>
    /// means <c>act(Wi[:I]·x) * (Wi[I:]·x)</c>.</summary>
    public required string FusedFfnLayout { get; init; }

    /// <summary>Tokenizer-level ids the config pins (pad/bos/eos/cls/sep),
    /// carried so an export can restore them.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyDictionary<string, int>? SpecialTokenIds { get; init; }
}

/// <summary>
/// The option-marker scoring head of a Von-style decision model. The
/// premise and K options are packed into one sequence, each option opened
/// by a marker token; the encoder's final hidden state at each marker goes
/// through <c>LayerNorm → Linear(H, S) → GELU → LayerNorm → Linear(S, 1)</c>
/// to give one logit per option, and a softmax over those is the answer.
/// It is not a label classifier: the number of outputs is the number of
/// options in the request, not a fixed label set.
/// </summary>
public sealed class OptionMarkerSurface
{
    public required int ScorerHiddenSize { get; init; }

    public required double NormEps { get; init; }

    public required Activation Activation { get; init; }

    /// <summary>The token whose final hidden state is scored.</summary>
    public required string MarkerToken { get; init; }

    /// <summary>Von 1.2+: each option attends only to the premise and itself,
    /// with position ids restarting after the premise, so its logit cannot
    /// depend on option order. Weights trained this way must run this way.</summary>
    public required bool IndependentOptions { get; init; }

    /// <summary>Digit runs are spaced out before tokenisation ("2026" → "2 0 2 6").</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool DigitSplit { get; init; }
}

// ── multi-stream residual, block-sparse attention, n-gram embedding ──────────

/// <summary>
/// Gated residual (hyper-connections): the residual stream is
/// <see cref="Streams"/> parallel copies of the hidden state (width
/// <c>Streams × hidden</c>). Each sub-block reads a gated mean of the streams
/// and writes its output back into every stream with its own gate:
/// <code>
/// n      = grouped_rmsnorm(x)                       (per stream, 1+w)
/// read   = mean_s( sigmoid(up(silu(down(n)/S))) ⊙ n )_s
/// x'     = x + out ⊗ 2·sigmoid(inject(n)/S)
/// </code>
/// There are no layer norms: <c>hc_norm</c> is the only normalisation, and a
/// final mixer (a read without the write) replaces the final norm. The
/// per-layer tensors are <c>attn_hyper_connection.*</c> and
/// <c>mlp_hyper_connection.*</c>; the final mixer is the final-norm object's
/// <c>hc_norm.weight</c>, <c>input_mix_weight_down.weight</c> and
/// <c>input_mix_weight_up.weight</c>.
/// </summary>
public sealed class HyperConnectionSurface
{
    public required int Streams { get; init; }
    public required int Lowrank { get; init; }
}

/// <summary>
/// Block-sparse attention (QSA in Qwen3.8-Flash-Next): each softmax layer carries an indexer that
/// picks which keys a query may attend to. Keys are pooled (mean) in blocks of
/// <see cref="CompressRatio"/> consecutive visible tokens, normalised and
/// rotated at the block's first position; a query scores every complete block
/// by <c>Σ_heads relu(q_h·k_b) / sqrt(head_dim)</c> and keeps the top
/// <c>Budget / CompressRatio</c> blocks plus the incomplete tail block. With
/// no more complete blocks than that, every visible key is kept and the
/// attention is exactly dense.
/// <para>Tensors: <c>self_attn.indexer.index_qk_proj.weight</c>
/// [(heads + kv_heads)·head_dim, hidden], <c>q_layernorm.weight</c> and
/// <c>k_layernorm.weight</c> [head_dim] (1+w).</para>
/// </summary>
public sealed class SparseAttentionSurface
{
    public required int Heads { get; init; }
    public required int KvHeads { get; init; }
    public required int HeadDim { get; init; }

    /// <summary>Tokens a query may select (<c>indexer_budget</c>).</summary>
    public required int Budget { get; init; }

    /// <summary>Tokens pooled into one key block (<c>indexer_compress_ratio</c>).</summary>
    public required int CompressRatio { get; init; }

    public int BlockTopK => Budget / CompressRatio;
}

/// <summary>
/// Per-layer hashed n-gram embedding (PLE): hashed 2…N-gram ids look up a
/// large table whose rows are projected into a value and one key per residual
/// stream; the normalised streams gate the value, and a dilated depthwise conv
/// adds local context. Applied before the layer's attention read on the
/// (0-based) <see cref="Layers"/>.
/// <para>The table is stored in <see cref="Shards"/> row-shards
/// (<c>ple.ple_embedding.ngram_embedding.shard_i.weight</c>), concatenated in
/// shard order along the rows. It is never materialised whole: a lookup reads
/// the rows it needs from the shard that holds them.</para>
/// </summary>
public sealed class NgramEmbeddingSurface
{
    /// <summary>Decoder layers (0-based) that carry a PLE block.</summary>
    public required IReadOnlyList<int> Layers { get; init; }

    /// <summary>The largest n-gram (<c>ngram_size</c>); 2…N-grams are hashed.</summary>
    public required int NgramSize { get; init; }

    public required int HeadsPerNgram { get; init; }
    public required long VocabSizeBase { get; init; }
    public required int VocabSizeDivisor { get; init; }
    public required long Seed { get; init; }

    /// <summary>Width of one token's concatenated n-gram embedding.</summary>
    public required int EmbedDim { get; init; }

    public required int ConvKernel { get; init; }

    /// <summary>The token that separates documents: an n-gram never reaches
    /// back across it (and the history before the first token reads as it).</summary>
    public required int EosTokenId { get; init; }

    /// <summary>The unigram vocabulary the hash multipliers are bounded by.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Row-shards the table is stored in; 1 is one unsharded tensor.</summary>
    public required int Shards { get; init; }

    public int NgramHeads => (NgramSize - 1) * HeadsPerNgram;
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

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ClassifierSurface? Classifier { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public EncoderSurface? Encoder { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public OptionMarkerSurface? OptionMarker { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Embedding { get; init; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public DiffusionSurface? Diffusion { get; init; }

    /// <summary>The gated residual: present means the residual stream is
    /// several streams wide and the layers carry no norms of their own.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public HyperConnectionSurface? HyperConnections { get; init; }

    /// <summary>Block-sparse attention: every softmax layer selects its keys with an indexer.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public SparseAttentionSurface? SparseAttention { get; init; }

    /// <summary>The hashed n-gram embedding on selected layers.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public NgramEmbeddingSurface? NgramEmbedding { get; init; }

    /// <summary>Keys of the surface this build does not read (a graph written
    /// by another build): the planner refuses a component that has any, naming
    /// them, rather than run it as something it is not. <c>amql-cli graph
    /// upgrade</c> rewrites a graph whose keys a config adapter recognises.</summary>
    [JsonExtensionData]
    public Dictionary<string, JsonElement>? Unread { get; init; }

    /// <summary>A range of layers run several times (see <see cref="DepthLoopSurface"/>).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public DepthLoopSurface? DepthLoop { get; init; }

    /// <summary>The residual attends over a bank of block inputs (see
    /// <see cref="BlockResidualSurface"/>); the layers' inputs and the final
    /// norm's are depth mixes.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public BlockResidualSurface? BlockResidual { get; init; }

    /// <summary>Looped layers gate their inputs on messages from the previous
    /// pass (see <see cref="MessageGateSurface"/>).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public MessageGateSurface? MessageGates { get; init; }

    /// <summary>Short residual convolutions (see <see cref="ResidualConvSurface"/>).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ResidualConvSurface? ResidualConv { get; init; }

    /// <summary>Which stack tensor plays which role, judged on import (see
    /// <see cref="TensorRoles"/>). Absent on containers written before roles
    /// existed; the planner then judges the map from the segment itself.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LayerRoleMap? Roles { get; init; }

    /// <summary>Attention and FFN read the same normalised input and both add
    /// to the residual (<c>x + attn(n(x)) + ffn(n(x))</c>), as in Cohere and
    /// GPT-J, instead of running one after the other.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool ParallelResidual { get; init; }

    /// <summary>Where each fact the importer could not read from the checkpoint
    /// came from (a family convention, by name), so a guess is never silent.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, string>? Provenance { get; init; }
}