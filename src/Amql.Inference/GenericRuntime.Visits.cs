using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The row-exact path: one token at a time through the plan's visit schedule
/// (<see cref="ComponentOpPlan.Schedule"/>), for a plan that
/// <see cref="ComponentOpPlan.RunsByVisit"/>. A visit runs its physical layer's
/// ops with the keys, values and convolution histories of its own cache slot;
/// a depth loop's later passes restart from the state saved before the loop;
/// under a block residual the stream carries its bank of block inputs. Every
/// product accumulates in double, row by row (see <see cref="RowMatrix"/>), so
/// a token's logits do not depend on how the work is spread over cores.
/// </summary>
public sealed partial class GenericRuntime
{
    /// <summary>One cache slot's state: a visit's keys and values and the
    /// histories of its short convolutions.</summary>
    private sealed class VisitSlot
    {
        public readonly List<float[]> Keys = new();
        public readonly List<float[]> Values = new();
        public readonly List<float[]> KeyHistory = new();
        public readonly List<float[]> ValueHistory = new();
        public readonly List<float[]> MixerHistory = new();
        public readonly List<float[]> FfnHistory = new();
    }

    private readonly Dictionary<int, VisitSlot> _visitSlots = new();

    /// <summary>The block residual's bank after the last token stepped: what the
    /// final norm's depth mix reads.</summary>
    private List<float[]>? _lastBank;

    private bool _rowPathChecked;

    /// <summary>Called with each visit's output (visit index, hidden state) on
    /// the row-exact path.</summary>
    public Action<int, float[]>? VisitOutputCapture { get; set; }

    private RowOperands Rows => _weights.Rows;

    private void ResetVisitState()
    {
        _visitSlots.Clear();
        _lastBank = null;
    }

    /// <summary>Why the row-exact path cannot run this plan, or null when it can.</summary>
    public static string? RowPathRefusal(ComponentOpPlan plan)
    {
        if (plan.Diffusion is not null || plan.FinalMixer is not null || plan.ResidualScale != 1.0)
        {
            return "masked diffusion, a gated residual or a residual scale";
        }
        if (plan.Embedding is null || plan.Output is null || plan.FinalNorm is null)
        {
            return "no embedding, final norm or output head";
        }
        if (plan.Embedding.Norm is { Kind: not NormType.RmsNorm } || plan.FinalNorm.Kind != NormType.RmsNorm || plan.FinalNorm.Bias is not null)
        {
            return "a LayerNorm at the embedding or the final norm";
        }
        if (plan.Output.Multiplier is not null)
        {
            return "an output multiplier";
        }
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            var layer = plan.Layers[l];
            if (layer.LinearAttention is not null || layer.Conv is not null || layer.MixerHyperConnection is not null ||
                layer.Ngram is not null || layer.OutputScale is not null || layer.ParallelResidual)
            {
                return $"layer {l}: a mixer or residual form other than softmax attention";
            }
            if (layer.PostAttentionNorm is not null || layer.PostFfnNorm is not null)
            {
                return $"layer {l}: post norms";
            }
            foreach (var norm in new[] { layer.PreAttentionNorm, layer.QInputNorm, layer.KInputNorm, layer.VInputNorm, layer.PreFfnNorm })
            {
                if (norm is { Kind: not NormType.RmsNorm } or { Bias: not null })
                {
                    return $"layer {l}: a LayerNorm";
                }
            }
            if (layer.Attention is not { } attn)
            {
                return $"layer {l}: no softmax attention";
            }
            if (attn.Position is not PositionNone)
            {
                return $"layer {l}: position policy {attn.Position.GetType().Name} (the row path runs position-free attention)";
            }
            if (!attn.Causal || attn.VFromK || attn.OutputGate || attn.HeadGate is not null || attn.Indexer is not null ||
                attn.QNorm is not null || attn.KNorm is not null || attn.ParameterFreeQkNorm.Q || attn.ParameterFreeQkNorm.K ||
                attn.LogitSoftcapping is not null || attn.QBias is not null || attn.KBias is not null || attn.VBias is not null ||
                attn.OBias is not null || attn.NumQHeads % attn.NumKvHeads != 0)
            {
                return $"layer {l}: an attention variant the row path does not implement";
            }
            if (layer.Ffn?.Routed is not { Packed: not null, GatePolicy: ExpertGateSoftcappedGlu } routed ||
                routed.RoutingPolicy != ExpertRoutingPolicy.SigmoidNormalisedOverSelected || routed.SharedExpert is not null ||
                routed.Hybrid is not null || routed.RouterScale is not null || routed.ExpertScale is not null || routed.RouterInputNormEps is not null)
            {
                return $"layer {l}: an FFN other than sigmoid-routed packed experts with a softcapped GLU";
            }
        }
        return null;
    }

    /// <summary>One token through the visit schedule; returns the last visit's
    /// output (1 × hidden) and keeps the bank for the final norm.</summary>
    private Tensor2D StepVisits(int token)
    {
        if (!_rowPathChecked)
        {
            if (RowPathRefusal(_plan) is { } refusal)
            {
                throw new UnsupportedOperatorException($"the row-exact path cannot run this plan: {refusal}");
            }
            _rowPathChecked = true;
        }
        var h = EmbedRow(token);
        var bank = new List<float[]>();
        var loop = _plan.DepthLoop;
        float[]? saved = null;
        List<float[]>? savedBank = null;
        List<float[]>? messageBank = null;
        float[]? messageOut = null;
        var schedule = _plan.Schedule;
        for (int v = 0; v < schedule.Count; v++)
        {
            var visit = schedule[v];
            if (loop is not null && visit.Layer == loop.Start)
            {
                if (visit.Pass == 0)
                {
                    saved = h;
                    savedBank = bank.ToList();
                }
                else
                {
                    // The previous pass's bank past the state the loop restarts
                    // from, and its output, are what this pass's gates read.
                    messageBank = bank.Skip(savedBank!.Count + loop.SkipBlocks).ToList();
                    messageOut = h;
                    h = saved!;
                    bank = savedBank.ToList();
                }
            }
            var layer = _plan.Layers[visit.Layer];
            var messages = visit.Pass > 0 && layer.MessageGates is not null ? (messageBank!, messageOut!) : ((List<float[]>, float[])?)null;
            h = RunRowLayer(visit.Layer, layer, Slot(visit.CacheSlot), h, bank, messages);
            VisitOutputCapture?.Invoke(v, h);
        }
        _lastBank = bank;
        SessionPosition++;
        return new Tensor2D(h, 1, h.Length);
    }

    private VisitSlot Slot(int index)
    {
        if (!_visitSlots.TryGetValue(index, out var slot))
        {
            _visitSlots[index] = slot = new VisitSlot();
        }
        return slot;
    }

    private float[] EmbedRow(int token)
    {
        var embed = _plan.Embedding!;
        if ((uint)token >= (uint)embed.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(token), $"token {token} is outside the vocabulary of {embed.VocabSize}");
        }
        var row = new float[embed.HiddenSize];
        Rows.Matrix(embed.Table, embed.VocabSize, embed.HiddenSize).Row(token, row);
        if (embed.Norm is { } norm)
        {
            row = embed.NormWeight is { } w
                ? RowMath.RmsNorm(row, Rows.Vector(w, embed.HiddenSize), norm.Eps)
                : RowMath.Unweighted(row, norm.Eps);
        }
        if (embed.Scale is { } scale)
        {
            RowMath.Scale(row, scale);
        }
        return row;
    }

    /// <summary>One visit of a layer: its inputs (depth mixes, message gates,
    /// norms), the attention and its residual conv, the residual join (or a new
    /// block), then the FFN the same way.</summary>
    private float[] RunRowLayer(int l, LayerPlan layer, VisitSlot slot, float[] h, List<float[]> bank,
        (List<float[]> Bank, float[] Output)? messages)
    {
        var attn = layer.Attention!;
        var blockInput = h;
        float[] xq = h, xk = h, xv = h;
        if (layer.MixerInputMix is { } mix)
        {
            var mixed = DepthMix(mix, bank, h);
            (xq, xk, xv) = mix.Count == 3 ? (mixed[0], mixed[1], mixed[2]) : (mixed[0], mixed[0], mixed[0]);
        }
        if (messages is { } m)
        {
            xq = GateInput(layer, "q", xq, m);
            xk = GateInput(layer, "k", xk, m);
            xv = GateInput(layer, "v", xv, m);
        }
        float[] qIn = RowNorm(xq, layer.QInputNorm ?? layer.PreAttentionNorm);
        // Inputs that are the same state under the same norm are normalised once.
        float[] kIn = ReferenceEquals(xk, xq) && layer.KInputNorm is null ? qIn : RowNorm(xk, layer.KInputNorm ?? layer.PreAttentionNorm);
        float[] vIn = ReferenceEquals(xv, xq) && layer.VInputNorm is null ? qIn : RowNorm(xv, layer.VInputNorm ?? layer.PreAttentionNorm);

        var mixerOut = RowAttention(attn, slot, qIn, kIn, vIn);
        if (layer.MixerOutputConv is { } mixerConv)
        {
            mixerOut = ResidualConv(mixerConv, slot.MixerHistory, mixerOut, null);
        }

        float[] residual;
        if (layer.OpensBlock)
        {
            bank.Add(blockInput);
            residual = mixerOut;
        }
        else
        {
            residual = RowMath.Add(blockInput, mixerOut);
        }

        var ffnIn = layer.FfnInputMix is { } ffnMix ? DepthMix(ffnMix, bank, residual)[0] : residual;
        if (messages is { } fm)
        {
            ffnIn = GateInput(layer, "ffn", ffnIn, fm);
        }
        var ffnOut = PackedExpertsRow(l, layer.Ffn!.Routed!, RowNorm(ffnIn, layer.PreFfnNorm));
        if (layer.FfnOutputConv is { } ffnConv)
        {
            ffnOut = ResidualConv(ffnConv, slot.FfnHistory, ffnOut, null);
        }
        return RowMath.Add(residual, ffnOut);
    }

    private float[] RowNorm(float[] x, NormOp? norm) =>
        norm is null ? x : RowMath.RmsNorm(x, Rows.Vector(norm.Weight, norm.Width), norm.Eps, norm.WeightOffset);

    /// <summary>The final norm on the row path: of the bank's depth mix under a
    /// block residual, else of the state itself.</summary>
    private float[] RowFinalNorm(float[] h)
    {
        if (_plan.BlockResidual is { } block)
        {
            h = DepthMix(block.OutputQuery, _lastBank ?? throw new InvalidOperationException(
                "the final norm reads the depth bank of the token last stepped, and none has been"), h)[0];
        }
        return RowNorm(h, _plan.FinalNorm!);
    }

    private Tensor2D RowFinalNorm(Tensor2D hidden)
    {
        if (hidden.Rows != 1)
        {
            throw new UnsupportedOperatorException(
                "the row-exact path keeps the depth bank of the last token only: its final norm reads one row at a time");
        }
        var h = RowFinalNorm(hidden.Row(0).ToArray());
        return new Tensor2D(h, 1, h.Length);
    }

    /// <summary>The head on the row path: the first <see cref="OutputOp.VocabSize"/>
    /// rows (padding past them is never scored), then the softcap.</summary>
    private Tensor2D RowHead(Tensor2D normed)
    {
        var output = _plan.Output!;
        var logits = Rows.Matrix(output.Projection, output.StoredRows, output.HiddenSize).MatVec(normed.Row(0), output.VocabSize);
        if (output.LogitSoftcapping is { } softcap)
        {
            double cap = softcap;
            for (int v = 0; v < logits.Length; v++)
            {
                logits[v] = (float)(Math.Tanh(logits[v] / cap) * cap);
            }
        }
        return new Tensor2D(logits, 1, logits.Length);
    }
}
