using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// A drafter that keeps no cache of its own (Gemma 4's assistant): a small stack whose
/// attention has queries only and reads the keys and values of the target's last layer of
/// the same kind. It reads the embedding of the latest token beside the target's
/// final-normed hidden state at the position before it, and predicts the token after; its
/// own hidden state, projected back to the target's width, stands in for the target's in the
/// next draft. Its query position stays where it started.
/// </summary>
public sealed class DrafterPlan
{
    /// <summary>The object a container keeps the drafter's tensors in.</summary>
    public const string Stack = "drafter.stack";

    public required int HiddenSize { get; init; }
    public required int VocabSize { get; init; }
    public required int IntermediateSize { get; init; }
    public required Activation Activation { get; init; }

    /// <summary>[hidden, 2 × target hidden] and [target hidden, hidden].</summary>
    public required OperandRef PreProjection { get; init; }
    public required OperandRef PostProjection { get; init; }

    public required NormOp FinalNorm { get; init; }

    /// <summary>The drafter's own head (its embedding table, used for nothing else).</summary>
    public required OperandRef Head { get; init; }

    public required List<DrafterLayer> Layers { get; init; }

    /// <summary>The container's drafter over this target, or null when it has none.</summary>
    public static DrafterPlan? Build(SystemGraph graph, OperandStore store, ComponentOpPlan target)
    {
        if (graph.Objects.FirstOrDefault(o => o.Id == Stack) is not { Representations.Count: > 0 } stack)
        {
            return null;
        }
        var component = graph.Component(stack.Component);
        var policies = component.Attention ?? throw new ContainerException($"the drafter '{component.Id}' has no attention table");
        int hidden = component.HiddenSize;
        var finalNorm = target.RequireFinalNorm("the drafter");
        OperandRef Ref(string name) => store.ContainsTensor(Stack, name)
            ? new OperandRef(Stack, name)
            : throw new ContainerException($"the drafter has no tensor '{name}'");
        NormOp Norm(string name, int width) => new()
        {
            Weight = Ref(name), Kind = finalNorm.Kind, Eps = finalNorm.Eps, WeightOffset = finalNorm.WeightOffset, Width = width,
        };
        var layers = new List<DrafterLayer>();
        for (int i = 0; i < component.NumLayers; i++)
        {
            var policy = policies[i];
            int source = target.Layers.FindLastIndex(l => l.Attention is { } a && (a.Window is not null) == (policy.Window is not null));
            if (source < 0)
            {
                throw new ContainerException($"the drafter's layer {i} reads a cache the model has no layer for");
            }
            var attn = target.Layers[source].Attention!;
            string p = $"layers.{i}.";
            long heads = store.ShapeOf(Stack, p + "self_attn.q_proj.weight")[0] / attn.HeadDim;
            layers.Add(new DrafterLayer
            {
                Source = source,
                Heads = checked((int)heads),
                Window = policy.Window,
                PreAttention = Norm(p + "input_layernorm.weight", hidden),
                PostAttention = Norm(p + "post_attention_layernorm.weight", hidden),
                PreFfn = Norm(p + "pre_feedforward_layernorm.weight", hidden),
                PostFfn = Norm(p + "post_feedforward_layernorm.weight", hidden),
                QNorm = Norm(p + "self_attn.q_norm.weight", attn.HeadDim),
                QProj = Ref(p + "self_attn.q_proj.weight"),
                OProj = Ref(p + "self_attn.o_proj.weight"),
                Gate = Ref(p + "mlp.gate_proj.weight"),
                Up = Ref(p + "mlp.up_proj.weight"),
                Down = Ref(p + "mlp.down_proj.weight"),
                OutputScale = store.ContainsTensor(Stack, p + "layer_scalar") ? Ref(p + "layer_scalar") : null,
            });
        }
        var dense = target.Layers.Select(l => l.Ffn?.Dense ?? l.Ffn?.Routed?.Hybrid?.Dense).FirstOrDefault(d => d is not null)
            ?? throw new ContainerException("the drafter takes its activation from the model's dense FFN, and the model has none");
        return new DrafterPlan
        {
            HiddenSize = hidden,
            VocabSize = checked((int)store.ShapeOf(Stack, "embed_tokens.weight")[0]),
            IntermediateSize = checked((int)store.ShapeOf(Stack, "layers.0.mlp.up_proj.weight")[0]),
            Activation = dense.Activation,
            PreProjection = Ref("pre_projection.weight"),
            PostProjection = Ref("post_projection.weight"),
            FinalNorm = Norm("norm.weight", hidden),
            Head = Ref("embed_tokens.weight"),
            Layers = layers,
        };
    }
}

/// <summary>
/// How many drafts a reply's next step asks for: one more when all held twice running, one
/// fewer when most did not. Prose keeps one or two at a time and code four.
/// </summary>
public sealed class DraftPace(int most)
{
    /// <summary>The least chance of holding, by the drafter's own reckoning, that a draft is checked at.</summary>
    public const float Floor = 0.45f;

    private int _held;

    public int Next { get; private set; } = Math.Min(2, most);

    public void Saw(int drafted, int kept)
    {
        if (drafted >= Next && kept == drafted)
        {
            if (++_held == 2)
            {
                Next = Math.Min(Next + 1, most);
                _held = 0;
            }
            return;
        }
        _held = 0;
        if (2 * kept < Next)
        {
            Next = Math.Max(1, Next - 1);
        }
    }
}

/// <summary>One layer of a <see cref="DrafterPlan"/>.</summary>
public sealed class DrafterLayer
{
    /// <summary>The target layer whose keys, values and rotary frequencies it reads with.</summary>
    public required int Source { get; init; }
    public required int Heads { get; init; }

    /// <summary>How far back it reads, or null for everything.</summary>
    public required long? Window { get; init; }
    public required NormOp PreAttention { get; init; }
    public required NormOp PostAttention { get; init; }
    public required NormOp PreFfn { get; init; }
    public required NormOp PostFfn { get; init; }
    public required NormOp QNorm { get; init; }
    public required OperandRef QProj { get; init; }
    public required OperandRef OProj { get; init; }
    public required OperandRef Gate { get; init; }
    public required OperandRef Up { get; init; }
    public required OperandRef Down { get; init; }
    public OperandRef? OutputScale { get; init; }
}
