using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The planner: container → <see cref="ComponentOpPlan"/>. It reads the
/// graph and surfaces only — per-layer policy rows are table reads, no
/// layer-pattern arithmetic; norms come from the declared surface; an
/// operator this build has not judged refuses with the primitive named.
/// Operand closure is enforced: every surface-implied tensor must exist in
/// the object's segment, or the whole component refuses.
/// </summary>
public static class Planner
{
    public static ComponentOpPlan Plan(Vindex3Container container, string componentId, OperandStore store)
    {
        var graph = container.Graph
            ?? throw new ContainerException("container records no system graph — cannot plan execution");
        var component = graph.Component(componentId);
        var surface = component.Execution
            ?? throw new ContainerException(
                $"component '{componentId}' has no execution surface — nothing to execute");
        var normSurface = surface.Norm;

        var objects = graph.Objects.Where(o => o.Component == componentId).ToList();
        var stack = objects.FirstOrDefault(o => o.Kind == ObjectKind.DecoderStack)
            ?? throw new ContainerException(
                $"component '{componentId}' owns no decoder_stack object — execution requires one");
        var embedObj = objects.FirstOrDefault(o => o.Kind == ObjectKind.Embedding);
        var finalNormObj = objects.FirstOrDefault(o => o.Kind == ObjectKind.FinalNorm)
            ?? throw new ContainerException(
                $"component '{componentId}' owns no final_norm object");
        var headObj = objects.FirstOrDefault(o => o.Kind == ObjectKind.OutputHead);

        var policies = component.Attention
            ?? throw new ContainerException(
                $"component '{componentId}' carries no per-layer attention table");
        if (policies.Count != component.NumLayers)
        {
            throw new ContainerException(
                $"component '{componentId}': attention table has {policies.Count} rows for {component.NumLayers} layers");
        }

        var attnSurface = surface.Attention
            ?? throw new ContainerException(
                $"component '{componentId}': a decoder stack implies an attention surface (required primitive: attention)");
        var ffnSurface = surface.Ffn
            ?? throw new ContainerException(
                $"component '{componentId}': a decoder stack implies an ffn surface (required primitive: ffn)");

        bool prePost = normSurface.Placement == NormPlacement.PrePost;
        if (normSurface.Post is null)
        {
            throw new ContainerException(
                $"component '{componentId}': no post norm spec was judged — a pre-ffn norm site is implied " +
                "and this build refuses to fabricate one");
        }

        var layers = new List<LayerPlan>(component.NumLayers);
        for (int l = 0; l < component.NumLayers; l++)
        {
            layers.Add(BuildLayer(store, stack.Id, l, policies[l], surface, ffnSurface, normSurface, prePost, component.HiddenSize));
        }

        return new ComponentOpPlan
        {
            ComponentId = componentId,
            Embedding = BuildEmbedding(store, embedObj, surface, component.HiddenSize),
            Layers = layers,
            FinalNorm = BindNorm(store, finalNormObj.Id, "weight", normSurface.FinalNorm, component.HiddenSize),
            Output = BuildOutput(store, embedObj, headObj, surface, component.HiddenSize),
        };
    }

    private static LayerPlan BuildLayer(
        OperandStore store,
        string stackId,
        int layer,
        AttentionLayerPolicy policy,
        ExecutionSurface surface,
        FfnSurface ffnSurface,
        NormSurface normSurface,
        bool prePost,
        int hiddenSize)
    {
        if (policy.Operator != LayerOperators.Softmax)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: operator '{policy.Operator}' has no managed implementation (required primitive: {policy.Operator})");
        }
        if (policy.Position is PositionUnresolved unresolved)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: position policy '{unresolved.Kind}' has no managed implementation");
        }
        if (ffnSurface.GatePolicy is not ExpertGateGated)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: expert gate policy '{ffnSurface.GatePolicy.GetType().Name}' has no managed implementation");
        }

        var attnSurface = surface.Attention!;
        var attn = new AttentionOp
        {
            NumQHeads = attnSurface.NumQHeads,
            NumKvHeads = attnSurface.NumKvHeads,
            HeadDim = attnSurface.HeadDim,
            ScoreScale = attnSurface.ScoreScale,
            LogitSoftcapping = attnSurface.LogitSoftcapping,
            Window = policy.Window,
            Position = policy.Position,
            VFromK = policy.VFromK,
            QkNormScope = attnSurface.QkNormScope,
            QkNormWeightOffset = attnSurface.QkNormWeightOffset,
            ParameterFreeQkNorm = attnSurface.ParameterFreeQkNorm,
            ParameterFreeQkNormEps = normSurface.Pre.Eps,
            QProj = new OperandRef(stackId, $"{layer}.self_attn.q_proj.weight"),
            KProj = new OperandRef(stackId, $"{layer}.self_attn.k_proj.weight"),
            VProj = new OperandRef(stackId, $"{layer}.self_attn.v_proj.weight"),
            OProj = new OperandRef(stackId, $"{layer}.self_attn.o_proj.weight"),
        };
        Require(store, stackId, $"{layer}.self_attn.q_proj.weight", layer, "attention (q_proj)");
        Require(store, stackId, $"{layer}.self_attn.k_proj.weight", layer, "attention (k_proj)");
        Require(store, stackId, $"{layer}.self_attn.v_proj.weight", layer, "attention (v_proj)");
        Require(store, stackId, $"{layer}.self_attn.o_proj.weight", layer, "attention (o_proj)");

        // Norm sites — bound from the placement estate, refused when the
        // operand is missing (closure).
        string preAttnName = $"{layer}.input_layernorm.weight";
        string postAttnName = $"{layer}.post_attention_layernorm.weight";
        Require(store, stackId, preAttnName, layer, "pre-attention norm");

        NormOp? preAttention = BindNorm(store, stackId, preAttnName, normSurface.Pre, 0);
        NormOp? postAttention = null;
        NormOp? preFfn;
        NormOp? postFfn = null;
        if (prePost)
        {
            Require(store, stackId, postAttnName, layer, "post-attention norm");
            postAttention = BindNorm(store, stackId, postAttnName, normSurface.Post!, 0);
            string preFfnName = $"{layer}.mlp.pre_layernorm.weight";
            string postFfnName = $"{layer}.mlp.post_layernorm.weight";
            Require(store, stackId, preFfnName, layer, "pre-ffn norm");
            Require(store, stackId, postFfnName, layer, "post-ffn norm");
            preFfn = BindNorm(store, stackId, preFfnName, normSurface.Post!, 0);
            postFfn = BindNorm(store, stackId, postFfnName, normSurface.Post!, 0);
        }
        else
        {
            preFfn = BindNorm(store, stackId, postAttnName, normSurface.Post!, 0);
        }

        return new LayerPlan
        {
            Attention = attn,
            PreAttentionNorm = preAttention,
            PostAttentionNorm = postAttention,
            PreFfnNorm = preFfn,
            PostFfnNorm = postFfn,
            Ffn = BuildFfn(store, stackId, layer, ffnSurface, hiddenSize),
        };
    }

    private static LayerFfn? BuildFfn(OperandStore store, string stackId, int layer, FfnSurface ffnSurface, int hiddenSize)
    {
        if (ffnSurface.Moe is { } moe)
        {
            Require(store, stackId, $"{layer}.mlp.router.weight", layer, "moe router");
            var routed = new RoutedFfnOp
            {
                Router = new OperandRef(stackId, $"{layer}.mlp.router.weight"),
                NumExperts = moe.Experts,
                TopK = moe.TopK,
                ExpertIntermediateSize = moe.ExpertIntermediateSize,
                HiddenSize = hiddenSize,
                Activation = ffnSurface.Activation,
                RoutingPolicy = moe.RoutingPolicy,
                ExpertGatePrefix = $"{layer}.mlp.experts.",
                ExpertUpPrefix = $"{layer}.mlp.experts.",
                ExpertDownPrefix = $"{layer}.mlp.experts.",
            };
            for (int e = 0; e < moe.Experts; e++)
            {
                Require(store, stackId, $"{layer}.mlp.experts.{e}.gate_proj.weight", layer, "expert gate");
                Require(store, stackId, $"{layer}.mlp.experts.{e}.up_proj.weight", layer, "expert up");
                Require(store, stackId, $"{layer}.mlp.experts.{e}.down_proj.weight", layer, "expert down");
            }
            return new LayerFfn { Routed = routed };
        }

        bool gated = ffnSurface.FfnType == FfnType.Gated;
        if (gated)
        {
            Require(store, stackId, $"{layer}.mlp.gate_proj.weight", layer, "ffn gate");
        }
        Require(store, stackId, $"{layer}.mlp.up_proj.weight", layer, "ffn up");
        Require(store, stackId, $"{layer}.mlp.down_proj.weight", layer, "ffn down");
        return new LayerFfn
        {
            Dense = new DenseFfnOp
            {
                Gate = gated ? new OperandRef(stackId, $"{layer}.mlp.gate_proj.weight") : null,
                Up = new OperandRef(stackId, $"{layer}.mlp.up_proj.weight"),
                Down = new OperandRef(stackId, $"{layer}.mlp.down_proj.weight"),
                Activation = ffnSurface.Activation,
                IntermediateSize = ffnSurface.IntermediateSize,
                HiddenSize = hiddenSize,
            },
        };
    }

    private static EmbeddingOp? BuildEmbedding(OperandStore store, LogicalObject? embedObj, ExecutionSurface surface, int hiddenSize)
    {
        if (embedObj is null)
        {
            return null;
        }
        Require(store, embedObj.Id, "weight", 0, "embedding table");
        return new EmbeddingOp
        {
            Table = new OperandRef(embedObj.Id, "weight"),
            VocabSize = surface.Head?.VocabSize ?? 0,
            HiddenSize = hiddenSize,
            Norm = surface.Head?.EmbeddingNorm,
            Scale = surface.Head?.EmbedScale,
        };
    }

    private static OutputOp? BuildOutput(OperandStore store, LogicalObject? embedObj, LogicalObject? headObj, ExecutionSurface surface, int hiddenSize)
    {
        var headSurface = surface.Head;
        if (headObj is null || headSurface is null)
        {
            return null;
        }
        if (headSurface.HeadReusesEmbedding)
        {
            var embed = embedObj ?? throw new ContainerException(
                "head surface reuses the embedding, but this component owns no embedding object");
            Require(store, embed.Id, "weight", 0, "embedding table (tied head)");
            return new OutputOp
            {
                Projection = new OperandRef(embed.Id, "weight"),
                VocabSize = headSurface.VocabSize,
                HiddenSize = hiddenSize,
                ReusesEmbedding = true,
                Multiplier = headSurface.OutputMultiplier,
                LogitSoftcapping = headSurface.FinalLogitSoftcapping,
            };
        }

        Require(store, headObj.Id, "weight", 0, "output head");
        return new OutputOp
        {
            Projection = new OperandRef(headObj.Id, "weight"),
            VocabSize = headSurface.VocabSize,
            HiddenSize = hiddenSize,
            ReusesEmbedding = false,
            Multiplier = headSurface.OutputMultiplier,
            LogitSoftcapping = headSurface.FinalLogitSoftcapping,
        };
    }

    private static NormOp BindNorm(OperandStore store, string objectId, string tensorName, NormSpec spec, int width)
    {
        Require(store, objectId, tensorName, 0, "norm");
        return NormOp.From(new OperandRef(objectId, tensorName), spec, width);
    }

    private static void Require(OperandStore store, string objectId, string tensorName, int layer, string what)
    {
        if (!store.ContainsTensor(objectId, tensorName))
        {
            string layerLabel = layer >= 0 ? $"layer {layer}: " : string.Empty;
            throw new ContainerException(
                $"operand closure: {layerLabel}surface implies '{objectId}/{tensorName}' ({what}) " +
                "but the segment carries no such tensor");
        }
    }
}