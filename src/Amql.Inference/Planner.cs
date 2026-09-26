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
            ResidualScale = surface.ResidualScale ?? 1.0,
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
        bool linear = policy.Operator == LayerOperators.LinearAttention;
        bool softmax = policy.Operator == LayerOperators.Softmax;
        bool conv = policy.Operator == LayerOperators.Conv;
        if (!linear && !softmax && !conv)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: operator '{policy.Operator}' has no managed implementation (required primitive: {policy.Operator})");
        }
        if (softmax && policy.Position is PositionUnresolved unresolved)
        {
            // Softmax layers embed position; an unresolved policy cannot be
            // executed. Linear layers are position-free — their policy row
            // is carried and ignored.
            throw new UnsupportedOperatorException(
                $"layer {layer}: position policy '{unresolved.Kind}' has no managed implementation");
        }
        if (ffnSurface.GatePolicy is not ExpertGateGated)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: expert gate policy '{ffnSurface.GatePolicy.GetType().Name}' has no managed implementation");
        }

        // ── norm sites + FFN (shared by both operator families) ──────────
        // Try canonical Qwen names first, then LFM2.5 variants
        string preAttnName = TryResolve(store, stackId, layer,
            "input_layernorm.weight", "operator_norm.weight")
            ?? throw new UnsupportedOperatorException($"layer {layer}: no pre-attention norm found");
        string postAttnName = TryResolve(store, stackId, layer,
            "post_attention_layernorm.weight", "ffn_norm.weight")
            ?? throw new UnsupportedOperatorException($"layer {layer}: no post-attention norm found");
        Require(store, stackId, preAttnName, layer, "pre-attention norm");

        NormOp? preAttention = BindNorm(store, stackId, preAttnName, normSurface.Pre, hiddenSize);
        NormOp? postAttention = null;
        NormOp? preFfn;
        NormOp? postFfn = null;
        if (prePost)
        {
            Require(store, stackId, postAttnName, layer, "post-attention norm");
            postAttention = BindNorm(store, stackId, postAttnName, normSurface.Post!, hiddenSize);
            string preFfnName = $"{layer}.mlp.pre_layernorm.weight";
            string postFfnName = $"{layer}.mlp.post_layernorm.weight";
            Require(store, stackId, preFfnName, layer, "pre-ffn norm");
            Require(store, stackId, postFfnName, layer, "post-ffn norm");
            preFfn = BindNorm(store, stackId, preFfnName, normSurface.Post!, hiddenSize);
            postFfn = BindNorm(store, stackId, postFfnName, normSurface.Post!, hiddenSize);
        }
        else
        {
            preFfn = BindNorm(store, stackId, postAttnName, normSurface.Post!, hiddenSize);
        }

        // ── the token mixer ─────────────────────────────────────────────
        return new LayerPlan
        {
            Attention = softmax ? BuildSoftmaxAttention(store, stackId, layer, policy, surface, normSurface) : null,
            LinearAttention = linear ? BuildLinearAttention(store, stackId, layer, surface, normSurface, hiddenSize) : null,
            Conv = conv ? BuildConv(store, stackId, layer, surface, hiddenSize) : null,
            PreAttentionNorm = preAttention,
            PostAttentionNorm = postAttention,
            PreFfnNorm = preFfn,
            PostFfnNorm = postFfn,
            Ffn = BuildFfn(store, stackId, layer, ffnSurface, hiddenSize),
        };
    }

    private static AttentionOp BuildSoftmaxAttention(
        OperandStore store,
        string stackId,
        int layer,
        AttentionLayerPolicy policy,
        ExecutionSurface surface,
        NormSurface normSurface)
    {
        var attnSurface = surface.Attention!;
        if (attnSurface.Sinks is not null)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: the persisted attention surface declares learned sinks with no managed implementation");
        }

        var op = new AttentionOp
        {
            NumQHeads = attnSurface.NumQHeads,
            NumKvHeads = attnSurface.NumKvHeads,
            HeadDim = attnSurface.HeadDim,
            ScoreScale = attnSurface.ScoreScale,
            LogitSoftcapping = attnSurface.LogitSoftcapping,
            Window = policy.Window,
            Position = policy.Position,
            VFromK = policy.VFromK,
            OutputGate = attnSurface.OutputGate is not null,
            QkNormScope = attnSurface.QkNormScope,
            QkNormWeightOffset = attnSurface.QkNormWeightOffset,
            ParameterFreeQkNorm = attnSurface.ParameterFreeQkNorm,
            ParameterFreeQkNormEps = normSurface.Pre.Eps,
            QNorm = BindOptionalQkNorm(store, stackId, layer, "q_norm", attnSurface, normSurface),
            KNorm = BindOptionalQkNorm(store, stackId, layer, "k_norm", attnSurface, normSurface),
            QProj = new OperandRef(stackId, TryResolve(store, stackId, layer, "self_attn.q_proj.weight") ?? $"{layer}.self_attn.q_proj.weight"),
            KProj = new OperandRef(stackId, TryResolve(store, stackId, layer, "self_attn.k_proj.weight") ?? $"{layer}.self_attn.k_proj.weight"),
            VProj = new OperandRef(stackId, TryResolve(store, stackId, layer, "self_attn.v_proj.weight") ?? $"{layer}.self_attn.v_proj.weight"),
            OProj = new OperandRef(stackId, TryResolve(store, stackId, layer, "self_attn.o_proj.weight", "self_attn.out_proj.weight") ?? $"{layer}.self_attn.o_proj.weight"),
        };
        string qName = TryResolve(store, stackId, layer, "self_attn.q_proj.weight") ?? $"{layer}.self_attn.q_proj.weight";
        string kName = TryResolve(store, stackId, layer, "self_attn.k_proj.weight") ?? $"{layer}.self_attn.k_proj.weight";
        string vName = TryResolve(store, stackId, layer, "self_attn.v_proj.weight") ?? $"{layer}.self_attn.v_proj.weight";
        string oName = TryResolve(store, stackId, layer, "self_attn.o_proj.weight", "self_attn.out_proj.weight") ?? $"{layer}.self_attn.o_proj.weight";
        Require(store, stackId, qName, layer, "attention (q_proj)");
        Require(store, stackId, kName, layer, "attention (k_proj)");
        Require(store, stackId, vName, layer, "attention (v_proj)");
        Require(store, stackId, oName, layer, "attention (o_proj)");
        return op;
    }

    /// <summary>Binds a weighted QK norm operand when the learned tensor
    /// exists. The pair is symmetric in the reference — one without the
    /// other is a defect this build refuses.</summary>
    private static NormOp? BindOptionalQkNorm(
        OperandStore store, string stackId, int layer, string kind,
        AttentionSurface attnSurface, NormSurface normSurface)
    {
        string name = $"{layer}.self_attn.{kind}.weight";
        string other = $"{layer}.self_attn.{(kind == "q_norm" ? "k_norm" : "q_norm")}.weight";
        bool present = store.ContainsTensor(stackId, name);
        bool otherPresent = store.ContainsTensor(stackId, other);
        if (present != otherPresent)
        {
            throw new ContainerException(
                $"operand closure: layer {layer}: weighted QK norm pair is asymmetric " +
                $"('{name}' present: {present}, '{other}' present: {otherPresent})");
        }
        if (!present)
        {
            return null;
        }
        Require(store, stackId, name, layer, $"weighted QK norm ({kind})");
        var spec = new NormSpec
        {
            Kind = NormType.RmsNorm,
            Eps = normSurface.Pre.Eps,
            WeightOffset = attnSurface.QkNormWeightOffset,
        };
        return new NormOp
        {
            Weight = new OperandRef(stackId, name),
            Kind = spec.Kind,
            Eps = spec.Eps,
            WeightOffset = spec.WeightOffset,
            Width = attnSurface.HeadDim, // per-head, one weight set shared by all heads
        };
    }

    /// <summary>Builds the linear-attention operator (GatedDeltaNet) for a
    /// linear layer. Geometry comes from the persisted linear surface; the
    /// tensor names are those of the Qwen3.5 GatedDeltaNet.</summary>
    private static LinearAttentionOp BuildLinearAttention(
        OperandStore store, string stackId, int layer,
        ExecutionSurface surface, NormSurface normSurface, int hiddenSize)
    {
        if (surface.LinearAttention is not { } json)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: operator 'linear_attention' declared but no linear attention surface was persisted");
        }

        int KeyHeads = Int(json, "key_heads") ?? throw new UnsupportedOperatorException($"layer {layer}: linear surface missing 'key_heads'");
        int KeyHeadDim = Int(json, "key_head_dim") ?? throw new UnsupportedOperatorException($"layer {layer}: linear surface missing 'key_head_dim'");
        int ValueHeads = Int(json, "value_heads") ?? throw new UnsupportedOperatorException($"layer {layer}: linear surface missing 'value_heads'");
        int ValueHeadDim = Int(json, "value_head_dim") ?? throw new UnsupportedOperatorException($"layer {layer}: linear surface missing 'value_head_dim'");
        int ConvKernel = Int(json, "conv_kernel") ?? throw new UnsupportedOperatorException($"layer {layer}: linear surface missing 'conv_kernel'");

        string[] operands =
        {
            "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
            "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight",
        };
        foreach (var name in operands)
        {
            Require(store, stackId, $"{layer}.linear_attn.{name}", layer, $"linear attention ({name})");
        }

        return new LinearAttentionOp
        {
            InProjQkv = new OperandRef(stackId, $"{layer}.linear_attn.in_proj_qkv.weight"),
            InProjZ = new OperandRef(stackId, $"{layer}.linear_attn.in_proj_z.weight"),
            InProjA = new OperandRef(stackId, $"{layer}.linear_attn.in_proj_a.weight"),
            InProjB = new OperandRef(stackId, $"{layer}.linear_attn.in_proj_b.weight"),
            Conv1d = new OperandRef(stackId, $"{layer}.linear_attn.conv1d.weight"),
            ALog = new OperandRef(stackId, $"{layer}.linear_attn.A_log"),
            DtBias = new OperandRef(stackId, $"{layer}.linear_attn.dt_bias"),
            NormWeight = new OperandRef(stackId, $"{layer}.linear_attn.norm.weight"),
            OutProj = new OperandRef(stackId, $"{layer}.linear_attn.out_proj.weight"),
            NumKHeads = KeyHeads,
            HeadKDim = KeyHeadDim,
            NumVHeads = ValueHeads,
            HeadVDim = ValueHeadDim,
            ConvKernel = ConvKernel,
            HiddenSize = hiddenSize,
            NormEps = normSurface.Pre.Eps,
        };

        static int? Int(System.Text.Json.JsonElement json, string name) =>
            json.ValueKind == System.Text.Json.JsonValueKind.Object &&
            json.TryGetProperty(name, out var v) && v.ValueKind == System.Text.Json.JsonValueKind.Number
                ? v.GetInt32()
                : null;
    }

    private static ConvOp BuildConv(
        OperandStore store, string stackId, int layer,
        ExecutionSurface surface, int hiddenSize)
    {
        // LFM2.5 uses conv_kernel=4 (short causal convolution)
        int convKernel = 4;

        // LFM2.5 conv tensor names: conv.in_proj, conv.conv, conv.out_proj
        string inProjName = $"{layer}.conv.in_proj.weight";
        string convWeightName = $"{layer}.conv.conv.weight";
        string convBiasName = $"{layer}.conv.conv.bias";
        string outProjName = $"{layer}.conv.out_proj.weight";

        Require(store, stackId, inProjName, layer, "conv in_proj");
        Require(store, stackId, convWeightName, layer, "conv weight");
        Require(store, stackId, outProjName, layer, "conv out_proj");

        // Conv bias is optional — try to resolve, null if missing
        OperandRef? convBias = null;
        try
        {
            store.Resolve(stackId, convBiasName);
            convBias = new OperandRef(stackId, convBiasName);
        }
        catch
        {
            // Bias not present — that's OK
        }

        // Conv dim = hidden size (depthwise conv preserves dimensionality)
        int convDim = hiddenSize;

        return new ConvOp
        {
            InProj = new OperandRef(stackId, inProjName),
            ConvWeight = new OperandRef(stackId, convWeightName),
            ConvBias = convBias,
            OutProj = new OperandRef(stackId, outProjName),
            ConvDim = convDim,
            KernelSize = convKernel,
            HiddenSize = hiddenSize,
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
        // Try canonical Qwen names first, then LFM2.5 variants
        string gateName = gated
            ? TryResolve(store, stackId, layer, "mlp.gate_proj.weight", "feed_forward.w1.weight")
                ?? throw new UnsupportedOperatorException($"layer {layer}: no FFN gate found")
            : null!;
        string upName = TryResolve(store, stackId, layer, "mlp.up_proj.weight", "feed_forward.w3.weight")
            ?? throw new UnsupportedOperatorException($"layer {layer}: no FFN up found");
        string downName = TryResolve(store, stackId, layer, "mlp.down_proj.weight", "feed_forward.w2.weight")
            ?? throw new UnsupportedOperatorException($"layer {layer}: no FFN down found");

        if (gated) Require(store, stackId, gateName, layer, "ffn gate");
        Require(store, stackId, upName, layer, "ffn up");
        Require(store, stackId, downName, layer, "ffn down");
        return new LayerFfn
        {
            Dense = new DenseFfnOp
            {
                Gate = gated ? new OperandRef(stackId, gateName) : null,
                Up = new OperandRef(stackId, upName),
                Down = new OperandRef(stackId, downName),
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

    /// <summary>Tries multiple tensor name spellings and returns the first
    /// that resolves. Used for cross-family compatibility (e.g., Qwen's
    /// input_layernorm vs LFM2.5's operator_norm).</summary>
    private static string? TryResolve(OperandStore store, string objectId, int layer, params string[] candidates)
    {
        foreach (var suffix in candidates)
        {
            string fullName = $"{layer}.{suffix}";
            if (store.ContainsTensor(objectId, fullName))
            {
                return fullName;
            }
        }
        return null;
    }
}