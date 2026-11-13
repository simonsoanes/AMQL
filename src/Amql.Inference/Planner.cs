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
        // Head-ownership contract: a score head and its surface come together
        // or not at all — one without the other is a container defect.
        bool ownsClassifierHead = objects.Any(o => o.Kind == ObjectKind.ClassifierHead);
        if (ownsClassifierHead != (surface.Classifier is not null) && surface.OptionMarker is null)
        {
            throw new ContainerException(ownsClassifierHead
                ? $"component '{componentId}' owns a classifier_head object but its surface declares no classifier"
                : $"component '{componentId}' declares a classifier surface but owns no classifier_head object");
        }
        if (objects.Any(o => o.Kind == ObjectKind.EncoderStack))
        {
            // A refusal by name, not a malformed container: encoders (nomic-bert,
            // ModernBERT/Von) import and export faithfully, they just do not run here.
            throw new UnsupportedOperatorException(
                $"component '{componentId}' is a bidirectional encoder_stack — this build's runtime executes causal decoders only");
        }
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

        // The diffusion surface and the bidirectional mask come together: the
        // objective reads logits that only a full-span stack produces, and a
        // bidirectional decoder has no generation procedure without one.
        if ((surface.Diffusion is not null) != attnSurface.Bidirectional)
        {
            throw new ContainerException(attnSurface.Bidirectional
                ? $"component '{componentId}': a bidirectional decoder stack declares no diffusion objective"
                : $"component '{componentId}': a masked diffusion surface over a causal attention surface");
        }
        if (surface.Diffusion is { Objective: not "masked_diffusion" } other)
        {
            throw new UnsupportedOperatorException(
                $"component '{componentId}': diffusion objective '{other.Objective}' has no managed implementation");
        }

        if (normSurface.Post is null)
        {
            throw new ContainerException(
                $"component '{componentId}': no post norm spec was judged — a pre-ffn norm site is implied " +
                "and this build refuses to fabricate one");
        }

        // Which tensor plays which role: judged on import and persisted; a
        // container from before roles existed has its map judged here from
        // the segment's own names — the same rules either way.
        var roles = surface.Roles ?? TensorRoles.Judge(store.TensorNames(stack.Id));

        var layers = new List<LayerPlan>(component.NumLayers);
        for (int l = 0; l < component.NumLayers; l++)
        {
            layers.Add(BuildLayer(store, stack.Id, l, policies[l], surface, ffnSurface, normSurface, roles, component.HiddenSize));
        }

        return new ComponentOpPlan
        {
            ComponentId = componentId,
            Embedding = BuildEmbedding(store, embedObj, surface, component.HiddenSize),
            Layers = layers,
            FinalNorm = NormOp.From(Bound(store, finalNormObj.Id, "weight"), normSurface.FinalNorm, component.HiddenSize,
                store.ContainsTensor(finalNormObj.Id, "bias") ? new OperandRef(finalNormObj.Id, "bias") : null),
            Output = BuildOutput(store, embedObj, headObj, surface, component.HiddenSize),
            ResidualScale = surface.ResidualScale ?? 1.0,
            Diffusion = surface.Diffusion is { } diffusion
                ? new MaskedDiffusionOp { MaskTokenId = diffusion.MaskTokenId }
                : null,
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
        LayerRoleMap roles,
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

        // ── norm sites, by role: which exist is the block's shape ─────────
        // (pre-only, sandwich, post-only); the pre-attention site reads the
        // pre spec, every other site the post spec.
        NormOp? preAttention = BindRoleNorm(store, stackId, layer, roles, TensorRoles.AttnIn, normSurface.Pre, hiddenSize);
        NormOp? postAttention = BindRoleNorm(store, stackId, layer, roles, TensorRoles.AttnOut, normSurface.Post!, hiddenSize);
        NormOp? preFfn = BindRoleNorm(store, stackId, layer, roles, TensorRoles.FfnIn, normSurface.Post!, hiddenSize);
        NormOp? postFfn = BindRoleNorm(store, stackId, layer, roles, TensorRoles.FfnOut, normSurface.Post!, hiddenSize);
        if (preAttention is null && postAttention is null && preFfn is null && postFfn is null)
        {
            throw new UnsupportedOperatorException($"layer {layer}: no norm of any role found — refusing an unnormalised block");
        }
        if (surface.ParallelResidual && (preAttention is null || preFfn is not null || postAttention is not null))
        {
            throw new ContainerException(
                $"layer {layer}: a parallel block reads one input norm, but the layer's norms do not have that shape");
        }

        // ── the token mixer ─────────────────────────────────────────────
        return new LayerPlan
        {
            Attention = softmax ? BuildSoftmaxAttention(store, stackId, layer, policy, surface, normSurface, roles) : null,
            LinearAttention = linear ? BuildLinearAttention(store, stackId, layer, surface, normSurface, hiddenSize) : null,
            Conv = conv ? BuildConv(store, stackId, layer, surface, hiddenSize) : null,
            PreAttentionNorm = preAttention,
            PostAttentionNorm = postAttention,
            PreFfnNorm = preFfn,
            PostFfnNorm = postFfn,
            Ffn = BuildFfn(store, stackId, layer, ffnSurface, roles, hiddenSize),
            ParallelResidual = surface.ParallelResidual,
        };
    }

    /// <summary>The layer-qualified tensor for a role, when this layer has it.</summary>
    private static string? RoleTensor(OperandStore store, string stackId, int layer, LayerRoleMap roles, string role, string part = "weight")
    {
        if (roles.Name(role, part) is not { } name)
        {
            return null;
        }
        string full = $"{layer}.{name}";
        return store.ContainsTensor(stackId, full) ? full : null;
    }

    private static OperandRef? RoleOperand(OperandStore store, string stackId, int layer, LayerRoleMap roles, string role, string part = "weight") =>
        RoleTensor(store, stackId, layer, roles, role, part) is { } name ? new OperandRef(stackId, name) : null;

    private static NormOp? BindRoleNorm(OperandStore store, string stackId, int layer, LayerRoleMap roles, string role, NormSpec spec, int width)
    {
        if (RoleTensor(store, stackId, layer, roles, role) is not { } weight)
        {
            return null;
        }
        return NormOp.From(new OperandRef(stackId, weight), spec, width,
            RoleOperand(store, stackId, layer, roles, role, "bias"));
    }

    private static AttentionOp BuildSoftmaxAttention(
        OperandStore store,
        string stackId,
        int layer,
        AttentionLayerPolicy policy,
        ExecutionSurface surface,
        NormSurface normSurface,
        LayerRoleMap roles)
    {
        var attnSurface = surface.Attention!;
        if (attnSurface.Sinks is not null)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: the persisted attention surface declares learned sinks with no managed implementation");
        }
        if (attnSurface.Bidirectional && policy.Window is not null)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: a sliding window over bidirectional attention has no managed implementation");
        }

        bool perHeadGate = attnSurface.OutputGate is { ValueKind: System.Text.Json.JsonValueKind.Object } gateFacts &&
                           gateFacts.TryGetProperty("per_head_gate", out var phg) && phg.ValueKind == System.Text.Json.JsonValueKind.True;
        OperandRef? headGate = null;
        if (perHeadGate)
        {
            string gateName = $"{layer}.self_attn.gate_proj.weight";
            Require(store, stackId, gateName, layer, "per-head output gate (gate_proj)");
            headGate = new OperandRef(stackId, gateName);
        }

        OperandRef Projection(string role) =>
            RoleOperand(store, stackId, layer, roles, role)
            ?? throw new ContainerException(
                $"operand closure: layer {layer}: a softmax layer implies the '{role}' projection but the segment carries none " +
                $"(looked for '{roles.Name(role) ?? string.Join("' / '", TensorRoles.Aliases[role].Select(s => s + ".weight"))}')");

        return new AttentionOp
        {
            NumQHeads = attnSurface.NumQHeads,
            NumKvHeads = attnSurface.NumKvHeads,
            HeadDim = attnSurface.HeadDim,
            ScoreScale = attnSurface.ScoreScale,
            LogitSoftcapping = attnSurface.LogitSoftcapping,
            Window = policy.Window,
            Causal = !attnSurface.Bidirectional,
            Position = policy.Position,
            RopeInterleaved = attnSurface.RopeInterleaved,
            VFromK = policy.VFromK,
            OutputGate = attnSurface.OutputGate is not null && !perHeadGate,
            HeadGate = headGate,
            QkNormScope = attnSurface.QkNormScope,
            QkNormWeightOffset = attnSurface.QkNormWeightOffset,
            ParameterFreeQkNorm = attnSurface.ParameterFreeQkNorm,
            ParameterFreeQkNormEps = normSurface.Pre.Eps,
            QNorm = BindOptionalQkNorm(store, stackId, layer, roles, TensorRoles.QNorm, attnSurface, normSurface),
            KNorm = BindOptionalQkNorm(store, stackId, layer, roles, TensorRoles.KNorm, attnSurface, normSurface),
            QProj = Projection(TensorRoles.Q),
            KProj = Projection(TensorRoles.K),
            VProj = Projection(TensorRoles.V),
            OProj = Projection(TensorRoles.O),
            QBias = RoleOperand(store, stackId, layer, roles, TensorRoles.Q, "bias"),
            KBias = RoleOperand(store, stackId, layer, roles, TensorRoles.K, "bias"),
            VBias = RoleOperand(store, stackId, layer, roles, TensorRoles.V, "bias"),
            OBias = RoleOperand(store, stackId, layer, roles, TensorRoles.O, "bias"),
        };
    }

    /// <summary>Binds a weighted QK norm operand when the learned tensor
    /// exists. The pair is symmetric in the reference — one without the
    /// other is a defect this build refuses. Its width — one head, or the
    /// whole projection (OLMo 2) — is read from the tensor's shape.</summary>
    private static NormOp? BindOptionalQkNorm(
        OperandStore store, string stackId, int layer, LayerRoleMap roles, string role,
        AttentionSurface attnSurface, NormSurface normSurface)
    {
        string otherRole = role == TensorRoles.QNorm ? TensorRoles.KNorm : TensorRoles.QNorm;
        string? name = RoleTensor(store, stackId, layer, roles, role);
        string? other = RoleTensor(store, stackId, layer, roles, otherRole);
        if ((name is null) != (other is null))
        {
            throw new ContainerException(
                $"operand closure: layer {layer}: weighted QK norm pair is asymmetric " +
                $"('{role}' present: {name is not null}, '{otherRole}' present: {other is not null})");
        }
        if (name is null)
        {
            return null;
        }
        int width = checked((int)store.ShapeOf(stackId, name)[0]);
        int heads = role == TensorRoles.QNorm ? attnSurface.NumQHeads : attnSurface.NumKvHeads;
        if (width != attnSurface.HeadDim && width != heads * attnSurface.HeadDim)
        {
            throw new ContainerException(
                $"layer {layer}: '{name}' has width {width} — neither one head ({attnSurface.HeadDim}) nor the projection ({heads * attnSurface.HeadDim})");
        }
        return new NormOp
        {
            Weight = new OperandRef(stackId, name),
            Kind = NormType.RmsNorm,
            Eps = normSurface.Pre.Eps,
            WeightOffset = attnSurface.QkNormWeightOffset,
            Width = width,
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
        // LFM2/LFM3 conv tensor names: conv.in_proj, conv.conv, conv.out_proj
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

        // Depthwise conv over hidden channels; the kernel width (LFM2: 3,
        // LFM3: 7) is the weight's last dim, [hidden, 1, K] or [hidden, K].
        int convDim = hiddenSize;
        var convShape = store.ShapeOf(stackId, convWeightName);
        int convKernel = checked((int)convShape[^1]);
        if (convShape[0] != convDim)
        {
            throw new ContainerException(
                $"layer {layer}: conv weight has {convShape[0]} channels, hidden size is {convDim}");
        }
        var inShape = store.ShapeOf(stackId, inProjName);
        if (inShape[0] != 3L * convDim)
        {
            throw new ContainerException(
                $"layer {layer}: conv in_proj has {inShape[0]} rows — the gated short conv needs 3×hidden (B, C, x)");
        }

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

    private static LayerFfn? BuildFfn(OperandStore store, string stackId, int layer, FfnSurface ffnSurface, LayerRoleMap roles, int hiddenSize)
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

        // Gated or plain is the checkpoint's shape: a gate projection exists or not.
        var gate = RoleOperand(store, stackId, layer, roles, TensorRoles.Gate);
        var up = RoleOperand(store, stackId, layer, roles, TensorRoles.Up);
        var down = RoleOperand(store, stackId, layer, roles, TensorRoles.Down);
        if (up is null && down is null && gate is null)
        {
            return null; // a layer with no FFN at all
        }
        if (up is null || down is null)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: the FFN is incomplete (up: {up is not null}, down: {down is not null})");
        }
        return new LayerFfn
        {
            Dense = new DenseFfnOp
            {
                Gate = gate,
                Up = up,
                Down = down,
                GateBias = RoleOperand(store, stackId, layer, roles, TensorRoles.Gate, "bias"),
                UpBias = RoleOperand(store, stackId, layer, roles, TensorRoles.Up, "bias"),
                DownBias = RoleOperand(store, stackId, layer, roles, TensorRoles.Down, "bias"),
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

    private static OperandRef Bound(OperandStore store, string objectId, string tensorName)
    {
        Require(store, objectId, tensorName, -1, "norm");
        return new OperandRef(objectId, tensorName);
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